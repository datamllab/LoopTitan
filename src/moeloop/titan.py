"""Finite integration with TorchTitan 086bf6c (v0.3.0).

TorchTitan owns the training engine. This file adapts our plain model's parameter
owners, returned sparse statistics and continuation state to that engine.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pickle
import socket
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

TORCHTITAN_COMMIT = "086bf6c166ec85c1298eb5596fa9bf95f6a2d840"


# Source archives used by offline clusters have no installed VCS metadata.
_PINNED_SOURCE_HASHES = {'train.py': '8ddd98849057f0e42608ee6c013026f0574a1c9ca06f5098c3a3d98d05ae272e',
 'trainer.py': '95530e42aa1f719f2df90d3d5bc3f15c9e4d215d24a50a8c4c6a8f74678b59c9',
 'protocols/model.py': 'bff5a093986d905b906d0152ce31a7ea40f6410d3938bcef1de857405bd255a4',
 'protocols/module.py': '5457e59a70e1e1f51faee4f6533d6b8be211209a089369f8b4947e16b8994693',
 'protocols/model_spec.py': '0946b4fd3586e37288a0f91dc52af7b6cf919ee98d9a6b3121765701161f1d01',
 'components/loss.py': '1257c86275f061dc66a8ffca2928c86ad00217553e30ba253f66aef57a189d94',
 'components/validate.py': '327bafa7feeaac3280d435534b4dbc64271cabe018b55cabafe2480d1636b7d6',
 'components/dataloader.py': 'be1d72585262f409fd12a67e99ac9eb673b9eeb208d286f6838b7af1f2b49390',
 'components/checkpointer/dcp.py': '6ddde395487149cd101faec00e09a42554dfd3e66423fa21f6a08c5cf1c1bddf',
 'components/optimizer/optimizer.py': '4b1d08d99e934a8fc2de5db9dfdbf839d9a8a0cdcc42ce00f8067ed65eed449e',
 'components/checkpointer/utils.py': '3b039581f7b38197a8c41e9f262bfe3e0f2121d8ab7c3ae4011ca40482bf4471',
 'distributed/flex_shard/dist_muon.py': '2a4821f3827358606d90bd65076a718d3a3f1d2968345200b3599f766d721f5f',
 'distributed/flex_shard/optimizer_reshard.py': '584618586e89c76592067c53c4e325501b42da1d905116d6f3cada0825f558a6',
 'distributed/flex_shard/_optimizer_reshard_runtime.py': '2c94d3f5d85fa6626f589402efe7d035ce9159fb50dd8feeab4148e4b1c897c8',
 'distributed/flex_shard/_optimizer_reshard_schedule.py': '1627236bb3485892d8b1ec0412f08d744bc31e79aef5655f05d264b6b56a7437',
 'distributed/fsdp.py': '0a0c2d99897ad94ec687416db61d1e7906679c08482a0798e5691dbd70e9cbf7',
 'distributed/activation_checkpoint.py': '44090d6c73621e28db6c9a24dc7a485929d6b8168ed0f707e77b11eaa62deca3',
 'distributed/parallel_dims.py': '7601fd0096f3320080b704b6ba10125e5ff54198feb3d2f3d959b364a5356ce9',
 'models/common/moe.py': '204e411e2233a70e9d736b538da6b63aa4b3450562cd514443d0ed43d8d7a6f1',
 'models/common/token_dispatcher.py': '8b58d293daa6c5fc0f6cd2cd6d87e8bcbfaccd26c33d07ae453266aa0371bcf8',
 'models/common/attention.py': '74638bc24b24d1cff1161a0be69a7421cbd6aa44f9f4ec9d4450ede0054eb83a',
 'config/configs.py': 'e58cfbcc1bf863c4aebd29fb4e5537fe5221aeadc3f08c2ed019f0bbdf492d41',
 'config/configurable.py': '21dda547b2fb9faf2c8560cb152feb3545cd22d09f60d16bae640ff27aaa1d0f',
 'config/manager.py': '7486f134f4c357818fbb140196b85f1c6729feeaccdc89b2c4f6e3f2ebaeb9de'}


def require_torchtitan_pin() -> None:
    """Entry points that train with TorchTitan call this first; importing this file does not."""
    # Validate the package Python will actually load, including offline archives.
    # Installed distribution metadata can refer to a different PYTHONPATH checkout.
    spec = importlib.util.find_spec("torchtitan")
    source = Path(spec.origin).parent if spec is not None and spec.origin else None
    if source is None:
        raise ImportError("Install the pinned TorchTitan train extra or put its source archive on PYTHONPATH")
    mismatches = [name for name, digest in _PINNED_SOURCE_HASHES.items()
                  if not (source / name).is_file()
                  or hashlib.sha256((source / name).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError(f"TorchTitan source must match {TORCHTITAN_COMMIT}; incompatible files: {mismatches}")
    if torch.__version__.split("+")[0] != "2.14.0":
        raise RuntimeError(f"The pinned training adapter requires PyTorch 2.14.0; found {torch.__version__}")


import spmd_types as spmd
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.validate import Validator
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.fsdp import disable_fsdp_gradient_division, get_fsdp_reshard_after_forward_policy
from torchtitan.models.common.moe import GroupedExperts, RoutedExperts
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from .data import STREAM_RULE, PackedDataset, PreparedManifest, build_document_mask, file_sha256
from .evaluate import bpb_fields, bpb_terms, token_byte_tensor
from .model import Attention, ModelConfig, RecursiveLM
from .training import CommittedProgress, TrainingRecipe, capture_rng, continuation_fingerprint, isolated_rng, objective, restore_rng, seed_host_rng, spec_identity
from .untie import check_untied


_compiled_flex = torch.compile(flex_attention, options={"max_autotune": False, "triton.cudagraphs": False})


_validated_manifests: dict[Path, tuple[tuple[int, int], PreparedManifest]] = {}


def training_manifest(path: str | Path) -> PreparedManifest:
    """Verify immutable shared data once on rank zero, then reuse its manifest.

    Other ranks validate metadata/file sizes against the same fingerprint. This
    avoids scanning a complete corpus per rank, per loader and per validation.
    A prepared data directory must remain immutable throughout the run.
    """
    path = Path(path).resolve()
    path = path / "manifest.json" if path.is_dir() else path
    stamp = (path.stat().st_mtime_ns, path.stat().st_size)
    cached = _validated_manifests.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    manifest = None
    if dist.is_initialized():
        result = [None]
        if dist.get_rank() == 0:
            try:
                manifest = PreparedManifest.load(path)
                result[0] = (manifest.fingerprint, None)
            except Exception as error:
                result[0] = (None, str(error))
        dist.broadcast_object_list(result, src=0)
        fingerprint, error = result[0]
        if error is not None:
            raise ValueError(f"prepared data validation failed: {error}")
        if manifest is None:
            manifest = PreparedManifest.load(path, verify_hashes=False)
        if manifest.fingerprint != fingerprint:
            raise ValueError("ranks do not share the same immutable data manifest")
    else:
        manifest = PreparedManifest.load(path)
    _validated_manifests[path] = (stamp, manifest)
    return manifest


class NativeExperts(nn.Module):
    """Native pure dispatch/compute/combine; deliberately excludes MoE counters."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.routed = RoutedExperts.Config(
            inner_experts=GroupedExperts.Config(dim=config.dim, hidden_dim=config.hidden_dim, num_experts=config.num_experts),
            token_dispatcher=LocalTokenDispatcher.Config(num_experts=config.num_experts, top_k=config.top_k),
        ).build()

    def forward(self, x, ids, weights, dispatch_counts):
        return self.routed(x, weights, ids, dispatch_counts)


class NativeAttention(Attention):
    """Keep QKV/RoPE in the core, bind only the document-masked GPU kernel."""

    def _attention(self, q, k, v, attention_mask, *, is_causal):
        if not isinstance(attention_mask, BlockMask):
            return super()._attention(q, k, v, attention_mask, is_causal=is_causal)
        # One bounded compilation shared by physical attention owners. Avoid the
        # upstream max-autotune search for quickstarts and short qualification.
        return _compiled_flex(q, k, v, block_mask=attention_mask, enable_gqa=q.shape[1] != k.shape[1])


def canonical_model_state(state: dict[str, Any]) -> dict[str, Any]:
    """Training wrapper/native-expert keys -> physical inference-core keys."""
    result = {}
    for key, value in state.items():
        key = key.removeprefix("core.")
        key = key.replace(".experts.routed.inner_experts.", ".experts.")
        if key in result:
            raise ValueError(f"duplicate canonical model key: {key}")
        result[key] = value
    return result


def scale_core_expert_lr(optimizers, model_parts, *, multiplier: float) -> None:
    """Split optimizer groups by actual core expert owners, before scheduling.

    Prefix/suffix experts and all non-expert parameters retain their LR. Keep
    canonical param_names aligned so native FQN-keyed optimizer checkpoints
    and scheduler base rates describe the same groups. Weight decay is unchanged.
    A DistMuon instance cannot split; ``moeloop.optim`` gives core experts their own.
    """
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("expert_lr_multiplier must be finite and positive")
    if multiplier == 1:
        return
    expert_ids = {id(parameter) for part in model_parts for cell in part.core.core
                  if cell.experts is not None for parameter in cell.experts.parameters()}
    for optimizer in optimizers:
        if optimizer.state:
            raise ValueError("expert learning rates must be configured before optimizer state is created")
        additions = []
        for group in optimizer.param_groups:
            params = group["params"]
            selected = [i for i, parameter in enumerate(params) if id(parameter) in expert_ids]
            if not selected:
                continue
            if len(selected) == len(params):
                group["lr"] *= multiplier
                continue
            remaining = [i for i, parameter in enumerate(params) if id(parameter) not in expert_ids]
            expert_group = {**group, "params": [params[i] for i in selected], "lr": group["lr"] * multiplier}
            group["params"] = [params[i] for i in remaining]
            if "param_names" in group:
                names = group["param_names"]
                expert_group["param_names"] = [names[i] for i in selected]
                group["param_names"] = [names[i] for i in remaining]
            additions.append(expert_group)
        for group in additions:
            optimizer.add_param_group(group)


def configure_optimizer(optimizers, model_parts, parallel_dims) -> None:
    scale_core_expert_lr(optimizers, model_parts, multiplier=model_parts[0].config.expert_lr_multiplier)


def ema_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """An fp32 copy of every parameter, keyed by its checkpoint FQN (``canonical_fqn``).

    Under FSDP2 each copy is a DTensor with its parameter's mesh and placement, so the
    EMA costs one sharded fp32 copy of the model per rank and its update communicates
    nothing; DCP saves and reshards it like the model.
    """
    return {canonical_fqn(name): parameter.detach().to(torch.float32, copy=True)
            for name, parameter in model.named_parameters()}


@torch.no_grad()
def update_ema(ema: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    """``ema = decay * ema + (1 - decay) * parameter`` on each local shard (``torch.lerp``)."""
    for name, parameter in model.named_parameters():
        ema[canonical_fqn(name)].lerp_(parameter, 1 - decay)


@torch.no_grad()
def swap_ema(model: nn.Module, ema: dict[str, torch.Tensor]) -> None:
    """Exchange every parameter with its EMA copy in place; a second call swaps back.

    FSDP2 keeps the root's parameters gathered after an eval forward, so every FSDP
    module is resharded first: the swap reaches the sharded values the next forward
    gathers. The round trip is exact for fp32 parameters (TorchTitan training.dtype).
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()
    for name, parameter in model.named_parameters():
        average = ema[canonical_fqn(name)]
        held = parameter.detach().clone()
        parameter.copy_(average)
        average.copy_(held)


class TitanModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        core: ModelConfig = field(default_factory=lambda: ModelConfig(vocab_size=50257))
        native_experts: bool = True
        flex_attention: bool = True
        expert_lr_multiplier: float = 1.0
        dist_muon: bool = False
        """Keep expert banks Shard(0) for any FSDP size: pinned DistMuon cannot
        reshard S(1) storage, and FSDP2 pads an uneven expert dimension."""

        def update_from_config(self, *, config, **kwargs):
            if config.training.seq_len > self.core.max_seq_len:
                raise ValueError("training.seq_len exceeds model max_seq_len")
            config.recipe.validate_model(self.core)
            manifest = training_manifest(config.dataloader.manifest_path)
            for relative, digest in manifest.tokenizer.get("files", {}).items():
                asset = Path(config.hf_assets_path) / Path(relative).name
                if not asset.is_file() or file_sha256(asset) != digest:
                    raise ValueError(f"hf_assets_path tokenizer differs from the prepared manifest: {asset.name}")
            if manifest.tokenizer["vocab_size"] != self.core.vocab_size:
                raise ValueError("prepared tokenizer vocab_size differs from model vocabulary")
            if config.training.seq_len % config.loss.num_chunks:
                raise ValueError("training.seq_len must be divisible by loss.num_chunks")

        def get_nparams_and_flops(self, model, seq_len):
            from .costs import estimate_cost
            cost = estimate_cost(self.core, seq_len=seq_len)
            # Titan's shape hook has no training recipe or AC policy. This is
            # nominal fixed-depth/full-BPTT matrix work; LoopMetrics omits MFU.
            # Formal comparisons pass the complete RecipeSpec to estimate_cost.
            return sum(p.numel() for p in model.parameters()), cost["flops"]["training_per_token"]

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.core = RecursiveLM(config.core, expert_factory=NativeExperts if config.native_experts else None, attention_factory=NativeAttention if config.flex_attention else None)
        self._skip_lm_head = False

    @property
    def lm_head(self):
        return self.core.lm_head

    def forward(self, tokens, **kwargs):
        return self.core(tokens, return_hidden=self._skip_lm_head, **kwargs)

    def verify_module_protocol(self):
        # This is the explicit plain-PyTorch boundary. Native Module recursion
        # does not initialize ordinary Linear/Norm children; init_states below does.
        if not isinstance(self.core, RecursiveLM):
            raise TypeError("TitanModel requires the plain RecursiveLM core")
        owners = [id(p) for _, p in self.named_parameters(remove_duplicate=False)]
        if len(owners) != len(set(owners)):
            raise ValueError("parameter aliases are unsupported; each parameter needs one physical owner")

    def init_states(self, *, buffer_device=None):
        # The run seed: TorchTitan's set_determinism applies debug.seed with
        # torch.manual_seed on every rank (LoopTrainer's host RNG reads it back the same
        # way). Each parameter is drawn whole on CPU and this rank keeps its shard, so
        # step-0 weights depend on the seed alone, not on rank count or GPU model.
        self.core.initialize_weights(torch.Generator().manual_seed(torch.initial_seed()))
        # RoPE is computed from positions in the core; no persistent lookup table.
        for name, buffer in self.named_buffers():
            if buffer.is_meta:
                raise RuntimeError(f"unmaterialized model buffer: {name}")


def model_spec(config: ModelConfig, *, native_experts=True, flex_attention=True, dist_muon=False) -> ModelSpec:
    return ModelSpec(name="moeloop", flavor="recursive_sparse", model=TitanModel.Config(core=config, native_experts=native_experts, flex_attention=flex_attention, dist_muon=dist_muon), parallelize_fn=parallelize_model, pipelining_fn=None, post_optimizer_build_fn=configure_optimizer, state_dict_adapter=None)


def parallelize_model(model, *, parallel_dims, training, parallelism, compile_config, ac_config, dump_folder):
    from torchtitan.distributed.activation_checkpoint import SelectiveAC
    if model.core.config.attention_per_visit and isinstance(ac_config, SelectiveAC.Config):
        raise ValueError("per-visit attention with nested FSDP supports FullAC or no AC; selective replay is not supported")
    if parallelism.spmd_backend != "partial_dtensor":
        raise ValueError("MoE-Loop v1 uses spmd_backend='partial_dtensor'")
    if any(getattr(parallel_dims, axis) != 1 for axis in ("tp", "ep", "cp", "pp")):
        raise ValueError("MoE-Loop v1 supports DP/FSDP; TP/EP/CP/PP degrees must be 1")
    core = model.core
    # AC wraps each actual owner once. Repeated calls return their own statistics,
    # collected outside replay; no hook mutates router state during recomputation.
    for region in (core.prefix, core.core, core.suffix):
        for index, cell in enumerate(region):
            if ac_config is not None:
                policy = ac_config.build(dump_folder=dump_folder)
                if not hasattr(policy, "_wrap_block"):
                    raise ValueError("only full/selective AC is supported by this adapter")
                cell = policy._wrap_block(cell, base_fqn=f"cell.{index}")
                region[index] = cell
            if compile_config.enable and "model" in compile_config.components:
                torch._dynamo.config.capture_scalar_outputs = True
                cell.compile(backend=compile_config.backend, fullgraph=True)
    names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    mesh = parallel_dims.get_mesh(names)
    kwargs = dict(mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param], reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce], cast_forward_inputs=False))
    if training.enable_cpu_offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()
    reshard = get_fsdp_reshard_after_forward_policy(parallelism.fsdp_reshard_after_forward, False)
    fully_shard(core.embedding, **kwargs, reshard_after_forward=reshard)
    fully_shard(core.final_norm, **kwargs, reshard_after_forward=reshard)
    # The native chunked head loss calls this owner through normal module hooks.
    fully_shard(core.lm_head, **kwargs, reshard_after_forward=True)
    for region in (core.prefix, core.core, core.suffix):
        for cell in region:
            if len(cell.attentions) > 1:
                for attention in cell.attentions:
                    fully_shard(attention, **kwargs, reshard_after_forward=reshard)
            expert_params = set(cell.experts.parameters()) if hasattr(cell, "experts") and cell.experts is not None else set()
            shard = None
            if expert_params and mesh.size() > cell.config.num_experts and not model.config.dist_muon:
                shard = lambda parameter, owners=expert_params: Shard(1) if parameter in owners else None
            fully_shard(cell, **kwargs, reshard_after_forward=reshard, **({"shard_placement_fn": shard} if shard else {}))
    fully_shard(model, **kwargs)
    disable_fsdp_gradient_division(model)
    return model


def rank_blocks(config, *, dp_world_size, local_batch_size):
    """Contiguous blocks one rank loads per round of the logical stream.

    Rank r loads the logical microbatches [r*A, (r+1)*A) of each update,
    A = accumulation steps; local_batch_size blocks form one microbatch.
    """
    update_blocks = config.global_batch_size if config.global_batch_size > 0 else local_batch_size * dp_world_size
    return update_blocks // dp_world_size


class PackedDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        manifest_path: str = "data/prepared/manifest.json"
        split: str = "train"
        source_weights: dict[str, float] | None = None
        shuffle: bool = True
        infinite: bool = True
        seed: int = 0
        global_batch_size: int = -1
        """Blocks of one optimizer update; -1 means one local batch per rank, as in
        TorchTitan. Derived: LoopTrainer overwrites it with training.global_batch_size,
        for the training and the validation loader."""

    def __init__(self, config: Config, *, dp_rank, dp_world_size, tokenizer, seq_len, local_batch_size, snapshot_every_n_steps=None):
        if config.num_workers != 0:
            raise ValueError("strict continuation currently supports num_workers=0; worker prefetch qualification follows separately")
        share = rank_blocks(config, dp_world_size=dp_world_size, local_batch_size=local_batch_size)
        dataset = PackedDataset(training_manifest(config.manifest_path), split=config.split, seq_len=seq_len, source_weights=config.source_weights, shuffle=config.shuffle, infinite=config.infinite, seed=config.seed, rank=dp_rank, world_size=dp_world_size, rank_blocks=share)
        self.config = config
        generator = torch.Generator().manual_seed(config.seed + dp_rank)
        super().__init__(dataset, dp_rank, dp_world_size, batch_size=local_batch_size, num_workers=0, pin_memory=config.pin_memory, drop_last=True, snapshot_every_n_steps=1, generator=generator)

    def load_state_dict(self, state_dict):
        if self._rank_id not in state_dict:
            raise ValueError(f"checkpoint is missing data cursor for {self._rank_id}")
        super().load_state_dict(state_dict)


def prepare_batch(input_dict, labels, *, flex_attention):
    inputs = input_dict["input"]
    positions = input_dict["positions"]
    document_ids = input_dict["document_ids"]
    valid = input_dict["input_mask"]
    if flex_attention:
        def allowed(batch, head, query, key):
            # create_block_mask may evaluate padded block coordinates.
            length = document_ids.shape[1]
            q = query.clamp(max=length - 1)
            k = key.clamp(max=length - 1)
            return (query < length) & (key < length) & (query >= key) & valid[batch, q] & valid[batch, k] & (document_ids[batch, q] == document_ids[batch, k])
        mask = create_block_mask(allowed, B=inputs.shape[0], H=None, Q_LEN=inputs.shape[1], KV_LEN=inputs.shape[1], device=str(inputs.device), _compile=True)
    else:
        mask = build_document_mask(document_ids, valid)
    return inputs, labels, {"positions": positions, "attention_mask": mask, "router_mask": valid}


class LoopChunkedLoss(ChunkedLossWrapper):
    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        recipe: TrainingRecipe = field(default_factory=TrainingRecipe)

    def __init__(self, config, *, compile_config=None):
        super().__init__(config, compile_config=compile_config)
        self.recipe = config.recipe
        self.dp_degree = 1
        self.accumulation = 1

    def __call__(self, output, labels, global_valid_tokens=None, **kwargs):
        if self.recipe.objective == "expected_ce" and global_valid_tokens is not None:
            return objective(output, labels, global_valid_tokens, self.recipe, dp_degree=self.dp_degree,
                             accumulation=self.accumulation, head=self.lm_head, num_chunks=self.num_chunks)
        # Native inner CE does the global-label normalization before its own
        # backward. Its bridge ignores grad_output: never rescale the returned CE.
        ce, metrics = super().__call__(output.hidden_states, labels, global_valid_tokens, **kwargs)
        loss, values = objective(output, labels, global_valid_tokens, self.recipe, dp_degree=self.dp_degree,
                                 accumulation=self.accumulation, ce=ce)
        return loss, {**metrics, **values}


class LoopMetrics(MetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        pass

    def __init__(self, config, *, dump_folder="./outputs", **kwargs):
        super().__init__(config, dump_folder=dump_folder, **kwargs)
        self.output_directory = Path(dump_folder)
        self.last_training = {}
        self.last_validation = {}
        self.ema_pass = False

    def log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
        elapsed = time.perf_counter() - self.time_last_log
        values = {"loss_metrics/global_avg_ce": global_avg_loss, "loss_metrics/global_max_ce": global_max_loss, "grad_norm": grad_norm, "throughput/padded_positions_per_second_per_rank": self.ntokens_since_last_log / max(elapsed, 1e-9), **(extra_metrics or {})}
        peak = self.device_memory_monitor.get_peak_stats()
        self.last_training = {"grad_norm": float(grad_norm), "padded_positions_per_second_per_rank": values["throughput/padded_positions_per_second_per_rank"], "peak_active_gib": peak.max_active_gib, "peak_reserved_gib": peak.max_reserved_gib}
        self.logger.log(values, step)
        logger.info("step=%s CE=%.6f grad_norm=%.4f padded_positions/s/rank=%.1f", step, global_avg_loss, grad_norm, values["throughput/padded_positions_per_second_per_rank"])
        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def log_validation(self, loss, step, extra_metrics=None):
        """The native validation loop reports its CE here. LoopValidator's pass on the EMA
        weights (``ema_pass``) keeps it as ``ema_ce`` and logs ``validation_metrics/ema_loss``,
        so the model's own validation log is unchanged."""
        if self.ema_pass:
            self.last_validation["ema_ce"] = float(loss)
            self.logger.log({"validation_metrics/ema_loss": loss}, step)
            self.ntokens_since_last_log, self.time_last_log = 0, time.perf_counter()  # as the native log
            return
        self.last_validation = {"step": step, "ce": float(loss)}
        return super().log_validation(loss, step, extra_metrics)


class LoopValidator(Validator):
    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        global_batches: int = 2
        """Global batches per validation. Ranks split each one as they split a training
        update (the loader's global_batch_size, set by LoopTrainer), so per-rank steps are
        global_batches x accumulation and every world size validates the same blocks."""
        target_label_tokens: int | None = None
        """Minimum global labels, rounded up to whole global batches.

        When set, this takes precedence over global_batches. Native ``steps`` is derived
        from one of the two and must stay unset (-1). Actual labels are checked and
        recorded; the dataloader's infinite flag is always respected.
        """

        def __post_init__(self):
            Validator.Config.__post_init__(self)
            if type(self.global_batches) is not int or self.global_batches <= 0:
                raise ValueError("global_batches must be a positive integer")
            if self.target_label_tokens is not None and (type(self.target_label_tokens) is not int or self.target_label_tokens <= 0):
                raise ValueError("target_label_tokens must be a positive integer or None")

    def __init__(self, config, **kwargs):
        if config.steps != -1:
            raise ValueError("validator.steps is derived per rank; set validator.global_batches "
                             "(or target_label_tokens) instead")
        world, local, seq_len = kwargs["dp_world_size"], kwargs["local_batch_size"], kwargs["seq_len"]
        self.global_batch_size = config.dataloader.global_batch_size if config.dataloader.global_batch_size > 0 else world * local
        if self.global_batch_size % (world * local):
            raise ValueError(f"validation global batch {self.global_batch_size} must be a multiple of "
                             f"world size {world} x local batch {local}")
        self.global_batches = (config.global_batches if config.target_label_tokens is None
                               else math.ceil(config.target_label_tokens / (self.global_batch_size * seq_len)))
        config = replace(config, steps=self.global_batches * self.global_batch_size // (world * local))
        super().__init__(config, **kwargs)
        # The base loop calls self.loss_fn on each batch; this method also sums BPB terms.
        self.ce_loss = self.loss_fn
        self.loss_fn = self.loss_and_text_ce
        # The trainer's EMA weights (``ema_weights``) when its config sets ema_decay.
        self.ema = None
        # The pinned base class forces infinite=True for every fixed-step run.
        # Keep the actual requested loader policy rather than inheriting that implicit wrap.
        self.dl_config = replace(config.dataloader)
        self.finite_batch_capacity = None
        if not self.dl_config.infinite:
            self.finite_batch_capacity = self._finite_capacity()
            if config.steps > min(self.finite_batch_capacity):
                raise ValueError(f"finite validation capacity is insufficient: requested {config.steps} steps, available per rank {self.finite_batch_capacity}; reduce the budget or prepare more heldout data")

    def _finite_capacity(self):
        """Check every rank before any rank can exit an FSDP validation loop.

        Only document offsets are read: each source packs its labels into
        ceil(labels / seq_len) blocks of the logical stream. Rank r takes blocks
        [r*K, (r+1)*K) of every round of world_size*K blocks (K = rank_blocks);
        the loader drops a final partial batch. Every process obtains the same
        capacities without an EOF collective.
        """
        data = self.dl_config
        dataset = PackedDataset(training_manifest(data.manifest_path), split=data.split,
            seq_len=self.seq_len, source_weights=data.source_weights, shuffle=data.shuffle,
            infinite=False, seed=data.seed)
        blocks = 0
        try:
            for source in dataset.sources.values():
                labels = 0
                for document in source.order:
                    _, begin, end = source._bounds(document)
                    labels += end - begin - 1
                blocks += (labels + self.seq_len - 1) // self.seq_len
        finally:
            dataset.close()
        share = rank_blocks(data, dp_world_size=self.dp_world_size, local_batch_size=self.local_batch_size)
        rounds, rest = divmod(blocks, self.dp_world_size * share)
        return [(rounds * share + min(max(rest - rank * share, 0), share)) // self.local_batch_size
                for rank in range(self.dp_world_size)]

    def post_dataloading_process(self, input_dict, labels, model_parts):
        self.observed_labels += int((labels != -100).sum().item())
        self.observed_steps += 1
        return prepare_batch(input_dict, labels, flex_attention=model_parts[0].config.flex_attention)

    def loss_and_text_ce(self, predictions, labels):
        """Return the base CE sum; with a byte table, also sum CE and bytes of text labels (``bpb_terms``).

        The native loss returns only a chunked sum, so a second loss call over only the
        text labels keeps the reported CE bit-identical, at the cost of one more chunked
        LM head per validation batch.
        """
        loss_sum, metrics = self.ce_loss(predictions, labels)
        if self.token_bytes is not None:
            label_bytes, text = bpb_terms(labels, self.token_bytes)
            self.ce_sum_text += self.ce_loss(predictions, labels.masked_fill(~text, -100))[0].double()
            self.label_bytes += label_bytes.sum()
        return loss_sum, metrics

    def validate(self, model_parts, step):
        """Run the native validation loop and append one record to ``validation.jsonl``.

        Bits per byte use ``moeloop.evaluate``'s ``token_byte_tensor``, ``bpb_terms`` and
        ``bpb_fields``, as ``evaluate_heldout`` does: labels whose token has 0 bytes (EOS,
        PAD, special and added tokens) leave both ``ce_sum_text`` and ``label_bytes``,
        ``bpb = ce_sum_text / (ln 2 * label_bytes)``, and ``byte_table_sha256`` hashes
        the tokenizer's unpadded table. Both terms are summed over ranks like
        ``valid_tokens``, so BPB does not depend on the rank count. Without a byte
        table the four fields are None and ``bpb_unavailable`` says why. BPB is also
        logged as ``validation_metrics/bpb``.

        With EMA weights (``self.ema``) the same blocks are validated again on them
        (``swap_ema``, then back): the record gains ``ema``, their ``ce`` and the five
        BPB fields, and their BPB is logged as ``validation_metrics/ema_bpb``.
        """
        modes = [model.training for model in model_parts]
        manifest = training_manifest(self.dl_config.manifest_path)
        try:
            result, count, fields = self._native_pass(model_parts, step, manifest)
            record = {**self.metrics_processor.last_validation, "valid_tokens": count,
                      "manifest_fingerprint": manifest.fingerprint,
                      "num_visits": model_parts[0].core.config.num_visits,
                      "target_label_tokens": self.config.target_label_tokens,
                      "global_batches": self.global_batches, "global_batch_size": self.global_batch_size,
                      "steps": self.observed_steps, "configured_steps": self.config.steps,
                      "world_size": self.dp_world_size, "local_batch_size": self.local_batch_size,
                      "seq_len": self.seq_len, "infinite": self.dl_config.infinite,
                      "finite_batch_capacity_per_rank": self.finite_batch_capacity, **fields}
            if fields["bpb"] is not None:
                self.metrics_processor.logger.log({"validation_metrics/bpb": fields["bpb"]}, step)
            if self.ema is not None:
                swap_ema(model_parts[0], self.ema)
                self.metrics_processor.ema_pass = True
                try:
                    _, _, fields = self._native_pass(model_parts, step, manifest)
                finally:
                    self.metrics_processor.ema_pass = False
                    swap_ema(model_parts[0], self.ema)
                record["ema"] = {"ce": self.metrics_processor.last_validation.pop("ema_ce"), **fields}
                if fields["bpb"] is not None:
                    self.metrics_processor.logger.log({"validation_metrics/ema_bpb": fields["bpb"]}, step)
            if not dist.is_initialized() or dist.get_rank() == 0:
                path = self.metrics_processor.output_directory / "validation.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            if self.config.target_label_tokens is not None and count < self.config.target_label_tokens:
                raise ValueError(f"finite validation observed {count} labels below target_label_tokens={self.config.target_label_tokens}; labels are never inferred from padded positions")
            return result
        finally:
            for model, training in zip(model_parts, modes):
                model.train(training)

    def _native_pass(self, model_parts, step, manifest):
        """One native validation loop: its result, the global label count and ``bpb_fields``."""
        self.observed_labels = 0
        self.observed_steps = 0
        # The base loop puts labels and outputs on this device, also under CPU offload.
        device = torch.device(utils.device_type)
        self.token_bytes, byte_table, bpb_unavailable = token_byte_tensor(manifest, device)
        self.ce_sum_text = torch.zeros((), dtype=torch.float64, device=device)
        self.label_bytes = torch.zeros((), dtype=torch.int64, device=device)
        with isolated_rng():
            result = super().validate(model_parts, step)
        count = torch.tensor(self.observed_labels, dtype=torch.int64, device=next(model_parts[0].parameters()).device)
        if dist.is_initialized():
            for total in (count, self.label_bytes, self.ce_sum_text):
                dist.all_reduce(total)
        fields = bpb_fields(byte_table, bpb_unavailable, float(self.ce_sum_text.item()), int(self.label_bytes.item()))
        return result, int(count.item()), fields


SCHEDULE_RUN_FIELDS = ("successful_updates", "schedule_total_steps", "schedule", "fork")


def lineage_free(identity):
    """A continuation identity without its LR schedule and lineage record.

    A fork shares this with its parent. A spec-built run also carries its horizon,
    schedule and fork record in ``recipe_metadata["run"]``; those fields go too.
    """
    shared = {key: value for key, value in identity.items() if key not in ("lr_scheduler", "fork")}
    metadata = identity["recipe_metadata"]
    if "run" in metadata:
        run = {key: value for key, value in metadata["run"].items() if key not in SCHEDULE_RUN_FIELDS}
        shared["recipe_metadata"] = {**metadata, "run": run}
    return shared


# Continuation-identity fields an untie may change: those that follow from the spec's model and
# depth, which check_untied compares as the spec, and the moeloop source, which gained the untie code.
# Of the optimizer only the factory kwargs go: Muon's per-parameter layouts and per-cell buckets name
# the core cells; its settings stay in param_groups.
UNTIED_FIELDS = ("model", "recipe", "loss", "recipe_metadata", "moeloop_source")


def schedule_factors(config, count):
    """The pinned native LR factors of update indices 0..count-1 under one schedule config.

    The native builder exposes its factor function only through a LambdaLR, so it is
    built here on a throwaway unit-LR optimizer.
    """
    optimizer = torch.optim.SGD([torch.zeros(())], lr=1.0)
    factor = config.build(optimizers=[optimizer], training_steps=config.total_steps).schedulers[0].lr_lambdas[0]
    return [factor(index) for index in range(count)]


class LoopTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        recipe: TrainingRecipe = field(default_factory=TrainingRecipe)
        recipe_metadata: dict[str, Any] = field(default_factory=dict)
        expert_lr_multiplier: float = 1.0
        fork: dict[str, Any] = field(default_factory=dict)
        """A new lineage from a parent's full checkpoint: ``checkpoint`` (path),
        ``metadata_sha256`` (digest of its DCP .metadata) and ``step`` (S). It enters
        the continuation identity; see LoopTrainer.check_fork. Empty for ordinary runs."""
        ema_decay: float | None = None
        """Keep an fp32 EMA of the parameters with this decay, updated after every optimizer
        update (``update_ema``), saved in every checkpoint, validated beside the model and
        exportable (``moeloop.export --weights ema``). It enters the continuation identity.
        None keeps no EMA."""

        def __post_init__(self):
            Trainer.Config.__post_init__(self)
            if not math.isfinite(self.expert_lr_multiplier) or self.expert_lr_multiplier <= 0:
                raise ValueError("expert_lr_multiplier must be finite and positive")
            if self.ema_decay is not None and not 0 < self.ema_decay < 1:
                raise ValueError("ema_decay must lie in (0, 1), or be None for no EMA")
            if self.parallelism.spmd_backend != "partial_dtensor":
                raise ValueError("select partial_dtensor for the plain model core")
            if not self.training.disable_cuda_graphs:
                raise ValueError("disable CUDA graphs for the v1 dynamic/masked training path")
            if self.checkpoint.last_save_model_only or self.checkpoint.last_save_in_hf:
                raise ValueError("the final training checkpoint must retain full state; HF export is separate")
            if self.lr_scheduler.total_steps is None:
                raise ValueError("set lr_scheduler.total_steps explicitly so continuation keeps the schedule horizon")
            if not 0 < self.training.steps <= self.lr_scheduler.total_steps:
                raise ValueError("training.steps must be positive and no larger than the schedule horizon")
            if self.checkpoint.exclude_from_loading:
                raise ValueError("strict continuation cannot exclude optimizer/scheduler/data/training state")

    def __init__(self, config):
        if not isinstance(config.loss, LoopChunkedLoss.Config):
            raise TypeError("LoopTrainer requires the native chunked CE and explicit auxiliary loss config")
        config.loss = replace(config.loss, recipe=config.recipe)
        config.model_spec.model.expert_lr_multiplier = config.expert_lr_multiplier
        config.dump_folder = str(Path(config.dump_folder).resolve())
        config.hf_assets_path = str(Path(config.hf_assets_path).resolve())
        manifest_path = Path(config.dataloader.manifest_path).resolve()
        config.dataloader.manifest_path = str(manifest_path / "manifest.json" if manifest_path.is_dir() else manifest_path)
        config.dataloader.global_batch_size = config.training.global_batch_size
        config.validator.dataloader.global_batch_size = config.training.global_batch_size
        self.progress = CommittedProgress()
        self.active_depth = config.recipe.num_visits
        self.active_microbatch = 0
        self.update_depths = []
        self.pending_counts = dict(padded_positions=0, input_tokens=0, label_tokens=0, recursive_input_tokens=0)
        self.update_metrics = []
        self.ema = None
        super().__init__(config)
        if not isinstance(self.loss_fn, LoopChunkedLoss):
            raise TypeError("LoopTrainer requires LoopChunkedLoss for explicit hidden/aux handling")
        self.loss_fn.recipe = config.recipe
        # dp_degree * accumulation is the number of logical microbatches per update.
        self.loss_fn.dp_degree = self.parallel_dims.get_mesh("batch").size() if self.parallel_dims.dp_enabled else 1
        self.loss_fn.accumulation = self.gradient_accumulation_steps
        microbatches = self.loss_fn.dp_degree * self.gradient_accumulation_steps
        self.rank = dist.get_rank()
        self.host_rng_seed = seed_host_rng(torch.initial_seed(), rank=self.rank)
        manifest = training_manifest(config.dataloader.manifest_path)
        self.identity = {
            **spec_identity(model=config.model_spec.model.core, recipe=config.recipe,
                            recipe_metadata=config.recipe_metadata,
                            expert_lr_multiplier=config.expert_lr_multiplier, fork=config.fork,
                            ema_decay=config.ema_decay),
            "schema": 1, "torchtitan_commit": TORCHTITAN_COMMIT,
            "debug_seed": config.debug.seed, "host_rng_policy": "torch_initial_seed_plus_global_rank_v1",
            "moeloop_source": {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
            "manifest": manifest.fingerprint, "dataloader": config.dataloader.to_dict(),
            "data_stream": {"rule": STREAM_RULE, "update_blocks": microbatches * config.training.local_batch_size,
                            "microbatch_blocks": config.training.local_batch_size},
            "parallelism": asdict(config.parallelism),
            "world_size": dist.get_world_size(), "local_batch_size": config.training.local_batch_size,
            "accumulation": self.gradient_accumulation_steps, "seq_len": config.training.seq_len,
            "optimizer": config.optimizer.to_dict(), "lr_scheduler": config.lr_scheduler.to_dict(),
            "loss": config.loss.to_dict(),
            "precision": [config.training.dtype, config.training.mixed_precision_param, config.training.mixed_precision_reduce],
            "max_norm": config.training.max_norm,
            "execution": {"native_experts": self.model_config.native_experts, "flex_attention": self.model_config.flex_attention, "compile": asdict(config.compile), "activation_checkpoint": config.activation_checkpoint.to_dict() if config.activation_checkpoint is not None else None},
        }
        self.identity_digest = continuation_fingerprint(self.identity)
        topology = {"rank": self.rank, "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
                    "hostname": socket.gethostname(), "cluster_node_id": os.environ.get("CLUSTER_NODE_ID")}
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, topology)
        if self.rank == 0:
            Path(config.dump_folder).mkdir(parents=True, exist_ok=True)
            self.attempt_directory = Path(config.dump_folder) / "attempts" / f"{time.time_ns()}-{os.getpid()}"
            self.attempt_directory.mkdir(parents=True)
            self._write_json("model_config.json", config.model_spec.model.core.to_dict())
            self._write_json("resolved_config.json", config.to_dict())
            self._write_json("continuation_identity.json", self.identity)
            topology = {"world_size": dist.get_world_size(), "ranks": ranks,
                        "continuation_identity_sha256": file_sha256(Path(config.dump_folder) / "continuation_identity.json")}
            self._write_json("native_topology.json", topology)
            logger.info("native_topology=%s", json.dumps(topology, sort_keys=True))
            if config.recipe_metadata:
                self._write_json("recipe.json", config.recipe_metadata)
        if config.ema_decay is not None:
            # Starts at the initialized weights; a resumed run loads it with the checkpoint.
            self.ema = ema_weights(self.model_parts[0])
            if config.validator.enable:
                self.validator.ema = self.ema

    def _write_json(self, filename, value):
        path = Path(self.config.dump_folder) / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        # Keep attempts when a resumed run fails or resolves a new stop target.
        if hasattr(self, "attempt_directory"):
            (self.attempt_directory / filename).write_text(path.read_text())

    def post_dataloading_process(self, input_dict, labels):
        inputs, labels, kwargs = prepare_batch(input_dict, labels, flex_attention=self.model_config.flex_attention)
        if self.config.recipe.depth_rule == "poisson_lognormal_micro_v1":
            # FSDP needs one depth per accumulation step on all ranks, so this rule
            # follows rank 0's microbatch index and depends on the number of ranks.
            self.active_depth = self.config.recipe.depth_at(self.progress.completed_updates, self.active_microbatch)
            depth = torch.tensor(self.active_depth, device=self.device, dtype=torch.int64)
            if dist.is_initialized():
                dist.broadcast(depth, src=0)
            self.active_depth = int(depth.item())
        self.update_depths.append(self.active_depth)
        self.active_microbatch += 1
        gradients = self.config.recipe.grad_visits
        kwargs.update(num_visits=self.active_depth, grad_visits=None if gradients is None else min(gradients, self.active_depth))
        self.ntokens_seen += labels.numel()
        valid = int(input_dict["input_mask"].sum().item())
        self.pending_counts["padded_positions"] += labels.numel()
        self.pending_counts["input_tokens"] += valid
        self.pending_counts["label_tokens"] += int((labels != -100).sum().item())
        cfg = self.model_config.core
        suffix_calls = cfg.suffix_cells * (self.active_depth if self.config.recipe.objective == "expected_ce" else 1)
        self.pending_counts["recursive_input_tokens"] += valid * (cfg.prefix_cells + suffix_calls + cfg.num_cells * self.active_depth)
        return inputs, labels, kwargs

    def _forward_backward_body(self, inputs, labels, global_valid_tokens, extra_kwargs):
        if global_valid_tokens.item() <= 0:
            raise ValueError("an optimizer update has zero globally valid labels")
        with self.train_context():
            output = self.model_parts[0](inputs, **extra_kwargs)
            loss, metrics = self.loss_fn(output, labels, global_valid_tokens)
            finite = torch.isfinite(loss.detach()).to(torch.int32)
            if dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f"nonfinite objective at attempted update {self.step}")
            counts = output.router_counts.detach()
            self.update_metrics.append({**metrics, "training_objective": loss.detach(),
                                        "router_counts": counts, "aux_calls": output.aux_calls})
            with spmd.no_typecheck():
                loss.backward()
        # Native Trainer logs/normalizes this as CE. Aux has its own scope.
        return metrics["ce"]

    def train_step(self, data_iterator):
        step_started = time.perf_counter()
        self.active_depth = self.config.recipe.depth_at(self.progress.completed_updates)
        self.active_microbatch = 0
        self.update_depths = []
        self.pending_counts = dict.fromkeys(self.pending_counts, 0)
        self.update_metrics = []
        # Native log() samples norm/throughput/memory only on logging steps.
        self.metrics_processor.last_training = {}
        super().train_step(data_iterator)
        if self.ema is not None:
            update_ema(self.ema, self.model_parts[0], self.config.ema_decay)
        self.progress.commit(**self.pending_counts)
        if self.step != self.progress.completed_updates:
            raise RuntimeError("native step differs from actually completed optimizer updates")
        global_progress, global_sources = self._global_observations()
        metrics = {"step": self.progress.completed_updates, "depth": self.active_depth, "grad_visits": min(self.config.recipe.grad_visits or self.active_depth, self.active_depth), "local_rank": self.rank, "local_progress": asdict(self.progress), "global_progress": global_progress, "step_elapsed_seconds": time.perf_counter() - step_started, **self.metrics_processor.last_training}
        metrics["microbatch_depths"] = self.update_depths
        metrics["microbatch_grad_visits"] = [min(self.config.recipe.grad_visits or d, d) for d in self.update_depths]
        for key in ("ce", "aux", "training_objective", "expected_ce", "entropy", "visit_ce", "mean_exit_probabilities"):
            if not any(key in item for item in self.update_metrics):
                continue
            value = sum((item[key].detach() for item in self.update_metrics if key in item), torch.zeros((), device=self.device))
            if dist.is_initialized():
                dist.all_reduce(value)
            metrics[key] = value.cpu().tolist() if value.ndim else float(value.item())
        metrics["aux_reduction"] = self.config.recipe.aux_reduction
        dataset = getattr(self.dataloader, "dataset", None)
        metrics["local_source_counts"] = getattr(dataset, "source_counts", {})
        metrics["global_source_counts"] = global_sources
        router_counts = sum((item["router_counts"] for item in self.update_metrics), torch.zeros_like(self.update_metrics[0]["router_counts"]))
        if router_counts.numel() and dist.is_initialized():
            dist.all_reduce(router_counts)
        metrics["global_router_assignments"] = router_counts.cpu().tolist()
        metrics["differentiated_router_calls_per_micro"] = [item["aux_calls"] for item in self.update_metrics]
        if self.rank == 0:
            with (Path(self.config.dump_folder) / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")

    def _global_observations(self):
        progress = asdict(self.progress)
        names = [name for name in progress if name != "completed_updates"]
        counts = torch.tensor([progress[name] for name in names], dtype=torch.int64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(counts)
        global_progress = {"completed_updates": self.progress.completed_updates, **dict(zip(names, counts.cpu().tolist()))}
        local_sources = getattr(getattr(self.dataloader, "dataset", None), "source_counts", {})
        source_ranks = [local_sources]
        if dist.is_initialized():
            source_ranks = [None] * dist.get_world_size()
            dist.all_gather_object(source_ranks, local_sources)
        global_sources = {}
        for sources in source_ranks:
            for source, fields in sources.items():
                totals = global_sources.setdefault(source, {})
                for name, count in fields.items():
                    totals[name] = totals.get(name, 0) + count
        return global_progress, global_sources

    def state_dict(self):
        rank = dist.get_rank() if dist.is_initialized() else getattr(self, "rank", 0)
        payload = {"schema": 1, "identity": self.identity, "fingerprint": self.identity_digest, "progress": asdict(self.progress), "rng": capture_rng()}
        state = {"step": self.step, "ntokens_seen": self.ntokens_seen, f"rank_{rank}": pickle.dumps(payload, protocol=5)}
        if self.ema is not None:  # DCP saves it as train_state.ema.<FQN> and loads it in place
            state["ema"] = self.ema
        return state

    def load_state_dict(self, state_dict):
        key = f"rank_{self.rank}"
        if key not in state_dict:
            raise ValueError(f"checkpoint lacks rank-qualified continuation state: {key}")
        payload = pickle.loads(state_dict[key])
        if payload["schema"] != 1:
            raise ValueError("unsupported continuation state schema")
        if payload["fingerprint"] != self.identity_digest:
            untie = self.identity.get("recipe_metadata", {}).get("run", {}).get("untie")
            if untie and not self.identity["fork"]:
                self.check_untie(payload["identity"], state_dict["step"])
            else:
                self.check_fork(payload["identity"], state_dict["step"])
        self.progress = CommittedProgress(**payload["progress"])
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]
        if self.step != self.progress.completed_updates:
            raise ValueError("checkpoint step does not represent completed optimizer updates")
        restore_rng(payload["rng"])

    def check_fork(self, parent, step):
        """Accept a parent's full state at the declared fork step S as this lineage's start.

        The two continuation identities must agree except for the LR schedule and the
        lineage record, and the new schedule must give the parent's LR factor at every
        index 0..S. Updates 1..S used indices 0..S-1; update S+1 still runs at the LR
        the restored optimizer carries, the parent's index S, because the native
        scheduler writes a new LR only after an update. In practice the fork step lies
        in the parent's stable phase, or in its warmup with a tail starting after it.
        """
        fork = self.identity["fork"]
        if not fork or step != fork["step"]:
            raise ValueError("strict continuation identity changed; use a new run for changed model/data/topology/optimizer/recipe")
        if lineage_free(parent) != lineage_free(self.identity):
            raise ValueError("fork parent differs beyond the LR schedule; a fork changes only the schedule tail")
        parent_schedule = type(self.config.lr_scheduler)(**parent["lr_scheduler"])
        if schedule_factors(parent_schedule, step + 1) != schedule_factors(self.config.lr_scheduler, step + 1):
            raise ValueError(f"fork schedule changes LRs the parent used up to step {step}; fork from a checkpoint before the parent's decay begins")

    def check_untie(self, source, step):
        """Accept a fully shared source's converted full state at the declared step S.

        ``recipe_experiments.py untie`` converted the source's step-S checkpoint
        (``moeloop.untie``). This run's spec must be the source spec's untied form
        (``check_untied``); the identity fields that follow from the spec's model and
        depth go with it (UNTIED_FIELDS, and the optimizer's factory kwargs, where the
        Muon layouts name every core cell). The moeloop source may differ too: sources
        trained before the untie code existed record other file digests; the source
        payload keeps its own. Every other field must be the source's: data, topology,
        optimizer settings, LR schedule, seed and execution.
        """
        untie = self.identity["recipe_metadata"]["run"]["untie"]
        if step != untie["step"] or source["recipe_metadata"].get("name") != untie["source"]:
            raise ValueError(f"an untie starts only from step {untie['step']} of {untie['source']}")
        check_untied(source["recipe_metadata"], self.identity["recipe_metadata"])
        def kept(identity):
            shared = {key: value for key, value in identity.items() if key not in UNTIED_FIELDS}
            shared["optimizer"] = {key: value for key, value in identity["optimizer"].items()
                                   if key != "optimizer_factory_kwargs_by_name"}
            return shared
        if kept(source) != kept(self.identity):
            raise ValueError("untie source differs beyond the untied model; an untie changes only the core cells' owners")

    def train(self):
        try:
            super().train()
            self.checkpointer.maybe_wait_for_saving()
            if self.progress.completed_updates != self.config.training.steps:
                raise RuntimeError(f"training ended after {self.progress.completed_updates} completed updates; requested {self.config.training.steps}")
            if self.config.validator.enable and not self.validator.should_validate(self.step):
                self.validator.validate(self.model_parts, self.step)
            if self.config.checkpoint.enable:
                checkpoint = Path(self.checkpointer.folder) / f"step-{self.progress.completed_updates}"
                if not (checkpoint / ".metadata").is_file():
                    raise RuntimeError(f"final full checkpoint is not durable: {checkpoint}")
            global_progress, global_sources = self._global_observations()
            if self.rank == 0:
                self._write_json("outcome.json", {"status": "completed", "completed_updates": self.progress.completed_updates, "global_progress": global_progress, "global_source_counts": global_sources, "full_checkpoint": str(checkpoint) if self.config.checkpoint.enable else None, "continuation_fingerprint": self.identity_digest})
        except BaseException as error:
            if self.rank == 0:
                self._write_json("outcome.json", {"status": "failed", "completed_updates": self.progress.completed_updates, "error": str(error)})
            raise
