"""Private eight-rank native-controls timing; explicit source/scripts PYTHONPATH.

Both original model forwards and both native FSDP policies remain unchanged.
The public BF16 router precision differs: this is timing, not gradient parity.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.attention.flex_attention import create_block_mask

import recipe_benchmark as bench
from check_release import verify_benchmark_math
from moeloop.recipes import p1_moe_spec
from moeloop.titan import TORCHTITAN_COMMIT, parallelize_model, require_torchtitan_pin
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.utils import clip_grad_norm_, set_spmd_backend
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.models.qwen3.parallelize import parallelize_qwen3

OUTPUT_PATH = None


class NativeRoot(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    @property
    def lm_head(self):
        return self.core.lm_head

    def forward(self, *args, **kwargs):
        return self.core(*args, **kwargs)


def profile(model, batches, settings, accumulation, *, reference):
    optimizer_config = {**bench.OPTIMIZER, "foreach": None, "fused": True}
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    lossfn = bench.make_loss(model, settings)
    denominator = torch.tensor(settings.batch_size * settings.seq_len * 8 * accumulation, device=settings.device)
    router_observation, handles = [], []
    from moeloop.moe import Router
    from torchtitan.models.common.moe import TokenChoiceTopKRouter
    for module in model.modules():
        if isinstance(module, TokenChoiceTopKRouter if reference else Router):
            def capture(module, inputs, output):
                router_observation.append(dict(input_dtype=str(inputs[0].dtype), gate_weight_dtype=str(module.gate.weight.dtype),
                    score_dtype=str(output[2].dtype) if reference else "FP32 explicit F.linear/softmax", natural_routes=True))
            handles.append(module.register_forward_hook(capture))
    durations, losses, norms, safety = [], [], [], {}
    started_warmup = time.perf_counter()
    for index in range(1 + settings.warmup + settings.steps):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=settings.device)
        for batch in batches:
            with bench.autocast(settings):
                hidden = bench.hidden_forward(model, batch, settings, reference=reference)
                loss, _ = lossfn(hidden, batch["labels"], denominator)
            loss.backward()
            total_loss += loss.detach()
        norm = clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        if index == 0:
            safety["all_gradients_present_and_finite"] = all(p.grad is not None and bool(torch.isfinite(p.grad.to_local()).all()) for p in model.parameters())
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if index == 0:
            safety["all_updated_parameters_finite"] = all(bool(torch.isfinite(p.to_local()).all()) for p in model.parameters())
            safety["optimizer_updates"] = min(int(state["step"]) for state in optimizer.state.values())
            for handle in handles:
                handle.remove()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            safety.update(global_ce=float(total_loss), grad_norm=float(norm), router_observation=router_observation)
        if index == settings.warmup:
            torch.cuda.reset_peak_memory_stats()
            warmup_seconds = time.perf_counter() - started_warmup
        if index > settings.warmup:
            durations.append(elapsed)
            losses.append(float(total_loss))
            norms.append(float(norm))
    safe = all(safety.get(key) is True for key in ("all_gradients_present_and_finite", "all_updated_parameters_finite"))
    safe = safe and safety["optimizer_updates"] == 1 and all(torch.isfinite(torch.tensor(losses + norms)))
    return dict(safe=bool(safe), safety=safety, full_step_seconds_raw=durations, local_ce_contributions=losses,
                grad_norms=norms, warmup_seconds=warmup_seconds, optimizer=optimizer_config,
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(), peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    global OUTPUT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("math-report", "hf-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--local-batch", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    require_torchtitan_pin()
    OUTPUT_PATH = args.output
    if int(os.environ.get("WORLD_SIZE", "0")) != 8 or min(args.local_batch, args.accumulation, args.steps) < 1 or args.warmup < 0:
        parser.error("requires eight torchrun ranks and positive batch/accumulation/steps")
    evidence = {}
    for name, path in (("single_gpu", args.math_report), ("hf_fp32", args.hf_report)):
        raw = json.loads(path.read_text())
        evidence[name] = dict(path=str(path), sha256=bench.digest(path), verification=verify_benchmark_math(raw, path),
                              settings=raw["settings"], correctness=raw["correctness"], component_math=raw.get("component_math"))
        if name == "hf_fp32":
            assert raw["settings"]["dtype"] == "float32" and raw["settings"]["reference"] == "transformers" and raw["correctness"]["status"] == "passed"
        else:
            math_report = raw
    config = p1_moe_spec().model
    assert config.to_dict() == math_report["model_config"], "math evidence must cover the P1 standard MoE geometry"
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    settings = bench.Settings(mode="benchmark", batch_size=args.local_batch, seq_len=4096, head_chunks=8,
                              device=str(device), warmup=args.warmup, steps=args.steps, component_math=True)
    descriptor = bench.describe(config, settings)
    for name in ("src/moeloop/model.py", "src/moeloop/moe.py", "scripts/recipe_benchmark.py"):
        assert descriptor["source_sha256"][name] == math_report["source_sha256"][name], "math source changed: " + name
    dist.init_process_group("nccl", device_id=device)
    set_spmd_backend("partial_dtensor")
    mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("fsdp",))
    pdims = SimpleNamespace(tp=1, ep=1, cp=1, pp=1, dp_replicate_enabled=False, tp_enabled=False, ep_enabled=False,
                            cp_enabled=False, pp_enabled=False, get_mesh=lambda names: mesh)
    wrapping = dict(parallel_dims=pdims, training=TrainingConfig(mixed_precision_param="bfloat16", mixed_precision_reduce="float32"),
                    parallelism=ParallelismConfig(spmd_backend="partial_dtensor"), compile_config=CompileConfig(enable=False), ac_config=None, dump_folder="/tmp")
    actual, reference, pairs = bench.make_models(config, settings)
    assert all(torch.equal(pair.actual, pair.reference_value()) for pair in pairs), "initial weights differ"
    weight_hash = hashlib.sha256()
    for pair in pairs:
        weight_hash.update(pair.name.encode()); weight_hash.update(pair.actual.detach().numpy().tobytes())
    models = dict(candidate=actual, reference=reference)
    del actual, reference, pairs, pair
    generator, batches = torch.Generator().manual_seed(42 + rank), []
    mask = create_block_mask(lambda b, h, q, k: q >= k, B=args.local_batch, H=None, Q_LEN=4096, KV_LEN=4096, device=str(device), _compile=True)
    for _ in range(args.accumulation):
        tokens = torch.randint(config.vocab_size, (args.local_batch, 4097), generator=generator)
        batches.append(dict(input=tokens[:, :-1].contiguous().to(device), labels=tokens[:, 1:].contiguous().to(device),
                            positions=torch.arange(4096, device=device).expand(args.local_batch, -1), block_mask=mask))
    results = {}
    for name in ("candidate", "reference"):
        model = models.pop(name).to(device)
        model = parallelize_model(NativeRoot(model), **wrapping) if name == "candidate" else parallelize_qwen3(model, **wrapping)
        results[name] = profile(model, batches, settings, args.accumulation, reference=name == "reference")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    gathered = [None] * 8
    dist.all_gather_object(gathered, dict(rank=rank, gpu=torch.cuda.get_device_name(), arms=results, initial_weights_sha256=weight_hash.hexdigest()))
    exit_code = [0]
    if rank == 0:
        assert len({row["initial_weights_sha256"] for row in gathered}) == 1, "initial weights differ across ranks"
        arms = {}
        for name in ("candidate", "reference"):
            rows = [row["arms"][name] for row in gathered]
            durations = [max(row["full_step_seconds_raw"][i] for row in rows) for i in range(args.steps)]
            stats = bench.summary(durations)
            arms[name] = dict(full_step_seconds=stats, full_step_seconds_max_rank_raw=durations,
                              global_labels_per_second=args.local_batch * 4096 * 8 * args.accumulation / stats["median"],
                              peak_memory_allocated_bytes=max(row["peak_memory_allocated_bytes"] for row in rows))
        ce = [gathered[0]["arms"][name]["safety"]["global_ce"] for name in ("candidate", "reference")]
        tol = bench.TOLERANCES["bfloat16"]
        runtime_safe = all(row["arms"][name]["safe"] for row in gathered for name in arms)
        runtime_safe = runtime_safe and abs(ce[0] - ce[1]) <= tol["loss_atol"] + tol["loss_rtol"] * abs(ce[1])
        ratio = arms["candidate"]["full_step_seconds"]["median"] / arms["reference"]["full_step_seconds"]["median"]
        report = dict(status=bench.performance_verdict(ratio, 1.0) if runtime_safe else "failed_runtime_controls", model_config=config.to_dict(),
            scope="single_node_8rank_fsdp_timing_with_native_precision_and_wrapping_differences", source_sha256=descriptor["source_sha256"], runtime=descriptor["runtime"],
            script_sha256=bench.digest(__file__), torchtitan_commit=TORCHTITAN_COMMIT, initial_weights_sha256=weight_hash.hexdigest(),
            mathematics_evidence=evidence, strict_distributed_bf16_gradient_parity="not_claimed", runtime_controls_passed=runtime_safe,
            conditions=dict(world_size=8, shard=8, replicate=1, local_batch=args.local_batch, seq_len=4096, accumulation=args.accumulation,
                global_batch=args.local_batch * 8 * args.accumulation, head_chunks=8, activation_checkpoint="none", model_compile=False,
                parameter_master_dtype="float32", fsdp_parameter_dtype="bfloat16", fsdp_reduce_dtype="float32", optimizer=results["candidate"]["optimizer"],
                clip_max_norm=1.0, auxiliary_coefficients=0, seed=42, timed_routing="natural_unmodified_router_outputs",
                timing="barrier outside; synchronized full update; maximum rank wall time per step", timed_steps=args.steps, warmup_updates=1+args.warmup,
                excluded=["data IO", "mask construction", "compilation/warmup", "checkpoint/eval", "first-update safety checks"]),
            native_differences=["router: ours explicit FP32 math on materialized BF16 gate; official unchanged BF16 gate/input path",
                "native FSDP grouping/reshard: ours separate final norm/head and head reshard; official grouped norm/head retained",
                "Flex internal autotune: ours off; official on; compilation excluded"],
            lag_policy=descriptor["lag_policy"], performance=dict(arms=arms, candidate_reference_step_ratio=ratio), per_rank=gathered)
        report["public_parallelize_sha256"] = bench.digest(inspect.getfile(parallelize_qwen3))
        report["public_fsdp_sha256"] = bench.digest(inspect.getfile(apply_fsdp_to_decoder))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("report", args.output, "status", report["status"], "ratio", ratio, flush=True)
        exit_code[0] = 0 if runtime_safe else 1
    dist.broadcast_object_list(exit_code, src=0)
    dist.destroy_process_group()
    return exit_code[0]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if OUTPUT_PATH is not None and int(os.environ.get("RANK", "0")) == 0:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_PATH.write_text(json.dumps(dict(status="failed", error=type(error).__name__ + ": " + str(error),
                traceback=traceback.format_exc(), script_sha256=bench.digest(__file__), forward_modified=False), indent=2) + "\n")
        raise
