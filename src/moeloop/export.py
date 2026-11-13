"""Select model weights from a local training checkpoint and save portable HF.

DCP directories are trusted local training artifacts; ``weights="ema"`` selects the
EMA of the parameters that a run with ``ema_decay`` saves there. Model-only torch files
are read with ``weights_only=True``; full arbitrary Python training pickles are not a
supported input. TorchTitan and tracking services are never imported here.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch
from torch import Tensor

from .data import PreparedManifest, file_sha256, validate_tokenizer_roles
from .model import ModelConfig, RecursiveLM


TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "config.json",
)


def _inference_license() -> str:
    source_license = Path(__file__).resolve().parents[2] / "LICENSE"
    if source_license.is_file():
        return source_license.read_text(encoding="utf-8")
    try:
        content = importlib.metadata.distribution("moeloop-lab").read_text("licenses/LICENSE")
    except importlib.metadata.PackageNotFoundError:
        content = None
    if not content:
        raise RuntimeError("MoE-Loop inference source LICENSE is missing from the checkout or installed distribution")
    return content


def _model_config(value: Mapping[str, Any]) -> ModelConfig:
    if "model_config" in value:
        value = value["model_config"]
    elif "model_spec" in value:
        value = value["model_spec"]["model"]["core"]
    return ModelConfig.from_dict(value)


def _metadata_directories(checkpoint: Path):
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    yield directory
    yield from directory.parents


def resolve_model_config(
    checkpoint: str | Path,
    model_config: ModelConfig | Mapping[str, Any] | str | Path | None = None,
) -> ModelConfig:
    if isinstance(model_config, ModelConfig):
        return model_config
    if isinstance(model_config, Mapping):
        return _model_config(model_config)
    if model_config is None:
        checkpoint = Path(checkpoint).resolve()
        model_config = next((
            directory / name
            for directory in _metadata_directories(checkpoint)
            for name in ("model_config.json", "resolved_config.json")
            if (directory / name).is_file()
        ), None)
        if model_config is None:
            raise ValueError("model configuration was not found; provide --model-config")
    return _model_config(json.loads(Path(model_config).read_text(encoding="utf-8")))


def _key_layout(keys: set[str], expected: Mapping[str, Tensor], *, model_only: bool) -> tuple[str, dict[str, str]]:
    """Return canonical->checkpoint keys for the finite supported owner layouts."""
    layouts = {
        "reference": {key: key for key in expected},
        "native_grouped": {key: "core." + key.replace(".experts.", ".experts.routed.inner_experts.") for key in expected},
        "wrapped_reference": {key: "core." + key for key in expected},
        "huggingface": {key: "model." + key for key in expected},
    }
    roots = {key.split(".", 1)[0] + "." for key in expected}
    failures = []
    for name, mapping in layouts.items():
        required = set(mapping.values())
        prefixes = roots if name == "reference" else {"model." if name == "huggingface" else "core."}
        actual = keys if model_only else {key for key in keys if any(key.startswith(prefix) for prefix in prefixes)}
        missing, extra = required - keys, actual - required
        if not missing and not extra:
            return name, mapping
        failures.append((len(missing) + len(extra), name, missing, extra))
    _, name, missing, extra = min(failures, key=lambda failure: failure[0])
    raise ValueError(f"checkpoint model keys do not match {name}: missing={sorted(missing)[:8]}, unexpected={sorted(extra)[:8]}")


EMA_PREFIX = "train_state.ema."


def _load_model_weights(checkpoint: Path, model: RecursiveLM, weights: str) -> tuple[dict[str, Tensor], dict[str, Any]]:
    expected = model.state_dict()
    if checkpoint.is_dir():
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata

        if not (checkpoint / ".metadata").is_file():
            raise ValueError("DCP checkpoint is incomplete: .metadata is missing")
        reader = dcp.FileSystemReader(checkpoint)
        metadata = reader.read_metadata()
        # LoopTrainer keys its EMA like the model parameters, under EMA_PREFIX.
        prefix = EMA_PREFIX if weights == "ema" else ""
        keys = {key.removeprefix(prefix) for key in metadata.state_dict_metadata if key.startswith(prefix)}
        if prefix and not keys:
            raise ValueError("checkpoint holds no EMA weights; the run kept none (ema_decay unset)")
        layout, mapping = _key_layout(keys, expected, model_only=bool(prefix))
        mapping = {canonical: prefix + stored for canonical, stored in mapping.items()}
        selected = {}
        for canonical, stored in mapping.items():
            tensor_metadata = metadata.state_dict_metadata[stored]
            if not isinstance(tensor_metadata, TensorStorageMetadata) or tuple(tensor_metadata.size) != tuple(expected[canonical].shape):
                raise ValueError(f"checkpoint shape/type differs for {canonical}")
            selected[stored] = torch.empty_like(expected[canonical], device="cpu", dtype=tensor_metadata.properties.dtype)
        # Flat model FQNs are top-level keys in native TorchTitan checkpoints.
        # No optimizer/data/training state is allocated or requested from DCP.
        dcp.load(selected, storage_reader=reader, planner=DefaultLoadPlanner(flatten_state_dict=False), no_dist=True)
        state = {canonical: selected[stored] for canonical, stored in mapping.items()}
        provenance = {"format": "dcp", "key_layout": layout, "metadata_sha256": file_sha256(checkpoint / ".metadata")}
    else:
        if weights == "ema":
            raise ValueError("EMA weights are read from a DCP training checkpoint")
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(loaded, Mapping) and set(loaded) in ({"model"}, {"state_dict"}):
            loaded = next(iter(loaded.values()))
        if not isinstance(loaded, Mapping) or any(not isinstance(key, str) or not isinstance(value, Tensor) for key, value in loaded.items()):
            raise ValueError("torch checkpoint must contain only a model state_dict of tensors")
        layout, mapping = _key_layout(set(loaded), expected, model_only=True)
        state = {canonical: loaded[stored] for canonical, stored in mapping.items()}
        for key, value in state.items():
            if value.shape != expected[key].shape:
                raise ValueError(f"checkpoint shape differs for {key}: {tuple(value.shape)} != {tuple(expected[key].shape)}")
        provenance = {"format": "torch_model_only", "key_layout": layout, "sha256": file_sha256(checkpoint)}
    return state, provenance


def _resolve_tokenizer(checkpoint: Path, tokenizer_path: str | Path | None) -> Path:
    if tokenizer_path is not None:
        path = Path(tokenizer_path).resolve()
        if not path.is_dir():
            raise ValueError(f"tokenizer directory does not exist: {path}")
        return path
    for directory in _metadata_directories(checkpoint):
        if (directory / "tokenizer" / "tokenizer_config.json").is_file():
            return directory / "tokenizer"
        resolved = directory / "resolved_config.json"
        if resolved.is_file():
            configured = json.loads(resolved.read_text(encoding="utf-8")).get("hf_assets_path")
            if configured:
                path = Path(configured)
                path = path if path.is_absolute() else directory / path
                if path.is_dir():
                    return path.resolve()
    raise ValueError("tokenizer assets were not found; provide --tokenizer")


def _training_tokenizer_identity(checkpoint: Path, tokenizer_directory: Path) -> dict[str, Any] | None:
    candidates = []
    for directory in _metadata_directories(checkpoint):
        resolved = directory / "resolved_config.json"
        if resolved.is_file():
            configured = json.loads(resolved.read_text()).get("dataloader", {}).get("manifest_path")
            if configured:
                path = Path(configured)
                candidates.append(path if path.is_absolute() else directory / path)
                break
    parent_manifest = tokenizer_directory.parent / "manifest.json"
    if parent_manifest.is_file():
        candidates.append(parent_manifest)
    identity = None
    for path in candidates:
        if not path.is_file():
            raise ValueError(f"training manifest is unavailable; cannot verify tokenizer identity: {path}")
        # Export verifies tokenizer bytes below, without rescanning all training
        # token shards merely to select the matching tokenizer identity.
        metadata = PreparedManifest.load(path, verify_hashes=False).tokenizer
        validate_tokenizer_roles(tokenizer_directory, metadata)
        expected = {Path(name).name: digest for name, digest in metadata.get("files", {}).items()}
        actual = {name: file_sha256(tokenizer_directory / name) for name in TOKENIZER_FILES if (tokenizer_directory / name).is_file()}
        if expected and actual != expected:
            raise ValueError("export tokenizer files differ from the training manifest; use the prepared tokenizer assets")
        if identity is not None and identity != metadata:
            raise ValueError("export tokenizer's parent manifest differs from the training tokenizer identity")
        identity = metadata
    return identity


def export_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    model_config: ModelConfig | Mapping[str, Any] | str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    num_visits: int | None = None,
    weights: str = "model",
) -> Path:
    """Export physical model weights, tokenizer, config, code, and provenance.

    ``model_config.json`` or ``resolved_config.json`` is discovered in checkpoint
    ancestors when omitted. Output must be a new directory. Depth changes default
    inference execution only; independent owners remain bounded by their capacity.
    ``weights`` is ``model`` or ``ema``, the checkpoint's EMA of the parameters; the
    provenance records which.
    """
    from transformers import AutoTokenizer, GenerationConfig
    from .hf import MoeloopConfig, MoeloopForCausalLM

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    inference_license = _inference_license()
    config = resolve_model_config(checkpoint, model_config)
    if num_visits is not None:
        config = replace(config, num_visits=num_visits)
    tokenizer_directory = _resolve_tokenizer(checkpoint, tokenizer_path)
    tokenizer_identity = _training_tokenizer_identity(checkpoint, tokenizer_directory)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory, local_files_only=True, trust_remote_code=False)
    if tokenizer_identity is not None:
        for role in ("bos", "eos", "pad"):
            if getattr(tokenizer, f"{role}_token_id") != tokenizer_identity[f"{role}_token_id"]:
                raise ValueError(f"HF tokenizer {role.upper()} ID differs from the training manifest")
    if len(tokenizer) > config.vocab_size:
        raise ValueError("tokenizer vocabulary is larger than the model vocabulary")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    hf_config = MoeloopConfig(config.to_dict(), bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    # Build shape/owner metadata first; do not allocate a random full CPU model
    # beside the checkpoint's full model tensors during a large export.
    with torch.device("meta"):
        model = MoeloopForCausalLM(hf_config)
    state, source = _load_model_weights(checkpoint, model.model, weights)
    model.model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    source["name"] = checkpoint.name
    source["weights"] = weights
    tokenizer_source = {name: file_sha256(tokenizer_directory / name) for name in TOKENIZER_FILES if (tokenizer_directory / name).is_file()}
    output.mkdir(parents=True, exist_ok=False)
    try:
        model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        GenerationConfig.from_model_config(hf_config).save_pretrained(output)
        (output / "LICENSE").write_text(inference_license, encoding="utf-8")
        provenance = {
            "format": "moeloop-hf-export", "version": 1,
            "checkpoint": source, "model_config": config.to_dict(),
            "tokenizer_source_files": tokenizer_source,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
            "artifact_sha256": {path.name: file_sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
        }
        (output / "export_provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(output)
        raise
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--num-visits", type=int)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    args = parser.parse_args(argv)
    output = export_checkpoint(args.checkpoint, args.output, args.model_config, args.tokenizer, args.num_visits,
                               args.weights)
    print(output)


if __name__ == "__main__":
    main()
