#!/usr/bin/env python3
"""Prepare local smoke text, verify receipts, or compare native continuations.

This script never allocates GPU resources. DCP comparisons load only CPU model
weights and selected continuation state, without constructing a trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata


TEXTS = (
    "recursive models reuse a block while attention keeps each logical cache separate",
    "training counts valid tokens and checkpoints preserve data cursors optimizer state and random state",
    "experts share physical weights while routers select sparse updates for each token",
    "clear recipes combine mechanisms and keep model code direct",
    "validation reads another split without changing training random state",
)


def prepare_fixture(output: str | Path) -> dict[str, Any]:
    """Write an offline tokenizer and immutable engineering-smoke dataset once."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from moeloop.prepare import prepare_dataset

    words = sorted({word for text in TEXTS for word in text.split()})
    vocabulary = {word: index for index, word in enumerate(("<pad>", "<s>", "</s>", "<unk>", *words))}
    if len(vocabulary) > 64:
        raise ValueError("qualification text exceeds the fixed 64-token vocabulary")
    while len(vocabulary) < 64:
        vocabulary[f"<unused{len(vocabulary)}>"] = len(vocabulary)
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
        pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        model_max_length=32)
    identity = hashlib.sha256(json.dumps({"vocabulary": vocabulary, "texts": TEXTS},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    sources = {
        name: {
            "train": [TEXTS[(number + offset) % len(TEXTS)] for number in range(64)],
            "validation": [TEXTS[(number + offset + 1) % len(TEXTS)] for number in range(16)],
        }
        for offset, name in enumerate(("architecture", "training"))
    }
    with tempfile.TemporaryDirectory(prefix="moeloop-qualification-tokenizer-") as temporary:
        tokenizer.save_pretrained(temporary)
        manifest = prepare_dataset(output, sources,
            encode=lambda text: backend.encode(text, add_special_tokens=False).ids,
            tokenizer={"id": "moeloop-qualification-wordlevel-v1", "revision": identity,
                       "vocab_size": 64, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0},
            tokenizer_directory=temporary, shard_tokens=512,
            provenance={"kind": "engineering_smoke_fixture", "version": 1,
                        "quality_evaluation": False, "contains_external_data": False})
    return {"status": "prepared", "manifest": str(manifest.path),
            "manifest_fingerprint": manifest.fingerprint, "vocab_size": 64,
            "sources": list(sources), "train_documents": 128, "validation_documents": 32}


def verify_receipt(directory: str | Path, *, expected_steps: int) -> dict[str, Any]:
    """Require a successful native outcome and its durable final full DCP."""
    directory = Path(directory).resolve()
    outcome_path = directory / "outcome.json"
    outcome = json.loads(outcome_path.read_text())
    if outcome.get("status") != "completed" or outcome.get("completed_updates") != expected_steps:
        raise AssertionError(f"{directory}: incomplete native outcome, expected {expected_steps} updates")
    checkpoint_value = outcome.get("full_checkpoint")
    if not checkpoint_value:
        raise AssertionError(f"{directory}: outcome lacks final full_checkpoint")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = directory / checkpoint
    if checkpoint.name != f"step-{expected_steps}" or not (checkpoint / ".metadata").is_file():
        raise AssertionError(f"{directory}: final full DCP metadata is missing or has the wrong step")
    saved_keys = dcp.FileSystemReader(checkpoint).read_metadata().state_dict_metadata
    for prefix in ("core.", "optimizer.", "lr_scheduler.", "train_state.", "dataloader."):
        if not any(name.startswith(prefix) for name in saved_keys):
            raise AssertionError(f"{directory}: final DCP is missing full-state component {prefix}")
    progress = outcome.get("global_progress", {})
    if progress.get("completed_updates") != expected_steps or progress.get("label_tokens", 0) <= 0:
        raise AssertionError(f"{directory}: receipt does not prove completed updates with valid labels")
    if not outcome.get("continuation_fingerprint") or not outcome.get("global_source_counts"):
        raise AssertionError(f"{directory}: receipt lacks continuation/data evidence")
    return {"directory": str(directory), "checkpoint": str(checkpoint.resolve()), "outcome": outcome}


def load_comparison_state(checkpoint: str | Path) -> dict[str, Any]:
    """Read native flattened model FQNs and bytes without loading optimizer tensors.

    DCP owns its serialization and metadata interpretation. The selected payloads
    are the rank-qualified pickle bytes produced by our native trainer/loader;
    compare only checkpoints created by the qualification job itself.
    """
    reader = dcp.FileSystemReader(checkpoint)
    metadata = reader.read_metadata()
    state: dict[str, Any] = {}
    for name, item in metadata.state_dict_metadata.items():
        if not name.startswith(("core.", "train_state.", "dataloader.", "lr_scheduler.")):
            continue
        if isinstance(item, TensorStorageMetadata):
            state[name] = torch.empty(item.size, dtype=item.properties.dtype, device="cpu")
        elif isinstance(item, BytesStorageMetadata):
            state[name] = b""
        else:
            raise TypeError(f"unsupported DCP metadata for {name}: {type(item)}")
    if not any(name.startswith("core.") for name in state):
        raise AssertionError("checkpoint contains no native moeloop model parameters")
    if not any(name.startswith("train_state.rank_") for name in state):
        raise AssertionError("checkpoint contains no rank-qualified trainer continuation state")
    if not any(name.startswith("dataloader.dp_rank_") for name in state):
        raise AssertionError("checkpoint contains no rank-qualified data cursors")
    dcp.load(state, storage_reader=reader, no_dist=True)
    train_ranks = {name for name in state if name.startswith("train_state.rank_")}
    identity = pickle.loads(state["train_state.rank_0"])["identity"] if "train_state.rank_0" in state else {}
    world_size = identity.get("world_size", 0)
    if not isinstance(world_size, int) or world_size < 1 or train_ranks != {f"train_state.rank_{rank}" for rank in range(world_size)}:
        raise AssertionError("checkpoint lacks continuation state for every training rank")
    data_world_size = state.get("dataloader.world_size", 0)
    data_ranks = {name for name in state if name.startswith("dataloader.dp_rank_")}
    if not isinstance(data_world_size, int) or data_world_size < 1 or data_ranks != {f"dataloader.dp_rank_{rank}" for rank in range(data_world_size)}:
        raise AssertionError("checkpoint lacks data cursors for every data-parallel rank")
    return state


def _assert_equal_tree(left: Any, right: Any, path: str) -> None:
    if type(left) is not type(right):
        raise AssertionError(f"continuation types differ at {path}")
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or not torch.equal(left, right):
            raise AssertionError(f"continuation tensor differs at {path}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"continuation keys differ at {path}")
        for key in left:
            _assert_equal_tree(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f"continuation lengths differ at {path}")
        for index, (one, two) in enumerate(zip(left, right)):
            _assert_equal_tree(one, two, f"{path}[{index}]")
    elif type(left).__module__.startswith("numpy") and hasattr(left, "shape"):
        if left.dtype != right.dtype or left.shape != right.shape or not bool((left == right).all()):
            raise AssertionError(f"continuation array differs at {path}")
    elif left != right:
        raise AssertionError(f"continuation value differs at {path}")


def compare_runs(uninterrupted: str | Path, resumed: str | Path, *, expected_steps: int = 4,
                 atol: float = 2e-5, rtol: float = 2e-4) -> dict[str, Any]:
    """Check a native uninterrupted trajectory against a stopped/resumed one."""
    one = verify_receipt(uninterrupted, expected_steps=expected_steps)
    two = verify_receipt(resumed, expected_steps=expected_steps)
    for name in ("continuation_fingerprint", "global_progress", "global_source_counts"):
        _assert_equal_tree(one["outcome"][name], two["outcome"][name], f"outcome.{name}")
    first = load_comparison_state(one["checkpoint"])
    second = load_comparison_state(two["checkpoint"])
    if first.keys() != second.keys():
        raise AssertionError("uninterrupted/resumed DCP model or continuation keys differ")
    errors: dict[str, float] = {}
    discrete_keys: list[str] = []
    for name, value in first.items():
        other = second[name]
        if name.startswith("core."):
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() or not torch.isfinite(other).all():
                raise AssertionError(f"nonfinite or non-tensor model state: {name}")
            torch.testing.assert_close(value, other, atol=atol, rtol=rtol,
                msg=lambda message: f"continuation model state {name}: {message}")
            errors[name] = float((value.float() - other.float()).abs().max())
        else:
            if name.startswith(("train_state.rank_", "dataloader.dp_rank_")):
                value, other = pickle.loads(value), pickle.loads(other)
            _assert_equal_tree(value, other, name)
            discrete_keys.append(name)
    return {"status": "passed", "expected_steps": expected_steps,
            "uninterrupted": str(Path(uninterrupted).resolve()), "resumed": str(Path(resumed).resolve()),
            "model_tensor_count": len(errors), "model_elements": sum(v.numel() for k, v in first.items() if k.startswith("core.")),
            "max_absolute_model_error": max(errors.values()),
            "worst_model_tensor": max(errors, key=errors.get),
            "atol": atol, "rtol": rtol, "exact_continuation_keys": discrete_keys,
            "training_rank_count": sum(name.startswith("train_state.rank_") for name in discrete_keys),
            "data_rank_count": sum(name.startswith("dataloader.dp_rank_") for name in discrete_keys),
            "global_progress": one["outcome"]["global_progress"],
            "global_source_counts": one["outcome"]["global_source_counts"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="write fresh offline smoke fixture")
    prepare.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify native completion receipt and final metadata")
    verify.add_argument("--directory", type=Path, required=True)
    verify.add_argument("--steps", type=int, required=True)
    verify.add_argument("--output", type=Path)
    compare = commands.add_parser("compare", help="compare uninterrupted and resumed final checkpoints")
    compare.add_argument("--uninterrupted", type=Path, required=True)
    compare.add_argument("--resumed", type=Path, required=True)
    compare.add_argument("--steps", type=int, default=4)
    compare.add_argument("--atol", type=float, default=2e-5)
    compare.add_argument("--rtol", type=float, default=2e-4)
    compare.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command == "prepare":
            result = prepare_fixture(arguments.output)
        elif arguments.command == "verify":
            result = {"status": "passed", **verify_receipt(arguments.directory, expected_steps=arguments.steps)}
        else:
            result = compare_runs(arguments.uninterrupted, arguments.resumed,
                expected_steps=arguments.steps, atol=arguments.atol, rtol=arguments.rtol)
    except Exception as error:
        if arguments.command != "prepare" and arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2) + "\n")
        raise
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if arguments.command != "prepare" and arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
