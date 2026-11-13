"""TorchTitan -> CVSM artifacts, without importing Torch, CUDA or MLflow.

capture_trainer is called after native Trainer construction and before train().
It observes the native config and batch mesh; it does not launch, resume, or
change training. Checkpoint settings are preserved, not certified as restored.
"""

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Mapping


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def build_record(native: Mapping, *, world_size: int, data_parallel_degree: int,
                 gradient_accumulation_steps: int, assets: Mapping, sources: Mapping,
                 record_kind: str = "training", environment: Mapping | None = None) -> dict:
    """Pure projection of already-observed inputs; budget counts input positions.

    Keep *all* native fields. Per-run locations belong in a study's Adaptable
    section with a rule; removing checkpoint inputs would hide possible resume.
    Target tokens use the original full schedule, including for resumed runs.
    """
    native = json.loads(_json_bytes(dict(native)))
    training = native["training"]
    world = _positive_int(world_size, "world_size")
    dp = _positive_int(data_parallel_degree, "data_parallel_degree")
    accumulation = _positive_int(gradient_accumulation_steps, "gradient_accumulation_steps")
    local = _positive_int(training["local_batch_size"], "training.local_batch_size")
    seq = _positive_int(training["seq_len"], "training.seq_len")
    steps = _positive_int(training["steps"], "training.steps")
    if world % dp:
        raise ValueError("world_size must be a multiple of the observed data-parallel degree")
    batch = local * dp * accumulation
    configured = training["global_batch_size"]
    if type(configured) is not int or (configured != -1 and configured != batch):
        raise ValueError("native global_batch_size disagrees with observed accumulation and batch mesh")
    if configured == -1 and accumulation != 1:
        raise ValueError("native automatic global_batch_size requires one accumulation step")
    record = {
        "schema_version": 1,
        "record_kind": record_kind,
        "native": native,
        "derived": {
            "world_size": world,
            "data_parallel_degree": dp,
            "gradient_accumulation_steps": accumulation,
            "global_batch_size": batch,
            "input_tokens_per_step": batch * seq,
            "target_input_tokens": batch * seq * steps,
        },
        "assets": dict(assets),
        "sources": dict(sources),
        "environment": dict(environment or {}),
    }
    # Validate the whole record before creating any artifact.
    return json.loads(_json_bytes(record))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observe_source(module_file: Path, *, expected_commit: str | None = None) -> dict:
    """Observe the Git checkout that actually contains an imported module file."""
    module_file = Path(module_file).resolve(strict=True)

    def git(*args):
        result = subprocess.run(["git", "-C", str(module_file.parent), *args],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise ValueError("runtime source must be a tracked file in an editable Git checkout")
        return result.stdout.strip()

    root = Path(git("rev-parse", "--show-toplevel"))
    git("ls-files", "--error-unmatch", "--full-name", str(module_file))
    commit = git("rev-parse", "HEAD")
    if expected_commit and commit != expected_commit:
        raise ValueError("runtime source commit differs from the project's TorchTitan pin")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("runtime model and TorchTitan source checkouts must be clean")
    if not module_file.is_relative_to(root):
        raise ValueError("runtime module is outside its observed source checkout")
    return {"commit": commit, "dirty": False}


def _torchtitan_pin():
    # An editable Git checkout is intentional: protected pins live alongside code.
    lock = Path(__file__).resolve().parents[2] / "configs" / "upstream.json"
    return json.loads(lock.read_text())["torchtitan"]["commit"]


def _rank_zero():
    rank = int(os.environ.get("RANK", "0"))
    if rank < 0:
        raise ValueError("RANK cannot be negative")
    return rank == 0


def _environment():
    packages = {}
    for name in ("torch", "triton", "torchdata", "numpy", "tokenizers", "spmd_types"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    version = getattr(sys.modules.get("torch"), "version", None)
    return {"python": platform.python_version(), "packages": packages,
            "torch_git_commit": getattr(version, "git_version", None),
            "torch_cuda_version": getattr(version, "cuda", None)}


def _write_once(path, value):
    """Publish complete JSON atomically, refusing to replace an existing artifact."""
    path = Path(path)
    payload = _json_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic and fails if destination already exists
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_record(record: Mapping) -> None:
    """Publish the one final resolved config that the CVSM launcher polls."""
    if not _rank_zero():
        return
    target = Path(os.environ["RUN_DIR"])
    _write_once(target / "native-config.json", record["native"])
    _write_once(target / "config.json", dict(record))


def capture_trainer(trainer, *, data_manifest: Path, tokenizer_files: Mapping[str, Path],
                    model_module: ModuleType, evaluation_manifest: Path | None = None) -> dict | None:
    """Observe a built v0.3.0 Trainer on global rank 0, then publish its config.

    Call on every rank after config.build(). No rank barrier or training mutation
    is introduced. Native torchrun error handling must terminate peers on failure.
    The model_module argument names the actual imported implementation, not a
    caller-supplied SHA. Manifest hashes identify files, not verified shard bytes.
    """
    if not _rank_zero():
        return None
    torchtitan = sys.modules.get("torchtitan")
    if torchtitan is None or not getattr(torchtitan, "__file__", None):
        raise ValueError("capture_trainer requires the actual imported TorchTitan package")
    if not getattr(model_module, "__file__", None):
        raise ValueError("model_module must be the actual imported model implementation")
    sources = {
        "torchtitan": observe_source(torchtitan.__file__, expected_commit=_torchtitan_pin()),
        "model": observe_source(model_module.__file__),
    }
    if not tokenizer_files or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name)
                                  for name in tokenizer_files):
        raise ValueError("tokenizer_files needs stable alphanumeric logical names")
    tokenizers = {name: sha256_file(path) for name, path in sorted(tokenizer_files.items())}
    assets = {
        "data_manifest_sha256": sha256_file(data_manifest),
        "tokenizer_files": tokenizers,
        "tokenizer_sha256": hashlib.sha256(_json_bytes(tokenizers)).hexdigest(),
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest) if evaluation_manifest else None,
    }
    dims = trainer.parallel_dims
    dp = dims.get_mesh("batch").size() if dims.dp_enabled else 1
    record = build_record(
        trainer.config.to_dict(), world_size=dims.world_size, data_parallel_degree=dp,
        gradient_accumulation_steps=trainer.gradient_accumulation_steps,
        assets=assets, sources=sources, environment=_environment(),
    )
    write_record(record)
    return record


def write_metrics(metrics: Mapping[str, float]) -> None:
    """Write one finite, flat final metrics snapshot on global rank 0."""
    if not _rank_zero():
        return
    if not metrics or any(not isinstance(key, str) or not key or
                          type(value) not in (float, int) or not math.isfinite(value)
                          for key, value in metrics.items()):
        raise ValueError("metrics must be a nonempty flat mapping of names to finite numbers")
    _write_once(Path(os.environ["RUN_DIR"]) / "metrics.json", dict(metrics))
