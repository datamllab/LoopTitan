#!/usr/bin/env python3
"""Build a source release, exercise its installed path, and collect full lm-eval.

Uses existing Python/GPU environments and ordinary files. No resource allocation,
experiment service, private source, or benchmark implementation is required.
Receipt verification uses only Python's standard library and never runs a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

TASK_SAMPLES = {"arc_easy": 2376, "hellaswag": 10042}
HARNESS_VERSION = "0.4.13"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def verify_file(record):
    path = Path(record["path"])
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def tree_record(path):
    path = Path(path).resolve()
    files = {str(item.relative_to(path)): sha256(item) for item in sorted(path.rglob("*"))
             if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"}
    if not files:
        raise ValueError(f"empty artifact tree: {path}")
    return {"path": str(path), "files": files,
            "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def verify_tree(record):
    if tree_record(record["path"]) != record:
        raise ValueError(f"artifact tree mismatch: {record['path']}")
    return Path(record["path"])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def clean_env(*, offline=False):
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "MOELOOP_ANCHOR_SPEC", "MOELOOP_ANCHOR_RESUME",
                "MOELOOP_QUAL_SPEC", "MOELOOP_QUAL_RESUME"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    return env


def run(argv, directory, name, *, env=None, cwd=None):
    directory = Path(directory).resolve()
    cwd = Path(cwd or directory).resolve()
    log = directory / f"{name}.log"
    receipt = {"argv": [str(arg) for arg in argv], "cwd": str(cwd),
               "started_at": datetime.now(timezone.utc).isoformat()}
    selected_env = env or clean_env()
    receipt["environment"] = {key: value for key, value in selected_env.items()
        if key.startswith("MOELOOP_EXPERIMENT_") or key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONNOUSERSITE", "CUDA_VISIBLE_DEVICES")}
    with log.open("x") as stream:
        result = subprocess.run(receipt["argv"], cwd=cwd, env=selected_env,
                                stdout=stream, stderr=subprocess.STDOUT)
    receipt.update(returncode=result.returncode, log=file_record(log),
                   finished_at=datetime.now(timezone.utc).isoformat())
    write_json(directory / f"{name}.process.json", receipt)
    if result.returncode:
        raise RuntimeError(f"process failed ({result.returncode}); see {log}")
    return receipt


def verify_process(record):
    if record.get("returncode") != 0 or not record.get("argv") or not record.get("cwd"):
        raise ValueError("failed or incomplete process receipt")
    verify_file(record["log"])


def _new_output(path, source=None):
    path = Path(path).resolve()
    if source is not None:
        roots = [Path(source).resolve()]
        checkout = Path(__file__).resolve().parents[1]
        if (checkout / ".git").exists():
            roots.append(checkout)
        if any(path == root or root in path.parents for root in roots):
            raise ValueError("output must be outside the source checkout")
    path.mkdir(parents=True, exist_ok=False)
    return path


def package(source, output, *, source_identity, python=sys.executable, uv="uv"):
    """Build a wheel *from the generated sdist*, then prove Torch-free import."""
    source = Path(source).resolve()
    if not (source / "pyproject.toml").is_file():
        raise ValueError("source must contain pyproject.toml")
    # A staged git archive is intentional: checkout metadata/private files must
    # not accidentally become a release source or an evidence input.
    forbidden = (".git", ".venv", "private", "build", "dist")
    if any((source / name).exists() for name in forbidden):
        raise ValueError("use a frozen source archive without .git/.venv/private/build/dist")
    identity = read_json(source_identity)
    source_record = tree_record(source)
    if not re.fullmatch(r"[0-9a-f]{40}", identity.get("commit", "")) or identity.get("dirty") is not False:
        raise ValueError("release source must identify a clean exact Git commit")
    if not identity.get("files") or any(source_record["files"].get(name) != value for name, value in identity["files"].items()):
        raise ValueError("frozen source identity files differ from the release archive")
    if any(name not in identity["files"] for name in source_record["files"] if name.startswith("src/") and name.endswith(".py")):
        raise ValueError("source identity must cover every public Python module")
    output = _new_output(output, source)
    processes = []
    build_source = output / "build-source"
    shutil.copytree(source, build_source)
    processes.append(run([uv, "build", "--sdist", "--out-dir", output / "sdist", build_source], output, "sdist"))
    sdist, = (output / "sdist").glob("*.tar.gz")
    unpack = output / "from-sdist"
    unpack.mkdir()
    with tarfile.open(sdist) as archive:
        archive.extractall(unpack, filter="data")
    rebuilt_source, = unpack.iterdir()
    processes.append(run([uv, "build", "--wheel", "--out-dir", output / "wheel", rebuilt_source], output, "wheel"))
    sdist_tree = tree_record(rebuilt_source)
    wheel, = (output / "wheel").glob("*.whl")
    # Check public Python bytes, rather than trusting a build-success exit code.
    with zipfile.ZipFile(wheel) as archive:
        module_hashes = {name: hashlib.sha256(archive.read(name)).hexdigest()
                         for name in archive.namelist() if name.startswith(("moeloop/", "moeloop_lab/")) and name.endswith(".py")}
    expected = {str(p.relative_to(source / "src")): sha256(p) for p in (source / "src").rglob("*.py")}
    if module_hashes != expected:
        raise ValueError("wheel public Python files differ from the frozen source")
    base = output / "base-env"
    processes.append(run([uv, "venv", "--python", python, base], output, "base-venv"))
    executable = base / "bin/python"
    processes.append(run([uv, "pip", "install", "--python", executable, "--no-deps", wheel], output, "base-install"))
    proof = output / "base-import.json"
    code = """import importlib.util,importlib.metadata,json,pathlib,sys
import moeloop,moeloop_lab
assert importlib.util.find_spec('torch') is None
assert 'torch' not in sys.modules
assert pathlib.Path(moeloop.__file__).is_relative_to(sys.prefix)
assert pathlib.Path(moeloop_lab.__file__).is_relative_to(sys.prefix)
pathlib.Path(sys.argv[1]).write_text(json.dumps({'torch_available':False,'torch_imported':False,'version':importlib.metadata.version('moeloop-lab'),'module':moeloop.__file__,'prefix':sys.prefix}))
"""
    processes.append(run([executable, "-I", "-c", code, proof], output, "base-import", env=clean_env(offline=True)))
    verify_tree(source_record)
    receipt = {"format": "moeloop-release-package", "schema_version": 1,
        "source": source_record, "source_identity": file_record(source_identity), "sdist_source": sdist_tree, "sdist": file_record(sdist),
        "wheel": file_record(wheel), "wheel_modules": module_hashes,
        "base_import": file_record(proof), "processes": processes}
    write_json(output / "package-receipt.json", receipt)
    return output / "package-receipt.json"


def verify_package_receipt(path):
    payload = read_json(path)
    if payload.get("format") != "moeloop-release-package" or payload.get("schema_version") != 1:
        raise ValueError("unsupported package receipt")
    for key in ("source", "sdist_source"):
        verify_tree(payload[key])
    identity = read_json(verify_file(payload["source_identity"]))
    if not re.fullmatch(r"[0-9a-f]{40}", identity.get("commit", "")) or identity.get("dirty") is not False or not identity.get("files"):
        raise ValueError("release source identity is not a clean exact Git commit")
    if any(payload["source"]["files"].get(name) != value for name, value in identity["files"].items()) or any(name not in identity["files"] for name in payload["source"]["files"] if name.startswith("src/") and name.endswith(".py")):
        raise ValueError("release source identity does not bind the package source")
    for key in ("sdist", "wheel", "base_import"):
        verify_file(payload[key])
    for process in payload["processes"]:
        verify_process(process)
    stages = [row["argv"] for row in payload["processes"]]
    if len(stages) != 5 or not any("--sdist" in row for row in stages) or not any(
            "--wheel" in row and payload["sdist_source"]["path"] in row for row in stages):
        raise ValueError("missing source→sdist→wheel process chain")
    with zipfile.ZipFile(payload["wheel"]["path"]) as archive:
        modules = {name: hashlib.sha256(archive.read(name)).hexdigest()
                   for name in archive.namelist() if name.startswith(("moeloop/", "moeloop_lab/")) and name.endswith(".py")}
    expected_modules = {name.removeprefix("src/"): value for name, value in payload["source"]["files"].items() if name.startswith("src/") and name.endswith(".py")}
    if modules != payload["wheel_modules"] or modules != expected_modules:
        raise ValueError("wheel modules differ from the exact frozen source")
    proof = read_json(payload["base_import"]["path"])
    if proof.get("torch_available") is not False or proof.get("torch_imported") is not False or proof.get("version") != "1.0.0rc1":
        raise ValueError("base install does not prove Torch-free release import")
    return payload


def verify_lm_eval_result(result):
    """A two-question fixture or --limit cannot count as complete evaluation."""
    if result.get("config", {}).get("limit", "missing") is not None:
        raise ValueError("full downstream evaluation must have limit=None")
    if result.get("config", {}).get("samples") is not None:
        raise ValueError("sample-selected downstream evaluation is incomplete")
    for task, count in TASK_SAMPLES.items():
        task_hash = result.get("task_hashes", {}).get(task)
        if not isinstance(task_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", task_hash):
            raise ValueError(f"missing native logged task hash: {task}")
        if result.get("n-shot", {}).get(task) != 0:
            raise ValueError(f"{task} must explicitly use 0-shot")
        if result.get("n-samples", {}).get(task) != {"original": count, "effective": count}:
            raise ValueError(f"{task} requires the complete {count}-example task")
        metrics = result.get("results", {}).get(task, {})
        for metric in ("acc,none", "acc_norm,none"):
            value = metrics.get(metric)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"missing or invalid {task}/{metric}")


def _harness_identity(output, python):
    """Hash installed harness and exact task YAML/helper bytes before execution."""
    destination = Path(output) / "harness.json"
    code = r'''import hashlib,importlib.metadata,json,pathlib,sys
import lm_eval
root=pathlib.Path(lm_eval.__file__).resolve().parent
version=importlib.metadata.version('lm_eval')
assert version=='0.4.13',f'expected lm_eval==0.4.13, got {version}'
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.yml')}
assert 'tasks/arc/arc_easy.yaml' in files and 'tasks/hellaswag/hellaswag.yaml' in files and 'tasks/hellaswag/utils.py' in files
payload={'version':version,'source_root':str(root),'files':files,'distribution_metadata':importlib.metadata.distribution('lm_eval').read_text('direct_url.json')}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
'''
    process = run([python, "-I", "-c", code, destination], output, "harness-identity")
    return file_record(destination), process


def _lm_eval_batch_size(value: int | str) -> str:
    """Validate the native harness syntax without choosing a batch size here."""
    batch = str(value)
    try:
        if type(value) not in (int, str):
            raise ValueError
        if batch.startswith("auto:"):
            schedule = float(batch[5:])
            if not math.isfinite(schedule) or schedule <= 0:
                raise ValueError
        elif batch != "auto" and int(batch) <= 0:
            raise ValueError
    except ValueError:
        raise ValueError("batch size must be a positive integer, auto, or auto:positive_number") from None
    return batch


def _terminal_readout_basis(config):
    """Recognize declared terminal CE or the pre-exit-gate MoE-Loop schemas."""
    if config.get("labels_objective") == "terminal_ce":
        return "explicit_terminal_ce"
    model = config.get("model_config", {})
    schema = model.get("schema_version")
    if ("labels_objective" not in config and config.get("model_type") == "moeloop"
            and type(schema) is int and schema in (1, 2)
            and all(part.get("exit_distribution", False) is False and not ({"gate", "exit_gate"} & part.keys())
                    for part in (config, model))):
        return f"legacy_moeloop_schema_{schema}_without_exit_gate"
    raise ValueError("downstream model has unknown or nonterminal readout semantics")


def downstream(models, output, *, python=sys.executable, device="cuda", batch_size=1):
    """Run the pinned existing CLI. Dataset download happens in this environment."""
    batch_size = _lm_eval_batch_size(batch_size)
    output = _new_output(output)
    harness, process = _harness_identity(output, python)
    rows = {}
    for name, model in models.items():
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("model names must be safe single path components")
        model = Path(model).resolve()
        if "," in str(model):
            raise ValueError("HF model path cannot contain a comma")
        config = read_json(model / "config.json")
        readout_basis = _terminal_readout_basis(config)
        depth = config["model_config"]["num_visits"]
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("model must pin a positive terminal evaluation depth")
        provenance = file_record(model / "export_provenance.json")
        artifact = tree_record(model)
        directory = output / name
        directory.mkdir()
        argv = [python, "-I", "-m", "lm_eval", "--model", "hf", "--model_args",
            f"pretrained={model},trust_remote_code=True,dtype=float32", "--tasks", ",".join(TASK_SAMPLES),
            "--num_fewshot", "0", "--batch_size", str(batch_size), "--device", device,
            "--log_samples", "--output_path", str(directory / "results")]
        # Models are local; dataset downloads are intentionally permitted here.
        execution = run(argv, directory, "lm-eval", env=clean_env())
        results = [p for p in (directory / "results").rglob("*.json") if "results" in read_json(p)]
        if len(results) != 1:
            raise ValueError(f"expected exactly one harness result under {directory}")
        result = read_json(results[0])
        verify_lm_eval_result(result)
        sample_logs = {}
        for task in TASK_SAMPLES:
            files = list((directory / "results").rglob(f"samples_{task}_*.jsonl"))
            if len(files) != 1:
                raise ValueError(f"missing or ambiguous private native sample log: {task}")
            sample_logs[task] = file_record(files[0])
        verify_tree(artifact)
        rows[name] = {"model": artifact, "export_provenance": provenance,
            "num_visits": depth, "readout": "terminal", "readout_basis": readout_basis, "dtype": "float32", "num_fewshot": 0, "limit": None,
            "result": file_record(results[0]), "sample_logs": sample_logs, "process": execution}
        write_json(directory / "model-receipt.json", rows[name])
    receipt = {"format": "moeloop-release-lm-eval", "schema_version": 1,
               "harness": harness, "harness_process": process, "models": rows}
    write_json(output / "lm-eval-receipt.json", receipt)
    verify_downstream_receipt(output / "lm-eval-receipt.json")
    return output / "lm-eval-receipt.json"



def collect_downstream(receipts, output):
    """Combine separately completed models without rerunning or downloading tasks."""
    combined, identities = {}, []
    harness, harness_process, harness_files = None, None, None
    for path in receipts:
        payload = verify_downstream_receipt(path)
        identity = read_json(payload["harness"]["path"])
        if harness_files is not None and identity["files"] != harness_files:
            raise ValueError("downstream inputs used different harness/task source bytes")
        if set(combined) & set(payload["models"]):
            raise ValueError("downstream input model names overlap")
        harness, harness_process, harness_files = payload["harness"], payload["harness_process"], identity["files"]
        combined.update(payload["models"])
        identities.append(file_record(path))
    if not combined:
        raise ValueError("provide at least one completed downstream receipt")
    write_json(output, {"format": "moeloop-release-lm-eval", "schema_version": 1,
        "harness": harness, "harness_process": harness_process, "models": combined,
        "input_receipts": identities})
    verify_downstream_receipt(output)
    return Path(output).resolve()

def verify_downstream_receipt(path, expected_models=None):
    payload = read_json(path)
    if payload.get("format") != "moeloop-release-lm-eval" or payload.get("schema_version") != 1:
        raise ValueError("unsupported downstream receipt")
    harness = read_json(verify_file(payload["harness"]))
    if harness.get("version") != HARNESS_VERSION:
        raise ValueError("downstream harness version mismatch")
    for name in ("tasks/arc/arc_easy.yaml", "tasks/hellaswag/hellaswag.yaml", "tasks/hellaswag/utils.py"):
        if len(harness.get("files", {}).get(name, "")) != 64:
            raise ValueError("missing exact task YAML/helper hash")
    verify_process(payload["harness_process"])
    for record in payload.get("input_receipts", []):
        verify_file(record)
    if expected_models is not None and set(payload["models"]) != set(expected_models):
        raise ValueError("downstream receipt does not contain the required endpoints")
    if not payload["models"]:
        raise ValueError("empty downstream receipt")
    common_task_hashes = None
    for name, row in payload["models"].items():
        if isinstance(expected_models, dict) and Path(row["model"]["path"]).resolve() != Path(expected_models[name]).resolve():
            raise ValueError("downstream receipt model differs from the required endpoint")
        if Path(row["export_provenance"]["path"]).resolve() != Path(row["model"]["path"]).resolve() / "export_provenance.json":
            raise ValueError("downstream provenance does not belong to the evaluated model")
        verify_tree(row["model"])
        verify_file(row["export_provenance"])
        result = read_json(verify_file(row["result"]))
        verify_lm_eval_result(result)
        task_hashes = {task: result["task_hashes"][task] for task in TASK_SAMPLES}
        if common_task_hashes is not None and task_hashes != common_task_hashes:
            raise ValueError("native downstream task hashes differ across models")
        common_task_hashes = task_hashes
        for task in TASK_SAMPLES:
            verify_file(row["sample_logs"][task])
        verify_process(row["process"])
        argv = row["process"]["argv"]
        if "--log_samples" not in argv:
            raise ValueError("formal downstream requires native sample logging")
        if any(arg == "--limit" or arg.startswith("--limit=") or arg == "--samples" or arg.startswith("--samples=") for arg in argv):
            raise ValueError("downstream process limited task samples")
        required_args = {"--model": "hf", "--model_args": f"pretrained={Path(row['model']['path']).resolve()},trust_remote_code=True,dtype=float32",
                         "--tasks": ",".join(TASK_SAMPLES), "--num_fewshot": "0"}
        if any(key not in argv or argv.index(key) + 1 >= len(argv) or argv[argv.index(key) + 1] != value for key, value in required_args.items()):
            raise ValueError("downstream command does not bind the required model/tasks/0-shot/float32 dtype")
        if row.get("num_fewshot") != 0 or row.get("limit", "missing") is not None or row.get("readout") != "terminal":
            raise ValueError("downstream protocol differs from frozen terminal 0-shot evaluation")
        if row.get("dtype") != "float32":
            raise ValueError("formal downstream requires fixed float32 dtype")
        config = read_json(Path(row["model"]["path"]) / "config.json")
        readout_basis = _terminal_readout_basis(config)
        if config["model_config"]["num_visits"] != row["num_visits"] or row.get("readout_basis", readout_basis) != readout_basis:
            raise ValueError("downstream model depth/readout differs from receipt")
    return payload



def journey(package_receipt, spec, manifest, output, *, train_python, hf_python,
            gpus=1, stop=4, device="cuda", seq_len=128, uv="uv", prepare_request=None):
    """Train real native updates, restore full DCP, export and consume the wheel.

    The training environment reuses caller-supplied dependencies, explicitly
    recorded in the install proof. The independent HF interpreter must contain
    torch/transformers but neither this package nor TorchTitan/tracking services.
    No random or constructed checkpoint substitutes for native training.
    """
    built = verify_package_receipt(package_receipt)
    spec = Path(spec).resolve()
    if (manifest is None) == (prepare_request is None):
        raise ValueError("supply exactly one of manifest or prepare_request")
    if manifest is not None:
        manifest = Path(manifest).resolve()
        if manifest.is_dir():
            manifest = manifest / "manifest.json"
    resolved = read_json(spec)
    horizon = resolved["run"]["successful_updates"]
    if not 0 < stop < horizon or gpus < 1:
        raise ValueError("journey needs 0 < stop < successful_updates and gpus >= 1")
    output = _new_output(output, built["source"]["path"])
    processes = []
    environment = output / "train-env"
    processes.append(run([uv, "venv", "--python", train_python, "--system-site-packages", environment], output, "train-venv"))
    python = environment / "bin/python"
    dependency_paths = output / "training-dependency-paths.json"
    code = "import json,pathlib,site,sys;pathlib.Path(sys.argv[1]).write_text(json.dumps(site.getsitepackages()))"
    processes.append(run([train_python, "-I", "-B", "-c", code, dependency_paths], output, "training-dependencies"))
    # A new venv based on another venv does not inherit that venv's packages.
    # Add only the explicitly supplied dependency sites, after the wheel's own
    # site-packages; the proof below requires the project itself to live here.
    dependency_sites = read_json(dependency_paths)
    # This local .pth runs under -I. Disable bytecode before caller hooks can
    # import modules through reused dependency paths and write into shared installs.
    code = r"import pathlib,site,sys;pathlib.Path(site.getsitepackages()[0], 'caller_dependencies.pth').write_text('import sys; sys.dont_write_bytecode = True; import site; ' + '; '.join('site.addsitedir(' + repr(p) + ')' for p in sys.argv[1:]) + '\n')"
    processes.append(run([python, "-I", "-c", code, *dependency_sites], output, "link-training-dependencies"))
    processes.append(run([uv, "pip", "install", "--python", python, "--no-deps", "--reinstall-package", "moeloop-lab", built["wheel"]["path"]], output, "train-install"))
    installed = output / "installed.json"
    code = r"""import hashlib,importlib.metadata,json,pathlib,sys
import moeloop
root=pathlib.Path(moeloop.__file__).parent
assert root.is_relative_to(sys.prefix),f'project not installed in journey venv: {root}'
assert sys.dont_write_bytecode is True
files={str(p.relative_to(root.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}
versions={name:importlib.metadata.version(name) for name in ('moeloop-lab','torch','transformers','tokenizers','safetensors','torchtitan')}
pathlib.Path(sys.argv[1]).write_text(json.dumps({'prefix':sys.prefix,'module':moeloop.__file__,'files':files,'versions':versions,'dependency_bytecode_writes':False,'dependencies':'caller supplied dependency sites via explicit pth; exact project wheel installed without dependencies'},sort_keys=True))
"""
    processes.append(run([python, "-I", "-c", code, installed], output, "installed-proof", env=clean_env(offline=True)))
    installed_files = read_json(installed)["files"]
    expected_modules = {name: value for name, value in built["wheel_modules"].items() if name.startswith("moeloop/")}
    if installed_files != expected_modules:
        raise ValueError("installed project modules differ from the exact release wheel")
    preparation = None
    if prepare_request is not None:
        request = read_json(prepare_request)
        allowed = {"recipe", "input", "validation", "test", "source_name", "text_field", "tokenizer",
                   "tokenizer_id", "tokenizer_revision", "bos_token_id", "eos_token_id", "shard_tokens"}
        if not request or set(request) - allowed:
            raise ValueError("prepare request must use public moeloop.prepare CLI argument names")
        prepare_argv = [python, "-I", "-m", "moeloop.prepare", "--output", str(output / "prepared")]
        inputs = {}
        for key, value in request.items():
            if key in ("input", "validation", "test", "tokenizer"):
                value = Path(value)
                if not value.is_absolute():
                    raise ValueError("prepare input/tokenizer paths must be absolute")
                inputs[key] = tree_record(value) if value.is_dir() else file_record(value)
            prepare_argv.extend(("--" + key.replace("_", "-"), str(value)))
        processes.append(run(prepare_argv, output, "prepare", env=clean_env()))
        manifest = output / "prepared" / "manifest.json"
        preparation = {"request": file_record(prepare_request), "inputs": inputs,
                       "prepared": tree_record(output / "prepared")}
    native = output / "native"
    native.mkdir()
    native_env = clean_env(offline=True)
    native_env.update(MOELOOP_EXPERIMENT_SPEC=str(spec), MOELOOP_EXPERIMENT_MANIFEST=str(manifest),
        MOELOOP_EXPERIMENT_OUTPUT=str(native), MOELOOP_DATA=str(manifest.parent))
    # The caller owns CUDA_VISIBLE_DEVICES and any ordinary torchrun networking.
    # This short installation journey is a single-node launch in assigned GPUs.
    argv = [python, "-I", "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={gpus}",
            "-m", "moeloop.train", "--module", "moeloop.experiment", "--config", "experiment"]
    native_env["MOELOOP_EXPERIMENT_STOP"] = str(stop)
    native_env.pop("MOELOOP_EXPERIMENT_RESUME", None)
    processes.append(run(argv, output, "native-stop", env=native_env))
    stopped = read_json(native / "outcome.json")
    write_json(output / "stop-outcome.json", stopped)
    checkpoint = Path(stopped["full_checkpoint"])
    native_env.update(MOELOOP_EXPERIMENT_STOP=str(horizon), MOELOOP_EXPERIMENT_RESUME=str(checkpoint))
    processes.append(run(argv, output, "native-resume", env=native_env))
    finished = read_json(native / "outcome.json")
    final_dcp = Path(finished["full_checkpoint"])
    # Read trusted checkpoints created above through the installed Torch reader.
    # Publish only key names; the checkpoint remains a normal full native DCP.
    state_proof = output / "full-state.json"
    code = r"""import json,pathlib,sys
import torch.distributed.checkpoint as dcp
rows=[]
for value in sys.argv[2:]:
 keys=sorted(dcp.FileSystemReader(value).read_metadata().state_dict_metadata)
 for prefix in ('core.','optimizer.','lr_scheduler.','train_state.rank_','dataloader.dp_rank_'):
  assert any(name.startswith(prefix) for name in keys),(value,prefix)
 rows.append({'checkpoint':value,'state_keys':keys})
pathlib.Path(sys.argv[1]).write_text(json.dumps(rows,sort_keys=True))
"""
    processes.append(run([python, "-I", "-c", code, state_proof, checkpoint, final_dcp], output, "full-state", env=clean_env(offline=True)))
    hf = output / "hf"
    processes.append(run([python, "-I", "-m", "moeloop.export", "--checkpoint", final_dcp,
        "--tokenizer", manifest.parent / "tokenizer", "--output", hf], output, "export", env=clean_env(offline=True)))
    depth = read_json(hf / "config.json")["model_config"]["num_visits"]
    heldout = output / "heldout.json"
    processes.append(run([python, "-I", "-m", "moeloop.evaluate", "--model", hf,
        "--manifest", manifest, "--split", "validation", "--seq-len", str(seq_len),
        "--num-visits", str(depth), "--device", device, "--output", heldout], output, "heldout", env=clean_env(offline=True)))
    standalone = output / "standalone-hf.json"
    # The HF consumer imports only Transformers/Torch, loads exported local code,
    # then proves deterministic fixed-depth greedy generation without a checkout.
    code = r"""import importlib.util,json,pathlib,sys
for name in ('moeloop','torchtitan','mlflow'):
 assert importlib.util.find_spec(name) is None,f'HF-only interpreter contains {name}'
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
artifact,output,device=sys.argv[1:]
model=AutoModelForCausalLM.from_pretrained(artifact,trust_remote_code=True,local_files_only=True).to(device).eval()
tokenizer=AutoTokenizer.from_pretrained(artifact,trust_remote_code=True,local_files_only=True)
ids=tokenizer.encode('The model',add_special_tokens=False)
if tokenizer.bos_token_id is not None: ids.insert(0,tokenizer.bos_token_id)
ids=ids[:min(8,model.config.max_position_embeddings-2)]
assert ids
inputs=torch.tensor([ids],device=device)
with torch.inference_mode():
 logits=model(inputs,use_cache=False).logits
 assert torch.isfinite(logits).all()
 kw=dict(attention_mask=torch.ones_like(inputs),max_new_tokens=2,do_sample=False,use_cache=False)
 first=model.generate(inputs,**kw)
 second=model.generate(inputs,**kw)
 assert torch.equal(first,second)
assert not any(name=='moeloop' or name.startswith('moeloop.') for name in sys.modules)
pathlib.Path(output).write_text(json.dumps({'offline':True,'repository_package_imported':False,'training_services_available':False,'readout':'terminal','num_visits':model.config.model_config['num_visits'],'input_ids':ids,'generated_ids':first.cpu().tolist(),'text':tokenizer.decode(first[0],skip_special_tokens=True),'deterministic_greedy':True,'python':sys.executable}))
"""
    processes.append(run([hf_python, "-I", "-c", code, hf, standalone, device], output, "standalone-hf", env=clean_env(offline=True)))
    receipt = {"format": "moeloop-release-journey", "schema_version": 1,
        "package_receipt": file_record(package_receipt), "spec": file_record(spec), "preparation": preparation,
        "manifest": file_record(manifest), "installed": file_record(installed), "dependency_paths": file_record(dependency_paths),
        "stop": stop, "horizon": horizon, "gpus": gpus, "stop_outcome": file_record(output / "stop-outcome.json"),
        "final_outcome": file_record(native / "outcome.json"), "full_state": file_record(state_proof),
        "resolved_config": file_record(native / "resolved_config.json"),
        "continuation_identity": file_record(native / "continuation_identity.json"),
        "stop_checkpoint": tree_record(checkpoint), "final_checkpoint": tree_record(final_dcp),
        "hf": tree_record(hf), "heldout": file_record(heldout), "standalone": file_record(standalone),
        "processes": processes}
    write_json(output / "journey-receipt.json", receipt)
    verify_journey_receipt(output / "journey-receipt.json")
    return output / "journey-receipt.json"


def verify_journey_receipt(path):
    payload = read_json(path)
    if payload.get("format") != "moeloop-release-journey" or payload.get("schema_version") != 1:
        raise ValueError("unsupported installed journey receipt")
    built = verify_package_receipt(verify_file(payload["package_receipt"]))
    for key in ("spec", "manifest", "installed", "dependency_paths", "stop_outcome", "final_outcome", "full_state", "resolved_config", "continuation_identity", "heldout", "standalone"):
        verify_file(payload[key])
    for key in ("stop_checkpoint", "final_checkpoint", "hf"):
        verify_tree(payload[key])
    for process in payload["processes"]:
        verify_process(process)
        cwd = Path(process["cwd"]).resolve()
        if cwd == Path(built["source"]["path"]) or Path(built["source"]["path"]) in cwd.parents:
            raise ValueError("installed journey ran inside the source checkout")
    preparation = payload.get("preparation")
    if preparation is not None:
        verify_file(preparation["request"])
        verify_tree(preparation["prepared"])
        if Path(payload["manifest"]["path"]).resolve() != Path(preparation["prepared"]["path"]) / "manifest.json":
            raise ValueError("journey did not consume its actual installed prepare output")
        for row in preparation["inputs"].values():
            (verify_tree if "files" in row else verify_file)(row)
        if not any("moeloop.prepare" in process["argv"] for process in payload["processes"]):
            raise ValueError("missing installed public prepare process")
    installed = read_json(payload["installed"]["path"])
    if installed.get("dependency_bytecode_writes") is not False:
        raise ValueError("installed journey does not protect reused dependency bytecode")
    if installed.get("versions", {}).get("moeloop-lab") != "1.0.0rc1" or installed["files"] != {
            k: v for k, v in built["wheel_modules"].items() if k.startswith("moeloop/")}:
        raise ValueError("journey did not use the release wheel modules")
    spec = read_json(payload["spec"]["path"])
    if spec["run"]["successful_updates"] != payload["horizon"] or not 0 < payload["stop"] < payload["horizon"]:
        raise ValueError("journey changed the frozen schedule/stop identity")
    stopped = read_json(payload["stop_outcome"]["path"])
    finished = read_json(payload["final_outcome"]["path"])
    for outcome, steps in ((stopped, payload["stop"]), (finished, payload["horizon"])):
        if outcome.get("status") != "completed" or outcome.get("completed_updates") != steps or outcome.get("global_progress", {}).get("label_tokens", 0) <= 0:
            raise ValueError("journey lacks actual completed native training")
    if not stopped.get("continuation_fingerprint") or stopped["continuation_fingerprint"] != finished.get("continuation_fingerprint"):
        raise ValueError("native full-state resume changed continuation identity")
    if finished["global_progress"]["label_tokens"] <= stopped["global_progress"]["label_tokens"]:
        raise ValueError("native resume did not advance training labels")
    for outcome, key in ((stopped, "stop_checkpoint"), (finished, "final_checkpoint")):
        if Path(outcome["full_checkpoint"]).resolve() != Path(payload[key]["path"]):
            raise ValueError("native outcome checkpoint differs from the retained DCP")
    continuation = read_json(payload["continuation_identity"]["path"])
    manifest = read_json(payload["manifest"]["path"])
    manifest_fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if continuation.get("manifest") != manifest_fingerprint or continuation.get("recipe_metadata") != spec:
        raise ValueError("native continuation does not bind the exact data/recipe")
    if continuation.get("moeloop_source") != {Path(name).name: value for name, value in installed["files"].items()}:
        raise ValueError("native execution source differs from the installed release wheel")
    config = read_json(payload["resolved_config"]["path"])
    if config["lr_scheduler"]["total_steps"] != spec["run"]["schedule_total_steps"]:
        raise ValueError("native installed journey changed the frozen schedule horizon")
    if Path(config["checkpoint"]["initial_load_path"]).resolve() != Path(payload["stop_checkpoint"]["path"]):
        raise ValueError("native resumed config does not load the actual stop checkpoint")
    for record, key in zip(read_json(payload["full_state"]["path"]), ("stop_checkpoint", "final_checkpoint"), strict=True):
        if Path(record["checkpoint"]).resolve() != Path(payload[key]["path"]):
            raise ValueError("full-state proof does not bind the real DCP")
        for prefix in ("core.", "optimizer.", "lr_scheduler.", "train_state.rank_", "dataloader.dp_rank_"):
            if not any(name.startswith(prefix) for name in record["state_keys"]):
                raise ValueError("DCP lacks a required full-state component")
    heldout = read_json(payload["heldout"]["path"])
    if heldout.get("complete_split") is not True or heldout.get("max_batches") is not None or heldout.get("valid_tokens", 0) <= 0:
        raise ValueError("installed heldout is incomplete")
    if heldout.get("manifest_fingerprint") != manifest_fingerprint:
        raise ValueError("heldout used a different prepared data manifest")
    standalone = read_json(payload["standalone"]["path"])
    if standalone.get("offline") is not True or standalone.get("repository_package_imported") is not False or standalone.get("training_services_available") is not False or standalone.get("deterministic_greedy") is not True:
        raise ValueError("standalone HF proof is incomplete")
    depth = read_json(Path(payload["hf"]["path"]) / "config.json")["model_config"]["num_visits"]
    if standalone.get("readout") != "terminal" or standalone.get("num_visits") != depth or heldout.get("num_visits") != depth:
        raise ValueError("journey evaluation depth/readout differs from exported model")
    provenance = read_json(Path(payload["hf"]["path"]) / "export_provenance.json")
    if provenance["checkpoint"].get("format") != "dcp" or provenance["checkpoint"].get("metadata_sha256") != payload["final_checkpoint"]["files"].get(".metadata"):
        raise ValueError("HF export does not come from the actual trained final DCP")
    return payload

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("package", help="source archive → sdist → rebuilt wheel → clean base import")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--source-identity", type=Path, required=True)
    build.add_argument("--python", default=sys.executable)
    build.add_argument("--uv", default="uv")
    evaluate = commands.add_parser("downstream", help="full pinned arc_easy+hellaswag; never accepts --limit")
    evaluate.add_argument("--model", action="append", required=True, metavar="NAME=HF_PATH")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--python", default=sys.executable)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=_lm_eval_batch_size, default=1, help="positive integer, auto or auto:N; native harness handles sizing")
    collect = commands.add_parser("collect-downstream", help="combine completed model receipts; no GPU or downloads")
    collect.add_argument("--receipt", type=Path, action="append", required=True)
    collect.add_argument("--output", type=Path, required=True)
    install = commands.add_parser("journey", help="installed native train/resume/DCP/offline HF journey")
    for name in ("package-receipt", "spec", "output"):
        install.add_argument("--" + name, type=Path, required=True)
    preparation = install.add_mutually_exclusive_group(required=True)
    preparation.add_argument("--manifest", type=Path)
    preparation.add_argument("--prepare-request", type=Path)
    install.add_argument("--train-python", required=True)
    install.add_argument("--hf-python", required=True)
    install.add_argument("--gpus", type=int, default=1)
    install.add_argument("--stop", type=int, default=4)
    install.add_argument("--device", default="cuda")
    install.add_argument("--seq-len", type=int, default=128)
    install.add_argument("--uv", default="uv")
    check = commands.add_parser("verify", help="read-only receipt validation")
    check.add_argument("--package", type=Path)
    check.add_argument("--downstream", type=Path)
    check.add_argument("--journey", type=Path)
    args = parser.parse_args(argv)
    if args.command == "package":
        print(package(args.source, args.output, source_identity=args.source_identity, python=args.python, uv=args.uv))
    elif args.command == "collect-downstream":
        print(collect_downstream(args.receipt, args.output))
    elif args.command == "journey":
        print(journey(args.package_receipt, args.spec, args.manifest, args.output,
            train_python=args.train_python, hf_python=args.hf_python, gpus=args.gpus,
            stop=args.stop, device=args.device, seq_len=args.seq_len, uv=args.uv, prepare_request=args.prepare_request))
    elif args.command == "downstream":
        models = {}
        for value in args.model:
            if "=" not in value:
                parser.error("--model must be NAME=HF_PATH")
            name, path = value.split("=", 1)
            if name in models:
                parser.error("model names must be unique")
            models[name] = path
        print(downstream(models, args.output, python=args.python, device=args.device, batch_size=args.batch_size))
    else:
        if not args.package and not args.downstream and not args.journey:
            parser.error("verify needs --package, --journey or --downstream")
        if args.package:
            verify_package_receipt(args.package)
        if args.downstream:
            verify_downstream_receipt(args.downstream)
        if args.journey:
            verify_journey_receipt(args.journey)
        print("receipt verification passed")


if __name__ == "__main__":
    main()
