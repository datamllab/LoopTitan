#!/usr/bin/env python3
"""Run recipe qualification on already visible GPUs; never allocate resources.

The same module supplies ``--module recipe_qualification --config qualify`` to
the existing native launcher. Failed attempts and their logs remain on disk.
Use --pilot for development geometry; pilot runs never satisfy release checks.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


METHODS = ("smelt", "meta", "foil", "expert_tying")
V2_METHODS = ("huginn", "loom", "ouro", "loom_attention")
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def snapshot_source(directory):
    """Keep the actual small source files, rather than an unverifiable git label."""
    directory = Path(directory)
    paths = (sorted((ROOT / "src/moeloop").glob("*.py"))
             + sorted((ROOT / "scripts").glob("recipe_*.py"))
             + [ROOT / "scripts" / name for name in ("check_release.py", "qualification.py")])
    files = {}
    for source in paths:
        relative = source.relative_to(ROOT)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[str(relative)] = digest(target)
    revision = ROOT / "source-revision.json"
    dirty = None
    if revision.is_file():
        identity = json.loads(revision.read_text())
        commit, dirty = identity.get("commit"), identity.get("dirty")
    elif (ROOT / ".git").exists():
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
        commit = result.stdout.strip() if result.returncode == 0 else None
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True)
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    else:
        commit = None
    return {"commit": commit, "dirty": dirty, "files": files}


def method_spec(method, *, vocab_size, seq_len, steps, global_batch_size):
    """Resolve the selected method path before entering TorchTitan."""
    from moeloop import recipes
    factories = {"smelt": recipes.smelt_spec, "meta": recipes.meta_spec,
                 "foil": recipes.foil_spec, "expert_tying": recipes.expert_tying_spec,
                 "huginn": recipes.huginn_spec, "loom": recipes.loom_spec,
                 "ouro": recipes.ouro_spec,
                 "loom_attention": lambda **geometry: recipes.loom_spec(attention_per_visit=True, **geometry)}
    if method not in factories:
        raise ValueError(f"unknown qualification method: {method}")
    spec = factories[method](vocab_size=vocab_size, dim=128, hidden_dim=256,
        num_heads=4, num_kv_heads=2, num_experts=4, max_seq_len=max(32, seq_len))
    return replace(spec, run=spec.run | {"seq_len": seq_len,
        "successful_updates": steps, "schedule_total_steps": steps,
        "global_batch_positions": seq_len * global_batch_size,
        "validation_interval": min(64, steps)})


def validate_run(*, steps, seq_len, global_batch_size, gpus, pilot):
    if min(steps, seq_len, global_batch_size, gpus) < 1 or global_batch_size % gpus:
        raise ValueError("positive steps/context/batch/ranks and batch divisible by GPU ranks are required")
    if not pilot:
        if steps < 512 or seq_len != 2048 or global_batch_size != 16:
            raise ValueError("use --pilot for development runs; method qualification requires >=512 updates, context 2048 and global batch 16")
        if gpus < 2:
            raise ValueError("method qualification requires at least two (2) FSDP ranks")


def qualification_batch(spec, world):
    """V2 uses the available H200 memory; historical V1 keeps its GA geometry."""
    batch = spec.run["global_batch_positions"] // spec.run["seq_len"]
    if type(world) is not int or world < 1 or batch % world:
        raise ValueError("qualification global batch must be divisible by world size")
    v2 = spec.name in V2_METHODS or spec.source.get("kind") == "attention_router_ownership_grid"
    local = batch // world if v2 else 1
    return local, batch // (world * local)


def qualify():
    """Native Python config factory. Horizon stays fixed across stop/resume."""
    from moeloop.recipe_spec import RecipeSpec
    from moeloop.recipes import trainer_config
    spec = RecipeSpec.from_dict(json.loads(Path(os.environ["MOELOOP_QUAL_SPEC"]).read_text()))
    manifest = Path(os.environ["MOELOOP_QUAL_MANIFEST"]).resolve()
    trainer = trainer_config(spec)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_batch, _ = qualification_batch(spec, world)
    trainer.dump_folder = os.environ["MOELOOP_QUAL_OUTPUT"]
    trainer.hf_assets_path = str(manifest.parent / "tokenizer")
    trainer.dataloader = replace(trainer.dataloader, manifest_path=str(manifest), num_workers=0)
    trainer.training = replace(trainer.training, steps=int(os.environ["MOELOOP_QUAL_STOP"]),
        local_batch_size=local_batch, global_batch_size=spec.run["global_batch_positions"] // spec.run["seq_len"],
        seq_len=spec.run["seq_len"], dtype="float32", mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32", disable_cuda_graphs=True)
    trainer.parallelism = replace(trainer.parallelism,
        data_parallel_replicate_degree=1, data_parallel_shard_degree=world)
    trainer.lr_scheduler = replace(trainer.lr_scheduler,
        total_steps=spec.run["schedule_total_steps"], warmup_steps=min(10, max(1, spec.run["schedule_total_steps"] // 8)))
    trainer.checkpoint = replace(trainer.checkpoint, interval=min(64, int(os.environ["MOELOOP_QUAL_STOP"])),
        keep_latest_k=2, initial_load_model_only=False, last_save_model_only=False,
        last_save_in_hf=False, async_mode="disabled",
        initial_load_path=os.environ.get("MOELOOP_QUAL_RESUME", trainer.checkpoint.initial_load_path))
    trainer.validator = replace(trainer.validator, freq=spec.run["validation_interval"], global_batches=2,
        dataloader=replace(trainer.validator.dataloader, manifest_path=str(manifest), num_workers=0))
    if os.environ.get("MOELOOP_QUAL_FULL_AC") == "1":
        from torchtitan.distributed.activation_checkpoint import FullAC
        trainer.activation_checkpoint = FullAC.Config()
    trainer.__post_init__()
    return trainer


def run_command(argv, directory, *, env=None, cwd=None, gpu_count=None, node_rank=None):
    """Keep the full log while showing bounded foreground progress."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = str(time.time_ns())
    log = directory / f"process-{stamp}.log"
    started = time.time()
    command = list(map(str, argv))
    print(f"[qualification] start {' '.join(command)}; log={log}", flush=True)
    tail = deque(maxlen=30)
    reported = set()
    with log.open("w") as stream:
        with subprocess.Popen(command, cwd=ROOT if cwd is None else cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                tail.append(line.rstrip())
                match = re.search(r"\bstep=(\d+)\s+CE=", line)
                if match:
                    step = int(match.group(1))
                    if (step == 1 or step % 64 == 0) and step not in reported:
                        print(line.rstrip(), flush=True)
                        reported.add(step)
            returncode = process.wait()
    metadata = {}
    if gpu_count is not None:
        metadata["gpu_count"] = gpu_count
    if node_rank is not None:
        metadata["node_rank"] = node_rank
    write_json(directory / f"process-{stamp}.json", {"argv": command, **metadata,
        "started_unix": started, "finished_unix": time.time(), "returncode": returncode,
        "log": str(log.resolve()), "log_sha256": digest(log)})
    print(f"[qualification] end returncode={returncode} elapsed={time.time() - started:.1f}s; log={log}", flush=True)
    if returncode:
        print("[qualification] final log lines:\n" + "\n".join(tail), flush=True)
        raise subprocess.CalledProcessError(returncode, argv)


def native_run(directory, spec, manifest, *, gpus, pilot, stop=None, resume=None, full_ac=False):
    """Invoke the existing training engine, preserving its full DCP and attempts."""
    import torch
    directory = Path(directory).resolve()
    if not directory.exists():
        directory.mkdir(parents=True)
        write_json(directory / "spec.json", spec.to_dict())
        local_batch, accumulation = qualification_batch(spec, gpus)
        write_json(directory / "request.json", {"method": spec.name, "pilot": pilot,
            "steps": spec.run["successful_updates"], "seq_len": spec.run["seq_len"],
            "global_batch_size": spec.run["global_batch_positions"] // spec.run["seq_len"],
            "world_size": gpus, "manifest": str(manifest.path),
            "manifest_fingerprint": manifest.fingerprint, "spec_sha256": digest(directory / "spec.json"),
            "source": snapshot_source(directory / "source"),
            "runtime": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda},
            "activation_checkpointing": "full" if full_ac else "none",
            **({"local_batch_size": local_batch, "gradient_accumulation_steps": accumulation}
               if spec.name in V2_METHODS or spec.source.get("kind") == "attention_router_ownership_grid" else {})})
    environment = os.environ.copy()
    environment.update(PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts"), environment.get("PYTHONPATH", "")]),
        MOELOOP_DATA=str(manifest.root), MOELOOP_QUAL_MANIFEST=str(manifest.path),
        MOELOOP_QUAL_SPEC=str(directory / "spec.json"), MOELOOP_QUAL_OUTPUT=str(directory),
        MOELOOP_QUAL_STOP=str(spec.run["successful_updates"] if stop is None else stop),
        MOELOOP_QUAL_FULL_AC="1" if full_ac else "0")
    environment.pop("MOELOOP_QUAL_RESUME", None)
    if resume is not None:
        environment["MOELOOP_QUAL_RESUME"] = str(Path(resume).resolve())
    run_command([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={gpus}",
        "-m", "moeloop.train", "--module", "recipe_qualification", "--config", "qualify"], directory, env=environment)
    from qualification import verify_receipt
    return verify_receipt(directory, expected_steps=spec.run["successful_updates"] if stop is None else stop)


def deliver_hf(directory, receipt, manifest, *, seq_len, device):
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts"), environment.get("PYTHONPATH", "")]))
    artifact = Path(directory) / "hf"
    run_command([sys.executable, "-m", "moeloop.export", "--checkpoint", receipt["checkpoint"],
        "--output", artifact, "--tokenizer", manifest.root / "tokenizer"], directory, env=environment)
    run_command([sys.executable, "-m", "moeloop.evaluate", "--model", artifact,
        "--manifest", manifest.path, "--output", Path(directory) / "heldout.json",
        "--seq-len", str(min(seq_len, 256)), "--max-batches", "4", "--device", device], directory, env=environment)
    environment.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="moeloop-hf-consumer-") as isolated:
        run_command([sys.executable, "-I", __file__, "--hf-check", artifact,
            "--hf-output", Path(directory) / "hf_check.json", "--device", device],
            directory, env=environment, cwd=isolated)


def hf_check(artifact, output, *, device):
    """An independent offline consumer loads only the exported HF model code."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    artifact = Path(artifact).resolve()
    model = AutoModelForCausalLM.from_pretrained(artifact, trust_remote_code=True,
                                               local_files_only=True).to(device).float().eval()
    tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
    tokens = tokenizer.encode("recursive models reuse a block", add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        tokens.insert(0, tokenizer.bos_token_id)
    tokens = tokens[:min(8, model.config.max_position_embeddings - 4)]
    if len(tokens) < 2:
        raise ValueError("HF cache check requires at least two prompt tokens")
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    cut = max(1, ids.size(1) // 2)
    cache_supported = bool(getattr(model.config, "use_cache", True))
    cache_result = {"supported": False, "reason": "Selected fixed-depth HF path explicitly disables cache."}
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
        options = dict(attention_mask=torch.ones_like(ids), max_new_tokens=4, do_sample=False)
        uncached = model.generate(ids, use_cache=False, **options)
        if cache_supported:
            first = model(ids[:, :cut], use_cache=True)
            rest = model(ids[:, cut:], past_key_values=first.past_key_values, use_cache=True).logits
            torch.testing.assert_close(rest, full[:, cut:], atol=1e-5, rtol=5e-5)
            cached = model.generate(ids, use_cache=True, **options)
            if not torch.equal(cached, uncached):
                raise AssertionError("cached and uncached greedy generation differ")
            cache_result = {"compared_tokens": ids.size(1) - cut,
                            "max_absolute_error": float((rest - full[:, cut:]).abs().max()),
                            "max_reference_absolute_value": float(full[:, cut:].abs().max()),
                            "atol": 1e-5, "rtol": 5e-5}
            generation = {"cached_ids": cached.cpu().tolist(), "uncached_ids": uncached.cpu().tolist()}
        else:
            repeated = model.generate(ids, use_cache=False, **options)
            if not torch.equal(uncached, repeated):
                raise AssertionError("uncached fixed-depth greedy recomputation differs")
            generation = {"uncached_ids": uncached.cpu().tolist(), "recomputed_ids": repeated.cpu().tolist()}
    if any(name == "moeloop" or name.startswith("moeloop.") for name in sys.modules):
        raise AssertionError("independent HF consumer imported the moeloop package")
    write_json(output, {"model_config": model.config.model_config, "repository_package_imported": False,
        "export_provenance_sha256": digest(artifact / "export_provenance.json"),
        "cache": cache_result,
        "generation": {"input_ids": ids.cpu().tolist(), **generation, "readout": "terminal",
                       "num_visits": model.config.model_config["num_visits"]}})


def sharing_specs(*, vocab_size, seq_len, steps, global_batch_size):
    """Four plain configurations; only attention/router ownership changes."""
    from moeloop.recipes import recursive_moe_spec
    base = recursive_moe_spec(vocab_size=vocab_size, dim=128, hidden_dim=256,
        num_heads=4, num_kv_heads=2, num_experts=4, max_seq_len=max(32, seq_len))
    return {
        f"a{attention}_r{router}": replace(base, name=f"sharing_a{attention}_r{router}",
            model=replace(base.model, attention_per_visit=bool(attention), router_per_visit=bool(router)),
            run=base.run | {"seq_len": seq_len, "successful_updates": steps,
                "schedule_total_steps": steps, "global_batch_positions": seq_len * global_batch_size,
                "validation_interval": min(64, steps)},
            source={"kind": "attention_router_ownership_grid", "attention_per_visit": bool(attention),
                    "router_per_visit": bool(router)})
        for attention in (0, 1) for router in (0, 1)
    }


def inherited_v1_reference(directory, source=None):
    """Bind already accepted V1 bytes, without reinterpreting its old schema."""
    from check_release import verify_inherited_v1
    directory = Path(directory).resolve()
    source = Path(source).resolve() if source else directory.parent / "frozen-p1-hsdp16/source.json"
    entry = {"directory": str(directory), "release_sha256": digest(directory / "release.json"),
             "release_check_sha256": digest(directory / "release-check.json"),
             "source_p1": str(source), "source_p1_sha256": digest(source)}
    verify_inherited_v1(entry)
    return entry


def run_v2_stage(args, manifest, gpus):
    from check_release import compare_optimizer
    from moeloop.costs import estimate_cost
    from qualification import compare_runs

    if args.v1_evidence is None:
        raise ValueError("V2 requires --v1-evidence for the accepted complete V1 bundle")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "release.json"
    inherited = inherited_v1_reference(args.v1_evidence, args.v1_source)
    index = json.loads(index_path.read_text()) if index_path.exists() else {
        "format": "moeloop-recipe-qualification", "version": 1, "stage": "v2",
        "manifest": str(manifest.path), "manifest_fingerprint": manifest.fingerprint,
        "inherited_v1": inherited, "methods": {}, "probes": {}, "sharing_grid": {}}
    if (index.get("stage") != "v2" or index.get("manifest_fingerprint") != manifest.fingerprint
            or index.get("inherited_v1") != inherited):
        raise ValueError("an evidence directory cannot mix stage/data/inherited V1 identities")
    try:
        for method in args.methods:
            spec = method_spec(method, vocab_size=manifest.tokenizer["vocab_size"], seq_len=args.seq_len,
                               steps=args.steps, global_batch_size=args.global_batch_size)
            spec = replace(spec, run=spec.run | {"manifest_identity": manifest.fingerprint})
            directory = output / method / str(time.time_ns())
            index["methods"].setdefault(method, []).append(str(directory))
            write_json(index_path, index)
            receipt = native_run(directory, spec, manifest, gpus=gpus, pilot=args.pilot)
            deliver_hf(directory, receipt, manifest, seq_len=args.seq_len, device=args.hf_device)

            probe = output / "continuation" / method / str(time.time_ns())
            one, two = probe / "uninterrupted", probe / "resumed"
            probe_spec = method_spec(method, vocab_size=manifest.tokenizer["vocab_size"],
                seq_len=args.seq_len, steps=8, global_batch_size=args.global_batch_size)
            probe_spec = replace(probe_spec, run=probe_spec.run | {"manifest_identity": manifest.fingerprint})
            index["probes"][method] = {"uninterrupted": str(one), "resumed": str(two),
                                       "report": str(probe / "comparison.json")}
            write_json(index_path, index)
            first = native_run(one, probe_spec, manifest, gpus=gpus, pilot=True)
            stopped = native_run(two, probe_spec, manifest, gpus=gpus, pilot=True, stop=4)
            last = native_run(two, probe_spec, manifest, gpus=gpus, pilot=True, resume=stopped["checkpoint"])
            report = compare_runs(one, two, expected_steps=8)
            report["optimizer"] = compare_optimizer(first["checkpoint"], last["checkpoint"])
            write_json(probe / "comparison.json", report)

        if args.sharing_grid:
            directory = output / "sharing-grid" / str(time.time_ns())
            specs = sharing_specs(vocab_size=manifest.tokenizer["vocab_size"], seq_len=args.seq_len,
                                  steps=8, global_batch_size=args.global_batch_size)
            index["sharing_grid"] = {"arms": {}, "comparison": str(directory / "comparison.json")}
            comparison = {"format": "moeloop-attention-router-grid", "version": 1,
                          "manifest_fingerprint": manifest.fingerprint, "seed": 42, "arms": {}}
            for name, spec in specs.items():
                spec = replace(spec, run=spec.run | {"manifest_identity": manifest.fingerprint})
                index["sharing_grid"]["arms"][name] = str(directory / name)
                write_json(index_path, index)
                native_run(directory / name, spec, manifest, gpus=gpus, pilot=True)
                comparison["arms"][name] = {"spec": spec.to_dict(),
                    "cost": estimate_cost(spec, seq_len=args.seq_len)}
            write_json(directory / "comparison.json", comparison)
    except BaseException as error:
        index["last_error"] = {"type": type(error).__name__, "message": str(error), "time_unix": time.time()}
        write_json(index_path, index)
        raise
    index.pop("last_error", None)
    write_json(index_path, index)
    print(index_path)


def run_stage(args):
    import torch
    from moeloop.data import PreparedManifest
    from moeloop.recipe_spec import RecipeSpec
    from moeloop.comparison import smelt_budget_pair, compare_recipes
    from check_release import compare_optimizer
    from qualification import compare_runs
    # The launcher needs metadata; every native run verifies the immutable
    # corpus on rank zero before training. Do not hash the full corpus twice.
    manifest = PreparedManifest.load(args.manifest, verify_hashes=False)
    gpus = torch.cuda.device_count() if args.gpus is None else args.gpus
    validate_run(steps=args.steps, seq_len=args.seq_len, global_batch_size=args.global_batch_size, gpus=gpus, pilot=args.pilot)
    if gpus > torch.cuda.device_count():
        raise ValueError("requested ranks exceed the GPUs already visible to this process")
    if args.stage == "v2":
        return run_v2_stage(args, manifest, gpus)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "release.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {
        "format": "moeloop-recipe-qualification", "version": 1, "stage": "v1",
        "manifest": str(manifest.path), "manifest_fingerprint": manifest.fingerprint,
        "methods": {}, "probes": {}, "pair": {}}
    if index.get("manifest_fingerprint") != manifest.fingerprint:
        raise ValueError("an evidence directory cannot mix prepared manifest identities")
    if index.get("stage") != "v1":
        raise ValueError("an evidence directory cannot mix release stages")
    try:
        for method in args.methods:
            spec = method_spec(method, vocab_size=manifest.tokenizer["vocab_size"], seq_len=args.seq_len,
                               steps=args.steps, global_batch_size=args.global_batch_size)
            spec = replace(spec, run=spec.run | {"manifest_identity": manifest.fingerprint})
            directory = output / method / str(time.time_ns())
            index["methods"].setdefault(method, []).append(str(directory))
            write_json(index_path, index)
            receipt = native_run(directory, spec, manifest, gpus=gpus, pilot=args.pilot, full_ac=method == "foil")
            deliver_hf(directory, receipt, manifest, seq_len=args.seq_len, device=args.hf_device)
            if method in ("foil", "expert_tying"):
                probe = output / "continuation" / method / str(time.time_ns())
                one, two = probe / "uninterrupted", probe / "resumed"
                probe_spec = method_spec(method, vocab_size=manifest.tokenizer["vocab_size"],
                    seq_len=args.seq_len, steps=8, global_batch_size=args.global_batch_size)
                probe_spec = replace(probe_spec, run=probe_spec.run | {"manifest_identity": manifest.fingerprint})
                index["probes"][method] = {"uninterrupted": str(one), "resumed": str(two), "report": str(probe / "comparison.json")}
                write_json(index_path, index)
                first = native_run(one, probe_spec, manifest, gpus=gpus, pilot=True, full_ac=method == "foil")
                stopped = native_run(two, probe_spec, manifest, gpus=gpus, pilot=True, stop=4, full_ac=method == "foil")
                last = native_run(two, probe_spec, manifest, gpus=gpus, pilot=True,
                    resume=stopped["checkpoint"], full_ac=method == "foil")
                report = compare_runs(one, two, expected_steps=8)
                report["optimizer"] = compare_optimizer(first["checkpoint"], last["checkpoint"])
                write_json(probe / "comparison.json", report)
        pair = smelt_budget_pair(vocab_size=manifest.tokenizer["vocab_size"], seq_len=args.seq_len)
        directory = output / "budget-pair" / str(time.time_ns())
        index["pair"] = {"contract": str(directory / "comparison.json")}
        arms = {}
        for role in ("reference", "candidate"):
            spec = RecipeSpec.from_dict(pair["comparison"]["arms"][role])
            spec = replace(spec, model=replace(spec.model, max_seq_len=max(2048, args.seq_len)),
                run=spec.run | {"seq_len": args.seq_len, "successful_updates": 8, "schedule_total_steps": 8,
                               "validation_interval": 8, "global_batch_positions": args.seq_len * args.global_batch_size,
                               "manifest_identity": manifest.fingerprint})
            arms[role] = spec
            index["pair"][role] = str(directory / role)
            write_json(index_path, index)
            native_run(directory / role, spec, manifest, gpus=gpus, pilot=True)
        pair["comparison"] = compare_recipes(arms["reference"], arms["candidate"], seq_len=args.seq_len,
            data={"manifest_fingerprint": manifest.fingerprint}, evaluation={"split": "validation", "manifest_fingerprint": manifest.fingerprint})
        write_json(directory / "comparison.json", pair)
    except BaseException as error:
        index["last_error"] = {"type": type(error).__name__, "message": str(error), "time_unix": time.time()}
        write_json(index_path, index)
        raise
    index.pop("last_error", None)
    write_json(index_path, index)
    print(index_path)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--hf-check" in argv:
        parser = argparse.ArgumentParser(description="Read an exported HF model in an independent offline process")
        parser.add_argument("--hf-check", type=Path, required=True)
        parser.add_argument("--hf-output", type=Path, required=True)
        parser.add_argument("--device", default="cpu")
        args = parser.parse_args(argv)
        hf_check(args.hf_check, args.hf_output, device=args.device)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("v1", "v2"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, help="number of already visible GPUs; default: all visible")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--methods", nargs="+", choices=METHODS + V2_METHODS)
    parser.add_argument("--v1-evidence", type=Path, help="accepted full V1 evidence directory; required for V2")
    parser.add_argument("--v1-source", type=Path, help="original frozen P1 source-identity JSON, if outside the default layout")
    parser.add_argument("--sharing-grid", action="store_true", help="run the four 8-update ownership qualifications")
    parser.add_argument("--sharing-grid-only", action="store_true", help="V2: run only the ownership grid, without repeating method qualifications")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--hf-device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.sharing_grid_only:
        if args.stage != "v2" or args.methods is not None:
            parser.error("--sharing-grid-only requires V2 and no --methods")
        args.methods, args.sharing_grid = [], True
    if args.methods is None:
        args.methods = list(METHODS if args.stage == "v1" else V2_METHODS)
        args.sharing_grid = args.sharing_grid or args.stage == "v2"
    if any(method not in (METHODS if args.stage == "v1" else V2_METHODS) for method in args.methods):
        parser.error("methods must belong to the selected release stage")
    run_stage(args)


if __name__ == "__main__":
    main()
