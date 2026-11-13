#!/usr/bin/env python3
"""Check source-bound V1/V2 qualification and V3 recipe release artifacts.

This command never launches training, exports a model or allocates GPUs. Native
DCP and continuation readers are reused from qualification.py; optimizer state
is additionally compared because V1 introduces expert-specific learning rates.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import pickle
import re
import statistics
import sys

from recipe_qualification import METHODS, V2_METHODS, digest


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_recipe(recorded, current):
    """Compare two recipe records by meaning, not key by key.

    A schema change can add fields with defaults (#29 took the model schema
    from 3 to 4), so this checkout's ``to_dict()`` has keys an archived record
    lacks. Both records go through ``RecipeSpec.from_dict``, which fills those
    in as the current code does; any value that differs still fails.
    """
    from moeloop.recipe_spec import RecipeSpec
    return RecipeSpec.from_dict(recorded) == RecipeSpec.from_dict(current)


def verify_source(directory, source):
    directory = Path(directory).resolve()
    files = source.get("files", {})
    required = {f"src/moeloop/{name}.py" for name in ("model", "moe", "cache", "training", "titan", "data", "recipes")}
    require(required <= files.keys(), "source snapshot lacks actual training modules")
    for relative, expected in files.items():
        path = (directory / relative).resolve()
        require(path.is_relative_to(directory) and path.is_file() and digest(path) == expected,
                f"source SHA256 differs or source file is missing: {relative}")


def compare_optimizer(first, second, *, atol=2e-5, rtol=2e-4):
    """Compare actual full-checkpoint optimizer tensors and scalar hyperparameters."""
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata, BytesStorageMetadata
    from qualification import _assert_equal_tree

    def load(path):
        reader = dcp.FileSystemReader(path)
        state = {}
        for name, item in reader.read_metadata().state_dict_metadata.items():
            if name.startswith("optimizer."):
                if isinstance(item, TensorStorageMetadata):
                    state[name] = torch.empty(item.size, dtype=item.properties.dtype, device="cpu")
                elif isinstance(item, BytesStorageMetadata):
                    state[name] = b""
                else:
                    raise TypeError(f"unknown optimizer DCP metadata: {name}")
        require(state, "full DCP contains no optimizer state")
        dcp.load(state, storage_reader=reader, no_dist=True)
        return state

    left, right = load(first), load(second)
    require(left.keys() == right.keys(), "optimizer continuation keys differ")
    count, maximum = 0, 0.
    for name, value in left.items():
        other = right[name]
        if isinstance(value, torch.Tensor):
            require(torch.isfinite(value).all() and torch.isfinite(other).all(), f"nonfinite optimizer tensor: {name}")
            if value.is_floating_point() and not name.endswith(".step"):
                torch.testing.assert_close(value, other, atol=atol, rtol=rtol,
                    msg=lambda message: f"optimizer continuation {name}: {message}")
            else:
                _assert_equal_tree(value, other, name)
            maximum = max(maximum, float((value.double() - other.double()).abs().max()))
            count += 1
        else:
            _assert_equal_tree(value, other, name)
    require(count > 0, "optimizer continuation contains no tensor moments")
    return {"tensor_count": count, "state_key_count": len(left), "max_absolute_error": maximum,
            "atol": atol, "rtol": rtol, "scalar_hyperparameters": "exact"}


def verify_processes(directory):
    records = sorted(Path(directory).glob("process-*.json"))
    require(records, f"{directory}: process receipts are missing")
    commands = []
    for path in records:
        value = read_json(path)
        log = Path(value["log"])
        require(value.get("returncode") == 0 and value.get("finished_unix", 0) >= value.get("started_unix", 1),
                f"{directory}: failed or incomplete process receipt")
        require(log.is_file() and digest(log) == value.get("log_sha256"), f"{directory}: process log hash differs")
        commands.append(value["argv"])
    return commands


def verify_training(directory, manifest, *, qualification=True, expected_steps=None):
    from qualification import verify_receipt
    from moeloop.model import ModelConfig
    from moeloop.recipe_spec import RecipeSpec
    from moeloop.training import continuation_fingerprint
    from recipe_qualification import validate_run

    directory = Path(directory)
    request = read_json(directory / "request.json")
    steps = request["steps"] if expected_steps is None else expected_steps
    require(request["steps"] == steps, f"{directory}: requested step identity differs")
    if qualification:
        require(request.get("pilot") is False, f"{directory}: pilot cannot satisfy method qualification")
        validate_run(steps=steps, seq_len=request["seq_len"], global_batch_size=request["global_batch_size"],
                     gpus=request["world_size"], pilot=False)
    require(request.get("manifest_fingerprint") == manifest.fingerprint, f"{directory}: manifest identity differs")
    require(digest(directory / "spec.json") == request.get("spec_sha256"), f"{directory}: resolved recipe hash differs")
    spec = RecipeSpec.from_dict(read_json(directory / "spec.json"))
    require(spec.name == request["method"] and spec.run["successful_updates"] == steps,
            f"{directory}: recipe or endpoint differs from request")
    require(spec.run["seq_len"] == request["seq_len"] and
            spec.run["global_batch_positions"] == request["seq_len"] * request["global_batch_size"],
            f"{directory}: recipe context/batch metadata is stale")
    verify_source(directory / "source", request["source"])
    receipt = verify_receipt(directory, expected_steps=steps)
    identity = read_json(directory / "continuation_identity.json")
    require(identity.get("manifest") == manifest.fingerprint and "model" in identity
            and ModelConfig.from_dict(identity["model"]) == spec.model, f"{directory}: native model/data identity differs")
    require("recipe_metadata" in identity and same_recipe(identity["recipe_metadata"], spec.to_dict()),
            f"{directory}: native recipe metadata differs")
    require("recipe" in identity
            and RecipeSpec.from_dict(spec.to_dict() | {"training": identity["recipe"]}).training == spec.training,
            f"{directory}: native objective/depth differs")
    require(identity.get("expert_lr_multiplier") == spec.optimizer["expert_lr_multiplier"],
            f"{directory}: expert learning-rate identity differs")
    require(continuation_fingerprint(identity) == receipt["outcome"]["continuation_fingerprint"],
            f"{directory}: continuation fingerprint differs")
    require({"model.py", "moe.py", "cache.py", "training.py", "titan.py", "data.py"} <= identity.get("moeloop_source", {}).keys(),
            f"{directory}: native source identity is incomplete")
    for name, value in identity.get("moeloop_source", {}).items():
        require(request["source"]["files"].get(f"src/moeloop/{name}") == value,
                f"{directory}: native code differs from source snapshot: {name}")
    require(identity.get("world_size") == request["world_size"] and identity.get("seq_len") == request["seq_len"],
            f"{directory}: native rank/context identity differs")
    if spec.name in V2_METHODS or spec.source.get("kind") == "attention_router_ownership_grid":
        require(request.get("local_batch_size") == identity.get("local_batch_size")
                and request.get("gradient_accumulation_steps") == identity.get("accumulation")
                and request["local_batch_size"] * request["gradient_accumulation_steps"] * request["world_size"]
                    == request["global_batch_size"], f"{directory}: actual local/global batch or accumulation differs")
    if qualification:
        require(identity.get("precision", [None, None])[1] == "bfloat16" and request["runtime"].get("cuda"),
                f"{directory}: missing actual BF16/CUDA runtime")
        require(identity["parallelism"].get("data_parallel_shard_degree", 0) >= 2,
                f"{directory}: replicated DP is not sharded FSDP evidence")
        if spec.name == "foil":
            require(identity["execution"].get("activation_checkpoint") is not None,
                    f"{directory}: heterogeneous/per-visit attention path lacks Full AC evidence")
    require(ModelConfig.from_dict(read_json(directory / "model_config.json")) == spec.model,
            f"{directory}: saved model config differs")
    resolved = read_json(directory / "resolved_config.json")
    require(resolved["training"]["steps"] == steps and resolved["lr_scheduler"]["total_steps"] == spec.run["schedule_total_steps"],
            f"{directory}: native stop target or fixed schedule horizon differs")
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
    require([item["step"] for item in metrics] == list(range(1, steps + 1)), f"{directory}: successful-update metric history has gaps/duplicates")
    for item in metrics:
        require(math.isfinite(item["ce"]) and math.isfinite(item["aux"]), f"{directory}: nonfinite training objective")
        if spec.name != "huginn":
            require(item["depth"] == spec.training.num_visits and item["grad_visits"] == spec.training.num_visits,
                    f"{directory}: fixed-depth/full-BPTT semantics differ")
    progress = receipt["outcome"]["global_progress"]
    require(progress["padded_positions"] == steps * request["seq_len"] * request["global_batch_size"],
            f"{directory}: actual packed-position budget differs")
    require(0 < progress["label_tokens"] <= progress["input_tokens"] <= progress["padded_positions"],
            f"{directory}: invalid input/label/position accounting")
    validations = [json.loads(line) for line in (directory / "validation.jsonl").read_text().splitlines() if line.strip()]
    interval = min(64, steps) if qualification else int(spec.run.get("validation_interval", min(64, steps)))
    expected = set(range(interval, steps + 1, interval)) | {steps}
    require(expected <= {item["step"] for item in validations}, f"{directory}: required real heldout validations are missing")
    for item in validations:
        require(item["valid_tokens"] > 0 and math.isfinite(item["ce"]) and item["manifest_fingerprint"] == manifest.fingerprint,
                f"{directory}: invalid heldout observation or data identity")
        require(item["num_visits"] == spec.model.num_visits, f"{directory}: heldout inference depth differs")
    commands = verify_processes(directory)
    require(any("moeloop.train" in command for command in commands), f"{directory}: native training invocation is missing")
    result = {"receipt": receipt, "request": request, "spec": spec, "identity": identity, "resolved": resolved,
              "steps": steps, "heldout_points": len(validations), "metrics": metrics, "validations": validations}
    if spec.name in V2_METHODS:
        helper = "latent" if spec.name == "huginn" else "halting" if spec.name == "ouro" else "memory"
        require(f"src/moeloop/{helper}.py" in request["source"]["files"],
                f"{directory}: dynamic method source snapshot is incomplete")
        result["dynamics"] = verify_v2_dynamics(result)
    return result


def verify_hf(directory, training):
    from moeloop.model import ModelConfig
    directory = Path(directory)
    artifact = directory / "hf"
    provenance = read_json(artifact / "export_provenance.json")
    checkpoint = Path(training["receipt"]["checkpoint"])
    require(provenance["checkpoint"].get("format") == "dcp" and
            provenance["checkpoint"].get("metadata_sha256") == digest(checkpoint / ".metadata"),
            f"{directory}: HF artifact does not identify the actual final native DCP")
    require(ModelConfig.from_dict(provenance["model_config"]) == training["spec"].model, f"{directory}: HF model config differs")
    hashes = provenance.get("artifact_sha256", {})
    require("config.json" in hashes and any(name.endswith(".safetensors") for name in hashes),
            f"{directory}: exported HF weights/config are missing")
    for name, expected in hashes.items():
        require((artifact / name).is_file() and digest(artifact / name) == expected,
                f"{directory}: HF artifact hash differs: {name}")
    for name in ("model.py", "moe.py", "cache.py"):
        require(hashes.get(name) == training["request"]["source"]["files"][f"src/moeloop/{name}"],
                f"{directory}: exported model math differs from native training source: {name}")
    for name, expected in hashes.items():
        source_name = f"src/moeloop/{name}"
        if name.endswith(".py") and source_name in training["request"]["source"]["files"]:
            require(expected == training["request"]["source"]["files"][source_name],
                    f"{directory}: exported helper math differs from native source: {name}")
    result = read_json(directory / "hf_check.json")
    require(result["export_provenance_sha256"] == digest(artifact / "export_provenance.json"), f"{directory}: independent HF source identity differs")
    require(ModelConfig.from_dict(result["model_config"]) == training["spec"].model,
            f"{directory}: independent HF model config differs")
    require(result.get("repository_package_imported") is False, f"{directory}: HF consumer was not isolated from the repository package")
    heldout = read_json(directory / "heldout.json")
    require(heldout["valid_tokens"] > 0 and math.isfinite(heldout["ce"]) and
            heldout["manifest_fingerprint"] == training["request"]["manifest_fingerprint"], f"{directory}: HF heldout result is missing or invalid")
    require(heldout["num_visits"] == training["spec"].model.num_visits, f"{directory}: HF heldout depth differs")
    cache_error = verify_hf_generation(result, training["spec"])
    commands = verify_processes(directory)
    require(any("--hf-check" in command and "-I" in command for command in commands) and
            any("moeloop.export" in command for command in commands) and any("moeloop.evaluate" in command for command in commands),
            f"{directory}: independent HF/export process receipts are missing")
    return {"valid_tokens": heldout["valid_tokens"], "ce": heldout["ce"], "cache_max_absolute_error": cache_error}


def verify_hf_generation(result, spec):
    """Preserve exact-cache checks, or accept an explicitly uncached path."""
    cache, generation = result["cache"], result["generation"]
    generated = generation.get("uncached_ids", [])
    inputs = generation.get("input_ids", [])
    require(generated and inputs and len(generated) == len(inputs)
            and all(len(output) > len(prompt) and output[:len(prompt)] == prompt
                    for output, prompt in zip(generated, inputs))
            and generation.get("readout") == "terminal"
            and generation.get("num_visits") == spec.model.num_visits,
            "fixed-depth generation identity or output is missing")
    if cache.get("supported") is False:
        require(spec.inference.get("cache") in ("none", "uncached")
                and isinstance(cache.get("reason"), str) and cache["reason"].strip(),
                "uncached HF delivery requires explicit cache support/adaptation metadata")
        require(generated == generation.get("recomputed_ids"), "uncached fixed-depth generation recomputation differs")
        return None
    require(cache.get("compared_tokens", 0) > 0 and cache.get("atol", math.inf) <= 1e-5
            and cache.get("rtol", math.inf) <= 5e-5 and math.isfinite(cache.get("max_absolute_error", math.nan))
            and cache["max_absolute_error"] <= cache["atol"] + cache["rtol"] * cache["max_reference_absolute_value"],
            "exact-cache parity is missing or failed")
    require(generation.get("cached_ids") == generated, "cached/uncached fixed-depth generation parity failed")
    return cache["max_absolute_error"]


def verify_engineering_release(directory):
    from moeloop.data import PreparedManifest
    from moeloop.comparison import compare_recipes
    from qualification import compare_runs
    directory = Path(directory).resolve()
    index = read_json(directory / "release.json")
    require(index.get("format") == "moeloop-recipe-qualification" and index.get("version") == 1 and index.get("stage") == "v1",
            "unsupported or missing release evidence format/schema")
    require(set(METHODS) <= index.get("methods", {}).keys(), "required method artifacts are missing")
    manifest = PreparedManifest.load(index["manifest"])
    require(index.get("manifest_fingerprint") == manifest.fingerprint, "release manifest identity differs")
    provenance = manifest.payload.get("provenance", {})
    require(provenance.get("kind") == "public_document_subset", "method qualification requires the fixed public text fixture")
    dataset = provenance.get("dataset", {})
    require(dataset.get("id") and dataset.get("revision") and dataset.get("license"), "public dataset source/revision/license are missing")
    require(provenance.get("selection", {}).get("deduplication") and
            all(manifest.source_names(split) for split in ("train", "validation", "test")), "isolated train/validation/test fixture evidence is missing")
    methods = {}
    for name in METHODS:
        attempts = index["methods"][name]
        require(isinstance(attempts, list) and attempts, f"{name}: no real attempt directory")
        training = verify_training(attempts[-1], manifest)
        require(training["spec"].name == name, f"{name}: method recipe identity differs")
        methods[name] = {"updates": training["steps"], "heldout_points": training["heldout_points"],
                         "hf": verify_hf(attempts[-1], training)}
    probes = {}
    for name in ("foil", "expert_tying"):
        require(name in index.get("probes", {}), f"{name}: strict continuation probe is missing")
        probe = index["probes"][name]
        one = verify_training(probe["uninterrupted"], manifest, qualification=False, expected_steps=8)
        two = verify_training(probe["resumed"], manifest, qualification=False, expected_steps=8)
        require(one["spec"].name == two["spec"].name == name, f"{name}: continuation recipe differs")
        resumed = Path(probe["resumed"])
        stop = [read_json(path) for path in resumed.glob("attempts/*/outcome.json")]
        require(any(value.get("status") == "completed" and value.get("completed_updates") == 4 for value in stop),
                f"{name}: stopped 4-update native attempt is missing")
        load = two["resolved"]["checkpoint"]
        require(load.get("initial_load_path") and Path(load["initial_load_path"]).name == "step-4" and
                not load.get("initial_load_model_only"), f"{name}: full state was not restored from step 4")
        compared = compare_runs(probe["uninterrupted"], probe["resumed"], expected_steps=8)
        compared["optimizer"] = compare_optimizer(one["receipt"]["checkpoint"], two["receipt"]["checkpoint"])
        original = read_json(probe["report"])
        require(original.get("status") == "passed" and original.get("expected_steps") == 8 and
                original.get("optimizer", {}).get("tensor_count", 0) > 0,
                f"{name}: original continuation comparison record is missing or failed")
        probes[name] = {"model_tensors": compared["model_tensor_count"], "optimizer": compared["optimizer"],
                        "training_ranks": compared["training_rank_count"], "data_ranks": compared["data_rank_count"]}
    pair = index.get("pair", {})
    require(all(pair.get(key) for key in ("reference", "candidate", "contract")), "budget-pair pilot artifacts are missing")
    reference = verify_training(pair["reference"], manifest, qualification=False, expected_steps=8)
    candidate = verify_training(pair["candidate"], manifest, qualification=False, expected_steps=8)
    require(reference["spec"].name == "moe" and candidate["spec"].name == "smelt", "budget-pair arms differ")
    contract = read_json(pair["contract"])
    require(0 < contract["candidate_count"] <= 32, "budget-pair screen exceeds the finite candidate bound")
    compared = compare_recipes(reference["spec"], candidate["spec"], seq_len=reference["request"]["seq_len"],
        data={"manifest_fingerprint": manifest.fingerprint}, evaluation={"split": "validation", "manifest_fingerprint": manifest.fingerprint})
    verify_legacy_comparison(compared, contract["comparison"])
    require({"unique_non_embedding_parameters", "training_flops_per_token", "logical_kv_bytes"} <= compared["matched"].keys(),
            "budget pair does not match the three declared estimates")
    return {"status": "passed", "stage": "v1", "scope": "four_method_engineering", "manifest_fingerprint": manifest.fingerprint,
            "methods": methods, "continuation": probes, "budget_pair": {"pilot_updates_per_arm": 8, "matched": compared["matched"]}}


def verify_legacy_comparison(actual, recorded):
    """Allow only additive reporting fields absent from archived engineering v1.

    Arm recipes compare by meaning (``same_recipe``); numeric costs, formulas,
    matches, tolerances and all other recorded values still compare exactly.
    New metadata never invalidates an unchanged run.
    """
    compatible = copy.deepcopy(actual)
    for role in ("reference", "candidate"):
        if "attention_pairs_per_sequence" not in recorded["costs"][role]["flops"]:
            compatible["costs"][role]["flops"].pop("attention_pairs_per_sequence", None)
        if role in recorded.get("arms", {}) and same_recipe(recorded["arms"][role], actual["arms"][role]):
            compatible["arms"][role] = recorded["arms"][role]
    for key in ("run.schedule", "run.global_batch_positions", "run.mixed_precision_param", "run.master_dtype", "run.reduce_dtype"):
        if key not in recorded["controlled"]:
            compatible["controlled"].pop(key, None)
    require(compatible == recorded, "budget-pair contract differs from actual resolved run arms")


def native_gpu_hours(directory, world_size):
    """Measured native-process wall time includes validation/checkpoint/compile."""
    verify_processes(directory)
    gpu_seconds = 0.
    for value in (read_json(path) for path in Path(directory).glob("process-*.json")):
        if "moeloop.train" not in value["argv"]:
            continue
        count = value.get("gpu_count", world_size)
        require(type(count) is int and 0 < count <= world_size, f"{directory}: process GPU accounting differs")
        gpu_seconds += count * (value["finished_unix"] - value["started_unix"])
    require(math.isfinite(gpu_seconds) and gpu_seconds > 0, f"{directory}: actual native-process runtime is missing")
    return gpu_seconds / 3600


def checkpoint_continuation(checkpoint, identity, *, expected_step, finite_data=False):
    """Read small native state/cursor payloads, without materializing model weights."""
    import torch.distributed.checkpoint as dcp
    from qualification import _assert_equal_tree
    from moeloop.training import continuation_fingerprint

    reader = dcp.FileSystemReader(checkpoint)
    metadata = reader.read_metadata().state_dict_metadata
    for prefix in ("core.", "optimizer.", "lr_scheduler.", "train_state.", "dataloader."):
        require(any(name.startswith(prefix) for name in metadata), f"{checkpoint}: full DCP lacks {prefix}")
    names = {f"train_state.rank_{rank}" for rank in range(identity["world_size"])}
    data_names = {name for name in metadata if name.startswith("dataloader.dp_rank_")}
    require(names <= metadata.keys() and len(data_names) == identity["world_size"],
            f"{checkpoint}: per-rank native continuation or data state is missing")
    state = dict.fromkeys(names | data_names, b"")
    dcp.load(state, storage_reader=reader, no_dist=True)
    labels = 0
    for name in names:
        record = pickle.loads(state[name])
        require(record.get("fingerprint") == continuation_fingerprint(identity), f"{checkpoint}: strict continuation identity differs")
        # The sidecar/fingerprint use JSON, whereas native pickle preserves
        # dataclass tuples. Compare identities at that JSON boundary only.
        saved_identity = json.loads(json.dumps(record["identity"], allow_nan=False))
        _assert_equal_tree(saved_identity, identity, f"{checkpoint}:{name}.identity")
        require(record["progress"]["completed_updates"] == expected_step and record.get("rng"),
                f"{checkpoint}: native optimizer progress/RNG is missing")
        labels += record["progress"]["label_tokens"]
    if finite_data:
        # Native dataloader payloads include the actual stateful dataset cursor.
        found = []
        def check(value):
            if isinstance(value, dict):
                if "epoch" in value and "cursor" in value:
                    require(value["epoch"] == 0, f"{checkpoint}: a training rank recycled documents")
                if "configuration" in value and "sources" in value:
                    found.append(True)
                    require(value["configuration"].get("infinite") is False, f"{checkpoint}: training data was infinite")
                for item in value.values():
                    check(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    check(item)
        for name in data_names:
            found.clear()
            check(pickle.loads(state[name]))
            require(found, f"{checkpoint}: actual finite dataset cursor is missing: {name}")
    return labels


def verify_muon_state(checkpoint, native):
    """Read the final DCP's optimizer state names (no tensors) for a Muon run.

    Every hidden matrix in the native DistMuon layouts holds exactly a
    ``momentum_buffer``; every other stored state is AdamW's. A side parameter
    that never received a gradient has no AdamW state, so only some side state
    is required, not state for every side parameter.
    """
    import torch.distributed.checkpoint as dcp

    matrices = {fqn for kwargs in native["optimizer_factory_kwargs_by_name"].values()
                for fqn in kwargs["compute_sharding_by_fqn"]}
    state = {}
    for name in dcp.FileSystemReader(checkpoint).read_metadata().state_dict_metadata:
        if name.startswith("optimizer.state."):
            fqn, key = name.removeprefix("optimizer.state.").rsplit(".", 1)
            state.setdefault(fqn, set()).add(key)
    require(matrices <= state.keys(), f"{checkpoint}: Muon optimizer state is missing for hidden matrices")
    require(state.keys() - matrices, f"{checkpoint}: side AdamW optimizer state is missing")
    for fqn, keys in state.items():
        expected = {"momentum_buffer"} if fqn in matrices else {"exp_avg", "exp_avg_sq", "step"}
        require(keys == expected, f"{checkpoint}: {fqn} optimizer state {sorted(keys)} is not {sorted(expected)}")


def verify_native_optimizer_execution(training, controls, role):
    """Bind optimizer, clipping and execution settings to one declared recipe.

    AdamW groups are written out as data and need no training stack. Muon
    groups name every hidden-matrix FQN, so the expected identity is
    ``trainer_config`` of the frozen spec (this imports TorchTitan, without
    which no Muon run exists), and the final checkpoint's state must match it.
    That expectation comes from this checkout's ``moeloop.optim``: a Muon run
    made before a grouping change no longer passes.
    """
    spec, identity = training["spec"], training["identity"]
    optimizer = spec.optimizer
    require(controls.get("optimizer") == optimizer, f"{role}: frozen common optimizer differs")
    if optimizer.get("name") == "Muon":
        from moeloop.recipes import trainer_config
        expected_optimizer = json.loads(json.dumps(trainer_config(spec).optimizer.to_dict()))
        require(identity.get("optimizer") == expected_optimizer, f"{role}: native optimizer differs from frozen spec"
                " (or this checkout's Muon grouping rule changed since the run)")
        verify_muon_state(training["receipt"]["checkpoint"], expected_optimizer)
    else:
        require(optimizer.get("name") == "AdamW" and set(optimizer.get("zero_weight_decay_roles", []))
                == {"norm", "embedding", "lm_head", "router"}, f"{role}: unsupported frozen optimizer roles")
        kwargs = {"lr": optimizer["base_lr"], "betas": list(optimizer["betas"]), "eps": optimizer["eps"]}
        expected_optimizer = {
            "param_groups": [
                {"pattern": "(?:norm|embedding|lm_head|router)", "optimizer_name": "AdamW",
                 "optimizer_kwargs": kwargs | {"weight_decay": 0.}},
                {"pattern": ".*", "optimizer_name": "AdamW",
                 "optimizer_kwargs": kwargs | {"weight_decay": optimizer["weight_decay"]}},
            ],
            "implementation": optimizer.get("implementation", "fused"),
            "optimizer_factory_kwargs_by_name": optimizer.get("optimizer_factory_kwargs_by_name", {}),
        }
        require(identity.get("optimizer") == expected_optimizer, f"{role}: native optimizer differs from frozen spec")
    require(identity.get("max_norm") == optimizer.get("max_norm", 1.),
            f"{role}: native gradient clipping differs from frozen spec")
    policy = spec.run.get("activation_checkpointing", "none")
    require(policy in ("none", "full_cells") and controls.get("activation_checkpointing") == policy,
            f"{role}: frozen activation-checkpoint policy differs from comparison")
    expected_execution = {
        "native_experts": spec.run.get("native_experts", True),
        "flex_attention": spec.run.get("flex_attention", True),
        "compile": {"enable": False, "enable_async_tensor_parallel": False,
                    "components": ["model", "loss"], "backend": "inductor"} | spec.run.get("compile", {}),
        "activation_checkpoint": None if policy == "none" else
            {"preserve_rng_state": True, "determinism_check": "default", "debug": False},
    }
    require(identity.get("execution") == expected_execution,
            f"{role}: native execution/AC differs from frozen declared settings")


def _verify_pair_controls(trainings, comparison, *, anchor_name):
    """Bind native settings to the frozen recipes, then compare the actual arms.

    These are the finite trainer_config/anchor defaults, expressed as data so
    checking the durable evidence of an AdamW run does not import the optional
    training stack; a Muon run's expected groups come from ``trainer_config``,
    which imports TorchTitan (see ``verify_native_optimizer_execution``).
    """
    controls = comparison["controlled"]
    validation_slices = {}
    for role, training in trainings.items():
        spec, identity = training["spec"], training["identity"]
        if anchor_name == "P1":
            frozen = spec.run.get("frozen_runtime", {})
            require({"world_size", "local_batch_size", "parallelism", "runtime", "gradient_accumulation_steps"} <= frozen.keys()
                    and controls.get("frozen_runtime") == frozen,
                    f"{role}: complete frozen runtime controls are missing or differ from comparison")
            require(all(identity.get(key) == frozen[key] for key in ("world_size", "local_batch_size"))
                    and identity.get("accumulation") == frozen["gradient_accumulation_steps"]
                    and all(key in frozen["parallelism"] and identity.get("parallelism", {}).get(key) == frozen["parallelism"][key]
                            for key in ("data_parallel_replicate_degree", "data_parallel_shard_degree"))
                    and training["request"].get("runtime") == frozen["runtime"],
                    f"{role}: actual native runtime/topology/batch differs from frozen runtime")
        else:
            verify_p2_runtime(training, role)
        require(same_recipe(comparison["arms"][role], spec.to_dict()), f"{role}: frozen comparison recipe differs")
        verify_native_optimizer_execution(training, controls, role)
        target = spec.run.get("validation_label_tokens")
        validator = training["resolved"]["validator"]
        loader = validator.get("dataloader", {})
        require(type(target) is int and target > 0 and validator.get("target_label_tokens") == target
                and spec.run.get("validation_infinite") is False and loader.get("infinite") is False
                and loader.get("shuffle") is False and loader.get("split") == "validation",
                f"{role}: frozen finite validation budget/loader differs from actual configuration")
        world, batch, seq_len = (identity[key] for key in ("world_size", "local_batch_size", "seq_len"))
        positions = world * batch * seq_len
        # LoopValidator: the budget rounds up to whole global batches of world*batch*accumulation
        # blocks; each rank runs accumulation steps per global batch.
        accumulation = identity["accumulation"]
        steps = math.ceil(target / (positions * accumulation)) * accumulation
        observed = []
        for point in training["validations"]:
            require(point.get("target_label_tokens") == target and point.get("infinite") is False
                    and point.get("steps") == point.get("configured_steps") == steps
                    and point.get("world_size") == world and point.get("local_batch_size") == batch
                    and point.get("seq_len") == seq_len,
                    f"{role}: actual validation target/steps/batch/finite policy differs from frozen budget")
            labels = point["valid_tokens"]
            require(type(labels) is int and target <= labels <= steps * positions,
                    f"{role}: actual validation labels exceed or fall short of the frozen rounded budget")
            capacity = point.get("finite_batch_capacity_per_rank", [])
            require(isinstance(capacity, list) and len(capacity) == world
                    and all(type(value) is int and value >= steps for value in capacity),
                    f"{role}: finite validation rank capacity is missing or insufficient")
            observed.append(labels)
        require(observed and len(set(observed)) == 1, f"{role}: repeated validation did not use a fixed slice")
        validation_slices[role] = (world, batch, seq_len, target, steps, observed[0])
    fields = ("optimizer", "max_norm", "execution")
    if anchor_name == "P2":
        for field in ("source", "runtime"):
            require(trainings["reference"]["request"][field] == trainings["candidate"]["request"][field],
                    f"P2 arms have different controlled {field} identities")
        fields += ("parallelism", "precision", "lr_scheduler", "torchtitan_commit")
    for field in fields:
        require(trainings["reference"]["identity"][field] == trainings["candidate"]["identity"][field],
                f"{anchor_name} arms have different controlled {field} identities")
    require(validation_slices["reference"] == validation_slices["candidate"],
            f"{anchor_name} arms have different controlled validation slices")
    if anchor_name == "P1":
        return controls["frozen_runtime"]
    reference = trainings["reference"]
    return {"world_size": 16, "local_batch_size": 16, "gradient_accumulation_steps": 1,
            "parallelism": reference["identity"]["parallelism"], "precision": reference["identity"]["precision"],
            "torchtitan_commit": reference["identity"]["torchtitan_commit"],
            "source": reference["request"]["source"], "runtime": reference["request"]["runtime"],
            "basis": "frozen requests and native identities checked against approved P2 controls"}


def verify_p1_controls(trainings, comparison):
    """Preserve the original P1 recipe/comparison frozen-runtime requirement."""
    return _verify_pair_controls(trainings, comparison, anchor_name="P1")


def verify_p2_controls(trainings, comparison):
    """P2 records runtime in frozen requests, independently of comparison JSON."""
    return _verify_pair_controls(trainings, comparison, anchor_name="P2")


def verify_p2_runtime(training, role):
    verify_p2_geometry(training, role)
    request, identity, resolved = training["request"], training["identity"], training["resolved"]
    runtime = request.get("runtime", {})
    # This is the approved H200/TorchTitan environment for the frozen P2 runs.
    expected_runtime = {"torch": "2.14.0+cu130", "cuda": "13.0", "gpu_name": "NVIDIA H200", "compute_capability": [9, 0]}
    require(all(runtime.get(key) == value for key, value in expected_runtime.items()) and runtime.get("python"),
            f"{role}: P2 actual software/hardware runtime differs from the frozen environment")
    require(identity.get("torchtitan_commit") == "086bf6c166ec85c1298eb5596fa9bf95f6a2d840",
            f"{role}: P2 native TorchTitan source differs")
    precision = ["float32", "bfloat16", "float32"]
    expected_training = {"local_batch_size": 16, "global_batch_size": 256, "seq_len": 4096,
                         "dtype": precision[0], "mixed_precision_param": precision[1], "mixed_precision_reduce": precision[2]}
    require(identity.get("precision") == precision and all(resolved["training"].get(key) == value for key, value in expected_training.items()),
            f"{role}: P2 resolved native batch/precision differs")
    parallelism = identity["parallelism"]
    expected_parallelism = {"data_parallel_replicate_degree": 2, "data_parallel_shard_degree": 8,
                            "tensor_parallel_degree": 1, "pipeline_parallel_degree": 1,
                            "expert_parallel_degree": 1, "context_parallel_degree": 1,
                            "enable_sequence_parallel": False, "spmd_backend": "partial_dtensor"}
    require(parallelism == resolved["parallelism"] and all(parallelism.get(key) == value for key, value in expected_parallelism.items()),
            f"{role}: P2 native parallelism differs from approved HSDP2x8")
    scheduler, schedule = identity["lr_scheduler"], training["spec"].run["schedule"]
    require(scheduler == resolved["lr_scheduler"] and all(scheduler.get(key) == schedule[key] for key in
            ("total_steps", "warmup_steps", "decay_ratio", "decay_type", "min_lr_factor")),
            f"{role}: P2 actual native scheduler differs from the full frozen horizon")


def verify_anchor_comparison(trainings, comparison, *, anchor_name):
    from moeloop.comparison import compare_recipes
    controls = comparison["controlled"]
    actual = compare_recipes(trainings["reference"]["spec"], trainings["candidate"]["spec"], seq_len=4096,
        tolerance=.05, batch_size=controls["batch_size"], bytes_per_element=controls["kv_bytes_per_element"],
        activation_checkpointing=controls["activation_checkpointing"],
        attention_pairs_per_sequence=controls.get("attention_pairs_per_sequence"), question=comparison["question"],
        data=comparison["data"], evaluation=comparison["evaluation"])
    if anchor_name == "P1":
        runtime = actual["controlled"]["frozen_runtime"] = verify_p1_controls(trainings, comparison)
    else:
        runtime = verify_p2_controls(trainings, comparison)
    # The arm recipes were compared by meaning in _verify_pair_controls.
    for field in ("costs", "matched", "controlled", "measured-different", "recipe_differences"):
        require(actual[field] == comparison[field], f"{anchor_name} {field} budget report differs from actual resolved run arms")
    return actual, runtime


def verify_p1_document_losses(evaluation, manifest, commands, evaluation_path, model_path, *, expected_documents=None):
    """Check full ordered document losses and the exact evaluation invocation."""
    reference = evaluation.get("document_losses", {})
    require(isinstance(reference.get("path"), str), "P1 actual document-loss sidecar is missing")
    path = Path(reference["path"])
    require(path.is_absolute() and path.is_file() and digest(path) == reference.get("sha256"),
            "P1 document-loss sidecar is missing or its SHA256 differs")
    sidecar = read_json(path)
    names = manifest.source_names(evaluation["split"])
    require(len(names) == 1 and sidecar.get("source") == names[0]
            and sidecar.get("format") == "moeloop-document-losses" and sidecar.get("version") == 1
            and sidecar.get("manifest_fingerprint") == manifest.fingerprint
            and sidecar.get("complete_split") is evaluation.get("complete_split") is True
            and all(sidecar.get(key) == evaluation.get(key) for key in ("split", "seq_len", "num_visits"))
            and sidecar.get("label_contract") == evaluation.get("label_contract") == "already_aligned_document_internal_next_token",
            "P1 document-loss source/evaluation metadata differs")
    shards = manifest.shards(names[0], evaluation["split"])
    count = sum(shard["num_documents"] for shard in shards)
    labels = sum(shard["num_tokens"] - shard["num_documents"] for shard in shards)
    documents = sidecar.get("documents", [])
    require(len(documents) == reference.get("document_count") == count
            and all(type(row.get("document_id")) is int and row["document_id"] == index
                    for index, row in enumerate(documents)),
            "P1 document-loss full ordered document coverage differs")
    for row in documents:
        require(type(row.get("valid_tokens")) is int and row["valid_tokens"] > 0
                and all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 0
                        for key in ("ce_sum", "ce"))
                and math.isclose(row["ce"] * row["valid_tokens"], row["ce_sum"], rel_tol=1e-9, abs_tol=1e-10),
                "P1 document-loss labels or finite CE sum/mean differs")
    require(sum(row["valid_tokens"] for row in documents) == evaluation["valid_tokens"] == labels
            and math.isclose(math.fsum(row["ce_sum"] for row in documents), evaluation["ce_sum"], rel_tol=1e-9, abs_tol=1e-6),
            "P1 document-loss totals differ from the complete heldout aggregate")
    identity = tuple((row["document_id"], row["valid_tokens"]) for row in documents)
    require(expected_documents is None or identity == expected_documents,
            "P1 paired document identities/label counts differ between arms")

    def option(command, name, default=None):
        return command[command.index(name) + 1] if name in command else default
    require(any("moeloop.evaluate" in command and option(command, "--split", "validation") == evaluation["split"]
                and Path(option(command, "--output", "/missing")).resolve() == Path(evaluation_path).resolve()
                and Path(option(command, "--model", "/missing")).resolve() == Path(model_path).resolve()
                and Path(option(command, "--manifest", "/missing")).resolve() == manifest.path
                and Path(option(command, "--document-loss-output", "/missing")).resolve() == path
                for command in commands), "P1 actual paired document-loss evaluation process receipt is missing")
    return identity


def verify_learning_observations(training, role):
    """Read successful-update progress, finite gradients and router activity."""
    positions = training["spec"].run["global_batch_positions"]
    previous_labels = 0
    for metric in training["metrics"]:
        progress = metric["global_progress"]
        require(progress["completed_updates"] == metric["step"]
                and previous_labels < progress["label_tokens"] <= progress["input_tokens"] <= progress["padded_positions"]
                and progress["padded_positions"] == metric["step"] * positions,
                f"{role}: native label/position learning curve is inconsistent or reset")
        previous_labels = progress["label_tokens"]
        require(math.isfinite(metric.get("grad_norm", math.nan)) and metric["grad_norm"] > 0,
                f"{role}: healthy finite nonzero native gradients are missing")
        counts = metric.get("global_router_assignments", [])
        require(counts and all(sum(row) > 0 and all(math.isfinite(value) and value >= 0 for value in row) for row in counts),
                f"{role}: actual nonempty finite router assignments are missing")
    require(training["metrics"][-1]["global_progress"] == training["receipt"]["outcome"]["global_progress"],
            f"{role}: final native metric progress differs from durable outcome")


def verify_scientific_anchor(path, *, anchor_name="P1"):
    from moeloop.data import PreparedManifest
    from moeloop.prepare import DCLM_ID, DCLM_TOKENIZER

    anchor = read_json(path)
    require(anchor.get("format") == "moeloop-scientific-anchor" and anchor.get("version") == 1
            and anchor.get("anchor") == anchor_name, "unsupported scientific anchor schema")
    require(set(anchor.get("arms", {})) == {"reference", "candidate"}, "P1 requires both ordinary MoE and SMELT arms")
    manifest = PreparedManifest.load(anchor["manifest"])
    require(anchor.get("manifest_fingerprint") == manifest.fingerprint, "P1 manifest fingerprint differs")
    provenance = manifest.payload.get("provenance", {})
    dataset = provenance.get("dataset", {})
    require(provenance.get("kind") == "public_document_subset" and dataset.get("id") == DCLM_ID
            and re.fullmatch(r"[0-9a-f]{40}", dataset.get("revision", "")) and dataset.get("license")
            and dataset.get("files"), "P1 requires publicly reconstructible pinned DCLM; old packed lineage is insufficient")
    require(provenance.get("selection", {}).get("deduplication"), "P1 document deduplication/split isolation is missing")
    require(all(item.get("size", 0) > 0 and re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))
                for item in dataset["files"]), "P1 public source shard content hashes are missing")
    require(manifest.tokenizer.get("id") == DCLM_TOKENIZER["id"]
            and manifest.tokenizer.get("revision") == DCLM_TOKENIZER["revision"]
            and manifest.tokenizer.get("source_files", {}).get("tokenizer.json") == DCLM_TOKENIZER["tokenizer_json_sha256"],
            "P1 fixed Qwen3 tokenizer identity differs")
    capacities = {split: sum(shard["num_tokens"] - shard["num_documents"]
        for name in manifest.source_names(split) for shard in manifest.shards(name, split))
        for split in ("train", "validation", "test")}
    target = 20_485_760_000
    require(capacities["train"] >= target and min(capacities["validation"], capacities["test"]) >= 10_000_000,
            "P1 fresh train/validation/test label capacity is insufficient")
    pair = read_json(anchor["comparison"])
    require(pair.get("anchor") == anchor_name and 0 < pair.get("candidate_count", 0) <= 32, "finite budget screen is missing")
    comparison = pair["comparison"]
    require(comparison.get("data", {}).get("manifest_fingerprint") == manifest.fingerprint
            and comparison.get("evaluation", {}).get("manifest_fingerprint") == manifest.fingerprint,
            "P1 frozen data/evaluation comparison identities are missing")
    trainings, arms, resumed, paired_documents = {}, {}, [], {}
    for role, name in (("reference", f"{anchor_name.lower()}_moe"), ("candidate", f"{anchor_name.lower()}_smelt")):
        directory = Path(anchor["arms"][role])
        training = verify_training(directory, manifest, qualification=False)
        spec, identity, resolved = training["spec"], training["identity"], training["resolved"]
        require(spec.name == name and spec.run.get("anchor") == anchor_name, f"{role}: anchor model scale/recipe is missing")
        if anchor_name == "P2":
            verify_p2_geometry(training, role)
        require(spec.run.get("seed") == identity.get("debug_seed") == 42 and spec.run.get("training_seeds") == [42],
                f"{role}: P1 requires the single frozen training seed 42")
        require(spec.run.get("data_infinite") is False and identity["dataloader"].get("infinite") is False,
                f"{role}: P1 fresh data must be finite")
        require(spec.run.get("seq_len") == 4096 and identity.get("precision") == ["float32", "bfloat16", "float32"]
                and training["request"]["runtime"].get("cuda"), f"{role}: P1 context/BF16/master/reduction runtime differs")
        require(identity["parallelism"].get("data_parallel_shard_degree", 0) >= 2,
                f"{role}: P1 needs actual sharded FSDP")
        schedule = spec.run.get("schedule", {})
        require(schedule.get("name") == "warmup_stable_cosine_decay" and schedule.get("target_label_tokens") == target
                and schedule.get("warmup_label_tokens") == 20_000_000
                and schedule.get("stable_end_label_tokens") == 10_485_760_000
                and schedule.get("decay_label_tokens") == 10_000_000_000
                and schedule.get("min_lr_factor") == .1 and schedule.get("decay_type") == "cosine"
                and schedule.get("total_steps") == spec.run["schedule_total_steps"] == training["steps"],
                f"{role}: full frozen P1 WSD schedule/endpoint is missing")
        native_schedule = resolved["lr_scheduler"]
        require(identity.get("lr_scheduler") == native_schedule and all(native_schedule.get(key) == schedule.get(key)
                    for key in ("total_steps", "warmup_steps", "decay_ratio", "decay_type", "min_lr_factor")),
                f"{role}: actual scheduler differs from the frozen full WSD schedule")
        labels = training["receipt"]["outcome"]["global_progress"]["label_tokens"]
        positions = spec.run["global_batch_positions"]
        require(target <= labels <= target + positions, f"{role}: actual P1 labels did not reach the complete endpoint within one update")
        saved_labels = checkpoint_continuation(training["receipt"]["checkpoint"], identity,
                                               expected_step=training["steps"], finite_data=True)
        require(saved_labels == labels, f"{role}: actual per-rank DCP labels differ from outcome")
        verify_learning_observations(training, role)
        hf = verify_hf(directory, training)
        endpoints = {}
        commands = verify_processes(directory)
        for split in ("validation", "test"):
            evaluation_path = Path(anchor["heldout"][role][split]).resolve()
            evaluation = read_json(evaluation_path)
            require(evaluation.get("split") == split and evaluation.get("complete_split") is True
                    and evaluation.get("valid_tokens") == capacities[split] and evaluation["valid_tokens"] >= 10_000_000
                    and evaluation.get("manifest_fingerprint") == manifest.fingerprint
                    and evaluation.get("seq_len") == 4096 and evaluation.get("num_visits") == spec.model.num_visits
                    and evaluation.get("attention") == "document_causal_block_local"
                    and math.isfinite(evaluation.get("ce", math.nan)) and math.isfinite(evaluation.get("ce_sum", math.nan))
                    and math.isclose(evaluation["ce"] * evaluation["valid_tokens"], evaluation["ce_sum"], rel_tol=1e-9),
                    f"{role}: complete frozen {split} heldout/source identity is missing")
            require(evaluation.get("tokenizer") == {key: manifest.tokenizer[key] for key in ("id", "revision", "vocab_size")},
                    f"{role}: heldout tokenizer identity differs")
            paired_documents[split] = verify_p1_document_losses(evaluation, manifest, commands, evaluation_path,
                directory / "hf", expected_documents=paired_documents.get(split))
            endpoints[split] = {"ce": evaluation["ce"], "valid_tokens": evaluation["valid_tokens"],
                                "document_losses": evaluation["document_losses"]}
        # A main-line stop/resume reuses its own saved prefix; no extra seed or
        # full uninterrupted duplicate trajectory is required.
        load = resolved["checkpoint"]
        if load.get("initial_load_path"):
            initial = Path(load["initial_load_path"])
            match = re.fullmatch(r"step-(\d+)", initial.name)
            require(match and not load.get("initial_load_model_only"), f"{role}: main-line resume was not full-state")
            stopped = int(match.group(1))
            require(0 < stopped < training["steps"], f"{role}: resume was not from a main-line prefix")
            checkpoint_continuation(initial, identity, expected_step=stopped, finite_data=True)
            attempts = [read_json(item) for item in directory.glob("attempts/*/outcome.json")]
            require(any(item.get("status") == "completed" and item.get("completed_updates") == stopped for item in attempts),
                    f"{role}: the actual stopped native attempt is missing")
            resumed.append(role)
        trainings[role] = training
        arms[role] = {"labels": labels, "updates": training["steps"], "hf": hf, "heldout": endpoints,
            "actual_gpu_hours": native_gpu_hours(directory, identity["world_size"])}
    verify_anchor_resumes(anchor_name, resumed)
    actual, runtime = verify_anchor_comparison(trainings, comparison, anchor_name=anchor_name)
    require({"unique_non_embedding_parameters", "training_flops_per_token", "logical_kv_bytes"} <= actual["matched"].keys(),
            "P1 does not match the three declared analytical budgets")
    require(actual["costs"]["reference"]["parameters"]["unique_non_embedding"] >= 500_000_000
            and actual["costs"]["candidate"]["parameters"]["unique_non_embedding"] >= 500_000_000,
            "P1 requires the research-scale model tier; a renamed tiny engineering pair is insufficient")
    for field in ("source", "world_size", "dataloader", "precision"):
        left = trainings["reference"]["request"] if field == "source" else trainings["reference"]["identity"]
        right = trainings["candidate"]["request"] if field == "source" else trainings["candidate"]["identity"]
        require(left[field] == right[field], f"P1 arms have different controlled {field} identities")
    for role, training in trainings.items():
        curve = [json.loads(line) for line in Path(anchor["curves"][role]).read_text().splitlines() if line.strip()]
        metrics = {item["step"]: item for item in training["metrics"]}
        validations = {item["step"]: item for item in training["validations"]}
        require([item["step"] for item in curve] == sorted(validations), f"{role}: complete native learning/compute curve is missing")
        per_position = actual["costs"][role]["flops"]["training_per_token"]
        for item in curve:
            metric = metrics[item["step"]]
            require(item["global_labels"] == metric["global_progress"]["label_tokens"]
                    and item["validation_ce"] == validations[item["step"]]["ce"]
                    and math.isclose(item["cumulative_model_flops_estimate"],
                        per_position * metric["global_progress"]["padded_positions"], rel_tol=1e-9),
                    f"{role}: curve differs from actual labels/validation/frozen model-compute estimate")
        arms[role]["curve_points"] = len(curve)
    if anchor_name == "P2":
        verify_p2_screen(pair, trainings)
    result = anchor.get("result", {})
    require(result.get("classification") and result.get("limitations"), "P1 scientific conclusion and single-seed/adaptation limitations are missing")
    return {"anchor": anchor_name, "manifest_fingerprint": manifest.fingerprint, "arms": arms, "resumed_arms": resumed,
        "matched": actual["matched"], "result": result, **({"runtime": runtime} if anchor_name == "P2" else {}),
        "compute_basis": "native_padded_positions_times_frozen_training_flops_per_token_estimate",
        "gpu_hours_scope": "measured native-process wall time including compile, validation and checkpoint"}


def verify_benchmark_error(error, tolerance, name, *, enforce_bound=True):
    fields = ("max_absolute", "relative_l2", "reference_peak", "max_absolute_allowed")
    require(all(type(error.get(key)) in (int, float) and math.isfinite(error[key]) and error[key] >= 0 for key in fields),
            f"{name}: complete finite raw error coverage is missing")
    bound = tolerance["peak_factor"] * error["reference_peak"] + tolerance["floor"]
    require(math.isclose(error["max_absolute_allowed"], bound, rel_tol=1e-9, abs_tol=1e-12),
            f"{name}: raw peak-scale bound differs from the original tolerance")
    if enforce_bound:
        require(error["relative_l2"] <= tolerance["relative_l2"] and error["max_absolute"] <= bound,
                f"{name}: component raw vector or peak error exceeds the original bound")


def verify_benchmark_math(report, path):
    """Preserve strict failures when accepting explicitly narrower math evidence."""
    from moeloop.costs import estimate_cost
    from moeloop.model import ModelConfig
    from recipe_benchmark import TOLERANCES

    settings, correctness = report.get("settings", {}), report.get("correctness") or {}
    require(correctness.get("all_parameter_gradients_checked") is True and correctness.get("optimizer_updates") == 1
            and correctness.get("physical_parameter_elements", 0) > 0
            and all(math.isfinite(correctness.get(key, math.nan)) for key in ("candidate_ce", "reference_ce")),
            f"{path}: real public-reference full-update mathematical parity is missing or failed")
    if correctness.get("status") == "passed":
        return {"mathematics_scope": "strict_full_network_parity", "strict_full_network_status": "passed"}
    scope = "component_equivalence_with_observed_end_to_end_bf16_route_divergence"
    components = report.get("component_math") or {}
    require(settings.get("reference") == "torchtitan" and settings.get("dtype") == "bfloat16"
            and settings.get("component_math") is True and report.get("mathematics_scope") == scope
            and components.get("status") == "passed" and components.get("scope") == scope
            and correctness.get("status") == "failed" and correctness.get("failures"),
            f"{path}: failed strict full-network parity needs explicit component evidence/scope; its failure must be retained")
    tolerance = TOLERANCES["bfloat16"]
    require(components.get("tolerances") == tolerance and report.get("tolerances_predeclared") == tolerance,
            f"{path}: component math changed the original BF16 tolerances")
    require(all(correctness.get(key) is True for key in ("initial_parameters_equal", "losses_finite", "ce_close",
                "all_gradients_finite", "all_updated_parameters_finite"))
            and math.isclose(correctness["candidate_ce"], correctness["reference_ce"],
                rel_tol=tolerance["loss_rtol"], abs_tol=tolerance["loss_atol"]),
            f"{path}: full-network equal-input/CE/finite-gradient/update evidence is missing or failed")
    require(report.get("conditions", {}).get("timed_routing") == components.get("timed_routing") == "natural_unmodified_router_outputs",
            f"{path}: component diagnostics must not lock or replace timed routes")
    gradients = correctness.get("gradient_errors", {})
    count = correctness.get("physical_parameter_tensors", 0)
    require(count > 0 and len(gradients) == count, f"{path}: complete full-network gradient error coverage is missing")
    for name, error in gradients.items():
        verify_benchmark_error(error, tolerance, f"{path}: retained full-network gradient {name}", enforce_bound=False)
    for name in ("hidden_states", "sampled_logits"):
        verify_benchmark_error(correctness.get(name, {}), tolerance, f"{path}: retained full-network {name}", enforce_bound=False)
    require(all(type(correctness.get(key)) in (int, float) and math.isfinite(correctness[key]) and correctness[key] >= 0
                for key in ("updated_parameters_max_absolute", "updated_parameters_max_relative_l2")),
            f"{path}: raw full-network updated-parameter errors are missing")
    coverage = components.get("coverage", {})
    require(coverage.get("representative_physical_cell") == "core.0" and coverage.get("homogeneous_geometry") is True
            and coverage.get("all_model_parameter_mapping_verified") is True
            and coverage.get("model_physical_parameter_tensors") == count
            and coverage.get("model_physical_parameter_elements") == correctness["physical_parameter_elements"],
            f"{path}: representative component versus whole-model mapping scope/coverage differs")
    config = ModelConfig.from_dict(report["model_config"])
    require(estimate_cost(config, seq_len=settings["seq_len"])["parameters"]["unique_total"] == correctness["physical_parameter_elements"],
            f"{path}: component coverage omits actual model parameter elements")
    for region in ("prefix", "core", "suffix"):
        local = config.region_config(region)
        require((local.hidden_dim, local.num_experts, local.top_k, local.always_on_hidden_dim)
                == (config.hidden_dim, config.num_experts, config.top_k, 0),
                f"{path}: representative component checks require the declared homogeneous expert geometry")
    routing = correctness.get("routing") or {}
    calls = routing.get("calls", [])
    expected_calls = config.prefix_cells + config.num_cells * config.num_visits + config.suffix_cells
    tokens_per_call = settings["batch_size"] * settings["seq_len"]
    require(routing.get("logical_calls") == len(calls) == expected_calls
            and {call.get("logical_call") for call in calls} == set(range(expected_calls)),
            f"{path}: observed full-network natural routing coverage is incomplete")
    for call in calls:
        require(call.get("tokens") == tokens_per_call and type(call.get("changed_tokens")) is int
                and 0 <= call["changed_tokens"] <= tokens_per_call
                and all(re.fullmatch(r"[0-9a-f]{64}", call.get(key, ""))
                        for key in ("candidate_expert_ids_sha256", "reference_expert_ids_sha256"))
                and (call["candidate_expert_ids_sha256"] == call["reference_expert_ids_sha256"])
                    == (call["changed_tokens"] == 0),
                f"{path}: natural router divergence count/hash differs")
    changed = sum(call["changed_tokens"] for call in calls)
    tokens = sum(call["tokens"] for call in calls)
    require(changed > 0 and routing.get("changed_tokens") == changed and routing.get("compared_tokens") == tokens
            and math.isclose(routing.get("changed_token_fraction", math.nan), changed / tokens, rel_tol=1e-9),
            f"{path}: the claimed observed BF16 route-divergence scope lacks actual consistent observations")
    expected = {
        "attention": {"inputs": ["query", "key", "value"], "parameters": [],
                      "errors": {"forward", "gradient.query", "gradient.key", "gradient.value"}},
        "router": {"inputs": ["input"], "parameters": ["gate.weight"],
                   "errors": {"logits", "weights", "gradient.input", "gradient.gate.weight"}},
        "experts": {"inputs": ["input", "route_weights"], "parameters": ["w1_EFD", "w3_EFD", "w2_EDF"],
                    "errors": {"forward", "gradient.input", "gradient.route_weights", "gradient.w1_EFD", "gradient.w3_EFD", "gradient.w2_EDF"}},
    }
    checks = components.get("checks", {})
    require(set(checks) == set(expected), f"{path}: attention/router/experts component coverage is incomplete")
    for name, contract in expected.items():
        check = checks[name]
        actual = check.get("coverage", {})
        require(check.get("status") == "passed" and check.get("reference") and not check.get("failures")
                and actual.get("input_gradients") == contract["inputs"]
                and actual.get("parameter_gradients") == contract["parameters"]
                and set(check.get("errors", {})) == contract["errors"],
                f"{path}: {name} component forward/gradient reference coverage is incomplete or failed")
        reference = check["reference"]
        reference_name = {"attention": "torch.nn.functional.scaled_dot_product_attention",
                          "router": "transformers.Qwen3MoeTopKRouter", "experts": "transformers.Qwen3MoeExperts"}[name]
        version = report.get("runtime", {}).get("torch") if name == "attention" else report.get("runtime", {}).get("packages", {}).get("transformers")
        require(reference.get("name") == reference_name and reference.get("version") == version and version
                and re.fullmatch(r"[0-9a-f]{64}", reference.get("file_sha256", "")),
                f"{path}: {name} independent operator source/version identity differs")
        for field, error in check["errors"].items():
            verify_benchmark_error(error, tolerance, f"{path}: component {name}.{field}")
    require(checks["router"].get("expert_sets_equal") is True
            and checks["experts"]["coverage"].get("experts") == config.num_experts
            and checks["experts"]["coverage"].get("top_k") == config.top_k
            and checks["router"]["reference"]["file_sha256"] == checks["experts"]["reference"]["file_sha256"]
            and checks["experts"].get("routing") == "fixed same-input FP32 router outputs for this operator derivative check only",
            f"{path}: exact router top-k or complete expert-bank gradient coverage differs")
    return {"mathematics_scope": scope, "strict_full_network_status": "failed",
            "strict_failures": correctness["failures"], "component_math_status": "passed",
            "observed_full_network_changed_route_fraction": changed / tokens,
            "component_scope": "one representative homogeneous physical cell; all expert-bank tensors and mapped full-model gradient finiteness"}


def verify_performance(evidence):
    from recipe_benchmark import TT_REFERENCE_HASHES

    paths = evidence.get("benchmarks", [])
    require(paths, "actual public-reference benchmark JSON is missing")
    primary, oracle, lagging = [], [], []
    for path in paths:
        report = read_json(path)
        require(report.get("schema_version") == 1, f"{path}: unsupported public benchmark schema")
        settings = report.get("settings", {})
        mathematics = verify_benchmark_math(report, path)
        require(report.get("source_sha256", {}).get("public_reference"), f"{path}: actual public reference source hash is missing")
        if settings.get("reference") == "transformers":
            require(settings.get("dtype") == "float32", f"{path}: HF mathematical oracle must retain FP32 evidence")
            oracle.append(str(path))
            continue
        if settings.get("reference") != "torchtitan":
            continue
        require(settings.get("mode") == "benchmark" and settings.get("candidate") == "native"
                and settings.get("attention") == "flex" and settings.get("dtype") == "bfloat16"
                and settings.get("device", "").startswith("cuda") and settings.get("warmup", 0) >= 2
                and settings.get("steps", 0) >= 5 and report.get("runtime", {}).get("cuda")
                and report.get("model_config", {}).get("num_visits") == 1,
                f"{path}: primary ordinary-MoE native/grouped Flex BF16 CUDA benchmark is missing")
        require(report.get("torchtitan_commit") == "086bf6c166ec85c1298eb5596fa9bf95f6a2d840"
                and report.get("public_reference_source_sha256") == TT_REFERENCE_HASHES,
                f"{path}: unchanged pinned official TorchTitan source identity differs")
        require(report.get("lag_policy", {}).get("max_step_ratio") == 1.2,
                f"{path}: the predeclared 0.20 lag policy changed")
        performance = report.get("performance", {})
        medians = {}
        for arm in ("reference", "candidate"):
            value = performance.get("arms", {}).get(arm, {})
            samples = value.get("full_step_samples_seconds", [])
            require(len(samples) == settings["steps"] and all(math.isfinite(item) and item > 0 for item in samples),
                    f"{path}: actual full-step CUDA timing samples are missing")
            medians[arm] = statistics.median(samples)
            require(math.isclose(medians[arm], value.get("full_step_seconds", {}).get("median", math.nan), rel_tol=1e-9)
                    and value.get("peak_memory_allocated_bytes", 0) > 0
                    and value.get("peak_memory_reserved_bytes", 0) >= value["peak_memory_allocated_bytes"],
                    f"{path}: timing summary or actual peak GPU memory differs")
        ratio = medians["candidate"] / medians["reference"]
        require(math.isclose(ratio, performance.get("candidate_reference_step_ratio", math.nan), rel_tol=1e-9),
                f"{path}: public-reference throughput ratio differs from raw timing samples")
        expected = "needs_investigation" if ratio > 1.2 else "passed"
        require(report.get("status") == performance.get("status") == expected,
                f"{path}: reported performance verdict hides or misstates the 20% lag")
        primary.append({"path": str(path), "candidate_reference_step_ratio": ratio, "status": expected, **mathematics})
        if ratio > 1.2:
            lagging.append(str(path))
    require(primary and oracle, "V1 needs both the official TorchTitan performance comparison and HF FP32 math oracle")
    fallback = None
    if lagging:
        require(evidence.get("fallback_scope"), "unresolved >20% public-reference lag requires an explicit restricted fallback scope")
        fallback = read_json(evidence["fallback_scope"])
        require(fallback.get("status") == "restricted_support" and fallback.get("reason")
                and fallback.get("supported") and fallback.get("unsupported")
                and fallback.get("no_universal_no_lag_claim") is True
                and set(lagging) <= set(fallback.get("unresolved_lag_benchmarks", [])),
                "fallback must identify supported/unsupported paths and preserve the unresolved lag explicitly")
    return {"status": "restricted_support" if lagging else "passed", "primary": primary,
            "hf_math_oracle": oracle, "lag_threshold": .20, "fallback_scope": fallback}


def verify_multinode(evidence):
    from moeloop.data import PreparedManifest

    require(isinstance(evidence, list) and evidence, "8/16-rank native qualification evidence is missing")
    results = {}
    for entry in evidence:
        directory = Path(entry["directory"])
        request = read_json(directory / "request.json")
        world = request.get("world_size")
        require(world in (8, 16) and world not in results, "multinode qualification must identify one 8-rank and one 16-rank run")
        manifest = PreparedManifest.load(request["manifest"])
        training = verify_training(directory, manifest, qualification=False)
        identity = training["identity"]
        require(training["steps"] >= 2 and request["runtime"].get("cuda")
                and identity["precision"] == ["float32", "bfloat16", "float32"]
                and identity["parallelism"].get("data_parallel_shard_degree") == world,
                f"{world} ranks: actual BF16/full-update FSDP evidence is missing")
        checkpoint_continuation(training["receipt"]["checkpoint"], identity, expected_step=training["steps"])
        topology = read_json(entry["topology"])
        require(topology.get("world_size") == world
                and topology.get("continuation_identity_sha256") == digest(directory / "continuation_identity.json"),
                f"{world} ranks: topology is not bound to the actual native continuation identity")
        ranks = topology.get("ranks", [])
        require(len(ranks) == world and {item.get("rank") for item in ranks} == set(range(world))
                and all(item.get("hostname") and type(item.get("local_rank")) is int for item in ranks),
                f"{world} ranks: actual per-rank host/local-rank receipt is missing")
        hosts = {item["hostname"] for item in ranks}
        for host in hosts:
            local = [item["local_rank"] for item in ranks if item["hostname"] == host]
            require(len(local) == len(set(local)) and set(local) == set(range(len(local))),
                    f"{world} ranks: invalid actual local-rank topology")
        require((world == 8 or len(hosts) >= 2)
                and (world != 16 or all(sum(item["hostname"] == host for item in ranks) == 8 for host in hosts)),
                "16 ranks require two actual hosts with eight ranks each")
        logs = topology.get("process_logs", [])
        require(logs, f"{world} ranks: hashed actual topology process logs are missing")
        native_logs = {str(Path(value["log"]).resolve()): value["log_sha256"] for value in
                       (read_json(path) for path in directory.glob("process-*.json"))}
        recorded = []
        for log in logs:
            path = Path(log["path"])
            require(path.is_file() and digest(path) == log.get("sha256")
                    and native_logs.get(str(path.resolve())) == log["sha256"],
                    f"{world} ranks: topology log hash/native process receipt differs")
            for line in path.read_text(errors="replace").splitlines():
                # A native launch can print one gathered JSON ranks list. Read
                # it from the hashed log rather than accepting a copied label.
                for match in re.finditer(r"\[", line):
                    try:
                        value, _ = json.JSONDecoder().raw_decode(line[match.start():])
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                        recorded.extend(value)
                        break
        require(all(item in recorded for item in ranks), f"{world} ranks: claimed rank/hostname topology is absent from actual logs")
        results[world] = {"updates": training["steps"], "hosts": len(hosts),
                          "actual_gpu_hours": native_gpu_hours(directory, world)}
    require(set(results) == {8, 16}, "both actual 8-rank and 16-rank native qualification receipts are required")
    return results


def verify_v2_dynamics(training):
    """Check observed execution/metrics, separately from the common LM CE."""
    spec, metrics = training["spec"], training["metrics"]
    require(metrics, "dynamic method has no actual metric history")
    model, recipe = spec.model, spec.training
    if spec.name == "huginn":
        require(model.initial_state == "random" and model.prefix_cells > 0 and model.suffix_cells > 0
                and model.input_injection in ("linear", "add")
                and recipe.depth_rule == "poisson_lognormal_micro_v1" and recipe.grad_visits,
                "Huginn random initial state/live anchor/sampled gradient path is missing")
        observed = []
        for item in metrics:
            depths, gradients = item.get("microbatch_depths", []), item.get("microbatch_grad_visits", [])
            require(depths and len(depths) == len(gradients) == training["identity"]["accumulation"],
                    "Huginn actual microbatch depth/gradient observations are missing")
            expected = [recipe.depth_at(item["step"] - 1, micro) for micro in range(len(depths))]
            require(depths == expected and gradients == [min(recipe.grad_visits, r) for r in expected],
                    "Huginn observed microbatch depth or gradient tail differs from its frozen sampler")
            observed.extend(depths)
        require(len(set(observed)) > 1 and max(observed) > recipe.grad_visits,
                "Huginn qualification did not exercise varying depths and a no-gradient prefix")
        return {"depth_min": min(observed), "depth_max": max(observed),
                "microbatches": len(observed), "gradient_tail": recipe.grad_visits}
    if spec.name in ("loom", "loom_attention"):
        require(model.state_update == "ema" and model.num_cells >= 2 and model.num_visits > 2
                and model.router_per_visit and model.attention_per_visit == (spec.name == "loom_attention")
                and model.memory_scale == .5 and model.memory_beta == .5
                and model.region_config("core").residual_scale == 1.
                and recipe.grad_visits is None and recipe.objective == "terminal_ce",
                "LOOM full-BPTT memory/reinjection/ownership configuration differs")
        calls = model.num_cells * model.num_visits
        calls += model.prefix_cells if model.region_config("prefix").num_experts else 0
        calls += model.suffix_cells if model.region_config("suffix").num_experts else 0
        for item in metrics:
            observed = item.get("differentiated_router_calls_per_micro", [])
            require(observed and len(observed) == training["identity"]["accumulation"]
                    and all(value == calls for value in observed),
                    "LOOM actual full-BPTT logical memory/router visits are missing")
        return {"state_update": "ema", "visits": model.num_visits, "cells": model.num_cells,
                "memory_scale": model.memory_scale, "memory_beta": model.memory_beta,
                "differentiated_calls_per_microbatch": calls}
    require(spec.name == "ouro" and model.exit_distribution and recipe.objective == "expected_ce"
            and recipe.grad_visits is None and model.num_visits > 1,
            "Ouro expected-loss/exit-distribution configuration is missing")
    entropies = []
    nondegenerate = False
    for item in metrics:
        require(all(type(item.get(key)) in (float, int) and math.isfinite(item[key])
                    for key in ("expected_ce", "entropy")), "Ouro expected CE/entropy observations are missing")
        probabilities, losses = item.get("mean_exit_probabilities", []), item.get("visit_ce", [])
        require(len(probabilities) == model.num_visits
                and all(type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                and math.isclose(sum(probabilities), 1., rel_tol=1e-5, abs_tol=1e-5),
                "Ouro actual tokenwise exit-probability aggregate is invalid")
        require(len(losses) == model.num_visits and all(math.isfinite(value) and value >= 0 for value in losses)
                and math.isclose(losses[-1], item["ce"], rel_tol=1e-5, abs_tol=1e-6),
                "Ouro per-visit readout losses or standard terminal CE differ")
        require(0 <= item["entropy"] <= math.log(model.num_visits) + 1e-5, "Ouro entropy is invalid")
        entropies.append(item["entropy"])
        nondegenerate |= sum(p > 1e-4 for p in probabilities) > 1
    require(max(entropies) > 0 and nondegenerate, "Ouro qualification has no nondegenerate exit distribution/entropy")
    return {"objective": "expected_ce", "entropy_max": max(entropies),
            "terminal_readout": "standard_lm_ce", "exit_probabilities_last": metrics[-1]["mean_exit_probabilities"]}


def verify_expected_updates(checkpoint):
    """Read real Adam moments for gate, head and backbone from the final DCP."""
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = dcp.FileSystemReader(checkpoint)
    state, groups = {}, {"gate": [], "head": [], "backbone": []}
    for name, item in reader.read_metadata().state_dict_metadata.items():
        if not name.startswith("optimizer.") or not name.endswith("exp_avg") or not isinstance(item, TensorStorageMetadata):
            continue
        group = ("gate" if "exit_gate" in name else "head" if "lm_head" in name else
                 "backbone" if any(part in name for part in (".attentions.", ".experts.", ".dense_ffn.")) else None)
        if group:
            state[name] = torch.empty(item.size, dtype=item.properties.dtype)
            groups[group].append(name)
    require(all(groups.values()), "Ouro checkpoint lacks actual gate/head/backbone optimizer moments")
    dcp.load(state, storage_reader=reader, no_dist=True)
    report = {}
    for group, names in groups.items():
        require(all(torch.isfinite(state[name]).all().item() for name in names), f"nonfinite {group} optimizer moments")
        count = sum(int(torch.count_nonzero(state[name]).item()) for name in names)
        require(count > 0, f"Ouro {group} has no observed nonzero optimizer update moments")
        report[group] = {"tensor_count": len(names), "nonzero_moment_elements": count}
    return report


def verify_inherited_v1(entry, *, accepted_path=None):
    """Authenticate accepted V1 artifacts without loading them in the new schema."""
    import torch.distributed.checkpoint as dcp
    accepted_path = (Path(__file__).resolve().parents[1] / "docs/evidence/p1.json"
                     if accepted_path is None else Path(accepted_path))
    accepted = read_json(accepted_path)
    require(isinstance(entry, dict) and all(entry.get(key) for key in (
        "directory", "release_sha256", "release_check_sha256", "source_p1", "source_p1_sha256")),
        "inherited V1 artifact identities are missing")
    directory, source_path = Path(entry["directory"]), Path(entry["source_p1"])
    publication = accepted["verification"]
    require(digest(directory / "release.json") == entry["release_sha256"], "inherited V1 release index SHA differs")
    require(digest(directory / "release-check.json") == entry["release_check_sha256"]
            == publication["retained_raw_reports"]["release-check.json"],
            "inherited V1 complete acceptance report SHA differs from the accepted publication")
    require(digest(directory / "finalization.json") == publication["retained_raw_reports"]["finalization.json"],
            "inherited V1 finalization SHA differs from the accepted publication")
    require(digest(source_path) == entry["source_p1_sha256"]
            == accepted["runtime"]["native"]["source_identity_sha256"],
            "inherited P1 original source identity SHA differs")
    source, report, index = read_json(source_path), read_json(directory / "release-check.json"), read_json(directory / "release.json")
    require(source["commit"] == accepted["training_source"]["commit"] and source.get("dirty") is False
            and source["files"] == accepted["training_source"]["files"], "inherited P1 frozen training source differs")
    require(report.get("status") == "passed" and report.get("stage") == "v1"
            and report.get("scope") == "v1_full_acceptance"
            and index.get("stage") == "v1" and index.get("scientific_anchor"),
            "inherited V1 is not the accepted complete stage gate")
    anchor = read_json(index["scientific_anchor"])
    fingerprint = accepted["data"]["manifest_fingerprint"]
    require(anchor.get("manifest_fingerprint") == report["scientific_anchor"]["manifest_fingerprint"] == fingerprint,
            "inherited P1 data identity differs")
    checker = anchor["verification_checker"]
    require(checker["sha256"] == publication["checker_sha256"] and digest(checker["path"]) == checker["sha256"],
            "inherited P1 original verification checker differs")
    for role in ("reference", "candidate"):
        run = Path(anchor["arms"][role])
        request, outcome = read_json(run / "request.json"), read_json(run / "outcome.json")
        require(request["source"].get("commit") == source["commit"]
                and all(source["files"].get(key) == value for key, value in request["source"]["files"].items()),
                f"inherited P1 {role} source differs")
        verify_source(run / "source", request["source"])
        require(outcome.get("status") == "completed"
                and outcome.get("completed_updates") == accepted["arms"][role]["completed_updates"]
                and outcome.get("global_progress", {}).get("label_tokens") == accepted["arms"][role]["actual_valid_labels"]
                and (Path(outcome["full_checkpoint"]) / ".metadata").is_file(),
                f"inherited P1 {role} final full checkpoint or actual progress is missing")
        checkpoint = Path(outcome["full_checkpoint"])
        provenance_path = run / "hf/export_provenance.json"
        require(digest(provenance_path) == accepted["arms"][role]["hf"]["export_provenance_sha256"],
                f"inherited P1 {role} published export provenance SHA differs")
        provenance = read_json(provenance_path)
        require(digest(checkpoint / ".metadata") == provenance["checkpoint"]["metadata_sha256"],
                f"inherited P1 {role} final checkpoint metadata SHA differs")
        metadata = dcp.FileSystemReader(checkpoint).read_metadata()
        shards = {value.relative_path for value in metadata.storage_data.values()}
        require(shards and all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0 for name in shards),
                f"inherited P1 {role} full checkpoint shards are missing")
    return {"status": "inherited", "scope": "v1_full_acceptance", "manifest_fingerprint": fingerprint,
            "training_source": source["commit"], "release_sha256": entry["release_sha256"],
            "release_check_sha256": entry["release_check_sha256"]}


def verify_sharing_specs(specs):
    require(set(specs) == {"a0_r0", "a0_r1", "a1_r0", "a1_r1"}, "four attention/router grid arms are missing")
    controlled = None
    for name, spec in specs.items():
        attention, router = bool(int(name[1])), bool(int(name[-1]))
        require(spec.model.attention_per_visit is attention and spec.model.router_per_visit is router,
                "sharing grid ownership does not identify its arm")
        require(spec.model.state_update == "residual" and not spec.model.exit_distribution
                and not spec.model.norm_per_visit and spec.training.grad_visits is None
                and not spec.training.sampled_visits and spec.training.objective == "terminal_ce",
                "sharing grid changed controlled state/normalization/objective semantics")
        model = spec.model.to_dict()
        for key in ("attention_per_visit", "router_per_visit", "visit_capacity"):
            model.pop(key)
        actual = (model, spec.to_dict()["training"], spec.optimizer, spec.inference, spec.run)
        require(controlled is None or actual == controlled, "sharing grid changed controlled geometry or training settings")
        controlled = actual


def verify_v2_release(directory):
    from moeloop.costs import estimate_cost
    from moeloop.data import PreparedManifest
    from qualification import compare_runs

    directory = Path(directory).resolve()
    index = read_json(directory / "release.json")
    require(index.get("format") == "moeloop-recipe-qualification" and index.get("version") == 1
            and index.get("stage") == "v2", "unsupported V2 release evidence format/schema")
    require(set(V2_METHODS) <= index.get("methods", {}).keys(), "required V2 method artifacts are missing")
    inherited = verify_inherited_v1(index.get("inherited_v1"))
    manifest = PreparedManifest.load(index["manifest"])
    require(index.get("manifest_fingerprint") == manifest.fingerprint == inherited["manifest_fingerprint"],
            "V2 must use the inherited public data/tokenizer identity")
    methods, probes = {}, {}
    for name in V2_METHODS:
        attempts = index["methods"][name]
        require(isinstance(attempts, list) and attempts, f"{name}: no actual native qualification run")
        training = verify_training(attempts[-1], manifest)
        require(training["spec"].name == name and training["spec"].run.get("seed") == 42,
                f"{name}: method/single-seed qualification identity differs")
        methods[name] = {"updates": training["steps"], "dynamics": training["dynamics"],
                         "hf": verify_hf(attempts[-1], training)}
        if name == "ouro":
            methods[name]["updates_observed"] = verify_expected_updates(training["receipt"]["checkpoint"])
        require(name in index.get("probes", {}), f"{name}: full-state continuation probe is missing")
        probe = index["probes"][name]
        one = verify_training(probe["uninterrupted"], manifest, qualification=False, expected_steps=8)
        two = verify_training(probe["resumed"], manifest, qualification=False, expected_steps=8)
        require(one["spec"].name == two["spec"].name == name, f"{name}: continuation recipe differs")
        stopped = [read_json(path) for path in Path(probe["resumed"]).glob("attempts/*/outcome.json")]
        load = two["resolved"]["checkpoint"]
        require(any(value.get("status") == "completed" and value.get("completed_updates") == 4 for value in stopped)
                and load.get("initial_load_path") and Path(load["initial_load_path"]).name == "step-4"
                and not load.get("initial_load_model_only"), f"{name}: actual step4 full-state resume is missing")
        compared = compare_runs(probe["uninterrupted"], probe["resumed"], expected_steps=8)
        compared["optimizer"] = compare_optimizer(one["receipt"]["checkpoint"], two["receipt"]["checkpoint"])
        original = read_json(probe["report"])
        require(original.get("status") == "passed" and original.get("expected_steps") == 8
                and original.get("optimizer", {}).get("tensor_count", 0) > 0,
                f"{name}: original continuation comparison is missing or failed")
        probes[name] = {"model_tensors": compared["model_tensor_count"], "optimizer": compared["optimizer"],
                        "training_ranks": compared["training_rank_count"], "data_ranks": compared["data_rank_count"]}
    grid = index.get("sharing_grid", {})
    require(grid.get("arms") and grid.get("comparison"), "attention/router grid artifacts are missing")
    report = read_json(grid["comparison"])
    require(report.get("format") == "moeloop-attention-router-grid" and report.get("version") == 1
            and report.get("manifest_fingerprint") == manifest.fingerprint and report.get("seed") == 42,
            "attention/router comparison identity is invalid")
    specs = {}
    for name, run in grid["arms"].items():
        training = verify_training(run, manifest, qualification=False, expected_steps=8)
        spec = specs[name] = training["spec"]
        cost = estimate_cost(spec, seq_len=training["request"]["seq_len"])
        arm = report["arms"].get(name, {})
        require("spec" in arm and same_recipe(arm["spec"], spec.to_dict()) and arm.get("cost") == json.loads(json.dumps(cost)),
                "attention/router comparison does not match actual recipes/costs")
    verify_sharing_specs(specs)
    return {"status": "passed", "stage": "v2", "scope": "v2_full_acceptance",
            "manifest_fingerprint": manifest.fingerprint, "inherited_v1": inherited,
            "methods": methods, "continuation": probes, "sharing_grid": {"arms": sorted(specs), "updates_per_arm": 8}}


def verify_release(directory):
    """The current V1 gate includes engineering, P1 and runtime evidence."""
    directory = Path(directory).resolve()
    index = read_json(directory / "release.json")
    require(index.get("format") == "moeloop-recipe-qualification" and index.get("version") == 1
            and index.get("stage") == "v1", "unsupported or missing release evidence format/schema")
    require(index.get("scientific_anchor"), "V1 requires the complete P1 scientific anchor; historical engineering passed is insufficient")
    require(index.get("performance") and index.get("multinode"), "V1 requires public-reference performance and actual multinode evidence")
    result = verify_engineering_release(directory)
    result["scope"] = "v1_full_acceptance"
    result["scientific_anchor"] = verify_scientific_anchor(index["scientific_anchor"])
    result["performance"] = verify_performance(index["performance"])
    result["multinode"] = verify_multinode(index["multinode"])
    return result


def verify_anchor_resumes(anchor_name, resumed):
    if anchor_name == "P2":
        require(set(resumed) == {"reference", "candidate"}, "P2 requires actual main-line full-state restore in both arms")
    else:
        require(resumed, "P1 requires at least one actual main-line strict full-state resume")


def verify_p2_geometry(training, role):
    spec, identity, request = training["spec"], training["identity"], training["request"]
    require(request["source"].get("dirty") is False and re.fullmatch(r"[0-9a-f]{40}", request["source"].get("commit", "")),
            f"{role}: P2 source is not frozen")
    model = spec.model
    expected = ((1280, 432, 64, 10, 0, 12, 1, 0) if role == "reference" else
                (1056, 384, 88, 12, 3, 6, 2, 3))
    actual = (model.dim, model.hidden_dim, model.num_experts, model.num_heads,
              model.prefix_cells, model.num_cells, model.num_visits, model.suffix_cells)
    require(actual == expected and model.top_k == 8 and model.num_kv_heads == 1,
            f"{role}: P2 approved second-scale geometry differs")
    frozen = read_json(Path(__file__).resolve().parents[1] / "recipes" / f"{spec.name}.json")
    require(spec.optimizer["base_lr"] in frozen["run"]["lr_trial_peak_lrs"],
            f"{role}: P2 common bounded pilot learning rate differs")
    require(training["steps"] == spec.run["schedule_total_steps"] == 19537
            and spec.run["global_batch_positions"] == 256 * 4096
            and request["world_size"] == identity["world_size"] == 16
            and request["global_batch_size"] == 256
            and request.get("local_batch_size") == identity["local_batch_size"] == 16
            and request.get("gradient_accumulation_steps") == identity["accumulation"] == 1,
            f"{role}: P2 full endpoint/data-rank/local batch/accumulation differs")
    require(identity["parallelism"].get("data_parallel_replicate_degree") == 2
            and identity["parallelism"].get("data_parallel_shard_degree") == 8,
            f"{role}: P2 approved HSDP2x8 topology differs")


def verify_p2_screen(pair, trainings):
    from moeloop.comparison import p1_attention_pairs, smelt_p2_pair
    from moeloop.model import ModelConfig
    controls = pair["comparison"]["controlled"]
    actual = smelt_p2_pair(base_lr=trainings["reference"]["spec"].optimizer["base_lr"],
        activation_checkpointing=controls["activation_checkpointing"],
        attention_pairs_per_sequence=controls.get("attention_pairs_per_sequence"),
        data=pair["comparison"]["data"], evaluation=pair["comparison"]["evaluation"])
    require(controls.get("attention_pairs_per_sequence") == p1_attention_pairs(),
            "P2 formal document-pair statistic differs from the frozen P1-prefix basis")
    require(pair.get("dense_comparison", {}).get("costs") == actual["dense_comparison"]["costs"],
            "P2 separate dense-S2 accounting differs")
    for field in ("candidate_count", "candidates", "selected_candidate_index", "selection_basis"):
        require(pair.get(field) == actual.get(field), f"P2 analytical candidate screen {field} differs")
    for role, training in trainings.items():
        require(same_recipe(pair["comparison"]["arms"][role], training["spec"].to_dict()),
                f"P2 selected {role} is not the frozen actual recipe")
        require(ModelConfig.from_dict(pair["comparison"]["arms"][role]["model"])
                == ModelConfig.from_dict(actual["comparison"]["arms"][role]["model"]),
                f"P2 {role} differs from pure-cost selected geometry")


def verify_inherited_v2(entry, *, accepted_path=None):
    """Retain accepted V2/P1 source identities while reading actual saved artifacts."""
    accepted_path = Path(accepted_path) if accepted_path else Path(__file__).resolve().parents[1] / "docs/evidence/recipes-v2.json"
    accepted = read_json(accepted_path)
    require(isinstance(entry, dict) and all(entry.get(key) for key in
            ("directory", "release_sha256", "release_check_sha256", "source_v2", "source_v2_sha256")),
            "inherited V2 index/report/source identities are missing")
    directory, source_path = Path(entry["directory"]), Path(entry["source_v2"])
    require(digest(directory / "release.json") == entry["release_sha256"], "inherited V2 release index SHA differs")
    require(digest(directory / "release-check.json") == entry["release_check_sha256"]
            == accepted["verification"]["release_check_sha256"], "inherited V2 accepted gate SHA differs")
    require(digest(source_path) == entry["source_v2_sha256"], "inherited V2 source receipt SHA differs")
    source, report, index = read_json(source_path), read_json(directory / "release-check.json"), read_json(directory / "release.json")
    require(source.get("dirty") is False and re.fullmatch(r"[0-9a-f]{40}", source.get("commit", "")),
            "inherited V2 source is not frozen")
    checker = source["checker"]
    require(digest(checker["path"]) == checker["sha256"] == accepted["verification"]["checker_sha256"],
            "inherited V2 original checker SHA differs")
    require(report.get("status") == "passed" and report.get("stage") == "v2"
            and report.get("scope") == "v2_full_acceptance" and index.get("stage") == "v2",
            "inherited V2 is not the complete accepted gate")
    for name in V2_METHODS:
        original = accepted["methods"][name]["training_source"]
        require(source["commit"] == original["commit"] and source["files"] == original["files"],
                f"inherited V2 {name} publication source differs")
        request = read_json(Path(index["methods"][name][-1]) / "request.json")
        require(request["source"]["commit"] == source["commit"]
                and request["source"]["files"] == source["files"], f"inherited V2 {name} actual source differs")
    result = verify_v2_release(directory)
    require(result["manifest_fingerprint"] == accepted["data"]["manifest_fingerprint"], "inherited V2 published data differs")
    return {"status": "inherited", "scope": "v2_full_acceptance", "manifest_fingerprint": result["manifest_fingerprint"],
            "training_source": source["commit"], "inherited_v1": result["inherited_v1"],
            "release_sha256": entry["release_sha256"], "release_check_sha256": entry["release_check_sha256"]}


def verify_v3_sharing(grid, manifest):
    from moeloop.costs import estimate_cost
    from moeloop.recipes import resolve, sharing_panel_specs
    expected = sharing_panel_specs()
    require(set(grid.get("arms", {})) == set(expected) and grid.get("comparison"), "complete C four-arm panel is missing")
    report = read_json(grid["comparison"])
    require(report.get("format") == "moeloop-attention-router-grid" and report.get("version") == 1
            and report.get("manifest_fingerprint") == manifest.fingerprint and report.get("seed") == 42,
            "C comparison identity differs")
    specs, observations = {}, {}
    for name, run in grid["arms"].items():
        training = verify_training(run, manifest, qualification=False, expected_steps=2048)
        spec, identity = training["spec"], training["identity"]
        verify_learning_observations(training, name)
        verify_native_optimizer_execution(training, {"optimizer": spec.optimizer,
            "activation_checkpointing": spec.run.get("activation_checkpointing", "none")}, name)
        resolved = resolve(spec)  # the archived spec predates RUN_DEFAULTS and carries removed keys
        require(spec.model == expected[name].model and spec.training == expected[name].training
                and resolved.optimizer == expected[name].optimizer
                and all(resolved.run.get(key) == value for key, value in expected[name].run.items() if key != "manifest_identity")
                and spec.run.get("manifest_identity") in (None, manifest.fingerprint),
                f"{name}: C frozen geometry/objective/optimizer/full cosine horizon differs")
        require(training["request"]["source"].get("dirty") is False
                and re.fullmatch(r"[0-9a-f]{40}", training["request"]["source"].get("commit", "")), f"{name}: C source is not frozen")
        require(identity.get("debug_seed") == 42 and identity.get("precision") == ["float32", "bfloat16", "float32"]
                and training["request"]["runtime"].get("cuda") and identity["world_size"] in (2, 4)
                and identity["parallelism"].get("data_parallel_shard_degree", 0) >= 2,
                f"{name}: C actual seed/BF16/CUDA/FSDP topology differs")
        require(identity["dataloader"].get("infinite") is False, f"{name}: C repeated training data")
        require(identity.get("lr_scheduler") == training["resolved"]["lr_scheduler"]
                and identity["lr_scheduler"].get("total_steps") == 2048
                and identity["lr_scheduler"].get("warmup_steps") == 64
                and identity["lr_scheduler"].get("min_lr_factor") == .1,
                f"{name}: C actual fixed scheduler differs")
        saved = checkpoint_continuation(training["receipt"]["checkpoint"], identity, expected_step=2048, finite_data=True)
        require(saved == training["receipt"]["outcome"]["global_progress"]["label_tokens"], f"{name}: C durable labels differ")
        cost = estimate_cost(spec, seq_len=2048)
        arm = report["arms"].get(name, {})
        require("spec" in arm and same_recipe(arm["spec"], spec.to_dict()) and arm.get("cost") == json.loads(json.dumps(cost)),
                f"{name}: C comparison does not match actual recipe/cost")
        specs[name] = spec
        observations[name] = {"request": training["request"], "identity": identity,
                              "terminal_ce": training["validations"][-1]["ce"], "updates": training["steps"],
                              "validation_slices": [(p["step"], p["valid_tokens"]) for p in training["validations"]]}
    verify_sharing_specs(specs)
    base = observations["a0_r0"]
    for name, arm in observations.items():
        require(arm["validation_slices"] == base["validation_slices"], f"{name}: C common validation slices differ")
        for field in ("source", "runtime", "local_batch_size", "gradient_accumulation_steps", "world_size"):
            require(arm["request"].get(field) == base["request"].get(field), f"{name}: C controlled {field} differs")
        for field in ("dataloader", "parallelism", "precision", "optimizer", "lr_scheduler", "execution"):
            require(arm["identity"].get(field) == base["identity"].get(field), f"{name}: C controlled native {field} differs")
    from recipe_report import sharing_interaction
    return sharing_interaction({name: arm["terminal_ce"] for name, arm in observations.items()}) | {"updates_per_arm": 2048}


def verify_p2_data_order(index):
    """Authenticate reuse of the original P1 document-pair cost statistic."""
    from moeloop.comparison import p1_attention_pairs
    v2 = read_json(Path(index["inherited_v2"]["directory"]) / "release.json")
    v1 = read_json(Path(v2["inherited_v1"]["directory"]) / "release.json")
    p1, p2 = read_json(v1["scientific_anchor"]), read_json(index["p2"])
    for role in ("reference", "candidate"):
        first, second = Path(p1["arms"][role]), Path(p2["arms"][role])
        original, actual = read_json(first / "continuation_identity.json"), read_json(second / "continuation_identity.json")
        require(all(original.get(key) == actual.get(key) for key in
                    ("manifest", "dataloader", "world_size", "local_batch_size", "accumulation", "seq_len")),
                f"{role}: P2 data rank streams/order/packing differ from frozen P1")
        before, after = read_json(first / "request.json"), read_json(second / "request.json")
        require(before["source"]["files"]["src/moeloop/data.py"] == after["source"]["files"]["src/moeloop/data.py"],
                f"{role}: reused P1 document statistics lack unchanged data source")
    return {"basis": "identical P1 manifest/loader/data source/world/local batch/accumulation/sequence streams",
            "attention_pairs_per_sequence": p1_attention_pairs()}


def verify_partial_sharing_profile(directory):
    """C already verifies shared experts/norms; profiling needs an independent A/R owner."""
    model = read_json(Path(directory) / "spec.json")["model"]
    require(model.get("attention_per_visit") is True or model.get("router_per_visit") is True,
            "partial-sharing performance requires an actual per-visit attention or router owner")


def verify_v3_performance_arm(index):
    selected = Path(read_json(index["performance"])["windows"]["sharing"]["run"]).resolve()
    require(selected in {Path(run).resolve() for run in index["sharing_grid"]["arms"].values()},
            "partial-sharing performance must use an actual declared C run")
    verify_partial_sharing_profile(selected)
    return selected


def verify_v3_analysis(path, index):
    from recipe_report import build_report
    actual = build_report(index)
    require(read_json(path) == actual, "analysis differs from raw-derived recomputation")
    require(all(panel["common_compute"] for panel in actual.get("anchors", {}).values()), "observed common-compute curve overlap is missing")
    return actual


def verify_v3_delivery(index):
    from recipe_delivery import verify_downstream_receipt, verify_journey_receipt
    p2 = read_json(index["p2"])
    v2 = read_json(Path(index["inherited_v2"]["directory"]) / "release.json")
    v1 = read_json(Path(v2["inherited_v1"]["directory"]) / "release.json")
    p1 = read_json(v1["scientific_anchor"])
    models = {f"p{number}_{kind}": str((Path(anchor["arms"][role]) / "hf").resolve())
              for number, anchor in ((1, p1), (2, p2)) for role, kind in (("reference", "moe"), ("candidate", "smelt"))}
    downstream = verify_downstream_receipt(index["lm_eval"], expected_models=models)
    package = verify_journey_receipt(index["package"])
    require(package.get("preparation") is not None,
            "installed journey must include the public preparation command")
    return {"lm_eval": {"path": index["lm_eval"], "sha256": digest(index["lm_eval"]),
                        "models": sorted(downstream["models"])},
            "package": {"path": index["package"], "sha256": digest(index["package"])}}


def verify_v3_release(directory):
    directory = Path(directory).resolve()
    index = read_json(directory / "release.json")
    require(index.get("format") == "moeloop-recipe-qualification" and index.get("version") == 1
            and index.get("stage") == "v3", "unsupported V3 release index")
    require(all(index.get(key) for key in ("manifest", "manifest_fingerprint", "inherited_v2", "p2", "sharing_grid",
                                         "performance", "lm_eval", "package", "analysis")),
            "V3 required P2/C/performance/downstream/package/analysis artifacts are missing")
    from moeloop.data import PreparedManifest
    inherited = verify_inherited_v2(index["inherited_v2"])
    manifest = PreparedManifest.load(index["manifest"])
    require(index["manifest_fingerprint"] == manifest.fingerprint == inherited["manifest_fingerprint"],
            "V3 must preserve the original P1/V2 manifest/tokenizer/data-order identity")
    p2 = verify_scientific_anchor(index["p2"], anchor_name="P2")
    require(p2["manifest_fingerprint"] == manifest.fingerprint, "P2 manifest differs from inherited P1")
    p2["data_order"] = verify_p2_data_order(index)
    sharing = verify_v3_sharing(index["sharing_grid"], manifest)
    verify_v3_performance_arm(index)
    analysis = verify_v3_analysis(index["analysis"], index)
    delivery = verify_v3_delivery(index)
    return {"status": "passed", "stage": "v3", "scope": "v3_full_acceptance", "manifest_fingerprint": manifest.fingerprint,
            "inherited_v2": inherited, "p2": p2, "sharing_grid": sharing, "performance": analysis["performance"],
            "analysis": {"path": index["analysis"], "sha256": digest(index["analysis"])}, "delivery": delivery}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("v1", "v2", "v3"), required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--engineering-only", action="store_true",
                        help="verify historical four-method engineering evidence only; does not satisfy V1 acceptance")
    args = parser.parse_args(argv)
    if args.stage != "v1" and args.engineering_only:
        parser.error("--engineering-only applies only to historical V1 evidence")
    try:
        result = (verify_v3_release(args.evidence) if args.stage == "v3" else
                  verify_v2_release(args.evidence) if args.stage == "v2" else
                  verify_engineering_release(args.evidence) if args.engineering_only else verify_release(args.evidence))
    except Exception as error:
        print(json.dumps({"status": "failed", "stage": args.stage, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
