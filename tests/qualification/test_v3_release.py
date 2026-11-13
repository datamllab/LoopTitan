"""V3 only accepts complete source-bound panels; no scientific winning gate."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_v3_requires_actual_panels_even_with_passed_marker(tmp_path):
    checker = importlib.import_module("check_release")
    (tmp_path / "release.json").write_text(json.dumps({
        "format": "moeloop-recipe-qualification", "version": 1, "stage": "v3", "passed": True,
        "inherited_v2": {"status": "passed"}}))
    with pytest.raises(ValueError, match="required|missing"):
        checker.verify_v3_release(tmp_path)
    result = subprocess.run([sys.executable, checker.__file__, "--stage", "v3", "--evidence", str(tmp_path)],
                            capture_output=True, text=True,
                            env=os.environ | {"PYTHONPATH": str(Path(checker.__file__).resolve().parents[1] / "src")})
    assert result.returncode == 1 and '"stage": "v3"' in result.stderr
    assert "required" in result.stderr or "missing" in result.stderr


def test_p2_requires_both_restored_arms():
    checker = importlib.import_module("check_release")
    checker.verify_anchor_resumes("P1", ["reference"])
    with pytest.raises(ValueError, match="both|each"):
        checker.verify_anchor_resumes("P2", ["reference"])
    checker.verify_anchor_resumes("P2", ["reference", "candidate"])


def test_analysis_gate_recomputes_instead_of_trusting_passed(tmp_path, monkeypatch):
    checker = importlib.import_module("check_release")
    reporter = importlib.import_module("recipe_report")
    raw = {"format": "moeloop-recipe-analysis", "version": 1, "sharing_grid": {"I_AR": -.2}}
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(reporter, "build_report", lambda index: raw)
    assert checker.verify_v3_analysis(path, {}) == raw
    path.write_text(json.dumps(raw | {"passed": True, "sharing_grid": {"I_AR": .5}}))
    with pytest.raises(ValueError, match="raw|recomput"):
        checker.verify_v3_analysis(path, {})


def _p2_training(role):
    from moeloop.recipes import p2_moe_spec, p2_smelt_spec
    spec = p2_moe_spec() if role == "reference" else p2_smelt_spec()
    return {"spec": spec, "steps": 19537,
            "request": {"source": {"commit": "a" * 40, "dirty": False}, "world_size": 16,
                        "global_batch_size": 256, "local_batch_size": 16, "gradient_accumulation_steps": 1},
            "identity": {"world_size": 16, "local_batch_size": 16, "accumulation": 1,
                         "parallelism": {"data_parallel_replicate_degree": 2, "data_parallel_shard_degree": 8}}}


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_p2_keeps_original_data_microbatch_groups_and_full_endpoint(role):
    pytest.importorskip("torch")
    checker = importlib.import_module("check_release")
    training = _p2_training(role)
    checker.verify_p2_geometry(training, role)
    training["identity"]["accumulation"] = 2
    with pytest.raises(ValueError, match="batch|accumulation"):
        checker.verify_p2_geometry(training, role)


def test_p2_screen_is_recomputed_from_costs_without_any_quality_input():
    pytest.importorskip("torch")
    from moeloop.comparison import smelt_p2_pair
    from moeloop.recipe_spec import RecipeSpec
    checker = importlib.import_module("check_release")
    pair = smelt_p2_pair(attention_pairs_per_sequence=4_180_216.9229844734)
    training = {role: {"spec": RecipeSpec.from_dict(spec)} for role, spec in pair["comparison"]["arms"].items()}
    checker.verify_p2_screen(pair, training)
    pair["candidates"][0]["hidden_dim"] += 8
    with pytest.raises(ValueError, match="screen"):
        checker.verify_p2_screen(pair, training)


def test_inherited_v2_source_is_bound_to_accepted_checker_and_publication(tmp_path, monkeypatch):
    checker = importlib.import_module("check_release")
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    old_checker = tmp_path / "original-checker.py"
    old_checker.write_text("# immutable accepted fixture\n")
    source = {"commit": "a" * 40, "dirty": False, "files": {"src/moeloop/model.py": "b" * 64},
              "checker": {"path": str(old_checker), "sha256": checker.digest(old_checker)}}
    source_path = put(tmp_path / "source-v2.json", source)
    directory = tmp_path / "accepted-v2"
    methods = {}
    for name in checker.V2_METHODS:
        run = tmp_path / name
        put(run / "request.json", {"source": source})
        methods[name] = [str(run)]
    release = put(directory / "release.json", {"stage": "v2", "methods": methods})
    report = put(directory / "release-check.json", {"status": "passed", "stage": "v2", "scope": "v2_full_acceptance"})
    accepted = put(tmp_path / "accepted.json", {
        "verification": {"release_check_sha256": checker.digest(report), "checker_sha256": checker.digest(old_checker)},
        "methods": {name: {"training_source": source} for name in methods}, "data": {"manifest_fingerprint": "c" * 64}})
    entry = {"directory": str(directory), "release_sha256": checker.digest(release),
             "release_check_sha256": checker.digest(report), "source_v2": str(source_path), "source_v2_sha256": checker.digest(source_path)}
    # Isolate identity-chain logic here; native V2 verification has its own tests.
    monkeypatch.setattr(checker, "verify_v2_release", lambda directory: {"manifest_fingerprint": "c" * 64, "inherited_v1": {"training_source": "1bea"}})
    result = checker.verify_inherited_v2(entry, accepted_path=accepted)
    assert result["training_source"] == "a" * 40
    assert result["inherited_v1"]["training_source"] == "1bea"
    old_checker.write_text("# replaced with new HEAD\n")
    with pytest.raises(ValueError, match="checker SHA"):
        checker.verify_inherited_v2(entry, accepted_path=accepted)


def _p2_native_controls():
    """Portable projection of the real source17 95-update pilot metadata shape.

    The frozen specs/comparison have no retroactive frozen_runtime field. The
    pilot stop is 95; each immutable recipe/request/scheduler keeps horizon19537.
    """
    import copy
    from moeloop.comparison import smelt_p2_pair
    from moeloop.recipe_spec import RecipeSpec
    fingerprint = "b" * 64
    comparison = smelt_p2_pair(base_lr=3e-4, activation_checkpointing="none",
        attention_pairs_per_sequence=4_180_216.9229844734,
        data={"manifest_fingerprint": fingerprint, "order": "P1 physical16/seed42/finite/rank-stride",
              "global_batch": 256, "local_batch": 16, "accumulation": 1},
        evaluation={"manifest_fingerprint": fingerprint, "context": 4096, "readout": "terminal",
                    "heldout": "complete validation/test; original native fixed subset"})["comparison"]
    runtime = {"python": "3.12.15", "torch": "2.14.0+cu130", "cuda": "13.0",
               "gpu_name": "NVIDIA H200", "compute_capability": [9, 0]}
    optimizer = {"param_groups": [
        {"pattern": "(?:norm|embedding|lm_head|router)", "optimizer_name": "AdamW",
         "optimizer_kwargs": {"lr": 3e-4, "betas": [.9, .95], "eps": 1e-8, "weight_decay": 0.}},
        {"pattern": ".*", "optimizer_name": "AdamW",
         "optimizer_kwargs": {"lr": 3e-4, "betas": [.9, .95], "eps": 1e-8, "weight_decay": .1}}],
        "implementation": "fused", "optimizer_factory_kwargs_by_name": {}}
    execution = {"native_experts": True, "flex_attention": True, "activation_checkpoint": None,
                 "compile": {"enable": False, "enable_async_tensor_parallel": False,
                             "components": ["model", "loss"], "backend": "inductor"}}
    parallelism = {"data_parallel_replicate_degree": 2, "data_parallel_shard_degree": 8,
                   "tensor_parallel_degree": 1, "pipeline_parallel_degree": 1,
                   "expert_parallel_degree": 1, "context_parallel_degree": 1,
                   "enable_sequence_parallel": False, "spmd_backend": "partial_dtensor"}
    source = {"commit": "a" * 40, "dirty": False, "files": {"src/moeloop/titan.py": "c" * 64}}
    trainings = {}
    for role, values in comparison["arms"].items():
        spec = RecipeSpec.from_dict(values)
        scheduler = {key: spec.run["schedule"][key] for key in
                     ("total_steps", "warmup_steps", "decay_ratio", "decay_type", "min_lr_factor")}
        trainings[role] = {"spec": spec, "steps": 19537,
            "request": {"source": copy.deepcopy(source), "runtime": runtime.copy(), "world_size": 16,
                        "global_batch_size": 256, "local_batch_size": 16, "gradient_accumulation_steps": 1},
            "identity": {"world_size": 16, "local_batch_size": 16, "accumulation": 1, "seq_len": 4096,
                         "optimizer": copy.deepcopy(optimizer), "max_norm": 1., "execution": copy.deepcopy(execution),
                         "parallelism": parallelism.copy(), "precision": ["float32", "bfloat16", "float32"],
                         "lr_scheduler": scheduler.copy(), "torchtitan_commit": "086bf6c166ec85c1298eb5596fa9bf95f6a2d840"},
            "resolved": {"training": {"steps": 95, "local_batch_size": 16, "global_batch_size": 256,
                                       "seq_len": 4096, "dtype": "float32", "mixed_precision_param": "bfloat16",
                                       "mixed_precision_reduce": "float32"},
                         "parallelism": parallelism.copy(), "lr_scheduler": scheduler.copy(),
                         "validator": {"target_label_tokens": 1_000_000,
                                       "dataloader": {"infinite": False, "shuffle": False, "split": "validation"}}},
            "validations": [{"step": 95, "target_label_tokens": 1_000_000, "valid_tokens": 1_048_576,
                             "steps": 1, "configured_steps": 1, "world_size": 16, "local_batch_size": 16,
                             "seq_len": 4096, "infinite": False, "finite_batch_capacity_per_rank": [8] * 16}]}
    return trainings, comparison


def test_p2_runtime_uses_frozen_requests_without_rewriting_original_comparison():
    pytest.importorskip("torch")
    import copy
    checker = importlib.import_module("check_release")
    trainings, comparison = _p2_native_controls()
    original = copy.deepcopy(comparison)
    assert all("frozen_runtime" not in t["spec"].run for t in trainings.values())
    assert "frozen_runtime" not in comparison["controlled"]
    actual, runtime = checker.verify_anchor_comparison(trainings, comparison, anchor_name="P2")
    assert comparison == original
    assert actual["controlled"] == comparison["controlled"]
    assert runtime["world_size"] == runtime["local_batch_size"] == 16
    assert runtime["gradient_accumulation_steps"] == 1
    assert runtime["source"]["commit"] == "a" * 40
    with pytest.raises(ValueError, match="frozen runtime"):
        checker.verify_p1_controls(trainings, comparison)


@pytest.mark.parametrize("mismatch", ["software", "topology", "precision", "optimizer", "validation", "scheduler"])
def test_p2_runtime_rejects_actual_controls_that_differ_from_frozen_contract(mismatch):
    pytest.importorskip("torch")
    checker = importlib.import_module("check_release")
    trainings, comparison = _p2_native_controls()
    candidate = trainings["candidate"]
    if mismatch == "software":
        candidate["request"]["runtime"]["torch"] = "2.13.0+cu130"
    elif mismatch == "topology":
        # Both arms reporting the same wrong topology must still fail.
        for training in trainings.values():
            training["identity"]["parallelism"]["data_parallel_replicate_degree"] = 1
            training["identity"]["parallelism"]["data_parallel_shard_degree"] = 16
            training["resolved"]["parallelism"] = training["identity"]["parallelism"].copy()
    elif mismatch == "precision":
        candidate["identity"]["precision"][2] = "bfloat16"
    elif mismatch == "optimizer":
        candidate["identity"]["optimizer"]["param_groups"][0]["optimizer_kwargs"]["lr"] = 1e-4
    elif mismatch == "validation":
        candidate["validations"][0]["valid_tokens"] = 10
    else:
        candidate["identity"]["lr_scheduler"]["total_steps"] = 95
    with pytest.raises(ValueError):
        checker.verify_anchor_comparison(trainings, comparison, anchor_name="P2")


@pytest.mark.parametrize("attention,router", [(False, False), (True, False), (False, True), (True, True)])
def test_v3_partial_profile_checks_actual_c_owner_semantics(tmp_path, attention, router):
    checker = importlib.import_module("check_release")
    # Flags come from the actual selected run's spec, not an arm-name promise.
    run = tmp_path / "selected-c-arm"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps({"model": {
        "attention_per_visit": attention, "router_per_visit": router}}))
    profile = tmp_path / "performance.json"
    profile.write_text(json.dumps({"windows": {"sharing": {"run": str(run)}}}))
    index = {"performance": str(profile), "sharing_grid": {"arms": {"a1_r0": str(run)}}}
    if attention or router:
        assert checker.verify_v3_performance_arm(index) == run.resolve()
    else:
        with pytest.raises(ValueError, match="partial|per.visit"):
            checker.verify_v3_performance_arm(index)
