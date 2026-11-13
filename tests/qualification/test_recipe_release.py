"""Release evidence is checked from artifacts, including optimizer continuation."""

import importlib
import copy
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_development_geometry_keeps_method_semantics_and_resolved_run_values():
    runner = importlib.import_module("recipe_qualification")
    spec = runner.method_spec("foil", vocab_size=64, seq_len=32, steps=8, global_batch_size=4)
    assert spec.model.dim == 128 and spec.model.hidden_dim == 256
    assert spec.model.region_config("core").num_experts == 8
    assert spec.model.region_config("prefix").num_experts == 4
    assert spec.model.attention_per_visit
    assert spec.run["successful_updates"] == spec.run["schedule_total_steps"] == 8
    assert spec.run["global_batch_positions"] == 128
    assert spec.run["seq_len"] == 32
    tying = runner.method_spec("expert_tying", vocab_size=64, seq_len=32, steps=8, global_batch_size=4)
    assert tying.optimizer["expert_lr_multiplier"] == pytest.approx(.5)
    assert tying.model.order == "layer" and tying.model.router_per_visit


def test_short_runs_require_explicit_pilot_and_release_requires_two_ranks():
    runner = importlib.import_module("recipe_qualification")
    with pytest.raises(ValueError, match="pilot"):
        runner.validate_run(steps=8, seq_len=2048, global_batch_size=16, gpus=2, pilot=False)
    with pytest.raises(ValueError, match="two|2"):
        runner.validate_run(steps=512, seq_len=2048, global_batch_size=16, gpus=1, pilot=False)
    runner.validate_run(steps=8, seq_len=32, global_batch_size=4, gpus=1, pilot=True)


def test_checker_rejects_passed_boolean_without_training_artifacts(tmp_path):
    checker = importlib.import_module("check_release")
    (tmp_path / "release.json").write_text(json.dumps({"stage": "v1", "passed": True}))
    with pytest.raises((ValueError, AssertionError), match="schema|format|method"):
        checker.verify_release(tmp_path)
    result = subprocess.run([sys.executable, str(Path(checker.__file__)),
        "--stage", "v1", "--evidence", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0


def test_v1_checker_rejects_historical_engineering_index_without_scientific_anchor(tmp_path):
    checker = importlib.import_module("check_release")
    (tmp_path / "release.json").write_text(json.dumps({
        "format": "moeloop-recipe-qualification", "version": 1, "stage": "v1",
        "methods": {name: ["historical-engineering-run"] for name in ("smelt", "meta", "foil", "expert_tying")},
        "passed": True,
    }))
    with pytest.raises(ValueError, match="P1|scientific|anchor"):
        checker.verify_release(tmp_path)
    result = subprocess.run([sys.executable, str(Path(checker.__file__)),
        "--stage", "v1", "--evidence", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 1 and "anchor" in result.stderr


def test_scientific_gate_rejects_a_passed_label_without_both_p1_arms(tmp_path):
    checker = importlib.import_module("check_release")
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps({"format": "moeloop-scientific-anchor", "version": 1,
        "anchor": "P1", "status": "passed", "arms": {"reference": "not-a-real-run"}}))
    with pytest.raises(ValueError, match="both|arms"):
        checker.verify_scientific_anchor(path)


def _p1_native_controls(*, base_lr=1e-4, activation_checkpointing="none"):
    from moeloop.comparison import compare_recipes
    from moeloop.recipes import p1_moe_spec, p1_smelt_spec

    trainings = {}
    runtime = {"python": "unit-python", "torch": "unit-torch", "cuda": "unit-cuda"}
    frozen_runtime = {"world_size": 8, "local_batch_size": 16,
        "parallelism": {"data_parallel_replicate_degree": 1, "data_parallel_shard_degree": 8},
        "runtime": runtime, "gradient_accumulation_steps": 2,
        "runtime_identity_sha256": "a" * 64, "source_identity_sha256": "b" * 64}
    for role, factory in (("reference", p1_moe_spec), ("candidate", p1_smelt_spec)):
        spec = factory(base_lr=base_lr)
        spec.run["activation_checkpointing"] = activation_checkpointing
        spec.run["frozen_runtime"] = copy.deepcopy(frozen_runtime)
        optimizer = {"param_groups": [
            {"pattern": "(?:norm|embedding|lm_head|router)", "optimizer_name": "AdamW",
             "optimizer_kwargs": {"lr": base_lr, "betas": [.9, .95], "eps": 1e-8, "weight_decay": 0.}},
            {"pattern": ".*", "optimizer_name": "AdamW",
             "optimizer_kwargs": {"lr": base_lr, "betas": [.9, .95], "eps": 1e-8, "weight_decay": .1}},
        ], "implementation": "fused", "optimizer_factory_kwargs_by_name": {}}
        execution = {"native_experts": True, "flex_attention": True,
            "compile": {"enable": False, "enable_async_tensor_parallel": False,
                        "components": ["model", "loss"], "backend": "inductor"},
            "activation_checkpoint": None if activation_checkpointing == "none" else
                {"preserve_rng_state": True, "determinism_check": "default", "debug": False}}
        trainings[role] = {"spec": spec, "identity": {"optimizer": optimizer, "max_norm": 1., "execution": execution,
                "world_size": 8, "local_batch_size": 16, "seq_len": 4096, "accumulation": 2,
                "parallelism": {"data_parallel_replicate_degree": 1, "data_parallel_shard_degree": 8}},
            "request": {"runtime": runtime.copy()},
            "resolved": {"validator": {"target_label_tokens": 1_000_000,
                "dataloader": {"infinite": False, "shuffle": False, "split": "validation"}}},
            "validations": [{"step": 1, "target_label_tokens": 1_000_000, "valid_tokens": 1_048_576,
                "steps": 2, "configured_steps": 2, "world_size": 8, "local_batch_size": 16,
                "seq_len": 4096, "infinite": False, "finite_batch_capacity_per_rank": [18] * 8}]}
    comparison = compare_recipes(trainings["reference"]["spec"], trainings["candidate"]["spec"],
        seq_len=4096, activation_checkpointing=activation_checkpointing)
    comparison["controlled"]["frozen_runtime"] = copy.deepcopy(frozen_runtime)
    return trainings, comparison


@pytest.mark.parametrize("policy", ["none", "full_cells"])
def test_p1_controls_accept_the_declared_lr_and_checkpoint_policy(policy):
    checker = importlib.import_module("check_release")
    trainings, comparison = _p1_native_controls(activation_checkpointing=policy)
    checker.verify_p1_controls(trainings, comparison)


@pytest.mark.parametrize("override", ["lr", "clip", "ac", "compile"])
def test_p1_controls_reject_equal_cli_overrides_with_stale_specs(override):
    checker = importlib.import_module("check_release")
    trainings, comparison = _p1_native_controls()
    # Both arms are changed alike: arm equality cannot validate an undeclared run.
    for training in trainings.values():
        identity = training["identity"]
        if override == "lr":
            for group in identity["optimizer"]["param_groups"]:
                group["optimizer_kwargs"]["lr"] = 3e-4
        elif override == "clip":
            identity["max_norm"] = 2.
        elif override == "ac":
            identity["execution"]["activation_checkpoint"] = {
                "preserve_rng_state": True, "determinism_check": "default", "debug": False}
        else:
            identity["execution"]["compile"]["enable"] = True
    with pytest.raises(ValueError, match="frozen|declared"):
        checker.verify_p1_controls(trainings, comparison)


@pytest.mark.parametrize("override", ["batch", "world"])
def test_p1_controls_reject_equal_runtime_overrides_after_freezing(override):
    checker = importlib.import_module("check_release")
    trainings, comparison = _p1_native_controls()
    for training in trainings.values():
        identity = training["identity"]
        point = training["validations"][0]
        if override == "batch":
            identity["local_batch_size"] = point["local_batch_size"] = 8
        else:
            identity["world_size"] = point["world_size"] = 4
            identity.setdefault("parallelism", {})["data_parallel_shard_degree"] = 4
            point["finite_batch_capacity_per_rank"] = [18] * 4
        identity["accumulation"] = 4
        point["steps"] = point["configured_steps"] = 4
    with pytest.raises(ValueError, match="frozen.*runtime|runtime.*frozen"):
        checker.verify_p1_controls(trainings, comparison)


def test_p1_controls_reject_different_declared_execution_between_arms():
    checker = importlib.import_module("check_release")
    trainings, comparison = _p1_native_controls()
    trainings["candidate"]["spec"].run["compile"] = {"enable": True}
    trainings["candidate"]["identity"]["execution"]["compile"]["enable"] = True
    comparison["arms"]["candidate"] = trainings["candidate"]["spec"].to_dict()
    with pytest.raises(ValueError, match="different controlled execution"):
        checker.verify_p1_controls(trainings, comparison)


@pytest.mark.parametrize("override", ["infinite", "steps", "labels", "saved_target"])
def test_p1_controls_reject_stale_or_wrapped_validation_budgets(override):
    checker = importlib.import_module("check_release")
    trainings, comparison = _p1_native_controls()
    training = trainings["candidate"]
    point = training["validations"][0]
    if override == "infinite":
        point["infinite"] = True
    elif override == "steps":
        point["steps"] = point["configured_steps"] = 31
    elif override == "labels":
        point["valid_tokens"] = 16_252_928
    else:
        training["resolved"]["validator"]["target_label_tokens"] = None
    with pytest.raises(ValueError, match="validation"):
        checker.verify_p1_controls(trainings, comparison)


def _document_loss_evaluation(tmp_path, manifest, *, name="reference", depth=1):
    from moeloop.evaluate import evaluate_heldout
    from moeloop.model import ModelConfig, RecursiveLM
    path, output, model = tmp_path / f"{name}-documents.json", tmp_path / f"{name}-evaluation.json", tmp_path / name / "hf"
    core = RecursiveLM(ModelConfig(vocab_size=8, dim=8, hidden_dim=16, num_heads=2,
        num_kv_heads=1, num_experts=0, num_cells=1, num_visits=depth, max_seq_len=8))
    evaluation = evaluate_heldout(core, manifest, seq_len=4, document_loss_output=path)
    sidecar = json.loads(path.read_text())
    command = ["python", "-m", "moeloop.evaluate", "--model", str(model), "--manifest", str(manifest.path),
               "--output", str(output), "--document-loss-output", str(path)]
    return evaluation, sidecar, path, output, model, [command]


def _document_loss_manifest(tmp_path):
    from moeloop.prepare import prepare_dataset
    return prepare_dataset(tmp_path / "data", {"public": {"validation": ["one", "two"]}},
        encode=lambda _: [3, 4], tokenizer={"id": "unit", "revision": "fixed", "vocab_size": 8,
            "bos_token_id": 1, "eos_token_id": 2})


def test_p1_accepts_real_document_pairing_with_different_losses_and_depths(tmp_path):
    checker = importlib.import_module("check_release")
    manifest = _document_loss_manifest(tmp_path)
    evaluation, _, _, output, model, commands = _document_loss_evaluation(tmp_path, manifest)
    expected = checker.verify_p1_document_losses(evaluation, manifest, commands, output, model)
    other, _, _, output, model, commands = _document_loss_evaluation(tmp_path, manifest, name="candidate", depth=2)
    assert other["ce"] != evaluation["ce"]
    checker.verify_p1_document_losses(other, manifest, commands, output, model, expected_documents=expected)


@pytest.mark.parametrize("failure", ["hash", "metadata", "coverage", "sum", "cli", "pair"])
def test_p1_document_losses_reject_unbound_or_unpaired_records(tmp_path, failure):
    checker = importlib.import_module("check_release")
    manifest = _document_loss_manifest(tmp_path)
    evaluation, sidecar, path, output, model, commands = _document_loss_evaluation(tmp_path, manifest)
    expected = ((0, 3), (1, 3))
    if failure == "hash":
        path.write_text(path.read_text() + "\n")
    else:
        if failure == "metadata":
            sidecar["source"] = "another-source"
        elif failure == "coverage":
            sidecar["documents"].pop()
            evaluation["document_losses"]["document_count"] = 1
        elif failure == "sum":
            row = sidecar["documents"][0]
            row["ce_sum"] *= .5
            row["ce"] = row["ce_sum"] / row["valid_tokens"]
        elif failure == "cli":
            del commands[0][-2:]
        else:
            for row, tokens in zip(sidecar["documents"], (2, 4)):
                row.update(valid_tokens=tokens, ce=row["ce_sum"] / tokens)
        path.write_text(json.dumps(sidecar))
        evaluation["document_losses"]["sha256"] = checker.digest(path)
    with pytest.raises(ValueError, match="document|paired"):
        checker.verify_p1_document_losses(evaluation, manifest, commands, output, model, expected_documents=expected)


def test_public_performance_gate_rejects_an_unmeasured_passed_label(tmp_path):
    checker = importlib.import_module("check_release")
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"schema_version": 1, "status": "passed",
        "settings": {"reference": "torchtitan", "mode": "describe"}, "correctness": None,
        "performance": {"status": "not_measured"}}))
    with pytest.raises(ValueError, match="mathematical|parity|missing"):
        checker.verify_performance({"benchmarks": [str(path)]})


def test_component_math_still_enforces_the_original_raw_vector_and_peak_bounds():
    from recipe_benchmark import TOLERANCES

    checker = importlib.import_module("check_release")
    tolerance = TOLERANCES["bfloat16"]
    raw = {"max_absolute": 1e-5, "relative_l2": .04, "reference_peak": 1.,
           "max_absolute_allowed": tolerance["peak_factor"] + tolerance["floor"]}
    with pytest.raises(ValueError, match="component|bound"):
        checker.verify_benchmark_error(raw, tolerance, "component vector")
    raw.update(relative_l2=.01, max_absolute=1.)
    with pytest.raises(ValueError, match="component|bound"):
        checker.verify_benchmark_error(raw, tolerance, "component peak")


def test_component_passed_label_cannot_replace_missing_retained_raw_gradient_coverage():
    from recipe_benchmark import TOLERANCES

    checker = importlib.import_module("check_release")
    scope = "component_equivalence_with_observed_end_to_end_bf16_route_divergence"
    report = {"settings": {"reference": "torchtitan", "dtype": "bfloat16", "component_math": True},
        "mathematics_scope": scope, "tolerances_predeclared": TOLERANCES["bfloat16"],
        "conditions": {"timed_routing": "natural_unmodified_router_outputs"},
        "correctness": {"status": "failed", "failures": ["gradient drift"],
            "all_parameter_gradients_checked": True, "optimizer_updates": 1,
            "physical_parameter_elements": 1, "physical_parameter_tensors": 1,
            "candidate_ce": 1., "reference_ce": 1., "gradient_errors": {},
            **dict.fromkeys(("initial_parameters_equal", "losses_finite", "ce_close",
                             "all_gradients_finite", "all_updated_parameters_finite"), True)},
        "component_math": {"status": "passed", "scope": scope, "tolerances": TOLERANCES["bfloat16"],
            "timed_routing": "natural_unmodified_router_outputs"}}
    with pytest.raises(ValueError, match="coverage"):
        checker.verify_benchmark_math(report, "unit-component-report")


def test_archived_engineering_budget_metadata_remains_compatible_but_numeric_changes_fail():
    from moeloop.comparison import compare_recipes

    checker = importlib.import_module("check_release")
    runner = importlib.import_module("recipe_qualification")
    first = runner.method_spec("smelt", vocab_size=64, seq_len=32, steps=8, global_batch_size=4)
    actual = compare_recipes(first, first, seq_len=32)
    archived = copy.deepcopy(actual)
    for cost in archived["costs"].values():
        del cost["flops"]["attention_pairs_per_sequence"]
    archived["controlled"].pop("run.global_batch_positions")
    checker.verify_legacy_comparison(actual, archived)
    archived["costs"]["candidate"]["flops"]["training_per_token"] += 1
    with pytest.raises(ValueError, match="budget-pair"):
        checker.verify_legacy_comparison(actual, archived)


def test_native_checkpoint_cursor_reader_rejects_recycled_training_documents(tmp_path):
    import torch.distributed.checkpoint as dcp
    from moeloop.training import continuation_fingerprint

    checker = importlib.import_module("check_release")
    identity = {"world_size": 1, "debug_seed": 42}
    for epoch in (0, 1):
        directory = tmp_path / f"epoch-{epoch}"
        train_state = {"identity": identity, "fingerprint": continuation_fingerprint(identity),
            "progress": {"completed_updates": 2, "label_tokens": 4}, "rng": {"torch": b"unit-reader-state"}}
        cursor = {"dataset": {"configuration": {"infinite": False},
                             "sources": {"public": {"epoch": epoch, "cursor": 2}}}}
        dcp.save({"core": {"weight": torch.ones(1)}, "optimizer": {"moment": torch.ones(1)},
            "lr_scheduler": {"step": 2}, "train_state": {"rank_0": pickle.dumps(train_state)},
            "dataloader": {"dp_rank_0": pickle.dumps(cursor)}}, checkpoint_id=directory, no_dist=True)
        if epoch == 0:
            assert checker.checkpoint_continuation(directory, identity, expected_step=2, finite_data=True) == 4
        else:
            with pytest.raises(ValueError, match="recycled"):
                checker.checkpoint_continuation(directory, identity, expected_step=2, finite_data=True)


def _identity_checkpoint(directory, native_identity, *, fingerprint_identity=None):
    import torch.distributed.checkpoint as dcp
    from moeloop.training import continuation_fingerprint

    record = {"identity": native_identity,
        "fingerprint": continuation_fingerprint(native_identity if fingerprint_identity is None else fingerprint_identity),
        "progress": {"completed_updates": 2, "label_tokens": 4}, "rng": {"torch": b"unit-reader-state"}}
    dcp.save({"core": {"weight": torch.ones(1)}, "optimizer": {"moment": torch.ones(1)},
        "lr_scheduler": {"step": 2}, "train_state": {"rank_0": pickle.dumps(record)},
        "dataloader": {"dp_rank_0": pickle.dumps({"cursor": 2})}}, checkpoint_id=directory, no_dist=True)


@pytest.mark.parametrize("sampled_visits", [(), (2, 4)])
def test_checkpoint_identity_accepts_native_tuples_against_its_json_sidecar(tmp_path, sampled_visits):
    from dataclasses import asdict
    from moeloop.training import TrainingRecipe

    checker = importlib.import_module("check_release")
    native = {"world_size": 1, "recipe": asdict(TrainingRecipe(num_visits=2, sampled_visits=sampled_visits)),
        "optimizer": {"betas": (.9, .95)}}
    sidecar = json.loads(json.dumps(native))
    _identity_checkpoint(tmp_path / "checkpoint", native)

    assert checker.checkpoint_continuation(tmp_path / "checkpoint", sidecar, expected_step=2) == 4


@pytest.mark.parametrize("changed_depth", [3, 2., True])
def test_checkpoint_identity_still_rejects_changed_values_and_scalar_types(tmp_path, changed_depth):
    checker = importlib.import_module("check_release")
    native = {"world_size": 1, "recipe": {"num_visits": 2, "sampled_visits": ()}}
    sidecar = json.loads(json.dumps(native))
    sidecar["recipe"]["num_visits"] = changed_depth
    # A claimed digest cannot substitute for checking the actual saved identity.
    _identity_checkpoint(tmp_path / "checkpoint", native, fingerprint_identity=sidecar)

    with pytest.raises(AssertionError, match=r"recipe\.num_visits"):
        checker.checkpoint_continuation(tmp_path / "checkpoint", sidecar, expected_step=2)


def _optimizer_checkpoint(path, moment):
    import torch.distributed.checkpoint as dcp
    dcp.save({"optimizer": {"state": {"weight": {"exp_avg": torch.tensor([moment]),
        "exp_avg_sq": torch.tensor([.2]), "step": torch.tensor(8.)}},
        "param_groups": [{"lr": 1e-4, "weight_decay": .1}]}},
        checkpoint_id=path, no_dist=True)


def test_optimizer_continuation_compares_real_moments_and_hyperparameters(tmp_path):
    checker = importlib.import_module("check_release")
    first, second = tmp_path / "one", tmp_path / "two"
    _optimizer_checkpoint(first, .1)
    _optimizer_checkpoint(second, .1)
    report = checker.compare_optimizer(first, second)
    assert report["tensor_count"] == 3 and report["max_absolute_error"] == 0
    changed = tmp_path / "changed"
    _optimizer_checkpoint(changed, .11)
    with pytest.raises(AssertionError, match="optimizer"):
        checker.compare_optimizer(first, changed)


def test_source_snapshot_detects_tampering_and_covers_actual_training_modules(tmp_path):
    runner = importlib.import_module("recipe_qualification")
    checker = importlib.import_module("check_release")
    source = runner.snapshot_source(tmp_path / "source")
    assert "scripts/recipe_anchor.py" in source["files"]
    checker.verify_source(tmp_path / "source", source)
    path = tmp_path / "source" / "src" / "moeloop" / "model.py"
    path.write_text(path.read_text() + "\n# changed after qualification\n")
    with pytest.raises(ValueError, match="source|SHA|hash"):
        checker.verify_source(tmp_path / "source", source)


def test_archived_source_without_git_has_a_verifiable_content_identity(tmp_path, monkeypatch):
    runner = importlib.import_module("recipe_qualification")
    checker = importlib.import_module("check_release")
    archive = tmp_path / "archive"
    for relative in [f"src/moeloop/{name}.py" for name in
                     ("model", "moe", "cache", "training", "titan", "data", "recipes")] + [
                     f"scripts/{name}.py" for name in ("recipe_qualification", "check_release", "qualification")]:
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# archived source fixture\n")
    monkeypatch.setattr(runner, "ROOT", archive)
    source = runner.snapshot_source(tmp_path / "snapshot")
    assert source["commit"] is None
    checker.verify_source(tmp_path / "snapshot", source)


def test_independent_hf_check_writes_actual_heldout_generation_and_cache_results(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    runner = importlib.import_module("recipe_qualification")
    from qualification import prepare_fixture
    from moeloop.model import ModelConfig, RecursiveLM, RegionConfig
    from moeloop.export import export_checkpoint
    from moeloop.data import PreparedManifest
    data = tmp_path / "data"
    prepare_fixture(data)
    config = ModelConfig(vocab_size=64, dim=16, hidden_dim=24, num_heads=2,
                         num_experts=2, num_cells=1, num_visits=2, max_seq_len=32,
                         prefix_cells=1, suffix_cells=1, attention_per_visit=True,
                         core=RegionConfig(num_experts=4))
    native = tmp_path / "model.pt"
    torch.save(RecursiveLM(config).state_dict(), native)
    artifact = export_checkpoint(native, tmp_path / "hf", config, data / "tokenizer")
    output = tmp_path / "hf_check.json"
    result = subprocess.run([sys.executable, "-I", runner.__file__, "--hf-check", str(artifact),
        "--hf-output", str(output), "--device", "cpu"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    observed = json.loads(output.read_text())
    assert observed["repository_package_imported"] is False
    assert observed["generation"]["cached_ids"] == observed["generation"]["uncached_ids"]
    assert observed["cache"]["compared_tokens"] > 0
    assert observed["cache"]["max_absolute_error"] < 1e-5
    heldout = tmp_path / "heldout.json"
    evaluated = subprocess.run([sys.executable, "-m", "moeloop.evaluate", "--model", str(artifact),
        "--manifest", str(data / "manifest.json"), "--output", str(heldout), "--seq-len", "16",
        "--max-batches", "2"], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(Path(runner.__file__).resolve().parents[1] / "src")})
    assert evaluated.returncode == 0, evaluated.stderr
    actual = json.loads(heldout.read_text())
    assert actual["valid_tokens"] > 0
    assert actual["manifest_fingerprint"] == PreparedManifest.load(data).fingerprint
