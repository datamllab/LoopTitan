"""V2 gates protect dynamic execution, inherited identities and real updates."""

import copy
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_v2_cannot_be_satisfied_by_a_passed_label_or_inherited_v1_alone(tmp_path):
    checker = importlib.import_module("check_release")
    (tmp_path / "release.json").write_text(json.dumps({
        "format": "moeloop-recipe-qualification", "version": 1, "stage": "v2",
        "passed": True, "inherited_v1": {"status": "passed"},
    }))
    with pytest.raises(ValueError, match="method|artifact"):
        checker.verify_v2_release(tmp_path)
    result = subprocess.run([sys.executable, checker.__file__, "--stage", "v2", "--evidence", str(tmp_path)],
                            capture_output=True, text=True,
                            env=os.environ | {"PYTHONPATH": str(Path(checker.__file__).resolve().parents[1] / "src")})
    assert result.returncode == 1 and "method" in result.stderr


def test_qualification_cli_exposes_v2_without_importing_a_training_engine():
    runner = importlib.import_module("recipe_qualification")
    result = subprocess.run([sys.executable, runner.__file__, "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "{v1,v2}" in result.stdout and "loom_attention" in result.stdout
    assert "--sharing-grid-only" in result.stdout


@pytest.mark.parametrize("name", ["huginn", "loom", "ouro", "loom_attention"])
def test_new_qualification_factories_retain_the_selected_dynamics(name):
    runner = importlib.import_module("recipe_qualification")
    spec = runner.method_spec(name, vocab_size=64, seq_len=32, steps=8, global_batch_size=4)
    assert spec.name == name
    assert spec.run["successful_updates"] == spec.run["schedule_total_steps"] == 8
    if name == "huginn":
        assert spec.model.initial_state == "random" and spec.model.input_injection == "linear"
        assert spec.training.depth_rule == "poisson_lognormal_micro_v1"
    elif name.startswith("loom"):
        assert spec.model.state_update == "ema" and spec.model.num_visits > 2
        assert spec.model.router_per_visit
        assert spec.model.attention_per_visit == (name == "loom_attention")
    else:
        assert spec.model.exit_distribution and spec.training.objective == "expected_ce"


@pytest.mark.parametrize("method,batch,expected", [("smelt", 16, (1, 8)), ("huginn", 16, (8, 1)),
                                                  ("ouro", 4, (2, 1)), ("grid", 16, (8, 1))])
def test_v2_batch_geometry_reduces_accumulation_without_changing_global_batch(method, batch, expected):
    runner = importlib.import_module("recipe_qualification")
    args = dict(vocab_size=64, seq_len=32, steps=8, global_batch_size=batch)
    spec = runner.sharing_specs(**args)["a1_r1"] if method == "grid" else runner.method_spec(method, **args)
    assert runner.qualification_batch(spec, 2) == expected


def _training(name):
    from moeloop import recipes
    factory = recipes.loom_spec if name == "loom_attention" else getattr(recipes, name + "_spec")
    kwargs = {"attention_per_visit": True} if name == "loom_attention" else {}
    spec = factory(vocab_size=32, dim=16, hidden_dim=24, num_heads=2, num_kv_heads=1,
                   num_experts=2, max_seq_len=32, **kwargs)
    metrics = []
    for step in range(1, 9):
        depths = [spec.training.depth_at(step - 1, micro) for micro in range(4)]
        grads = [min(spec.training.grad_visits or r, r) for r in depths]
        item = {"step": step, "ce": 3., "aux": .01, "depth": depths[-1], "grad_visits": grads[-1],
                "microbatch_depths": depths, "microbatch_grad_visits": grads}
        if name == "ouro":
            item.update(expected_ce=3.1, entropy=1.03,
                        mean_exit_probabilities=[.5, .25, .25], visit_ce=[3.2, 3.1, 3.])
        elif name.startswith("loom"):
            item["differentiated_router_calls_per_micro"] = [spec.model.num_cells * spec.model.num_visits] * 4
        metrics.append(item)
    return {"spec": spec, "metrics": metrics, "identity": {"accumulation": 4}}


@pytest.mark.parametrize("name", ["huginn", "loom", "ouro", "loom_attention"])
def test_dynamic_gate_accepts_the_declared_actual_observations(name):
    checker = importlib.import_module("check_release")
    checker.verify_v2_dynamics(_training(name))


@pytest.mark.parametrize("failure", ["missing", "fixed", "grad"])
def test_huginn_rejects_missing_or_changed_microbatch_sampling(failure):
    checker = importlib.import_module("check_release")
    training = _training("huginn")
    for item in training["metrics"]:
        if failure == "missing":
            item.pop("microbatch_depths")
        elif failure == "fixed":
            item["microbatch_depths"] = [4] * 4
        else:
            item["microbatch_grad_visits"] = list(item["microbatch_depths"])
    with pytest.raises(ValueError, match="microbatch|depth|gradient"):
        checker.verify_v2_dynamics(training)


@pytest.mark.parametrize("failure", ["entropy", "distribution", "readout"])
def test_ouro_rejects_degenerate_or_unbound_expected_loss_metrics(failure):
    checker = importlib.import_module("check_release")
    training = _training("ouro")
    for item in training["metrics"]:
        if failure == "entropy":
            item["entropy"] = 0.
        elif failure == "distribution":
            item["mean_exit_probabilities"] = [.8, .8, .8]
        else:
            item["visit_ce"][-1] = 6.
    with pytest.raises(ValueError, match="entropy|probabilit|terminal|readout"):
        checker.verify_v2_dynamics(training)


def test_hf_uncached_path_is_explicit_and_checks_recomputed_generation():
    from moeloop.recipes import huginn_spec
    checker = importlib.import_module("check_release")
    spec = huginn_spec()
    result = {"cache": {"supported": False, "reason": "fixed-depth deterministic evaluation adaptation"},
              "generation": {"input_ids": [[1, 2]], "uncached_ids": [[1, 2, 3]],
                             "recomputed_ids": [[1, 2, 3]], "readout": "terminal", "num_visits": 4}}
    assert checker.verify_hf_generation(result, spec) is None
    bad = copy.deepcopy(result)
    bad["generation"]["recomputed_ids"][0][-1] = 7
    with pytest.raises(ValueError, match="generation"):
        checker.verify_hf_generation(bad, spec)
    false_claim = replace(spec, inference=spec.inference | {"cache": "logical_visit_slots"})
    with pytest.raises(ValueError, match="cache"):
        checker.verify_hf_generation(result, false_claim)


@pytest.mark.parametrize("zero_gate", [False, True])
def test_expected_loss_update_proof_reads_real_optimizer_moments(tmp_path, zero_gate):
    import torch.distributed.checkpoint as dcp
    checker = importlib.import_module("check_release")
    state = {"optimizer": {"state": {
        "core.exit_gate.weight": {"exp_avg": torch.zeros(1, 3) if zero_gate else torch.ones(1, 3)},
        "core.lm_head.weight": {"exp_avg": torch.ones(8, 3)},
        "core.core.0.attentions.0.wq.weight": {"exp_avg": torch.ones(3, 3)},
    }}}
    checkpoint = tmp_path / "checkpoint"
    dcp.save(state, checkpoint_id=checkpoint)
    if zero_gate:
        with pytest.raises(ValueError, match="gate"):
            checker.verify_expected_updates(checkpoint)
    else:
        report = checker.verify_expected_updates(checkpoint)
        assert set(report) == {"gate", "head", "backbone"}
        assert all(item["nonzero_moment_elements"] > 0 for item in report.values())


def test_sharing_four_grid_has_only_attention_and_router_ownership_changes():
    runner = importlib.import_module("recipe_qualification")
    checker = importlib.import_module("check_release")
    specs = runner.sharing_specs(vocab_size=64, seq_len=32, steps=8, global_batch_size=4)
    checker.verify_sharing_specs(specs)
    broken = dict(specs)
    selected = broken["a1_r1"]
    broken["a1_r1"] = replace(selected, model=replace(selected.model, hidden_dim=selected.model.hidden_dim * 2))
    with pytest.raises(ValueError, match="controlled|geometry"):
        checker.verify_sharing_specs(broken)


def _inherited_identity_fixture(tmp_path):
    """Small identity-chain fixture, not a claim of native GPU qualification."""
    checker = importlib.import_module("check_release")
    import torch.distributed.checkpoint as dcp

    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    directory = tmp_path / "accepted-v1"
    source_files = {f"src/moeloop/{name}.py": f"# unit source {name}\n"
                    for name in ("model", "moe", "cache", "training", "titan", "data", "recipes")}
    files = {}
    arms = {}
    exported = {}
    for role in ("reference", "candidate"):
        run = tmp_path / role
        for relative, content in source_files.items():
            target = run / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            files[relative] = checker.digest(target)
        source = {"commit": "a" * 40, "dirty": False, "files": files.copy()}
        put(run / "request.json", {"source": source})
        checkpoint = run / "checkpoint/step-19537"
        dcp.save({"core": {"unit": torch.ones(2)}}, checkpoint_id=checkpoint)
        provenance = put(run / "hf/export_provenance.json", {
            "checkpoint": {"metadata_sha256": checker.digest(checkpoint / ".metadata")}})
        exported[role] = checker.digest(provenance)
        put(run / "outcome.json", {"status": "completed", "completed_updates": 19537,
                                  "global_progress": {"label_tokens": 20486029312},
                                  "full_checkpoint": str(checkpoint)})
        arms[role] = str(run)
    source_path = put(tmp_path / "original-source.json", source)
    original_checker = tmp_path / "original-checker.py"
    original_checker.write_text("# immutable original checker fixture\n")
    fingerprint = "b" * 64
    anchor = put(directory / "anchor.json", {"manifest_fingerprint": fingerprint, "arms": arms,
                 "verification_checker": {"path": str(original_checker), "sha256": checker.digest(original_checker)}})
    put(directory / "release.json", {"stage": "v1", "scientific_anchor": str(anchor)})
    put(directory / "release-check.json", {"status": "passed", "stage": "v1", "scope": "v1_full_acceptance",
                                           "scientific_anchor": {"manifest_fingerprint": fingerprint}})
    put(directory / "finalization.json", {"status": "verified", "scope": "v1_full_acceptance"})
    accepted = put(tmp_path / "publication.json", {
        "training_source": source, "data": {"manifest_fingerprint": fingerprint},
        "runtime": {"native": {"source_identity_sha256": checker.digest(source_path)}},
        "verification": {"checker_sha256": checker.digest(original_checker), "retained_raw_reports": {
            name: checker.digest(directory / name) for name in ("release-check.json", "finalization.json")}},
        "arms": {role: {"completed_updates": 19537, "actual_valid_labels": 20486029312,
                        "hf": {"export_provenance_sha256": exported[role]}} for role in arms},
    })
    entry = {"directory": str(directory), "release_sha256": checker.digest(directory / "release.json"),
             "release_check_sha256": checker.digest(directory / "release-check.json"),
             "source_p1": str(source_path), "source_p1_sha256": checker.digest(source_path)}
    return entry, accepted, original_checker


def test_inheritance_verifies_accepted_bytes_without_loading_the_old_model_schema(tmp_path):
    checker = importlib.import_module("check_release")
    entry, accepted, _ = _inherited_identity_fixture(tmp_path)
    result = checker.verify_inherited_v1(entry, accepted_path=accepted)
    assert result["status"] == "inherited" and result["training_source"] == "a" * 40


@pytest.mark.parametrize("changed", ["release", "report", "source", "checker"])
def test_inherited_v1_cannot_relabel_modified_original_artifacts(tmp_path, changed):
    checker = importlib.import_module("check_release")
    entry, accepted, original_checker = _inherited_identity_fixture(tmp_path)
    paths = {"release": Path(entry["directory"]) / "release.json",
             "report": Path(entry["directory"]) / "release-check.json",
             "source": Path(entry["source_p1"]), "checker": original_checker}
    paths[changed].write_text(paths[changed].read_text() + "\n")
    with pytest.raises(ValueError, match="SHA|source|checker"):
        checker.verify_inherited_v1(entry, accepted_path=accepted)
