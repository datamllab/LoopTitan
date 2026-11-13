"""Synthetic artifacts exercise assembly contracts, never qualify a training run."""
import json
from pathlib import Path

import pytest

from recipe_qualification import digest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def run_fixture(path, spec, fingerprint="b" * 64):
    """Deliberately incomplete native fixture: useful for arithmetic, never acceptance."""
    put(path / "spec.json", spec)
    put(path / "request.json", {"source": {"commit": "a" * 40, "dirty": False},
        "manifest_fingerprint": fingerprint, "world_size": 2,
        "seq_len": spec["run"]["seq_len"], "spec_sha256": digest(path / "spec.json"),
        "global_batch_size": 16, "runtime": {"gpu_name": "synthetic"}})
    for name in ("continuation_identity.json", "outcome.json", "resolved_config.json", "model_config.json"):
        put(path / name, {"synthetic_fixture": True})
    for name, rows in {
        "metrics.jsonl": [{"step": s, "global_progress": {"label_tokens": 90 * s, "padded_positions": 100 * s},
            "step_elapsed_seconds": 2., "peak_active_gib": 3., "peak_reserved_gib": 4.} for s in range(1, 31)],
        "validation.jsonl": [{"step": 10, "ce": 4.}, {"step": 30, "ce": 3.9}],
    }.items():
        (path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    log = path / "native.log"
    log.write_text("synthetic native-process accounting fixture\n")
    put(path / "process-0.json", {"argv": ["python", "-m", "moeloop.train"],
        "log": str(log), "log_sha256": digest(log), "returncode": 0,
        "started_unix": 100, "finished_unix": 110, "gpu_count": 2})
    return path


def test_cli_requires_concrete_inputs_and_fresh_output(tmp_path):
    from recipe_release import main, new_output
    with pytest.raises(SystemExit) as error:
        main(["--output", str(tmp_path / "release")])
    assert error.value.code == 2
    assert not (tmp_path / "release").exists()
    output = new_output(tmp_path / "release")
    with pytest.raises(FileExistsError):
        new_output(output)


def test_source_v2_retains_original_checker_and_frozen_commit(tmp_path):
    from recipe_release import inherited_v2
    old_checker = tmp_path / "accepted-checker.py"
    old_checker.write_text("# synthetic accepted checker\n")
    source = put(tmp_path / "source.json", {"commit": "a" * 40, "dirty": False,
        "files": {"src/moeloop/model.py": "c" * 64},
        "checker": {"path": str(old_checker), "sha256": digest(old_checker)}})
    v2 = tmp_path / "v2"
    put(v2 / "release.json", {"stage": "v2"})
    put(v2 / "release-check.json", {"stage": "v2", "status": "passed", "scope": "v2_full_acceptance"})
    result = inherited_v2(v2, source)
    assert result["source_v2_sha256"] == digest(source)
    assert result["release_check_sha256"] == digest(v2 / "release-check.json")
    old_checker.write_text("# changed publication checker\n")
    with pytest.raises(ValueError, match="checker"):
        inherited_v2(v2, source)


def test_missing_raw_input_is_not_replaced_with_a_positive_marker(tmp_path):
    from recipe_release import load_run
    put(tmp_path / "request.json", {"passed": True})
    with pytest.raises(FileNotFoundError, match="spec.json"):
        load_run(tmp_path, "b" * 64)


def test_anchor_curves_use_frozen_pair_costs_and_real_report_helper(tmp_path):
    pytest.importorskip("torch")
    from moeloop.comparison import smelt_p2_pair
    from recipe_release import assemble_anchor
    pair = smelt_p2_pair(attention_pairs_per_sequence=4_180_216.9229844734)
    comparison = pair["comparison"]
    comparison["data"] = comparison["evaluation"] = {"manifest_fingerprint": "b" * 64}
    pair_path = put(tmp_path / "pair.json", pair)
    arms = {role: run_fixture(tmp_path / role, spec) for role, spec in comparison["arms"].items()}
    heldout = {role: {split: str(put(tmp_path / role / f"{split}.json", {"split": split}))
        for split in ("validation", "test")} for role in arms}
    output = tmp_path / "release"
    output.mkdir()
    anchor = assemble_anchor(output, pair_path, arms, heldout, tmp_path / "manifest.json", "b" * 64)
    assert anchor["format"] == "moeloop-scientific-anchor" and anchor["anchor"] == "P2"
    assert anchor["comparison"] == str(pair_path)
    rows = [json.loads(line) for line in Path(anchor["curves"]["reference"]).read_text().splitlines()]
    assert rows[0]["global_labels"] == 900 and rows[0]["padded_positions"] == 1000
    assert rows[0]["cumulative_model_flops_estimate"] == comparison["costs"]["reference"]["flops"]["training_per_token"] * 1000
    assert pair_path.read_text() == json.dumps(pair) + "\n"
    # The actual gate must reject this incomplete synthetic run; no validator mocks.
    from check_release import verify_scientific_anchor
    anchor_path = put(output / "p2.json", anchor)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_scientific_anchor(anchor_path, anchor_name="P2")
    changed = json.loads((arms["candidate"] / "spec.json").read_text())
    changed["run"]["seed"] = 7
    put(arms["candidate"] / "spec.json", changed)
    with pytest.raises(ValueError, match="frozen|spec"):
        assemble_anchor(output, pair_path, arms, heldout, tmp_path / "manifest.json", "b" * 64)


def test_c_comparison_uses_actual_specs_and_costs(tmp_path):
    pytest.importorskip("torch")
    from moeloop.recipes import sharing_panel_specs
    from moeloop.costs import estimate_cost
    from recipe_release import sharing_comparison
    arms = {name: run_fixture(tmp_path / name, spec.to_dict()) for name, spec in sharing_panel_specs().items()}
    result = sharing_comparison(arms, "b" * 64)
    assert result["format"] == "moeloop-attention-router-grid"
    for name, spec in sharing_panel_specs().items():
        assert result["arms"][name] == json.loads(json.dumps({"spec": spec.to_dict(), "cost": estimate_cost(spec, seq_len=2048)}))


def test_resumed_performance_window_measures_10_through_29(tmp_path):
    from recipe_release import performance_spec
    from recipe_report import performance_report
    arms = {role: tmp_path / role for role in ("reference", "candidate")}
    sharing = {"a1_r0": tmp_path / "sharing"}
    for directory in list(arms.values()) + list(sharing.values()):
        put(directory / "request.json", {"world_size": 2, "seq_len": 2048, "global_batch_size": 16,
            "runtime": {"gpu_name": "synthetic"}, "source": {"commit": "a" * 40}})
        put(directory / "spec.json", {"run": {}, "model": {"attention_per_visit": True, "router_per_visit": False}})
        (directory / "metrics.jsonl").write_text("".join(json.dumps({"step": s,
            "global_progress": {"label_tokens": s * 100}, "step_elapsed_seconds": 2.,
            "peak_active_gib": 3., "peak_reserved_gib": 4.}) + "\n" for s in range(1, 31)))
    performance = performance_spec(arms, sharing)
    assert performance["windows"]["sharing"]["run"] == str(sharing["a1_r0"].resolve())
    path = put(tmp_path / "performance.json", performance)
    expected = {"p2_reference": [arms["reference"]], "p2_candidate": [arms["candidate"]], "sharing": list(sharing.values())}
    report = performance_report(path, expected)
    assert report["p2_reference"]["warmup_steps"] == list(range(5, 10))
    assert report["p2_reference"]["measured_steps"] == list(range(10, 30))
    assert report["sharing"]["measured_steps"] == list(range(6, 26))


def test_complete_assembly_writes_real_gate_failure_without_validator_mocks(tmp_path):
    pytest.importorskip("torch")
    from moeloop.comparison import smelt_p2_pair
    from moeloop.recipes import sharing_panel_specs
    from recipe_release import main
    pair = smelt_p2_pair(attention_pairs_per_sequence=4_180_216.9229844734)
    pair["comparison"]["data"] = pair["comparison"]["evaluation"] = {"manifest_fingerprint": "b" * 64}
    pair_path = put(tmp_path / "pair.json", pair)
    arms = {role: run_fixture(tmp_path / role, spec) for role, spec in pair["comparison"]["arms"].items()}
    heldout = {role: {split: str(put(tmp_path / role / f"{split}.json", {"split": split}))
        for split in ("validation", "test")} for role in arms}
    p1 = put(tmp_path / "p1.json", {"comparison": str(pair_path), "arms": {key: str(value) for key, value in arms.items()},
        "heldout": heldout})
    v1 = tmp_path / "v1"
    put(v1 / "release.json", {"scientific_anchor": str(p1)})
    manifest = put(tmp_path / "manifest.json", {"synthetic_fixture": True})
    v2 = tmp_path / "v2"
    put(v2 / "release.json", {"stage": "v2", "manifest": str(manifest), "manifest_fingerprint": "b" * 64,
        "inherited_v1": {"directory": str(v1)}})
    put(v2 / "release-check.json", {"stage": "v2", "status": "passed", "scope": "v2_full_acceptance"})
    checker = tmp_path / "original-checker.py"
    checker.write_text("# synthetic fixture, not accepted publication evidence\n")
    source = put(tmp_path / "source-v2.json", {"commit": "a" * 40, "dirty": False,
        "files": {"src/moeloop/model.py": "c" * 64}, "checker": {"path": str(checker), "sha256": digest(checker)}})
    receipt = put(tmp_path / "incomplete-delivery.json", {"synthetic_fixture": True})
    output = tmp_path / "release"
    argv = ["--output", str(output), "--v2", str(v2), "--source-v2", str(source), "--p2-pair", str(pair_path),
        "--lm-eval", str(receipt), "--package", str(receipt)]
    for role in arms:
        argv += [f"--p2-{role}", str(arms[role])]
        for split in heldout[role]:
            argv += [f"--p2-{role}-{split}", heldout[role][split]]
    for name, spec in sharing_panel_specs().items():
        argv += ["--sharing-" + name.replace("_", "-"), str(run_fixture(tmp_path / name, spec.to_dict()))]
    assert main(argv) == 1
    # Producer/report contracts agree, but synthetic V2 cannot pass the actual gate.
    assert (output / "analysis.json").is_file()
    check = json.loads((output / "release-check.json").read_text())
    assert check["status"] == "failed" and "accepted gate SHA" in check["error"]
    index = json.loads((output / "release.json").read_text())
    assert set(index) >= {"p2", "sharing_grid", "performance", "analysis", "lm_eval", "package"}
    assert index["inherited_v2"]["directory"] == str(v2)
    assert not (output / "p1.json").exists()
    before = (output / "release-check.json").read_bytes()
    assert main(argv) == 1  # Fresh-only output never overwrites a prior failed identity.
    assert (output / "release-check.json").read_bytes() == before


def test_assembler_rejects_all_shared_profile_but_retains_ar_as_partial(tmp_path):
    from recipe_release import performance_spec
    arms = {role: tmp_path / role for role in ("reference", "candidate")}
    sharing = {}
    for name, attention, router in (("a0_r0", False, False), ("a1_r0", True, False), ("a1_r1", True, True)):
        sharing[name] = tmp_path / name
        put(sharing[name] / "spec.json", {"model": {"attention_per_visit": attention, "router_per_visit": router}})
    assert performance_spec(arms, sharing)["windows"]["sharing"]["run"] == str(sharing["a1_r0"].resolve())
    with pytest.raises(ValueError, match="partial|per.visit"):
        performance_spec(arms, sharing, sharing_arm="a0_r0")
    assert performance_spec(arms, sharing, sharing_arm="a1_r1")["windows"]["sharing"]["run"] == str(sharing["a1_r1"].resolve())
