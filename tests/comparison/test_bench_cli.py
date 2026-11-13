"""scripts/bench.py on frozen specs: parsing, fixed settings, the LR rule, validate and leaderboard.

Track P2 has one calibrated LR per optimizer, so ``run`` is its whole submission; ``sweep`` and the LR
rule are tested on Track M, which keeps its three-point grid.

No GPU: ``bench_cli.launch`` is replaced by a fake that writes the files a finished
``recipe_experiments.py run --evaluate`` leaves behind.
"""
import importlib
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from moeloop import bench

bench_cli = importlib.import_module("bench")
recipe_experiments = importlib.import_module("recipe_experiments")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def track():
    return bench.load_tracks()["P2"]


def test_run_and_sweep_parse_a_p2_spec():
    spec = "recipes/phase1/m0-adamw.json"
    args = bench_cli.parser().parse_args(["sweep", "--spec", spec, "--gpus", "8", "--output", "/out"])
    assert (args.command, args.spec, args.track, args.gpus, args.output) == ("sweep", Path(spec), "P2", 8, Path("/out"))
    assert args.manifest.name == "manifest.json"
    args = bench_cli.parser().parse_args(["run", "--spec", spec, "--lr", "1.2e-3", "--gpus", "8", "--output", "/out"])
    assert (args.command, args.lr, args.track) == ("run", 1.2e-3, "P2")


def test_fixed_settings_are_written_where_the_spec_has_none(track):
    spec = json.loads((ROOT / "recipes/phase1/m3-muon.json").read_text())
    assert bench_cli.track_spec(spec, track) == spec  # Already complete: unchanged.
    del spec["optimizer"]["max_norm"], spec["optimizer"]["muon"]["nesterov"]
    spec["optimizer"]["eps"] = 1e-6  # A spec's own value stays; the checker reports it.
    filled = bench_cli.track_spec(spec, track)["optimizer"]
    assert (filled["max_norm"], filled["muon"]["nesterov"], filled["eps"]) == (1.0, True, 1e-6)
    assert filled["muon"]["lr"] == spec["optimizer"]["muon"]["lr"]


def test_selection_rule_on_fake_results():
    assert bench_cli.select({1.5e-4: 1.02, 3e-4: 1.01, 6e-4: 1.01}) == 3e-4  # Ties go to the smaller LR.
    assert bench_cli.next_lr({1.5e-4: 1.03, 3e-4: 1.01, 6e-4: 1.02}) is None
    assert bench_cli.next_lr({1.5e-4: 1.03, 3e-4: 1.02, 6e-4: 1.01}) == 1.2e-3
    assert bench_cli.next_lr({1.5e-4: 1.01, 3e-4: 1.02, 6e-4: 1.03}) == 7.5e-5
    assert bench_cli.next_lr({7.5e-5: 1.00, 1.5e-4: 1.01, 3e-4: 1.02, 6e-4: 1.03}) == 3.75e-5


def bpb(lr):
    """A U-shaped validation curve with its minimum at 1.2e-3, beyond the AdamW grid."""
    return 1.0 + 0.01 * math.log2(lr / 1.2e-3) ** 2


@pytest.fixture
def launched(monkeypatch, tmp_path, track):
    """Fake launcher: writes one finished, evaluated experiment output per command; records each LR.

    ``state["bpb"]`` is the validation curve; tests may replace it.
    """
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tokenizer": {"files": {
        "tokenizer/tokenizer.json": track.tokenizer["tokenizer_json_sha256"]}}}))
    lrs = []
    state = {"manifest": manifest, "lrs": lrs, "bpb": bpb, "track": track}

    def launch(command):
        assert command[2:4] == ["run", "--spec"] and command[-1] == "--evaluate"
        values = dict(zip(command[3:-1:2], command[4:-1:2]))
        lr, output = float(values["--lr"]), Path(values["--output"])
        assert (values["--gpus"], values["--manifest"]) == ("8", str(manifest))
        lrs.append(lr)
        spec = recipe_experiments.load_spec(values["--spec"], lr)
        steps = spec.run["successful_updates"]
        (output / f"checkpoint/step-{steps}").mkdir(parents=True)
        (output / f"checkpoint/step-{steps}/.metadata").write_bytes(b"dcp")
        files = {"spec.json": spec.to_dict(),
                 "request.json": {"world_size": 8, "local_batch_size": 16, "gradient_accumulation_steps": 2,
                                  "manifest": str(manifest), "manifest_fingerprint": track.data["manifest_fingerprint"],
                                  "source": {"commit": "a" * 40, "dirty": False}},
                 "outcome.json": {"status": "completed", "completed_updates": steps,
                                  "full_checkpoint": f"checkpoint/step-{steps}"},
                 "process-1.json": {"log_sha256": "3" * 64}}
        for split, labels in (("validation", "selection_label_tokens"), ("test", "ranking_label_tokens")):
            files[f"eval-{split}.json"] = {
                "split": split, "seq_len": track.seq_len, "complete_split": True, "valid_tokens": track.evaluation[labels],
                "bpb": state["bpb"](lr) + (0.005 if split == "test" else 0), "byte_table_sha256": track.evaluation["byte_table_sha256"],
                "tokenizer": {key: track.tokenizer[key] for key in ("id", "revision", "vocab_size")}}
        for name, value in files.items():
            (output / name).write_text(json.dumps(value))

    monkeypatch.setattr(bench_cli, "launch", launch)
    monkeypatch.delenv("RUN_DIR", raising=False)
    return state


def sweep(tmp_path, manifest, arm):
    """The Track M LR sweep of a frozen P1 spec."""
    output = tmp_path / arm
    bench_cli.main(["sweep", "--track", "M", "--spec", str(ROOT / f"recipes/p1/{arm}.json"), "--gpus", "8",
                    "--output", str(output), "--manifest", str(manifest)])
    return output / "submission.json"


def run_p2(tmp_path, manifest, arm, lr):
    """One run of a frozen Phase 1 spec at the Track P2 LR."""
    output = tmp_path / f"p2-{arm}"
    bench_cli.main(["run", "--spec", str(ROOT / f"recipes/phase1/{arm}.json"), "--lr", str(lr), "--gpus", "8",
                    "--output", str(output), "--manifest", str(manifest)])
    return output / "submission.json"


@pytest.mark.parametrize("arm", ["m0-adamw", "m3-adamw"])
def test_sweep_extends_the_edge_and_submits_an_eligible_track_m_entry(launched, tmp_path, monkeypatch, capsys, arm):
    manifest, lrs = launched["manifest"], launched["lrs"]
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    path = sweep(tmp_path, manifest, arm)
    # Grid, then 1.2e-3 beyond the upper edge, then 2.4e-3 because 1.2e-3 became the edge.
    assert lrs == [1.5e-4, 3e-4, 6e-4, 1.2e-3, 2.4e-3]
    value = json.loads(path.read_text())
    assert [trial["lr"] for trial in value["lr_trials"]] == lrs
    assert value["recipe"]["optimizer"]["base_lr"] == 1.2e-3 and value["seeds"][0]["test_bpb"] == pytest.approx(1.005)
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "eligible", report["reasons"]
    assert all(abs(entry["relative_difference"]) <= 0.05 for entry in report["budgets"].values())
    assert bench_cli.main(["validate", str(path)]) == 0
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert (metrics["lr"], metrics["lr_trials"]) == (1.2e-3, 5) and len(metrics) == 8
    config = json.loads((run_dir / "config.json").read_text())
    assert config["optimizer"]["base_lr"] == 1.2e-3 and config["launch"]["world_size"] == 8
    sweep(tmp_path, manifest, arm)  # A rerun reads the finished trials and launches nothing.
    assert len(lrs) == 5


def test_run_at_the_p2_lr_is_an_eligible_entry_without_trials(launched, tmp_path, capsys):
    for arm, lr in (("m3-adamw", 1.2e-3), ("m3-muon", 1.7e-3)):
        path = run_p2(tmp_path, launched["manifest"], arm, lr)
        value = json.loads(path.read_text())
        assert value["lr_trials"] == [] and value["seeds"][0]["final_update"] == launched["track"].horizon_updates
        assert value["recipe"]["optimizer"]["base_lr"] == lr
        report = json.loads(capsys.readouterr().out)
        assert report["verdict"] == "eligible", report["reasons"]
    assert launched["lrs"] == [1.2e-3, 1.7e-3]


def test_run_at_another_lr_is_not_the_p2_lr(launched, tmp_path, capsys):
    run_p2(tmp_path, launched["manifest"], "m3-adamw", 6e-4)
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "protocol-mismatch"
    assert any("is not the track LR" in line for line in report["reasons"]["protocol-mismatch"])


def test_run_launches_one_lr_and_one_lr_is_not_the_grid(launched, tmp_path, capsys):
    manifest, lrs = launched["manifest"], launched["lrs"]
    bench_cli.main(["run", "--track", "M", "--spec", str(ROOT / "recipes/p1/m3-muon.json"), "--lr", "1.2e-3", "--gpus", "8",
                    "--output", str(tmp_path / "one"), "--manifest", str(manifest)])
    assert lrs == [1.2e-3]
    value = json.loads((tmp_path / "one/submission.json").read_text())
    assert value["recipe"]["optimizer"]["muon"]["lr"] == value["recipe"]["optimizer"]["base_lr"] == 1.2e-3
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "protocol-mismatch"
    assert any("lack base grid points" in line for line in report["reasons"]["protocol-mismatch"])


@pytest.mark.parametrize("curve, expected, selected", [
    # Minimum at 3.75e-5, below the grid: two halvings, then 1.875e-5 to make it interior.
    (lambda lr: 1.0 + 0.01 * math.log2(lr / 3.75e-5) ** 2, [1.5e-4, 3e-4, 6e-4, 7.5e-5, 3.75e-5, 1.875e-5], 3.75e-5),
    # 3e-4 and the upper edge 6e-4 tie: the smaller LR wins, it is interior, so nothing is added.
    (lambda lr: {1.5e-4: 1.02, 3e-4: 1.01, 6e-4: 1.01}[lr], [1.5e-4, 3e-4, 6e-4], 3e-4),
    # The lower edge 1.5e-4 ties with 3e-4 and wins; one halving makes it interior.
    (lambda lr: {7.5e-5: 1.03, 1.5e-4: 1.01, 3e-4: 1.01, 6e-4: 1.02}[lr], [1.5e-4, 3e-4, 6e-4, 7.5e-5], 1.5e-4),
])
def test_sweep_downward_extension_and_ties_reach_the_final_selection(launched, tmp_path, capsys, curve, expected, selected):
    launched["bpb"] = curve
    value = json.loads(sweep(tmp_path, launched["manifest"], "m0-adamw").read_text())
    assert launched["lrs"] == expected
    assert value["recipe"]["optimizer"]["base_lr"] == selected
    assert [trial["lr"] for trial in value["lr_trials"]] == sorted(expected)
    assert json.loads(capsys.readouterr().out)["verdict"] == "eligible"


def test_run_on_track_l_is_an_eligible_entry_without_trials(launched, tmp_path, capsys):
    track = bench.load_tracks()["L"]
    assert track.lr_grid["AdamW"] == (3e-4,)
    bench_cli.main(["run", "--track", "L", "--spec", str(ROOT / "recipes/p1_moe.json"), "--lr", "3e-4", "--gpus", "8",
                    "--output", str(tmp_path / "one"), "--manifest", str(launched["manifest"])])
    assert launched["lrs"] == [3e-4]
    value = json.loads((tmp_path / "one/submission.json").read_text())
    assert value["lr_trials"] == [] and value["seeds"][0]["final_update"] == track.horizon_updates
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "eligible", report["reasons"]


def test_validate_a_budget_mismatched_spec(launched, tmp_path, capsys):
    manifest = launched["manifest"]
    path = run_p2(tmp_path, manifest, "m0-2e-adamw", 1.2e-3)  # Twice the expert parameters, outside the budget by design.
    capsys.readouterr()
    assert bench_cli.main(["validate", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "budget-mismatch"
    assert any(line.startswith("unique_non_embedding_parameters differs") for line in report["reasons"]["budget-mismatch"])


def test_leaderboard_of_two_submissions(launched, tmp_path):
    manifest = launched["manifest"]
    paths = [run_p2(tmp_path, manifest, arm, 1.2e-3) for arm in ("m0-adamw", "m3-adamw")]
    value = json.loads(paths[1].read_text())
    paths[1].write_text(json.dumps(value | {"seeds": [value["seeds"][0] | {"test_bpb": 0.99}], "gpu_hours": 7.5}))
    assert bench_cli.main(["leaderboard", *map(str, paths), "--output", str(tmp_path / "board")]) == 0
    board = json.loads((tmp_path / "board/leaderboard.json").read_text())
    assert [(row["rank"], row["submission"], row["n_seeds"]) for row in board["rows"]] == [
        (1, "phase1-m3-adamw", 1), (2, "phase1-m0-adamw", 1)]
    assert board["excluded"] == []
    table = (tmp_path / "board/leaderboard.md").read_text().splitlines()
    assert table[0].startswith("| Track | Rank | Submission | Test BPB |")
    assert table[2] == "| P2 | 1 | phase1-m3-adamw | 0.9900 | 1 |  | 1.000 | 1 | 7.50 |  |"
    assert table[3].startswith("| P2 | 2 | phase1-m0-adamw | 1.0050 | 1 |  |")
