"""scripts/bench.py on frozen specs: parsing, fixed settings, the LR rule, the submission folder, validate and leaderboard.

Track P2 has one calibrated LR per optimizer, so ``run`` is its whole submission; ``sweep`` and the LR
rule are tested on Track M, which keeps its three-point grid.

No GPU: ``bench_cli.launch`` is replaced by a fake that writes the files a finished
``recipe_experiments.py run --evaluate`` leaves behind.
"""
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess

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

    ``state["bpb"]`` is the validation curve; tests may replace it. Each trial records the trainer's
    package versions, so these tests need TorchTitan installed; the test checkout may have local
    edits, so its tracked-file status is reported clean.
    """
    pytest.importorskip("torchtitan")
    monkeypatch.setattr(bench_cli, "tracked_changes", lambda: False)
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
                                  "source": {"commit": "a" * 40, "dirty": False},
                                  "runtime": {"python": "3.12", "torch": torch.__version__, "cuda": None}},
                 "outcome.json": {"status": "completed", "completed_updates": steps,
                                  "full_checkpoint": f"checkpoint/step-{steps}"}}
        log = f"step=1 CE=11.9 lr={lr}\n".encode()
        (output / "process-1.log").write_bytes(log)
        files["process-1.json"] = {"log_sha256": hashlib.sha256(log).hexdigest()}
        for split, labels in (("validation", "selection_label_tokens"), ("test", "ranking_label_tokens")):
            files[f"eval-{split}.json"] = {
                "split": split, "seq_len": track.seq_len, "complete_split": True, "valid_tokens": track.evaluation[labels],
                "num_visits": spec.inference.get("num_visits", spec.model.num_visits),
                "manifest_fingerprint": track.data["manifest_fingerprint"],
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
    return output / "submission"


def run_p2(tmp_path, manifest, arm, lr):
    """One run of a frozen Phase 1 spec at the Track P2 LR."""
    output = tmp_path / f"p2-{arm}"
    bench_cli.main(["run", "--spec", str(ROOT / f"recipes/phase1/{arm}.json"), "--lr", str(lr), "--gpus", "8",
                    "--output", str(output), "--manifest", str(manifest)])
    return output / "submission"


@pytest.mark.parametrize("arm", ["m0-adamw", "m3-adamw"])
def test_sweep_extends_the_edge_and_submits_an_eligible_track_m_entry(launched, tmp_path, monkeypatch, capsys, arm):
    manifest, lrs = launched["manifest"], launched["lrs"]
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    folder = sweep(tmp_path, manifest, arm)
    # Grid, then 1.2e-3 beyond the upper edge, then 2.4e-3 because 1.2e-3 became the edge.
    assert lrs == [1.5e-4, 3e-4, 6e-4, 1.2e-3, 2.4e-3]
    value = json.loads((folder / "submission.json").read_text())
    assert [trial["lr"] for trial in value["lr_trials"]] == lrs
    # The selected trial is the top of the folder; the others have their own run folders.
    assert [trial["run"] for trial in value["lr_trials"]] == ["lr-0.00015", "lr-0.0003", "lr-0.0006", ".", "lr-0.0024"]
    assert value["recipe"]["optimizer"]["base_lr"] == 1.2e-3 and value["seeds"][0]["test_bpb"] == pytest.approx(1.005)
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "eligible", report["reasons"]
    assert all(abs(entry["relative_difference"]) <= 0.05 for entry in report["budgets"].values())
    assert bench_cli.main(["validate", "--dir", str(folder)]) == 0
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert (metrics["lr"], metrics["lr_trials"], metrics["completed_updates"]) == (1.2e-3, 5, 4769) and len(metrics) == 9
    config = json.loads((run_dir / "config.json").read_text())
    assert config == json.loads((folder / "config.json").read_text())
    assert config["optimizer"]["base_lr"] == 1.2e-3 and config["launch"]["world_size"] == 8
    sweep(tmp_path, manifest, arm)  # A rerun reads the finished trials and launches nothing.
    assert len(lrs) == 5


def test_run_at_the_p2_lr_is_an_eligible_entry_without_trials(launched, tmp_path, capsys):
    for arm, lr in (("m3-adamw", 1.2e-3), ("m3-muon", 1.7e-3)):
        value = json.loads((run_p2(tmp_path, launched["manifest"], arm, lr) / "submission.json").read_text())
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
    value = json.loads((tmp_path / "one/submission/submission.json").read_text())
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
    value = json.loads((sweep(tmp_path, launched["manifest"], "m0-adamw") / "submission.json").read_text())
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
    value = json.loads((tmp_path / "one/submission/submission.json").read_text())
    assert value["lr_trials"] == [] and value["seeds"][0]["final_update"] == track.horizon_updates
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "eligible", report["reasons"]


def test_validate_a_budget_mismatched_spec(launched, tmp_path, capsys):
    manifest = launched["manifest"]
    folder = run_p2(tmp_path, manifest, "m0-2e-adamw", 1.2e-3)  # Twice the expert parameters, outside the budget by design.
    capsys.readouterr()
    assert bench_cli.main(["validate", "--dir", str(folder)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "budget-mismatch"
    assert any(line.startswith("unique_non_embedding_parameters differs") for line in report["reasons"]["budget-mismatch"])


def test_leaderboard_of_two_submissions(launched, tmp_path):
    manifest = launched["manifest"]
    paths = [run_p2(tmp_path, manifest, "m0-adamw", 1.2e-3)]
    launched["bpb"] = lambda lr: 0.985  # The second arm's evaluation outputs carry test BPB 0.99.
    paths.append(run_p2(tmp_path, manifest, "m3-adamw", 1.2e-3))
    value = json.loads((paths[1] / "submission.json").read_text())
    (paths[1] / "submission.json").write_text(json.dumps(value | {"gpu_hours": 7.5}))  # Reported, not bound to a file.
    assert bench_cli.main(["leaderboard", *map(str, paths), "--output", str(tmp_path / "board")]) == 0
    board = json.loads((tmp_path / "board/leaderboard.json").read_text())
    assert [(row["rank"], row["submission"], row["n_seeds"]) for row in board["rows"]] == [
        (1, "phase1-m3-adamw", 1), (2, "phase1-m0-adamw", 1)]
    assert board["excluded"] == []
    table = (tmp_path / "board/leaderboard.md").read_text().splitlines()
    assert table[0].startswith("| Track | Rank | Submission | Test BPB |")
    assert table[2] == "| P2 | 1 | phase1-m3-adamw | 0.9900 | 1 |  | 1.000 | 1 | 7.50 |  |"
    assert table[3].startswith("| P2 | 2 | phase1-m0-adamw | 1.0050 | 1 |  |")


def validate_dir(folder, capsys):
    capsys.readouterr()
    code = bench_cli.main(["validate", "--dir", str(folder)])
    report = json.loads(capsys.readouterr().out)
    assert code == (0 if report["verdict"] == "eligible" else 1)
    return report


def rehash(folder, name):
    """Declare a changed file's new sha256, as a forger would."""
    value = json.loads((folder / "submission.json").read_text())
    value["files"][name] = bench.sha256(folder / name)
    (folder / "submission.json").write_text(json.dumps(value))


def edit(path, change):
    path.write_text(json.dumps(change(json.loads(path.read_text()))))


def test_made_up_submission_folder_is_rejected_and_not_ranked(tmp_path, track, capsys):
    """Placeholder hashes, a configuration of {"x": 1} and an unrecorded commit around a frozen recipe."""
    folder = tmp_path / "made-up"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"x": 1}))
    (folder / "submission.json").write_text(json.dumps({
        "name": "made-up", "track": "P2", "recipe": json.loads((ROOT / "recipes/phase1/m0-adamw.json").read_text()),
        "files": {"config.json": bench.sha256(folder / "config.json"), "train.log": "1" * 64},
        "source_commit": "0" * 40, "manifest_fingerprint": track.data["manifest_fingerprint"],
        "tokenizer": dict(track.tokenizer), "evaluation": dict(track.evaluation),
        "seeds": [{"seed": 42, "final_update": track.horizon_updates, "validation_bpb": 0.5, "test_bpb": 0.5,
                   "log_sha256": "1" * 64, "checkpoint_sha256": "2" * 64, "run": "."}]}))
    report = validate_dir(folder, capsys)
    assert report["verdict"] == "missing-evidence"
    assert report["reasons"] == {"missing-evidence": [
        "train.log: declared sha256 has no file in the folder",
        "seed 42: run evidence ['metrics.json', 'eval-validation.json', 'eval-test.json', 'train.log'] "
        "is not a file matching its declared sha256"]}
    assert bench_cli.main(["leaderboard", str(folder), "--output", str(tmp_path / "board")]) == 0
    board = json.loads((tmp_path / "board/leaderboard.json").read_text())
    assert board["rows"] == [] and board["excluded"][0]["verdict"] == "missing-evidence"


def test_a_folder_written_by_run_is_eligible_and_says_what_it_checked(launched, tmp_path, capsys):
    folder = run_p2(tmp_path, launched["manifest"], "m0-adamw", 1.2e-3)
    assert sorted(path.name for path in folder.iterdir()) == [
        "config.json", "eval-test.json", "eval-validation.json", "metrics.json", "submission.json", "train.log"]
    value = json.loads((folder / "submission.json").read_text())
    assert sorted(value["files"]) == ["config.json", "eval-test.json", "eval-validation.json", "metrics.json", "train.log"]
    assert value["seeds"][0]["log_sha256"] == value["files"]["train.log"]
    config = json.loads((folder / "config.json").read_text())
    assert config["trainer"] == bench.trainer_pins() | {"torch": torch.__version__}
    report = validate_dir(folder, capsys)
    assert report["verdict"] == "eligible", report["reasons"]
    assert report["checked"][0] == "sha256 of the 5 files in the folder against the 5 declared"
    assert report["checked"][1].startswith("seed 42: config.json in '.' against the declared recipe")
    assert report["unchecked"][-1] == (f"seed 42: checkpoint_sha256 {value['seeds'][0]['checkpoint_sha256']} (of the DCP "
                                       ".metadata file only: tensor layout, not weights) at no URL declared: not checked here")
    assert not any("checkpoint" in line for line in report["checked"])
    assert any("needs rerunning training" in line for line in report["unchecked"])


def field(name, path, value):
    """Set a dotted key of a folder JSON file."""
    def change(folder):
        data = json.loads((folder / name).read_text())
        *parents, key = path.split(".")
        target = data
        for parent in parents:
            target = target[parent]
        target[key] = value
        (folder / name).write_text(json.dumps(data))
    return change


def rehashed(name, path, value):
    """Edit a file and declare its new sha256, as a forger would: the content check must catch it."""
    return lambda folder: (field(name, path, value)(folder), rehash(folder, name))


A = "seed 42: "
TAMPERING = [  # One row per check: the edit, the verdict, the reason.
    # Hashes: an edited file left unrehashed, a missing file, an undeclared file, a run without its config.
    (field("eval-test.json", "bpb", 0.9), "missing-evidence", "eval-test.json: file does not match its declared sha256"),
    (field("metrics.json", "test_bpb", 0.9), "missing-evidence", "metrics.json: file does not match its declared sha256"),
    (field("config.json", "run.seed", 43), "missing-evidence", "config.json: file does not match its declared sha256"),
    (lambda f: (f / "train.log").unlink(), "missing-evidence", "train.log: declared sha256 has no file in the folder"),
    (lambda f: (f / "notes.txt").write_text("x"), "missing-evidence", "notes.txt: file has no declared sha256"),
    (lambda f: ((f / "config.json").unlink(), field("submission.json", "files", {
        k: h for k, h in json.loads((f / "submission.json").read_text())["files"].items() if k != "config.json"})(f)),
     "missing-evidence", A + "run evidence ['config.json'] is not a file matching its declared sha256"),
    (lambda f: field("submission.json", "seeds", [{k: v for k, v in json.loads((f / "submission.json").read_text())[
        "seeds"][0].items() if k != "run"}])(f), "missing-evidence", A + "run folder is not declared"),
    (lambda f: ((f / "metrics.json").write_text("[]"), rehash(f, "metrics.json")),
     "protocol-mismatch", A + "config.json and metrics.json must be JSON objects"),
    (lambda f: ((f / "eval-test.json").write_text("[]"), rehash(f, "eval-test.json")),
     "protocol-mismatch", A + "eval-test.json must be a JSON object"),
    (lambda f: ((f / "metrics.json").write_text("{"), rehash(f, "metrics.json")),
     "protocol-mismatch", A + "run evidence is not valid JSON"),
    # The captured configuration against the declared recipe and the track.
    (rehashed("config.json", "run.seed", 43), "protocol-mismatch", A + "config.json run.seed is 43, the declared recipe has 42"),
    (rehashed("config.json", "optimizer.base_lr", 6e-4), "protocol-mismatch",
     A + "config.json optimizer.base_lr is 0.0006, the declared recipe has 0.0012"),
    (rehashed("config.json", "launch.local_batch_size", 8), "protocol-mismatch",
     A + "config.json launch global batch positions is 524288, expected 1048576"),
    (rehashed("config.json", "launch.manifest_fingerprint", "0" * 64), "protocol-mismatch",
     A + "config.json launch.manifest_fingerprint is '" + "0" * 64),
    (rehashed("config.json", "source.commit", "b" * 40), "protocol-mismatch",
     A + f"config.json source.commit is {'b' * 40!r}, expected {'a' * 40!r}"),
    (rehashed("config.json", "source.dirty", True), "protocol-mismatch",
     A + "config.json source.dirty (tracked files) is True, expected False"),
    (rehashed("config.json", "trainer.torchtitan", "0.2.0"), "protocol-mismatch",
     A + "config.json trainer.torchtitan is '0.2.0', expected '0.3.0'"),
    (rehashed("config.json", "runtime.torch", "2.9.0+cu128"), "protocol-mismatch",
     A + "config.json runtime.torch is '2.9.0', expected '2.14.0'"),
    # metrics.json and the log against the declaration.
    (rehashed("metrics.json", "completed_updates", 4000), "protocol-mismatch",
     A + "metrics.json completed_updates is 4000, expected 4769"),
    (rehashed("metrics.json", "validation_bpb", 0.9), "protocol-mismatch", A + "metrics.json validation_bpb is 0.9, expected 1.0"),
    (rehashed("metrics.json", "test_bpb", 0.9), "protocol-mismatch", A + "metrics.json test_bpb is 0.9, expected 1.005"),
    (lambda f: ((f / "train.log").write_text("step=1 CE=1.0\n"), rehash(f, "train.log")), "protocol-mismatch",
     A + "train.log sha256 is"),
    # Each evaluation output against the declaration, the recipe's inference depth and the track.
    (rehashed("eval-test.json", "bpb", 0.9), "protocol-mismatch", A + "eval-test.json bpb is 0.9, expected 1.005"),
    (lambda f: edit(f / "submission.json", lambda v: v | {"seeds": [v["seeds"][0] | {"test_bpb": 0.9}]}),
     "protocol-mismatch", A + "eval-test.json bpb is 1.005, expected 0.9"),
    (rehashed("eval-test.json", "num_visits", 4), "protocol-mismatch", A + "eval-test.json num_visits is 4, expected 1"),
    (rehashed("eval-test.json", "split", "validation"), "protocol-mismatch",
     A + "eval-test.json split is 'validation', expected 'test'"),
    (rehashed("eval-validation.json", "seq_len", 2048), "protocol-mismatch",
     A + "eval-validation.json seq_len is 2048, expected 4096"),
    (rehashed("eval-test.json", "complete_split", False), "protocol-mismatch",
     A + "eval-test.json complete_split is False, expected True"),
    (rehashed("eval-test.json", "valid_tokens", 1000), "protocol-mismatch",
     A + "eval-test.json valid_tokens is 1000, expected 10009889"),
    (rehashed("eval-test.json", "byte_table_sha256", "0" * 64), "protocol-mismatch",
     A + "eval-test.json byte_table_sha256 is '" + "0" * 64),
    (rehashed("eval-validation.json", "manifest_fingerprint", "0" * 64), "protocol-mismatch",
     A + "eval-validation.json manifest_fingerprint is '" + "0" * 64),
    (rehashed("eval-test.json", "tokenizer.revision", "main"), "protocol-mismatch",
     A + "eval-test.json tokenizer.revision is 'main'"),
]


@pytest.mark.parametrize("tamper, verdict, reason", TAMPERING)
def test_each_kind_of_tampering_has_its_verdict(launched, tmp_path, capsys, tamper, verdict, reason):
    folder = run_p2(tmp_path, launched["manifest"], "m0-adamw", 1.2e-3)
    tamper(folder)
    report = validate_dir(folder, capsys)
    assert report["verdict"] == verdict, report["reasons"]
    assert any(line.startswith(reason) for line in report["reasons"][verdict]), report["reasons"]


def test_an_edited_lr_trial_in_a_sweep_folder_is_caught(launched, tmp_path, capsys):
    folder = sweep(tmp_path, launched["manifest"], "m0-adamw")
    assert validate_dir(folder, capsys)["verdict"] == "eligible"
    edit(folder / "submission.json", lambda v: v | {"lr_trials": [v["lr_trials"][0] | {"validation_bpb": 0.5}]
                                                                  + v["lr_trials"][1:]})
    report = validate_dir(folder, capsys)
    assert "LR trial 0.00015: eval-validation.json bpb is 1.09, expected 0.5" in report["reasons"]["protocol-mismatch"]


def test_only_tracked_files_make_a_source_tree_dirty(tmp_path):
    """Prepared data or outputs inside the checkout leave an honest run clean; an edited tracked file does not."""
    git = lambda *args: subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "model.py").write_text("x = 1\n")
    git("add", "model.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    (tmp_path / "data/prepared").mkdir(parents=True)
    (tmp_path / "data/prepared/manifest.json").write_text("{}")
    assert bench_cli.tracked_changes(tmp_path) is False
    (tmp_path / "model.py").write_text("x = 2\n")
    assert bench_cli.tracked_changes(tmp_path) is True
