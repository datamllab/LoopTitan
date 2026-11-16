#!/usr/bin/env python3
"""Looped MoE Bench command line: run, sweep, validate and leaderboard.

    python scripts/bench.py run --spec recipes/phase1/m0-adamw.json --lr 1.2e-3 --gpus 8 --output DIR
    python scripts/bench.py sweep --track M --spec recipes/p1/m3-muon.json --gpus 8 --output DIR
    python scripts/bench.py validate --dir DIR/submission
    python scripts/bench.py leaderboard A/submission B/submission --output DIR

``run`` and ``sweep`` take a frozen recipe spec (default track P2) and write it to
``DIR/spec.json`` with the track's fixed optimizer settings for its optimizer filled in
where the spec leaves them out; the checker reports any the spec sets differently. Each
trial is ``recipe_experiments.py run --spec DIR/spec.json --lr LR --evaluate`` into
``DIR/lr-<LR>``; its results go to ``DIR/lr-<LR>.json``, and a trial with that file is
not launched again. ``sweep`` runs the track's LR grid for the spec's optimizer and its
lr_selection rule: lowest full-validation BPB, ties to the smaller LR, and while that LR
is on an edge one more trial beyond it (x2 above, /2 below). ``run`` launches one LR.
Track P2 has one calibrated LR per optimizer, so a P2 submission is a single ``run`` at that LR.
On a track with one LR per optimizer, ``run`` at that LR is the whole submission, with
no LR trials. Both write the submission folder ``DIR/submission`` (replacing an earlier
one) and print its ``moeloop.bench.validate`` report. The folder holds the selected
trial, which is the seed-42 run, at its top and every other LR trial in ``lr-<LR>/``; each
run has its captured ``config.json`` (spec, launch topology, source and runtime from the
trial's request, trainer package versions; ``source.dirty`` says whether a tracked file
differed from the commit when the trial was launched, untracked files do not count), ``metrics.json``, ``eval-validation.json``,
``eval-test.json`` and ``train.log`` (the last training launch's log), and
``submission.json`` declares the sha256 of every file. Under ``scripts/run.py`` they
also write, at the end, ``$RUN_DIR/config.json`` (the selected run's) and
``metrics.json`` (BPB, LR, GPU hours of all trials, budget residuals); the trials do not
see ``RUN_DIR``.

One output directory is for one spec and one checkout: finished ``lr-*.json`` records are
reused without re-checking them. If a trial dies partway, delete its partial ``lr-<LR>``
directory and rerun bench.py, which launches that LR afresh, or resume it by hand with
``recipe_experiments.py run --resume``: bench.py never resumes, and relaunching over a
partial directory fails.

``validate --dir`` prints the report of one submission folder (its reasons, what it
checked and what it cannot check without rerunning) and exits 1 unless it is eligible.
``leaderboard`` validates every submission folder and writes ``leaderboard.json``
(``moeloop.bench.leaderboard``) and ``leaderboard.md``.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moeloop import bench
from recipe_qualification import digest, write_json


def track_spec(spec: dict, track: bench.Track) -> dict:
    """``spec`` with the track's fixed settings of its optimizer wherever the spec has none."""
    fixed = track.optimizers["fixed"][spec["optimizer"]["name"]]
    optimizer = {key: value for key, value in fixed.items() if key != "muon"} | spec["optimizer"]
    if "muon" in fixed:
        optimizer["muon"] = fixed["muon"] | optimizer.get("muon", {})
    return spec | {"optimizer": optimizer}


def launch(command: list[str]) -> None:
    """Run one trial; RUN_DIR stays with the bench process."""
    env = {key: value for key, value in os.environ.items() if key != "RUN_DIR"}
    subprocess.run(command, env=env, check=True)


def tracked_changes(root: Path = ROOT) -> bool | None:
    """Whether a tracked file differs from the checked-out commit; None outside a git checkout.

    Untracked files (prepared data, run outputs) do not count: they cannot change the code that ran.
    """
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                            text=True, capture_output=True)
    return bool(status.stdout.strip()) if status.returncode == 0 else None


def trial(output: Path, manifest: Path, lr: float, gpus: int) -> dict:
    """Launch one LR unless its record exists; return the record read from its experiment output."""
    path, directory = output / f"lr-{lr}.json", output / f"lr-{lr}"
    if path.is_file():
        return json.loads(path.read_text())
    started, changes = time.monotonic(), tracked_changes()  # The tree at launch, not when the run ends.
    launch([sys.executable, str(ROOT / "scripts" / "recipe_experiments.py"), "run", "--spec", str(output / "spec.json"),
            "--manifest", str(manifest), "--output", str(directory), "--gpus", str(gpus), "--lr", str(lr), "--evaluate"])
    outcome = json.loads((directory / "outcome.json").read_text())
    bpb = {split: json.loads((directory / f"eval-{split}.json").read_text())["bpb"] for split in ("validation", "test")}
    process = json.loads(sorted(directory.glob("process-*.json"))[-1].read_text())  # The last training launch.
    value = {"lr": lr, "seed": json.loads((directory / "spec.json").read_text())["run"]["seed"],
             "final_update": outcome["completed_updates"], "validation_bpb": bpb["validation"], "test_bpb": bpb["test"],
             "log_sha256": process["log_sha256"],
             "checkpoint_sha256": digest(directory / outcome["full_checkpoint"] / ".metadata"),
             "gpu_hours": (time.monotonic() - started) * gpus / 3600,
             "trainer": {name: importlib.metadata.version(name) for name in bench.trainer_pins()},
             "tracked_changes": changes}
    write_json(path, value)
    return value


def select(trials: dict[float, float]) -> float:
    """Lowest validation BPB, ties to the smaller LR."""
    return min(trials, key=lambda lr: (trials[lr], lr))


def next_lr(trials: dict[float, float]) -> float | None:
    """The trial the track rule adds next, or None once the selected LR is interior."""
    selected = select(trials)
    if selected == max(trials):
        return selected * 2
    if selected == min(trials):
        return selected / 2
    return None


def write_run(folder: Path, directory: Path, track: bench.Track, trials: list[dict], record: dict) -> None:
    """One run's evidence: captured config, metrics, evaluation outputs and training log."""
    spec = json.loads((directory / "spec.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    config = {key: spec[key] for key in ("name", "model", "training", "optimizer", "inference", "run")}
    config["launch"] = {key: request[key] for key in
                        ("world_size", "local_batch_size", "gradient_accumulation_steps", "manifest_fingerprint")}
    # The trial's own dirty flag also counts untracked files; a source-revision.json install keeps it.
    dirty = request["source"]["dirty"] if record["tracked_changes"] is None else record["tracked_changes"]
    config |= {"source": request["source"] | {"dirty": dirty}, "runtime": request["runtime"], "trainer": record["trainer"],
               "bench": {"track": track.name, "trial_lrs": [t["lr"] for t in trials]}}
    write_json(folder / "config.json", config)
    write_json(folder / "metrics.json", {"completed_updates": record["final_update"], "lr": record["lr"],
        "validation_bpb": record["validation_bpb"], "test_bpb": record["test_bpb"], "gpu_hours": record["gpu_hours"]})
    for split in ("validation", "test"):
        shutil.copyfile(directory / f"eval-{split}.json", folder / f"eval-{split}.json")
    shutil.copyfile(sorted(directory.glob("process-*.json"))[-1].with_suffix(".log"), folder / "train.log")


def submit(output: Path, track: bench.Track, trials: list[dict], selected: dict) -> dict:
    """Write the folder DIR/submission from the trials' outputs, print its report; RUN_DIR records."""
    folder = output / "submission"
    shutil.rmtree(folder, ignore_errors=True)
    runs = {t["lr"]: "." if t["lr"] == selected["lr"] else f"lr-{t['lr']}" for t in trials}
    for t in trials:
        write_run(folder / runs[t["lr"]], output / f"lr-{t['lr']}", track, trials, t)
    directory = output / f"lr-{selected['lr']}"
    recipe = json.loads((directory / "spec.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    validation, test = (json.loads((directory / f"eval-{split}.json").read_text()) for split in ("validation", "test"))
    manifest = json.loads(Path(request["manifest"]).read_text())
    searched = len(track.lr_grid[recipe["optimizer"]["name"]]) > 1  # A one-LR track takes no LR trials.
    value = {
        "schema_version": 2, "name": recipe["name"], "track": track.name, "recipe": recipe,
        "files": {path.relative_to(folder).as_posix(): bench.sha256(path)
                  for path in sorted(folder.rglob("*")) if path.is_file()},
        "source_commit": request["source"]["commit"], "manifest_fingerprint": request["manifest_fingerprint"],
        "tokenizer": test["tokenizer"] | {"tokenizer_json_sha256": manifest["tokenizer"]["files"]["tokenizer/tokenizer.json"]},
        "evaluation": {"metric": "bpb", "complete_split": validation["complete_split"] and test["complete_split"],
                       "seq_len": test["seq_len"], "selection_split": validation["split"],
                       "selection_label_tokens": validation["valid_tokens"], "ranking_split": test["split"],
                       "ranking_label_tokens": test["valid_tokens"], "byte_table_sha256": test["byte_table_sha256"]},
        "seeds": [{key: selected[key] for key in
                   ("seed", "final_update", "validation_bpb", "test_bpb", "log_sha256", "checkpoint_sha256")} | {"run": "."}],
        "lr_trials": [{key: t[key] for key in ("lr", "seed", "final_update", "validation_bpb", "log_sha256")}
                      | {"run": runs[t["lr"]]} for t in trials] if searched else [],
        "gpu_hours": math.fsum(t["gpu_hours"] for t in trials)}
    write_json(folder / "submission.json", value)
    report = bench.validate(bench.Submission.from_dict(value), track, folder)
    print(json.dumps(report.to_dict(), indent=1))
    if os.environ.get("RUN_DIR"):
        run_dir = Path(os.environ["RUN_DIR"])
        shutil.copyfile(folder / "config.json", run_dir / "config.json")
        write_json(run_dir / "metrics.json", json.loads((folder / "metrics.json").read_text()) | {
            "lr_trials": len(trials), "gpu_hours": value["gpu_hours"],
            **{f"budget_residual_{key}": entry["relative_difference"] for key, entry in report.budgets.items()}})
    return value


def run(args, track: bench.Track) -> None:
    value = trial(args.output, args.manifest, args.lr, args.gpus)
    submit(args.output, track, [value], value)


def sweep(args, track: bench.Track, spec: dict) -> None:
    trials, pending = {}, list(track.lr_grid[spec["optimizer"]["name"]])
    while pending:
        for lr in pending:
            trials[lr] = trial(args.output, args.manifest, lr, args.gpus)
        following = next_lr({lr: value["validation_bpb"] for lr, value in trials.items()}) if len(trials) > 1 else None
        pending = [] if following is None else [following]
    selected = trials[select({lr: value["validation_bpb"] for lr, value in trials.items()})]
    submit(args.output, track, [trials[lr] for lr in sorted(trials)], selected)


def validate(folder: Path) -> bench.Report:
    value = bench.Submission.from_dict(json.loads((folder / "submission.json").read_text()))
    return bench.validate(value, bench.load_tracks()[value.track], folder)


def cell(value, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def markdown(board: dict) -> str:
    lines = ["| Track | Rank | Submission | Test BPB | Seeds | Range | Expert LR multiplier | Expert group | GPU hours | CORE |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in board["rows"]:
        spread = "" if row["range_test_bpb"] is None else "–".join(cell(v, 4) for v in row["range_test_bpb"])
        lines.append(f"| {row['track']} | {row['rank']} | {row['submission']} | {cell(row['mean_test_bpb'], 4)} | "
                     f"{row['n_seeds']} | {spread} | {cell(row['expert_lr_multiplier'], 3)} | "
                     f"{row['expert_group_size'] or ''} | {cell(row['gpu_hours'], 2)} | {cell(row['core'], 3)} |")
    if board["excluded"]:
        lines += ["", "Excluded: " + ", ".join(f"{row['submission']} (Track {row['track']}, {row['verdict']})"
                                              for row in board["excluded"])]
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "sweep"):
        command = sub.add_parser(name, help="one LR of a frozen spec" if name == "run" else "the track LR grid and rule")
        command.add_argument("--spec", type=Path, required=True, help="frozen RecipeSpec JSON")
        command.add_argument("--track", default="P2", choices=sorted(bench.load_tracks()))
        if name == "run":
            command.add_argument("--lr", type=float, required=True, help="peak LR; Muon sets muon.lr to it too")
        command.add_argument("--gpus", type=int, required=True, help="total ranks per trial")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--manifest", type=Path,
                             default=Path(os.environ.get("MOELOOP_DATA", "data/prepared")) / "manifest.json",
                             help="prepared data manifest (default: $MOELOOP_DATA/manifest.json)")
    sub.add_parser("validate", help="check one submission folder").add_argument(
        "--dir", type=Path, required=True, dest="folder",
        help="submission folder: submission.json and the files whose sha256 it declares")
    command = sub.add_parser("leaderboard", help="leaderboard JSON and Markdown of eligible submissions")
    command.add_argument("submissions", type=Path, nargs="+", help="submission folders")
    command.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command in ("run", "sweep"):
        track = bench.load_tracks()[args.track]
        spec = track_spec(json.loads(args.spec.read_text()), track)
        args.output = args.output.resolve()
        write_json(args.output / "spec.json", spec)
        if args.command == "run":
            run(args, track)
        else:
            sweep(args, track, spec)
        return 0
    if args.command == "validate":
        report = validate(args.folder)
        print(json.dumps(report.to_dict(), indent=1))
        return 0 if report.verdict == "eligible" else 1
    board = bench.leaderboard([validate(path) for path in args.submissions])
    write_json(args.output / "leaderboard.json", board)
    (args.output / "leaderboard.md").write_text(markdown(board))
    print(args.output / "leaderboard.json", args.output / "leaderboard.md", sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
