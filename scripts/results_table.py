#!/usr/bin/env python3
"""Write the Phase 1 results table from its data: python scripts/results_table.py [--check]

docs/results/phase1.json holds, for every design and optimizer, the frozen configuration, the
learning rate, the test and validation bits per byte copied from the results comment of the run
issue it names, and the three budgets relative to recipes/phase1/m0-<optimizer>.json: the
relative differences moeloop.comparison.compare_recipes reports with the P2 track's seq_len and
accounting (eval/bench/tracks.json), rounded to 6 decimals. An entry whose loss fell slowly at the
protocol learning rate keeps that value and adds rerun_at_other_lr (learning rate,
test and validation BPB, issue), which becomes a numbered note under its optimizer's tables. This
writes docs/results/phase1.md from the data. --check writes nothing and exits 1 if a configuration
does not exist, a stored budget differs from the cost tool's, a group's budgets disagree with its
matched flag, a rerun uses the protocol learning rate, or phase1.md is not what this writes.
"""
import argparse
import json
import sys
from pathlib import Path

from moeloop.bench import load_tracks
from moeloop.comparison import compare_recipes
from moeloop.recipe_spec import RecipeSpec

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "results" / "phase1.json"
TABLE = ROOT / "docs" / "results" / "phase1.md"
REFERENCE = "recipes/phase1/m0-{}.json"
METRICS = ("unique_non_embedding_parameters", "training_flops_per_token", "logical_kv_bytes")
OPTIMIZERS = {"adamw": "AdamW", "muon": "Muon"}
ISSUES = "https://github.com/datamllab/LoopTitan/issues/"
NOTES = [
    "- Test, Validation: bits per byte (BPB) on the complete held-out test and validation splits after the last update.",
    "- Δ test: test BPB minus the ordinary MoE's with the same optimizer, in mBPB (0.001 BPB).",
    "- Parameters, FLOPs, KV: unique non-embedding parameters, training FLOPs per token and logical KV cache "
    "relative to `recipes/phase1/m0-<optimizer>.json`, from `moeloop.comparison.compare_recipes` with the P2 "
    "track's accounting in `eval/bench/tracks.json`. Matched budgets are within 5% on all three.",
    "- Source: the entry of the method map (see [Sources](#sources)) that the design follows.",
    "- Run: the issue whose results comment holds the values. The design links to its frozen configuration.",
]
HEADER = ["| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]


def spec(path: str) -> RecipeSpec:
    return RecipeSpec.from_dict(json.loads((ROOT / path).read_text()))


def budgets(config: str, optimizer: str) -> dict[str, float]:
    """The three budgets of a configuration relative to the reference, counted as the P2 track counts them."""
    track = load_tracks()["P2"]
    result = compare_recipes(spec(REFERENCE.format(optimizer)), spec(config), seq_len=track.seq_len,
                             tolerance=track.budget["tolerance"], **track.budget["accounting"])
    entries = result["matched"] | result["measured-different"]
    return {metric: round(entries[metric]["relative_difference"], 6) for metric in METRICS}


def link(issue: int) -> str:
    return f"[#{issue}]({ISSUES}{issue})"


def lr(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa.removesuffix('.0')}e{int(exponent)}"  # 1.2e-3, 3e-4


def signed(value: float, digits: int) -> str:
    return f"{round(value, digits) + 0:+.{digits}f}"  # + 0 turns -0.0 into 0.0


def render(data: dict) -> str:
    rows = [row for group in data["groups"] for row in group["rows"]]
    sources = list(dict.fromkeys(row["source"] for row in rows if row["source"]))
    lines = ["# Phase 1 results", "",
             "<!-- Written by scripts/results_table.py from phase1.json; edit the data, not this file. -->", "",
             data["protocol"], "", *NOTES]
    for optimizer, name in OPTIMIZERS.items():
        reference = next(row["runs"][optimizer]["test_bpb"] for row in rows
                         if row["runs"][optimizer]["config"] == REFERENCE.format(optimizer))
        lines += ["", f"## {name}"]
        notes = []
        for group in data["groups"]:
            lines += ["", f"### {group['title']}", "", *HEADER]
            for row in group["rows"]:
                run = row["runs"][optimizer]
                rerun = run.get("rerun_at_other_lr")
                if rerun:
                    notes.append(f"At the protocol learning rate the loss of this configuration fell slowly under {name}; "
                                 f"a rerun at learning rate {lr(rerun['lr'])} gave test BPB {rerun['test_bpb']:.6f} "
                                 f"and validation BPB {rerun['validation_bpb']:.6f} ({link(rerun['issue'])}).")
                cells = [f"[{row['design']}](../../{run['config']})",
                         f"[{sources.index(row['source']) + 1}](#sources)" if row["source"] else "",
                         f"{run['test_bpb']:.6f}" + (f" (note {len(notes)})" if rerun else ""), signed(1000 * (run["test_bpb"] - reference), 2),
                         f"{run['validation_bpb']:.6f}",
                         *(signed(100 * run["budgets"][metric], 1) + "%" for metric in METRICS),
                         lr(run["lr"]), link(run["issue"])]
                lines.append("| " + " | ".join(cells) + " |")
        if notes:
            lines += ["", f"Notes ({name}):", "", *(f"{i}. {note}" for i, note in enumerate(notes, 1))]
    lines += ["", "## Sources", "", *(f"{i}. [{source}](../method-map.md)" for i, source in enumerate(sources, 1))]
    return "\n".join(lines) + "\n"


def problems(data: dict) -> list[str]:
    tolerance = load_tracks()["P2"].budget["tolerance"]
    found = []
    for group in data["groups"]:
        for row in group["rows"]:
            for optimizer, run in row["runs"].items():
                if not (ROOT / run["config"]).is_file():
                    found.append(f"{run['config']} does not exist")
                    continue
                if run["budgets"] != budgets(run["config"], optimizer):
                    found.append(f"{run['config']}: stored budgets differ from the cost tool's")
                if run.get("rerun_at_other_lr", {}).get("lr") == run["lr"]:
                    found.append(f"{run['config']}: the rerun uses the protocol learning rate")
                if group["matched"] != all(abs(value) <= tolerance for value in run["budgets"].values()):
                    found.append(f"{run['config']}: budgets disagree with matched={group['matched']}")
    if not TABLE.is_file() or TABLE.read_text() != render(data):
        found.append(f"{TABLE.relative_to(ROOT)} is not current: run python scripts/results_table.py")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the data or the table is not current")
    args = parser.parse_args()
    data = json.loads(DATA.read_text())
    if args.check:
        found = problems(data)
        print("\n".join(found) or "phase1 results: current")
        sys.exit(1 if found else 0)
    TABLE.write_text(render(data))


if __name__ == "__main__":
    main()
