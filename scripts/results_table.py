#!/usr/bin/env python3
"""Write the Phase 1 results table from its data: python scripts/results_table.py [--check]

docs/results/phase1.json holds, for every design and optimizer, the frozen configuration, the
learning rate, the test and validation bits per byte copied from the results comment of the run
issue it names, and the three budgets relative to recipes/phase1/m0-<optimizer>.json: the
relative differences moeloop.comparison.compare_recipes reports with the P2 track's seq_len and
accounting (eval/bench/tracks.json), rounded to 6 decimals. A row's source marks are derived, not
stored: the published sources (method, version, url) of its configuration and of every ancestor in
scripts/lineage.py, where a recipe's source is its own ``source`` field or, through base_recipe,
the one it is built from. An entry that was rerun at another learning rate keeps its protocol value
and adds rerun_at_other_lr (learning rate, test and validation BPB, issue, and the note's
sentence), which becomes a numbered note under its optimizer's tables. This writes
docs/results/phase1.md from the data. --check writes nothing and exits 1 if a configuration does
not exist, a stored budget differs from the cost tool's, a group's budgets disagree with its
matched flag, a rerun uses the protocol learning rate or its note leaves out its values, or, once
the data is consistent, phase1.md is not what this writes.
"""
import argparse
import json
import sys
from pathlib import Path

from moeloop.bench import load_tracks
from moeloop.comparison import compare_recipes
from moeloop.recipe_spec import RecipeSpec

import lineage

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
    "- Source: the published designs the configuration and its ancestors in `scripts/lineage.py` follow, "
    "numbered in [Sources](#sources).",
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


def published(path: Path) -> str | None:
    """A recipe's published source as the reference list prints it: its own, or its base recipe's."""
    entry = json.loads(path.read_text())["source"]
    while "url" not in entry and "base_recipe" in entry:
        entry = json.loads((ROOT / entry["base_recipe"]).read_text())["source"]
    return f"[{entry['method']} {entry['version']}]({entry['url']})" if "url" in entry else None


def sources(config: str, nodes: dict) -> list[str]:
    """The published sources of a configuration and of its lineage ancestors."""
    node, found = config.removeprefix("recipes/").rsplit("-", 1)[0], []
    while node:
        found.append(published(lineage.recipe_files(node)[0]))
        node = nodes[node][0]
    return list(dict.fromkeys(entry for entry in found if entry))


def rerun_note(rerun: dict) -> str:
    return f"{rerun['note']} ({link(rerun['issue'])})."


def render(data: dict) -> str:
    rows = [row for group in data["groups"] for row in group["rows"]]
    nodes = lineage.lineage()
    marks = {row["design"]: sources(row["runs"]["adamw"]["config"], nodes) for row in rows}
    references = list(dict.fromkeys(entry for row in rows for entry in marks[row["design"]]))
    lines = ["# Phase 1 results", "",
             "<!-- Written by scripts/results_table.py from phase1.json; edit the data, not this file. -->", "",
             data["protocol"], "", data["not_shown"], "", *NOTES]
    for optimizer, name in OPTIMIZERS.items():
        base = next(row["runs"][optimizer]["test_bpb"] for row in rows
                         if row["runs"][optimizer]["config"] == REFERENCE.format(optimizer))
        lines += ["", f"## {name}"]
        notes = []
        for group in data["groups"]:
            lines += ["", f"### {group['title']}", "", *HEADER]
            for row in group["rows"]:
                run = row["runs"][optimizer]
                rerun = run.get("rerun_at_other_lr")
                if rerun:
                    notes.append(rerun_note(rerun))
                numbers = sorted(references.index(entry) + 1 for entry in marks[row["design"]])
                cells = [f"[{row['design']}](../../{run['config']})",
                         ", ".join(f"[{number}](#sources)" for number in numbers),
                         f"{run['test_bpb']:.6f}" + (f" (note {len(notes)})" if rerun else ""), signed(1000 * (run["test_bpb"] - base), 2),
                         f"{run['validation_bpb']:.6f}",
                         *(signed(100 * run["budgets"][metric], 1) + "%" for metric in METRICS),
                         lr(run["lr"]), link(run["issue"])]
                lines.append("| " + " | ".join(cells) + " |")
        if notes:
            lines += ["", f"Notes ({name}):", "", *(f"{i}. {note}" for i, note in enumerate(notes, 1))]
    lines += ["", "## Sources", "", *(f"{i}. {entry}" for i, entry in enumerate(references, 1)), "",
              "What each configuration keeps from its source and what it adapts: [method map](../method-map.md)."]
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
                rerun = run.get("rerun_at_other_lr")
                if rerun and rerun["lr"] == run["lr"]:
                    found.append(f"{run['config']}: the rerun uses the protocol learning rate")
                if rerun and not all(value in rerun["note"] for value in (
                        lr(rerun["lr"]), f"{rerun['test_bpb']:.6f}", f"{rerun['validation_bpb']:.6f}")):
                    found.append(f"{run['config']}: the rerun note leaves out its learning rate or BPB")
                if group["matched"] != all(abs(value) <= tolerance for value in run["budgets"].values()):
                    found.append(f"{run['config']}: budgets disagree with matched={group['matched']}")
    if found:  # the table is compared once the data is consistent
        return found
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
