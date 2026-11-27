"""Phase 1 results (scripts/results_table.py): the data names existing configurations and the generated table is current."""
import json
import re
import subprocess
import sys

import yaml

import results_table

DATA = json.loads(results_table.DATA.read_text())
RUNS = [(optimizer, run) for group in DATA["groups"] for row in group["rows"]
        for optimizer, run in row["runs"].items()]


def test_every_configuration_in_the_data_exists():
    assert [run["config"] for _, run in RUNS if not (results_table.ROOT / run["config"]).is_file()] == []


def test_every_configuration_is_its_optimizer_file():
    assert all(run["config"].endswith(f"-{optimizer}.json") for optimizer, run in RUNS)


def test_every_method_configuration_is_reported():
    methods = {str(path.relative_to(results_table.ROOT)) for path in (results_table.ROOT / "recipes/methods").glob("*.json")}
    assert methods <= {run["config"] for _, run in RUNS}


def test_the_table_is_current():
    assert results_table.TABLE.read_text() == results_table.render(DATA)


def test_the_page_opens_with_the_paper_of_the_citation():
    paper = yaml.safe_load((results_table.ROOT / "CITATION.cff").read_text())["preferred-citation"]
    assert results_table.PAPER == (paper["title"], paper["url"])
    assert results_table.TABLE.read_text().split("\n\n")[2] == (
        "Paper: [LoopTitan: What Does Parameter Revisit Buy a Mixture-of-Experts?](https://arxiv.org/abs/2611.34821) "
        "(arXiv:2611.34821)")


def test_the_check_passes_on_the_committed_data():
    result = subprocess.run([sys.executable, str(results_table.ROOT / "scripts/results_table.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_check_reports_a_missing_configuration_and_a_wrong_budget():
    data = json.loads(json.dumps(DATA))
    first, second = data["groups"][0]["rows"][1]["runs"]["adamw"], data["groups"][0]["rows"][2]["runs"]["adamw"]
    first["config"] = "recipes/phase1/none-adamw.json"
    second["budgets"]["logical_kv_bytes"] += 0.01
    found = results_table.problems(data)
    assert any("none-adamw.json does not exist" in line for line in found)
    assert any(second["config"] in line and "budgets differ" in line for line in found)


def sections(table: str) -> dict[str, str]:
    return {optimizer: table.split(f"## {name}\n")[1].split("\n## ")[0]
            for optimizer, name in results_table.OPTIMIZERS.items()}


def test_every_rerun_is_a_numbered_note_marked_on_its_row():
    reruns = [(optimizer, run) for optimizer, run in RUNS if "rerun_at_other_lr" in run]
    assert reruns
    for optimizer, run in reruns:
        section = sections(results_table.render(DATA))[optimizer]
        notes = section.split("\n\nNotes (")[1].splitlines()[2:]
        number = notes.index(next(line for line in notes if results_table.rerun_note(run["rerun_at_other_lr"]) in line)) + 1
        row = next(line for line in section.splitlines() if f"(../../{run['config']})" in line)
        assert f"{run['test_bpb']:.6f} (note {number})" in row


def test_the_check_reports_a_rerun_at_the_protocol_learning_rate_or_without_its_values():
    data = json.loads(json.dumps(DATA))
    run = data["groups"][1]["rows"][0]["runs"]["adamw"]
    run["rerun_at_other_lr"] = {"lr": run["lr"], "test_bpb": 1.0, "validation_bpb": 1.0, "issue": 1, "note": "a note"}
    found = results_table.problems(data)
    assert any("rerun uses the protocol learning rate" in line for line in found)
    assert any("rerun note leaves out" in line for line in found)


def test_source_marks_follow_the_lineage_in_ascending_order():
    nodes = results_table.lineage.lineage()
    smelt = results_table.published(results_table.ROOT / "recipes/p2_smelt.json")
    for config in ("phase1/m3", "phase1/m4", "phase1/m7", "phase1/m5", "shape/u", "shape/r",
                   "methods/smelt-v3", "methods/latentloop-residual", "methods/m0-loop"):
        assert smelt in results_table.sources(f"recipes/{config}-adamw.json", nodes)
    for config in ("phase1/m0", "shape/g", "phase1/m0-2e"):
        assert results_table.sources(f"recipes/{config}-adamw.json", nodes) == []
    for line in results_table.TABLE.read_text().splitlines():
        if line.startswith("| ["):
            numbers = [int(mark) for mark in re.findall(r"\[(\d+)\]\(#sources\)", line.split(" | ")[1])]
            assert numbers == sorted(numbers)
