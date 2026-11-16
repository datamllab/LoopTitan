"""Phase 1 results (scripts/results_table.py): the data names existing configurations and the generated table is current."""
import json
import subprocess
import sys

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


def test_every_row_source_has_a_url_and_covers_its_configurations_own_source():
    data = json.loads(json.dumps(DATA))
    row = next(row for group in data["groups"] for row in group["rows"] if row["runs"]["adamw"]["config"].startswith("recipes/methods/"))
    row["sources"] = ["recipes/p2_moe.json"]
    found = results_table.problems(data)
    assert any("recipes/p2_moe.json: its source field has no url" in line for line in found)
    assert any(row["runs"]["adamw"]["config"] in line and "not among the row's sources" in line for line in found)
