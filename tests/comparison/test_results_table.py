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
