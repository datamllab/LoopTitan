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


def test_every_rerun_is_a_note_under_its_optimizer():
    table = results_table.TABLE.read_text()
    sections = {name: table.split(f"## {name}\n")[1].split("\n## ")[0] for name in results_table.OPTIMIZERS.values()}
    reruns = [(optimizer, run["rerun_at_other_lr"]) for optimizer, run in RUNS if "rerun_at_other_lr" in run]
    assert reruns
    for optimizer, rerun in reruns:
        assert rerun["lr"] != next(run["lr"] for o, run in RUNS if o == optimizer)
        section = sections[results_table.OPTIMIZERS[optimizer]]
        assert f"a rerun at learning rate {results_table.lr(rerun['lr'])} gave test BPB {rerun['test_bpb']:.6f}" in section
        assert f"issues/{rerun['issue']})" in section


def test_the_check_reports_a_rerun_at_the_protocol_learning_rate():
    data = json.loads(json.dumps(DATA))
    run = data["groups"][1]["rows"][0]["runs"]["adamw"]
    run["rerun_at_other_lr"] = {"lr": run["lr"], "test_bpb": 1.0, "validation_bpb": 1.0, "issue": 1}
    assert any("rerun uses the protocol learning rate" in line for line in results_table.problems(data))
