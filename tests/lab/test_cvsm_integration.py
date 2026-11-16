import json
import os
from pathlib import Path
import subprocess
import sys

from mlflow import MlflowClient

import doctor
from run_record_helpers import Repo
from smoke import SPEC


ROOT = Path(__file__).resolve().parents[2]


def test_cpu_fixture_through_real_mlflow_and_unmodified_spec_checker(tmp_path):
    template = tmp_path / "template.db"
    MlflowClient(f"sqlite:///{template}").search_experiments()
    repo = Repo(tmp_path, template)
    config = repo.root / "scripts" / "run.toml"
    config.write_text(config.read_text().replace('seed_key = "seed"', 'seed_key = "debug.seed"')
                      + "require_tracking_uri = true\n")
    repo.commit("native seed option and explicit backend")
    for repeat in (1, 2):
        for seed in (1, 2):
            command = [sys.executable, str(ROOT / "scripts" / "smoke.py"),
                       "--loop-repeat", str(repeat), "--debug.seed", str(seed)]
            result = repo.run(issue=9, cmd=command)
            assert result.returncode == 0, result.stderr
    runs = repo.runs()
    assert len(runs) == 4
    for run in runs:
        assert run.info.status == "FINISHED"
        assert run.data.params["cvsm.issue"] == "9"
        assert run.data.params["cvsm.seed"] in ("1", "2")
        assert run.data.params["cfg.derived.target_input_tokens"] == "64"
        assert run.data.params["cfg.record_kind"] == "engineering-fixture"
        assert run.data.metrics == {"fixture.completed": 1.0}
        config = json.loads(repo.artifact(run, "config.json").read_text())
        assert config["derived"]["global_batch_size"] == 8
        assert json.loads(repo.artifact(run, "native-config.json").read_text()) == config["native"]
    results = repo.results("--spec", "-", issue=9, input=SPEC)
    assert "4 runs in 2 groups" in results
    assert "Spec check:\n- No deviations" in results

    # A deviation is a new recorded attempt, never an edit to historical runs.
    result = repo.run(issue=9, cmd=[sys.executable, str(ROOT / "scripts" / "smoke.py"),
                                    "--debug.seed", "3"])
    assert result.returncode == 0
    result = repo.run(issue=9, cmd=[sys.executable, "-c", "raise SystemExit(7)"])
    assert result.returncode == 7
    results = repo.results("--spec", "-", issue=9, input=SPEC)
    assert "6 runs" in results
    assert "`native.debug.seed in 1,2` does not hold" in results
    assert len([r for r in repo.runs() if r.info.status == "FAILED"]) == 1


def test_both_commands_refuse_implicit_local_database_before_launch(tmp_path):
    config = tmp_path / "run.toml"
    config.write_text('require_tracking_uri = true\nexperiment = "test"\ngpus = []\n')
    marker = tmp_path / "must-not-launch"
    env = {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}
    for script, options in (
        ("run.py", ["--", sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]),
        ("results.py", []),
    ):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script),
                                 "--issue", "9", "--config", str(config), *options],
                                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode != 0
        assert "Set MLFLOW_TRACKING_URI explicitly" in result.stderr
    assert not marker.exists()
    assert list(tmp_path.iterdir()) == [config]


def test_doctor_uses_bounded_read_only_http_and_sdk_calls(monkeypatch, capsys):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            assert limit == 1024
            return b"OK"

    def health(url, timeout):
        calls.append((url, timeout))
        return Response()

    class ReadOnlyClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://fixture.invalid:5000"

        def search_experiments(self, max_results):
            assert max_results == 1
            calls.append("search")
            return []

    monkeypatch.setattr(doctor, "urlopen", health)
    monkeypatch.setattr("mlflow.MlflowClient", ReadOnlyClient)
    # Isolate the CLI's process-local timeout settings from the rest of pytest.
    for name in ("MLFLOW_HTTP_REQUEST_TIMEOUT", "MLFLOW_HTTP_REQUEST_MAX_RETRIES", "MLFLOW_DISABLE_AGENT_HINT"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    assert doctor.main(["--tracking-uri", "http://fixture.invalid:5000"]) == 0
    assert calls == [("http://fixture.invalid:5000/health", 5), "search"]
    assert json.loads(capsys.readouterr().out)["writes"] == 0
