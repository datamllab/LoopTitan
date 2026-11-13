import json
import subprocess
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import pytest

from moeloop_lab import protocol


def native_config():
    return {
        "training": {"local_batch_size": 1, "global_batch_size": 128,
                     "seq_len": 4096, "steps": 80},
        "debug": {"seed": 47},
        "model_spec": {"name": "loopmoe", "model": {"loop_repeat": 2}},
        "checkpoint": {"folder": "/private/checkpoints", "load_step": -1},
    }


def build(native=None, **overrides):
    args = dict(world_size=32, data_parallel_degree=16, gradient_accumulation_steps=8,
                assets={"data_manifest_sha256": "a" * 64}, sources={})
    return protocol.build_record(native or native_config(), **(args | overrides))


def test_uses_batch_mesh_degree_instead_of_total_gpu_count():
    record = build()
    assert record["derived"] == {
        "world_size": 32, "data_parallel_degree": 16, "gradient_accumulation_steps": 8,
        "global_batch_size": 128, "input_tokens_per_step": 524288,
        "target_input_tokens": 41943040,
    }
    assert record["native"] == native_config()  # full runtime config, including checkpoint inputs


def test_native_automatic_batch_is_resolved_from_observed_accumulation():
    native = native_config()
    native["training"]["global_batch_size"] = -1
    record = build(native, gradient_accumulation_steps=1)
    assert record["native"]["training"]["global_batch_size"] == -1
    assert record["derived"]["global_batch_size"] == 16


@pytest.mark.parametrize("overrides", [
    {"gradient_accumulation_steps": 4}, {"data_parallel_degree": 64},
    {"world_size": 31}, {"gradient_accumulation_steps": True},
])
def test_rejects_inconsistent_or_nonintegral_observed_budget(overrides):
    with pytest.raises(ValueError):
        build(**overrides)


def test_does_not_change_native_settings_and_refuses_nan():
    native = native_config()
    original = deepcopy(native)
    build(native)
    assert native == original
    native["optimizer"] = {"lr": float("nan")}
    with pytest.raises(ValueError):
        build(native)


def test_only_global_rank_zero_touches_run_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.delenv("RUN_DIR", raising=False)
    assert protocol.capture_trainer(None, data_manifest=None, tokenizer_files={},
                                    model_module=None) is None
    protocol.write_metrics({"loss": float("nan")})
    assert list(tmp_path.iterdir()) == []


def test_final_artifacts_are_atomic_and_cannot_be_overwritten(monkeypatch, tmp_path):
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    record = build()
    protocol.write_record(record)
    original = (tmp_path / "config.json").read_bytes()
    assert json.loads(original) == record
    with pytest.raises(FileExistsError):
        protocol.write_record(build(gradient_accumulation_steps=8))
    assert (tmp_path / "config.json").read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        protocol.write_metrics({"loss": float("inf")})
    assert not (tmp_path / "metrics.json").exists()
    protocol.write_metrics({"completed_steps": 80})
    assert json.loads((tmp_path / "metrics.json").read_text()) == {"completed_steps": 80}


def test_capture_observes_real_modules_assets_and_trainer_mesh(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    module_file = source / "model.py"
    module_file.write_text("# tiny source fixture, not TorchTitan\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "model.py"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=test", "-c",
                    "user.email=test@example.com", "commit", "-qm", "fixture"], check=True)
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    module = ModuleType("fixture")
    module.__file__ = str(module_file)
    monkeypatch.setitem(sys.modules, "torchtitan", module)
    monkeypatch.setattr(protocol, "_torchtitan_pin", lambda: sha)
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("RUN_DIR", str(output))
    monkeypatch.setenv("RANK", "0")
    data = tmp_path / "manifest.json"
    data.write_text('{"format":"fixture"}')
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"fixture":true}')
    mesh_requests = []

    def get_mesh(name):
        mesh_requests.append(name)
        return SimpleNamespace(size=lambda: 16)

    trainer = SimpleNamespace(config=SimpleNamespace(to_dict=native_config),
                              parallel_dims=SimpleNamespace(world_size=32, dp_enabled=True,
                                                            get_mesh=get_mesh),
                              gradient_accumulation_steps=8)
    record = protocol.capture_trainer(trainer, data_manifest=data,
                                      tokenizer_files={"tokenizer": tokenizer}, model_module=module)
    assert mesh_requests == ["batch"]
    assert record["sources"]["torchtitan"] == {"commit": sha, "dirty": False}
    assert record["sources"]["model"] == {"commit": sha, "dirty": False}
    assert record["assets"]["data_manifest_sha256"] == protocol.sha256_file(data)
    assert record["assets"]["tokenizer_files"]["tokenizer"] == protocol.sha256_file(tokenizer)
    assert json.loads((output / "native-config.json").read_text()) == native_config()
    assert json.loads((output / "config.json").read_text()) == record
    module_file.write_text("# changed runtime\n")
    with pytest.raises(ValueError, match="clean"):
        protocol.observe_source(module_file, expected_commit=sha)
    module_file.write_text("# tiny source fixture, not TorchTitan\n")
    with pytest.raises(ValueError, match="commit"):
        protocol.observe_source(module_file, expected_commit="0" * 40)


def test_empty_or_non_numeric_final_metrics_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    for metrics in ({}, {"ok": True}, {"loss": "1"}):
        with pytest.raises(ValueError):
            protocol.write_metrics(metrics)
    assert not (tmp_path / "metrics.json").exists()


def test_protocol_does_not_import_training_or_tracking_dependencies():
    result = subprocess.run([sys.executable, "-c", "import sys; import moeloop_lab.protocol; "
                             "assert 'torch' not in sys.modules; assert 'mlflow' not in sys.modules"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
