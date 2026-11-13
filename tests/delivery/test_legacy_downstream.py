"""Exact P1 schema2 HF metadata; no model execution or artifact migration."""
import json
from pathlib import Path

import pytest

import recipe_delivery as delivery


@pytest.fixture(params=("reference", "candidate"))
def p1_artifact(tmp_path, request):
    source = Path(__file__).parent / "fixtures" / f"p1-{request.param}-hf-config.json"
    artifact = tmp_path / f"p1-{request.param}"
    artifact.mkdir()
    (artifact / "config.json").write_bytes(source.read_bytes())
    delivery.write_json(artifact / "export_provenance.json", {"unit_fixture": True})
    return artifact


def native_receipt_files(directory, argv):
    """Ordinary synthetic process/result files for this CPU contract test only."""
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "fixture.log"
    log.write_text("unit fixture; no scoring or model execution")
    harness = directory / "harness.json"
    delivery.write_json(harness, {"version": "0.4.13", "files": {name: "a" * 64 for name in
        ("tasks/arc/arc_easy.yaml", "tasks/hellaswag/hellaswag.yaml", "tasks/hellaswag/utils.py")}})
    results = directory / "results"
    results.mkdir()
    result = results / "results_fixture.json"
    delivery.write_json(result, {"config": {"limit": None},
        "n-shot": {task: 0 for task in delivery.TASK_SAMPLES},
        "n-samples": {task: {"original": count, "effective": count} for task, count in delivery.TASK_SAMPLES.items()},
        "results": {task: {"acc,none": .25, "acc_norm,none": .26} for task in delivery.TASK_SAMPLES},
        "task_hashes": {task: "a" * 64 for task in delivery.TASK_SAMPLES}})
    samples = {}
    for task in delivery.TASK_SAMPLES:
        path = results / f"samples_{task}_fixture.jsonl"
        path.write_text("unit fixture\n")
        samples[task] = delivery.file_record(path)
    process = {"returncode": 0, "argv": argv, "cwd": str(directory), "log": delivery.file_record(log)}
    return delivery.file_record(harness), process, delivery.file_record(result), samples


def test_downstream_accepts_exact_legacy_p1_config_without_rewriting(p1_artifact, tmp_path, monkeypatch):
    before = delivery.tree_record(p1_artifact)
    identity = tmp_path / "identity"
    harness, process, _, _ = native_receipt_files(identity, ["python"])
    monkeypatch.setattr(delivery, "_harness_identity", lambda *args: (harness, process))
    def run(argv, directory, name, **kwargs):
        assert name == "lm-eval" and "--limit" not in argv and "--log_samples" in argv
        assert argv[argv.index("--model_args") + 1] == f"pretrained={p1_artifact},trust_remote_code=True,dtype=float32"
        _, completed, _, _ = native_receipt_files(directory, argv)
        return completed
    monkeypatch.setattr(delivery, "run", run)
    path = delivery.downstream({"p1_moe": p1_artifact}, tmp_path / "downstream", batch_size="auto:4")
    payload = delivery.verify_downstream_receipt(path, expected_models={"p1_moe": p1_artifact})
    assert payload["models"]["p1_moe"]["dtype"] == "float32"
    assert payload["models"]["p1_moe"]["readout_basis"] == "legacy_moeloop_schema_2_without_exit_gate"
    assert delivery.tree_record(p1_artifact) == before


def test_verifier_accepts_exact_legacy_p1_config(p1_artifact, tmp_path):
    config = json.loads((p1_artifact / "config.json").read_text())
    argv = ["python", "-m", "lm_eval", "--model", "hf", "--model_args", f"pretrained={p1_artifact},trust_remote_code=True,dtype=float32",
        "--tasks", "arc_easy,hellaswag", "--num_fewshot", "0", "--log_samples"]
    harness, process, result, samples = native_receipt_files(tmp_path / "evidence", argv)
    row = {"model": delivery.tree_record(p1_artifact), "export_provenance": delivery.file_record(p1_artifact / "export_provenance.json"),
        "result": result, "sample_logs": samples, "process": process, "readout": "terminal",
        "num_fewshot": 0, "limit": None, "dtype": "float32", "num_visits": config["model_config"]["num_visits"]}
    receipt = tmp_path / "receipt.json"
    delivery.write_json(receipt, {"format": "moeloop-release-lm-eval", "schema_version": 1,
        "harness": harness, "harness_process": process, "models": {"p1_moe": row}})
    delivery.verify_downstream_receipt(receipt)
    row["dtype"] = "bfloat16"
    changed = tmp_path / "wrong-dtype.json"
    delivery.write_json(changed, {"format": "moeloop-release-lm-eval", "schema_version": 1,
        "harness": harness, "harness_process": process, "models": {"p1_moe": row}})
    with pytest.raises(ValueError, match="float32"):
        delivery.verify_downstream_receipt(changed)
    row["dtype"] = "float32"
    row["process"] = process | {"argv": [value.replace(",dtype=float32", ",dtype=bfloat16") for value in argv]}
    wrong_command = tmp_path / "wrong-command-dtype.json"
    delivery.write_json(wrong_command, {"format": "moeloop-release-lm-eval", "schema_version": 1,
        "harness": harness, "harness_process": process, "models": {"p1_moe": row}})
    with pytest.raises(ValueError, match="float32"):
        delivery.verify_downstream_receipt(wrong_command)


@pytest.mark.parametrize("change", ("nonterminal", "unknown_schema", "exit_gate"))
def test_terminal_semantics_reject_unknown_or_nonterminal_legacy_metadata(change):
    source = Path(__file__).parent / "fixtures" / "p1-reference-hf-config.json"
    config = json.loads(source.read_text())
    if change == "nonterminal": config["labels_objective"] = "expected_ce"
    if change == "unknown_schema": config["model_config"]["schema_version"] = 3
    if change == "exit_gate": config["model_config"]["exit_distribution"] = True
    with pytest.raises(ValueError, match="readout semantics"):
        delivery._terminal_readout_basis(config)


def test_terminal_semantics_accept_known_schema1_and_explicit_terminal_ce():
    source = Path(__file__).parent / "fixtures" / "p1-reference-hf-config.json"
    config = json.loads(source.read_text())
    config["model_config"]["schema_version"] = 1
    assert delivery._terminal_readout_basis(config) == "legacy_moeloop_schema_1_without_exit_gate"
    config["model_config"].update(schema_version=3, exit_distribution=True)
    config["labels_objective"] = "terminal_ce"
    assert delivery._terminal_readout_basis(config) == "explicit_terminal_ce"
