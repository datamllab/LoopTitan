"""Fail-closed checks for ordinary release receipts (no model/GPU dependency)."""
import json
from pathlib import Path
import shutil

import pytest

from recipe_delivery import verify_lm_eval_result, tree_record, verify_tree, verify_process


def complete_result():
    return {"task_hashes": {"arc_easy": "a" * 64, "hellaswag": "b" * 64}, "config": {"limit": None}, "n-shot": {"arc_easy": 0, "hellaswag": 0},
        "n-samples": {"arc_easy": {"original": 2376, "effective": 2376},
                      "hellaswag": {"original": 10042, "effective": 10042}},
        "results": {task: {"acc,none": .25, "acc_norm,none": .26} for task in ("arc_easy", "hellaswag")}}


def test_downstream_requires_actual_full_tasks_not_small_fixture():
    verify_lm_eval_result(complete_result())
    for mutation in ("limit", "fewshot", "count", "missing"):
        result = complete_result()
        if mutation == "limit": result["config"]["limit"] = 2
        if mutation == "fewshot": result["n-shot"]["arc_easy"] = 1
        if mutation == "count": result["n-samples"]["arc_easy"]["effective"] = 2
        if mutation == "missing": del result["results"]["hellaswag"]
        with pytest.raises(ValueError): verify_lm_eval_result(result)


def test_artifact_tree_detects_changed_and_added_files(tmp_path):
    # Change only fresh copies, never a directory listed before: an NFS client can keep a cached
    # listing that omits a file added within the server's timestamp granularity (#83).
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "weights").write_bytes(b"real artifact")
    record = tree_record(tree)
    verify_tree(record)
    verify_tree(dict(record, path=str(shutil.copytree(tree, tmp_path / "same"))))
    added = shutil.copytree(tree, tmp_path / "added")
    (added / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="tree"): verify_tree(dict(record, path=str(added)))
    changed = shutil.copytree(tree, tmp_path / "changed")
    (changed / "weights").write_bytes(b"other artifact")
    with pytest.raises(ValueError, match="tree"): verify_tree(dict(record, path=str(changed)))


def test_process_receipt_rejects_failure(tmp_path):
    log = tmp_path / "log"
    log.write_text("failed")
    from recipe_delivery import file_record
    with pytest.raises(ValueError, match="process"):
        verify_process({"returncode": 1, "argv": ["false"], "cwd": str(tmp_path), "log": file_record(log)})


def test_downstream_receipt_binds_expected_endpoint_before_accepting_metrics(tmp_path):
    from recipe_delivery import file_record, verify_downstream_receipt, write_json
    harness = tmp_path / "harness.json"
    write_json(harness, {"version": "0.4.13", "files": {name: "a" * 64 for name in
        ("tasks/arc/arc_easy.yaml", "tasks/hellaswag/hellaswag.yaml", "tasks/hellaswag/utils.py")}})
    log = tmp_path / "identity.log"
    log.write_text("fixture")
    receipt = tmp_path / "receipt.json"
    write_json(receipt, {"format": "moeloop-release-lm-eval", "schema_version": 1,
        "harness": file_record(harness), "harness_process": {"returncode": 0,
            "argv": ["python"], "cwd": str(tmp_path), "log": file_record(log)},
        "models": {"p1_moe": {"model": {"path": str(tmp_path / "different")}}}})
    with pytest.raises(ValueError, match="required endpoint"):
        verify_downstream_receipt(receipt, expected_models={"p1_moe": tmp_path / "expected"})


def test_downstream_collection_requires_identical_harness_and_unique_models(tmp_path, monkeypatch):
    import recipe_delivery as delivery
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    delivery.write_json(one, {"version": "0.4.13", "files": {"task.yaml": "a" * 64}})
    delivery.write_json(two, {"version": "0.4.13", "files": {"task.yaml": "b" * 64}})
    monkeypatch.setattr(delivery, "verify_downstream_receipt", lambda path: {
        "harness": delivery.file_record(path), "harness_process": {}, "models": {str(path): {}}})
    with pytest.raises(ValueError, match="different harness"):
        delivery.collect_downstream([one, two], tmp_path / "combined.json")
    with pytest.raises(ValueError, match="overlap"):
        delivery.collect_downstream([one, one], tmp_path / "combined.json")


def test_journey_dependency_hook_disables_bytecode_before_caller_hooks(tmp_path, monkeypatch):
    import subprocess
    import sys
    import venv
    import recipe_delivery as delivery

    dependencies = tmp_path / "caller-dependencies"
    dependencies.mkdir()
    (dependencies / "source_hook.py").write_text("import sys\nBYTECODE_DISABLED = sys.dont_write_bytecode\n")
    (dependencies / "source_hook.pth").write_text("import source_hook\n")
    spec, manifest = tmp_path / "spec.json", tmp_path / "manifest.json"
    delivery.write_json(spec, {"run": {"successful_updates": 8}})
    delivery.write_json(manifest, {})
    monkeypatch.setattr(delivery, "verify_package_receipt", lambda _: {
        "source": {"path": str(tmp_path / "frozen-source")}, "wheel": {"path": "unused.whl"}})

    class StopBeforeInstall(Exception):
        pass

    def run(argv, directory, name, **kwargs):
        environment = Path(directory) / "train-env"
        if name == "train-venv":
            venv.EnvBuilder(with_pip=False).create(environment)
        elif name == "training-dependencies":
            Path(argv[-1]).write_text(json.dumps([str(dependencies)]))
        elif name == "link-training-dependencies":
            subprocess.run([str(value) for value in argv], check=True)
        elif name == "train-install":
            # -I ignores inherited PYTHONDONTWRITEBYTECODE. The environment's
            # own hook must disable writes before processing caller .pth hooks.
            subprocess.run([str(environment / "bin/python"), "-I", "-c",
                "import source_hook,sys;assert source_hook.BYTECODE_DISABLED;assert sys.dont_write_bytecode"], check=True)
            assert not list(dependencies.rglob("*.pyc"))
            raise StopBeforeInstall
        else:
            raise AssertionError(f"unexpected stage {name}")
        return {}

    monkeypatch.setattr(delivery, "run", run)
    with pytest.raises(StopBeforeInstall):
        delivery.journey(tmp_path / "package.json", spec, manifest, tmp_path / "journey",
            train_python=sys.executable, hf_python=sys.executable)


def test_full_downstream_requires_native_logged_task_hashes():
    for hashes in ({}, {"arc_easy": "a" * 64}, {"arc_easy": "a" * 64, "hellaswag": "invalid"}, {"arc_easy": "a" * 64, "hellaswag": None}):
        result = complete_result()
        result["task_hashes"] = hashes
        with pytest.raises(ValueError, match="task hash"):
            verify_lm_eval_result(result)


def test_combined_downstream_requires_matching_native_task_hashes(tmp_path):
    import recipe_delivery as delivery
    harness = tmp_path / "harness.json"
    delivery.write_json(harness, {"version": "0.4.13", "files": {name: "a" * 64 for name in
        ("tasks/arc/arc_easy.yaml", "tasks/hellaswag/hellaswag.yaml", "tasks/hellaswag/utils.py")}})
    log = tmp_path / "fixture.log"
    log.write_text("unit fixture; no scoring or model execution")
    process = {"returncode": 0, "argv": ["python"], "cwd": str(tmp_path), "log": delivery.file_record(log)}
    rows = {}
    for name in ("p1_moe", "p2_moe"):
        model = tmp_path / name
        model.mkdir()
        delivery.write_json(model / "config.json", {"labels_objective": "terminal_ce", "model_config": {"num_visits": 2}})
        delivery.write_json(model / "export_provenance.json", {})
        result = tmp_path / f"{name}.json"
        delivery.write_json(result, complete_result() | {"task_hashes": {"arc_easy": "a" * 64, "hellaswag": "b" * 64}})
        samples = {}
        for task in ("arc_easy", "hellaswag"):
            path = tmp_path / f"samples-{name}-{task}.jsonl"
            path.write_text("unit fixture\n")
            samples[task] = delivery.file_record(path)
        rows[name] = {"model": delivery.tree_record(model), "export_provenance": delivery.file_record(model / "export_provenance.json"),
            "result": delivery.file_record(result), "sample_logs": samples, "num_fewshot": 0, "limit": None,
            "readout": "terminal", "dtype": "float32", "num_visits": 2, "process": process | {"argv": ["python", "-m", "lm_eval", "--model", "hf",
                "--model_args", f"pretrained={model},trust_remote_code=True,dtype=float32", "--tasks", "arc_easy,hellaswag", "--num_fewshot", "0", "--log_samples"]}}
    payload = {"format": "moeloop-release-lm-eval", "schema_version": 1, "harness": delivery.file_record(harness),
               "harness_process": process, "models": rows}
    same = tmp_path / "same.json"
    delivery.write_json(same, payload)
    delivery.verify_downstream_receipt(same)
    changed = tmp_path / "changed-result.json"
    delivery.write_json(changed, complete_result() | {"task_hashes": {"arc_easy": "a" * 64, "hellaswag": "c" * 64}})
    rows["p2_moe"]["result"] = delivery.file_record(changed)
    mismatch = tmp_path / "mismatch.json"
    delivery.write_json(mismatch, payload)
    with pytest.raises(ValueError, match="task hashes differ"):
        delivery.verify_downstream_receipt(mismatch)


@pytest.mark.parametrize("batch_size", [None, "8", "auto", "auto:4", "auto:0.5"])
def test_downstream_cli_passes_native_batch_syntax(tmp_path, monkeypatch, batch_size):
    import recipe_delivery as delivery
    captured = {}
    def downstream(models, output, **kwargs):
        captured.update(models=models, **kwargs)
        return output / "not-executed.json"
    monkeypatch.setattr(delivery, "downstream", downstream)
    argv = ["downstream", "--model", f"p1_moe={tmp_path}", "--output", str(tmp_path / "results")]
    if batch_size is not None:
        argv.extend(("--batch-size", batch_size))
    delivery.main(argv)
    assert captured["batch_size"] == (1 if batch_size is None else batch_size)


@pytest.mark.parametrize("batch_size", ["0", "-1", "auto:0", "auto:-1", "auto:nan", "auto:inf", "automatic", "auto:", "auto:4:2", "1.5"])
def test_downstream_cli_rejects_invalid_native_batch_syntax(tmp_path, monkeypatch, batch_size):
    import recipe_delivery as delivery
    monkeypatch.setattr(delivery, "downstream", lambda *args, **kwargs: pytest.fail("invalid batch reached execution"))
    with pytest.raises(SystemExit) as error:
        delivery.main(["downstream", "--model", f"p1_moe={tmp_path}", "--output", str(tmp_path / "results"), "--batch-size", batch_size])
    assert error.value.code == 2
