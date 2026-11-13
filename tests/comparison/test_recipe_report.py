"""Finite report arithmetic uses raw observations, never extrapolated gains."""
import importlib

import pytest


def test_local_compute_interpolation_keeps_actual_brackets_and_weights():
    report = importlib.import_module("recipe_report")
    curve = [{"step": 1, "global_labels": 10, "validation_ce": 5., "cumulative_model_flops_estimate": 100.},
             {"step": 2, "global_labels": 20, "validation_ce": 3., "cumulative_model_flops_estimate": 200.}]
    point = report.interpolate_compute(curve, 125.)
    assert point["validation_ce"] == 4.5
    assert point["weights"] == [.75, .25]
    assert point["brackets"] == curve
    assert report.interpolate_compute(curve, 50.) is None
    assert report.interpolate_compute(curve, 250.) is None


def test_common_compute_retains_the_shared_observed_domain_only():
    report = importlib.import_module("recipe_report")
    def curve(xs):
        return [{"step": i, "global_labels": i * 10, "validation_ce": 5. - i,
                 "cumulative_model_flops_estimate": x} for i, x in enumerate(xs, 1)]
    points = report.common_compute(curve([100., 200., 300.]), curve([150., 250., 350.]))
    assert [p["compute"] for p in points] == [150., 200., 250., 300.]
    assert all(p["reference"]["brackets"] and p["candidate"]["brackets"] for p in points)


def test_continuous_window_uses_actual_label_deltas_and_memory():
    report = importlib.import_module("recipe_report")
    metrics = [{"step": s, "global_progress": {"label_tokens": s * 100},
                "step_elapsed_seconds": 2., "peak_active_gib": 3., "peak_reserved_gib": 4.}
               for s in range(1, 26)]
    result = report.summarize_window(metrics, start_step=1, warmup_updates=5, measurement_updates=20, world_size=2)
    assert result["effective_labels_per_second"] == {"median": 50., "min": 50., "max": 50.}
    assert result["measured_steps"] == list(range(6, 26))
    assert result["measured_gpu_hours"] == 80 / 3600
    assert result["peak_allocated_gib"] == 3.
    with pytest.raises(ValueError, match="continuous|missing"):
        report.summarize_window(metrics[:-1], start_step=1, warmup_updates=5, measurement_updates=20, world_size=2)
    with pytest.raises(ValueError, match="warmup"):
        report.summarize_window(metrics, start_step=1, warmup_updates=4, measurement_updates=20, world_size=2)


def test_negative_interaction_is_preserved_with_four_raw_values():
    report = importlib.import_module("recipe_report")
    values = {"a0_r0": 4., "a1_r0": 3.9, "a0_r1": 3.8, "a1_r1": 3.5}
    result = report.sharing_interaction(values)
    assert result["raw_terminal_ce"] == values
    assert result["I_AR"] == pytest.approx(-.2)


def test_curve_rebuild_uses_raw_progress_and_retains_config_data_source(tmp_path, monkeypatch):
    import json
    checker = importlib.import_module("check_release")
    report = importlib.import_module("recipe_report")
    def put(name, value):
        (tmp_path / name).write_text(json.dumps(value))
    request = {"source": {"commit": "a" * 40, "files": {}}, "manifest_fingerprint": "b" * 64, "world_size": 2}
    put("request.json", request)
    for name in ("spec.json", "continuation_identity.json", "outcome.json"):
        put(name, {"fixture": True})
    metric = {"step": 1, "global_progress": {"label_tokens": 90, "padded_positions": 100}}
    put("metrics.jsonl", metric)
    put("validation.jsonl", {"step": 1, "ce": 4.})
    monkeypatch.setattr(checker, "native_gpu_hours", lambda directory, world_size: .25)
    result = report.analyze_run(tmp_path, {"flops": {"training_per_token": 12.}, "parameters": {"unique_non_embedding": 42}})
    assert result["curve"][0]["global_labels"] == 90
    assert result["curve"][0]["cumulative_model_flops_estimate"] == 1200.
    assert result["basis"]["source"] == request["source"]
    assert result["basis"]["manifest_fingerprint"] == "b" * 64
    assert result["basis"]["sha256"]["metrics.jsonl"] == checker.digest(tmp_path / "metrics.jsonl")
    metric["global_progress"]["label_tokens"] = 80
    put("metrics.jsonl", metric)
    recomputed = report.analyze_run(tmp_path, {"flops": {"training_per_token": 12.}, "parameters": {"unique_non_embedding": 42}})
    assert recomputed["curve"][0]["global_labels"] == 80
    assert recomputed["basis"]["sha256"] != result["basis"]["sha256"]


def test_public_replot_reads_only_report_points_without_raw_evidence(tmp_path, monkeypatch):
    import json
    from types import ModuleType, SimpleNamespace
    report = importlib.import_module("recipe_report")
    published = {"anchors": {"P2": {"arms": {"reference": {
        "run": "/unavailable/private/run", "curve": [
            {"global_labels": 1e9, "validation_ce": 4., "cumulative_model_flops_estimate": 2e18}]}}}}}
    path, output = tmp_path / "published-analysis.json", tmp_path / "curves.png"
    path.write_text(json.dumps(published))
    def no_raw_evidence(*args, **kwargs):
        pytest.fail("published replot must not load raw run evidence")
    monkeypatch.setattr(report, "build_report", no_raw_evidence)
    monkeypatch.setattr(report, "read_jsonl", no_raw_evidence)
    plotted = []
    axis = SimpleNamespace(plot=lambda xs, ys, **kwargs: plotted.append((xs, ys, kwargs["label"])),
                           set=lambda **kwargs: None, legend=lambda: None)
    figure = SimpleNamespace(tight_layout=lambda: None, savefig=lambda target: target.write_bytes(b"figure fixture"))
    pyplot = ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda *args, **kwargs: (figure, [axis, axis])
    pyplot.close = lambda figure: None
    monkeypatch.setitem(__import__("sys").modules, "matplotlib", ModuleType("matplotlib"))
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.pyplot", pyplot)
    report.main(["--report", str(path), "--plot", str(output)])
    assert plotted == [([1.], [4.], "P2 reference"), ([2.], [4.], "P2 reference")]
    assert output.read_bytes() == b"figure fixture"
