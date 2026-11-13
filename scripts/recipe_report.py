#!/usr/bin/env python3
"""Rebuild finite P1/P2 curves, C interaction and measured performance from raw files.

Model FLOPs are analytical padded-position estimates. Local common-compute
interpolation stays inside actual observations; no scaling-law fit is made.
Matplotlib is optional and imported only by --plot. --report replots supplied
public analysis points without authenticating or loading their raw evidence.
--detail-from-step selects a separate contrast figure; the default overview
retains every observation.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
import statistics


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interpolate_compute(curve, compute):
    """Retain both actual local endpoints and their linear interpolation weights."""
    xs = [p["cumulative_model_flops_estimate"] for p in curve]
    require(xs and all(math.isfinite(x) for x in xs) and all(a < b for a, b in zip(xs, xs[1:])),
            "compute curve must have strictly increasing finite observations")
    if not xs[0] <= compute <= xs[-1]:
        return None
    index = bisect.bisect_left(xs, compute)
    if xs[index] == compute:
        return {"validation_ce": curve[index]["validation_ce"], "brackets": [curve[index]], "weights": [1.]}
    lower, upper = curve[index - 1], curve[index]
    weight = (compute - xs[index - 1]) / (xs[index] - xs[index - 1])
    weights = [1. - weight, weight]
    return {"validation_ce": sum(w * p["validation_ce"] for w, p in zip(weights, (lower, upper))),
            "brackets": [lower, upper], "weights": weights}


def common_compute(reference, candidate):
    lower = max(reference[0]["cumulative_model_flops_estimate"], candidate[0]["cumulative_model_flops_estimate"])
    upper = min(reference[-1]["cumulative_model_flops_estimate"], candidate[-1]["cumulative_model_flops_estimate"])
    xs = sorted({point["cumulative_model_flops_estimate"] for curve in (reference, candidate) for point in curve
                 if lower <= point["cumulative_model_flops_estimate"] <= upper})
    result = []
    for x in xs:
        left, right = interpolate_compute(reference, x), interpolate_compute(candidate, x)
        result.append({"compute": x, "reference": left, "candidate": right,
                       "candidate_minus_reference_ce": right["validation_ce"] - left["validation_ce"]})
    return result


def sharing_interaction(values):
    require(set(values) == {"a0_r0", "a0_r1", "a1_r0", "a1_r1"}
            and all(math.isfinite(value) for value in values.values()), "four finite C endpoint CEs are required")
    return {"raw_terminal_ce": values,
            "I_AR": values["a1_r1"] - values["a1_r0"] - values["a0_r1"] + values["a0_r0"],
            "definition": "CE_AR-CE_A-CE_R+CE_B; one scale/budget/seed, descriptive only"}


def summarize_window(metrics, *, start_step, warmup_updates, measurement_updates, world_size):
    require(type(warmup_updates) is int and warmup_updates >= 5 and
            type(measurement_updates) is int and measurement_updates >= 20, "performance requires >=5 warmup and >=20 measured updates")
    require(type(start_step) is int and start_step >= 1 and type(world_size) is int and world_size > 0,
            "invalid continuous performance window geometry")
    by_step = {item["step"]: item for item in metrics}
    steps = list(range(start_step, start_step + warmup_updates + measurement_updates))
    require(all(s in by_step for s in steps) and len(by_step) == len(metrics), "continuous performance window has missing/duplicate updates")
    measured = steps[warmup_updates:]
    seconds, rates, active, reserved = [], [], [], []
    for step in measured:
        item, previous = by_step[step], by_step[step - 1]
        duration = item.get("step_elapsed_seconds", math.nan)
        labels = item["global_progress"]["label_tokens"] - previous["global_progress"]["label_tokens"]
        allocated, cached = item.get("peak_active_gib", math.nan), item.get("peak_reserved_gib", math.nan)
        require(math.isfinite(duration) and duration > 0 and labels > 0
                and math.isfinite(allocated) and math.isfinite(cached) and 0 < allocated <= cached,
                "performance window lacks actual step timing/effective labels/allocated and reserved peaks")
        seconds.append(duration)
        rates.append(labels / duration)
        active.append(allocated)
        reserved.append(cached)
    def summary(values):
        return {"median": statistics.median(values), "min": min(values), "max": max(values)}
    return {"warmup_steps": steps[:warmup_updates], "measured_steps": measured,
            "step_seconds": summary(seconds), "effective_labels_per_second": summary(rates),
            "aggregate_effective_labels_per_second": sum(by_step[s]["global_progress"]["label_tokens"] - by_step[s - 1]["global_progress"]["label_tokens"] for s in measured) / sum(seconds),
            "peak_allocated_gib": max(active), "peak_reserved_gib": max(reserved),
            "measured_gpu_hours": sum(seconds) * world_size / 3600,
            "scope": "continuous native update window; process GPUHr separately includes initialization/validation/checkpoint"}


def analyze_run(directory, cost):
    from recipe_qualification import digest
    from check_release import native_gpu_hours
    directory = Path(directory).resolve()
    request = read_json(directory / "request.json")
    metrics = read_jsonl(directory / "metrics.jsonl")
    validations = read_jsonl(directory / "validation.jsonl")
    by_step = {item["step"]: item for item in metrics}
    curve = []
    for point in validations:
        progress = by_step[point["step"]]["global_progress"]
        curve.append({"step": point["step"], "global_labels": progress["label_tokens"],
                      "padded_positions": progress["padded_positions"], "validation_ce": point["ce"],
                      "cumulative_model_flops_estimate": cost["flops"]["training_per_token"] * progress["padded_positions"]})
    require(curve and all(math.isfinite(p["validation_ce"]) for p in curve), "raw heldout curve missing/nonfinite")
    require(all(a["step"] < b["step"] and a["global_labels"] < b["global_labels"] for a, b in zip(curve, curve[1:])),
            "raw heldout curve order/label progression differs")
    basis_files = ("request.json", "spec.json", "metrics.jsonl", "validation.jsonl", "continuation_identity.json", "outcome.json")
    return {"run": str(directory), "basis": {"source": request["source"], "manifest_fingerprint": request["manifest_fingerprint"],
             "sha256": {name: digest(directory / name) for name in basis_files}, "cost": cost},
            "curve": curve, "unique_non_embedding_parameters": cost["parameters"]["unique_non_embedding"],
            "actual_gpu_hours": native_gpu_hours(directory, request["world_size"])}


def analyze_anchor(path):
    anchor = read_json(path)
    comparison = read_json(anchor["comparison"])["comparison"]
    arms = {role: analyze_run(run, comparison["costs"][role]) for role, run in anchor["arms"].items()}
    return {"arms": arms, "common_compute": common_compute(arms["reference"]["curve"], arms["candidate"]["curve"]),
            "heldout": {role: {split: read_json(file) for split, file in endpoints.items()}
                        for role, endpoints in anchor["heldout"].items()}}


def performance_report(path, expected_runs):
    from recipe_qualification import digest
    value = read_json(path)
    require(value.get("format") == "moeloop-recipe-performance" and value.get("version") == 1
            and set(value.get("windows", {})) == {"p2_reference", "p2_candidate", "sharing"},
            "required plain/loop/partial-sharing performance windows are missing")
    results, runtimes = {}, []
    for name, window in value["windows"].items():
        run = Path(window["run"]).resolve()
        allowed = expected_runs[name]
        require(str(run) in {str(Path(p).resolve()) for p in allowed}, "performance window does not reuse an actual declared run")
        request, spec = read_json(run / "request.json"), read_json(run / "spec.json")
        require(request["runtime"].get("gpu_name"), "performance actual GPU hardware identity is missing")
        runtimes.append(request["runtime"])
        summary = summarize_window(read_jsonl(run / "metrics.jsonl"), start_step=window["start_step"],
            warmup_updates=window["warmup_updates"], measurement_updates=window["measurement_updates"], world_size=request["world_size"])
        summary["basis"] = {"run": str(run), "metrics_sha256": digest(run / "metrics.jsonl"), "source": request["source"],
                            "runtime": request["runtime"], "world_size": request["world_size"],
                            "seq_len": request["seq_len"], "global_batch_size": request["global_batch_size"],
                            "activation_checkpointing": spec["run"].get("activation_checkpointing", "none")}
        results[name] = summary
    require(all(runtime == runtimes[0] for runtime in runtimes), "performance hardware/software runtime identities differ")
    return results


def build_report(index):
    from moeloop.costs import estimate_cost
    from moeloop.recipe_spec import RecipeSpec
    p2 = read_json(index["p2"])
    inherited = read_json(Path(index["inherited_v2"]["directory"]) / "release.json")
    p1_index = read_json(Path(inherited["inherited_v1"]["directory"]) / "release.json")
    anchors = {"P1": analyze_anchor(p1_index["scientific_anchor"]), "P2": analyze_anchor(index["p2"])}
    grid = {}
    for name, run in index["sharing_grid"]["arms"].items():
        spec = RecipeSpec.from_dict(read_json(Path(run) / "spec.json"))
        grid[name] = analyze_run(run, estimate_cost(spec, seq_len=spec.run["seq_len"],
            activation_checkpointing=spec.run.get("activation_checkpointing", "none")))
    interaction = sharing_interaction({name: arm["curve"][-1]["validation_ce"] for name, arm in grid.items()})
    base = grid["a0_r0"]["unique_non_embedding_parameters"]
    interaction.update(arms=grid, parameter_deltas={name: arm["unique_non_embedding_parameters"] - base for name, arm in grid.items()})
    expected = {"p2_reference": [p2["arms"]["reference"]], "p2_candidate": [p2["arms"]["candidate"]],
                "sharing": list(index["sharing_grid"]["arms"].values())}
    return {"format": "moeloop-recipe-analysis", "version": 1, "manifest_fingerprint": index["manifest_fingerprint"],
            "anchors": anchors, "sharing_grid": interaction, "performance": performance_report(index["performance"], expected),
            "limitations": ["One predeclared seed42; no scaling-law inference or extrapolation.",
                            "Analytical matrix FLOPs use padded positions; measured native GPUHr includes compile, validation and checkpoint.",
                            "Common-compute CE is locally linearly interpolated inside observed brackets; weights and raw endpoints are retained.",
                            "C is a same-shape component panel, not an iso-parameter comparison or full Foil/LOOM reproduction."]}


def plot_report(report, path):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for anchor, panel in report["anchors"].items():
        for role, arm in panel["arms"].items():
            curve = arm["curve"]
            for axis, key, scale in ((axes[0], "global_labels", 1e9), (axes[1], "cumulative_model_flops_estimate", 1e18)):
                axis.plot([p[key] / scale for p in curve], [p["validation_ce"] for p in curve], label=f"{anchor} {role}")
    for axis, label in zip(axes, ("Valid training labels (B)", "Analytical model compute (EFLOP)")):
        axis.set(xlabel=label, ylabel="Validation CE")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_detail(report, path, from_step):
    """Show observed late curves and the supplied bracketed compute contrasts."""
    import matplotlib.pyplot as plt

    selected = {}
    for anchor in ("P1", "P2"):
        panel = report["anchors"][anchor]
        curves = {role: [p for p in panel["arms"][role]["curve"] if p["step"] >= from_step]
                  for role in ("reference", "candidate")}
        require(all(curves.values()), f"{anchor}: no observations at/after step {from_step}")
        labels = {role: {p["global_labels"]: p for p in curve} for role, curve in curves.items()}
        require(all(len(labels[role]) == len(curve) for role, curve in curves.items()),
                f"{anchor}: duplicate observed training label counts")
        shared = sorted(labels["reference"].keys() & labels["candidate"].keys())
        require(shared, f"{anchor}: no exact shared observed label counts after cutoff")
        compute = [p for p in panel["common_compute"]
                   if all(p[role]["brackets"] and
                          all(b["step"] >= from_step for b in p[role]["brackets"])
                          for role in ("reference", "candidate"))]
        require(compute, f"{anchor}: no common-compute points with all brackets after cutoff")
        require(all(len(p[role]["brackets"]) == len(p[role]["weights"])
                    and all(0 <= w <= 1 for w in p[role]["weights"])
                    and math.isclose(sum(p[role]["weights"]), 1.)
                    for p in compute for role in ("reference", "candidate")),
                f"{anchor}: common-compute bracket weights are invalid")
        selected[anchor] = curves, labels, shared, compute

    style = {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
             "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
             "svg.fonttype": "none", "pdf.fonttype": 42}
    with plt.rc_context(style):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))
        native, label_delta, compute_delta, sharing = axes.flat
        colors = ("#0072B2", "#D55E00")
        for (anchor, (curves, labels, shared, compute)), color in zip(selected.items(), colors):
            for role, name, linestyle, marker in (("reference", "MoE", "-", "o"),
                                                  ("candidate", "SMELT", "--", "s")):
                curve = curves[role]
                native.plot([p["global_labels"] / 1e9 for p in curve],
                            [p["validation_ce"] for p in curve], color=color,
                            linestyle=linestyle, marker=marker, markersize=3,
                            linewidth=1.5, label=f"{anchor} {name}")
            label_delta.plot([x / 1e9 for x in shared],
                             [labels["candidate"][x]["validation_ce"] -
                              labels["reference"][x]["validation_ce"] for x in shared],
                             color=color, marker="o", markersize=3, linewidth=1.5, label=anchor)
            # Reuse public analysis values, brackets and weights; do not interpolate anew.
            compute_delta.plot([p["compute"] / 1e18 for p in compute],
                               [p["candidate_minus_reference_ce"] for p in compute],
                               color=color, marker="o", markersize=2.5, linewidth=1.5, label=anchor)

        native.set(title="a  Native validation curves · observed points", ylabel="Validation CE",
                   xlabel="Actual training labels (B)")
        native.legend(ncol=2, frameon=False)
        label_delta.set(title="b  Exact shared observed labels · no interpolation",
                        ylabel="CE difference (SMELT − MoE)", xlabel="Actual training labels (B)")
        compute_delta.set(title="c  Common compute · supplied local interpolation",
                          ylabel="CE difference (SMELT − MoE)",
                          xlabel="Document-aware analytical model compute (EFLOP)")
        for axis in (label_delta, compute_delta):
            axis.axhline(0, color="#555555", linewidth=0.8, linestyle=":", zorder=0)
            axis.legend(frameon=False, loc="upper right" if axis is label_delta else "lower left")
        compute_delta.text(0.98, 0.95, "Both arms' brackets at/after cutoff\nNo extrapolation or fit",
                           transform=compute_delta.transAxes, ha="right", va="top", fontsize=8)

        grid = report["sharing_grid"]
        names = (("a0_r0", "B"), ("a0_r1", "R"), ("a1_r0", "A"), ("a1_r1", "AR"))
        values = [grid["raw_terminal_ce"][key] for key, _ in names]
        require(all(math.isfinite(x) for x in values), "C endpoints must be finite")
        for y, ((key, name), value) in enumerate(zip(names, values)):
            sharing.scatter(value, y, color="#444444", s=35, zorder=3)
            sharing.annotate(f"{value:.6f}", (value, y), xytext=(7, 0),
                             textcoords="offset points", va="center", fontsize=9)
        sharing.set_yticks(range(4),
                          [f"{name}   Δ non-emb. params: +{grid['parameter_deltas'][key]:,}"
                           for key, name in names], fontsize=8)
        sharing.set(title="d  C endpoints · same shape, not iso-parameter", xlabel="Native validation CE",
                    ylim=(3.6, -0.6))
        span = max(values) - min(values)
        sharing.set_xlim(min(values) - max(span * 0.15, 0.005), max(values) + max(span * 0.5, 0.02))
        steps = sorted({grid["arms"][key]["curve"][-1]["step"] for key, _ in names})
        sharing.text(0.02, 0.03, "C terminal step: " + ", ".join(f"{s:,}" for s in steps),
                     transform=sharing.transAxes, fontsize=8)
        for axis in axes.flat:
            axis.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
            axis.tick_params(labelsize=9)

        figure.suptitle(f"Native validation contrasts | P1/P2 observed step ≥ {from_step:,}",
                        fontsize=15, x=0.055, ha="left", y=0.98)
        boundary = ("10,000 is the predeclared WSD decay boundary. " if from_step == 10000 else "")
        figure.text(0.055, 0.935, boundary + "Full-history overview and all public observations remain available.",
                    fontsize=9, color="#444444")
        figure.text(0.055, 0.035,
                    "P1/P2 use a fixed ~1M-label native validation slice; their full ~10M-label HF evaluations are separate. "
                    "One predeclared seed (42); descriptive only.\n"
                    "P1/P2 cutoff applies to a–c only. Compute uses padded positions with document-aware attention; "
                    "C shows raw endpoint dots and parameter increments vs B.\n"
                    "C: A = per-visit attention; R = per-visit router; experts and norms remain shared. "
                    "Its step-2,048 endpoints are independent of the P1/P2 cutoff.", fontsize=8, linespacing=1.6)
        figure.tight_layout(rect=(0.02, 0.09, 0.99, 0.91), h_pad=2.8, w_pad=2.8)
        figure.savefig(path, dpi=220)
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence", type=Path, help="rebuild analysis from raw release evidence")
    source.add_argument("--report", type=Path, help="replot supplied analysis JSON; does not verify raw evidence")
    parser.add_argument("--output", type=Path, help="required with --evidence")
    parser.add_argument("--plot", type=Path, help="optional Matplotlib figure; install matplotlib separately")
    parser.add_argument("--detail-from-step", type=int,
                        help="with --plot, draw native contrasts from this observed step (e.g. WSD boundary 10000)")
    args = parser.parse_args(argv)
    if args.detail_from_step is not None and (args.detail_from_step < 0 or args.plot is None):
        parser.error("--detail-from-step requires --plot and a nonnegative step")
    def plot(report):
        if args.detail_from_step is None:
            plot_report(report, args.plot)
        else:
            plot_detail(report, args.plot, args.detail_from_step)
    if args.report is not None:
        if args.plot is None or args.output is not None:
            parser.error("--report requires --plot and does not use --output")
        plot(read_json(args.report))
        return
    if args.output is None:
        parser.error("--evidence requires --output")
    report = build_report(read_json(args.evidence / "release.json"))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot(report)


if __name__ == "__main__":
    main()
