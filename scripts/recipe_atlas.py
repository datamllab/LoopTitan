"""Zero-GPU design-space atlas: costs of every recipe over a small geometry grid.

For each recipe the script sweeps width, FFN width, expert count and the
structural knobs (layers, visits) that recipe allows. The width, FFN and expert
axes are multiples of the reference's own values (``default_grid``), so every
reference, from the C grid to P1/P2, gets a grid at its own scale; ``--grid``
overrides the width, FFN, expert and recipe lists. It records the analytical
estimates of ``moeloop.costs.estimate_cost``, and marks the shapes that match a
reference on all three budgets within a tolerance, using
``moeloop.comparison.compare_recipes`` unchanged. A cost the estimator cannot
give (``None``) stays ``null``/empty and is never matched, counted as zero or
drawn; it is listed under ``unknown``, and only then does ``unknown_reason`` say why.

The grid is coarse, so "no match" only means none on this grid. The reference is an
explicit RecipeSpec JSON (``--reference``; default: the C-grid baseline
``recipes/c_a0_r0.json``), one run per track. Combinations a constructor cannot
build (top_k above num_experts, heads not dividing dim) are skipped.

Everything is arithmetic and needs neither a GPU nor torch. The estimates are the matrix-FLOP and logical-KV
counts documented in ``moeloop.costs``, not measured hardware work.

Output files in ``--output DIR`` (sorted, fixed format, byte-identical reruns):
``atlas.json``, ``atlas.csv`` and, if matplotlib is installed (``report``
extra), ``atlas.svg``. ``--table FILE`` writes the one-row-per-recipe table
(``budget_table``): the nearest shape of each recipe, a matching one
if the grid has any, with its three relative differences to the reference.
``--arm SPEC.json`` (repeatable) adds a trained arm that lies off the grid to that table, compared
with the reference by ``compare_recipes`` like every grid point. ``--table FILE --check`` writes
nothing and exits 1 if FILE differs from a fresh run.

Recipes: the ordinary MoE and dense controls and the nine named recipes (``sandwich`` and
``relaxed`` on their MoE path only, with prelude, coda and LoRA rank at the constructor defaults).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moeloop import recipes
from moeloop.comparison import compare_recipes
from moeloop.costs import estimate_cost
from moeloop.recipe_spec import RecipeSpec

# default_grid scales width, FFN width, experts and depth around the reference.
DIM_FACTORS = (0.75, 0.875, 1, 1.125, 1.25)
HIDDEN_FACTORS = (0.5, 0.75, 0.875, 1, 1.125, 1.25, 1.5)
EXPERT_FACTORS = (0.5, 0.75, 1, 1.5, 2)
DEPTH_FACTORS = (0.5, 0.75, 1, 1.25, 1.5)  # moe/dense layers vs the reference's logical depth
SMELT_FACTORS = (0.5, 0.625, 0.75, 0.875, 1)  # SMELT physical layers (its middle half runs twice)
# Structural axes of the other recipes; visits are the passes / eval_passes / group_size axes.
STRUCTURE = {
    "meta": {"core_layers": (1, 2, 4, 8), "passes": (2, 3, 4)},
    "foil": {"core_layers": (2, 4, 8), "passes": (1, 2, 4)},
    "expert_tying": {"groups": (1, 2, 4, 8), "group_size": (2, 4)},
    "huginn": {"core_layers": (1, 2, 4, 8), "eval_passes": (2, 4, 6)},
    "loom": {"core_layers": (1, 2, 4, 8), "passes": (2, 3, 4)},
    "ouro": {"core_layers": (1, 2, 4, 8), "passes": (2, 3, 4)},
    "sandwich": {"core_layers": (1, 2, 3, 4), "passes": (2, 3, 4, 8)},  # prelude and coda stay at the 2 layers each
    "relaxed": {"core_layers": (1, 2, 4, 8), "passes": (2, 3, 4)},  # LoRA rank stays at the default 16
}

# Contract metric (see compare_recipes) -> atlas field.
METRICS = {"unique_non_embedding_parameters": "unique_non_embedding_parameters",
           "training_flops_per_token": "training_flops_per_position",
           "logical_kv_bytes": "kv_bytes_per_token"}
DEFAULT_REFERENCE = "recipes/c_a0_r0.json"


def _scaled(value: int, factors, *, multiple: int = 1) -> list[int]:
    """``value`` times each factor, rounded to a multiple; factor 1 keeps ``value`` exactly."""
    return sorted({value if f == 1 else max(multiple, int(value * f / multiple + 0.5) * multiple) for f in factors})


def default_grid(reference) -> dict:
    """Axes at the reference's scale: the reference's own dim/hidden_dim/experts/depth are on the grid."""
    model = reference.model
    if model.num_experts < 1:
        raise ValueError("the reference needs experts; pass --grid num_experts=... for a dense reference")
    depth = model.prefix_cells + model.num_cells * model.num_visits + model.suffix_cells
    layers = {"layers": _scaled(depth, DEPTH_FACTORS)}
    smelt = sorted({max(4, 2 * int(depth * f / 2 + 0.5)) for f in SMELT_FACTORS})
    return {"dims": _scaled(model.dim, DIM_FACTORS, multiple=model.dim // model.num_heads),
            "hidden_dims": _scaled(model.hidden_dim, HIDDEN_FACTORS, multiple=8),
            "num_experts": _scaled(model.num_experts, EXPERT_FACTORS),
            "structure": {"moe": layers, "dense": layers, "smelt": {"physical_layers": smelt},
                          **{name: {axis: list(values) for axis, values in axes.items()} for name, axes in STRUCTURE.items()}}}


def override_grid(grid: dict, items: list[str]) -> dict:
    """Apply ``KEY=v1,v2`` items: ``dims``, ``hidden_dims``, ``num_experts`` (ints) or ``recipes`` (names)."""
    grid = dict(grid)
    for item in items:
        key, _, text = item.partition("=")
        values = text.split(",")
        if key == "recipes":
            grid["structure"] = {name: grid["structure"][name] for name in values}  # KeyError names an unknown recipe
        elif key in ("dims", "hidden_dims", "num_experts"):
            grid[key] = sorted({int(value) for value in values})
        else:
            raise ValueError(f"unknown grid key {key!r}; use dims, hidden_dims, num_experts or recipes")
    return grid


def _values(cost: dict) -> dict:
    return {"unique_non_embedding_parameters": cost["parameters"]["unique_non_embedding"],
            "unique_total_parameters": cost["parameters"]["unique_total"],
            "training_flops_per_position": cost["flops"]["training_per_token"],
            "kv_bytes_per_token": cost["kv"]["bytes_per_cached_token"]}


def _residual(point: dict) -> float | None:
    """Largest absolute relative difference of the three budgets; None if any is unknown."""
    values = list(point["relative_differences"].values())
    return None if None in values else max(map(abs, values))


def _nearness(point: dict) -> tuple:
    """Sort key of "nearest": the absolute differences sorted largest first, compared lexicographically (the
    smaller largest residual wins, a tie goes to the smaller next one), then the shape's JSON.
    Only for points whose three costs are known."""
    return (sorted(map(abs, point["relative_differences"].values()), reverse=True),
            json.dumps(point["shape"], sort_keys=True, separators=(",", ":")))


def _differences(comparison: dict) -> dict:
    """The three relative differences of a ``compare_recipes`` result, keyed by atlas field (None if unknown)."""
    groups = {key: comparison[group][key] for group in ("matched", "measured-different", "unknown")
              for key in METRICS if key in comparison[group]}
    return {METRICS[key]: groups[key]["relative_difference"] for key in METRICS}


def build_atlas(reference, *, seq_len: int, grid: dict, tolerance: float = 0.05,
                label: str = DEFAULT_REFERENCE) -> dict:
    """Sweep ``grid``; ``matched`` is true iff ``compare_recipes`` matches all three budgets."""
    model = reference.model
    head_dim = model.dim // model.num_heads
    points, summary = [], {}
    reference_cost = estimate_cost(reference, seq_len=seq_len)
    if None in _values(reference_cost).values():
        raise ValueError("the reference needs all three costs known; an unknown reference cost cannot anchor a match")
    for name, structure in grid["structure"].items():
        constructor = getattr(recipes, f"{name}_spec")
        summary[name] = {"points": 0, "matched": 0, "unknown_points": 0}
        sparse = name != "dense"
        axes = {"dim": grid["dims"], "hidden_dim": grid["hidden_dims"],
                **({"num_experts": grid["num_experts"]} if sparse else {}), **structure}
        for combination in itertools.product(*(axes[key] for key in sorted(axes))):
            shape = dict(zip(sorted(axes), combination))
            heads = shape["dim"] // head_dim
            if shape["dim"] % head_dim or heads % model.num_kv_heads or (sparse and model.top_k > shape["num_experts"]):
                continue  # the constructor would reject it
            # The constructors set each recipe's topology; the reference's other geometry carries over.
            spec = constructor(**shape, num_heads=heads, num_kv_heads=model.num_kv_heads, top_k=model.top_k,
                               vocab_size=model.vocab_size, max_seq_len=model.max_seq_len,
                               qk_norm=model.qk_norm, rms_eps=model.rms_eps, rope_theta=model.rope_theta,
                               init_std=model.init_std,
                               **({"always_on_hidden_dim": model.always_on_hidden_dim} if sparse else {}))
            comparison = compare_recipes(reference, spec, seq_len=seq_len, tolerance=tolerance)
            unknown = sorted(METRICS[key] for key in METRICS if key in comparison["unknown"])
            matched = all(key in comparison["matched"] for key in METRICS)
            reason = comparison["costs"]["candidate"]["unknown"]
            points.append({"recipe": name, "shape": shape,
                           **_values(comparison["costs"]["candidate"]),
                           "relative_differences": _differences(comparison),
                           "matched": matched, "unknown": unknown,
                           "unknown_reason": (reason.get("training_depth") or reason.get("training_path")) if unknown else None})
            row = summary[name]
            row["points"] += 1
            row["matched"] += matched
            row["unknown_points"] += bool(unknown)
    matches = {}
    for name in summary:
        rows = [p for p in points if p["recipe"] == name]
        near = min((p for p in rows if not p["matched"] and _residual(p) is not None), key=_nearness, default=None)
        matches[name] = {
            "matched": [{"shape": p["shape"], "max_abs_relative_residual": _residual(p)} for p in rows if p["matched"]],
            "nearest_unmatched": None if near is None else {
                "shape": near["shape"], "max_abs_relative_residual": _residual(near),
                "relative_differences": near["relative_differences"]}}
    return {"schema_version": 1, "kind": "recipe_atlas",
            "reference": {"source": label, "name": reference.name, "seq_len": seq_len, "tolerance": tolerance,
                          "cost_formula_version": reference_cost["formula_version"], **_values(reference_cost)},
            "match_rule": "abs(candidate-reference)/reference <= tolerance on unique non-embedding parameters, "
                          "training FLOPs per position and logical KV (moeloop.comparison.compare_recipes); "
                          "a cost that is unknown never matches. Residual = largest absolute relative difference.",
            "grid": {**grid, "fixed": {"head_dim": head_dim, "num_kv_heads": model.num_kv_heads, "top_k": model.top_k,
                                       "vocab_size": model.vocab_size, "max_seq_len": model.max_seq_len,
                                       "qk_norm": model.qk_norm, "rms_eps": model.rms_eps,
                                       "rope_theta": model.rope_theta, "init_std": model.init_std,
                                       "always_on_hidden_dim": model.always_on_hidden_dim}},
            "summary": summary, "matches": matches, "points": points}


def arm_row(reference, arm, *, source: str, seq_len: int, tolerance: float = 0.05) -> dict:
    """A trained arm (a RecipeSpec off the grid) against the reference, by ``compare_recipes`` unchanged."""
    comparison = compare_recipes(reference, arm, seq_len=seq_len, tolerance=tolerance)
    model = arm.model
    shape = {"dim": model.dim, "hidden_dim": model.hidden_dim, "num_experts": model.num_experts,
             "head_dim": model.dim // model.num_heads, "num_kv_heads": model.num_kv_heads, "top_k": model.top_k,
             "prefix_cells": model.prefix_cells, "num_cells": model.num_cells, "num_visits": model.num_visits,
             "suffix_cells": model.suffix_cells}
    return {"recipe": arm.name, "source": source, "shape": shape, "relative_differences": _differences(comparison),
            "matched": all(key in comparison["matched"] for key in METRICS)}


def budget_table(atlas: dict, arms: list[dict] = ()) -> dict:
    """The docs table: per recipe, the nearest shape on the grid (a matching one if there is any) and its three
    relative differences to the reference. ``nearest`` is ``None`` if no shape has all three costs known.
    ``arms`` are ``arm_row`` results; each is listed under the atlas recipe its name ends with (``p1_smelt``: smelt),
    and an arm whose name ends with none is a ValueError."""
    rows = []
    for name, count in atlas["summary"].items():
        known = [p for p in atlas["points"] if p["recipe"] == name and _residual(p) is not None]
        near = min(known, key=_nearness, default=None)
        rows.append({"recipe": name, **count,
                     "nearest": None if near is None else {"shape": near["shape"], "matched": near["matched"],
                                                           "relative_differences": near["relative_differences"]}})
    placed = []
    for arm in arms:
        after = next((name for name in atlas["summary"] if arm["recipe"].endswith(f"_{name}")), None)
        if after is None:
            raise ValueError(f"--arm {arm['source']}: recipe name {arm['recipe']!r} does not end with _<recipe> for an "
                             f"atlas recipe: {', '.join(f'_{name}' for name in atlas['summary'])} "
                             "(e.g. p1_smelt is listed after smelt)")
        placed.append({**arm, "after": after})
    reference = atlas["reference"]
    return {"reference": {key: reference[key] for key in ("source", "name", "seq_len", "tolerance", "cost_formula_version",
                                                          *METRICS.values())},
            "grid": {**{key: atlas["grid"][key] for key in ("dims", "hidden_dims", "num_experts")},
                     **{key: atlas["grid"]["fixed"][key] for key in ("head_dim", "num_kv_heads", "top_k")}},
            "rows": rows, "arms": placed}


def table_text(table: dict) -> str:
    return json.dumps(table, indent=1, sort_keys=True, allow_nan=False) + "\n"


CSV_COLUMNS = ("recipe", "shape", "dim", "hidden_dim", "num_experts", *METRICS.values(), "unique_total_parameters",
               "matched", "unknown", *(f"relative_difference_{field}" for field in METRICS.values()))


def write_csv(atlas: dict, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for p in atlas["points"]:
            cells = [p["recipe"], json.dumps(p["shape"], sort_keys=True, separators=(",", ":")),
                     p["shape"]["dim"], p["shape"]["hidden_dim"], p["shape"].get("num_experts"),
                     *(p[field] for field in METRICS.values()), p["unique_total_parameters"],
                     str(p["matched"]).lower(), ";".join(p["unknown"]),
                     *(p["relative_differences"][field] for field in METRICS.values())]
            writer.writerow("" if cell is None else cell for cell in cells)  # unknown is empty, never 0


# Eight validated categorical hues (dataviz reference palette) plus neutral gray; the tenth and eleventh
# recipe reuse the first two. Every recipe has its own marker shape, so identity is never colour alone.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#8a8a85")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h", "p", "<", ">")


def write_svg(atlas: dict, path: Path) -> bool:
    """Scatter of FLOPs/position against unique parameters; False if matplotlib is missing.

    Every grid point is a faint dot. Per recipe the matched shapes are solid and outlined, and the nearest
    unmatched shape is hollow; the best shape of each recipe is labelled (with its residual if unmatched).
    """
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update({"svg.hashsalt": "moeloop-atlas", "svg.fonttype": "none"})
    ref, tol = atlas["reference"], atlas["reference"]["tolerance"]
    x_name, y_name = "unique_non_embedding_parameters", "training_flops_per_position"
    ink, muted = "#0b0b0b", "#52514e"
    index = {(p["recipe"], json.dumps(p["shape"], sort_keys=True)): p for p in atlas["points"]}
    kv_max = max([p["kv_bytes_per_token"] for p in atlas["points"]] + [ref["kv_bytes_per_token"]])
    area = lambda p: 30 + 170 * p["kv_bytes_per_token"] / kv_max
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.set_facecolor("white")
    ax.axvspan(ref[x_name] * (1 - tol), ref[x_name] * (1 + tol), color="#2a78d6", alpha=0.10, lw=0)
    ax.axhspan(ref[y_name] * (1 - tol), ref[y_name] * (1 + tol), color="#2a78d6", alpha=0.10, lw=0)
    drawn_unknown, handles = 0, []
    for number, name in enumerate(atlas["summary"]):
        rows = [p for p in atlas["points"] if p["recipe"] == name]
        known = [p for p in rows if p[y_name] is not None]
        drawn_unknown += len(rows) - len(known)
        color, marker = COLORS[number % len(COLORS)], MARKERS[number]
        ax.scatter([p[x_name] for p in known], [p[y_name] for p in known], s=10, color=color, marker=marker,
                   alpha=0.07, linewidths=0, zorder=1)
        info = atlas["matches"][name]
        chosen = [(index[(name, json.dumps(m["shape"], sort_keys=True))], True) for m in info["matched"]]
        if info["nearest_unmatched"]:
            chosen.append((index[(name, json.dumps(info["nearest_unmatched"]["shape"], sort_keys=True))], False))
        for p, matched in chosen:
            ax.scatter([p[x_name]], [p[y_name]], s=area(p), marker=marker, zorder=3, linewidths=1.5,
                       **({"color": color, "edgecolors": ink} if matched else {"facecolors": "none", "edgecolors": color}))
        if chosen:
            p, matched = chosen[0]
            residual = _residual(p)
            angle = 2 * math.pi * number / len(MARKERS)
            dx, dy = math.cos(angle) * 80, math.sin(angle) * 80
            ax.annotate(name if matched else f"{name} (nearest, {residual:.0%})", (p[x_name], p[y_name]),
                        xytext=(dx, dy), textcoords="offset points", ha="left" if dx >= 0 else "right",
                        va="center", fontsize=8, color=ink, zorder=4,
                        arrowprops={"arrowstyle": "-", "color": muted, "lw": 0.6, "shrinkA": 0, "shrinkB": 3})
        handles.append(Line2D([], [], color=color, marker=marker, ls="",
                              label=name if known else f"{name} (FLOPs unknown, not drawn)"))
    ax.scatter([ref[x_name]], [ref[y_name]], s=220, marker="*", color=ink, zorder=5)
    handles.append(Line2D([], [], color=ink, marker="*", ls="", markersize=12, label=f"reference ({ref['name']})"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Unique non-embedding parameters", color=muted)
    ax.set_ylabel("Training matrix FLOPs per position", color=muted)
    ax.set_title(f"Recipe atlas: shaded = reference ±{tol:.0%}; solid = matches all three budgets, hollow = nearest miss;\n"
                 f"faint = grid; marker area = logical KV bytes per token ({drawn_unknown} points with unknown FLOPs not drawn)",
                 color=ink, fontsize=9, loc="left")
    ax.grid(color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)
    return True


def write_outputs(atlas: dict, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "atlas.json").write_text(json.dumps(atlas, indent=1, sort_keys=True, allow_nan=False) + "\n")
    write_csv(atlas, output / "atlas.csv")
    written = ["atlas.json", "atlas.csv"]
    if write_svg(atlas, output / "atlas.svg"):
        written.append("atlas.svg")
    else:
        print("matplotlib is not installed (report extra); skipping atlas.svg", file=sys.stderr)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="directory for atlas.json/csv/svg")
    parser.add_argument("--arm", action="append", default=[], metavar="SPEC.json",
                        help="a trained arm (RecipeSpec JSON, repeatable) listed in the --table beside its recipe, "
                             "e.g. recipes/p1_smelt.json; its name must end with an atlas recipe name")
    parser.add_argument("--table", type=Path, metavar="FILE",
                        help="the docs table (budget_table) as JSON: written to FILE, or with --check compared to it")
    parser.add_argument("--check", action="store_true",
                        help="with --table: write nothing, exit 1 if FILE differs from a fresh run")
    parser.add_argument("--reference", help=f"RecipeSpec JSON of the track baseline (default {DEFAULT_REFERENCE})")
    parser.add_argument("--seq-len", type=int, help="default: the reference run.seq_len")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2",
                        help="override an axis (repeatable): dims, hidden_dims, num_experts "
                             "(absolute values) or recipes=moe,smelt")
    args = parser.parse_args()
    if args.output is None and args.table is None:
        parser.error("give --output, --table or both")
    if args.check and args.table is None:
        parser.error("--check needs --table")
    path = Path(args.reference) if args.reference else ROOT / DEFAULT_REFERENCE
    reference = RecipeSpec.from_dict(json.loads(path.read_text()))
    seq_len = args.seq_len if args.seq_len is not None else reference.run.get("seq_len")
    if seq_len is None:
        parser.error("--seq-len is required: the reference JSON has no run.seq_len")
    grid = override_grid(default_grid(reference), args.grid)
    atlas = build_atlas(reference, seq_len=seq_len, grid=grid, tolerance=args.tolerance,
                        label=args.reference or DEFAULT_REFERENCE)
    if args.output is not None:
        for name in write_outputs(atlas, args.output):
            print(args.output / name)
    for name, row in atlas["summary"].items():
        print(f"{name}: {row['matched']}/{row['points']} matched, {row['unknown_points']} with unknown cost")
    if args.table is not None:
        arms = [arm_row(reference, RecipeSpec.from_dict(json.loads(Path(path).read_text())), source=path,
                        seq_len=seq_len, tolerance=args.tolerance) for path in args.arm]
        try:
            text = table_text(budget_table(atlas, arms))
        except ValueError as error:  # an --arm the table cannot place
            parser.error(str(error))
        if not args.check:
            args.table.write_text(text)
            print(args.table)
        elif not args.table.exists() or args.table.read_text() != text:
            sys.exit(f"{args.table} differs from a fresh run; regenerate it without --check")
        else:
            print(f"{args.table} matches a fresh run")


if __name__ == "__main__":
    main()
