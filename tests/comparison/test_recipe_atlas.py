"""The atlas is the estimator and the comparison contract swept over a grid; it never invents costs."""
import csv
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

atlas = importlib.import_module("recipe_atlas")
ROOT = Path(__file__).resolve().parents[2]
SEQ = 256


def load(name):
    return atlas.RecipeSpec.from_dict(json.loads((ROOT / "recipes" / f"{name}.json").read_text()))


def reference():
    return atlas.recipes.moe_spec(layers=4, dim=128, hidden_dim=256, num_heads=4, num_kv_heads=2,
                                  num_experts=8, top_k=2, max_seq_len=2048)


def build(structure, *, tolerance=0.05, ref=None, **axes):
    grid = {"dims": (128,), "hidden_dims": (256,), "num_experts": (8,), "structure": structure} | axes
    return atlas.build_atlas(ref or reference(), seq_len=SEQ, grid=grid, tolerance=tolerance)


def rebuild(point):
    """The spec a point describes, built independently of build_atlas."""
    return getattr(atlas.recipes, f"{point['recipe']}_spec")(
        **point["shape"], num_heads=point["shape"]["dim"] // 32, num_kv_heads=2, top_k=2,
        vocab_size=50257, max_seq_len=2048)


def test_points_equal_estimate_cost():
    structure = {"moe": {"layers": (2, 4)}, "smelt": {"physical_layers": (4,)},
                 "loom": {"core_layers": (2,), "passes": (3,)},
                 "sandwich": {"core_layers": (2,), "passes": (3,)}, "relaxed": {"core_layers": (2,), "passes": (3,)}}
    result = build(structure, hidden_dims=(192, 256))
    assert len(result["points"]) == 2 * 2 + 2 + 2 + 2 + 2
    for point in result["points"]:
        cost = atlas.estimate_cost(rebuild(point), seq_len=SEQ)
        assert point["unique_non_embedding_parameters"] == cost["parameters"]["unique_non_embedding"]
        assert point["unique_total_parameters"] == cost["parameters"]["unique_total"]
        assert point["training_flops_per_position"] == cost["flops"]["training_per_token"]
        assert point["kv_bytes_per_token"] == cost["kv"]["bytes_per_cached_token"]
    expected = atlas.estimate_cost(reference(), seq_len=SEQ)
    assert result["reference"]["training_flops_per_position"] == expected["flops"]["training_per_token"]
    assert result["reference"]["cost_formula_version"] == expected["formula_version"]


def test_candidates_inherit_the_reference_geometry(monkeypatch):
    base = reference()
    ref = replace(base, model=replace(base.model, rope_theta=500000.0, qk_norm=True, rms_eps=1e-5,
                                      init_std=0.03, always_on_hidden_dim=64))
    built = []
    for name in ("moe", "dense"):
        constructor = getattr(atlas.recipes, f"{name}_spec")
        monkeypatch.setattr(atlas.recipes, f"{name}_spec",
                            lambda *a, _c=constructor, **k: built.append(_c(*a, **k)) or built[-1])
    result = build({"moe": {"layers": (4,)}, "dense": {"layers": (4,)}}, ref=ref)
    moe, dense = built
    for model in (moe.model, dense.model):
        assert (model.rope_theta, model.qk_norm, model.rms_eps, model.init_std) == (500000.0, True, 1e-5, 0.03)
    assert moe.model.always_on_hidden_dim == 64 and dense.model.always_on_hidden_dim == 0  # dense has no experts
    default = build({"moe": {"layers": (4,)}})["points"][0]  # qk_norm and always-on cost parameters
    assert result["points"][0]["unique_non_embedding_parameters"] > default["unique_non_embedding_parameters"]
    assert result["grid"]["fixed"]["rope_theta"] == 500000.0


def test_match_rule_on_fixture():
    result = build({"moe": {"layers": (3, 4)}}, hidden_dims=(160, 224, 240, 256, 288, 384))
    base = atlas.estimate_cost(reference(), seq_len=SEQ)
    base = (base["parameters"]["unique_non_embedding"], base["flops"]["training_per_token"],
            base["kv"]["bytes_per_cached_token"])
    oracle = set()
    for point in result["points"]:
        got = (point["unique_non_embedding_parameters"], point["training_flops_per_position"],
               point["kv_bytes_per_token"])
        differences = [(value - ref) / ref for value, ref in zip(got, base)]
        assert list(point["relative_differences"].values()) == pytest.approx(differences, abs=1e-12)
        if all(abs(d) <= 0.05 for d in differences):
            oracle.add((point["shape"]["layers"], point["shape"]["hidden_dim"]))
    found = {(p["shape"]["layers"], p["shape"]["hidden_dim"]) for p in result["points"] if p["matched"]}
    assert found == oracle
    assert (4, 256) in found  # the reference itself
    assert 0 < len(found) < len(result["points"])  # the rule both accepts and rejects
    assert result["summary"]["moe"] == {"points": len(result["points"]), "matched": len(found), "unknown_points": 0}


def test_tolerance_boundary_is_inclusive():
    structure = {"moe": {"layers": (4,)}}
    point = build(structure, hidden_dims=(240,))["points"][0]
    worst = max(abs(d) for d in point["relative_differences"].values())
    assert 0.05 < worst < 0.5  # outside the default tolerance, inside the one used below
    assert not point["matched"]
    assert build(structure, hidden_dims=(240,), tolerance=worst)["points"][0]["matched"]
    assert not build(structure, hidden_dims=(240,), tolerance=worst * 0.999)["points"][0]["matched"]


def residual(recipe, **shape):
    """Largest absolute relative budget difference to the reference, from estimate_cost alone."""
    base, got = (atlas.estimate_cost(spec, seq_len=SEQ) for spec in (reference(), rebuild({"recipe": recipe, "shape": shape})))
    pairs = [(got["parameters"]["unique_non_embedding"], base["parameters"]["unique_non_embedding"]),
             (got["flops"]["training_per_token"], base["flops"]["training_per_token"]),
             (got["kv"]["bytes_per_cached_token"], base["kv"]["bytes_per_cached_token"])]
    return max(abs(value / ref - 1) for value, ref in pairs)


def test_matches_section_lists_matched_and_nearest_unmatched():
    result = build({"moe": {"layers": (3, 4)}, "huginn": {"core_layers": (2,), "eval_passes": (4,)}},
                   hidden_dims=(160, 240, 256))
    # Only the reference's own shape matches (residual 0); the nearest miss changes just the FFN width,
    # so its residual is the parameter difference (-5.87%), above the 5% tolerance.
    own = {"dim": 128, "hidden_dim": 256, "layers": 4, "num_experts": 8}
    near = {"dim": 128, "hidden_dim": 240, "layers": 4, "num_experts": 8}
    moe = result["matches"]["moe"]
    assert moe["matched"] == [{"shape": own, "max_abs_relative_residual": 0.0}]
    assert moe["nearest_unmatched"]["shape"] == near
    assert moe["nearest_unmatched"]["max_abs_relative_residual"] == pytest.approx(residual("moe", **near), abs=1e-12)
    assert moe["nearest_unmatched"]["max_abs_relative_residual"] == pytest.approx(0.0587313, abs=1e-6)
    assert moe["nearest_unmatched"]["relative_differences"]["training_flops_per_position"] == pytest.approx(-0.0063982, abs=1e-6)
    # Sampled depth has an expected-FLOPs residual; every huginn shape has 2x4 visits against the reference's 4
    # layers, so KV is +150% for all three and the nearest miss is the one whose next largest residual is smallest:
    # hidden_dim 240 (FLOPs +12.95%), not 256 (+14.10%) or 160 (parameters -34.26%).
    huginn = result["matches"]["huginn"]
    assert huginn["matched"] == []
    assert huginn["nearest_unmatched"]["shape"]["hidden_dim"] == 240
    assert huginn["nearest_unmatched"]["relative_differences"]["training_flops_per_position"] == pytest.approx(0.1295117, abs=1e-6)
    assert huginn["nearest_unmatched"]["max_abs_relative_residual"] == pytest.approx(1.5, abs=1e-12)
    assert huginn["nearest_unmatched"]["max_abs_relative_residual"] == pytest.approx(
        residual("huginn", **huginn["nearest_unmatched"]["shape"]), abs=1e-12)


def test_nearest_unmatched_breaks_a_tie_on_the_largest_residual_by_the_next_largest():
    # P1 reference, SMELT with 6 physical layers: KV is -10% for every shape. 480/128 and 616/96 both have the
    # largest residual 10%; the other two shapes are further away (parameters -20.9% and +34.8%).
    p1 = atlas.RecipeSpec.from_dict(json.loads((ROOT / "recipes" / "p1_moe.json").read_text()))
    grid = {"dims": (672,), "hidden_dims": (480, 616), "num_experts": (96, 128),
            "structure": {"smelt": {"physical_layers": (6,)}}}
    result = atlas.build_atlas(p1, seq_len=p1.run["seq_len"], grid=grid)
    residuals = {(p["shape"]["hidden_dim"], p["shape"]["num_experts"]):
                 sorted((abs(d) for d in p["relative_differences"].values()), reverse=True) for p in result["points"]}
    assert residuals[(480, 96)][0] == pytest.approx(0.2086335, abs=1e-6)
    assert residuals[(616, 128)][0] == pytest.approx(0.3477224, abs=1e-6)
    assert residuals[(480, 128)] == pytest.approx([0.1, 0.0949522, 0.0522545], abs=1e-6)
    assert residuals[(616, 96)] == pytest.approx([0.1, 0.0184349, 0.0129675], abs=1e-6)
    # The shape string would pick hidden_dim 480 first; the next-largest residual picks 616/96.
    nearest = result["matches"]["smelt"]["nearest_unmatched"]
    assert nearest["shape"] == {"dim": 672, "hidden_dim": 616, "num_experts": 96, "physical_layers": 6}
    assert nearest["max_abs_relative_residual"] == 0.1


def test_nearest_unmatched_falls_back_to_the_shape_when_all_residuals_tie():
    # P2 reference: Huginn at 6 and at 4 eval passes has the same parameters and FLOPs (sampled depth is the same
    # expectation) and a KV difference of the same size with the opposite sign (-1/6 and +1/6), so the two
    # shapes have identical absolute residuals and only the shape's JSON can order them.
    p2 = load("p2_moe")
    grid = {"dims": (1664,), "hidden_dims": (432,), "num_experts": (128,),
            "structure": {"huginn": {"core_layers": (2,), "eval_passes": (6, 4)}}}
    result = atlas.build_atlas(p2, seq_len=p2.run["seq_len"], grid=grid)
    differences = {p["shape"]["eval_passes"]: [abs(d) for d in p["relative_differences"].values()]
                   for p in result["points"]}
    assert differences[4] == differences[6]
    assert result["matches"]["huginn"]["nearest_unmatched"]["shape"]["eval_passes"] == 4


def test_sampled_depth_cost_is_an_expected_flops_residual(tmp_path):
    result = build({"huginn": {"core_layers": (2,), "eval_passes": (4,)}}, tolerance=0.99)
    point = result["points"][0]
    assert point["training_flops_per_position"] > 0
    assert point["relative_differences"]["training_flops_per_position"] is not None
    assert point["unknown"] == []
    assert point["unknown_reason"] is None  # the cost is known: the expectation note is not an unknown
    assert point["matched"] is False  # KV differs by 150%, beyond even a 99% tolerance
    assert point["unique_non_embedding_parameters"] > 0 and point["kv_bytes_per_token"] > 0
    assert result["summary"]["huginn"]["unknown_points"] == 0
    atlas.write_outputs(result, tmp_path)
    assert json.loads((tmp_path / "atlas.json").read_text())["points"][0]["training_flops_per_position"] == point["training_flops_per_position"]
    row = next(csv.DictReader((tmp_path / "atlas.csv").open()))
    assert float(row["training_flops_per_position"]) == point["training_flops_per_position"]
    assert float(row["relative_difference_training_flops_per_position"]) == point["relative_differences"]["training_flops_per_position"]
    assert row["unknown"] == "" and row["matched"] == "false"


def test_unknown_reason_is_filled_only_for_a_point_with_an_unknown_cost(monkeypatch):
    built, constructor = [], atlas.recipes.ouro_spec

    def other_training_depth(*args, **kwargs):  # a fixed training depth other than the model's: no estimate
        spec = constructor(*args, **kwargs)
        built.append(replace(spec, training=replace(spec.training, num_visits=spec.training.num_visits + 1)))
        return built[-1]

    monkeypatch.setattr(atlas.recipes, "ouro_spec", other_training_depth)
    result = build({"moe": {"layers": (4,)}, "ouro": {"core_layers": (2,), "passes": (2,)}})
    moe, ouro = result["points"]
    assert moe["unknown"] == [] and moe["unknown_reason"] is None
    assert ouro["unknown"] == ["training_flops_per_position"]
    assert ouro["training_flops_per_position"] is None and ouro["relative_differences"]["training_flops_per_position"] is None
    assert ouro["unknown_reason"] == atlas.estimate_cost(built[0], seq_len=SEQ)["unknown"]["training_path"]
    assert ouro["matched"] is False
    assert result["summary"]["ouro"] == {"points": 1, "matched": 0, "unknown_points": 1}
    assert result["matches"]["ouro"]["nearest_unmatched"] is None  # an unknown residual is never ranked


def test_output_is_deterministic(tmp_path):
    structure = {"moe": {"layers": (2, 4)}, "huginn": {"core_layers": (2,), "eval_passes": (4,)},
                 "foil": {"core_layers": (2,), "passes": (1,)}}
    names = []
    for run in ("a", "b"):
        result = build(structure, hidden_dims=(192, 256), num_experts=(4, 8))
        names = atlas.write_outputs(result, tmp_path / run)
    assert {"atlas.json", "atlas.csv"} <= set(names)
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    order = list(structure)
    keys = [(order.index(p["recipe"]), sorted(p["shape"].items())) for p in result["points"]]
    assert keys == sorted(keys)  # recipes in declared order, shapes ascending
    text = (tmp_path / "a" / "atlas.json").read_text()
    assert text.endswith("}\n") and "NaN" not in text and "Infinity" not in text


def test_svg_is_drawn_when_matplotlib_is_installed(tmp_path):
    pytest.importorskip("matplotlib")
    result = build({"moe": {"layers": (2, 4)}, "huginn": {"core_layers": (2,), "eval_passes": (4,)}})
    assert "atlas.svg" in atlas.write_outputs(result, tmp_path)
    assert "<svg" in (tmp_path / "atlas.svg").read_text()


def test_svg_is_skipped_with_a_message_without_matplotlib(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # makes `import matplotlib` raise ImportError
    names = atlas.write_outputs(build({"moe": {"layers": (2, 4)}}), tmp_path)
    assert names == ["atlas.json", "atlas.csv"]
    assert not (tmp_path / "atlas.svg").exists()
    assert "matplotlib is not installed" in capsys.readouterr().err


# name, recipe, the reference's own shape, and the huginn result on the same grid: (matched shape or None,
# residual of the matched or nearest shape). Huginn's FLOPs are expectations over its sampled depth.
TRACKS = [
    ("p1_moe", "moe", {"dim": 672, "hidden_dim": 544, "layers": 10, "num_experts": 64},
     {"core_layers": 4, "dim": 576, "eval_passes": 2, "hidden_dim": 544, "num_experts": 128}, True, 0.0211128),
    ("p1_smelt", "smelt", {"dim": 576, "hidden_dim": 440, "num_experts": 96, "physical_layers": 10},
     {"core_layers": 2, "dim": 704, "eval_passes": 6, "hidden_dim": 440, "num_experts": 192}, False, 0.1843451),
    ("p2_moe", "moe", {"dim": 1280, "hidden_dim": 432, "layers": 12, "num_experts": 64},
     {"core_layers": 4, "dim": 1280, "eval_passes": 2, "hidden_dim": 432, "num_experts": 128}, False, 1 / 6),
]


@pytest.mark.parametrize("name, recipe, own, shape, huginn_matches, huginn_residual", TRACKS)
def test_track_references_at_p1_p2_scale_match_themselves(name, recipe, own, shape, huginn_matches, huginn_residual):
    ref = load(name)
    grid = atlas.default_grid(ref)
    assert ref.model.dim in grid["dims"] and ref.model.hidden_dim in grid["hidden_dims"]
    assert ref.model.num_experts in grid["num_experts"]
    grid = atlas.override_grid(grid, [f"hidden_dims={ref.model.hidden_dim}"])  # keep the test quick
    result = atlas.build_atlas(ref, seq_len=ref.run["seq_len"], grid=grid)
    assert set(result["summary"]) == set(grid["structure"])
    assert result["matches"][recipe]["matched"][0] == {"shape": own, "max_abs_relative_residual": 0.0}
    assert result["summary"]["huginn"]["unknown_points"] == 0
    huginn = result["matches"]["huginn"]
    found = huginn["matched"][0] if huginn_matches else huginn["nearest_unmatched"]
    assert [m["shape"] for m in huginn["matched"]] == ([shape] if huginn_matches else [])
    assert found["shape"] == shape
    assert found["max_abs_relative_residual"] == pytest.approx(huginn_residual, abs=1e-6)
    point = next(p for p in result["points"] if p["recipe"] == "huginn" and p["shape"] == shape)
    assert max(map(abs, point["relative_differences"].values())) == found["max_abs_relative_residual"]
    assert point["matched"] is huginn_matches
    assert result["reference"]["unique_non_embedding_parameters"] > 5e8  # P1/P2 scale, not the C grid


def test_default_grid_is_relative_to_the_reference_and_overridable():
    grid = atlas.default_grid(load("c_a0_r0"))
    assert grid["dims"] == [384, 448, 512, 576, 640] and grid["num_experts"] == [4, 6, 8, 12, 16]
    assert grid["structure"]["moe"] == {"layers": [4, 6, 8, 10, 12]}
    assert grid["structure"]["smelt"] == {"physical_layers": [4, 6, 8]}
    out = atlas.override_grid(grid, ["dims=512,640", "recipes=moe,loom"])
    assert out["dims"] == [512, 640]
    assert list(out["structure"]) == ["moe", "loom"]
    assert grid["dims"] == [384, 448, 512, 576, 640]  # the input is not modified
    with pytest.raises(ValueError):
        atlas.override_grid(grid, ["nope=1"])


def test_missing_seq_len_is_a_clear_error(tmp_path, monkeypatch, capsys):
    spec = json.loads((ROOT / "recipes/c_a0_r0.json").read_text())
    del spec["run"]["seq_len"]
    path = tmp_path / "no_seq_len.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr("sys.argv", ["recipe_atlas.py", "--output", str(tmp_path / "out"), "--reference", str(path)])
    with pytest.raises(SystemExit) as stop:
        atlas.main()
    assert stop.value.code == 2
    assert "--seq-len is required" in capsys.readouterr().err


def test_cli_default_reference_is_the_c_grid_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["recipe_atlas.py", "--output", str(tmp_path), "--grid", "hidden_dims=1024"])
    atlas.main()
    result = json.loads((tmp_path / "atlas.json").read_text())
    expected = atlas.estimate_cost(load("c_a0_r0"), seq_len=2048)
    assert result["reference"]["source"] == "recipes/c_a0_r0.json"
    assert result["reference"]["unique_non_embedding_parameters"] == expected["parameters"]["unique_non_embedding"]
    assert set(result["summary"]) == set(atlas.default_grid(load("c_a0_r0"))["structure"])
    assert result["matches"]["loom"]["matched"]  # the C-grid baseline's own looped shape


NAMED = {"moe", "dense", "smelt", "meta", "foil", "expert_tying", "huginn", "loom", "ouro", "sandwich", "relaxed"}


def test_the_default_grid_covers_every_named_recipe():
    assert set(atlas.default_grid(load("p1_moe"))["structure"]) == NAMED
    for name in NAMED:
        assert callable(getattr(atlas.recipes, f"{name}_spec"))


def test_budget_table_is_the_nearest_shape_per_recipe_with_its_three_differences():
    result = build({"moe": {"layers": (3, 4)}, "huginn": {"core_layers": (2,), "eval_passes": (4,)}},
                   hidden_dims=(160, 240, 256))
    table = atlas.budget_table(result)
    assert table["reference"]["tolerance"] == 0.05 and table["grid"]["hidden_dims"] == (160, 240, 256)
    assert (table["grid"]["head_dim"], table["grid"]["num_kv_heads"], table["grid"]["top_k"]) == (32, 2, 2)  # fixed axes
    assert table["arms"] == []
    moe, huginn = table["rows"]
    # A recipe with a matching shape shows it: the reference's own shape, all three differences zero.
    assert moe["nearest"] == {"shape": {"dim": 128, "hidden_dim": 256, "layers": 4, "num_experts": 8}, "matched": True,
                              "relative_differences": dict.fromkeys(atlas.METRICS.values(), 0.0)}
    assert (moe["points"], moe["matched"]) == (6, 1)
    # A recipe without one shows its nearest miss, the same shape as the matches section names.
    assert huginn["matched"] == 0 and huginn["nearest"]["matched"] is False
    assert huginn["nearest"]["shape"] == result["matches"]["huginn"]["nearest_unmatched"]["shape"]
    assert huginn["nearest"]["relative_differences"] == result["matches"]["huginn"]["nearest_unmatched"]["relative_differences"]


def test_budget_table_has_no_nearest_shape_where_every_cost_is_unknown(monkeypatch):
    constructor = atlas.recipes.ouro_spec
    monkeypatch.setattr(atlas.recipes, "ouro_spec", lambda *a, **k: _other_training_depth(constructor(*a, **k)))
    table = atlas.budget_table(build({"ouro": {"core_layers": (2,), "passes": (2,)}}))
    assert table["rows"][0]["nearest"] is None and table["rows"][0]["points"] == 1


def _other_training_depth(spec):
    return replace(spec, training=replace(spec.training, num_visits=spec.training.num_visits + 1))


def test_table_check_compares_with_a_fresh_run(tmp_path, monkeypatch, capsys):
    path = tmp_path / "table.json"
    argv = ["recipe_atlas.py", "--reference", str(ROOT / "recipes/c_a0_r0.json"), "--table", str(path),
            "--grid", "dims=512", "--grid", "hidden_dims=1024", "--grid", "num_experts=8", "--grid", "recipes=moe,loom"]
    monkeypatch.setattr("sys.argv", argv + ["--check"])
    with pytest.raises(SystemExit) as missing:
        atlas.main()
    assert "differs from a fresh run" in str(missing.value) and not path.exists()
    monkeypatch.setattr("sys.argv", argv)
    atlas.main()
    assert [row["recipe"] for row in json.loads(path.read_text())["rows"]] == ["moe", "loom"]
    monkeypatch.setattr("sys.argv", argv + ["--check"])
    atlas.main()
    assert "matches a fresh run" in capsys.readouterr().out
    path.write_text(path.read_text().replace('"points"', '"pointz"'))  # a stale table
    with pytest.raises(SystemExit):
        atlas.main()


@pytest.fixture(scope="module")
def p1_atlas():
    p1 = load("p1_moe")
    return atlas.build_atlas(p1, seq_len=p1.run["seq_len"], grid=atlas.default_grid(p1), label="recipes/p1_moe.json")


def test_a_trained_arm_off_the_grid_is_compared_by_the_contract():
    # P1 SMELT (recipes/p1_smelt.json: head size 64, hidden width 440, layout 2/5x2/3) is within 5% of the P1 MoE on
    # all three budgets, although no grid shape is: the grid fixes the reference's head size 96.
    ref, arm = load("p1_moe"), load("p1_smelt")
    row = atlas.arm_row(ref, arm, source="recipes/p1_smelt.json", seq_len=4096)
    base, got = (atlas.estimate_cost(spec, seq_len=4096) for spec in (ref, arm))
    expected = {"unique_non_embedding_parameters": (got["parameters"]["unique_non_embedding"], base["parameters"]["unique_non_embedding"]),
                "training_flops_per_position": (got["flops"]["training_per_token"], base["flops"]["training_per_token"]),
                "kv_bytes_per_token": (got["kv"]["bytes_per_cached_token"], base["kv"]["bytes_per_cached_token"])}
    for field, (value, reference_value) in expected.items():
        assert row["relative_differences"][field] == pytest.approx(value / reference_value - 1, abs=1e-12)
    assert row["relative_differences"] == pytest.approx({"unique_non_embedding_parameters": 0.03534, "training_flops_per_position": 0.02273,
                                                         "kv_bytes_per_token": 0.0}, abs=1e-5)
    assert row["matched"] is True and row["source"] == "recipes/p1_smelt.json"
    assert row["shape"] == {"dim": 576, "hidden_dim": 440, "num_experts": 96, "head_dim": 64, "num_kv_heads": 1, "top_k": 8,
                            "prefix_cells": 2, "num_cells": 5, "num_visits": 2, "suffix_cells": 3}
    assert atlas.budget_table(build({"smelt": {"physical_layers": (4,)}}, ref=ref), [row])["arms"][0]["after"] == "smelt"
    assert atlas.arm_row(ref, replace(arm, model=replace(arm.model, hidden_dim=1024)), source="x", seq_len=4096)["matched"] is False


def test_the_docs_table_is_the_p1_atlas(p1_atlas):
    committed = (ROOT / "eval/atlas/recipe-atlas-p1.json").read_text()
    p1 = load("p1_moe")
    arm = atlas.arm_row(p1, load("p1_smelt"), source="recipes/p1_smelt.json", seq_len=p1.run["seq_len"])
    assert committed == atlas.table_text(atlas.budget_table(p1_atlas, [arm])), (
        "regenerate: python scripts/recipe_atlas.py --reference recipes/p1_moe.json --arm recipes/p1_smelt.json "
        "--table eval/atlas/recipe-atlas-p1.json")
    table = json.loads(committed)
    rows = table["rows"]
    assert [row["recipe"] for row in rows] == list(p1_atlas["summary"]) and {row["recipe"] for row in rows} == NAMED
    # The trained P1 SMELT arm is matched, while the grid has no matching smelt shape.
    assert [(a["recipe"], a["after"], a["matched"]) for a in table["arms"]] == [("p1_smelt", "smelt", True)]
    assert next(row for row in rows if row["recipe"] == "smelt")["matched"] == 0
