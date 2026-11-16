"""Bench tracks, the four verdicts with real analytical budgets, and leaderboard arithmetic.

These tests check the declaration rules on in-memory submissions without a folder, so every
report also carries ``bench.NO_FOLDER``; ``declared`` drops it. The folder binding is tested
in test_bench_cli.py on folders written by ``scripts/bench.py run``.
"""
import copy
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from moeloop import bench
from moeloop.prepare import DCLM_TOKENIZER
from moeloop.tokens import byte_table_sha256, token_byte_lengths

ROOT = Path(__file__).resolve().parents[2]
BYTE_TABLE = "b504f8826a5ed6fc6a83ceb55c1c96f6c757ada30c6e38c46995ae2e5b74e17c"


def submission(track, arm="c_a0_r1", **changes):
    recipe = json.loads((ROOT / f"recipes/{arm}.json").read_text())
    recipe["run"] |= {"seq_len": track.seq_len, "global_batch_positions": track.global_batch_positions,
                      "successful_updates": track.horizon_updates,
                      "schedule_total_steps": track.horizon_updates,
                      "schedule": {key: track.schedule[key] for key in
                                   ("name", "coordinate", "total_steps", "warmup_steps",
                                    "decay_ratio", "decay_type", "min_lr_factor")}}
    grid = track.lr_grid["AdamW"]  # three points: the middle one wins; one point: no trials
    trials = [{"lr": lr, "seed": 42, "final_update": track.horizon_updates, "validation_bpb": bpb,
               "log_sha256": "3" * 64} for lr, bpb in zip(grid, (1.03, 1.01, 1.02)) if len(grid) > 1]
    recipe["optimizer"] |= {"base_lr": grid[len(grid) // 2], "max_norm": 1.0}  # C-grid recipes leave max_norm to the trainer default.
    values = {
        "name": arm, "track": track.name, "recipe": recipe,
        "source_commit": "a" * 40, "manifest_fingerprint": track.data["manifest_fingerprint"],
        "tokenizer": dict(track.tokenizer), "evaluation": dict(track.evaluation),
        "seeds": [{"seed": seed, "final_update": track.horizon_updates, "validation_bpb": bpb,
                   "test_bpb": bpb + 0.005, "log_sha256": "1" * 64, "checkpoint_sha256": "2" * 64}
                  for seed, bpb in zip(track.seeds, (1.01, 1.00, 1.02))],
        "lr_trials": trials, "gpu_hours": 2.5, "core": 0.12,
    }
    return bench.Submission.from_dict(values | changes)


def p1_submission(track, name):
    """A frozen P1/P2 recipe, untouched, as a one-seed Track L submission."""
    value = submission(track, arm=name, lr_trials=[], seeds=[
        {"seed": 42, "final_update": 19537, "validation_bpb": 1.0, "test_bpb": 1.0,
         "log_sha256": "1" * 64, "checkpoint_sha256": "2" * 64}])
    return replace(value, recipe=json.loads((ROOT / f"recipes/{name}.json").read_text()))


def declared(report):
    """The reasons about the declaration itself; without a folder the report is never eligible."""
    assert report.verdict != "eligible" and bench.NO_FOLDER in report.reasons["missing-evidence"]
    reasons = {key: tuple(line for line in value if line != bench.NO_FOLDER) for key, value in report.reasons.items()}
    return {key: value for key, value in reasons.items() if value}


@pytest.fixture
def tracks():
    return bench.load_tracks()


def test_verdicts_in_tracks_json_match_the_validator():
    assert tuple(json.loads((ROOT / "eval/bench/tracks.json").read_text())["verdicts"]) == bench.VERDICTS


def test_tracks_freeze_the_p1_data_tokenizer_and_schedule_rules(tracks):
    manifest_text = (ROOT / "docs/evidence/p1-data-manifest.json").read_text()
    manifest = json.loads(manifest_text)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False).encode()).hexdigest()
    assert fingerprint == "e67a956d2fe51dd1784505a6f58d7790f397a09624330700c0ebaffcd344f152"
    assert set(tracks) == {"S", "M", "L", "P2"}
    for track in tracks.values():
        assert track.data["manifest_fingerprint"] == fingerprint
        assert {k: manifest["tokenizer"][k] for k in ("id", "revision", "vocab_size")} == {
            k: track.tokenizer[k] for k in ("id", "revision", "vocab_size")}
        assert track.tokenizer["tokenizer_json_sha256"] == manifest["tokenizer"]["files"]["tokenizer/tokenizer.json"]
        labels = manifest["provenance"]["selection"]["actual_label_tokens"]
        assert (track.evaluation["selection_split"], track.evaluation["ranking_split"]) == ("validation", "test")
        assert track.evaluation["selection_label_tokens"] == labels["validation"] == 10_001_497
        assert track.evaluation["ranking_label_tokens"] == labels["test"] == 10_009_889
        assert track.evaluation["seq_len"] == track.seq_len and track.evaluation["byte_table_sha256"] == BYTE_TABLE
        assert track.budget["tolerance"] == 0.05 and len(track.budget["metrics"]) == 3
        assert track.horizon_updates == -(-track.target_label_tokens // track.global_batch_positions)
        assert track.nominal_positions == track.horizon_updates * track.global_batch_positions
        assert track.global_batch_positions % track.seq_len == 0
        assert track.schedule["total_steps"] == track.horizon_updates
    assert [(t.seq_len, t.horizon_updates, t.seeds, t.required_seed, t.min_seeds) for t in tracks.values()] == [
        (2048, 15259, (42, 43, 44), 42, 1), (4096, 4769, (42, 43, 44), 42, 1), (4096, 19537, (42, 43, 44), 42, 1),
        (4096, 4769, (42, 43, 44), 42, 1)]
    for track in (tracks["S"], tracks["M"], tracks["P2"]):  # WSD: 2% warmup, last 20% linear to 0.
        schedule = track.schedule
        assert schedule["warmup_steps"] == round(0.02 * track.horizon_updates)
        assert (schedule["decay_ratio"], schedule["decay_type"], schedule["min_lr_factor"]) == (0.2, "linear", 0.0)
    for track in (tracks["S"], tracks["M"]):  # LR trials on a 3-point grid, Muon = 2 x AdamW.
        assert (track.lr_selection["trial_updates"], track.lr_selection["trial_seed"]) == (track.horizon_updates, 42)
        assert track.lr_grid == {"AdamW": (1.5e-4, 3e-4, 6e-4), "Muon": (3e-4, 6e-4, 1.2e-3)}
        assert track.lr_grid["Muon"] == tuple(2 * lr for lr in track.lr_grid["AdamW"])
    for name in ("p1_moe", "p1_smelt"):  # Track L is the frozen P1 schedule, key for key.
        frozen = json.loads((ROOT / f"recipes/{name}.json").read_text())["run"]["schedule"]
        assert {key: tracks["L"].schedule[key] for key in frozen} == frozen
    assert (tracks["L"].schedule["warmup_steps"], tracks["L"].schedule["stable_end_label_tokens"],
            tracks["L"].schedule["decay_type"], tracks["L"].schedule["min_lr_factor"]) == (20, 10_485_760_000, "cosine", 0.1)
    assert tracks["S"].token_budget["status"] == "provisional"
    assert tracks["L"].lr_grid == {"AdamW": (3e-4,), "Muon": (6e-4,)}
    assert tracks["P2"].lr_grid == {"AdamW": (1.2e-3,), "Muon": (1.7e-3,)}  # Calibrated in #180.
    for name in ("L", "P2"):
        assert set(tracks[name].lr_selection) == {"rule", "grid_basis"}  # No trials on a single-LR track.


def test_byte_table_hash_from_local_hf_cache(tracks):
    hub = pytest.importorskip("huggingface_hub")
    path = hub.try_to_load_from_cache(DCLM_TOKENIZER["id"], "tokenizer.json")
    if not isinstance(path, str) or hashlib.sha256(Path(path).read_bytes()).hexdigest() != DCLM_TOKENIZER["tokenizer_json_sha256"]:
        pytest.skip("fixed Qwen3 tokenizer.json is not in the local HF cache")
    table = token_byte_lengths(Path(path).parent)
    assert len(table) == tracks["S"].tokenizer["vocab_size"]
    assert byte_table_sha256(table) == BYTE_TABLE


def test_track_s_geometry_is_the_c_grid_baseline(tracks):
    c_grid = json.loads((ROOT / "recipes/c_a0_r0.json").read_text())
    geometry = tracks["S"].reference["geometry"]
    assert geometry == {key: c_grid["model"][key] for key in geometry}
    assert (c_grid["run"]["seq_len"], c_grid["run"]["global_batch_positions"]) == (
        tracks["S"].seq_len, tracks["S"].global_batch_positions)


def test_track_references_resolve_from_recipe_constructors(tracks):
    from moeloop import recipes
    assert tracks["S"].reference_spec() == recipes.sharing_panel_specs()["a0_r0"]
    assert tracks["M"].reference_spec() == tracks["L"].reference_spec() == recipes.p1_moe_spec()
    assert tracks["P2"].reference_spec() == recipes.p2_moe_spec()
    drifted = replace(tracks["S"], reference=tracks["S"].reference | {
        "geometry": tracks["S"].reference["geometry"] | {"dim": 640}})
    with pytest.raises(ValueError, match="drifted"):
        drifted.reference_spec()


def test_eligible_submission_with_real_budgets(tracks):
    track = tracks["S"]
    report = bench.validate(submission(track), track)
    assert declared(report) == {} and report.seed_test_bpb == {}
    params = report.budgets["unique_non_embedding_parameters"]
    assert params["candidate"] - params["reference"] == 6 * 512 * 8  # 2 banks x 3 extra visit routers, d x E
    assert report.budgets["training_flops_per_token"]["relative_difference"] == 0
    assert report.budgets["logical_kv_bytes"]["relative_difference"] == 0


def test_budget_mismatch_reports_the_relative_difference(tracks):
    track = tracks["S"]
    report = bench.validate(submission(track, arm="c_a1_r1"), track)
    assert report.verdict == "budget-mismatch" and report.seed_test_bpb == {}
    assert declared(report) == {"budget-mismatch": (
        "unique_non_embedding_parameters differs by +0.1494 (tolerance 0.05)",)}


def test_unknown_training_flops_are_missing_evidence(tracks):
    track = tracks["S"]
    sampled = submission(track, arm="c_a0_r0")
    sampled.recipe["training"]["sampled_visits"] = [2, 4]
    sampled.recipe["training"]["grad_visits"] = 1
    report = bench.validate(sampled, track)
    assert report.verdict == "missing-evidence"
    assert declared(report) == {"missing-evidence": (
        "training_flops_per_token is unknown: Missing supported cost estimate; unknown is never matched.",)}
    assert report.budgets["training_flops_per_token"]["candidate"] is None


def test_uncostable_recipe_is_protocol_mismatch(tracks):
    track = tracks["M"]
    report = bench.validate(submission(track), track)
    assert report.verdict == "protocol-mismatch"
    assert any("cannot be costed at seq_len 4096" in r for r in report.reasons["protocol-mismatch"])


@pytest.mark.parametrize("change, reason", [
    (lambda s: replace(s, manifest_fingerprint="0" * 64), "manifest_fingerprint differs"),
    (lambda s: replace(s, tokenizer=s.tokenizer | {"revision": "main"}), "tokenizer.revision is 'main'"),
    (lambda s: replace(s, evaluation=s.evaluation | {"ranking_split": "validation"}),
     "evaluation.ranking_split is 'validation', track requires 'test'"),
    (lambda s: replace(s, seeds=(s.seeds[0] | {"final_update": 15000},) + s.seeds[1:]),
     "seed 42: final_update 15000 is not the schedule horizon 15259; only completed schedules count"),
    (lambda s: replace(s, seeds=s.seeds[:2] + (s.seeds[2] | {"seed": 7},)), "seed 7 is not a track seed"),
    (lambda s: replace(s, seeds=s.seeds[1:]), "seed 42 is required"),
    (lambda s: replace(s, recipe=s.recipe | {"optimizer": s.recipe["optimizer"] | {"base_lr": 6e-4}}),
     "optimizer.base_lr 0.0006 is not the selected LR 0.0003"),
    (lambda s: replace(s, recipe=s.recipe | {"optimizer": s.recipe["optimizer"] | {"name": "SGD"}}),
     "no LR grid for optimizer 'SGD'"),
    (lambda s: replace(s, recipe={k: v for k, v in s.recipe.items() if k != "optimizer"}),
     "recipe must state its optimizer and LR"),
    (lambda s: replace(s, recipe=s.recipe | {"optimizer": {}}), "recipe must state its optimizer and LR"),
    (lambda s: replace(s, recipe=s.recipe | {"optimizer": {k: v for k, v in s.recipe["optimizer"].items() if k != "base_lr"}}),
     "recipe must state its optimizer and LR"),
    (lambda s: replace(s, lr_trials=s.lr_trials[:2]), "LR trials lack base grid points [0.0006]"),
    (lambda s: replace(s, lr_trials=s.lr_trials + s.lr_trials[:1]), "LR trial 0.00015 is not a unique positive LR"),
    (lambda s: replace(s, lr_trials=(s.lr_trials[0] | {"final_update": 1000},) + s.lr_trials[1:]),
     "ends at update 1000, not 15259"),
    (lambda s: replace(s, recipe=s.recipe | {"run": s.recipe["run"] | {"mixed_precision_param": "float32"}}),
     "run.mixed_precision_param is 'float32', track requires 'bfloat16'"),
    (lambda s: replace(s, recipe=s.recipe | {"run": s.recipe["run"] | {"schedule_total_steps": 2048}}),
     "run.schedule_total_steps is 2048, track requires 15259"),
    (lambda s: replace(s, recipe=s.recipe | {"run": s.recipe["run"] | {"schedule": s.recipe["run"]["schedule"] | {"decay_type": "cosine"}}}),
     "run.schedule.decay_type is 'cosine', track requires 'linear'"),
    (lambda s: replace(s, track="M"), "submitted for track 'M'"),
])
def test_protocol_mismatches(tracks, change, reason):
    track = tracks["S"]
    report = bench.validate(change(submission(track)), track)
    assert report.verdict == "protocol-mismatch" and report.seed_test_bpb == {}
    assert any(reason in r for r in report.reasons["protocol-mismatch"]), report.reasons


@pytest.mark.parametrize("change, reason", [
    (lambda s: replace(s, seeds=(s.seeds[0] | {"checkpoint_sha256": None},) + s.seeds[1:]),
     "seed 42: checkpoint_sha256 must be a 64-hex sha256"),
    (lambda s: replace(s, seeds=()), "0 seeds reported, track needs at least 1"),
    (lambda s: replace(s, seeds=({k: v for k, v in s.seeds[0].items() if k != "test_bpb"},) + s.seeds[1:]),
     "seed 42: final test_bpb is missing"),
    (lambda s: replace(s, lr_trials=({k: v for k, v in s.lr_trials[0].items() if k != "validation_bpb"},) + s.lr_trials[1:]),
     "LR trial 0.00015: validation_bpb is missing"),
    (lambda s: replace(s, lr_trials=(s.lr_trials[0] | {"log_sha256": None},) + s.lr_trials[1:]),
     "LR trial 0.00015: log_sha256 must be a 64-hex sha256"),
    (lambda s: replace(s, source_commit=None), "source_commit must be a 40-hex git commit"),
    (lambda s: replace(s, evaluation={k: v for k, v in s.evaluation.items() if k != "byte_table_sha256"}),
     "evaluation.byte_table_sha256 is missing"),
    (lambda s: replace(s, lr_trials=()), "LR trials are missing for grid [0.00015, 0.0003, 0.0006]"),
])
def test_missing_evidence(tracks, change, reason):
    track = tracks["S"]
    report = bench.validate(change(submission(track)), track)
    assert report.verdict == "missing-evidence"
    assert reason in report.reasons["missing-evidence"], report.reasons


def test_definite_failures_rank_before_missing_evidence(tracks):
    track = tracks["L"]
    base = p1_submission(track, "p1_moe")
    both = replace(base, manifest_fingerprint="0" * 64, source_commit=None)
    report = bench.validate(both, track)
    assert report.verdict == "protocol-mismatch" and set(report.reasons) == {"protocol-mismatch", "missing-evidence"}


def test_submission_rejects_unknown_fields_and_roundtrips(tracks):
    value = submission(tracks["S"])
    assert bench.Submission.from_dict(copy.deepcopy(value.to_dict())) == value
    with pytest.raises(ValueError, match="unknown submission fields"):
        bench.Submission.from_dict(value.to_dict() | {"bpb": 1.0})
    for name in ("name", "track"):
        with pytest.raises(ValueError, match="required"):
            bench.Submission.from_dict({k: v for k, v in value.to_dict().items() if k != name})


def report(name, values, track="S", verdict="eligible"):
    return bench.Report(submission=name, track=track, verdict=verdict,
                        seed_test_bpb=dict(zip((42, 43, 44, 45), values)) if verdict == "eligible" else {},
                        expert_lr_multiplier=1.0, expert_group_size=1, gpu_hours=2.5, core=0.12)


def test_leaderboard_numbers_on_a_fixed_fixture():
    board = bench.leaderboard([
        report("loop", [1.00, 1.02, 0.98]), report("moe", [0.95, 0.97, 0.96]),
        report("bad", [], verdict="budget-mismatch"), report("solo", [0.93]),
        report("m-arm", [2.0, 2.0], track="M"), report("m-alt", [2.0, 2.0], track="M")])
    rows = [(r["track"], r["rank"], r["submission"], r["seeds"], r["n_seeds"], r["mean_test_bpb"], r["range_test_bpb"])
            for r in board["rows"]]
    assert rows == [("M", 1, "m-alt", [42, 43], 2, 2.0, [2.0, 2.0]),  # Equal means: by name, consecutive ranks.
                    ("M", 2, "m-arm", [42, 43], 2, 2.0, [2.0, 2.0]),
                    ("S", 1, "solo", [42], 1, 0.93, None),  # One seed: no range.
                    ("S", 2, "moe", [42, 43, 44], 3, pytest.approx(0.96), [0.95, 0.97]),
                    ("S", 3, "loop", [42, 43, 44], 3, pytest.approx(1.00), [0.98, 1.02])]
    assert board["excluded"] == [{"submission": "bad", "track": "S", "verdict": "budget-mismatch"}]
    assert board["metric"] == "test_bpb"
    assert all((r["expert_lr_multiplier"], r["expert_group_size"], r["gpu_hours"], r["core"]) == (1.0, 1, 2.5, 0.12)
               for r in board["rows"])
    assert set(board) == {"schema_version", "metric", "lower_is_better", "rows", "excluded"}  # No interval estimate.


def test_lr_selection_reads_validation_bpb_only(tracks):
    track = tracks["S"]
    # Validation selects 3e-4; a test-split argmin (6e-4 here) must not move the selection.
    trials = tuple(t | {"test_bpb": {1.5e-4: 1.2, 3e-4: 1.1, 6e-4: 0.9}[t["lr"]]}
                   for t in submission(track).lr_trials)
    assert declared(bench.validate(submission(track, lr_trials=trials), track)) == {}
    tied = tuple(t | {"validation_bpb": 1.0} for t in trials)
    report = bench.validate(submission(track, lr_trials=tied), track)
    assert declared(report) == {"protocol-mismatch": (  # A tie goes to the smaller LR, here an edge.
        "lowest validation BPB is at the edge LR 0.00015; add the trial at 7.5e-05 and repeat until the minimum is interior",
        "optimizer.base_lr 0.0003 is not the selected LR 0.00015")}


@pytest.mark.parametrize("name, parameters, flops", [("p1_moe", 0.0, 0.0), ("p1_smelt", 0.035344, 0.022732)])
def test_frozen_p1_recipes_are_eligible_on_track_l(tracks, name, parameters, flops):
    track = tracks["L"]
    report = bench.validate(p1_submission(track, name), track)
    assert declared(report) == {}
    relative = {metric: entry["relative_difference"] for metric, entry in report.budgets.items()}
    assert relative == pytest.approx({"unique_non_embedding_parameters": parameters,
                                      "training_flops_per_token": flops, "logical_kv_bytes": 0.0}, abs=1e-6)


def lr_submission(track, optimizer, base_lr, trials):
    """LRs are decimal literals, parsed as JSON would parse them, never computed doubles."""
    value = submission(track, lr_trials=[{"lr": lr, "seed": 42, "final_update": track.horizon_updates,
                                          "validation_bpb": bpb, "log_sha256": "3" * 64} for lr, bpb in trials])
    settings = {"name": optimizer, "base_lr": base_lr}
    if optimizer == "Muon":
        settings["muon"] = track.optimizers["fixed"]["Muon"]["muon"] | {"lr": base_lr}
    return replace(value, recipe=value.recipe | {"optimizer": value.recipe["optimizer"] | settings})


@pytest.mark.parametrize("optimizer, base_lr, trials, reason", [
    ("Muon", 6e-4, [(3e-4, 1.03), (6e-4, 1.01), (1.2e-3, 1.02)], None),
    # The minimum stops at the upper edge: one x2 step is required.
    ("Muon", 1.2e-3, [(3e-4, 1.03), (6e-4, 1.02), (1.2e-3, 1.01)],
     "lowest validation BPB is at the edge LR 0.0012; add the trial at 0.0024 and repeat until the minimum is interior"),
    ("Muon", 1.2e-3, [(3e-4, 1.03), (6e-4, 1.02), (1.2e-3, 1.01), (2.4e-3, 1.04)], None),
    # Two steps: 2.4e-3 is still the edge minimum, 4.8e-3 brackets it.
    ("Muon", 2.4e-3, [(3e-4, 1.03), (6e-4, 1.02), (1.2e-3, 1.01), (2.4e-3, 1.0), (4.8e-3, 1.05)], None),
    ("Muon", 2.4e-3, [(3e-4, 1.03), (6e-4, 1.02), (1.2e-3, 1.01), (4.8e-3, 1.0)],
     "LR trials [0.0048] are not an unbroken x2/÷2 extension of the grid [0.0003, 0.0006, 0.0012]"),
    ("AdamW", 1.5e-4, [(7.5e-5, 1.02), (1.5e-4, 1.0), (3e-4, 1.01), (6e-4, 1.03)], None),
    ("AdamW", 1.5e-4, [(3.75e-5, 1.02), (1.5e-4, 1.0), (3e-4, 1.01), (6e-4, 1.03)],
     "LR trials [3.75e-05] are not an unbroken x2/÷2 extension of the grid [0.00015, 0.0003, 0.0006]"),
    ("AdamW", 4e-4, [(1.5e-4, 1.03), (3e-4, 1.01), (4e-4, 1.0), (6e-4, 1.02)],
     "LR trials [0.0004] are not an unbroken x2/÷2 extension of the grid [0.00015, 0.0003, 0.0006]"),
])
def test_edge_minimum_must_be_bracketed_by_x2_steps(tracks, optimizer, base_lr, trials, reason):
    track = tracks["S"]
    report = bench.validate(lr_submission(track, optimizer, base_lr, trials), track)
    assert declared(report) == ({} if reason is None else {"protocol-mismatch": (reason,)})


def test_single_lr_tracks_take_no_trials(tracks):
    track = tracks["L"]
    value = p1_submission(track, "p1_moe")
    assert declared(bench.validate(value, track)) == {}
    with_trials = replace(value, lr_trials=({"lr": 3e-4, "seed": 42, "final_update": 19537, "validation_bpb": 1.0},))
    assert declared(bench.validate(with_trials, track)) == {"protocol-mismatch": (
        "track L fixes AdamW LR 0.0003; LR trials are not part of it",)}
    muon = replace(value, recipe=value.recipe | {"optimizer": value.recipe["optimizer"] | {
        "name": "Muon", "muon": track.optimizers["fixed"]["Muon"]["muon"] | {"lr": 3e-4}}})
    assert declared(bench.validate(muon, track)) == {"protocol-mismatch": (
        "optimizer.base_lr 0.0003 is not the track LR 0.0006",)}


@pytest.mark.parametrize("change, reason", [
    (lambda o: o | {"betas": [0.9, 0.99]}, "optimizer.betas is [0.9, 0.99], track fixes [0.9, 0.95]"),
    (lambda o: o | {"weight_decay": 0.0}, "optimizer.weight_decay is 0.0, track fixes 0.1"),
    (lambda o: {k: v for k, v in o.items() if k != "max_norm"}, "optimizer.max_norm is None, track fixes 1.0"),
    (lambda o: o | {"zero_weight_decay_roles": []},
     "optimizer.zero_weight_decay_roles is [], track fixes ['norm', 'embedding', 'lm_head', 'router']"),
    (lambda o: {k: v for k, v in o.items() if k != "expert_lr_multiplier"}, "recipe must state optimizer.expert_lr_multiplier"),
    (lambda o: {k: v for k, v in o.items() if k != "expert_group_size"}, "recipe must state optimizer.expert_group_size"),
    (lambda o: o | {"muon": o["muon"] | {"ns_steps": 6}}, "optimizer.muon.ns_steps is 6, track fixes 5"),
    (lambda o: o | {"muon": o["muon"] | {"adjust_lr_fn": None}}, "optimizer.muon.adjust_lr_fn is None, track fixes 'match_rms_adamw'"),
    (lambda o: o | {"muon": o["muon"] | {"lr": 1.2e-3}}, "optimizer.muon.lr 0.0012 must equal optimizer.base_lr 0.0006"),
])
def test_non_lr_optimizer_settings_are_fixed(tracks, change, reason):
    track = tracks["S"]
    value = lr_submission(track, "Muon", 6e-4, [(3e-4, 1.03), (6e-4, 1.01), (1.2e-3, 1.02)])
    value = replace(value, recipe=value.recipe | {"optimizer": change(value.recipe["optimizer"])})
    assert declared(bench.validate(value, track)) == {"protocol-mismatch": (reason,)}


def test_method_level_expert_settings_are_free_but_reported(tracks):
    track = tracks["S"]
    value = submission(track)
    value = replace(value, recipe=value.recipe | {"optimizer": value.recipe["optimizer"] | {
        "expert_lr_multiplier": 0.5, "expert_group_size": 4}})
    report = bench.validate(value, track)
    assert declared(report) == {}
    assert (report.expert_lr_multiplier, report.expert_group_size, report.gpu_hours, report.core) == (0.5, 4, 2.5, 0.12)


def test_muon_settings_are_the_pinned_dist_muon_defaults(tracks):
    pytest.importorskip("torchtitan")
    from torchtitan.distributed.flex_shard.dist_muon import DistMuon
    defaults = {name: p.default for name, p in inspect.signature(DistMuon.__init__).parameters.items()}
    for track in tracks.values():
        muon = track.optimizers["fixed"]["Muon"]["muon"]
        for key in ("momentum", "nesterov", "ns_steps", "eps"):
            assert muon[key] == defaults[key], key
        assert tuple(muon["ns_coefficients"]) == defaults["ns_coefficients"]
        assert muon["weight_decay"] == track.optimizers["fixed"]["AdamW"]["weight_decay"]


@pytest.mark.parametrize("name", ["p1_moe", "p1_smelt"])
def test_p1_recipes_are_eligible_on_track_m_under_its_wsd_rules(tracks, name):
    track = tracks["M"]
    value = submission(track, arm=name)
    assert value.recipe["run"]["schedule"] | {"total_steps": 4769} == {
        "name": "warmup_stable_decay", "coordinate": "successful_optimizer_updates", "total_steps": 4769,
        "warmup_steps": 95, "decay_ratio": 0.2, "decay_type": "linear", "min_lr_factor": 0.0}
    assert declared(bench.validate(value, track)) == {}


@pytest.mark.parametrize("name, differences", [("p2_moe", ("+0.8498", "+1.0491", "+0.6000")),
                                               ("p2_smelt", ("+0.8449", "+1.0990", "+0.6500"))])
def test_p2_recipes_are_budget_mismatch_on_track_l(tracks, name, differences):
    track = tracks["L"]
    report = bench.validate(p1_submission(track, name), track)
    assert declared(report) == {"budget-mismatch": tuple(
        f"{metric} differs by {difference} (tolerance 0.05)" for metric, difference in
        zip(("unique_non_embedding_parameters", "training_flops_per_token", "logical_kv_bytes"), differences))}


@pytest.mark.parametrize("name", ["S", "M", "L", "P2"])
def test_one_seed_42_is_eligible_on_every_track(tracks, name):
    track = tracks[name]
    if name == "L":
        value = p1_submission(track, "p1_moe")
    else:
        value = submission(track, arm={"S": "c_a0_r1", "M": "p1_moe", "P2": "p2_moe"}[name], seeds=[
            {"seed": 42, "final_update": track.horizon_updates, "validation_bpb": 1.0, "test_bpb": 1.0,
             "log_sha256": "1" * 64, "checkpoint_sha256": "2" * 64}])
    assert declared(bench.validate(value, track)) == {}


def test_a_made_up_submission_without_files_is_not_ranked(tracks):
    """Placeholder hashes and an unrecorded commit around a frozen recipe: nothing binds them to a run."""
    track = tracks["P2"]
    value = bench.Submission.from_dict({
        "name": "made-up", "track": "P2", "recipe": json.loads((ROOT / "recipes/phase1/m0-adamw.json").read_text()),
        "source_commit": "0" * 40, "manifest_fingerprint": track.data["manifest_fingerprint"],
        "tokenizer": dict(track.tokenizer), "evaluation": dict(track.evaluation),
        "seeds": [{"seed": 42, "final_update": track.horizon_updates, "validation_bpb": 0.5, "test_bpb": 0.5,
                   "log_sha256": "1" * 64, "checkpoint_sha256": "2" * 64}]})
    report = bench.validate(value, track)
    assert report.reasons == {"missing-evidence": (bench.NO_FOLDER,)}
    assert bench.leaderboard([report])["rows"] == []
    assert any("not checked here" in line for line in report.unchecked)
