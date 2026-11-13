"""Looped MoE Bench: frozen tracks, submission checks and the leaderboard.

Budgets come from ``moeloop.comparison``, so this module needs the model extra
(CPU PyTorch is enough).

A verdict covers only what a submission declares: the three analytical budgets,
data/tokenizer/evaluation identities, schedule completion and LR selection, and
the presence of evidence hashes. It never re-runs training or evaluation, never
opens the hashed files, and never judges quality.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import recipes
from .comparison import compare_recipes
from .recipe_spec import RecipeSpec


TRACKS_PATH = Path(__file__).resolve().parents[2] / "eval" / "bench" / "tracks.json"
VERDICTS = ("eligible", "budget-mismatch", "protocol-mismatch", "missing-evidence")
_SHA1, _SHA256 = re.compile(r"[0-9a-f]{40}"), re.compile(r"[0-9a-f]{64}")
# Schedule keys that moeloop.recipes.trainer_config passes to the LR scheduler.
_SCHEDULE_KEYS = ("total_steps", "warmup_steps", "decay_ratio", "decay_type", "min_lr_factor")
_IDENTITY_KEYS = {"tokenizer": ("id", "revision", "vocab_size", "tokenizer_json_sha256"),
                  "evaluation": ("metric", "complete_split", "seq_len", "selection_split",
                                 "selection_label_tokens", "ranking_split", "ranking_label_tokens",
                                 "byte_table_sha256")}


@dataclass(frozen=True)
class Track:
    """One frozen track of ``eval/bench/tracks.json``; see that file for meanings."""

    name: str
    description: str
    hardware: str
    reference: dict[str, Any]
    seq_len: int
    global_batch_positions: int
    mixed_precision_param: str
    target_label_tokens: int
    horizon_updates: int
    nominal_positions: int
    token_budget: dict[str, Any]
    schedule: dict[str, Any]
    seeds: tuple[int, ...]
    required_seed: int
    min_seeds: int
    seed_rule: str
    lr_grid: dict[str, tuple[float, ...]]
    optimizers: dict[str, Any]
    lr_selection: dict[str, Any]
    data: dict[str, Any]
    tokenizer: dict[str, Any]
    evaluation: dict[str, Any]
    budget: dict[str, Any]
    endpoint: str

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Track:
        value = dict(values)
        value["seeds"] = tuple(value["seeds"])
        value["lr_grid"] = {name: tuple(grid) for name, grid in value["lr_grid"].items()}
        return cls(**value)

    def reference_spec(self) -> RecipeSpec:
        """Resolve the baseline RecipeSpec and check it still has the frozen geometry.

        A drift between ``moeloop.recipes`` and the frozen geometry raises,
        rather than silently moving the budget target.
        """
        spec = getattr(recipes, self.reference["constructor"])()
        if "arm" in self.reference:
            spec = spec[self.reference["arm"]]
        actual = spec.model.to_dict()
        drift = {key: (value, actual[key]) for key, value in self.reference["geometry"].items()
                 if actual[key] != value}
        if drift:
            raise ValueError(f"track {self.name} reference geometry drifted: {drift}")
        return spec


def load_tracks(path: str | Path = TRACKS_PATH) -> dict[str, Track]:
    """Read the frozen tracks. The default path exists in a source checkout only."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported tracks schema version")
    return {name: Track.from_dict(track) for name, track in payload["tracks"].items()}


@dataclass(frozen=True)
class Submission:
    """One result JSON. Absent fields stay None and become missing-evidence.

    ``recipe`` is the RecipeSpec dict shared by every seed (its ``run.seed`` is
    not read); ``config`` is the resolved launch config, kept as evidence.
    ``seeds`` entries carry seed, final_update, validation_bpb, test_bpb,
    log_sha256 and checkpoint_sha256; ``lr_trials`` entries carry lr, seed,
    final_update, validation_bpb and log_sha256. Validation BPB selects the
    LR; test BPB is reported and ranked. ``gpu_hours`` is reported, never an
    eligibility condition, and so is ``core``: the mean over seeds of the
    ``core`` value from ``moeloop.core.evaluate_core``.
    """

    name: str
    track: str
    recipe: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    source_commit: str | None = None
    manifest_fingerprint: str | None = None
    tokenizer: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    seeds: tuple[dict[str, Any], ...] = ()
    lr_trials: tuple[dict[str, Any], ...] = ()
    gpu_hours: float | None = None
    core: float | None = None
    schema_version: int = 1

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Submission:
        unknown = set(values) - {entry.name for entry in fields(cls)}
        if unknown:
            raise ValueError(f"unknown submission fields: {sorted(unknown)}")
        absent = [name for name in ("name", "track") if name not in values]
        if absent:
            raise ValueError(f"submission fields are required: {absent}")
        if values.get("schema_version", 1) != 1:
            raise ValueError("unsupported submission schema version")
        value = dict(values)
        for name in ("seeds", "lr_trials"):
            value[name] = tuple(value.get(name) or ())
            if not all(isinstance(entry, Mapping) for entry in value[name]):
                raise ValueError(f"submission {name} entries must be objects")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["seeds"], value["lr_trials"] = list(self.seeds), list(self.lr_trials)
        return value


@dataclass(frozen=True)
class Report:
    """A verdict with every reason found; ``seed_test_bpb`` feeds the leaderboard."""

    submission: str
    track: str
    verdict: str
    reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    seed_test_bpb: dict[int, float] = field(default_factory=dict)
    expert_lr_multiplier: float | None = None
    expert_group_size: int | None = None
    gpu_hours: float | None = None
    core: float | None = None

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")

    def to_dict(self) -> dict[str, Any]:
        return {"submission": self.submission, "track": self.track, "verdict": self.verdict,
                "reasons": {key: list(value) for key, value in self.reasons.items()},
                "budgets": self.budgets, "seed_test_bpb": {str(k): v for k, v in self.seed_test_bpb.items()},
                "expert_lr_multiplier": self.expert_lr_multiplier, "expert_group_size": self.expert_group_size,
                "gpu_hours": self.gpu_hours, "core": self.core}


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _budgets(track: Track, recipe: Mapping[str, Any], protocol: list[str],
             missing: list[str], budget: list[str]) -> dict[str, Any]:
    """Compare the three analytical budgets against the track baseline."""
    reference = track.reference_spec()  # A drifted track raises; it is not the submission's fault.
    try:
        comparison = compare_recipes(reference, RecipeSpec.from_dict(recipe), seq_len=track.seq_len,
                                     tolerance=track.budget["tolerance"], **track.budget["accounting"])
    except (KeyError, TypeError, ValueError) as error:
        protocol.append(f"recipe cannot be costed at seq_len {track.seq_len}: {error}")
        return {}
    result = {}
    for metric in track.budget["metrics"]:
        group = next(name for name in ("matched", "measured-different", "unknown")
                     if metric in comparison[name])
        entry = comparison[group][metric]
        result[metric] = {key: entry[key] for key in ("reference", "candidate", "relative_difference")}
        if group == "measured-different":
            budget.append(f"{metric} differs by {entry['relative_difference']:+.4f} "
                          f"(tolerance {track.budget['tolerance']})")
        elif group == "unknown":
            missing.append(f"{metric} is unknown: {entry['reason']}")
    return result


def _lr_selection(track: Track, optimizer: Mapping[str, Any], submitted: Sequence[Mapping[str, Any]],
                  protocol: list[str], missing: list[str]) -> None:
    """Check the LR grid, its x2/÷2 edge extensions and the validation argmin.

    Trials must hold the whole base grid. Beyond either edge only the unbroken
    chain grid_max*2, *4, ... or grid_min/2, /4, ... may appear, and the
    selected LR (lowest validation BPB, ties to the smaller LR) needs a trial
    on each side. Halving and doubling are exact in binary floating point, so
    LRs compare exactly. Extensions beyond what the rule requires are not
    rejected.
    """
    name, base_lr = optimizer.get("name"), optimizer.get("base_lr")
    grid = track.lr_grid.get(name)
    if grid is None:
        protocol.append(f"track {track.name} has no LR grid for optimizer {name!r}")
        return
    if len(grid) == 1:
        if submitted:
            protocol.append(f"track {track.name} fixes {name} LR {grid[0]}; LR trials are not part of it")
        if base_lr != grid[0]:
            protocol.append(f"optimizer.base_lr {base_lr!r} is not the track LR {grid[0]}")
        return
    trials: dict[float, Any] = {}
    for trial in submitted:
        lr = trial.get("lr")
        if trial.get("seed") != track.lr_selection["trial_seed"]:
            protocol.append(f"LR trial {lr!r} uses seed {trial.get('seed')!r}, "
                            f"track requires {track.lr_selection['trial_seed']}")
        if trial.get("final_update") != track.lr_selection["trial_updates"]:
            protocol.append(f"LR trial {lr!r} ends at update {trial.get('final_update')!r}, "
                            f"not {track.lr_selection['trial_updates']}")
        if not _number(trial.get("validation_bpb")):
            missing.append(f"LR trial {lr!r}: validation_bpb is missing")
        if not isinstance(trial.get("log_sha256"), str) or not _SHA256.fullmatch(trial["log_sha256"]):
            missing.append(f"LR trial {lr!r}: log_sha256 must be a 64-hex sha256")
        if not _number(lr) or lr <= 0 or lr in trials:
            protocol.append(f"LR trial {lr!r} is not a unique positive LR")
            continue
        trials[lr] = trial.get("validation_bpb")
    if not trials:
        missing.append(f"LR trials are missing for grid {list(grid)}")
        return
    allowed = set(grid)
    for edge, step in ((max(grid), 2), (min(grid), 0.5)):
        lr = edge * step
        while lr in trials:
            allowed.add(lr)
            lr *= step
    absent = [lr for lr in grid if lr not in trials]
    stray = sorted(set(trials) - allowed)
    if absent:
        protocol.append(f"LR trials lack base grid points {absent} of {list(grid)}")
    if stray:
        protocol.append(f"LR trials {stray} are not an unbroken x2/÷2 extension of the grid {list(grid)}")
    if absent or stray or not all(map(_number, trials.values())):
        return
    selected = min(trials, key=lambda lr: (trials[lr], lr))
    if selected in (min(trials), max(trials)):
        following = selected * 2 if selected == max(trials) else selected / 2
        protocol.append(f"lowest validation BPB is at the edge LR {selected}; "
                        f"add the trial at {following} and repeat until the minimum is interior")
    if base_lr != selected:
        protocol.append(f"optimizer.base_lr {base_lr!r} is not the selected LR {selected}")


def _optimizer_settings(track: Track, optimizer: Mapping[str, Any], protocol: list[str]) -> None:
    """The track fixes every listed setting; the two method-level expert settings are free but stated."""
    fixed = track.optimizers["fixed"].get(optimizer["name"])
    if fixed is None:
        return  # _lr_selection already rejects an optimizer without a grid.
    muon = optimizer.get("muon") or {}
    expected = {f"optimizer.{key}": value for key, value in fixed.items() if key != "muon"}
    expected |= {f"optimizer.muon.{key}": value for key, value in fixed.get("muon", {}).items()}
    declared = {f"optimizer.{key}": optimizer.get(key) for key in fixed if key != "muon"}
    declared |= {f"optimizer.muon.{key}": muon.get(key) for key in fixed.get("muon", {})}
    for key, value in expected.items():
        if declared[key] != value:
            protocol.append(f"{key} is {declared[key]!r}, track fixes {value!r}")
    if "muon" in fixed and muon.get("lr") != optimizer["base_lr"]:
        protocol.append(f"optimizer.muon.lr {muon.get('lr')!r} must equal optimizer.base_lr {optimizer['base_lr']!r}")
    for key in ("expert_lr_multiplier", "expert_group_size"):
        if not _number(optimizer.get(key)):
            protocol.append(f"recipe must state optimizer.{key}")


def validate(submission: Submission, track: Track) -> Report:
    """Check one submission against one track and collect every reason.

    The verdict is the most definite failure: protocol-mismatch, then
    budget-mismatch, then missing-evidence; eligible only when no reason exists.
    """
    protocol: list[str] = []
    missing: list[str] = []
    budget: list[str] = []
    budgets: dict[str, Any] = {}
    if submission.track != track.name:
        protocol.append(f"submitted for track {submission.track!r}, checked against {track.name!r}")

    # Evidence and identities.
    if not submission.config:
        missing.append("resolved config is missing")
    if not isinstance(submission.source_commit, str) or not _SHA1.fullmatch(submission.source_commit):
        missing.append("source_commit must be a 40-hex git commit")
    if submission.manifest_fingerprint is None:
        missing.append("manifest_fingerprint is missing")
    elif submission.manifest_fingerprint != track.data["manifest_fingerprint"]:
        protocol.append("manifest_fingerprint differs from the track data manifest")
    for name, keys in _IDENTITY_KEYS.items():
        declared, frozen = getattr(submission, name), getattr(track, name)
        if not isinstance(declared, Mapping):
            missing.append(f"{name} identity is missing")
            continue
        for key in keys:
            if declared.get(key) is None:
                missing.append(f"{name}.{key} is missing")
            elif declared[key] != frozen[key]:
                protocol.append(f"{name}.{key} is {declared[key]!r}, track requires {frozen[key]!r}")

    # Recipe, schedule and endpoint.
    recipe = submission.recipe
    method: dict[str, Any] = {}
    if not isinstance(recipe, Mapping):
        missing.append("recipe is missing")
    else:
        run, schedule = recipe.get("run", {}), recipe.get("run", {}).get("schedule", {})
        optimizer = recipe.get("optimizer") or {}
        method = {key: optimizer.get(key) for key in ("expert_lr_multiplier", "expert_group_size")}
        if isinstance(optimizer.get("name"), str) and _number(optimizer.get("base_lr")):
            _lr_selection(track, optimizer, submission.lr_trials, protocol, missing)
            _optimizer_settings(track, optimizer, protocol)
        else:
            protocol.append("recipe must state its optimizer and LR")
        required = {"run.seq_len": (run.get("seq_len"), track.seq_len),
                    "run.global_batch_positions": (run.get("global_batch_positions"), track.global_batch_positions),
                    "run.mixed_precision_param": (run.get("mixed_precision_param"), track.mixed_precision_param),
                    "run.successful_updates": (run.get("successful_updates"), track.horizon_updates),
                    "run.schedule_total_steps": (run.get("schedule_total_steps"), track.horizon_updates),
                    "model.vocab_size": (recipe.get("model", {}).get("vocab_size"), track.tokenizer["vocab_size"])}
        required |= {f"run.schedule.{key}": (schedule.get(key), track.schedule[key]) for key in _SCHEDULE_KEYS}
        for key, (declared, frozen) in required.items():
            if declared != frozen:
                protocol.append(f"{key} is {declared!r}, track requires {frozen!r}")
        budgets = _budgets(track, recipe, protocol, missing, budget)

    seed_bpb: dict[int, Any] = {}
    for entry in submission.seeds:
        seed = entry.get("seed")
        if seed in seed_bpb or type(seed) is not int:
            protocol.append(f"seed {seed!r} is not a unique integer")
            continue
        if seed not in track.seeds:
            protocol.append(f"seed {seed} is not a track seed {list(track.seeds)}")
        for key, pattern in (("log_sha256", _SHA256), ("checkpoint_sha256", _SHA256)):
            if not isinstance(entry.get(key), str) or not pattern.fullmatch(entry[key]):
                missing.append(f"seed {seed}: {key} must be a 64-hex sha256")
        for key in ("validation_bpb", "test_bpb"):
            if not _number(entry.get(key)):
                missing.append(f"seed {seed}: final {key} is missing")
        if entry.get("final_update") is None:
            missing.append(f"seed {seed}: final_update is missing")
        elif entry["final_update"] != track.horizon_updates:
            protocol.append(f"seed {seed}: final_update {entry['final_update']} is not the "
                            f"schedule horizon {track.horizon_updates}; only completed schedules count")
        seed_bpb[seed] = entry.get("test_bpb")
    if len(seed_bpb) < track.min_seeds:
        missing.append(f"{len(seed_bpb)} seeds reported, track needs at least {track.min_seeds}")
    elif track.required_seed not in seed_bpb:
        protocol.append(f"seed {track.required_seed} is required")

    reasons = {name: tuple(values) for name, values in
               (("protocol-mismatch", protocol), ("budget-mismatch", budget), ("missing-evidence", missing)) if values}
    verdict = next(iter(reasons), "eligible")
    return Report(submission=submission.name, track=track.name, verdict=verdict, reasons=reasons,
                  budgets=budgets, seed_test_bpb=seed_bpb if verdict == "eligible" else {},
                  gpu_hours=submission.gpu_hours, core=submission.core, **method)


def leaderboard(reports: Sequence[Report]) -> dict[str, Any]:
    """Rank eligible reports by mean final test BPB per track; list the rest as excluded.

    Each row reports the mean over the submitted seeds, their count, and their
    range (min, max) when there are several; one seed (42) is enough, so no
    interval estimate is given. Equal means are ordered by submission name and
    still get consecutive ranks.
    """
    rows = []
    for report in reports:
        if report.verdict != "eligible":
            continue
        values = [report.seed_test_bpb[key] for key in sorted(report.seed_test_bpb)]
        rows.append({"track": report.track, "submission": report.submission,
                     "seeds": sorted(report.seed_test_bpb), "n_seeds": len(values),
                     "mean_test_bpb": math.fsum(values) / len(values),
                     "range_test_bpb": [min(values), max(values)] if len(values) > 1 else None,
                     "expert_lr_multiplier": report.expert_lr_multiplier,
                     "expert_group_size": report.expert_group_size, "gpu_hours": report.gpu_hours,
                     "core": report.core})
    rows.sort(key=lambda row: (row["track"], row["mean_test_bpb"], row["submission"]))
    for index, row in enumerate(rows):
        row["rank"] = sum(other["track"] == row["track"] for other in rows[:index]) + 1
    return {"schema_version": 1, "metric": "test_bpb", "lower_is_better": True,
            "rows": rows,
            "excluded": [{"submission": r.submission, "track": r.track, "verdict": r.verdict}
                         for r in reports if r.verdict != "eligible"]}
