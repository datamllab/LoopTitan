"""Looped MoE Bench: frozen tracks, submission checks and the leaderboard.

Budgets come from ``moeloop.comparison``, so this module needs the model extra
(CPU PyTorch is enough).

A submission is a folder: ``submission.json`` declares the entry and the sha256
of every other file in the folder; each seed and LR trial names its run folder
inside it, which holds the run's captured ``config.json``, ``metrics.json``, one
``eval-<split>.json`` per split it reports, and ``train.log``. A verdict covers the
declaration (the three analytical budgets, data/tokenizer/evaluation identities,
schedule completion and LR selection) and its binding to those files: every hash
is recomputed, and each captured configuration and evaluation output is compared
with the declared recipe, values and the frozen track. It never re-runs training
or evaluation, never uses the network, and never judges quality; the report lists
what it checked and what it cannot check without a rerun.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import recipes
from .comparison import compare_recipes
from .recipe_spec import RecipeSpec


TRACKS_PATH = Path(__file__).resolve().parents[2] / "eval" / "bench" / "tracks.json"
PYPROJECT_PATH = TRACKS_PATH.parents[2] / "pyproject.toml"
VERDICTS = ("eligible", "budget-mismatch", "protocol-mismatch", "missing-evidence")
_SHA1, _SHA256 = re.compile(r"[0-9a-f]{40}"), re.compile(r"[0-9a-f]{64}")
# Schedule keys that moeloop.recipes.trainer_config passes to the LR scheduler.
_SCHEDULE_KEYS = ("total_steps", "warmup_steps", "decay_ratio", "decay_type", "min_lr_factor")
_IDENTITY_KEYS = {"tokenizer": ("id", "revision", "vocab_size", "tokenizer_json_sha256"),
                  "evaluation": ("metric", "complete_split", "seq_len", "selection_split",
                                 "selection_label_tokens", "ranking_split", "ranking_label_tokens",
                                 "byte_table_sha256")}
NO_FOLDER = "no submission folder: declared hashes are not bound to files"
UNCHECKED = ("that training this configuration produced the log, metrics and evaluation outputs (needs rerunning training)",
             "which weights the evaluation outputs scored: they carry no model identity (needs the checkpoint and a re-evaluation)",
             "that source_commit exists (no network access here)",
             "the recorded source file hashes (source.files) against the files of the declared commit",
             "source.dirty is the state of the tracked files at launch, not a comparison of the code that ran with the commit",
             "the content of train.log (only its sha256 is compared) and runtime fields other than package versions",
             "the TorchTitan version as the training process saw it: it is recorded by the bench process only",
             "the submission-level gpu_hours and core: declared values not bound to any file",
             "the prepared data behind the manifest fingerprint (needs the data)")


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


def trainer_pins(path: str | Path = PYPROJECT_PATH) -> dict[str, str]:
    """The exact pins (``name==version``) of the ``train`` extra in pyproject.toml."""
    extra = tomllib.loads(Path(path).read_text(encoding="utf-8"))["project"]["optional-dependencies"]["train"]
    return dict(requirement.split("==") for requirement in extra)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Submission:
    """One ``submission.json``. Absent fields stay None and become missing-evidence.

    ``recipe`` is the RecipeSpec dict shared by every seed (its ``run.seed`` is
    not read); ``files`` maps each other file of the folder (a POSIX path
    relative to it) to its sha256. ``seeds`` entries carry seed, final_update,
    validation_bpb, test_bpb, log_sha256, checkpoint_sha256, optionally
    checkpoint_url, and run: the folder (relative, ``.`` for the top) holding
    that run's evidence. ``lr_trials`` entries carry lr, seed, final_update,
    validation_bpb, log_sha256 and run. Validation BPB selects the
    LR; test BPB is reported and ranked. ``gpu_hours`` is reported, never an
    eligibility condition, and so is ``core``: the mean over seeds of the
    ``core`` value from ``moeloop.core.evaluate_core``.
    """

    name: str
    track: str
    recipe: dict[str, Any] | None = None
    files: dict[str, str] | None = None
    source_commit: str | None = None
    manifest_fingerprint: str | None = None
    tokenizer: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    seeds: tuple[dict[str, Any], ...] = ()
    lr_trials: tuple[dict[str, Any], ...] = ()
    gpu_hours: float | None = None
    core: float | None = None
    schema_version: int = 2

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Submission:
        unknown = set(values) - {entry.name for entry in fields(cls)}
        if unknown:
            raise ValueError(f"unknown submission fields: {sorted(unknown)}")
        absent = [name for name in ("name", "track") if name not in values]
        if absent:
            raise ValueError(f"submission fields are required: {absent}")
        if values.get("schema_version", 2) != 2:
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
    """A verdict with every reason found, what was checked and what was not; ``seed_test_bpb`` feeds the leaderboard."""

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
    checked: tuple[str, ...] = ()
    unchecked: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")

    def to_dict(self) -> dict[str, Any]:
        return {"submission": self.submission, "track": self.track, "verdict": self.verdict,
                "reasons": {key: list(value) for key, value in self.reasons.items()},
                "budgets": self.budgets, "seed_test_bpb": {str(k): v for k, v in self.seed_test_bpb.items()},
                "expert_lr_multiplier": self.expert_lr_multiplier, "expert_group_size": self.expert_group_size,
                "gpu_hours": self.gpu_hours, "core": self.core,
                "checked": list(self.checked), "unchecked": list(self.unchecked)}


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


def _differences(declared: Any, captured: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Every leaf path where two JSON values differ, with both values."""
    if isinstance(declared, Mapping) and isinstance(captured, Mapping):
        return [difference for key in sorted(declared.keys() | captured.keys())
                for difference in _differences(declared.get(key), captured.get(key), f"{path}.{key}" if path else key)]
    return [] if declared == captured else [(path, declared, captured)]


def _run_evidence(label: str, entry: Mapping[str, Any], lr: Any, splits: tuple[str, ...], files: Mapping[str, Any],
                  log_sha256: str, submission: Submission, track: Track, protocol: list[str]) -> None:
    """Compare one run's captured config, metrics and evaluation outputs with its declaration and the track."""
    config, metrics = files["config.json"], files["metrics.json"]
    if not isinstance(config, Mapping) or not isinstance(metrics, Mapping):
        protocol.append(f"{label}: config.json and metrics.json must be JSON objects")
        return
    recipe = submission.recipe if isinstance(submission.recipe, Mapping) else {}
    expected = {key: recipe.get(key) for key in ("name", "model", "training", "optimizer", "inference", "run")}
    expected["run"] = dict(expected["run"] or {}) | {"seed": entry.get("seed")}
    expected["optimizer"] = dict(expected["optimizer"] or {}) | {"base_lr": lr}
    if expected["optimizer"].get("name") == "Muon":  # As recipe_experiments.load_spec sets a trial LR.
        expected["optimizer"]["muon"] = dict(expected["optimizer"].get("muon") or {}) | {"lr": lr}
    for path, declared, captured in _differences(expected, {key: config.get(key) for key in expected}):
        protocol.append(f"{label}: config.json {path} is {captured!r}, the declared recipe has {declared!r}")
    launch, source, trainer, runtime = (config.get(key) or {} for key in ("launch", "source", "trainer", "runtime"))
    try:
        positions = (launch["world_size"] * launch["local_batch_size"] * launch["gradient_accumulation_steps"]
                     * track.seq_len)
    except (KeyError, TypeError):
        positions = None
    required = {"config.json launch global batch positions": (positions, track.global_batch_positions),
                "config.json launch.manifest_fingerprint": (launch.get("manifest_fingerprint"),
                                                            track.data["manifest_fingerprint"]),
                "config.json source.commit": (source.get("commit"), submission.source_commit),
                "config.json source.dirty (tracked files)": (source.get("dirty"), False),
                "metrics.json completed_updates": (metrics.get("completed_updates"), entry.get("final_update")),
                "train.log sha256": (log_sha256, entry.get("log_sha256"))}
    for name, pin in trainer_pins().items():  # The training process's own runtime record too, where it has the package.
        required[f"config.json trainer.{name}"] = (str(trainer.get(name)).split("+")[0], pin)
        if name in runtime:
            required[f"config.json runtime.{name}"] = (str(runtime[name]).split("+")[0], pin)
    depth = (recipe.get("inference") or {}).get("num_visits", (recipe.get("model") or {}).get("num_visits"))
    labels = {"validation": "selection_label_tokens", "test": "ranking_label_tokens"}
    for split in splits:
        output, name = files[f"eval-{split}.json"], f"eval-{split}.json"
        if not isinstance(output, Mapping):
            protocol.append(f"{label}: {name} must be a JSON object")
            continue
        tokenizer = output.get("tokenizer") or {}
        required |= {f"{name} split": (output.get("split"), split),
                     f"{name} seq_len": (output.get("seq_len"), track.evaluation["seq_len"]),
                     f"{name} num_visits": (output.get("num_visits"), depth),
                     f"{name} complete_split": (output.get("complete_split"), track.evaluation["complete_split"]),
                     f"{name} valid_tokens": (output.get("valid_tokens"), track.evaluation[labels[split]]),
                     f"{name} byte_table_sha256": (output.get("byte_table_sha256"), track.evaluation["byte_table_sha256"]),
                     f"{name} manifest_fingerprint": (output.get("manifest_fingerprint"), track.data["manifest_fingerprint"]),
                     f"{name} bpb": (output.get("bpb"), entry.get(f"{split}_bpb")),
                     f"metrics.json {split}_bpb": (metrics.get(f"{split}_bpb"), entry.get(f"{split}_bpb"))}
        required |= {f"{name} tokenizer.{key}": (tokenizer.get(key), track.tokenizer[key])
                     for key in ("id", "revision", "vocab_size")}
    for key, (captured, expected_value) in required.items():
        if captured != expected_value:
            protocol.append(f"{label}: {key} is {captured!r}, expected {expected_value!r}")


def _bind(submission: Submission, track: Track, folder: Path, protocol: list[str], missing: list[str],
          checked: list[str]) -> None:
    """Hash every file of the folder and check each run's evidence."""
    declared = submission.files if isinstance(submission.files, Mapping) else {}
    present = {path.relative_to(folder).as_posix(): path for path in sorted(folder.rglob("*")) if path.is_file()}
    present.pop("submission.json", None)
    bound = {}
    for name in sorted(declared.keys() | present.keys()):
        if name not in present:
            missing.append(f"{name}: declared sha256 has no file in the folder")
        elif name not in declared:
            missing.append(f"{name}: file has no declared sha256")
        elif sha256(present[name]) != declared[name]:
            missing.append(f"{name}: file does not match its declared sha256")
        else:
            bound[name] = declared[name]
    checked.append(f"sha256 of the {len(present)} files in the folder against the {len(declared)} declared")
    recipe = submission.recipe if isinstance(submission.recipe, Mapping) else {}
    recipe_lr = (recipe.get("optimizer") or {}).get("base_lr")
    runs = [(f"seed {entry.get('seed')!r}", entry, recipe_lr, ("validation", "test")) for entry in submission.seeds]
    runs += [(f"LR trial {entry.get('lr')!r}", entry, entry.get("lr"), ("validation",)) for entry in submission.lr_trials]
    for label, entry, lr, splits in runs:
        run = entry.get("run")
        if not isinstance(run, str):
            missing.append(f"{label}: run folder is not declared")
            continue
        names = ["config.json", "metrics.json", *(f"eval-{split}.json" for split in splits), "train.log"]
        paths = {name: posixpath.normpath(posixpath.join(run, name)) for name in names}
        absent = [path for path in paths.values() if path not in bound]
        if absent:
            missing.append(f"{label}: run evidence {absent} is not a file matching its declared sha256")
            continue
        try:
            files = {name: json.loads(present[path].read_text(encoding="utf-8"))
                     for name, path in paths.items() if name.endswith(".json")}
        except ValueError as error:
            protocol.append(f"{label}: run evidence is not valid JSON: {error}")
            continue
        _run_evidence(label, entry, lr, splits, files, bound[paths["train.log"]], submission, track, protocol)
        checked.append(f"{label}: config.json in {run!r} against the declared recipe (seed, LR, schedule, updates, "
                       "global batch, model, optimizer) and the track (launch topology, data manifest, trainer and runtime "
                       "package pins, source commit with no tracked change); metrics.json, " + ", ".join(f"eval-{split}.json" for split in splits)
                       + " and train.log sha256 against the declared values, inference depth and the track's "
                       "tokenizer and evaluation identity")


def validate(submission: Submission, track: Track, folder: str | Path | None = None) -> Report:
    """Check one submission and the files of its folder against one track and collect every reason.

    Without ``folder`` no declared hash is bound to a file, so the verdict is at
    best missing-evidence. The verdict is the most definite failure:
    protocol-mismatch, then budget-mismatch, then missing-evidence; eligible
    only when no reason exists.
    """
    protocol: list[str] = []
    missing: list[str] = []
    budget: list[str] = []
    budgets: dict[str, Any] = {}
    if submission.track != track.name:
        protocol.append(f"submitted for track {submission.track!r}, checked against {track.name!r}")

    # Evidence and identities.
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

    # Evidence bound to files.
    checked, unchecked = [], list(UNCHECKED)
    if folder is None:
        missing.append(NO_FOLDER)
    else:
        _bind(submission, track, Path(folder), protocol, missing, checked)
    unchecked += [f"seed {entry.get('seed')!r}: checkpoint_sha256 {entry.get('checkpoint_sha256')} (of the DCP .metadata "
                  f"file only: tensor layout, not weights) at {entry.get('checkpoint_url') or 'no URL declared'}: "
                  "not checked here" for entry in submission.seeds]

    reasons = {name: tuple(values) for name, values in
               (("protocol-mismatch", protocol), ("budget-mismatch", budget), ("missing-evidence", missing)) if values}
    verdict = next(iter(reasons), "eligible")
    return Report(submission=submission.name, track=track.name, verdict=verdict, reasons=reasons,
                  budgets=budgets, seed_test_bpb=seed_bpb if verdict == "eligible" else {},
                  gpu_hours=submission.gpu_hours, core=submission.core, checked=tuple(checked),
                  unchecked=tuple(unchecked), **method)


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
