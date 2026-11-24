#!/usr/bin/env python3
"""Write the 0.1B size-study specs to recipes/p1/: python scripts/p1_specs.py [--check]

The size study (D/N curve at about 100M active parameters) compares M0, the standard MoE
p1_moe_spec(), with M3, the SMELT p1_smelt_spec(), under AdamW and Muon. Track M of
eval/bench/tracks.json gives every spec the same schedule and optimizer settings, as
scripts/phase1_specs.py does for Track P2.

The base LR is the middle of the track grid; a run sets its grid point at launch
(recipe_experiments.py run --lr). Every expert LR multiplier is 1. A checkpoint is saved every
238 steps and all are kept (keep_latest_k 0), so decay forks can start at steps 238, 476, 952,
1904 and 3332. --check exits 1 if a file differs from what this script writes or recipes/p1/
has other files.
"""
import argparse
import json
from pathlib import Path

from moeloop.bench import load_tracks
from moeloop.recipe_spec import RecipeSpec
from moeloop.recipes import p1_moe_spec, p1_smelt_spec

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "recipes" / "p1"
ARMS = {"m0": p1_moe_spec, "m3": p1_smelt_spec}
# P1's note on its own schedule, which the track replaces.
P1_SCHEDULE_NOTE = "Normal matrix initialization"
TRACK_NOTE = ("Matrix initialization std=.02, unit norm scales and aux coefficient .01 are P1's; the schedule "
              "(95-update linear warmup, constant peak, last 20% linear to 0) and the AdamW or Muon settings "
              "are Track M's.")


def arm_spec(arm: str, optimizer: str, track) -> dict:
    base = ARMS[arm]().to_dict()
    fixed, lr = track.optimizers["fixed"][optimizer], track.lr_grid[optimizer][1]
    settings = (base["optimizer"] | {key: value for key, value in fixed.items() if key != "muon"}
                | {"name": optimizer, "base_lr": lr, "expert_lr_multiplier": 1.0})
    if optimizer == "Muon":
        settings["muon"] = fixed["muon"] | {"lr": lr}
    notes = [line for line in base["adaptations"] if not line.startswith(P1_SCHEDULE_NOTE)]
    if optimizer == "Muon":  # Muon updates the hidden matrices; everything else keeps AdamW.
        notes = [line.replace("Common AdamW", "Common Muon for hidden matrices (AdamW for embedding, head, "
                              "routers, norms and other 1-D parameters)") for line in notes]
    schedule = {key: track.schedule[key] for key in
                ("name", "coordinate", "total_steps", "warmup_steps", "decay_ratio", "decay_type", "min_lr_factor")}
    run = base["run"] | {
        "successful_updates": track.horizon_updates, "schedule_total_steps": track.horizon_updates,
        "schedule": schedule, "local_batch_size": 16, "checkpoint_interval": 238, "keep_latest_k": 0}
    spec = base | {
        "name": f"p1-{arm}-{optimizer.lower()}", "optimizer": settings, "run": run,
        "adaptations": notes + [TRACK_NOTE],
        "source": base["source"] | {"kind": "p1_arm", "arm": arm, "base_recipe": ARMS[arm].__name__}}
    return RecipeSpec.from_dict(spec).to_dict()  # from_dict validates the model and its depth


def specs() -> dict[str, dict]:
    track = load_tracks()["M"]
    return {f"{arm}-{optimizer.lower()}": arm_spec(arm, optimizer, track)
            for arm in ARMS for optimizer in ("AdamW", "Muon")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with recipes/p1/ instead of writing")
    args = parser.parse_args()
    files = {OUTPUT / f"{name}.json": json.dumps(spec, indent=2) + "\n" for name, spec in specs().items()}
    if args.check:
        stale = sorted(str(path.relative_to(ROOT)) for path in set(files) | set(OUTPUT.glob("*"))
                       if not path.is_file() or path.read_text() != files.get(path))
        if stale:
            raise SystemExit(f"not what scripts/p1_specs.py writes: {', '.join(stale)}")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
