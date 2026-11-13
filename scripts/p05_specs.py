#!/usr/bin/env python3
"""Write the 0.5B size-study specs to recipes/p05/: python scripts/p05_specs.py [--check]

The size study (#223) compares M0, an ordinary MoE, with M3, SMELT, at about 0.5B active
parameters under AdamW and Muon. Each arm scales a frozen P2 recipe (ARMS), so the P2
initialization, aux coefficient, norms and objective carry over unchanged. U (Muon only, #255)
is the shape control for pre-registration 4: M3's 24 logical layers unrolled into 24 unshared
layers of M3's width, heads and expert width, with no loop and no residual scaling, and 60
experts per layer so its unique non-embedding parameters match M0's. U against M3 then changes
the sharing and the experts per layer (1440 unique experts against 1472), as
recipes/shape/u-*.json does against recipes/phase1/m3-*.json at 0.2B.

M0 has 497,092,608 active non-embedding parameters per token: attention, norms, router and
8 of 64 experts in each of 16 layers, plus the final norm. M3 differs from M0 by +0.9% in
unique non-embedding parameters, +2.7% in training FLOPs per token and +3.1% in logical KV
(moeloop.comparison.compare_recipes with Track P2's accounting: dense attention, no recompute).
U differs from M0 by +0.4%, +2.5% and +3.1%: its 24 layers have M3's logical KV and nearly
its compute, and 24 x 60 experts hold about M3's 16 x 92.

Track P2's rules give the schedule and optimizer settings; only the length changes. The run
uses the whole train split of the track's DCLM manifest once: H = floor(train labels /
1,048,576 positions) = 22888 updates, a round(0.02 H) = 458-update warmup and the last 20%
linear to 0. A checkpoint is saved every 1144 updates and all are kept (keep_latest_k 0),
so decay forks start at 1144, 2288, 4576 and 9152, each decaying a quarter of its step
(horizons 1430, 2860, 5720 and 11440, D/N 3.02, 6.03, 12.07 and 24.13 for M0's active N);
the full run ends at D/N 48.28. The base LR is the track's calibrated LR (issue #180); a run
may set another at launch (recipe_experiments.py run --lr).

No arm fits in 141 GB without activation checkpointing at local batch 16: a peak of
about 195 GiB per GPU for M0 and 230 GiB for M3 (an activation model scaled to P2's measured
no-checkpointing peaks, 103 GiB for its MoE and 124 GiB for its SMELT). U saves activations for
the same 24 layers of M3's width, and at 0.2B U at M3's width and depth needed about M3's memory
(125 against 124 GiB, scripts/shape_specs.py), so U needs about M3's 230 GiB. The specs therefore
set activation_checkpointing full_cells, which brings all three to about 35 GiB; the budgets
above count no recompute, as Track P2 does, and stay within 5% with recompute counted.

--check exits 1 if a file differs from what this script writes or recipes/p05/ has other files.
"""
import argparse
import json
from pathlib import Path

from moeloop.bench import load_tracks
from moeloop.recipe_spec import RecipeSpec

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "recipes" / "p05"
CHECKPOINT_INTERVAL = 1144  # the D/N 3 fork step, 0.8 H/16; the other forks are multiples

# arm: (frozen P2 recipe, geometry changes, note)
ARMS = {
    "m0": ("p2_moe", {"dim": 2048, "num_heads": 16, "hidden_dim": 448, "num_cells": 16},
           "Ordinary MoE: 16 layers of 64 experts (width 448), top-8."),
    "m3": ("p2_smelt", {"dim": 1672, "num_heads": 19, "hidden_dim": 392, "num_experts": 92,
                        "prefix_cells": 4, "num_cells": 8, "suffix_cells": 4},
           "SMELT: 4 prefix layers, 8 core layers of 92 experts (width 392) each visited twice and shared "
           "whole, 4 suffix layers."),
    "u": ("p2_smelt", {"dim": 1672, "num_heads": 19, "hidden_dim": 392, "num_experts": 60, "prefix_cells": 0,
                       "num_cells": 24, "suffix_cells": 0, "num_visits": 1, "core": None},
          "M3's 24 logical layers unrolled into one ordinary MoE stack: 24 unshared layers of width 1672 (19 heads "
          "of 88), 60 experts of width 392 each, top-8, residual not scaled."),
}
OPTIMIZERS = {"u": ("Muon",)}  # every other arm: AdamW and Muon
# P2's notes on its own schedule, AdamW-only controls and P2 geometry, which this script replaces.
P2_NOTES = ("Normal matrix initialization", "P1 initialization, AdamW", "P2 public-stack", "SMELT expert count")
GEOMETRY_NOTE = (
    "The 0.5B geometry continues P1 -> P2, which grew wider faster than deeper (dim 672 -> 1280, 10 -> 12 "
    "layers): M0 takes dim 2048 (16 heads of 128) and 16 layers, a multiple of 4 for the 4 + 8 + 4 SMELT split, "
    "with expert width 448 for about 0.5B active parameters. M3 keeps p2_smelt's proportions to M0 (width 0.82, "
    "head dim 88, 1.4x the experts at 0.9x the width); among those shapes E92/F392 is balanced on all three "
    "budgets, within 5% of M0 with dense attention and with P2's measured document pairs. Expert pool and "
    "head dimensions differ, so this is a recipe comparison, not a pure loop ablation.")
# Notes an arm adds after the shared ones.
ARM_NOTES = {"u": [
    "U is the shape control for #223's pre-registration 4 (#255): M3 with each logical layer given its own "
    "attention, router, norms and experts, no loop and no residual scaling, as recipes/shape/u-*.json unrolls "
    "recipes/phase1/m3-*.json at 0.2B. The expert count 60 matches M0's unique non-embedding parameters "
    "(+0.4%): 24 x 60 = 1440 unique experts against M3's 16 x 92 = 1472. With 24 layers of M3's width, U has "
    "M3's logical KV and nearly its compute, so U against M3 changes the sharing and, to keep the parameters, "
    "the experts per layer. Only Muon is written (#223 Decision, 2026-11-05).",
]}
# P2's own study policy (LR trials, diagnostics, release identity); the track rules replace it.
P2_POLICY = ("anchor", "diagnostic_updates", "final_test_label_tokens", "final_validation_label_tokens",
             "formal_identity_status", "lr_trial_max_label_tokens", "lr_trial_max_updates", "lr_trial_peak_lrs")


def schedule(track) -> dict:
    """Track P2's rule over the whole train split of its manifest: no label is seen twice."""
    manifest = json.loads((ROOT / track.data["manifest"]).read_text())
    horizon = manifest["provenance"]["selection"]["actual_label_tokens"]["train"] // track.global_batch_positions
    rule = track.schedule
    return {"name": rule["name"], "coordinate": rule["coordinate"], "total_steps": horizon,
            "warmup_steps": round(rule["warmup_fraction"] * horizon), "decay_ratio": rule["decay_fraction"],
            "decay_type": rule["decay_type"], "min_lr_factor": rule["min_lr_factor"]}


def arm_spec(arm: str, optimizer: str, track) -> dict:
    base_name, changes, note = ARMS[arm]
    base = json.loads((ROOT / "recipes" / f"{base_name}.json").read_text())
    fixed, (lr,) = track.optimizers["fixed"][optimizer], track.lr_grid[optimizer]  # one calibrated point
    settings = (base["optimizer"] | {key: value for key, value in fixed.items() if key != "muon"}
                | {"name": optimizer, "base_lr": lr, "expert_lr_multiplier": 1.0})
    if optimizer == "Muon":
        settings["muon"] = fixed["muon"] | {"lr": lr}
    notes = [line for line in base["adaptations"] if not line.startswith(P2_NOTES)]
    if optimizer == "Muon":  # Muon updates the hidden matrices; everything else keeps AdamW.
        notes = [line.replace("Common AdamW", "Common Muon for hidden matrices (AdamW for embedding, head, "
                              "routers, norms and other 1-D parameters)") for line in notes]
    plan = schedule(track)
    horizon = plan["total_steps"]
    track_note = (f"Matrix initialization std=.02, unit norm scales and aux coefficient .01 are P2's; the schedule "
                  f"is Track P2's rule over the whole train split ({horizon} updates, {plan['warmup_steps']}-update "
                  f"linear warmup, constant peak, last 20% linear to 0) and the AdamW or Muon settings are Track P2's.")
    run = {key: value for key, value in base["run"].items() if key not in P2_POLICY} | {
        "successful_updates": horizon, "schedule_total_steps": horizon, "schedule": plan, "panel": "p05",
        "checkpoint_interval": CHECKPOINT_INTERVAL, "keep_latest_k": 0, "activation_checkpointing": "full_cells"}
    model = base["model"] | changes
    spec = base | {
        "name": f"p05-{arm}-{optimizer.lower()}", "model": model, "optimizer": settings, "run": run,
        "training": base["training"] | {"num_visits": model["num_visits"]},
        "inference": base["inference"] | {"num_visits": model["num_visits"]},
        "preserved": [note], "adaptations": notes + [GEOMETRY_NOTE, track_note] + ARM_NOTES.get(arm, []),
        "source": {"kind": "p05_arm", "version": 1, "arm": arm, "base_recipe": f"recipes/{base_name}.json"}}
    return RecipeSpec.from_dict(spec).to_dict()  # from_dict validates the model and its depth


def specs() -> dict[str, dict]:
    track = load_tracks()["P2"]
    return {f"{arm}-{optimizer.lower()}": arm_spec(arm, optimizer, track)
            for arm in ARMS for optimizer in OPTIMIZERS.get(arm, ("AdamW", "Muon"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with recipes/p05/ instead of writing")
    args = parser.parse_args()
    files = {OUTPUT / f"{name}.json": json.dumps(spec, indent=2) + "\n" for name, spec in specs().items()}
    if args.check:
        stale = sorted(str(path.relative_to(ROOT)) for path in set(files) | set(OUTPUT.glob("*"))
                       if not path.is_file() or path.read_text() != files.get(path))
        if stale:
            raise SystemExit(f"not what scripts/p05_specs.py writes: {', '.join(stale)}")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
