#!/usr/bin/env python3
"""Write the Phase 1 arm specs to recipes/phase1/: python scripts/phase1_specs.py [--check]

Phase 1 asks where the looped-MoE gain over an ordinary MoE comes from at P2 scale
(about 200M active parameters, 5B labels). Every arm starts from a frozen P2 recipe and
changes only which modules its layers own or share (ARMS). The P2 track of
eval/bench/tracks.json then gives every arm the same schedule and optimizer settings.

The ladder takes SMELT (p2_smelt: 3 prefix layers, 6 core layers visited twice, 3 suffix
layers) apart one shared module at a time: m4 unties the core, m7 shares only the experts,
m5 also shares the router, and m3 is SMELT, which also shares attention and norms. The rails
keep the ordinary MoE shape (p2_moe) and change only how the experts are organised. m0-2e
doubles the expert parameters: it is the one arm outside the budget.

m7-elr and m4-elr (AdamW only, issue #264) are m7 and m4 with the core experts' LR times
sqrt(2): the step-size test of study #177. The base LR is the track's calibrated LR
(issue #180); a run may set another at launch (recipe_experiments.py run --lr). Every other
arm has expert LR multiplier 1. All checkpoints are
kept (keep_latest_k 0), so decay forks can start from any of them. --check exits 1 if a
file differs from what this script writes.
"""
import argparse
import json
import math
from pathlib import Path

from moeloop.bench import load_tracks
from moeloop.recipe_spec import RecipeSpec

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "recipes" / "phase1"
BOTH = ("AdamW", "Muon")
OWN = {"attention_per_visit": True, "router_per_visit": True, "norm_per_visit": True}

# arm: (frozen base recipe, model changes, layers sharing one expert bank, optimizers, note)
ARMS = {
    "m0": ("p2_moe", {}, 1, BOTH,
           "Ordinary MoE baseline: 12 layers of 64 experts, top-8."),
    "m4": ("p2_smelt", {"num_cells": 12, "num_visits": 1, "core": {
               "hidden_dim": None, "num_experts": 44, "top_k": None, "always_on_hidden_dim": None,
               "residual_scale": 0.5}}, 1, BOTH,
           "Untied twin of SMELT: 12 independent core layers of 44 experts (the 6 x 88 experts, untied)."),
    "m7": ("p2_smelt", OWN, 1, BOTH,
           "SMELT sharing only the experts: attention, router and norms are owned per visit."),
    "m5": ("p2_smelt", {"attention_per_visit": True, "norm_per_visit": True}, 1, BOTH,
           "SMELT sharing experts and router: attention and norms are owned per visit."),
    "m3": ("p2_smelt", {}, 1, BOTH,
           "SMELT: 6 core layers of 88 experts, each visited twice, shared whole."),
    "m1": ("p2_moe", {"num_experts": 128, "num_cells": 6, "num_visits": 2, "order": "layer"} | OWN, 2, BOTH,
           "6 expert banks of 128, each shared by 2 adjacent layers; everything else per layer."),
    "m1s": ("p2_moe", {"num_experts": 128, "num_cells": 6, "num_visits": 2, "order": "stack"} | OWN, 2, ("AdamW",),
            "The 6 banks of m1 in loop order: layers 1-6 use banks 1-6, layers 7-12 use them again."),
    "m2": ("p2_moe", {"num_experts": 256, "num_cells": 3, "num_visits": 4, "order": "layer"} | OWN, 4, ("AdamW",),
           "3 expert banks of 256, each shared by 4 adjacent layers; everything else per layer."),
    "m6": ("p2_moe", {"num_experts": 128, "hidden_dim": 216, "top_k": 16}, 1, BOTH,
           "Fine-grained MoE: 128 experts of width 216 per layer, top-16; the pool of m1 without sharing."),
    "m0-2e": ("p2_moe", {"num_experts": 128}, 1, BOTH,
              "128 full-width experts per layer, top-8: twice the expert parameters, outside the budget."),
}
# The step-size test of study #177: the same arms with the core experts' LR times sqrt(2).
EXPERT_LR = {"m7-elr": ("m7", "SMELT sharing only the experts (m7) with the core experts' LR times sqrt(2): "
                        "tests whether AdamW gives each use of a shared expert about 1/sqrt(2) of a step (#177)."),
             "m4-elr": ("m4", "Untied twin of SMELT (m4) with the core experts' LR times sqrt(2): "
                        "the control for m7-elr, whose unshared experts should gain nothing from the larger step (#177).")}
for _arm, (_source, _note) in EXPERT_LR.items():
    ARMS[_arm] = (*ARMS[_source][:3], ("AdamW",), _note)
# P2's notes on its own schedule and AdamW-only controls, which the P2 track replaces.
P2_SCHEDULE_NOTES = ("Normal matrix initialization", "P1 initialization, AdamW")
TRACK_NOTE = ("Matrix initialization std=.02, unit norm scales and aux coefficient .01 are P2's; the schedule "
              "(95-update linear warmup, constant peak, last 20% linear to 0) and the AdamW or Muon settings "
              "are Track P2's.")
# P2's own study policy (LR trials, diagnostics, release identity); the track replaces it.
P2_POLICY = ("anchor", "diagnostic_updates", "final_test_label_tokens", "final_validation_label_tokens",
             "formal_identity_status", "lr_trial_max_label_tokens", "lr_trial_max_updates", "lr_trial_peak_lrs")


def arm_spec(arm: str, optimizer: str, track) -> dict:
    base_name, changes, bank, _, note = ARMS[arm]
    base = json.loads((ROOT / "recipes" / f"{base_name}.json").read_text())
    model = base["model"] | changes
    fixed, (lr,) = track.optimizers["fixed"][optimizer], track.lr_grid[optimizer]  # one calibrated point
    settings = (base["optimizer"] | {key: value for key, value in fixed.items() if key != "muon"}
                | {"name": optimizer, "base_lr": lr, "expert_group_size": bank,
                   "expert_lr_multiplier": math.sqrt(2) if arm in EXPERT_LR else 1.0})
    if optimizer == "Muon":
        settings["muon"] = fixed["muon"] | {"lr": lr}
    notes = [line for line in base["adaptations"] if not line.startswith(P2_SCHEDULE_NOTES)]
    if optimizer == "Muon":  # Muon updates the hidden matrices; everything else keeps AdamW.
        notes = [line.replace("Common AdamW", "Common Muon for hidden matrices (AdamW for embedding, head, "
                              "routers, norms and other 1-D parameters)") for line in notes]
    schedule = {key: track.schedule[key] for key in
                ("name", "coordinate", "total_steps", "warmup_steps", "decay_ratio", "decay_type", "min_lr_factor")}
    run = {key: value for key, value in base["run"].items() if key not in P2_POLICY} | {
        "successful_updates": track.horizon_updates, "schedule_total_steps": track.horizon_updates,
        "schedule": schedule, "panel": "phase1", "keep_latest_k": 0}
    spec = base | {
        "name": f"phase1-{arm}-{optimizer.lower()}", "model": model,
        "training": base["training"] | {"num_visits": model["num_visits"]},
        "inference": base["inference"] | {"num_visits": model["num_visits"]},
        "optimizer": settings, "run": run, "preserved": [note], "adaptations": notes + [TRACK_NOTE],
        "source": {"kind": "phase1_arm", "version": 1, "arm": arm, "base_recipe": f"recipes/{base_name}.json"}}
    return RecipeSpec.from_dict(spec).to_dict()  # from_dict validates the model and its depth


def specs() -> dict[str, dict]:
    track = load_tracks()["P2"]
    return {f"{arm}-{optimizer.lower()}": arm_spec(arm, optimizer, track)
            for arm, (*_, optimizers, _) in ARMS.items() for optimizer in optimizers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with recipes/phase1/ instead of writing")
    args = parser.parse_args()
    files = {OUTPUT / f"{name}.json": json.dumps(spec, indent=2) + "\n" for name, spec in specs().items()}
    if args.check:
        stale = sorted(str(path.relative_to(ROOT)) for path in set(files) | set(OUTPUT.glob("*.json"))
                       if not path.is_file() or path.read_text() != files.get(path))
        if stale:
            raise SystemExit(f"not what scripts/phase1_specs.py writes: {', '.join(stale)}")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
