#!/usr/bin/env python3
"""Write the method specs to recipes/methods/: python scripts/method_specs.py [--check]

Study #225 compares the published looped methods of moeloop.recipes on Track P2 (about
200M active parameters, 5B labels), each (except M0-Loop below) within 5% of recipes/phase1/m0-<optimizer>.json on
unique non-embedding parameters, training FLOPs per token and logical KV (compare_recipes,
the Bench budget rule). Every method is its constructor with the constructor's mechanism:
loop count, which modules are shared or per visit, special parts and training objective.
Only geometry changes (METHODS): P2's vocabulary, width, heads and KV heads where the method
allows, then layer count, expert count and expert width. Each spec says in one sentence what
changed from the constructor default and why. SMELT-V3/V4 take p2_smelt to 3 and 4 core
visits at its 18 effective layers, with core residuals scaled by 1/visits and the core expert
count set to meet the parameter budget. LatentLoop-Residual is LatentLoop Section 3.1's fixed
per-visit residual finding on M3: M3's geometry with the core residuals scaled by 0.25 in the
first visit and 1 in the second, not the paper's final LatentLoop method (2x pooled first
visit, scaled decoder, wider model), which needs sequence pooling. Sampled-depth
training (Huginn) is costed by the expectation moeloop.costs defines over the declared depth
distribution.

M0-Loop is the study's budget-convention control and the one spec outside the 5% rule: m0's
cells unchanged in M3's loop (SMELT's 12-layer split, 3 prefix, 6 core layers run twice, 3
suffix: 18 effective layers, core residuals scaled by 1/2, no per-visit owners). Its unique
non-embedding parameters equal m0's; its training FLOPs per token are 1.31x and its logical KV
1.5x m0's, on purpose. M3 already splits 12 layers, so its 3/6/3 split carries over exactly and
no other split is as close. Memory on 8 x H200 (141 GiB) at local batch 16 and seq 4096:
m0's run (#180) peaked at 103.3 GiB with 12 layer calls (#239: about 109 KiB per token per
call); 18 calls of the same cells need about 144 GiB (155 GiB scaling the whole peak), so M0-Loop
checkpoints activations (full_cells).

The P2 track of eval/bench/tracks.json gives every spec the same schedule and optimizer
settings, as scripts/phase1_specs.py does: the base LR is the track's calibrated LR (issue
#180), and a run may set another at launch. The expert LR multiplier is the constructor's. --check
exits 1 if a file differs from what this script writes or is not written by it.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from moeloop import recipes
from moeloop.bench import load_tracks
from moeloop.config import RegionConfig
from moeloop.recipe_spec import RecipeSpec
from moeloop.recipes import REMOVED_RUN_KEYS
from phase1_specs import P2_POLICY, P2_SCHEDULE_NOTES

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "recipes" / "methods"
OPTIMIZERS = ("AdamW", "Muon")
P2_MOE = json.loads((ROOT / "recipes" / "p2_moe.json").read_text())  # m0's base recipe
P2 = {key: P2_MOE["model"][key] for key in
      ("vocab_size", "dim", "num_heads", "num_kv_heads", "hidden_dim", "num_experts", "top_k", "max_seq_len")}
P2_SMELT = json.loads((ROOT / "recipes" / "p2_smelt.json").read_text())["model"]  # m3's base recipe
M3 = {key: P2_SMELT[key] for key in P2} | {"physical_layers": 12}


def smelt_visits_spec(*, visits: int, core_layers: int, core_experts: int) -> RecipeSpec:
    """p2_smelt with its core of ``core_layers`` visited ``visits`` times, residuals scaled by 1/visits."""
    spec = recipes.p2_smelt_spec()
    model = replace(spec.model, num_cells=core_layers, num_visits=visits,
                    core=RegionConfig(num_experts=core_experts, residual_scale=1 / visits))
    return replace(spec, name=f"smelt_v{visits}", model=model,
        training=replace(spec.training, num_visits=visits), inference=spec.inference | {"num_visits": visits},
        source=spec.source | {"selected_path": f"R{visits}_core_only_scale_1/{visits}_effective_depth_18"},
        preserved=(f"{core_layers} core layers run {visits} times with shared core owners between 3 independent prefix "
                   f"and 3 suffix layers: 18 effective layers, as in SMELT's 2-visit P2 path.",
                   f"Both core attention/FFN residuals use 1/{visits}; independent prelude/coda use 1.",
                   "Fixed depth, full BPTT, terminal cross entropy."))


# method: (constructor, arguments, what geometry changed from the constructor default and why)
METHODS = {
    "meta": (recipes.meta_spec, P2 | {"prelude_layers": 2, "coda_layers": 2, "num_experts": 130},
             "Geometry: prelude and coda of 2 layers instead of 1 give m0's 12 logical layers (2 + 2 x 4 + 2), so the "
             "KV matches at P2's width and heads, and 130 experts per layer instead of 64 give m0's unique parameters "
             "with 6 physical layers."),
    "foil": (recipes.foil_spec, P2 | {"core_layers": 8, "prelude_layers": 2, "coda_layers": 2},
             "Geometry: Base (E, D, L) = (64, 8, 1) with a prelude and coda of 2 layers instead of (4, 4, 1) with 1 "
             "keeps the constructor's 1:4:1 prelude/core/coda ratio and gives m0's 12 logical layers and expert pool "
             "at P2's width (core banks of 128 experts)."),
    "expert-tying": (recipes.expert_tying_spec, P2 | {"groups": 3, "num_experts": 256},
                     "Geometry: 3 groups of g = 4 layers instead of 2 give m0's 12 logical layers, and 256 experts per "
                     "bank instead of 64 give m0's expert pool; the expert LR stays base_lr/sqrt(4)."),
    "huginn": (recipes.huginn_spec, P2 | {"num_heads": 8, "hidden_dim": 984, "num_experts": 86},
               "Geometry: 8 heads of width 160 instead of 10 of 128 at P2's width, because the fixed 1 + 2 x 4 + 1 = 10 "
               "inference layers need wider K/V to meet m0's KV; experts of width 984 instead of 432 make up the "
               "expected training FLOPs that the 2-visit gradient tail saves, and 86 experts per layer meet m0's "
               "unique parameters."),
    "loom": (recipes.loom_spec, P2 | {"core_layers": 4, "num_experts": 196},
             "Geometry: 4 core layers instead of 2 give m0's 12 logical layers (4 x 3), and 196 experts per layer "
             "instead of 64 give m0's unique parameters."),
    "ouro": (recipes.ouro_spec, P2 | {"core_layers": 4, "dim": 640, "num_heads": 5, "hidden_dim": 304,
                                      "num_experts": 562},
             "Geometry: width 640 (5 heads of 128) instead of P2's 1280, because the expected-CE readout at each of "
             "the 3 visits projects onto the 151669-token vocabulary three times per token and only half of P2's "
             "width fits m0's training FLOPs; 4 core layers instead of 2 give m0's 12 logical layers (4 x 3), "
             "experts of width 304 fill the FLOPs and 562 experts per layer meet m0's unique parameters; so the "
             "vocabulary readout is 74% of its training FLOPs (m0: 37%), its other compute is 41% of m0's, and top-8 "
             "of 562 experts activates 1.4% of them (m0: 12.5%)."),
    "sandwich": (recipes.sandwich_spec, P2 | {"core_layers": 1, "top_k": 2, "hidden_dim": 1728, "num_experts": 39},
                 "Geometry: the prelude/core/coda ratio becomes 2:1:2 (P2-R1x8-C2 instead of the constructor's "
                 "2:3:2 P2-R3x8-C2) to give m0's 12 logical layers at P2's width and heads, so 8 of the 12 logical "
                 "layers are core visits against the paper's 86% (96 of 112 in P8-R12x8-C8); the preserved top-2 "
                 "banks use 39 experts of width 1728 to meet m0's unique parameters, and 1728 = 4 x 432, so top-2 x "
                 "1728 equals m0's active 8 x 432."),
    "relaxed": (recipes.relaxed_spec, P2 | {"core_layers": 6, "rank": 160, "num_experts": 129},
                "Geometry: 6 shared layers instead of 4 give m0's 12 logical layers (6 x 2), LoRA rank 160 keeps the "
                "constructor's rank/width ratio of 1/8 at width 1280, and 129 experts per layer meet m0's unique "
                "parameters."),
    "smelt-v3": (smelt_visits_spec, {"visits": 3, "core_layers": 4, "core_experts": 134},
                 "Geometry: P2 SMELT with 4 core layers run 3 times instead of 6 run twice keeps 3 + 4 x 3 + 3 = 18 "
                 "effective layers, and 134 core experts instead of 88 meet m0's unique parameters."),
    "smelt-v4": (smelt_visits_spec, {"visits": 4, "core_layers": 3, "core_experts": 179},
                 "Geometry: P2 SMELT with 3 core layers run 4 times instead of 6 run twice keeps 3 + 3 x 4 + 3 = 18 "
                 "effective layers, and 179 core experts instead of 88 meet m0's unique parameters."),
    "m0-loop": (recipes.smelt_spec, P2 | {"physical_layers": 12},
                "Not a SMELT configuration: SMELT's loop topology only, on m0's cells. Geometry: m0's 12 layers of width 1280 (10 heads of 128, 64 experts of width 432, top-8) in M3's "
                "loop, 3 prefix, 6 core layers run twice and 3 suffix layers (18 effective layers), as the "
                "budget-convention control: unique non-embedding parameters equal m0's, while training FLOPs per "
                "token are 1.31x (model work, without recompute) and logical KV 1.5x m0's, on purpose; the 18 layer calls "
                "are estimated not to fit 8 x H200 at local batch 16, so activations are checkpointed (full_cells)."),
    "latentloop-residual": (recipes.latentloop_residual_spec, M3,
                   "Geometry: M3's, unchanged, so this spec is LatentLoop Section 3.1's (0.25, 1) residual on M3: P2 SMELT's 12 physical layers as 3 prefix, 6 core layers run twice and "
                   "3 suffix (18 effective layers) at width 1056 (12 heads of 88), 88 experts of width 384, top-8, so "
                   "LatentLoop-Residual - M3 differs only in the core residual scales, (0.25, 1) against SMELT's (0.5, 0.5), "
                   "and parameters, training FLOPs and KV equal M3's; like the paper's MoE Loop against its Base "
                   "(1184 against 1408 wide, 92 against 64 experts at 3B; Table 6), M3 is narrower than m0 with more "
                   "experts (1056 against 1280, 88 against 64). M3 trains without activation checkpointing on "
                   "8 x H200 at local batch 16 (#180), and the residual scales change no activation, so neither does this."),
}
FULL_CELLS = {"m0-loop"}  # methods that do not fit in memory without activation checkpointing
DATA_NOTE = ("Publicly reconstructible DCLM replaces the original private corpus; the frozen manifest and tokenizer "
             "are separate comparison identities.")
# Constructor notes that this geometry or Track P2 makes false: (text, what the spec says instead), applied in
# order; {optimizer} is the track optimizer. P2_SCHEDULE_NOTES are dropped, as in phase1_specs.py.
REWRITES = (
    ("Public prepared-text fixture replaces the original data/training stack; its manifest is a separate run "
     "identity.", DATA_NOTE),
    ("Common AdamW and independently owned", "{optimizer} and independently owned"),
    ("Common AdamW replaces the paper's Muon optimizer background.", "The paper trains with Muon; this spec uses {optimizer}."),
    ("Original optimizer background is replaced by AdamW.", "Original optimizer background is replaced by {optimizer}."),
    ("The common AdamW replaces the paper's AdamW", "{optimizer} replaces the paper's AdamW"),
    ("decay to 0.01x rather than the common 0.1x", "decay to 0.01x"),
    ("The common AdamW, schedule and run defaults replace", "{optimizer}, schedule and run settings replace"),
    ("are replaced by the common run defaults and vocabulary", "are replaced by Track P2's 4096-token sequences, 256 "
     "sequences per update and the 151669-token vocabulary"),
    ("P2-R3x8-C2 keeps the paper's 2:3:2 prelude/core/coda ratio and eight passes at the common small geometry; the",
     "The"),
    ("become the common expert count and width", "become the expert count and width of the geometry note"),
    ("at the common small geometry", "at P2 geometry"),
    ("are explicit small-model adaptations", "are explicit small-model adaptations, kept at P2 scale"),
    ("One rank for all attention projections and one for the FFN replace", "One rank for all attention projections replaces"),
    ("SMELT expert count (default E88 instead of the paper's E96) is an explicit resource-matching adaptation",
     "SMELT's prefix and suffix expert count (E88 instead of the paper's E96) and core expert count are explicit "
     "resource-matching adaptations"),
)
TRACK_NOTE = ("Matrix initialization std=.02 and unit norm scales are P2's; seq 4096, 1,048,576 positions per update, "
              "the schedule (95-update linear warmup, constant peak, last 20% linear to 0) and the AdamW or Muon "
              "settings are Track P2's; the training recipe (objective, depth, aux coefficient, expert LR multiplier) "
              "is the constructor's.")
# Muon updates the hidden matrices; everything else keeps AdamW.
OPTIMIZER_TEXT = {"AdamW": "Track P2's AdamW", "Muon": "Track P2's Muon for hidden matrices (AdamW for embedding, "
                  "head, routers, norms and other 1-D parameters)"}


def method_spec(name: str, optimizer: str, track) -> dict:
    constructor, arguments, geometry = METHODS[name]
    spec = constructor(**arguments)
    run = P2_MOE["run"]  # m0's P2 run settings
    fixed, (lr,) = track.optimizers["fixed"][optimizer], track.lr_grid[optimizer]  # one calibrated point
    settings = (spec.optimizer | {key: value for key, value in fixed.items() if key != "muon"}
                | {"name": optimizer, "base_lr": lr})
    if optimizer == "Muon":
        settings["muon"] = fixed["muon"] | {"lr": lr}
    notes = [line for line in spec.adaptations if not line.startswith(P2_SCHEDULE_NOTES)]
    for text, instead in REWRITES:
        notes = [line.replace(text, instead.replace("{optimizer}", OPTIMIZER_TEXT[optimizer])) for line in notes]
    schedule = {key: track.schedule[key] for key in
                ("name", "coordinate", "total_steps", "warmup_steps", "decay_ratio", "decay_type", "min_lr_factor")}
    run = {key: value for key, value in run.items() if key not in P2_POLICY + REMOVED_RUN_KEYS} | {
        "successful_updates": track.horizon_updates, "schedule_total_steps": track.horizon_updates,
        "schedule": schedule, "panel": "methods", "keep_latest_k": 0}
    if name in FULL_CELLS:
        run["activation_checkpointing"] = "full_cells"
    result = replace(spec, name=f"method-{name}-{optimizer.lower()}", optimizer=settings, run=run,
                     adaptations=(*notes, geometry, TRACK_NOTE),
                     source=spec.source | {"track": "P2", "constructor": constructor.__name__})
    return RecipeSpec.from_dict(result.to_dict()).to_dict()  # from_dict validates the model and its depth


def specs() -> dict[str, dict]:
    track = load_tracks()["P2"]
    return {f"{name}-{optimizer.lower()}": method_spec(name, optimizer, track)
            for name in METHODS for optimizer in OPTIMIZERS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with recipes/methods/ instead of writing")
    args = parser.parse_args()
    files = {OUTPUT / f"{name}.json": json.dumps(spec, indent=2) + "\n" for name, spec in specs().items()}
    if args.check:
        stale = sorted(str(path.relative_to(ROOT)) for path in set(files) | set(OUTPUT.glob("*.json"))
                       if not path.is_file() or path.read_text() != files.get(path))
        if stale:
            raise SystemExit(f"not what scripts/method_specs.py writes: {', '.join(stale)}")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
