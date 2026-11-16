#!/usr/bin/env python3
"""Write the shape-decomposition specs to recipes/shape/: python scripts/shape_specs.py [--check]

The chain M0 -> G -> U -> R -> M4 changes one shape setting per step, plus M0R (M0 with only
the middle residual scaled); this writes G, U, R and M0R. The depth scan adds D24 and D36, plain
MoEs deeper and narrower than U. M0LR and M0I (AdamW only) split M0R's residual x 0.5: under Adam
it is the same training as LR x 0.5 (weight decay x 2) plus init x 0.5 on the middle layers'
branch output projections, so each keeps M0R's 3/6/3 split and changes one of the two. Each arm is a Phase 1 spec (recipes/phase1/m0-*.json or
m4-*.json) with the model fields below replaced; everything else, the optimizer, schedule and run
settings included, is the source's, except that the arms in FULL_CELLS checkpoint activations
and ``source`` names the arm and its Phase 1 spec.
The description is the spec's ``preserved`` sentence. A run sets its LR at launch
(recipe_experiments.py run --lr).

D24 and D36 keep U's proportions to M0. Logical KV (1 KV head) grows with depth, so the head
dim is about 1536 / layers (M0's 12 x 128): 64 at 24 layers, and 44 at 36 layers, half of U's 88,
with U's KV (+3.1%). Head dim 42 would match KV more closely, but width 714 (17 x 42) is not a
multiple of 8, and a bf16 matmul at that width ran at about 85% of width 704's throughput. Expert
width is about 0.36 x dim (U's 384 / 1056), and the expert count matches M0's parameters. Against
M0 (moeloop.comparison.compare_recipes with Track P2's accounting) D24 differs by +0.3% in unique
non-embedding parameters, -0.7% in training FLOPs per token and 0% in logical KV; D36 by +0.5%,
-2.6% and +3.1%.

Memory on 8 x H200 (140 GiB usable) at local batch 16 and seq 4096, without checkpointing: M0's
run (#180) peaked at 103 GiB. One layer saves about 109 KiB per token for M0, 92 for U, 78 for
D24 and 61 for D36 (measured in eager bf16 with native experts and flex attention), so D24 needs
about 136 GiB and D36 about 156 GiB, against 125 GiB for U, which ran. M6 (170 KiB per layer, 12
layers, about 156 GiB) ran out of memory at 136.5 GiB allocated. Neither fits with margin, so both
set activation_checkpointing full_cells, as M6's rerun did.

--check exits 1 if a file differs from what this script writes or recipes/shape/ has other files.
"""
import argparse
import json
from pathlib import Path

from moeloop.recipe_spec import RecipeSpec

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "recipes" / "shape"
NO_CORE = {"hidden_dim": None, "num_experts": None, "top_k": None, "always_on_hidden_dim": None}
ARMS = {  # arm: (source arm, model fields it changes, description)
    "g": ("m0", dict(hidden_dim=384, num_experts=72, top_k=9),
          "M0 with finer experts: 72 per layer of width 384, top-9."),
    "u": ("m4", dict(prefix_cells=0, num_cells=18, suffix_cells=0, num_experts=58, core=None),
          "M4's depth and width as one uniform stack: 18 layers of width 1056, 58 experts each, residual not scaled."),
    "r": ("m4", dict(core=NO_CORE | {"num_experts": 44, "residual_scale": 1.0}),
          "M4's expert layout (88 experts in each of the first 3 and last 3 layers, 44 in each of the middle 12)"
          " with the middle residual not scaled."),
    "m0r": ("m0", dict(prefix_cells=3, num_cells=6, suffix_cells=3, core=NO_CORE | {"residual_scale": 0.5}),
            "M0's shape with only the residual of the middle 6 layers scaled by 0.5."),
    "m0lr": ("m0", dict(prefix_cells=3, num_cells=6, suffix_cells=3,
                        core=NO_CORE | {"residual_scale": 1.0, "output_lr_scale": 0.5}),
             "M0's shape with only the LR of the middle 6 layers' branch output projections (attention out_proj,"
             " expert down projections) scaled by 0.5 and their weight decay by 2. Under Adam, M0R's residual x 0.5"
             " is this LR change plus init x 0.5 on the same projections; this arm keeps only the LR change, to test"
             " whether M0R's gain comes from the LR."),
    "m0i": ("m0", dict(prefix_cells=3, num_cells=6, suffix_cells=3,
                       core=NO_CORE | {"residual_scale": 1.0, "output_init_scale": 0.5}),
            "M0's shape with only the init of the middle 6 layers' branch output projections (attention out_proj,"
            " expert down projections) scaled by 0.5. Under Adam, M0R's residual x 0.5 is this init change plus"
            " LR x 0.5 (weight decay x 2) on the same projections; this arm keeps only the init change, to test"
            " whether M0R's gain comes from the init."),
    "d24": ("m0", dict(dim=896, num_heads=14, hidden_dim=320, num_experts=62, num_cells=24),
            "M0 made deeper and narrower: 24 layers of width 896, 14 heads of dim 64 (smaller heads keep the"
            " KV cache of more layers at M0's size), 62 experts of width 320 each, top-8, residual not scaled."),
    "d36": ("m0", dict(dim=704, num_heads=16, hidden_dim=256, num_experts=66, num_cells=36),
            "M0 made deeper and narrower: 36 layers of width 704, 16 heads of dim 44 (smaller heads keep the"
            " KV cache of more layers within 5% of M0's), 66 experts of width 256 each, top-8, residual not scaled."),
}
FULL_CELLS = {"d24", "d36"}  # arms that do not fit in memory without activation checkpointing
OPTIMIZERS = {"m0lr": ("adamw",), "m0i": ("adamw",)}  # every other arm: AdamW and Muon


def spec_for(arm: str, optimizer: str) -> dict:
    source, changes, description = ARMS[arm]
    base = f"recipes/phase1/{source}-{optimizer}.json"
    spec = json.loads((ROOT / base).read_text())
    spec |= {"name": f"shape-{arm}-{optimizer}", "preserved": [description], "model": spec["model"] | changes,
             "source": {"kind": "shape_arm", "version": 1, "arm": arm, "base_recipe": base}}
    if arm in FULL_CELLS:
        spec["run"] |= {"activation_checkpointing": "full_cells"}
    return RecipeSpec.from_dict(spec).to_dict()  # from_dict validates the model and its depth


def specs() -> dict[str, dict]:
    return {f"{arm}-{optimizer}": spec_for(arm, optimizer)
            for arm in ARMS for optimizer in OPTIMIZERS.get(arm, ("adamw", "muon"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with recipes/shape/ instead of writing")
    args = parser.parse_args()
    files = {OUTPUT / f"{name}.json": json.dumps(spec, indent=2) + "\n" for name, spec in specs().items()}
    if args.check:
        stale = sorted(str(path.relative_to(ROOT)) for path in set(files) | set(OUTPUT.glob("*"))
                       if not path.is_file() or path.read_text() != files.get(path))
        if stale:
            raise SystemExit(f"not what scripts/shape_specs.py writes: {', '.join(stale)}")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
