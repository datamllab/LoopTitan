#!/usr/bin/env python3
"""Write the untied twins of the Phase 1 M3 specs to recipes/untie/: python scripts/untie_specs.py [--check]

U0 is M3 (recipes/phase1/m3-*.json) untied (moeloop.untie.untied): the 6 shared core cells
of 88 experts, visited twice, become 12 core cells visited once, and core cell 6 + c is the
second visit of M3's cell c. Depth, width, experts per cell, routing and residual scales
are M3's, so training FLOPs per token and the KV cache are too; only the core parameters
double. With equal copies U0 computes M3's function. ``run.untie`` is null: U0 trains from
scratch; an untie from M3 records its source there. A run sets its LR at launch, from
scratch (recipe_experiments.py run --lr) or from an M3 checkpoint (recipe_experiments.py
untie --lr). --check exits 1 if a file differs from what this script writes.
"""
import argparse
import json
from pathlib import Path

from moeloop.recipe_spec import RecipeSpec
from moeloop.untie import untied

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "recipes" / "untie"
NOTE = ("Untied twin of M3: the 6 shared core cells of 88 experts, visited twice, as 12 core cells visited once "
        "(the second visit of cell c is cell 6 + c); with equal copies it computes M3's function.")


def u0_spec(optimizer: str) -> dict:
    base = f"recipes/phase1/m3-{optimizer}.json"
    spec = untied(json.loads((ROOT / base).read_text()))
    spec |= {"name": f"untie-u0-{optimizer}", "preserved": [NOTE], "run": spec["run"] | {"untie": None},
             "source": {"kind": "untie", "version": 1, "base_recipe": base}}
    return RecipeSpec.from_dict(spec).to_dict()  # from_dict validates the model and its depth


def specs() -> dict[str, dict]:
    return {f"u0-{optimizer}": u0_spec(optimizer) for optimizer in ("adamw", "muon")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with recipes/untie/ instead of writing")
    args = parser.parse_args()
    files = {OUTPUT / f"{name}.json": json.dumps(spec, indent=2) + "\n" for name, spec in specs().items()}
    if args.check:
        stale = sorted(str(path.relative_to(ROOT)) for path in set(files) | set(OUTPUT.glob("*.json"))
                       if not path.is_file() or path.read_text() != files.get(path))
        if stale:
            raise SystemExit(f"not what scripts/untie_specs.py writes: {', '.join(stale)}")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
