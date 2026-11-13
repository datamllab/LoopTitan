"""Print portable recipe/cost/comparison JSON; never launch training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# A checkout invocation works before installation; installed imports still work.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moeloop import recipes
from moeloop.comparison import compare_recipes, diff_recipes, smelt_budget_pair
from moeloop.costs import estimate_cost
from moeloop.recipe_spec import RecipeSpec


NAMES = ("tiny", "dense", "moe", "recursive_moe", "smelt", "meta", "foil_base",
         "foil", "expert_tying", "per_visit_router", "input_anchor",
         "huginn", "loom", "loom_attention", "ouro", "sandwich", "relaxed")


def _read(value: str) -> RecipeSpec:
    if value == "loom_attention":
        return recipes.loom_spec(attention_per_visit=True)
    if value in NAMES:
        return getattr(recipes, f"{value}_spec")()
    return RecipeSpec.from_dict(json.loads(Path(value).read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "diff", "compare", "pair"))
    parser.add_argument("recipes", nargs="*", help="built-in name or RecipeSpec JSON path")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--activation-checkpointing", choices=("none", "full_cells", "unknown"), default="none")
    parser.add_argument("--attention-pairs-per-sequence", type=float,
        help="Measured mean within-document causal attention pairs per packed sequence; omit for explicit dense-S² arithmetic")
    parser.add_argument("--tolerance", type=float, default=.05)
    args = parser.parse_args()
    if args.attention_pairs_per_sequence is not None and args.action == "diff":
        parser.error("--attention-pairs-per-sequence requires inspect or compare")
    expected = {"inspect": 1, "diff": 2, "compare": 2, "pair": 0}[args.action]
    if len(args.recipes) != expected:
        parser.error(f"{args.action} requires {expected} recipe argument(s)")
    specs = [_read(value) for value in args.recipes]
    seq_len = args.seq_len if args.seq_len is not None else min([2048] + [spec.model.max_seq_len for spec in specs])
    options = {"seq_len": seq_len, "batch_size": args.batch_size,
               "bytes_per_element": args.bytes_per_element,
               "activation_checkpointing": None if args.activation_checkpointing == "unknown" else args.activation_checkpointing,
               "attention_pairs_per_sequence": args.attention_pairs_per_sequence}
    if args.action == "inspect":
        result = {"recipe": specs[0].to_dict(), "cost": estimate_cost(specs[0], **options)}
    elif args.action == "diff":
        result = diff_recipes(*specs)
    elif args.action == "compare":
        result = compare_recipes(*specs, tolerance=args.tolerance, **options)
    else:
        if (args.batch_size, args.bytes_per_element, args.activation_checkpointing, args.tolerance,
                args.attention_pairs_per_sequence) != (1, 2, "none", .05, None):
            parser.error("pair reproduces its fixed batch=1, BF16-KV, no-AC, dense-S², 5% contract; use compare for other contracts")
        result = smelt_budget_pair(seq_len=seq_len)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
