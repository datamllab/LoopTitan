#!/usr/bin/env python3
"""Resolve P1 recipe/comparison JSON, or load it through the native launcher.

Offline: python scripts/recipe_anchor.py --output /path/to/new-config-directory
Native: MOELOOP_ANCHOR_SPEC=/path/to/p1_moe.json PYTHONPATH=src:scripts
        python -m moeloop.train --module recipe_anchor --config anchor

This entry uses already assigned resources. It neither allocates GPUs nor
starts training when generating JSON. A stop is an absolute update coordinate;
it never changes the frozen full WSD horizon.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def anchor():
    """Native config factory; CLI flags can set batch, parallelism and profiling."""
    from moeloop.recipe_spec import RecipeSpec
    from moeloop.recipes import trainer_config

    spec = RecipeSpec.from_dict(json.loads(Path(os.environ["MOELOOP_ANCHOR_SPEC"]).read_text()))
    if spec.run.get("anchor") != "P1":
        raise ValueError("anchor requires a resolved P1 RecipeSpec")
    config = trainer_config(spec)
    stop = int(os.environ.get("MOELOOP_ANCHOR_STOP", spec.run["successful_updates"]))
    if not 0 < stop <= spec.run["successful_updates"]:
        raise ValueError("anchor stop must lie within the frozen full schedule")
    config.training = replace(config.training, steps=stop)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    config.parallelism = replace(config.parallelism,
        data_parallel_replicate_degree=1, data_parallel_shard_degree=world)
    if "MOELOOP_ANCHOR_OUTPUT" in os.environ:
        config.dump_folder = os.environ["MOELOOP_ANCHOR_OUTPUT"]
    if "MOELOOP_ANCHOR_MANIFEST" in os.environ:
        manifest = Path(os.environ["MOELOOP_ANCHOR_MANIFEST"]).resolve()
        if manifest.is_dir():
            manifest = manifest / "manifest.json"
        config.dataloader = replace(config.dataloader, manifest_path=str(manifest))
        config.validator.dataloader = replace(config.validator.dataloader, manifest_path=str(manifest))
        config.hf_assets_path = str(manifest.parent / "tokenizer")
    config.checkpoint = replace(config.checkpoint, keep_latest_k=2, async_mode="disabled",
        initial_load_path=os.environ.get("MOELOOP_ANCHOR_RESUME", config.checkpoint.initial_load_path))
    config.__post_init__()
    return config


def main(argv=None):
    from moeloop.comparison import smelt_p1_pair

    # The predeclared peak-LR trials, shared by both frozen P1 arms; the first is the formal peak.
    trials = json.loads((ROOT / "recipes" / "p1_moe.json").read_text())["run"]["lr_trial_peak_lrs"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peak-lr", type=float, choices=trials, default=trials[0])
    parser.add_argument("--activation-checkpointing", choices=("none", "full_cells"), default="none")
    parser.add_argument("--attention-pairs-per-sequence", type=float,
        help="Measured mean within-document causal attention pairs per packed sequence; omit for explicit dense-S² arithmetic")
    args = parser.parse_args(argv)
    pair = smelt_p1_pair(base_lr=args.peak_lr, activation_checkpointing=args.activation_checkpointing,
        attention_pairs_per_sequence=args.attention_pairs_per_sequence)
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, value in (("pair.json", pair),
            ("p1_moe.json", pair["comparison"]["arms"]["reference"]),
            ("p1_smelt.json", pair["comparison"]["arms"]["candidate"])):
        path = args.output / filename
        content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f"refusing to replace a different resolved config: {path}")
        path.write_text(content)
    print(json.dumps({"output": str(args.output), "candidate_count": pair["candidate_count"],
        "selected_hidden_dim": pair["comparison"]["arms"]["candidate"]["model"]["hidden_dim"],
        "formal_freeze_status": pair["formal_freeze_status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
