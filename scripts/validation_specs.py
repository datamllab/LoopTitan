"""Write the GPU-validation specs of #49: python scripts/validation_specs.py --output DIR

Both take Track S's reference recipe (recipes/c_a0_r0.json) over its own 2048 updates,
with Track S's warmup-stable-decay schedule rule (2% warmup, constant peak, the last 20%
linear to 0) and Track S's fixed optimizer settings. The reference's cosine schedule has
no stable phase, so a fork from the checkpoint at update 150 would be refused; under this
schedule update 150 is at peak. b5-adamw.json keeps the recipe's AdamW learning rate;
b5-muon.json uses Muon at the middle of Track S's Muon grid.

The local batch is fixed so a 4-GPU and an 8-GPU run see the same micro-batches: the
router aux loss is a mean over each rank's micro calls (local_micro_call_mean), so it is
independent of the GPU count only when the local batch is.
"""
import argparse
import json
from pathlib import Path

from moeloop.bench import load_tracks

ROOT = Path(__file__).resolve().parents[1]
LOCAL_BATCH = 2  # divides the 16-sequence global batch on 4 and on 8 GPUs


def specs() -> dict[str, dict]:
    base = json.loads((ROOT / "recipes/c_a0_r0.json").read_text())
    track = load_tracks()["S"]
    total = base["run"]["successful_updates"]
    schedule = {key: track.schedule[key] for key in ("name", "coordinate", "decay_type", "decay_ratio", "min_lr_factor")}
    schedule |= {"total_steps": total, "warmup_steps": round(track.schedule["warmup_fraction"] * total)}
    base["run"] |= {"local_batch_size": LOCAL_BATCH, "schedule": schedule}
    fixed, lr = track.optimizers["fixed"], track.lr_grid["Muon"][1]
    adamw = base["optimizer"] | fixed["AdamW"]
    muon = (base["optimizer"] | {key: value for key, value in fixed["Muon"].items() if key != "muon"}
            | {"name": "Muon", "base_lr": lr, "muon": fixed["Muon"]["muon"] | {"lr": lr}})
    return {"b5-adamw": base | {"name": "b5-adamw", "optimizer": adamw},
            "b5-muon": base | {"name": "b5-muon", "optimizer": muon}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, spec in specs().items():
        path = args.output / f"{name}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
