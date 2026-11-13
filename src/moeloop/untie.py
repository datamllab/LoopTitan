"""Untie a fully shared looped model: every core visit gets its own copy of the core cells.

A stack-order model with C core cells visited R times, every owner shared, computes the
same function as the model with C·R core cells visited once whose cell v·C + c is a copy
of cell c: the call of core cell c in visit v becomes core layer v·C + c. ``untied`` gives
that model's spec, ``untie_state`` copies state entries by this map and
``untie_checkpoint`` converts a full training checkpoint, optimizer state included.

Only the plain residual stream keeps the function: a Cell's residual scale is a constant
per region, but per-visit core scales (``visit_residual_scales``) differ between visits, the
EMA memory cell scales by ``residual_gain(num_cells, depth)``, and input injection,
``visit_final_norm`` and exit distributions act between visits.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .config import ModelConfig

CORE_CELL = re.compile(r"(?:^|\.)core\.(\d+)\.")  # the cell index of a core-cell key, also inside optimizer keys
DESCRIPTION = ("name", "source", "preserved", "adaptations")


def untied(spec: dict[str, Any]) -> dict[str, Any]:
    """The untied form of a spec (``RecipeSpec.to_dict()``): C·R core cells, one visit, the rest unchanged."""
    model, training = spec["model"], spec["training"]
    if (model["order"] != "stack" or ModelConfig.from_dict(model).independent_visits or model["num_visits"] < 2
            or model["state_update"] != "residual" or model["input_injection"] != "none"
            or model.get("visit_final_norm") or model.get("visit_residual_scales") or model["exit_distribution"]
            or training["num_visits"] != model["num_visits"] or training["sampled_visits"]
            or training["grad_visits"] is not None or training["depth_rule"] != "uniform_update_v1"):
        raise ValueError("untie needs a fully shared stack-order source with at least 2 visits on the plain "
                         "residual stream, trained at its fixed depth with full BPTT")
    return spec | {"model": model | {"num_cells": model["num_cells"] * model["num_visits"], "num_visits": 1},
                   "training": training | {"num_visits": 1}, "inference": spec["inference"] | {"num_visits": 1}}


def check_untied(source: dict[str, Any], target: dict[str, Any]) -> None:
    """Refuse a target spec that is not ``untied(source)`` apart from its description and ``run.untie``."""
    def comparable(spec):
        kept = {key: value for key, value in spec.items() if key not in DESCRIPTION}
        return kept | {"run": {key: value for key, value in spec["run"].items() if key != "untie"}}
    if comparable(target) != comparable(untied(source)):
        raise ValueError("the untie target is not the source's untied form: only name, description and "
                         "run.untie may differ; data, optimizer and schedule stay the source's")


def untie_state(state: dict[str, Any], *, num_cells: int, num_visits: int) -> dict[str, Any]:
    """Each core cell c's entries again as cells v·num_cells + c, one per visit v; other entries unchanged.

    Works on any flat FQN-keyed state: model, optimizer (``state.<fqn>.exp_avg``,
    ``param_groups.<fqn>.lr``), EMA, or a full DCP checkpoint's keys. Values are shared, not copied.
    """
    result = {}
    for key, value in state.items():
        match = CORE_CELL.search(key)
        if match is None:
            result[key] = value
            continue
        for visit in range(num_visits):
            result[key[:match.start(1)] + str(visit * num_cells + int(match[1])) + key[match.end(1):]] = value
    return result


def untie_checkpoint(source: Path, target: Path, *, num_cells: int, num_visits: int) -> None:
    """Save the full DCP checkpoint ``source`` at ``target`` with its core cells untied, in one process.

    Every entry is loaded whole on CPU and saved under its untied keys; dataloader,
    LR scheduler and train state (step, RNG, continuation payloads) keep their keys.
    The trainer reshards the plain tensors at load like any DCP checkpoint.
    """
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = dcp.FileSystemReader(source)
    state = {key: torch.empty(entry.size, dtype=entry.properties.dtype) if isinstance(entry, TensorStorageMetadata)
             else None for key, entry in reader.read_metadata().state_dict_metadata.items()}
    # Flat keys in, flat keys out. With either flattening the planner works on a copy of ``state``:
    # tensors still load in place, but byte entries (dataloader, train state) land in the copy and stay None.
    planner = DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False)
    dcp.load(state, storage_reader=reader, planner=planner, no_dist=True)
    dcp.save(untie_state(state, num_cells=num_cells, num_visits=num_visits), checkpoint_id=target,
             planner=DefaultSavePlanner(flatten_state_dict=False), no_dist=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Untie the core cells of one full DCP checkpoint.")
    parser.add_argument("--from", dest="source", type=Path, required=True,
                        help="<run>/checkpoint/step-S; the cell counts come from <run>/spec.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    model = json.loads((args.source.parent.parent / "spec.json").read_text())["model"]
    untie_checkpoint(args.source, args.output, num_cells=model["num_cells"], num_visits=model["num_visits"])


if __name__ == "__main__":
    main()
