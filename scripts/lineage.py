#!/usr/bin/env python3
"""Print the method lineage of the published recipes: python scripts/lineage.py [--json PATH]

Every node is a frozen recipe arm, recipes/<family>/<arm>-<optimizer>.json (its optimizer files
share one model). Its parent is the configuration it builds on, and it changes the listed
ModelConfig fields; the root is the ordinary MoE, phase1/m0. The study families keep their own
declarations (ARMS of phase1_specs.py, shape_specs.py and p05_specs.py): a P2 base recipe stands
for the Phase 1 arm with its model. The other nodes are declared in DECLARED, each parent chosen
from what the method's paper and constructor build on, which is not always the configuration
nearest by field count: untie/u0 is one field from phase1/m4 but is M3 untied, and
methods/expert-tying has phase1/m2's model but ties the layers of an ordinary MoE.
tests/comparison/test_lineage.py checks every node against its recipe files. --json writes the
nodes, parents and changed fields with their old and new values.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from moeloop.config import ModelConfig
import p05_specs
import phase1_specs
import shape_specs

ROOT = Path(__file__).resolve().parents[1]
ROOT_NODE = "phase1/m0"
LOOP = ("prefix_cells", "num_cells", "suffix_cells", "num_visits")  # a loop's split and visit count
# ModelConfig sets visit_capacity to num_visits whenever an owner is per visit; no declaration sets it.
DERIVED = ("visit_capacity",)
BASES = {"p2_moe": "phase1/m0", "p2_smelt": "phase1/m3"}  # the Phase 1 arm with each P2 base's model
# node: (parent, the model fields it changes, why this parent); checked against the recipe files
DECLARED = {
    "phase1/m3": ("phase1/m0", ("dim", "hidden_dim", "num_heads", "num_experts", *LOOP, "core"),
                  "SMELT runs the middle half of an ordinary MoE twice, at P2's matched width and experts."),
    "untie/u0": ("phase1/m3", ("num_cells", "num_visits"), "M3 untied: each core visit gets its own cells."),
    "p1/m0": ("phase1/m0", ("dim", "hidden_dim", "num_heads", "num_cells"), "The ordinary MoE at P1's 0.1B scale."),
    "p1/m3": ("p1/m0", ("dim", "hidden_dim", "num_heads", "num_experts", *LOOP, "core"),
              "SMELT on P1's ordinary MoE, at P1's matched width and experts."),
    "methods/smelt-v3": ("phase1/m3", ("num_cells", "num_visits", "core"), "SMELT's P2 core run 3 times."),
    "methods/smelt-v4": ("phase1/m3", ("num_cells", "num_visits", "core"), "SMELT's P2 core run 4 times."),
    "methods/latentloop-residual": ("phase1/m3", ("visit_residual_scales", "core"),
        "LatentLoop Section 3.1's per-visit residual on SMELT's loop (latentloop_residual_spec builds on smelt_spec)."),
    "methods/m0-loop": ("phase1/m0", (*LOOP, "core"), "m0's cells in SMELT's loop, the budget-convention control."),
    "methods/meta": ("phase1/m0", ("num_experts", *LOOP),
                     "A shared middle stack between an independent prelude and coda of an ordinary MoE."),
    "methods/foil": ("phase1/m0", ("attention_per_visit", *LOOP, "prefix", "core", "suffix"),
                     "Foil flattens the core of its Base, an ordinary MoE with prelude and coda: (E,D,L)->(2E,D/2,2L)."),
    "methods/expert-tying": ("phase1/m0", ("num_experts", "num_cells", "num_visits", "order",
                                           "attention_per_visit", "router_per_visit", "norm_per_visit"),
                             "Tying shares one expert bank across consecutive layers of an ordinary MoE."),
    "methods/huginn": ("phase1/m0", ("hidden_dim", "num_heads", "num_experts", *LOOP, "input_injection",
                                     "initial_state"),
                       "A recurrent core between prelude and coda, with a live input anchor and a random latent start."),
    "methods/loom": ("phase1/m0", ("num_experts", "num_cells", "num_visits", "router_per_visit",
                                   "state_update"),
                     "The whole ordinary MoE stack looped, with per-visit routers and two EMA memories."),
    "methods/ouro": ("phase1/m0", ("dim", "hidden_dim", "num_heads", "num_experts", "num_cells", "num_visits",
                                   "exit_distribution"),
                     "The whole ordinary MoE stack looped, with a learned exit distribution over visits."),
    "methods/sandwich": ("methods/huginn", ("hidden_dim", "num_heads", "num_experts", "top_k", *LOOP,
                                            "input_injection", "injection_dt", "initial_state"),
                         "Part I's MoE results report the Huginn MoE; this path replaces its linear injection and "
                         "random start with diagonal-decay injection and z_0 = e."),
    "methods/relaxed": ("phase1/m0", ("num_experts", "num_cells", "num_visits", "lora_attention_rank"),
                        "The ordinary MoE stack shared in a cycle (a Recursive Transformer) with per-visit LoRA."),
}


def family(parent: str, changes: dict, note: str) -> tuple:
    """A family declaration as a node; a declared value equal to the parent's (m1s's stack order) is no change."""
    old = model(parent)
    return parent, tuple(field for field, value in changes.items() if value != old[field]), note


def lineage() -> dict[str, tuple]:
    """node: (parent, changed fields, note); the root's parent is None."""
    nodes = {ROOT_NODE: (None, (), "The ordinary MoE: 12 layers of 64 experts, top-8 (P2).")}
    for arm, (base, changes, _, _, note) in phase1_specs.ARMS.items():
        if arm in phase1_specs.EXPERT_LR:  # the same model as its source arm; only the expert LR differs
            nodes[f"phase1/{arm}"] = (f"phase1/{phase1_specs.EXPERT_LR[arm][0]}", (), note)
        elif changes:
            nodes[f"phase1/{arm}"] = family(BASES[base], changes, note)
    for arm, (source, changes, note) in shape_specs.ARMS.items():
        nodes[f"shape/{arm}"] = family(f"phase1/{source}", changes, note)
    for arm, (base, changes, note) in p05_specs.ARMS.items():
        nodes[f"p05/{arm}"] = family(BASES[base], changes, note)
    return nodes | DECLARED


def recipe_files(node: str) -> list[Path]:
    return [path for optimizer in ("adamw", "muon") if (path := ROOT / "recipes" / f"{node}-{optimizer}.json").is_file()]


def model(node: str) -> dict:
    """Every ModelConfig field of the node's first recipe file, defaults filled in, except DERIVED."""
    values = asdict(ModelConfig.from_dict(json.loads(recipe_files(node)[0].read_text())["model"]))
    return {field: value for field, value in values.items() if field not in DERIVED}


def document() -> dict:
    nodes = {}
    for node, (parent, fields, note) in lineage().items():
        new, old = model(node), parent and model(parent)
        nodes[node] = {"parent": parent, "note": note,
                       "recipes": [str(path.relative_to(ROOT)) for path in recipe_files(node)],
                       "changes": {field: {"old": old[field], "new": new[field]} for field in fields}}
    return {"root": ROOT_NODE, "nodes": nodes}


def tree(node: str = ROOT_NODE, depth: int = 0, nodes: dict | None = None) -> list[str]:
    nodes = nodes or lineage()
    lines = ["  " * depth + node + (f"  [{', '.join(nodes[node][1])}]" if nodes[node][1] else "")]
    for child, (parent, _, _) in nodes.items():
        if parent == node:
            lines += tree(child, depth + 1, nodes)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="also write nodes, parents and changed fields as JSON")
    args = parser.parse_args()
    print("\n".join(tree()))
    if args.json:
        args.json.write_text(json.dumps(document(), indent=2) + "\n")


if __name__ == "__main__":
    main()
