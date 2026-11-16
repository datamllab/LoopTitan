"""Method lineage (scripts/lineage.py): every node differs from its parent in exactly its declared model fields."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import lineage

ROOT = Path(__file__).resolve().parents[2]
NODES = lineage.lineage()
FAMILIES = ("phase1", "shape", "untie", "p05", "p1", "methods")


@pytest.mark.parametrize("node", NODES)
def test_each_node_differs_from_its_parent_in_exactly_the_declared_fields(node):
    parent, fields, _ = NODES[node]
    if parent is None:
        assert node == lineage.ROOT_NODE and fields == ()
        return
    new, old = lineage.model(node), lineage.model(parent)
    assert {field for field in new if new[field] != old[field]} == set(fields)


@pytest.mark.parametrize("node", NODES)
def test_every_recipe_file_of_a_node_has_its_model(node):
    files = lineage.recipe_files(node)
    assert files
    assert {json.dumps(json.loads(path.read_text())["model"], sort_keys=True) for path in files} == {
        json.dumps(json.loads(files[0].read_text())["model"], sort_keys=True)}


def test_the_tree_is_connected_to_the_root():
    for node in NODES:
        seen = [node]
        while NODES[seen[-1]][0] is not None:
            assert NODES[seen[-1]][0] not in seen, f"cycle through {seen}"
            seen.append(NODES[seen[-1]][0])
        assert seen[-1] == lineage.ROOT_NODE


def test_every_recipe_of_the_released_families_is_a_node():
    files = {path for family in FAMILIES for path in (ROOT / "recipes" / family).glob("*.json")}
    assert files == {path for node in NODES for path in lineage.recipe_files(node)}


def test_the_command_prints_the_tree_and_writes_the_json(tmp_path):
    out = tmp_path / "lineage.json"
    printed = subprocess.run([sys.executable, str(ROOT / "scripts/lineage.py"), "--json", str(out)],
                             env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True,
                             check=True).stdout.splitlines()
    assert printed[0] == lineage.ROOT_NODE and len(printed) == len(NODES)
    assert sorted(line.split()[0] for line in printed) == sorted(NODES)
    document = json.loads(out.read_text())
    assert document["root"] == lineage.ROOT_NODE and set(document["nodes"]) == set(NODES)
    meta = document["nodes"]["methods/meta"]
    assert meta["parent"] == "phase1/m0" and meta["changes"]["num_visits"] == {"old": 1, "new": 4}
