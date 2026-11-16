"""compare_recipes.py: each action takes its own recipe count and says what to use instead."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def compare(*args):
    return subprocess.run([sys.executable, "scripts/compare_recipes.py", *args], cwd=ROOT,
                          capture_output=True, text=True)


def test_pair_with_recipes_points_to_compare():
    # pair is the fixed SMELT budget screen; two recipes are budget-matched by compare.
    result = compare("pair", "recipes/p1_moe.json", "recipes/p1_smelt.json")
    assert result.returncode == 2 and not result.stdout
    assert "pair takes no recipe arguments" in result.stderr
    assert "compare REFERENCE CANDIDATE" in result.stderr


def test_pair_without_recipes_is_the_fixed_screen():
    from moeloop.comparison import smelt_budget_pair

    result = compare("pair")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads(json.dumps(smelt_budget_pair(seq_len=2048)))
