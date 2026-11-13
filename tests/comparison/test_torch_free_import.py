"""Recipe specs, costs, comparison and the atlas import and run where torch is not installed.

Each check runs in a fresh, empty virtual environment, as the package CI job does,
with the source tree on PYTHONPATH; ``find_spec("torch") is None`` holds there even
when the test itself runs in an environment with torch.
"""
import os
import subprocess
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NO_TORCH = 'import importlib.util; assert importlib.util.find_spec("torch") is None\n'


@pytest.fixture(scope="module")
def clean_python(tmp_path_factory):
    directory = tmp_path_factory.mktemp("clean")
    venv.create(directory, with_pip=False)
    return directory / "bin" / "python"


def run(python, *args):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([str(python), *args], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_specs_costs_and_comparison_run_without_torch(clean_python):
    program = NO_TORCH + """
from moeloop import comparison, costs, recipes
from moeloop.config import ModelConfig, RegionConfig, TrainingRecipe
from moeloop.recipe_spec import RecipeSpec
spec = recipes.p1_smelt_spec()
assert RecipeSpec.from_dict(spec.to_dict()) == spec
assert type(spec.model) is ModelConfig and type(spec.model.core) is RegionConfig
assert type(spec.training) is TrainingRecipe
pair = comparison.smelt_p1_pair()
print(pair["candidates"][pair["selected_candidate_index"]]["hidden_dim"])
assert costs.estimate_cost(spec, seq_len=4096)["flops"]["training_per_token"] > 0
"""
    # smelt_p1_pair documents its dense-S^2 screen as selecting F424.
    assert run(clean_python, "-c", program).strip() == "424"


def test_atlas_runs_without_torch_and_reproduces_the_report_csv(clean_python, tmp_path):
    run(clean_python, "-c", NO_TORCH)
    run(clean_python, "scripts/recipe_atlas.py", "--reference", "recipes/p1_moe.json", "--output", str(tmp_path))
    # The report CSV was written before sandwich and relaxed joined the atlas; their rows come last.
    published = (ROOT / "eval/atlas/atlas-p1.csv").read_bytes()
    fresh = (tmp_path / "atlas.csv").read_bytes()
    assert fresh.startswith(published)
    assert {line.split(b",")[0] for line in fresh[len(published):].splitlines()} == {b"sandwich", b"relaxed"}
