"""A per-microbatch depth rule draws one depth per accumulation step. At the same global
batch, changing the local batch changes the accumulation, and with it the depth sequence.
The continuation identity records both the spec's local batch (``recipe_metadata``) and
the launch's ``accumulation``, so it refuses such a strict resume; the same local batch
resumes bitwise. A world-size change is not exercised: these tests run on one rank.

Runs go through the launcher and LoopTrainer on one CPU gloo rank (test_fork's harness).
Global batch 2 blocks: local batch 1 gives 2 accumulation steps, local batch 2 gives 1.
"""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchtitan")

import recipe_experiments as runner
from test_fork import assert_same_state, lab  # noqa: F401  (lab is the fixture)

from moeloop.recipe_spec import RecipeSpec


def micro_spec(lab, local_batch_size):
    spec = json.loads((lab.root / "parent.json").read_text())
    spec["training"] |= {"depth_rule": "poisson_lognormal_micro_v1", "sampled_visits": []}
    spec["run"]["local_batch_size"] = local_batch_size
    path = lab.root / f"micro-local{local_batch_size}.json"
    path.write_text(json.dumps(spec))
    return path


def test_per_microbatch_depths_refuse_a_changed_local_batch_and_resume_exactly(lab):
    spec = micro_spec(lab, 1)
    runner.run(spec, lab.manifest, lab.root / "micro", gpus=1, stop=4)
    runner.run(spec, lab.manifest, lab.root / "micro-split", gpus=1, stop=2)
    checkpoint = lab.root / "micro-split/checkpoint/step-2"
    # Local batch 1 -> 2 at the same global batch: accumulation 2 -> 1, another depth sequence.
    with pytest.raises(ValueError, match="strict continuation identity changed"):
        runner.run(micro_spec(lab, 2), lab.manifest, lab.root / "micro-accumulation-1", gpus=1,
                   stop=4, resume=checkpoint)
    # Same local batch: 2 + 2 resumed updates equal 4 uninterrupted ones bitwise.
    runner.run(spec, lab.manifest, lab.root / "micro-split", gpus=1, stop=4, resume=checkpoint)
    continuous, resumed = lab.trainers["micro"][0], lab.trainers["micro-split"][-1]
    # The loaded checkpoint equals the state after update 2.
    assert_same_state(resumed.loaded, lab.trainers["micro-split"][0].states[2])
    assert_same_state(resumed.states[4], continuous.states[4])
    # Oracle: one depth per update and accumulation step, as depth_at draws it.
    recipe = RecipeSpec.from_dict(json.loads(spec.read_text())).training
    expected = [[recipe.depth_at(update, micro) for micro in range(2)] for update in range(4)]
    for run in ("micro", "micro-split"):
        lines = (lab.root / run / "metrics.jsonl").read_text().splitlines()
        assert [json.loads(line)["microbatch_depths"] for line in lines] == expected
