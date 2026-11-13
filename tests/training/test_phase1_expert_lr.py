"""The Phase 1 step-size arms (#264): expert_lr_multiplier sqrt(2) reaches exactly the core experts' groups."""
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan")

from moeloop import recipes
from moeloop.recipe_spec import RecipeSpec
from moeloop.titan import TitanModel, configure_optimizer

PHASE1 = Path(__file__).resolve().parents[2] / "recipes" / "phase1"
CORE_EXPERT = re.compile(r"^core\.core\.\d+\.experts\.")
OUTSIDE_EXPERT = re.compile(r"^core\.(prefix|suffix)\.\d+\.experts\.")
# arm: (source arm, core layers that own experts)
ARMS = {"m7-elr": ("m7", 6), "m4-elr": ("m4", 12)}


def group_settings(arm):
    """Every parameter's (lr, weight_decay) from the optimizer a LoopTrainer builds, on the meta device."""
    spec = RecipeSpec.from_dict(json.loads((PHASE1 / f"{arm}-adamw.json").read_text()))
    config = recipes.trainer_config(spec)
    with torch.device("meta"):  # as LoopTrainer, which copies expert_lr_multiplier into the model config
        model = TitanModel(replace(config.model_spec.model, flex_attention=False,
                                   expert_lr_multiplier=config.expert_lr_multiplier))
    optimizers = replace(config.optimizer, implementation="for-loop").build(model_parts=[model])
    configure_optimizer(optimizers, [model], None)
    settings = {name: (group["lr"], group["weight_decay"]) for optimizer in optimizers.optimizers
                for group in optimizer.param_groups for name in group["param_names"]}
    assert len(settings) == len(list(model.parameters()))  # every parameter is in exactly one group
    return spec.optimizer, settings


@pytest.mark.parametrize("arm", ARMS)
def test_sqrt2_reaches_exactly_the_core_experts(arm):
    source, core_layers = ARMS[arm]
    optimizer, settings = group_settings(arm)
    reference_optimizer, reference = group_settings(source)
    base_lr = optimizer["base_lr"]
    assert optimizer["expert_lr_multiplier"] == math.sqrt(2) and reference_optimizer["expert_lr_multiplier"] == 1.0
    assert base_lr == reference_optimizer["base_lr"] and settings.keys() == reference.keys()
    core = {name for name in settings if CORE_EXPERT.match(name)}
    assert {name.split(".")[2] for name in core} == {str(i) for i in range(core_layers)}  # w1, w2, w3 of each layer
    assert len(core) == 3 * core_layers
    for name, (lr, decay) in settings.items():
        assert decay == reference[name][1], name  # weight decay is unchanged
        assert lr == pytest.approx(base_lr * math.sqrt(2) if name in core else base_lr, rel=1e-12), name
        assert reference[name][0] == base_lr, name
    outside = [name for name in settings if OUTSIDE_EXPERT.match(name)]
    assert len(outside) == 3 * 6 and {settings[name][0] for name in outside} == {base_lr}  # 3 prefix + 3 suffix layers
    assert {settings[name][0] for name in settings if name not in core} == {base_lr}  # attention, routers, norms, embeddings
