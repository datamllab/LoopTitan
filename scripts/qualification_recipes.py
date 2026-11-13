"""Tiny native recipes for a bounded, already allocated GPU qualification job.

torchrun --nproc-per-node=4 -m moeloop.train \
    --module qualification_recipes --config smoke

Add the checkout's scripts directory to PYTHONPATH. The caller owns allocation,
the output directory and the process time limit; this module allocates no job.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from moeloop.model import ModelConfig
from moeloop.training import TrainingRecipe


VARIANTS = ("shared", "partial_attention", "partial_router", "expert_tying", "injected", "dense")


def variant_config(name: str) -> tuple[ModelConfig, TrainingRecipe]:
    if name not in VARIANTS:
        raise ValueError(f"qualification variant must be one of {VARIANTS}, got {name!r}")
    model = ModelConfig(vocab_size=64, dim=64, hidden_dim=128, num_heads=4,
                        num_kv_heads=2, num_experts=4, top_k=2,
                        num_cells=2, num_visits=2, max_seq_len=32, qk_norm=True)
    recipe = TrainingRecipe(num_visits=2, balance_coeff=0.01)
    if name == "partial_attention":
        model = replace(model, attention_per_visit=True)
    elif name == "partial_router":
        model = replace(model, router_per_visit=True)
    elif name == "expert_tying":
        model = replace(model, order="layer", attention_per_visit=True,
                        router_per_visit=True, norm_per_visit=True)
    elif name == "injected":
        model = replace(model, input_injection="linear", initial_state="zero",
                        prefix_cells=1, suffix_cells=1)
        # First two updates use depths [3, 2], actually exercising depth changes.
        recipe = TrainingRecipe(num_visits=2, sampled_visits=(2, 3), grad_visits=1,
                                depth_seed=0, balance_coeff=0.01)
    elif name == "dense":
        model = replace(model, num_experts=0)
        recipe = replace(recipe, balance_coeff=0.0)
    recipe.validate_model(model)
    return model, recipe


def _required_directory(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"set {name} explicitly for the qualification job")
    return Path(value).expanduser().resolve()


def smoke():
    """Return the existing native trainer with tiny bounded qualification inputs."""
    from moeloop.recipe_spec import RecipeSpec
    from moeloop.recipes import trainer_config

    variant = os.environ.get("MOELOOP_QUAL_VARIANT", "shared")
    model, recipe = variant_config(variant)
    prepared = _required_directory("MOELOOP_DATA")
    output = _required_directory("MOELOOP_OUTPUT")
    steps = int(os.environ.get("MOELOOP_STEPS", "2"))
    if steps not in (2, 4):
        raise ValueError("qualification MOELOOP_STEPS must be 2 or 4")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    # Two sequences per rank, one per microbatch; the eight CE chunks come from
    # RUN_DEFAULTS, native experts and FlexAttention from model_spec.
    spec = RecipeSpec(f"qualification_{variant}", model, recipe, run={
        "seq_len": 32, "global_batch_positions": 2 * world_size * 32, "local_batch_size": 1,
        "successful_updates": steps, "schedule_total_steps": 16,
        "schedule": {"warmup_steps": 1, "decay_ratio": None, "decay_type": "cosine", "min_lr_factor": 0.1},
        "checkpoint_interval": 1, "keep_latest_k": 0, "validation_interval": 1, "validation_global_batches": 1,
        "activation_checkpointing": "full_cells" if variant in ("partial_attention", "injected") else "none"})
    trainer = trainer_config(spec)
    trainer.dump_folder = str(output)
    trainer.hf_assets_path = str(prepared / "tokenizer")
    trainer.dataloader = replace(trainer.dataloader, manifest_path=str(prepared / "manifest.json"), num_workers=0)
    trainer.validator.dataloader = replace(trainer.validator.dataloader,
        manifest_path=str(prepared / "manifest.json"), num_workers=0)
    resume = os.environ.get("MOELOOP_RESUME")
    if resume:
        trainer.checkpoint = replace(trainer.checkpoint,
            initial_load_path=str(Path(resume).expanduser().resolve()))
    trainer.__post_init__()
    return trainer
