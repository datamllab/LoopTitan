"""Load one ordinary frozen recipe through the installed native trainer.

This config factory owns no training loop or scheduler. An absolute stop only
changes when the native engine returns, never the recipe's schedule horizon.
A spec with ``run.fork`` (written by ``recipe_experiments.py fork``) starts a new
lineage from its parent's checkpoint, given as the resume path.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path


def experiment():
    """``trainer_config(spec)`` with only the launch facts set: output, data, stop and topology."""
    from .recipe_spec import RecipeSpec
    from .recipes import trainer_config

    spec_path = Path(os.environ['MOELOOP_EXPERIMENT_SPEC']).resolve()
    spec = RecipeSpec.from_dict(json.loads(spec_path.read_text()))
    if not spec.run.get('global_batch_positions'):
        raise ValueError('a frozen experiment needs explicit global_batch_positions')
    config = trainer_config(spec)
    manifest = Path(os.environ['MOELOOP_EXPERIMENT_MANIFEST']).resolve()
    if manifest.is_dir():
        manifest = manifest / 'manifest.json'
    stop = int(os.environ.get('MOELOOP_EXPERIMENT_STOP', spec.run['successful_updates']))
    if not 0 < stop <= spec.run['successful_updates']:
        raise ValueError('experiment stop must be inside the frozen recipe horizon')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', str(world)))
    if min(world, local_world) < 1 or world % local_world:
        raise ValueError('invalid native DP/FSDP topology')
    replicate = int(os.environ.get('MOELOOP_EXPERIMENT_REPLICATE', str(world // local_world)))
    if min(world, local_world, replicate) < 1 or world % replicate:
        raise ValueError('invalid native DP/FSDP topology')
    config.training = replace(config.training, steps=stop)
    config.parallelism = replace(config.parallelism,
        data_parallel_replicate_degree=replicate, data_parallel_shard_degree=world // replicate)
    config.dump_folder = str(Path(os.environ['MOELOOP_EXPERIMENT_OUTPUT']).resolve())
    config.hf_assets_path = str(manifest.parent / 'tokenizer')
    config.dataloader = replace(config.dataloader, manifest_path=str(manifest), num_workers=0)
    config.validator.dataloader = replace(config.validator.dataloader,
        manifest_path=str(manifest), num_workers=0)
    config.fork = spec.run.get('fork', {})
    if config.fork and 'MOELOOP_EXPERIMENT_RESUME' not in os.environ:
        raise ValueError('a fork starts from its parent checkpoint; launch it with recipe_experiments.py fork')
    config.checkpoint = replace(config.checkpoint,
        initial_load_path=os.environ.get('MOELOOP_EXPERIMENT_RESUME', config.checkpoint.initial_load_path))
    config.__post_init__()
    return config
