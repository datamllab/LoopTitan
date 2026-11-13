"""Actual two-rank CPU FSDP/AC/chunked-loss gradient semantics."""
import os
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from moeloop.model import ModelConfig, RecursiveLM
from moeloop.titan import TitanModel, canonical_model_state, parallelize_model, LoopChunkedLoss
from moeloop.training import TrainingRecipe, objective
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.config import TrainingConfig, ParallelismConfig, CompileConfig
from torchtitan.distributed.utils import set_spmd_backend


def worker(rank, world, init):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=init, rank=rank, world_size=world)
    mesh = init_device_mesh('cpu', (world,), mesh_dim_names=('fsdp',))
    pdims = SimpleNamespace(tp=1, ep=1, cp=1, pp=1, dp_replicate_enabled=False, get_mesh=lambda names: mesh)
    set_spmd_backend('partial_dtensor')
    for case in ['dense', 'sparse', 'partial', 'truncated', 'memory', 'expected', 'latent',
                 'memory_sharded', 'expected_sharded']:
        # One expert across two shards exercises the alternative expert axis;
        # larger M/H and independent attention also change owner/replay order.
        sharded = case.endswith('_sharded')
        mode = case.removesuffix('_sharded')
        policies = [None, FullAC.Config()]
        if mode in ('dense', 'sparse', 'truncated'):
            policies.append(SelectiveAC.Config())
        if mode == 'latent':
            policies = [FullAC.Config()]
        for ac in policies:
            cfg = ModelConfig(vocab_size=19, dim=16, hidden_dim=24, num_heads=2, num_kv_heads=1,
                              num_experts=1 if sharded else 0 if mode == 'dense' else 2,
                              top_k=1 if sharded else 2, num_cells=3 if sharded else 2,
                              num_visits=4 if sharded else 3,
                              attention_per_visit=sharded or mode == 'partial', prefix_cells=1, suffix_cells=1,
                              input_injection='linear' if mode in ('truncated', 'latent') else 'none',
                              initial_state='random' if mode == 'latent' else 'input',
                              state_update='ema' if mode == 'memory' else 'residual',
                              router_per_visit=mode == 'memory', exit_distribution=mode == 'expected')
            torch.manual_seed(37)
            bridge = TitanModel(TitanModel.Config(core=cfg, native_experts=False, flex_attention=False))
            reference = RecursiveLM(cfg)
            reference.load_state_dict(canonical_model_state(bridge.state_dict()))
            bridge = parallelize_model(bridge, parallel_dims=pdims,
                        training=TrainingConfig(mixed_precision_param='float32', mixed_precision_reduce='float32'),
                        parallelism=ParallelismConfig(spmd_backend='partial_dtensor'),
                        compile_config=CompileConfig(enable=False), ac_config=ac, dump_folder='/tmp')
            # Gloo lacks NCCL PREMUL_SUM; preserve native divide_factor=1 with SUM.
            if world > 1:
                from torch.distributed.fsdp import FSDPModule
                for module in bridge.modules():
                    if isinstance(module, FSDPModule):
                        module.set_force_sum_reduction_for_comms(True)
            bridge._skip_lm_head = True
            recipe = TrainingRecipe(num_visits=cfg.num_visits, grad_visits=2 if mode in ('truncated', 'latent') else None,
                                    balance_coeff=0.17, z_loss_coeff=0.03,
                                    objective='expected_ce' if mode == 'expected' else 'terminal_ce',
                                    entropy_coeff=.1 if mode == 'expected' else 0)
            lossfn = LoopChunkedLoss(LoopChunkedLoss.Config(num_chunks=2, recipe=recipe))
            lossfn.set_lm_head(bridge.lm_head)
            lossfn.dp_degree = world
            lossfn.accumulation = 2
            torch.manual_seed(101 + rank)
            for update in range(2):
                bridge.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                denominator = torch.tensor(7 * world * 2)
                for micro in range(2):
                    tokens = torch.randint(0, 19, (2, 4))
                    labels = torch.randint(0, 19, (2, 4))
                    labels[0, 0] = -100
                    rng = torch.get_rng_state()
                    actual, _ = lossfn(bridge(tokens, grad_visits=recipe.grad_visits), labels, denominator)
                    torch.set_rng_state(rng)
                    expected, _ = objective(reference(tokens, grad_visits=recipe.grad_visits), labels, denominator,
                                            recipe, dp_degree=world, accumulation=2, head=reference.lm_head)
                    actual.backward()
                    expected.backward()
                reference_parameters = dict(reference.named_parameters())
                for name, parameter in bridge.core.named_parameters():
                    other = reference_parameters[name.replace('._checkpoint_wrapped_module', '')]
                    if parameter.grad is None or other.grad is None:
                        assert parameter.grad is None and other.grad is None, name
                    else:
                        dist.all_reduce(other.grad)
                        torch.testing.assert_close(parameter.grad.full_tensor(), other.grad,
                                                   rtol=2e-5, atol=3e-7, msg=name)
                if rank == 0:
                    print(case, 'none' if ac is None else ac._owner.__name__, 'update', update, 'OK', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    world = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    with tempfile.TemporaryDirectory() as tmp:
        mp.start_processes(worker, args=(world, 'file://' + tmp + '/init'), nprocs=world, start_method='fork')
