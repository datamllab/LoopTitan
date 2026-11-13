#!/usr/bin/env python3
"""Run one frozen pretraining recipe on already assigned GPUs, fork one of a run's
full checkpoints into a new lineage with a new decay tail, or untie a fully shared
run's checkpoint and continue it with every core visit owning its own cells.

Each concurrent arm/worktree needs its own output directory and rendezvous.
Multi-node launchers call this once per node with standard torchrun topology
variables. No cluster allocator, service or CVSM protocol is required.

``run --lr`` sets the peak learning rate (``optimizer.base_lr``, and ``optimizer.muon.lr``
for Muon) before the spec is written to the output. ``--evaluate`` exports the final
checkpoint of a complete schedule to ``<output>/hf`` and measures BPB on the full
validation and test splits at the recipe's inference depth. If export or evaluation fails
after training, rerun those ``python -m moeloop.export``/``moeloop.evaluate`` commands by
hand with the same arguments. Under ``scripts/run.py``, node 0 writes the launched spec
and topology to ``$RUN_DIR/config.json`` and the results to ``$RUN_DIR/metrics.json``:
completed updates, BPB, and the GPU hours of the whole launch (training, export and
evaluation).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from recipe_qualification import digest, run_command, snapshot_source, write_json


def topology_arguments(world):
    nodes = int(os.environ.get('NNODES', '1'))
    local = int(os.environ.get('NPROC_PER_NODE', str(world)))
    rank = int(os.environ.get('NODE_RANK', '0'))
    if min(nodes, local, world) < 1 or nodes * local != world or not 0 <= rank < nodes:
        raise ValueError('requested GPU count differs from launcher topology')
    if nodes == 1:
        return ['--standalone', f'--nproc-per-node={local}'], rank
    return [f'--nnodes={nodes}', f'--nproc-per-node={local}', f'--node-rank={rank}',
            f'--master-addr={os.environ["MASTER_ADDR"]}',
            f'--master-port={os.environ["MASTER_PORT"]}'], rank


def prepare_run(directory, spec, manifest, *, world, local_batch, runtime):
    directory = Path(directory).resolve()
    batch = spec.run['global_batch_positions'] // spec.run['seq_len']
    if batch % (world * local_batch):
        raise ValueError('batch is not divisible by world and local batch')
    identity = {'method': spec.name, 'pilot': False,
        'steps': spec.run['successful_updates'], 'seq_len': spec.run['seq_len'],
        'global_batch_size': batch, 'world_size': world, 'local_batch_size': local_batch,
        'gradient_accumulation_steps': batch // (world * local_batch),
        'manifest': str(manifest.path), 'manifest_fingerprint': manifest.fingerprint}
    if (directory / 'request.json').is_file():
        request = json.loads((directory / 'request.json').read_text())
        if (any(request.get(k) != v for k, v in identity.items())
                or json.loads((directory / 'spec.json').read_text()) != spec.to_dict()
                or request.get('runtime') != runtime):
            raise ValueError('existing experiment output identity differs; use a new directory')
        return request
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('nonempty output has no experiment identity; use a new directory')
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'spec.json', spec.to_dict())
    request = {**identity, 'source': snapshot_source(directory / 'source'),
               'spec_sha256': digest(directory / 'spec.json'), 'runtime': runtime,
               'activation_checkpointing': spec.run.get('activation_checkpointing', 'none')}
    write_json(directory / 'request.json', request)
    return request



@contextmanager
def coordinator_lock(directory, node_rank):
    """Fail fast when two independent arms select the same output directory."""
    if node_rank:
        yield
        return
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / ('.' + directory.name + '.lock')).open('a') as lock:
        try:
            fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('experiment output has another active coordinator') from error
        yield


def runtime_identity(python):
    code = '''import json, sys, torch
value = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda)
if torch.cuda.is_available():
    value.update(gpu_name=torch.cuda.get_device_name(0),
                 compute_capability=list(torch.cuda.get_device_capability(0)))
print(json.dumps(value))'''
    return json.loads(subprocess.check_output([python, '-c', code], text=True))


def protect_resume(directory, checkpoint, *, node_rank, timeout=3600):
    """Retain an immutable full DCP outside native rotation before launch.

    The coordinator calls this while holding the output lock. Only the small
    metadata file is hashed; checkpoint shards are copied unchanged. Publication
    renames a complete directory, so peers cannot observe a partial checkpoint.
    Failed partial copies remain available for diagnosis.
    """
    checkpoint = Path(checkpoint).resolve()
    if not re.fullmatch(r'step-\d+', checkpoint.name):
        raise ValueError('resume checkpoint must be a native step-N directory')
    target = Path(directory).resolve() / 'protected' / checkpoint.name
    deadline = time.monotonic() + timeout
    if node_rank:
        while not target.is_dir():
            if time.monotonic() >= deadline:
                raise TimeoutError('coordinator did not publish the retained resume checkpoint')
            time.sleep(.1)
    if target.exists():
        receipt_path = target / 'retention.json'
        if not receipt_path.is_file() or not (target / '.metadata').is_file():
            raise ValueError('protected checkpoint identity is incomplete')
        receipt = json.loads(receipt_path.read_text())
        expected = receipt.get('metadata_sha256')
        if (digest(target / '.metadata') != expected
                or receipt.get('protected_path') != str(target)
                or (checkpoint != target and receipt.get('original_path') != str(checkpoint))
                or ((checkpoint / '.metadata').is_file()
                    and digest(checkpoint / '.metadata') != expected)):
            raise ValueError('protected checkpoint identity differs from requested resume')
        return target
    metadata_sha256 = digest(checkpoint / '.metadata')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f'.{checkpoint.name}.partial-', dir=target.parent))
    shutil.copytree(checkpoint, partial, dirs_exist_ok=True)
    if (digest(partial / '.metadata') != metadata_sha256
            or digest(checkpoint / '.metadata') != metadata_sha256):
        raise ValueError('resume checkpoint identity changed during retention')
    write_json(partial / 'retention.json', {'original_path': str(checkpoint),
        'protected_path': str(target), 'metadata_sha256': metadata_sha256})
    partial.rename(target)
    return target


def load_spec(spec_path, lr):
    """The spec at ``spec_path`` with peak LR ``lr`` (and Muon's muon.lr), unless None."""
    from moeloop.recipe_spec import RecipeSpec
    spec = RecipeSpec.from_dict(json.loads(Path(spec_path).read_text()))
    if lr is not None:
        optimizer = spec.optimizer | {'base_lr': lr}
        if optimizer['name'] == 'Muon':
            optimizer['muon'] = optimizer['muon'] | {'lr': lr}
        spec = replace(spec, optimizer=optimizer)
    return spec


def run(spec_path, manifest_path, output, *, gpus, stop=None, resume=None, lr=None,
        evaluate=False, python=None, installed=False):
    return launch(load_spec(spec_path, lr), manifest_path, output, gpus=gpus, stop=stop, resume=resume,
                  evaluate=evaluate, python=python, installed=installed)


def fork(checkpoint, output, *, decay_steps, decay_type, gpus, min_lr_factor=None,
         decay_start=None, evaluate=False, python=None, installed=False):
    """Start a new lineage from a parent's full step-S checkpoint with a new decay tail.

    The parent is the run directory holding the checkpoint (``<run>/checkpoint/step-S``
    or ``<run>/protected/step-S``). The fork's spec is the parent's with a WSD tail of
    ``decay_steps`` updates after ``decay_start`` (default S) toward ``min_lr_factor``
    (default the parent's resolved one). The tail must start at or after both S and
    the end of the parent's warmup: earlier, it would change LRs the parent already
    used, or the native builder would silently shorten it. In the pinned native
    LambdaLR convention update ``decay_start + 1`` still runs at peak and the virtual
    index ``decay_start + decay_steps``, one past the last update, reaches the floor.
    ``run.fork`` records the parent checkpoint, its DCP metadata digest and S. The
    trainer accepts the parent's state only at S, only if nothing but the schedule
    changed (source, data, topology and optimizer included), and only if the new
    schedule matches the parent's LRs through S (``LoopTrainer.check_fork``): a
    stable-phase checkpoint, or a warmup checkpoint with a tail after warmup. The
    parent's outputs are only read. Continue an interrupted fork with the same
    command, or with ``run`` on the fork's own spec.json and checkpoint once the
    parent's is gone.
    """
    from moeloop.recipe_spec import RecipeSpec
    checkpoint = Path(checkpoint).resolve()
    parent = checkpoint.parent.parent
    spec = json.loads((parent / 'spec.json').read_text())
    schedule = json.loads((parent / 'continuation_identity.json').read_text())['lr_scheduler']
    step = int(checkpoint.name.removeprefix('step-'))
    start = step if decay_start is None else decay_start
    warmup = schedule['warmup_steps']
    if start < max(step, warmup):
        raise ValueError(f'a fork tail must start at or after both the fork step {step} '
                         f'and the end of warmup {warmup}; got {start}')
    total = start + decay_steps
    spec['run'] |= {'successful_updates': total, 'schedule_total_steps': total,
        'schedule': {'total_steps': total, 'warmup_steps': warmup,
                     'stable_end_steps': start, 'decay_steps': decay_steps,
                     'decay_ratio': decay_steps / total, 'decay_type': decay_type,
                     'min_lr_factor': schedule['min_lr_factor'] if min_lr_factor is None else min_lr_factor},
        'fork': {'checkpoint': str(checkpoint), 'metadata_sha256': digest(checkpoint / '.metadata'),
                 'step': step}}
    manifest = json.loads((parent / 'request.json').read_text())['manifest']
    return launch(RecipeSpec.from_dict(spec), manifest, output, gpus=gpus, resume=checkpoint,
                  evaluate=evaluate, python=python, installed=installed)


def untie(checkpoint, spec_path, output, *, gpus, lr=None, evaluate=False, python=None, installed=False):
    """Continue a fully shared stack-order run from its full step-S checkpoint, untied.

    The source is the run directory holding the checkpoint (``<run>/checkpoint/step-S``
    or ``<run>/protected/step-S``). ``spec_path`` with ``lr`` applied as for ``run``
    must be the source spec's untied form (``moeloop.untie.check_untied``): core cell
    c of visit v becomes core cell v·C + c, and everything but the name, the
    description and ``run.untie`` is the source's, data, optimizer and schedule
    included. ``run.untie`` records the source spec's name, S, the checkpoint and its
    DCP metadata digest. The coordinator converts the checkpoint once into
    ``<output>/protected/step-S`` (``python -m moeloop.untie``): model and optimizer
    state copied by the cell map, dataloader, RNG and train state unchanged, so update
    S+1 reads the source's update S+1 data. The trainer accepts it only at S and only
    if the rest of the continuation identity, topology included, is the source's,
    apart from the moeloop source digests (``LoopTrainer.check_untie``). The source's outputs are only read. Continue an
    interrupted untie with the same command, or with ``run`` on the untie's own
    spec.json and checkpoint once the source's is gone.
    """
    from moeloop.recipe_spec import RecipeSpec
    from moeloop.untie import check_untied
    checkpoint = Path(checkpoint).resolve()
    parent = checkpoint.parent.parent
    source = json.loads((parent / 'spec.json').read_text())
    spec = load_spec(spec_path, lr).to_dict()
    check_untied(source, spec)
    step = int(checkpoint.name.removeprefix('step-'))
    spec['run'] |= {'untie': {'source': source['name'], 'step': step, 'checkpoint': str(checkpoint),
                              'metadata_sha256': digest(checkpoint / '.metadata')}}
    manifest = json.loads((parent / 'request.json').read_text())['manifest']
    output = Path(output).resolve()
    return launch(RecipeSpec.from_dict(spec), manifest, output, gpus=gpus, resume=output / 'protected' / checkpoint.name,
                  untie=checkpoint, evaluate=evaluate, python=python, installed=installed)


def launch(spec, manifest_path, output, *, gpus, stop=None, resume=None, untie=None, evaluate=False,
           python=None, installed=False):
    """``untie``: a fully shared source checkpoint that the coordinator first converts into ``resume``."""
    from moeloop.data import PreparedManifest
    from moeloop.recipes import local_batch_size
    from qualification import verify_receipt

    started = time.monotonic()
    output = Path(output).resolve()
    manifest = PreparedManifest.load(manifest_path, verify_hashes=False)
    stop = spec.run['successful_updates'] if stop is None else stop
    if not 0 < stop <= spec.run['successful_updates']:
        raise ValueError('stop must lie inside the fixed recipe horizon')
    if evaluate and stop != spec.run['successful_updates']:
        raise ValueError('--evaluate needs the whole schedule; drop --stop')
    if resume is None and spec.run.get('untie'):
        raise ValueError('an untie starts from its converted source checkpoint; launch it with recipe_experiments.py untie')
    launcher, node = topology_arguments(gpus)
    batch = spec.run['global_batch_positions'] // spec.run['seq_len']
    local = local_batch_size(spec, gpus)
    python = python or sys.executable
    runtime = runtime_identity(python)
    run_dir = os.environ.get('RUN_DIR')
    with coordinator_lock(output, node):
        if node == 0:
            request = prepare_run(output, spec, manifest, world=gpus, local_batch=local, runtime=runtime)
            if run_dir:
                values = spec.to_dict()
                config = {key: values[key] for key in
                          ('name', 'model', 'training', 'optimizer', 'inference', 'run')}
                config['launch'] = {'world_size': gpus, 'local_batch_size': local,
                    'gradient_accumulation_steps': batch // (gpus * local),
                    'manifest_fingerprint': manifest.fingerprint}
                write_json(Path(run_dir) / 'config.json', config)
        else:
            deadline = time.monotonic() + 120
            while not (output / 'request.json').is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError('coordinator did not publish the shared experiment request')
                time.sleep(.1)
            request = prepare_run(output, spec, manifest, world=gpus, local_batch=local, runtime=runtime)
        environment = os.environ.copy()
        if installed:
            environment.pop('PYTHONPATH', None)
        else:
            source = output / 'source'
            environment['PYTHONPATH'] = os.pathsep.join([
                str(source / 'src'), str(source / 'scripts'), environment.get('PYTHONPATH', '')])
        environment.update(MOELOOP_EXPERIMENT_SPEC=str(output / 'spec.json'),
            MOELOOP_EXPERIMENT_MANIFEST=str(manifest.path), MOELOOP_EXPERIMENT_OUTPUT=str(output),
            MOELOOP_EXPERIMENT_STOP=str(stop))
        environment.pop('MOELOOP_EXPERIMENT_RESUME', None)
        if resume is None and (output / 'metrics.jsonl').is_file():
            raise ValueError('existing updates require an explicit --resume checkpoint')
        if untie is not None and node == 0 and not resume.exists():
            # Published like protect_resume's copies: a complete directory renamed whole, with its receipt.
            resume.parent.mkdir(parents=True, exist_ok=True)
            partial = Path(tempfile.mkdtemp(prefix=f'.{resume.name}.partial-', dir=resume.parent))
            subprocess.run([python, '-m', 'moeloop.untie', '--from', untie, '--output', partial],
                           env=environment, cwd=output, check=True)
            write_json(partial / 'retention.json', {'original_path': str(untie), 'protected_path': str(resume),
                'metadata_sha256': digest(partial / '.metadata')})
            partial.rename(resume)
        if resume is not None:
            environment['MOELOOP_EXPERIMENT_RESUME'] = str(protect_resume(output, resume, node_rank=node))
        run_command([python, '-m', 'torch.distributed.run', *launcher,
            '-m', 'moeloop.train', '--module', 'moeloop.experiment', '--config', 'experiment'],
            output, env=environment, cwd=output,
            gpu_count=int(os.environ.get('NPROC_PER_NODE', str(gpus))), node_rank=node)
        if node:
            deadline = time.monotonic() + 120
            while True:
                outcome = output / 'outcome.json'
                if outcome.is_file() and json.loads(outcome.read_text()).get('completed_updates') == stop:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('coordinator did not publish the final training outcome')
                time.sleep(.1)
            return verify_receipt(output, expected_steps=stop)
        result = verify_receipt(output, expected_steps=stop)
        metrics = {}
        if evaluate:
            subprocess.run([python, '-m', 'moeloop.export', '--checkpoint',
                result['outcome']['full_checkpoint'], '--output', output / 'hf'],
                env=environment, cwd=output, check=True)
            for split in ('validation', 'test'):
                path = output / f'eval-{split}.json'
                subprocess.run([python, '-m', 'moeloop.evaluate', '--model', output / 'hf',
                    '--manifest', manifest.path, '--split', split,
                    '--seq-len', str(spec.run['seq_len']), '--batch-size', '4',
                    '--num-visits', str(spec.inference.get('num_visits', spec.model.num_visits)),
                    '--device', 'cuda', '--output', path], env=environment, cwd=output, check=True)
                metrics[f'{split}_bpb'] = json.loads(path.read_text())['bpb']
        if run_dir:
            write_json(Path(run_dir) / 'metrics.json', {'completed_updates':
                result['outcome']['completed_updates'],
                'gpu_hours': (time.monotonic() - started) * gpus / 3600, **metrics})
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('run', help='execute one already-frozen recipe')
    for name in ('spec', 'manifest', 'output'):
        command.add_argument('--' + name, type=Path, required=True)
    command.add_argument('--gpus', type=int, required=True, help='total ranks across all nodes')
    command.add_argument('--stop', type=int, help='absolute update, preserving full schedule')
    command.add_argument('--resume', type=Path)
    command.add_argument('--lr', type=float, help='peak learning rate; Muon sets muon.lr to it too')
    command.add_argument('--evaluate', action='store_true', help='export and measure validation/test BPB after the whole schedule')
    command.add_argument('--python', help='native training interpreter')
    command.add_argument('--installed', action='store_true', help='use installed wheel rather than source snapshot')
    command = sub.add_parser('fork', help='start a new lineage from a full step-S checkpoint with a new decay tail')
    command.add_argument('--from', dest='checkpoint', type=Path, required=True, help='parent <run>/checkpoint/step-S')
    command.add_argument('--decay-steps', type=int, required=True)
    command.add_argument('--decay-type', choices=('linear', 'sqrt', 'cosine'), required=True)
    command.add_argument('--min-lr-factor', type=float, help="default: the parent's")
    command.add_argument('--decay-start', type=int, help='update where the tail begins; default: S')
    command.add_argument('--evaluate', action='store_true', help='export and measure validation/test BPB after the whole schedule')
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--gpus', type=int, required=True, help='total ranks across all nodes')
    command.add_argument('--python', help='native training interpreter')
    command.add_argument('--installed', action='store_true', help='use installed wheel rather than source snapshot')
    command = sub.add_parser('untie', help='continue a fully shared run from a full step-S checkpoint, untied')
    command.add_argument('--from', dest='checkpoint', type=Path, required=True, help='source <run>/checkpoint/step-S')
    command.add_argument('--spec', type=Path, required=True, help="the source spec's untied form")
    command.add_argument('--lr', type=float, help='peak learning rate; Muon sets muon.lr to it too')
    command.add_argument('--evaluate', action='store_true', help='export and measure validation/test BPB after the whole schedule')
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--gpus', type=int, required=True, help='total ranks across all nodes')
    command.add_argument('--python', help='native training interpreter')
    command.add_argument('--installed', action='store_true', help='use installed wheel rather than source snapshot')
    args = parser.parse_args(argv)
    if args.command == 'run':
        result = run(args.spec, args.manifest, args.output, gpus=args.gpus, stop=args.stop,
                     resume=args.resume, lr=args.lr, evaluate=args.evaluate,
                     python=args.python, installed=args.installed)
    elif args.command == 'untie':
        result = untie(args.checkpoint, args.spec, args.output, gpus=args.gpus, lr=args.lr,
                       evaluate=args.evaluate, python=args.python, installed=args.installed)
    else:
        result = fork(args.checkpoint, args.output, decay_steps=args.decay_steps,
                      decay_type=args.decay_type, min_lr_factor=args.min_lr_factor,
                      decay_start=args.decay_start, evaluate=args.evaluate, gpus=args.gpus,
                      python=args.python, installed=args.installed)
    print(json.dumps({'status':'completed', 'output':str(args.output.resolve()),
                      'completed_updates':result['outcome']['completed_updates']}))


if __name__ == '__main__':
    main()
