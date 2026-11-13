"""Under scripts/run.py a launch records its config and results in RUN_DIR.

The launcher runs for real; only the GPU parts are replaced: the torchrun command
(run_command), the receipt check and the export/evaluate subprocesses. The launcher's
clock is fixed: each training command takes one hour on it, each export or evaluation
twenty minutes.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('torch')

import qualification
import recipe_experiments as runner
from moeloop.bench import load_tracks
from moeloop.data import PreparedManifest
from moeloop.recipe_spec import RecipeSpec

ROOT = Path(__file__).resolve().parents[2]


def adamw_spec():
    return json.loads((ROOT / 'recipes/p2_moe.json').read_text())


def muon_spec():
    spec = adamw_spec()
    fixed = load_tracks()['S'].optimizers['fixed']['Muon']
    spec['optimizer'] |= {key: value for key, value in fixed.items() if key != 'muon'}
    spec['optimizer'] |= {'name': 'Muon', 'muon': fixed['muon'] | {'lr': 6e-4}}
    return spec


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    """A launch whose training, receipt check and subprocesses are recorded."""
    manifest = SimpleNamespace(fingerprint='prepared-data', path=tmp_path / 'manifest.json')
    monkeypatch.setattr(PreparedManifest, 'load', lambda *args, **kwargs: manifest)
    monkeypatch.setattr(runner, 'snapshot_source', lambda p: {'commit': 'a' * 40, 'dirty': False, 'files': {}})
    monkeypatch.setattr(runner, 'runtime_identity', lambda python: {'torch': 'test'})
    monkeypatch.setattr(qualification, 'verify_receipt', lambda directory, expected_steps:
        {'outcome': {'completed_updates': expected_steps,
                     'full_checkpoint': str(directory / f'checkpoint/step-{expected_steps}')}})
    for name in ('NNODES', 'NPROC_PER_NODE', 'NODE_RANK', 'RUN_DIR'):
        monkeypatch.delenv(name, raising=False)
    calls = SimpleNamespace(training=[], subprocesses=[], config_at_launch=None)
    run_dir, output = tmp_path / 'run_dir', tmp_path / 'out'
    clock = [0.0]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])

    def train(argv, directory, **kwargs):
        calls.training.append(argv)
        config = run_dir / 'config.json'
        calls.config_at_launch = json.loads(config.read_text()) if config.is_file() else None
        clock[0] += 3600
        # What the coordinator's trainer publishes; a peer node waits for it.
        (output / 'outcome.json').write_text(
            json.dumps({'completed_updates': int(kwargs['env']['MOELOOP_EXPERIMENT_STOP'])}))

    def subprocess_run(argv, **kwargs):
        argv = list(map(str, argv))
        calls.subprocesses.append((argv, kwargs))
        clock[0] += 1200
        if 'moeloop.evaluate' in argv:
            split = argv[argv.index('--split') + 1]
            Path(argv[argv.index('--output') + 1]).write_text(
                json.dumps({'bpb': {'validation': .9, 'test': .8}[split]}))

    monkeypatch.setattr(runner, 'run_command', train)
    monkeypatch.setattr(runner.subprocess, 'run', subprocess_run)

    def main(argv, run_dir_set):
        if run_dir_set:
            run_dir.mkdir()
            monkeypatch.setenv('RUN_DIR', str(run_dir))
        runner.main(argv)

    def launch(spec, *args, gpus=8, run_dir_set=True):
        path = tmp_path / 'spec.json'
        path.write_text(json.dumps(spec))
        main(['run', '--spec', str(path), '--manifest', str(manifest.path), '--output', str(output),
              '--gpus', str(gpus), *args], run_dir_set)

    def fork(checkpoint, *args):
        main(['fork', '--from', str(checkpoint), '--output', str(output), '--gpus', '8', *args], True)

    calls.launch, calls.fork, calls.run_dir, calls.output, calls.manifest = launch, fork, run_dir, output, manifest
    return calls


@pytest.mark.parametrize('make_spec', [adamw_spec, muon_spec])
def test_lr_sets_the_learning_rate_written_to_the_output(launcher, make_spec):
    spec = make_spec()
    launcher.launch(spec, '--lr', '1.2e-3')
    written = json.loads((launcher.output / 'spec.json').read_text())['optimizer']
    assert written['base_lr'] == 1.2e-3
    if spec['optimizer']['name'] == 'Muon':
        assert written['muon']['lr'] == 1.2e-3
    else:
        assert 'muon' not in written
    assert launcher.config_at_launch['optimizer'] == written


def test_config_is_written_before_training_with_exactly_the_spec_and_launch(launcher):
    spec = adamw_spec()
    launcher.launch(spec)
    config = launcher.config_at_launch
    assert len(launcher.training) == 1 and config is not None
    assert list(config) == ['name', 'model', 'training', 'optimizer', 'inference', 'run', 'launch']
    launched = json.loads((launcher.output / 'spec.json').read_text())
    assert {key: config[key] for key in list(config)[:-1]} == {key: launched[key] for key in list(config)[:-1]}
    # p2_moe: 1048576 / 4096 = 256 sequences per update, local batch 16 on 8 GPUs.
    assert config['launch'] == {'world_size': 8, 'local_batch_size': 16,
                                'gradient_accumulation_steps': 2,
                                'manifest_fingerprint': 'prepared-data'}


@pytest.mark.parametrize('gpus', [4, 8])
def test_metrics_after_a_completed_run(launcher, gpus):
    launcher.launch(adamw_spec(), '--stop', '8', gpus=gpus)
    metrics = json.loads((launcher.run_dir / 'metrics.json').read_text())
    # One hour of launch on the fixed clock costs one GPU hour per GPU.
    assert metrics == {'completed_updates': 8, 'gpu_hours': gpus}
    assert launcher.subprocesses == []


def test_evaluate_exports_and_measures_both_splits(launcher):
    spec = adamw_spec()
    launcher.launch(spec, '--evaluate')
    out, steps = launcher.output, spec['run']['successful_updates']
    metrics = json.loads((launcher.run_dir / 'metrics.json').read_text())
    assert metrics['completed_updates'] == steps
    assert metrics['validation_bpb'] == .9 and metrics['test_bpb'] == .8
    (export, export_kwargs), *evaluations = launcher.subprocesses
    assert export[1:] == ['-m', 'moeloop.export', '--checkpoint', str(out / f'checkpoint/step-{steps}'),
                          '--output', str(out / 'hf')]
    assert [argv[1:] for argv, _ in evaluations] == [
        ['-m', 'moeloop.evaluate', '--model', str(out / 'hf'), '--manifest', str(launcher.manifest.path),
         '--split', split, '--seq-len', '4096', '--batch-size', '4', '--num-visits', '1',
         '--device', 'cuda', '--output', str(out / f'eval-{split}.json')] for split in ('validation', 'test')]
    # Evaluation runs the run's own source snapshot, like training.
    for argv, kwargs in launcher.subprocesses:
        assert kwargs['check'] is True
        assert kwargs['env']['PYTHONPATH'].startswith(str(out / 'source/src'))


def test_evaluate_with_an_early_stop_raises_before_launch(launcher):
    with pytest.raises(ValueError, match='whole schedule'):
        launcher.launch(adamw_spec(), '--evaluate', '--stop', '8')
    assert launcher.training == [] and launcher.subprocesses == []
    assert not launcher.output.exists()
    assert list(launcher.run_dir.iterdir()) == []


def test_without_run_dir_nothing_is_recorded(launcher, tmp_path):
    launcher.launch(adamw_spec(), '--stop', '8', run_dir_set=False)
    assert len(launcher.training) == 1
    assert not list(tmp_path.rglob('config.json')) and not list(tmp_path.rglob('metrics.json'))


def test_fork_records_its_own_spec_and_evaluates(launcher, tmp_path):
    parent = tmp_path / 'parent'
    (parent / 'checkpoint/step-100').mkdir(parents=True)
    (parent / 'checkpoint/step-100/.metadata').write_bytes(b'parent DCP metadata')
    (parent / 'spec.json').write_text(json.dumps(adamw_spec()))
    (parent / 'request.json').write_text(json.dumps({'manifest': str(launcher.manifest.path)}))
    (parent / 'continuation_identity.json').write_text(
        json.dumps({'lr_scheduler': {'warmup_steps': 20, 'min_lr_factor': 0.0}}))
    launcher.fork(parent / 'checkpoint/step-100', '--decay-steps', '50', '--decay-type', 'linear',
                  '--evaluate')
    config = json.loads((launcher.run_dir / 'config.json').read_text())
    assert config['run']['fork']['step'] == 100
    assert config['run']['fork']['checkpoint'] == str(parent / 'checkpoint/step-100')
    assert config['run']['successful_updates'] == 150
    metrics = json.loads((launcher.run_dir / 'metrics.json').read_text())
    # Training, export and two evaluations take two hours of the launch on 8 GPUs.
    assert metrics == {'completed_updates': 150, 'gpu_hours': 16,
                       'validation_bpb': .9, 'test_bpb': .8}
    export = launcher.subprocesses[0][0]
    assert export[export.index('--checkpoint') + 1] == str(launcher.output / 'checkpoint/step-150')


def test_peer_node_records_and_evaluates_nothing(launcher, monkeypatch):
    for name, value in {'NNODES': '2', 'NPROC_PER_NODE': '4', 'NODE_RANK': '1',
                        'MASTER_ADDR': '192.0.2.1', 'MASTER_PORT': '29501'}.items():
        monkeypatch.setenv(name, value)
    spec = adamw_spec()
    # The coordinator node publishes the shared request first.
    runner.prepare_run(launcher.output, RecipeSpec.from_dict(spec), launcher.manifest,
                       world=8, local_batch=16, runtime={'torch': 'test'})
    launcher.launch(spec, '--evaluate')
    assert len(launcher.training) == 1
    assert list(launcher.run_dir.iterdir()) == []
    assert launcher.subprocesses == []
