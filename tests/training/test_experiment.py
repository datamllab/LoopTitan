"""Frozen panel launches retain schedule, topology and independent output identity."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')


def test_native_factory_keeps_full_wsd_when_stopping_early(tmp_path, monkeypatch):
    pytest.importorskip('torchtitan')
    from moeloop.experiment import experiment
    from moeloop.recipes import p2_moe_spec
    spec = p2_moe_spec()
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec.to_dict()))
    values = {'SPEC': str(path), 'MANIFEST': str(tmp_path / 'data/manifest.json'),
              'OUTPUT': str(tmp_path / 'run'), 'STOP': '8'}
    for name, value in values.items(): monkeypatch.setenv('MOELOOP_EXPERIMENT_' + name, value)
    monkeypatch.setenv('WORLD_SIZE', '16')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '8')
    config = experiment()
    assert config.training.steps == 8
    assert config.training.local_batch_size == 16
    assert config.training.global_batch_size == 256
    assert config.lr_scheduler.total_steps == 19537
    assert config.lr_scheduler.warmup_steps == spec.run['schedule']['warmup_steps']
    assert config.parallelism.data_parallel_replicate_degree == 2
    assert config.parallelism.data_parallel_shard_degree == 8
    assert config.validator.target_label_tokens == 1_000_000
    assert config.recipe_metadata == spec.to_dict()
    assert config.checkpoint.last_save_model_only is False


def test_native_factory_keeps_the_checkpoints_the_spec_asks_for(tmp_path, monkeypatch):
    pytest.importorskip('torchtitan')
    from dataclasses import replace
    from moeloop.experiment import experiment
    from moeloop.recipes import p2_moe_spec
    for name, value in {'MANIFEST': str(tmp_path / 'data/manifest.json'), 'OUTPUT': str(tmp_path / 'run'),
                        'STOP': '8'}.items():
        monkeypatch.setenv('MOELOOP_EXPERIMENT_' + name, value)
    monkeypatch.setenv('WORLD_SIZE', '8')
    for keep, expected in ((None, 2), (0, 0)):  # The default rotates; 0 keeps every checkpoint.
        spec = p2_moe_spec()
        if keep is not None:
            spec = replace(spec, run=spec.run | {'keep_latest_k': keep})
        path = tmp_path / f'spec-{expected}.json'
        path.write_text(json.dumps(spec.to_dict()))
        monkeypatch.setenv('MOELOOP_EXPERIMENT_SPEC', str(path))
        assert experiment().checkpoint.keep_latest_k == expected


def test_multinode_launcher_uses_only_standard_topology_environment(monkeypatch):
    from recipe_experiments import topology_arguments
    for name, value in {'NNODES':'2','NPROC_PER_NODE':'8','NODE_RANK':'1',
                        'MASTER_ADDR':'192.0.2.1','MASTER_PORT':'29501'}.items():
        monkeypatch.setenv(name, value)
    args, rank = topology_arguments(16)
    assert rank == 1
    assert '--nnodes=2' in args and '--nproc-per-node=8' in args
    assert '--master-addr=192.0.2.1' in args and '--master-port=29501' in args
    with pytest.raises(ValueError, match='topology'):
        topology_arguments(8)


def test_existing_output_cannot_silently_accept_a_different_recipe(tmp_path, monkeypatch):
    import recipe_experiments as runner
    from moeloop.recipes import sharing_panel_specs
    spec = sharing_panel_specs()['a0_r0']
    manifest = SimpleNamespace(fingerprint='prepared-data', path=tmp_path/'manifest.json')
    monkeypatch.setattr(runner, 'snapshot_source', lambda p: {'commit':'a'*40, 'dirty':False, 'files':{}})
    output = tmp_path / 'run'
    output.mkdir()  # Pre-created empty output is a normal caller workflow.
    runner.prepare_run(output, spec, manifest, world=2, local_batch=8, runtime={'torch':'test'})
    assert json.loads((output/'request.json').read_text())['steps'] == 2048
    runner.prepare_run(output, spec, manifest, world=2, local_batch=8, runtime={'torch':'test'})
    with pytest.raises(ValueError, match='identity'):
        runner.prepare_run(output, sharing_panel_specs()['a1_r0'], manifest,
                           world=2, local_batch=8, runtime={'torch':'test'})
    assert len(list(output.glob('request.json'))) == 1


def test_independent_process_cannot_share_an_active_output(tmp_path):
    import subprocess
    import sys
    from recipe_experiments import coordinator_lock
    path = tmp_path / 'run'
    program = """import fcntl, sys
with open(sys.argv[1], 'a') as lock:
    try:
        fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(3)
"""
    def attempt():
        return subprocess.run([sys.executable, '-c', program, str(tmp_path / '.run.lock')]).returncode
    with coordinator_lock(path, 0):
        assert attempt() == 3
    assert attempt() == 0


def checkpoint_fixture(path, metadata=b'immutable DCP metadata'):
    path.mkdir(parents=True)
    (path / '.metadata').write_bytes(metadata)
    (path / '__0_0.distcp').write_bytes(b'full native state fixture')
    return path


def test_protected_resume_survives_actual_native_rotation(tmp_path):
    pytest.importorskip('torchtitan')
    from queue import Queue
    import shutil
    from torchtitan.components.checkpointer.dcp import CheckpointManager, purge_thread
    from recipe_experiments import protect_resume
    from recipe_qualification import digest

    output = tmp_path / 'run'
    source = checkpoint_fixture(output / 'checkpoint/step-4')
    protected = protect_resume(output, source, node_rank=0)
    checkpoint_fixture(output / 'checkpoint/step-8')
    checkpoint_fixture(output / 'checkpoint/step-12')
    queue = Queue()
    manager = SimpleNamespace(folder=str(output / 'checkpoint'), keep_latest_k=2,
        _should_purge=lambda: True, purge_thread=object(), purge_queue=queue)
    CheckpointManager._purge_stale_checkpoints(manager)
    queue.put(None)
    purge_thread(queue, shutil.rmtree)
    assert not source.exists()
    assert protected == output / 'protected/step-4'
    assert (protected / '__0_0.distcp').read_bytes() == b'full native state fixture'
    receipt = json.loads((protected / 'retention.json').read_text())
    assert receipt['original_path'] == str(source)
    assert receipt['metadata_sha256'] == digest(protected / '.metadata')
    # Repeated resume can use either the original or the durable path.
    assert protect_resume(output, source, node_rank=0) == protected
    assert protect_resume(output, protected, node_rank=0) == protected


def test_protected_resume_rejects_same_step_different_identity(tmp_path):
    from recipe_experiments import protect_resume
    output = tmp_path / 'run'
    original = checkpoint_fixture(output / 'checkpoint/step-4')
    protect_resume(output, original, node_rank=0)
    other = checkpoint_fixture(tmp_path / 'another/step-4')
    with pytest.raises(ValueError, match='identity'):
        protect_resume(output, other, node_rank=0)
    (original / '.metadata').write_bytes(b'changed DCP identity')
    with pytest.raises(ValueError, match='identity'):
        protect_resume(output, original, node_rank=0)


def test_peer_waits_for_complete_atomic_retention(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    import recipe_experiments as runner
    output = tmp_path / 'run'
    original = checkpoint_fixture(output / 'checkpoint/step-4')
    copying, release = Event(), Event()
    copytree = runner.shutil.copytree

    def delayed_copy(*args, **kwargs):
        copying.set()
        assert release.wait(5)
        return copytree(*args, **kwargs)

    monkeypatch.setattr(runner.shutil, 'copytree', delayed_copy)
    with ThreadPoolExecutor(max_workers=2) as pool:
        coordinator = pool.submit(runner.protect_resume, output, original, node_rank=0)
        assert copying.wait(5)
        peer = pool.submit(runner.protect_resume, output, original, node_rank=1)
        assert not (output / 'protected/step-4').exists()
        assert not peer.done()
        release.set()
        assert peer.result(timeout=5) == coordinator.result(timeout=5)
    assert (peer.result() / 'retention.json').is_file()


def test_retention_copy_failure_keeps_partial_and_never_publishes(tmp_path, monkeypatch):
    import recipe_experiments as runner
    output = tmp_path / 'run'
    original = checkpoint_fixture(output / 'checkpoint/step-4')

    def failed_copy(source, target, **kwargs):
        (Path(target) / 'failure-evidence').write_text('partial copy')
        raise OSError('storage failure')

    monkeypatch.setattr(runner.shutil, 'copytree', failed_copy)
    with pytest.raises(OSError, match='storage failure'):
        runner.protect_resume(output, original, node_rank=0)
    assert not (output / 'protected/step-4').exists()
    assert list((output / 'protected').glob('.step-4.partial-*/failure-evidence'))


@pytest.mark.parametrize('node_rank', [0, 1])
def test_public_runner_launches_with_durable_resume_and_node_gpu_count(tmp_path, monkeypatch, node_rank):
    import recipe_experiments as runner
    import qualification
    from moeloop.data import PreparedManifest
    from moeloop.recipes import p2_moe_spec

    output = tmp_path / 'run'
    manifest = SimpleNamespace(fingerprint='prepared-data', path=tmp_path / 'manifest.json')
    spec = p2_moe_spec()
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(spec.to_dict()))
    monkeypatch.setattr(PreparedManifest, 'load', lambda *args, **kwargs: manifest)
    monkeypatch.setattr(runner, 'snapshot_source', lambda p: {'commit':'a'*40, 'dirty':False, 'files':{}})
    monkeypatch.setattr(runner, 'runtime_identity', lambda python: {'torch':'test'})
    monkeypatch.setattr(qualification, 'verify_receipt', lambda *args, **kwargs: {'ok':True})
    for key, value in {'NNODES':'2', 'NPROC_PER_NODE':'8', 'NODE_RANK':str(node_rank),
                       'MASTER_ADDR':'192.0.2.1', 'MASTER_PORT':'29501'}.items():
        monkeypatch.setenv(key, value)
    runner.prepare_run(output, spec, manifest, world=16, local_batch=16, runtime={'torch':'test'})
    original = checkpoint_fixture(output / 'checkpoint/step-4')
    if node_rank:
        runner.protect_resume(output, original, node_rank=0)
    history = output / 'metrics.jsonl'
    history.write_text('{"step":4}\n')
    raw = output / 'process-old.json'
    raw.write_text('{"retained":"old receipt"}\n')
    launched = []

    def launch(argv, directory, **kwargs):
        protected = Path(kwargs['env']['MOELOOP_EXPERIMENT_RESUME'])
        assert (protected / '.metadata').is_file()
        assert (protected / 'retention.json').is_file()
        assert kwargs['gpu_count'] == 8 and kwargs['node_rank'] == node_rank
        assert protected == output / 'protected/step-4'
        launched.append(argv)
        (output / 'outcome.json').write_text('{"completed_updates":8}')

    monkeypatch.setattr(runner, 'run_command', launch)
    assert runner.run(spec_path, manifest.path, output, gpus=16, stop=8, resume=original) == {'ok':True}
    assert len(launched) == 1
    assert history.read_text() == '{"step":4}\n'
    assert raw.read_text() == '{"retained":"old receipt"}\n'


def test_peer_retention_timeout_never_copies(tmp_path):
    from recipe_experiments import protect_resume
    original = checkpoint_fixture(tmp_path / 'run/checkpoint/step-4')
    with pytest.raises(TimeoutError, match='retained resume'):
        protect_resume(tmp_path / 'run', original, node_rank=1, timeout=0)
    assert not (tmp_path / 'run/protected').exists()
