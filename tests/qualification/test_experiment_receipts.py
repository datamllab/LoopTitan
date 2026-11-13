"""Native launcher receipts count physical GPUs once per node."""
import json
import sys
import time
from types import SimpleNamespace


def test_two_eight_gpu_node_hours_are_sixteen_gpu_hours(tmp_path, monkeypatch):
    import recipe_qualification as commands
    from check_release import native_gpu_hours

    # Run real short children; fix only the module's accounting clock to one hour/node.
    # Global time.time stays real, so other callers cannot exhaust this sequence.
    clock = iter([0., 3600., 3600., 0., 3600., 3600.])
    monkeypatch.setattr(commands, 'time', SimpleNamespace(time=lambda: next(clock), time_ns=time.time_ns))
    for rank in (0, 1):
        commands.run_command([sys.executable, '-c', 'pass', 'moeloop.train'], tmp_path,
                             gpu_count=8, node_rank=rank)
    receipts = [json.loads(path.read_text()) for path in tmp_path.glob('process-*.json')]
    assert sorted(value['node_rank'] for value in receipts) == [0, 1]
    assert all(value['gpu_count'] == 8 for value in receipts)
    assert native_gpu_hours(tmp_path, 16) == 16.


def test_legacy_command_receipt_keeps_world_size_fallback(tmp_path, monkeypatch):
    import recipe_qualification as commands
    from check_release import native_gpu_hours

    clock = iter([0., 3600., 3600.])
    monkeypatch.setattr(commands, 'time', SimpleNamespace(time=lambda: next(clock), time_ns=time.time_ns))
    commands.run_command([sys.executable, '-c', 'pass', 'moeloop.train'], tmp_path)
    receipt = json.loads(next(tmp_path.glob('process-*.json')).read_text())
    assert 'gpu_count' not in receipt and 'node_rank' not in receipt
    assert native_gpu_hours(tmp_path, 16) == 16.
