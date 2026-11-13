import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchtitan")


@pytest.mark.skipif(sys.platform != "linux", reason="the CPU FSDP probe uses Linux fork")
def test_actual_two_rank_fsdp_ac_and_chunked_loss_gradients():
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"), "OMP_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("fsdp_semantics.py")), "2"], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "partial FullAC update 1 OK" in result.stdout
    assert "truncated SelectiveAC update 1 OK" in result.stdout
    assert "memory FullAC update 1 OK" in result.stdout
    assert "expected none update 1 OK" in result.stdout
    assert "expected FullAC update 1 OK" in result.stdout
    assert "memory_sharded FullAC update 1 OK" in result.stdout
    assert "expected_sharded FullAC update 1 OK" in result.stdout
    assert "latent FullAC update 1 OK" in result.stdout


def test_selective_ac_rejects_independent_attention_before_wrapping():
    from types import SimpleNamespace
    from moeloop.model import ModelConfig
    from moeloop.titan import TitanModel, parallelize_model
    from torchtitan.distributed.activation_checkpoint import SelectiveAC
    model = TitanModel(TitanModel.Config(core=ModelConfig(vocab_size=19, dim=16, hidden_dim=24,
                      num_heads=2, num_experts=2, attention_per_visit=True), native_experts=False, flex_attention=False))
    with pytest.raises(ValueError, match="FullAC or no AC"):
        parallelize_model(model, parallel_dims=None, training=None, parallelism=None, compile_config=None,
                          ac_config=SelectiveAC.Config(), dump_folder="/tmp")
