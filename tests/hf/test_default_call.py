import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from moeloop.hf import MoeloopConfig, MoeloopForCausalLM
from moeloop.model import ModelConfig


def test_eval_full_prefix_supports_gradients_and_respects_config_cache_choice():
    cfg = ModelConfig(vocab_size=19, dim=16, hidden_dim=24, num_heads=2, num_experts=2)
    model = MoeloopForCausalLM(MoeloopConfig(cfg.to_dict(), use_cache=False)).eval()
    assert model.config.use_cache is False
    tokens = torch.tensor([[1, 3, 4]])
    output = model(tokens)
    assert output.past_key_values is None
    output.logits.sum().backward()
    assert model.model.embedding.weight.grad is not None
    model.config.use_cache = True
    assert model(tokens).past_key_values is None
    with torch.no_grad():
        assert model(tokens).past_key_values.get_seq_length() == 3
