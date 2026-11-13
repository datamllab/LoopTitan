"""HF delivery checks over the same core, including an isolated offline load."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from moeloop.hf import MoeloopCache, MoeloopConfig, MoeloopForCausalLM
from moeloop.model import ModelConfig


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(20)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    config = ModelConfig(vocab_size=31, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2,
                         num_experts=4, top_k=2, num_cells=1, num_visits=2,
                         prefix_cells=1, suffix_cells=1)
    hf_config = MoeloopConfig(config.to_dict(), bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return MoeloopForCausalLM(hf_config).eval()


def test_hf_cached_padding_and_full_prefix_match(model):
    ids = torch.tensor([[0, 0, 4, 5], [0, 6, 7, 8]])
    mask = ids != 0
    with torch.no_grad():
        full = model(ids, attention_mask=mask, use_cache=False)
        prefix = model(ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
        continuation = model(ids[:, 3:], attention_mask=mask, past_key_values=prefix.past_key_values)
    assert isinstance(continuation.past_key_values, MoeloopCache)
    torch.testing.assert_close(continuation.logits, full.logits[:, -1:], atol=3e-7, rtol=2e-5)
    assert continuation.past_key_values.get_seq_length() == 4


def test_hf_labels_shift_exactly_once(model):
    ids = torch.tensor([[1, 4, 5, 2], [1, 8, 2, 0]])
    labels = ids.clone()
    labels[1, -1] = -100
    output = model(ids, labels=labels, use_cache=False)
    expected = torch.nn.functional.cross_entropy(output.logits[:, :-1].float().reshape(-1, 31),
                                                labels[:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(output.loss, expected)
    assert output.past_key_values is None
    output.loss.backward()
    assert model.model.embedding.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="no valid"):
        model(ids, labels=torch.full_like(labels, -100), use_cache=False)


@pytest.mark.parametrize("num_beams", [1, 3])
def test_generate_greedy_and_beam_cache_equal_recomputation(model, num_beams):
    ids = torch.tensor([[0, 4, 5], [6, 7, 8]])
    mask = ids != 0
    kwargs = dict(attention_mask=mask, max_new_tokens=3, do_sample=False,
                  num_beams=num_beams, eos_token_id=None, pad_token_id=0)
    cached = model.generate(ids, use_cache=True, **kwargs)
    recomputed = model.generate(ids, use_cache=False, **kwargs)
    assert torch.equal(cached, recomputed)
    assert cached.shape == (2, 6)


def tokenizer_fixture():
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocabulary.update({f"w{i}": i for i in range(4, 31)})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
        unk_token="<unk>", padding_side="left",
    )


def test_save_pretrained_loads_in_offline_process_without_checkout(model, tmp_path):
    artifact = tmp_path / "artifact"
    model.save_pretrained(artifact, safe_serialization=True)
    tokenizer_fixture().save_pretrained(artifact)
    config = json.loads((artifact / "config.json").read_text())
    assert config["auto_map"]["AutoModelForCausalLM"] == "hf.MoeloopForCausalLM"
    assert config["tie_word_embeddings"] is False
    assert config["architectures"] == ["MoeloopForCausalLM"]  # the core's class name is not in the export
    assert "RecursiveLM" not in json.dumps(config) and "LoopedLM" not in json.dumps(config)
    assert all((artifact / name).is_file() for name in ("hf.py", "model.py", "moe.py", "cache.py", "model.safetensors"))
    ids = torch.tensor([[4, 5, 6]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
    torch.save(expected, tmp_path / "expected.pt")
    script = r'''
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(1)
artifact = Path(sys.argv[1])
model = AutoModelForCausalLM.from_pretrained(artifact, trust_remote_code=True, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
assert tokenizer.encode("w4 w5 w6", add_special_tokens=False) == [4, 5, 6]
with torch.no_grad():
    actual = model(torch.tensor([[4, 5, 6]]), use_cache=False).logits
torch.testing.assert_close(actual, torch.load(sys.argv[2], weights_only=True))
result = model.generate(torch.tensor([[4, 5, 6]]), attention_mask=torch.ones((1,3),dtype=torch.long),
                        max_new_tokens=2, do_sample=False, eos_token_id=None, pad_token_id=0)
assert result.shape == (1, 5)
for forbidden in ("moeloop", "moeloop_lab", "torchtitan", "mlflow"):
    assert forbidden not in sys.modules, forbidden
print("offline standalone parity and generation passed")
'''
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(tmp_path / "hf-home"))
    result = subprocess.run([sys.executable, "-I", "-c", script, str(artifact), str(tmp_path / "expected.pt")],
                            cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "offline standalone parity" in result.stdout
