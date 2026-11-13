"""Vocabulary changes remain trainable and survive portable HF serialization."""

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from moeloop import recipes
from moeloop.hf import MoeloopConfig, MoeloopForCausalLM
from test_portable import tokenizer_fixture


@pytest.fixture(autouse=True)
def isolated_rng():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("name", ["recursive_moe", "ouro", "loom"])
@pytest.mark.parametrize("pad_to_multiple_of", [None, 8])
def test_resized_vocabulary_roundtrip_keeps_tokens_weights_and_logits(name, pad_to_multiple_of, tmp_path):
    spec = getattr(recipes, name + "_spec")(
        vocab_size=31, dim=16, hidden_dim=24, num_heads=4,
        num_kv_heads=2, num_experts=4, top_k=2, max_seq_len=32)
    model = MoeloopForCausalLM(MoeloopConfig(spec.model.to_dict())).eval()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    tokenizer = tokenizer_fixture()
    tokenizer.add_tokens([f"added{i}" for i in range(6)])
    embedding = model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=pad_to_multiple_of, mean_resizing=False)
    size = 37 if pad_to_multiple_of is None else 40
    assert embedding.num_embeddings == model.get_output_embeddings().out_features == size
    assert model.config.vocab_size == model.config.model_config["vocab_size"] == model.model.config.vocab_size == size
    assert spec.model.vocab_size == 31
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    for key, value in model.state_dict().items():
        previous_rows = value[:31] if key in ("model.embedding.weight", "model.lm_head.weight") else value
        torch.testing.assert_close(previous_rows, original[key], atol=0, rtol=0)

    ids = torch.tensor([tokenizer.encode("added5 w4 added2", add_special_tokens=False)])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
    artifact = tmp_path / "resized"
    model.save_pretrained(artifact)
    tokenizer.save_pretrained(artifact)
    restored = MoeloopForCausalLM.from_pretrained(artifact, local_files_only=True).eval()
    restored_tokenizer = transformers.AutoTokenizer.from_pretrained(artifact, local_files_only=True)
    assert restored_tokenizer.encode("added5 w4 added2", add_special_tokens=False) == ids[0].tolist()
    assert restored.config.vocab_size == restored.config.model_config["vocab_size"] == restored.model.config.vocab_size == size
    assert restored.get_input_embeddings().weight is not restored.get_output_embeddings().weight
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    with torch.no_grad():
        actual = restored(ids, use_cache=False).logits
    assert actual.shape == (1, 3, size)
    # Serialized weights are exact; FP32 kernels after rematerialization need
    # the same arithmetic tolerance as the other standalone HF consumers.
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-5)
    if name == "ouro" and pad_to_multiple_of is None:
        expected_file = tmp_path / "expected.pt"
        torch.save(expected, expected_file)
        program = r'''
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(1)
model = AutoModelForCausalLM.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True).eval()
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
ids = torch.tensor([tokenizer.encode("added5 w4 added2", add_special_tokens=False)])
with torch.no_grad():
    actual = model(ids, use_cache=False).logits
torch.testing.assert_close(actual, torch.load(sys.argv[2], weights_only=True), atol=3e-7, rtol=2e-5)
assert model.config.model_config["vocab_size"] == 37
for package in ("moeloop", "moeloop_lab", "torchtitan", "mlflow"):
    assert package not in sys.modules, package
'''
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(tmp_path / "hf-home"))
        result = subprocess.run([sys.executable, "-I", "-c", program, str(artifact), str(expected_file)],
                                cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr


def test_added_rows_use_declared_initialization_and_shrinking_preserves_remaining_rows(tmp_path):
    spec = recipes.ouro_spec(vocab_size=31, dim=16, hidden_dim=24, num_heads=4,
                             num_kv_heads=2, num_experts=4, top_k=2, init_std=.007)
    model = MoeloopForCausalLM(MoeloopConfig(spec.model.to_dict())).eval()
    model.resize_token_embeddings(1031, mean_resizing=False)
    for owner in (model.get_input_embeddings(), model.get_output_embeddings()):
        added = owner.weight[31:].detach()
        assert torch.isfinite(added).all()
        assert abs(added.mean().item()) < .0003
        assert added.std().item() == pytest.approx(.007, rel=.03)
    embedding = model.get_input_embeddings().weight[:23].detach().clone()
    head = model.get_output_embeddings().weight[:23].detach().clone()
    model.resize_token_embeddings(23)
    assert model.config.vocab_size == model.config.model_config["vocab_size"] == model.model.config.vocab_size == 23
    torch.testing.assert_close(model.get_input_embeddings().weight, embedding, atol=0, rtol=0)
    torch.testing.assert_close(model.get_output_embeddings().weight, head, atol=0, rtol=0)
    model.save_pretrained(tmp_path)
    restored = MoeloopForCausalLM.from_pretrained(tmp_path, local_files_only=True)
    assert restored(torch.tensor([[4, 5, 22]]), use_cache=False).logits.shape == (1, 3, 23)
