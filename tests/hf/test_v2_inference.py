"""New state/objective recipes remain portable terminal-logit consumers."""

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from moeloop import recipes
from moeloop.hf import MoeloopConfig, MoeloopForCausalLM
from test_portable import tokenizer_fixture


@pytest.fixture(autouse=True)
def isolated_cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name", ["huginn", "loom", "loom_attention", "ouro"])
def test_new_method_terminal_inference_survives_standalone_hf_delivery(name, tmp_path):
    geometry = dict(vocab_size=31, dim=16, hidden_dim=24, num_heads=4,
                    num_kv_heads=2, num_experts=4, top_k=2, max_seq_len=32)
    if name == "loom_attention":
        spec = recipes.loom_spec(attention_per_visit=True, **geometry)
    else:
        spec = getattr(recipes, name + "_spec")(**geometry)
    config = MoeloopConfig(spec.model.to_dict(), bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = MoeloopForCausalLM(config).eval()
    ids = torch.tensor([[4, 5, 6]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        terminal = model(ids, attention_mask=mask, use_cache=False).logits
        repeated = model(ids, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(repeated, terminal, atol=0, rtol=0)
        default_output = model(ids, attention_mask=mask)
        generated = model.generate(ids, attention_mask=mask, max_new_tokens=2,
                                   do_sample=False, eos_token_id=None, pad_token_id=0)
        recomputed = model.generate(ids, attention_mask=mask, max_new_tokens=2,
                                    do_sample=False, eos_token_id=None, pad_token_id=0,
                                    use_cache=False)
    assert torch.equal(generated, recomputed)
    assert generated.shape == (1, 5)
    if name == "ouro":
        assert model.config.use_cache and default_output.past_key_values is not None
        torch.testing.assert_close(default_output.logits, terminal, atol=3e-7, rtol=2e-5)
    else:
        assert not model.config.use_cache and default_output.past_key_values is None

    # HF's labels API evaluates ordinary terminal next-token CE, including for
    # the expected-loss training recipe. It must not train the exit gate.
    labels = ids.clone()
    labels[:, -1] = -100
    labelled = model(ids, attention_mask=mask, labels=labels)
    expected_ce = torch.nn.functional.cross_entropy(
        terminal[:, :-1].float().reshape(-1, 31), labels[:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(labelled.loss, expected_ce)
    labelled.loss.backward()
    assert model.model.embedding.weight.grad is not None
    if name == "ouro":
        assert model.model.exit_gate.weight.grad is None
        assert model.model.exit_gate.bias.grad is None

    artifact = tmp_path / "artifact"
    model.save_pretrained(artifact, safe_serialization=True)
    tokenizer_fixture().save_pretrained(artifact)
    # Transitive state modules must accompany weights/config so the artifact
    # can execute its dynamics without the training checkout.
    for module in ("hf.py", "model.py", "latent.py", "memory.py"):
        assert (artifact / module).is_file(), module
    expected = tmp_path / "expected.pt"
    torch.save({"logits": terminal, "generated": generated}, expected)
    script = r'''
import importlib.abc
import sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
class NoTrainingPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('moeloop', 'moeloop_lab', 'torchtitan', 'mlflow'):
            raise AssertionError('standalone artifact imported ' + fullname)
sys.meta_path.insert(0, NoTrainingPackages())
torch.set_num_threads(1)
artifact, expected_file = Path(sys.argv[1]), Path(sys.argv[2])
config = AutoConfig.from_pretrained(artifact, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(artifact, config=config,
    trust_remote_code=True, local_files_only=True).eval()
tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
assert tokenizer.encode('w4 w5 w6', add_special_tokens=False) == [4, 5, 6]
expected = torch.load(expected_file, weights_only=True)
ids = torch.tensor([[4, 5, 6]])
with torch.no_grad():
    actual = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    generated = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=2,
        do_sample=False, eos_token_id=None, pad_token_id=0)
torch.testing.assert_close(actual, expected['logits'])
assert torch.equal(generated, expected['generated'])
for forbidden in ('moeloop', 'moeloop_lab', 'torchtitan', 'mlflow'):
    assert forbidden not in sys.modules, forbidden
assert config.use_cache == (sys.argv[3] == 'ouro')
print('standalone terminal inference passed')
'''
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_HOME=str(tmp_path / "hf-home"))
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(artifact), str(expected), name],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "standalone terminal inference passed" in result.stdout
