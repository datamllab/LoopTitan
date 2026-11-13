"""The public names of `moeloop` and the cost of importing it."""

import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import moeloop


def test_looped_lm_is_the_same_class_as_recursive_lm():
    pytest.importorskip("torch")
    from moeloop.model import RecursiveLM

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert moeloop.LoopedLM is moeloop.RecursiveLM is RecursiveLM


def test_the_class_is_named_looped_lm_and_old_pickles_still_resolve():
    torch = pytest.importorskip("torch")
    from moeloop.model import ModelConfig, RecursiveLM

    model = RecursiveLM(ModelConfig(vocab_size=11, dim=8, hidden_dim=8, num_heads=2, num_kv_heads=1, num_experts=2, top_k=1))
    assert type(model).__name__ == "LoopedLM"
    assert repr(model).startswith("LoopedLM(")
    # An archived pickle names the class by module and name, as written before the rename.
    assert pickle.loads(b"cmoeloop.model\nRecursiveLM\n.") is RecursiveLM
    clone = pickle.loads(pickle.dumps(model))
    assert type(clone) is RecursiveLM
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), clone.state_dict().values()))


def test_all_lists_the_public_names():
    pytest.importorskip("torch")
    assert moeloop.__all__ == ["ModelConfig", "ModelOutput", "LoopedLM", "RecursiveLM", "DecodeCache"]
    for name in moeloop.__all__:
        assert getattr(moeloop, name) is not None
    namespace: dict = {}
    exec("from moeloop import *", namespace)
    assert set(moeloop.__all__) <= set(namespace)


def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="module 'moeloop' has no attribute 'NoSuchName'"):
        moeloop.NoSuchName


def test_import_does_not_import_torch():
    code = "import sys, moeloop; print('torch' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": str(Path(moeloop.__file__).parents[1])}
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "False"
