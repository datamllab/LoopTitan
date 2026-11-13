import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan")

from moeloop.prepare import prepare_dataset
from moeloop.titan import PackedDataLoader


def test_real_loader_iterator_resume_does_not_consume_training_rng(tmp_path):
    manifest = prepare_dataset(
        tmp_path / "prepared", {"text": {"train": ["a b c"] * 32}},
        encode=lambda _: [4, 5, 6],
        tokenizer={"id": "test", "revision": "fixed", "vocab_size": 19, "bos_token_id": 1, "eos_token_id": 2},
    )
    cfg = PackedDataLoader.Config(manifest_path=str(manifest.path), pin_memory=False)

    def build():
        return PackedDataLoader(cfg, dp_rank=0, dp_world_size=1, tokenizer=None, seq_len=4, local_batch_size=2)

    loader = build()
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    rng = torch.get_rng_state()
    expected_batch = next(iterator)
    expected_random = torch.rand(5)
    resumed = build()
    resumed.load_state_dict(state)
    torch.set_rng_state(rng)
    actual_batch = next(iter(resumed))
    for key in expected_batch[0]:
        assert torch.equal(expected_batch[0][key], actual_batch[0][key])
    assert torch.equal(expected_batch[1], actual_batch[1])
    assert torch.equal(expected_random, torch.rand(5))
