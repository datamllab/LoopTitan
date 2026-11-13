import pytest

torch = pytest.importorskip("torch")

from moeloop.cache import DecodeCache
from moeloop.model import ModelConfig, RecursiveLM


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(83)
    yield
    torch.set_num_threads(previous)


def model_config(**changes):
    values = dict(vocab_size=31, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2,
                  num_experts=4, top_k=2, num_cells=2, num_visits=3,
                  prefix_cells=1, suffix_cells=1)
    return ModelConfig(**(values | changes))


@pytest.mark.parametrize("order", ["stack", "layer"])
@pytest.mark.parametrize("per_visit", [False, True])
def test_exact_chunked_cache_matches_full_prefix(order, per_visit):
    model = RecursiveLM(model_config(order=order, attention_per_visit=per_visit)).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [3, 7, 1, 9, 4, 2]])
    cache = DecodeCache(3)
    with torch.no_grad():
        full = model(ids).logits
        chunks = [model(ids[:, :3], cache=cache).logits,
                  model(ids[:, 3:5], cache=cache).logits,
                  model(ids[:, 5:], cache=cache).logits]
    torch.testing.assert_close(torch.cat(chunks, 1), full, atol=3e-7, rtol=2e-5)
    assert cache.length == 6
    assert len(cache.slots) == 8
    assert len({id(key) for key, _ in cache.slots.values()}) == 8
    assert cache.slots[("core", 0, 0)][0].size(1) == 2  # KV heads, not Q heads


def test_padded_cache_and_reorder_preserve_positions_and_all_slots():
    model = RecursiveLM(model_config()).eval()
    ids = torch.tensor([[0, 0, 4, 5], [0, 6, 7, 8]])
    valid = ids != 0
    cache = DecodeCache(3)
    with torch.no_grad():
        model(ids, attention_mask=valid, cache=cache)
        order = torch.tensor([1, 0, 1])
        cache.reorder(order)
        ids, valid = ids[order], valid[order]
        next_token = torch.tensor([[9], [10], [11]])
        full_valid = torch.cat((valid, torch.ones((3, 1), dtype=torch.bool)), 1)
        cached = model(next_token, attention_mask=full_valid, cache=cache).logits
        full = model(torch.cat((ids, next_token), 1), attention_mask=full_valid).logits[:, -1:]
    torch.testing.assert_close(cached, full, atol=3e-7, rtol=2e-5)
    assert cache.positions.tolist() == [[0, 0, 1, 2, 3], [0, 0, 0, 1, 2], [0, 0, 1, 2, 3]]
    assert all(key.size(0) == 3 for key, _ in cache.slots.values())


def test_cache_rejects_a_forward_that_skips_a_logical_slot():
    model = RecursiveLM(model_config()).eval()
    no_suffix = RecursiveLM(model_config(suffix_cells=0)).eval()
    cache = DecodeCache(3)
    with torch.no_grad():
        model(torch.tensor([[1, 2]]), cache=cache)
        with pytest.raises(ValueError, match="every logical cache slot"):
            no_suffix(torch.tensor([[3]]), cache=cache)


def test_advance_rejects_a_slot_with_extra_columns():
    cache = DecodeCache(1)
    kv = torch.zeros(1, 1, 2, 4)
    slots = (("core", 0, 0), ("core", 1, 0))
    for slot in slots + slots[:1]:
        cache.append(slot, kv, kv)
    with pytest.raises(ValueError, match="every logical cache slot"):
        cache.advance(torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.bool), slots)


def test_forward_after_a_failed_forward_raises_until_clear(monkeypatch):
    model = RecursiveLM(model_config()).eval()
    ids = torch.tensor([[1, 2, 3]])
    cache = DecodeCache(3)
    attention = model.suffix[0].attentions[0]
    with torch.no_grad():
        with monkeypatch.context() as patch:
            def fail(*args, **kwargs):
                raise RuntimeError("injected attention failure")
            patch.setattr(attention, "forward", fail)
            with pytest.raises(RuntimeError, match="injected"):
                model(ids, cache=cache)
        with pytest.raises(ValueError, match="clear"):
            model(ids, cache=cache)
        cache.clear()
        logits = model(ids, cache=cache).logits
        fresh = model(ids, cache=DecodeCache(3)).logits
    assert torch.equal(logits, fresh)


def test_cache_rejects_training_and_depth_change():
    model = RecursiveLM(model_config()).eval()
    cache = DecodeCache(3)
    ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="no_grad"):
        model(ids, cache=cache)
    with torch.no_grad():
        model(ids, cache=cache)
        with pytest.raises(ValueError, match="num_visits"):
            model(ids[:, :1], num_visits=2, cache=cache)
    cache.clear()
    assert cache.length == 0 and cache.slots == {} and cache.positions is None
