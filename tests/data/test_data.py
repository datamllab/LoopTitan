from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from moeloop.data import PackedDataset, PreparedManifest, build_document_mask
from moeloop.prepare import prepare_dataset, read_texts


TOKENIZER = {"id": "test-tokenizer", "revision": "fixed-v1", "vocab_size": 64, "bos_token_id": 0, "eos_token_id": 63, "pad_token_id": 63}


def prepare(path: Path, sources, tokenizer=None, **kwargs):
    return prepare_dataset(path, sources, encode=lambda text: [int(value) for value in text.split()], tokenizer=tokenizer or TOKENIZER, **kwargs)


def assert_sample_equal(left, right):
    assert left[0].keys() == right[0].keys()
    for name in left[0]:
        assert torch.equal(left[0][name], right[0][name]), name
    assert torch.equal(left[1], right[1])


def test_document_shift_positions_padding_and_no_cross_document_attention(tmp_path):
    manifest = prepare(tmp_path / "data", {"text": {"validation": ["1 2", "3"]}})
    dataset = PackedDataset(manifest, split="validation", seq_len=6)
    inputs, labels = next(dataset)
    assert inputs["input"].tolist() == [0, 1, 2, 0, 3, 63]
    assert labels.tolist() == [1, 2, 63, 3, 63, -100]
    assert inputs["positions"].tolist() == [0, 1, 2, 0, 1, 0]
    assert inputs["document_ids"].tolist() == [0, 0, 0, 1, 1, -1]
    assert inputs["input_mask"].tolist() == [True] * 5 + [False]
    allowed = build_document_mask(inputs["document_ids"][None], inputs["input_mask"][None])
    assert allowed.shape == (1, 1, 6, 6)
    assert allowed[0, 0, 2, :].tolist() == [True, True, True, False, False, False]
    assert allowed[0, 0, 4, :].tolist() == [False, False, False, True, True, False]
    assert not allowed[0, 0, 5, :].any()
    with pytest.raises(StopIteration):
        next(dataset)


def test_long_document_continuation_preserves_every_target_and_resets_context_positions(tmp_path):
    manifest = prepare(tmp_path / "data", {"text": {"test": ["1 2 3 4 5"]}})
    blocks = list(PackedDataset(manifest, split="test", seq_len=3))
    assert len(blocks) == 2
    assert [block[0]["input"].tolist() for block in blocks] == [[0, 1, 2], [3, 4, 5]]
    assert [block[1].tolist() for block in blocks] == [[1, 2, 3], [4, 5, 63]]
    assert all(block[0]["positions"].tolist() == [0, 1, 2] for block in blocks)
    assert blocks[0][0]["document_ids"][0] == blocks[1][0]["document_ids"][0]


def test_eos_token_is_not_a_padding_detector(tmp_path):
    manifest = prepare(tmp_path / "data", {"text": {"test": ["63"]}})
    inputs, labels = next(PackedDataset(manifest, split="test", seq_len=4))
    assert inputs["input"].tolist() == [0, 63, 63, 63]
    assert inputs["input_mask"].tolist() == [True, True, False, False]
    assert labels.tolist() == [63, 63, -100, -100]


def test_source_weights_mix_token_blocks_instead_of_documents(tmp_path):
    manifest = prepare(tmp_path / "data", {
        "short": {"train": ["1"]},
        "long": {"train": [" ".join(["2"] * 99)]},
    })
    dataset = PackedDataset(manifest, seq_len=8, source_weights={"short": 3, "long": 1}, seed=8)
    for _ in range(2000):
        next(dataset)
    counts = dataset.source_counts
    assert counts["short"]["input_tokens"] + counts["long"]["input_tokens"] == 16000
    assert 0.71 < counts["short"]["input_tokens"] / 16000 < 0.79
    assert counts["short"]["documents"] > 30 * counts["long"]["documents"]
    assert counts["short"]["input_tokens"] == 8 * counts["short"]["blocks"]


def test_exact_resume_includes_partial_document_mixture_shuffle_and_epochs(tmp_path):
    manifest = prepare(tmp_path / "data", {
        "a": {"train": ["1 2 3 4 5 6 7 8 9", "11 12 13"]},
        "b": {"train": ["21 22 23 24 25 26", "31 32 33 34 35"]},
    }, shard_tokens=10)
    kwargs = dict(seq_len=5, source_weights={"a": 0.4, "b": 0.6}, seed=117)
    original = PackedDataset(manifest, **kwargs)
    for _ in range(7):
        next(original)
    state = json.loads(json.dumps(original.state_dict()))
    assert any(source["active_offset"] > 0 for source in state["sources"].values())
    expected = [next(original) for _ in range(30)]
    expected_counts = original.source_counts
    restored = PackedDataset(manifest, **kwargs)
    restored.load_state_dict(state)
    for sample in expected:
        assert_sample_equal(sample, next(restored))
    assert restored.source_counts == expected_counts
    assert any(source["epoch"] > 1 for source in restored.state_dict()["sources"].values())


def test_stateful_dataloader_resumes_the_next_collated_batches(tmp_path):
    torchdata = pytest.importorskip("torchdata.stateful_dataloader")
    manifest = prepare(tmp_path / "data", {
        "a": {"train": ["1 2 3 4 5 6 7 8", "9 10"]},
        "b": {"train": ["11 12 13 14 15", "16 17 18"]},
    })
    kwargs = dict(seq_len=5, source_weights={"a": 2, "b": 1}, seed=28)
    loader = torchdata.StatefulDataLoader(PackedDataset(manifest, **kwargs), batch_size=3, num_workers=0)
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    state = copy.deepcopy(loader.state_dict())
    expected = [next(iterator) for _ in range(5)]
    restored = torchdata.StatefulDataLoader(PackedDataset(manifest, **kwargs), batch_size=3, num_workers=0)
    restored.load_state_dict(state)
    actual = iter(restored)
    for sample in expected:
        assert_sample_equal(sample, next(actual))


def test_worker_prefetch_is_rejected_instead_of_claiming_incomplete_resume(tmp_path, monkeypatch):
    manifest = prepare(tmp_path / "data", {"a": {"train": ["1"]}})
    dataset = PackedDataset(manifest, seq_len=2)
    monkeypatch.setattr("moeloop.data.get_worker_info", lambda: object())
    with pytest.raises(ValueError, match="num_workers=0"):
        iter(dataset)


def test_finite_sources_exhaust_once_and_exhausted_state_stays_exhausted(tmp_path):
    manifest = prepare(tmp_path / "data", {"a": {"validation": ["1", "2"]}, "b": {"validation": ["3"]}})
    dataset = PackedDataset(manifest, split="validation", seq_len=3, seed=3)
    samples = list(dataset)
    assert sum(int((labels != -100).sum()) for _, labels in samples) == 6
    assert dataset.source_counts["a"]["documents"] == 2
    assert dataset.source_counts["b"]["documents"] == 1
    restored = PackedDataset(manifest, split="validation", seq_len=3, seed=3)
    restored.load_state_dict(dataset.state_dict())
    assert list(restored) == []


def test_rank_shards_are_disjoint_and_epoch_shuffle_is_reproducible(tmp_path):
    manifest = prepare(tmp_path / "data", {"a": {"train": [str(value) for value in range(1, 9)]}})
    rank0 = PackedDataset(manifest, seq_len=2, rank=0, world_size=2, seed=41)
    rank1 = PackedDataset(manifest, seq_len=2, rank=1, world_size=2, seed=41)
    values0 = {next(rank0)[1][0].item() for _ in range(4)}
    values1 = {next(rank1)[1][0].item() for _ in range(4)}
    assert not values0 & values1
    assert values0 | values1 == set(range(1, 9))
    same_rank = PackedDataset(manifest, seq_len=2, rank=0, world_size=2, seed=41)
    for _ in range(4):
        next(same_rank)
    for _ in range(12):
        assert_sample_equal(next(rank0), next(same_rank))


@pytest.mark.parametrize("changed", ["tokenizer", "data", "seq_len", "seed", "rank"])
def test_resume_rejects_changed_identity_or_configuration(tmp_path, changed):
    sources = {"a": {"train": ["1 2 3", "4 5 6"]}}
    manifest = prepare(tmp_path / "first", sources)
    original = PackedDataset(manifest, seq_len=2, seed=4)
    next(original)
    kwargs = dict(seq_len=2, seed=4)
    if changed in ("tokenizer", "data"):
        tokenizer = dict(TOKENIZER, revision="different") if changed == "tokenizer" else TOKENIZER
        documents = {"a": {"train": ["7 2 3", "4 5 6"]}} if changed == "data" else sources
        manifest = prepare(tmp_path / "second", documents, tokenizer=tokenizer)
    elif changed == "seq_len":
        kwargs["seq_len"] = 3
    elif changed == "seed":
        kwargs["seed"] = 5
    else:
        kwargs.update(rank=1, world_size=2)
    restored = PackedDataset(manifest, **kwargs)
    with pytest.raises(ValueError, match="strict resume"):
        restored.load_state_dict(original.state_dict())


def test_manifest_hash_checks_and_location_independent_identity(tmp_path):
    sources = {"a": {"train": ["1 2", "3"]}}
    first = prepare(tmp_path / "first", sources)
    second = prepare(tmp_path / "second", sources)
    assert first.fingerprint == second.fingerprint
    shard = second.root / second.shards("a", "train")[0]["tokens_path"]
    content = bytearray(shard.read_bytes())
    content[4] ^= 1
    shard.write_bytes(content)
    with pytest.raises(ValueError, match="SHA256"):
        PreparedManifest.load(second.root)
    assert PreparedManifest.load(second.root, verify_hashes=False).fingerprint == first.fingerprint


def test_prepare_refuses_overwrite_and_cleans_only_own_failed_directory(tmp_path):
    output = tmp_path / "data"
    original = prepare(output, {"a": {"train": ["1"]}})
    with pytest.raises(FileExistsError):
        prepare(output, {"a": {"train": ["2"]}})
    assert PreparedManifest.load(output).fingerprint == original.fingerprint
    with pytest.raises(ValueError, match="outside vocab"):
        prepare(tmp_path / "bad", {"a": {"train": ["64"]}})
    assert not (tmp_path / "bad").exists()


def test_text_and_jsonl_record_rules(tmp_path):
    text = tmp_path / "text.txt"
    text.write_text("one\n\n two  \n", encoding="utf-8")
    assert list(read_texts(text)) == ["one", " two  "]
    jsonl = tmp_path / "text.jsonl"
    jsonl.write_text('{"body":"first"}\n\n{"body":"second"}\n', encoding="utf-8")
    assert list(read_texts(jsonl, text_field="body")) == ["first", "second"]
    jsonl.write_text('{"body":4}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-string"):
        list(read_texts(jsonl, text_field="body"))


@pytest.mark.parametrize("weights", [{}, {"a": 0}, {"a": -1}, {"a": float("nan")}, {"a": float("inf")}, {"other": 1}])
def test_invalid_mixture_is_rejected(tmp_path, weights):
    manifest = prepare(tmp_path / "data", {"a": {"train": ["1"]}})
    with pytest.raises(ValueError):
        PackedDataset(manifest, seq_len=2, source_weights=weights)


def test_saved_source_order_is_checked_not_silently_accepted(tmp_path):
    manifest = prepare(tmp_path / "data", {"a": {"train": ["1", "2"]}})
    dataset = PackedDataset(manifest, seq_len=1)
    state = copy.deepcopy(dataset.state_dict())
    state["sources"]["a"]["order"].reverse()
    with pytest.raises(ValueError, match="shuffle order"):
        dataset.load_state_dict(state)
