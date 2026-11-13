"""One packed block stream per run; ranks take contiguous slices of each update."""

import pytest

torch = pytest.importorskip("torch")

from moeloop.data import PackedDataset
from moeloop.prepare import prepare_dataset


TOKENIZER = {"id": "test-tokenizer", "revision": "fixed-v1", "vocab_size": 64, "bos_token_id": 0, "eos_token_id": 63}
UPDATE_BLOCKS = 8


def manifest(path):
    # Unequal documents span block boundaries; few documents force epoch wraps.
    texts = {"short": [" ".join(str(1 + (i + j) % 60) for j in range(1 + i % 4)) for i in range(5)],
             "long": [" ".join(str(1 + (3 * i + j) % 60) for j in range(6 + 2 * i)) for i in range(3)]}
    return prepare_dataset(path, {name: {"train": items, "validation": items} for name, items in texts.items()},
                           encode=lambda text: [int(value) for value in text.split()], tokenizer=TOKENIZER)


def ranks(prepared, world, **kwargs):
    return [PackedDataset(prepared, seq_len=5, source_weights={"short": 2, "long": 1}, seed=17, rank=rank,
                          world_size=world, rank_blocks=UPDATE_BLOCKS // world, **kwargs) for rank in range(world)]


def update(datasets):
    """Blocks of one update, in rank order: rank r holds logical slice r."""
    return [next(dataset) for dataset in datasets for _ in range(UPDATE_BLOCKS // len(datasets))]


def flat(blocks):
    return [(tuple(tuple(t.tolist()) for t in inputs.values()), tuple(labels.tolist())) for inputs, labels in blocks]


def test_every_update_holds_the_same_blocks_in_the_same_order_for_1_2_4_ranks(tmp_path):
    prepared = manifest(tmp_path / "data")
    streams = {}
    for world in (1, 2, 4):
        datasets = ranks(prepared, world)
        streams[world] = [flat(update(datasets)) for _ in range(12)]
        counts = {}
        for dataset in datasets:
            for name, fields in dataset.source_counts.items():
                for field, value in fields.items():
                    counts[name, field] = counts.get((name, field), 0) + value
        streams[world, "counts"] = counts
    assert streams[1] == streams[2] == streams[4]
    assert streams[1, "counts"] == streams[2, "counts"] == streams[4, "counts"]
    # 12 updates of 40 positions cover every document several times.
    labels = sum(count for (_, field), count in streams[1, "counts"].items() if field == "label_tokens")
    assert labels == sum(sum(label != -100 for label in block[1]) for blocks in streams[1] for block in blocks)
    assert streams[1, "counts"]["short", "documents"] > 5 and streams[1, "counts"]["long", "documents"] > 3


def test_rank_resume_continues_its_slice_and_refuses_a_new_topology(tmp_path):
    prepared = manifest(tmp_path / "data")
    datasets = ranks(prepared, 4)
    for _ in range(5):
        update(datasets)
    state = datasets[3].state_dict()
    expected = [next(datasets[3]) for _ in range(6)]
    restored = ranks(prepared, 4)[3]
    restored.load_state_dict(state)
    assert flat(next(restored) for _ in range(6)) == flat(expected)
    with pytest.raises(ValueError, match="strict resume"):
        ranks(prepared, 2)[1].load_state_dict(state)


def test_finite_stream_ends_inside_a_round_and_ranks_keep_their_slices(tmp_path):
    prepared = manifest(tmp_path / "data")
    whole = flat(ranks(prepared, 1, split="validation")[0])
    datasets = ranks(prepared, 4, split="validation")
    slices = [flat(dataset) for dataset in datasets]
    assert len(whole) % UPDATE_BLOCKS  # the last round is partial
    for rank, blocks in enumerate(slices):
        expected = [block for index, block in enumerate(whole) if index % UPDATE_BLOCKS // 2 == rank]
        assert blocks == expected
