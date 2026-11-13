import json

import pytest

torch = pytest.importorskip("torch")

from moeloop.data import PackedDataset, PreparedManifest
from moeloop.prepare import prepare_dataset

from subset_manifest import main

TOKENIZER = {"id": "test-tokenizer", "revision": "fixed-v1", "vocab_size": 64, "bos_token_id": 0, "eos_token_id": 63, "pad_token_id": 63}


@pytest.fixture
def data(tmp_path):
    """Two sources, 3-4 train shards each (about 10 tokens per shard), plus validation and test."""
    def texts(first, count):
        return [" ".join(str(first + i) for i in range(3)) for i in range(count)]
    sources = {
        "a": {"train": texts(1, 12), "validation": texts(20, 3), "test": texts(30, 3)},
        "b": {"train": texts(40, 9), "test": texts(50, 2)},
    }
    prepare_dataset(tmp_path / "data", sources, encode=lambda text: [int(value) for value in text.split()],
                    tokenizer=TOKENIZER, shard_tokens=10)
    return tmp_path / "data"


def train_shards(manifest):
    return [shard for name in manifest.source_names("train") for shard in manifest.shards(name, "train")]


def subset(data, tokens, name="subset.json"):
    assert main(["--manifest", str(data / "manifest.json"), "--train-tokens", str(tokens), "--output", str(data / name)]) == 0
    return PreparedManifest.load(data / name, verify_hashes=True)


def test_selected_shards_are_a_prefix_and_tokens_reach_the_target(data):
    full = PreparedManifest.load(data / "manifest.json")
    shards = train_shards(full)
    assert len(shards) >= 6
    target = shards[0]["num_tokens"] + shards[1]["num_tokens"] + 1  # needs a third shard
    kept = train_shards(subset(data, target))
    assert kept == shards[:len(kept)] and len(kept) == 3
    assert sum(shard["num_tokens"] for shard in kept) >= target > sum(shard["num_tokens"] for shard in kept[:-1])


def test_rule_runs_over_sources_in_order_and_drops_unused_train_splits(data):
    full = PreparedManifest.load(data / "manifest.json")
    a_tokens = sum(shard["num_tokens"] for shard in full.shards("a", "train"))
    # Exactly all of source a: b gets no train shard but keeps its test split.
    payload = subset(data, a_tokens).payload
    assert [source["name"] for source in payload["sources"]] == ["a", "b"]
    assert "train" not in payload["sources"][1]["splits"]
    assert payload["sources"][1]["splits"]["test"] == full.payload["sources"][1]["splits"]["test"]
    # One token more reaches into source b.
    assert subset(data, a_tokens + 1, "more.json").source_names("train") == ("a", "b")


def test_validation_test_and_tokenizer_are_unchanged(data, capsys):
    full = PreparedManifest.load(data / "manifest.json")
    capsys.readouterr()
    small = subset(data, 1)
    for old, new in zip(full.payload["sources"], small.payload["sources"]):
        assert {k: v for k, v in new["splits"].items() if k != "train"} == {k: v for k, v in old["splits"].items() if k != "train"}
    assert small.tokenizer == full.tokenizer
    assert {k: v for k, v in small.payload.items() if k != "sources"} == {k: v for k, v in full.payload.items() if k != "sources"}
    assert small.fingerprint != full.fingerprint


def test_report_counts_and_fingerprint(data, capsys):
    shards = train_shards(PreparedManifest.load(data / "manifest.json"))
    capsys.readouterr()
    small = subset(data, shards[0]["num_tokens"] + 1)
    report = json.loads(capsys.readouterr().out)
    assert report["train_shards"] == 2
    assert report["train_tokens"] == shards[0]["num_tokens"] + shards[1]["num_tokens"]
    assert report["train_documents"] == shards[0]["num_documents"] + shards[1]["num_documents"]
    assert report["fingerprint"] == small.fingerprint
    assert not list(data.glob("*.tmp"))


def test_full_train_split_is_allowed(data):
    full = PreparedManifest.load(data / "manifest.json")
    total = sum(shard["num_tokens"] for shard in train_shards(full))
    assert train_shards(subset(data, total)) == train_shards(full)


def test_subset_stream_starts_a_new_shuffled_epoch_after_its_tokens(data):
    small = subset(data, 50)  # 10 of source a's 12 documents; b is dropped from train
    dataset = PackedDataset(small, seq_len=4, seed=3, source_weights={"a": 1})
    source = dataset.sources["a"]
    source_documents = sum(shard["num_documents"] for shard in small.shards("a", "train"))
    assert source_documents == 10 and small.source_names("train") == ("a",)
    first_order = list(source.order)
    assert sorted(first_order) == list(range(source_documents))
    while source.epoch == 0:
        next(dataset)
    assert source.counts["documents"] >= source_documents
    assert sorted(source.order) == sorted(first_order) and source.order != first_order


def test_refuses_existing_output(data):
    (data / "taken.json").write_text("{}")
    with pytest.raises(SystemExit):
        main(["--manifest", str(data / "manifest.json"), "--train-tokens", "10", "--output", str(data / "taken.json")])
    assert (data / "taken.json").read_text() == "{}"


def test_refuses_more_than_the_train_split_and_nonpositive_counts(data):
    total = sum(shard["num_tokens"] for shard in train_shards(PreparedManifest.load(data / "manifest.json")))
    for tokens in (total + 1, 0):
        with pytest.raises(SystemExit):
            main(["--manifest", str(data / "manifest.json"), "--train-tokens", str(tokens), "--output", str(data / "x.json")])
        assert not (data / "x.json").exists()


def test_refuses_output_outside_the_manifest_directory(data, tmp_path):
    with pytest.raises(SystemExit):
        main(["--manifest", str(data / "manifest.json"), "--train-tokens", "10", "--output", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()
