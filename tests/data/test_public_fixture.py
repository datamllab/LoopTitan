from __future__ import annotations

import json
import hashlib
import struct

import pytest

torch = pytest.importorskip("torch")

from moeloop.data import PackedDataset, PreparedManifest
from moeloop import prepare as prepare_module


TOKENIZER = {"id": "fixture", "revision": "fixed", "vocab_size": 256, "bos_token_id": 0, "eos_token_id": 255}
LIMITS = {"train": 6, "validation": 4, "test": 4}


def records():
    # SHA256(text UTF-8), first 8 bytes big endian, modulo 100:
    # train: "1 2" -> 93, "3 4" -> 43; validation -> 1; test -> 0.
    yield {"id": "first", "text": "1 2"}
    yield {"id": "first", "text": "50 1 199"}  # Repeated ID, different content.
    yield {"id": "copy", "text": "1 2"}  # Repeated content, different ID.
    yield {"id": "valid", "text": "16 17 65"}
    yield {"id": "heldout", "text": "50 1 199"}
    yield {"id": "last", "text": "3 4"}
    raise AssertionError("preparation read beyond the fulfilled document budgets")


def prepare(path, *, documents=None, limits=None):
    return prepare_module.prepare_document_subset(
        path, records() if documents is None else documents,
        encode=lambda text: [int(token) for token in text.split()],
        tokenizer=TOKENIZER, label_tokens=LIMITS if limits is None else limits,
        source_manifest={"id": "public-fixture", "revision": "immutable-source"},
        shard_tokens=8,
    )


def identities(manifest, split):
    return [
        json.loads(line)
        for shard in manifest.shards("public", split)
        for line in (manifest.root / shard["documents_path"]).read_text().splitlines()
    ]


def test_public_subset_deduplicates_ids_and_content_before_three_way_isolation(tmp_path):
    manifest = prepare(tmp_path / "fixture")
    expected = {"train": (["first", "last"], [1, 2, 255, 3, 4, 255]),
                "validation": (["valid"], [16, 17, 65, 255]),
                "test": (["heldout"], [50, 1, 199, 255])}
    hashes = []
    for split, (ids, targets) in expected.items():
        docs = identities(manifest, split)
        assert [doc["id"] for doc in docs] == ids
        hashes.extend(doc["content_sha256"] for doc in docs)
        dataset = PackedDataset(manifest, split=split, seq_len=4, infinite=False, shuffle=False)
        actual = [token for _, labels in dataset for token in labels.tolist() if token != -100]
        assert actual == targets
    assert len(hashes) == len(set(hashes))
    assert manifest.payload["provenance"]["selection"]["actual_label_tokens"] == LIMITS
    assert manifest.payload["provenance"]["selection"]["duplicate_documents"] == 2


def test_split_for_takes_the_first_eight_digest_bytes_big_endian_modulo_100():
    digest = lambda value: value.to_bytes(8, "big") + b"\xff" * 24  # Later bytes never matter.
    assert [prepare_module.split_for(digest(value)) for value in (0, 1, 2, 99, 100, 101, 102, 2**64 - 1)] == [
        "test", "validation", "train", "train", "test", "validation", "train", "train"]  # 2**64-1 % 100 == 15
    # The fixture texts' documented buckets: "1 2" -> 93, "16 17 65" -> 1, "50 1 199" -> 0.
    content = lambda text: hashlib.sha256(text.encode()).digest()
    assert [prepare_module.split_for(content(text)) for text in ("1 2", "16 17 65", "50 1 199")] == ["train", "validation", "test"]
    assert prepare_module.SPLIT_BUCKETS == {"test": [0], "validation": [1], "train": [2, 99]}


def test_a_document_skipped_by_a_full_split_still_claims_its_id_and_content(tmp_path):
    # "1 10" -> bucket 0 (test), like "50 1 199"; "1 2" and "3 4" -> train.
    documents = [{"id": "heldout", "text": "50 1 199"}, {"id": "valid", "text": "16 17 65"},
                 {"id": "late", "text": "1 10"},  # Test is full: skipped, yet its ID is now taken.
                 {"id": "late", "text": "1 2"},  # Train has room, but the ID repeats: a duplicate.
                 {"id": "first", "text": "3 4"}]
    manifest = prepare(tmp_path / "fixture", documents=documents, limits={"train": 3, "validation": 4, "test": 4})
    assert [doc["id"] for doc in identities(manifest, "train")] == ["first"]
    assert [doc["id"] for doc in identities(manifest, "test")] == ["heldout"]
    assert manifest.payload["provenance"]["selection"]["duplicate_documents"] == 1


def test_every_preparation_writes_the_document_rules_the_loader_requires(tmp_path, monkeypatch):
    from moeloop.data import DOCUMENT_RULES

    subset = prepare(tmp_path / "subset")
    source, assets = packed_source(tmp_path, monkeypatch)
    packed = import_packed(tmp_path / "imported", source, assets)
    for manifest in (subset, packed):
        assert manifest.payload["document_rules"] == DOCUMENT_RULES
        assert manifest.payload["provenance"]["selection"]["buckets"] == prepare_module.SPLIT_BUCKETS
    payload = json.loads((subset.root / "manifest.json").read_text())
    payload["document_rules"] = dict(DOCUMENT_RULES, add_bos=False)
    (subset.root / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="document packing rules"):
        PreparedManifest.load(subset.root)


def test_document_identity_is_location_independent_and_protects_strict_resume(tmp_path):
    first = prepare(tmp_path / "one")
    second = prepare(tmp_path / "two")
    assert first.fingerprint == second.fingerprint
    original = PackedDataset(first, seq_len=2, seed=13)
    next(original)
    saved = json.loads(json.dumps(original.state_dict()))
    restored = PackedDataset(second, seq_len=2, seed=13)
    restored.load_state_dict(saved)
    for _ in range(5):
        left, right = next(original), next(restored)
        assert torch.equal(left[1], right[1])
        assert all(torch.equal(left[0][key], right[0][key]) for key in left[0])
    changed = list(records_without_sentinel())
    changed[0] = dict(changed[0], id="changed-original-id")
    # Keep the duplicate ID duplicate, so token documents remain identical.
    changed[1] = dict(changed[1], id="changed-original-id")
    altered = prepare(tmp_path / "three", documents=changed)
    assert altered.shards("public", "train")[0]["tokens_sha256"] == first.shards("public", "train")[0]["tokens_sha256"]
    with pytest.raises(ValueError, match="identity changed"):
        PackedDataset(altered, seq_len=2, seed=13).load_state_dict(saved)


def records_without_sentinel():
    iterator = records()
    return [next(iterator) for _ in range(6)]


def test_insufficient_public_source_leaves_no_manifest_and_cannot_overwrite(tmp_path):
    output = tmp_path / "fixture"
    with pytest.raises(ValueError, match="insufficient.*train"):
        prepare(output, documents=records_without_sentinel(), limits=dict(LIMITS, train=100))
    assert not output.exists()
    manifest = prepare(output)
    with pytest.raises(FileExistsError):
        prepare(output)
    assert PreparedManifest.load(output).fingerprint == manifest.fingerprint


def test_manifest_checks_document_identity_files(tmp_path):
    manifest = prepare(tmp_path / "fixture")
    path = manifest.root / manifest.shards("public", "test")[0]["documents_path"]
    path.write_text(path.read_text().replace("heldout", "different"))
    with pytest.raises(ValueError, match="documents SHA256"):
        PreparedManifest.load(manifest.root)


def test_preparation_closes_bounded_source_when_budgets_stop_iteration(tmp_path):
    closed = []

    def source():
        try:
            yield from records()
        finally:
            closed.append(True)

    documents = source()
    prepare(tmp_path / "fixture", documents=documents)
    assert closed == [True]


def test_cached_parquet_checks_frozen_source_bytes_before_reading(tmp_path, monkeypatch):
    parquet = pytest.importorskip("pyarrow.parquet")
    arrow = pytest.importorskip("pyarrow")
    import hashlib

    path = tmp_path / "cached.parquet"
    parquet.write_table(arrow.Table.from_pylist(records_without_sentinel()), path)
    monkeypatch.setattr(prepare_module, "FINEWEB_EDU_FILES", ({
        "path": "fixed-source.parquet", "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    },))
    loaded = prepare_module._fineweb_records(source_file=path)
    manifest = prepare(tmp_path / "fixture", documents=loaded)
    assert [item["id"] for item in identities(manifest, "test")] == ["heldout"]
    corrupted = bytearray(path.read_bytes())
    corrupted[8] ^= 1
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="source size/SHA256 mismatch"):
        list(prepare_module._fineweb_records(source_file=path))


def packed_source(tmp_path, monkeypatch, *, terminal=255):
    tokenizers = pytest.importorskip("tokenizers")
    source, assets = tmp_path / "packed", tmp_path / "assets"
    source.mkdir()
    assets.mkdir()
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {str(index): index for index in range(255)} | {"<eos>": 255}, unk_token="0"))
    backend.save(str(assets / "tokenizer.json"))
    (assets / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast",
        "eos_token": "<eos>", "bos_token": None, "pad_token": "<eos>"}))
    tokenizer_sha = hashlib.sha256((assets / "tokenizer.json").read_bytes()).hexdigest()
    monkeypatch.setattr(prepare_module, "DCLM_TOKENIZER", {"id": "fixture", "revision": "fixed",
        "tokenizer_json_sha256": tokenizer_sha, "vocab_size": 256, "eos_token_id": 255}, raising=False)
    # Source-token SHA256 buckets: [1,2,255]->5, [3,4,255]->27,
    # [67,3,255]->1 (validation), [74,3,255]->0 (test).
    chunks = []
    documents = ([[1, 2, terminal], [1, 2, terminal], [67, 3, 255], [74, 3, 255]], [[3, 4, 255]])
    for number, docs in enumerate(documents):
        name = f"train-{number:05d}"
        content = {"tokens": b"".join(struct.pack("<3I", *doc) for doc in docs),
                   "docs": b"".join(struct.pack("<q", 3 * (index + 1)) for index in range(len(docs))),
                   "dochash": b"".join(struct.pack("<Q", index + 11) for index in range(len(docs)))}
        for kind, payload in content.items():
            (source / f"{name}.{kind}.bin").write_bytes(payload)
        chunks.append({"name": name, "tokens": 3 * len(docs), "docs": len(docs),
                       **{f"{kind}_sha256": hashlib.sha256(payload).hexdigest() for kind, payload in content.items()}})
    manifest = {"format": "loopmoe-packed-v1", "dtype": "uint32", "tokenizer": {
        "tokenizer_json_sha256": tokenizer_sha, "eos_id": 255, "bos_id": None, "vocab_size": 256},
        "tool_commit": "0" * 40, "splits": {"train": {"chunks": chunks}}}
    path = source / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, assets


def import_packed(output, source, assets, **kwargs):
    return prepare_module.prepare_dclm_packed(output, source_manifest=source,
        tokenizer_directory=assets, label_tokens={"train": 6, "validation": 3, "test": 3}, **kwargs)


def test_packed_import_preserves_encoded_documents_and_freezes_new_isolated_splits(tmp_path, monkeypatch):
    source, assets = packed_source(tmp_path, monkeypatch)
    manifest = import_packed(tmp_path / "one", source, assets, shard_tokens=8)
    second = import_packed(tmp_path / "two", source, assets, shard_tokens=8)
    assert manifest.fingerprint == second.fingerprint
    expected = {"train": [1, 2, 255, 3, 4, 255], "validation": [67, 3, 255], "test": [74, 3, 255]}
    all_hashes = []
    for split, labels in expected.items():
        dataset = PackedDataset(manifest, split=split, seq_len=4, infinite=False, shuffle=False)
        assert [token for _, target in dataset for token in target.tolist() if token != -100] == labels
        docs = identities(manifest, split)
        all_hashes.extend(doc["token_sha256"] for doc in docs)
        assert all("source_text_blake2b64" in doc and "content_sha256" not in doc for doc in docs)
    assert len(all_hashes) == len(set(all_hashes)) == 4
    provenance = manifest.payload["provenance"]
    assert provenance["selection"]["duplicate_documents"] == 1
    assert provenance["dataset"]["upstream_revision"] is None
    assert provenance["dataset"]["revision"].startswith("sha256:")
    assert [chunk["name"] for chunk in provenance["dataset"]["chunks"]] == ["train-00000", "train-00001"]
    assert manifest.tokenizer["bos_token_id"] == manifest.tokenizer["eos_token_id"] == manifest.tokenizer["pad_token_id"] == 255


def test_packed_import_accepts_explicit_chunk_order_beyond_the_legacy_first_two(tmp_path, monkeypatch):
    source, assets = packed_source(tmp_path, monkeypatch)
    parent = json.loads(source.read_text())
    content = {"tokens": struct.pack("<3I", 7, 8, 255), "docs": struct.pack("<q", 3), "dochash": struct.pack("<Q", 99)}
    chunk = {"name": "train-00002", "tokens": 3, "docs": 1}
    for kind, raw in content.items():
        (source.parent / f"train-00002.{kind}.bin").write_bytes(raw)
        chunk[f"{kind}_sha256"] = hashlib.sha256(raw).hexdigest()
    parent["splits"]["train"]["chunks"].append(chunk)
    source.write_text(json.dumps(parent))
    names = ["train-00002", "train-00000", "train-00001"]
    manifest = prepare_module.prepare_dclm_packed(tmp_path / "explicit", source_manifest=source,
        tokenizer_directory=assets, label_tokens={"train": 9, "validation": 3, "test": 3}, chunk_names=names)
    assert [item["name"] for item in manifest.payload["provenance"]["dataset"]["chunks"]] == names
    dataset = PackedDataset(manifest, seq_len=4, infinite=False, shuffle=False)
    assert [token for _, targets in dataset for token in targets.tolist() if token != -100] == [7, 8, 255, 1, 2, 255, 3, 4, 255]


@pytest.mark.parametrize("damage", ["payload", "eos", "boundary"])
def test_packed_import_rejects_corrupt_payload_eos_and_document_boundaries(tmp_path, monkeypatch, damage):
    source, assets = packed_source(tmp_path, monkeypatch, terminal=254 if damage == "eos" else 255)
    value = json.loads(source.read_text())
    if damage == "payload":
        (source.parent / "train-00000.tokens.bin").write_bytes(b"changed source")
    elif damage == "boundary":
        path = source.parent / "train-00000.docs.bin"
        path.write_bytes(struct.pack("<4q", 3, 2, 9, 12))
        value["splits"]["train"]["chunks"][0]["docs_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        source.write_text(json.dumps(value))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="SHA256|size|EOS|offset|boundar"):
        import_packed(output, source, assets)
    assert not output.exists()
