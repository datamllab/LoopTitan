from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("torch")
arrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")

from moeloop import prepare as preparation
from moeloop.data import PackedDataset, PreparedManifest


def source_fixture(tmp_path, monkeypatch):
    tokenizers = pytest.importorskip("tokenizers")
    assets = tmp_path / "assets"
    assets.mkdir()
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"<unk>": 0, "<eos>": 255} | {str(index): index for index in range(1, 255)}, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.save(str(assets / "tokenizer.json"))
    monkeypatch.setattr(preparation, "DCLM_TOKENIZER", {
        "id": "fixture", "revision": "fixed", "vocab_size": 256, "eos_token_id": 255,
        "tokenizer_json_sha256": hashlib.sha256((assets / "tokenizer.json").read_bytes()).hexdigest(),
    })
    source = tmp_path / "source"
    source.mkdir()
    # Exact UTF-8 SHA256 buckets: train=93/43, validation=1, test=0.
    rows = [
        [{"id": "first", "text": "1 2"}, {"id": "copy", "text": "1 2"}],
        [{"id": "first", "text": "duplicate id"}, {"id": "valid", "text": "16 17 65"},
         {"id": "heldout", "text": "50 1 199"}, {"id": "last", "text": "3 4"}],
    ]
    files = []
    for index, documents in enumerate(rows):
        path = source / f"{index:03d}.parquet"
        parquet.write_table(arrow.Table.from_pylist(documents), path, row_group_size=1)
        files.append({"path": path.name, "size": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"format": "moeloop-public-parquet-source-v1", "id": preparation.DCLM_ID,
                "revision": "1" * 40, "files": files}
    return manifest, source, assets


def prepare(output, frozen, source, assets, **kwargs):
    return preparation.prepare_dclm_public(output, source_manifest=frozen,
        source_directory=source, tokenizer_directory=assets,
        label_tokens={"train": 6, "validation": 4, "test": 4}, shard_tokens=8, **kwargs)


def test_ordered_dclm_shards_preserve_whole_documents_and_global_split_isolation(tmp_path, monkeypatch):
    frozen, source, assets = source_fixture(tmp_path, monkeypatch)
    manifest = prepare(tmp_path / "out", frozen, source, assets)
    second = prepare(tmp_path / "another", frozen, source, assets)
    assert second.fingerprint == manifest.fingerprint
    expected = {"train": [1, 2, 255, 3, 4, 255], "validation": [16, 17, 65, 255],
                "test": [50, 1, 199, 255]}
    hashes = []
    for split, labels in expected.items():
        dataset = PackedDataset(manifest, split=split, seq_len=4, infinite=False, shuffle=False)
        assert [value for _, targets in dataset for value in targets.tolist() if value != -100] == labels
        for shard in manifest.shards("public", split):
            for line in (manifest.root / shard["documents_path"]).read_text().splitlines():
                identity = json.loads(line)
                hashes.append(identity["content_sha256"])
                assert identity["source_file"] in [item["path"] for item in frozen["files"]]
                assert identity["source_row"] >= 0
    assert len(hashes) == len(set(hashes)) == 4
    provenance = manifest.payload["provenance"]
    assert provenance["dataset"]["revision"] == "1" * 40
    assert provenance["dataset"]["files"] == frozen["files"]
    assert provenance["selection"]["duplicate_documents"] == 2
    assert provenance["selection"]["actual_label_tokens"] == {"train": 6, "validation": 4, "test": 4}


def test_interrupted_dclm_resumes_at_committed_source_shard_with_identical_manifest(tmp_path, monkeypatch):
    frozen, source, assets = source_fixture(tmp_path, monkeypatch)
    uninterrupted = prepare(tmp_path / "reference", frozen, source, assets)
    original_records = preparation._dclm_records
    original_append = preparation._SplitWriter.append

    def interrupted(writer, tokens, identity=None):
        original_append(writer, tokens, identity)
        if identity["id"] == "valid":
            raise KeyboardInterrupt("interrupted after partial output")

    monkeypatch.setattr(preparation._SplitWriter, "append", interrupted)
    output = tmp_path / "resume"
    with pytest.raises(KeyboardInterrupt):
        prepare(output, frozen, source, assets)
    assert not (output / "manifest.json").exists()
    assert list((output / "public" / "validation").glob("*.tokens.bin"))
    with sqlite3.connect(output / ".prepare.sqlite3") as connection:
        state = json.loads(connection.execute("SELECT state FROM checkpoint").fetchone()[0])
    assert state["next_file"] == 1
    with pytest.raises(FileExistsError):
        prepare(output, frozen, source, assets)
    with pytest.raises(ValueError, match="identity"):
        prepare(output, dict(frozen, revision="2" * 40), source, assets, resume=True)

    # Resumption must neither tokenize nor read the committed first source shard.
    def resumed(path, source_file):
        assert source_file == frozen["files"][1]["path"]
        yield from original_records(path, source_file)

    monkeypatch.setattr(preparation._SplitWriter, "append", original_append)
    monkeypatch.setattr(preparation, "_dclm_records", resumed)
    restored = prepare(output, frozen, source, assets, resume=True)
    assert restored.fingerprint == uninterrupted.fingerprint
    assert PreparedManifest.load(output).fingerprint == uninterrupted.fingerprint


def test_dclm_source_corruption_and_insufficient_supply_never_publish_manifest(tmp_path, monkeypatch):
    frozen, source, assets = source_fixture(tmp_path, monkeypatch)
    output = tmp_path / "short"
    with pytest.raises(ValueError, match="insufficient.*train"):
        prepare(output, dict(frozen, files=frozen["files"][:1]), source, assets)
    assert not (output / "manifest.json").exists()
    path = source / frozen["files"][0]["path"]
    path.write_bytes(b"corrupted source")
    with pytest.raises(ValueError, match="size/SHA256"):
        prepare(tmp_path / "corrupted", frozen, source, assets)
    assert not (tmp_path / "corrupted" / "manifest.json").exists()


def test_dclm_cli_uses_explicit_pinned_source_and_refuses_an_implicit_fineweb_default(tmp_path, monkeypatch, capsys):
    from recipe_data import main

    frozen, source, assets = source_fixture(tmp_path, monkeypatch)
    source_json = tmp_path / "source.json"
    source_json.write_text(json.dumps(frozen))
    output = tmp_path / "cli"
    main(["--dclm-source", str(source_json), "--source-directory", str(source),
          "--tokenizer", str(assets), "--output", str(output), "--train-label-tokens", "6",
          "--validation-label-tokens", "4", "--test-label-tokens", "4", "--shard-tokens", "8"])
    assert json.loads(capsys.readouterr().out)["fingerprint"] == PreparedManifest.load(output).fingerprint
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "implicit-source")])


def test_dclm_resume_rechecks_committed_output_hashes(tmp_path, monkeypatch):
    frozen, source, assets = source_fixture(tmp_path, monkeypatch)
    original = preparation._dclm_records

    def interrupted(path, source_file):
        if source_file == frozen["files"][1]["path"]:
            raise KeyboardInterrupt
        yield from original(path, source_file)

    monkeypatch.setattr(preparation, "_dclm_records", interrupted)
    output = tmp_path / "resume-corruption"
    with pytest.raises(KeyboardInterrupt):
        prepare(output, frozen, source, assets)
    shard = next((output / "public" / "train").glob("*.tokens.bin"))
    shard.write_bytes(b"corrupted committed output")
    with pytest.raises(ValueError, match="committed output SHA256"):
        prepare(output, frozen, source, assets, resume=True)
