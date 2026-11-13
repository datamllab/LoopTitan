from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from moeloop.data import PackedDataset, PreparedManifest
from moeloop.prepare import GPT2_ID, GPT2_REVISION, WIKITEXT_ID, WIKITEXT_REVISION, main, prepare_wikitext2


def make_tokenizer(path: Path):
    tokenizers = pytest.importorskip("tokenizers")
    path.mkdir()
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2, "first": 3, "second": 4, "third": 5}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    # Input tokenizer defaults must not silently truncate or pad stored documents.
    tokenizer.enable_truncation(max_length=1)
    tokenizer.enable_padding(length=10, pad_id=2, pad_token="</s>")
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}))


def test_local_cli_uses_same_packer_and_copies_checked_portable_tokenizer(tmp_path, capsys):
    tokenizer_path = tmp_path / "tokenizer"
    make_tokenizer(tokenizer_path)
    train = tmp_path / "train.jsonl"
    train.write_text('{"text":"first second third"}\n', encoding="utf-8")
    validation = tmp_path / "validation.txt"
    validation.write_text("second\n", encoding="utf-8")
    output = tmp_path / "prepared"
    main(["--input", str(train), "--validation", str(validation), "--tokenizer", str(tokenizer_path), "--bos-token-id", "1", "--eos-token-id", "2", "--output", str(output)])
    manifest = PreparedManifest.load(output)
    assert manifest.tokenizer["path"] == "tokenizer"
    assert (output / "tokenizer" / "tokenizer.json").is_file()
    assert set(manifest.tokenizer["files"]) == {"tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json"}
    inputs, labels = next(PackedDataset(manifest, seq_len=6, infinite=False, shuffle=False))
    assert inputs["input"].tolist() == [1, 3, 4, 5, 2, 2]
    assert labels.tolist() == [3, 4, 5, 2, -100, -100]
    assert json.loads(capsys.readouterr().out)["fingerprint"] == manifest.fingerprint
    with pytest.raises(SystemExit):
        main(["--input", str(train), "--tokenizer", str(tokenizer_path), "--bos-token-id", "1", "--eos-token-id", "2", "--output", str(output)])


def test_public_recipe_uses_pinned_identities_and_each_original_split(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("huggingface_hub")
    calls = {}
    tokenizer_path = tmp_path / "downloaded"
    tokenizer_path.mkdir()
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"<unk>": 0, "1": 1, "2": 2, "3": 3, "<end>": 50256}, unk_token="<unk>"))
    backend.save(str(tokenizer_path / "tokenizer.json"))

    def download(repo_id, *, revision, allow_patterns):
        calls["tokenizer"] = (repo_id, revision, allow_patterns)
        return str(tokenizer_path)

    def dataset(repo_id, config, *, revision):
        calls["dataset"] = (repo_id, config, revision)
        return {"train": [{"text": "1"}, {"text": "  "}], "validation": [{"text": "2"}], "test": [{"text": "3"}]}

    tokenizer = SimpleNamespace(get_vocab_size=lambda: 50257, encode=lambda text, add_special_tokens: SimpleNamespace(ids=[int(text)]))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr("datasets.load_dataset", dataset)
    monkeypatch.setattr("moeloop.prepare._load_tokenizer", lambda path: tokenizer)
    manifest = prepare_wikitext2(tmp_path / "data")
    assert calls["dataset"] == (WIKITEXT_ID, "wikitext-2-raw-v1", WIKITEXT_REVISION)
    assert calls["tokenizer"][:2] == (GPT2_ID, GPT2_REVISION)
    assert "tokenizer.json" in calls["tokenizer"][2]
    assert not any("safetensors" in path or "pytorch_model" in path for path in calls["tokenizer"][2])
    for split, expected in (("train", 1), ("validation", 2), ("test", 3)):
        inputs, labels = next(PackedDataset(manifest, split=split, seq_len=3, infinite=False))
        assert inputs["input"].tolist() == [50256, expected, 50256]
        assert labels.tolist() == [expected, 50256, -100]


def test_manifest_rejects_role_and_decoder_conflicts_even_without_full_hash_scan(tmp_path):
    tokenizer_path = tmp_path / "tokenizer"
    make_tokenizer(tokenizer_path)
    text = tmp_path / "train.txt"
    text.write_text("first second\n")
    output = tmp_path / "prepared"
    main(["--input", str(text), "--tokenizer", str(tokenizer_path), "--bos-token-id", "1", "--eos-token-id", "2", "--output", str(output)])
    config_path = output / "tokenizer" / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["added_tokens_decoder"]["1"]["content"] = "third"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="added_tokens_decoder"):
        PreparedManifest.load(output, verify_hashes=False)
