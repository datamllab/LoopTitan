from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from moeloop.tokens import NotByteLevelBPE, byte_table_sha256, token_byte_lengths


# The legacy layout of GPT-2's official tokenizer.json: the model has no "type"
# and its merges are "a b" strings. Whole-character tokens spell é (C3 A9),
# 你 (E4 BD A0) and 🙂 (F0 9F 99 82) in GPT-2's byte alphabet, so every entry is
# valid UTF-8 on its own and decodes losslessly.
SPECIAL = "<|endoftext|>"
VOCAB = {"h": 0, "e": 1, "l": 2, "o": 3, "Ġ": 4, "he": 5, "ll": 6, "hell": 7, "hello": 8, "Ġhello": 9,
         "Ã©": 10, "ä½ł": 11, "ðŁĻĤ": 12, SPECIAL: 13}
MERGES = ["h e", "l l", "he ll", "hell o", "Ġ hello"]
TABLE = (1, 1, 1, 1, 1, 2, 2, 4, 5, 6, 2, 3, 4, 0)
BPB_FIXTURE = Path(__file__).parent / "fixtures" / "bpb"


def untyped_backend():
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": 13, "special": True, "content": SPECIAL, "single_word": False, "lstrip": False,
                          "rstrip": False, "normalized": True}],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True},
        "model": {"dropout": None, "unk_token": None, "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": False, "vocab": VOCAB, "merges": MERGES},
    }


def write(directory, backend):
    (directory / "tokenizer.json").write_text(json.dumps(backend), encoding="utf-8")
    return directory


def test_untyped_bpe_gives_the_same_table_as_typed_bpe(tmp_path):
    backend = untyped_backend()
    assert token_byte_lengths(write(tmp_path, backend)) == TABLE
    backend["model"]["type"] = "BPE"
    assert token_byte_lengths(write(tmp_path, backend)) == TABLE


def test_untyped_model_without_merges_is_refused(tmp_path):
    backend = untyped_backend()
    del backend["model"]["merges"]
    with pytest.raises(NotByteLevelBPE, match="byte-level BPE"):
        token_byte_lengths(write(tmp_path, backend))


def test_table_matches_tokenizers_model_and_decoding(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    backend = untyped_backend()
    library = tokenizers.Tokenizer.from_file(str(write(tmp_path, backend) / "tokenizer.json"))
    assert isinstance(library.model, tokenizers.models.BPE)  # tokenizers reads the untyped model as BPE too
    table = token_byte_lengths(tmp_path)
    for token_id in range(library.get_vocab_size()):
        decoded = library.decode([token_id], skip_special_tokens=True)
        assert "�" not in decoded and table[token_id] == len(decoded.encode("utf-8")), token_id
    assert library.encode("hello hello").ids == [8, 9]

    del backend["model"]["merges"]
    with pytest.raises(Exception, match="ModelUntagged"):  # without merges tokenizers finds no BPE either
        tokenizers.Tokenizer.from_file(str(write(tmp_path, backend) / "tokenizer.json"))


def test_existing_tables_keep_their_hashes(tmp_path):
    assert byte_table_sha256(token_byte_lengths(BPB_FIXTURE)) == "a1953a9c4ec6adca5974a7ffefec9306acf53957bdc79a39bde48da43cfb40c6"
    backend = json.loads((BPB_FIXTURE / "tokenizer.json").read_text(encoding="utf-8"))
    del backend["model"]["type"]
    assert token_byte_lengths(write(tmp_path, backend)) == token_byte_lengths(BPB_FIXTURE)


def test_qwen3_table_hash_from_local_hf_cache():
    pytest.importorskip("torch")  # moeloop.prepare, which pins the Qwen3 tokenizer.json, imports torch
    hub = pytest.importorskip("huggingface_hub")
    from moeloop.prepare import DCLM_TOKENIZER

    path = hub.try_to_load_from_cache(DCLM_TOKENIZER["id"], "tokenizer.json")
    if not isinstance(path, str) or hashlib.sha256(Path(path).read_bytes()).hexdigest() != DCLM_TOKENIZER["tokenizer_json_sha256"]:
        pytest.skip("fixed Qwen3 tokenizer.json is not in the local HF cache")
    assert byte_table_sha256(token_byte_lengths(Path(path).parent)) == "b504f8826a5ed6fc6a83ceb55c1c96f6c757ada30c6e38c46995ae2e5b74e17c"
