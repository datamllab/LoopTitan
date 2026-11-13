"""Exact UTF-8 byte lengths of tokenizer vocabulary entries, for bits-per-byte.

Only byte-level BPE (GPT-2, Qwen3, Llama 3) is supported: its vocabulary strings
spell raw bytes through GPT-2's printable byte alphabet, so each string maps back
to exactly one byte sequence. Other tokenizer types raise instead of guessing.
Reads ``tokenizer.json`` directly; neither tokenizers nor Transformers is imported.
A model without ``type`` (the legacy layout of GPT-2's official file) is BPE when
it has ``vocab`` and ``merges``, as tokenizers' own deserializer decides.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Sequence


class NotByteLevelBPE(ValueError):
    """The tokenizer is not a byte-level BPE whose byte lengths are exact."""


def _byte_decoder() -> dict[str, int]:
    """Invert GPT-2 ``bytes_to_unicode``: printable byte -> itself, others -> 256 + n."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    shifted = [byte for byte in range(256) if byte not in printable]
    pairs = [(byte, byte) for byte in printable] + [(byte, 256 + index) for index, byte in enumerate(shifted)]
    return {chr(character): byte for byte, character in pairs}


BYTE_DECODER = _byte_decoder()


def _parts(node: Any, key: str) -> list[dict[str, Any]]:
    if not node:
        return []
    return [part for child in node[key] for part in _parts(child, key)] if node.get("type") == "Sequence" else [node]


def token_byte_lengths(tokenizer_dir: str | Path) -> tuple[int, ...]:
    """Return the UTF-8 byte length of every token ID of a byte-level BPE tokenizer.

    Special and added tokens count 0 bytes, so EOS/BOS/PAD never enter a byte
    denominator. Limits: bytes are those of the text after the tokenizer's
    normalizer (Qwen3 applies NFC; GPT-2 and Llama 3 have none), and an added
    non-special token written literally in a document (e.g. Qwen3 ``<think>``)
    is undercounted by its length.
    """
    backend = json.loads((Path(tokenizer_dir) / "tokenizer.json").read_text(encoding="utf-8"))
    model = backend.get("model", {})
    model_type = model.get("type", "BPE" if "vocab" in model and "merges" in model else None)
    pre_tokenizers = [part for part in _parts(backend.get("pre_tokenizer"), "pretokenizers") if part.get("type") == "ByteLevel"]
    normalizers = [part.get("type") for part in _parts(backend.get("normalizer"), "normalizers")]
    if (model_type != "BPE" or model.get("byte_fallback") or not pre_tokenizers
            or "ByteLevel" not in [part.get("type") for part in _parts(backend.get("decoder"), "decoders")]):
        raise NotByteLevelBPE("byte lengths need a byte-level BPE tokenizer (ByteLevel pre-tokenizer and decoder)")
    if any(part.get("add_prefix_space") for part in pre_tokenizers):
        raise NotByteLevelBPE("ByteLevel add_prefix_space inserts bytes absent from the text")
    if normalizers not in ([], ["NFC"]):
        raise NotByteLevelBPE(f"normalizer {normalizers} changes text bytes; only none or NFC is supported")
    added = {token["id"] for token in backend.get("added_tokens", [])}
    lengths = dict.fromkeys(added, 0)
    for token, token_id in model["vocab"].items():
        if token_id in added:
            continue
        if any(character not in BYTE_DECODER for character in token):
            raise NotByteLevelBPE(f"vocabulary entry {token!r} is outside the byte-level alphabet")
        lengths[token_id] = len(token)  # One alphabet character spells exactly one byte.
    if sorted(lengths) != list(range(len(lengths))):
        raise ValueError("tokenizer IDs are not contiguous from 0")
    return tuple(lengths[token_id] for token_id in range(len(lengths)))


def byte_table_sha256(lengths: Sequence[int]) -> str:
    """SHA-256 of the table as consecutive little-endian uint32 values, ID order."""
    return hashlib.sha256(struct.pack(f"<{len(lengths)}I", *lengths)).hexdigest()
