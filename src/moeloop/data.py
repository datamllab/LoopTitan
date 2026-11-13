"""Immutable token documents, document-local packing, and resumable block mixtures.

Preparation writes uint32 token files and uint64 document offsets, both little-endian.
Each stored document includes its explicit BOS and EOS. A training sample contains
already aligned input/target pairs; the last EOS never predicts another document.
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
import mmap
import random
import struct
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .tokens import NotByteLevelBPE, token_byte_lengths


FORMAT = "moeloop-token-documents"
FORMAT_VERSION = 1
# How every stored document is framed and packed; each manifest records it.
DOCUMENT_RULES = {
    "add_bos": True,
    "append_eos": True,
    "label_shift": "within_document",
    "positions": "block_document_segment_zero",
    "padding": "explicit_input_mask",
}
# One packed block stream per run, split into contiguous rank slices (PackedDataset).
STREAM_RULE = "logical_blocks_v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _local_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"manifest file escapes its directory: {relative}")
    return path


def validate_tokenizer_roles(directory: str | Path, metadata: Mapping[str, Any]) -> None:
    """Check stored vocabulary IDs and both HF role maps without Transformers."""
    directory = Path(directory)
    backend = json.loads((directory / "tokenizer.json").read_text())
    vocab = backend.get("model", {}).get("vocab", {})
    vocab = vocab if isinstance(vocab, dict) else {row[0]: index for index, row in enumerate(vocab)}
    vocab.update({item["content"]: item["id"] for item in backend.get("added_tokens", [])})
    config_path = directory / "tokenizer_config.json"
    if not config_path.is_file():
        raise ValueError("prepared tokenizer assets lack tokenizer_config.json; prepare the dataset again")
    configs = [json.loads(config_path.read_text())]
    for token_id, token in configs[0].get("added_tokens_decoder", {}).items():
        token = token.get("content") if isinstance(token, dict) else token
        if vocab.get(token) != int(token_id):
            raise ValueError("tokenizer_config added_tokens_decoder differs from tokenizer.json vocabulary IDs")
    added_tokens = directory / "added_tokens.json"
    if added_tokens.is_file():
        for token, token_id in json.loads(added_tokens.read_text()).items():
            if vocab.get(token) != token_id:
                raise ValueError("added_tokens.json differs from tokenizer.json vocabulary IDs")
    legacy = directory / "special_tokens_map.json"
    if legacy.is_file():
        configs.append(json.loads(legacy.read_text()))
    for role in ("bos", "eos", "pad"):
        expected = metadata[f"{role}_token_id"]
        for config in configs:
            token = config.get(f"{role}_token")
            token = token.get("content") if isinstance(token, dict) else token
            if token is None or vocab.get(token) != expected:
                raise ValueError(f"prepared tokenizer {role.upper()} role differs from manifest ID {expected}; prepare the dataset again")
        declared = configs[0].get(f"{role}_token_id", expected)
        if declared != expected:
            raise ValueError(f"tokenizer_config {role}_token_id differs from manifest")


class PreparedManifest:
    """A checked manifest. Its fingerprint includes data, tokenizer, and packing rules."""

    def __init__(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.path = path.resolve()
        self.root = self.path.parent
        self._payload = copy.deepcopy(dict(payload))
        self.fingerprint = hashlib.sha256(_canonical(self._payload)).hexdigest()

    @classmethod
    def load(cls, path: str | Path, *, verify_hashes: bool = True) -> PreparedManifest:
        path = Path(path)
        if path.is_dir():
            path = path / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT or payload.get("version") != FORMAT_VERSION:
            raise ValueError("unsupported token-document manifest format/version")
        tokenizer = payload.get("tokenizer", {})
        for key in ("id", "revision", "vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
            if key not in tokenizer:
                raise ValueError(f"tokenizer metadata is missing {key}")
        vocab_size = tokenizer["vocab_size"]
        if not isinstance(vocab_size, int) or vocab_size <= 0 or vocab_size > 2**32:
            raise ValueError("vocab_size must fit unsigned 32-bit token IDs")
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = tokenizer[key]
            if not isinstance(value, int) or not 0 <= value < vocab_size:
                raise ValueError(f"invalid tokenizer {key}")
        if payload.get("document_rules") != DOCUMENT_RULES:
            raise ValueError("unsupported document packing rules")
        manifest = cls(path, payload)
        names: set[str] = set()
        sources = payload.get("sources", [])
        if not sources:
            raise ValueError("manifest has no sources")
        for source in sources:
            name = source.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("source names must be nonempty and unique")
            names.add(name)
            for split, shards in source.get("splits", {}).items():
                if not isinstance(split, str) or not shards:
                    raise ValueError(f"empty split for source {name}")
                for shard in shards:
                    documents, tokens = shard["num_documents"], shard["num_tokens"]
                    if not isinstance(documents, int) or not isinstance(tokens, int) or documents <= 0 or tokens < 2 * documents:
                        raise ValueError(f"invalid document/token counts for {name}/{split}")
                    for kind, width, length in (("tokens", 4, tokens), ("index", 8, documents + 1)):
                        data_path = _local_file(manifest.root, shard[f"{kind}_path"])
                        if data_path.stat().st_size != width * length:
                            raise ValueError(f"{kind} file size mismatch: {data_path}")
                        if verify_hashes and file_sha256(data_path) != shard[f"{kind}_sha256"]:
                            raise ValueError(f"{kind} SHA256 mismatch: {data_path}")
                    # Optional v1 metadata freezes original IDs/content hashes for
                    # public subsets without changing existing manifest identities.
                    if "documents_path" in shard:
                        documents_path = _local_file(manifest.root, shard["documents_path"])
                        if not documents_path.is_file():
                            raise ValueError(f"documents file is missing: {documents_path}")
                        if verify_hashes and file_sha256(documents_path) != shard.get("documents_sha256"):
                            raise ValueError(f"documents SHA256 mismatch: {documents_path}")
        for relative, expected in tokenizer.get("files", {}).items():
            tokenizer_path = _local_file(manifest.root, relative)
            if not tokenizer_path.is_file():
                raise ValueError(f"tokenizer file is missing: {relative}")
            if verify_hashes and file_sha256(tokenizer_path) != expected:
                raise ValueError(f"tokenizer SHA256 mismatch: {relative}")
        if tokenizer.get("files"):
            validate_tokenizer_roles(_local_file(manifest.root, tokenizer.get("path", "tokenizer")), tokenizer)
        return manifest

    @property
    def tokenizer(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload["tokenizer"])

    @property
    def payload(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)

    def source_names(self, split: str) -> tuple[str, ...]:
        return tuple(source["name"] for source in self._payload["sources"] if split in source["splits"])

    def shards(self, source_name: str, split: str) -> list[dict[str, Any]]:
        for source in self._payload["sources"]:
            if source["name"] == source_name:
                return copy.deepcopy(source["splits"][split])
        raise KeyError(source_name)


def manifest_byte_table(manifest: PreparedManifest) -> tuple[tuple[int, ...] | None, str | None]:
    """Return ``(byte_table, None)`` from the manifest's tokenizer assets, else ``(None, reason)``.

    The table holds the UTF-8 byte length of every tokenizer ID (``moeloop.tokens``) and
    is unpadded: it may be shorter than the manifest ``vocab_size``, never longer. The
    reason is that the manifest has no tokenizer assets or the tokenizer is not byte-level BPE.
    """
    if not manifest.tokenizer.get("files"):
        return None, "manifest has no tokenizer assets"
    try:
        table = token_byte_lengths(_local_file(manifest.root, manifest.tokenizer.get("path", "tokenizer")))
    except NotByteLevelBPE as error:
        return None, str(error)
    if len(table) > manifest.tokenizer["vocab_size"]:
        raise ValueError("tokenizer byte table is longer than the manifest vocab_size")
    return table, None


class _TokenShard:
    def __init__(self, root: Path, metadata: Mapping[str, Any]) -> None:
        self.root, self.metadata = root, dict(metadata)
        self._files: list[Any] = []
        self._tokens: mmap.mmap | None = None
        self._index: mmap.mmap | None = None

    def _open(self) -> None:
        if self._tokens is None:
            for key in ("tokens", "index"):
                stream = _local_file(self.root, self.metadata[f"{key}_path"]).open("rb")
                self._files.append(stream)
                setattr(self, f"_{key}", mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ))

    def document_bounds(self, index: int) -> tuple[int, int]:
        self._open()
        assert self._index is not None
        begin, end = struct.unpack_from("<QQ", self._index, 8 * index)
        if not 0 <= begin < end <= self.metadata["num_tokens"] or end - begin < 2:
            raise ValueError("invalid document offsets in token shard")
        return begin, end

    def read_tokens(self, begin: int, length: int) -> list[int]:
        self._open()
        assert self._tokens is not None
        return list(struct.unpack_from(f"<{length}I", self._tokens, 4 * begin))

    def close(self) -> None:
        for key in ("tokens", "index"):
            mapping = getattr(self, f"_{key}")
            if mapping is not None:
                mapping.close()
                setattr(self, f"_{key}", None)
        for stream in self._files:
            stream.close()
        self._files.clear()


def _seed(seed: int, source: str, epoch: int) -> int:
    message = f"moeloop-document-order-v1\0{seed}\0{source}\0{epoch}".encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little")


class _PackedSource:
    def __init__(
        self,
        manifest: PreparedManifest,
        name: str,
        split: str,
        seq_len: int,
        *,
        shuffle: bool,
        infinite: bool,
        seed: int,
    ) -> None:
        self.name, self.seq_len = name, seq_len
        self.shuffle, self.infinite, self.seed = shuffle, infinite, seed
        self.pad_id = manifest.tokenizer["pad_token_id"]
        self.shards = [_TokenShard(manifest.root, item) for item in manifest.shards(name, split)]
        self._open_shards: list[_TokenShard] = []
        self.ends: list[int] = []
        for shard in self.shards:
            self.ends.append((self.ends[-1] if self.ends else 0) + shard.metadata["num_documents"])
        self.epoch = 0
        self.cursor = 0
        self.order = self._order()
        self.active_document: int | None = None
        self.active_offset = 0
        self.document_serial = 0
        self.active_serial = -1
        self.finished = not bool(self.order)
        self.counts = {"blocks": 0, "input_tokens": 0, "label_tokens": 0, "documents": 0}

    def _order(self) -> list[int]:
        order = list(range(self.ends[-1]))
        if self.shuffle:
            random.Random(_seed(self.seed, self.name, self.epoch)).shuffle(order)
        return order

    def _bounds(self, document: int) -> tuple[_TokenShard, int, int]:
        shard_number = bisect.bisect_right(self.ends, document)
        local_index = document - (self.ends[shard_number - 1] if shard_number else 0)
        shard = self.shards[shard_number]
        if shard in self._open_shards:
            self._open_shards.remove(shard)
        elif len(self._open_shards) == 8:
            self._open_shards.pop(0).close()
        self._open_shards.append(shard)
        begin, end = shard.document_bounds(local_index)
        return shard, begin, end

    def _start_document(self) -> bool:
        if self.cursor == len(self.order):
            if not self.infinite or not self.order:
                self.finished = True
                return False
            self.epoch += 1
            self.order = self._order()
            self.cursor = 0
        self.active_document = self.order[self.cursor]
        self.cursor += 1
        self.active_offset = 0
        self.active_serial = self.document_serial
        self.document_serial += 1
        return True

    def next_block(self, read: bool = True) -> tuple[dict[str, torch.Tensor], torch.Tensor] | None:
        """Pack the next block. ``read=False`` passes another rank's block using offsets only."""
        inputs: list[int] = []
        labels: list[int] = []
        document_ids: list[int] = []
        positions: list[int] = []
        valid = 0
        while valid < self.seq_len:
            if self.active_document is None and not self._start_document():
                break
            assert self.active_document is not None
            shard, begin, end = self._bounds(self.active_document)
            remaining_pairs = end - begin - 1 - self.active_offset
            length = min(self.seq_len - valid, remaining_pairs)
            if read:
                if self.active_offset == 0:
                    self.counts["documents"] += 1
                tokens = shard.read_tokens(begin + self.active_offset, length + 1)
                inputs.extend(tokens[:-1])
                labels.extend(tokens[1:])
                document_ids.extend([self.active_serial] * length)
                # A continuation segment has no previous-block attention or KV state.
                positions.extend(range(length))
            valid += length
            self.active_offset += length
            if self.active_offset == end - begin - 1:
                self.active_document = None
                self.active_offset = 0
        if valid == 0:
            raise StopIteration
        if not read:
            return None
        padding = self.seq_len - valid
        inputs.extend([self.pad_id] * padding)
        labels.extend([-100] * padding)
        document_ids.extend([-1] * padding)
        positions.extend([0] * padding)
        self.counts["blocks"] += 1
        self.counts["input_tokens"] += valid
        self.counts["label_tokens"] += valid
        return {
            "input": torch.tensor(inputs, dtype=torch.int64),
            "positions": torch.tensor(positions, dtype=torch.int64),
            "document_ids": torch.tensor(document_ids, dtype=torch.int64),
            "input_mask": torch.arange(self.seq_len) < valid,
        }, torch.tensor(labels, dtype=torch.int64)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "order": self.order.copy(),
            "cursor": self.cursor,
            "active_document": self.active_document,
            "active_offset": self.active_offset,
            "document_serial": self.document_serial,
            "active_serial": self.active_serial,
            "finished": self.finished,
            "counts": self.counts.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        epoch = state["epoch"]
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("invalid source epoch")
        self.epoch = epoch
        if state["order"] != self._order():
            raise ValueError("source shuffle order does not match its saved epoch/configuration")
        cursor = state["cursor"]
        if not isinstance(cursor, int) or not 0 <= cursor <= len(state["order"]):
            raise ValueError("invalid source cursor")
        document, offset = state["active_document"], state["active_offset"]
        if document is not None:
            if cursor == 0 or document != state["order"][cursor - 1]:
                raise ValueError("active document does not match source cursor")
            _, begin, end = self._bounds(document)
            if not isinstance(offset, int) or not 0 <= offset < end - begin - 1:
                raise ValueError("invalid partial-document offset")
        elif offset != 0:
            raise ValueError("partial-document offset without an active document")
        for name in ("order", "cursor", "active_document", "active_offset", "document_serial", "active_serial", "finished", "counts"):
            setattr(self, name, copy.deepcopy(state[name]))

    def close(self) -> None:
        for shard in self._open_shards:
            shard.close()
        self._open_shards.clear()


def _as_tuple(value: Any) -> Any:
    return tuple(_as_tuple(item) for item in value) if isinstance(value, (tuple, list)) else value


class PackedDataset(IterableDataset):
    """Stateful equal-length source-block mixture; weights are input-token probabilities.

    Train splits shuffle documents and reloop by default. Other splits are finite
    and ordered. All ranks pack one logical block stream, which depends only on
    the data, seed, weights and ``seq_len``. It is split into rounds of
    ``world_size * rank_blocks`` blocks, and rank r takes the contiguous blocks
    ``[r * rank_blocks, (r + 1) * rank_blocks)`` of each round. With
    ``rank_blocks = global batch / world_size`` a round is one optimizer update,
    so each update holds the same blocks in the same order for any world size.
    Every rank walks the whole stream but reads tokens only for its own blocks.
    Strict continuation currently supports ``num_workers=0``; no worker prefetch
    state is hidden behind this contract. Save only at completed batch boundaries.
    """

    def __init__(
        self,
        manifest: str | Path | PreparedManifest,
        *,
        split: str = "train",
        seq_len: int,
        source_weights: Mapping[str, float] | None = None,
        shuffle: bool | None = None,
        infinite: bool | None = None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        rank_blocks: int = 1,
        verify_hashes: bool = True,
    ) -> None:
        super().__init__()
        self.manifest = manifest if isinstance(manifest, PreparedManifest) else PreparedManifest.load(manifest, verify_hashes=verify_hashes)
        if seq_len <= 0 or world_size <= 0 or rank_blocks <= 0 or not 0 <= rank < world_size:
            raise ValueError("require seq_len > 0, rank_blocks > 0 and 0 <= rank < world_size")
        self.seq_len = seq_len
        self.split, self.rank, self.world_size, self.seed = split, rank, world_size, seed
        self.rank_blocks = rank_blocks
        self.shuffle = (split == "train") if shuffle is None else shuffle
        self.infinite = (split == "train") if infinite is None else infinite
        names = self.manifest.source_names(split)
        if not names:
            raise ValueError(f"manifest has no {split!r} split")
        weights = dict(source_weights) if source_weights is not None else dict.fromkeys(names, 1.0)
        if set(weights) - set(names):
            raise ValueError(f"unknown source weights: {sorted(set(weights) - set(names))}")
        if not weights or any(not isinstance(value, (int, float)) or not 0 <= value < float("inf") for value in weights.values()):
            raise ValueError("source weights must be finite, nonnegative, and have positive sum")
        total = sum(weights.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("source weights must have a finite positive sum")
        self.weights = {name: float(weights[name] / total) for name in names if weights.get(name, 0) > 0}
        self.sources = {
            name: _PackedSource(self.manifest, name, split, seq_len, shuffle=self.shuffle, infinite=self.infinite, seed=seed)
            for name in self.weights
        }
        self._active = list(self.sources)
        self._rng = random.Random(_seed(seed, "mixture", 0))
        self._blocks = 0  # logical stream position, over all ranks' blocks

    @property
    def fingerprint(self) -> str:
        return self.manifest.fingerprint

    @property
    def source_counts(self) -> dict[str, dict[str, int]]:
        return {name: source.counts.copy() for name, source in self.sources.items()}

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        if get_worker_info() is not None:
            raise ValueError("PackedDataset strict resume requires num_workers=0")
        return self

    def __next__(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        while self._active:
            selected = self._rng.choices(self._active, weights=[self.weights[name] for name in self._active], k=1)[0]
            own = self._blocks % (self.world_size * self.rank_blocks) // self.rank_blocks == self.rank
            try:
                block = self.sources[selected].next_block(read=own)
            except StopIteration:
                self._active.remove(selected)
                continue
            self._blocks += 1
            if own:
                return block
        raise StopIteration

    def _configuration(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "seq_len": self.seq_len,
            "weights": self.weights.copy(),
            "shuffle": self.shuffle,
            "infinite": self.infinite,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "rank_blocks": self.rank_blocks,
            "num_workers": 0,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "manifest_fingerprint": self.fingerprint,
            "configuration": self._configuration(),
            "blocks": self._blocks,
            "active_sources": self._active.copy(),
            "mixture_rng": copy.deepcopy(self._rng.getstate()),
            "sources": {name: source.state_dict() for name, source in self.sources.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 2 or state.get("manifest_fingerprint") != self.fingerprint:
            raise ValueError("data state version (2: logical block stream) or manifest/tokenizer identity changed; strict resume is refused")
        if state.get("configuration") != self._configuration():
            raise ValueError("data configuration/topology changed; strict resume is refused")
        if set(state["sources"]) != set(self.sources) or len(set(state["active_sources"])) != len(state["active_sources"]) or set(state["active_sources"]) - set(self.sources):
            raise ValueError("saved source set does not match this dataset")
        for name, source in self.sources.items():
            source.load_state_dict(state["sources"][name])
        self._rng.setstate(_as_tuple(state["mixture_rng"]))
        self._active = list(state["active_sources"])
        self._blocks = state["blocks"]

    def close(self) -> None:
        for source in self.sources.values():
            source.close()


def build_document_mask(document_ids: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
    """Reference bool ``[B,1,S,S]`` allowed-attention mask, built once per batch.

    The native training adapter can use the same predicate to create a BlockMask,
    avoiding this quadratic materialization at long production sequence lengths.
    Padding remains explicit even when the pad token equals EOS.
    """
    if document_ids.ndim != 2 or document_ids.shape != input_mask.shape or input_mask.dtype != torch.bool:
        raise ValueError("document_ids and bool input_mask must have matching [B,S] shapes")
    if document_ids.dtype != torch.int64:
        raise ValueError("document_ids must be int64")
    length = document_ids.shape[1]
    causal = torch.arange(length, device=document_ids.device)[:, None] >= torch.arange(length, device=document_ids.device)[None, :]
    same_document = document_ids[:, :, None] == document_ids[:, None, :]
    valid = input_mask[:, :, None] & input_mask[:, None, :]
    return (same_document & valid & causal).unsqueeze(1)
