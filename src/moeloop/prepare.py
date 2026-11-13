"""Explicit preparation of immutable token documents; never used during import/train.

``python -m moeloop.prepare --recipe wikitext2_gpt2 --output /data/moeloop-demo``
``python -m moeloop.prepare --input train.jsonl --validation valid.txt
       --tokenizer /data/gpt2-tokenizer --bos-token-id 50256
       --eos-token-id 50256 --output /data/my-corpus``
"""

from __future__ import annotations

import argparse
import array
import hashlib
import itertools
import json
import mmap
import os
import re
import shutil
import sqlite3
import struct
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .data import DOCUMENT_RULES, FORMAT, FORMAT_VERSION, PreparedManifest, file_sha256


WIKITEXT_ID = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
GPT2_ID = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
FINEWEB_EDU_ID = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
# One explicitly bounded file, not the sample-10BT glob or the complete dataset.
FINEWEB_EDU_FILES = ({
    "path": "sample/10BT/000_00000.parquet",
    "size": 2_152_819_114,
    "sha256": "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871",
},)
PUBLIC_LABEL_TOKENS = {"train": 300_000_000, "validation": 1_000_000, "test": 1_000_000}
DCLM_ID = "mlfoundations/dclm-baseline-1.0-parquet"
DCLM_REVISION = "817d6752765f6a41261085171dd546b104f60626"
DCLM_SOURCE_FORMAT = "moeloop-public-parquet-source-v1"
# Corpus headroom for a ~20.486B-label run: document/rank length imbalance must
# not force an early wrap on one rank. The training horizon remains separate.
P1_LABEL_TOKENS = {"train": 24_000_000_000, "validation": 10_000_000, "test": 10_000_000}
DCLM_TOKENIZER = {
    "id": "Qwen/Qwen3-30B-A3B", "revision": "ae659febe817e4b3ebd7355f47792725801204c9",
    "tokenizer_json_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "vocab_size": 151669, "eos_token_id": 151645,
}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "config.json",
)


def _write_atomic(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # link is atomic and, unlike replace(), refuses to overwrite an existing file.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _u32_bytes(tokens: Iterable[int]) -> bytes:
    values = array.array("I", tokens)
    if values.itemsize != 4:
        raise RuntimeError("this platform does not provide 32-bit unsigned int arrays")
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _normalized_tokenizer_assets(directory: Path, metadata: Mapping[str, Any]) -> dict[str, bytes]:
    from tokenizers import AddedToken, Tokenizer

    backend = Tokenizer.from_file(str(directory / "tokenizer.json"))
    roles = {}
    for role in ("bos", "eos", "pad"):
        token_id = metadata[f"{role}_token_id"]
        token = backend.id_to_token(token_id)
        if token is None:
            raise ValueError(f"{role}_token_id={token_id} is absent from tokenizer.json")
        roles[f"{role}_token"] = token
    original_vocab = backend.get_vocab()
    backend.add_special_tokens([AddedToken(token, normalized=False, special=True) for token in dict.fromkeys(roles.values())])
    if backend.get_vocab() != original_vocab:
        raise ValueError("special-token normalization must not change vocabulary IDs")
    backend.no_padding()
    backend.no_truncation()
    assets = {name: (directory / name).read_bytes() for name in TOKENIZER_FILES if (directory / name).is_file()}
    config = json.loads(assets.get("tokenizer_config.json", b"{}"))
    config.setdefault("tokenizer_class", "PreTrainedTokenizerFast")
    config.update(roles)
    for role in ("bos", "eos", "pad"):
        config[f"{role}_token_id"] = metadata[f"{role}_token_id"]
    decoder = config.setdefault("added_tokens_decoder", {})
    for role, token in roles.items():
        decoder[str(metadata[f"{role}_id"])] = {"content": token, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
    special = json.loads(assets.get("special_tokens_map.json", b"{}"))
    special.update(roles)
    assets["tokenizer.json"] = backend.to_str().encode()
    assets["tokenizer_config.json"] = (json.dumps(config, sort_keys=True, indent=2) + "\n").encode()
    assets["special_tokens_map.json"] = (json.dumps(special, sort_keys=True, indent=2) + "\n").encode()
    return assets


class _SplitWriter:
    """A single-pass shard sink; the public fixture routes documents to three sinks."""

    def __init__(self, output: Path, name: str, split: str, shard_tokens: int, *, identities: bool = False,
                 shards: list[dict[str, Any]] | None = None) -> None:
        self.output, self.name, self.split = output, name, split
        self.directory = output / name / split
        self.directory.mkdir(parents=True, exist_ok=shards is not None)
        self.shard_tokens, self.identities = shard_tokens, identities
        self.shards: list[dict[str, Any]] = list(shards or [])
        self.token_stream = self.index_stream = self.document_stream = None
        self.token_count = self.document_count = 0

    def finish(self) -> None:
        if self.token_stream is None:
            return
        for stream in (self.token_stream, self.index_stream, self.document_stream):
            if stream is None:
                continue
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        shard = {
            "tokens_path": self.token_path.relative_to(self.output).as_posix(),
            "tokens_sha256": file_sha256(self.token_path),
            "index_path": self.index_path.relative_to(self.output).as_posix(),
            "index_sha256": file_sha256(self.index_path),
            "num_documents": self.document_count,
            "num_tokens": self.token_count,
        }
        if self.identities:
            shard.update(documents_path=self.document_path.relative_to(self.output).as_posix(),
                         documents_sha256=file_sha256(self.document_path))
        self.shards.append(shard)
        self.token_stream = self.index_stream = self.document_stream = None

    def append(self, tokens: list[int] | array.array, identity: Mapping[str, Any] | None = None) -> None:
        if self.token_stream is not None and self.token_count + len(tokens) > self.shard_tokens:
            self.finish()
        if self.token_stream is None:
            stem = self.directory / f"{len(self.shards):06d}"
            self.token_path = stem.with_suffix(".tokens.bin")
            self.index_path = stem.with_suffix(".index.bin")
            self.token_stream, self.index_stream = self.token_path.open("xb"), self.index_path.open("xb")
            if self.identities:
                self.document_path = stem.with_suffix(".documents.jsonl")
                self.document_stream = self.document_path.open("xb")
            self.token_count = self.document_count = 0
            self.index_stream.write(struct.pack("<Q", 0))
        self.token_stream.write(_u32_bytes(tokens))
        self.token_count += len(tokens)
        self.document_count += 1
        self.index_stream.write(struct.pack("<Q", self.token_count))
        if self.document_stream is not None:
            self.document_stream.write((json.dumps(dict(identity or {}), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode())

    def close(self) -> None:
        for stream in (self.token_stream, self.index_stream, self.document_stream):
            if stream is not None:
                stream.close()


def _frame_document(ids: Iterable[int], metadata: Mapping[str, Any]) -> list[int]:
    """BOS, the tokenizer's IDs, EOS (DOCUMENT_RULES), after checking every ID fits the vocabulary."""
    encoded = list(ids)
    if any(not isinstance(token, int) or not 0 <= token < metadata["vocab_size"] for token in encoded):
        raise ValueError("tokenizer emitted an ID outside vocab_size")
    return [metadata["bos_token_id"], *encoded, metadata["eos_token_id"]]


# Public subsets split by content hash: bucket 0 is test, 1 validation, 2..99 train.
SPLIT_BUCKETS = {"test": [0], "validation": [1], "train": [2, 99]}


def split_for(digest: bytes) -> str:
    """The split of a public-subset document: first eight digest bytes, big endian, modulo 100 (SPLIT_BUCKETS)."""
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def _seen_tables(connection: sqlite3.Connection) -> sqlite3.Connection:
    """Create the deduplication state of ``_fill_splits``: every taken original ID and content digest."""
    connection.execute("CREATE TABLE seen_ids (key TEXT PRIMARY KEY) WITHOUT ROWID")
    connection.execute("CREATE TABLE seen_content (key BLOB PRIMARY KEY) WITHOUT ROWID")
    return connection


def _fill_splits(documents: Iterable[tuple[str, bytes, Any, Mapping[str, Any]]], seen: sqlite3.Connection,
                 writers: Mapping[str, _SplitWriter], selection: dict[str, Any], to_tokens: Callable[[Any], Any]) -> None:
    """Take whole documents into their splits, in source order, until every split meets its label target.

    ``documents`` yields ``(original_id, digest, payload, identity)`` per nonempty
    document; the caller counts scanned and empty ones. The first original ID and
    the first content digest win globally, before ``split_for(digest)`` picks the
    split; ``seen`` (``_seen_tables``) holds both. A split that already met its
    target skips the document without building its tokens; ``to_tokens(payload)``
    builds the stored tokens of the others. Counts accumulate in ``selection``, so
    a resumed preparation continues them. Returns without reading further once
    every target is met; a source that runs out first leaves some target unmet.
    """
    targets, counts = selection["target_label_tokens"], selection["actual_label_tokens"]
    next_progress = (counts["train"] // 10_000_000 + 1) * 10_000_000
    for original_id, digest, payload, identity in documents:
        if (seen.execute("SELECT 1 FROM seen_ids WHERE key=?", (original_id,)).fetchone()
                or seen.execute("SELECT 1 FROM seen_content WHERE key=?", (digest,)).fetchone()):
            selection["duplicate_documents"] += 1
            continue
        seen.execute("INSERT INTO seen_ids VALUES (?)", (original_id,))
        seen.execute("INSERT INTO seen_content VALUES (?)", (digest,))
        split = split_for(digest)
        if counts[split] >= targets[split]:
            continue
        tokens = to_tokens(payload)
        writers[split].append(tokens, identity)
        counts[split] += len(tokens) - 1
        if counts["train"] >= next_progress:
            print(f"prepared labels: {json.dumps(counts, sort_keys=True)}", file=sys.stderr, flush=True)
            next_progress += 10_000_000
        if all(counts[name] >= targets[name] for name in targets):
            return


def _write_split(output: Path, name: str, split: str, texts: Iterable[str], encode: Callable[[str], list[int]],
                 metadata: Mapping[str, Any], shard_tokens: int) -> list[dict[str, Any]]:
    writer = _SplitWriter(output, name, split, shard_tokens)
    try:
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("document text must be a string")
            if not text.strip():
                continue
            writer.append(_frame_document(encode(text), metadata))
        writer.finish()
    finally:
        writer.close()
    if not writer.shards:
        raise ValueError(f"no nonempty documents for {name}/{split}")
    return writer.shards


def prepare_dataset(
    output: str | Path,
    sources: Mapping[str, Mapping[str, Iterable[str]]],
    *,
    encode: Callable[[str], list[int]],
    tokenizer: Mapping[str, Any],
    tokenizer_directory: str | Path | None = None,
    shard_tokens: int = 1_000_000,
    provenance: Mapping[str, Any] | None = None,
) -> PreparedManifest:
    """Prepare text iterables once, refusing overwrite; a manifest marks completion.

    ``sources`` is ``{source_name: {split_name: texts}}``. The caller supplies
    explicit tokenizer identity and special IDs. Production token IDs remain in
    mmap-friendly shards; text and complete token corpora are never collected.
    ``encode`` must not insert special tokens or truncate documents.
    """
    if shard_tokens <= 0 or not sources:
        raise ValueError("require positive shard_tokens and at least one source")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name == "manifest.json" for name in sources):
        raise ValueError("source names must use letters, digits, '.', '_' or '-'")
    for splits in sources.values():
        if not splits or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", split) for split in splits):
            raise ValueError("source needs valid split names")

    def write_sources(root, metadata, encode):
        return [{"name": name, "splits": {
            split: _write_split(root, name, split, texts, encode, metadata, shard_tokens)
            for split, texts in splits.items()
        }} for name, splits in sources.items()]

    return _prepare_dataset(output, encode=encode, tokenizer=tokenizer, tokenizer_directory=tokenizer_directory,
                            provenance=provenance, write_sources=write_sources)


def _prepare_dataset(output, *, encode, tokenizer, tokenizer_directory, provenance, write_sources) -> PreparedManifest:
    output = Path(output).resolve()
    metadata = dict(tokenizer)
    for key in ("id", "revision", "vocab_size", "bos_token_id", "eos_token_id"):
        if key not in metadata:
            raise ValueError(f"tokenizer metadata is missing {key}")
    metadata.setdefault("pad_token_id", metadata["eos_token_id"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        metadata["files"] = {}
        if tokenizer_directory is not None:
            directory = Path(tokenizer_directory).resolve()
            if not (directory / "tokenizer.json").is_file():
                raise ValueError("tokenizer_directory must contain tokenizer.json")
            (output / "tokenizer").mkdir()
            metadata["source_files"] = {name: file_sha256(directory / name) for name in TOKENIZER_FILES if (directory / name).is_file()}
            for filename, content in _normalized_tokenizer_assets(directory, metadata).items():
                target = output / "tokenizer" / filename
                _write_atomic(target, content)
                metadata["files"][target.relative_to(output).as_posix()] = file_sha256(target)
            metadata["path"] = "tokenizer"
            normalized_backend = _load_tokenizer(output / "tokenizer")
            encode = lambda text: normalized_backend.encode(text, add_special_tokens=False).ids
        prepared_sources = write_sources(output, metadata, encode)
        payload = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "tokenizer": metadata,
            "document_rules": DOCUMENT_RULES,
            "provenance": dict(provenance or {}),
            "sources": prepared_sources,
        }
        _write_atomic(output / "manifest.json", (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode())
        return PreparedManifest.load(output)
    except BaseException:
        # We exclusively created this directory; never remove a pre-existing dataset.
        shutil.rmtree(output)
        raise


def _dclm_source(source_manifest: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    source = dict(source_manifest) if isinstance(source_manifest, Mapping) else json.loads(Path(source_manifest).read_text())
    if source.get("format") != DCLM_SOURCE_FORMAT or source.get("id") != DCLM_ID:
        raise ValueError(f"DCLM source must use {DCLM_SOURCE_FORMAT} and {DCLM_ID}")
    if not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
        raise ValueError("DCLM source revision must be an immutable 40-character commit")
    files = source.get("files", [])
    names = set()
    if not files:
        raise ValueError("DCLM source needs an explicit ordered Parquet file list")
    for item in files:
        name = item.get("path", "")
        path = Path(name)
        if (not name or path.is_absolute() or ".." in path.parts or path.suffix != ".parquet"
                or name in names or type(item.get("size")) is not int or item["size"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))):
            raise ValueError(f"invalid or duplicate DCLM source file: {name}")
        names.add(name)
    return source


def freeze_dclm_source(output: str | Path, *, num_shards: int = 1024,
                       revision: str = DCLM_REVISION, shard_paths: Iterable[str] | None = None) -> dict[str, Any]:
    """Freeze public HF metadata only; never download Parquet or tokenizer bytes.

    By default select the lexicographically first ``num_shards`` paths. Explicit
    ``shard_paths`` preserve the caller's order. The completed source JSON, not
    a live glob or repository HEAD, is the input to preparation.
    """
    from huggingface_hub import HfApi

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or num_shards <= 0:
        raise ValueError("require an immutable DCLM commit and positive num_shards")
    api = HfApi()
    paths = (list(shard_paths) if shard_paths is not None else sorted(
        name for name in api.list_repo_files(DCLM_ID, repo_type="dataset", revision=revision)
        if name.endswith(".parquet"))[:num_shards])
    if not paths or (shard_paths is None and len(paths) != num_shards):
        raise ValueError("insufficient public DCLM files for num_shards")
    files = {}
    for begin in range(0, len(paths), 100):
        for item in api.get_paths_info(DCLM_ID, paths[begin:begin + 100], repo_type="dataset", revision=revision):
            lfs = getattr(item, "lfs", None)
            if lfs is None:
                raise ValueError(f"public source lacks an LFS content SHA256: {item.path}")
            files[item.path] = {"path": item.path, "size": item.size, "sha256": lfs.sha256}
    source = _dclm_source({"format": DCLM_SOURCE_FORMAT, "id": DCLM_ID, "revision": revision,
        "files": [files[name] for name in paths],
        "card_url": f"https://huggingface.co/datasets/{DCLM_ID}/blob/{revision}/README.md",
        "license": {"id": "CC-BY-4.0", "basis": "upstream_dataset_card"}})
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(output, (json.dumps(source, sort_keys=True, indent=2) + "\n").encode())
    return source


def _dclm_records(path: Path, source_file: str) -> Iterable[Mapping[str, Any]]:
    import pyarrow.parquet as parquet

    with parquet.ParquetFile(path) as reader:
        row = 0
        for batch in reader.iter_batches(batch_size=256, columns=["id", "text"]):
            for record in batch.to_pylist():
                yield {"id": record["id"], "text": record["text"], "source_file": source_file, "source_row": row}
                row += 1


def prepare_dclm_public(
    output: str | Path, *, source_manifest: str | Path | Mapping[str, Any],
    tokenizer_directory: str | Path | None = None, source_directory: str | Path | None = None,
    cache_dir: str | Path | None = None, label_tokens: Mapping[str, int] | None = None,
    shard_tokens: int = 128_000_000, resume: bool = False,
) -> PreparedManifest:
    """Prepare pinned, ordered public DCLM Parquet shards on CPU, once.

    SQLite holds exact global ID/content deduplication on disk with a 64 MiB
    page cache; bounded 256-document batches use the tokenizer's Rust thread pool.
    Each source file commits its dedup state, counts and fsynced output shards
    together. ``resume=True`` verifies completed output, removes uncommitted
    shards, and replays at most the interrupted source file. The final manifest
    has the existing v1 format and is published only after every budget is met.
    An insufficient source is an error, never a license to cycle documents.
    """
    import fcntl
    from huggingface_hub import hf_hub_download, snapshot_download

    output = Path(output).resolve()
    source = _dclm_source(source_manifest)
    targets = dict(P1_LABEL_TOKENS if label_tokens is None else label_tokens)
    if (set(targets) != {"train", "validation", "test"}
            or any(type(value) is not int or value <= 0 for value in targets.values()) or shard_tokens <= 0):
        raise ValueError("require positive train, validation, test label targets and shard_tokens")
    if output.exists() and (not resume or (output / "manifest.json").exists()):
        raise FileExistsError(output)
    if resume and not (output / ".prepare.sqlite3").is_file():
        raise ValueError("resume requires an existing DCLM preparation checkpoint")
    expected = DCLM_TOKENIZER
    directory = Path(tokenizer_directory) if tokenizer_directory is not None else Path(snapshot_download(
        expected["id"], revision=expected["revision"], allow_patterns=list(TOKENIZER_FILES),
        cache_dir=str(cache_dir) if cache_dir is not None else None))
    if file_sha256(directory / "tokenizer.json") != expected["tokenizer_json_sha256"]:
        raise ValueError("tokenizer.json SHA256 differs from the fixed Qwen3 tokenization identity")
    eos = expected["eos_token_id"]
    metadata = {"id": expected["id"], "revision": expected["revision"], "vocab_size": expected["vocab_size"],
        "bos_token_id": eos, "eos_token_id": eos, "pad_token_id": eos,
        "source_bos_token_id": None, "role_adaptation": "source_eos_id_as_explicit_bos_and_eos",
        "source_files": {name: file_sha256(directory / name) for name in TOKENIZER_FILES if (directory / name).is_file()},
        "path": "tokenizer"}
    assets = _normalized_tokenizer_assets(directory, metadata)
    metadata["files"] = {f"tokenizer/{name}": hashlib.sha256(content).hexdigest() for name, content in assets.items()}
    contract = {"source": source, "tokenizer": metadata, "label_tokens": targets, "shard_tokens": shard_tokens,
                "selection_algorithm": "sha256-utf8-first8-big-endian-mod100-v1"}
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".prepare.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another process owns this DCLM preparation") from error
        connection = sqlite3.connect(output / ".prepare.sqlite3")
        writers = {}
        try:
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA synchronous=FULL")
            if resume:
                row = connection.execute("SELECT contract, state FROM checkpoint").fetchone()
                if row is None or json.loads(row[0]) != contract:
                    raise ValueError("DCLM preparation identity changed; source, tokenizer, budgets and sharding must match")
                state = json.loads(row[1])
            else:
                state = {"next_file": 0, "shards": {split: [] for split in targets}, "selection": {
                    "algorithm": contract["selection_algorithm"],
                    "buckets": SPLIT_BUCKETS,
                    "deduplication": "first_original_id_and_first_exact_utf8_content_sha256_global",
                    "stop": "minimum_shifted_labels_per_split_at_whole_document_boundary",
                    "target_label_tokens": targets, "actual_label_tokens": dict.fromkeys(targets, 0),
                    "scanned_documents": 0, "duplicate_documents": 0, "empty_documents": 0}}
                connection.execute("CREATE TABLE checkpoint (contract TEXT NOT NULL, state TEXT NOT NULL)")
                _seen_tables(connection)
                connection.execute("INSERT INTO checkpoint VALUES (?, ?)", (json.dumps(contract, sort_keys=True), json.dumps(state)))
                connection.commit()
            (output / "tokenizer").mkdir(exist_ok=True)
            for name, content in assets.items():
                target = output / "tokenizer" / name
                if target.exists():
                    if file_sha256(target) != metadata["files"][f"tokenizer/{name}"]:
                        raise ValueError(f"prepared tokenizer SHA256 mismatch: {name}")
                else:
                    _write_atomic(target, content)
            backend = _load_tokenizer(output / "tokenizer")
            if backend.get_vocab_size() != metadata["vocab_size"]:
                raise ValueError("tokenizer vocabulary size differs from the fixed identity")
            for split in targets:
                retained = set()
                for shard in state["shards"][split]:
                    for kind in ("tokens", "index", "documents"):
                        path = output / shard[f"{kind}_path"]
                        if file_sha256(path) != shard[f"{kind}_sha256"]:
                            raise ValueError(f"committed output SHA256 mismatch: {path}")
                        retained.add(path.name)
                split_directory = output / "public" / split
                if split_directory.exists():
                    for path in split_directory.iterdir():
                        if path.name not in retained:
                            path.unlink()
                writers[split] = _SplitWriter(output, "public", split, shard_tokens, identities=True,
                                              shards=state["shards"][split])
            selection = state["selection"]
            counts = selection["actual_label_tokens"]
            complete = lambda: all(counts[split] >= targets[split] for split in targets)

            def nonempty(records, source_file):
                while True:
                    documents = list(itertools.islice(records, 256))
                    if not documents:
                        return
                    # encode_batch uses tokenizers/Rayon's bounded Rust pool;
                    # its result order is stable regardless of worker count.
                    encoded = backend.encode_batch([row["text"] if isinstance(row["text"], str) else ""
                                                    for row in documents], add_special_tokens=False)
                    for document, encoding in zip(documents, encoded, strict=True):
                        selection["scanned_documents"] += 1
                        text, original_id = document["text"], document["id"]
                        if not isinstance(text, str) or not isinstance(original_id, str) or not original_id:
                            raise ValueError(f"DCLM requires original string id/text: {source_file}:{document['source_row']}")
                        if not text.strip():
                            selection["empty_documents"] += 1
                            continue
                        digest = hashlib.sha256(text.encode()).digest()
                        yield original_id, digest, encoding, {"id": original_id, "content_sha256": digest.hex(),
                            "source_file": source_file, "source_row": document["source_row"]}

            for file_index in range(state["next_file"], len(source["files"])):
                if complete():
                    break
                item = source["files"][file_index]
                path = (Path(source_directory) / item["path"] if source_directory is not None else Path(hf_hub_download(
                    DCLM_ID, item["path"], repo_type="dataset", revision=source["revision"],
                    cache_dir=str(cache_dir) if cache_dir is not None else None)))
                if path.stat().st_size != item["size"] or file_sha256(path) != item["sha256"]:
                    raise ValueError(f"DCLM source size/SHA256 mismatch: {item['path']}")
                records = _dclm_records(path, item["path"])
                try:
                    _fill_splits(nonempty(records, item["path"]), connection, writers, selection,
                                 lambda encoding: _frame_document(encoding.ids, metadata))
                finally:
                    records.close()
                for writer in writers.values():
                    writer.finish()
                    descriptor = os.open(writer.directory, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                state["next_file"] = file_index + 1
                state["shards"] = {split: writer.shards for split, writer in writers.items()}
                connection.execute("UPDATE checkpoint SET state=?", (json.dumps(state),))
                connection.commit()
                print(f"committed DCLM source {file_index + 1}/{len(source['files'])}; labels: {json.dumps(counts, sort_keys=True)}",
                      file=sys.stderr, flush=True)
            missing = {split: {"required": targets[split], "actual": counts[split]} for split in targets if counts[split] < targets[split]}
            if missing:
                raise ValueError(f"insufficient frozen public DCLM source for label targets: {missing}; freeze a larger source list for a new preparation")
            payload = {"format": FORMAT, "version": FORMAT_VERSION, "tokenizer": metadata,
                "document_rules": DOCUMENT_RULES,
                "provenance": {"kind": "public_document_subset", "dataset": source, "selection": selection,
                    "document_definition": "one_nonempty_original_record_no_text_normalization"},
                "sources": [{"name": "public", "splits": state["shards"]}]}
            _write_atomic(output / "manifest.json", (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode())
            # Every output hash was computed over fsynced bytes before its
            # checkpoint commit. Avoid rereading ~96 GB a second time here.
            manifest = PreparedManifest.load(output, verify_hashes=False)
        finally:
            connection.close()
            for writer in writers.values():
                writer.close()
        (output / ".prepare.sqlite3").unlink()
        return manifest


def prepare_dclm_packed(
    output: str | Path, *, source_manifest: str | Path, tokenizer_directory: str | Path,
    label_tokens: Mapping[str, int] | None = None, shard_tokens: int = 1_000_000,
    chunk_names: Iterable[str] | None = None,
) -> PreparedManifest:
    """Import ordered DCLM packed chunks as whole documents, without reencoding.

    Source documents already end in EOS and have no BOS. The common format
    prepends an explicit BOS using the existing EOS ID; token content is otherwise
    unchanged. Token SHA256 drives deduplication and split isolation. The old
    UTF-8 blake2b64 remains a separate, weaker source identity.

    Serves archived data only: it reproduces manifests imported from the legacy
    ``loopmoe-packed-v1`` chunks. New corpora use ``prepare_dclm_public``.
    """
    if Path(output).exists():
        raise FileExistsError(output)
    targets = dict(PUBLIC_LABEL_TOKENS if label_tokens is None else label_tokens)
    if set(targets) != {"train", "validation", "test"} or any(type(value) is not int or value <= 0 for value in targets.values()):
        raise ValueError("label_tokens needs positive train, validation and test targets")
    if shard_tokens <= 0:
        raise ValueError("shard_tokens must be positive")
    source_manifest = Path(source_manifest).resolve()
    if source_manifest.is_dir():
        source_manifest = source_manifest / "manifest.json"
    parent = json.loads(source_manifest.read_text())
    if parent.get("format") != "loopmoe-packed-v1" or parent.get("dtype") != "uint32":
        raise ValueError("source must use loopmoe-packed-v1 uint32 token documents")
    source_tokenizer = parent["tokenizer"]
    expected = DCLM_TOKENIZER
    if (source_tokenizer.get("tokenizer_json_sha256") != expected["tokenizer_json_sha256"]
            or source_tokenizer.get("eos_id") != expected["eos_token_id"]
            or source_tokenizer.get("vocab_size") != expected["vocab_size"]
            or source_tokenizer.get("bos_id") is not None):
        raise ValueError("source tokenizer differs from the fixed Qwen3 tokenization identity")
    tokenizer_directory = Path(tokenizer_directory).resolve()
    if file_sha256(tokenizer_directory / "tokenizer.json") != expected["tokenizer_json_sha256"]:
        raise ValueError("tokenizer.json SHA256 differs from the packed source")
    available = parent["splits"]["train"]["chunks"]
    if chunk_names is None:
        chunks = available[:2]
        if len(chunks) != 2 or [chunk["name"] for chunk in chunks] != ["train-00000", "train-00001"]:
            raise ValueError("legacy DCLM import requires the fixed first two train chunks")
    else:
        names = list(chunk_names)
        known = {chunk["name"]: chunk for chunk in available}
        if not names or len(names) != len(set(names)) or any(name not in known for name in names):
            raise ValueError("packed chunk names must be nonempty, unique and present in the source manifest")
        chunks = [known[name] for name in names]
    references = []
    for chunk in chunks:
        paths = {kind: source_manifest.parent / f"{chunk['name']}.{kind}.bin" for kind in ("tokens", "docs", "dochash")}
        sizes = {"tokens": chunk["tokens"] * 4, "docs": chunk["docs"] * 8, "dochash": chunk["docs"] * 8}
        for kind, path in paths.items():
            if not path.is_file() or path.stat().st_size != sizes[kind] or file_sha256(path) != chunk[f"{kind}_sha256"]:
                raise ValueError(f"packed source size/SHA256 mismatch: {chunk['name']}.{kind}.bin")
        ends = array.array("q")
        ends.frombytes(paths["docs"].read_bytes())
        if sys.byteorder != "little":
            ends.byteswap()
        if not ends or ends[0] <= 0 or ends[-1] != chunk["tokens"] or any(right <= left for left, right in zip(ends, ends[1:])):
            raise ValueError(f"invalid packed document offsets/boundaries: {chunk['name']}")
        references.append((chunk, paths["tokens"], ends, paths["dochash"].read_bytes()))
        print(f"checked packed source {chunk['name']}: {chunk['tokens']} tokens, {chunk['docs']} documents", flush=True)
    frozen_chunks = [{key: chunk[key] for key in ("name", "tokens", "docs", "tokens_sha256", "docs_sha256", "dochash_sha256")} for chunk in chunks]
    source_identity = {"manifest_sha256": file_sha256(source_manifest), "chunks": frozen_chunks,
                       "tokenizer_json_sha256": expected["tokenizer_json_sha256"]}
    revision = hashlib.sha256(json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    dataset = {"id": "mlfoundations/dclm-baseline-1.0-parquet", "local_parent": "dclm-oebm-v1",
        "revision": f"sha256:{revision}", "upstream_revision": None,
        "source_family": "https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0",
        "license": {"id": "CC-BY-4.0", "basis": "upstream_family_dataset_card",
                    "url": "https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0"},
        **source_identity}
    selection = {"algorithm": "sha256-u32le-token-document-first8-big-endian-mod100-v1",
        "buckets": SPLIT_BUCKETS,
        "deduplication": "first_exact_source_token_sha256_global",
        "stop": "minimum_shifted_labels_per_split_at_whole_document_boundary",
        "target_label_tokens": targets, "actual_label_tokens": dict.fromkeys(targets, 0),
        "scanned_documents": 0, "duplicate_documents": 0, "empty_documents": 0}
    provenance = {"kind": "public_document_subset", "dataset": dataset, "selection": selection,
        "document_definition": "whole_parent_token_document_including_source_eos_no_decode_reencode",
        "document_identity_scope": "local_parent_chunk_and_ordinal_not_original_hf_id"}
    eos = expected["eos_token_id"]
    tokenizer = {"id": expected["id"], "revision": expected["revision"], "vocab_size": expected["vocab_size"],
        "bos_token_id": eos, "eos_token_id": eos, "pad_token_id": eos,
        "source_bos_token_id": None, "role_adaptation": "prepend_source_eos_id_as_bos_keep_existing_terminal_eos"}

    def write_sources(root, metadata, encode):
        writers = {split: _SplitWriter(root, "public", split, shard_tokens, identities=True) for split in targets}
        seen = _seen_tables(sqlite3.connect(":memory:"))

        def nonempty():
            for chunk, path, ends, text_hashes in references:
                with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as stored:
                    start = 0
                    for index, end in enumerate(ends):
                        raw = stored[4 * start:4 * end]
                        start = end
                        selection["scanned_documents"] += 1
                        if struct.unpack_from("<I", raw, len(raw) - 4)[0] != eos:
                            raise ValueError(f"packed document lacks terminal EOS: {chunk['name']}:{index}")
                        if len(raw) <= 4:
                            selection["empty_documents"] += 1
                            continue
                        digest = hashlib.sha256(raw).digest()
                        # Positional IDs never repeat, so only the token SHA256 deduplicates.
                        identity = {"id": f"{chunk['name']}:{index}", "source_chunk": chunk["name"],
                            "source_document": index, "source_tokens_sha256": chunk["tokens_sha256"],
                            "source_token_count": len(raw) // 4, "token_sha256": digest.hex(),
                            "source_text_blake2b64": text_hashes[8 * index:8 * (index + 1)].hex()}
                        yield identity["id"], digest, raw, identity

        def to_tokens(raw):
            tokens = array.array("I")
            tokens.frombytes(raw)
            if sys.byteorder != "little":
                tokens.byteswap()
            if max(tokens) >= metadata["vocab_size"]:
                raise ValueError("packed source token exceeds tokenizer vocabulary")
            tokens.insert(0, metadata["bos_token_id"])
            return tokens

        try:
            _fill_splits(nonempty(), seen, writers, selection, to_tokens)
            counts = selection["actual_label_tokens"]
            missing = {split: {"required": target, "actual": counts[split]} for split, target in targets.items() if counts[split] < target}
            if missing:
                raise ValueError(f"insufficient fixed packed source for label targets: {missing}")
            for writer in writers.values():
                writer.finish()
            return [{"name": "public", "splits": {split: writer.shards for split, writer in writers.items()}}]
        finally:
            seen.close()
            for writer in writers.values():
                writer.close()

    return _prepare_dataset(output, encode=None, tokenizer=tokenizer, tokenizer_directory=tokenizer_directory,
                            provenance=provenance, write_sources=write_sources)


def prepare_document_subset(
    output: str | Path,
    documents: Iterable[Mapping[str, Any]],
    *,
    encode: Callable[[str], list[int]],
    tokenizer: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    label_tokens: Mapping[str, int] | None = None,
    tokenizer_directory: str | Path | None = None,
    shard_tokens: int = 1_000_000,
) -> PreparedManifest:
    """Freeze one bounded, deduplicated train/validation/test document subset.

    Records must have original ``id`` and exact ``text``. First original ID and
    first UTF-8 content SHA256 win globally, before ``split_for`` picks the split
    from the content hash. Each split takes whole original documents until its
    minimum number of shifted labels is reached. No epoch, RNG or local cache path
    changes the subset identity. Whitespace-only records are skipped before the ID
    check and counted only as scanned; ``prepare_dclm_public`` checks the ID first
    and counts them as empty. Each keeps its order so existing manifests stay
    byte-identical.
    """
    targets = dict(PUBLIC_LABEL_TOKENS if label_tokens is None else label_tokens)
    if set(targets) != {"train", "validation", "test"} or any(type(value) is not int or value <= 0 for value in targets.values()):
        raise ValueError("label_tokens needs positive train, validation and test targets")
    if shard_tokens <= 0:
        raise ValueError("shard_tokens must be positive")
    selection = {
        "algorithm": "sha256-utf8-first8-big-endian-mod100-v1",
        "buckets": SPLIT_BUCKETS,
        "deduplication": "first_original_id_and_first_exact_utf8_content_sha256_global",
        "stop": "minimum_shifted_labels_per_split_at_whole_document_boundary",
        "target_label_tokens": targets,
        "actual_label_tokens": dict.fromkeys(targets, 0),
        "scanned_documents": 0,
        "duplicate_documents": 0,
    }
    provenance = {"kind": "public_document_subset", "dataset": dict(source_manifest), "selection": selection,
                  "document_definition": "one_nonempty_original_record_no_text_normalization"}

    def write_sources(root, metadata, encode):
        writers = {split: _SplitWriter(root, "public", split, shard_tokens, identities=True) for split in targets}
        seen = _seen_tables(sqlite3.connect(":memory:"))
        iterator = iter(documents)

        def nonempty():
            for row, document in enumerate(iterator):
                selection["scanned_documents"] += 1
                text, original_id = document.get("text"), document.get("id")
                if not isinstance(text, str):
                    raise ValueError(f"public document text must be a string at source row {row}")
                if not text.strip():
                    continue
                if not isinstance(original_id, str) or not original_id:
                    raise ValueError(f"public document needs its original nonempty string id at source row {row}")
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                identity = {"id": original_id, "content_sha256": digest.hex(), "source_row": row}
                if "source_file" in document:
                    identity["source_file"] = document["source_file"]
                yield original_id, digest, text, identity

        try:
            _fill_splits(nonempty(), seen, writers, selection, lambda text: _frame_document(encode(text), metadata))
            counts = selection["actual_label_tokens"]
            missing = {split: {"required": target, "actual": counts[split]} for split, target in targets.items() if counts[split] < target}
            if missing:
                raise ValueError(f"insufficient fixed public source for label targets: {missing}")
            for writer in writers.values():
                writer.finish()
            return [{"name": "public", "splits": {split: writer.shards for split, writer in writers.items()}}]
        finally:
            seen.close()
            for writer in writers.values():
                writer.close()
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    return _prepare_dataset(output, encode=encode, tokenizer=tokenizer, tokenizer_directory=tokenizer_directory,
                            provenance=provenance, write_sources=write_sources)


def read_texts(path: str | Path, *, text_field: str = "text") -> Iterable[str]:
    """Each line is a document; JSONL uses an explicit text field."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if path.suffix.lower() == ".jsonl":
                try:
                    text = json.loads(line)[text_field]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(f"invalid JSONL record at {path}:{number}") from error
                if not isinstance(text, str):
                    raise ValueError(f"non-string {text_field} at {path}:{number}")
                yield text
            else:
                yield line.rstrip("\r\n")


def _load_tokenizer(path: Path) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("text preparation needs tokenizers; install moeloop-lab[data]") from error
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    # A saved tokenizer can contain truncation/padding settings. Packing owns both.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def _fineweb_records(*, source_file: str | Path | None = None,
                     cache_dir: str | Path | None = None) -> Iterable[Mapping[str, Any]]:
    """Read only the frozen file list in bounded batches from a reusable cache."""
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    for source in FINEWEB_EDU_FILES:
        path = Path(source_file) if source_file is not None else Path(hf_hub_download(
            FINEWEB_EDU_ID, source["path"], repo_type="dataset", revision=FINEWEB_EDU_REVISION,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        ))
        if path.stat().st_size != source["size"] or file_sha256(path) != source["sha256"]:
            raise ValueError(f"fixed FineWeb-Edu source size/SHA256 mismatch: {path}")
        with parquet.ParquetFile(path) as reader:
            for batch in reader.iter_batches(batch_size=1024, columns=["id", "text"]):
                for record in batch.to_pylist():
                    yield {"id": record["id"], "text": record["text"], "source_file": source["path"]}


def prepare_fineweb_edu(output: str | Path, *, source_file: str | Path | None = None,
                        cache_dir: str | Path | None = None, label_tokens: Mapping[str, int] | None = None,
                        shard_tokens: int = 1_000_000) -> PreparedManifest:
    """Prepare the versioned public fixture once, entirely on CPU.

    The default downloads and verifies one 2.15 GB Parquet file into the HF
    cache. ``source_file`` reuses that exact file from any local location. Only
    the necessary prefix is read in bounded Parquet batches. Local/cache paths
    do not enter the manifest or change the selected original rows.
    """
    if Path(output).exists():
        raise FileExistsError(f"output already exists: {output}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("the public fixture needs tokenizers, datasets and pyarrow; install moeloop-lab[data]") from error
    print(f"dataset: {FINEWEB_EDU_ID}@{FINEWEB_EDU_REVISION}; fixed files: {[item['path'] for item in FINEWEB_EDU_FILES]}", flush=True)
    print(f"tokenizer: {GPT2_ID}@{GPT2_REVISION}; no model weights; target labels: {dict(label_tokens or PUBLIC_LABEL_TOKENS)}", flush=True)
    tokenizer_directory = Path(snapshot_download(GPT2_ID, revision=GPT2_REVISION, allow_patterns=list(TOKENIZER_FILES),
                                                 cache_dir=str(cache_dir) if cache_dir is not None else None))
    tokenizer = _load_tokenizer(tokenizer_directory)
    return prepare_document_subset(
        output, _fineweb_records(source_file=source_file, cache_dir=cache_dir),
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        tokenizer={"id": GPT2_ID, "revision": GPT2_REVISION, "license": "mit", "vocab_size": tokenizer.get_vocab_size(),
                   "bos_token_id": 50256, "eos_token_id": 50256, "pad_token_id": 50256},
        tokenizer_directory=tokenizer_directory, label_tokens=label_tokens, shard_tokens=shard_tokens,
        source_manifest={
            "id": FINEWEB_EDU_ID, "revision": FINEWEB_EDU_REVISION, "config": "sample-10BT", "split": "train",
            "files": list(FINEWEB_EDU_FILES),
            "card_url": f"https://huggingface.co/datasets/{FINEWEB_EDU_ID}/blob/{FINEWEB_EDU_REVISION}/README.md",
            "license": {"id": "ODC-By-1.0", "url": "https://opendatacommons.org/licenses/by/1-0/",
                        "additional_terms_url": "https://commoncrawl.org/terms-of-use"},
        },
    )


def prepare_wikitext2(output: str | Path, *, shard_tokens: int = 1_000_000) -> PreparedManifest:
    """Download only the pinned public data/tokenizer, then use the local writer."""
    if Path(output).exists():
        raise FileExistsError(f"output already exists: {output}")
    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("the WikiText example needs datasets and tokenizers; install moeloop-lab[data]") from error
    print(f"dataset: {WIKITEXT_ID}@{WIKITEXT_REVISION} (wikitext-2-raw-v1)", flush=True)
    print(f"tokenizer: {GPT2_ID}@{GPT2_REVISION}; no model weights", flush=True)
    tokenizer_directory = Path(snapshot_download(GPT2_ID, revision=GPT2_REVISION, allow_patterns=list(TOKENIZER_FILES)))
    tokenizer = _load_tokenizer(tokenizer_directory)
    dataset = load_dataset(WIKITEXT_ID, "wikitext-2-raw-v1", revision=WIKITEXT_REVISION)
    return prepare_dataset(
        output,
        {"wikitext": {split: (record["text"] for record in dataset[split]) for split in ("train", "validation", "test")}},
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        tokenizer={"id": GPT2_ID, "revision": GPT2_REVISION, "vocab_size": tokenizer.get_vocab_size(), "bos_token_id": 50256, "eos_token_id": 50256, "pad_token_id": 50256},
        tokenizer_directory=tokenizer_directory,
        shard_tokens=shard_tokens,
        provenance={"dataset_id": WIKITEXT_ID, "dataset_revision": WIKITEXT_REVISION, "dataset_config": "wikitext-2-raw-v1", "document_definition": "each_nonempty_original_row"},
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing paths are refused")
    parser.add_argument("--recipe", choices=("wikitext2_gpt2", "fineweb_edu_gpt2"))
    parser.add_argument("--input", type=Path, help="local training text or JSONL, one document per line")
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--source-name", default="local")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer", type=Path, help="local directory containing tokenizer.json")
    parser.add_argument("--tokenizer-id", default="local")
    parser.add_argument("--tokenizer-revision", default="local-files-sha256")
    parser.add_argument("--bos-token-id", type=int)
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--shard-tokens", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.recipe:
        if any(value is not None for value in (args.input, args.validation, args.test, args.tokenizer, args.bos_token_id, args.eos_token_id)):
            parser.error("--recipe cannot be combined with local data/tokenizer inputs")
        recipe = prepare_wikitext2 if args.recipe == "wikitext2_gpt2" else prepare_fineweb_edu
        manifest = recipe(args.output, shard_tokens=args.shard_tokens)
    else:
        if args.input is None or args.tokenizer is None or args.bos_token_id is None or args.eos_token_id is None:
            parser.error("local preparation needs --input, --tokenizer, --bos-token-id and --eos-token-id")
        tokenizer = _load_tokenizer(args.tokenizer)
        paths = {name: path for name, path in (("train", args.input), ("validation", args.validation), ("test", args.test)) if path is not None}
        manifest = prepare_dataset(
            args.output,
            {args.source_name: {split: read_texts(path, text_field=args.text_field) for split, path in paths.items()}},
            encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
            tokenizer={"id": args.tokenizer_id, "revision": args.tokenizer_revision, "vocab_size": tokenizer.get_vocab_size(), "bos_token_id": args.bos_token_id, "eos_token_id": args.eos_token_id, "pad_token_id": args.eos_token_id},
            tokenizer_directory=args.tokenizer,
            shard_tokens=args.shard_tokens,
            provenance={"input_files": {split: {"name": path.name, "sha256": file_sha256(path)} for split, path in paths.items()}, "document_definition": "each_nonempty_line"},
        )
    print(json.dumps({"manifest": str(manifest.path), "fingerprint": manifest.fingerprint, "tokenizer": manifest.tokenizer}, indent=2))


if __name__ == "__main__":
    main()
