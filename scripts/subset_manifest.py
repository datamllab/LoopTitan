#!/usr/bin/env python3
"""Write a prepared manifest whose train split is a leading prefix of the original shards.

Train shards are taken in manifest order (sources in listed order, then shards in listed
order) until their cumulative tokens first reach ``--train-tokens``; a source left with no
train shard loses its train split. Validation, test and tokenizer are copied unchanged and
shard paths stay as they are, so OUT must sit in the same directory as the original
manifest: ``PreparedManifest`` resolves every path against its own directory.
"""

import argparse
import json
import sys
from pathlib import Path

from moeloop.data import PreparedManifest


def subset_payload(payload, train_tokens):
    """Return ``payload`` with only the train prefix, plus the kept train shards."""
    total = sum(shard["num_tokens"] for source in payload["sources"] for shard in source["splits"].get("train", []))
    if not 0 < train_tokens <= total:
        raise ValueError(f"--train-tokens must be in 1..{total} (the full train split)")
    kept, taken, sources = [], 0, []
    for source in payload["sources"]:
        splits = dict(source["splits"])
        shards = []
        for shard in splits.pop("train", []):
            if taken < train_tokens:
                shards.append(shard)
                taken += shard["num_tokens"]
        if shards:
            splits["train"] = shards
            kept += shards
        if splits:
            sources.append({**source, "splits": splits})
    return {**payload, "sources": sources}, kept


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-tokens", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    source = args.manifest / "manifest.json" if args.manifest.is_dir() else args.manifest
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    if output.parent != source.resolve().parent:
        parser.error(f"--output must be in {source.resolve().parent}, where the shard paths resolve")
    original = PreparedManifest.load(source, verify_hashes=False)
    try:
        payload, kept = subset_payload(original.payload, args.train_tokens)
    except ValueError as error:
        parser.error(str(error))
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        subset = PreparedManifest.load(temporary, verify_hashes=True)
    except Exception:
        temporary.unlink()
        raise
    temporary.replace(output)
    tokens = sum(shard["num_tokens"] for shard in kept)
    documents = sum(shard["num_documents"] for shard in kept)
    print(json.dumps({
        "output": str(output),
        "requested_train_tokens": args.train_tokens,
        "train_tokens": tokens,  # includes BOS/EOS; exceeds the request by at most one shard
        "train_labels": tokens - documents,  # the unit training counts (tokens - documents)
        "train_documents": documents,
        "train_shards": len(kept),
        "fingerprint": subset.fingerprint,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
