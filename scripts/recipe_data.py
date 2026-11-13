#!/usr/bin/env python3
"""Freeze public DCLM source shards and prepare immutable token documents on CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moeloop.prepare import (DCLM_REVISION, P1_LABEL_TOKENS, PUBLIC_LABEL_TOKENS,
                             freeze_dclm_source, prepare_dclm_packed, prepare_dclm_public,
                             prepare_fineweb_edu)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = os.environ.get("MOELOOP_DATA")
    parser.add_argument("--output", type=Path, default=default_output,
                        help="new preparation directory; defaults to MOELOOP_DATA")
    parser.add_argument("--freeze-dclm-source", type=Path,
                        help="write an ordered commit/size/SHA256 source JSON using HF metadata only, then exit")
    parser.add_argument("--dclm-shards", type=int, default=1024,
                        help="number of sorted public Parquet paths to freeze; no label capacity is assumed")
    parser.add_argument("--dclm-revision", default=DCLM_REVISION)
    parser.add_argument("--dclm-shard-list", type=Path,
                        help="optional explicit ordered paths, one per line, for --freeze-dclm-source")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dclm-source", type=Path,
                        help="frozen public DCLM source JSON; defaults 24B/10M/10M labels with rank-capacity headroom for P1")
    source.add_argument("--source-file", type=Path, help="reuse the exact pinned FineWeb Parquet file, checked by size and SHA256")
    source.add_argument("--fineweb-edu", action="store_true", help="explicitly prepare the legacy FineWeb-Edu/GPT2 fixture")
    source.add_argument("--packed-source", type=Path, help="existing DCLM loopmoe-packed-v1 directory/manifest; reuse tokens without decoding")
    parser.add_argument("--packed-chunk", action="append", help="explicit ordered packed chunk name; repeat; legacy default is the first two")
    parser.add_argument("--tokenizer", type=Path, help="local pinned Qwen3 tokenizer assets; required with --packed-source")
    parser.add_argument("--source-directory", type=Path, help="local root containing exact frozen DCLM Parquet paths; otherwise use the HF cache")
    parser.add_argument("--resume", action="store_true", help="resume --dclm-source at its last committed input shard")
    parser.add_argument("--cache-dir", type=Path, help="optional persistent Hugging Face cache for source/tokenizer files")
    parser.add_argument("--shard-tokens", type=int, help="output shard limit; DCLM public default 128M, legacy default 1M")
    for split in PUBLIC_LABEL_TOKENS:
        parser.add_argument(f"--{split}-label-tokens", type=int)
    args = parser.parse_args(argv)
    if args.freeze_dclm_source is not None:
        paths = args.dclm_shard_list.read_text().splitlines() if args.dclm_shard_list is not None else None
        frozen = freeze_dclm_source(args.freeze_dclm_source, num_shards=args.dclm_shards,
                                    revision=args.dclm_revision, shard_paths=paths)
        print(json.dumps({"source_manifest": str(args.freeze_dclm_source), "revision": frozen["revision"],
                          "shards": len(frozen["files"]), "source_bytes": sum(item["size"] for item in frozen["files"])}, indent=2))
        return
    if args.output is None:
        parser.error("preparation requires --output or MOELOOP_DATA")
    if not any((args.dclm_source, args.packed_source, args.source_file, args.fineweb_edu)):
        parser.error("choose --dclm-source, --packed-source, or an explicit legacy FineWeb source")
    if (args.resume or args.source_directory is not None) and args.dclm_source is None:
        parser.error("--resume and --source-directory require --dclm-source")
    if args.packed_chunk is not None and args.packed_source is None:
        parser.error("--packed-chunk requires --packed-source")
    defaults = P1_LABEL_TOKENS if args.dclm_source is not None else PUBLIC_LABEL_TOKENS
    limits = {split: getattr(args, f"{split}_label_tokens") if getattr(args, f"{split}_label_tokens") is not None else count
              for split, count in defaults.items()}
    shard_tokens = args.shard_tokens if args.shard_tokens is not None else (128_000_000 if args.dclm_source is not None else 1_000_000)
    if args.dclm_source is not None:
        manifest = prepare_dclm_public(args.output, source_manifest=args.dclm_source,
            tokenizer_directory=args.tokenizer, source_directory=args.source_directory, cache_dir=args.cache_dir,
            label_tokens=limits, shard_tokens=shard_tokens, resume=args.resume)
    elif args.packed_source is not None:
        if args.tokenizer is None:
            parser.error("--packed-source requires --tokenizer; the packed-data importer never downloads assets")
        manifest = prepare_dclm_packed(args.output, source_manifest=args.packed_source,
            tokenizer_directory=args.tokenizer, label_tokens=limits, shard_tokens=shard_tokens,
            chunk_names=args.packed_chunk)
    else:
        manifest = prepare_fineweb_edu(args.output, source_file=args.source_file, cache_dir=args.cache_dir,
            label_tokens=limits, shard_tokens=shard_tokens)
    print(json.dumps({"manifest": str(manifest.path), "fingerprint": manifest.fingerprint,
                      "tokenizer": manifest.tokenizer,
                      "selection": manifest.payload["provenance"]["selection"]}, indent=2))


if __name__ == "__main__":
    main()
