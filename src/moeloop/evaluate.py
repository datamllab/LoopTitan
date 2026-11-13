"""Heldout document-local CE and a thin command adapter to lm-evaluation-harness."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from itertools import islice
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PackedDataset, PreparedManifest, build_document_mask, file_sha256, manifest_byte_table
from .tokens import byte_table_sha256
from .training import isolated_rng


def token_byte_tensor(manifest: PreparedManifest, device) -> tuple[Tensor | None, tuple[int, ...] | None, str | None]:
    """Return ``(token_bytes, byte_table, bpb_unavailable)`` for a manifest.

    ``token_bytes`` holds the UTF-8 byte count of every ID as int64 on ``device``,
    zero-padded to the manifest ``vocab_size``: padded IDs never occur as labels.
    ``byte_table`` is the tokenizer's unpadded table (``manifest_byte_table``), which
    ``byte_table_sha256`` hashes. Without a table both are None and the reason says why.
    """
    byte_table, bpb_unavailable = manifest_byte_table(manifest)
    if byte_table is None:
        return None, None, bpb_unavailable
    token_bytes = torch.zeros(manifest.tokenizer["vocab_size"], dtype=torch.int64, device=device)
    token_bytes[:len(byte_table)] = torch.tensor(byte_table, dtype=torch.int64)
    return token_bytes, byte_table, None


def bpb_terms(labels: Tensor, token_bytes: Tensor) -> tuple[Tensor, Tensor]:
    """Return per-label byte counts and the mask of byte-bearing ("text") labels.

    Ignored labels (-100) and tokens with 0 bytes (EOS, PAD, every special or added
    token) count 0 bytes and are not text. BPB is the CE summed over text labels
    (``ce_sum_text``) divided by ``ln 2`` times the summed byte counts (``label_bytes``).
    """
    label_bytes = torch.where(labels == -100, 0, token_bytes[labels.clamp(min=0)])
    return label_bytes, label_bytes > 0


def bpb_fields(byte_table: tuple[int, ...] | None, bpb_unavailable: str | None,
               ce_sum_text: float, label_bytes: int) -> dict[str, Any]:
    """The five reported BPB fields; without a table the first four are None."""
    if byte_table is None:
        return {"label_bytes": None, "ce_sum_text": None, "bpb": None, "byte_table_sha256": None,
                "bpb_unavailable": bpb_unavailable}
    return {"label_bytes": label_bytes, "ce_sum_text": ce_sum_text,
            "bpb": ce_sum_text / (math.log(2) * label_bytes) if label_bytes else None,
            "byte_table_sha256": byte_table_sha256(byte_table),
            "bpb_unavailable": None if label_bytes else "no byte-bearing labels evaluated"}


@contextmanager
def heldout_batches(model, manifest: PreparedManifest, *, split: str, seq_len: int, batch_size: int,
                    max_batches: int | None) -> Iterator[SimpleNamespace]:
    """Yield a finite, ordered split as ``held`` with ``dataset``, ``device``, ``batches`` and ``complete``.

    ``held.batches`` gives at most ``max_batches`` batches ``(inputs, labels, forward
    kwargs)`` on the model's ``device``, with document-local causal attention.
    ``held.complete`` turns True when the split runs out before ``max_batches``.
    Inside, the model is in eval mode under an isolated RNG and inference mode; on
    exit its mode is restored and the split closed.
    """
    if batch_size < 1 or seq_len < 1 or (max_batches is not None and max_batches < 1):
        raise ValueError("batch_size, seq_len and max_batches must be positive")
    core = getattr(model, "model", model)
    if seq_len > core.config.max_seq_len:
        raise ValueError("seq_len exceeds the model context length")
    if manifest.tokenizer["vocab_size"] > core.config.vocab_size:
        raise ValueError("data tokenizer vocabulary exceeds the model vocabulary")
    device = next(core.parameters()).device
    held = SimpleNamespace(dataset=PackedDataset(manifest, split=split, seq_len=seq_len, shuffle=False, infinite=False),
                           device=device, complete=False)

    def batches():
        for inputs, labels in DataLoader(held.dataset, batch_size=batch_size, num_workers=0):
            inputs = {name: value.to(device) for name, value in inputs.items()}
            yield inputs, labels.to(device), dict(positions=inputs["positions"], router_mask=inputs["input_mask"],
                attention_mask=build_document_mask(inputs["document_ids"], inputs["input_mask"]))
        held.complete = True

    held.batches = islice(batches(), max_batches)
    was_training = model.training
    try:
        model.eval()
        with isolated_rng(), torch.inference_mode():
            yield held
    finally:
        held.dataset.close()
        model.train(was_training)


def evaluate_heldout(
    model,
    manifest: str | Path | PreparedManifest,
    *,
    split: str = "validation",
    seq_len: int = 256,
    batch_size: int = 1,
    max_batches: int | None = None,
    num_visits: int | None = None,
    document_loss_output: str | Path | None = None,
) -> dict[str, Any]:
    """Return raw CE sum/count under prepared-data labels, without a second shift.

    The function uses a finite, ordered split and restores model mode and RNG.
    It accepts a plain RecursiveLM or its HF wrapper already on the desired device.
    This data protocol is not a claim to reproduce an external WikiText PPL table.
    Optional per-document losses use stable single-source split ordinals, merge
    block continuations, and publish a new JSON file without overwriting.

    Bits per byte follow nanochat ``evaluate_bpb`` (:func:`bpb_terms`): labels whose
    token has 0 bytes (EOS, PAD, every special or added token) leave both ``ce_sum_text``
    and ``label_bytes``, and ``bpb = ce_sum_text / (ln 2 * label_bytes)``. A
    convention that keeps EOS CE in the numerator reports ``ce_sum / (ln 2 *
    label_bytes)``, which is larger. The byte table comes from the manifest's
    tokenizer assets (``moeloop.tokens``); without assets, or for a tokenizer
    that is not byte-level BPE, the four byte fields and per-document ``bytes``
    are None and ``bpb_unavailable`` says why; ``bpb`` is also None when no
    evaluated label has bytes. For a tokenizer with an NFC
    normalizer (Qwen3), bytes are counted on the NFC-normalized text. A
    manifest ``vocab_size`` padded above the tokenizer counts 0 bytes for the
    padded IDs, and ``byte_table_sha256`` hashes the tokenizer's own unpadded
    table, so it identifies the byte accounting independent of padding. Text CE
    comes from a second, unreduced CE: summing that one instead of the reduced CE
    changes ``ce_sum`` in its last bits.
    """
    core = getattr(model, "model", model)
    depth = core.config.num_visits if num_visits is None else num_visits
    manifest = manifest if isinstance(manifest, PreparedManifest) else PreparedManifest.load(manifest)
    document_path = Path(document_loss_output).resolve() if document_loss_output is not None else None
    if document_path is not None:
        if document_path.exists():
            raise FileExistsError(f"document loss output already exists: {document_path}")
        if len(manifest.source_names(split)) != 1:
            raise ValueError("document loss output requires a single-source split")
    documents: dict[int, dict[str, Any]] = {}
    ce_sum, valid_tokens, input_tokens, batches = 0.0, 0, 0, 0
    ce_sum_text, label_bytes = 0.0, 0
    with heldout_batches(model, manifest, split=split, seq_len=seq_len, batch_size=batch_size,
                         max_batches=max_batches) as held:
        token_bytes, byte_table, bpb_unavailable = token_byte_tensor(manifest, held.device)
        for inputs, labels, kwargs in held.batches:
            output = core(inputs["input"], **kwargs, num_visits=depth)
            logits, targets = output.logits.flatten(0, 1).float(), labels.flatten()
            losses = F.cross_entropy(logits, targets, reduction="sum" if document_path is None else "none", ignore_index=-100)
            ce = losses if document_path is None else losses.double().sum()
            if not torch.isfinite(ce).item():
                raise ValueError("heldout CE is non-finite")
            target_bytes = torch.zeros_like(targets)
            if token_bytes is not None:
                token_losses = losses if document_path is not None else F.cross_entropy(logits, targets, reduction="none", ignore_index=-100)
                target_bytes, text = bpb_terms(targets, token_bytes)
                ce_sum_text += token_losses[text].double().sum().item()
                label_bytes += int(target_bytes.sum().item())
            if document_path is None:
                ce_sum += ce.double().item()
            else:
                valid = labels.flatten() != -100
                ids, inverse, counts = torch.unique(inputs["document_ids"].flatten()[valid],
                    sorted=True, return_inverse=True, return_counts=True)
                sums = torch.zeros(ids.numel(), dtype=torch.float64, device=held.device)
                sums.scatter_add_(0, inverse, losses[valid].double())
                sizes = torch.zeros_like(ids).scatter_add_(0, inverse, target_bytes[valid])
                for document_id, count, total, size in zip(ids.tolist(), counts.tolist(), sums.tolist(), sizes.tolist()):
                    row = documents.setdefault(document_id, {"document_id": document_id, "valid_tokens": 0, "ce_sum": 0.0, "bytes": 0})
                    row["valid_tokens"] += count
                    row["ce_sum"] += total
                    row["bytes"] += size
            valid_tokens += int((labels != -100).sum().item())
            input_tokens += int(inputs["input_mask"].sum().item())
            batches += 1
    if valid_tokens == 0:
        raise ValueError("heldout split produced no valid targets")
    if document_path is not None:
        ce_sum = math.fsum(row["ce_sum"] for row in documents.values())
    mean = ce_sum / valid_tokens
    result = {
        "split": split, "seq_len": seq_len, "num_visits": depth,
        "batch_size": batch_size, "max_batches": max_batches,
        "complete_split": held.complete,
        "batches": batches, "ce_sum": ce_sum, "valid_tokens": valid_tokens,
        "input_tokens": input_tokens, "ce": mean,
        "perplexity": math.exp(mean) if mean < math.log(sys.float_info.max) else None,
        "manifest_fingerprint": manifest.fingerprint,
        "label_contract": "already_aligned_document_internal_next_token",
        "attention": "document_causal_block_local",
        "source_counts": held.dataset.source_counts,
        "tokenizer": {key: manifest.tokenizer[key] for key in ("id", "revision", "vocab_size")},
        **bpb_fields(byte_table, bpb_unavailable, ce_sum_text, label_bytes),
    }
    if document_path is not None:
        payload = {key: result[key] for key in ("manifest_fingerprint", "split", "seq_len", "num_visits", "complete_split", "label_contract")}
        payload.update(format="moeloop-document-losses", version=1, source=manifest.source_names(split)[0],
            documents=[{**row, "ce": row["ce_sum"] / row["valid_tokens"], "bytes": row["bytes"] if byte_table is not None else None}
                       for _, row in sorted(documents.items())])
        _write_document_losses(document_path, payload)
        result["document_losses"] = {"path": str(document_path), "sha256": file_sha256(document_path), "document_count": len(documents)}
    return result


def _write_document_losses(path: Path, payload: dict[str, Any]) -> None:
    """Publish a fully written file atomically, including an overwrite race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def lm_eval_command(
    model_path: str | Path,
    tasks: Sequence[str],
    *,
    batch_size: int | str = 1,
    device: str = "cuda",
    output_path: str | Path | None = None,
    log_samples: bool = False,
) -> list[str]:
    """Build the existing harness's HF command; no custom benchmark runner.

    Install ``lm_eval[hf]`` separately, pin version/tasks for a reported comparison,
    and execute the returned argv with ``subprocess.run(argv, check=True)``.
    Opt-in ``log_samples`` saves private raw samples and native document/prompt/target
    task hashes; the default keeps existing lightweight callers unchanged. Native
    ``batch_size`` accepts positive integers, ``auto`` or ``auto:N``. The
    official CLI supports this legacy form as well as its newer ``run`` subcommand:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
    """
    path = Path(model_path).resolve()
    if not path.is_dir() or "," in str(path):
        raise ValueError("model_path must be an existing directory without commas")
    if not tasks or any(not task or "," in task for task in tasks):
        raise ValueError("provide nonempty task names")
    batch = str(batch_size)
    try:
        if type(batch_size) not in (int, str):
            raise ValueError
        if batch.startswith("auto:"):
            schedule = float(batch[5:])
            if not math.isfinite(schedule) or schedule <= 0:
                raise ValueError
        elif batch != "auto" and int(batch) <= 0:
            raise ValueError
    except ValueError:
        raise ValueError("batch_size must be a positive integer, auto, or auto:positive_number") from None
    command = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", f"pretrained={path},trust_remote_code=True", "--tasks", ",".join(tasks), "--num_fewshot", "0", "--batch_size", str(batch_size), "--device", device]
    if log_samples:
        command.append("--log_samples")
    if output_path is not None:
        command.extend(("--output_path", str(output_path)))
    return command


def _check_tokenizer_identity(model_path: Path, manifest: PreparedManifest) -> None:
    provenance_path = model_path / "export_provenance.json"
    if not provenance_path.is_file():
        return
    source_files = json.loads(provenance_path.read_text())["tokenizer_source_files"]
    prepared_files = {Path(name).name: digest for name, digest in manifest.tokenizer.get("files", {}).items()}
    if prepared_files and source_files != prepared_files:
        raise ValueError("evaluation data tokenizer files differ from the exported model's source tokenizer")


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local exported HF directory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--num-visits", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="new JSON result file, without overwrite")
    parser.add_argument("--document-loss-output", type=Path, help="new per-document loss JSON for a single-source split, without overwrite")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.document_loss_output is not None:
        if args.document_loss_output.exists():
            parser.error(f"document loss output already exists: {args.document_loss_output}")
        if args.output is not None and args.output.resolve() == args.document_loss_output.resolve():
            parser.error("aggregate and document loss outputs must have different paths")
    manifest = PreparedManifest.load(args.manifest)
    _check_tokenizer_identity(args.model, manifest)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, local_files_only=True).to(args.device)
    result = evaluate_heldout(model, manifest, split=args.split, seq_len=args.seq_len, batch_size=args.batch_size, max_batches=args.max_batches, num_visits=args.num_visits, document_loss_output=args.document_loss_output)
    content = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(content, end="")


if __name__ == "__main__":
    main()
