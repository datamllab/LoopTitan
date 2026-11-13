"""CORE in-context evaluation (DCLM v2 baselines) of a local Transformers causal LM.

CORE is the DCLM paper's mean centered accuracy over 22 low-variance tasks
(arXiv 2406.11794, Section 3 and Appendix G): each task accuracy is rescaled as
``(acc - random) / (1 - random)`` so 0 is random guessing and 1 is perfect.
The task list equals ``low_variance_datasets`` in DCLM's
``eval/additional_aggregation.json``; few-shot counts, task types and
continuation delimiters equal the eval bundle's ``core.yaml``.

Prompting and scoring follow nanochat ``nanochat/core_eval.py`` and
``scripts/base_eval.py`` (MIT, checked at commit 92d63d4e, 2026-07-03), not
llm-foundry: BOS is prepended to every prompt; a multiple-choice continuation is
the tokens after the options' common token prefix, a schema continuation the
options' common token suffix; the option with the lowest mean continuation CE
wins; a language-modeling example is correct only if every continuation token is
the greedy argmax. Data is shuffled with ``random.Random(1337)`` and truncated to
``max_per_task``; few-shot examples for item ``i`` are
``random.Random(1234 + i).sample`` of the other items of that same (truncated)
list, so a truncated run also changes its few-shot pool, as in nanochat. Prompts
longer than the model context keep their last tokens, as in nanochat; a
continuation that would lose its predicting position raises instead.

Random baselines are DCLM's corrected ``eval_meta_data.csv`` (PR #115, merged
2025-09-09 as commit 361714b): commonsense_qa 40.3 (was 20), agi_eval_lsat_ar 25
(was 20), bigbench_language_identification 25 (was 9.1). The bundle's own
``eval_meta_data.csv`` still holds the v1 values and is not read. ``core_v1``
is also reported, because nanochat and pre-September-2025 DCLM numbers use v1.

Known limits: nanochat reports matching the DCLM reference on all tasks except
squad (31% versus 37%); this module inherits that convention. Scores depend on
numeric precision: the model runs in its loaded dtype without autocast, while
the bundle's GPT-2 reference CSVs came from an older nanochat run under bf16
autocast, and near-chance multiple-choice tasks flip a few examples between the
two (GPT-2: copa 0.60 in that CSV, 0.63 here in fp32 on CPU; its
language-modeling tasks match exactly). ``--batch-size`` bounds memory, since full
``batch x tokens x vocab`` logits are materialized. The bundle is only read from
a local directory; ``--bundle-info`` prints the official URL, hashes and manual
download steps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .data import file_sha256

CORE_VERSION = "dclm-v2"
BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
BUNDLE_ZIP_SHA256 = "90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200"
BUNDLE_SHA256 = "7233898952f8fa5b59293cbf7b2cd7afa5ed0eec14feabf815d243d55800ad8c"
SHUFFLE_SEED = 1337
FEWSHOT_SEED = 1234


@dataclass(frozen=True)
class Task:
    label: str
    dataset: str
    kind: str
    shots: int
    delimiter: str
    baseline: str  # DCLM v2 random baseline, in percent as in eval_meta_data.csv
    baseline_v1: str


_MC, _SCHEMA, _LM, _ANSWER = "multiple_choice", "schema", "language_modeling", "\nAnswer: "
TASKS = (
    Task("hellaswag_zeroshot", "language_understanding/hellaswag.jsonl", _MC, 0, " ", "25", "25"),
    Task("jeopardy", "world_knowledge/jeopardy_all.jsonl", _LM, 10, _ANSWER, "0", "0"),
    Task("bigbench_qa_wikidata", "world_knowledge/bigbench_qa_wikidata.jsonl", _LM, 10, " ", "0", "0"),
    Task("arc_easy", "world_knowledge/arc_easy.jsonl", _MC, 10, _ANSWER, "25", "25"),
    Task("arc_challenge", "world_knowledge/arc_challenge.jsonl", _MC, 10, _ANSWER, "25", "25"),
    Task("copa", "commonsense_reasoning/copa.jsonl", _MC, 0, " ", "50", "50"),
    Task("commonsense_qa", "commonsense_reasoning/commonsense_qa.jsonl", _MC, 10, " ", "40.3", "20"),
    Task("piqa", "commonsense_reasoning/piqa.jsonl", _MC, 10, _ANSWER, "50", "50"),
    Task("openbook_qa", "commonsense_reasoning/openbook_qa.jsonl", _MC, 0, " ", "25", "25"),
    Task("lambada_openai", "language_understanding/lambada_openai.jsonl", _LM, 0, " ", "0", "0"),
    Task("hellaswag", "language_understanding/hellaswag.jsonl", _MC, 10, " ", "25", "25"),
    Task("winograd", "language_understanding/winograd_wsc.jsonl", _SCHEMA, 0, " ", "50", "50"),
    Task("winogrande", "language_understanding/winogrande.jsonl", _SCHEMA, 0, " ", "50", "50"),
    Task("bigbench_dyck_languages", "symbolic_problem_solving/bigbench_dyck_languages.jsonl", _LM, 10, " ", "0", "0"),
    Task("agi_eval_lsat_ar", "symbolic_problem_solving/agi_eval_lsat_ar.jsonl", _MC, 3, " ", "25", "20"),
    Task("bigbench_cs_algorithms", "symbolic_problem_solving/bigbench_cs_algorithms.jsonl", _LM, 10, " ", "0", "0"),
    Task("bigbench_operators", "symbolic_problem_solving/bigbench_operators.jsonl", _LM, 10, " ", "0", "0"),
    Task("bigbench_repeat_copy_logic", "symbolic_problem_solving/bigbench_repeat_copy_logic.jsonl", _LM, 10, " ", "0", "0"),
    Task("squad", "reading_comprehension/squad.jsonl", _LM, 10, " ", "0", "0"),
    Task("coqa", "reading_comprehension/coqa.jsonl", _LM, 0, " ", "0", "0"),
    Task("boolq", "reading_comprehension/boolq.jsonl", _MC, 10, _ANSWER, "62", "62"),
    Task("bigbench_language_identification", "language_understanding/bigbench_language_identification.jsonl", _MC, 10, " ", "25", "9.1"),
)
TASK_BY_LABEL = {task.label: task for task in TASKS}


def directory_sha256(path: str | Path) -> str:
    """SHA-256 of the ``sha256sum`` listing of every file, sorted by relative path.

    Equals ``cd DIR && find -L . -type f -printf '%P\\n' | LC_ALL=C sort |
    xargs -d '\\n' sha256sum | sha256sum``.
    """
    root = Path(path)
    files = sorted((Path(parent) / name).relative_to(root).as_posix()
                   for parent, _, names in os.walk(root, followlinks=True) for name in names
                   if (Path(parent) / name).is_file())
    if not files:
        raise ValueError(f"directory has no files: {root}")
    return hashlib.sha256("".join(f"{file_sha256(root / name)}  {name}\n" for name in files).encode()).hexdigest()


def centered(accuracy: float, baseline: str) -> float:
    """DCLM's centered accuracy for a baseline given in percent, as in its aggregation."""
    return (accuracy - 0.01 * float(baseline)) / (1.0 - 0.01 * float(baseline))


def load_examples(bundle: str | Path, task: Task, max_per_task: int | None = None) -> list[dict[str, Any]]:
    with (Path(bundle) / "eval_data" / task.dataset).open(encoding="utf-8") as stream:
        data = [json.loads(line) for line in stream if line.strip()]
    random.Random(SHUFFLE_SEED).shuffle(data)
    return data if max_per_task is None else data[:max_per_task]


def fewshot_indices(index: int, size: int, shots: int) -> list[int]:
    if shots == 0:
        return []
    if size <= shots:
        raise ValueError(f"{shots}-shot sampling needs more than {shots} examples, got {size}")
    return random.Random(FEWSHOT_SEED + index).sample([i for i in range(size) if i != index], shots)


def render_multiple_choice(item: dict, delimiter: str, shots: Sequence[dict] = ()) -> list[str]:
    prefix = "".join(f"{shot['query']}{delimiter}{shot['choices'][shot['gold']]}\n\n" for shot in shots)
    return [f"{prefix}{item['query']}{delimiter}{choice}" for choice in item["choices"]]


def render_schema(item: dict, delimiter: str, shots: Sequence[dict] = ()) -> list[str]:
    prefix = "".join(f"{shot['context_options'][shot['gold']]}{delimiter}{shot['continuation']}\n\n" for shot in shots)
    return [f"{prefix}{context}{delimiter}{item['continuation']}" for context in item["context_options"]]


def render_language_modeling(item: dict, delimiter: str, shots: Sequence[dict] = ()) -> list[str]:
    """Return the prompt without (stripped, as in nanochat) and with the continuation."""
    prefix = "".join(f"{shot['context'].strip()}{delimiter}{shot['continuation']}\n\n" for shot in shots)
    without = f"{prefix}{item['context'].strip()}{delimiter}"
    return [without.strip(), without + item["continuation"]]


RENDER = {_MC: render_multiple_choice, _SCHEMA: render_schema, _LM: render_language_modeling}


def _common_length(sequences: Sequence[Sequence[int]], *, from_end: bool) -> int:
    width = min(len(sequence) for sequence in sequences)
    for offset in range(width):
        position = -1 - offset if from_end else offset
        if any(sequence[position] != sequences[0][position] for sequence in sequences):
            return offset
    return width


def continuation_spans(kind: str, tokens: Sequence[list[int]]) -> list[tuple[list[int], int, int]]:
    """Return ``(tokens, start, end)`` rows; tokens[start:end] is the scored continuation."""
    if kind == _MC:
        start = _common_length(tokens, from_end=False)
        spans = [(row, start, len(row)) for row in tokens]
    elif kind == _SCHEMA:
        suffix = _common_length(tokens, from_end=True)
        spans = [(row, len(row) - suffix, len(row)) for row in tokens]
    else:
        without, with_continuation = tokens
        if with_continuation[:len(without)] != without:
            raise ValueError("the prompt without continuation must tokenize to a prefix of the prompt with it")
        spans = [(with_continuation, len(without), len(with_continuation))]
    # One empty span is valid data (a piqa option that is a token prefix of the other, duplicate
    # arc choices): its nan mean CE then loses unless listed first, as in nanochat. No scorable
    # span at all means a broken tokenizer or example.
    if not any(1 <= start < end for _, start, end in spans):
        raise ValueError("no option has a scorable continuation span; check the tokenizer and the example")
    return spans


def task_rows(task: Task, data: Sequence[dict], tokenizer, bos_token_id: int,
              max_tokens: int | None = None) -> list[tuple[int, list[int], int, int]]:
    """Render, tokenize and crop every example into ``(example, tokens, start, end)`` rows."""
    rows = []
    for index, item in enumerate(data):
        shots = [data[i] for i in fewshot_indices(index, len(data), task.shots)]
        encoded = tokenizer(RENDER[task.kind](item, task.delimiter, shots), add_special_tokens=False)["input_ids"]
        try:
            spans = continuation_spans(task.kind, [[bos_token_id, *ids] for ids in encoded])
        except ValueError as error:
            raise ValueError(f"{task.label} example {index}: {error}") from None
        for tokens, start, end in spans:
            crop = 0 if max_tokens is None else max(0, len(tokens) - max_tokens)
            if start - crop < 1:
                raise ValueError(f"{task.label} example {index}: continuation does not fit the model context")
            rows.append((index, tokens[crop:], start - crop, end - crop))
    return rows


@torch.inference_mode()
def score_rows(model, rows: Sequence[tuple[list[int], int, int]], *, pad_token_id: int,
               batch_size: int = 8) -> list[tuple[float, bool]]:
    """Mean continuation CE and greedy exact match per row, right-padded under a mask."""
    device = next(model.parameters()).device
    scores = []
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]
        ids = torch.full((len(chunk), max(len(tokens) for tokens, _, _ in chunk)), pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, (tokens, _, _) in enumerate(chunk):
            ids[row, :len(tokens)] = torch.tensor(tokens)
            mask[row, :len(tokens)] = 1
        ids, mask = ids.to(device), mask.to(device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        for row, (_, start, end) in enumerate(chunk):
            predicted, target = logits[row, start - 1:end - 1].float(), ids[row, start:end]
            scores.append((F.cross_entropy(predicted, target).item(), torch.equal(predicted.argmax(-1), target)))
    return scores


def example_correct(task: Task, data: Sequence[dict], rows, scores) -> list[bool]:
    by_example: list[list[tuple[float, bool]]] = [[] for _ in data]
    for (index, *_), score in zip(rows, scores):
        by_example[index].append(score)
    if task.kind == _LM:
        return [options[0][1] for options in by_example]
    correct = []
    for item, options in zip(data, by_example):
        losses = [loss for loss, _ in options]
        correct.append(losses.index(min(losses)) == item["gold"])
    return correct


def evaluate_core(model, tokenizer, bundle: str | Path, *, tasks: Sequence[str] | None = None,
                  max_per_task: int | None = None, batch_size: int = 8) -> dict[str, Any]:
    """Evaluate the named CORE tasks (all 22 by default) on a model already on its device."""
    labels = [task.label for task in TASKS] if tasks is None else list(dict.fromkeys(tasks))
    unknown = sorted(set(labels) - set(TASK_BY_LABEL))
    if unknown or not labels:
        raise ValueError(f"unknown or empty CORE task selection: {unknown}")
    if batch_size < 1 or (max_per_task is not None and max_per_task < 1):
        raise ValueError("batch_size and max_per_task must be positive")
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else model.config.bos_token_id
    if bos is None:
        raise ValueError("CORE prompts start with BOS; neither the tokenizer nor the model config defines it")
    max_tokens = getattr(model.config, "max_position_embeddings", None)
    selected = [task for task in TASKS if task.label in labels]
    if max_per_task is not None and max_per_task <= max(task.shots for task in selected):
        raise ValueError(f"max_per_task must exceed the largest selected few-shot count ({max(task.shots for task in selected)})")
    bundle = Path(bundle)
    bundle_sha256 = directory_sha256(bundle)
    results = {}
    was_training = model.training
    try:
        model.eval()
        for task in selected:
            data = load_examples(bundle, task, max_per_task)
            rows = task_rows(task, data, tokenizer, bos, max_tokens)
            scores = score_rows(model, [row[1:] for row in rows], pad_token_id=bos, batch_size=batch_size)
            correct = sum(example_correct(task, data, rows, scores))
            accuracy = correct / len(data)
            results[task.label] = {
                "task_type": task.kind, "num_fewshot": task.shots, "continuation_delimiter": task.delimiter,
                "dataset": task.dataset, "dataset_sha256": file_sha256(bundle / "eval_data" / task.dataset),
                "examples": len(data), "correct": correct, "accuracy": accuracy,
                "random_baseline": 0.01 * float(task.baseline), "centered": centered(accuracy, task.baseline),
                "random_baseline_v1": 0.01 * float(task.baseline_v1), "centered_v1": centered(accuracy, task.baseline_v1),
            }
    finally:
        model.train(was_training)
    complete = len(selected) == len(TASKS) and max_per_task is None
    mean = math.fsum(row["centered"] for row in results.values()) / len(results)
    mean_v1 = math.fsum(row["centered_v1"] for row in results.values()) / len(results)
    return {
        "format": "moeloop-core", "version": 1, "core_version": CORE_VERSION,
        "core": mean if complete else None, "core_v1": mean_v1 if complete else None,
        "mean_centered": mean, "mean_centered_v1": mean_v1, "complete": complete,
        "tasks": results, "max_per_task": max_per_task, "batch_size": batch_size,
        "bos_token_id": bos, "max_tokens": max_tokens,
        "fewshot": {"shuffle_seed": SHUFFLE_SEED, "sample_seed": FEWSHOT_SEED,
                    "rule": "Random(shuffle_seed).shuffle(data); data[:max_per_task]; "
                            "item i uses Random(sample_seed + i).sample(other indices, num_fewshot)"},
        "bundle": {"path": str(bundle.resolve()), "sha256": bundle_sha256, "official": bundle_sha256 == BUNDLE_SHA256},
    }


def bundle_info() -> str:
    return "\n".join((
        "CORE eval bundle (nanochat's copy of the DCLM/llm-foundry ICL data)",
        f"url: {BUNDLE_URL}",
        f"zip sha256: {BUNDLE_ZIP_SHA256}",
        f"unzipped eval_bundle/ directory sha256: {BUNDLE_SHA256}",
        "manual download (the evaluator never downloads):",
        f"  curl -fLO {BUNDLE_URL} && sha256sum eval_bundle.zip",
        "  python -c \"import zipfile; zipfile.ZipFile('eval_bundle.zip').extractall('.')\"",
        "  python -m moeloop.core --model <HF export dir> --bundle eval_bundle --output <new json>",
    ))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CORE (DCLM v2) evaluation of a local HF export")
    parser.add_argument("--model", type=Path, help="local exported HF directory")
    parser.add_argument("--bundle", type=Path, help="local unzipped eval_bundle directory")
    parser.add_argument("--output", type=Path, help="new JSON result file, without overwrite")
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_BY_LABEL), help="subset of CORE tasks; default all 22")
    parser.add_argument("--max-per-task", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="sequences per forward")
    parser.add_argument("--bundle-info", action="store_true", help="print the official bundle URL, hashes and download steps")
    args = parser.parse_args(argv)
    if args.bundle_info:
        print(bundle_info())
        return
    if args.model is None or args.bundle is None or args.output is None:
        parser.error("--model, --bundle and --output are required")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_sha256 = directory_sha256(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, local_files_only=True).to(args.device)
    result = evaluate_core(model, tokenizer, args.bundle, tasks=args.tasks, max_per_task=args.max_per_task, batch_size=args.batch_size)
    result["model"] = {"path": str(args.model.resolve()), "sha256": model_sha256,
                       "dtype": str(next(model.parameters()).dtype), "device": args.device}
    result["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers")}
    content = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(content)
    print(content, end="")


if __name__ == "__main__":
    main()
