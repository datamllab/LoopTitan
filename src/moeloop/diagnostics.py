"""Per-visit diagnostics of one recursive model on one prepared split.

``python -m moeloop.diagnostics --model <HF export dir> --manifest <prepared> --output <new JSON>``
prints and saves one JSON result; ``--plot <file.svg|file.png>`` adds a figure
when matplotlib is installed (the ``report`` extra). In Python, :func:`diagnose`
also accepts an in-memory ``RecursiveLM`` or its HF wrapper.

Visits are 0-based, as in ``ModelOutput.router_call_slots``. The quantities are:

* ``logit_lens``: the head on ``RecursiveLM.readout`` of the state after visit v
  (the remaining suffix cells and the final norm), the readout ``forward`` builds
  with ``collect_readouts`` (which itself requires ``exit_distribution``).
  States are read with forward hooks on the first and last core cells, so this
  needs whole-visit states: ``order="stack"`` or a single core cell.
* ``residual``: the update ``u_v`` added by the core cells in visit v, i.e. the
  state leaving the last core cell minus the state entering the first one
  (after input injection/reinjection, which is not part of ``u_v``). Token means
  of ``|u_v|``, ``|x_v|``, ``|u_v|/|x_v|`` (tokens with ``|x_v|>0``) and
  ``cos(u_{v-1}, u_v)``.
* ``routing``: per sparse bank (one physical cell) and visit, the entropy in
  nats of its expert load (the returned per-call router counts); the multiset
  Jaccard ``sum min/sum max`` of two visits' loads; and the token-mean Jaccard
  of the selected top-k sets, read with a forward hook on the routers. Load
  entropy is not the mean per-token router entropy.
* ``effective_depth``: the loss of the forward truncated to ``num_visits=r``
  for r in 1..K. For whole-visit models without EMA state this is exactly the
  logit lens at visit r-1, which is reused (``effective_depth_from``). EMA gains
  depend on the executed depth, so there the truncated curve is a separate
  forward per r and differs from the lens; layer order also runs one per r.

Losses use prepared-data labels (``-100`` ignored) and document-local attention,
as :func:`moeloop.evaluate.evaluate_heldout` does, over the same batches
(:func:`moeloop.evaluate.heldout_batches`). BPB uses the same terms
(:func:`moeloop.evaluate.bpb_terms`): labels with a zero-byte table entry
(special tokens) leave both the numerator and the denominator. The byte table
comes from the manifest's tokenizer assets (``moeloop.tokens``); without assets,
or for a tokenizer that is not byte-level BPE, only CE is reported and
``bpb_unavailable`` says why.

``--fit-readout`` adds ``readout`` (:func:`fit_readout`): a mixed readout over the
whole-visit states, fitted on the frozen model. Under ``scripts/run.py`` (``RUN_DIR``
set) the command writes ``$RUN_DIR/config.json``, the source run's config with an
``analysis`` block (:func:`source_config`), and a flat ``$RUN_DIR/metrics.json``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from itertools import combinations, islice
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PackedDataset, PreparedManifest, build_document_mask
from .evaluate import _check_tokenizer_identity, bpb_fields, bpb_terms, heldout_batches, token_byte_tensor


def load_entropy(counts: Tensor) -> float:
    """Entropy in nats of a nonnegative count vector, with 0 log 0 = 0."""
    p = counts.double() / counts.sum()
    return float(-(p[p > 0] * p[p > 0].log()).sum())


def multiset_jaccard(left: Tensor, right: Tensor) -> float:
    """Jaccard of two expert-selection multisets given as count vectors."""
    return float(torch.minimum(left, right).sum().double() / torch.maximum(left, right).sum().double())


def token_jaccard(left: Tensor, right: Tensor, valid: Tensor, experts: int) -> Tensor:
    """Sum over valid tokens of |A∩B|/|A∪B| for top-k id tensors [..., k]."""
    a = F.one_hot(left, experts).sum(-2) > 0
    b = F.one_hot(right, experts).sum(-2) > 0
    ratio = (a & b).sum(-1).double() / (a | b).sum(-1).double()
    return ratio[valid].sum()


@contextmanager
def _record(core) -> Iterator[dict[str, list]]:
    """Record core inputs/outputs per call and routed top-k ids in call order."""
    record: dict[str, list] = {"inputs": [], "outputs": [], "ids": []}
    handles = [
        core.core[0].register_forward_pre_hook(lambda _, args: record["inputs"].append(args[0])),
        core.core[-1].register_forward_hook(lambda _, args, output: record["outputs"].append(output[0])),
    ]
    handles += [module.register_forward_hook(lambda _, args, output: record["ids"].append(output.ids))
                for group in (core.prefix, core.core, core.suffix) for cell in group for module in cell.routers]
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


class _Loss:
    """Accumulates CE over valid labels and, with a byte table, text CE and bytes."""

    def __init__(self, token_bytes: Tensor | None, byte_table: tuple[int, ...] | None) -> None:
        self.token_bytes, self.byte_table = token_bytes, byte_table
        self.ce_sum, self.tokens, self.text_sum, self.bytes = 0.0, 0, 0.0, 0

    def add(self, logits: Tensor, labels: Tensor) -> None:
        losses = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), reduction="none", ignore_index=-100).double()
        valid = labels.flatten() != -100
        if not torch.isfinite(losses[valid]).all():
            raise ValueError("diagnostic CE is non-finite")
        self.ce_sum += float(losses.sum())
        self.tokens += int(valid.sum())
        if self.token_bytes is not None:
            label_bytes, text = bpb_terms(labels.flatten(), self.token_bytes)
            self.text_sum += float(losses[text].sum())
            self.bytes += int(label_bytes.sum())

    def result(self) -> dict[str, Any]:
        row = {"ce_sum": self.ce_sum, "valid_tokens": self.tokens, "ce": self.ce_sum / self.tokens}
        if self.byte_table is not None:
            fields = bpb_fields(self.byte_table, None, self.text_sum, self.bytes)
            row.update((key, fields[key]) for key in ("ce_sum_text", "label_bytes", "bpb"))
        return row


def load_export(path: str | Path, manifest: PreparedManifest, device: str):
    """An HF export on ``device``, after its tokenizer is checked against the manifest's."""
    from transformers import AutoModelForCausalLM
    _check_tokenizer_identity(Path(path), manifest)
    return AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, local_files_only=True).to(device)


def diagnose(
    model, manifest: str | Path | PreparedManifest, *, split: str = "validation", seq_len: int = 256,
    batch_size: int = 1, max_batches: int | None = None, num_visits: int | None = None, device: str = "cpu",
) -> dict[str, Any]:
    """Return the JSON-ready diagnostics; model mode and RNG are restored.

    ``model`` is a RecursiveLM, its HF wrapper, or a local HF export directory,
    which is loaded onto ``device`` after the export's tokenizer is checked
    against the manifest's. ``device`` applies only to a model loaded from a
    path; an in-memory model stays on its own device.
    """
    manifest = manifest if isinstance(manifest, PreparedManifest) else PreparedManifest.load(manifest)
    if isinstance(model, (str, Path)):
        model = load_export(model, manifest, device)
    core = getattr(model, "model", model)
    config = core.config
    depth = config.num_visits if num_visits is None else num_visits
    whole_visits = config.order == "stack" or config.num_cells == 1
    skipped = {} if whole_visits else {name: "per-visit states need order='stack' or one core cell"
                                       for name in ("logit_lens", "residual")}
    # Without whole visits or with depth-dependent EMA gains, visit r-1 of the
    # K-visit forward is not the r-visit forward, so the curve needs its own forwards.
    truncate = not whole_visits or config.state_update == "ema"
    residual = [dict(update=0.0, state=0.0, relative=0.0, relative_tokens=0, cosine=0.0, cosine_tokens=0)
                for _ in range(depth)]
    loads: dict[tuple, Tensor] = {}
    overlaps: dict[tuple, list[float]] = {}
    widths: dict[tuple, int] = {}
    valid_inputs = batches = 0
    with heldout_batches(model, manifest, split=split, seq_len=seq_len, batch_size=batch_size,
                         max_batches=max_batches) as held:
        token_bytes, byte_table, bpb_unavailable = token_byte_tensor(manifest, held.device)
        lens = [_Loss(token_bytes, byte_table) for _ in range(depth)]
        truncated = [_Loss(token_bytes, byte_table) for _ in range(depth)] if truncate else lens
        for inputs, labels, kwargs in held.batches:
            valid = inputs["input_mask"]
            with _record(core) as record:
                output = core(inputs["input"], **kwargs, num_visits=depth)
            if truncate:
                truncated[-1].add(output.logits, labels)
                for r in range(1, depth):
                    truncated[r - 1].add(core(inputs["input"], **kwargs, num_visits=r).logits, labels)
            if whole_visits:
                cell_inputs, *_ = core.parse_inputs(inputs["input"], **kwargs, num_visits=depth)
                previous = None
                for visit, (state, after) in enumerate(zip(record["inputs"], record["outputs"])):
                    lens[visit].add(core.lm_head(core.readout(after, visit, cell_inputs)), labels)
                    update = (after - state).float()
                    update_norm, state_norm = update.norm(dim=-1)[valid], state.float().norm(dim=-1)[valid]
                    row = residual[visit]
                    row["update"] += float(update_norm.double().sum())
                    row["state"] += float(state_norm.double().sum())
                    row["relative"] += float((update_norm / state_norm)[state_norm > 0].double().sum())
                    row["relative_tokens"] += int((state_norm > 0).sum())
                    if previous is not None:
                        cosine = F.cosine_similarity(previous, update, dim=-1)[valid]
                        row["cosine"] += float(cosine.double().sum())
                        row["cosine_tokens"] += cosine.numel()
                    previous = update
            sparse = [(slot, width, counts) for slot, width, counts in zip(
                output.router_call_slots, output.router_call_widths, output.router_call_counts) if width]
            if len(sparse) != len(record["ids"]):
                raise RuntimeError("router hook calls differ from the model's sparse router calls")
            ids: dict[tuple, dict[int, Tensor]] = {}
            for ((region, index, visit), width, counts), routed in zip(sparse, record["ids"]):
                bank = (region, index)
                widths[bank] = width
                loads[bank + (visit,)] = loads.get(bank + (visit,), 0) + counts[:width].cpu()
                ids.setdefault(bank, {})[visit] = routed
            for bank, routed in ids.items():
                for i, j in combinations(sorted(routed), 2):
                    totals = overlaps.setdefault(bank + (i, j), [0.0, 0])
                    totals[0] += float(token_jaccard(routed[i], routed[j], valid, widths[bank]))
                    totals[1] += int(valid.sum())
            valid_inputs += int(valid.sum())
            batches += 1
    if batches == 0 or truncated[-1].tokens == 0:
        raise ValueError("diagnostic split produced no valid targets")
    final = bpb_fields(byte_table, bpb_unavailable, truncated[-1].text_sum, truncated[-1].bytes)

    def mean(total: float, count: int) -> float | None:
        return total / count if count else None

    routing = []
    for bank in sorted(widths):
        visits = sorted(visit for (*key, visit) in loads if tuple(key) == bank)
        entropies = [load_entropy(loads[bank + (visit,)]) for visit in visits]
        routing.append({
            "region": bank[0], "index": bank[1], "num_experts": widths[bank], "visits": visits,
            "load_entropy": entropies,
            "normalized_load_entropy": [value / math.log(widths[bank]) if widths[bank] > 1 else None for value in entropies],
            "load_jaccard": [[multiset_jaccard(loads[bank + (i,)], loads[bank + (j,)]) for j in visits] for i in visits],
            "token_jaccard": [[1.0 if i == j else mean(*overlaps[bank + (min(i, j), max(i, j))]) for j in visits] for i in visits],
        })
    return {
        "format": "moeloop-diagnostics", "version": 1, "split": split, "seq_len": seq_len,
        "batch_size": batch_size, "max_batches": max_batches, "batches": batches,
        "complete_split": held.complete,
        "num_visits": depth, "input_tokens": valid_inputs, "manifest_fingerprint": manifest.fingerprint,
        "model_config": config.to_dict(), "bpb_unavailable": final["bpb_unavailable"], "skipped": skipped,
        "byte_table_sha256": final["byte_table_sha256"],
        "logit_lens": None if not whole_visits else [{"visit": visit, **loss.result()} for visit, loss in enumerate(lens)],
        "residual": None if not whole_visits else [{
            "visit": visit, "update_norm": mean(row["update"], valid_inputs), "state_norm": mean(row["state"], valid_inputs),
            "relative_update": mean(row["relative"], row["relative_tokens"]),
            "cosine_previous": mean(row["cosine"], row["cosine_tokens"]),
        } for visit, row in enumerate(residual)],
        "routing": routing,
        "effective_depth_from": "truncated_forward" if truncate else "logit_lens",
        "effective_depth": [{"num_visits": r + 1, **loss.result()} for r, loss in enumerate(truncated)],
    }


QUANTILES = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def sample_batches(manifest: PreparedManifest, *, split: str, seq_len: int, sequences: int, seed: int,
                   batch_size: int) -> list[tuple[dict[str, Tensor], Tensor]]:
    """The first ``sequences`` blocks of ``split``, packed in seed-shuffled document order, as CPU batches.

    ``document_ids`` are packing serials, which a document's continuation segments keep.
    """
    dataset = PackedDataset(manifest, split=split, seq_len=seq_len, shuffle=True, infinite=False, seed=seed)
    try:
        batches = list(islice(DataLoader(dataset, batch_size=batch_size, num_workers=0), sequences // batch_size))
    finally:
        dataset.close()
    if sum(len(labels) for _, labels in batches) != sequences:
        raise ValueError(f"need {sequences} blocks of {seq_len} tokens from {split!r} in batches of {batch_size}")
    return batches


def device_batch(inputs: dict[str, Tensor], labels: Tensor, device) -> tuple[dict[str, Tensor], Tensor, dict[str, Any]]:
    """One CPU batch as ``heldout_batches`` yields it: ``(inputs, labels, forward kwargs)`` on ``device``."""
    inputs = {name: value.to(device) for name, value in inputs.items()}
    return inputs, labels.to(device), dict(positions=inputs["positions"], router_mask=inputs["input_mask"],
        attention_mask=build_document_mask(inputs["document_ids"], inputs["input_mask"]))


def visit_states(core, inputs: dict[str, Tensor], kwargs: dict[str, Any], depth: int):
    """The forward's output, the states ``[entering visit 1, after visit 1, ..., after visit R]`` and the readout inputs."""
    with _record(core) as record:
        output = core(inputs["input"], **kwargs, num_visits=depth)
    cell_inputs, *_ = core.parse_inputs(inputs["input"], **kwargs, num_visits=depth)
    return output, torch.stack([record["inputs"][0], *record["outputs"]]), cell_inputs


def mixed_readout(core, states: Tensor, weights: Tensor, cell_inputs: dict[str, Any]) -> Tensor:
    """Logits of ``sum_s weights[s] states[s]`` through the suffix, the final norm and the head."""
    mixed = torch.einsum("s,s...->...", weights.to(states.dtype), states)
    return core.lm_head(core.readout(mixed, len(states) - 2, cell_inputs))


def scaled_readout(core, states: Tensor, control: Tensor, cell_inputs: dict[str, Any]) -> Tensor:
    """The terminal-only control: the terminal state times ``control[0]``, logits times ``control[1]``."""
    scale, inverse_temperature = control.to(states.dtype)
    return core.lm_head(core.readout(scale * states[-1], len(states) - 2, cell_inputs)) * inverse_temperature


def token_ce(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), reduction="none", ignore_index=-100)


def fit_readout(
    model, manifest: str | Path | PreparedManifest, *, seq_len: int, batch_size: int = 1,
    fit_split: str = "validation", report_split: str = "test", fit_sequences: int = 1024, seed: int = 0,
    epochs: int = 2, lr: float = 0.01, num_visits: int | None = None,
) -> dict[str, Any]:
    """Fit a mixed readout over the visit states of the frozen model, then report it on ``report_split``.

    States: the state entering visit 1 (the prefix output when ``initial_state="input"``
    without input injection), then the state after each of the R visits.

    * ``mixture``: ``softmax(a)`` over the R+1 states (R+1 logits, R free), summed and read
      out through the suffix, the final norm and the head; ``a`` starts at 0 (equal weights).
    * ``control``: the terminal state only, with a state scale and an inverse head
      temperature, the same R = 2 free scalars of a two-visit model; starts at (1, 1),
      the native readout.

    Data: the first ``fit_sequences`` blocks of ``fit_split`` (:func:`sample_batches`).
    Documents with an even packing serial form the fit half, odd ones the selection half;
    attention is document-local, so the halves never see each other. Each epoch takes one
    Adam step (``lr``) per batch on the fit half's mean CE, both heads together; after it
    the selection-half CE is measured (also at epoch 0, before any step). Each head keeps
    its parameters of lowest selection CE, and fitting stops when neither head improved
    or after ``epochs`` epochs. Then one pass over the whole ``report_split``
    (``heldout_batches``) reports, as ``_Loss`` rows: ``native`` (the model's own logits),
    ``first_visit`` (the readout after visit 1), ``terminal_only_mixture`` (the mixture at
    the one-hot terminal weights, which must equal ``native``), ``mixture`` and ``control``.

    ``per_token``, over the valid labels of ``report_split``: ``d`` = CE after visit 1
    minus native CE (nats), its quantiles (nearest rank), the share of the summed ``d``
    held by the 20% of tokens with the largest ``d``, and the mean ``d`` over five equal
    token groups ordered by the label's frequency in ``report_split``, rarest first.
    """
    manifest = manifest if isinstance(manifest, PreparedManifest) else PreparedManifest.load(manifest)
    core = getattr(model, "model", model)
    depth = core.config.num_visits if num_visits is None else num_visits
    if core.config.order != "stack" and core.config.num_cells != 1:
        raise ValueError("a mixed readout needs whole-visit states: order='stack' or one core cell")
    device = next(core.parameters()).device
    batches = sample_batches(manifest, split=fit_split, seq_len=seq_len, sequences=fit_sequences, seed=seed,
                             batch_size=batch_size)
    halves = [[(labels != -100) & (inputs["document_ids"] % 2 == parity) for inputs, labels in batches]
              for parity in (0, 1)]
    mixture = torch.zeros(depth + 1, device=device, requires_grad=True)
    control = torch.ones(2, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([mixture, control], lr=lr)

    def readouts(states, cell_inputs):
        return {"mixture": mixed_readout(core, states, mixture.softmax(0), cell_inputs),
                "control": scaled_readout(core, states, control, cell_inputs)}

    def selection_ce() -> dict[str, float]:
        sums = {"mixture": 0.0, "control": 0.0}
        with torch.no_grad():
            for (inputs, labels), half in zip(batches, halves[1]):
                inputs, labels, kwargs = device_batch(inputs, labels, device)
                _, states, cell_inputs = visit_states(core, inputs, kwargs, depth)
                for name, logits in readouts(states, cell_inputs).items():
                    sums[name] += float(token_ce(logits, labels)[half.flatten().to(device)].double().sum())
        tokens = sum(int(half.sum()) for half in halves[1])
        return {name: total / tokens for name, total in sums.items()}

    history, best = [], {}
    was_training = model.training
    model.eval()
    try:
        for epoch in range(epochs + 1):
            if epoch:
                for (inputs, labels), half in zip(batches, halves[0]):
                    if not half.any():
                        continue
                    inputs, labels, kwargs = device_batch(inputs, labels, device)
                    with torch.no_grad():
                        _, states, cell_inputs = visit_states(core, inputs, kwargs, depth)
                    fit = half.flatten().to(device)
                    loss = sum(token_ce(logits, labels)[fit].mean() for logits in readouts(states, cell_inputs).values())
                    mixture.grad, control.grad = torch.autograd.grad(loss, (mixture, control))
                    optimizer.step()
            losses = selection_ce()
            history.append({"epoch": epoch, **losses})
            improved = [name for name, value in losses.items() if name not in best or value < best[name][0]]
            for name in improved:
                best[name] = (losses[name], epoch, (mixture if name == "mixture" else control).detach().clone())
            if not improved:
                break
    finally:
        model.train(was_training)

    weights, scales = best["mixture"][2].softmax(0), best["control"][2]
    terminal = F.one_hot(torch.tensor(depth), depth + 1).to(device)
    deltas, targets = [], []
    with heldout_batches(model, manifest, split=report_split, seq_len=seq_len, batch_size=batch_size,
                         max_batches=None) as held:
        token_bytes, byte_table, bpb_unavailable = token_byte_tensor(manifest, held.device)
        rows = {name: _Loss(token_bytes, byte_table)
                for name in ("native", "first_visit", "terminal_only_mixture", "mixture", "control")}
        for inputs, labels, kwargs in held.batches:
            output, states, cell_inputs = visit_states(core, inputs, kwargs, depth)
            first = core.lm_head(core.readout(states[1], 0, cell_inputs))
            for name, logits in (("native", output.logits), ("first_visit", first),
                                 ("terminal_only_mixture", mixed_readout(core, states, terminal, cell_inputs)),
                                 ("mixture", mixed_readout(core, states, weights, cell_inputs)),
                                 ("control", scaled_readout(core, states, scales, cell_inputs))):
                rows[name].add(logits, labels)
            valid = labels.flatten() != -100
            deltas.append((token_ce(first, labels) - token_ce(output.logits, labels))[valid].double().cpu())
            targets.append(labels.flatten()[valid].cpu())
    delta, target = torch.cat(deltas), torch.cat(targets)
    ordered = delta.sort().values
    frequency = torch.bincount(target)[target]
    groups = delta[frequency.argsort(stable=True)].tensor_split(5)
    return {
        "fit_split": fit_split, "report_split": report_split, "fit_sequences": fit_sequences, "seed": seed,
        "seq_len": seq_len, "batch_size": batch_size, "epochs": epochs, "lr": lr, "num_visits": depth,
        "fit_tokens": sum(int(half.sum()) for half in halves[0]),
        "selection_tokens": sum(int(half.sum()) for half in halves[1]),
        "selection_ce": history, "bpb_unavailable": bpb_unavailable,
        "mixture": {"epoch": best["mixture"][1], "weights": weights.tolist(), **rows["mixture"].result()},
        "control": {"epoch": best["control"][1], "state_scale": float(scales[0]), "temperature": float(1 / scales[1]),
                    **rows["control"].result()},
        **{name: rows[name].result() for name in ("native", "first_visit", "terminal_only_mixture")},
        "per_token": {
            "tokens": len(delta),
            "quantiles": {str(q): float(ordered[round(q * (len(ordered) - 1))]) for q in QUANTILES},
            "top_20_share": float(ordered[len(ordered) - len(ordered) // 5:].sum() / delta.sum()),
            "frequency_quintiles": [float(group.mean()) for group in groups],
        },
    }


def source_config(run: str | Path) -> dict[str, Any]:
    """The source run's ``config.json``, rebuilt from its spec.json and request.json.

    Training writes that file only into its own ``$RUN_DIR`` (an MLflow artifact); this mirrors
    ``launch`` in ``scripts/recipe_experiments.py``, which takes the same keys of the spec and the
    same ``launch`` facts.
    """
    run = Path(run)
    spec = json.loads((run / "spec.json").read_text())
    request = json.loads((run / "request.json").read_text())
    config = {key: spec[key] for key in ("name", "model", "training", "optimizer", "inference", "run")}
    config["launch"] = {key: request[key] for key in
                        ("world_size", "local_batch_size", "gradient_accumulation_steps", "manifest_fingerprint")}
    return config


def write_run_json(name: str, value: dict[str, Any]) -> None:
    """Write ``$RUN_DIR/<name>`` when ``RUN_DIR`` is set (under ``scripts/run.py``)."""
    if os.environ.get("RUN_DIR"):
        content = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        (Path(os.environ["RUN_DIR"]) / name).write_text(content, encoding="utf-8")


def flat_metrics(result: dict[str, Any]) -> dict[str, float]:
    """The headline numbers of a diagnostics result as ``{name: number}``, without None values."""
    metrics = {}
    for row in result["logit_lens"] or ():
        metrics |= {f"lens_ce_{row['visit'] + 1}": row["ce"], f"lens_bpb_{row['visit'] + 1}": row.get("bpb")}
    for row in result["residual"] or ():
        metrics |= {f"update_norm_{row['visit'] + 1}": row["update_norm"],
                    f"relative_update_{row['visit'] + 1}": row["relative_update"]}
    readout = result.get("readout")
    if readout is not None:
        for name in ("native", "first_visit", "terminal_only_mixture", "mixture", "control"):
            metrics |= {f"{name}_ce": readout[name]["ce"], f"{name}_bpb": readout[name].get("bpb")}
        metrics |= {f"mixture_weight_{s}": weight for s, weight in enumerate(readout["mixture"]["weights"])}
        metrics |= {"control_state_scale": readout["control"]["state_scale"],
                    "control_temperature": readout["control"]["temperature"],
                    "top_20_share": readout["per_token"]["top_20_share"]}
        metrics |= {f"quantile_{q}": value for q, value in readout["per_token"]["quantiles"].items()}
        metrics |= {f"frequency_quintile_{i + 1}": value
                    for i, value in enumerate(readout["per_token"]["frequency_quintiles"])}
    return {name: value for name, value in metrics.items() if value is not None}


def plot_diagnostics(result: dict[str, Any], path: str | Path) -> Path:
    """Write six small panels (format from the suffix); needs matplotlib.

    The x axis is the 1-based visit, so r on the truncation curve and visit
    r-1 of the logit lens share a point; an overlap point at visit v compares
    visits v-1 and v. At most eight banks are drawn per routing panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    metric = "bpb" if result["byte_table_sha256"] else "ce"
    banks = [bank for bank in result["routing"] if len(bank["visits"]) > 1][:len(colors)]

    def per_bank(values):
        return [(f"{bank['region']} {bank['index']}", values(bank)) for bank in banks]

    def consecutive(matrix):
        return lambda bank: [(bank["visits"][k] + 1, bank[matrix][k - 1][k]) for k in range(1, len(bank["visits"]))]

    panels = (
        (f"Loss ({metric.upper()}): logit lens and truncated depth",
         [("truncated forward" if result["effective_depth_from"] == "truncated_forward" else "logit lens = truncated forward",
           [(row["num_visits"], row[metric]) for row in result["effective_depth"]]),
          ("logit lens", [(row["visit"] + 1, row[metric]) for row in result["logit_lens"] or ()]
           if result["effective_depth_from"] == "truncated_forward" else [])]),
        ("Relative core update |u|/|x|",
         [("core update", [(row["visit"] + 1, row["relative_update"]) for row in result["residual"] or ()])]),
        ("Cosine of consecutive core updates",
         [("core update", [(row["visit"] + 1, row["cosine_previous"]) for row in result["residual"] or ()])]),
        ("Normalized expert-load entropy",
         per_bank(lambda bank: [(visit + 1, value) for visit, value in zip(bank["visits"], bank["normalized_load_entropy"])])),
        ("Token top-k Jaccard, consecutive visits", per_bank(consecutive("token_jaccard"))),
        ("Expert-load Jaccard, consecutive visits", per_bank(consecutive("load_jaccard"))),
    )
    for axis, (title, series) in zip(axes.flat, panels):
        drawn = 0
        for (label, points), color in zip(series, colors):
            points = [(x, y) for x, y in points if y is not None]
            if points:
                axis.plot(*zip(*points), color=color, linewidth=2, marker="o", markersize=5, label=label)
                drawn += 1
        axis.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
        axis.set_xlabel("visit (1-based)", fontsize=9, color="#52514e")
        axis.set_xlim(0.5, result["num_visits"] + 0.5)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_facecolor("#fcfcfb")
        axis.grid(True, color="#e4e3df", linewidth=0.8)
        axis.tick_params(colors="#52514e", labelsize=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
        if drawn:
            axis.legend(fontsize=8, frameon=False)
        else:
            axis.text(0.5, 0.5, "not available", ha="center", va="center", color="#52514e", transform=axis.transAxes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor="#fcfcfb")
    plt.close(figure)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="local exported HF directory, <run>/hf for RUN_DIR")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--num-visits", type=int, help="K; default is the model's num_visits")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="new JSON result file, without overwrite")
    parser.add_argument("--plot", type=Path, help="new .svg or .png figure (needs matplotlib)")
    parser.add_argument("--fit-readout", action="store_true", help="add the mixed readout (fit_readout) at --seq-len")
    parser.add_argument("--fit-split", default="validation")
    parser.add_argument("--report-split", default="test")
    parser.add_argument("--fit-sequences", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0, help="document shuffle of the fit sequences")
    parser.add_argument("--readout-epochs", type=int, default=2)
    parser.add_argument("--readout-lr", type=float, default=0.01)
    args = parser.parse_args(argv)
    for path in (args.output, args.plot):
        if path is not None and path.exists():
            parser.error(f"output already exists: {path}")
    if args.plot is not None and args.plot.suffix not in (".svg", ".png"):
        parser.error("--plot must end in .svg or .png")
    if os.environ.get("RUN_DIR"):
        provenance = json.loads((args.model / "export_provenance.json").read_text())
        analysis = {"model": str(args.model.resolve()), "step": int(provenance["checkpoint"]["name"].removeprefix("step-")),
                    "split": args.split, "seq_len": args.seq_len, "batch_size": args.batch_size,
                    "max_batches": args.max_batches, "num_visits": args.num_visits, "fit_readout": args.fit_readout}
        if args.fit_readout:
            analysis |= {"seed": args.seed, "fit_split": args.fit_split, "report_split": args.report_split,
                         "fit_sequences": args.fit_sequences, "readout_epochs": args.readout_epochs,
                         "readout_lr": args.readout_lr}
        write_run_json("config.json", {**source_config(args.model.resolve().parent), "analysis": analysis})
    manifest = PreparedManifest.load(args.manifest)
    model = load_export(args.model, manifest, args.device)
    result = diagnose(model, manifest, split=args.split, seq_len=args.seq_len, batch_size=args.batch_size,
                      max_batches=args.max_batches, num_visits=args.num_visits)
    if args.fit_readout:
        result["readout"] = fit_readout(model, manifest, seq_len=args.seq_len, batch_size=args.batch_size,
                                        fit_split=args.fit_split, report_split=args.report_split,
                                        fit_sequences=args.fit_sequences, seed=args.seed, epochs=args.readout_epochs,
                                        lr=args.readout_lr, num_visits=args.num_visits)
    content = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    if args.plot is not None:
        plot_diagnostics(result, args.plot)
    write_run_json("metrics.json", flat_metrics(result))
    print(content, end="")


if __name__ == "__main__":
    main()
