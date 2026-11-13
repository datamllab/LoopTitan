"""Conformance battery for any recipe: ``check_recipe(spec)``.

The checks reuse the oracles of tests/model and tests/hf:

- ``ownership``: no parameter is registered twice or shares storage;
- ``shared_gradients``: each shared gradient equals its per-call sum;
- ``cost_calls``: ``estimate_cost`` call and parameter counts equal one forward;
- ``cache``: chunked cached decoding equals full recomputation, and its KV
  slots/bytes equal the cost model (not applicable to uncached recipes);
- ``hf_roundtrip``: ``save_pretrained``/``from_pretrained`` give identical logits;
- ``identity``: a JSON-rebuilt spec keeps the continuation identity and weights.

The model is always ``RecursiveLM(spec.model)`` on the spec's own geometry,
so pass a small instantiation (recipe constructors take geometry keywords).
The battery runs the reference CPU path in float32; it says nothing about
quality, training stability, native GPU kernels or the on-disk export CLI.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import fields
import json
from typing import Any, Iterator

import torch
from torch import nn

from .cache import DecodeCache
from .costs import estimate_cost
from .model import Attention, Cell, RecursiveLM
from .moe import Router
from .recipe_spec import RecipeSpec
from .training import continuation_fingerprint, objective, spec_identity


class ConformanceError(AssertionError):
    """At least one check failed; ``report`` holds every check's result."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        failed = [f"{name}: {result['detail']}" for name, result in report["checks"].items()
                  if result["status"] == "fail"]
        super().__init__(f"recipe {report['recipe']!r} failed conformance:\n  " + "\n  ".join(failed))


def check_recipe(spec: RecipeSpec, *, seq_len: int = 8, seed: int = 0) -> dict[str, Any]:
    """Run every check in ``CHECKS``; return the report or raise ConformanceError.

    The report is ``{"recipe", "seq_len", "seed", "checks": {name: result}}``;
    each result has ``status`` (pass, fail or not_applicable) and ``detail``.
    Each check builds its own model after ``torch.manual_seed(seed)`` and the
    caller's RNG state is restored. A check that raises counts as failed, and
    every check runs, so one report lists every broken contract.
    """
    if not isinstance(spec, RecipeSpec):
        raise TypeError("check_recipe takes a RecipeSpec")
    if type(seq_len) is not int or not 3 <= seq_len <= spec.model.max_seq_len:
        raise ValueError("seq_len must be an integer in [3, max_seq_len] (the cache check uses three chunks)")
    checks = {}
    with torch.random.fork_rng(), torch.enable_grad():
        for name, check in _BATTERY.items():
            try:
                checks[name] = check(spec, seq_len, seed)
            except Exception as error:  # a crash is a failed contract
                checks[name] = {"status": "fail", "detail": f"{type(error).__name__}: {error}"}
    report = {"recipe": spec.name, "seq_len": seq_len, "seed": seed, "checks": checks}
    if any(result["status"] == "fail" for result in checks.values()):
        raise ConformanceError(report)
    return report


def _build(spec: RecipeSpec, seed: int) -> RecursiveLM:
    torch.manual_seed(seed)
    return RecursiveLM(spec.model)


def _tokens(spec: RecipeSpec, seq_len: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(spec.model.vocab_size, (2, seq_len), generator=generator)


def _training_forward(model: RecursiveLM, spec: RecipeSpec, ids: torch.Tensor):
    """One training forward at the recipe's fixed (or nominal sampled) depth."""
    return model(ids, num_visits=spec.training.num_visits, grad_visits=spec.training.grad_visits)


def _ownership(spec, seq_len, seed):
    """Unique owners: no parameter under two names, no two sharing storage."""
    model = _build(spec, seed)
    names = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        names[id(parameter)].append(name)
    aliases = [group for group in names.values() if len(group) > 1]
    if aliases:
        raise AssertionError(f"parameters registered under several names: {aliases}")
    storages = defaultdict(list)
    canonical = list(model.canonical_parameters())
    for name, parameter in canonical:
        if parameter.numel():
            storages[parameter.untyped_storage().data_ptr()].append(name)
    shared = [group for group in storages.values() if len(group) > 1]
    if shared:
        raise AssertionError(f"distinct parameters share storage: {shared}")
    return {"status": "pass", "detail": f"{len(canonical)} canonical parameters, one owner each"}


@contextmanager
def _per_call_copies(model: RecursiveLM, copies: dict[str, list[nn.Parameter]]) -> Iterator[None]:
    """Explicit unrolling by hooks: every call reads its own leaf copy.

    test_semantics.py unrolls a shared cell into independent cells with copied
    weights. Recipes with EMA memory, per-visit injection or readouts cannot be
    rewritten as such a stack, so each call of a Cell or top-level module gets
    a fresh copy (equal values) of every parameter it reaches instead.
    """
    owners = {id(parameter): name for name, parameter in model.canonical_parameters()}
    units = [module for module in model.modules() if isinstance(module, Cell)]
    units += [module for module in model.children() if not isinstance(module, nn.ModuleList)]
    saved: list[list[tuple[nn.Module, str, nn.Parameter]]] = []

    def swap(module, args):
        originals = []
        for owner in module.modules():
            for key, parameter in owner._parameters.items():
                if parameter is not None:
                    copy = nn.Parameter(parameter.detach().clone(), parameter.requires_grad)
                    copies[owners[id(parameter)]].append(copy)
                    originals.append((owner, key, parameter))
                    owner._parameters[key] = copy
        saved.append(originals)

    def restore(module, args, output):
        for owner, key, parameter in saved.pop():
            owner._parameters[key] = parameter

    handles = [hook for unit in units for hook in
               (unit.register_forward_pre_hook(swap), unit.register_forward_hook(restore))]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _shared_gradients(spec, seq_len, seed):
    """The shared gradient equals the sum of per-call copy gradients.

    This catches gradient handling that depends on parameter identity, such as
    parameter hooks or custom backward functions that overwrite rather than
    accumulate. It cannot catch an error both passes share (the copies run the
    same code); that is the independent unrolled-stack oracle of
    tests/model/test_semantics.py. Every parameter must be used inside a module
    call (a Cell or a top-level child); one read outside such a call gets no
    copy and is reported as a mismatch.
    """
    model = _build(spec, seed).train()
    ids = _tokens(spec, seq_len, seed)
    labels = ids.roll(-1, 1)

    def loss() -> torch.Tensor:
        torch.manual_seed(seed)  # identical random initial latents in both passes
        value, _ = objective(_training_forward(model, spec, ids), labels,
                             torch.tensor(labels.numel()), spec.training, head=model.lm_head)
        return value

    loss().backward()
    canonical = list(model.canonical_parameters())
    shared = {name: parameter.grad for name, parameter in canonical}
    model.zero_grad(set_to_none=True)
    copies: dict[str, list[nn.Parameter]] = defaultdict(list)
    with _per_call_copies(model, copies):
        loss().backward()
    for name, parameter in canonical:
        zero = torch.zeros_like(parameter)
        total = sum((copy.grad for copy in copies[name] if copy.grad is not None), zero)
        expected = zero if shared[name] is None else shared[name]
        torch.testing.assert_close(expected, total, atol=2e-6, rtol=2e-5,
                                   msg=lambda message, name=name: f"{name}: {message}")
    reused = sum(sum(copy.grad is not None for copy in copies[name]) > 1 for name, _ in canonical)
    return {"status": "pass", "reused_parameters": reused,
            "detail": f"{reused} parameters reused across calls; gradients equal per-call sums"}


def _cost_calls(spec, seq_len, seed):
    """estimate_cost call counts and unique parameters equal one executed forward.

    ``head`` calls are not compared: lm_head module calls count loss chunks
    (and checkpoint replays), not the cost model's logical readouts.
    """
    model = _build(spec, seed).train()
    cost = estimate_cost(spec, seq_len=seq_len)
    calls = cost["calls"].get("nominal", cost["calls"])  # sampled depth: nominal forward
    kinds = {"forward_cells": Cell, "attention": Attention, "router": Router}
    counted = [(key, module) for module in model.modules() for key, kind in kinds.items() if isinstance(module, kind)]
    keys = list(kinds)
    if model.exit_gate is not None:
        keys.append("exit_gate")
        counted.append(("exit_gate", model.exit_gate))
    executed: Counter[str] = Counter()
    handles = [module.register_forward_hook(lambda *_, key=key: executed.update([key]))
               for key, module in counted]
    ids = _tokens(spec, seq_len, seed)
    try:
        output = _training_forward(model, spec, ids)
    finally:
        for handle in handles:
            handle.remove()
    keys.append("expert_assignments_per_input_token")
    executed["expert_assignments_per_input_token"] = output.router_counts.sum().item() / ids.numel()
    estimated = {key: calls[key] for key in keys}
    actual = {key: executed[key] for key in keys}
    if actual != estimated:
        raise AssertionError(f"estimate_cost counts {estimated} but one forward executed {actual}")
    unique = sum(parameter.numel() for _, parameter in model.canonical_parameters())
    if unique != cost["parameters"]["unique_total"]:
        raise AssertionError(f"estimate_cost counts {cost['parameters']['unique_total']} unique parameters, model has {unique}")
    depth = "nominal-depth" if "nominal" in cost["calls"] else "fixed-depth"
    return {"status": "pass", "detail": f"{depth} calls {actual} (head calls not compared), {unique} unique parameters"}


def _cache(spec, seq_len, seed):
    """Chunked cached decoding equals full recomputation; KV matches the cost model."""
    if spec.inference.get("cache") == "uncached":
        reason = next((text for text in spec.adaptations if "uncached" in text), "")
        return {"status": "not_applicable", "detail": f"spec.inference['cache'] is 'uncached'. {reason}".strip()}
    model = _build(spec, seed).eval()
    depth = spec.inference.get("num_visits", spec.model.num_visits)
    ids = _tokens(spec, seq_len, seed)
    cache = DecodeCache(depth)
    cuts = (0, seq_len // 2, seq_len - 1, seq_len)
    with torch.no_grad():
        full = model(ids, num_visits=depth).logits
        chunks = [model(ids[:, start:stop], num_visits=depth, cache=cache).logits
                  for start, stop in zip(cuts, cuts[1:])]
    torch.testing.assert_close(torch.cat(chunks, 1), full)  # float32 defaults: atol 1e-5, rtol 1.3e-6
    kv = estimate_cost(spec, seq_len=seq_len, batch_size=ids.size(0), bytes_per_element=full.element_size())["kv"]
    actual = sum(tensor.numel() * tensor.element_size() for pair in cache.slots.values() for tensor in pair)
    if (len(cache.slots), actual) != (kv["logical_slots"], kv["bytes_at_context"]):
        raise AssertionError(f"cache holds {len(cache.slots)} slots/{actual} bytes, estimate_cost "
                             f"{kv['logical_slots']} slots/{kv['bytes_at_context']} bytes")
    return {"status": "pass", "detail": f"{len(cache.slots)} slots, three chunks equal full prefix"}


def _hf_roundtrip(spec, seq_len, seed):
    """``save_pretrained`` then ``from_pretrained`` (the export path) keeps the model.

    Weights and keys must survive bitwise; logits use float32 default tolerances
    (atol 1e-5, rtol 1.3e-6) because file-loaded weights can take another kernel path.
    """
    from tempfile import TemporaryDirectory
    from .hf import MoeloopConfig, MoeloopForCausalLM

    model = _build(spec, seed).eval()
    exported = MoeloopForCausalLM(MoeloopConfig(spec.model.to_dict()))
    exported.model.load_state_dict(model.state_dict(), strict=True)
    # Cleanup may fail on network filesystems while the loaded weights are memory-mapped.
    with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        exported.save_pretrained(directory, safe_serialization=True)
        restored, info = MoeloopForCausalLM.from_pretrained(directory, local_files_only=True, output_loading_info=True)
    state = restored.model.state_dict()
    changed = [key for key, value in model.state_dict().items() if not torch.equal(state[key], value)]
    if changed or any(info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise AssertionError(f"from_pretrained did not restore every weight exactly: {changed or info}")
    ids = _tokens(spec, seq_len, seed)
    with torch.no_grad():
        torch.testing.assert_close(restored.eval()(ids, use_cache=False).logits, model(ids).logits)
    return {"status": "pass", "detail": "save_pretrained/from_pretrained: bitwise weights, equal logits"}


def _spec_identity(spec: RecipeSpec) -> dict[str, Any]:
    """``spec_identity`` with the values trainer_config and experiment take from ``spec``."""
    return spec_identity(model=spec.model, recipe=spec.training, recipe_metadata=spec.to_dict(),
                         expert_lr_multiplier=spec.optimizer.get("expert_lr_multiplier", 1.0),
                         fork=spec.run.get("fork", {}), ema_decay=spec.run.get("ema_decay"))


def _identity(spec, seq_len, seed):
    """A JSON-reconstructed spec has the same identity and the same initial weights.

    The fingerprint comparison mainly verifies that the spec-derived identity
    fields round-trip through JSON.
    """
    again = RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    changed = [entry.name for entry in fields(RecipeSpec) if getattr(again, entry.name) != getattr(spec, entry.name)]
    if changed:
        raise AssertionError(f"JSON reconstruction changes {changed}; a resumed run would not match this spec")
    first, second = (continuation_fingerprint(_spec_identity(value)) for value in (spec, again))
    if first != second:
        raise AssertionError(f"continuation fingerprints differ: {first} != {second}")
    states = [_build(value, seed).state_dict() for value in (spec, again)]
    if states[0].keys() != states[1].keys() or any(not torch.equal(states[0][k], states[1][k]) for k in states[0]):
        raise AssertionError("two constructions with the same seed have different initial weights")
    return {"status": "pass", "detail": f"fingerprint {first[:16]}"}


_BATTERY = {"ownership": _ownership, "shared_gradients": _shared_gradients, "cost_calls": _cost_calls,
            "cache": _cache, "hf_roundtrip": _hf_roundtrip, "identity": _identity}
CHECKS = tuple(_BATTERY)
