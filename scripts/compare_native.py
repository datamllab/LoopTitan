#!/usr/bin/env python3
"""Compare native grouped/Flex operators with the plain PyTorch model in BF16.

Requires a CUDA GPU, Torch 2.14.0 and the pinned TorchTitan 086bf6c source.
Checks shared and injected-suffix execution, including all auxiliary gradients.
BF16 backward reductions are compared by vector error and tensor-peak scale;
relative error at a nearly cancelled individual coordinate is not meaningful.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import NamedTuple

import torch

from moeloop.model import ModelConfig, RecursiveLM
from moeloop.moe import Router
from moeloop.titan import TitanModel, canonical_model_state, prepare_batch, require_torchtitan_pin
from moeloop.training import TrainingRecipe, objective


TOLERANCES = {
    "logits": {"atol": 0.01, "rtol": 0.03},
    "loss": {"atol": 0.005, "rtol": 0.01},
    "gradient": {"relative_l2": 0.03, "peak_eps_multiplier": 2.0, "absolute_floor": 1e-6},
    "global_gradient_relative_l2": 0.03,
}


class RouterCall(NamedTuple):
    owner: str
    valid_mask: torch.Tensor
    ids: torch.Tensor


@contextmanager
def routing_trace(model):
    """Observe each logical call without changing its outputs or gradients."""
    calls = []

    def capture(owner):
        def hook(module, args, output):
            mask = args[1] if len(args) > 1 else None
            if mask is None:
                mask = torch.ones(output.ids.shape[:-1], dtype=torch.bool, device=output.ids.device)
            calls.append(RouterCall(owner, mask.detach().cpu().clone(),
                                    output.ids.detach()[mask].cpu()))
        return hook

    handles = [module.register_forward_hook(capture(name))
               for name, module in model.named_modules() if isinstance(module, Router)]
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def compare_routing(actual, reference):
    """Compare valid token expert sets in invocation order, before pooling."""
    failures, calls = [], []
    if not actual or not reference:
        failures.append("no logical Router calls were captured")
    if len(actual) != len(reference):
        failures.append("logical Router call count differs")
    for index, (observed, expected) in enumerate(zip(actual, reference)):
        # Top-k order is arbitrary; token order and call identity are not.
        ids = observed.ids.sort(dim=-1).values
        expected_ids = expected.ids.sort(dim=-1).values
        equal = (observed.owner == expected.owner
                 and torch.equal(observed.valid_mask, expected.valid_mask)
                 and torch.equal(ids, expected_ids))
        if not equal:
            failures.append(f"routing call {index} ({expected.owner}) differs")
        calls.append({
            "call": index, "router": expected.owner,
            "valid_tokens": expected_ids.shape[0], "top_k": expected_ids.shape[-1],
            "expert_sets_equal": equal,
            "native_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            "reference_ids_sha256": hashlib.sha256(expected_ids.numpy().tobytes()).hexdigest(),
        })
    return {"equal": not failures, "logical_calls": len(reference),
            "valid_token_expert_assignments": sum(call.ids.numel() for call in reference),
            "calls": calls, "failures": failures}


def errors(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual, reference = actual.detach().float(), reference.detach().float()
    difference = actual - reference
    norm = float(torch.linalg.vector_norm(difference))
    expected = float(torch.linalg.vector_norm(reference))
    return {
        "max_absolute": float(difference.abs().max()),
        "rms_absolute": float(difference.square().mean().sqrt()),
        "relative_l2": norm / max(expected, 1e-20),
        "reference_max_absolute": float(reference.abs().max()),
        "reference_l2": expected,
    }


def run_case(cfg: ModelConfig, grad_visits: int | None = None) -> dict:
    torch.manual_seed(71)
    reference = RecursiveLM(cfg).cuda().bfloat16().train()
    native = TitanModel(
        TitanModel.Config(core=cfg, native_experts=True, flex_attention=True)
    ).cuda().bfloat16().train()
    reference_state = reference.state_dict()
    mapped = canonical_model_state(dict(native.named_parameters()))
    if set(mapped) != set(reference_state):
        raise AssertionError("canonical physical parameter mismatch")
    with torch.no_grad():
        for key, param in mapped.items():
            param.copy_(reference_state[key])

    tokens = (torch.arange(64, device="cuda").reshape(2, 32) * 7 + 3) % cfg.vocab_size
    valid = torch.arange(32, device="cuda")[None, :] < torch.tensor(
        [27, 23], device="cuda"
    )[:, None]
    documents = torch.zeros_like(tokens)
    documents[0, 9:] = 1
    documents[1, 13:] = 1
    positions = torch.arange(32, device="cuda")[None, :].expand(2, -1).clone()
    positions[0, 9:] -= 9
    positions[1, 13:] -= 13
    positions.masked_fill_(~valid, 0)
    labels = tokens.roll(-1, 1).clone()
    labels.masked_fill_(~valid, -100)
    labels[0, 8] = labels[0, 26] = labels[1, 12] = labels[1, 22] = -100
    labels[:, -1] = -100
    inputs = {
        "input": tokens,
        "input_mask": valid,
        "positions": positions,
        "document_ids": documents,
    }
    n_tokens, n_labels, n_kwargs = prepare_batch(inputs, labels, flex_attention=True)
    r_tokens, r_labels, r_kwargs = prepare_batch(inputs, labels, flex_attention=False)
    recipe = TrainingRecipe(
        num_visits=cfg.num_visits,
        grad_visits=grad_visits,
        balance_coeff=0.17,
        z_loss_coeff=0.03,
    )
    with routing_trace(native.core) as native_routes, routing_trace(reference) as reference_routes:
        native_out = native(n_tokens, grad_visits=grad_visits, **n_kwargs)
        expected_out = reference(r_tokens, grad_visits=grad_visits, **r_kwargs)
    routing = compare_routing(native_routes, reference_routes)
    denominator = (labels != -100).sum()
    native_loss, _ = objective(native_out, n_labels, denominator, recipe)
    expected_loss, _ = objective(expected_out, r_labels, denominator, recipe)
    report = {
        "model_config": cfg.to_dict(),
        "grad_visits": grad_visits,
        "valid_inputs": int(valid.sum()),
        "valid_labels": int(denominator),
        "document_count": 4,
        "padding_positions": int((~valid).sum()),
        "logits": errors(native_out.logits, expected_out.logits),
        "loss": errors(native_loss, expected_loss),
        "balance_sum": errors(native_out.balance_sum, expected_out.balance_sum),
        "z_sum": errors(native_out.z_sum, expected_out.z_sum),
        "native_loss": float(native_loss.detach()),
        "reference_loss": float(expected_loss.detach()),
    }
    failures = list(routing["failures"])
    for label, actual, expected, tolerance in (
        ("logits", native_out.logits, expected_out.logits, TOLERANCES["logits"]),
        ("loss", native_loss, expected_loss, TOLERANCES["loss"]),
    ):
        try:
            torch.testing.assert_close(actual, expected, **tolerance)
        except AssertionError as error:
            failures.append(label + ": " + str(error))
    if native_out.aux_calls != expected_out.aux_calls:
        failures.append("auxiliary call count differs")
    if not torch.equal(native_out.router_counts, expected_out.router_counts):
        failures.append("valid-input router assignment counts differ")

    native_loss.backward()
    expected_loss.backward()
    gradients = {}
    difference_squared = reference_squared = 0.0
    reference_params = dict(reference.named_parameters())
    for key, param in mapped.items():
        expected = reference_params[key]
        if param.grad is None or expected.grad is None:
            if param.grad is not None or expected.grad is not None:
                failures.append("missing gradient: " + key)
            gradients[key] = {"none": param.grad is None and expected.grad is None}
            continue
        if not torch.isfinite(param.grad).all() or not torch.isfinite(expected.grad).all():
            failures.append("nonfinite gradient: " + key)
        error = errors(param.grad, expected.grad)
        # Independent BF16 reduction orders can differ by a few rounding units.
        # Bound both the complete vector and its worst coordinate; do not use
        # coordinate-relative error to judge values near cancellation.
        tolerance = TOLERANCES["gradient"]
        peak_bound = (tolerance["peak_eps_multiplier"] * torch.finfo(torch.bfloat16).eps
                      * error["reference_max_absolute"] + tolerance["absolute_floor"])
        gradients[key] = {**error, "max_absolute_allowed": peak_bound}
        difference_squared += float((param.grad.float() - expected.grad.float()).square().sum())
        reference_squared += float(expected.grad.float().square().sum())
        if error["relative_l2"] > tolerance["relative_l2"] or error["max_absolute"] > peak_bound:
            failures.append("gradient " + key + ": vector or BF16 peak-scale error exceeds its bound")
    relative = (difference_squared / max(reference_squared, 1e-40)) ** 0.5
    if relative > TOLERANCES["global_gradient_relative_l2"]:
        failures.append("global gradient relative L2 threshold exceeded")
    report.update({
        "status": "failed" if failures else "passed",
        "physical_parameter_tensors": len(mapped),
        "auxiliary_calls": native_out.aux_calls,
        "router_assignment_counts_equal": torch.equal(native_out.router_counts, expected_out.router_counts),
        "router_assignments_equal": routing["equal"],
        "routing": routing,
        "global_gradient_relative_l2": relative,
        "gradient_errors": gradients,
        "failures": failures,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    require_torchtitan_pin()
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg = ModelConfig(
        vocab_size=64,
        dim=64,
        hidden_dim=128,
        num_heads=4,
        num_kv_heads=2,
        num_experts=4,
        top_k=2,
        num_cells=2,
        num_visits=2,
        max_seq_len=32,
        qk_norm=True,
    )
    cases = {
        "shared": run_case(cfg),
        "injected_suffix": run_case(
            replace(
                cfg,
                input_injection="linear",
                initial_state="zero",
                prefix_cells=1,
                suffix_cells=1,
                num_visits=3,
            ),
            grad_visits=1,
        ),
    }
    report = {
        "schema_version": 2,
        "status": (
            "passed" if all(case["status"] == "passed" for case in cases.values()) else "failed"
        ),
        "scope": (
            "same canonical BF16 parameters; CUDA native grouped experts and "
            "document-masked FlexAttention versus ordinary PyTorch expert loop "
            "and SDPA; all gradients including balance/z objectives"
        ),
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "dtype": "bfloat16",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "routing_scope": "valid token expert sets at every logical Router call, including no-grad visits; top-k order ignored",
        "tolerances_predeclared": TOLERANCES,
        "cases": cases,
        "elapsed_seconds": time.perf_counter() - started,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    for name, case in cases.items():
        print(name, case["status"], case["global_gradient_relative_l2"], case["failures"])
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
