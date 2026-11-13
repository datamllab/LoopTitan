"""Inspectable owners, fixed- and sampled-depth matrix estimates, and logical inference KV.

These are arithmetic estimates, not measured hardware work. In particular the
default attention convention counts dense S-by-S matmuls. An explicit observed
causal pair count gives a separate logical document-mask estimate; neither is
measured kernel work. Dispatch, elementwise and optimizer work remain unknown.
Sampled depth and tail gradients give expectations and quantiles over the
declared depth distribution, never a run's realized depths.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from .config import ModelConfig, TrainingRecipe
from .recipe_spec import RecipeSpec


FORMULA_VERSION = "fixed_terminal_matrix_v1"
DEPTH_QUANTILES = {"p50": 0.5, "p95": 0.95}
# Standard-normal grid for the lognormal rate integral: the trapezoidal rule on
# a uniform grid converges geometrically for smooth Gaussian integrands.
NORMAL_STEP = 0.01
NORMAL_GRID = [index * NORMAL_STEP for index in range(-1200, 1201)]
# Every explanatory text of the report. Keys of the ``unknown`` section are its own keys.
NOTES = {
    "parameters": "unique_non_embedding excludes both independently owned input embedding and output head; all other owners counted once.",
    "trainable": "Current unfrozen model default; an externally frozen model needs actual parameter inspection.",
    "terminal_head": "One terminal logical projection; chunked native head kernel-call count is runtime dependent.",
    "expected_head": "One logical projection per training visit; fixed-depth inference uses only the terminal head. Chunk count is runtime dependent.",
    "dense_attention": "Two dense S-by-S matrix products; no causal-triangle or document-mask discount.",
    "document_attention": "Logical within-document causal query/key pairs; not measured masked-kernel work.",
    "full_backward": "Two backward matrix products per forward matrix product; full BPTT.",
    # Conservative convention: with a gradient tail, the first differentiated visit's linear-injection A h has a
    # detached input and needs no input gradient; backward still counts it (2 d^2 FLOPs per position too many).
    "tail_backward": ("Two backward matrix products per differentiated forward product; prefix, suffix, head and "
                      "the last min(grad_visits, k) core visits differentiate."),
    "expected_recompute": "Expected CE uses nonreentrant head/CE checkpointing; one additional head matrix forward per visit.",
    "depth_quantiles": ("Training FLOPs per padded position of one pass at the p50/p95 depth of the declared rule, "
                        "per update or microbatch; not a run total."),
    "kv": ("K+V per logical call at the declared fixed inference depth; parameter sharing never merges cache slots. "
           "Excludes allocator overhead."),
    "non_matmul_flops": "Norm/RoPE/softmax/top-k/CE/activations/residuals/dispatch are not counted.",
    "exit_non_matmul_flops": "Exit sigmoid, survival, entropy and loss weighting are also excluded.",
    "optimizer_flops": "AdamW and distributed optimizer work are not estimated.",
    "measured_flops": "Native kernels, masks, padding and hardware need runtime measurement.",
    "runtime_memory": "Activation, allocator, optimizer-state and communication memory are not inferred from KV.",
    "actual_tokens": "Supply run counts for valid input/labels and document masking; this report uses padded positions.",
    "input_injection": ("Decay softplus/exp, channel products and the OrthoInj projection "
                        "(two length-d reductions per position and visit) are non-matrix work, not included in FLOPs."),
    "ema_state": "Forward-local EMA updates, normalization and reinjection are non-matrix work, not included in FLOPs.",
    "inference_cache": "KV is a logical geometry budget; the selected uncached path does not establish cache allocation or cache parity.",
    "training_path": ("Training estimates cover fixed depth at the model's num_visits, sampled depth, and tail gradients "
                      "with input injection, under the declared objective; this path requires actual execution counts."),
    "training_depth": ("Training FLOPs are expectations over the declared depth distribution, with p50/p95 passes in "
                       "flops.training_depth; calls show one nominal-depth forward. Realized depths and work need run records."),
    "nominal_depth": "Nominal depth illustrates one fixed forward, not the sampled process's expectation, bound or observed work.",
    "recompute_flops": "Activation-checkpoint policy was not specified.",
}


def depth_probability(recipe: TrainingRecipe, visits: int) -> float:
    """Probability that one training pass runs ``visits`` core visits.

    ``uniform_update_v1`` draws one of ``sampled_visits`` per update, or always
    uses ``num_visits``. ``poisson_lognormal_micro_v1`` draws, per microbatch,
    ``1 + Poisson(rate)`` with ``log rate ~ Normal(log(mean - 1) - log_std^2/2,
    log_std)``, as ``latent.sampled_depth`` does. The rate integral uses the
    trapezoidal rule over +-12 standard deviations with spacing 0.01; it is
    accurate while ``log_std * sqrt(visits)`` stays well below 100.
    """
    if recipe.depth_rule != "poisson_lognormal_micro_v1":
        depths = recipe.sampled_visits or (recipe.num_visits,)
        return depths.count(visits) / len(depths)
    if visits < 1:
        return 0.0
    n, sigma = visits - 1, recipe.depth_log_std
    location = math.log(recipe.depth_mean - 1) - 0.5 * sigma * sigma
    total = 0.0
    for z in NORMAL_GRID:
        log_rate = location + sigma * z
        total += math.exp(-0.5 * z * z + n * log_rate - math.exp(log_rate) - math.lgamma(n + 1))
    return total * NORMAL_STEP / math.sqrt(2 * math.pi)


def training_depth(recipe: TrainingRecipe) -> dict[str, Any]:
    """Expected visits k, expected differentiated visits min(grad_visits, k), quantile depths.

    The Poisson-lognormal mean is exact (``depth_mean``); its expected tail
    and quantiles sum ``depth_probability``. A finite depth list gives the
    exact ``ceil(level * n)``-th sorted depth.
    """
    sampled_per = ("microbatch" if recipe.depth_rule == "poisson_lognormal_micro_v1"
                   else "update" if recipe.sampled_visits else None)
    depths = recipe.sampled_visits or (recipe.num_visits,)
    expected = float(recipe.depth_mean) if sampled_per == "microbatch" else sum(depths) / len(depths)
    tail = recipe.grad_visits
    if tail is None:
        expected_tail = expected
    else:
        below = [depth_probability(recipe, visits) for visits in range(1, tail)]
        expected_tail = sum(visits * p for visits, p in enumerate(below, 1)) + tail * (1 - sum(below))
    if sampled_per == "microbatch":
        quantiles, visits, cumulative = {}, 0, 0.0
        for name, level in DEPTH_QUANTILES.items():
            while cumulative < level:
                visits += 1
                cumulative += depth_probability(recipe, visits)
            quantiles[name] = visits
    else:
        ordered = sorted(depths)
        quantiles = {name: ordered[math.ceil(level * len(ordered)) - 1] for name, level in DEPTH_QUANTILES.items()}
    return {"rule": recipe.depth_rule, "sampled_per": sampled_per, "grad_visits": tail,
            "expected_visits": expected, "expected_grad_visits": expected_tail,
            "quantile_visits": quantiles}


def _attention_widths(model: ModelConfig) -> tuple[int, int, int]:
    """Head dim, K/V width, and the q, k, v and out projection parameters of one attention."""
    d, h = model.dim, model.dim // model.num_heads
    kv_width = model.num_kv_heads * h
    return h, kv_width, 2 * d * d + 2 * d * kv_width


def _region_calls(model: ModelConfig, core_visits: float, suffix_visits: int) -> tuple[tuple[str, int, float], ...]:
    """(region, physical cells, visits per cell): the prefix runs once per forward."""
    return (("prefix", model.prefix_cells, 1), ("core", model.num_cells, core_visits),
            ("suffix", model.suffix_cells, suffix_visits))


def _lora_per_visit(model: ModelConfig) -> int:
    """LoRA parameters of one core cell at one visit (``model.Cell``).

    Attention: rank * (in + out) for q, k, v and out; FFN: rank * (in + out)
    for w1, w3 and w2 of the dense or always-on FFN. Each parameter is one MAC
    per position: the delta runs unmerged as ``B (A x)``.
    """
    core = model.region_config("core")
    _, kv_width, _ = _attention_widths(model)
    ffn = core.always_on_hidden_dim if core.num_experts else core.hidden_dim
    return model.lora_attention_rank * (6 * model.dim + 2 * kv_width) + model.lora_ffn_rank * 3 * (model.dim + ffn)


def parameter_counts(model: ModelConfig) -> dict[str, Any]:
    """Unique parameters: every physical owner once, however many calls use it.

    Experts are shared by all visits of their cell. Attention, router and
    norms flagged per visit, and per-visit LoRA, have one owner per core cell
    and visit of ``visit_capacity``. ``unique_non_embedding`` excludes both the
    input embedding and the independently owned output head.
    """
    d = model.dim
    h, kv_width, projections = _attention_widths(model)
    embedding = model.vocab_size * d
    adapter = 0
    if model.input_injection == "linear":
        adapter = 2 * d * d  # A and W.
    elif model.input_injection in ("diag_decay", "orthogonal"):
        adapter = d * d + 2 * d  # W plus per-channel A_log and dt_bias.
    components = {"input_embedding": embedding, "lm_head": embedding,
                  "final_norm": d, "attention": 0, "norms": 0, "router": 0,
                  "routed_experts": 0, "dense_ffn": 0, "always_on_ffn": 0,
                  "input_adapter": adapter}
    if model.exit_distribution:
        components["exit_gate"] = d + 1  # One shared Linear(dim, 1), including its bias.
    lora = _lora_per_visit(model)
    if lora:
        components["lora"] = 0
    by_region = {}
    for region, physical, _ in _region_calls(model, 1, 1):  # Owners do not depend on visits.
        local = model.region_config(region)
        capacity = (model.visit_capacity or 1) if region == "core" else 1
        owners = {"attention": physical * (capacity if local.attention_per_visit else 1),
                  "router": physical * (capacity if local.router_per_visit else 1) if local.num_experts else 0,
                  "norm_pairs": physical * (capacity if local.norm_per_visit else 1),
                  "expert_banks": physical if local.num_experts else 0}
        parameters = {
            "attention": owners["attention"] * (projections + (2 * h if local.qk_norm else 0) + (d + kv_width if local.qk_bias else 0)),
            # Each norm owner holds the attention and FFN input norms, with sandwich_norm their output norms
            # and with residual_norm their residual norms.
            "norms": 2 * (1 + local.sandwich_norm + local.residual_norm) * d * owners["norm_pairs"]
                     + (physical * d if region == "core" and model.state_update == "ema" else 0),
            "router": owners["router"] * d * local.num_experts,
            "routed_experts": physical * 3 * local.num_experts * d * local.hidden_dim,
            "dense_ffn": physical * 3 * d * local.hidden_dim if not local.num_experts else 0,
            "always_on_ffn": physical * 3 * d * local.always_on_hidden_dim,
        }
        if region == "core" and model.state_update == "ema":
            owners["ffn_output_norm"] = physical
        if region == "core" and lora:
            owners["lora"] = physical * capacity
            parameters["lora"] = physical * capacity * lora
        for name, value in parameters.items():
            components[name] += value
        by_region[region] = {"physical_banks": physical, "num_experts": local.num_experts,
                             "top_k": local.top_k if local.num_experts else None,
                             "hidden_dim": local.hidden_dim, "owners": owners,
                             "parameters": parameters, "unique_total": sum(parameters.values())}
    total = sum(components.values())
    return {"unique_total": total, "unique_trainable": total,
            "unique_non_embedding": total - embedding - embedding,
            "non_input_embedding": total - embedding,
            "input_embedding": embedding, "lm_head": embedding,
            "definition": NOTES["parameters"], "trainable_definition": NOTES["trainable"],
            "by_component": components, "by_region": by_region}


def _forward_components(model: ModelConfig, *, visits: float, readouts: int,
                        pairs_per_position: float) -> dict[str, float]:
    """Forward matrix FLOPs per padded position with ``visits`` core visits.

    Linear in ``visits``, so the expected depth gives the expected forward.
    """
    d = model.dim
    _, _, projections = _attention_widths(model)
    embedding = model.vocab_size * d
    forward = {"attention_projections": 0, "attention_scores_values": 0,
               "router": 0, "routed_experts": 0, "dense_ffn": 0,
               "always_on_ffn": 0, "input_adapter": 0, "head": readouts * 2 * embedding}
    if model.exit_distribution:
        forward["exit_gate"] = readouts * 2 * d
    for region, physical, region_visits in _region_calls(model, visits, readouts):
        local = model.region_config(region)
        calls = physical * region_visits
        forward["attention_projections"] += calls * 2 * projections
        # QK^T and AV: two MACs for each declared query/key pair and dimension.
        forward["attention_scores_values"] += calls * 4 * pairs_per_position * d
        if local.num_experts:
            forward["router"] += calls * 2 * d * local.num_experts
            forward["routed_experts"] += calls * 6 * d * local.hidden_dim * local.top_k
        else:
            forward["dense_ffn"] += calls * 6 * d * local.hidden_dim
        forward["always_on_ffn"] += calls * 6 * d * local.always_on_hidden_dim
    if model.input_injection == "linear":
        forward["input_adapter"] = visits * 4 * d * d
    elif model.input_injection in ("diag_decay", "orthogonal"):
        forward["input_adapter"] = visits * 2 * d * d  # W e, recomputed at every visit.
    lora = _lora_per_visit(model)
    if lora:
        forward["lora"] = model.num_cells * visits * 2 * lora
    return forward


def _pass_flops(forward: dict[str, float], differentiated: dict[str, float], *,
                activation_checkpointing: str | None, expected: bool) -> tuple[float, float, float | None]:
    """Forward, backward and replay FLOPs of one pass from its forward components.

    Backward is two products per differentiated forward product. ``full_cells``
    replays every differentiated Cell once, not the adapter, head or gate; the
    expected-CE loss replays its head projections under either policy.
    """
    differentiated_total = sum(differentiated.values())
    replay = (0 if activation_checkpointing == "none" else
              differentiated_total - differentiated["head"] - differentiated["input_adapter"]
              - differentiated.get("exit_gate", 0)
              if activation_checkpointing == "full_cells" else None)
    if expected and replay is not None:
        replay += differentiated["head"]
    return sum(forward.values()), 2 * differentiated_total, replay


def training_flops(
    model: ModelConfig, recipe: TrainingRecipe, *, seq_len: int, batch_size: int = 1,
    activation_checkpointing: Literal["none", "full_cells"] | None = "none",
    attention_pairs_per_sequence: float | None = None,
) -> dict[str, Any]:
    """Matrix FLOPs per padded position and cell calls of one training pass.

    Fixed depth at the model's ``num_visits`` with full BPTT uses the
    recipe's objective. Expected CE has a head/coda readout per visit and its
    memory-bounded loss replays every head projection in backward, even
    without Cell checkpointing. Sampled depth k and tail gradients (terminal
    CE) use the sampled-depth formula: forward at k visits; backward and
    replay over the prefix, suffix, head and last min(grad_visits, k) visits,
    which differentiate only with input injection. Its FLOPs are expectations
    over the depth rule (the work is affine in k and min(grad_visits, k)),
    ``flops.training_depth`` gives the work of a pass at the p50/p95 depths
    (work never decreases with depth), and its calls are given only at the
    nominal depth. Any other path has no backward estimate. ``None``
    checkpointing means unknown recomputation, not zero; ``full_cells``
    replays every Cell's matrix operations once, not the input adapter or
    terminal head. ``attention_pairs_per_sequence`` is the mean number of
    allowed causal query/key pairs in a packed sequence, including its
    diagonal; None counts dense S-by-S matmuls. ``region_calls`` gives the
    logical calls of each region. ``estimate_cost`` validates the arguments,
    and ``RecipeSpec`` validates that the recipe fits the model.
    """
    dense_attention = attention_pairs_per_sequence is None
    pairs = seq_len * seq_len if dense_attention else attention_pairs_per_sequence
    pairs_per_position = seq_len if dense_attention else pairs / seq_len
    expected = recipe.objective == "expected_ce"
    depth = model.num_visits
    readouts = depth if expected else 1
    sampled = bool(recipe.sampled_visits) or recipe.depth_rule == "poisson_lognormal_micro_v1"
    truncated = recipe.grad_visits is not None and (sampled or recipe.grad_visits != depth)
    supported = (sampled or recipe.num_visits == depth) and not (truncated and model.input_injection == "none")
    sampled_formula = supported and (sampled or truncated)

    def forward_at(visits: float) -> dict[str, float]:
        return _forward_components(model, visits=visits, readouts=readouts, pairs_per_position=pairs_per_position)

    nominal = forward_at(depth)
    forward = differentiated = nominal
    if sampled_formula:
        summary = training_depth(recipe)
        forward, differentiated = forward_at(summary["expected_visits"]), forward_at(summary["expected_grad_visits"])
        quantiles = {}
        for name, visits in summary["quantile_visits"].items():
            tail = min(recipe.grad_visits or visits, visits)
            parts = _pass_flops(forward_at(visits), forward_at(tail),
                                activation_checkpointing=activation_checkpointing, expected=False)
            quantiles[name] = {"visits": visits, "grad_visits": tail,
                               "training_per_token": None if parts[2] is None else sum(parts)}
    fwd, bwd, replay = (_pass_flops(forward, differentiated, activation_checkpointing=activation_checkpointing,
                                    expected=expected) if supported else (sum(forward.values()), None, None))
    train = fwd + bwd + replay if bwd is not None and replay is not None else None
    forward_known = sampled_formula or not sampled
    # full_cells replays every Cell, not the adapter, head or gate; expected CE replays its head.
    replayed = (set(differentiated) - {"head", "input_adapter", "exit_gate"}
                if activation_checkpointing == "full_cells" else set())
    if expected:
        replayed.add("head")
    flops = {
        "mac_to_flops": 2,
        "normalization": "per_padded_input_position",
        "attention_convention": NOTES["dense_attention" if dense_attention else "document_attention"],
        "attention_pairs_per_sequence": pairs,
        "backward_convention": NOTES["tail_backward" if sampled_formula else "full_backward"],
        "forward_components": forward if forward_known else None,
        "backward_components": {name: 2 * value for name, value in differentiated.items()} if supported else None,
        "recompute_components": None if replay is None else {
            name: value if name in replayed else 0 for name, value in differentiated.items()},
        "forward_per_token": fwd if forward_known else None, "backward_per_token": bwd,
        "recompute_per_token": replay, "training_per_token": train,
        "training_per_forward_batch": None if train is None else train * batch_size * seq_len,
        "activation_checkpointing": activation_checkpointing,
    }
    if expected:
        flops["objective_recompute_policy"] = NOTES["expected_recompute"]
    if sampled_formula:
        flops["training_depth"] = summary | {"quantiles": quantiles, "definition": NOTES["depth_quantiles"]}
    if sampled:
        flops.update(nominal_forward_components=nominal, nominal_forward_per_token=sum(nominal.values()))

    cells = routers = assignments = 0
    region_calls = {}
    for region, physical, visits in _region_calls(model, depth, readouts):
        local = model.region_config(region)
        cells += physical * visits
        if local.num_experts:
            routers += physical * visits
            assignments += physical * visits * local.top_k
        region_calls[region] = ({"logical_calls": None, "nominal_logical_calls": physical * visits}
                                if sampled and region == "core" else {"logical_calls": physical * visits})
    counts = {"forward_cells": cells, "attention": cells, "router": routers, "head": readouts,
              "expert_assignments_per_input_token": assignments}
    if sampled:  # Only the head count is known; the others are those of the nominal depth.
        counts = {"forward_cells": None, "attention": None, "router": None, "head": readouts,
                  "expert_assignments_per_input_token": None, "nominal": counts}
    if sampled or not supported:
        backward_cells = None
    elif sampled_formula:  # A fixed depth has a deterministic differentiated tail.
        backward_cells = model.prefix_cells + model.num_cells * recipe.grad_visits + model.suffix_cells
    else:
        backward_cells = cells
    calls = counts | {
        "backward_cells": backward_cells,
        "recompute_cells": None if backward_cells is None else (
            backward_cells if activation_checkpointing == "full_cells" else
            0 if activation_checkpointing == "none" else None),
        "head_definition": NOTES["expected_head" if expected else "terminal_head"],
    }
    if expected:
        calls.update(exit_gate=readouts, recompute_head=readouts if supported else None, inference_head=1)

    unknown = {name: NOTES[name] for name in
               ("non_matmul_flops", "optimizer_flops", "measured_flops", "runtime_memory", "actual_tokens")}
    if expected:
        unknown["non_matmul_flops"] += " " + NOTES["exit_non_matmul_flops"]
    if model.input_injection in ("diag_decay", "orthogonal"):
        unknown["input_injection"] = NOTES["input_injection"]
    if model.state_update == "ema":
        unknown["ema_state"] = NOTES["ema_state"]
    if not supported:
        unknown["training_path"] = NOTES["training_path"]
    if sampled:
        unknown["training_depth"] = NOTES["training_depth" if sampled_formula else "nominal_depth"]
    if activation_checkpointing is None:
        unknown["recompute_flops"] = NOTES["recompute_flops"]

    if sampled_formula:
        formula = "sampled_depth_matrix_v1" if dense_attention else "sampled_depth_document_matrix_v1"
    elif expected:
        formula = "fixed_expected_matrix_v1" if dense_attention else "fixed_expected_document_matrix_v1"
    else:
        formula = FORMULA_VERSION if dense_attention else "fixed_terminal_document_matrix_v1"
    tokens = {"batch_size": batch_size, "seq_len": seq_len,
              "padded_positions_per_forward": batch_size * seq_len,
              "logical_cell_positions_per_forward": None if sampled else cells * batch_size * seq_len,
              "valid_input_tokens": None, "valid_label_tokens": None}
    return {"formula_version": formula, "calls": calls, "region_calls": region_calls,
            "flops": flops, "tokens": tokens, "unknown": unknown}


def kv_bytes(model: ModelConfig, depth: int, *, seq_len: int, batch_size: int = 1,
             bytes_per_element: int = 2) -> dict[str, Any]:
    """Logical inference KV at a fixed depth of ``depth`` core visits.

    K and V for every logical cell call: parameter sharing never merges cache
    slots. ``estimate_cost`` validates the arguments, including ``depth``
    against the model's ``visit_capacity``.
    """
    h, kv_width, _ = _attention_widths(model)
    slots = sum(physical * visits for _, physical, visits in _region_calls(model, depth, 1))
    per_token = slots * 2 * kv_width * bytes_per_element
    return {"logical_slots": slots, "num_visits": depth, "num_kv_heads": model.num_kv_heads,
            "head_dim": h, "bytes_per_element": bytes_per_element,
            "bytes_per_cached_token": per_token,
            "bytes_at_context": per_token * seq_len * batch_size,
            "definition": NOTES["kv"]}


def estimate_cost(
    config: ModelConfig | RecipeSpec, *, seq_len: int, batch_size: int = 1,
    bytes_per_element: int = 2,
    activation_checkpointing: Literal["none", "full_cells"] | None = "none",
    attention_pairs_per_sequence: float | None = None,
) -> dict[str, Any]:
    """Unique parameters, training FLOPs and logical inference KV of one recipe or model.

    Validates the arguments and assembles ``parameter_counts``,
    ``training_flops`` and ``kv_bytes``. A ``RecipeSpec`` supplies the
    training recipe and the separately declared fixed inference depth, which
    independent owners cannot run beyond their allocated visit capacity. A
    bare ``ModelConfig`` trains at its own ``num_visits`` with full BPTT and
    the objective its exit gate selects. Supply
    ``attention_pairs_per_sequence`` from actual document masks; it does not
    predict masked-kernel efficiency.
    """
    for name, value in (("seq_len", seq_len), ("batch_size", batch_size),
                        ("bytes_per_element", bytes_per_element)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if activation_checkpointing not in (None, "none", "full_cells"):
        raise ValueError("activation_checkpointing must be none, full_cells or None (unknown)")
    if attention_pairs_per_sequence is not None and (
        type(attention_pairs_per_sequence) not in (int, float)
        or not math.isfinite(attention_pairs_per_sequence)
        or not 0 <= attention_pairs_per_sequence <= seq_len * (seq_len + 1) / 2
    ):
        raise ValueError("attention_pairs_per_sequence must be finite and within the causal triangle")
    spec = config if isinstance(config, RecipeSpec) else None
    model = config if spec is None else spec.model
    if seq_len > model.max_seq_len:
        raise ValueError("seq_len exceeds model max_seq_len")
    inference_depth = model.num_visits if spec is None else spec.inference.get("num_visits", model.num_visits)
    if type(inference_depth) is not int or inference_depth < 1:
        raise ValueError("inference num_visits must be a positive integer")
    if model.visit_capacity is not None and inference_depth > model.visit_capacity:
        raise ValueError("inference num_visits exceeds independent owner visit_capacity")
    recipe = spec.training if spec is not None else TrainingRecipe(
        num_visits=model.num_visits, objective="expected_ce" if model.exit_distribution else "terminal_ce")

    parameters = parameter_counts(model)
    training = training_flops(model, recipe, seq_len=seq_len, batch_size=batch_size,
                              activation_checkpointing=activation_checkpointing,
                              attention_pairs_per_sequence=attention_pairs_per_sequence)
    kv = kv_bytes(model, inference_depth, seq_len=seq_len, batch_size=batch_size,
                  bytes_per_element=bytes_per_element)
    for region, physical, visits in _region_calls(model, inference_depth, 1):
        parameters["by_region"][region] |= (training["region_calls"][region]
                                            | {"inference_logical_calls": physical * visits})
    unknown = training["unknown"]
    if spec is not None and spec.inference.get("cache") == "uncached":
        unknown["inference_cache"] = NOTES["inference_cache"]
    return {"schema_version": 1, "formula_version": training["formula_version"],
            "kind": "analytical_matrix_estimate", "parameters": parameters,
            "calls": training["calls"] | {"inference_cells": kv["logical_slots"]},
            "flops": training["flops"], "tokens": training["tokens"], "kv": kv, "unknown": unknown}
