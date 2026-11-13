"""Plain comparison records; resource tolerances never imply a quality result."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .costs import estimate_cost
from .recipe_spec import RecipeSpec
from .recipes import moe_spec, p1_moe_spec, p1_smelt_spec, p2_moe_spec, p2_smelt_spec, smelt_spec


def diff_recipes(reference: RecipeSpec, candidate: RecipeSpec) -> dict[str, Any]:
    """Return actual resolved model/training/optimizer/run/inference differences."""
    result: dict[str, Any] = {}

    def walk(path: str, left: Any, right: Any) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                walk(f"{path}.{key}" if path else key, left.get(key), right.get(key))
        elif left != right:
            result[path] = {"reference": left, "candidate": right}

    left, right = reference.to_dict(), candidate.to_dict()
    for values, spec in ((left, reference), (right, candidate)):
        # Compare resolved regional values, so an explicit inherited value is
        # not mistaken for a different model mechanism.
        for region in ("prefix", "core", "suffix"):
            actual = spec.model.region_config(region)
            values["model"][region] = {key: getattr(actual, key) for key in
                                       ("hidden_dim", "num_experts", "top_k",
                                        "always_on_hidden_dim", "residual_scale",
                                        "output_lr_scale", "output_init_scale")}
    for field in ("model", "training", "optimizer", "run", "inference"):
        walk(field, left[field], right[field])
    return result


def compare_recipes(
    reference: RecipeSpec, candidate: RecipeSpec, *, seq_len: int,
    tolerance: float = 0.05, batch_size: int = 1, bytes_per_element: int = 2,
    activation_checkpointing: str | None = "none",
    attention_pairs_per_sequence: float | None = None,
    question: str = "Compare common held-out LM CE at the declared recipe resource budgets.",
    data: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify declared estimates, controls, measured differences and unknowns.

    ``matched`` only concerns the three named analytical budgets. Actual data,
    evaluation identity, throughput and quality stay unknown until supplied by
    a run. The full objective and inference readout remain separate records.
    A sampled-depth arm matches on expected training FLOPs; the entry also
    reports each arm's p95-depth pass (a fixed-depth arm's only pass).
    """
    if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("tolerance must be finite and in [0, 1)")
    attention_options = ({} if attention_pairs_per_sequence is None else
                         {"attention_pairs_per_sequence": attention_pairs_per_sequence})
    costs = {name: estimate_cost(spec, seq_len=seq_len, batch_size=batch_size,
                                bytes_per_element=bytes_per_element,
                                activation_checkpointing=activation_checkpointing,
                                **attention_options)
             for name, spec in (("reference", reference), ("candidate", candidate))}
    metrics = {
        "unique_non_embedding_parameters": ("parameters", "unique_non_embedding"),
        "training_flops_per_token": ("flops", "training_per_token"),
        "logical_kv_bytes": ("kv", "bytes_at_context"),
    }
    depths = {name: costs[name]["flops"].get("training_depth") for name in ("reference", "candidate")}
    matched, different, unknown = {}, {}, {}
    for metric, (section, field) in metrics.items():
        left, right = (costs[name][section][field] for name in ("reference", "candidate"))
        entry = {"reference": left, "candidate": right, "relative_tolerance": tolerance,
                 "relative_difference": None if left is None or right is None else (right - left) / left,
                 "basis": "analytical_estimate"}
        if metric == "training_flops_per_token" and any(depths.values()):
            entry["basis"] = "analytical_expectation_over_training_depth"
            entry["p95"] = {name: depth["quantiles"]["p95"]["training_per_token"] if depth
                            else costs[name]["flops"]["training_per_token"] for name, depth in depths.items()}
        if left is None or right is None:
            unknown[metric] = entry | {"reason": "Missing supported cost estimate; unknown is never matched."}
        elif abs(entry["relative_difference"]) <= tolerance:
            matched[metric] = entry
        else:
            different[metric] = entry
    controls: dict[str, Any] = {
        "seq_len": seq_len, "batch_size": batch_size,
        "kv_bytes_per_element": bytes_per_element,
    }
    formulas = {name: costs[name]["formula_version"] for name in ("reference", "candidate")}
    if formulas["reference"] == formulas["candidate"]:
        controls["cost_formula_version"] = formulas["reference"]
    else:
        different["cost_formula_version"] = formulas | {"basis": "declared_formula"}
    objectives = {name: spec.training.objective for name, spec in
                  (("reference", reference), ("candidate", candidate))}
    if objectives["reference"] != objectives["candidate"]:
        different["training.objective"] = objectives | {"basis": "declared_objective"}
    if "expected_ce" in objectives.values():
        unknown["objective_comparability"] = "Expected CE/entropy is a training objective, not standard terminal held-out CE; quality needs a separately declared common evaluation."
    execution = {name: costs[name]["unknown"]["training_depth"] for name in ("reference", "candidate")
                 if "training_depth" in costs[name]["unknown"]}
    if execution:
        unknown["training_execution"] = execution
    if activation_checkpointing is not None:
        controls["activation_checkpointing"] = activation_checkpointing
    else:
        unknown["activation_checkpointing"] = "Recompute policy is unspecified."
    if attention_pairs_per_sequence is not None:
        controls["attention_pairs_per_sequence"] = attention_pairs_per_sequence
    for field in ("vocab_size", "dim", "num_heads", "num_kv_heads"):
        left, right = getattr(reference.model, field), getattr(candidate.model, field)
        if left == right:
            controls[f"model.{field}"] = left
    for region in ("prefix", "core", "suffix"):
        for field in ("hidden_dim", "top_k"):
            left = getattr(reference.model.region_config(region), field)
            right = getattr(candidate.model.region_config(region), field)
            if left == right:
                controls[f"model.{region}.{field}"] = left
    for field in ("optimizer", "training"):
        left, right = reference.to_dict()[field], candidate.to_dict()[field]
        if left == right:
            controls[field] = left
    for field in ("seed", "successful_updates", "schedule_total_steps", "schedule", "objective",
                  "global_batch_positions", "mixed_precision_param", "master_dtype", "reduce_dtype"):
        left, right = reference.run.get(field), candidate.run.get(field)
        if left is not None and left == right:
            controls[f"run.{field}"] = left
    if not data:
        unknown["data"] = "Manifest/tokenizer revisions, split isolation, exposure and valid-token counts require run identities."
    if not evaluation:
        unknown["evaluation"] = "Held-out manifest/revision and evaluation masks must be frozen before quality comparison."
    unknown["runtime"] = "Native FLOPs, throughput, peak memory, and total training compute are not measured by this record."
    unknown["quality"] = "No held-out observations supplied; no scientific ranking or scaling-law conclusion."
    return {
        "schema_version": 1, "question": question,
        "arms": {"reference": reference.to_dict(), "candidate": candidate.to_dict()},
        "costs": costs, "data": None if data is None else dict(data),
        "evaluation": None if evaluation is None else dict(evaluation),
        "matched": matched, "controlled": controls,
        "measured-different": different, "unknown": unknown,
        "recipe_differences": diff_recipes(reference, candidate),
        "tolerance_definition": "abs(candidate-reference)/reference <= tolerance; an engineering rule, not a paper tolerance.",
        "endpoint": {"kind": "successful_optimizer_updates",
                     "reference": reference.run.get("successful_updates"),
                     "candidate": candidate.run.get("successful_updates")},
        "selection_rule": "Freeze arms, data/eval, seeds, endpoint and common held-out CE before observing quality.",
        "scientific_conclusion": None,
    }


def smelt_budget_pair(*, vocab_size: int = 50257, seq_len: int = 2048) -> dict[str, Any]:
    """Reproduce one fixed five-shape screen, with the E=12 pair selected.

    This finite list is the entire candidate family, not an expandable solver.
    Both selected arms have 12 logical layers. SMELT has 8 physical layers
    (2 prelude, 4 core repeated twice, 2 coda) and more experts per bank.
    """
    reference = moe_spec(layers=12, num_experts=8, vocab_size=vocab_size)
    candidates = []
    selected = None
    for experts in (8, 10, 12, 14, 16):
        candidate = smelt_spec(physical_layers=8, num_experts=experts, vocab_size=vocab_size)
        comparison = compare_recipes(reference, candidate, seq_len=seq_len)
        candidates.append({"num_experts": experts,
                           "unique_non_embedding_parameters": comparison["costs"]["candidate"]["parameters"]["unique_non_embedding"],
                           "training_flops_per_token": comparison["costs"]["candidate"]["flops"]["training_per_token"],
                           "logical_kv_bytes": comparison["costs"]["candidate"]["kv"]["bytes_at_context"],
                           "matched_budgets": list(comparison["matched"]),
                           "relative_differences": {key: value["relative_difference"]
                                                    for group in ("matched", "measured-different")
                                                    for key, value in comparison[group].items()}})
        if experts == 12:
            selected = comparison
    return {"schema_version": 1, "candidate_count": len(candidates),
            "candidate_family": "Only SMELT E in (8,10,12,14,16); all other geometry fixed.",
            "candidates": candidates,
            "selected_shape": {"baseline_physical_layers": 12, "baseline_experts": 8,
                               "smelt_physical_layers": 8, "smelt_experts": 12,
                               "vocab_size": vocab_size,
                               "dim": 128, "hidden_dim": 256, "num_heads": 4,
                               "num_kv_heads": 2, "top_k": 2},
            "comparison": selected}


P1_STATISTICS = Path(__file__).resolve().parents[2] / "docs" / "evidence" / "p1-data-statistics.json"
_BUDGETS = ("unique_non_embedding_parameters", "training_flops_per_token", "logical_kv_bytes")


def p1_attention_pairs() -> float:
    """Mean document-causal attention pairs per sequence of the frozen P1 world16 training prefix.

    Read from ``docs/evidence/p1-data-statistics.json``, so it exists in a
    source checkout only.
    """
    return json.loads(P1_STATISTICS.read_text())["attention_pairs_per_sequence"]


def _screen(reference: RecipeSpec, candidates: list[tuple[dict[str, int], RecipeSpec]], *,
            activation_checkpointing: str, sum_residuals: bool = False,
            **options: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare every candidate shape with the reference on the three budgets at 4096 context.

    Both arms take ``activation_checkpointing``; ``options`` go to
    ``compare_recipes``. Returns one row per candidate (its shape, costs and
    relative residuals) and the full comparisons. ``sum_residuals`` adds the
    sum of absolute residuals that P2's selection rule breaks ties with.
    """
    reference = replace(reference, run=reference.run | {"activation_checkpointing": activation_checkpointing})
    rows, comparisons = [], []
    for shape, candidate in candidates:
        candidate = replace(candidate, run=candidate.run | {"activation_checkpointing": activation_checkpointing})
        comparison = compare_recipes(reference, candidate, seq_len=4096,
                                     activation_checkpointing=activation_checkpointing, **options)
        values = comparison["matched"] | comparison["measured-different"]
        relative = {metric: values[metric]["relative_difference"] for metric in _BUDGETS}
        cost = comparison["costs"]["candidate"]
        row = shape | {"unique_non_embedding_parameters": cost["parameters"]["unique_non_embedding"],
                       "training_flops_per_token": cost["flops"]["training_per_token"],
                       "logical_kv_bytes": cost["kv"]["bytes_at_context"],
                       "relative_differences": relative,
                       "maximum_absolute_relative_difference": max(map(abs, relative.values()))}
        if sum_residuals:
            row["sum_absolute_relative_differences"] = sum(map(abs, relative.values()))
        rows.append(row | {"matched_budgets": list(comparison["matched"])})
        comparisons.append(comparison)
    return rows, comparisons


def _endpoint(comparison: dict[str, Any], completion_rule: str) -> dict[str, Any]:
    """The label-token endpoint of both arms' frozen schedules, with its update rounding."""
    reference, candidate = (comparison["arms"][arm]["run"]["schedule"] for arm in ("reference", "candidate"))
    return {
        "kind": "global_valid_label_tokens_with_nominal_update_rounding",
        "reference": reference["target_label_tokens"], "candidate": candidate["target_label_tokens"],
        "nominal_successful_updates": reference["total_steps"],
        "nominal_padded_positions": reference["rounded_nominal_positions"],
        "actual_valid_label_tokens": None,
        "completion_rule": completion_rule,
    }


def _active_parameters(costs: dict[str, Any]) -> dict[str, Any]:
    """Physical: one top-k traversal of each unique cell plus final norm; logical: every core call."""
    active = {}
    for arm, cost in costs.items():
        parameters = cost["parameters"]
        physical = logical = parameters["by_component"]["final_norm"]
        for region in parameters["by_region"].values():
            if not region["physical_banks"]:
                continue
            local = region["unique_total"] - region["parameters"]["routed_experts"]
            local += region["parameters"]["routed_experts"] * region["top_k"] // region["num_experts"]
            physical += local
            logical += local * region["logical_calls"] // region["physical_banks"]
        active[arm] = {"physical_non_embedding": physical, "logical_visited_non_embedding": logical}
    return active


def smelt_p1_pair(
    *, base_lr: float = 3e-4, activation_checkpointing: str = "none",
    attention_pairs_per_sequence: float | None = None,
    data: Mapping[str, Any] | None = None, evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One fixed 12-shape P1 arithmetic screen, with no held-out selection.

    With the measured pairs ``p1_attention_pairs()`` it selects F440, the
    trained ``p1_smelt_spec()``; the default dense-S² estimate selects F424.
    Another measured statistic or checkpoint policy reruns the same finite
    screen; the resulting shape must be frozen before formal training, never
    after quality observations. Actual hardware work and throughput remain
    separate evidence.
    """
    rows, comparisons = _screen(p1_moe_spec(base_lr=base_lr),
        [({"hidden_dim": width}, p1_smelt_spec(base_lr=base_lr, hidden_dim=width))
         for width in (384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472)],
        activation_checkpointing=activation_checkpointing,
        attention_pairs_per_sequence=attention_pairs_per_sequence,
        question="At the P1 public-DCLM endpoint, does middle-half SMELT improve held-out terminal CE under the three declared resource budgets?",
        data=data, evaluation=evaluation)
    selected_index = min(range(len(rows)),
        key=lambda index: (rows[index]["maximum_absolute_relative_difference"], rows[index]["hidden_dim"]))
    selected = comparisons[selected_index]
    selected["endpoint"] = _endpoint(selected, "Verify actual labels at the completed full-update endpoint in both arms; report padding and rounding separately, with at most one update discrepancy.")
    selected["selection_rule"] = "Choose geometry by the minimum maximum absolute analytical budget difference in the fixed 12-shape family; only the two predeclared equal-opportunity LR pilots may use validation. Freeze data/eval/source, one seed42, full WSD and endpoint before formal training; test never selects anything."
    return {
        "schema_version": 1, "anchor": "P1", "candidate_count": len(rows),
        "candidate_family": "Baseline H672/L10/E64/top8/F544/heads7/KV1; SMELT H576/L10(2+5+3)/E96/top8/heads9/KV1, only intermediate width in (384,392,400,408,416,424,432,440,448,456,464,472).",
        "candidates": rows, "selected_candidate_index": selected_index,
        "selection_basis": "Analytical budgets only; no quality observations or additional search candidates.",
        "formal_freeze_status": "Requires actual DCLM document statistics, fixed data/eval/source and target-shape runtime qualification before formal training.",
        "active_parameters": _active_parameters(selected["costs"]),
        "active_definition": "Physical counts one top-k traversal of each unique physical cell and final norm, excluding input embedding/head; logical counts repeated core calls too. Neither is a training-FLOP estimate.",
        "comparison": selected,
    }


def smelt_p2_pair(
    *, base_lr: float = 3e-4, activation_checkpointing: str = "none",
    attention_pairs_per_sequence: float | None = None,
    data: Mapping[str, Any] | None = None, evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Screen exactly 32 public-stack shapes, before any quality observations.

    The default screens dense-S², consistently with the cost API. The formal
    project choice E88/F384 uses the explicitly supplied document-pair mean
    ``p1_attention_pairs()`` of the frozen P1 world16 training prefix. Reusing
    it requires identical P2 data-rank streams and masks; another measured
    statistic recomputes the same finite screen. The selected pair's dense
    estimate is always reported separately, never mixed into selection.
    The record names that mean, read from ``docs/evidence/p1-data-statistics.json``,
    so this function works in a source checkout only.
    """
    if activation_checkpointing not in ("none", "full_cells"):
        raise ValueError("activation_checkpointing must be none or full_cells for P2 budget selection")
    reference = p2_moe_spec(base_lr=base_lr)
    rows, comparisons = _screen(reference,
        [({"num_experts": experts, "hidden_dim": width},
          p2_smelt_spec(base_lr=base_lr, hidden_dim=width, num_experts=experts))
         for experts in (88, 92, 96, 100) for width in (336, 344, 352, 360, 368, 376, 384, 392)],
        activation_checkpointing=activation_checkpointing, sum_residuals=True,
        attention_pairs_per_sequence=attention_pairs_per_sequence,
        question="At the P2 public-DCLM endpoint, does middle-half SMELT improve held-out terminal CE under the three declared resource budgets?",
        data=data, evaluation=evaluation)
    feasible = [i for i, row in enumerate(rows) if set(row["matched_budgets"]) == set(_BUDGETS)]
    if not feasible:
        raise ValueError("No P2 candidate meets all three 5% budgets under the supplied cost assumptions")
    selected_index = min(feasible, key=lambda i: (
        rows[i]["maximum_absolute_relative_difference"],
        rows[i]["sum_absolute_relative_differences"],
        abs(rows[i]["num_experts"] - 96), rows[i]["hidden_dim"]))
    selected = comparisons[selected_index]
    selected["endpoint"] = _endpoint(selected, "Both arms must complete the frozen full-update endpoint; verify actual labels, padding and rounding separately, with at most one update discrepancy.")
    selected["selection_rule"] = "Filter all three analytical budget residuals to <=5%; minimize (maximum absolute residual, sum absolute residuals, abs(E-96), F). No quality selection; freeze source/data/eval/seed42/full WSD before training."
    reference, candidate = (RecipeSpec.from_dict(selected["arms"][arm]) for arm in ("reference", "candidate"))
    dense = compare_recipes(reference, candidate, seq_len=4096,
        activation_checkpointing=activation_checkpointing, data=data, evaluation=evaluation,
        question="Separate dense-S² accounting for the selected P2 pair; not measured kernel work.")
    return {
        "schema_version": 1, "anchor": "P2", "candidate_count": len(rows),
        "candidate_family": "Baseline H1280/F432/L12/E64/top8/Q10/KV1; SMELT H1056/L12(3+6x2+3)/top8/Q12/KV1; only E in (88,92,96,100) and F in (336,344,352,360,368,376,384,392).",
        "candidates": rows, "selected_candidate_index": selected_index,
        "selection_basis": "Analytical budgets only; no quality observations or additional search candidates.",
        "attention_statistics": {
            "attention_pairs_per_sequence": attention_pairs_per_sequence,
            "formal_p1_prefix_mean": p1_attention_pairs(),
            "formal_statistic_provenance": "docs/evidence/p1-data-statistics.json; exact frozen P1 world16 prefix only, not arbitrary DCLM packing.",
            "reuse_condition": "Prove identical P2 data-rank streams/document masks before reusing the formal P1 statistic; otherwise supply measured pairs and freeze the resulting identity.",
        },
        "formal_freeze_status": "Requires frozen P2 source/data/evaluation and exact P1-prefix data verification, bounded stability pilots and target-shape runtime qualification.",
        "active_parameters": _active_parameters(selected["costs"]),
        "active_definition": "Physical counts one top-k traversal per unique cell plus final norm, excluding embedding/head; logical counts repeated core calls. Neither is a FLOP estimate.",
        "comparison": selected, "dense_comparison": dense,
    }
