#!/usr/bin/env python3
"""Per-visit gradients, gradient noise and loss curvature of one checkpoint; nothing is trained.

    python scripts/visit_gradients.py (--checkpoint <run>/checkpoint/step-S | --model <run>/hf) \\
        --manifest M --output OUT.json

Sample: the first ``--sequences`` blocks of ``--split`` in seed-shuffled document order
(``moeloop.diagnostics.sample_batches``), in ``--groups`` equal groups of consecutive
blocks. Loss L: the run's training objective (``moeloop.training.objective``) for the
whole sample as one optimizer update, i.e. terminal CE summed over all valid labels and
divided by their number, plus the router auxiliaries averaged over the microbatches of
``--batch-size`` blocks, at the training depth. The model runs in FP32 and eval mode,
which is the training forward for every arm without a random initial state.

Per-visit gradients (models with more than one visit): ``untie`` gives every visit of a
core cell its own copy of each owner the visits share (the experts; attention, router and
norms unless they are owned per visit). The copy visit v uses receives exactly g_v, and
the shared owner's gradient is the sum over visits. Per component (attention, router,
experts, norm, ffn): ``norm[v]`` = |g_v|, ``cosine[i][j]`` = cos(g_i, g_j), ``snr[v]`` for
g_v and ``snr_sum`` for the sum over visits. Owners a visit has to itself are included with
``shared: false``: their g_v are distinct tensors of one shape. SNR is |m|^2 divided by
the mean over groups of |G_k - m|^2, where G_k is group k's part of the gradient of L
times the number of groups, so that the mean m of the G_k is the gradient of L.

Untied models (one visit): ``untied_cosine`` gives, per attention and norm, the cosine of
the gradients of core layer c and layer c + num_cells // 2, for c below num_cells // 2,
and their mean. Experts have no one-to-one correspondence and are not compared.

Curvature: ``--power-iterations`` steps of power iteration on the Hessian of L over all
parameters, from a seeded normal vector. Each Hessian-vector product is the central
difference (g(theta + h w) - g(theta - h w)) / (2 h) of two gradient passes, with
|h w| = ``--fd-step`` |theta|; the perturbation actually stored in FP32 is used as the
probe direction, so rounding the parameters cannot bias the product. MoE routing is
discontinuous in theta, so the top-k sets selected at theta are recorded once and reused
at theta +- h w: the routed weights and the balance term still follow theta, and the loss
is smooth along every probe. Power iteration converges to the eigenvalue of largest
magnitude; ``rayleigh`` lists the Rayleigh quotient of every step and ``last_change`` is
the relative change of the last one. For an AdamW DCP checkpoint, ``preconditioned``
repeats this for D^(-1/2) H D^(-1/2) with D = sqrt(exp_avg_sq) + eps from the optimizer
state. An HF export has no optimizer state, so it is null there.

Under ``scripts/run.py`` (``RUN_DIR`` set) this writes ``$RUN_DIR/config.json``, the source
run's config with an ``analysis`` block, and a flat ``$RUN_DIR/metrics.json``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from moeloop.config import ModelConfig
from moeloop.data import PreparedManifest
from moeloop.diagnostics import device_batch, sample_batches, source_config, write_run_json
from moeloop.evaluate import _check_tokenizer_identity
from moeloop.export import _key_layout, _load_model_weights, resolve_model_config
from moeloop.model import RecursiveLM
from moeloop.moe import Router
from moeloop.recipe_spec import RecipeSpec
from moeloop.training import objective

# Cell owners: per-visit lists, which Cell indexes by visit, and single modules shared by every visit.
OWNER_LISTS = {"attentions": "attention", "routers": "router", "attention_norms": "norm", "ffn_norms": "norm",
               "attention_output_norms": "norm", "ffn_output_norms": "norm",
               "attention_residual_norms": "norm", "ffn_residual_norms": "norm"}
OWNER_MODULES = {"experts": "experts", "dense_ffn": "ffn", "always_on_ffn": "ffn"}


def load_model(checkpoint: Path | None, export: Path | None, manifest: PreparedManifest, device: str):
    """The FP32 ``RecursiveLM``, its run directory and its step, from a DCP checkpoint or an HF export.

    Both build this package's classes, which ``frozen_routing`` finds by type, and load the
    weights strictly; the remote code an HF export carries would define classes of another module.
    """
    if checkpoint is not None:
        with torch.device("meta"):
            model = RecursiveLM(resolve_model_config(checkpoint))
        state, _ = _load_model_weights(checkpoint, model, "model")
        model.load_state_dict(state, strict=True, assign=True)
        return model.float().to(device), checkpoint.parent.parent, int(checkpoint.name.removeprefix("step-"))
    from safetensors.torch import load_file
    _check_tokenizer_identity(export, manifest)
    with torch.device("meta"):
        model = RecursiveLM(ModelConfig.from_dict(json.loads((export / "config.json").read_text())["model_config"]))
    state = {key.removeprefix("model."): value for path in sorted(export.glob("*.safetensors"))
             for key, value in load_file(path).items()}
    model.load_state_dict(state, strict=True, assign=True)
    name = json.loads((export / "export_provenance.json").read_text())["checkpoint"]["name"]
    return model.float().to(device), export.parent, int(name.removeprefix("step-"))


def adam_root(checkpoint: Path, model: nn.Module, eps: float) -> list[torch.Tensor]:
    """R = (sqrt(exp_avg_sq) + eps)^(-1/2) per parameter, in ``model.parameters()`` order, from the DCP optimizer state."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    reader = dcp.FileSystemReader(checkpoint)
    _, mapping = _key_layout(set(reader.read_metadata().state_dict_metadata), model.state_dict(), model_only=False)
    keys = [f"optimizer.state.{mapping[name]}.exp_avg_sq" for name, _ in model.named_parameters()]
    state = {key: torch.empty_like(p, device="cpu") for key, p in zip(keys, model.parameters())}
    dcp.load(state, storage_reader=reader, planner=DefaultLoadPlanner(flatten_state_dict=False), no_dist=True)
    device = next(model.parameters()).device
    return [(state[key].to(device).sqrt() + eps).rsqrt() for key in keys]


def gradient(model, batches, recipe, sample: tuple[int, int], device) -> float:
    """Accumulate into ``.grad`` the part of the gradient of L that ``batches`` hold; return their part of L.

    ``sample`` is ``(valid labels, microbatches)`` of the whole sample, which normalize L.
    """
    labels_total, microbatches = torch.tensor(sample[0], device=device), sample[1]
    total = 0.0
    for inputs, labels in batches:
        inputs, labels, kwargs = device_batch(inputs, labels, device)
        output = model(inputs["input"], **kwargs, num_visits=recipe.num_visits)
        loss, _ = objective(output, labels, labels_total, recipe, accumulation=microbatches)
        loss.backward()
        total += float(loss.detach())
    return total


def untie(model: RecursiveLM, visits: int):
    """A copy of ``model`` whose core cells give every visit its own copy of each shared owner.

    Returns the copy and ``(component, shared, [module visit v uses])`` per core owner with
    parameters. A shared owner list becomes one copy per visit, which Cell indexes by visit;
    a single module (the expert bank) is swapped in by a pre-hook before each core call.
    Only the per-visit modules require gradients.
    """
    untied = copy.deepcopy(model).requires_grad_(False)
    owners = []
    for cell in untied.core:
        for name, component in OWNER_LISTS.items():
            modules = getattr(cell, name)
            if not list(modules.parameters()):
                continue
            shared = len(modules) == 1
            if shared:
                modules = nn.ModuleList(copy.deepcopy(modules[0]) for _ in range(visits))
                setattr(cell, name, modules)
            owners.append((component, shared, list(modules[:visits])))
        copies = {name: nn.ModuleList(copy.deepcopy(getattr(cell, name)) for _ in range(visits))
                  for name in OWNER_MODULES if getattr(cell, name) is not None}
        owners += [(OWNER_MODULES[name], True, list(modules)) for name, modules in copies.items()]
        cell.visit_copies = nn.ModuleDict(copies)

        def use_visit_copies(cell, args, kwargs):
            for name, modules in cell.visit_copies.items():
                setattr(cell, name, modules[kwargs["visit"]])
        cell.register_forward_pre_hook(use_visit_copies, with_kwargs=True)
    for _, _, modules in owners:
        for module in modules:
            module.requires_grad_(True)
    return untied, owners


def visit_statistics(model, batches, recipe, sample: tuple[int, int], groups: int, device) -> dict:
    """Per-component norms, cosines and SNRs of the per-visit gradients (module docstring)."""
    visits = recipe.num_visits
    untied, owners = untie(model, visits)
    components = sorted({component for component, _, _ in owners})
    parameters = {(component, v): [p for c, _, modules in owners if c == component for p in modules[v].parameters()]
                  for component in components for v in range(visits)}
    series = [*range(visits), "sum"]

    def dot(left, right):
        return sum(float((a.double() * b.double()).sum()) for a, b in zip(left, right))

    # The group parts c_k of the gradient sum to g_v; G_k = groups * c_k, so sum_k |G_k|^2 = groups^2 sum_k |c_k|^2.
    sums = {key: [torch.zeros_like(p) for p in ps] for key, ps in parameters.items()}
    squares = {(component, s): 0.0 for component in components for s in series}
    size = len(batches) // groups
    for k in range(groups):
        gradient(untied, batches[k * size:(k + 1) * size], recipe, sample, device)
        for component in components:
            parts = [[p.grad for p in parameters[component, v]] for v in range(visits)]
            for v, part in enumerate(parts):
                squares[component, v] += groups ** 2 * dot(part, part)
                for running, c in zip(sums[component, v], part):
                    running += c
            squares[component, "sum"] += groups ** 2 * sum(float(sum(cs).double().square().sum()) for cs in zip(*parts))
        for ps in parameters.values():
            for p in ps:
                p.grad = None

    result = {}
    for component in components:
        g = [sums[component, v] for v in range(visits)]
        total = [sum(gs) for gs in zip(*g)]
        norms = [dot(gv, gv) for gv in g]
        signal = {**dict(enumerate(norms)), "sum": dot(total, total)}
        snr = {s: signal[s] / (squares[component, s] / groups - signal[s]) for s in series}
        result[component] = {
            "shared": all(shared for c, shared, _ in owners if c == component),
            "norm": [value ** 0.5 for value in norms],
            "cosine": [[dot(g[i], g[j]) / (norms[i] * norms[j]) ** 0.5 for j in range(visits)] for i in range(visits)],
            "snr": [snr[v] for v in range(visits)], "snr_sum": snr["sum"],
        }
    return result


def untied_cosine(model: RecursiveLM) -> dict:
    """Per attention and norm: cos of the ``.grad`` of core layers c and c + num_cells // 2."""
    offset = len(model.core) // 2
    result = {}
    for component in ("attention", "norm"):
        def vector(cell):
            return torch.cat([p.grad.flatten() for name, c in OWNER_LISTS.items() if c == component
                              for p in getattr(cell, name).parameters()])
        cosines = [float(F.cosine_similarity(vector(model.core[c]).double(), vector(model.core[c + offset]).double(), dim=0))
                   for c in range(offset)]
        result[component] = {"offset": offset, "cosine": cosines, "mean": sum(cosines) / len(cosines)}
    return result


@contextmanager
def frozen_routing(model):
    """Record every router call's top-k selection; with ``state["replay"] = 0`` the calls reuse them in order.

    A replayed call keeps the recorded ids and counts and recomputes from the current
    weights what ``Router.forward`` computes from them: the selected-set weights and the
    balance term. The z-loss does not depend on the selection and is kept.
    """
    state = {"calls": [], "replay": None}

    def hook(router, args, output):
        if state["replay"] is None:
            state["calls"].append((output.ids, output.counts, output.dispatch_counts))
            return output
        ids, counts, dispatch_counts = state["calls"][state["replay"]]
        state["replay"] += 1
        x, router_mask = args
        probs = F.linear(x.float(), router.gate.weight.float()).softmax(-1)
        selected = probs.gather(-1, ids)
        valid = router_mask.float()
        denominator = valid.sum().clamp_min(1)
        probability_mean = (probs * valid.unsqueeze(-1)).sum(tuple(range(probs.ndim - 1))) / denominator
        balance = probs.size(-1) * (counts.float() / (denominator * router.top_k) * probability_mean).sum()
        return output._replace(ids=ids, weights=(selected / selected.sum(-1, keepdim=True)).to(x.dtype),
                               balance=balance, counts=counts, dispatch_counts=dispatch_counts)

    handles = [module.register_forward_hook(hook) for module in model.modules() if isinstance(module, Router)]
    try:
        yield state
    finally:
        for handle in handles:
            handle.remove()


def top_eigenvalue(model, batches, recipe, sample: tuple[int, int], routing, iterations: int, seed: int,
                   fd_step: float, device, root: list[torch.Tensor] | None = None) -> dict:
    """Power iteration on H, or on R H R for the diagonal ``root`` R = D^(-1/2) (module docstring).

    Besides the model and its gradient, a step holds five parameter-sized buffers, six with R.
    """
    parameters = list(model.parameters())
    theta = [p.detach().clone() for p in parameters]
    root = root or [1.0] * len(parameters)
    generator = torch.Generator().manual_seed(seed)
    u = [torch.randn(p.shape, generator=generator).to(device) for p in parameters]

    def dot(left, right):
        return sum(float((a.double() * b.double()).sum()) for a, b in zip(left, right))

    def move(step, direction):
        with torch.no_grad():
            for p, t, w in zip(parameters, theta, direction):
                p.copy_(t + step * w)

    def grad():
        model.zero_grad(set_to_none=True)
        routing["replay"] = 0
        gradient(model, batches, recipe, sample, device)
        return [p.grad for p in parameters]

    step = fd_step * dot(theta, theta) ** 0.5
    rayleigh = []
    for _ in range(iterations):
        w = [r * x for r, x in zip(root, u)]
        scale = step / dot(w, w) ** 0.5
        move(scale, w)
        probe = [p.detach() - t for p, t in zip(parameters, theta)]
        product = grad()
        move(-scale, w)
        minus = grad()
        with torch.no_grad():
            # The probe is the perturbation actually stored in FP32, mapped back by 1/R.
            for d, p, t, r in zip(probe, parameters, theta, root):
                d.sub_(p - t).div_(2 * scale * r)
            for y, g, r in zip(product, minus, root):
                y.sub_(g).mul_(r / (2 * scale))
        rayleigh.append(dot(probe, product) / dot(probe, probe))
        length = dot(product, product) ** 0.5
        u = [y.div_(length) for y in product]
        del w, probe, product, minus
    move(0.0, theta)
    model.zero_grad(set_to_none=True)
    return {"lambda_max": rayleigh[-1], "rayleigh": rayleigh, "step_norm": step,
            "last_change": abs(rayleigh[-1] - rayleigh[-2]) / abs(rayleigh[-1])}


def flat_metrics(result: dict) -> dict:
    metrics = {"loss": result["loss"], "lambda_max": result["curvature"]["lambda_max"],
               "lambda_max_last_change": result["curvature"]["last_change"]}
    if result["preconditioned"] is not None:
        metrics |= {"lambda_max_preconditioned": result["preconditioned"]["lambda_max"],
                    "lambda_max_preconditioned_last_change": result["preconditioned"]["last_change"]}
    for component, row in (result["visits"] or {}).items():
        visits = len(row["norm"])
        metrics |= {f"{component}.norm_{v + 1}": row["norm"][v] for v in range(visits)}
        metrics |= {f"{component}.snr_{v + 1}": row["snr"][v] for v in range(visits)}
        metrics |= {f"{component}.cos_{i + 1}_{j + 1}": row["cosine"][i][j] for i in range(visits) for j in range(i + 1, visits)}
        metrics[f"{component}.snr_sum"] = row["snr_sum"]
    for component, row in (result["untied_cosine"] or {}).items():
        metrics[f"untied.{component}.cos_mean"] = row["mean"]
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="DCP training checkpoint <run>/checkpoint/step-S")
    source.add_argument("--model", type=Path, help="HF export <run>/hf")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON result file, without overwrite")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0, help="document shuffle and power-iteration start")
    parser.add_argument("--groups", type=int, default=8, help="equal groups of sequences for the SNR")
    parser.add_argument("--batch-size", type=int, default=1, help="blocks per microbatch")
    parser.add_argument("--power-iterations", type=int, default=20)
    parser.add_argument("--fd-step", type=float, default=1e-5, help="finite-difference step relative to |theta|")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.sequences % (args.groups * args.batch_size):
        parser.error("--sequences must split into --groups groups of whole --batch-size microbatches")
    checkpoint = None if args.checkpoint is None else args.checkpoint.resolve()
    manifest = PreparedManifest.load(args.manifest)
    model, run, step = load_model(checkpoint, None if args.model is None else args.model.resolve(), manifest, args.device)
    analysis = {"checkpoint": str(checkpoint or args.model.resolve()), "step": step, "seed": args.seed,
                "split": args.split, "sequences": args.sequences, "seq_len": args.seq_len, "groups": args.groups,
                "batch_size": args.batch_size, "power_iterations": args.power_iterations, "fd_step": args.fd_step}
    write_run_json("config.json", {**source_config(run), "analysis": analysis})
    spec = RecipeSpec.from_dict(json.loads((run / "spec.json").read_text()))
    recipe = spec.training
    batches = sample_batches(manifest, split=args.split, seq_len=args.seq_len, sequences=args.sequences,
                             seed=args.seed, batch_size=args.batch_size)
    sample = (sum(int((labels != -100).sum()) for _, labels in batches), len(batches))
    model.eval()
    visits = visit_statistics(model, batches, recipe, sample, args.groups, args.device) if recipe.num_visits > 1 else None
    with frozen_routing(model) as routing:
        loss = gradient(model, batches, recipe, sample, args.device)
        untied = untied_cosine(model) if recipe.num_visits == 1 else None
        curvature = top_eigenvalue(model, batches, recipe, sample, routing, args.power_iterations, args.seed,
                                   args.fd_step, args.device)
        preconditioned = None
        if checkpoint is not None and spec.optimizer["name"] == "AdamW":
            preconditioned = top_eigenvalue(model, batches, recipe, sample, routing, args.power_iterations, args.seed,
                                            args.fd_step, args.device, adam_root(checkpoint, model, spec.optimizer["eps"]))
    result = {"format": "moeloop-visit-gradients", "version": 1, "analysis": analysis,
              "num_visits": recipe.num_visits, "valid_labels": sample[0], "loss": loss,
              "visits": visits, "untied_cosine": untied, "curvature": curvature, "preconditioned": preconditioned}
    content = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(content)
    write_run_json("metrics.json", flat_metrics(result))
    print(content, end="")


if __name__ == "__main__":
    main()
