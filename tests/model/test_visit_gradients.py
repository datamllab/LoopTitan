"""scripts/visit_gradients.py pinned against untied duplicates, exact Hessians and its run files."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random

import pytest

torch = pytest.importorskip("torch")
from torch import nn
import torch.distributed.checkpoint as dcp
from torch.nn.attention import SDPBackend, sdpa_kernel

from moeloop.diagnostics import device_batch, sample_batches
from moeloop.model import RecursiveLM, per_visit
from moeloop.prepare import prepare_dataset
from moeloop.recipe_spec import RecipeSpec
from moeloop.training import objective
import visit_gradients
from visit_gradients import OWNER_LISTS, OWNER_MODULES, frozen_routing, gradient, top_eigenvalue, untie, visit_statistics

ROOT = Path(__file__).parents[2]
TOKENIZER = {"id": "test-tokenizer", "revision": "fixed-v1", "vocab_size": 16, "bos_token_id": 0, "eos_token_id": 15,
             "pad_token_id": 15}
TINY = dict(vocab_size=16, dim=8, hidden_dim=4, num_heads=2, num_kv_heads=1, num_experts=4, top_k=2, max_seq_len=64)
SEQ_LEN = 8


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(5)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def manifest(tmp_path):
    rng = random.Random(0)
    texts = [" ".join(str(rng.randrange(1, 15)) for _ in range(rng.randrange(5, 30))) for _ in range(16)]
    return prepare_dataset(tmp_path / "data", {"text": {"validation": texts, "test": texts[:6]}},
                           encode=lambda text: [int(value) for value in text.split()], tokenizer=TOKENIZER)


def arm(name: str, optimizer: str = "adamw", **changes) -> RecipeSpec:
    """A phase-1 arm at tiny width: one third of its prefix, core and suffix cells."""
    spec = json.loads((ROOT / "recipes" / "phase1" / f"{name}-{optimizer}.json").read_text())
    model = spec["model"] | TINY | {key: spec["model"][key] // 3 for key in ("prefix_cells", "num_cells", "suffix_cells")}
    if model["core"] and model["core"]["num_experts"]:
        model["core"] = model["core"] | {"num_experts": 2}
    return RecipeSpec.from_dict(spec | {"model": model | changes})


def sample(manifest, sequences=4, batch_size=1):
    batches = sample_batches(manifest, split="validation", seq_len=SEQ_LEN, sequences=sequences, seed=0,
                             batch_size=batch_size)
    return batches, (sum(int((labels != -100).sum()) for _, labels in batches), len(batches))


def unrolled(model: RecursiveLM, visits: int) -> RecursiveLM:
    """The explicitly untied duplicate: core cell i in visit v becomes single-visit core cell v * C + i."""
    cells = len(model.core)
    config = replace(model.config, num_cells=cells * visits, num_visits=1, visit_capacity=None,
                     attention_per_visit=False, router_per_visit=False, norm_per_visit=False)
    duplicate = RecursiveLM(config)
    duplicate.load_state_dict({key: value for key, value in model.state_dict().items() if not key.startswith("core.")},
                              strict=False)
    for v in range(visits):
        for i, cell in enumerate(model.core):
            target = duplicate.core[v * cells + i]
            for name in OWNER_LISTS:
                getattr(target, name)[0].load_state_dict(per_visit(getattr(cell, name), v).state_dict())
            target.experts.load_state_dict(cell.experts.state_dict())
    return duplicate


def visit_modules(cell, v):
    """The owners an untied cell uses in visit v, by name."""
    return {**{name: per_visit(getattr(cell, name), v) for name in OWNER_LISTS},
            **{name: modules[v] for name, modules in cell.visit_copies.items()}}


@pytest.mark.parametrize("name", ["m3", "m5", "m7"])
def test_per_visit_gradients_sum_to_the_gradient_and_match_an_untied_duplicate(manifest, name):
    spec = arm(name)
    model = RecursiveLM(spec.model).eval()
    duplicate = unrolled(model, 2).eval()
    batches, totals = sample(manifest)
    with torch.no_grad():
        inputs, labels, kwargs = device_batch(*batches[0], "cpu")
        assert torch.equal(duplicate(inputs["input"], **kwargs).logits, model(inputs["input"], **kwargs).logits)
    untied, owners = untie(model, 2)
    losses = [gradient(module, batches, spec.training, totals, "cpu") for module in (model, untied)]
    losses.append(gradient(duplicate, batches, replace(spec.training, num_visits=1), totals, "cpu"))
    assert losses == pytest.approx([losses[0]] * 3, rel=1e-6)
    checked = 0
    for i, (cell, untied_cell) in enumerate(zip(model.core, untied.core)):
        for owner in [*OWNER_LISTS, *OWNER_MODULES]:
            if getattr(cell, owner) is None or not list(getattr(cell, owner).parameters()):
                continue
            per_visit_grads = []
            for v in range(2):
                copy = visit_modules(untied_cell, v)[owner]
                twin = getattr(duplicate.core[v * len(model.core) + i], owner)
                twin = twin[0] if isinstance(twin, nn.ModuleList) else twin
                grads = [p.grad for p in copy.parameters()]
                for g, expected in zip(grads, twin.parameters()):
                    torch.testing.assert_close(g, expected.grad, rtol=1e-5, atol=1e-8)
                per_visit_grads.append(grads)
                checked += 1
            tied = getattr(cell, owner)
            if isinstance(tied, nn.ModuleList) and len(tied) > 1:  # owned per visit: g_v is visit v's own gradient
                for v, grads in enumerate(per_visit_grads):
                    for g, own in zip(grads, tied[v].parameters()):
                        torch.testing.assert_close(g, own.grad, rtol=1e-5, atol=1e-8)
            else:
                for (first, second), full in zip(zip(*per_visit_grads), tied.parameters()):
                    torch.testing.assert_close(first + second, full.grad, rtol=1e-5, atol=1e-8)
    assert checked
    shared = {component for component, shared, _ in owners if shared}
    assert shared == {"m3": {"attention", "router", "experts", "norm"}, "m5": {"router", "experts"},
                      "m7": {"experts"}}[name]


def test_visit_statistics_match_their_definitions_on_the_duplicate(manifest):
    spec = arm("m3")
    model = RecursiveLM(spec.model).eval()
    batches, totals = sample(manifest)
    duplicate = unrolled(model, 2).eval()
    groups = []
    for k in range(2):
        duplicate.zero_grad(set_to_none=True)
        gradient(duplicate, batches[2 * k:2 * k + 2], replace(spec.training, num_visits=1), totals, "cpu")
        groups.append([2 * torch.cat([p.grad.flatten() for c in range(v * 2, v * 2 + 2)
                                      for p in duplicate.core[c].experts.parameters()]).double() for v in range(2)])
    result = visit_statistics(model, batches, spec.training, totals, 2, "cpu")["experts"]
    mean = [(groups[0][v] + groups[1][v]) / 2 for v in range(2)]
    assert result["shared"] and result["norm"] == pytest.approx([float(m.norm()) for m in mean], rel=1e-5)
    assert result["cosine"][0][1] == pytest.approx(float(torch.dot(*mean) / (mean[0].norm() * mean[1].norm())), rel=1e-5)
    for series, value in [*enumerate(result["snr"]), ("sum", result["snr_sum"])]:
        g = [group[0] + group[1] if series == "sum" else group[series] for group in groups]
        m = (g[0] + g[1]) / 2
        noise = ((g[0] - m).square().sum() + (g[1] - m).square().sum()) / 2
        assert value == pytest.approx(float(m.square().sum() / noise), rel=1e-4)


def flat(tensors):
    return torch.cat([t.flatten() for t in tensors]).double()


def exact_hessian(model, batches, recipe, totals):
    """Double backward of L; routing is piecewise constant, so this is the frozen-routing Hessian."""
    parameters = list(model.parameters())
    with sdpa_kernel(SDPBackend.MATH):  # the CPU flash kernel has no double backward
        loss = 0
        for inputs, labels in batches:
            inputs, labels, kwargs = device_batch(inputs, labels, "cpu")
            output = model(inputs["input"], **kwargs, num_visits=recipe.num_visits)
            loss = loss + objective(output, labels, torch.tensor(totals[0]), recipe, accumulation=totals[1])[0]
        grads = torch.cat([g.flatten() for g in torch.autograd.grad(loss, parameters, create_graph=True)])
        rows = [flat(torch.autograd.grad(g, parameters, retain_graph=True, materialize_grads=True)) for g in grads]
    hessian = torch.stack(rows)
    return (hessian + hessian.T) / 2


def largest_magnitude(matrix):
    values = torch.linalg.eigvalsh(matrix)
    return float(values[values.abs().argmax()])


def test_power_iteration_matches_the_exact_top_eigenvalue_plain_and_preconditioned(manifest):
    spec = arm("m3", prefix_cells=0, num_cells=1, suffix_cells=0)
    model = RecursiveLM(spec.model).eval()
    batches, totals = sample(manifest)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):  # near a minimum the top eigenvalue is a clear outlier, as in trained checkpoints
        model.zero_grad(set_to_none=True)
        gradient(model, batches, spec.training, totals, "cpu")
        optimizer.step()
    hessian = exact_hessian(model, batches, spec.training, totals)
    root = [(torch.rand(p.shape) * 0.1 + 0.01).rsqrt() for p in model.parameters()]
    with frozen_routing(model) as routing:
        model.zero_grad(set_to_none=True)
        gradient(model, batches, spec.training, totals, "cpu")
        plain = top_eigenvalue(model, batches, spec.training, totals, routing, 20, 0, 1e-5, "cpu")
        scaled = top_eigenvalue(model, batches, spec.training, totals, routing, 20, 0, 1e-5, "cpu", root)
    assert len(routing["calls"]) == len(batches) * 2
    assert plain["lambda_max"] == pytest.approx(largest_magnitude(hessian), rel=0.05)
    r = flat(root)
    assert scaled["lambda_max"] == pytest.approx(largest_magnitude(r[:, None] * hessian * r[None, :]), rel=0.05)


def test_frozen_routing_replays_the_recorded_selection_and_keeps_the_loss(manifest):
    spec = arm("m3")
    model = RecursiveLM(spec.model).eval()
    batches, totals = sample(manifest)
    with frozen_routing(model) as routing:
        recorded = gradient(model, batches, spec.training, totals, "cpu")
        grads = flat(p.grad for p in model.parameters())
        model.zero_grad(set_to_none=True)
        routing["replay"] = 0
        assert gradient(model, batches, spec.training, totals, "cpu") == pytest.approx(recorded, rel=1e-6)
        torch.testing.assert_close(flat(p.grad for p in model.parameters()), grads, rtol=1e-5, atol=1e-9)
        assert routing["replay"] == len(routing["calls"])


def write_run(tmp_path, spec, model, optimizer_state: bool):
    """A run directory as recipe_experiments.py and the native trainer leave it, with one DCP step."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps(spec.to_dict()))
    (run / "request.json").write_text(json.dumps({"world_size": 8, "local_batch_size": 16,
        "gradient_accumulation_steps": 1, "manifest_fingerprint": "abc"}))
    (run / "model_config.json").write_text(json.dumps(spec.model.to_dict()))
    state = {"core." + name: value.detach().clone() for name, value in model.state_dict().items()}
    if optimizer_state:
        state["optimizer"] = {f"state.core.{name}.exp_avg_sq": torch.rand(value.shape) * 1e-4
                              for name, value in model.named_parameters()}
    dcp.save(state, checkpoint_id=str(run / "checkpoint" / "step-8"), no_dist=True)
    return run, state


@pytest.mark.parametrize("name,optimizer", [("m3", "adamw"), ("m4", "muon"), ("m0", "adamw")])
def test_checkpoint_run_writes_the_source_config_analysis_and_flat_metrics(tmp_path, manifest, monkeypatch, name, optimizer):
    spec = arm(name, optimizer)
    model = RecursiveLM(spec.model).eval()
    run, state = write_run(tmp_path, spec, model, optimizer == "adamw")
    run_dir = tmp_path / "RUN_DIR"
    run_dir.mkdir()
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    output = tmp_path / "out.json"
    visit_gradients.main(["--checkpoint", str(run / "checkpoint" / "step-8"), "--manifest", str(manifest.path),
                          "--output", str(output), "--sequences", "4", "--seq-len", str(SEQ_LEN), "--groups", "2",
                          "--power-iterations", "2"])
    config = json.loads((run_dir / "config.json").read_text())
    values = spec.to_dict()
    assert {key: config[key] for key in ("name", "model", "training", "optimizer", "inference", "run")} == {
        key: values[key] for key in ("name", "model", "training", "optimizer", "inference", "run")}
    assert config["launch"] == {"world_size": 8, "local_batch_size": 16, "gradient_accumulation_steps": 1,
                                "manifest_fingerprint": "abc"}
    assert {key: config["analysis"][key] for key in ("seed", "split", "sequences", "seq_len", "power_iterations", "step")} == {
        "seed": 0, "split": "validation", "sequences": 4, "seq_len": SEQ_LEN, "power_iterations": 2, "step": 8}
    result = json.loads(output.read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert all(isinstance(value, float) for value in metrics.values())
    assert metrics["lambda_max"] == result["curvature"]["lambda_max"] and len(result["curvature"]["rayleigh"]) == 2
    if name == "m3":
        assert set(result["visits"]) == {"attention", "router", "experts", "norm"} and result["untied_cosine"] is None
        assert metrics["experts.cos_1_2"] == result["visits"]["experts"]["cosine"][0][1]
    else:
        assert result["visits"] is None
        assert [row["offset"] for row in result["untied_cosine"].values()] == [len(model.core) // 2] * 2
        assert "untied.attention.cos_mean" in metrics
    if optimizer == "adamw":
        expected = [(state["optimizer"][f"state.core.{name}.exp_avg_sq"].sqrt() + spec.optimizer["eps"]).rsqrt()
                    for name, _ in model.named_parameters()]
        for loaded, value in zip(visit_gradients.adam_root(run / "checkpoint" / "step-8", model, spec.optimizer["eps"]),
                                 expected):
            torch.testing.assert_close(loaded, value)
        assert "lambda_max_preconditioned" in metrics
    else:
        assert result["preconditioned"] is None and "lambda_max_preconditioned" not in metrics


def export_run(tmp_path, spec, model):
    """``write_run`` with optimizer state, and its step exported to ``run/hf`` as scripts/export.py does."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from moeloop.export import export_checkpoint
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3} | {f"w{i}": i for i in range(4, 16)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"))
    transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>",
                                         eos_token="<eos>", unk_token="<unk>").save_pretrained(tmp_path / "tokenizer")
    run, _ = write_run(tmp_path, spec, model, optimizer_state=True)
    export_checkpoint(run / "checkpoint" / "step-8", run / "hf", spec.model, tmp_path / "tokenizer")
    return run


def routers(model):
    return [router for group in (model.prefix, model.core, model.suffix) for cell in group for router in cell.routers]


def test_hf_export_routers_all_get_the_frozen_routing_hook(tmp_path, manifest):
    spec = arm("m3")
    run = export_run(tmp_path, spec, RecursiveLM(spec.model).eval())
    model, _, _ = visit_gradients.load_model(None, run / "hf", manifest, "cpu")
    assert routers(model)
    with frozen_routing(model):
        assert [len(router._forward_hooks) for router in routers(model)] == [1] * len(routers(model))


def test_hf_and_checkpoint_paths_give_the_same_gradients_and_curvature(tmp_path, manifest):
    spec = arm("m3")
    run = export_run(tmp_path, spec, RecursiveLM(spec.model).eval())
    results = []
    for source in (["--checkpoint", str(run / "checkpoint" / "step-8")], ["--model", str(run / "hf")]):
        output = tmp_path / f"{source[0][2:]}.json"
        visit_gradients.main([*source, "--manifest", str(manifest.path), "--output", str(output), "--sequences", "4",
                              "--seq-len", str(SEQ_LEN), "--groups", "2", "--power-iterations", "3", "--fd-step", "0.1"])
        results.append(json.loads(output.read_text()))
    checkpoint, hf = results
    assert checkpoint["preconditioned"] is not None and hf["preconditioned"] is None
    for key in ("loss", "valid_labels", "visits", "curvature"):
        assert hf[key] == checkpoint[key]


@pytest.mark.parametrize("source", ["model", "hf"])
def test_replay_feeds_the_recorded_selection_to_the_experts_after_the_router_would_change_it(tmp_path, manifest, source):
    spec = arm("m3")
    model = RecursiveLM(spec.model).eval()
    if source == "hf":
        model, _, _ = visit_gradients.load_model(None, export_run(tmp_path, spec, model) / "hf", manifest, "cpu")
    batches, totals = sample(manifest)
    banks = [cell.experts for group in (model.prefix, model.core, model.suffix) for cell in group]
    dispatched = []
    handles = [bank.register_forward_pre_hook(lambda _, args: dispatched.append(args[1])) for bank in banks]

    def loss_and_ids(routing=None):
        """L and the expert ids each bank received, replaying ``routing`` when given."""
        dispatched.clear()
        if routing is not None:
            routing["replay"] = 0
        model.zero_grad(set_to_none=True)
        return gradient(model, batches, spec.training, totals, "cpu"), list(dispatched)

    with frozen_routing(model) as routing:
        _, recorded = loss_and_ids()  # replay is None: this pass records
        assert [ids for ids, _, _ in routing["calls"]] == recorded
        with torch.no_grad():
            for router in routers(model):
                router.gate.weight.add_(torch.randn_like(router.gate.weight))
        replayed_loss, replayed = loss_and_ids(routing)
    unfrozen_loss, unfrozen = loss_and_ids()
    for handle in handles:
        handle.remove()
    assert any(not torch.equal(a, b) for a, b in zip(unfrozen, recorded))  # the router's own top-k changed
    assert all(torch.equal(a, b) for a, b in zip(replayed, recorded)) and len(replayed) == len(recorded)
    assert replayed_loss != pytest.approx(unfrozen_loss, rel=1e-6)


def test_hf_export_run_has_no_preconditioned_curvature(tmp_path, manifest, monkeypatch):
    spec = arm("m7")
    run = export_run(tmp_path, spec, RecursiveLM(spec.model).eval())
    run_dir = tmp_path / "RUN_DIR"
    run_dir.mkdir()
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    output = tmp_path / "out.json"
    visit_gradients.main(["--model", str(run / "hf"), "--manifest", str(manifest.path), "--output", str(output),
                          "--sequences", "4", "--seq-len", str(SEQ_LEN), "--groups", "2", "--power-iterations", "2"])
    result = json.loads(output.read_text())
    assert result["analysis"]["step"] == 8 and result["preconditioned"] is None
    assert {component: row["shared"] for component, row in result["visits"].items()} == {
        "attention": False, "router": False, "experts": True, "norm": False}
    assert json.loads((run_dir / "config.json").read_text())["analysis"] == result["analysis"]
