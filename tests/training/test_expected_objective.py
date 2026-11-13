"""Independent dense-logits oracles for tokenwise expected exit losses."""

import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn
import torch.nn.functional as F

from moeloop.halting import expected_lm_loss, hazard_probabilities
from moeloop.training import TrainingRecipe


class Readouts(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(5, 5)
        self.gate = nn.Linear(5, 1)
        self.head = nn.Sequential(nn.Linear(5, 9), nn.Tanh())

    def forward(self, inputs, visits):
        hidden = []
        state = inputs
        for _ in range(visits):
            state = torch.tanh(self.backbone(state))
            hidden.append(state)
        return tuple(hidden), torch.stack([self.gate(h).squeeze(-1) for h in hidden])


def dense_reference(hidden, gate_logits, head, labels, denominator, beta):
    """Literal product formula, materializing all logits for this tiny oracle."""
    survival = torch.ones_like(gate_logits[0], dtype=torch.float32)
    probabilities = []
    for visit in range(len(hidden) - 1):
        stop = torch.sigmoid(gate_logits[visit].float())
        probabilities.append(survival * stop)
        survival = survival * (1 - stop)
    probabilities.append(survival)
    p = torch.stack(probabilities)
    ce = torch.stack([
        F.cross_entropy(head(h).flatten(0, 1).float(), labels.flatten(),
                        reduction="none", ignore_index=-100).reshape_as(labels)
        for h in hidden
    ])
    valid = labels != -100
    entropy = -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(0)
    expected_ce = (p * ce).sum() / denominator
    entropy = entropy[valid].sum() / denominator
    return expected_ce - beta * entropy, {
        "ce": ce[-1].sum() / denominator,
        "expected_ce": expected_ce,
        "entropy": entropy,
        "mean_exit_probabilities": (p * valid).sum((1, 2)) / denominator,
    }


def fixture():
    torch.manual_seed(17)  # Unit oracle fixture, not another training replicate.
    model = Readouts()
    inputs = torch.linspace(-1.1, 1.2, 40).reshape(2, 4, 5)
    labels = torch.tensor([[0, 2, -100, 4], [8, -100, 1, 3]])
    return model, inputs, labels


def test_hazards_are_tokenwise_and_final_probability_is_residual():
    logits = torch.logit(torch.tensor([[[0.2, 0.8]], [[0.3, 0.6]], [[0.9, 0.1]]])).requires_grad_()
    expected = torch.tensor([[[0.2, 0.8]], [[0.24, 0.12]], [[0.56, 0.08]]])
    p = hazard_probabilities(logits)
    torch.testing.assert_close(p, expected)
    torch.testing.assert_close(p.sum(0), torch.ones_like(p[0]))
    changed_final = logits.detach().clone()
    changed_final[-1] = torch.tensor([[-1000.0, 1000.0]])
    torch.testing.assert_close(hazard_probabilities(changed_final), p)
    (p * torch.arange(3).reshape(3, 1, 1)).sum().backward()
    assert logits.grad[:-1].abs().sum() > 0
    assert torch.equal(logits.grad[-1], torch.zeros_like(logits.grad[-1]))


@pytest.mark.parametrize("chunks", [1, 3, 19])
def test_checkpointed_expected_loss_matches_independent_full_autograd(chunks):
    actual, inputs, labels = fixture()
    reference = copy.deepcopy(actual)
    x = inputs.clone().requires_grad_()
    y = inputs.clone().requires_grad_()
    hidden, gates = actual(x, 3)
    expected_hidden, expected_gates = reference(y, 3)
    denominator = torch.tensor(19)  # A whole-update/global denominator, not this local count.
    loss, metrics = expected_lm_loss(hidden, gates, actual.head, labels, denominator,
                                     entropy_coeff=0.17, num_chunks=chunks)
    expected, expected_metrics = dense_reference(expected_hidden, expected_gates, reference.head,
                                                 labels, denominator, 0.17)
    torch.testing.assert_close(loss, expected, rtol=2e-6, atol=2e-7)
    for name, value in expected_metrics.items():
        torch.testing.assert_close(metrics[name], value.detach(), rtol=2e-6, atol=2e-7)
        assert not metrics[name].requires_grad
    loss.backward()
    expected.backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=2e-5, atol=2e-7)
    for (name, parameter), (_, other) in zip(actual.named_parameters(), reference.named_parameters()):
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name
        torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-5, atol=2e-7, msg=name)


def test_uneven_microbatches_use_one_global_label_denominator():
    full, inputs, labels = fixture()
    split = copy.deepcopy(full)
    labels[0, 1:] = -100
    denominator = (labels != -100).sum()
    hidden, gates = full(inputs, 3)
    total, _ = expected_lm_loss(hidden, gates, full.head, labels, denominator, num_chunks=3)
    total.backward()
    for start, end in ((0, 1), (1, 2)):
        hidden, gates = split(inputs[start:end], 3)
        local, _ = expected_lm_loss(hidden, gates, split.head, labels[start:end], denominator, num_chunks=2)
        local.backward()
    for (name, p), (_, q) in zip(full.named_parameters(), split.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=3e-7, msg=name)


def test_one_visit_is_terminal_ce_with_zero_entropy():
    model, inputs, labels = fixture()
    hidden, gates = model(inputs, 1)
    loss, metrics = expected_lm_loss(hidden, gates, model.head, labels,
                                     (labels != -100).sum(), entropy_coeff=30.0, num_chunks=3)
    expected = F.cross_entropy(model.head(hidden[0]).flatten(0, 1).float(), labels.flatten(),
                               reduction="mean", ignore_index=-100)
    torch.testing.assert_close(loss, expected)
    assert metrics["entropy"].item() == 0
    torch.testing.assert_close(metrics["mean_exit_probabilities"], torch.ones(1))
    loss.backward()
    assert model.gate.weight.grad is None or torch.equal(model.gate.weight.grad, torch.zeros_like(model.gate.weight))


def test_extreme_hazards_have_finite_objective_and_gradients():
    model, inputs, labels = fixture()
    hidden, _ = model(inputs, 4)
    gates = torch.tensor([[[1000., -1000., 1000., -1000.]],
                          [[-1000., 1000., -1000., 1000.]],
                          [[1000., -1000., -1000., 1000.]],
                          [[1000., 1000., 1000., 1000.]]]).expand(4, 2, 4).clone().requires_grad_()
    p = hazard_probabilities(gates)
    torch.testing.assert_close(p.sum(0), torch.ones_like(p[0]))
    loss, metrics = expected_lm_loss(hidden, gates, model.head, labels,
                                     (labels != -100).sum(), num_chunks=3)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["entropy"])
    assert torch.isfinite(gates.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_empty_local_labels_contribute_zero_with_positive_global_count():
    model, inputs, labels = fixture()
    hidden, gates = model(inputs, 3)
    loss, metrics = expected_lm_loss(hidden, gates, model.head, torch.full_like(labels, -100),
                                     torch.tensor(7), num_chunks=3)
    loss.backward()
    assert loss.item() == 0
    assert all(torch.count_nonzero(v) == 0 for v in metrics.values())
    assert all(torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_head_parameters_train_without_hidden_input_gradients():
    model, inputs, labels = fixture()
    hidden = tuple(torch.full_like(inputs, value) for value in (0.3, 0.7))
    gates = torch.zeros((2, *labels.shape))
    loss, _ = expected_lm_loss(hidden, gates, model.head, labels, (labels != -100).sum(), num_chunks=3)
    loss.backward()
    assert model.head[0].weight.grad.abs().sum() > 0


def test_bfloat16_autocast_keeps_probability_and_loss_math_float32():
    model, inputs, labels = fixture()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        hidden, gates = model(inputs, 3)
        loss, metrics = expected_lm_loss(hidden, gates, model.head, labels,
                                         (labels != -100).sum(), num_chunks=3)
    assert hazard_probabilities(gates).dtype == torch.float32
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("denominator", [0, -1])
def test_invalid_global_denominator_is_rejected(denominator):
    model, inputs, labels = fixture()
    hidden, gates = model(inputs, 3)
    with pytest.raises(ValueError, match="global.*label"):
        expected_lm_loss(hidden, gates, model.head, labels, torch.tensor(denominator))


def test_mismatched_visit_and_token_shapes_are_rejected():
    model, inputs, labels = fixture()
    hidden, gates = model(inputs, 3)
    with pytest.raises(ValueError, match="hazard"):
        expected_lm_loss(hidden, gates[:2], model.head, labels, torch.tensor(6))
    with pytest.raises(ValueError, match="hidden"):
        expected_lm_loss((hidden[0][:, :-1], *hidden[1:]), gates, model.head, labels, torch.tensor(6))
    with pytest.raises(ValueError, match="num_chunks"):
        expected_lm_loss(hidden, gates, model.head, labels, torch.tensor(6), num_chunks=0)


# expected_lm_loss does not recheck entropy_coeff; TrainingRecipe rejects it.
@pytest.mark.parametrize("entropy_coeff", [-.1, float("nan"), float("inf")])
def test_training_recipe_rejects_invalid_entropy_coefficient(entropy_coeff):
    with pytest.raises(ValueError, match="entropy_coeff"):
        TrainingRecipe(num_visits=3, objective="expected_ce", entropy_coeff=entropy_coeff)
