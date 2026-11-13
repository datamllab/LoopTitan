import random
import pytest

torch = pytest.importorskip("torch")

from moeloop.model import ModelConfig, RecursiveLM
from moeloop.training import (
    CommittedProgress,
    TrainingRecipe,
    continuation_fingerprint,
    isolated_rng,
    objective,
    updates_for_token_budget,
)


def tiny_config(**kwargs):
    return ModelConfig(vocab_size=19, dim=16, hidden_dim=24, num_heads=2, num_kv_heads=1, num_experts=2, top_k=2, num_cells=1, num_visits=2, **kwargs)


def test_update_depth_sampling_is_stateless_and_restarts_at_committed_update():
    recipe = TrainingRecipe(num_visits=4, sampled_visits=(2, 4, 6), grad_visits=2, depth_seed=91)
    before = random.getstate()
    sequence = [recipe.depth_at(step) for step in range(60)]
    assert random.getstate() == before
    random.seed(123)
    assert [recipe.depth_at(step) for step in range(23, 60)] == sequence[23:]
    assert set(sequence) == {2, 4, 6}


@pytest.mark.parametrize("kwargs", [{"num_visits": 0}, {"sampled_visits": (2, 2)}, {"sampled_visits": (2, 4), "grad_visits": 3}, {"balance_coeff": float("nan")}, {"z_loss_coeff": -1}])
def test_invalid_training_controls_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        TrainingRecipe(**kwargs)


def test_first_release_owner_and_gradient_support_is_checked():
    with pytest.raises(ValueError, match="per-visit owners"):
        TrainingRecipe(num_visits=2, sampled_visits=(1, 2)).validate_model(tiny_config(attention_per_visit=True))
    with pytest.raises(ValueError, match="stack"):
        TrainingRecipe(num_visits=2, grad_visits=1).validate_model(tiny_config(order="layer"))
    TrainingRecipe(num_visits=2).validate_model(tiny_config(attention_per_visit=True))
    TrainingRecipe(num_visits=2, grad_visits=1).validate_model(tiny_config())


def test_global_label_normalization_matches_uneven_microbatch_gradients():
    torch.manual_seed(14)
    model = RecursiveLM(tiny_config()).double()
    copy = RecursiveLM(tiny_config()).double()
    copy.load_state_dict(model.state_dict())
    tokens = torch.randint(0, 19, (3, 4))
    labels = torch.randint(0, 19, (3, 4))
    labels[0, 2:] = -100
    labels[1, 3:] = -100
    denominator = (labels != -100).sum()
    recipe = TrainingRecipe(num_visits=2, balance_coeff=0)
    full, _ = objective(model(tokens), labels, denominator, recipe)
    full.backward()
    for start, end in ((0, 1), (1, 3)):
        loss, _ = objective(copy(tokens[start:end]), labels[start:end], denominator, recipe, accumulation=2)
        loss.backward()
    for (name, parameter), (_, other) in zip(model.named_parameters(), copy.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-5, atol=3e-7, msg=name)


def test_auxiliary_is_call_mean_then_rank_micro_mean_and_eval_is_raw_ce():
    model = RecursiveLM(tiny_config())
    tokens = torch.randint(0, 19, (2, 4))
    labels = tokens.clone()
    labels[0, 0] = -100
    output = model(tokens)
    recipe = TrainingRecipe(num_visits=2, balance_coeff=0.3, z_loss_coeff=0.2)
    total, metrics = objective(output, labels, (labels != -100).sum(), recipe, dp_degree=3, accumulation=2)
    expected = (0.3 * output.balance_sum + 0.2 * output.z_sum) / (output.aux_calls * 3 * 2)
    torch.testing.assert_close(total - metrics["ce"], expected)
    evaluation, evaluation_metrics = objective(output, labels, None, recipe)
    torch.testing.assert_close(evaluation, metrics["ce"] * (labels != -100).sum())
    assert set(evaluation_metrics) == {"ce"}
    with pytest.raises(ValueError, match="valid label"):
        objective(output, labels, torch.tensor(0), recipe)


def test_validation_rng_isolation_including_failure():
    random.seed(12)
    torch.manual_seed(12)
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    with pytest.raises(RuntimeError):
        with isolated_rng():
            random.random()
            torch.rand(8)
            raise RuntimeError("validation failed")
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_committed_work_and_budget_never_round_or_count_failed_attempts():
    progress = CommittedProgress()
    progress.commit(padded_positions=16, input_tokens=13, label_tokens=11, recursive_input_tokens=52)
    assert progress.completed_updates == 1
    assert progress.input_tokens == 13
    with pytest.raises(ValueError):
        progress.commit(padded_positions=16, input_tokens=-1, label_tokens=11, recursive_input_tokens=52)
    assert progress.completed_updates == 1
    assert updates_for_token_budget(128, seq_len=8, global_batch_size=4) == 4
    with pytest.raises(ValueError):
        updates_for_token_budget(129, seq_len=8, global_batch_size=4)
    assert continuation_fingerprint({"a": 1, "b": 2}) == continuation_fingerprint({"b": 2, "a": 1})
    assert continuation_fingerprint({"a": 1}) != continuation_fingerprint({"a": 2})


def test_host_rng_seed_is_reproducible_per_rank_without_resetting_torch():
    from moeloop.training import seed_host_rng
    np = pytest.importorskip("numpy")
    with isolated_rng():
        before = torch.get_rng_state().clone()
        seed_host_rng(42, rank=0)
        expected = (random.random(), np.random.rand())
        seed_host_rng(42, rank=0)
        assert expected == (random.random(), np.random.rand())
        seed_host_rng(42, rank=1)
        assert expected != (random.random(), np.random.rand())
        assert torch.equal(torch.get_rng_state(), before)
