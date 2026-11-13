"""Closed-form checks for forward-local normalized exponential memories."""

import math

import pytest

torch = pytest.importorskip("torch")


def test_first_write_is_unbiased_and_keeps_its_gradient():
    from moeloop.memory import EMACarry, ema_update

    value = torch.tensor([[2.0, -3.0]], requires_grad=True)
    empty = EMACarry()
    carry, read = ema_update(empty, value)
    torch.testing.assert_close(read, value)
    assert carry.denominator == 1.0
    assert empty.numerator is None and empty.denominator == 0.0
    read.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


def test_two_memories_use_all_visits_and_current_visit_respectively():
    from moeloop.memory import EMACarry, ema_update

    writes = tuple(torch.tensor([float(i), float(7 - i)], requires_grad=True)
                   for i in range(1, 7))
    global_carry = EMACarry()
    for visit in range(3):
        local_carry = EMACarry()
        for write in writes[2 * visit:2 * visit + 2]:
            global_carry, global_read = ema_update(global_carry, write)
            local_carry, local_read = ema_update(local_carry, write)

    global_weights = torch.tensor([.5 ** i for i in range(5, -1, -1)])
    local_weights = torch.tensor([.5, 1.0])
    expected_global = (torch.stack(writes) * global_weights[:, None]).sum(0) / global_weights.sum()
    expected_local = (torch.stack(writes[-2:]) * local_weights[:, None]).sum(0) / local_weights.sum()
    torch.testing.assert_close(global_read, expected_global)
    torch.testing.assert_close(local_read, expected_local)
    assert global_carry.denominator == global_weights.sum().item()
    assert local_carry.denominator == local_weights.sum().item()

    # The two summaries add independently; they are not averaged or detached.
    actual_grad = torch.autograd.grad((global_read + local_read).sum(), writes)
    expected_grad = torch.autograd.grad((expected_global + expected_local).sum(), writes)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected)
    assert (actual_grad[0] > 0).all()


def test_reset_and_new_carry_do_not_mutate_previous_forward():
    from moeloop.memory import EMACarry, ema_update

    first, _ = ema_update(EMACarry(), torch.tensor([3.0]))
    previous = first.numerator.clone()
    continued, _ = ema_update(first, torch.tensor([9.0]))
    fresh, fresh_read = ema_update(EMACarry(), torch.tensor([-4.0]))
    torch.testing.assert_close(first.numerator, previous)
    assert first.denominator == 1.0 and continued.denominator == 1.5
    torch.testing.assert_close(fresh_read, torch.tensor([-4.0]))
    assert fresh.denominator == 1.0


def test_zero_decay_discards_history_and_preserves_write_dtype():
    from moeloop.memory import EMACarry, ema_update

    carry, _ = ema_update(EMACarry(), torch.tensor([3.0], dtype=torch.bfloat16), beta=0)
    carry, read = ema_update(carry, torch.tensor([7.0], dtype=torch.bfloat16), beta=0)
    assert carry.denominator == 1.0 and read.dtype == torch.bfloat16
    torch.testing.assert_close(read, torch.tensor([7.0], dtype=torch.bfloat16))


def test_gains_follow_geometry_and_one_based_visit():
    from moeloop.memory import reinjection_gain, residual_gain

    assert residual_gain(num_cells=2, num_visits=3) == pytest.approx(.5 / (3 * math.sqrt(2)))
    assert reinjection_gain(num_cells=2, visit=2) == pytest.approx(.5 / (2 * math.sqrt(2)))
    assert reinjection_gain(num_cells=2, visit=3) == pytest.approx(.5 / (3 * math.sqrt(2)))
    assert residual_gain(num_cells=2, num_visits=3, scale=0) == 0


def test_invalid_carry_cannot_silently_broadcast_or_normalize():
    from moeloop.memory import EMACarry, ema_update

    for carry in (EMACarry(None, 1.0), EMACarry(torch.ones(2), 0.0),
                  EMACarry(torch.ones(1), 1.0)):
        with pytest.raises(ValueError, match="carry"):
            ema_update(carry, torch.ones(2))


def test_first_visit_has_no_reinjection_gain():
    from moeloop.memory import reinjection_gain

    with pytest.raises(ValueError, match="visit"):
        reinjection_gain(num_cells=2, visit=1)
