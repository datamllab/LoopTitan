import math

import pytest

torch = pytest.importorskip("torch")

from moeloop.costs import estimate_cost
from moeloop.data import build_document_mask
from moeloop.model import ModelConfig


def model():
    return ModelConfig(vocab_size=17, dim=8, hidden_dim=12, num_heads=2,
                       num_kv_heads=1, num_experts=4, top_k=2,
                       num_cells=1, num_visits=2, max_seq_len=8)


def test_document_cost_agrees_with_real_mask_and_preserves_non_attention_work():
    ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.int64)
    mask = build_document_mask(ids, torch.ones_like(ids, dtype=torch.bool))
    pairs = int(mask.sum())
    assert pairs == 6
    dense = estimate_cost(model(), seq_len=4, activation_checkpointing='full_cells')
    logical = estimate_cost(model(), seq_len=4, activation_checkpointing='full_cells',
                            attention_pairs_per_sequence=pairs)
    assert logical['formula_version'] == 'fixed_terminal_document_matrix_v1'
    assert dense['formula_version'] == 'fixed_terminal_matrix_v1'
    assert logical['parameters'] == dense['parameters']
    assert logical['kv'] == dense['kv']
    forward = logical['flops']['forward_components']
    assert forward['attention_scores_values'] == 4 * pairs / 4 * 8 * 2
    assert forward['head'] == dense['flops']['forward_components']['head']
    # Full backward is two matrix products and checkpointing adds one replay.
    removed = dense['flops']['forward_components']['attention_scores_values'] - forward['attention_scores_values']
    assert dense['flops']['training_per_token'] - logical['flops']['training_per_token'] == 4 * removed
    assert 'measured_flops' in logical['unknown']


def test_fractional_mean_pairs_can_describe_multiple_packed_sequences():
    result = estimate_cost(model(), seq_len=4, attention_pairs_per_sequence=7.5)
    assert result['flops']['attention_pairs_per_sequence'] == 7.5
    assert result['flops']['forward_components']['attention_scores_values'] == 120


@pytest.mark.parametrize('pairs', [True, -1, math.nan, math.inf, 11, '6'])
def test_invalid_causal_pair_counts_are_rejected(pairs):
    with pytest.raises(ValueError, match='attention_pairs_per_sequence'):
        estimate_cost(model(), seq_len=4, attention_pairs_per_sequence=pairs)
