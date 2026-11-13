"""CPU checks for bounded job inputs and actual native DCP receipt reading."""

import json
import pickle

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

import torch.distributed.checkpoint as dcp

from qualification import compare_runs, load_comparison_state, prepare_fixture, verify_receipt
from moeloop.data import PackedDataset, PreparedManifest


def test_offline_fixture_is_loadable_and_refuses_overwriting(tmp_path):
    output = tmp_path / "data"
    result = prepare_fixture(output)
    manifest = PreparedManifest.load(output)
    assert result["vocab_size"] == manifest.tokenizer["vocab_size"] == 64
    assert result["train_documents"] == 128
    assert manifest.payload["provenance"]["contains_external_data"] is False
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(output / "tokenizer", local_files_only=True)
    assert len(tokenizer) == 64
    assert tokenizer.bos_token_id == 1 and tokenizer.eos_token_id == 2
    for rank in range(4):
        dataset = PackedDataset(output / "manifest.json", split="train", seq_len=32,
            shuffle=True, infinite=True, seed=42, rank=rank, world_size=4)
        inputs, labels = next(iter(dataset))
        assert inputs["input"].shape == labels.shape == (32,)
        assert inputs["input"].max() < 64 and (labels != -100).any()
        dataset.close()
    with pytest.raises(FileExistsError):
        prepare_fixture(output)


def test_native_recipes_resolve_six_variants_without_starting_a_trainer(tmp_path, monkeypatch):
    pytest.importorskip("torchtitan")
    from qualification_recipes import VARIANTS, smoke
    from torchtitan.distributed.activation_checkpoint import FullAC
    prepared = tmp_path / "data"
    prepare_fixture(prepared)
    monkeypatch.setenv("MOELOOP_DATA", str(prepared))
    monkeypatch.setenv("MOELOOP_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("MOELOOP_STEPS", "2")
    for name in VARIANTS:
        monkeypatch.setenv("MOELOOP_QUAL_VARIANT", name)
        config = smoke()
        config.model_spec.model.update_from_config(config=config)
        assert config.training.steps == 2 and config.lr_scheduler.total_steps == 16
        assert config.training.global_batch_size == 8 and config.training.local_batch_size == 1
        assert config.training.seq_len == 32 and config.loss.num_chunks == 8
        assert config.training.mixed_precision_param == "bfloat16"
        assert config.model_spec.model.native_experts and config.model_spec.model.flex_attention
        assert not config.compile.enable and config.training.disable_cuda_graphs
        assert config.checkpoint.interval == config.validator.freq == config.validator.global_batches == 1
        assert not config.checkpoint.last_save_model_only
        assert isinstance(config.activation_checkpoint, FullAC.Config) == (name in ("partial_attention", "injected"))
        if name == "injected":
            assert [config.recipe.depth_at(i) for i in range(2)] == [3, 2]
            assert config.recipe.grad_visits == 1
    monkeypatch.setenv("MOELOOP_STEPS", "4")
    monkeypatch.setenv("MOELOOP_RESUME", str(tmp_path / "old" / "checkpoint" / "step-2"))
    resumed = smoke()
    assert resumed.training.steps == 4 and resumed.lr_scheduler.total_steps == 16
    assert resumed.checkpoint.initial_load_path.endswith("/old/checkpoint/step-2")
    monkeypatch.setenv("MOELOOP_STEPS", "1000")
    with pytest.raises(ValueError, match="must be 2 or 4"):
        smoke()


def _write_native_checkpoint(directory, *, weight_offset=0.0, cursor=8, completed=4):
    checkpoint = directory / "checkpoint" / "step-4"
    progress = {"completed_updates": completed, "input_tokens": 128, "label_tokens": 128,
                "padded_positions": 128, "recursive_input_tokens": 512}
    payload = {"schema": 1, "identity": {"world_size": 1}, "fingerprint": "qualified-fixture",
               "progress": progress, "rng": {"torch": torch.tensor([1, 2, 3], dtype=torch.uint8)}}
    state = {
        # Same flat-model/nested-aux grammar as TorchTitan's DCP manager.
        "core.embedding.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) + weight_offset,
        "core.final_norm.weight": torch.ones(4),
        "train_state": {"step": 4, "ntokens_seen": 128, "rank_0": pickle.dumps(payload)},
        "dataloader": {"world_size": 1, "dp_rank_0": pickle.dumps({"dataset_state": {"cursor": cursor}})},
        "lr_scheduler": {"last_epoch": 4, "_step_count": 5},
        "optimizer": {"state": {"large_unused_moment": torch.zeros(23)}},
    }
    dcp.save(state, checkpoint_id=checkpoint, no_dist=True)
    outcome = {"status": "completed", "completed_updates": 4, "global_progress": progress,
               "global_source_counts": {"engineering": {"blocks": 8}},
               "full_checkpoint": str(checkpoint), "continuation_fingerprint": "qualified-fixture"}
    (directory / "outcome.json").write_text(json.dumps(outcome))
    return checkpoint


def test_native_dcp_comparison_reads_model_and_rank_cursors_without_optimizer(tmp_path):
    first, second = tmp_path / "uninterrupted", tmp_path / "resumed"
    checkpoint = _write_native_checkpoint(first)
    _write_native_checkpoint(second)
    selected = load_comparison_state(checkpoint)
    assert not any(key.startswith("optimizer") for key in selected)
    assert selected["train_state.step"] == 4
    report = compare_runs(first, second)
    assert report["status"] == "passed" and report["max_absolute_model_error"] == 0
    assert report["model_tensor_count"] == 2
    assert "dataloader.dp_rank_0" in report["exact_continuation_keys"]
    (checkpoint / ".metadata").unlink()
    with pytest.raises(AssertionError, match="metadata"):
        verify_receipt(first, expected_steps=4)


def test_comparison_rejects_changed_cursor_and_model(tmp_path):
    first, cursor_changed, weight_changed = tmp_path / "one", tmp_path / "cursor", tmp_path / "weight"
    _write_native_checkpoint(first)
    _write_native_checkpoint(cursor_changed, cursor=9)
    _write_native_checkpoint(weight_changed, weight_offset=0.1)
    with pytest.raises(AssertionError, match="cursor"):
        compare_runs(first, cursor_changed)
    with pytest.raises(AssertionError, match="model state"):
        compare_runs(first, weight_changed)
