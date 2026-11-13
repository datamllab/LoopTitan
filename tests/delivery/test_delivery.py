from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from moeloop.data import PackedDataset, PreparedManifest
from moeloop.evaluate import _check_tokenizer_identity, evaluate_heldout, lm_eval_command
from moeloop.export import export_checkpoint, resolve_model_config
from moeloop.generate import generate_text
from moeloop.model import ModelConfig, RecursiveLM
from moeloop.prepare import main as prepare_main, prepare_dataset


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(92)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return ModelConfig(vocab_size=17, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2, num_experts=4, top_k=2, num_cells=1, num_visits=2, max_seq_len=32)


@pytest.fixture
def tokenizer_path(tmp_path):
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocabulary.update({f"w{i}": i for i in range(4, 17)})
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
    directory = tmp_path / "tokenizer"
    tokenizer.save_pretrained(directory)
    return directory


def state_layout(state, layout):
    if layout == "native_grouped":
        return {"core." + key.replace(".experts.", ".experts.routed.inner_experts."): value.clone() for key, value in state.items()}
    if layout == "wrapped_reference":
        return {"core." + key: value.clone() for key, value in state.items()}
    if layout == "huggingface":
        return {"model." + key: value.clone() for key, value in state.items()}
    return {key: value.clone() for key, value in state.items()}


def save_dcp(path, state):
    import torch.distributed.checkpoint as dcp
    dcp.save({**state, "optimizer": {"moments": torch.ones(1024)}, "data": {"cursor": b"local cursor"}, "train_state": {"step": 7}}, checkpoint_id=path, no_dist=True)


@pytest.mark.parametrize("layout", ["reference", "native_grouped", "wrapped_reference", "huggingface"])
def test_model_only_weights_export_strictly_and_match_core(tmp_path, config, tokenizer_path, layout):
    previously_imported = set(sys.modules)
    model = RecursiveLM(config).eval()
    checkpoint = tmp_path / "weights.pt"
    torch.save(state_layout(model.state_dict(), layout), checkpoint)
    output = export_checkpoint(checkpoint, tmp_path / "artifact", config, tokenizer_path)
    loaded = transformers.AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True, local_files_only=True).eval()
    ids = torch.tensor([[1, 4, 5, 6]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids, use_cache=False).logits, model(ids).logits)
    provenance = json.loads((output / "export_provenance.json").read_text())
    assert provenance["checkpoint"]["key_layout"] == layout
    assert provenance["artifact_sha256"]["model.safetensors"]
    license_bytes = (output / "LICENSE").read_bytes()
    assert license_bytes == (Path(__file__).resolve().parents[2] / "LICENSE").read_bytes()
    assert provenance["artifact_sha256"]["LICENSE"] == hashlib.sha256(license_bytes).hexdigest()
    assert all((output / name).is_file() for name in ("model.py", "hf.py", "moe.py", "cache.py", "generation_config.json", "tokenizer.json"))
    assert not ({"torchtitan", "mlflow"} & (set(sys.modules) - previously_imported))


@pytest.mark.parametrize("license_available", [True, False])
def test_export_license_from_installed_metadata_or_clean_failure(tmp_path, config, tokenizer_path, monkeypatch, license_available):
    import moeloop.export as exporter

    license_text = (Path(__file__).resolve().parents[2] / "LICENSE").read_text(encoding="utf-8")
    original_distribution = exporter.importlib.metadata.distribution
    requested = []

    class InstalledDistribution:
        def read_text(self, filename):
            requested.append(filename)
            return license_text if license_available else None

    def distribution(name):
        return InstalledDistribution() if name == "moeloop-lab" else original_distribution(name)

    monkeypatch.setattr(exporter, "__file__", str(tmp_path / "site-packages" / "moeloop" / "export.py"))
    monkeypatch.setattr(exporter.importlib.metadata, "distribution", distribution)
    checkpoint = tmp_path / "weights.pt"
    torch.save(RecursiveLM(config).state_dict(), checkpoint)
    output = tmp_path / "artifact"
    if license_available:
        export_checkpoint(checkpoint, output, config, tokenizer_path)
        assert (output / "LICENSE").read_text(encoding="utf-8") == license_text
    else:
        with pytest.raises(RuntimeError, match="LICENSE is missing"):
            export_checkpoint(checkpoint, output, config, tokenizer_path)
        assert not output.exists()
    assert requested == ["licenses/LICENSE"]


def test_native_flat_dcp_loads_only_model_and_discovers_metadata(tmp_path, config, tokenizer_path, monkeypatch):
    import torch.distributed.checkpoint as dcp
    model = RecursiveLM(config).eval()
    root = tmp_path / "training"
    checkpoint = root / "checkpoint" / "step-7"
    state = state_layout(model.state_dict(), "native_grouped")
    save_dcp(checkpoint, state)
    (root / "model_config.json").write_text(json.dumps(config.to_dict()))
    (root / "resolved_config.json").write_text(json.dumps({"hf_assets_path": str(tokenizer_path)}))
    requested = []
    original = dcp.FileSystemReader.read_data

    def read_data(reader, plan, planner):
        requested.extend(item.storage_index.fqn for item in plan.items)
        return original(reader, plan, planner)

    monkeypatch.setattr(dcp.FileSystemReader, "read_data", read_data)
    output = export_checkpoint(checkpoint, tmp_path / "artifact")
    assert requested and set(requested) == set(state)
    assert all(key.startswith("core.") for key in requested)
    loaded = transformers.AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True, local_files_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[name])


@pytest.mark.parametrize("problem", ["missing", "extra", "shape"])
def test_native_dcp_refuses_incomplete_or_different_model(tmp_path, config, tokenizer_path, problem):
    state = state_layout(RecursiveLM(config).state_dict(), "native_grouped")
    if problem == "missing":
        state.pop(next(iter(state)))
    elif problem == "extra":
        state["core.unexpected.weight"] = torch.ones(2)
    else:
        state["core.embedding.weight"] = torch.ones(2)
    checkpoint = tmp_path / "checkpoint"
    save_dcp(checkpoint, state)
    with pytest.raises(ValueError, match="keys|shape"):
        export_checkpoint(checkpoint, tmp_path / "artifact", config, tokenizer_path)
    assert not (tmp_path / "artifact").exists()


def test_export_config_inputs_depth_override_and_no_silent_overwrite(tmp_path, config, tokenizer_path):
    checkpoint = tmp_path / "weights.pt"
    torch.save(RecursiveLM(config).state_dict(), checkpoint)
    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(json.dumps({"model_spec": {"model": {"core": config.to_dict()}}}))
    assert resolve_model_config(checkpoint) == config
    assert resolve_model_config(checkpoint, resolved) == config
    assert resolve_model_config(checkpoint, config.to_dict()) == config
    output = export_checkpoint(checkpoint, tmp_path / "artifact", tokenizer_path=tokenizer_path, num_visits=3)
    assert json.loads((output / "config.json").read_text())["model_config"]["num_visits"] == 3
    with pytest.raises(FileExistsError):
        export_checkpoint(checkpoint, output, config, tokenizer_path)
    independent = replace(config, attention_per_visit=True)
    torch.save(RecursiveLM(independent).state_dict(), checkpoint)
    with pytest.raises(ValueError, match="visit_capacity"):
        export_checkpoint(checkpoint, tmp_path / "too-deep", independent, tokenizer_path, num_visits=3)


def test_torch_input_refuses_full_training_mapping_and_extra_parameters(tmp_path, config, tokenizer_path):
    checkpoint = tmp_path / "weights.pt"
    state = RecursiveLM(config).state_dict()
    torch.save({"model": state, "optimizer": {"anything": 1}}, checkpoint)
    with pytest.raises(ValueError, match="only a model"):
        export_checkpoint(checkpoint, tmp_path / "full", config, tokenizer_path)
    torch.save({**state, "surprise": torch.ones(1)}, checkpoint)
    with pytest.raises(ValueError, match="unexpected"):
        export_checkpoint(checkpoint, tmp_path / "extra", config, tokenizer_path)


def test_closest_resolved_config_precedes_a_more_distant_model_config(tmp_path, config):
    training = tmp_path / "training"
    checkpoint = training / "checkpoint"
    checkpoint.mkdir(parents=True)
    (training / "resolved_config.json").write_text(json.dumps({"model_spec": {"model": {"core": config.to_dict()}}}))
    (tmp_path / "model_config.json").write_text(json.dumps(replace(config, num_visits=7).to_dict()))
    assert resolve_model_config(checkpoint) == config


def prepared_fixture(path, tokenizer_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return prepare_dataset(path, {"text": {"validation": ["w4 w5", "w6"]}}, encode=lambda text: tokenizer.encode(text, add_special_tokens=False), tokenizer={"id": "test", "revision": "fixed-v1", "vocab_size": len(tokenizer), "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}, tokenizer_directory=tokenizer_path)


def test_heldout_uses_document_local_aligned_targets_and_preserves_mode_rng(tmp_path, config, tokenizer_path):
    model = RecursiveLM(config).train()
    manifest = prepared_fixture(tmp_path / "data", tokenizer_path)
    with torch.no_grad():
        manual = sum(torch.nn.functional.cross_entropy(model(torch.tensor([inputs])).logits.flatten(0, 1).float(), torch.tensor(labels), reduction="sum").item() for inputs, labels in (([1, 4, 5], [4, 5, 2]), ([1, 6], [6, 2])))
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    result = evaluate_heldout(model, manifest, seq_len=6, batch_size=2)
    assert result["complete_split"] is True
    assert result["valid_tokens"] == result["input_tokens"] == 5
    assert result["ce_sum"] == pytest.approx(manual, abs=2e-6)
    assert result["ce"] == pytest.approx(manual / 5, abs=1e-6)
    assert result["perplexity"] == pytest.approx(math.exp(manual / 5), rel=1e-6)
    assert "document_losses" not in result
    assert model.training
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert python_rng == random.getstate()
    partial = evaluate_heldout(model, manifest, seq_len=2, max_batches=1)
    assert partial["complete_split"] is False
    assert partial["valid_tokens"] == 2


def test_heldout_document_losses_reconcile_continuations_and_stable_ids(tmp_path, config, tokenizer_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    texts = ["w4 w5", " ".join(["w6", "w7", "w8"] * 4), "w9"]
    manifest = prepare_dataset(tmp_path / "data", {"text": {"validation": texts}},
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
        tokenizer={"id": "test", "revision": "fixed-v1", "vocab_size": len(tokenizer), "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0})
    model = RecursiveLM(config).train()
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    for length in (4, 7):
        output = tmp_path / "losses" / f"seq-{length}.json"
        result = evaluate_heldout(model, manifest, seq_len=length, batch_size=2, document_loss_output=output)
        losses = json.loads(output.read_text())
        assert losses["format"] == "moeloop-document-losses" and losses["version"] == 1
        assert losses["manifest_fingerprint"] == manifest.fingerprint and losses["source"] == "text"
        assert losses["split"] == "validation" and losses["seq_len"] == length
        assert losses["num_visits"] == config.num_visits and losses["complete_split"] is True
        assert losses["label_contract"] == result["label_contract"]
        documents = losses["documents"]
        assert [row["document_id"] for row in documents] == [0, 1, 2]
        assert [row["valid_tokens"] for row in documents] == [3, 13, 2]
        assert sum(row["valid_tokens"] for row in documents) == result["valid_tokens"] == 18
        assert sum(row["ce_sum"] for row in documents) == pytest.approx(result["ce_sum"], abs=1e-10)
        assert result["document_losses"] == {"path": str(output.resolve()),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "document_count": 3}
        # Independent forwards for each document segment: block boundaries
        # reset context, while adjacent documents cannot attend to each other.
        offset = 0
        with torch.no_grad():
            for row, text in zip(documents, texts):
                tokens = [1, *tokenizer.encode(text, add_special_tokens=False), 2]
                cursor, expected = 0, 0.0
                while cursor < len(tokens) - 1:
                    count = min(length - offset, len(tokens) - 1 - cursor)
                    logits = model(torch.tensor([tokens[cursor:cursor + count]])).logits
                    expected += torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(),
                        torch.tensor(tokens[cursor + 1:cursor + count + 1]), reduction="sum").item()
                    cursor += count
                    offset = (offset + count) % length
                assert row["ce_sum"] == pytest.approx(expected, abs=3e-6)
                assert row["ce"] == row["ce_sum"] / row["valid_tokens"]
    assert model.training
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()


def test_heldout_document_losses_reject_multiple_sources(tmp_path, config):
    manifest = prepare_dataset(tmp_path / "data", {name: {"validation": ["one"]} for name in ("a", "b")},
        encode=lambda text: [4], tokenizer={"id": "unit", "revision": "fixed", "vocab_size": 17, "bos_token_id": 1, "eos_token_id": 2})
    output = tmp_path / "documents.json"
    model = RecursiveLM(config).train()
    with pytest.raises(ValueError, match="single.source"):
        evaluate_heldout(model, manifest, document_loss_output=output, seq_len=8)
    assert model.training and not output.exists()


def test_export_generate_and_tokenizer_identity_check(tmp_path, config, tokenizer_path):
    checkpoint = tmp_path / "weights.pt"
    torch.save(RecursiveLM(config).state_dict(), checkpoint)
    manifest = prepared_fixture(tmp_path / "data", tokenizer_path)
    output = export_checkpoint(checkpoint, tmp_path / "artifact", config, manifest.root / "tokenizer")
    _check_tokenizer_identity(output, manifest)
    text = generate_text(output, "w4 w5", max_new_tokens=2)
    assert text.startswith("w4 w5")
    with pytest.raises(ValueError, match="context"):
        generate_text(output, "w4 w5", max_new_tokens=40)
    payload = manifest.payload
    payload["tokenizer"]["files"]["tokenizer/tokenizer.json"] = "different"
    modified = PreparedManifest(manifest.path, payload)
    with pytest.raises(ValueError, match="tokenizer files differ"):
        _check_tokenizer_identity(output, modified)


def test_lm_eval_adapter_is_the_existing_hf_cli(tmp_path):
    command = lm_eval_command(tmp_path, ["hellaswag", "arc_easy"], device="cpu", batch_size=2, output_path=tmp_path / "results.json")
    assert command[:5] == [sys.executable, "-m", "lm_eval", "--model", "hf"]
    assert f"pretrained={tmp_path},trust_remote_code=True" in command
    assert "hellaswag,arc_easy" in command


@pytest.mark.parametrize("bare_json", [False, True])
def test_explicit_prepare_special_ids_survive_export_legacy_map_and_generation(tmp_path, config, tokenizer_path, bare_json):
    source = tokenizer_path
    if bare_json:
        source = tmp_path / "bare"
        source.mkdir()
        (source / "tokenizer.json").write_bytes((tokenizer_path / "tokenizer.json").read_bytes())
    else:
        (source / "special_tokens_map.json").write_text(json.dumps({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}))
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    text = tmp_path / "local.txt"
    text.write_text("w4 w7\n")
    prepared = tmp_path / "prepared"
    prepare_main(["--input", str(text), "--tokenizer", str(source), "--bos-token-id", "5", "--eos-token-id", "6", "--output", str(prepared)])
    manifest = PreparedManifest.load(prepared)
    assert next(PackedDataset(manifest, seq_len=4, infinite=False, shuffle=False))[0]["input"][0].item() == 5
    checkpoint = tmp_path / "weights.pt"
    torch.save(RecursiveLM(config).state_dict(), checkpoint)
    output = export_checkpoint(checkpoint, tmp_path / "artifact", config, prepared / "tokenizer")
    exported = transformers.AutoTokenizer.from_pretrained(output, local_files_only=True, trust_remote_code=True)
    assert (exported.bos_token_id, exported.eos_token_id, exported.pad_token_id) == (5, 6, 6)
    for filename in ("config.json", "generation_config.json"):
        artifact_config = json.loads((output / filename).read_text())
        assert (artifact_config["bos_token_id"], artifact_config["eos_token_id"], artifact_config["pad_token_id"]) == (5, 6, 6)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original
    assert generate_text(output, "w4", max_new_tokens=1).startswith("w4")


def test_export_rejects_tokenizer_other_than_training_manifest(tmp_path, config, tokenizer_path):
    manifest = prepared_fixture(tmp_path / "data", tokenizer_path)
    checkpoint = tmp_path / "weights.pt"
    torch.save(RecursiveLM(config).state_dict(), checkpoint)
    (tmp_path / "resolved_config.json").write_text(json.dumps({"dataloader": {"manifest_path": str(manifest.path)}}))
    with pytest.raises(ValueError, match="training manifest|role differs"):
        export_checkpoint(checkpoint, tmp_path / "artifact", config, tokenizer_path)


def test_export_eval_generate_cli_roundtrip_offline(tmp_path, config, tokenizer_path):
    model = RecursiveLM(config)
    checkpoint = tmp_path / "checkpoint"
    save_dcp(checkpoint, state_layout(model.state_dict(), "native_grouped"))
    (tmp_path / "model_config.json").write_text(json.dumps(config.to_dict()))
    manifest = prepared_fixture(tmp_path / "data", tokenizer_path)
    output = tmp_path / "artifact"
    document_losses = tmp_path / "document-losses.json"
    aggregate = tmp_path / "heldout.json"
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1")
    commands = [
        ["moeloop.export", "--checkpoint", str(checkpoint), "--output", str(output), "--tokenizer", str(manifest.root / "tokenizer")],
        ["moeloop.evaluate", "--model", str(output), "--manifest", str(manifest.path), "--seq-len", "4", "--max-batches", "1", "--output", str(aggregate), "--document-loss-output", str(document_losses)],
        ["moeloop.generate", "--model", str(output), "--prompt", "w4 w5", "--max-new-tokens", "2"],
    ]
    for command in commands:
        result = subprocess.run([sys.executable, "-m", *command], cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=35)
        assert result.returncode == 0, result.stdout + result.stderr
        if command[0] == "moeloop.evaluate":
            evaluation = json.loads(result.stdout)
            assert evaluation["valid_tokens"] == 4 and evaluation == json.loads(aggregate.read_text())
            assert evaluation["document_losses"]["sha256"] == hashlib.sha256(document_losses.read_bytes()).hexdigest()
            assert evaluation["document_losses"]["path"] == str(document_losses.resolve())
            sidecar = json.loads(document_losses.read_text())
            assert sidecar["complete_split"] is False
            assert [row["valid_tokens"] for row in sidecar["documents"]] == [3, 1]
        elif command[0] == "moeloop.generate":
            assert result.stdout.strip().startswith("w4 w5")
    # Reject the document path independently of the aggregate output path.
    repeated = commands[1].copy()
    repeated[repeated.index("--output") + 1] = str(tmp_path / "new-aggregate.json")
    original = document_losses.read_bytes()
    result = subprocess.run([sys.executable, "-m", *repeated], cwd=tmp_path, env=environment,
        text=True, capture_output=True, timeout=35)
    assert result.returncode != 0 and "already exists" in result.stderr
    assert document_losses.read_bytes() == original and not (tmp_path / "new-aggregate.json").exists()


def test_lm_eval_sample_logging_is_opt_in(tmp_path):
    assert "--log_samples" not in lm_eval_command(tmp_path, ["arc_easy"])
    command = lm_eval_command(tmp_path, ["arc_easy"], output_path=tmp_path / "results", log_samples=True)
    assert "--log_samples" in command


@pytest.mark.parametrize("batch_size", [1, 16, "32", "auto", "auto:4", "auto:0.5"])
def test_lm_eval_passes_native_batch_syntax_without_limiting_tasks(tmp_path, batch_size):
    command = lm_eval_command(tmp_path, ["arc_easy", "hellaswag"], batch_size=batch_size, log_samples=True)
    assert command[command.index("--batch_size") + 1] == str(batch_size)
    assert "--log_samples" in command and "--limit" not in command
    assert command[command.index("--tasks") + 1] == "arc_easy,hellaswag"


@pytest.mark.parametrize("batch_size", [0, -1, "0", "auto:0", "auto:-1", "auto:nan", "auto:inf", "automatic", "auto:", "auto:4:2", 1.5, True, None])
def test_lm_eval_rejects_invalid_native_batch_syntax(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch"):
        lm_eval_command(tmp_path, ["arc_easy"], batch_size=batch_size)
