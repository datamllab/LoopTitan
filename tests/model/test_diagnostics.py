"""Per-visit diagnostics pinned against the model's own forward and closed forms."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")
import torch.nn.functional as F
from torch.utils.data import DataLoader

from moeloop import recipes
from moeloop.data import PackedDataset, build_document_mask
from moeloop.diagnostics import (_record, diagnose, fit_readout, load_entropy, main, multiset_jaccard, plot_diagnostics,
                                 token_jaccard)
from moeloop.evaluate import evaluate_heldout
from moeloop.export import export_checkpoint
from moeloop.model import RecursiveLM
from moeloop.prepare import prepare_dataset
from moeloop.tokens import byte_table_sha256, token_byte_lengths


GEOMETRY = dict(vocab_size=64, dim=16, hidden_dim=16, num_heads=2, num_kv_heads=1, num_experts=4, top_k=2, max_seq_len=64)
TOKENIZER = {"id": "test-tokenizer", "revision": "fixed-v1", "vocab_size": 64, "bos_token_id": 0, "eos_token_id": 63, "pad_token_id": 63}
SEQ_LEN, BATCH = 8, 2
# The byte-level BPE fixture of tests/data/test_bpb.py (vocabulary 321, EOT id 0).
BPB_FIXTURE = Path(__file__).parents[1] / "data" / "fixtures" / "bpb"
BPB_TEXTS = ["hello world", "naïve café — 你好，世界 🙂", "tabs\tand\nnewlines  and   spaces", "quick 🦊 jumps over the lazy dog"]
BPB_TOKENIZER = {"id": "bpb-fixture", "revision": "fixed-v1", "vocab_size": 321, "bos_token_id": 0, "eos_token_id": 0, "pad_token_id": 0}
RECIPES = {
    "recursive_moe": lambda: recipes.recursive_moe_spec(**GEOMETRY),
    "smelt": lambda: recipes.smelt_spec(physical_layers=4, **GEOMETRY),
    "huginn": lambda: recipes.huginn_spec(**GEOMETRY),
    "ouro": lambda: recipes.ouro_spec(**GEOMETRY),
    "loom": lambda: recipes.loom_spec(**GEOMETRY),
    "per_visit_router": lambda: recipes.per_visit_router_spec(**GEOMETRY),
    "expert_tying": lambda: recipes.expert_tying_spec(groups=2, group_size=2, **GEOMETRY),
}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(31)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def manifest(tmp_path):
    texts = ["1 2 3 4 5 6 7", "8 9 10", " ".join(str(value) for value in range(11, 30)), "30 31 32 33"]
    return prepare_dataset(tmp_path / "data", {"text": {"validation": texts}},
                           encode=lambda text: [int(value) for value in text.split()], tokenizer=TOKENIZER)


def model(name):
    return RecursiveLM(RECIPES[name]().model).eval()


def batches(manifest):
    dataset = PackedDataset(manifest, split="validation", seq_len=SEQ_LEN, shuffle=False, infinite=False)
    try:
        yield from DataLoader(dataset, batch_size=BATCH, num_workers=0)
    finally:
        dataset.close()


def forward(core, inputs, **kwargs):
    return core(inputs["input"], positions=inputs["positions"], router_mask=inputs["input_mask"],
                attention_mask=build_document_mask(inputs["document_ids"], inputs["input_mask"]), **kwargs)


def run(core, manifest, **kwargs):
    return diagnose(core, manifest, seq_len=SEQ_LEN, batch_size=BATCH, **kwargs)


def test_closed_forms_for_entropy_and_jaccard():
    assert load_entropy(torch.tensor([2, 1, 1, 0])) == pytest.approx(1.5 * math.log(2), abs=1e-12)
    assert load_entropy(torch.tensor([5, 5, 5, 5])) == pytest.approx(math.log(4), abs=1e-12)
    assert multiset_jaccard(torch.tensor([2, 1, 0]), torch.tensor([1, 1, 1])) == pytest.approx(0.5)
    left, right = torch.tensor([[[0, 1], [0, 1]]]), torch.tensor([[[1, 2], [1, 0]]])
    valid = torch.tensor([[True, True]])
    assert float(token_jaccard(left, right, valid, 4)) == pytest.approx(1 / 3 + 1)
    assert float(token_jaccard(left, right, ~valid, 4)) == 0


@pytest.mark.parametrize("name", ["recursive_moe", "smelt", "huginn", "ouro", "per_visit_router"])
def test_logit_lens_and_truncation_match_the_models_own_losses(manifest, name):
    core = model(name)
    result = run(core, manifest)
    depth = core.config.num_visits
    assert result["skipped"] == {} and len(result["logit_lens"]) == len(result["effective_depth"]) == depth
    assert result["effective_depth_from"] == "logit_lens"
    for r in range(1, depth + 1):
        reference = evaluate_heldout(core, manifest, seq_len=SEQ_LEN, batch_size=BATCH, num_visits=r)
        truncated = result["effective_depth"][r - 1]
        assert truncated["num_visits"] == r and truncated["valid_tokens"] == reference["valid_tokens"]
        assert truncated["ce_sum"] == pytest.approx(reference["ce_sum"], rel=1e-6)
    assert result["logit_lens"] == [{"visit": row["num_visits"] - 1, **{k: v for k, v in row.items() if k != "num_visits"}}
                                    for row in result["effective_depth"]]


def test_logit_lens_equals_the_models_collected_readouts(manifest):
    core = model("ouro")
    sums = [0.0] * core.config.num_visits
    with torch.no_grad():
        for inputs, labels in batches(manifest):
            output = forward(core, inputs, collect_readouts=True)
            for visit, hidden in enumerate(output.visit_hidden_states):
                sums[visit] += F.cross_entropy(core.lm_head(hidden).flatten(0, 1), labels.flatten(), reduction="sum").item()
    lens = run(core, manifest)["logit_lens"]
    assert [row["ce_sum"] for row in lens] == pytest.approx(sums, rel=1e-6)


def test_ema_truncation_runs_its_own_forwards_and_differs_from_the_lens(manifest):
    core = model("loom")
    result = run(core, manifest)
    assert result["effective_depth_from"] == "truncated_forward" and result["skipped"] == {}
    curve = [row["ce_sum"] for row in result["effective_depth"]]
    references = [evaluate_heldout(core, manifest, seq_len=SEQ_LEN, batch_size=BATCH, num_visits=r)["ce_sum"]
                  for r in range(1, core.config.num_visits + 1)]
    assert curve == pytest.approx(references, rel=1e-6)
    assert result["logit_lens"][-1]["ce_sum"] == pytest.approx(curve[-1], rel=1e-6)
    assert result["logit_lens"][0]["ce_sum"] != pytest.approx(curve[0], rel=1e-6)  # depth-dependent gains


def test_layer_order_reports_routing_and_depth_but_no_visit_states(manifest):
    core = model("expert_tying")
    result = run(core, manifest)
    assert result["logit_lens"] is None and result["residual"] is None
    assert set(result["skipped"]) == {"logit_lens", "residual"}
    assert [bank["visits"] for bank in result["routing"] if bank["region"] == "core"] == [[0, 1], [0, 1]]
    assert result["effective_depth_from"] == "truncated_forward"
    references = [evaluate_heldout(core, manifest, seq_len=SEQ_LEN, batch_size=BATCH, num_visits=r)["ce_sum"]
                  for r in (1, 2)]
    assert [row["ce_sum"] for row in result["effective_depth"]] == pytest.approx(references, rel=1e-6)


def test_identical_routing_has_unit_jaccard_and_top_k_load_entropy(manifest):
    core = model("per_visit_router")
    with torch.no_grad():
        for module in core.modules():
            if hasattr(module, "gate"):
                module.gate.weight.zero_()  # uniform probabilities: every call selects the same top-k set
    result = run(core, manifest)
    assert result["routing"] and all(len(bank["visits"]) == 4 for bank in result["routing"])
    for bank in result["routing"]:
        assert bank["token_jaccard"] == [[1.0] * 4] * 4
        assert bank["load_jaccard"] == [[1.0] * 4] * 4
        assert bank["load_entropy"] == pytest.approx([math.log(2)] * 4)
        assert bank["normalized_load_entropy"] == pytest.approx([0.5] * 4)


def test_routing_rows_reconcile_with_router_counts(manifest):
    core = model("recursive_moe")
    expected = {}
    with torch.no_grad():
        for inputs, _ in batches(manifest):
            with _record(core) as record:
                output = forward(core, inputs)
            sparse = [(slot, counts) for slot, counts, width in zip(
                output.router_call_slots, output.router_call_counts, output.router_call_widths) if width]
            assert len(sparse) == len(record["ids"])
            for (slot, counts), ids in zip(sparse, record["ids"]):
                # The hooked top-k ids of valid tokens are the call's own router counts.
                assert torch.equal(torch.bincount(ids[inputs["input_mask"]].flatten(), minlength=counts.numel()), counts)
                expected[slot] = expected.get(slot, 0) + counts
    result = run(core, manifest)
    for bank in result["routing"]:
        for visit, entropy in zip(bank["visits"], bank["load_entropy"]):
            assert entropy == pytest.approx(load_entropy(expected[(bank["region"], bank["index"], visit)]))
        assert all(bank["token_jaccard"][i][i] == bank["load_jaccard"][i][i] == 1 for i in range(len(bank["visits"])))
        assert all(0 <= value <= 1 for row in bank["token_jaccard"] for value in row)


@pytest.mark.parametrize("name", ["recursive_moe", "huginn"])
def test_residual_updates_match_a_manual_core_unroll(manifest, name):
    core = model(name)  # huginn: prefix cell, zero initial state at evaluation, linear input injection
    depth = core.config.num_visits
    norms, relative, cosine, tokens = [0.0] * depth, [0.0] * depth, [0.0] * depth, 0
    with torch.no_grad():
        for inputs, _ in batches(manifest):
            valid = inputs["input_mask"]
            kwargs = dict(positions=inputs["positions"], router_mask=valid, cache=None,
                          attention_mask=build_document_mask(inputs["document_ids"], valid))
            anchor, previous = core.embedding(inputs["input"]), None
            for index, cell in enumerate(core.prefix):
                anchor = cell(anchor, visit=0, slot=("prefix", index, 0), **kwargs)[0]
            state = anchor if core.config.initial_state == "input" else torch.zeros_like(anchor)
            for visit in range(depth):
                if core.input_adapter is not None:
                    state = core.input_adapter(state, anchor)
                entering = state
                for index, cell in enumerate(core.core):
                    state = cell(state, visit=visit, slot=("core", index, visit), **kwargs)[0]
                update = state - entering
                norms[visit] += update.norm(dim=-1)[valid].sum().item()
                relative[visit] += (update.norm(dim=-1) / entering.norm(dim=-1))[valid].sum().item()
                if previous is not None:
                    cosine[visit] += F.cosine_similarity(previous, update, dim=-1)[valid].sum().item()
                previous = update
            tokens += int(valid.sum())
    rows = run(core, manifest)["residual"]
    assert [row["update_norm"] for row in rows] == pytest.approx([value / tokens for value in norms], rel=1e-5)
    assert [row["relative_update"] for row in rows] == pytest.approx([value / tokens for value in relative], rel=1e-5)
    assert rows[0]["cosine_previous"] is None
    assert [row["cosine_previous"] for row in rows[1:]] == pytest.approx([value / tokens for value in cosine[1:]], rel=1e-5)


def test_bpb_matches_a_hand_oracle_and_heldout_evaluation(tmp_path):
    tokenizer = tokenizers.Tokenizer.from_file(str(BPB_FIXTURE / "tokenizer.json"))
    manifest = prepare_dataset(tmp_path / "bpb", {"text": {"validation": BPB_TEXTS}},
                               encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
                               tokenizer=BPB_TOKENIZER, tokenizer_directory=BPB_FIXTURE)
    core = RecursiveLM(recipes.smelt_spec(physical_layers=4, **(GEOMETRY | {"vocab_size": 321})).model).eval()
    lengths = token_byte_lengths(BPB_FIXTURE)
    text_sum, label_bytes = 0.0, 0
    with torch.no_grad():
        for inputs, labels in batches(manifest):
            losses = F.cross_entropy(forward(core, inputs).logits.flatten(0, 1), labels.flatten(), reduction="none", ignore_index=-100)
            for loss, label in zip(losses.tolist(), labels.flatten().tolist()):
                if label != -100 and lengths[label]:
                    text_sum, label_bytes = text_sum + loss, label_bytes + lengths[label]
    result = run(core, manifest)
    reference = evaluate_heldout(core, manifest, seq_len=SEQ_LEN, batch_size=BATCH)
    final = result["effective_depth"][-1]
    assert final["label_bytes"] == result["logit_lens"][-1]["label_bytes"] == reference["label_bytes"] == label_bytes
    assert final["ce_sum_text"] == pytest.approx(text_sum, rel=1e-6)
    assert final["bpb"] == pytest.approx(text_sum / (math.log(2) * label_bytes), rel=1e-6)
    assert final["bpb"] == pytest.approx(reference["bpb"], rel=1e-6)
    assert result["byte_table_sha256"] == reference["byte_table_sha256"] == byte_table_sha256(lengths)
    assert result["bpb_unavailable"] is None


@pytest.mark.parametrize("name", ["loom", "expert_tying"])
def test_truncated_forward_bpb_is_heldout_evaluation_bit_for_bit(tmp_path, name):
    """The K-visit truncated row is evaluate_heldout's forward and BPB terms, also above the manifest vocabulary."""
    tokenizer = tokenizers.Tokenizer.from_file(str(BPB_FIXTURE / "tokenizer.json"))
    manifest = prepare_dataset(tmp_path / "bpb", {"text": {"validation": BPB_TEXTS}},
                               encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
                               tokenizer=BPB_TOKENIZER, tokenizer_directory=BPB_FIXTURE)
    core = RecursiveLM(replace(RECIPES[name]().model, vocab_size=384))
    result = run(core, manifest)
    reference = evaluate_heldout(core, manifest, seq_len=SEQ_LEN, batch_size=BATCH)
    assert result["effective_depth_from"] == "truncated_forward"
    final = result["effective_depth"][-1]
    assert {key: final[key] for key in ("ce_sum_text", "label_bytes", "bpb")} == {
        key: reference[key] for key in ("ce_sum_text", "label_bytes", "bpb")}
    assert result["byte_table_sha256"] == reference["byte_table_sha256"]


def test_no_byte_bearing_labels_match_heldout_evaluation(tmp_path):
    tokenizer = tokenizers.Tokenizer.from_file(str(BPB_FIXTURE / "tokenizer.json"))
    manifest = prepare_dataset(tmp_path / "special", {"text": {"validation": ["<|endoftext|>"]}},
                               encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
                               tokenizer=BPB_TOKENIZER, tokenizer_directory=BPB_FIXTURE)
    core = RecursiveLM(recipes.smelt_spec(physical_layers=4, **(GEOMETRY | {"vocab_size": 321})).model)
    result = run(core, manifest)
    reference = evaluate_heldout(core, manifest, seq_len=SEQ_LEN, batch_size=BATCH)
    assert result["bpb_unavailable"] == reference["bpb_unavailable"] == "no byte-bearing labels evaluated"
    assert result["effective_depth"][-1]["bpb"] is reference["bpb"] is None


def test_ce_only_without_tokenizer_assets(manifest):
    result = run(model("smelt"), manifest)
    assert result["byte_table_sha256"] is None and result["bpb_unavailable"] == "manifest has no tokenizer assets"
    assert "bpb" not in result["logit_lens"][0]
    json.dumps(result, allow_nan=False)


def test_non_finite_loss_is_refused(manifest):
    core = model("smelt")
    with torch.no_grad():
        core.lm_head.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        run(core, manifest)


def test_mode_and_rng_are_restored(manifest):
    core = model("huginn").train()
    state = torch.get_rng_state().clone()
    run(core, manifest, max_batches=1)
    assert core.training and torch.equal(state, torch.get_rng_state())


@pytest.fixture
def word_tokenizer(tmp_path):
    """A WordLevel (not byte-level BPE) tokenizer with the manifest's 64 ids."""
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3} | {f"w{i}": i for i in range(4, 64)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>",
                                         eos_token="<eos>", unk_token="<unk>").save_pretrained(tmp_path / "tokenizer")
    return tmp_path / "tokenizer"


@pytest.fixture
def tokenized_manifest(tmp_path, word_tokenizer):
    tokenizer = transformers.AutoTokenizer.from_pretrained(word_tokenizer, local_files_only=True)
    return prepare_dataset(tmp_path / "tokenized", {"text": {"validation": ["w4 w5 w6 w7", "w8 w9"]}},
                           encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
                           tokenizer={**TOKENIZER, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0},
                           tokenizer_directory=word_tokenizer)


def test_non_byte_level_tokenizer_gives_ce_only_with_the_reason(tokenized_manifest):
    result = run(model("smelt"), tokenized_manifest)
    reference = evaluate_heldout(model("smelt"), tokenized_manifest, seq_len=SEQ_LEN, batch_size=BATCH)
    assert result["byte_table_sha256"] is None and result["bpb_unavailable"] == reference["bpb_unavailable"]
    assert "bpb" not in result["effective_depth"][-1]


def test_cli_on_an_hf_export_matches_the_in_memory_result(tmp_path, manifest, word_tokenizer):
    core = model("smelt")
    torch.save(core.state_dict(), tmp_path / "weights.pt")
    exported = export_checkpoint(tmp_path / "weights.pt", tmp_path / "export", core.config, word_tokenizer)
    output = tmp_path / "out" / "diagnostics.json"
    main(["--model", str(exported), "--manifest", str(manifest.path), "--seq-len", str(SEQ_LEN),
          "--batch-size", str(BATCH), "--output", str(output)])
    saved = json.loads(output.read_text())
    expected = run(core, manifest)
    assert saved["logit_lens"][-1]["ce_sum"] == pytest.approx(expected["logit_lens"][-1]["ce_sum"], rel=1e-6)
    assert saved["routing"] == json.loads(json.dumps(expected["routing"]))
    with pytest.raises(SystemExit):
        main(["--model", str(exported), "--manifest", str(manifest.path), "--output", str(output)])
    tokenizer = tokenizers.Tokenizer.from_file(str(BPB_FIXTURE / "tokenizer.json"))
    other = prepare_dataset(tmp_path / "other", {"text": {"validation": BPB_TEXTS}},
                            encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
                            tokenizer=BPB_TOKENIZER, tokenizer_directory=BPB_FIXTURE)
    with pytest.raises(ValueError, match="tokenizer files differ"):
        diagnose(exported, other, seq_len=SEQ_LEN)


def test_plot_writes_svg_and_png(tmp_path, manifest):
    pytest.importorskip("matplotlib")
    result = run(model("smelt"), manifest)
    for suffix in ("svg", "png"):
        path = plot_diagnostics(result, tmp_path / f"diagnostics.{suffix}")
        assert path.stat().st_size > 0



@pytest.fixture
def bpb_splits(tmp_path):
    tokenizer = tokenizers.Tokenizer.from_file(str(BPB_FIXTURE / "tokenizer.json"))
    return prepare_dataset(tmp_path / "splits", {"text": {"validation": BPB_TEXTS * 3, "test": BPB_TEXTS[::-1]}},
                           encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
                           tokenizer=BPB_TOKENIZER, tokenizer_directory=BPB_FIXTURE)


def test_mixed_readout_at_terminal_weights_is_the_native_readout(bpb_splits):
    core = RecursiveLM(recipes.smelt_spec(physical_layers=4, **(GEOMETRY | {"vocab_size": 321})).model)
    result = fit_readout(core, bpb_splits, seq_len=SEQ_LEN, batch_size=BATCH, fit_sequences=8, epochs=2, lr=0.05)
    reference = evaluate_heldout(core, bpb_splits, split="test", seq_len=SEQ_LEN, batch_size=BATCH)
    assert result["terminal_only_mixture"] == result["native"]
    # BPB terms bit for bit; evaluate_heldout's ce_sum is a reduced FP32 sum (see its docstring).
    assert {key: result["native"][key] for key in ("valid_tokens", "ce_sum_text", "label_bytes", "bpb")} == {
        key: reference[key] for key in ("valid_tokens", "ce_sum_text", "label_bytes", "bpb")}
    assert result["native"]["ce_sum"] == pytest.approx(reference["ce_sum"], rel=1e-6)
    assert sum(result["mixture"]["weights"]) == pytest.approx(1) and len(result["mixture"]["weights"]) == 3
    epochs = {row["epoch"]: row for row in result["selection_ce"]}
    for name in ("mixture", "control"):  # each head keeps its epoch of lowest selection CE
        assert epochs[result[name]["epoch"]][name] == min(row[name] for row in result["selection_ce"])
    assert result["fit_tokens"] > 0 and result["selection_tokens"] > 0
    # Per-token improvement of the second visit, recomputed from the model's own readouts.
    deltas, labels_all = [], []
    with torch.no_grad():
        dataset = PackedDataset(bpb_splits, split="test", seq_len=SEQ_LEN, shuffle=False, infinite=False)
        for inputs, labels in DataLoader(dataset, batch_size=BATCH, num_workers=0):
            with _record(core) as record:
                output = forward(core, inputs)
            cell_inputs, *_ = core.parse_inputs(inputs["input"], positions=inputs["positions"], router_mask=inputs["input_mask"],
                attention_mask=build_document_mask(inputs["document_ids"], inputs["input_mask"]))
            first = core.lm_head(core.readout(record["outputs"][0], 0, cell_inputs))
            loss = lambda logits: F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none", ignore_index=-100)
            valid = labels.flatten() != -100
            deltas.append((loss(first) - loss(output.logits))[valid].double())
            labels_all.append(labels.flatten()[valid])
        dataset.close()
    delta, target = torch.cat(deltas), torch.cat(labels_all)
    per_token = result["per_token"]
    assert per_token["tokens"] == len(delta) == reference["valid_tokens"]
    top = delta.sort(descending=True).values[:len(delta) // 5]
    assert per_token["top_20_share"] == pytest.approx(float(top.sum() / delta.sum()), rel=1e-6)
    assert per_token["quantiles"]["0.5"] == pytest.approx(float(delta.sort().values[round(0.5 * (len(delta) - 1))]), rel=1e-6)
    counts = torch.bincount(target)
    quintiles = delta[counts[target].argsort(stable=True)].tensor_split(5)
    assert per_token["frequency_quintiles"] == pytest.approx([float(q.mean()) for q in quintiles], rel=1e-6)


def test_fit_readout_needs_enough_fit_blocks(bpb_splits):
    core = RecursiveLM(recipes.smelt_spec(physical_layers=4, **(GEOMETRY | {"vocab_size": 321})).model)
    with pytest.raises(ValueError, match="need 1024 blocks"):
        fit_readout(core, bpb_splits, seq_len=SEQ_LEN, batch_size=BATCH)


def test_cli_fit_readout_under_run_dir_writes_source_config_analysis_and_metrics(tmp_path, monkeypatch, word_tokenizer):
    import torch.distributed.checkpoint as dcp
    texts = [" ".join(str(1 + (i * 7 + j) % 60) for j in range(5 + i % 9)) for i in range(24)]
    manifest = prepare_dataset(tmp_path / "numbers", {"text": {"validation": texts, "test": texts[:6]}},
                               encode=lambda text: [int(value) for value in text.split()], tokenizer=TOKENIZER)
    spec = recipes.smelt_spec(physical_layers=4, **GEOMETRY)
    core = RecursiveLM(spec.model)
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps(spec.to_dict()))
    (run / "request.json").write_text(json.dumps({"world_size": 8, "local_batch_size": 16,
        "gradient_accumulation_steps": 1, "manifest_fingerprint": "abc"}))
    dcp.save({key: value.clone() for key, value in core.state_dict().items()},
             checkpoint_id=str(run / "checkpoint" / "step-4769"), no_dist=True)
    export_checkpoint(run / "checkpoint" / "step-4769", run / "hf", core.config, word_tokenizer)
    run_dir = tmp_path / "RUN_DIR"
    run_dir.mkdir()
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    output = tmp_path / "readout.json"
    main(["--model", str(run / "hf"), "--manifest", str(manifest.path), "--seq-len", str(SEQ_LEN),
          "--batch-size", str(BATCH), "--fit-readout", "--fit-sequences", "4", "--output", str(output)])
    config = json.loads((run_dir / "config.json").read_text())
    values = spec.to_dict()
    assert {key: config[key] for key in ("name", "model", "training", "optimizer", "inference", "run")} == {
        key: values[key] for key in ("name", "model", "training", "optimizer", "inference", "run")}
    assert {key: config["analysis"][key] for key in ("seed", "fit_split", "report_split", "fit_sequences", "seq_len", "step")} == {
        "seed": 0, "fit_split": "validation", "report_split": "test", "fit_sequences": 4, "seq_len": SEQ_LEN, "step": 4769}
    result = json.loads(output.read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["native_ce"] == metrics["terminal_only_mixture_ce"] == result["readout"]["native"]["ce"]
    assert {"mixture_weight_0", "control_temperature", "top_20_share", "frequency_quintile_5", "update_norm_2"} <= set(metrics)
    assert all(isinstance(value, (int, float)) for value in metrics.values())
