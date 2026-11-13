"""CORE (DCLM v2) contract: nanochat prompts and spans, seeded few-shot, centering, CLI output."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from moeloop import core
from moeloop.hf import MoeloopConfig, MoeloopForCausalLM
from moeloop.model import ModelConfig

BUNDLE = Path(__file__).parent / "fixtures" / "core" / "eval_bundle"
SOURCE = Path(__file__).resolve().parents[2] / "src"
FIXTURE_TASKS = ("copa", "winograd", "lambada_openai")  # one 0-shot task per type


def byte_tokenizer():
    """Byte-level BPE without merges: text tokenizes to exactly its UTF-8 bytes."""
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    vocabulary.update({symbol: 3 + index for index, symbol in enumerate(sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()))})
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocabulary, merges=[]))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(25)
        config = ModelConfig(vocab_size=259, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2, num_experts=4,
                             top_k=2, num_cells=1, num_visits=2, prefix_cells=1, suffix_cells=1, max_seq_len=128)
        model = MoeloopForCausalLM(MoeloopConfig(config.to_dict(), bos_token_id=1, eos_token_id=2, pad_token_id=0)).eval()
    tokenizer = byte_tokenizer()
    directory = tmp_path_factory.mktemp("core") / "model"
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    return model, tokenizer, directory


def test_task_table_is_the_dclm_core_definition():
    assert len(core.TASKS) == 22 and len(core.TASK_BY_LABEL) == 22
    kinds = [task.kind for task in core.TASKS]
    assert (kinds.count("multiple_choice"), kinds.count("schema"), kinds.count("language_modeling")) == (11, 2, 9)
    assert {task.label: task.shots for task in core.TASKS if task.shots not in (0, 10)} == {"agi_eval_lsat_ar": 3}
    assert {task.label for task in core.TASKS if task.shots == 0} == {
        "hellaswag_zeroshot", "copa", "openbook_qa", "lambada_openai", "winograd", "winogrande", "coqa"}
    assert {task.label for task in core.TASKS if task.delimiter != " "} == {"jeopardy", "arc_easy", "arc_challenge", "piqa", "boolq"}
    # DCLM PR #115 (2025-09) changed exactly these CORE baselines.
    assert {task.label: (task.baseline_v1, task.baseline) for task in core.TASKS if task.baseline != task.baseline_v1} == {
        "commonsense_qa": ("20", "40.3"), "agi_eval_lsat_ar": ("20", "25"), "bigbench_language_identification": ("9.1", "25")}


def test_rendering_matches_nanochat_templates():
    shots = [{"query": "Q: 1+1?", "choices": ["2", "5"], "gold": 0}, {"query": "Q: 3+3?", "choices": ["9", "6"], "gold": 1}]
    assert core.render_multiple_choice({"query": "Q: 2+2?", "choices": ["3", "4"], "gold": 1}, "\nAnswer: ", shots) == [
        "Q: 1+1?\nAnswer: 2\n\nQ: 3+3?\nAnswer: 6\n\nQ: 2+2?\nAnswer: 3",
        "Q: 1+1?\nAnswer: 2\n\nQ: 3+3?\nAnswer: 6\n\nQ: 2+2?\nAnswer: 4",
    ]
    assert core.render_multiple_choice({"query": "Go", "choices": ["left", "right"], "gold": 0}, " ") == ["Go left", "Go right"]
    item = {"context_options": ["Ann thanked Bob because Ann", "Ann thanked Bob because Bob"], "continuation": "was kind.", "gold": 1}
    assert core.render_schema(item, " ", [{"context_options": ["X", "Y"], "continuation": "z.", "gold": 1}]) == [
        "Y z.\n\nAnn thanked Bob because Ann was kind.", "Y z.\n\nAnn thanked Bob because Bob was kind."]
    # LM contexts are trimmed; the prompt without continuation is stripped, the one with it is not.
    assert core.render_language_modeling({"context": "  The sky is ", "continuation": "blue"}, " ",
                                         [{"context": "Grass is ", "continuation": "green "}]) == [
        "Grass is green \n\nThe sky is", "Grass is green \n\nThe sky is blue"]
    assert core.render_language_modeling({"context": "Capital of France?", "continuation": "Paris"}, "\nAnswer: ") == [
        "Capital of France?\nAnswer:", "Capital of France?\nAnswer: Paris"]


def test_continuation_spans_and_context_cropping():
    assert core.continuation_spans("multiple_choice", [[1, 5, 6, 7], [1, 5, 8]]) == [([1, 5, 6, 7], 2, 4), ([1, 5, 8], 2, 3)]
    assert core.continuation_spans("schema", [[1, 5, 9, 9], [1, 6, 6, 9, 9]]) == [([1, 5, 9, 9], 2, 4), ([1, 6, 6, 9, 9], 3, 5)]
    assert core.continuation_spans("language_modeling", [[1, 5], [1, 5, 6, 7]]) == [([1, 5, 6, 7], 2, 4)]
    with pytest.raises(ValueError, match="prefix"):
        core.continuation_spans("language_modeling", [[1, 5], [1, 6, 7]])
    # An option that is a token prefix of another keeps nanochat's empty span (real piqa data).
    assert core.continuation_spans("multiple_choice", [[1, 5, 6], [1, 5]]) == [([1, 5, 6], 2, 3), ([1, 5], 2, 2)]
    # A tokenizer that encodes everything to [] leaves only BOS, so no option can be scored.
    for kind, tokens in (("multiple_choice", [[1], [1]]), ("schema", [[1], [1]]), ("schema", [[1, 5], [1, 5]]),
                         ("language_modeling", [[1], [1]]), ("language_modeling", [[1, 5], [1, 5]])):
        with pytest.raises(ValueError, match="no option has a scorable continuation span"):
            core.continuation_spans(kind, tokens)
    # Byte tokens make spans checkable from string lengths: BOS + common byte prefix.
    task = core.TASK_BY_LABEL["copa"]
    data = [{"query": "Pick", "choices": ["a cat", "a dog!"], "gold": 0}]
    rows = core.task_rows(task, data, byte_tokenizer(), 1)
    assert [(index, start, end) for index, _, start, end in rows] == [(0, 1 + len("Pick a "), 1 + len("Pick a cat")), (0, 8, 1 + len("Pick a dog!"))]
    cropped = core.task_rows(task, data, byte_tokenizer(), 1, max_tokens=6)
    assert [(len(tokens), start, end) for _, tokens, start, end in cropped] == [(6, 3, 6), (6, 2, 6)]
    assert cropped[1][1] == rows[1][1][-6:]
    with pytest.raises(ValueError, match="does not fit"):
        core.task_rows(task, data, byte_tokenizer(), 1, max_tokens=4)


def test_fewshot_sampling_and_subsampling_are_seeded(tmp_path):
    assert core.fewshot_indices(3, 10, 4) == [2, 7, 9, 1] == core.fewshot_indices(3, 10, 4)
    assert core.fewshot_indices(3, 10, 0) == []
    with pytest.raises(ValueError, match="more than 10"):
        core.fewshot_indices(0, 10, 10)
    path = tmp_path / "eval_data" / "commonsense_reasoning" / "copa.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(10)))
    task = core.TASK_BY_LABEL["copa"]
    assert [row["id"] for row in core.load_examples(tmp_path, task)] == [0, 3, 7, 2, 1, 6, 4, 5, 8, 9]
    assert [row["id"] for row in core.load_examples(tmp_path, task, 4)] == [0, 3, 7, 2]


def test_centered_accuracy_and_core_mean_against_hand_values():
    assert core.centered(0.5, "25") == pytest.approx(1 / 3)
    assert core.centered(0.62, "62") == pytest.approx(0.0)
    assert core.centered(1.0, "40.3") == pytest.approx(1.0)
    assert core.centered(0.2, "0") == 0.2
    commonsense_qa = core.TASK_BY_LABEL["commonsense_qa"]  # v2 baseline 40.3, v1 20
    assert core.centered(0.5, commonsense_qa.baseline) == pytest.approx((0.5 - 0.403) / 0.597) == pytest.approx(0.162479, abs=1e-6)
    assert core.centered(0.5, commonsense_qa.baseline_v1) == pytest.approx(0.375)
    # GPT-2 (124M) accuracies published by nanochat in the bundle's openai-community-gpt2.csv;
    # that file's CORE row is 0.113891 under v1 baselines.
    gpt2 = {
        "hellaswag_zeroshot": 0.308504, "jeopardy": 0.001889, "bigbench_qa_wikidata": 0.282663, "arc_easy": 0.412458,
        "arc_challenge": 0.221843, "copa": 0.6, "commonsense_qa": 0.224406, "piqa": 0.62296, "openbook_qa": 0.278,
        "lambada_openai": 0.322919, "hellaswag": 0.307907, "winograd": 0.578755, "winogrande": 0.486188,
        "bigbench_dyck_languages": 0.155, "agi_eval_lsat_ar": 0.226087, "bigbench_cs_algorithms": 0.419697,
        "bigbench_operators": 0.085714, "bigbench_repeat_copy_logic": 0.03125, "squad": 0.058278, "coqa": 0.133659,
        "boolq": 0.552294, "bigbench_language_identification": 0.2566,
    }
    v1 = sum(core.centered(gpt2[task.label], task.baseline_v1) for task in core.TASKS) / 22
    v2 = sum(core.centered(gpt2[task.label], task.baseline) for task in core.TASKS) / 22
    assert v1 == pytest.approx(0.113891, abs=2e-6)
    # v2 = v1 + [((.224406-.403)/.597 - (.224406-.2)/.8) + ((.226087-.25)/.75 - (.226087-.2)/.8)
    #            + ((.2566-.25)/.75 - (.2566-.091)/.909)] / 22
    assert v2 == pytest.approx(0.088094, abs=2e-6)


def test_directory_sha256_is_the_sorted_sha256sum_listing(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_bytes(b"bee")
    (tmp_path / "a.txt").write_bytes(b"ay")
    listing = f"{hashlib.sha256(b'ay').hexdigest()}  a.txt\n{hashlib.sha256(b'bee').hexdigest()}  sub/b.txt\n"
    assert core.directory_sha256(tmp_path) == hashlib.sha256(listing.encode()).hexdigest()


def _oracle(model, tokens, start, end):
    """One unpadded forward per row: mean continuation CE and greedy exact match."""
    with torch.no_grad():
        logits = model(torch.tensor([tokens]), use_cache=False).logits[0, start - 1:end - 1]
    target = torch.tensor(tokens[start:end])
    return torch.nn.functional.cross_entropy(logits, target).item(), bool((logits.argmax(-1) == target).all())


def test_scores_are_batch_invariant_and_match_unpadded_forwards(exported):
    model, tokenizer, _ = exported
    for label in FIXTURE_TASKS:
        task = core.TASK_BY_LABEL[label]
        data = core.load_examples(BUNDLE, task)
        rows = core.task_rows(task, data, tokenizer, 1, 128)
        expected = [_oracle(model, *row[1:]) for row in rows]
        for batch_size in (1, 3):
            scores = core.score_rows(model, [row[1:] for row in rows], pad_token_id=1, batch_size=batch_size)
            assert [match for _, match in scores] == [match for _, match in expected]
            torch.testing.assert_close(torch.tensor([loss for loss, _ in scores]), torch.tensor([loss for loss, _ in expected]))


def test_choice_and_lm_decision_rules():
    # Lowest mean continuation CE wins; a nan (empty-span) option loses unless listed first, as in nanochat.
    data = [{"gold": 0}, {"gold": 1}, {"gold": 0}, {"gold": 0}]
    rows = [(0,), (0,), (1,), (1,), (1,), (2,), (2,), (3,), (3,)]
    scores = [(1.0, False), (2.0, False), (0.9, False), (0.5, False), (0.7, False),
              (3.0, False), (1.0, False), (1.0, False), (float("nan"), False)]
    for label in ("copa", "winograd"):
        assert core.example_correct(core.TASK_BY_LABEL[label], data, rows, scores) == [True, True, False, True]
    # A language-modeling example counts only its greedy flag.
    lambada = core.TASK_BY_LABEL["lambada_openai"]
    assert core.example_correct(lambada, [{}, {}], [(0,), (1,)], [(9.0, True), (0.1, False)]) == [True, False]


def test_lm_match_requires_every_continuation_token_to_be_greedy(exported):
    model, _, _ = exported
    prefix = [1, 40, 41, 42]
    with torch.no_grad():
        first = int(model(torch.tensor([prefix]), use_cache=False).logits[0, -1].argmax())
        second = int(model(torch.tensor([prefix + [first]]), use_cache=False).logits[0, -1].argmax())
    wrong_second = (second + 1) % 259
    rows = [(prefix + [first, second], 4, 6), (prefix + [first, wrong_second], 4, 6), (prefix + [wrong_second, second], 4, 6)]
    assert [match for _, match in core.score_rows(model, rows, pad_token_id=1, batch_size=3)] == [True, False, False]


def test_complete_run_reports_core_with_v2_baselines(exported, tmp_path):
    model, tokenizer, _ = exported
    for task in core.TASKS:  # 11 examples: enough for every 10-shot task
        if task.kind == "multiple_choice":
            items = [{"query": f"q{i}", "choices": ["a", "b"], "gold": i % 2} for i in range(11)]
        elif task.kind == "schema":
            items = [{"context_options": [f"x{i}", f"y{i}"], "continuation": "z", "gold": i % 2} for i in range(11)]
        else:
            items = [{"context": f"c{i}", "continuation": "w"} for i in range(11)]
        path = tmp_path / "eval_data" / task.dataset
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(item) + "\n" for item in items))
    result = core.evaluate_core(model, tokenizer, tmp_path)
    assert result["complete"] is True and len(result["tasks"]) == 22
    assert result["core"] == result["mean_centered"] == pytest.approx(sum(row["centered"] for row in result["tasks"].values()) / 22)
    assert result["core_v1"] == pytest.approx(sum(row["centered_v1"] for row in result["tasks"].values()) / 22)
    row = result["tasks"]["commonsense_qa"]
    assert row["examples"] == 11 and row["random_baseline"] == pytest.approx(0.403)
    assert row["centered"] == pytest.approx((row["correct"] / 11 - 0.403) / 0.597)
    assert row["centered_v1"] == pytest.approx((row["correct"] / 11 - 0.2) / 0.8)


def test_selection_is_checked_before_any_evaluation(exported):
    model, tokenizer, _ = exported
    with pytest.raises(ValueError, match="largest selected few-shot count"):
        core.evaluate_core(model, tokenizer, BUNDLE, tasks=["copa", "arc_easy"], max_per_task=10)
    with pytest.raises(ValueError, match="unknown"):
        core.evaluate_core(model, tokenizer, BUNDLE, tasks=["mmlu"])


def test_cli_writes_versioned_core_result_offline(exported, tmp_path):
    model, tokenizer, directory = exported
    output = tmp_path / "core.json"
    command = [sys.executable, "-m", "moeloop.core", "--model", str(directory), "--bundle", str(BUNDLE),
               "--output", str(output), "--tasks", *FIXTURE_TASKS, "--batch-size", "2"]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(SOURCE), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(tmp_path / "hf-home"))
    result = subprocess.run(command, cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text())
    assert set(payload) == {"format", "version", "core_version", "core", "core_v1", "mean_centered", "mean_centered_v1",
                            "complete", "tasks", "max_per_task", "batch_size", "bos_token_id", "max_tokens", "fewshot",
                            "bundle", "model", "versions"}
    assert (payload["format"], payload["version"], payload["core_version"]) == ("moeloop-core", 1, "dclm-v2")
    assert payload["complete"] is False and payload["core"] is None and payload["core_v1"] is None
    assert (payload["bos_token_id"], payload["max_tokens"], payload["max_per_task"], payload["batch_size"]) == (1, 128, None, 2)
    assert payload["fewshot"]["shuffle_seed"] == 1337 and payload["fewshot"]["sample_seed"] == 1234
    assert payload["bundle"]["sha256"] == core.directory_sha256(BUNDLE) and payload["bundle"]["official"] is False
    assert payload["model"]["sha256"] == core.directory_sha256(directory) and payload["model"]["dtype"] == "torch.float32"
    assert set(payload["tasks"]) == set(FIXTURE_TASKS)
    for label, row in payload["tasks"].items():
        task = core.TASK_BY_LABEL[label]
        data = core.load_examples(BUNDLE, task)
        rows = core.task_rows(task, data, tokenizer, 1, 128)
        correct = sum(core.example_correct(task, data, rows, [_oracle(model, *row[1:]) for row in rows]))
        assert row == {
            "task_type": task.kind, "num_fewshot": 0, "continuation_delimiter": " ", "dataset": task.dataset,
            "dataset_sha256": hashlib.sha256((BUNDLE / "eval_data" / task.dataset).read_bytes()).hexdigest(),
            "examples": 2, "correct": correct, "accuracy": correct / 2,
            "random_baseline": pytest.approx(0.01 * float(task.baseline)), "centered": pytest.approx(core.centered(correct / 2, task.baseline)),
            "random_baseline_v1": pytest.approx(0.01 * float(task.baseline_v1)), "centered_v1": pytest.approx(core.centered(correct / 2, task.baseline_v1)),
        }
    assert payload["mean_centered"] == pytest.approx(sum(row["centered"] for row in payload["tasks"].values()) / 3)
    again = subprocess.run(command, cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=120)
    assert again.returncode != 0 and "already exists" in again.stderr


def test_bundle_info_names_the_official_source(capsys):
    core.main(["--bundle-info"])
    printed = capsys.readouterr().out
    assert core.BUNDLE_URL in printed and core.BUNDLE_ZIP_SHA256 in printed and core.BUNDLE_SHA256 in printed
    with pytest.raises(SystemExit):
        core.main(["--model", "m"])
