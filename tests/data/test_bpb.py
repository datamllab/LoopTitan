from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

import pytest

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")

from moeloop.data import manifest_byte_table
from moeloop.evaluate import bpb_fields, bpb_terms, evaluate_heldout, token_byte_tensor
from moeloop.model import ModelConfig, RecursiveLM
from moeloop.prepare import DCLM_TOKENIZER, prepare_dataset
from moeloop.tokens import BYTE_DECODER, NotByteLevelBPE, byte_table_sha256, token_byte_lengths


# tokenizer.json: a GPT-2-style byte-level BPE (tokenizers 0.23.3), trained with
# BpeTrainer(vocab_size=320, special_tokens=["<|endoftext|>"],
# initial_alphabet=ByteLevel.alphabet()) on four short mixed-script lines, plus
# one non-special added token "<think>" (ID 320).
FIXTURE = Path(__file__).parent / "fixtures" / "bpb"
EOT, THINK = 0, 320
TEXTS = ["hello world", "naïve café — 你好，世界 🙂", "tabs\tand\nnewlines  and   spaces", "quick 🦊 jumps over the lazy dog"]
TOKENIZER = {"id": "bpb-fixture", "revision": "fixed-v1", "vocab_size": 321, "bos_token_id": EOT, "eos_token_id": EOT, "pad_token_id": EOT}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(24)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    return RecursiveLM(ModelConfig(vocab_size=321, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2, num_experts=4,
                                   top_k=2, num_cells=1, num_visits=2, max_seq_len=64)).train()


def encode(text):
    return tokenizers.Tokenizer.from_file(str(FIXTURE / "tokenizer.json")).encode(text, add_special_tokens=False).ids


def prepare(path, *, assets=True, vocab_size=321):
    return prepare_dataset(path, {"text": {"validation": TEXTS}}, encode=encode, tokenizer={**TOKENIZER, "vocab_size": vocab_size},
                           tokenizer_directory=FIXTURE if assets else None)


def test_byte_decoder_is_gpt2_bytes_to_unicode():
    assert sorted(BYTE_DECODER.values()) == list(range(256))
    assert BYTE_DECODER["Ġ"] == 32 and BYTE_DECODER["Ċ"] == 10 and BYTE_DECODER["ĉ"] == 9
    assert BYTE_DECODER["!"] == 33 and BYTE_DECODER["ÿ"] == 255 and BYTE_DECODER["Ā"] == 0


def test_fixture_table_spells_exact_utf8_bytes_and_zeroes_special_and_added_tokens():
    table = token_byte_lengths(FIXTURE)
    assert len(table) == 321 and table[EOT] == table[THINK] == 0
    vocab = json.loads((FIXTURE / "tokenizer.json").read_text())["model"]["vocab"]
    id_to_token = {token_id: token for token, token_id in vocab.items()}
    for text in TEXTS:
        ids = encode(text)
        assert sum(table[token] for token in ids) == len(text.encode("utf-8"))
        assert b"".join(bytes(BYTE_DECODER[c] for c in id_to_token[token]) for token in ids) == text.encode("utf-8")
    # Documented limit: a literal added token in text is matched whole and counts 0 bytes.
    literal = "think <think> twice"
    assert sum(table[token] for token in encode(literal)) == len(literal.encode()) - len("<think>")
    expected = hashlib.sha256(b"".join(struct.pack("<I", length) for length in table)).hexdigest()
    assert byte_table_sha256(table) == expected == "a1953a9c4ec6adca5974a7ffefec9306acf53957bdc79a39bde48da43cfb40c6"


@pytest.mark.parametrize("change, message", [
    (lambda backend: backend["model"].update(type="WordPiece"), "byte-level BPE"),
    (lambda backend: backend.update(pre_tokenizer={"type": "Whitespace"}), "byte-level BPE"),
    (lambda backend: backend["pre_tokenizer"].update(add_prefix_space=True), "add_prefix_space"),
    (lambda backend: backend.update(normalizer={"type": "Sequence", "normalizers": [{"type": "NFC"}, {"type": "Lowercase"}]}), "normalizer"),
    (lambda backend: backend["model"]["vocab"].update({"a b": 300}), "alphabet"),
])
def test_non_byte_level_tokenizers_are_refused(tmp_path, change, message):
    backend = json.loads((FIXTURE / "tokenizer.json").read_text())
    change(backend)
    (tmp_path / "tokenizer.json").write_text(json.dumps(backend))
    with pytest.raises(NotByteLevelBPE, match=message):
        token_byte_lengths(tmp_path)


def test_qwen3_table_from_local_hf_cache():
    hub = pytest.importorskip("huggingface_hub")
    path = hub.try_to_load_from_cache(DCLM_TOKENIZER["id"], "tokenizer.json")
    if not isinstance(path, str) or hashlib.sha256(Path(path).read_bytes()).hexdigest() != DCLM_TOKENIZER["tokenizer_json_sha256"]:
        pytest.skip("fixed Qwen3 tokenizer.json is not in the local HF cache")
    table = token_byte_lengths(Path(path).parent)
    assert len(table) == DCLM_TOKENIZER["vocab_size"]
    assert table[DCLM_TOKENIZER["eos_token_id"]] == 0 and all(length == 0 for length in table[151643:])
    backend = tokenizers.Tokenizer.from_file(path)
    for text in [*TEXTS, "Ünïcödé — 日本語のテキスト، عربى 🧪\r\n"]:
        assert sum(table[token] for token in backend.encode(text, add_special_tokens=False).ids) == len(text.encode())


def hand_oracle(model):
    """Independent per-document forwards: (all-label CE, text-label CE, EOS CE, bytes)."""
    total = text = eos = 0.0
    with torch.no_grad():
        for document in TEXTS:
            tokens = [EOT, *encode(document), EOT]
            losses = torch.nn.functional.cross_entropy(model(torch.tensor([tokens[:-1]])).logits[0].float(),
                                                       torch.tensor(tokens[1:]), reduction="none").double()
            total += losses.sum().item()
            text += losses[:-1].sum().item()
            eos += losses[-1].item()
    return total, text, eos, sum(len(document.encode()) for document in TEXTS)


def test_heldout_bpb_matches_hand_computation_and_excludes_special_tokens(tmp_path, model):
    manifest = prepare(tmp_path / "data")
    result = evaluate_heldout(model, manifest, seq_len=64, batch_size=2)
    total, text, eos, size = hand_oracle(model)
    assert result["valid_tokens"] == sum(len(encode(document)) + 1 for document in TEXTS)
    assert result["label_bytes"] == size == 113
    assert result["ce_sum"] == pytest.approx(total, rel=1e-6)  # float32 sum reduction
    assert result["ce_sum_text"] == pytest.approx(text, rel=1e-6)
    assert result["ce_sum"] - result["ce_sum_text"] == pytest.approx(eos, abs=1e-4)
    assert result["bpb"] == pytest.approx(text / (math.log(2) * size), rel=1e-6)
    assert result["bpb"] == result["ce_sum_text"] / (math.log(2) * result["label_bytes"])
    assert result["byte_table_sha256"] == byte_table_sha256(token_byte_lengths(FIXTURE))
    assert result["bpb_unavailable"] is None


def test_manifest_byte_table_is_the_tokenizers_unpadded_table_or_a_reason(tmp_path):
    assert manifest_byte_table(prepare(tmp_path / "padded", vocab_size=384)) == (token_byte_lengths(FIXTURE), None)
    assert manifest_byte_table(prepare(tmp_path / "plain", assets=False)) == (None, "manifest has no tokenizer assets")
    with pytest.raises(ValueError, match="longer than the manifest vocab_size"):
        manifest_byte_table(prepare(tmp_path / "short", vocab_size=320))


def test_token_byte_tensor_pads_to_the_manifest_vocabulary(tmp_path):
    table = token_byte_lengths(FIXTURE)
    token_bytes, byte_table, reason = token_byte_tensor(prepare(tmp_path / "padded", vocab_size=384), torch.device("cpu"))
    assert byte_table == table and reason is None and token_bytes.dtype == torch.int64
    assert token_bytes.tolist() == [*table, *[0] * (384 - len(table))]
    assert token_byte_tensor(prepare(tmp_path / "plain", assets=False), "cpu") == (None, None, "manifest has no tokenizer assets")


def test_bpb_terms_and_fields_on_hand_labels():
    table = token_byte_lengths(FIXTURE)
    word = encode("café")
    labels = torch.tensor([[-100, EOT, *word], [THINK, word[0], -100, EOT]])
    label_bytes, text = bpb_terms(labels, torch.tensor([*table, 0, 0]))
    expected = [[0, 0, *(table[token] for token in word)], [0, table[word[0]], 0, 0]]
    assert label_bytes.tolist() == expected and sum(expected[0]) == len("café".encode())
    assert text.tolist() == [[value > 0 for value in row] for row in expected]
    fields = bpb_fields(table, None, 7.5, 6)
    assert fields == {"label_bytes": 6, "ce_sum_text": 7.5, "bpb": 7.5 / (math.log(2) * 6),
                      "byte_table_sha256": byte_table_sha256(table), "bpb_unavailable": None}
    assert bpb_fields(table, None, 0.0, 0)["bpb_unavailable"] == "no byte-bearing labels evaluated"
    assert bpb_fields(None, "reason", 0.0, 0) == {"label_bytes": None, "ce_sum_text": None, "bpb": None,
                                                  "byte_table_sha256": None, "bpb_unavailable": "reason"}


def test_padded_vocab_size_keeps_bpb_and_unpadded_table_hash(tmp_path):
    model = RecursiveLM(ModelConfig(vocab_size=384, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2, num_experts=4,
                                    top_k=2, num_cells=1, num_visits=2, max_seq_len=64))
    exact = evaluate_heldout(model, prepare(tmp_path / "exact"), seq_len=7, batch_size=2)
    padded = evaluate_heldout(model, prepare(tmp_path / "padded", vocab_size=384), seq_len=7, batch_size=2)
    assert padded["tokenizer"]["vocab_size"] == 384
    for key in ("ce_sum", "valid_tokens", "label_bytes", "ce_sum_text", "bpb", "byte_table_sha256"):
        assert padded[key] == exact[key], key
    assert padded["byte_table_sha256"] == byte_table_sha256(token_byte_lengths(FIXTURE))
    with pytest.raises(ValueError, match="longer than the manifest vocab_size"):
        evaluate_heldout(model, prepare(tmp_path / "short", vocab_size=320, assets=True), seq_len=7)


def test_existing_fields_are_unchanged_by_the_byte_table(tmp_path, model):
    with_assets = evaluate_heldout(model, prepare(tmp_path / "assets"), seq_len=7, batch_size=2)
    without = evaluate_heldout(model, prepare(tmp_path / "plain", assets=False), seq_len=7, batch_size=2)
    new = ("label_bytes", "ce_sum_text", "bpb", "byte_table_sha256", "bpb_unavailable")
    assert all(without[key] is None for key in new[:-1]) and all(with_assets[key] is not None for key in new[:-1])
    assert without["bpb_unavailable"] == "manifest has no tokenizer assets" and with_assets["bpb_unavailable"] is None
    old = set(without) - set(new) - {"manifest_fingerprint"}
    assert {key: with_assets[key] for key in old} == {key: without[key] for key in old}


def test_document_bytes_reconcile_across_block_continuations(tmp_path, model):
    manifest = prepare(tmp_path / "data")
    aggregate = evaluate_heldout(model, manifest, seq_len=5, batch_size=3)
    output = tmp_path / "documents.json"
    result = evaluate_heldout(model, manifest, seq_len=5, batch_size=3, document_loss_output=output)
    documents = json.loads(output.read_text())["documents"]
    assert [row["bytes"] for row in documents] == [len(document.encode()) for document in TEXTS]
    assert result["label_bytes"] == aggregate["label_bytes"] == sum(row["bytes"] for row in documents)
    assert result["ce_sum_text"] == pytest.approx(aggregate["ce_sum_text"], abs=1e-9)
    assert result["bpb"] == pytest.approx(aggregate["bpb"], rel=1e-12)


def test_bpb_is_unavailable_without_byte_bearing_labels(tmp_path, model):
    # "<think>" is one added token: its labels are THINK and EOS, both 0 bytes.
    manifest = prepare_dataset(tmp_path / "data", {"text": {"validation": ["<think>"]}}, encode=encode,
                               tokenizer=TOKENIZER, tokenizer_directory=FIXTURE)
    result = evaluate_heldout(model, manifest, seq_len=8)
    assert result["valid_tokens"] == 2 and result["label_bytes"] == 0 and result["ce_sum_text"] == 0.0
    assert result["bpb"] is None and result["byte_table_sha256"] is not None
    assert result["bpb_unavailable"] == "no byte-bearing labels evaluated"


def test_non_byte_level_manifest_reports_no_byte_fields(tmp_path, model):
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"<eot>": 0, "<unk>": 1, "hello": 2, "world": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    directory = tmp_path / "wordlevel"
    directory.mkdir()
    backend.save(str(directory / "tokenizer.json"))
    with pytest.raises(NotByteLevelBPE):
        token_byte_lengths(directory)
    manifest = prepare_dataset(tmp_path / "data", {"text": {"validation": ["hello world"]}}, encode=None,
        tokenizer={**TOKENIZER, "vocab_size": 4}, tokenizer_directory=directory)
    output = tmp_path / "documents.json"
    result = evaluate_heldout(model, manifest, seq_len=8, document_loss_output=output)
    assert result["valid_tokens"] == 3 and result["ce_sum"] > 0
    assert [result[key] for key in ("label_bytes", "ce_sum_text", "bpb", "byte_table_sha256")] == [None] * 4
    assert result["bpb_unavailable"] == "byte lengths need a byte-level BPE tokenizer (ByteLevel pre-tokenizer and decoder)"
    assert json.loads(output.read_text())["documents"][0]["bytes"] is None
