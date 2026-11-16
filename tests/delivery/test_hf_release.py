"""HF paired release: card numbers, dry-run listing, and the upload gates (mocked HfApi only)."""
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import hf_release
import readme_figures as figures

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = "acme"
EXPORT_PROVENANCE = b'{"fixture": "export provenance"}'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def p1_evidence(dclm_sha=""):
    """A P1-shaped evidence file with numbers that appear nowhere in the repo; it binds the real P1 recipe files."""
    def arm(updates, labels, hours, provenance):
        return {"completed_updates": updates, "actual_valid_labels": labels, "actual_native_gpu_hours": hours,
                "hf": {"export_provenance_sha256": provenance}}
    heldout = {split: {"reference_ce": 3.1111 + shift, "candidate_ce": 3.0555 + shift, "valid_labels": 1234567 + shift_labels}
               for split, shift, shift_labels in (("validation", 0.0, 0), ("test", 0.0123, 5))}
    budget = lambda name, ref, cand: {"budget": name, "reference": ref, "candidate": cand,
        "relative_difference": cand / ref - 1, "relative_tolerance": 0.05, "basis": "analytical_estimate"}
    recipes = {role: ROOT / "recipes" / name for role, name in (("reference", "p1_moe.json"), ("candidate", "p1_smelt.json"))}
    artifacts = [{"path": f"recipes/{path.name}", "sha256": sha(path.read_bytes())} for path in recipes.values()]
    return {"status": "passed", "pair": {"reference_recipe": "recipes/p1_moe.json", "candidate_recipe": "recipes/p1_smelt.json"},
        "artifacts": artifacts + [{"path": "docs/evidence/p1-dclm-source.json", "sha256": dclm_sha}],
        "training_source": {"commit": "a" * 40, "dirty": False},
        "data": {"manifest_fingerprint": "f" * 64}, "hardware": {"gpu_model": "Test GPU", "nodes": 3, "gpus_per_node": 4,
            "world_size": 12, "dp_replicate": 3, "dp_shard": 4},
        "arms": {"reference": arm(1111, 2_222_222, 12.5, sha(EXPORT_PROVENANCE)),
                 "candidate": arm(1111, 2_222_222, 16.25, sha(EXPORT_PROVENANCE))},
        "same_D_full_HF_heldout": heldout,
        "matched_analytical_budgets": [budget("unique_non_embedding_parameters", 1_000_000, 1_034_000),
                                       budget("training_flops_per_token", 2_000_000.0, 1_950_000.0),
                                       budget("logical_kv_bytes", 4096, 4096)]}


@pytest.fixture
def plan(tmp_path):
    """Plan on disk with two export directories; each has one extra file the plan does not list."""
    dclm = json.dumps({"id": "org/dclm-fixture", "revision": "r" * 40, "license": {"id": "CC-BY-4.0"}})
    (tmp_path / "dclm.json").write_text(dclm)
    (tmp_path / "evidence.json").write_text(json.dumps(p1_evidence(sha(dclm.encode()))))
    (tmp_path / "extra.json").write_text(json.dumps({"bpb": {"core": [0.5, 0.25]}, "a/b": {"c~d": 0.125}}))
    models = {}
    for role, repo, recipe in (("looped", "fixture-smelt", "p1_smelt"), ("control", "fixture-moe", "p1_moe")):
        export = tmp_path / repo
        export.mkdir()
        contents = {"config.json": f'{{"role": "{role}"}}'.encode(), "model.safetensors": role.encode() * 100,
                    "export_provenance.json": EXPORT_PROVENANCE}
        for name, data in contents.items():
            (export / name).write_bytes(data)
        (export / "unlisted.bin").write_bytes(b"never upload me")
        models[role] = {"repo": repo, "recipe": "SMELT" if role == "looped" else "MoE", "export_dir": repo,
            "recipe_file": str(ROOT / "recipes" / f"{recipe}.json"), "checkpoint_sha256": sha(role.encode()),
            "files": {name: sha(data) for name, data in contents.items()}}
    models["looped"]["paper"] = "https://arxiv.org/abs/2609.01343"
    models["looped"]["metrics"] = {"BPB": "extra.json#/bpb/core/1"}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"pairs": [{"name": "fixture", "evidence": {"anchor": "P1", "file": "evidence.json"},
                                           "dclm_source": "dclm.json", **models}]}))
    return path


def release(plan, role):
    return next(item for item in hf_release.load_plan(plan, NAMESPACE) if item["role"] == role)


def test_card_has_required_sections_and_evidence_numbers(plan):
    evidence = p1_evidence()
    card = release(plan, "looped")["card"]
    assert card.startswith("---\nlicense: mit\nlibrary_name: transformers\n")
    assert '- "org/dclm-fixture"' in card and '- "looped-moe"' in card
    for heading in ("## Recipe", "## Matched analytical budgets", "## Data", "## Training", "## Evaluation",
                    "## Provenance", "## Usage", "## Scope", "## Paired model"):
        assert heading in card
    assert "SMELT (`p1_smelt`)" in card and "https://arxiv.org/abs/2609.01343" in card
    # Budgets: this model, paired model, residual against the control.
    assert "| `unique_non_embedding_parameters` | 1,034,000 | 1,000,000 | +3.40% | 5.00% |" in card
    assert "| `training_flops_per_token` | 1.9500e+06 | 2.0000e+06 | -2.50% | 5.00% |" in card
    assert "| `logical_kv_bytes` | 4,096 | 4,096 | +0.00% | 5.00% |" in card
    # Data, training, evaluation.
    assert "revision `" + "r" * 40 + "` (license CC-BY-4.0)" in card and "`" + "f" * 64 + "`" in card
    assert "Training label tokens: 2,222,222" in card and "Updates: 1,111" in card
    assert "12 x Test GPU (3 nodes), HSDP 3 x 4; measured native GPU hours 16.25" in card
    heldout = evidence["same_D_full_HF_heldout"]
    assert f"| validation | {heldout['validation']['candidate_ce']:.4f} | 1,234,567 |" in card
    assert f"| test | {heldout['test']['candidate_ce']:.4f} | 1,234,572 |" in card
    assert "- BPB: 0.2500" in card
    # Provenance and scope.
    assert "`" + "a" * 40 + "` (clean)" in card and sha(b"looped") in card
    assert f"`model.safetensors`: `{sha(b'looped' * 100)}`" in card
    assert "trust_remote_code=True" in card and "'acme/fixture-smelt'" in card
    assert "Single seed" in card and "seed 42" in card
    # The plan gives an absolute recipe path; the public card shows the file name only.
    assert "- Recipe file `p1_smelt.json`: SHA-256 `" in card and str(ROOT) not in card and str(plan.parent) not in card
    assert "16.25 (looped) and 12.50 (control), +30.0%" in card


def test_cards_are_paired_and_share_residuals(plan):
    looped, control = release(plan, "looped")["card"], release(plan, "control")["card"]
    assert "[acme/fixture-moe](https://huggingface.co/acme/fixture-moe)" in looped
    assert "[acme/fixture-smelt](https://huggingface.co/acme/fixture-smelt)" in control
    assert "| `unique_non_embedding_parameters` | 1,000,000 | 1,034,000 | +3.40% | 5.00% |" in control
    assert '- "moe-control"' in control and "none (ordinary MoE control)" in control
    assert "measured native GPU hours 12.50" in control


def test_real_evidence_p1_and_p2_are_read_without_new_parsers():
    p1 = hf_release.pair_facts("P1", ROOT / "docs/evidence/p1.json")
    assert p1["budgets"]["unique_non_embedding_parameters"] == {"reference": 712656672, "candidate": 737845056,
        "relative_difference": 0.03534434600789088, "relative_tolerance": 0.05}
    assert p1["arms"]["candidate"]["heldout"]["validation"] == {"ce": 3.0083626831157475, "valid_labels": 10001497}
    assert p1["arms"]["reference"]["heldout"]["test"]["ce"] == 3.0092780095669243
    assert (p1["arms"]["reference"]["updates"], p1["arms"]["reference"]["labels"]) == (19537, 20486029312)
    assert p1["gpus"] == "16 x NVIDIA H200 (2 nodes), HSDP 2 x 8"
    assert p1["manifest_fingerprint"] == "e67a956d2fe51dd1784505a6f58d7790f397a09624330700c0ebaffcd344f152"
    p2 = hf_release.pair_facts("P2", ROOT / "docs/evidence/recipes-v3.json")
    assert p2["budgets"]["training_flops_per_token"]["relative_difference"] == -0.022556881807818414
    assert p2["arms"]["candidate"]["heldout"]["test"] == {"ce": 2.8730771860706414, "valid_labels": 10009889}
    assert p2["arms"]["candidate"]["gpu_hours"] == 148.03375262949203
    assert p2["gpus"] == "16 x NVIDIA H200, HSDP 2 x 8" and p2["source"]["commit"] == "17d51e4df9e807f81c5d31301441daba9ce527b2"


def test_real_p1_card_renders_literal_evidence_numbers(tmp_path):
    facts = hf_release.pair_facts("P1", ROOT / "docs/evidence/p1.json")
    source = json.loads((ROOT / "docs/evidence/p1-dclm-source.json").read_text())
    model = {"repo": "p1-smelt", "recipe": "SMELT", "recipe_file": "recipes/p1_smelt.json",
             "checkpoint_sha256": "c" * 64, "files": {"config.json": "d" * 64}}
    repos = {"looped": "acme/p1-smelt", "control": "acme/p1-moe"}
    card = hf_release.render_card("looped", model, repos, facts, source, ROOT)
    assert "| `unique_non_embedding_parameters` | 737,845,056 | 712,656,672 | +3.53% | 5.00% |" in card
    assert "| validation | 3.0084 | 10,001,497 |" in card and "| test | 3.0034 | 10,009,889 |" in card
    assert "revision `817d6752765f6a41261085171dd546b104f60626` (license CC-BY-4.0)" in card
    assert "Updates: 19,537" in card and "measured native GPU hours 103.43" in card
    assert "103.43 (looped) and 78.02 (control), +32.6%" in card


def test_dry_run_lists_exact_files_and_hashes(plan, capsys, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    api = MagicMock()
    monkeypatch.setattr(hf_release, "hf_client", api)
    assert hf_release.main([str(plan), "--namespace", NAMESPACE]) == 0
    out = capsys.readouterr().out
    export = plan.parent / "fixture-smelt"
    assert "== acme/fixture-smelt (fixture, looped)" in out and "== acme/fixture-moe (fixture, control)" in out
    for name in ("config.json", "model.safetensors", "export_provenance.json"):
        data = (export / name).read_bytes()
        assert f"{sha(data)}  {len(data):>14,}  {name}" in out
    assert "unlisted.bin" not in out and out.count("generated README.md follows") == 2
    assert "--- README.md ---\n---\nlicense: mit" in out
    api.assert_not_called()


def test_dry_run_flags_hash_mismatch_with_nonzero_exit(plan, capsys):
    (plan.parent / "fixture-moe" / "config.json").write_text("tampered")
    assert hf_release.main([str(plan), "--namespace", NAMESPACE, "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert out.count("MISMATCH") == 1 and "config.json  MISMATCH" in out


def test_upload_needs_the_flag_and_a_token(plan, monkeypatch, capsys):
    api = MagicMock()
    monkeypatch.setattr(hf_release, "hf_client", api)
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    assert hf_release.main([str(plan), "--namespace", NAMESPACE]) == 0  # token present, no flag: dry-run
    monkeypatch.delenv("HF_TOKEN")
    assert hf_release.main([str(plan), "--namespace", NAMESPACE, "--upload"]) == 1
    assert "HF_TOKEN" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        hf_release.main([str(plan), "--namespace", NAMESPACE, "--upload", "--dry-run"])
    api.assert_not_called()


def test_upload_refuses_when_plan_hash_differs(plan, monkeypatch, capsys):
    api = MagicMock()
    monkeypatch.setattr(hf_release, "hf_client", api)
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    (plan.parent / "fixture-smelt" / "model.safetensors").write_bytes(b"changed after the plan was made")
    assert hf_release.main([str(plan), "--namespace", NAMESPACE, "--upload"]) == 1
    assert "hash differs from plan: acme/fixture-smelt/model.safetensors" in capsys.readouterr().err
    api.assert_not_called()  # nothing is sent, not even the intact control repo


def test_upload_rechecks_hashes_after_the_plan_was_loaded(plan, monkeypatch):
    api = MagicMock()
    monkeypatch.setattr(hf_release, "hf_client", api)
    releases = hf_release.load_plan(plan, NAMESPACE)
    (plan.parent / "fixture-moe" / "config.json").write_text("changed later")
    with pytest.raises(ValueError, match="refusing upload.*acme/fixture-moe/config.json"):
        hf_release.upload(releases, "fixture-token")
    api.assert_not_called()


def mock_client(monkeypatch):
    api = MagicMock()
    add = lambda path_in_repo, path_or_fileobj: (path_in_repo, path_or_fileobj)
    client = MagicMock(return_value=(api, add))
    monkeypatch.setattr(hf_release, "hf_client", client)
    return api, client


def test_upload_commits_only_listed_files_and_cards_once_per_repo(plan, monkeypatch):
    api, client = mock_client(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    assert hf_release.main([str(plan), "--namespace", NAMESPACE, "--upload"]) == 0
    client.assert_called_once_with("fixture-token")
    api.upload_file.assert_not_called()
    assert [call.kwargs["repo_id"] for call in api.create_commit.call_args_list] == ["acme/fixture-smelt", "acme/fixture-moe"]
    for call in api.create_commit.call_args_list:
        assert {name for name, _ in call.kwargs["operations"]} == {"README.md", "config.json", "model.safetensors", "export_provenance.json"}
    assert all(call.kwargs["private"] is True for call in api.create_repo.call_args_list)
    control = next(call for call in api.create_commit.call_args_list if call.kwargs["repo_id"] == "acme/fixture-moe")
    assert dict(control.kwargs["operations"])["README.md"] == release(plan, "control")["card"].encode()
    assert dict(control.kwargs["operations"])["config.json"] == str(plan.parent / "fixture-moe" / "config.json")


def test_public_flag_creates_public_repos(plan, monkeypatch):
    api, _ = mock_client(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    assert hf_release.main([str(plan), "--namespace", NAMESPACE, "--upload", "--public"]) == 0
    assert [call.kwargs["private"] for call in api.create_repo.call_args_list] == [False, False]


def test_failed_commit_leaves_no_partial_upload_calls(plan, monkeypatch):
    api, _ = mock_client(monkeypatch)
    api.create_commit.side_effect = RuntimeError("network")
    with pytest.raises(RuntimeError):
        hf_release.upload(hf_release.load_plan(plan, NAMESPACE), "fixture-token")
    assert api.create_commit.call_count == 1 and api.upload_file.call_count == 0


def test_json_pointer_escapes_and_list_index(plan):
    assert hf_release.metric("extra.json#/bpb/core/0", plan.parent) == 0.5
    assert hf_release.metric("extra.json#/a~1b/c~0d", plan.parent) == 0.125
    with pytest.raises(ValueError, match="not a number"):
        hf_release.metric("extra.json#/a~1b", plan.parent)


def test_card_paper_line_follows_the_role(plan):
    document = json.loads(plan.read_text())
    del document["pairs"][0]["looped"]["paper"]
    plan.write_text(json.dumps(document))
    assert "- Paper: none recorded" in release(plan, "looped")["card"]
    assert "- Paper: none (ordinary MoE control)" in release(plan, "control")["card"]


def test_swapped_recipe_or_dclm_file_is_refused(plan):
    original = plan.read_text()
    document = json.loads(original)
    document["pairs"][0]["looped"]["recipe_file"] = str(ROOT / "recipes" / "p1_moe.json")  # control recipe on the looped model
    plan.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="fixture-smelt: recipe file differs from the evidence"):
        hf_release.load_plan(plan, NAMESPACE)
    plan.write_text(original)
    (plan.parent / "dclm.json").write_text(json.dumps({"id": "org/dclm-fixture", "revision": "x" * 40, "license": {"id": "CC-BY-4.0"}}))
    with pytest.raises(ValueError, match="DCLM source differs from the evidence"):
        hf_release.load_plan(plan, NAMESPACE)


def test_real_p1_evidence_binds_the_repo_recipes_and_dclm_source():
    evidence = ROOT / "docs/evidence/p1.json"
    facts = hf_release.pair_facts("P1", evidence)
    for role, name in (("reference", "p1_moe"), ("candidate", "p1_smelt")):
        assert facts["recipes"][role] == sha((ROOT / "recipes" / f"{name}.json").read_bytes())
    assert hf_release.bound_source(facts, evidence, ROOT / "docs/evidence/p1-dclm-source.json")["revision"].startswith("817d6752")
    with pytest.raises(ValueError, match="differs from the evidence"):
        hf_release.bound_source(facts, evidence, ROOT / "docs/evidence/dclm-v1.json")


def test_real_p2_evidence_binds_recipes_and_source_through_the_data_manifest(tmp_path):
    evidence = ROOT / "docs/evidence/recipes-v3.json"
    facts = hf_release.pair_facts("P2", evidence)
    for role, name in (("reference", "p2_moe"), ("candidate", "p2_smelt")):
        assert facts["recipes"][role] == sha((ROOT / "recipes" / f"{name}.json").read_bytes())
    source_file = ROOT / "docs/evidence/p1-dclm-source.json"
    assert hf_release.bound_source(facts, evidence, source_file)["id"] == "mlfoundations/dclm-baseline-1.0-parquet"
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**json.loads(source_file.read_text()), "revision": "0" * 40}))
    with pytest.raises(ValueError, match="differs from the data manifest"):
        hf_release.bound_source(facts, evidence, other)


def test_real_p2_card_renders_both_roles_with_evidence_numbers():
    evidence = ROOT / "docs/evidence/recipes-v3.json"
    facts = hf_release.pair_facts("P2", evidence)
    source = json.loads((ROOT / "docs/evidence/p1-dclm-source.json").read_text())
    repos = {"looped": "acme/p2-smelt", "control": "acme/p2-moe"}
    model = lambda recipe: {"repo": recipe, "recipe": recipe, "recipe_file": f"recipes/{recipe}.json",
                            "checkpoint_sha256": "c" * 64, "files": {"config.json": "d" * 64}}
    looped = hf_release.render_card("looped", model("p2_smelt"), repos, facts, source, ROOT)
    control = hf_release.render_card("control", model("p2_moe"), repos, facts, source, ROOT)
    assert "| `unique_non_embedding_parameters` | 1,314,771,744 | 1,318,288,640 | -0.27% | 5.00% |" in looped
    assert "| `unique_non_embedding_parameters` | 1,318,288,640 | 1,314,771,744 | -0.27% | 5.00% |" in control
    assert "| `training_flops_per_token` | 2.5158e+09 | 2.5739e+09 | -2.26% | 5.00% |" in looped
    assert "| validation | 2.8780 | 10,001,497 |" in looped and "| validation | 2.8923 | 10,001,497 |" in control
    assert "| test | 2.8731 | 10,009,889 |" in looped and "| test | 2.8873 | 10,009,889 |" in control
    assert "16 x NVIDIA H200, HSDP 2 x 8; measured native GPU hours 148.03" in looped
    assert "measured native GPU hours 135.02" in control
    assert "148.03 (looped) and 135.02 (control), +9.6%" in looped
    assert "`17d51e4df9e807f81c5d31301441daba9ce527b2` (clean)" in looped
    assert "Training label tokens: 20,486,029,312" in control and "Updates: 19,537" in control
    assert "[acme/p2-moe](https://huggingface.co/acme/p2-moe)" in looped and "[acme/p2-smelt]" in control


@pytest.mark.parametrize("name", ["README.md", "../outside.txt", "/etc/passwd", "missing.bin"])
def test_plan_rejects_unsafe_or_missing_file_names(plan, name):
    document = json.loads(plan.read_text())
    document["pairs"][0]["looped"]["files"][name] = "0" * 64
    (plan.parent / "outside.txt").write_text("outside the export directory")
    plan.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        hf_release.load_plan(plan, NAMESPACE)


def test_export_must_carry_the_evidence_bound_provenance_hash(plan):
    document = json.loads(plan.read_text())
    del document["pairs"][0]["looped"]["files"]["export_provenance.json"]
    plan.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="export_provenance.json is not listed"):
        hf_release.load_plan(plan, NAMESPACE)


def test_dirty_or_unpassed_evidence_is_refused(plan):
    evidence = plan.parent / "evidence.json"
    document = json.loads(evidence.read_text())
    document["training_source"]["dirty"] = True
    evidence.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="clean exact commit"):
        hf_release.load_plan(plan, NAMESPACE)
    document["training_source"]["dirty"], document["status"] = False, "failed"
    evidence.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="not 'passed'"):
        hf_release.load_plan(plan, NAMESPACE)


# --- single models bound to the Phase 1 results ---------------------------------------------

EXAMPLE = ROOT / "docs/release/phase1-models.example.json"
RESULTS = json.loads((ROOT / "docs/results/phase1.json").read_text())
RUNS = {run["config"]: run for group in RESULTS["groups"] for row in group["rows"] for run in row["runs"].values()}


@pytest.fixture
def single_plan(tmp_path):
    """The example plan with two models filled in from made-up exports whose evaluation files hold the recorded BPB."""
    plan = json.loads(EXAMPLE.read_text())
    plan["results"], plan["dataset"] = str(ROOT / "docs/results/phase1.json"), str(ROOT / "docs/evidence/p1-dclm-source.json")
    models = {model["config"]: model for model in plan["models"]}
    plan["models"] = [models["recipes/phase1/m3-muon.json"], models["recipes/phase1/m0-muon.json"]]
    for model in plan["models"]:
        run, export = RUNS[model["config"]], tmp_path / model["repo"]
        contents = {"config.json": model["repo"].encode(), "model.safetensors": b"weights" * 10,
                    "eval/validation.json": json.dumps({"split": "validation", "bpb": run["validation_bpb"]}).encode(),
                    "eval/test.json": json.dumps({"split": "test", "bpb": run["test_bpb"]}).encode()}
        for name, data in contents.items():
            (export / name).parent.mkdir(parents=True, exist_ok=True)
            (export / name).write_bytes(data)
        model.update(export_dir=model["repo"], evaluation={"validation": "eval/validation.json", "test": "eval/test.json"},
                     files={name: sha(data) for name, data in contents.items()})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_example_plan_names_the_sixteen_models_and_binds_each_to_one_run():
    plan = json.loads(EXAMPLE.read_text())
    stems = ["moe12", "moe12-finer", "moe18", "moe18-layout", "twin", "loop6x2", "moe12-loop6x2-unmatched", "moe12-2e"]
    assert [model["repo"] for model in plan["models"]] == [f"looptitan-p1-{stem}-{optimizer}" for stem in stems
                                                           for optimizer in ("adamw", "muon")]
    assert all(model["config"] in RUNS and model["config"].endswith(f"-{model['repo'].rsplit('-', 1)[1]}.json")
               for model in plan["models"])
    assert len({model["config"] for model in plan["models"]}) == 16


def test_example_dry_run_renders_every_card_and_the_collection(capsys):
    assert hf_release.main([str(EXAMPLE), "--namespace", "example"]) == 0
    out = capsys.readouterr().out
    assert out.count("--- README.md ---") == 16 and out.count("not filled in: export_dir, files and evaluation") == 16
    assert "== collection example: LoopTitan Phase 1" in out
    assert "  1. example/looptitan-p1-moe12-adamw\n" in out and "  16. example/looptitan-p1-moe12-2e-muon\n" in out


def test_single_card_states_the_run_from_the_results_file(single_plan):
    card = hf_release.load_plan(single_plan, NAMESPACE)[0]["card"]
    run, reference = RUNS["recipes/phase1/m3-muon.json"], RUNS["recipes/phase1/m0-muon.json"]
    assert card.startswith("---\nlicense: mit\nlibrary_name: transformers\npipeline_tag: text-generation\n")
    assert '- "looped-moe"' in card and '- "muon"' in card and '- "mlfoundations/dclm-baseline-1.0-parquet"' in card
    assert f"    - name: Test BPB\n      type: bpb\n      value: {run['test_bpb']:.6f}" in card
    assert f"    - name: Validation BPB\n      type: bpb\n      value: {run['validation_bpb']:.6f}" in card
    assert "![12 layers stored, 18 run](banner.svg)" in card
    assert "\n# Looped MoE: the middle 6 layers run twice\n\nMuon · LoopTitan Phase 1\n\nChain-end loop: 12 layers at width 1056" in card
    assert f"| 0.970676 | 0.972425 | {1000 * (run['test_bpb'] - reference['test_bpb']):+.2f} mBPB |" in card
    assert "| -3.70 mBPB |" in card and "Matched budgets: all three within 5% of the ordinary MoE (Muon)." in card
    assert ("![Budgets against the ordinary MoE (Muon): unique non-embedding parameters -0.27%, training FLOPs per "
            "token +2.44%, KV cache +3.12%; tolerance ±5%](budgets.svg)") in card
    assert ("5.0B tokens of [mlfoundations/dclm-baseline-1.0-parquet](https://huggingface.co/datasets/mlfoundations/"
            "dclm-baseline-1.0-parquet) at revision `817d6752765f6a41261085171dd546b104f60626` (CC-BY-4.0): 4,769 updates "
            "of 1,048,576 tokens in sequences of 4,096, warmup-stable-decay schedule, seed 42. Muon at learning rate "
            "1.7e-3, the value tuned once per optimizer on the ordinary MoE; the same grid on this model selects the "
            "same value ([#180](https://github.com/datamllab/LoopTitan/issues/180)).") in card
    assert RESULTS["protocol"] not in card and "every row" not in card and card.lower().count("chain-end") == 1
    assert 'repo = "acme/looptitan-p1-loop6x2-muon"' in card and "trust_remote_code=True" in card
    for link in ("https://github.com/datamllab/LoopTitan/blob/main/recipes/phase1/m3-muon.json",
                 "https://github.com/datamllab/LoopTitan/blob/main/docs/results/phase1.md",
                 "https://github.com/datamllab/LoopTitan/tree/main/docs", "https://github.com/datamllab/LoopTitan/issues/180",
                 "https://github.com/datamllab/LoopTitan/blob/main/LICENSE"):
        assert link in card
    assert f"## Citation\n\n```bibtex\n{figures.bibtex(yaml.safe_load((ROOT / 'CITATION.cff').read_text()))}\n```\n" in card
    assert str(single_plan.parent) not in card and str(ROOT) not in card


PAPER = ("@misc{looptitan2026,\n  title = {{A Paper Title With Capitals}},\n  author = {{The LoopTitan authors}},\n"
         "  year = {2026},\n  eprint = {2611.00001},\n  archivePrefix = {arXiv},\n  primaryClass = {cs.LG},\n"
         "  url = {https://arxiv.org/abs/2611.00001}\n}")


def test_card_cites_the_preferred_citation_of_citation_cff_and_only_that(single_plan, monkeypatch):
    fixture = ROOT / "tests" / "delivery" / "fixtures" / "citation-paper.cff"
    cff = yaml.safe_load(fixture.read_text())
    assert figures.bibtex(cff) == PAPER
    monkeypatch.setattr(hf_release, "CITATION", fixture)
    card = hf_release.load_plan(single_plan, NAMESPACE)[0]["card"]
    assert f"## Citation\n\n```bibtex\n{PAPER}\n```\n" in card and "@software" not in card
    journal = cff["preferred-citation"] | {"journal": "Journal of Tests", "identifiers": [], "doi": "10.0000/test"}
    assert figures.bibtex(cff | {"preferred-citation": journal}) == (
        "@article{looptitan2026,\n  title = {{A Paper Title With Capitals}},\n  author = {{The LoopTitan authors}},\n"
        "  year = {2026},\n  journal = {Journal of Tests},\n  doi = {10.0000/test},\n"
        "  url = {https://arxiv.org/abs/2611.00001}\n}")


def test_unmatched_row_says_so():
    plan = json.loads(EXAMPLE.read_text())
    plan["models"] = [model for model in plan["models"] if model["repo"] == "looptitan-p1-moe12-2e-adamw"]
    plan["results"], plan["dataset"] = str(ROOT / "docs/results/phase1.json"), str(ROOT / "docs/evidence/p1-dclm-source.json")
    card = hf_release.single_releases(plan, EXAMPLE.parent, NAMESPACE)[0]["card"]
    assert "# Ordinary MoE with twice the experts, outside matched budgets\n\nAdamW · LoopTitan Phase 1" in card
    assert ("Outside matched budgets: more than 5% from the ordinary MoE (AdamW) in unique non-embedding "
            "parameters (+96.7%).") in card and '- "moe"' in card and "![12 layers](banner.svg)" in card


@pytest.mark.parametrize("repo, sentence", [
    ("looptitan-p1-moe12-muon", "Muon at learning rate 1.7e-3, tuned on this model, once per optimizer ([#180]"),
    ("looptitan-p1-loop6x2-adamw", "AdamW at learning rate 1.2e-3, the value tuned once per optimizer on the ordinary MoE; "
                                   "the same grid on this model selects the same value ([#180]"),
    ("looptitan-p1-twin-muon", "Muon at learning rate 1.7e-3, inherited from the ordinary MoE, where it was tuned once "
                               "per optimizer ([#180]"),
    ("looptitan-p1-moe12-2e-adamw", "AdamW at learning rate 1.2e-3, inherited from the ordinary MoE")])
def test_learning_rate_sentence_states_where_the_value_was_tuned(repo, sentence):
    plan = json.loads(EXAMPLE.read_text())
    plan["models"] = [model for model in plan["models"] if model["repo"] == repo]
    card = hf_release.single_releases(plan, EXAMPLE.parent, NAMESPACE)[0]["card"]
    assert sentence in card and card.count("learning rate") == 1


@pytest.mark.parametrize("name", ["README.md", "banner.svg", "budgets.svg"])
def test_single_plan_refuses_a_file_with_a_generated_name(single_plan, name):
    document = json.loads(single_plan.read_text())
    document["models"][0]["files"][name] = "0" * 64
    single_plan.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="README.md, banner.svg, budgets.svg are generated"):
        hf_release.load_plan(single_plan, NAMESPACE)


def test_budgets_figure_uses_the_readme_band_and_marks_values_outside_it():
    def draw(config, optimizer):
        run = RUNS[config]
        return hf_release.budgets(config, f"recipes/phase1/m0-{optimizer}.json", run, optimizer)
    inside, outside = draw("recipes/phase1/m3-muon.json", "muon"), draw("recipes/methods/m0-loop-adamw.json", "adamw")
    square = 'width="10" height="10" rx="2" style="fill:#16181D"'
    assert "within ±5%" in inside and square not in inside and inside.count('r="5.5"') == 3
    assert "outside ±5%" in outside and outside.count(square) == 2 and outside.count('r="5.5"') == 1
    assert "+31.46%" in outside and "+50.00%" in outside and "tolerance ±5%" in outside


def test_banner_draws_layers_and_the_loop_from_the_configuration():
    def draw(config):
        return hf_release.banner(config, json.loads((ROOT / config).read_text())["model"])
    looped, plain = draw("recipes/phase1/m3-muon.json"), draw("recipes/phase1/m0-muon.json")
    acc, bar = (f"fill:{figures.LIGHT[key]}\"/>" for key in ("acc", "bar"))
    assert looped.count(acc) == 6 and looped.count(bar) == 6 and "×2</text>" in looped
    assert "12 layers stored, 18 run</text>" in looped and "#3346D3" not in plain
    assert plain.count(bar) == 12 and "12 layers</text>" in plain
    assert figures.glyph("phase1/m3", 0, 0, figures.LIGHT, k=2)[0] in looped  # the README's drawing, light variant
    mark = (ROOT / "docs" / "assets" / "brand" / "looptitan-lockup-medium-ink.svg").read_text()
    mark = mark[mark.index(">") + 1:mark.rindex("</svg>")]
    assert mark in looped and mark in plain  # the lockup, placed as it is


def test_evaluation_bpb_must_reproduce_the_results(single_plan):
    document = json.loads(single_plan.read_text())
    model = document["models"][1]
    test = single_plan.parent / model["export_dir"] / "eval/test.json"
    test.write_text(json.dumps({"split": "test", "bpb": RUNS[model["config"]]["test_bpb"] + 2e-6}))
    model["files"]["eval/test.json"] = sha(test.read_bytes())
    single_plan.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="looptitan-p1-moe12-muon: test BPB 0.974378 of eval/test.json differs from 0.974376"):
        hf_release.load_plan(single_plan, NAMESPACE)
    assert hf_release.main([str(single_plan), "--namespace", NAMESPACE]) == 1


def test_evaluation_files_must_be_listed_and_configs_must_be_runs(single_plan):
    original = single_plan.read_text()
    document = json.loads(original)
    del document["models"][0]["files"]["eval/validation.json"]
    single_plan.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="evaluation file 'eval/validation.json' is not listed"):
        hf_release.load_plan(single_plan, NAMESPACE)
    document = json.loads(original)
    document["models"][0]["config"] = "recipes/phase1/m1-muon.json"
    single_plan.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="recipes/phase1/m1-muon.json is not a run of"):
        hf_release.load_plan(single_plan, NAMESPACE)


def test_single_dry_run_lists_files_and_generated_banner(single_plan, capsys):
    assert hf_release.main([str(single_plan), "--namespace", NAMESPACE]) == 0
    out = capsys.readouterr().out
    assert "== acme/looptitan-p1-loop6x2-muon (recipes/phase1/m3-muon.json)" in out
    assert "  eval/test.json\n" in out and out.count("  generated banner.svg (") == 2
    assert "== collection acme: LoopTitan Phase 1\n" in out and "  2. acme/looptitan-p1-moe12-muon" in out


def test_single_upload_commits_card_banner_and_files_then_the_collection(single_plan, monkeypatch):
    api, client = mock_client(monkeypatch)
    repos = ["acme/looptitan-p1-loop6x2-muon", "acme/looptitan-p1-moe12-muon"]
    api.create_collection.return_value.slug = "acme/looptitan-phase-1-0123"
    api.add_collection_item.return_value.items = [MagicMock(item_id=repo, item_object_id=f"id-{repo}") for repo in reversed(repos)]
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    assert hf_release.main([str(single_plan), "--namespace", NAMESPACE, "--upload"]) == 0
    releases = hf_release.load_plan(single_plan, NAMESPACE)
    assert [call.kwargs["repo_id"] for call in api.create_commit.call_args_list] == repos
    first = dict(api.create_commit.call_args_list[0].kwargs["operations"])
    assert set(first) == {"README.md", "banner.svg", "budgets.svg", "config.json", "model.safetensors", "eval/validation.json", "eval/test.json"}
    assert first["README.md"] == releases[0]["card"].encode() and first["banner.svg"] == releases[0]["assets"]["banner.svg"]
    plan = json.loads(EXAMPLE.read_text())["collection"]
    api.create_collection.assert_called_once_with(title=plan["title"], namespace=NAMESPACE, description=plan["description"],
                                                  private=True, exists_ok=True)
    assert [call.kwargs["item_id"] for call in api.add_collection_item.call_args_list] == repos
    assert [(call.args[1], call.kwargs["position"]) for call in api.update_collection_item.call_args_list] == \
        [(f"id-{repo}", position) for position, repo in enumerate(repos)]


def test_upload_refuses_a_plan_with_an_export_not_filled_in(monkeypatch, capsys):
    api, _ = mock_client(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    assert hf_release.main([str(EXAMPLE), "--namespace", NAMESPACE, "--upload"]) == 1
    assert "export not filled in: acme/looptitan-p1-moe12-adamw" in capsys.readouterr().err
    api.create_repo.assert_not_called()
    api.create_collection.assert_not_called()
