#!/usr/bin/env python3
"""Plan, render and (only on request) upload Hugging Face releases of single models or pairs; never train.

Example: python scripts/hf_release.py plan.json --namespace my-org            # dry-run
         HF_TOKEN=... python scripts/hf_release.py plan.json --namespace my-org --upload

A plan of single models (docs/release/phase1-models.example.json) names a results file
(docs/results/phase1.json), the dataset's source receipt, a collection (title, description) and a
list of `models`. Each model gives its repo name, a `title` in plain words, the `config` path that binds it to one run of the
results file, its local `export_dir`, the exact `files` to upload with their SHA-256, and its
`evaluation`: the names, among those files, of the `python -m moeloop.evaluate` outputs for the
validation and test splits. Their `bpb` must equal the run's validation_bpb and test_bpb to 6
decimals, or the whole run is refused. The card and its figures under assets/ (banner, where the
model stands among the designs of the plan, budgets, the README's destination capsules; light, dark
and narrow files) are generated with the README figures' functions; every number on them comes from
the results file, the recipe and the P2 track, and the reference of the BPB difference and of the
budgets is the ordinary MoE under the same optimizer. The collection title names the release on each
card; each model's `short` name labels the figures and its `note` opens its collection note. The
plan's `org_card` is the organisation card's text.
A model whose `export_dir` is still empty is rendered in the dry-run and refused by `--upload`.
With `--upload`, after the repos exist, one collection in the namespace lists them in plan order.
`--cards` uploads only each card and its figures to existing repos (no export needed, earlier
banner.svg and budgets.svg removed); `--collection` applies the collection's texts; `--org-card`
uploads the organisation card to the existing Space <namespace>/README. None creates a model repo or
changes visibility.

A pair plan lists pairs. Each pair names one evidence file (anchor P1: docs/evidence/p1.json,
anchor P2: docs/evidence/recipes-v3.json), a DCLM source receipt, and two models: `looped`
(the evidence `candidate` arm) and its matched ordinary-MoE `control` (the `reference` arm).
Every model gives its repo name, display recipe, local export directory, recipe file, checkpoint
SHA-256 and the exact files to upload with their SHA-256. Relative paths resolve against the
plan file. Optional model keys: `paper` (URL) and `metrics` ({"BPB": "evidence.json#/json/pointer"}).

The default mode only prints each file's size and SHA-256, the target repo id and the rendered
model card. `--upload` needs the flag and HF_TOKEN; it re-hashes every listed file, refuses the
whole run if one differs from the plan, then uploads only the listed files plus the generated
README.md. One commit per repo, so a failed
upload leaves no half-filled repo (a pair is not atomic). New repos are private unless `--public`.
A file changed between the check and its transfer is not detected. `huggingface_hub` arrives
through `transformers` (the `hf` extra); it is imported only for `--upload`.

The evidence binds the recipe files, the DCLM source and (P1) the export provenance file by hash;
a plan whose files differ is refused. It records no checkpoint hash, so `checkpoint_sha256` is
shown on the card as declared by the plan. The card states what the evidence supports; it makes no judgement.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys

import yaml

from check_release import require
from moeloop.bench import load_tracks
from moeloop.comparison import compare_recipes
from recipe_delivery import read_json, sha256
import readme_figures as figures
from results_table import METRICS, OPTIMIZERS, REFERENCE, lr, signed, spec

ROOT = Path(__file__).resolve().parents[1]
CITATION = ROOT / "CITATION.cff"
ROLES = {"looped": "candidate", "control": "reference"}
SPLITS = ("validation", "test")
NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
GENERATED = ("README.md", "assets/")  # written by this script for a single model: the card and its figures
SUPERSEDED = ("banner.svg", "budgets.svg")  # figures of earlier cards, removed by a card update; nothing else is
# The learning rate was tuned on the ordinary MoE (REFERENCE) and the same grid run on this loop; the track's
# lr_selection (eval/bench/tracks.json) names both models only in prose (M0 and M3), so the path is named here.
SAME_GRID = "recipes/phase1/m3-{}.json"


# --- evidence readers: one normalized view over the P1 and P2 publications ---------------

def _budgets(rows):
    return {name: {key: row[key] for key in ("reference", "candidate", "relative_difference", "relative_tolerance")}
            for name, row in rows.items()}


def read_p1(doc):
    arms = {}
    for role in ROLES.values():
        arm = doc["arms"][role]
        arms[role] = {"updates": arm["completed_updates"], "labels": arm["actual_valid_labels"],
            "gpu_hours": arm["actual_native_gpu_hours"], "export_provenance_sha256": arm["hf"]["export_provenance_sha256"],
            "heldout": {split: {"ce": doc["same_D_full_HF_heldout"][split][f"{role}_ce"],
                "valid_labels": doc["same_D_full_HF_heldout"][split]["valid_labels"]} for split in SPLITS}}
    hardware, artifacts = doc["hardware"], {item["path"]: item["sha256"] for item in doc["artifacts"]}
    return {"source": doc["training_source"], "data_manifest": None,
        "dclm_sha256": artifacts["docs/evidence/p1-dclm-source.json"],
        "recipes": {role: artifacts[doc["pair"][f"{role}_recipe"]] for role in ROLES.values()}, "manifest_fingerprint": doc["data"]["manifest_fingerprint"],
        "gpus": f"{hardware['world_size']} x {hardware['gpu_model']} ({hardware['nodes']} nodes), "
                f"HSDP {hardware['dp_replicate']} x {hardware['dp_shard']}",
        "budgets": _budgets({row["budget"]: row for row in doc["matched_analytical_budgets"]}), "arms": arms}


def read_p2(doc):
    p2, runtime = doc["p2"], doc["p2"]["runtime"]
    arms = {role: {"updates": p2["arms"][role]["updates"], "labels": p2["arms"][role]["labels"],
        "gpu_hours": p2["arms"][role]["actual_gpu_hours"],
        "heldout": {split: {"ce": p2["arms"][role]["heldout"][split]["ce"],
            "valid_labels": p2["arms"][role]["heldout"][split]["valid_tokens"]} for split in SPLITS}}
        for role in ROLES.values()}
    parallelism = runtime["parallelism"]
    return {"source": runtime["source"], "dclm_sha256": None, "data_manifest": doc["data_publication"],
        "recipes": {role: p2["public_recipe_files"][role]["sha256"] for role in ROLES.values()},
        "manifest_fingerprint": doc["manifest_fingerprint"],
        "gpus": f"{runtime['world_size']} x {runtime['runtime']['gpu_name']}, "
                f"HSDP {parallelism['data_parallel_replicate_degree']} x {parallelism['data_parallel_shard_degree']}",
        "budgets": _budgets(p2["matched"]), "arms": arms}


READERS = {"P1": read_p1, "P2": read_p2}


def pair_facts(anchor, path):
    require(anchor in READERS, f"unknown evidence anchor {anchor!r}; known: {sorted(READERS)}")
    doc = read_json(path)
    require(doc.get("status") == "passed", f"{path}: evidence status is not 'passed'")
    facts = READERS[anchor](doc)
    require(facts["source"].get("dirty") is False and re.fullmatch(r"[0-9a-f]{40}", facts["source"].get("commit", "")),
            f"{path}: training source must be a clean exact commit")
    return {**facts, "anchor": anchor, "evidence_sha256": sha256(path)}


def metric(reference, base):
    """Resolve 'file#/json/pointer' (RFC 6901) against `base`; return the number found there."""
    name, _, pointer = reference.partition("#")
    value = read_json(base / name)
    for part in pointer.split("/")[1:]:
        value = value[int(part)] if isinstance(value, list) else value[part.replace("~1", "/").replace("~0", "~")]
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{reference}: not a number")
    return value


# --- plan ---------------------------------------------------------------------------------

def load_plan(path, namespace, cards_only=False):
    """Return one release per model: repo id, rendered card and the checked file listing."""
    require(NAMESPACE.fullmatch(namespace), f"invalid namespace {namespace!r}")
    path = Path(path).resolve()
    base, plan = path.parent, read_json(path)
    if "models" in plan:
        return single_releases(plan, base, namespace, cards_only)
    require(not cards_only, "--cards needs a plan of single models")
    require(plan.get("pairs"), "plan has no pairs")
    releases = []
    for pair in plan["pairs"]:
        evidence = base / pair["evidence"]["file"]
        facts = pair_facts(pair["evidence"]["anchor"], evidence)
        source = bound_source(facts, evidence, base / pair["dclm_source"])
        models = {name: pair[name] for name in ROLES}
        repos = {name: f"{namespace}/{model['repo']}" for name, model in models.items()}
        require(repos["looped"] != repos["control"], f"{pair.get('name')}: looped and control share a repo")
        for name, model in models.items():
            recipe = base / model["recipe_file"]
            require(sha256(recipe) == facts["recipes"][ROLES[name]], f"{model['repo']}: recipe file differs from the evidence")
            files = checked_files(model, base, facts["arms"][ROLES[name]].get("export_provenance_sha256"))
            card = render_card(name, model, repos, facts, source, base)
            releases.append({"pair": pair["name"], "role": name, "label": f"{pair['name']}, {name}",
                             "repo_id": repos[name], "card": card, "assets": {}, "files": files})
    return releases


def family_of(plan, runs, matched):
    """The designs of the plan in plan order (models sharing a `short` name), each with its run per optimizer."""
    family = {}
    for model in plan["models"]:
        row, optimizer, run = runs[model["config"]]
        reference = runs[REFERENCE.format(optimizer)][2]
        design = family.setdefault(model["short"], {"short": model["short"], "title": model["title"], "configs": [],
                                                     "matched": matched[model["config"]], "runs": {}})
        design["configs"].append(model["config"])
        design["runs"][optimizer] = {"config": model["config"], "repo": model["repo"], "test_bpb": run["test_bpb"],
                                     "delta": 1000 * (run["test_bpb"] - reference["test_bpb"])}
    return list(family.values())


def single_releases(plan, base, namespace, cards_only=False):
    """One release per model of the plan, in plan order, bound to its run of the results file. With cards_only,
    only the card and its figures: the export (weights, evaluation files) is neither needed nor checked."""
    results = read_json(base / plan["results"])
    page = (base / plan["results"]).resolve().relative_to(ROOT).with_suffix(".md").as_posix()
    runs = {run["config"]: (row, optimizer, run) for group in results["groups"] for row in group["rows"]
            for optimizer, run in row["runs"].items()}
    matched = {run["config"]: group["matched"] for group in results["groups"] for row in group["rows"]
               for run in row["runs"].values()}
    dataset, cff = read_json(base / plan["dataset"]), yaml.safe_load(CITATION.read_text())
    for model in plan["models"]:
        require(model["config"] in runs, f"{model['repo']}: {model['config']} is not a run of {plan['results']}")
    family = family_of(plan, runs, matched)
    releases = []
    for model in plan["models"]:
        row, optimizer, run = runs[model["config"]]
        files = None
        if model["export_dir"] and not cards_only:
            require(not any(name == "README.md" or name.startswith("assets/") for name in model["files"]),
                    f"{model['repo']}: {', '.join(GENERATED)} are generated")
            files = checked_files(model, base, None)
            for split in SPLITS:
                name, expected = model["evaluation"][split], run[f"{split}_bpb"]
                require(name in model["files"], f"{model['repo']}: evaluation file {name!r} is not listed in files")
                bpb = read_json(base / model["export_dir"] / name)["bpb"]
                require(f"{bpb:.6f}" == f"{expected:.6f}", f"{model['repo']}: {split} BPB {bpb:.6f} of {name} "
                        f"differs from {expected:.6f} in {plan['results']}")
        reference, recipe = REFERENCE.format(optimizer), read_json(ROOT / model["config"])
        card = model_card(f"{namespace}/{model['repo']}", model, row, optimizer, run, runs[reference][2],
                          plan["collection"]["title"], page, dataset, cff, recipe, family)
        releases.append({"repo_id": f"{namespace}/{model['repo']}", "label": model["config"], "card": card,
                         "title": model["title"], "optimizer": optimizer,
                         "assets": card_assets(model, recipe, run, reference, optimizer, family), "files": files,
                         "note": collection_note(model, run, runs[reference][2], matched[model["config"]])})
    return releases


def collection_note(model, run, reference, matched):
    """The collection's plain-text note of one model: the plan's words, then the numbers of the results file."""
    delta, tolerance = 1000 * (run["test_bpb"] - reference["test_bpb"]), load_tracks()["P2"].budget["tolerance"]
    if model["config"] == reference["config"]:
        return f"{model['note']} Test BPB {run['test_bpb']:.4f}."
    return (f"{model['note']} Test BPB {run['test_bpb']:.4f}, {signed(delta, 1)} mBPB against the ordinary MoE with "
            f"the same optimizer; " + (f"budgets matched within {tolerance:.0%}." if matched else "outside matched budgets."))


def bound_source(facts, evidence, source_file):
    """The DCLM source receipt, required to be the one the evidence records (P2: through its data manifest)."""
    source = read_json(source_file)
    if facts["dclm_sha256"]:
        require(sha256(source_file) == facts["dclm_sha256"], f"{source_file}: DCLM source differs from the evidence")
    else:
        manifest = evidence.parent / facts["data_manifest"]["path"]
        require(sha256(manifest) == facts["data_manifest"]["sha256"], f"{manifest}: differs from the evidence")
        dataset = read_json(manifest)["provenance"]["dataset"]
        require((dataset["id"], dataset["revision"]) == (source["id"], source["revision"]),
                f"{source_file}: DCLM source differs from the data manifest")
    return source


def checked_files(model, base, export_provenance):
    """List the planned files with their actual size and hash; `ok` is False on any difference."""
    directory = (base / model["export_dir"]).resolve()
    require(model["files"], f"{model['repo']}: no files listed")
    rows = []
    for name, expected in sorted(model["files"].items()):
        pure = PurePosixPath(name)
        require(name != "README.md" and not pure.is_absolute() and ".." not in pure.parts,
                f"{model['repo']}: file name {name!r} is not an allowed relative path (README.md is generated)")
        require(re.fullmatch(r"[0-9a-f]{64}", expected), f"{model['repo']}/{name}: plan hash is not a SHA-256")
        file = (directory / pure).resolve()
        require(file.is_relative_to(directory) and file.is_file(), f"{model['repo']}: missing file {name}")
        actual = sha256(file)
        rows.append({"name": name, "path": file, "size": file.stat().st_size, "sha256": actual,
                     "expected": expected, "ok": actual == expected})
    if export_provenance:  # the evidence binds the export through this file's hash
        require(model["files"].get("export_provenance.json") == export_provenance,
                f"{model['repo']}: export_provenance.json is not listed with the evidence hash")
    return rows


# --- model card ---------------------------------------------------------------------------

def count(value):
    return f"{value:,}" if isinstance(value, int) else f"{value:.4e}"


def card_metric(value):
    return f"{value:.4f}" if isinstance(value, float) else f"{value:,}"


def render_card(role, model, repos, facts, source, base):
    """Markdown model card with HF front matter; every number is read from the evidence."""
    mine, theirs = ROLES[role], ROLES["control" if role == "looped" else "looped"]
    arm, paired = facts["arms"][mine], repos["control" if role == "looped" else "looped"]
    recipe = read_json(base / model["recipe_file"])
    run, optimizer, schedule = recipe["run"], recipe["optimizer"], recipe["run"]["schedule"]
    looped, control = facts["arms"]["candidate"], facts["arms"]["reference"]
    kind = "Looped (weight-shared depth) MoE" if role == "looped" else "Ordinary MoE control"
    tags = ["moeloop", "mixture-of-experts", "looped-moe" if role == "looped" else "moe-control",
            "custom_code", facts["anchor"].lower()]
    lines = ["---", "license: mit", "library_name: transformers", "pipeline_tag: text-generation",
             "tags:", *[f"- {json.dumps(tag)}" for tag in tags], "datasets:", f"- {json.dumps(source['id'])}", "---", "",
             f"# {model['repo']}", "",
             f"{kind}, trained from scratch for the {facts['anchor']} paired comparison of "
             f"[moeloop-lab](https://github.com/datamllab/LoopTitan). Base language model; not instruction tuned.", "",
             "## Recipe", "", f"- Recipe: {model['recipe']} (`{recipe['name']}`)",
             f"- Paper: {model['paper']}" if model.get("paper") else
             ("- Paper: none recorded" if role == "looped" else "- Paper: none (ordinary MoE control)"), "",
             "## Matched analytical budgets", "",
             f"Residual = (looped - control) / control, from analytical estimates; the paired model is "
             f"[{paired}](https://huggingface.co/{paired}).", "",
             "| Budget | This model | Paired model | Residual | Tolerance |", "|---|---|---|---|---|"]
    for name, row in facts["budgets"].items():
        lines.append(f"| `{name}` | {count(row[mine])} | {count(row[theirs])} | "
                     f"{row['relative_difference']:+.2%} | {row['relative_tolerance']:.2%} |")
    lines += ["", "## Data", "",
              f"- Dataset: [{source['id']}](https://huggingface.co/datasets/{source['id']}) "
              f"revision `{source['revision']}` (license {source['license']['id']})",
              f"- Prepared-data manifest fingerprint: `{facts['manifest_fingerprint']}`",
              f"- Training label tokens: {arm['labels']:,}", "", "## Training", "",
              f"- Updates: {arm['updates']:,} (sequence length {run['seq_len']}, "
              f"global batch {run['global_batch_positions']:,} positions)",
              f"- GPUs: {facts['gpus']}; measured native GPU hours {arm['gpu_hours']:.2f}",
              f"- Optimizer: {optimizer['name']}, peak LR {optimizer['base_lr']}, weight decay {optimizer['weight_decay']}",
              f"- Schedule: `{schedule['name']}`, {schedule['total_steps']:,} updates "
              f"({schedule['warmup_steps']} warmup, {schedule['decay_steps']:,} {schedule['decay_type']} decay "
              f"to {schedule['min_lr_factor']} x peak)", f"- Seed: {run['seed']}", "", "## Evaluation", "",
              "Full held-out cross-entropy (nats per label token), same data for both models.", "",
              "| Split | CE | Label tokens |", "|---|---|---|"]
    lines += [f"| {split} | {arm['heldout'][split]['ce']:.4f} | {arm['heldout'][split]['valid_labels']:,} |" for split in SPLITS]
    lines += [f"- {name}: {card_metric(metric(reference, base))}" for name, reference in model.get("metrics", {}).items()]
    lines += ["", "## Provenance", "",
              f"- Training source commit: `{facts['source']['commit']}` (clean)",
              f"- Native checkpoint SHA-256 (declared by the release plan): `{model['checkpoint_sha256']}`",
              f"- Evidence: `{facts['anchor']}`, SHA-256 `{facts['evidence_sha256']}`",
              f"- Recipe file `{Path(model['recipe_file']).name}`: SHA-256 `{sha256(base / model['recipe_file'])}`",
              "- Export files (SHA-256):", "",
              *[f"  - `{name}`: `{digest}`" for name, digest in sorted(model["files"].items())], "",
              "## Usage", "",
              "The model runs repository Python code: read it first, or pin `revision=`.", "", "```python",
              "from transformers import AutoModelForCausalLM, AutoTokenizer", "",
              f"model = AutoModelForCausalLM.from_pretrained({repos[role]!r}, trust_remote_code=True)",
              f"tokenizer = AutoTokenizer.from_pretrained({repos[role]!r})", "```", "", "## Scope", "",
              f"- Single seed: one training run (seed {run['seed']}) per model. Differences against the paired model "
              f"are descriptive, not a significance estimate.",
              f"- Cost: matching is analytical. Measured native GPU hours were {looped['gpu_hours']:.2f} (looped) and "
              f"{control['gpu_hours']:.2f} (control), {looped['gpu_hours'] / control['gpu_hours'] - 1:+.1%}; "
              f"equal analytical budgets do not mean equal wall-clock cost.",
              "- One scale and one public dataset; a declared adaptation of the recipe, not the paper's training stack.",
              "- No instruction tuning, safety evaluation or claim of state-of-the-art quality.", "",
              "## Paired model", "",
              f"[{paired}](https://huggingface.co/{paired}): "
              f"the {'ordinary MoE control' if role == 'looped' else 'looped model'} matched to this one.", ""]
    return "\n".join(lines)


BUDGET_NAMES = {"unique_non_embedding_parameters": "unique non-embedding parameters",
                "training_flops_per_token": "training FLOPs per token", "logical_kv_bytes": "KV cache"}


def model_card(repo_id, model, row, optimizer, run, reference, release, page, dataset, cff, recipe, family):
    """Markdown card of one model: every number comes from its row of the results file, its recipe or the track.
    The figures are the README's drawing, one file per theme (assets/*-light.svg, assets/*-dark.svg)."""
    repo, name, config, train = cff["repository-code"], OPTIMIZERS[optimizer], recipe["model"], recipe["run"]
    tolerance = load_tracks()["P2"].budget["tolerance"]
    outside = {key: value for key, value in run["budgets"].items() if abs(value) > tolerance}
    tags = ["looptitan", "mixture-of-experts", "looped-moe" if config["num_visits"] > 1 else "moe", "custom_code", optimizer]
    designs = [design for design in family if design["runs"]]
    place = next(i for i, design in enumerate(designs, 1) if model["config"] in design["configs"])
    delta = 1000 * (run["test_bpb"] - reference["test_bpb"])
    lines = ["---", f"license: {cff['license'].lower()}", "library_name: transformers", "pipeline_tag: text-generation",
             "tags:", *[f"- {json.dumps(tag)}" for tag in tags], "datasets:", f"- {json.dumps(dataset['id'])}",
             "model-index:", f"- name: {json.dumps(model['repo'])}", "  results:",
             "  - task:", "      type: text-generation", "    dataset:", f"      name: {json.dumps(dataset['id'])}",
             f"      type: {json.dumps(dataset['id'])}", "    metrics:"]
    for split in ("test", "validation"):
        lines += [f"    - name: {split.capitalize()} BPB", "      type: bpb", f"      value: {run[f'{split}_bpb']:.6f}"]
    lines += ["---", "", themed("banner", banner_text(config), 720), "", capsule_row(repo, page), "",
              f"# {model['title']}", "",
              f"**{name}** · {release} · design {place} of {len(designs)}, each trained with AdamW and with Muon", "",
              f"{row['design'][0].upper()}{row['design'][1:]}. A base language model trained from scratch, "
              "not instruction tuned.", "",
              "## Results", "", f"| Test BPB | Validation BPB | Δ test vs ordinary MoE ({name}) |", "|---:|---:|---:|",
              f"| {run['test_bpb']:.6f} | {run['validation_bpb']:.6f} | {signed(delta, 2)} mBPB |", "",
              themed("standing", standing_text(model, family), 720), "",
              "Bits per byte on the complete held-out test and validation splits after the last update; lower is better. "
              f"The ordinary MoE is [`{reference['config']}`]({repo}/blob/main/{reference['config']}); "
              "1 mBPB is 0.001 BPB.", "", "## Budgets", "",
              f"Outside matched budgets: more than {tolerance:.0%} from the ordinary MoE ({name}) in " +
              " and ".join(f"{BUDGET_NAMES[key]} ({signed(100 * value, 1)}%)" for key, value in outside.items()) + "."
              if outside else f"Matched budgets: all three within {tolerance:.0%} of the ordinary MoE ({name}).", "",
              themed("budgets", budgets_text(run, name, tolerance), 720), "",
              "## Training", "",
              f"{train['successful_updates'] * train['global_batch_positions'] / 1e9:.1f}B tokens of "
              f"[{dataset['id']}](https://huggingface.co/datasets/{dataset['id']}) at revision `{dataset['revision']}` "
              f"({dataset['license']['id']}): {train['successful_updates']:,} updates of {train['global_batch_positions']:,} "
              f"tokens in sequences of {train['seq_len']:,}, {train['schedule']['name'].replace('_', '-')} schedule, "
              f"seed {train['seed']}. {name} at learning rate {lr(recipe['optimizer']['base_lr'])}, " +
              ("tuned on this model, once per optimizer" if model["config"] == reference["config"] else
               "the value tuned once per optimizer on the ordinary MoE; the same grid on this model selects the same value"
               if model["config"] == SAME_GRID.format(optimizer) else
               "inherited from the ordinary MoE, where it was tuned once per optimizer") +
              f" ([#{reference['issue']}]({repo}/issues/{reference['issue']})). The full protocol: "
              f"[results page]({repo}/blob/main/{page}). Configuration: [`{model['config']}`]({repo}/blob/main/{model['config']}); "
              f"run record: [#{run['issue']}]({repo}/issues/{run['issue']}).", "",
              "## Use", "", "```python", "import torch", "from transformers import AutoModelForCausalLM, AutoTokenizer", "",
              f"repo = {json.dumps(repo_id)}", "tokenizer = AutoTokenizer.from_pretrained(repo)",
              "model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()",
              "ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer.encode(\"The model\", add_special_tokens=False)])",
              "out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=32, do_sample=False)",
              "print(tokenizer.decode(out[0], skip_special_tokens=True))", "```", "",
              "The model runs Python code from this repository: read it first, or pin `revision=`.", "",
              f"## The {release} models", "",
              f"{len(designs)} designs, each trained with AdamW and with Muon on the same data, schedule and seed. "
              "Test BPB; the model of this page in bold.", "",
              "| Design | AdamW | Muon |", "|:---|---:|---:|"]
    namespace = repo_id.split("/")[0]
    for design in designs:
        cells = []
        for opt in OPTIMIZERS:
            other = design["runs"].get(opt)
            if other is None:
                cells.append("")
                continue
            link = f"[{other['test_bpb']:.4f}](https://huggingface.co/{namespace}/{other['repo']})"
            cells.append(f"**{other['test_bpb']:.4f}**" if other["config"] == model["config"] else link)
        lines.append(f"| {design['title']} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Citation", "",
              "If you use this model or its results, please cite LoopTitan; this applies to people and to agents that "
              "build on it:", "", "```bibtex", figures.bibtex(cff), "```", "",
              "## Licence", "", f"[{cff['license']}]({repo}/blob/main/LICENSE). The LoopTitan authors.", ""]
    return "\n".join(lines)


# --- card figures: the README's drawing, one file per Hugging Face theme ---------------------

# Hugging Face shows card images in its own light or dark theme, chosen by a class on the page, not by the
# reader's system: each figure is drawn twice and the card shows one by Tailwind's dark: classes. The dark
# variant is drawn on Hugging Face's dark ground.
THEMES = {"light": figures.LIGHT, "dark": figures.DARK | {"card": "#0B0F19", "panel": "#0B0F19"}}
W = 720  # the card's content width on a desktop page; the images scale down on a phone
# The destination row of the README, for a model page: (key, label, colour family, mark, link or None)
HF_PLACES = [("homepage", "Homepage", "ours", "brand/looptitan-shell-small", None),
             ("code", "Code", "github", "octicons-mark-github-16", "{repo}"),
             ("docs", "Documentation", "ours", "octicons-book-16", "{repo}/tree/main/docs"),
             ("models", "Models", "hf", "huggingface-logo", None),
             ("paper", "Paper", "arxiv", "arxiv-logomark-small", None),
             ("results", "Results", "ours", "bars", "{repo}/blob/main/{page}")]


def themed(name, alt, width, narrow=True):
    """The light and the dark file of one figure, Hugging Face showing the one of its theme; and, below its
    md breakpoint (a phone), the narrow drawing of each, wrapped so that width and theme choose together."""
    def pair(suffix, w):
        return (f'<img class="not-prose dark:hidden" src="assets/{name}-light{suffix}.svg" width="{w}" alt="{alt}">'
                f'<img class="not-prose hidden dark:block" src="assets/{name}-dark{suffix}.svg" width="{w}" alt="{alt}">')
    if not narrow:
        return f'<p align="center">{pair("", width)}</p>'
    return (f'<div class="hidden md:block"><p align="center">{pair("", width)}</p></div>\n'
            f'<div class="md:hidden"><p align="center">{pair("-narrow", 360)}</p></div>')


def capsule_row(repo, page):
    """The README's destination capsules; a destination that is not public yet carries no link."""
    rows = []
    for theme, cls in (("light", "dark:hidden"), ("dark", "hidden dark:block")):
        parts = []
        for key, label, *_, link in HF_PLACES:
            img = (f'<img class="not-prose inline-block" src="assets/capsule-{key}-{theme}.svg" height="32" '
                   f'alt="{label}">')
            parts.append(f'<a href="{link.format(repo=repo, page=page)}">{img}</a>' if link else img)
        rows.append(f'<p align="center" class="{cls}">' + "".join(parts) + "</p>")
    return "\n".join(rows)


def card_assets(model, recipe, run, reference_path, optimizer, family):
    """Every figure of one card, both themes, by file name under assets/."""
    out = {}
    for theme, t in THEMES.items():
        for suffix, narrow in (("", False), ("-narrow", True)):
            out[f"assets/banner-{theme}{suffix}.svg"] = banner(model["config"], recipe["model"], optimizer, t, narrow)
            out[f"assets/standing-{theme}{suffix}.svg"] = standing(model, family, t, narrow)
            out[f"assets/budgets-{theme}{suffix}.svg"] = budgets(model["config"], reference_path, run,
                                                                  OPTIMIZERS[optimizer], t, narrow)
        for i, (key, label, colour, mark, _) in enumerate(HF_PLACES):
            out[f"assets/capsule-{key}-{theme}.svg"] = figures.capsule(key, label, colour, mark, i == 0, t,
                                                                         figures.HOSTS.get(key))
    return {name: svg.encode() for name, svg in out.items()}


def banner_text(config):
    stored = config["prefix_cells"] + config["num_cells"] + config["suffix_cells"]
    run = stored + (config["num_visits"] - 1) * config["num_cells"]
    return f"{stored} layers stored, {run} run" if run > stored else f"{stored} layers"


def banner(config_path, config, optimizer, t, narrow=False):
    """The lockup, the release, and the model as the README draws it: one bar per stored layer, its area the
    layer's expert parameters; looped layers in indigo under an arc with the visit count; the counts as numerals."""
    w, h, scale, k = (360, 212, 1.0, 2) if narrow else (W, 268, 1.6, 2)
    body, total = figures.glyph(config_path.removeprefix("recipes/").rsplit("-", 1)[0], 0, 0, t, k=k)
    scale = min(scale, (w - 48) / total)
    stored = config["prefix_cells"] + config["num_cells"] + config["suffix_cells"]
    run = stored + (config["num_visits"] - 1) * config["num_cells"]
    pad = 18 if narrow else 24
    out = [figures.card(0, 0, w, h, t),
           figures.lockup("medium-indigo" if t is THEMES["light"] else "medium-dark", pad, pad - 2, 22)[0],
           figures.mlabel(w - pad, pad + 14, f"Phase 1 · {OPTIMIZERS[optimizer]}", t, "end"),
           f'<g transform="translate({figures.num((w - scale * total) / 2)} {(h - 64) / 2 + 22}) scale({figures.num(scale)})">{body}</g>',
           figures.line(1, h - 64.5, w - 1, h - 64.5, t["line"], 1)]
    if not narrow:
        out.append(figures.text(pad, h - 26, config_path, 12, t["mid"], 400, mono=True))
    cols = [(w / 2 - 70 if narrow else w - 160, "layers stored", stored, t["ink"]),
            (w / 2 + 70 if narrow else w - pad, "layers run", run, t["acc"] if run > stored else t["ink"])]
    for x, label, value, colour in cols:
        anchor = "middle" if narrow else "end"
        out.append(figures.mlabel(x, h - 40, label, t, anchor))
        out.append(figures.text(x, h - 14, str(value), 22, colour, 600, anchor))
    return figures.svg(w, h, banner_text(config), banner_text(config), t, "".join(out))


def standing_text(model, family):
    rows = [f"{design['short']}: " + ", ".join(f"{OPTIMIZERS[opt]} {r['delta']:+.1f}" for opt, r in design["runs"].items())
            for design in family]
    return "Test mBPB of each Phase 1 design against the ordinary MoE, by optimizer: " + "; ".join(rows) + "."


def standing(model, family, t, narrow=False):
    """Where the model stands: every design of the release, test mBPB against the ordinary MoE with the same
    optimizer, AdamW and Muon side by side (one above the other when narrow); the row of this model on the
    loop's ground, its bar in ink and the others grey; a design outside matched budgets drawn open."""
    deltas = [r["delta"] for design in family for r in design["runs"].values()]
    lo, hi = 2 * math.floor(min(deltas) / 2), max(2, 2 * math.ceil(max(deltas) / 2))
    mine = next(i for i, design in enumerate(family) if model["config"] in design["configs"])
    pitch, n = (26, len(family)) if narrow else (30, len(family))
    if narrow:
        w, label_x, panels, top0 = 360, 18, [], 70
        for opt in OPTIMIZERS:
            panels.append((opt, 172, 342, top0))
            top0 += (n - 1) * pitch + 86
        h = top0 - 10
    else:
        w, label_x = W, 24
        panels = [("adamw", 268, 474, 84), ("muon", 504, 696, 84)]
        h = 84 + (n - 1) * pitch + 70
    out = [figures.card(0, 0, w, h, t),
           figures.text(label_x, 34, "Where it stands" if narrow else "Where it stands: test mBPB against the ordinary MoE",
                        14, t["ink"], 600),
           figures.text(w - label_x, 34, "test mBPB, ← lower is better" if narrow else "← lower is better", 12, t["mid"], 400, "end")]
    for opt, x0, x1, top in panels:
        def X(v):
            return x0 + (v - lo) / (hi - lo) * (x1 - x0)
        if narrow or opt == "adamw":
            out.append(figures.rect(10, top + mine * pitch - pitch / 2 + 2, w - 20, pitch - 4, 6, t["tint"]))
        out.append(figures.text(label_x if narrow else x0, top - 18, OPTIMIZERS[opt], 13, t["mid"], 600))
        bottom = top + (n - 1) * pitch + 12
        for v in range(lo, hi + 1, 4 if hi - lo > 12 else 2):
            out.append(figures.line(X(v), top - 14, X(v), bottom, t["hair"] if v == 0 else t["line"], 1))
            out.append(figures.text(X(v), bottom + 16, f"{v:+d}".replace("-", "−").replace("+0", "0"), 11, t["softt"], 400,
                                    "middle", mono=True))
        for i, design in enumerate(family):
            y = top + i * pitch
            if narrow or opt == "adamw":
                out.append(figures.text(label_x, y + 4.5, design["short"], 12.5 if narrow else 13, t["ink"],
                                        600 if i == mine else 400))
            r = design["runs"].get(opt)
            if r is None:
                continue
            here = r["config"] == model["config"]
            colour = t["ink"] if here else t["grey"]
            a, b = sorted((X(0), X(r["delta"])))
            out.append(figures.rect(a, y - 5, max(b - a, 2.5), 10, 1.5, colour) if design["matched"] else
                       figures.rect(a + .5, y - 4.5, max(b - a, 2.5) - 1, 9, 1.5, t["card"], colour, 1))
            value = "ref" if design["runs"][opt]["config"] == REFERENCE.format(opt) else f"{r['delta']:+.1f}".replace("-", "−")
            out.append(figures.text(a - 6, y + 4, value, 11 if not here else 12, t["ink"] if here else t["softt"],
                                    600 if here else 400, "end", mono=True))
    out.append(figures.rect(label_x, h - 25, 12, 9, 1.5, t["card"], t["grey"], 1))
    out.append(figures.text(label_x + 18, h - 17, "outside matched budgets", 12, t["mid"]))
    return figures.svg(w, h, "Where it stands", standing_text(model, family), t, "".join(out))


def budgets_text(run, name, tolerance):
    return (f"Budgets against the ordinary MoE ({name}): " +
            ", ".join(f"{BUDGET_NAMES[key]} {100 * run['budgets'][key]:+.2f}%" for key in METRICS) +
            f"; tolerance ±{tolerance:.0%}")


def budgets(config_path, reference_path, run, name, t=figures.LIGHT, narrow=False):
    """The three budgets in the README's band drawing, as the P2 track counts them; a budget outside its
    tolerance is drawn as the homepage's out-of-band mark, an ink square past the band's end."""
    track = load_tracks()["P2"]
    result = compare_recipes(spec(reference_path), spec(config_path), seq_len=track.seq_len,
                             tolerance=track.budget["tolerance"], **track.budget["accounting"])
    entries = result["matched"] | result["measured-different"]
    w, h, x, span = (360, 300, 18, 360 - 36 - 22) if narrow else (W, 248, 24, W - 64)
    inside = sum(abs(entry["relative_difference"]) <= entry["relative_tolerance"] for entry in entries.values())
    state = f"{inside} of {len(entries)} within ±{figures.pct(track.budget['tolerance'])}"
    out = [figures.card(0, 0, w, h, t),
           figures.text(x, 38, f"Budgets ({name})" if narrow else f"Budgets against the ordinary MoE ({name})", 14, t["ink"], 600),
           figures.okline(w - x, 34, entries, t, state)[0]]
    for i, (key, label) in enumerate(figures.BUDGETS):
        entry, y = entries[key], (84 + i * 72 if narrow else 92 + i * 54)
        row = figures.budget_row(t, entry, key, label, x, y, span, narrow)
        if abs(entry["relative_difference"]) > entry["relative_tolerance"]:
            row[-1] = figures.rect(x + span / 2 + math.copysign(span / 2 + 16, entry["relative_difference"]) - 5,
                                   y + 13, 10, 10, 2, t["ink"])
        out += row
    return figures.svg(w, h, "Budgets", budgets_text(run, name, track.budget["tolerance"]), t, "".join(out))


# --- dry-run and upload -------------------------------------------------------------------

def dry_run_report(releases, collection=None, org_card=None):
    lines = []
    for release in releases:
        lines += [f"== {release['repo_id']} ({release['label']})", "files to upload:"]
        if release["files"] is None:
            lines.append("  not filled in: export_dir, files and evaluation")
        lines += [f"  {row['sha256']}  {row['size']:>14,}  {row['name']}" + ("" if row["ok"] else
                  f"  MISMATCH (plan {row['expected']})") for row in release["files"] or []]
        lines += [f"  generated {name} ({len(data):,} bytes)" for name, data in release["assets"].items()]
        lines += ["  generated README.md follows", "--- README.md ---", release["card"], "--- end ---", ""]
    if collection:
        lines += [f"== collection {collection['namespace']}: {collection['title']}", f"  {collection['description']}"]
        for i, release in enumerate(releases, 1):
            lines += [f"  {i}. {release['repo_id']}"] + ([f"     note: {release['note']}"] if release.get("note") else [])
        lines.append("")
    if org_card:
        lines += [f"== organisation card {org_card['repo_id']}",
                  *[f"  generated {name} ({len(data):,} bytes)" for name, data in org_card["assets"].items()],
                  "--- README.md ---", org_card["card"], "--- end ---", ""]
    return "\n".join(lines)


def mismatches(releases):
    """Re-hash every listed file; names of those that differ from the plan."""
    return [f"{release['repo_id']}/{row['name']}" for release in releases for row in release["files"] or []
            if sha256(row["path"]) != row["expected"]]


def hf_client(token):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    return HfApi(token=token), CommitOperationAdd, CommitOperationDelete


def upload(releases, token, *, public=False, collection=None):
    """Upload exactly the listed files, each card and its assets, then the collection; nothing is sent
    if an export is not filled in or any hash differs."""
    require(token, "upload needs HF_TOKEN in the environment")
    empty = [release["repo_id"] for release in releases if release["files"] is None]
    require(not empty, "refusing upload, export not filled in: " + ", ".join(empty))
    bad = mismatches(releases)
    require(not bad, "refusing upload, hash differs from plan: " + ", ".join(bad))
    api, add, delete = hf_client(token)
    for release in releases:
        repo_id = release["repo_id"]
        api.create_repo(repo_id=repo_id, repo_type="model", private=not public, exist_ok=True)
        operations = [add(path_in_repo="README.md", path_or_fileobj=release["card"].encode())]
        operations += [add(path_in_repo=name, path_or_fileobj=data) for name, data in release["assets"].items()]
        operations += [add(path_in_repo=row["name"], path_or_fileobj=str(row["path"])) for row in release["files"]]
        api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                          commit_message="Add paired release files" if "pair" in release else "Add release files")
    if collection:
        apply_collection(releases, collection, token, public)


def upload_cards(releases, token):
    """Card-only update: each card and its figures into a repository that already exists, and the figures of
    the earlier card (SUPERSEDED) removed; no weights are read or sent, no other file is deleted, no repository
    is created and no visibility is changed."""
    require(token, "--cards needs HF_TOKEN in the environment")
    api, add, delete = hf_client(token)
    missing = [release["repo_id"] for release in releases if not api.repo_exists(release["repo_id"], repo_type="model")]
    require(not missing, "refusing card update, repository does not exist: " + ", ".join(missing))
    for release in releases:
        operations = [add(path_in_repo="README.md", path_or_fileobj=release["card"].encode())]
        operations += [add(path_in_repo=name, path_or_fileobj=data) for name, data in release["assets"].items()]
        operations += [delete(path_in_repo=name) for name in SUPERSEDED
                       if api.file_exists(release["repo_id"], name, repo_type="model")]
        api.create_commit(repo_id=release["repo_id"], repo_type="model", operations=operations,
                          commit_message="Update the model card")


def apply_collection(releases, collection, token, public=False):
    """The collection: title, description, and the models in plan order, each with its note."""
    require(token, "--collection needs HF_TOKEN in the environment")
    api, *_ = hf_client(token)
    made = api.create_collection(title=collection["title"], namespace=collection["namespace"],
                                 description=collection["description"], private=not public, exists_ok=True)
    api.update_collection_metadata(made.slug, title=collection["title"], description=collection["description"])
    for release in releases:
        made = api.add_collection_item(made.slug, item_id=release["repo_id"], item_type="model",
                                       note=release.get("note"), exists_ok=True)
    items = {item.item_id: item.item_object_id for item in made.items}
    for position, release in enumerate(releases):  # an existing collection may hold another order or older notes
        api.update_collection_item(made.slug, items[release["repo_id"]], note=release.get("note"), position=position)


def org_card(plan_path, namespace, releases):
    """The organisation card: the text kept in docs/release/, the README's banner and capsules, and the
    models of the plan with their links, in place of the line <!-- models -->."""
    plan = read_json(plan_path)
    text = (Path(plan_path).resolve().parent / plan["org_card"]).read_text()
    rows = ["| Design | AdamW | Muon |", "|:---|:---|:---|"]
    for design in family_of_releases(releases):
        cells = [f"[{design[opt]}](https://huggingface.co/{namespace}/{design[opt]})" if opt in design else ""
                 for opt in OPTIMIZERS]
        rows.append(f"| {design['title']} | {cells[0]} | {cells[1]} |")
    require("<!-- models -->" in text, f"{plan['org_card']}: the line <!-- models --> is missing")
    repo = yaml.safe_load(CITATION.read_text())["repository-code"]
    page = "docs/results/phase1.md"
    text = (text.replace("<!-- models -->", "\n".join(rows)).replace("<!-- banner -->", themed("banner", "LoopTitan", 880))
            .replace("<!-- capsules -->", capsule_row(repo, page)))
    assets = {}
    for theme, t in THEMES.items():
        assets[f"assets/banner-{theme}.svg"] = figures.banner(t)
        for i, (key, label, colour, mark, _) in enumerate(HF_PLACES):
            assets[f"assets/capsule-{key}-{theme}.svg"] = figures.capsule(key, label, colour, mark, i == 0, t,
                                                                            figures.HOSTS.get(key))
    return {"repo_id": f"{namespace}/README", "card": text, "assets": {name: svg.encode() for name, svg in assets.items()}}


def family_of_releases(releases):
    out = {}
    for release in releases:
        design = out.setdefault(release["title"], {"title": release["title"]})
        design[release["optimizer"]] = release["repo_id"].split("/", 1)[1]
    return list(out.values())


def upload_org_card(card, token):
    """The organisation card into the organisation's existing README Space; its visibility is not changed."""
    require(token, "--org-card needs HF_TOKEN in the environment")
    api, add, delete = hf_client(token)
    require(api.repo_exists(card["repo_id"], repo_type="space"), f"refusing, Space {card['repo_id']} does not exist")
    operations = [add(path_in_repo="README.md", path_or_fileobj=card["card"].encode())]
    operations += [add(path_in_repo=name, path_or_fileobj=data) for name, data in card["assets"].items()]
    api.create_commit(repo_id=card["repo_id"], repo_type="space", operations=operations,
                      commit_message="Update the organisation card")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--namespace", required=True, help="HF user or organization; repo id is <namespace>/<repo>")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print files, hashes and cards (the default)")
    mode.add_argument("--upload", action="store_true", help="upload the listed files; needs HF_TOKEN")
    mode.add_argument("--cards", action="store_true",
                      help="upload only each card and its figures to existing repos; needs no export; needs HF_TOKEN")
    mode.add_argument("--collection", action="store_true", help="apply the collection's title, description and notes")
    mode.add_argument("--org-card", action="store_true", help="upload the organisation card to the README Space")
    parser.add_argument("--public", action="store_true", help="create new repos public instead of private")
    args = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    try:
        require(token or not (args.upload or args.cards or args.collection or args.org_card), "uploads need HF_TOKEN in the environment")
        releases = load_plan(args.plan, args.namespace, cards_only=args.cards or args.collection or args.org_card)
        collection = read_json(args.plan).get("collection")
        collection = collection and {**collection, "namespace": args.namespace}
        card = org_card(args.plan, args.namespace, releases) if read_json(args.plan).get("org_card") else None
        if args.cards:
            upload_cards(releases, token)
            print(f"updated {len(releases)} cards: " + ", ".join(release["repo_id"] for release in releases))
            return 0
        if args.collection:
            apply_collection(releases, collection, token)
            print(f"applied collection {collection['title']!r} to {len(releases)} models")
            return 0
        if args.org_card:
            require(card, "the plan names no org_card")
            upload_org_card(card, token)
            print(f"updated the organisation card {card['repo_id']}")
            return 0
        if args.upload:
            upload(releases, token, public=args.public, collection=collection)
            print(f"uploaded {len(releases)} repos: " + ", ".join(release["repo_id"] for release in releases))
            return 0
        print(dry_run_report(releases, collection, card))
        return 0 if all(row["ok"] for release in releases for row in release["files"] or []) else 1
    except (ValueError, KeyError, OSError) as error:
        print(f"error: {error!r}" if isinstance(error, KeyError) else f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
