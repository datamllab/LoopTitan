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
decimals, or the whole run is refused. The card, a banner (banner.svg) and a budgets
figure (budgets.svg), both drawn with the README figures' functions, are generated; every number on
them comes from the results file, the recipe and the P2 track, and the reference of the BPB
difference and of the budgets is the ordinary MoE under the same optimizer. The collection title
names the release on each card.
A model whose `export_dir` is still empty is rendered in the dry-run and refused by `--upload`.
With `--upload`, after the repos exist, one collection in the namespace lists them in plan order.

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
ROLES = {"looped": "candidate", "control": "reference"}
SPLITS = ("validation", "test")
NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
GENERATED = ("README.md", "banner.svg", "budgets.svg")  # written by this script for a single model
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

def load_plan(path, namespace):
    """Return one release per model: repo id, rendered card and the checked file listing."""
    require(NAMESPACE.fullmatch(namespace), f"invalid namespace {namespace!r}")
    path = Path(path).resolve()
    base, plan = path.parent, read_json(path)
    if "models" in plan:
        return single_releases(plan, base, namespace)
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


def single_releases(plan, base, namespace):
    """One release per model of the plan, in plan order, bound to its run of the results file."""
    results = read_json(base / plan["results"])
    page = (base / plan["results"]).resolve().relative_to(ROOT).with_suffix(".md").as_posix()
    runs = {run["config"]: (row, optimizer, run) for group in results["groups"] for row in group["rows"]
            for optimizer, run in row["runs"].items()}
    dataset, cff = read_json(base / plan["dataset"]), yaml.safe_load((ROOT / "CITATION.cff").read_text())
    releases = []
    for model in plan["models"]:
        require(model["config"] in runs, f"{model['repo']}: {model['config']} is not a run of {plan['results']}")
        row, optimizer, run = runs[model["config"]]
        files = None
        if model["export_dir"]:
            require(not set(GENERATED) & set(model["files"]), f"{model['repo']}: {', '.join(GENERATED)} are generated")
            files = checked_files(model, base, None)
            for split in SPLITS:
                name, expected = model["evaluation"][split], run[f"{split}_bpb"]
                require(name in model["files"], f"{model['repo']}: evaluation file {name!r} is not listed in files")
                bpb = read_json(base / model["export_dir"] / name)["bpb"]
                require(f"{bpb:.6f}" == f"{expected:.6f}", f"{model['repo']}: {split} BPB {bpb:.6f} of {name} "
                        f"differs from {expected:.6f} in {plan['results']}")
        reference, recipe = REFERENCE.format(optimizer), read_json(ROOT / model["config"])
        card = model_card(f"{namespace}/{model['repo']}", model, row, optimizer, run, runs[reference][2],
                          plan["collection"]["title"], page, dataset, cff, recipe)
        releases.append({"repo_id": f"{namespace}/{model['repo']}", "label": model["config"], "card": card,
                         "assets": {"banner.svg": banner(model["config"], recipe["model"]).encode(),
                                    "budgets.svg": budgets(model["config"], reference, run, OPTIMIZERS[optimizer]).encode()},
                         "files": files})
    return releases


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


def model_card(repo_id, model, row, optimizer, run, reference, release, page, dataset, cff, recipe):
    """Markdown card of one model: every number comes from its row of the results file, its recipe or the track."""
    repo, name, config, train = cff["repository-code"], OPTIMIZERS[optimizer], recipe["model"], recipe["run"]
    tolerance = load_tracks()["P2"].budget["tolerance"]
    outside = {key: value for key, value in run["budgets"].items() if abs(value) > tolerance}
    authors = " and ".join("{" + a["name"] + "}" if "name" in a else f"{a['family-names']}, {a['given-names']}"
                           for a in cff["authors"])
    tags = ["looptitan", "mixture-of-experts", "looped-moe" if config["num_visits"] > 1 else "moe", "custom_code", optimizer]
    lines = ["---", f"license: {cff['license'].lower()}", "library_name: transformers", "pipeline_tag: text-generation",
             "tags:", *[f"- {json.dumps(tag)}" for tag in tags], "datasets:", f"- {json.dumps(dataset['id'])}",
             "model-index:", f"- name: {json.dumps(model['repo'])}", "  results:",
             "  - task:", "      type: text-generation", "    dataset:", f"      name: {json.dumps(dataset['id'])}",
             f"      type: {json.dumps(dataset['id'])}", "    metrics:"]
    for split in ("test", "validation"):
        lines += [f"    - name: {split.capitalize()} BPB", "      type: bpb", f"      value: {run[f'{split}_bpb']:.6f}"]
    lines += ["---", "", f"![{banner_text(config)}](banner.svg)", "", f"# {model['title']}", "", f"{name} · {release}", "",
              f"{row['design'][0].upper()}{row['design'][1:]}. A base language model trained from scratch, "
              "not instruction tuned.", "",
              "## Results", "", f"| Test BPB | Validation BPB | Δ test vs ordinary MoE ({name}) |", "|---:|---:|---:|",
              f"| {run['test_bpb']:.6f} | {run['validation_bpb']:.6f} | "
              f"{signed(1000 * (run['test_bpb'] - reference['test_bpb']), 2)} mBPB |", "",
              "Bits per byte on the complete held-out test and validation splits after the last update. "
              f"The ordinary MoE is [`{reference['config']}`]({repo}/blob/main/{reference['config']}); "
              "1 mBPB is 0.001 BPB.", "", "## Budgets", "",
              f"Outside matched budgets: more than {tolerance:.0%} from the ordinary MoE ({name}) in " +
              " and ".join(f"{BUDGET_NAMES[key]} ({signed(100 * value, 1)}%)" for key, value in outside.items()) + "."
              if outside else f"Matched budgets: all three within {tolerance:.0%} of the ordinary MoE ({name}).", "",
              f"![{budgets_text(run, name, tolerance)}](budgets.svg)", "",
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
              f"[results page]({repo}/blob/main/{page}).", "",
              "## Use", "", "```python", "import torch", "from transformers import AutoModelForCausalLM, AutoTokenizer", "",
              f"repo = {json.dumps(repo_id)}", "tokenizer = AutoTokenizer.from_pretrained(repo)",
              "model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()",
              "ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer.encode(\"The model\", add_special_tokens=False)])",
              "out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=32, do_sample=False)",
              "print(tokenizer.decode(out[0], skip_special_tokens=True))", "```", "",
              "The model runs Python code from this repository: read it first, or pin `revision=`.", "",
              "## Links", "", f"- Code: [{cff['title']}]({repo})",
              f"- Configuration: [`{model['config']}`]({repo}/blob/main/{model['config']})",
              f"- Documentation: [docs]({repo}/tree/main/docs)", f"- Results: [results page]({repo}/blob/main/{page})",
              f"- Run record: [#{run['issue']}]({repo}/issues/{run['issue']})", "",
              "## Citation", "", "```bibtex", f"@software{{{cff['title'].lower()},", f"  title = {{{cff['title']}}},",
              f"  author = {{{authors}}},", f"  version = {{{cff['version']}}},", f"  url = {{{repo}}}", "}", "```", "",
              "## Licence", "", f"[{cff['license']}]({repo}/blob/main/LICENSE).", ""]
    return "\n".join(lines)


def banner_text(config):
    stored = config["prefix_cells"] + config["num_cells"] + config["suffix_cells"]
    run = stored + (config["num_visits"] - 1) * config["num_cells"]
    return f"{stored} layers stored, {run} run" if run > stored else f"{stored} layers"


def banner(config_path, config):
    """The README's model glyph of the configuration, light variant, enlarged on a card: one bar per stored
    layer, its area the layer's expert parameters; looped layers in indigo under an arc with the visit count."""
    w, h, scale = 880, 280, 2.2
    body, total = figures.glyph(config_path.removeprefix("recipes/").rsplit("-", 1)[0], 0, 0, figures.LIGHT, k=2)
    return figures.svg(w, h, banner_text(config), banner_text(config), figures.LIGHT,
                       figures.card(0, 0, w, h, figures.LIGHT) +
                       f'<g transform="translate({figures.num((w - scale * total) / 2)} 145) scale({figures.num(scale)})">{body}</g>' +
                       figures.text(w / 2, h - 20, banner_text(config), 14, figures.LIGHT["mid"], 400, "middle"))


def budgets_text(run, name, tolerance):
    return (f"Budgets against the ordinary MoE ({name}): " +
            ", ".join(f"{BUDGET_NAMES[key]} {100 * run['budgets'][key]:+.2f}%" for key in METRICS) +
            f"; tolerance ±{tolerance:.0%}")


def budgets(config_path, reference_path, run, name):
    """The three budgets in the README's band drawing, light variant, as the P2 track counts them; a budget
    outside its tolerance is drawn as the homepage's out-of-band mark, an ink square past the band's end."""
    track, t = load_tracks()["P2"], figures.LIGHT
    result = compare_recipes(spec(reference_path), spec(config_path), seq_len=track.seq_len,
                             tolerance=track.budget["tolerance"], **track.budget["accounting"])
    entries = result["matched"] | result["measured-different"]
    w, h, x, span = 880, 248, 28, 800
    inside = all(abs(entry["relative_difference"]) <= entry["relative_tolerance"] for entry in entries.values())
    out = [figures.card(0, 0, w, h, t), figures.text(x, 42, f"Budgets against the ordinary MoE ({name})", 14, t["ink"], 600),
           figures.text(x + span, 42, f"{'within' if inside else 'outside'} ±{figures.pct(track.budget['tolerance'])}",
                        13, t["acc"] if inside else t["ink"], 600, "end", mono=True)]
    for i, (key, label) in enumerate(figures.BUDGETS):
        entry, y = entries[key], 92 + i * 54
        row = figures.budget_row(t, entry, key, label, x, y, span)
        if abs(entry["relative_difference"]) > entry["relative_tolerance"]:
            row[-1] = figures.rect(x + span / 2 + math.copysign(span / 2 + 16, entry["relative_difference"]) - 5,
                                   y + 13, 10, 10, 2, t["ink"])
        out += row
    return figures.svg(w, h, "Budgets", budgets_text(run, name, track.budget["tolerance"]), t, "".join(out))


# --- dry-run and upload -------------------------------------------------------------------

def dry_run_report(releases, collection=None):
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
        lines += [f"== collection {collection['namespace']}: {collection['title']}", f"  {collection['description']}",
                  *[f"  {i}. {release['repo_id']}" for i, release in enumerate(releases, 1)], ""]
    return "\n".join(lines)


def mismatches(releases):
    """Re-hash every listed file; names of those that differ from the plan."""
    return [f"{release['repo_id']}/{row['name']}" for release in releases for row in release["files"] or []
            if sha256(row["path"]) != row["expected"]]


def hf_client(token):
    from huggingface_hub import CommitOperationAdd, HfApi
    return HfApi(token=token), CommitOperationAdd


def upload(releases, token, *, public=False, collection=None):
    """Upload exactly the listed files, each card and its assets, then the collection; nothing is sent
    if an export is not filled in or any hash differs."""
    require(token, "upload needs HF_TOKEN in the environment")
    empty = [release["repo_id"] for release in releases if release["files"] is None]
    require(not empty, "refusing upload, export not filled in: " + ", ".join(empty))
    bad = mismatches(releases)
    require(not bad, "refusing upload, hash differs from plan: " + ", ".join(bad))
    api, add = hf_client(token)
    for release in releases:
        repo_id = release["repo_id"]
        api.create_repo(repo_id=repo_id, repo_type="model", private=not public, exist_ok=True)
        operations = [add(path_in_repo="README.md", path_or_fileobj=release["card"].encode())]
        operations += [add(path_in_repo=name, path_or_fileobj=data) for name, data in release["assets"].items()]
        operations += [add(path_in_repo=row["name"], path_or_fileobj=str(row["path"])) for row in release["files"]]
        api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                          commit_message="Add paired release files" if "pair" in release else "Add release files")
    if collection:
        made = api.create_collection(title=collection["title"], namespace=collection["namespace"],
                                     description=collection["description"], private=not public, exists_ok=True)
        for release in releases:
            made = api.add_collection_item(made.slug, item_id=release["repo_id"], item_type="model", exists_ok=True)
        items = {item.item_id: item.item_object_id for item in made.items}
        for position, release in enumerate(releases):  # an existing collection may hold another order
            api.update_collection_item(made.slug, items[release["repo_id"]], position=position)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--namespace", required=True, help="HF user or organization; repo id is <namespace>/<repo>")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print files, hashes and cards (the default)")
    mode.add_argument("--upload", action="store_true", help="upload the listed files; needs HF_TOKEN")
    parser.add_argument("--public", action="store_true", help="create new repos public instead of private")
    args = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    try:
        require(token or not args.upload, "--upload needs HF_TOKEN in the environment")
        releases = load_plan(args.plan, args.namespace)
        collection = read_json(args.plan).get("collection")
        collection = collection and {**collection, "namespace": args.namespace}
        if args.upload:
            upload(releases, token, public=args.public, collection=collection)
            print(f"uploaded {len(releases)} repos: " + ", ".join(release["repo_id"] for release in releases))
            return 0
        print(dry_run_report(releases, collection))
        return 0 if all(row["ok"] for release in releases for row in release["files"] or []) else 1
    except (ValueError, KeyError, OSError) as error:
        print(f"error: {error!r}" if isinstance(error, KeyError) else f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
