#!/usr/bin/env python3
"""Plan, render and (only on request) upload paired Hugging Face releases; never train.

Example: python scripts/hf_release.py plan.json --namespace my-org            # dry-run
         HF_TOKEN=... python scripts/hf_release.py plan.json --namespace my-org --upload

A release plan lists pairs. Each pair names one evidence file (anchor P1: docs/evidence/p1.json,
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
import os
from pathlib import Path, PurePosixPath
import re
import sys

from check_release import require
from recipe_delivery import read_json, sha256

ROLES = {"looped": "candidate", "control": "reference"}
SPLITS = ("validation", "test")
NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


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
            releases.append({"pair": pair["name"], "role": name, "repo_id": repos[name], "card": card, "files": files})
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
    lines = ["---", "license: apache-2.0", "library_name: transformers", "pipeline_tag: text-generation",
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


# --- dry-run and upload -------------------------------------------------------------------

def dry_run_report(releases):
    lines = []
    for release in releases:
        lines += [f"== {release['repo_id']} ({release['pair']}, {release['role']})", "files to upload:"]
        lines += [f"  {row['sha256']}  {row['size']:>14,}  {row['name']}" + ("" if row["ok"] else
                  f"  MISMATCH (plan {row['expected']})") for row in release["files"]]
        lines += ["  generated README.md follows", "--- README.md ---", release["card"], "--- end ---", ""]
    return "\n".join(lines)


def mismatches(releases):
    """Re-hash every listed file; names of those that differ from the plan."""
    return [f"{release['repo_id']}/{row['name']}" for release in releases for row in release["files"]
            if sha256(row["path"]) != row["expected"]]


def hf_client(token):
    from huggingface_hub import CommitOperationAdd, HfApi
    return HfApi(token=token), CommitOperationAdd


def upload(releases, token, *, public=False):
    """Upload exactly the listed files and each card; nothing is sent if any hash differs."""
    require(token, "upload needs HF_TOKEN in the environment")
    bad = mismatches(releases)
    require(not bad, "refusing upload, hash differs from plan: " + ", ".join(bad))
    api, add = hf_client(token)
    for release in releases:
        repo_id = release["repo_id"]
        api.create_repo(repo_id=repo_id, repo_type="model", private=not public, exist_ok=True)
        operations = [add(path_in_repo="README.md", path_or_fileobj=release["card"].encode())]
        operations += [add(path_in_repo=row["name"], path_or_fileobj=str(row["path"])) for row in release["files"]]
        api.create_commit(repo_id=repo_id, repo_type="model", operations=operations, commit_message="Add paired release files")


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
        if args.upload:
            upload(releases, token, public=args.public)
            print(f"uploaded {len(releases)} repos: " + ", ".join(release["repo_id"] for release in releases))
            return 0
        print(dry_run_report(releases))
        return 0 if all(row["ok"] for release in releases for row in release["files"]) else 1
    except (ValueError, KeyError, OSError) as error:
        print(f"error: {error!r}" if isinstance(error, KeyError) else f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
