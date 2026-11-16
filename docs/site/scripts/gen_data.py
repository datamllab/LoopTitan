#!/usr/bin/env python3
"""Generate the homepage's data files (home/data/*.js) from the repository checkout. Nothing is typed in here.

    python scripts/gen_data.py [--refresh-record] [--out DIR]       (build.sh runs it before every build)

Sources, all read from REPO (default ../.., the checkout this folder sits in) at its checked-out commit:
  configs.js  scripts/lineage.py (document(): parents and changed fields), the recipe files (model, source),
              moeloop.comparison.compare_recipes(<family m0>, <config>, seq_len=4096) (budgets)
  compare.js  the command shown on the page, run through compare_recipes (its matched and differences sections)
  checks.js   docs/evidence/qwen3-cpu.json, moeloop.testing.CHECKS, pytest --collect-only (test count)
  data.js     docs/results/phase1.json (test BPB), the frozen recipes (schedule, batch, seed, learning rates),
              moeloop.costs.estimate_cost (stored and active parameters of the ordinary MoE)
  refs.js     the reference list of README.md (titles, years), the source URLs of the recipe files
  record.js   data/record-snapshot.json, a committed snapshot of the GitHub issue record; --refresh-record
              rewrites it with gh (read-only; the token and GH_REPO from the environment)
The output keeps exactly the shape app.js reads. It needs the repository's model environment (torch and the data,
hf, train and report extras, as CI's model job): the test count is what pytest collects there.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("REPO", SITE / "../..")).resolve()
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import lineage  # noqa: E402
from moeloop import testing  # noqa: E402
from moeloop.comparison import compare_recipes  # noqa: E402
from moeloop.config import ModelConfig  # noqa: E402
from moeloop.costs import estimate_cost  # noqa: E402
from moeloop.recipe_spec import RecipeSpec  # noqa: E402

SNAPSHOT = SITE / "data" / "record-snapshot.json"
BUDGETS = ("logical_kv_bytes", "training_flops_per_token", "unique_non_embedding_parameters")
# The page's command in the Compare section (shown verbatim; its output is computed here).
COMPARE_COMMAND = "python scripts/compare_recipes.py compare recipes/phase1/m3-adamw.json recipes/phase1/m4-adamw.json"
# Arms of the one-change chain and the published designs as app.js names them, by frozen configuration.
CHAIN = {"M0": "phase1/m0", "G": "shape/g", "U": "shape/u", "R": "shape/r", "M4": "phase1/m4",
         "M7": "phase1/m7", "M5": "phase1/m5", "M3": "phase1/m3"}
EXCHANGE = {"M0-Loop": "methods/m0-loop", "M0-2E": "phase1/m0-2e"}
PUBLISHED = {"V3": "methods/smelt-v3", "V4": "methods/smelt-v4", "LatentLoop-Residual": "methods/latentloop-residual",
             "Meta": "methods/meta", "Foil": "methods/foil", "Tying": "methods/expert-tying", "Huginn": "methods/huginn",
             "Sandwich": "methods/sandwich", "LOOM": "methods/loom", "Ouro": "methods/ouro", "Relaxed": "methods/relaxed"}
QUANTITIES = ("logits", "loss", "gradient", "updated_parameter")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def recipe(path: str) -> dict:
    return json.loads((REPO / path).read_text())


def spec(path: str) -> RecipeSpec:
    return RecipeSpec.from_dict(recipe(path))


def published_url(path: str) -> str | None:
    """A recipe's published source: its own source URL or, through base_recipe, its base's (results_table.py)."""
    entry = recipe(path)["source"]
    while "url" not in entry and (REPO / str(entry.get("base_recipe", ""))).is_file() and entry.get("base_recipe"):
        entry = recipe(entry["base_recipe"])["source"]
    return entry.get("url")


def compare(reference: str, candidate: str, seq_len: int | None = None) -> dict:
    ref, cand = spec(reference), spec(candidate)
    seq_len = seq_len or min([2048, ref.model.max_seq_len, cand.model.max_seq_len])  # the CLI's default
    return compare_recipes(ref, cand, tolerance=0.05, seq_len=seq_len, batch_size=1, bytes_per_element=2,
                           activation_checkpointing="none", attention_pairs_per_sequence=None)


def plain(value):
    return json.loads(json.dumps(value))


# ---------------------------------------------------------------- configs.js
def reference_of(node: str) -> str:
    fam = node.split("/")[0]
    return f"recipes/{fam}/m0-adamw.json" if fam in ("p05", "p1") else "recipes/phase1/m0-adamw.json"


def config(node: str, entry: dict) -> dict:
    path = entry["recipes"][0]
    m = ModelConfig.from_dict(recipe(path)["model"])
    regions = {}
    for region in ("prefix", "core", "suffix"):
        r = m.region_config(region)
        regions[region] = {"experts": r.num_experts, "expertParamsM": round(3 * r.num_experts * r.hidden_dim * m.dim / 1e6, 2),
                           "residual": r.residual_scale}
    ref = reference_of(node)
    result = compare(ref, path, seq_len=4096)
    budgets = {}
    for key in BUDGETS:  # a budget inside its tolerance is in "matched", one outside in "measured-different"
        b = result["matched"].get(key) or result["measured-different"][key]
        budgets[key] = {"candidate": b["candidate"], "reference": b["reference"], "rel": b["relative_difference"],
                        "tol": b["relative_tolerance"], "raw": {k: v for k, v in b.items() if k not in ("candidate", "reference")},
                        "within": abs(b["relative_difference"]) <= b["relative_tolerance"]}
    return {"parent": entry["parent"], "recipes": entry["recipes"], "changes": plain(entry["changes"]),
            "pre": m.prefix_cells, "core": m.num_cells, "suf": m.suffix_cells, "visits": m.num_visits, "order": m.order,
            "dim": m.dim, "topK": m.top_k, "regions": regions,
            "visitResidualScales": list(m.visit_residual_scales) or None,
            "attentionPerVisit": m.attention_per_visit, "routerPerVisit": m.router_per_visit, "normPerVisit": m.norm_per_visit,
            "loraAttentionRank": m.lora_attention_rank, "inputInjection": m.input_injection, "initialState": m.initial_state,
            "stateUpdate": m.state_update, "exitDistribution": m.exit_distribution, "source": published_url(path),
            "budgets": budgets, "reference": ref}


def configs_js(commit: str) -> str:
    doc = lineage.document()
    out = {node: config(node, entry) for node, entry in doc["nodes"].items()}
    head = (f"/* Generated by docs/site/scripts/gen_data.py from the repository at commit {commit}:\n"
            "   scripts/lineage.py (parents and changed fields), the recipe files (model, source),\n"
            "   compare_recipes(<reference m0>, <config>, seq_len=4096) (budgets).\n"
            "   expertParamsM = 3 x experts x hidden x dim / 1e6 (routed expert weights per layer). Do not edit by hand. */\n")
    return head + f'window.REPO_COMMIT = "{commit}";\nwindow.CONFIGS = {json.dumps(out, indent=1)};\n'


# ---------------------------------------------------------------- compare.js
def compare_js(commit: str) -> str:
    args = COMPARE_COMMAND.split()
    result = compare(args[-2], args[-1])
    out = {"command": COMPARE_COMMAND, "reference": result["arms"]["reference"]["name"],
           "candidate": result["arms"]["candidate"]["name"], "matched": result["matched"],
           "differences": result["recipe_differences"]}
    head = (f"/* Excerpt of the real output of:\n   {COMPARE_COMMAND}\n"
            f"   computed by docs/site/scripts/gen_data.py from the repository at commit {commit}. */\n")
    return head + f"window.COMPARE = {json.dumps(plain(out), indent=1, sort_keys=True)};\n"


# ---------------------------------------------------------------- checks.js
def test_count() -> int:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=REPO,
                         capture_output=True, text=True, env=env)
    found = re.search(r"(\d+) tests? collected", run.stdout)
    if not found:
        raise SystemExit("pytest --collect-only gave no count:\n" + run.stdout[-2000:] + run.stderr[-2000:])
    print(f"homepage data: pytest --collect-only: {run.stdout.strip().splitlines()[-1]}")
    return int(found.group(1))


def checks_js(commit: str) -> str:
    ev = json.loads((REPO / "docs/evidence/qwen3-cpu.json").read_text())
    cases = {}
    for name, case in ev["cases"].items():
        errors = {q: case["max_absolute_errors"][q] for q in QUANTITIES}
        errors["cached_logits"] = case["cache"]["logits_max_absolute_error"]
        cases[name] = {"errors": errors, "updates": case["updates"]}
    tests = test_count()
    out = {"reference": ev["reference"], "transformers": ev["transformers_version"], "source_url": ev["source_url"],
           "device": ev["device"], "dtype": ev["dtype"], "optimizer": ev["optimizer"]["name"],
           "tolerances": ev["tolerances"], "cases": cases, "tests": tests, "checks": list(testing.CHECKS)}
    head = (f"/* Generated by docs/site/scripts/gen_data.py from docs/evidence/qwen3-cpu.json, moeloop.testing.CHECKS and\n"
            f"   pytest --collect-only ({tests:,} tests) at commit {commit}. */\n")
    return head + f"window.CHECKS = {json.dumps(out, indent=1)};\n"


# ---------------------------------------------------------------- data.js
def data_js(commit: str) -> str:
    results = json.loads((REPO / "docs/results/phase1.json").read_text())
    runs = {}
    for group in results["groups"]:
        for row in group["rows"]:
            node = row["runs"]["adamw"]["config"].removeprefix("recipes/").removesuffix("-adamw.json")
            runs[node] = row["runs"]
    bpb = lambda node: {o: runs[node][o]["test_bpb"] for o in ("adamw", "muon")}  # noqa: E731
    m0 = {o: recipe(f"recipes/phase1/m0-{o}.json") for o in ("adamw", "muon")}
    run = m0["adamw"]["run"]
    cost = estimate_cost(spec("recipes/phase1/m0-adamw.json"), seq_len=run["seq_len"])
    par = cost["parameters"]
    comp = par["by_component"]
    experts = comp["routed_experts"]
    m = spec("recipes/phase1/m0-adamw.json").model
    active = par["unique_non_embedding"] - experts + experts * m.top_k / m.num_experts
    out = {
        "_about": f"Generated by docs/site/scripts/gen_data.py from docs/results/phase1.json and the frozen recipes at commit {commit}. "
                  "test_bpb: complete held-out test split after the last update.",
        "fig1_decomposition": {"test_bpb": {arm: bpb(node) for arm, node in CHAIN.items()}},
        "fig3_instrument": {"training": {
            "tokens_per_update": run["global_batch_positions"], "horizon_5B_updates": run["schedule_total_steps"],
            "warmup_updates": run["schedule"]["warmup_steps"], "decay_fraction": run["schedule"]["decay_ratio"],
            "seq_len": run["seq_len"], "seed": run["seed"],
            "lr_adamw": m0["adamw"]["optimizer"]["base_lr"], "lr_muon": m0["muon"]["optimizer"]["base_lr"]}},
        "fig6_exchange": {arm: bpb(node) for arm, node in EXCHANGE.items()},
        "fig8_recipes": {"recipes": {arm: bpb(node) for arm, node in PUBLISHED.items()}},
        "reruns": {node: {k: v for k, v in r["adamw"]["rerun_at_other_lr"].items() if k != "note"}
                   for node, r in runs.items() if "rerun_at_other_lr" in r["adamw"]},
        "scale": {"stored_non_embedding": par["unique_non_embedding"], "active_non_embedding": round(active),
                  "definition": "ordinary MoE phase1/m0: unique non-embedding parameters, and those used per token (routed experts x top_k / experts)"},
    }
    head = f"/* Generated by docs/site/scripts/gen_data.py from docs/results/phase1.json and the frozen recipes at commit {commit}. */\n"
    return head + f"window.PAPER = {json.dumps(out, indent=1)};\n"


# ---------------------------------------------------------------- refs.js
def refs_js(commit: str) -> str:
    readme = (REPO / "README.md").read_text()
    urls = {}
    for path in sorted((REPO / "recipes").rglob("*.json")):
        url = json.loads(path.read_text()).get("source", {}).get("url")
        if url and (aid := re.search(r"(\d{4}\.\d{5})", url)):
            urls.setdefault(aid.group(1), url)
    out = []
    for m in re.finditer(r"^\d+\. (.+?)\. (?:.*?)(\d{4})\. \[arXiv:(\d{4}\.\d{5})\]\((\S+)\)$", readme, re.M):
        title, year, aid, abs_url = m.groups()
        out.append({"id": aid, "title": title, "year": year, "url": urls.get(aid, abs_url)})
    head = (f"/* Generated by docs/site/scripts/gen_data.py at commit {commit}: titles and years from the reference list of\n"
            "   README.md, links from the source field of the frozen recipes. In the README's order. */\n")
    return head + f"window.REFS = {json.dumps(out, indent=1)};\n"


# ---------------------------------------------------------------- record.js
ORIG = re.compile(r"^> 原记录时间：(\d{4}-\d{2}-\d{2})")
ORIG_PR = re.compile(r"^> 原为合并请求（pull request）#\d+，创建于 (\d{4}-\d{2}-\d{2})[^，]*，于 (\d{4}-\d{2}-\d{2})")
KINDS = ("study", "plan", "run", "code", "pr")


def gh(path: str) -> list:
    run = subprocess.run(["gh", "api", "--paginate", "--slurp", path], check=True, capture_output=True, text=True)
    return [item for page in json.loads(run.stdout) for item in page]


def refresh_record(repo: str) -> dict:
    """The public issue record: one row per study, plan, run and code thread and per merged pull request."""
    threads = []
    issues = gh(f"repos/{repo}/issues?state=all&per_page=100")
    merged = {p["number"]: p["merged_at"] for p in gh(f"repos/{repo}/pulls?state=closed&per_page=100") if p["merged_at"]}
    for it in sorted(issues, key=lambda x: x["number"]):
        labels = [lab["name"] for lab in it["labels"]]
        body = it.get("body") or ""
        if it.get("pull_request"):
            if it["number"] not in merged:
                continue
            kind, opened, closed = "pr", it["created_at"][:10], merged[it["number"]][:10]
        else:
            kind = next((k for k in KINDS if k in labels), None)
            if kind is None:
                continue
            opened, closed = it["created_at"][:10], (it["closed_at"] or "")[:10] or None
            if (pr := ORIG_PR.match(body)):  # a merged pull request recorded as an issue when the record moved
                opened, closed = pr.group(1), pr.group(2)
            elif (orig := ORIG.match(body)):  # moved issue: its original dates
                opened = orig.group(1)
                if closed:
                    last = None
                    if it["comments"]:
                        comments = gh(f"repos/{repo}/issues/{it['number']}/comments?per_page=100")
                        stamps = [m.group(1) for c in comments if (m := ORIG.match(c.get("body") or ""))]
                        last = stamps[-1] if stamps else None
                    closed = last or opened
        parent = None
        if it.get("parent_issue_url"):
            parent = int(it["parent_issue_url"].rsplit("/", 1)[1])
        threads.append({"n": it["number"], "kind": kind, "opened": opened, "closed": closed, "comments": it["comments"],
                        "parent": parent, "title": it["title"] if kind in ("study", "plan") else None})
    return {"as_of": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"), "repo": f"https://github.com/{repo}",
            "threads": threads}


def record_js(snapshot: dict) -> str:
    out = {"_about": f"The public issue record of {snapshot['repo'].removeprefix('https://')} on {snapshot['as_of']}: one row per "
                     "thread (number, kind, original open/close dates, number of comments, parent study or plan). kind pr = record "
                     "of a merged pull request. Study and plan titles are in Chinese.",
           "as_of": snapshot["as_of"], "repo": snapshot["repo"], "threads": snapshot["threads"]}
    head = ("/* Generated by docs/site/scripts/gen_data.py from docs/site/data/record-snapshot.json "
            f"(the GitHub issue record on {snapshot['as_of']}). */\n")
    return head + f"window.RECORD = {json.dumps(out, indent=0, ensure_ascii=False)};\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=SITE / "home/data")
    parser.add_argument("--refresh-record", action="store_true", help="rewrite data/record-snapshot.json with gh")
    args = parser.parse_args()
    commit = git("rev-parse", "--short=7", "HEAD")
    if args.refresh_record:
        snap = refresh_record(os.environ.get("GH_REPO", "datamllab/LoopTitan"))
        SNAPSHOT.write_text(json.dumps(snap, indent=1, ensure_ascii=False) + "\n")
    files = {"configs.js": configs_js(commit), "compare.js": compare_js(commit), "checks.js": checks_js(commit),
             "data.js": data_js(commit), "refs.js": refs_js(commit), "record.js": record_js(json.loads(SNAPSHOT.read_text()))}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out / name).write_text(text)
    print(f"homepage data: {len(files)} files from {REPO} at {commit} -> {args.out}")


if __name__ == "__main__":
    main()
