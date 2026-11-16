#!/usr/bin/env python3
"""Print a Markdown results table for the MLflow runs of some issues, checked against a spec.

    python scripts/results.py --issue N [N ...] [--metric M ...] [--spec FILE] [--config scripts/run.toml]
    gh issue view P --json body -q .body | python scripts/results.py --issue N --spec -

Reads the runs whose cvsm.issue param is one of the Ns, deleted ones included, from the experiment run.py
writes to (MLFLOW_EXPERIMENT_NAME, else `experiment` in run.toml) in the store it writes to
(MLFLOW_TRACKING_URI, else <main clone>/mlflow.db). Runs are grouped by command with the seed
tokens removed. Per group: n (runs that exited 0, which the statistics use), n_failed (runs
that ended any other way), and the mean and sample std of each metric's latest value.
Provenance differences (commit, env lock, data manifest), dirty trees, successful runs without
metrics, incomplete runs (still RUNNING) and deleted runs are flagged; whether runs are
comparable is for the reader to decide.

The section Spec check checks every selected run, running ones included, against the study spec
in FILE (a study issue's body; - reads stdin), using the cfg.* params run.py logs from
config.json. It reads the lines of ### Arms (`<arm>: <key>=<value> ...`), ### Controlled
(`<key>=<value>  # <reason>` or `<key> in <v1>,<v2>,...  # <reason>`) and ### Adaptable
(`<key>  # <rule>`), and lists as facts: problems of the spec itself (unreadable lines,
Controlled lines without a reason, a missing Arms or Controlled section, keys in several
sections); runs where a Controlled line does not hold or whose config lacks its key (not recorded);
runs matching no arm or several; arms missing values of an `in` line among their running or
successful runs that meet every Controlled line; keys in no section whose values differ across
the runs that recorded a config; arms with no run; and the values each Adaptable key took per
arm. Values compare as JSON when they parse as JSON (numbers by value), else as strings. With a
spec, the table gains each group's arms.
"""
import argparse
import json
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

from run import MlflowClient, experiment_name, load_config, num, split_seed, store_setup, tracking_uri
from mlflow.entities import ViewType


def fetch(client, exp_id, issue):
    runs, token = [], None
    while True:
        page = client.search_runs([exp_id], f"params.`cvsm.issue` = '{issue}'",
                                  run_view_type=ViewType.ALL, max_results=1000, page_token=token)
        runs += page
        if not (token := page.token):
            return runs


def fmt(x):
    return f"{x:.4g}"


def short(values, n):
    return ", ".join(dict.fromkeys(v[:n] if v else "none" for v in values))


def ids(runs):
    return ", ".join(r["id"] for r in runs)


def count(n, noun="run"):
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def norm(value):
    """A value as it compares: canonical JSON when it parses as JSON (1 and 1.0 alike), else itself."""
    try:
        v = json.loads(value, parse_float=lambda x: int(f) if (f := float(x)).is_integer() else f)
    except ValueError:
        return value
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True)


def top_level_items(text):
    """The comma-separated items of text, splitting only outside brackets and braces (JSON lists)."""
    items, depth, start = [], 0, 0
    for i, c in enumerate(text):
        depth += (c in "[{") - (c in "]}")
        if c == "," and not depth:
            items.append(text[start:i])
            start = i + 1
    return [item.strip() for item in items + [text[start:]]]


def read_spec(text):
    """Arms {arm: {key: value}}, Controlled [(line, key, values)], Adaptable [key], the declared
    keys and the spec's own problems, from a study body; `  # ...` ends a line's item."""
    spec = {"arms": {}, "controlled": [], "adaptable": [], "problems": []}
    section, seen = None, set()
    for line in text.splitlines():
        if line.startswith("### "):
            section = line[4:].strip()
            seen.add(section)
            continue
        if section not in ("Arms", "Controlled", "Adaptable") or line.strip() in ("", "_No response_"):
            continue
        item, *note = re.split(r"\s+#", line.strip(), maxsplit=1)
        if section == "Arms":
            arm, colon, rest = item.partition(":")
            pairs = [p.partition("=") for p in re.split(r"\s+(?=[^\s=]+=)", rest.strip())]
            if colon and arm.strip() and all(k and eq for k, eq, _ in pairs):
                spec["arms"][arm.strip()] = {k: v for k, _, v in pairs}
                continue
        elif section == "Controlled":
            m = re.fullmatch(r"([^\s=]+)\s+in\s+(.+)|([^\s=]+)\s*=(.*)", item)
            if m:
                values = top_level_items(m[2]) if m[1] else [m[4].strip()]
                spec["controlled"].append((item, m[1] or m[3], values))
                if not (note and note[0].strip()):
                    spec["problems"].append(f"Controlled line without a reason: {line.strip()}")
                continue
        elif re.fullmatch(r"[^\s=]+", item):
            spec["adaptable"].append(item)
            continue
        spec["problems"].append(f"Unreadable spec line: {section}: {line.strip()}")
    spec["problems"] += [f"No {s} section in the spec" for s in ("Arms", "Controlled") if s not in seen]
    keys = {"Arms": {k for kv in spec["arms"].values() for k in kv},
            "Controlled": {key for _, key, _ in spec["controlled"]}, "Adaptable": set(spec["adaptable"])}
    spec["declared"] = set().union(*keys.values())
    for key in sorted(spec["declared"]):
        if len(where := [s for s, ks in keys.items() if key in ks]) > 1:
            spec["problems"].append(f"`{key}` is in more than one section: {', '.join(where)}")
    return spec


def tally(runs, key):
    """The values runs took for key, equal ones merged: [`value (n runs)`]."""
    seen = {}
    for r in runs:
        v = r["cfg"].get(key)
        seen.setdefault(None if v is None else norm(v), ["not recorded" if v is None else v, 0])[1] += 1
    return [f"{v} ({count(n)})" for v, n in seen.values()]


def check(spec, runs):
    """The Spec check lines: deviations from the spec as facts, then the Adaptable values."""
    if not spec:
        return ["- No spec given (--spec)"]
    arms = spec["arms"]
    out = [f"- {problem}" for problem in spec["problems"]]
    broken = set()  # ids of runs where some Controlled line does not hold or its key is unrecorded
    for item, key, values in spec["controlled"]:
        allowed = {norm(v) for v in values}
        if missing := [r for r in runs if key not in r["cfg"]]:
            out.append(f"- `{item}`: {count(len(missing))} with no recorded `{key}`: {ids(missing)}")
        if wrong := [r for r in runs if key in r["cfg"] and norm(r["cfg"][key]) not in allowed]:
            out.append(f"- `{item}` does not hold for {count(len(wrong))}: "
                       + ", ".join(f"{r['id']} ({r['cfg'][key]})" for r in wrong))
        broken |= {r["id"] for r in missing + wrong}
    if none := [r for r in runs if not r["arms"]]:
        out.append(f"- {count(len(none))} with no matching arm: {ids(none)}")
    if several := [r for r in runs if len(r["arms"]) > 1]:
        out.append(f"- {count(len(several))} in more than one arm: "
                   + ", ".join(f"{r['id']} ({', '.join(r['arms'])})" for r in several))
    member = {arm: [r for r in runs if arm in r["arms"]] for arm in arms}
    for item, key, values in spec["controlled"]:
        for arm, rs in member.items():
            took = {norm(r["cfg"][key]) for r in rs if r["id"] not in broken and not r["deleted"]
                    and (r["ok"] or r["status"] == "RUNNING")}  # failed or broken runs cover nothing
            if len(values) > 1 and rs and (lack := [v for v in values if norm(v) not in took]):
                out.append(f"- Arm {arm} is missing values of `{item}`: {', '.join(lack)}")
    recorded = [r for r in runs if r["cfg"]]
    for key in sorted({k for r in recorded for k in r["cfg"]} - spec["declared"]):
        if len(values := tally(recorded, key)) > 1:
            out.append(f"- Undeclared difference `{key}`: {', '.join(values)}")
    if empty := [arm for arm, rs in member.items() if not rs]:
        out.append(f"- Arms with no run: {', '.join(empty)}")
    if not out:
        out.append("- No deviations")
    for key in spec["adaptable"]:
        per = [f"{arm}: {', '.join(tally(rs, key))}" for arm, rs in member.items() if rs]
        out.append(f"- Adaptable `{key}`: {'; '.join(per) or 'no run in an arm'}")
    return out


def main():
    ap = argparse.ArgumentParser(prog="results.py")
    ap.add_argument("--issue", type=int, nargs="+", required=True)
    ap.add_argument("--metric", nargs="+", action="extend", help="metrics to show (default all)")
    ap.add_argument("--spec", help="study issue body to check the runs against; - reads stdin")
    ap.add_argument("--config", default="scripts/run.toml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    spec = a.spec and read_spec(sys.stdin.read() if a.spec == "-" else Path(a.spec).read_text())

    uri = tracking_uri(cfg)
    with store_setup(uri):
        client = MlflowClient(uri)
        exp = client.get_experiment_by_name(experiment_name(cfg))
    issues = list(dict.fromkeys(a.issue))
    found = sorted((r for i in issues for r in fetch(client, exp.experiment_id, i)) if exp else [],
                   key=lambda r: r.info.start_time)
    # One dict per run: the cvsm.* params without the prefix, plus status, deletion, metrics and config.
    runs = [{k[len("cvsm."):]: v for k, v in r.data.params.items() if k.startswith("cvsm.")}
            | {"id": r.info.run_id, "status": r.info.status, "metrics": r.data.metrics,
               "deleted": r.info.lifecycle_stage == "deleted",
               "cfg": {k[4:]: v for k, v in r.data.params.items() if k.startswith("cfg.")}} for r in found]
    groups = {}
    for r in runs:
        r["ok"] = r.get("exit_code") == "0"
        r["arms"] = [arm for arm, kv in spec["arms"].items() if all(
            k in r["cfg"] and norm(r["cfg"][k]) == norm(v) for k, v in kv.items())] if spec else []
        cmd = split_seed(shlex.split(r.get("command", "")), cfg["seed_key"])[1]
        groups.setdefault(shlex.join(cmd), []).append(r)
    metrics = a.metric or sorted({m for r in runs for m, v in r["metrics"].items() if num(v)})

    head = ["command", *(["arm"] if spec else []), "n", "n_failed",
            *(f"{m} {s}" for m in metrics for s in ("mean", "std")), "commit", "env lock", "data manifest"]
    out = [f"Results of {', '.join(f'#{i}' for i in issues)}: {count(len(runs))} in {count(len(groups), 'group')},"
           " grouped by command with the seed removed.", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for cmd, rs in groups.items():
        ok = [r for r in rs if r["ok"]]
        failed = [r for r in rs if not r["ok"] and r["status"] != "RUNNING"]
        row = ["`" + cmd.replace("|", "\\|") + "`"]
        if spec:
            row.append(", ".join(dict.fromkeys(arm for r in rs for arm in r["arms"] or ["none"])))
        row += [str(len(ok)), str(len(failed))]
        for m in metrics:
            v = [r["metrics"][m] for r in ok if num(r["metrics"].get(m))]
            row += [fmt(statistics.mean(v)) if v else "—", fmt(statistics.stdev(v)) if len(v) > 1 else "—"]
        row += [short((r.get("commit") for r in rs), 7), short((r.get("env_lock_sha") for r in rs), 8),
                short((r.get("data_manifest_sha") for r in rs), 8)]
        out.append("| " + " | ".join(row) + " |")

    flags = []
    commits = list(dict.fromkeys(r["commit"] for r in runs if r.get("commit")))  # by run start
    if len(commits) > 1:
        first, last = commits[0], commits[-1]
        p = subprocess.run(["git", "diff", "--stat", first, last], capture_output=True, text=True,
                           errors="replace")
        flags += [f"- Commit differs: {short(commits, 7)}; `git diff --stat {first[:7]} {last[:7]}`:",
                  "", "```", (p.stdout or p.stderr).rstrip(), "```"]
    for key, name in (("env_lock_sha", "Env lock"), ("data_manifest_sha", "Data manifest")):
        if len({r[key] for r in runs if key in r}) > 1:  # runs killed before launch have none
            flags.append(f"- {name} differs: {short((r[key] for r in runs if key in r), 8)}")
    missing = [r for r in runs if r.get("env_lock_missing")]
    if missing:
        files = sorted({f for r in missing for f in r["env_lock_missing"].split(",")})
        flags.append(f"- {count(len(missing))} with missing env lock files: {', '.join(files)}")
    if dirty := [r for r in runs if r.get("dirty") == "true"]:
        flags.append(f"- {count(len(dirty))} with a dirty worktree (see the artifacts diff.patch and untracked.txt):"
                     f" {ids(dirty)}")
    if nometrics := [r for r in runs if r["ok"] and not r["metrics"]]:
        flags.append(f"- {count(len(nometrics))} with exit code 0 and no metrics: {ids(nometrics)}")
    if incomplete := [r for r in runs if r["status"] == "RUNNING"]:
        flags.append(f"- {count(len(incomplete))} not finished (still running or killed): {ids(incomplete)}")
    if deleted := [r for r in runs if r["deleted"]]:
        flags.append(f"- {count(len(deleted))} deleted in MLflow but still counted in the table: {ids(deleted)}")
    print("\n".join(out + ["", "Flags:", *(flags or ["- none"]), "", "Spec check:", *check(spec, runs)]))


if __name__ == "__main__":
    main()
