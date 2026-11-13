#!/usr/bin/env python3
"""Assemble finished V3 files and run the read-only release gate; never train.

Example: python scripts/recipe_release.py --help
Inputs are explicit durable paths. P1 stays linked through accepted V2/V1.
Use recipe_report.py --report OUTPUT/analysis.json --plot curves.png to replot.
A failed gate leaves a factual release-check.json and exits nonzero. Output must
be a fresh directory; this command never overwrites an earlier release identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from check_release import read_json, require, verify_partial_sharing_profile
from recipe_delivery import write_json
from recipe_qualification import digest

ROLES = ("reference", "candidate")
SHARING = ("a0_r0", "a0_r1", "a1_r0", "a1_r1")
RUN_FILES = ("request.json", "spec.json", "metrics.jsonl", "validation.jsonl",
             "continuation_identity.json", "outcome.json", "resolved_config.json", "model_config.json")


def existing_file(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    return path


def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_run(directory, fingerprint):
    directory = Path(directory).resolve()
    for name in RUN_FILES:
        existing_file(directory / name)
    request, spec = read_json(directory / "request.json"), read_json(directory / "spec.json")
    require(request.get("manifest_fingerprint") == fingerprint, f"{directory}: manifest differs")
    require(request.get("spec_sha256") == digest(directory / "spec.json"), f"{directory}: request/spec hash differs")
    return request, spec


def inherited_v2(directory, source_path):
    directory, source_path = Path(directory).resolve(), existing_file(source_path)
    release, check = existing_file(directory / "release.json"), existing_file(directory / "release-check.json")
    source, report = read_json(source_path), read_json(check)
    require(source.get("dirty") is False and re.fullmatch(r"[0-9a-f]{40}", source.get("commit", ""))
            and source.get("files") and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in source["files"].values()),
            "original V2 source receipt must contain a clean exact commit and file hashes")
    checker = source.get("checker", {})
    require(checker.get("path") and digest(existing_file(checker["path"])) == checker.get("sha256"),
            "original V2 checker SHA differs")
    require(read_json(release).get("stage") == report.get("stage") == "v2"
            and report.get("status") == "passed" and report.get("scope") == "v2_full_acceptance",
            "V2 requires its original full accepted release-check")
    return {"directory": str(directory), "release_sha256": digest(release), "release_check_sha256": digest(check),
            "source_v2": str(source_path), "source_v2_sha256": digest(source_path)}


def assemble_anchor(output, pair_path, arms, heldout, manifest, fingerprint):
    from recipe_report import analyze_run
    pair_path = existing_file(pair_path)
    pair = read_json(pair_path)
    require(pair.get("anchor") == "P2" and 0 < pair.get("candidate_count", 0) <= 32,
            "frozen P2 finite screen is missing")
    comparison = pair["comparison"]
    require(all(comparison.get(key, {}).get("manifest_fingerprint") == fingerprint for key in ("data", "evaluation")),
            "frozen P2 data/evaluation identities differ")
    curves = {}
    # Check all input identities before writing. Never select a new pair from CE.
    for role in ROLES:
        _, spec = load_run(arms[role], fingerprint)
        require(spec == comparison["arms"][role], f"{role}: actual spec differs from frozen P2 pair")
        for split in ("validation", "test"):
            existing_file(heldout[role][split])
    for role in ROLES:
        curve = analyze_run(arms[role], comparison["costs"][role])["curve"]
        path = output / f"p2-{role}-curve.jsonl"
        with path.open("x") as stream:
            for point in curve:
                stream.write(json.dumps(point, sort_keys=True, allow_nan=False) + "\n")
        curves[role] = str(path)
    return {"format": "moeloop-scientific-anchor", "version": 1, "anchor": "P2",
            "manifest": str(Path(manifest).resolve()), "manifest_fingerprint": fingerprint,
            "comparison": str(pair_path), "arms": {role: str(Path(arms[role]).resolve()) for role in ROLES},
            "heldout": {role: {split: str(Path(path).resolve()) for split, path in heldout[role].items()} for role in ROLES},
            "curves": curves, "result": {"classification": "observed_single_seed_comparison",
                "limitations": ["One predeclared seed42; no scaling-law inference.",
                    "P2 E88 is a declared public-stack adaptation; quality differences are descriptive.",
                    "Analytical budget matching does not establish a winning recipe."]}}


def sharing_comparison(arms, fingerprint):
    from moeloop.costs import estimate_cost
    from moeloop.recipe_spec import RecipeSpec
    require(set(arms) == set(SHARING), "complete C four-arm panel is required")
    rows = {}
    for name in SHARING:
        request, raw = load_run(arms[name], fingerprint)
        spec = RecipeSpec.from_dict(raw)
        require(request["seq_len"] == spec.run["seq_len"] == 2048, f"{name}: C context differs")
        rows[name] = {"spec": spec.to_dict(), "cost": estimate_cost(spec, seq_len=request["seq_len"])}
    # Normalize tuples exactly as the verifier's JSON comparison does.
    return json.loads(json.dumps({"format": "moeloop-attention-router-grid", "version": 1,
        "manifest_fingerprint": fingerprint, "seed": 42, "arms": rows}, allow_nan=False))


def performance_spec(arms, sharing, *, sharing_arm="a1_r0", p2_start=5, sharing_start=1):
    require(sharing_arm in sharing and type(p2_start) is int and p2_start >= 1
            and type(sharing_start) is int and sharing_start >= 1, "invalid performance window")
    verify_partial_sharing_profile(sharing[sharing_arm])
    runs = {f"p2_{role}": (arms[role], p2_start) for role in ROLES}
    runs["sharing"] = (sharing[sharing_arm], sharing_start)
    return {"format": "moeloop-recipe-performance", "version": 1,
            "windows": {name: {"run": str(Path(run).resolve()), "start_step": start,
                "warmup_updates": 5, "measurement_updates": 20} for name, (run, start) in runs.items()}}


def assemble(args):
    from recipe_report import build_report
    from check_release import verify_v3_release
    inherited = inherited_v2(args.v2, args.source_v2)
    v2 = read_json(Path(inherited["directory"]) / "release.json")
    manifest = existing_file(args.manifest or v2["manifest"])
    fingerprint = v2["manifest_fingerprint"]
    arms = {role: getattr(args, f"p2_{role}").resolve() for role in ROLES}
    heldout = {role: {split: str(existing_file(getattr(args, f"p2_{role}_{split}")))
        for split in ("validation", "test")} for role in ROLES}
    sharing = {name: getattr(args, f"sharing_{name}").resolve() for name in SHARING}
    for run in list(arms.values()) + list(sharing.values()):
        load_run(run, fingerprint)
    pair = existing_file(args.p2_pair)
    downstream, journey = existing_file(args.lm_eval), existing_file(args.package)
    comparison = sharing_comparison(sharing, fingerprint)
    if args.sharing_comparison:
        require(read_json(existing_file(args.sharing_comparison)) == comparison,
                "supplied C comparison differs from actual specs/costs")
    publication = existing_file(args.publication_source) if args.publication_source else None
    output = new_output(args.output)
    try:
        anchor = assemble_anchor(output, pair, arms, heldout, manifest, fingerprint)
        write_json(output / "p2.json", anchor)
        write_json(output / "sharing-comparison.json", comparison)
        write_json(output / "performance.json", performance_spec(arms, sharing, sharing_arm=args.performance_sharing_arm,
            p2_start=args.p2_performance_start, sharing_start=args.sharing_performance_start))
        index = {"format": "moeloop-recipe-qualification", "version": 1, "stage": "v3",
            "manifest": str(manifest), "manifest_fingerprint": fingerprint, "inherited_v2": inherited,
            "p2": str(output / "p2.json"), "sharing_grid": {"arms": {name: str(run) for name, run in sharing.items()},
                "comparison": str(output / "sharing-comparison.json")}, "performance": str(output / "performance.json"),
            "lm_eval": str(downstream), "package": str(journey), "analysis": str(output / "analysis.json")}
        if publication:
            # Publication identity is separate from immutable native training source.
            index["publication_source"] = {"path": str(publication), "sha256": digest(publication)}
        write_json(output / "release.json", index)
        write_json(output / "analysis.json", build_report(index))
        result = verify_v3_release(output)
    except Exception as error:
        write_json(output / "release-check.json", {"status": "failed", "stage": "v3", "error": str(error)})
        raise
    write_json(output / "release-check.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "v2", "source-v2", "p2-pair", "p2-reference", "p2-candidate",
                 "p2-reference-validation", "p2-reference-test", "p2-candidate-validation", "p2-candidate-test",
                 "sharing-a0-r0", "sharing-a0-r1", "sharing-a1-r0", "sharing-a1-r1", "lm-eval", "package"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="defaults to accepted V2 manifest; gate enforces inherited identity")
    parser.add_argument("--sharing-comparison", type=Path, help="optional frozen C JSON; must match actual specs/costs")
    parser.add_argument("--publication-source", type=Path, help="optional publication source receipt, separate from training source")
    parser.add_argument("--performance-sharing-arm", choices=SHARING, default="a1_r0")
    parser.add_argument("--p2-performance-start", type=int, default=5,
                        help="first warmup update after actual native resume (default 5 after stop4; measured10-29)")
    parser.add_argument("--sharing-performance-start", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        result = assemble(args)
    except Exception as error:
        print(json.dumps({"status": "failed", "stage": "v3", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
