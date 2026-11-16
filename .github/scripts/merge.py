"""Merge a PR through the gate: python .github/scripts/merge.py <PR> [--human]

Without --human (an agent merging a `Merge: agent` task) it merges only if the issue
the PR closes says `Merge: agent`, no changed path is protected in the base branch's
guard.toml, and the newest `Verdict:` comment passes the current head; otherwise it
refuses (exit 2). With --human the human decided, so it merges without these checks.
Either way it squash-merges the current head; branch protection still runs the checks.
"""
import base64
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard import any_match, closing_refs, parse_sections  # noqa: E402

PASS = re.compile(r"Verdict: pass @ ([0-9a-fA-F]{7,40})")


def merge_mode(issue_body: str | None) -> str:
    return "agent" if parse_sections(issue_body).get("Merge", "").strip().lower() == "agent" else "human"


def check(pr: dict, files: list[dict], comments: list[str], issue: dict | None, cfg: dict) -> list[str]:
    reasons, refs = [], closing_refs(pr.get("body") or "", pr.get("repo"))
    if len(refs) != 1:
        reasons.append(f"the PR body must close exactly one issue with `Closes #N` (found {len(refs)})")
    elif issue is None:
        reasons.append(f"cannot read issue #{refs[0]}")
    elif merge_mode(issue.get("body")) != "agent":
        reasons.append(f"the Merge field of issue #{refs[0]} is not agent")

    for f in files:
        for p in [f["filename"]] + ([f["previous_filename"]] if f.get("previous_filename") else []):
            if any_match(p, cfg["protected"]):
                reasons.append(f"changes the protected path {p}")

    firsts = [(c or "").strip().split("\n")[0].strip() for c in comments]
    verdict = next((line for line in reversed(firsts) if line.startswith("Verdict:")), None)
    m = PASS.fullmatch(verdict or "")
    if verdict is None:
        reasons.append("no review verdict on the PR (a comment whose first line is `Verdict: pass @ <head SHA>`)")
    elif m is None or not pr["head"].lower().startswith(m.group(1).lower()):
        reasons.append(f"the newest verdict is not a pass for the current head {pr['head'][:7]}: {verdict}")
    return reasons


def gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def gh_api(path: str, *flags: str):
    p = gh("api", path, *flags)
    return json.loads(p.stdout) if p.returncode == 0 else None


def main(argv: list[str]) -> int:
    human, args = "--human" in argv, [a for a in argv if a != "--human"]
    if len(args) != 1 or not args[0].isdigit():
        print("usage: python .github/scripts/merge.py <PR> [--human] (PR number)")
        return 1
    number = args[0]
    pr = gh_api(f"repos/{{owner}}/{{repo}}/pulls/{number}")
    if pr is None:
        print(f"cannot read PR #{number}; not merged")
        return 1
    head = pr["head"]["sha"]
    if not human:
        repo, ref = pr["base"]["repo"]["full_name"], pr["base"]["ref"]
        pages = gh_api(f"repos/{repo}/pulls/{number}/files", "--paginate", "--slurp")
        notes = gh_api(f"repos/{repo}/issues/{number}/comments", "--paginate", "--slurp")
        toml = gh_api(f"repos/{repo}/contents/.github/guard.toml?ref={quote(ref, safe='')}")
        if pages is None or notes is None or toml is None:
            print(f"cannot read the files or comments of PR #{number}, or guard.toml on {ref}; not merged")
            return 1
        refs = closing_refs(pr.get("body") or "", repo)
        issue = gh_api(f"repos/{repo}/issues/{refs[0]}") if len(refs) == 1 else None
        reasons = check({"body": pr.get("body") or "", "head": head, "repo": repo},
                        [f for page in pages for f in page], [c.get("body") or "" for page in notes for c in page],
                        issue, tomllib.loads(base64.b64decode(toml["content"]).decode()))
        if reasons:
            print(f"PR #{number} not merged:")
            for r in reasons:
                print(f"- {r}")
            print("waiting for a human to merge")
            return 2
    p = gh("pr", "merge", number, "--squash", "--match-head-commit", head,
           "--subject", f"{pr['title']} (#{number})", "--body", "")
    if p.returncode != 0:
        print(f"merge of PR #{number} failed: {(p.stderr or p.stdout).strip()}")
        return 1
    print(f"merged PR #{number} on the human's instruction (squash, head {head[:7]})" if human
          else f"merged PR #{number} (squash, head {head[:7]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
