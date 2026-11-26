"""Packaging contract: pyproject metadata, the two workflows, and (if uv exists) a real wheel.

Torch-free. The wheel test builds a copy of the sources in a temp dir, so the checkout stays clean.
"""

import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
REPO_GUARD = "github.repository == 'datamllab/LoopTitan'"
PINNED_TWINE = re.compile(r"twine==\d+\.\d+\.\d+ twine check")


def project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def workflow(name):
    doc = yaml.safe_load((WORKFLOWS / name).read_text())
    doc["triggers"] = doc.pop(True, doc.pop("on", None))  # YAML 1.1 reads the key `on` as True
    return doc


def run_text(job):
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_metadata():
    p = project()
    assert p["name"] == "moeloop-lab"
    assert p["version"] == "1.0.0rc1"
    assert p["description"]
    assert p["readme"] == "README.md" and (ROOT / "README.md").is_file()
    assert p["license"] == "MIT" and p["license-files"] == ["LICENSE"]
    assert p["requires-python"] == ">=3.11"
    assert p["dependencies"] == []  # the base install stays torch-free
    assert p["urls"] == {"Homepage": "https://datamllab.github.io/LoopTitan/",
                         "Documentation": "https://datamllab.github.io/LoopTitan/docs/",
                         "Repository": "https://github.com/datamllab/LoopTitan",
                         "Paper": "https://arxiv.org/abs/2611.34821",
                         "Models": "https://huggingface.co/NaturalScale"}
    assert "Programming Language :: Python :: 3.11" in p["classifiers"]
    assert not [c for c in p["classifiers"] if c.startswith("License ::")]  # the SPDX field is the license
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert cff["version"] == p["version"] and cff["license"] == p["license"]


def test_package_discovery_is_exactly_the_two_import_packages():
    tool = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"]["find"]
    assert tool == {"where": ["src"], "include": ["moeloop", "moeloop_lab"]}
    assert sorted(p.parent.name for p in (ROOT / "src").glob("*/__init__.py")) == ["moeloop", "moeloop_lab"]


def test_workflows_parse():
    for name in ("test.yml", "release.yml"):
        doc = workflow(name)
        assert doc["jobs"], name
    assert "package" in workflow("test.yml")["jobs"]


def test_test_workflow_package_job():
    text = run_text(workflow("test.yml")["jobs"]["package"])
    assert "uv build" in text and PINNED_TWINE.search(text) and "dist/*.whl" in text
    assert "importlib.util.find_spec(\"torch\") is None" in text  # imports must work without torch


def test_release_triggers_only_on_tags():
    doc = workflow("release.yml")
    assert doc["triggers"] == {"push": {"tags": ["v*"]}}  # no workflow_dispatch, schedule or branch push


def test_release_runs_only_in_the_main_repository():
    for name, job in workflow("release.yml")["jobs"].items():
        assert job["if"] == REPO_GUARD, name


def test_release_permissions_and_publish_job():
    doc = workflow("release.yml")
    jobs = doc["jobs"]
    assert doc["permissions"] == {"contents": "read"}
    assert jobs["publish"]["permissions"] == {"id-token": "write"}
    assert jobs["publish"]["environment"]["name"] == "pypi"
    assert [name for name, job in jobs.items() if "id-token" in job.get("permissions", {})] == ["publish"]
    uses = [s["uses"] for s in jobs["publish"]["steps"] if "uses" in s]
    assert any(re.fullmatch(r"pypa/gh-action-pypi-publish@v\d+\.\d+\.\d+", u) for u in uses)
    assert "password" not in str(jobs["publish"]) and "secrets." not in (WORKFLOWS / "release.yml").read_text()
    # build checks before anything is published; the GitHub release is a draft made after PyPI.
    assert PINNED_TWINE.search(run_text(jobs["build"])) and "uv build" in run_text(jobs["build"])
    assert jobs["publish"]["needs"] == "build" and jobs["github-release"]["needs"] == "publish"
    assert jobs["github-release"]["permissions"] == {"contents": "write"}
    assert "--draft" in run_text(jobs["github-release"])


@pytest.mark.parametrize("tag, matches", [("v1.0.0rc1", True), ("v1.0.0", False), ("v1.0.0-rc1", False)])
def test_release_tag_must_equal_the_pyproject_version(tmp_path, tag, matches):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1.0.0rc1"\n')
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}", GITHUB_REF_NAME=tag)
    [step] = [s for s in workflow("release.yml")["jobs"]["build"]["steps"] if s.get("name") == "Tag matches pyproject version"]
    done = subprocess.run(["bash", "-e", "-c", step["run"]], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert (done.returncode == 0) is matches, done.stderr


@pytest.mark.parametrize("tag, prerelease", [
    ("v1.0.0rc1", True), ("v1.0.0a2", True), ("v2.1b3", True), ("v1.0.0.dev1", True),
    ("v1.0.0", False), ("v1.2.3", False), ("v1.0.0.post1", False)])
def test_github_release_flags_only_pre_releases(tmp_path, tag, prerelease):
    (tmp_path / "gh").write_text('#!/bin/sh\necho "$@" > "$GH_LOG"\n')
    (tmp_path / "gh").chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}", GH_LOG=str(tmp_path / "log"),
               GITHUB_REF_NAME=tag, GITHUB_REPOSITORY="datamllab/LoopTitan")
    script = run_text(workflow("release.yml")["jobs"]["github-release"])
    done = subprocess.run(["bash", "-e", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    args = (tmp_path / "log").read_text().split()
    assert args[:2] == ["release", "create"] and args[2] == tag and "--draft" in args
    assert ("--prerelease" in args) is prerelease


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_built_wheel_contents(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(ROOT / name, src / name)
    shutil.copytree(ROOT / "src", src / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    out = tmp_path / "dist"
    done = subprocess.run(["uv", "build", "--out-dir", str(out), str(src)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    [wheel] = out.glob("*.whl")
    assert wheel.name == "moeloop_lab-1.0.0rc1-py3-none-any.whl"
    assert len(list(out.glob("*.tar.gz"))) == 1
    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
        meta = Parser().parsestr(z.read("moeloop_lab-1.0.0rc1.dist-info/METADATA").decode())
        license_text = z.read("moeloop_lab-1.0.0rc1.dist-info/licenses/LICENSE")  # export.py reads it from here
    dist_info = {n for n in names if n.startswith("moeloop_lab-1.0.0rc1.dist-info/")}
    assert license_text == (ROOT / "LICENSE").read_bytes() and license_text.startswith(b"MIT License\n")
    # The code is exactly src/*.py: no tests, docs, data, recipes or configs.
    assert names - dist_info == {str(p.relative_to(ROOT / "src")) for p in (ROOT / "src").rglob("*.py")}
    assert meta["Name"] == "moeloop-lab" and meta["Version"] == "1.0.0rc1"
    assert meta["Requires-Python"] == ">=3.11" and meta["License-Expression"] == "MIT"
    assert meta["Description-Content-Type"].startswith("text/markdown")
    assert {u.split(", ")[0] for u in meta.get_all("Project-URL")} == {"Homepage", "Documentation", "Repository", "Paper", "Models"}
    requires = meta.get_all("Requires-Dist")
    assert all("extra ==" in r for r in requires)  # nothing is required by default
    assert any(r.startswith("torchtitan==0.3.0") for r in requires)
    assert not [r for r in requires if "@" in r or "://" in r]  # PyPI rejects direct-URL dependencies
