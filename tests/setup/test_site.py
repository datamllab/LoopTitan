"""The site under docs/site (docs/site/README.md): what its build reads from the repository is still there."""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "docs" / "site"
HOME = SITE / "home"


def nav_sources():
    # nav.mjs lists each page as [source path in the repository, slug, sidebar label]
    return re.findall(r"\['([^']+)', '[^']*', '[^']*'\]", (SITE / "nav.mjs").read_text())


def test_every_page_in_the_navigation_exists():
    sources = nav_sources()
    assert "docs/architecture.md" in sources
    assert [s for s in sources if not (ROOT / s).is_file()] == []


def test_every_document_is_in_the_navigation():
    # docs/assets/ holds notes beside the asset files and docs/release/ the texts uploaded to Hugging Face: not pages
    inputs = (SITE, ROOT / "docs" / "assets", ROOT / "docs" / "release")
    documents = {str(p.relative_to(ROOT)) for p in (ROOT / "docs").rglob("*.md") if not any(d in p.parents for d in inputs)}
    assert documents - set(nav_sources()) == set()


def test_the_homepage_links_nowhere_outside_the_base():
    # Served under a base path (/LoopTitan/ on GitHub Pages), so every link and asset of the homepage is relative.
    html = (HOME / "index.html").read_text()
    assert re.findall(r"""(?:href|src)\s*=\s*["']/(?!/)[^"']*""", html) == []
    for path in [*HOME.glob("*.css"), *HOME.glob("*.js")]:
        assert re.findall(r"""url\(\s*["']?/(?!/)""", path.read_text()) == [], path.name
        assert re.findall(r"""["'`]/(?:docs|brand|fonts|data)\b""", path.read_text()) == [], path.name


def test_every_font_carries_its_licence():
    fonts = sorted(HOME.glob("fonts/*.woff2"))
    assert fonts
    for font in fonts:
        licence = HOME / "fonts" / f"OFL-{font.stem.split('-')[0]}.txt"
        assert "SIL Open Font License, Version 1.1" in licence.read_text(), font.name


def test_the_typed_expert_learning_rate_matches_the_recipes():
    typed = re.search(r"expertLr: '×([\d.]+)'", (HOME / "content.js").read_text()).group(1)
    for arm in ("m7-elr", "m4-elr"):
        recipe = json.loads((ROOT / "recipes" / "phase1" / f"{arm}-adamw.json").read_text())
        assert f"{recipe['optimizer']['expert_lr_multiplier']:.2f}" == typed, arm


def test_the_site_is_published_only_from_main_while_the_repository_is_public():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "pages.yml").read_text())
    on = wf[True]  # YAML 1.1 reads the key `on` as True
    assert on["push"] == {"branches": ["main"]}
    public = "github.event_name == 'push' && github.event.repository.private == false"
    assert public in wf["jobs"]["deploy"]["if"] and public in wf["jobs"]["build"]["env"]["PUBLISH"]
    assert set(wf["permissions"].values()) == {"read"}
