"""The README figures are current: scripts/readme_figures.py draws them from the repository's data."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_figures_are_current():
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "readme_figures.py"), "--check"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_readme_uses_every_figure():
    readme = (ROOT / "README.md").read_text()
    import readme_figures
    for name in readme_figures.figures():
        assert f"docs/assets/{name}" in readme, name


def test_budget_counts_follow_the_compare_result(monkeypatch):
    import readme_figures
    real = readme_figures.budgets

    def one_outside(reference, candidate):
        matched = real(reference, candidate)
        matched["logical_kv_bytes"] = matched["logical_kv_bytes"] | {"relative_difference": 0.2}
        return matched
    monkeypatch.setattr(readme_figures, "budgets", one_outside)
    assert "2 of 3 budgets within ±5%" in readme_figures.pillars(readme_figures.LIGHT)
    assert "2 of 3 within ±5%" in readme_figures.compare_fig(readme_figures.LIGHT)


def test_banners_place_the_brand_lockups_as_they_are():
    import re
    import readme_figures

    def inner(name):
        text = (ROOT / "docs" / "assets" / "brand" / f"looptitan-lockup-{name}.svg").read_text()
        return re.search(r"<svg[^>]*>(.*)</svg>", text, re.S).group(1)
    light, dark, narrow = (readme_figures.banner(t) for t in (readme_figures.LIGHT, readme_figures.DARK, readme_figures.AUTO))
    assert inner("medium-indigo") in light and inner("medium-dark") in dark
    assert inner("medium-indigo") in narrow and inner("medium-dark") in narrow
