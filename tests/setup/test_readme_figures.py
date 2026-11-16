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
