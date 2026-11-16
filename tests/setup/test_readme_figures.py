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
