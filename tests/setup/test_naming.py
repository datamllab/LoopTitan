"""Reader-facing pages call the reference design the standard MoE, as the paper does.

Frozen records (recipes/, configs/, eval/, data/) keep their text and are not read here.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGES = ["README.md", "docs/*.md", "docs/results/*.md", "docs/release/*.md",
         "docs/site/home/index.html", "docs/site/home/content.js", "docs/site/home/app.js"]
OLD = re.compile(r"\bordinary[\s-]+MoE\b", re.I)


def test_pages_say_standard_moe():
    files = sorted({path for pattern in PAGES for path in ROOT.glob(pattern)})
    assert files
    hits = [str(path.relative_to(ROOT)) for path in files if OLD.search(path.read_text())]  # also across a line break
    assert hits == []
