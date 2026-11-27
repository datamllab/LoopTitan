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


def test_record_tile_counts_follow_the_snapshot(monkeypatch):
    import readme_figures
    snapshot = readme_figures.threads()
    snapshot["threads"] = snapshot["threads"][:100]
    monkeypatch.setattr(readme_figures, "threads", lambda: snapshot)
    figure = readme_figures.pillars(readme_figures.LIGHT)
    assert ">100</text>" in figure and "Every path on the record. 100 issues, failures included." in figure


def test_banners_place_the_brand_lockups_as_they_are():
    import re
    import readme_figures

    def inner(name):
        text = (ROOT / "docs" / "assets" / "brand" / f"looptitan-lockup-{name}.svg").read_text()
        return re.search(r"<svg[^>]*>(.*)</svg>", text, re.S).group(1)
    light, dark, narrow = (readme_figures.banner(t) for t in (readme_figures.LIGHT, readme_figures.DARK, readme_figures.AUTO))
    assert inner("medium-indigo") in light and inner("medium-dark") in dark
    assert inner("medium-indigo") in narrow and inner("medium-dark") in narrow


def test_capsules_place_the_marks_as_they_are():
    import re
    import readme_figures as rf
    for key, label, family, mark in rf.PLACES + rf.ACTIONS:
        if mark in ("bars", "play", "tree"):
            continue
        primary = key == rf.PRIMARY
        for theme in (rf.LIGHT, rf.DARK):
            variant = "white" if (theme is rf.LIGHT) == primary else "ink"
            name = f"{mark}-{variant}" if mark.startswith("brand/") else f"brand/external/{mark}"
            text = (ROOT / "docs" / "assets" / f"{name}.svg").read_text()
            inner = re.search(r"<svg[^>]*>(.*)</svg>", text, re.S).group(1)
            assert inner in rf.capsule(key, label, family, mark, primary, theme, rf.HOSTS.get(key))  # as published, (key, name)


def test_tree_draws_every_issue_of_the_snapshot(monkeypatch):
    """The tree's counts come from the record snapshot, and it follows the snapshot when it changes."""
    import readme_figures as rf
    snapshot = rf.threads()
    figure = rf.tree_fig(rf.LIGHT)
    assert rf.TREE_TITLE in figure and rf.TREE_NEXT in figure
    assert all(part in rf.tree_fig(rf.AUTO) for part in ("Many agents, many directions,", ">one record.<"))
    for kind, name in rf.TREE_KINDS:
        count = sum(1 for th in snapshot["threads"] if th["kind"] == kind)
        assert f">{count}</text>" in figure and f">{name.upper()}</text>" in figure
    snapshot["threads"] = snapshot["threads"][:100]
    monkeypatch.setattr(rf, "threads", lambda: snapshot)
    assert f"{sum(1 for th in snapshot['threads'] if th['kind'] == 'pr')} pull requests" in rf.tree_fig(rf.LIGHT)


def test_film_capsule_gives_the_film_length():
    import readme_figures
    assert readme_figures.film_length() == "0:16"
    assert ">0:16</text>" in (ROOT / "docs" / "assets" / "capsule-film-light.svg").read_text()


def test_only_the_tree_moves_and_it_rests():
    """The tree grows, holds as the finished drawing and grows again; nothing runs during the hold (its clock waits on
    the end of the growth), reduced motion shows the still twin, and the file stays small. No other figure moves."""
    import re
    import readme_figures as rf
    for name, content in rf.figures().items():
        moves = bool(re.search(r"<animate|<set |animation:", content))
        assert moves == name.startswith("tree-"), name
        assert "indefinite" not in content and "infinite" not in content, name
        if moves:
            assert len(content.encode()) < 150_000 and "prefers-reduced-motion: reduce" in content, name
            assert 'begin="0s;fd.end"' in content and f'begin="g0.end+{rf.HOLD}ms"' in content, name
            assert content.count('<g class="st">') == content.count('<g class="mo">') >= 1, name
    assert rf.HOLD <= 10_000


def test_the_growing_tree_ends_on_the_still_one():
    """Every growth animation of a mark ends on that mark's static value: a radius on r, a path drawn to offset 0."""
    import re
    import readme_figures as rf
    tree = rf.tree_fig(rf.LIGHT)
    for r, values in re.findall(r'<circle [^>]*r="([\d.]+)"[^>]*><animate attributeName="r" values="([^"]+)"', tree):
        assert float(values.split(";")[-1]) == float(r)
    assert all(v.endswith(";0") for v in re.findall(r'attributeName="stroke-dashoffset" values="([^"]+)"', tree))
