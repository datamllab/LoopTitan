"""The README stays true for this tree: links, named paths, the Phase 1 chain table, no CJK text."""
import json
import re
from pathlib import Path

import hf_release
import yaml

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
FENCE = re.compile(r"^(```|~~~).*?^\1", re.M | re.S)
CODE_SPAN = re.compile(r"`[^`\n]*`")
TARGET = re.compile(r"\]\(([^)\s]+)\)|(?:href|src|srcset)=\"([^\"]+)\"")
CJK = re.compile("[぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯가-힯]")


def prose(text):
    """Markdown outside fenced code blocks and code spans."""
    return CODE_SPAN.sub("", FENCE.sub("", text))


def anchors(path):
    """GitHub heading anchors of a Markdown file, with -1, -2 for repeats."""
    seen, out = {}, set()
    for line in FENCE.sub("", path.read_text()).splitlines():
        m = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        slug = re.sub(r"[^\w\- ]", "", m.group(1).lower()).replace(" ", "-")
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        out.add(slug if n == 0 else f"{slug}-{n}")
    return out


def test_relative_links_and_anchors_resolve():
    targets = [a or b for a, b in TARGET.findall(prose(README.read_text()))]
    local = [t for t in targets if not re.match(r"[a-z]+:", t)]
    assert local, "no relative links found"
    for target in local:
        path, _, anchor = target.partition("#")
        file = (ROOT / path).resolve() if path else README
        assert file.exists(), target
        if anchor:
            assert anchor in anchors(file), target


def test_named_recipes_and_scripts_exist():
    names = set(re.findall(r"(?<![\w/.-])((?:recipes|scripts)/[\w./-]*\w)", README.read_text()))
    assert names
    for name in names:
        assert (ROOT / name).exists(), name


def test_chain_table_equals_phase1_data():
    data = json.loads((ROOT / "docs" / "results" / "phase1.json").read_text())
    chain = data["groups"][0]["rows"]
    by_config = {row["runs"]["adamw"]["config"]: row for row in chain}
    base = {opt: chain[0]["runs"][opt]["test_bpb"] for opt in ("adamw", "muon")}
    section = README.read_text().split("### Phase 1", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if re.match(r"\|\s*\d+\s*\|", line)]
    assert len(rows) >= 2
    for line in rows:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        config = re.search(r"\]\(([^)]+)\)", cells[1]).group(1)
        row = by_config[config]
        for opt, value, delta in (("adamw", cells[2], cells[3]), ("muon", cells[4], cells[5])):
            bpb = row["runs"][opt]["test_bpb"]
            assert value == f"{bpb:.4f}", (config, opt)
            expected = "" if row is chain[0] else f"{1000 * (bpb - base[opt]):+.1f}"
            assert delta.replace("−", "-") == expected, (config, opt)


def test_no_cjk_characters():
    assert not CJK.search(README.read_text())


def test_bibtex_entry_equals_the_model_card_entry():
    """README, model cards and CITATION.cff give one citation: the card's entry is rendered from the cff."""
    block = re.compile(r"^```bibtex\n(.*?)^```", re.M | re.S)
    plan = json.loads((ROOT / "docs" / "release" / "phase1-models.example.json").read_text())
    plan["results"] = str(ROOT / "docs" / "results" / "phase1.json")
    plan["dataset"] = str(ROOT / "docs" / "evidence" / "p1-dclm-source.json")
    plan["models"] = plan["models"][:1]
    card = hf_release.single_releases(plan, ROOT / "docs" / "release", "acme")[0]["card"]
    entry = block.findall(README.read_text())
    assert len(entry) == 1
    assert entry == block.findall(card)
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert f"@software{{{cff['title'].lower()}," in entry[0] and f"title = {{{cff['title']}}}" in entry[0]
    assert f"author = {{{{{cff['authors'][0]['name']}}}}}" in entry[0]
    assert f"version = {{{cff['version']}}}" in entry[0] and f"url = {{{cff['url']}}}" in entry[0]
