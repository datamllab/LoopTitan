#!/usr/bin/env python3
"""Draw the README figures from the repository's data: python scripts/readme_figures.py [--check]

Writes SVG files to docs/assets/: each figure in a light and a dark variant (README.md switches them with
<picture> and prefers-color-scheme), and the large figures also in a narrow variant for phones, which
follows the reader's colour scheme by itself. Every value drawn comes from the repository: the frozen
recipes (recipes/), the lineage of scripts/lineage.py, the cost model (moeloop.comparison), the CPU
record docs/evidence/qwen3-cpu.json, the Phase 1 results docs/results/phase1.json, pyproject.toml and
eval/bench/tracks.json; the layout (positions, sizes, axis ranges) is chosen in this script. The destination
capsules place the marks of docs/assets/brand/ and docs/assets/brand/external/ unchanged. It also
writes README.md's BibTeX block (between the citation markers) from CITATION.cff with bibtex(), the entry
the model cards carry too. The mark
comes from the lockup files in docs/assets/brand/, placed as they are. --check writes nothing and exits 1
if a file differs from what the data gives; tests/setup/test_readme_figures.py runs it. Text uses system
fonts, because GitHub shows these files as images.

The drawing follows the homepage: state shown by the thing itself (a dot on its tolerance band, a square
per value), mono labels, hairlines, stations on a shaft with hanging numerals, and indigo only for the loop.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from moeloop.comparison import compare_recipes  # noqa: E402
from moeloop.recipe_spec import RecipeSpec  # noqa: E402
import lineage  # noqa: E402

ASSETS = ROOT / "docs" / "assets"
BRAND = ASSETS / "brand"
README = ROOT / "README.md"
CITATION = ROOT / "CITATION.cff"
# README.md's BibTeX block, between these markers, is rendered from CITATION.cff by bibtex()
CITE = re.compile(r"(<!-- citation: [^\n]* -->\n```bibtex\n).*?(\n```\n<!-- /citation -->)", re.S)
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

# The homepage palette (landing tokens); the dark one is the mark's dark set on GitHub's dark ground.
# Indigo marks the loop only; data marks are ink, the second optimizer grey.
LIGHT = dict(ink="#16181D", mid="#5B616E", softt="#737985", soft="#9AA0AB", hair="#D5D8DE", line="#E3E5E9",
             faint="#ECEEF1", band="#EEF0F3", card="#FFFFFF", panel="#FFFFFF", acc="#3346D3", acc2="#7C89E4",
             acc3="#AAB3F0", tint="#F2F4FD", tint2="#E6E9FB", bar="#16181D", grey="#9AA0AB", on_acc="#FFFFFF",
             spark="#FFFFFF", lit="#7C89E4", inkhi="#1F2757")
DARK = dict(ink="#E7EBF1", mid="#9BA3AF", softt="#848B97", soft="#5E6672", hair="#363D48", line="#272D36",
            faint="#1B2028", band="#1B2028", card="#0D1117", panel="#0D1117", acc="#7486F2", acc2="#5866C9",
            acc3="#3B458B", tint="#151A2E", tint2="#1D2442", bar="#DCE0E7", grey="#6B7280", on_acc="#0D1117",
            spark="#FFFFFF", lit="#B9C2FB", inkhi="#BBC3EA")
AUTO = {key: f"var(--{key})" for key in LIGHT}  # the narrow variant: colours chosen inside the file

# The one-change chain of Phase 1, in README order: (step name, detail from the model fields, recipe node)
CHAIN = [("Ordinary MoE", "{num_experts} experts", "phase1/m0"),
         ("Finer experts", "{num_experts}, top-{top_k}", "shape/g"),
         ("Deeper, narrower", "{layers} layers", "shape/u"),
         ("Expert layout", "{prefix_experts} · {core_experts} · {suffix_experts}", "shape/r"),
         ("Residual scale", "middle ×{core_residual}", "phase1/m4"),
         ("The loop", "middle ×{num_visits}", "phase1/m3")]
# Published designs as children in scripts/lineage.py, named by what they do
DESIGNS = {"methods/meta": "Prelude, loop, coda", "methods/foil": "Flattened core",
           "methods/expert-tying": "Tied expert banks", "methods/huginn": "Recurrent core",
           "methods/sandwich": "Decaying re-entry", "methods/loom": "Whole stack, averaged state",
           "methods/ouro": "Whole stack, learned exit", "methods/relaxed": "Whole stack, low-rank delta",
           "methods/smelt-v3": "Middle loop", "methods/smelt-v4": "Middle loop",
           "methods/latentloop-residual": "Residual per visit"}
MECHANISMS = [("Where the loop sits, how often", ["prefix_cells · num_cells", "suffix_cells · num_visits"]),
              ("Experts per region", ["prefix · core · suffix"]),
              ("What a visit shares", ["attention_per_visit", "router_per_visit · norm_per_visit"]),
              ("Residual scale", ["core.residual_scale", "visit_residual_scales"]),
              ("Input re-entry", ["input_injection"]),
              ("Initial state and update", ["initial_state · state_update"]),
              ("Depth: fixed, sampled, learned", ["depth_rule · grad_visits", "exit_distribution"]),
              ("Low-rank delta per visit", ["lora_attention_rank", "lora_ffn_rank"]),
              ("Experts tied across layers", ['order="layer"'])]
CASES = {"ordinary_moe": "Ordinary MoE", "shared_stack": "Shared stack", "shared_layer": "Shared layer",
         "partial_attention": "Attention per visit", "partial_router": "Router per visit",
         "expert_tying": "Expert tying"}
QUANTITIES = [("logits", "Logits", "logits_and_cached_logits"), ("loss", "Loss", "loss"),
              ("gradient", "Gradients", "gradient"), ("updated_parameter", "Optimizer step", "updated_parameter"),
              ("cached_logits", "Cached logits", "logits_and_cached_logits")]
BUDGETS = [("unique_non_embedding_parameters", "Parameters"), ("training_flops_per_token", "Training FLOPs"),
           ("logical_kv_bytes", "KV cache")]


# ---------------------------------------------------------------- drawing primitives

def num(v):
    return f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def paint(fill="none", stroke=None, sw=None, op=None, dash=None):
    parts = [f"fill:{fill}"]
    if stroke:
        parts += [f"stroke:{stroke}", f"stroke-width:{num(sw or 1)}"]
    if dash:
        parts.append(f"stroke-dasharray:{dash}")
    if op is not None:
        parts.append(f"opacity:{num(op)}")
    return ";".join(parts)


def rect(x, y, w, h, r=0, fill="none", stroke=None, sw=1, op=None, dash=None, cls=None):
    c = f' class="{cls}"' if cls else ""
    return (f'<rect{c} x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" rx="{num(r)}" '
            f'style="{paint(fill, stroke, sw, op, dash)}"/>')


def circle(cx, cy, r, fill="none", stroke=None, sw=1, op=None):
    return f'<circle cx="{num(cx)}" cy="{num(cy)}" r="{num(r)}" style="{paint(fill, stroke, sw, op)}"/>'


def path(d, stroke, sw=1.5, fill="none", dash=None, op=None, cls=None, extra=""):
    c = f' class="{cls}"' if cls else ""
    return (f'<path{c} d="{d}" style="{paint(fill, stroke, sw, op, dash)};stroke-linecap:round;'
            f'stroke-linejoin:round"{extra}/>')


def line(x1, y1, x2, y2, stroke, sw=1, dash=None):
    return path(f"M{num(x1)} {num(y1)}L{num(x2)} {num(y2)}", stroke, sw, dash=dash)


def text(x, y, s, size, fill, weight=400, anchor="start", mono=False, ls=None):
    spacing = f";letter-spacing:{num(ls)}px" if ls else ""
    return (f'<text class="{"m" if mono else "s"}" x="{num(x)}" y="{num(y)}" text-anchor="{anchor}" '
            f'style="font-size:{num(size)}px;font-weight:{weight};fill:{fill}{spacing}">{esc(s)}</text>')


def width(s, size, mono=False, weight=400):
    """A generous estimate of the rendered width, for layout only (the widest common fallback fonts)."""
    return len(s) * size * (0.61 if mono else 0.6 if weight >= 600 else 0.57)


def mlabel(x, y, s, t, anchor="start", fill=None):
    """A mono label in capitals, as the homepage's keys."""
    return text(x, y, s.upper(), 11, fill or t["mid"], 500, anchor, mono=True, ls=.66)


def mlabel_width(s):
    return width(s, 11, mono=True) + .66 * len(s)


def arrow_head(x, y, dx, dy, stroke, sw=1.5, size=5):
    """A chevron at (x, y) pointing along (dx, dy)."""
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    a = (x - size * ux + size * 0.8 * uy, y - size * uy - size * 0.8 * ux)
    b = (x - size * ux - size * 0.8 * uy, y - size * uy + size * 0.8 * ux)
    return path(f"M{num(a[0])} {num(a[1])}L{num(x)} {num(y)}L{num(b[0])} {num(b[1])}", stroke, sw)


def card(x, y, w, h, t, r=20):
    """A panel: the page's ground with the one hairline border."""
    return rect(x + .5, y + .5, w - 1, h - 1, r, t["card"], t["line"])


def station(x, y, t, fill=None):
    """A station on a shaft: an upright tick 4 x 16 whose foot stands on y."""
    return rect(x, y - 16, 4, 16, 1, fill or t["ink"])


def tolerance_band(x, y, span, t, h=8):
    """The tolerance band, centred on y, with its zero tick."""
    return (rect(x, y - h / 2, span, h, h / 2, t["band"], t["line"]) +
            line(x + span / 2, y - h / 2 - 3, x + span / 2, y + h / 2 + 3, t["soft"], 1))


def okline(x, y, matched, t, label, anchor="end"):
    """The budget state shown by its band: the real dots on one small band, then the words. Returns (elements, width)."""
    bw = 56
    w = bw + 10 + width(label, 12, mono=True)
    x0 = x - w if anchor == "end" else x
    out = tolerance_band(x0, y, bw, t, 6)
    for entry in matched.values():
        rel, tol = entry["relative_difference"], entry["relative_tolerance"]
        inside = abs(rel) <= tol
        cx = x0 + bw / 2 + max(-1, min(1, rel / tol)) * bw / 2
        out += circle(cx, y, 2.6, t["ink"] if inside else t["panel"], t["ink"] if not inside else None, 1.2)
    out += text(x0 + bw + 10, y + 4, label, 12, t["ink"], 500, mono=True)
    return out, w


def squares(x, y, flags, t, size=5, gap=2):
    """One square per value, ink when it is inside its tolerance, open when it is not. Returns (elements, width)."""
    out = "".join(rect(x + i * (size + gap), y - size / 2, size, size, 1, t["ink"]) if ok else
                  rect(x + i * (size + gap) + .5, y - size / 2 + .5, size - 1, size - 1, 1, t["panel"], t["ink"], 1)
                  for i, ok in enumerate(flags))
    return out, len(flags) * (size + gap) - gap


def lockup(name, x, y, h):
    """A lockup file of docs/assets/brand/, placed as it is, h pixels high. Returns (elements, width)."""
    src = (BRAND / f"looptitan-lockup-{name}.svg").read_text()
    vw, vh = (float(v) for v in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', src).groups())
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S).group(1)
    s = h / vh
    return f'<g transform="translate({num(x)} {num(y)}) scale({s:.5f})">{inner}</g>', vw * s


def svg(w, h, title, desc, t, body, css=""):
    style = f".s{{font-family:{SANS}}}.m{{font-family:{MONO}}}"
    if t is AUTO:
        light = ";".join(f"--{k}:{v}" for k, v in LIGHT.items())
        dark = ";".join(f"--{k}:{v}" for k, v in DARK.items())
        style += f":root{{{light}}}@media (prefers-color-scheme: dark){{:root{{{dark}}}}}"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-labelledby="title desc">\n<title id="title">{esc(title)}</title>\n'
            f'<desc id="desc">{esc(desc)}</desc>\n<style>{style}{css}</style>\n{body}\n</svg>\n')


# ---------------------------------------------------------------- data

def recipe(node, optimizer=None):
    files = lineage.recipe_files(node)
    path_ = next(p for p in files if optimizer is None or p.name.endswith(f"-{optimizer}.json"))
    return json.loads(path_.read_text())


def layers(node):
    """(model fields, [(region, expert parameters of the layer)] in stored order)."""
    m = recipe(node)["model"]

    def params(region):
        o = m.get(region) or {}
        return 3 * (o.get("num_experts") or m["num_experts"]) * (o.get("hidden_dim") or m["hidden_dim"]) * m["dim"]
    out = [("prefix", params("prefix"))] * m["prefix_cells"] + [("core", params("core"))] * m["num_cells"]
    return m, out + [("suffix", params("suffix"))] * m["suffix_cells"]


def budgets(reference, candidate):
    specs = [RecipeSpec.from_dict(recipe(node)) for node in (reference, candidate)]
    return compare_recipes(*specs, seq_len=specs[0].model.max_seq_len)["matched"]


def checks():
    record = json.loads((ROOT / "docs" / "evidence" / "qwen3-cpu.json").read_text())
    rows = []
    for case, label in CASES.items():
        errors = record["cases"][case]["max_absolute_errors"]
        errors = errors | {"cached_logits": record["cases"][case]["cache"]["logits_max_absolute_error"]}
        rows.append((label, [(errors[q], record["tolerances"][tol]["atol"]) for q, _, tol in QUANTITIES]))
    return record, rows


def chain_deltas():
    """For each optimizer: mBPB of each chain step against step 0."""
    rows = {row["runs"]["adamw"]["config"]: row for row in
            json.loads((ROOT / "docs" / "results" / "phase1.json").read_text())["groups"][0]["rows"]}
    out = {}
    for opt in ("adamw", "muon"):
        bpb = [rows[f"recipes/{node}-adamw.json"]["runs"][opt]["test_bpb"] for _, _, node in CHAIN]
        out[opt] = [1000 * (b - bpb[0]) for b in bpb]
    return out


def pct(fraction):
    return f"{100 * fraction:g}%"


def within(matched):
    """(budgets inside their tolerance, budgets, the tolerance) of one compare result."""
    inside = sum(abs(v["relative_difference"]) <= v["relative_tolerance"] for v in matched.values())
    return inside, len(matched), next(iter(matched.values()))["relative_tolerance"]


def p0():
    return layers("phase1/m0")[1][0][1]


# ---------------------------------------------------------------- the model glyph

def glyph(node, x, cy, t, k=1.0, label=True):
    """A model as bars on a line: one bar per stored layer, its area the layer's expert parameters;
    looped layers in indigo under an arc with the visit count. Returns (elements, width)."""
    m, cells = layers(node)
    base = p0()
    sizes = [(region, 5 * k * math.sqrt(p / base), 26 * k * math.sqrt(p / base)) for region, p in cells]
    gap = 3 * k
    total = sum(w for _, w, _ in sizes) + gap * (len(sizes) - 1)
    out = [line(x - 5 * k, cy, x + total + 5 * k, cy, t["hair"], 1.2)]
    looped = m["num_visits"] > 1
    xs, cx = [], x
    for region, w, h in sizes:
        loop = looped and region == "core"
        if loop and m["order"] == "layer":  # one bank serves num_visits consecutive layers: a comb
            n = m["num_visits"]
            tw = 3 * k
            for i in range(n):
                out.append(rect(cx + i * (tw + 1.4 * k), cy - h / 2, tw, h - 4 * k, 1.2 * k, t["bar"]))
            w = n * tw + (n - 1) * 1.4 * k
            out.append(rect(cx - .6 * k, cy + h / 2 - 5 * k, w + 1.2 * k, 5 * k, 2 * k, t["acc"]))
        else:
            out.append(rect(cx, cy - h / 2, w, h, min(w / 2, 2.5 * k), t["acc"] if loop else t["bar"]))
        xs.append((cx, w, h, loop))
        cx += w + gap
    total = cx - gap - x
    out[0] = line(x - 5 * k, cy, x + total + 5 * k, cy, t["hair"], 1.2)
    core = [(a, w, h) for a, w, h, loop in xs if loop]
    if core and m["order"] != "layer":
        top = cy - max(h for _, _, h in core) / 2 - 3 * k
        x1, x2 = core[-1][0] + core[-1][1] / 2, core[0][0] + core[0][1] / 2
        if x1 - x2 < 6 * k:  # a single looped layer: the arc returns over it
            x1, x2 = x1 + 4 * k, x2 - 4 * k
        c = 9 + 5 * k
        out.append(path(f"M{num(x1)} {num(top)}C{num(x1)} {num(top - c)} {num(x2)} {num(top - c)} {num(x2)} {num(top)}",
                        t["acc"], 1.3))
        out.append(arrow_head(x2, top, 0, 1, t["acc"], 1.3, 3.2))
        if label:
            out.append(text((x1 + x2) / 2, top - .75 * c - 3.5, f"×{m['num_visits']}", 12, t["acc"], 600, "middle"))
    return "".join(out), total


# ---------------------------------------------------------------- figures

def banner(t):
    """The lockup as the homepage's header carries it, and the homepage's display line under it."""
    narrow = t is AUTO
    w, h = (360, 232) if narrow else (880, 272)
    cx = w / 2
    lh = 40 if narrow else 52  # the mark's height in pixels: the medium master (22 to 95 px)
    _, lw = lockup("medium-indigo", 0, 0, lh)
    top = 20 if narrow else 24
    css = ""
    if narrow:  # both colourings of the lockup; the reader's scheme shows one
        light, _ = lockup("medium-indigo", cx - lw / 2, top, lh)
        dark, _ = lockup("medium-dark", cx - lw / 2, top, lh)
        out = [f'<g class="lk">{light}</g><g class="lk-d">{dark}</g>']
        css = ".lk-d{display:none}@media (prefers-color-scheme: dark){.lk{display:none}.lk-d{display:inline}}"
    else:
        out = [lockup("medium-indigo" if t is LIGHT else "medium-dark", cx - lw / 2, top, lh)[0]]
    # one soft light behind the whole group: a single ellipse, brightest at its centre
    gy, grx, gry = (h / 2, 176, 112) if narrow else (h / 2, 380, 128)
    out.insert(0, '<defs><radialGradient id="glow" cx="50%" cy="50%" r="50%">' + "".join(
        f'<stop offset="{o}" style="stop-color:{t["acc"]};stop-opacity:{a}"/>'
        for o, a in [(0, .085), (.4, .06), (.75, .022), (1, 0)]) + '</radialGradient></defs>'
        f'<ellipse cx="{num(cx)}" cy="{num(gy)}" rx="{grx}" ry="{gry}" fill="url(#glow)"/>')
    if narrow:
        out.append(text(cx, 112, "A common ground for", 27, t["ink"], 650, "middle", ls=-.8))
        out.append(text(cx, 145, "looped MoE research.", 27, t["ink"], 650, "middle", ls=-.8))
        out.append(text(cx, 182, "The design space of looped MoE in one", 14.5, t["mid"], 400, "middle"))
        out.append(text(cx, 204, "model, compared on equal terms.", 14.5, t["mid"], 400, "middle"))
    else:
        out.append(text(cx, 146, "A common ground for", 44, t["ink"], 650, "middle", ls=-1.6))
        out.append(text(cx, 196, "looped MoE research.", 44, t["ink"], 650, "middle", ls=-1.6))
        out.append(text(cx, 240, "The design space of looped MoE in one model, compared on equal terms.", 17,
                        t["mid"], 400, "middle"))
    return svg(w, h, "LoopTitan", "A common ground for looped MoE research. The design space of looped MoE in "
               "one model, compared on equal terms.", t, "".join(out), css)


def badge(label, value, t):
    """A mono key and its value over a hairline, as the homepage's release line."""
    kw = mlabel_width(label)
    end = math.ceil(kw + 8 + width(value, 12, mono=True))
    body = (mlabel(0, 14, label, t) + text(kw + 8, 14, value, 12, t["ink"], 500, mono=True) +
            line(0, 21.5, end, 21.5, t["hair"], 1))
    return svg(end + 12, 22, f"{label}: {value}", f"{label}: {value}", t, body)


# The README's destination capsules, in row order: (key, label, colour family, mark). Marks of other organisations
# are their own files in docs/assets/brand/external/, placed unchanged (SOURCES.md there gives source and terms).
PLACES = [("homepage", "Homepage", "ours", "brand/looptitan-shell-small"), ("docs", "Documentation", "ours", "octicons-book-16"),
          ("models", "Models", "hf", "huggingface-logo"), ("paper", "Paper", "ours", "octicons-file-16"),
          ("results", "Results", "ours", "bars"), ("record", "Record", "github", "octicons-issue-opened-16"),
          ("contribute", "Contribute", "github", "octicons-git-pull-request-16")]
HOSTS = {"paper": "arXiv"}  # the service named in words where the mark cannot name it
# per family and theme: (ground, border, mark colour); LoopTitan's own places take the identity's indigo
HUES = {"ours": {"light": ("#F2F4FD", "#C8CFF5", "#3346D3"), "dark": ("#151A2E", "#2C3672", "#7486F2")},
        "hf": {"light": ("#FFF8DD", "#F3D77A", None), "dark": ("#241F0E", "#5E4F1C", None)},
        "github": {"light": ("#EEF7F1", "#B5DCC1", "#1A7F37"), "dark": ("#0F2418", "#21543A", "#3FB950")}}


def placed(file, x, y, size, colour=None):
    """An SVG file of docs/assets/, unchanged, scaled into a size x size box at (x, y); colour fills a one-colour icon."""
    src = (ASSETS / file).read_text()
    vw, vh = (float(v) for v in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', src).groups())
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S).group(1)
    k = size / max(vw, vh)
    fill = f' style="fill:{colour}"' if colour else ""
    return f'<g transform="translate({num(x + (size - vw * k) / 2)} {num(y + (size - vh * k) / 2)}) scale({k:.5f})"{fill}>{inner}</g>'


# Advance widths of a common sans (per 1000 em), so that a label's length is set, not left to the viewer's font
ADVANCE = dict(zip("abcdefghijklmnopqrstuvwxyz", [556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833,
                                                   556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500]))
ADVANCE |= dict(zip("CDHMPR", [722, 722, 722, 833, 667, 722]))


def label_text(x, y, s, size, fill, weight=500, mono=False):
    """A label held to one length in every font: the letter spacing absorbs the difference between fonts."""
    w = len(s) * size * .6 if mono else sum(ADVANCE[c] for c in s) * size / 1000 * 1.03
    return (f'<text class="{"m" if mono else "s"}" x="{num(x)}" y="{num(y)}" textLength="{num(w)}" lengthAdjust="spacing" '
            f'style="font-size:{num(size)}px;font-weight:{weight};fill:{fill}">{esc(s)}</text>'), w


def capsule(key, label, family, mark, primary, t, host=None):
    """A destination capsule: the place's mark and name on its colour's light ground. One height, one type and one
    rhythm: 12 px, the 16 px mark, 8 px, the label held to its length, 14 px; host names the service in mono."""
    h, size, pad, gap = 32, 13, 12, 8
    ground, border, colour = HUES[family]["light" if t is LIGHT else "dark"]
    ink, soft = t["ink"], t["mid"]
    if primary:
        ground, border, ink, colour, soft = t["acc"], None, t["on_acc"], t["on_acc"], t["on_acc"]
    x = pad + 16 + gap
    words, tw = label_text(x, 21, label, size, ink)
    if host:
        tag, hw = label_text(x + tw + 7, 21, host, 11.5, soft, 400, mono=True)
        words += tag
        tw += 7 + hw
    w = round(x + tw + pad + 2)
    out = [rect(.5, .5, w - 1, h - 1, (h - 1) / 2, ground, border)]
    if mark == "bars":  # results: bars of a result, the last in the loop's colour
        out += [rect(pad + 1.5 + i * 4.6, 22 - bh, 3.2, bh, 1, colour if i == 2 else ink) for i, bh in enumerate([6, 9, 12])]
    elif mark.startswith("brand/"):
        out.append(placed(f"{mark}-{'white' if t is LIGHT else 'ink'}.svg", pad, 8.5, 15))
    elif mark == "huggingface-logo":  # its artwork leaves a margin in its square: drawn 2 px larger to match
        out.append(placed(f"brand/external/{mark}.svg", pad - 1, 7, 18))
    else:
        out.append(placed(f"brand/external/{mark}.svg", pad, 8, 16, colour))
    out.append(words)
    return svg(w + 6, h, label, label + (f" ({host})" if host else ""), t, "".join(out))  # 6 px of air after each capsule


def hero(t):
    """The Phase 1 loop: bars on a shaft, the middle layers run twice; a light passes the looped ones twice."""
    narrow = t is AUTO
    m, cells = layers("phase1/m3")
    pre, core, suf, visits = m["prefix_cells"], m["num_cells"], m["suffix_cells"], m["num_visits"]
    if narrow:
        w, h, bw, bh, gap, ggap, cy, r = 360, 288, 12, 72, 9, 20, 104, 2.5
    else:
        w, h, bw, bh, gap, ggap, cy, r = 880, 336, 16, 104, 30, 52, 146, 3
    groups = [pre, core, suf]
    total = sum(n * bw + (n - 1) * gap for n in groups) + 2 * ggap
    x = (w - total) / 2
    bars = []
    for g, n in enumerate(groups):
        for i in range(n):
            bars.append((x, g == 1))
            x += bw + (gap if i < n - 1 else ggap)
    first, last = bars[pre][0], bars[pre + core - 1][0]
    loop_l, loop_r = first - ggap / 2, last + bw + ggap / 2
    xs, xe = (16, w - 16) if narrow else (64, w - 64)
    top = cy - bh / 2
    a0 = top - (6 if narrow else 8)
    c = 46 if narrow else 72
    ax1, ax2 = last + bw / 2, first + bw / 2
    arc = f"M{num(ax1)} {num(a0)}C{num(ax1)} {num(a0 - c)} {num(ax2)} {num(a0 - c)} {num(ax2)} {num(a0)}"
    # the light: the token's path through the layers, timed as the homepage's pulse. One quick hop per layer
    # with the motion easing, the way back along the arc, out, then a rest; each layer flashes as it is reached.
    T, hop, back, edge, start, fade = 4.0, .075, .54, .12, .3, .06
    route = f"M{num(xs)} {num(cy)}H{num(ax1)}V{num(a0)}C{num(ax1)} {num(a0 - c)} {num(ax2)} {num(a0 - c)} {num(ax2)} {num(a0)}V{num(cy)}H{num(xe)}"
    ctrl = [(ax1, a0), (ax1, a0 - c), (ax2, a0 - c), (ax2, a0)]
    pts = [tuple((1 - u) ** 3 * p[0] + 3 * (1 - u) ** 2 * u * p[1] + 3 * (1 - u) * u ** 2 * p[2] + u ** 3 * p[3]
                 for p in zip(*ctrl)) for u in (i / 400 for i in range(401))]
    second = ax1 - xs + 2 * (cy - a0) + sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))  # where the second visit starts
    length = second + xe - ax2
    seq = list(range(pre)) + list(range(pre, pre + core)) * visits + list(range(pre + core, pre + core + suf))
    keys = [(0, 0, None), (start, 0, None)]  # (time, distance along the route, easing of the move that ends here)
    hits = {}
    for j, i in enumerate(seq):
        centre = bars[i][0] + bw / 2
        again = j >= pre + core  # past the first visit: measured from where the way back lands
        d = (second + centre - ax2) if again else (centre - xs)
        move = edge if j == 0 else back if i < seq[j - 1] else hop * (i - seq[j - 1])
        keys.append((keys[-1][0] + move, d, None if j == 0 else ".65 0 .35 1"))
        hits.setdefault(i, []).append(keys[-1][0])
    keys.append((keys[-1][0] + edge, length, None))
    end = keys[-1][0]
    keys.append((T, length, None))
    times = ";".join(f"{k / T:.4f}" for k, _, _ in keys)
    splines = ";".join(e or "0 0 1 1" for _, _, e in keys[1:])
    timing = f'dur="{num(T)}s" repeatCount="indefinite" keyTimes="{times}" calcMode="spline" keySplines="{splines}"'
    tail = (14 if narrow else 22) / length * 1000
    css = ["@media (prefers-reduced-motion: reduce){.mo{display:none}}"]
    out = ['<defs><radialGradient id="glow" cx="50%" cy="50%" r="50%">'
           f'<stop offset="0" style="stop-color:{t["acc"]};stop-opacity:.10"/>'
           f'<stop offset="1" style="stop-color:{t["acc"]};stop-opacity:0"/></radialGradient></defs>']
    out.append(f'<ellipse cx="{num((loop_l + loop_r) / 2)}" cy="{num(cy)}" rx="{num((loop_r - loop_l) * .72)}" '
               f'ry="{num(bh * .9)}" fill="url(#glow)"/>')
    out.append(line(xs, cy, xe, cy, t["hair"], 1))
    for i, (bx, loop) in enumerate(bars):
        out.append(rect(bx, top, bw, bh, r, t["acc"] if loop else t["bar"]))
        flash = [(0, 0)] + [(at + dt, v) for at in hits[i] for dt, v in [(-.01, 0), (0, 1), (.15, .49), (.3, .16), (.5, 0)]] + [(T, 0)]
        out.append(f'<rect class="mo" x="{num(bx)}" y="{num(top)}" width="{bw}" height="{bh}" rx="{num(r)}" opacity="0" '
                   f'style="fill:{t["lit"] if loop else t["inkhi"]}"><animate attributeName="opacity" dur="{num(T)}s" '
                   f'repeatCount="indefinite" keyTimes="{";".join(f"{a / T:.4f}" for a, _ in flash)}" '
                   f'values="{";".join(num(float(v)) for _, v in flash)}"/></rect>')
    out.append(path(arc, t["acc"], 1.6 if narrow else 2))
    out.append(arrow_head(ax2, a0, 0, 1, t["acc"], 1.6 if narrow else 2, 5.5 if narrow else 7))
    apex = a0 - .75 * c
    out.append(text((ax1 + ax2) / 2, apex - (9 if narrow else 12), f"×{visits}", 18 if narrow else 22, t["acc"], 600, "middle"))
    show = [(0, 0), (start, 0), (start + fade, 1), (end - fade, 1), (end, 0), (T, 0)]
    out.append(f'<g class="mo" opacity="0"><animate attributeName="opacity" dur="{num(T)}s" repeatCount="indefinite" '
               f'keyTimes="{";".join(f"{a / T:.4f}" for a, _ in show)}" values="{";".join(str(v) for _, v in show)}"/>'
               f'<path d="{route}" pathLength="1000" style="fill:none;stroke:{t["lit"]};stroke-width:{3 if narrow else 4};'
               f'stroke-linecap:round;stroke-dasharray:{num(tail)} 2000;stroke-opacity:.6">'
               f'<animate attributeName="stroke-dashoffset" {timing} '
               f'values="{";".join(num(tail - 1000 * d / length) for _, d, _ in keys)}"/></path>'
               f'<g><animateMotion path="{route}" {timing} keyPoints="{";".join(f"{d / length:.4f}" for _, d, _ in keys)}"/>'
               + circle(0, 0, 8 if narrow else 11, t["acc"], op=.18) + circle(0, 0, 3.2 if narrow else 4.5, t["spark"], t["acc"], 1.5 if narrow else 1.75)
               + "</g></g>")
    # each group: a hairline span under it and its label
    yb = cy + bh / 2 + (14 if narrow else 20)
    starts = [bars[0][0], first, bars[pre + core][0]]
    ends = [bars[pre - 1][0] + bw, last + bw, bars[-1][0] + bw]
    size = 12.5 if narrow else 14
    run_words = "twice" if visits == 2 else f"{visits} times"
    labels = [f"{pre} layers", f"{core} layers, one set of weights, run {run_words}", f"{suf} layers"]
    if narrow:
        labels[1] = f"{core} layers, run {run_words}"
    for i, (a, b) in enumerate(zip(starts, ends)):
        col = t["acc"] if i == 1 else t["hair"]
        out.append(path(f"M{num(a)} {num(yb - 4)}V{num(yb)}H{num(b)}V{num(yb - 4)}", col, 1))
        out.append(text((a + b) / 2, yb + (19 if narrow else 23), labels[i], size, t["acc"] if i == 1 else t["mid"],
                        500 if i == 1 else 400, "middle"))
    if narrow:
        out.append(text(w / 2, yb + 37, "one set of weights", size, t["acc"], 500, "middle"))
    # the two counts, as hanging numerals: the label above, the number under it
    stored, run = pre + core + suf, pre + core * visits + suf
    ny = h - (18 if narrow else 22)
    for dx, label, value, col in [(-1, "layers stored", stored, t["ink"]), (1, "layers run", run, t["acc"])]:
        x = w / 2 + dx * (54 if narrow else 64)
        out.append(mlabel(x, ny - (30 if narrow else 34), label, t, "middle"))
        out.append(text(x, ny, str(value), 24 if narrow else 28, col, 600, "middle"))
    desc = (f"{pre} outer layers, {core} middle layers that run {visits} times with one set of weights, "
            f"{suf} outer layers: {stored} layers stored, {run} run. A light passes the middle layers twice.")
    return svg(w, h, "The Phase 1 loop", desc, t, "".join(out), "".join(css))


# ---- pillars (each figure in a local box 256 x 120)

def pillar_designs(t):
    """Many designs, one model: three looped designs, drawn as in the lineage figure, into one ModelConfig."""
    out = []
    for node, y in zip(["phase1/m3", "methods/huginn", "methods/ouro"], [22, 60, 98]):
        g, gw = glyph(node, 4, y, t, .56, label=False)
        out.append(g)
        visits = recipe(node)["model"]["num_visits"]
        out.append(text(4 + gw + 10, y + 4.3, f"×{visits}", 12, t["acc"], 600, mono=True))
        x = 4 + gw + 34
        out.append(path(f"M{num(x)} {y}C{num(x + 30)} {y} 128 60 152 60", t["hair"], 1))
    out.append(station(156, 68, t) + text(168, 64.5, "ModelConfig", 12, t["ink"], 500, mono=True))
    return "".join(out)


def pillar_budgets(t, matched):
    out = []
    for i, (key, label) in enumerate(BUDGETS):
        y = 24 + i * 32
        rel = matched[key]["relative_difference"]
        tol = matched[key]["relative_tolerance"]
        x0, x1 = 96, 184
        out.append(text(0, y + 4.2, label.replace("Training ", ""), 12, t["mid"]))
        out.append(tolerance_band(x0, y, x1 - x0, t))
        out.append(circle((x0 + x1) / 2 + rel / tol * (x1 - x0) / 2, y, 4.5, t["ink"], t["panel"], 2))
        out.append(text(256, y + 4.2, f"{100 * rel:+.2f}%".replace("-", "−"), 12, t["ink"], 500, "end", mono=True))
    tol = within(matched)[2]
    for x, s in [(96, f"−{pct(tol)}"), (140, "0"), (184, f"+{pct(tol)}")]:
        out.append(text(x, 114, s, 11, t["softt"], 400, "middle", mono=True))
    return "".join(out)


def pillar_checks(t, rows):
    """A square per value: a row per configuration, a column per quantity."""
    out = []
    for i, (_, cells) in enumerate(rows):
        for j, (value, tol) in enumerate(cells):
            x, y = 4 + j * 17, 10 + i * 17
            out.append(rect(x, y, 12, 12, 1.5, t["ink"]) if value <= tol else
                       rect(x + .5, y + .5, 11, 11, 1.5, t["panel"], t["ink"]))
    out.append(text(108, 50, f"{len(rows)} configurations", 12, t["mid"], 400, mono=True))
    out.append(text(108, 68, f"× {len(rows[0][1])} quantities", 12, t["mid"], 400, mono=True))
    return "".join(out)


def pillars(t):
    narrow = t is AUTO
    matched = budgets("phase1/m0", "phase1/m3")
    record, rows = checks()
    inside = sum(v <= tol for _, cells in rows for v, tol in cells)
    n = sum(len(cells) for _, cells in rows)
    b_in, b_n, b_tol = within(matched)
    tiles = [(pillar_designs(t), str(len(lineage.lineage())), "configurations, one model",
              f"{len(MECHANISMS)} mechanisms, each a field"),
             (pillar_budgets(t, matched), f"{b_in} of {b_n}", f"budgets within ±{pct(b_tol)}",
              "The loop against the ordinary MoE"),
             (pillar_checks(t, rows), f"{inside} of {n}", "within tolerance", "Against Transformers Qwen3-MoE")]
    cw, ch, gap = (360, 228, 28) if narrow else (272, 228, 32)
    w = cw if narrow else 3 * cw + 2 * gap
    h = 3 * ch + 2 * gap if narrow else ch
    out = []
    for i, (art, big, head, sub) in enumerate(tiles):
        x, y = (0, i * (ch + gap)) if narrow else (i * (cw + gap), 0)
        if narrow and i:
            out.append(line(0, y - gap / 2 + .5, w, y - gap / 2 + .5, t["line"], 1))
        out.append(f'<g transform="translate({num(x)} {y + 8})">{art}</g>')
        out.append(text(x, y + 168, big, 30, t["ink"], 600, ls=-.8))
        out.append(text(x, y + 194, head, 14, t["ink"], 500))
        out.append(text(x, y + 215, sub, 13, t["mid"]))
    desc = (f"Three tiles. {tiles[0][1]} {tiles[0][2]}: {tiles[0][3]}. {tiles[1][1]} {tiles[1][2]}: parameters, "
            f"training FLOPs and KV cache of the loop against the ordinary MoE. {tiles[2][1]} {tiles[2][2]}: six "
            "configurations, five quantities, against Transformers Qwen3-MoE.")
    return svg(w, h, "LoopTitan at a glance", desc, t, "".join(out))


# ---- mechanisms (local box 256 x 116)

def bars_row(t, x, cy, spec, w=10, h=40, gap=8, ggap=18):
    """spec: [(count, colour)]; returns (elements, [(left, right) of each group], [(x) of each bar])."""
    out, spans, xs = [], [], []
    for count, colour in spec:
        left = x
        for i in range(count):
            out.append(rect(x, cy - h / 2, w, h, 2.5, colour))
            xs.append(x)
            x += w + (gap if i < count - 1 else 0)
        spans.append((left, x))
        x += ggap
    return "".join(out), spans, xs


def loop_arc(t, x1, x2, y, c, label=None, sw=1.5):
    out = path(f"M{num(x1)} {num(y)}C{num(x1)} {num(y - c)} {num(x2)} {num(y - c)} {num(x2)} {num(y)}", t["acc"], sw)
    out += arrow_head(x2, y, 0, 1, t["acc"], sw, 4.5)
    if label:
        out += text((x1 + x2) / 2, y - .75 * c - 6, label, 12, t["acc"], 600, "middle")
    return out


def mech_loop(t):
    m = recipe("phase1/m3")["model"]
    body, spans, xs = bars_row(t, 59, 62, [(2, t["bar"]), (3, t["acc"]), (2, t["bar"])])
    out = [line(30, 62, 226, 62, t["hair"], 1), body,
           loop_arc(t, xs[4] + 5, xs[2] + 5, 38, 22, f"×{m['num_visits']}")]
    for (a, b), s, col in zip(spans, ["prelude", "loop", "coda"], [t["mid"], t["acc"], t["mid"]]):
        out.append(text((a + b) / 2, 102, s, 12, col, 500 if s == "loop" else 400, "middle", mono=True))
    return "".join(out)


def mech_experts(t):
    m, cells = layers("shape/r")
    out = [line(18, 56, 238, 56, t["hair"], 1)]
    top = max(p for _, p in cells)
    x, spans = 25, {}
    for i, (region, p) in enumerate(cells):
        h = 50 * p / top
        out.append(rect(x, 56 - h / 2, 7, h, 2, t["bar"]))
        spans.setdefault(region, [x, x])[1] = x + 7
        x += 7 + (10 if i + 1 < len(cells) and cells[i + 1][0] != region else 4)
    for region in ("prefix", "core", "suffix"):
        o = m.get(region) or {}
        e = o.get("num_experts") or m["num_experts"]
        a, b = spans[region]
        out.append(text((a + b) / 2, 102, f"{e} experts" if region == "core" else str(e), 12, t["mid"], 400, "middle", mono=True))
    return "".join(out)


def mech_shares(t):
    out = []
    for i, y in enumerate([36, 70]):
        out.append(text(0, y + 4.2, f"visit {i + 1}", 12, t["mid"], 400, mono=True))
        out.append(rect(58, y - 13, 72, 26, 6, t["tint"]))
        out.append(text(94, y + 4.2, "attention", 12, t["acc"], 500, "middle", mono=True))
        out.append(rect(134, y - 13, 54, 26, 6, t["tint"]))
        out.append(text(161, y + 4.2, "router", 12, t["acc"], 500, "middle", mono=True))
    out.append(rect(192, 23, 60, 60, 6, t["acc"]))
    out.append(text(222, 57.2, "experts", 12, t["on_acc"], 600, "middle", mono=True))
    out.append(text(123, 106, "owned per visit", 12, t["mid"], 400, "middle", mono=True))
    out.append(text(222, 106, "shared", 12, t["mid"], 400, "middle", mono=True))
    return "".join(out)


def mech_residual(t):
    scale = recipe("phase1/m3")["model"]["core"]["residual_scale"]
    out = [rect(44, 16, 72, 24, 12, t["bar"]), text(80, 32.2, "layer", 12, t["panel"], 500, "middle", mono=True),
           rect(132, 16, 104, 24, 12, t["acc"]), text(184, 32.2, "looped layer", 12, t["on_acc"], 500, "middle", mono=True),
           line(80, 46, 80, 74, t["bar"], 2.4), arrow_head(80, 76, 0, 1, t["bar"], 2.4, 6),
           text(90, 66, "×1", 12, t["ink"], 500, mono=True),
           line(184, 46, 184, 74, t["acc"], 1.3), arrow_head(184, 76, 0, 1, t["acc"], 1.3, 6),
           text(194, 66, f"×{num(scale)}", 12, t["acc"], 500, mono=True),
           rect(28, 82, 200, 6, 3, t["band"], t["line"]),
           text(128, 106, "the stream", 12, t["mid"], 400, "middle", mono=True)]
    return "".join(out)


def mech_input(t):
    m = recipe("methods/huginn")["model"]
    out = [line(22, 50, 234, 50, t["hair"], 1), rect(36, 32, 10, 36, 2.5, t["bar"])]
    out.append(circle(74, 50, 7.5, t["panel"], t["acc"], 1.4))
    out.append(path("M70 50H78M74 46V54", t["acc"], 1.4))
    for x in (98, 118):
        out.append(rect(x, 32, 10, 36, 2.5, t["acc"]))
    out.append(rect(156, 32, 10, 36, 2.5, t["bar"]))
    out.append(path("M123 28C123 6 74 6 74 38", t["acc"], 1.5) + arrow_head(74, 40, 0, 1, t["acc"], 1.5, 4.5))
    out.append(text(128, 16, f"×{m['num_visits']}", 12, t["acc"], 600))
    out.append(line(74, 92, 74, 62, t["acc"], 1.3, dash="3 3") + arrow_head(74, 60, 0, -1, t["acc"], 1.3, 4.5))
    out.append(text(84, 96, "input, at every visit", 12, t["acc"], 500, mono=True))
    return "".join(out)


def mech_state(t):
    out = [circle(40, 46, 18, t["panel"], t["hair"], 1)]
    for dx, dy in [(-8, -6), (5, -9), (9, 3), (-3, 4), (-10, 8), (3, 12), (11, -3)]:
        out.append(circle(40 + dx * .9, 46 + dy * .9, 1.8, t["acc"]))
    out.append(line(64, 46, 92, 46, t["hair"], 1) + arrow_head(94, 46, 1, 0, t["hair"], 1, 4))
    for x in (104, 124):
        out.append(rect(x, 28, 10, 36, 2.5, t["acc"]))
    out.append(loop_arc(t, 129, 109, 24, 18))
    out.append(line(144, 46, 172, 46, t["hair"], 1) + arrow_head(174, 46, 1, 0, t["hair"], 1, 4))
    out.append(circle(200, 46, 18, t["panel"], t["acc"], 1.3))
    out.append(f'<path d="M200 28A18 18 0 0 0 200 64Z" style="fill:{t["acc3"]}"/>')
    out.append(text(40, 88, "start", 12, t["mid"], 400, "middle", mono=True))
    out.append(text(40, 104, "random", 12, t["acc"], 500, "middle", mono=True))
    out.append(text(200, 88, "update", 12, t["mid"], 400, "middle", mono=True))
    out.append(text(200, 104, "average", 12, t["acc"], 500, "middle", mono=True))
    return "".join(out)


def mech_depth(t):
    out = [line(26, 40, 230, 40, t["hair"], 1)]
    for i, x in enumerate([56, 98, 140, 182]):
        out.append(text(x + 6, 15, str(i + 1), 12, t["softt"], 400, "middle", mono=True))
        out.append(rect(x, 23, 12, 34, 2.5, t["acc"]))
        out.append(line(x + 6, 61, x + 6, 74, t["acc3"], 1.2, dash="2 2") + arrow_head(x + 6, 77, 0, 1, t["acc3"], 1.2, 3.5))
    out.append(rect(46, 80, 164, 6, 3, t["band"], t["line"]))
    out.append(text(128, 106, "exit after any visit", 12, t["mid"], 400, "middle", mono=True))
    return "".join(out)


def mech_lora(t):
    out = [rect(38, 22, 24, 58, 4, t["acc"]), text(50, 104, "shared", 12, t["mid"], 400, "middle", mono=True),
           text(88, 56, "+", 14, t["mid"], 400, "middle")]
    for i, x in enumerate([116, 184]):
        out.append(rect(x, 22, 24, 58, 4, t["acc3"], op=.55))
        out.append(rect(x + 30, 22, 6, 58, 2, t["acc"]))
        out.append(text(x + 33, 14, "Δ", 12, t["acc"], 600, "middle"))
        out.append(text(x + 18, 104, f"visit {i + 1}", 12, t["mid"], 400, "middle", mono=True))
    return "".join(out)


def mech_tying(t):
    m = recipe("methods/expert-tying")["model"]
    k = m["num_visits"]
    xs = [62 + i * 32 for i in range(k)]
    out = []
    for x in xs:
        out.append(rect(x, 14, 12, 32, 2.5, t["bar"]))
        out.append(line(x + 6, 46, x + 6, 68, t["acc2"], 1.2))
    out.append(text(xs[-1] + 26, 34.5, f"{k} layers", 12, t["mid"], 400, mono=True))
    out.append(rect(46, 68, 140, 26, 6, t["acc"]))
    out.append(text(116, 85.2, "one expert bank", 12, t["on_acc"], 600, "middle", mono=True))
    return "".join(out)


MECH_ART = [mech_loop, mech_experts, mech_shares, mech_residual, mech_input, mech_state, mech_depth, mech_lora, mech_tying]


def mechanisms(t):
    """A specimen sheet: one panel, one hairline grid; in each cell the figure, its name, its fields."""
    narrow = t is AUTO
    cols = 1 if narrow else 3
    w = 360 if narrow else 880
    cw, ch = w / cols, 204
    rows = math.ceil(len(MECHANISMS) / cols)
    h = rows * ch
    out = [card(0, 0, w, h, t)]
    for i in range(1, cols):
        out.append(line(i * cw, 1, i * cw, h - 1, t["line"], 1))
    for j in range(1, rows):
        out.append(line(1, j * ch, w - 1, j * ch, t["line"], 1))
    for i, ((title, fields), art) in enumerate(zip(MECHANISMS, MECH_ART)):
        x, y = (i % cols) * cw, (i // cols) * ch
        out.append(f'<g transform="translate({num(x + (cw - 256) / 2)} {y + 14})">{art(t)}</g>')
        out.append(text(x + 24, y + 152, title, 15, t["ink"], 600))
        for j, field in enumerate(fields):
            out.append(text(x + 24, y + 174 + j * 17, field, 12, t["softt"], 400, mono=True))
    desc = "Nine mechanisms, each a field of one model: " + "; ".join(title for title, _ in MECHANISMS) + "."
    return svg(w, int(h), "Nine mechanisms", desc, t, "".join(out))


# ---- lineage

def chain_detail(fmt, node):
    m, cells = layers(node)

    def experts(region):
        return (m.get(region) or {}).get("num_experts") or m["num_experts"]
    return fmt.format(**m, layers=len(cells), prefix_experts=experts("prefix"), core_experts=experts("core"),
                      suffix_experts=experts("suffix"), core_residual=num((m["core"] or {}).get("residual_scale") or 1.0))


def design_row(t, node, x, y, label_x, k=.56):
    """A published design: its glyph, its visit count and its name."""
    m = recipe(node)["model"]
    g, gw = glyph(node, x, y, t, k, label=False)
    out = [g]
    if m["num_visits"] > 1 and m["order"] != "layer":  # tied layers repeat in place; no loop to count
        out.append(text(label_x - 12, y + 4.3, f"×{m['num_visits']}", 12, t["acc"], 600, "end", mono=True))
    out.append(text(label_x, y + 4.5, DESIGNS[node], 13, t["ink"], 500))
    return "".join(out)


def legend(t, x, y):
    """The key of the glyph, as the homepage's instrument gives it. Returns (elements, width)."""
    out = [rect(x, y - 9, 4, 12, 1, t["bar"]), text(x + 12, y + 1.5, "a stored layer, area = its expert parameters", 12, t["mid"])]
    x2 = x + 12 + width("a stored layer, area = its expert parameters", 12) + 20
    out += [rect(x2, y - 9, 4, 12, 1, t["acc"]), text(x2 + 12, y + 1.5, "looped layers", 12, t["mid"])]
    x3 = x2 + 12 + width("looped layers", 12) + 20
    out += [text(x3, y + 1.5, "×n", 12, t["acc"], 600, mono=True), text(x3 + 24, y + 1.5, "visits per token", 12, t["mid"])]
    return "".join(out), x3 + 24 + width("visits per token", 12) - x


def lineage_fig(t):
    narrow = t is AUTO
    nodes = lineage.lineage()
    kids = {}
    for node, (parent, _, _) in nodes.items():
        kids.setdefault(parent, []).append(node)
    left_rows = []
    for n in kids["phase1/m0"]:
        if n in DESIGNS:
            left_rows.append((n, 0))
            left_rows += [(c, 1) for c in kids.get(n, []) if c in DESIGNS]
    right_rows = [(n, 0) for n in kids["phase1/m3"] if n in DESIGNS]
    out = [""]
    if narrow:
        w = 360
        out.append(text(20, 34, "One change per step", 13, t["ink"], 600))
        y = 74
        out.append(line(20, y, 20, y + 60 * (len(CHAIN) - 1), t["line"], 1))
        for i, (name, fmt, node) in enumerate(CHAIN):
            last = i == len(CHAIN) - 1
            out.append(rect(18, y - 8, 4, 16, 1, t["acc"] if last else t["ink"]))
            g, gw = glyph(node, 40, y + 2, t, .62)
            out.append(g)
            out.append(text(160, y - 1, name, 13, t["acc"] if last else t["ink"], 600))
            out.append(text(160, y + 17, chain_detail(fmt, node), 12, t["mid"], 400, mono=True))
            y += 60
        y += 6
        for title, rows in [("From the ordinary MoE", left_rows), ("From the loop", right_rows)]:
            out.append(line(1, y - 18.5, w - 1, y - 18.5, t["line"], 1))
            out.append(text(20, y + 6, title, 13, t["ink"], 600))
            y += 40
            for node, depth in rows:
                x0 = 20 + depth * 22
                if depth:
                    out.append(path(f"M{x0 - 12} {y - 30}V{y - 6}Q{x0 - 12} {y} {x0 - 6} {y}", t["hair"], 1))
                out.append(design_row(t, node, x0, y, 150))
                y += 44
            y += 16
        out.append(line(1, y - 18.5, w - 1, y - 18.5, t["line"], 1))
        out.append(rect(20, y - 3, 4, 12, 1, t["bar"]) + text(32, y + 7.5, "a stored layer, area = its expert parameters", 12, t["mid"]))
        out.append(rect(20, y + 19, 4, 12, 1, t["acc"]) + text(32, y + 29.5, "looped layers", 12, t["mid"]))
        out.append(text(140, y + 29.5, "×n", 12, t["acc"], 600, mono=True) + text(164, y + 29.5, "visits per token", 12, t["mid"]))
        h = y + 52
    else:
        w = 880
        out.append(text(28, 40, "One change per step", 13, t["ink"], 600))
        out.append(text(w - 28, 40, "Phase 1, every step at equal budgets", 12, t["mid"], 400, "end"))
        cxs = [86 + i * 141.6 for i in range(len(CHAIN))]
        out.append(line(28, 104, w - 28, 104, t["line"], 1))
        for i, ((name, fmt, node), cx) in enumerate(zip(CHAIN, cxs)):
            _, gw = glyph(node, 0, 0, t, .78)
            g, _ = glyph(node, cx - gw / 2, 104, t, .78)
            out.append(g)
            col = t["acc"] if i == len(CHAIN) - 1 else t["ink"]
            out.append(text(cx, 152, name, 13, col, 600, "middle"))
            out.append(text(cx, 171, chain_detail(fmt, node), 12, t["mid"], 400, "middle", mono=True))
        top, pitch = 262, 44
        spine = cxs[0]
        rows_y = [top + i * pitch for i in range(len(left_rows))]
        out.append(rect(spine - 2, 188, 4, 12, 1, t["ink"]))
        out.append(path(f"M{num(spine)} 200V{num(rows_y[-1] - 10)}Q{num(spine)} {num(rows_y[-1])} {num(spine + 10)} {num(rows_y[-1])}",
                        t["hair"], 1))
        out.append(text(spine + 18, 236, "Published designs from the ordinary MoE", 12, t["mid"], 600))
        prev = {}
        for (node, depth), y in zip(left_rows, rows_y):
            x0 = spine + 28 + depth * 30
            if depth:
                py = prev[0]
                out.append(path(f"M{num(spine + 40)} {num(py + 14)}V{num(y - 8)}Q{num(spine + 40)} {num(y)} {num(spine + 48)} {num(y)}H{num(x0 - 6)}",
                                t["hair"], 1))
            else:
                out.append(line(spine, y, x0 - 6, y, t["hair"], 1))
            prev[depth] = y
            out.append(design_row(t, node, x0, y, spine + 136))
        rspine = 486
        ry = [top + i * pitch for i in range(len(right_rows))]
        out.append(rect(cxs[-1] - 2, 188, 4, 12, 1, t["acc"]))
        out.append(path(f"M{num(cxs[-1])} 200V{num(204)}Q{num(cxs[-1])} 212 {num(cxs[-1] - 8)} 212H{num(rspine + 8)}"
                        f"Q{rspine} 212 {rspine} 220V{num(ry[-1] - 10)}Q{rspine} {num(ry[-1])} {rspine + 10} {num(ry[-1])}",
                        t["hair"], 1))
        out.append(text(rspine + 18, 236, "Published designs from the loop", 12, t["mid"], 600))
        for (node, _), y in zip(right_rows, ry):
            out.append(line(rspine, y, rspine + 22, y, t["hair"], 1))
            out.append(design_row(t, node, rspine + 28, y, rspine + 136))
        h = rows_y[-1] + 92
        out.append(line(1, h - 48.5, w - 1, h - 48.5, t["line"], 1))
        out.append(legend(t, 28, h - 22)[0])
    out[0] = card(0, 0, w, h, t)
    desc = ("The one-change chain of Phase 1: " + ", ".join(name for name, _, _ in CHAIN) +
            ". Published designs drawn as children of the ordinary MoE or of the loop, as scripts/lineage.py declares: " +
            "; ".join(DESIGNS[n] for n, _ in left_rows + right_rows) + ".")
    return svg(w, int(h), "Lineage of the configurations", desc, t, "".join(out))


# ---- compare

def fmt_budget(key, value):
    if key == "unique_non_embedding_parameters":
        return f"{value / 1e9:.3f}B"
    if key == "training_flops_per_token":
        return f"{value / 1e9:.2f} GFLOP"
    return f"{value / 1e6:.2f} MB"


def compare_fig(t):
    narrow = t is AUTO
    ref, cand = "phase1/m3", "phase1/m4"
    matched = budgets(ref, cand)
    inside, n, tol = within(matched)
    state = f"{inside} of {n} within ±{pct(tol)}"
    out = [""]
    models = []
    for node, name in [(ref, "The loop"), (cand, "Its untied twin")]:
        m, cells = layers(node)
        models.append((node, name, len(cells), m["prefix_cells"] + m["num_cells"] * m["num_visits"] + m["suffix_cells"]))
    if narrow:
        w = 360
        y = 0
        for node, name, stored, run in models:
            out.append(text(20, y + 34, name, 14, t["acc"] if node == ref else t["ink"], 600))
            out.append(text(20, y + 54, f"{stored} stored · {run} run", 12, t["mid"], 400, mono=True))
            _, gw = glyph(node, 0, 0, t, .62)
            g, _ = glyph(node, w - 22 - gw, y + 58, t, .62)
            out.append(g)
            y += 88
            out.append(line(1, y + .5, w - 1, y + .5, t["line"], 1))
        out.append(text(20, y + 36, "Budgets", 14, t["ink"], 600))
        out.append(okline(w - 20, y + 32, matched, t, state)[0])
        y += 72
        for key, label in BUDGETS:
            out += budget_row(t, matched[key], key, label, 20, y, w - 40, narrow=True)
            y += 70
        h = y
    else:
        w, h = 880, 280
        for i, (node, name, stored, run) in enumerate(models):
            y = i * 140
            out.append(text(28, y + 44, name, 14, t["acc"] if node == ref else t["ink"], 600))
            out.append(text(28, y + 64, f"{stored} layers stored · {run} run", 12, t["mid"], 400, mono=True))
            _, gw = glyph(node, 0, 0, t, .86)
            g, _ = glyph(node, 404 - gw, y + 84, t, .86)
            out.append(g)
        out.append(line(1, 140.5, 432, 140.5, t["line"], 1))
        out.append(line(432.5, 1, 432.5, h - 1, t["line"], 1))
        out.append(text(460, 44, "Budgets of the twin against the loop", 14, t["ink"], 600))
        out.append(okline(460, 70, matched, t, state, anchor="start")[0])
        for i, (key, label) in enumerate(BUDGETS):
            out += budget_row(t, matched[key], key, label, 460, 120 + i * 54, w - 28 - 460)
    out[0] = card(0, 0, w, h, t)
    desc = (f"The loop ({models[0][2]} layers stored, {models[0][3]} run) and its untied twin ({models[1][2]} stored, "
            f"{models[1][3]} run). " + "; ".join(f"{label}: {fmt_budget(k, matched[k]['reference'])} against "
                                              f"{fmt_budget(k, matched[k]['candidate'])}, "
                                              f"{100 * matched[k]['relative_difference']:+.2f}%" for k, label in BUDGETS) +
            f". {state}.")
    return svg(w, int(h), "The loop and its untied twin", desc, t, "".join(out))


def budget_row(t, entry, key, label, x, y, span, narrow=False):
    """One budget: its name, its values, and its dot on the tolerance band; the dot is the last element."""
    rel, tol = entry["relative_difference"], entry["relative_tolerance"]
    out = [text(x, y, label, 13, t["ink"], 600),
           text(x + span, y, f"{100 * rel:+.2f}%".replace("-", "−"), 13, t["ink"], 500, "end", mono=True)]
    values = f"{fmt_budget(key, entry['reference'])} → {fmt_budget(key, entry['candidate'])}"
    if not narrow:
        out.append(text(x + 128, y, values, 12, t["mid"], 400, mono=True))
    ty = y + 18
    out.append(tolerance_band(x, ty, span, t))
    mid = x + span / 2
    if narrow:
        out.append(text(x, ty + 22, values, 12, t["mid"], 400, mono=True))
    out.append(circle(mid + rel / tol * span / 2, ty, 5.5, t["ink"], t["panel"], 2))
    return out


# ---- check

def fmt_err(v):
    if v == 0:
        return "0 exact"
    m, e = f"{v:.1e}".split("e")
    return f"{m}e{int(e)}".replace("-", "−")


def check_fig(t):
    narrow = t is AUTO
    record, rows = checks()
    flags = [v <= tol for _, cells in rows for v, tol in cells]
    state = f"{sum(flags)} of {len(flags)} within tolerance"
    updates = {case["updates"] for case in record["cases"].values()}
    setup = (f"{'/'.join(map(str, sorted(updates)))} {record['optimizer']['name']} updates",
             f"{record['dtype']} on {record['device'].upper()}")
    out = [""]
    if narrow:
        w = 360
        lab, cw0, cw = 128, 128, 46
        h = 60 + 46 + len(rows) * 40 + 72
        out.append(text(20, 34, "Against Transformers Qwen3-MoE", 14, t["ink"], 600))
        half = len(flags) // 2
        sq, sw = squares(20, 55, flags[:half], t, 6, 2)
        out.append(sq + squares(20, 63, flags[half:], t, 6, 2)[0])
        out.append(text(20 + sw + 12, 63, state, 12, t["ink"], 500, mono=True))
        heads = ["Logits", "Loss", "Grad.", "Step", "Cache"]
        for j, head in enumerate(heads):
            out.append(text(cw0 + j * cw + 19, 100, head, 12, t["mid"], 500, "middle"))
        for i, (label, cells) in enumerate(rows):
            y = 130 + i * 40
            out.append(line(1, y - 20.5, w - 1, y - 20.5, t["line"], 1))
            out.append(text(20, y + 4.5, label, 12.5, t["ink"], 500))
            for j, (v, tol) in enumerate(cells):
                x = cw0 + j * cw
                out.append(rect(x, y - 2, 38, 4, 2, t["band"]))
                if v > 0:
                    out.append(rect(x, y - 2, max(3, 38 * v / tol), 4, 2, t["ink"]))
        out.append(line(1, h - 70.5, w - 1, h - 70.5, t["line"], 1))
        out.append(text(20, h - 42, f"A bar fills to its tolerance. {setup[0]},", 12, t["mid"]))
        out.append(text(20, h - 24, f"{setup[1]}, Transformers {record['transformers_version']}.", 12, t["mid"]))
    else:
        w = 880
        h = 140 + (len(rows) - 1) * 50 + 94
        out.append(text(28, 42, "Largest difference from Transformers Qwen3-MoE", 14, t["ink"], 600))
        sw = len(flags) * 7 - 2
        out.append(squares(w - 28 - width(state, 12, mono=True) - 12 - sw, 38, flags, t)[0])
        out.append(text(w - 28, 42, state, 12, t["ink"], 500, "end", mono=True))
        x0, cw = 196, 134
        for j, (_, head, tol) in enumerate(QUANTITIES):
            x = x0 + j * cw
            out.append(text(x, 86, head, 13, t["ink"], 600))
            out.append(text(x, 104, "tol " + fmt_err(record["tolerances"][tol]["atol"]), 12, t["softt"], 400, mono=True))
        for i, (label, cells) in enumerate(rows):
            y = 140 + i * 50
            out.append(line(28, y - 22.5, w - 28, y - 22.5, t["line"], 1))
            out.append(text(28, y + 10.5, label, 13, t["ink"], 500))
            for j, (v, tol) in enumerate(cells):
                x = x0 + j * cw
                out.append(text(x, y + 4, fmt_err(v), 12, t["ink"] if v else t["mid"], 400, mono=True))
                out.append(rect(x, y + 12, 104, 4, 2, t["band"]))
                if v > 0:
                    out.append(rect(x, y + 12, max(3, 104 * v / tol), 4, 2, t["ink"]))
        out.append(line(1, h - 46.5, w - 1, h - 46.5, t["line"], 1))
        out.append(text(28, h - 20, f"Largest absolute difference after {setup[0]}, {setup[1]}, Transformers "
                                    f"{record['transformers_version']}. A bar fills to its tolerance.", 12, t["mid"]))
    out[0] = card(0, 0, w, h, t)
    desc = f"{state}: " + "; ".join(
        f"{label}: " + ", ".join(f"{q} {fmt_err(v)}" for (_, q, _), (v, _) in zip(QUANTITIES, cells)) for label, cells in rows) + "."
    return svg(w, int(h), "Check against an independent implementation", desc, t, "".join(out))


# ---- results

def waterfall(t, deltas, x0, x1, top, pitch, colour, lo=-10, hi=2):
    """One optimizer's chain as a waterfall: each step a bar from the previous total to the new one."""
    n = len(deltas) + 1
    loop_row = len(deltas) - 1

    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)
    out = []
    bottom = top + (n - 1) * pitch + 16
    for v in range(lo, hi + 1, 2):
        out.append(line(X(v), top - 18, X(v), bottom, t["hair"] if v == 0 else t["line"], 1))
        out.append(text(X(v), bottom + 18, f"{v:+d}".replace("-", "−").replace("+0", "0"), 11, t["softt"], 400, "middle", mono=True))
    for i in range(n):
        y = top + i * pitch
        if i == 0:
            out.append(rect(X(0) - 1.5, y - 5, 3, 10, 1, colour))
            continue
        a, b = (0, deltas[-1]) if i == n - 1 else (deltas[i - 1], deltas[i])
        loop = i == loop_row
        out.append(rect(min(X(a), X(b)), y - 5, max(abs(X(b) - X(a)), 2.5), 10, 1.5, t["acc"] if loop else colour))
        out.append(text(min(X(a), X(b)) - 7, y + 4.3, f"{b - a:+.1f}".replace("-", "−"), 12,
                        t["acc"] if loop else t["ink"] if i == n - 1 else t["mid"], 600 if i == n - 1 or loop else 400,
                        "end", mono=True))
        if i < n - 2:
            out.append(line(X(b), y + 5, X(b), y + pitch - 5, t["soft"], 1, dash="2 2"))
    return out


def results_fig(t):
    narrow = t is AUTO
    deltas = chain_deltas()
    run = recipe(CHAIN[0][2])["run"]
    tokens = f"{run['successful_updates'] * run['global_batch_positions'] / 1e9:.1f}B"
    labels = ["Ordinary MoE"] + [f"+ {name}" for name, _, _ in CHAIN[1:]] + ["All five steps"]
    n = len(labels)
    loop_row = len(CHAIN) - 1
    optimizers = [("adamw", "AdamW", t["bar"]), ("muon", "Muon", t["grey"])]
    out = [""]

    def row_labels(x, top, pitch, size):
        o = []
        for i, label in enumerate(labels):
            y = top + i * pitch
            loop, total = i == loop_row, i == n - 1
            o.append(text(x, y + 4.5, label, size, t["acc"] if loop else t["ink"], 600 if loop or total else 400))
        return o

    def band(x, w_, top, pitch, i):
        return rect(x, top + i * pitch - pitch / 2 + 2, w_, pitch - 4, 6, t["tint"])

    def key(x, y, name, colour):
        return rect(x, y - 8, 8, 8, 1, colour) + text(x + 14, y, name, 13, t["ink"], 600)
    if narrow:
        w, pitch = 360, 34
        y = 34
        out.append(text(20, y, "Test mBPB against the ordinary MoE", 14, t["ink"], 600))
        y += 16
        for opt, name, colour in optimizers:
            top = y + 52
            out.append(key(20, y + 28, name, colour))
            out.append(band(8, w - 16, top, pitch, loop_row))
            out += row_labels(20, top, pitch, 12)
            out += waterfall(t, deltas[opt], 176, 340, top, pitch, colour)
            y = top + (n - 1) * pitch + 50
        out.append(line(1, y - 12.5, w - 1, y - 12.5, t["line"], 1))
        out.append(text(20, y + 14, f"← lower is better. {tokens} tokens, equal budgets.", 12, t["mid"]))
        h = y + 34
    else:
        w, pitch, top = 880, 38, 112
        h = top + (n - 1) * pitch + 86
        out.append(text(28, 42, "Test mBPB against the ordinary MoE, one change at a time", 14, t["ink"], 600))
        out.append(text(w - 28, 42, "← lower is better", 12, t["mid"], 400, "end"))
        out.append(band(14, w - 28, top, pitch, loop_row))
        out += row_labels(28, top, pitch, 13)
        for (opt, name, colour), (x0, x1) in zip(optimizers, [(250, 518), (586, 852)]):
            out.append(key(x0, 84, name, colour))
            out += waterfall(t, deltas[opt], x0, x1, top, pitch, colour)
        out.append(line(1, h - 44.5, w - 1, h - 44.5, t["line"], 1))
        out.append(text(28, h - 18, f"Test bits per byte after {tokens} tokens; every step at equal parameters, FLOPs "
                                    "and KV cache.", 12, t["mid"]))
    out[0] = card(0, 0, w, h, t)
    desc = ("A waterfall of the one-change chain under AdamW and Muon, test mBPB against the ordinary MoE: " +
            "; ".join(f"{name}: AdamW {deltas['adamw'][i]:+.1f}, Muon {deltas['muon'][i]:+.1f}"
                      for i, (name, _, _) in enumerate(CHAIN)) + ".")
    return svg(w, int(h), "Phase 1: one change at a time", desc, t, "".join(out))


# ---- how research runs here

FLOW = [("A study", [("Declare", "arms and budget"), ("Run", "each run recorded"), ("Check", "against the design")]),
        ("An outside entry", [("Submit", "a result folder"), ("Check", "the bench workflow"), ("Merge", "complete records")])]
RECORD = ("The record", ["Issues, runs and results.", "Failed runs and refuted", "predictions stay."])


def flow_fig(t):
    """Stations on a shaft: each lane's steps with hanging numerals, both shafts ending in the record."""
    narrow = t is AUTO
    out = []
    if narrow:
        w = 360
        y = 22
        sx = 2
        first = None
        for lane, steps in FLOW:
            out.append(mlabel(26, y, lane, t))
            y += 30
            for i, (title, sub) in enumerate(steps):
                first = first or y
                out.append(rect(sx, y - 2, 14, 4, 1, t["ink"]))
                out.append(text(26, y + 4.5, str(i + 1), 12, t["softt"], 400, mono=True))
                out.append(text(46, y + 5, title, 15, t["ink"], 600))
                out.append(text(46, y + 25, sub, 13, t["mid"]))
                y += 58
            y += 8
        out.append(rect(sx, y - 2, 14, 4, 1, t["ink"]))
        out.insert(0, line(sx + .5, first, sx + .5, y, t["hair"], 1))
        title, lines_ = RECORD
        out.append(text(26, y + 5, title, 15, t["ink"], 600))
        out.append(text(26, y + 25, " ".join(lines_[:2]), 13, t["mid"]))
        out.append(text(26, y + 43, lines_[2], 13, t["mid"]))
        h = y + 56
    else:
        w, h = 880, 252
        col, rx = 196, 636
        for li, (lane, steps) in enumerate(FLOW):
            y = 58 + li * 132  # the shaft
            out.append(mlabel(0, y - 38, lane, t))
            out.append(line(0, y + .5, rx, y + .5, t["hair"], 1))
            for i, (title, sub) in enumerate(steps):
                x = i * col
                out.append(station(x, y, t))
                out.append(text(x + 12, y - 5, str(i + 1), 12, t["softt"], 400, mono=True))
                out.append(text(x, y + 28, title, 15, t["ink"], 600))
                out.append(text(x, y + 48, sub, 13, t["mid"]))
        # the record: one station that both shafts reach
        out.append(rect(rx, 42, 4, 148 + 1, 1, t["ink"]))
        title, lines_ = RECORD
        out.append(text(rx + 24, 100, title, 15, t["ink"], 600))
        for j, s in enumerate(lines_):
            out.append(text(rx + 24, 124 + j * 19, s, 13, t["mid"]))
    desc = ("How research runs here. A study: declare, run, check against the declaration, record. "
            "An outside entry: a submission folder, the bench workflow's check, merge. Both end in the record, "
            "where failed runs and refuted predictions stay.")
    return svg(w, int(h), "How research runs here", desc, t, "".join(out))


# ---------------------------------------------------------------- citation

def bibtex(cff):
    """The one BibTeX entry of README.md and of every model card. If CITATION.cff has a preferred-citation (the
    paper), that entry and only that: @article with its journal, otherwise @misc, with an arXiv identifier
    "arXiv:<id> [<class>]" as eprint, archivePrefix and primaryClass. Otherwise the software entry."""
    def authors(entry):
        return " and ".join("{" + a["name"] + "}" if "name" in a else f"{a['family-names']}, {a['given-names']}"
                            for a in entry["authors"])
    paper = cff.get("preferred-citation")
    if paper is None:
        kind, key = "software", cff["title"].lower()
        fields = [("title", "{" + cff["title"] + "}"), ("author", authors(cff)), ("version", cff["version"]),
                  ("url", cff["url"])]
    else:
        year = str(paper.get("year") or paper["date-published"])[:4]
        kind, key = ("article" if "journal" in paper else "misc"), f"{cff['title'].lower()}{year}"
        fields = [("title", "{" + paper["title"] + "}"), ("author", authors(paper)), ("year", year)]
        if "journal" in paper:
            fields.append(("journal", paper["journal"]))
        arxiv = [m for item in paper.get("identifiers", [])
                 if (m := re.fullmatch(r"arXiv:(\S+)(?: \[(\S+)\])?", str(item["value"])))]
        if arxiv:
            fields += [("eprint", arxiv[0][1]), ("archivePrefix", "arXiv")]
            fields += [("primaryClass", arxiv[0][2])] if arxiv[0][2] else []
        if "doi" in paper:
            fields.append(("doi", paper["doi"]))
        if "url" in paper:
            fields.append(("url", paper["url"]))
    return f"@{kind}{{{key},\n" + ",\n".join(f"  {name} = {{{value}}}" for name, value in fields) + "\n}"


def cited(readme, cff):
    """README.md's text with the BibTeX block between the citation markers rendered from the cff."""
    out, n = CITE.subn(lambda m: m[1] + bibtex(cff) + m[2], readme)
    if n != 1:
        sys.exit("README.md: the citation markers are missing")
    return out


# ---------------------------------------------------------------- files

def badges():
    project = (ROOT / "pyproject.toml").read_text()
    pythons = re.findall(r"Programming Language :: Python :: (3\.\d+)", project)
    torch = re.search(r'"torch==([\d.]+)"', project).group(1)
    licence = re.search(r'^license = "([^"]+)"', project, re.M).group(1)
    tracks = json.loads((ROOT / "eval" / "bench" / "tracks.json").read_text())["tracks"]
    return {"licence": ("licence", licence), "python": ("python", " · ".join(pythons)),
            "pytorch": ("pytorch", torch), "benchmark": ("benchmark", f"{len(tracks)} tracks")}


def figures():
    out = {}
    for name, draw in [("banner", banner), ("hero", hero), ("pillars", pillars), ("mechanisms", mechanisms),
                       ("lineage", lineage_fig), ("compare", compare_fig), ("check", check_fig),
                       ("results", results_fig), ("flow", flow_fig)]:
        for variant, theme in [("light", LIGHT), ("dark", DARK), ("narrow", AUTO)]:
            out[f"{name}-{variant}.svg"] = draw(theme)
    for i, (key, label, family, mark) in enumerate(PLACES):
        for variant, theme in [("light", LIGHT), ("dark", DARK)]:
            out[f"capsule-{key}-{variant}.svg"] = capsule(key, label, family, mark, i == 0, theme, HOSTS.get(key))
    for key, (label, value) in badges().items():
        for variant, theme in [("light", LIGHT), ("dark", DARK)]:
            out[f"badge-{key}-{variant}.svg"] = badge(label, value, theme)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if a figure or the README citation is not current")
    args = parser.parse_args()
    stale = []
    for name, content in figures().items():
        path_ = ASSETS / name
        if args.check:
            if not path_.is_file() or path_.read_text() != content:
                stale.append(name)
        else:
            path_.write_text(content)
    readme = README.read_text()
    current = cited(readme, yaml.safe_load(CITATION.read_text()))
    if current != readme:
        if args.check:
            stale.append("README.md (citation)")
        else:
            README.write_text(current)
    if stale:
        sys.exit("not current, run python scripts/readme_figures.py: " + ", ".join(stale))


if __name__ == "__main__":
    main()
