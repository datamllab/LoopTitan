#!/usr/bin/env python3
"""Draw the README figures from the repository's data: python scripts/readme_figures.py [--check]

Writes SVG files to docs/assets/: each figure in a light and a dark variant (README.md switches them with
<picture> and prefers-color-scheme), and the large figures also in a narrow variant for phones, which
follows the reader's colour scheme by itself. Every value drawn comes from the repository: the frozen
recipes (recipes/), the lineage of scripts/lineage.py, the cost model (moeloop.comparison), the CPU
record docs/evidence/qwen3-cpu.json, the Phase 1 results docs/results/phase1.json, pyproject.toml and
eval/bench/tracks.json; the layout (positions, sizes, axis ranges) is chosen in this script. --check
writes nothing and exits 1 if a file differs from what the data gives; tests/setup/test_readme_figures.py
runs it. Text uses system fonts, because GitHub shows these files
as images, which cannot load web fonts.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from moeloop.comparison import compare_recipes  # noqa: E402
from moeloop.recipe_spec import RecipeSpec  # noqa: E402
import lineage  # noqa: E402

ASSETS = ROOT / "docs" / "assets"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

# The homepage palette; the dark one matches docs/assets/mark-dark.svg. Indigo marks the loop only.
LIGHT = dict(ink="#16181D", mid="#5B616E", soft="#9AA0AB", hair="#D5D8DE", line="#E3E5E9", faint="#EEF0F3",
             card="#FAFBFC", panel="#FFFFFF", acc="#3346D3", acc2="#7C89E4", acc3="#AAB3F0", tint="#F2F4FD",
             tint2="#E6E9FB", bar="#16181D", grey="#A9AFB9", on_acc="#FFFFFF", glow="#3346D3", spark="#FFFFFF", lit="#AAB3F0")
DARK = dict(ink="#E7EBF1", mid="#9BA3AF", soft="#5E6672", hair="#363D48", line="#272D36", faint="#1B2028",
            card="#11161D", panel="#0D1117", acc="#7486F2", acc2="#5866C9", acc3="#3B458B", tint="#151A2E",
            tint2="#1D2442", bar="#DCE0E7", grey="#59606C", on_acc="#0D1117", glow="#8E9CF7", spark="#FFFFFF", lit="#B9C2FB")
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


def arrow_head(x, y, dx, dy, stroke, sw=1.5, size=5):
    """A chevron at (x, y) pointing along (dx, dy)."""
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    a = (x - size * ux + size * 0.8 * uy, y - size * uy - size * 0.8 * ux)
    b = (x - size * ux - size * 0.8 * uy, y - size * uy + size * 0.8 * ux)
    return path(f"M{num(a[0])} {num(a[1])}L{num(x)} {num(y)}L{num(b[0])} {num(b[1])}", stroke, sw)


def card(x, y, w, h, t, r=20):
    return rect(x + .5, y + .5, w - 1, h - 1, r, t["card"], t["line"])


def panel(x, y, w, h, t, r=14):
    return rect(x + .5, y + .5, w - 1, h - 1, r, t["panel"], t["line"])


def check_icon(x, y, t, r=8):
    return (circle(x, y, r, t["tint2"]) +
            path(f"M{num(x - r * .42)} {num(y + r * .02)}L{num(x - r * .1)} {num(y + r * .34)}"
                 f"L{num(x + r * .45)} {num(y - r * .32)}", t["acc"], 1.6))


def pill(x, y, label, t, size=12, anchor="start"):
    """A tint chip with a check mark and a label; returns (elements, width)."""
    w = width(label, size) + 34
    x0 = x - w if anchor == "end" else x
    return rect(x0, y - 12, w, 24, 12, t["tint"]) + check_icon(x0 + 14, y, t, 6.5) + \
        text(x0 + 25, y + 4.2, label, size, t["acc"], 600), w


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
    narrow = t is AUTO
    w, h = (360, 272) if narrow else (880, 288)
    cx = w / 2
    mark = (ASSETS / ("mark-light.svg" if t is not DARK else "mark-dark.svg")).read_text()
    vw, vh = (float(v) for v in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', mark).groups())
    inner = re.search(r"<svg[^>]*>(.*)</svg>", mark, re.S).group(1)
    if narrow:  # the mark's two colours follow the scheme too
        inner = inner.replace('fill="#3346d3"', 'style="fill:var(--acc)"').replace('fill="#16181d"', 'style="fill:var(--ink)"')
    mh = 46 if narrow else 54
    s = mh / vh
    out = []
    # a quiet field of dots that fades out from the centre
    out.append('<defs><pattern id="dots" width="16" height="16" patternUnits="userSpaceOnUse">'
               f'<circle cx="2" cy="2" r="1" style="fill:{t["hair"]}"/></pattern>'
               '<radialGradient id="fade" cx="50%" cy="46%" r="58%"><stop offset="0" stop-color="#fff" stop-opacity=".9"/>'
               '<stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>'
               f'<mask id="m"><rect width="{w}" height="{h}" fill="url(#fade)"/></mask>'
               '<radialGradient id="halo" cx="50%" cy="50%" r="50%">'
               f'<stop offset="0" style="stop-color:{t["acc"]};stop-opacity:.10"/>'
               f'<stop offset="1" style="stop-color:{t["acc"]};stop-opacity:0"/></radialGradient></defs>')
    out.append(f'<rect width="{w}" height="{h}" fill="url(#dots)" mask="url(#m)"/>')
    top = 26 if narrow else 40
    out.append(f'<ellipse cx="{num(cx)}" cy="{num(top + mh / 2)}" rx="{110 if narrow else 160}" ry="{60 if narrow else 70}" fill="url(#halo)"/>')
    out.append(f'<g transform="translate({num(cx - vw * s / 2)} {top}) scale({s:.5f})">{inner}</g>')
    if narrow:
        out.append(text(cx, 126, "LoopTitan", 40, t["ink"], 650, "middle", ls=-1.2))
        out.append(text(cx, 168, "A common ground for", 21, t["ink"], 600, "middle", ls=-.3))
        out.append(text(cx, 195, "looped MoE research.", 21, t["ink"], 600, "middle", ls=-.3))
        out.append(text(cx, 226, "The design space of looped MoE in one", 14, t["mid"], 400, "middle"))
        out.append(text(cx, 246, "model, compared on equal terms.", 14, t["mid"], 400, "middle"))
    else:
        out.append(text(cx, 156, "LoopTitan", 56, t["ink"], 650, "middle", ls=-1.8))
        out.append(text(cx, 208, "A common ground for looped MoE research.", 26, t["ink"], 600, "middle", ls=-.5))
        out.append(text(cx, 242, "The design space of looped MoE in one model, compared on equal terms.", 17,
                        t["mid"], 400, "middle"))
    return svg(w, h, "LoopTitan", "A common ground for looped MoE research. The design space of looped MoE in "
               "one model, compared on equal terms.", t, "".join(out))


def badge(label, value, t):
    lw, vw = width(label, 12) + 18, width(value, 12, weight=600) + 18
    w, h = lw + vw, 22
    body = (rect(.5, .5, w - 1, h - 1, 11, t["panel"], t["hair"]) +
            f'<path d="M{num(lw)} .5H{num(w - 11)}A10.5 10.5 0 0 1 {num(w - 11)} {num(h - .5)}H{num(lw)}Z" '
            f'style="fill:{t["tint"]}"/>' +
            line(lw, 1, lw, h - 1, t["hair"], 1) +
            text(lw / 2, 15.2, label, 12, t["mid"], 400, "middle") +
            text(lw + vw / 2, 15.2, value, 12, t["acc"], 600, "middle"))
    return svg(round(w), h, f"{label}: {value}", f"{label}: {value}", t, body)


def institution(t, n):
    """A neutral slot for the logo of affiliation n; the authors replace the file's use in README.md."""
    w, h = 176, 52
    body = (rect(.75, .75, w - 1.5, h - 1.5, 14, t["card"], t["hair"], 1.5, dash="4 4") +
            circle(34, 26, 10, t["faint"]) + text(34, 30.3, str(n), 12, t["mid"], 600, "middle") +
            text(54, 31, "Institution", 14, t["mid"], 500))
    return svg(w, h, f"Institution {n}", f"A placeholder for the logo of institution {n}.", t, body)


def hero(t):
    """The Phase 1 loop: bars on a line, the middle layers run twice; a light passes the looped ones twice."""
    narrow = t is AUTO
    m, cells = layers("phase1/m3")
    pre, core, suf, visits = m["prefix_cells"], m["num_cells"], m["suffix_cells"], m["num_visits"]
    if narrow:
        w, h, bw, bh, gap, ggap, cy, r = 360, 238, 14, 72, 8, 18, 108, 5
    else:
        w, h, bw, bh, gap, ggap, cy, r = 880, 316, 30, 124, 18, 40, 166, 9
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
    a0 = top - (6 if narrow else 9)
    c = 46 if narrow else 78
    ax1, ax2 = last + bw / 2, first + bw / 2
    arc = f"M{num(ax1)} {num(a0)}C{num(ax1)} {num(a0 - c)} {num(ax2)} {num(a0 - c)} {num(ax2)} {num(a0)}"
    # timing: the light moves at one speed on the line and takes a fixed time on the arc
    T, fade, t_arc = 10.0, .35, 1.5
    d1, d2 = loop_r - xs, xe - loop_l
    v = (d1 + d2) / 6.2
    t1 = d1 / v
    t2 = t1 + t_arc
    t3 = t2 + d2 / v

    def pct(s):
        return f"{100 * s / T:.2f}%"
    dot = [(0, xs, 0), (fade, xs + v * fade, 1), (t1 - .05, loop_r - v * .05, 1), (t1 + .2, loop_r + v * .2, 0),
           (t2 - .25, loop_l - v * .25, 0), (t2, loop_l, 1), (t3 - .3, xe - v * .3, 1), (t3, xe, 0), (T, xe, 0)]
    css = ["@keyframes dot{" + "".join(f"{pct(s)}{{transform:translate({num(px)}px,0);opacity:{o}}}" for s, px, o in dot) + "}",
           f".dot{{animation:dot {num(T)}s linear infinite}}",
           f"@keyframes arc{{0%,{pct(t1 - .1)}{{stroke-dashoffset:16;opacity:0}}{pct(t1)}{{stroke-dashoffset:16;opacity:1}}"
           f"{pct(t2)}{{stroke-dashoffset:-100;opacity:1}}{pct(t2 + .1)},100%{{stroke-dashoffset:-100;opacity:0}}}}",
           f".comet{{opacity:0;stroke-dasharray:16 200;animation:arc {num(T)}s linear infinite}}",
           "@media (prefers-reduced-motion: reduce){.dot,.comet,.lit{animation:none!important}}"]
    out = ['<defs><radialGradient id="glow" cx="50%" cy="50%" r="50%">'
           f'<stop offset="0" style="stop-color:{t["acc"]};stop-opacity:.13"/>'
           f'<stop offset="1" style="stop-color:{t["acc"]};stop-opacity:0"/></radialGradient>'
           '<radialGradient id="light" cx="50%" cy="50%" r="50%">'
           f'<stop offset="0" style="stop-color:{t["glow"]};stop-opacity:.55"/>'
           f'<stop offset=".45" style="stop-color:{t["glow"]};stop-opacity:.16"/>'
           f'<stop offset="1" style="stop-color:{t["glow"]};stop-opacity:0"/></radialGradient></defs>']
    out.append(f'<ellipse cx="{num((loop_l + loop_r) / 2)}" cy="{num(cy)}" rx="{num((loop_r - loop_l) * .78)}" '
               f'ry="{num(bh * .95)}" fill="url(#glow)"/>')
    out.append(line(xs, cy, xe, cy, t["hair"], 2 if not narrow else 1.5))
    out.append(arrow_head(xe, cy, 1, 0, t["hair"], 2 if not narrow else 1.5, 7 if not narrow else 5))
    for i, (bx, loop) in enumerate(bars):
        out.append(rect(bx, top, bw, bh, r, t["acc"] if loop else t["bar"]))
        passes = [(bx + bw / 2 - xs) / v] if bx < loop_r else []
        if bx > loop_l:
            passes.append(t2 + (bx + bw / 2 - loop_l) / v)
        frames = ["0%{opacity:0}"] + [f"{pct(max(p - .3, .01))}{{opacity:0}}{pct(p)}{{opacity:1}}{pct(p + .7)}{{opacity:0}}"
                                      for p in passes] + ["100%{opacity:0}"]
        css.append(f"@keyframes b{i}{{{''.join(frames)}}}.b{i}{{animation:b{i} {num(T)}s linear infinite}}")
        out.append(rect(bx, top, bw, bh, r, t["lit"] if loop else t["acc"], op=0, cls=f"lit b{i}"))
    out.append(path(arc, t["acc"], 2.5 if not narrow else 2))
    out.append(path(arc, t["glow"], 9 if not narrow else 6, cls="comet", extra=' pathLength="100" stroke-opacity=".22"'))
    out.append(path(arc, t["spark"], 1.6 if not narrow else 1.2, cls="comet", extra=' pathLength="100"'))
    out.append(arrow_head(ax2, a0, 0, 1, t["acc"], 2.5 if not narrow else 2, 8 if not narrow else 6))
    apex = a0 - .75 * c
    out.append(text((ax1 + ax2) / 2, apex - (9 if narrow else 13), f"×{visits}", 20 if narrow else 26, t["acc"], 650, "middle"))
    out.append(f'<g class="dot" style="opacity:0"><circle cx="0" cy="{num(cy)}" r="{24 if not narrow else 15}" fill="url(#light)"/>'
               + circle(0, cy, 5.5 if not narrow else 3.5, t["spark"], t["glow"], 2 if not narrow else 1.5) + "</g>")
    # brackets and labels under the three groups
    yb = cy + bh / 2 + (16 if narrow else 22)
    starts = [bars[0][0], first, bars[pre + core][0]]
    ends = [bars[pre - 1][0] + bw, last + bw, bars[-1][0] + bw]
    size = 12.5 if narrow else 15
    labels = [f"{pre} layers", f"{core} layers, one set of weights, run {'twice' if visits == 2 else f'{visits} times'}", f"{suf} layers"]
    if narrow:
        labels[1] = f"{core} layers, run {'twice' if visits == 2 else f'{visits} times'}"
    for i, (a, b) in enumerate(zip(starts, ends)):
        col = t["acc"] if i == 1 else t["hair"]
        out.append(path(f"M{num(a)} {num(yb - 5)}V{num(yb)}H{num(b)}V{num(yb - 5)}", col, 1.3))
        out.append(text((a + b) / 2, yb + (19 if narrow else 25), labels[i], size, t["acc"] if i == 1 else t["mid"],
                        500 if i == 1 else 400, "middle"))
    stored, run = pre + core + suf, pre + core * visits + suf
    if narrow:
        out.append(text(w / 2, yb + 37, "one set of weights", size, t["acc"], 500, "middle"))
    out.append(text(w / 2, h - (14 if narrow else 18), f"{stored} layers stored  ·  {run} layers run", 12 if narrow else 13,
                    t["mid"], 400, "middle"))
    desc = (f"{pre} outer layers, {core} middle layers that run {visits} times with one set of weights, "
            f"{suf} outer layers: {stored} layers stored, {run} run. A light passes the middle layers twice.")
    return svg(w, h, "The Phase 1 loop", desc, t, "".join(out), "".join(css))


# ---- pillars

def pillar_designs(t):
    """Many designs, one model (local box 256 x 140): three looped designs, drawn as in the lineage figure."""
    out = []
    for node, y in zip(["phase1/m3", "methods/huginn", "methods/ouro"], [31, 75, 118]):
        g, gw = glyph(node, 24, y, t, .56, label=False)
        out.append(g)
        visits = recipe(node)["model"]["num_visits"]
        out.append(text(24 + gw + 10, y + 4.3, f"×{visits}", 12, t["acc"], 600, mono=True))
        x = 24 + gw + 34
        out.append(path(f"M{num(x)} {y}C{num(x + 30)} {y} 120 75 146 75", t["hair"], 1.2))
    out.append(rect(146, 58, 96, 34, 10, t["tint"], t["acc3"], 1))
    out.append(text(194, 79.5, "ModelConfig", 12, t["acc"], 500, "middle", mono=True))
    return "".join(out)


def pillar_budgets(t, matched):
    out = []
    for i, (key, label) in enumerate(BUDGETS):
        y = 34 + i * 30
        rel = matched[key]["relative_difference"]
        tol = matched[key]["relative_tolerance"]
        x0, x1 = 100, 186
        mid = (x0 + x1) / 2
        out.append(text(14, y + 4.2, label.replace("Training ", ""), 12, t["mid"]))
        out.append(rect(x0, y - 4, x1 - x0, 8, 4, t["tint2"]))
        out.append(line(mid, y - 7, mid, y + 7, t["soft"], 1))
        out.append(circle(mid + rel / tol * (x1 - x0) / 2, y, 4.6, t["acc"], t["panel"], 2))
        out.append(text(244, y + 4.2, f"{100 * rel:+.2f}%".replace("-", "−"), 12, t["ink"], 500, "end", mono=True))
    tol = within(matched)[2]
    for x, s in [(100, f"−{pct(tol)}"), (143, "0"), (186, f"+{pct(tol)}")]:
        out.append(text(x, 126, s, 12, t["mid"], 400, "middle", mono=True))
    return "".join(out)


def pillar_checks(t, rows):
    out = []
    heads = ["logits", "loss", "grad", "step", "cache"]
    for j, head in enumerate(heads):
        out.append(text(22 + j * 44 + 18, 30, head, 12, t["mid"], 400, "middle", mono=True))
    for i, (_, cells) in enumerate(rows):
        y = 48 + i * 15.5
        for j, (value, tol) in enumerate(cells):
            x = 22 + j * 44
            out.append(rect(x, y - 2, 36, 4, 2, t["faint"]))
            if value > 0:
                out.append(rect(x, y - 2, max(3, 36 * value / tol), 4, 2, t["acc"]))
    return "".join(out)


def pillars(t):
    narrow = t is AUTO
    matched = budgets("phase1/m0", "phase1/m3")
    record, rows = checks()
    inside = sum(v <= tol for _, cells in rows for v, tol in cells)
    n = sum(len(cells) for _, cells in rows)
    tiles = [(pillar_designs(t), f"{len(lineage.lineage())} configurations, one model",
              f"{len(MECHANISMS)} mechanisms, each a field"),
             (pillar_budgets(t, matched), "{} of {} budgets within ±{}".format(*within(matched)[:2], pct(within(matched)[2])),
              "The loop against the ordinary MoE"),
             (pillar_checks(t, rows), f"{inside} of {n} within tolerance", "Against Transformers Qwen3-MoE")]
    cw, ch, gap = (360, 222, 14) if narrow else (280, 224, 20)
    w = cw if narrow else 3 * cw + 2 * gap
    h = 3 * ch + 2 * gap if narrow else ch
    out = []
    for i, (art, head, sub) in enumerate(tiles):
        x, y = (0, i * (ch + gap)) if narrow else (i * (cw + gap), 0)
        out.append(card(x, y, cw, ch, t))
        out.append(panel(x + 12, y + 12, cw - 24, 140, t))
        out.append(f'<g transform="translate({num(x + (cw - 256) / 2)} {y + 12})">{art}</g>')
        out.append(text(x + 20, y + 182, head, 15, t["ink"], 600))
        out.append(text(x + 20, y + 204, sub, 13, t["mid"]))
    desc = (f"Three tiles. {tiles[0][1]}: {tiles[0][2]}. {tiles[1][1]}: parameters, training FLOPs and KV cache "
            f"of the loop against the ordinary MoE. {tiles[2][1]}: six configurations, five quantities, against "
            "Transformers Qwen3-MoE.")
    return svg(w, h, "LoopTitan at a glance", desc, t, "".join(out))


# ---- mechanisms (local box 256 x 116)

def bars_row(t, x, cy, spec, w=10, h=40, gap=8, ggap=18):
    """spec: [(count, colour)]; returns (elements, [(left, right) of each group], [(x) of each bar])."""
    out, spans, xs = [], [], []
    for count, colour in spec:
        left = x
        for i in range(count):
            out.append(rect(x, cy - h / 2, w, h, 3.5, colour))
            xs.append(x)
            x += w + (gap if i < count - 1 else 0)
        spans.append((left, x))
        x += ggap
    return "".join(out), spans, xs


def loop_arc(t, x1, x2, y, c, label=None, sw=1.6):
    out = path(f"M{num(x1)} {num(y)}C{num(x1)} {num(y - c)} {num(x2)} {num(y - c)} {num(x2)} {num(y)}", t["acc"], sw)
    out += arrow_head(x2, y, 0, 1, t["acc"], sw, 4.5)
    if label:
        out += text((x1 + x2) / 2, y - .75 * c - 6, label, 12, t["acc"], 600, "middle")
    return out


def mech_loop(t):
    m = recipe("phase1/m3")["model"]
    body, spans, xs = bars_row(t, 59, 62, [(2, t["bar"]), (3, t["acc"]), (2, t["bar"])])
    out = [line(30, 62, 226, 62, t["hair"], 1.2), body,
           loop_arc(t, xs[4] + 5, xs[2] + 5, 38, 22, f"×{m['num_visits']}")]
    for (a, b), s, col in zip(spans, ["prelude", "loop", "coda"], [t["mid"], t["acc"], t["mid"]]):
        out.append(text((a + b) / 2, 102, s, 12, col, 500 if s == "loop" else 400, "middle", mono=True))
    return "".join(out)


def mech_experts(t):
    m, cells = layers("shape/r")
    out = [line(18, 56, 238, 56, t["hair"], 1.2)]
    top = max(p for _, p in cells)
    x, spans = 25, {}
    for i, (region, p) in enumerate(cells):
        h = 50 * p / top
        out.append(rect(x, 56 - h / 2, 7, h, 2.5, t["bar"]))
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
        out.append(circle(22, y, 9, t["faint"]))
        out.append(text(22, y + 4.2, str(i + 1), 12, t["mid"], 500, "middle", mono=True))
        out.append(rect(38, y - 13, 74, 26, 8, t["tint"]))
        out.append(text(75, y + 4.2, "attention", 12, t["acc"], 500, "middle", mono=True))
        out.append(rect(118, y - 13, 56, 26, 8, t["tint"]))
        out.append(text(146, y + 4.2, "router", 12, t["acc"], 500, "middle", mono=True))
    out.append(rect(180, 23, 62, 60, 10, t["acc"]))
    out.append(text(211, 57.2, "experts", 12, t["on_acc"], 600, "middle", mono=True))
    out.append(text(106, 106, "owned per visit", 12, t["mid"], 400, "middle", mono=True))
    out.append(text(211, 106, "shared", 12, t["mid"], 400, "middle", mono=True))
    return "".join(out)


def mech_residual(t):
    scale = recipe("phase1/m3")["model"]["core"]["residual_scale"]
    out = [rect(44, 16, 72, 24, 12, t["bar"]), text(80, 32.2, "layer", 12, t["panel"], 500, "middle", mono=True),
           rect(132, 16, 104, 24, 12, t["acc"]), text(184, 32.2, "looped layer", 12, t["on_acc"], 500, "middle", mono=True),
           line(80, 46, 80, 74, t["bar"], 2.6), arrow_head(80, 76, 0, 1, t["bar"], 2.6, 6),
           text(90, 66, "×1", 12, t["ink"], 500, mono=True),
           line(184, 46, 184, 74, t["acc"], 1.4), arrow_head(184, 76, 0, 1, t["acc"], 1.4, 6),
           text(194, 66, f"×{num(scale)}", 12, t["acc"], 500, mono=True),
           rect(28, 82, 200, 6, 3, t["line"]),
           text(128, 106, "the stream", 12, t["mid"], 400, "middle", mono=True)]
    return "".join(out)


def mech_input(t):
    m = recipe("methods/huginn")["model"]
    out = [line(22, 50, 234, 50, t["hair"], 1.2), rect(36, 32, 10, 36, 3.5, t["bar"])]
    out.append(circle(74, 50, 7.5, t["panel"], t["acc"], 1.5))
    out.append(path("M70 50H78M74 46V54", t["acc"], 1.5))
    for x in (98, 118):
        out.append(rect(x, 32, 10, 36, 3.5, t["acc"]))
    out.append(rect(156, 32, 10, 36, 3.5, t["bar"]))
    out.append(path("M123 28C123 6 74 6 74 38", t["acc"], 1.6) + arrow_head(74, 40, 0, 1, t["acc"], 1.6, 4.5))
    out.append(text(128, 16, f"×{m['num_visits']}", 12, t["acc"], 600))
    out.append(line(74, 92, 74, 62, t["acc"], 1.4, dash="3 3") + arrow_head(74, 60, 0, -1, t["acc"], 1.4, 4.5))
    out.append(text(84, 96, "input, at every visit", 12, t["acc"], 500, mono=True))
    return "".join(out)


def mech_state(t):
    out = [circle(40, 46, 18, t["panel"], t["hair"], 1.2)]
    for dx, dy in [(-8, -6), (5, -9), (9, 3), (-3, 4), (-10, 8), (3, 12), (11, -3)]:
        out.append(circle(40 + dx * .9, 46 + dy * .9, 1.8, t["acc"]))
    out.append(line(64, 46, 92, 46, t["hair"], 1.3) + arrow_head(94, 46, 1, 0, t["hair"], 1.3, 4))
    for x in (104, 124):
        out.append(rect(x, 28, 10, 36, 3.5, t["acc"]))
    out.append(loop_arc(t, 129, 109, 24, 18))
    out.append(line(144, 46, 172, 46, t["hair"], 1.3) + arrow_head(174, 46, 1, 0, t["hair"], 1.3, 4))
    out.append(circle(200, 46, 18, t["panel"], t["acc"], 1.4))
    out.append(f'<path d="M200 28A18 18 0 0 0 200 64Z" style="fill:{t["acc3"]}"/>')
    out.append(text(40, 88, "start", 12, t["mid"], 400, "middle", mono=True))
    out.append(text(40, 104, "random", 12, t["acc"], 500, "middle", mono=True))
    out.append(text(200, 88, "update", 12, t["mid"], 400, "middle", mono=True))
    out.append(text(200, 104, "average", 12, t["acc"], 500, "middle", mono=True))
    return "".join(out)


def mech_depth(t):
    out = [line(26, 40, 230, 40, t["hair"], 1.2)]
    for i, x in enumerate([56, 98, 140, 182]):
        out.append(text(x + 6, 15, str(i + 1), 12, t["mid"], 400, "middle", mono=True))
        out.append(rect(x, 23, 12, 34, 3.5, t["acc"]))
        out.append(line(x + 6, 61, x + 6, 74, t["acc3"], 1.3, dash="2 2") + arrow_head(x + 6, 77, 0, 1, t["acc3"], 1.3, 3.5))
    out.append(rect(46, 80, 164, 6, 3, t["tint2"]))
    out.append(text(128, 106, "exit after any visit", 12, t["mid"], 400, "middle", mono=True))
    return "".join(out)


def mech_lora(t):
    out = [rect(38, 22, 24, 58, 5, t["acc"]), text(50, 104, "shared", 12, t["mid"], 400, "middle", mono=True),
           text(88, 56, "+", 14, t["mid"], 400, "middle")]
    for i, x in enumerate([116, 184]):
        out.append(rect(x, 22, 24, 58, 5, t["acc3"], op=.55))
        out.append(rect(x + 30, 22, 6, 58, 3, t["acc"]))
        out.append(text(x + 33, 14, "Δ", 12, t["acc"], 600, "middle"))
        out.append(text(x + 18, 104, f"visit {i + 1}", 12, t["mid"], 400, "middle", mono=True))
    return "".join(out)


def mech_tying(t):
    m = recipe("methods/expert-tying")["model"]
    k = m["num_visits"]
    xs = [62 + i * 32 for i in range(k)]
    out = []
    for x in xs:
        out.append(rect(x, 14, 12, 32, 3.5, t["bar"]))
        out.append(line(x + 6, 46, x + 6, 68, t["acc2"], 1.3))
    out.append(text(xs[-1] + 26, 34.5, f"{k} layers", 12, t["mid"], 400, mono=True))
    out.append(rect(46, 68, 140, 26, 9, t["acc"]))
    out.append(text(116, 85.2, "one expert bank", 12, t["on_acc"], 600, "middle", mono=True))
    return "".join(out)


MECH_ART = [mech_loop, mech_experts, mech_shares, mech_residual, mech_input, mech_state, mech_depth, mech_lora, mech_tying]


def mechanisms(t):
    narrow = t is AUTO
    cw, ch, gap = (360, 196, 12) if narrow else (280, 204, 20)
    w = cw if narrow else 3 * cw + 2 * gap
    h = 9 * ch + 8 * gap if narrow else 3 * ch + 2 * gap
    out = []
    for i, ((title, fields), art) in enumerate(zip(MECHANISMS, MECH_ART)):
        x, y = (0, i * (ch + gap)) if narrow else ((i % 3) * (cw + gap), (i // 3) * (ch + gap))
        out.append(card(x, y, cw, ch, t))
        out.append(panel(x + 12, y + 12, cw - 24, 116, t))
        out.append(f'<g transform="translate({num(x + (cw - 256) / 2)} {y + 12})">{art(t)}</g>')
        out.append(text(x + 20, y + 152, title, 14, t["ink"], 600))
        for j, f in enumerate(fields):
            out.append(text(x + 20, y + 173 + j * 17, f, 12, t["mid"], 400, mono=True))
    desc = "Nine mechanisms, each a field of one model: " + "; ".join(title for title, _ in MECHANISMS) + "."
    return svg(w, h, "Nine mechanisms", desc, t, "".join(out))


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
    out = []
    if narrow:
        w = 360
        out.append(text(16, 30, "One change per step", 13, t["mid"], 600))
        y = 66
        out.append(line(24, y, 24, y + 60 * (len(CHAIN) - 1), t["line"], 1.2))
        for i, (name, fmt, node) in enumerate(CHAIN):
            last = i == len(CHAIN) - 1
            out.append(circle(24, y, 4, t["acc"] if last else t["panel"], t["acc"] if last else t["hair"], 1.4))
            g, gw = glyph(node, 44, y + 2, t, .62)
            out.append(g)
            out.append(text(160, y - 1, name, 13, t["acc"] if last else t["ink"], 600))
            out.append(text(160, y + 17, chain_detail(fmt, node), 12, t["mid"], 400, mono=True))
            y += 60
        y += 6
        for title, rows in [("From the ordinary MoE", left_rows), ("From the loop", right_rows)]:
            out.append(line(16, y - 18.5, w - 16, y - 18.5, t["line"], 1))
            out.append(text(16, y + 6, title, 13, t["mid"], 600))
            y += 40
            for node, depth in rows:
                x0 = 24 + depth * 22
                if depth:
                    out.append(path(f"M{x0 - 12} {y - 30}V{y - 6}Q{x0 - 12} {y} {x0 - 6} {y}", t["hair"], 1.2))
                out.append(design_row(t, node, x0, y, 150))
                y += 44
            y += 16
        h = y - 24
    else:
        w = 880
        out.append("")
        out.append(text(28, 40, "One change per step", 13, t["mid"], 600))
        out.append(text(w - 28, 40, "Phase 1, every step at equal budgets", 12, t["mid"], 400, "end"))
        cxs = [86 + i * 141.6 for i in range(len(CHAIN))]
        for i, ((name, fmt, node), cx) in enumerate(zip(CHAIN, cxs)):
            _, gw = glyph(node, 0, 0, t, .78)
            g, _ = glyph(node, cx - gw / 2, 104, t, .78)
            out.append(g)
            col = t["acc"] if i == len(CHAIN) - 1 else t["ink"]
            out.append(text(cx, 152, name, 13, col, 600, "middle"))
            out.append(text(cx, 171, chain_detail(fmt, node), 12, t["mid"], 400, "middle", mono=True))
            if i:
                out.append(line(cxs[i - 1] + 60, 104, cx - 62, 104, t["hair"], 1.2) +
                           arrow_head(cx - 61, 104, 1, 0, t["hair"], 1.2, 4))
        top, pitch = 268, 44
        spine = cxs[0]
        rows_y = [top + i * pitch for i in range(len(left_rows))]
        out.append(circle(spine, 192, 3.5, t["panel"], t["hair"], 1.4))
        out.append(path(f"M{num(spine)} 196V{num(rows_y[-1] - 10)}Q{num(spine)} {num(rows_y[-1])} {num(spine + 10)} {num(rows_y[-1])}",
                        t["hair"], 1.2))
        out.append(text(spine + 18, 238, "Published designs from the ordinary MoE", 12, t["mid"], 600))
        prev = {}
        for (node, depth), y in zip(left_rows, rows_y):
            x0 = spine + 28 + depth * 30
            if depth:
                py = prev[0]
                out.append(path(f"M{num(spine + 40)} {num(py + 14)}V{num(y - 8)}Q{num(spine + 40)} {num(y)} {num(spine + 48)} {num(y)}H{num(x0 - 6)}",
                                t["hair"], 1.2))
            else:
                out.append(line(spine, y, x0 - 6, y, t["hair"], 1.2))
            prev[depth] = y
            out.append(design_row(t, node, x0, y, spine + 136))
        rspine = 486
        ry = [top + i * pitch for i in range(len(right_rows))]
        out.append(circle(cxs[-1], 192, 3.5, t["panel"], t["acc3"], 1.4))
        out.append(path(f"M{num(cxs[-1])} 196V{num(204)}Q{num(cxs[-1])} 212 {num(cxs[-1] - 8)} 212H{num(rspine + 8)}"
                        f"Q{rspine} 212 {rspine} 220V{num(ry[-1] - 10)}Q{rspine} {num(ry[-1])} {rspine + 10} {num(ry[-1])}",
                        t["acc3"], 1.2))
        out.append(text(rspine + 18, 240, "Published designs from the loop", 12, t["mid"], 600))
        for (node, _), y in zip(right_rows, ry):
            out.append(line(rspine, y, rspine + 22, y, t["acc3"], 1.2))
            out.append(design_row(t, node, rspine + 28, y, rspine + 136))
        # legend
        lx, ly, lw = rspine, rows_y[-1] - 76, w - 28 - rspine
        out.append(rect(lx + .5, ly + .5, lw - 1, 92, 14, t["panel"], t["line"]))
        out.append(rect(lx + 20, ly + 23, 6, 18, 2, t["bar"]) + rect(lx + 29, ly + 27, 4, 10, 2, t["bar"]))
        out.append(text(lx + 48, ly + 36.5, "a stored layer, area = its expert parameters", 12, t["mid"]))
        out.append(path(f"M{lx + 32} {ly + 58}C{lx + 32} {ly + 50} {lx + 22} {ly + 50} {lx + 22} {ly + 58}", t["acc"], 1.2))
        out.append(rect(lx + 19, ly + 60, 6, 16, 2, t["acc"]) + rect(lx + 29, ly + 60, 6, 16, 2, t["acc"]))
        out.append(text(lx + 48, ly + 72.5, "looped layers, ×n visits per token", 12, t["mid"]))
        h = rows_y[-1] + 36
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
    out = []
    models = []
    for node, name in [(ref, "The loop"), (cand, "Its untied twin")]:
        m, cells = layers(node)
        models.append((node, name, len(cells), m["prefix_cells"] + m["num_cells"] * m["num_visits"] + m["suffix_cells"]))
    if narrow:
        w = 360
        y = 12
        for node, name, stored, run in models:
            out.append(card(0, y, w, 92, t))
            out.append(text(18, y + 30, name, 14, t["acc"] if node == ref else t["ink"], 600))
            out.append(text(18, y + 50, f"{stored} stored · {run} run", 12, t["mid"], 400, mono=True))
            _, gw = glyph(node, 0, 0, t, .62)
            g, _ = glyph(node, w - 22 - gw, y + 54, t, .62)
            out.append(g)
            y += 104
        out.append(card(0, y, w, 244, t))
        out.append(text(18, y + 32, "Budgets", 14, t["ink"], 600))
        p, _ = pill(w - 16, y + 27, f"{inside} of {n} within ±{pct(tol)}", t, anchor="end")
        out.append(p)
        y += 64
        for key, label in BUDGETS:
            out += budget_row(t, matched[key], key, label, 18, y, w - 36, narrow=True)
            y += 58
        h = y + 8
    else:
        w, h = 880, 260
        out.append(card(0, 0, w, h, t))
        for i, (node, name, stored, run) in enumerate(models):
            y = 20 + i * 116
            out.append(panel(16, y, 384, 104, t))
            out.append(text(36, y + 32, name, 14, t["acc"] if node == ref else t["ink"], 600))
            out.append(text(36, y + 52, f"{stored} layers stored · {run} run", 12, t["mid"], 400, mono=True))
            _, gw = glyph(node, 0, 0, t, .86)
            g, _ = glyph(node, 384 - gw, y + 68, t, .86)
            out.append(g)
        out.append(text(436, 50, "Budgets of the twin against the loop", 14, t["ink"], 600))
        p, _ = pill(w - 24, 45, f"{inside} of {n} within ±{pct(tol)}", t, anchor="end")
        out.append(p)
        for i, (key, label) in enumerate(BUDGETS):
            out += budget_row(t, matched[key], key, label, 436, 96 + i * 54, w - 24 - 436)
    desc = (f"The loop ({models[0][2]} layers stored, {models[0][3]} run) and its untied twin ({models[1][2]} stored, "
            f"{models[1][3]} run). " + "; ".join(f"{label}: {fmt_budget(k, matched[k]['reference'])} against "
                                              f"{fmt_budget(k, matched[k]['candidate'])}, "
                                              f"{100 * matched[k]['relative_difference']:+.2f}%" for k, label in BUDGETS) +
            f". {inside} of {n} within ±{pct(tol)}.")
    return svg(w, int(h), "The loop and its untied twin", desc, t, "".join(out))


def budget_row(t, entry, key, label, x, y, span, narrow=False):
    rel, tol = entry["relative_difference"], entry["relative_tolerance"]
    out = [text(x, y, label, 13, t["ink"], 600),
           text(x + span, y, f"{100 * rel:+.2f}%".replace("-", "−"), 13, t["acc"], 600, "end", mono=True)]
    values = f"{fmt_budget(key, entry['reference'])} → {fmt_budget(key, entry['candidate'])}"
    if not narrow:
        out.append(text(x + 128, y, values, 12, t["mid"], 400, mono=True))
    ty = y + 18
    out.append(rect(x, ty - 4, span, 8, 4, t["tint2"]))
    mid = x + span / 2
    out.append(line(mid, ty - 8, mid, ty + 8, t["soft"], 1))
    out.append(circle(mid + rel / tol * span / 2, ty, 5.5, t["acc"], t["panel"], 2))
    if narrow:
        out.append(text(x, ty + 22, values, 12, t["mid"], 400, mono=True))
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
    inside = sum(v <= tol for _, cells in rows for v, tol in cells)
    n = sum(len(cells) for _, cells in rows)
    updates = {case["updates"] for case in record["cases"].values()}
    setup = (f"{'/'.join(map(str, sorted(updates)))} {record['optimizer']['name']} updates",
             f"{record['dtype']} on {record['device'].upper()}")
    out = []
    if narrow:
        w = 360
        lab, cw0, cw = 128, 128, 46
        h = 60 + 34 + len(rows) * 40 + 70
        out.append(card(0, 0, w, h, t))
        p, _ = pill(16, 32, f"{inside} of {n} within tolerance", t)
        out.append(p)
        heads = ["Logits", "Loss", "Grad.", "Step", "Cache"]
        for j, head in enumerate(heads):
            out.append(text(cw0 + j * cw + 19, 80, head, 12, t["mid"], 500, "middle"))
        for i, (label, cells) in enumerate(rows):
            y = 110 + i * 40
            out.append(line(16, y - 20.5, w - 16, y - 20.5, t["line"], 1))
            out.append(text(16, y + 4.5, label, 12.5, t["ink"], 500))
            for j, (v, tol) in enumerate(cells):
                x = cw0 + j * cw
                out.append(rect(x, y - 2.5, 38, 5, 2.5, t["faint"]))
                if v > 0:
                    out.append(rect(x, y - 2.5, max(3, 38 * v / tol), 5, 2.5, t["acc"]))
        out.append(text(16, h - 42, f"A bar fills to its tolerance. {setup[0]},", 12, t["mid"]))
        out.append(text(16, h - 24, f"{setup[1]}, Transformers {record['transformers_version']}.", 12, t["mid"]))
    else:
        w = 880
        h = 140 + (len(rows) - 1) * 50 + 94
        out.append(card(0, 0, w, h, t))
        out.append(text(28, 42, "Largest difference from Transformers Qwen3-MoE", 14, t["ink"], 600))
        p, _ = pill(w - 24, 37, f"{inside} of {n} within tolerance", t, anchor="end")
        out.append(p)
        x0, cw = 196, 134
        for j, (_, head, tol) in enumerate(QUANTITIES):
            x = x0 + j * cw
            out.append(text(x, 86, head, 13, t["ink"], 600))
            out.append(text(x, 104, "tol " + fmt_err(record["tolerances"][tol]["atol"]), 12, t["mid"], 400, mono=True))
        for i, (label, cells) in enumerate(rows):
            y = 140 + i * 50
            out.append(line(28, y - 22.5, w - 28, y - 22.5, t["line"], 1))
            out.append(text(28, y + 10.5, label, 13, t["ink"], 500))
            for j, (v, tol) in enumerate(cells):
                x = x0 + j * cw
                out.append(text(x, y + 4, fmt_err(v), 12, t["ink"] if v else t["mid"], 400, mono=True))
                out.append(rect(x, y + 12, 104, 4, 2, t["faint"]))
                if v > 0:
                    out.append(rect(x, y + 12, max(3, 104 * v / tol), 4, 2, t["acc"]))
        out.append(line(28, h - 46.5, w - 28, h - 46.5, t["line"], 1))
        out.append(text(28, h - 20, f"Largest absolute difference after {setup[0]}, {setup[1]}, Transformers "
                                    f"{record['transformers_version']}. A bar fills to its tolerance.", 12, t["mid"]))
    desc = f"{inside} of {n} values within tolerance: " + "; ".join(
        f"{label}: " + ", ".join(f"{q} {fmt_err(v)}" for (_, q, _), (v, _) in zip(QUANTITIES, cells)) for label, cells in rows) + "."
    return svg(w, int(h), "Check against an independent implementation", desc, t, "".join(out))


# ---- results

def waterfall(t, deltas, x0, x1, top, pitch, lo=-10, hi=2):
    """One optimizer's chain as a waterfall: each step a bar from the previous total to the new one."""
    n = len(deltas) + 1
    loop_row = len(deltas) - 1

    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)
    out = []
    bottom = top + (n - 1) * pitch + 16
    for v in range(lo, hi + 1, 2):
        out.append(line(X(v), top - 18, X(v), bottom, t["hair"] if v == 0 else t["line"], 1))
        out.append(text(X(v), bottom + 18, f"{v:+d}".replace("-", "−").replace("+0", "0"), 12, t["mid"], 400, "middle", mono=True))
    for i in range(n):
        y = top + i * pitch
        if i == 0:
            out.append(rect(X(0) - 1.5, y - 5, 3, 10, 1.5, t["bar"]))
            continue
        a, b = (0, deltas[-1]) if i == n - 1 else (deltas[i - 1], deltas[i])
        loop = i == loop_row
        out.append(rect(min(X(a), X(b)), y - 5, max(abs(X(b) - X(a)), 2.5), 10, 2.5, t["acc"] if loop else t["bar"]))
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
    out = [""]

    def row_labels(x, top, pitch, size):
        o = []
        for i, label in enumerate(labels):
            y = top + i * pitch
            loop, total = i == loop_row, i == n - 1
            o.append(text(x, y + 4.5, label, size, t["acc"] if loop else t["ink"], 600 if loop or total else 400))
        return o

    def band(x, w_, top, pitch, i):
        return rect(x, top + i * pitch - pitch / 2 + 2, w_, pitch - 4, 8, t["tint"])
    if narrow:
        w, pitch = 360, 34
        y = 30
        out.append(text(16, y, "Test mBPB against the ordinary MoE", 13.5, t["ink"], 600))
        y += 20
        for opt, name in [("adamw", "AdamW"), ("muon", "Muon")]:
            top = y + 52
            out.append(text(16, y + 24, name, 13, t["mid"], 600))
            out.append(band(10, w - 20, top, pitch, loop_row))
            out += row_labels(18, top, pitch, 12)
            out += waterfall(t, deltas[opt], 176, 340, top, pitch)
            y = top + (n - 1) * pitch + 50
        out.append(text(16, y + 6, f"← lower is better. {tokens} tokens, equal budgets.", 12, t["mid"]))
        h = y + 26
    else:
        w, pitch, top = 880, 38, 112
        h = top + (n - 1) * pitch + 86
        out.append(text(28, 42, "Test mBPB against the ordinary MoE, one change at a time", 14, t["ink"], 600))
        out.append(text(w - 28, 42, "← lower is better", 12, t["mid"], 400, "end"))
        out.append(band(14, w - 28, top, pitch, loop_row))
        out += row_labels(28, top, pitch, 13)
        for (opt, name), (x0, x1) in zip([("adamw", "AdamW"), ("muon", "Muon")], [(250, 518), (586, 852)]):
            out.append(text(x0, 80, name, 13, t["mid"], 600))
            out += waterfall(t, deltas[opt], x0, x1, top, pitch)
        out.append(text(28, h - 18, f"Test bits per byte after {tokens} tokens; every step at equal parameters, FLOPs "
                                    "and KV cache.", 12, t["mid"]))
    out[0] = card(0, 0, w, h, t)
    desc = ("A waterfall of the one-change chain under AdamW and Muon, test mBPB against the ordinary MoE: " +
            "; ".join(f"{name}: AdamW {deltas['adamw'][i]:+.1f}, Muon {deltas['muon'][i]:+.1f}"
                      for i, (name, _, _) in enumerate(CHAIN)) + ".")
    return svg(w, int(h), "Phase 1: one change at a time", desc, t, "".join(out))


# ---- how research runs here

def icon(kind, x, y, col):
    """16 px line icons."""
    p = {"declare": f"M{x+3} {y+1}H{x+10}L{x+14} {y+5}V{y+15}H{x+3}ZM{x+6} {y+8}H{x+11}M{x+6} {y+11}H{x+11}",
         "run": f"M{x+2} {y+14}V{y+9}M{x+6} {y+14}V{y+4}M{x+10} {y+14}V{y+7}M{x+14} {y+14}V{y+2}",
         "check": f"M{x+2} {y+8.5}L{x+6} {y+12.5}L{x+14} {y+3.5}",
         "folder": f"M{x+1} {y+4}H{x+6}L{x+8} {y+6}H{x+15}V{y+14}H{x+1}Z",
         "merge": f"M{x+4} {y+4}V{y+12}M{x+4} {y+4}C{x+4} {y+9} {x+8} {y+9} {x+11.5} {y+9}"
                  f"M{x+4} {y+1.5}a1.6 1.6 0 1 0 0.01 0M{x+4} {y+12.5}a1.6 1.6 0 1 0 0.01 0M{x+13} {y+9}a1.6 1.6 0 1 0 0.01 0",
         "record": f"M{x+2} {y+3}H{x+14}M{x+2} {y+8}H{x+14}M{x+2} {y+13}H{x+10}"}[kind]
    return path(p, col, 1.5)


def flow_fig(t):
    narrow = t is AUTO
    lanes = [("A study", [("declare", "Declare", "arms and budget"), ("run", "Run", "each run recorded"),
                          ("check", "Check", "against the design")]),
             ("An outside entry", [("folder", "Submit", "a result folder"),
                                   ("check", "Check", "the bench workflow"), ("merge", "Merge", "complete records")])]
    out = []

    def node(x, y, w, h, kind, title, sub, accent=False):
        o = [rect(x + .5, y + .5, w - 1, h - 1, 14, t["tint"] if accent else t["panel"], t["acc3"] if accent else t["line"])]
        o.append(rect(x + 14, y + 14, 28, 28, 8, t["tint2"] if not accent else t["panel"]))
        o.append(icon(kind, x + 20, y + 20, t["acc"]))
        o.append(text(x + 54, y + 27, title, 13.5, t["acc"] if accent else t["ink"], 600))
        o.append(text(x + 54, y + 45, sub, 12, t["mid"]))
        return o
    if narrow:
        w = 360
        y = 16
        for lane, steps in lanes:
            out.append(text(16, y + 14, lane, 12, t["mid"], 600))
            y += 26
            for i, (kind, title, sub) in enumerate(steps):
                out += node(0, y, w, 60, kind, title, sub)
                y += 60
                out.append(line(28, y + 2, 28, y + 14, t["hair"], 1.3) + arrow_head(28, y + 16, 0, 1, t["hair"], 1.3, 4))
                y += 20
            out.pop()
            y += 10
        out.append(text(16, y + 14, "Both end in", 12, t["mid"], 600))
        y += 26
        out += node(0, y, w, 76, "record", "The record", "issues, runs and results; failed", accent=True)
        out.append(text(54, y + 63, "runs and refuted predictions kept", 12, t["mid"]))
        h = y + 86
    else:
        w, h = 880, 252
        out.append(card(0, 0, w, h, t))
        nw, nh, gap = 186, 60, 30
        rx = 28 + 3 * nw + 3 * gap
        for li, (lane, steps) in enumerate(lanes):
            y = 52 + li * 112
            out.append(text(28, y - 14, lane, 12, t["mid"], 600))
            for i, (kind, title, sub) in enumerate(steps):
                x = 28 + i * (nw + gap)
                out += node(x, y, nw, nh, kind, title, sub)
                xa = x + nw + 6
                ya = y + nh / 2
                if i < 2:
                    out.append(line(xa, ya, xa + gap - 12, ya, t["hair"], 1.3) + arrow_head(xa + gap - 11, ya, 1, 0, t["hair"], 1.3, 4))
            out.append(path(f"M{num(xa)} {num(ya)}H{num(rx - 8)}", t["acc3"], 1.3) + arrow_head(rx - 7, ya, 1, 0, t["acc3"], 1.3, 4))
        out.append(rect(rx + .5, 52.5, w - 28 - rx - 1, 172 - 1, 16, t["tint"], t["acc3"]))
        out.append(rect(rx + 16, 68, 28, 28, 8, t["panel"]) + icon("record", rx + 22, 74, t["acc"]))
        out.append(text(rx + 16, 128, "The record", 15, t["acc"], 650))
        for j, s in enumerate(["Issues, runs and results.", "Failed runs and refuted", "predictions stay."]):
            out.append(text(rx + 16, 152 + j * 18, s, 12.5, t["mid"]))
    desc = ("How research runs here. A study: declare, run, check against the declaration, record. "
            "An outside entry: a submission folder, the bench workflow's check, merge. Both end in the record, "
            "where failed runs and refuted predictions stay.")
    return svg(w, int(h), "How research runs here", desc, t, "".join(out))


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
    for key, (label, value) in badges().items():
        for variant, theme in [("light", LIGHT), ("dark", DARK)]:
            out[f"badge-{key}-{variant}.svg"] = badge(label, value, theme)
    for n in (1, 2):
        for variant, theme in [("light", LIGHT), ("dark", DARK)]:
            out[f"institution-{n}-{variant}.svg"] = institution(theme, n)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if a figure is not current")
    args = parser.parse_args()
    stale = []
    for name, content in figures().items():
        path_ = ASSETS / name
        if args.check:
            if not path_.is_file() or path_.read_text() != content:
                stale.append(name)
        else:
            path_.write_text(content)
    if stale:
        sys.exit("not current, run python scripts/readme_figures.py: " + ", ".join(stale))


if __name__ == "__main__":
    main()
