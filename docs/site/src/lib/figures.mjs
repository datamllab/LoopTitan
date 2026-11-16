// Figures at the top of documentation pages, rendered at build time as static HTML and SVG from the homepage's
// data files (public/data/*.js, content.js) with the homepage's drawing rules and vocabulary. No value is typed in.
import fs from 'node:fs';
import path from 'node:path';
import { BKEYS, C, P, T, modelSvg, pct, sgn } from './home.mjs';
import { bandSvg } from './vocab.mjs';
import { BASE } from '../../nav.mjs';

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const REPO = 'https://github.com/datamllab/LoopTitan/';
/** A mono path to a file of the repository: the homepage's evidence link. */
const ev = (file, label = file) => `<a class="lt-ev" href="${REPO}blob/main/${file}" rel="noopener">${esc(label)}</a>`;
const run = (cfg) => cfg.pre + cfg.core * cfg.visits + cfg.suf;
const stored = (cfg) => cfg.pre + cfg.core + cfg.suf;
const key = () =>
	`<p class="lt-key"><span class="lt-key__i"><span class="lt-key__k"></span>one layer, area = its expert parameters</span><span class="lt-key__i"><span class="lt-key__k lt-key__k--acc"></span>looped layers</span><span class="lt-key__i"><span class="lt-key__arc">×2</span>visits per token</span></p>`;

/* The mechanism drawings: read from the homepage's app.js (const MF = {...}), so both stay one drawing. */
function mechanismDrawings() {
	const src = fs.readFileSync(path.resolve(process.cwd(), 'public/app.js'), 'utf8');
	const m = src.match(/const MF = (\{[\s\S]*?\n\t\});/);
	if (!m) throw new Error('figures: the mechanism drawings (const MF = {...}) are not in public/app.js any more');
	return new Function(`return (${m[1]});`)();
}

/* the homepage's mechs(): what a configuration adds, in words */
function mechs(cfg) {
	const M = T.mech;
	const out = [];
	const r = cfg.regions.core.residual;
	if (cfg.visitResidualScales) out.push(M.visitResidual);
	else if (r !== 1) out.push(`residual ×${Number(r.toFixed(2))}`);
	if (cfg.visits > 1 && cfg.order === 'layer') out.push(M.layerOrder);
	if (cfg.visits > 1 && cfg.attentionPerVisit) out.push(M.attentionPerVisit);
	if (cfg.visits > 1 && cfg.routerPerVisit) out.push(M.routerPerVisit);
	if (cfg.visits > 1 && cfg.normPerVisit && !cfg.attentionPerVisit) out.push(M.normPerVisit);
	if (cfg.loraAttentionRank > 0) out.push(M.lora);
	if (cfg.inputInjection && cfg.inputInjection !== 'none') out.push(M.injection);
	if (cfg.initialState === 'random') out.push(M.random);
	if (cfg.stateUpdate === 'ema') out.push(M.ema);
	if (cfg.exitDistribution) out.push(M.exit);
	return out;
}

/* ---------------------------------------------------------------- architecture: one model, its regions, two orders */
function architecture() {
	const k = 'phase1/m3';
	const cfg = C[k];
	const big = modelSvg(cfg, { slots: 14, pitch: 30, bw: 15, hU: 110, top: 50, bottom: 26, sw: 1.7, hk: 1.25, label: true, regions: ['prefix · runs once', `core · runs ×${cfg.visits}`, 'suffix · runs once'] });
	const order = (kk, name, seq) =>
		`<div class="lt-order"><div class="lt-order__fig">${modelSvg(C[kk], { slots: 8, pitch: 12, bw: 7, hU: 40, top: 22, sw: 1.3, hk: 0.75, label: true })}</div><div class="lt-order__t"><p class="lt-order__n">${name}<code>${seq}</code></p><p class="lt-order__d">${esc(T.design[kk])}</p>${ev(C[kk].recipes[0])}</div></div>`;
	return `<figure class="not-content lt-fig lt-fig--arch" aria-label="${esc(`${T.design[k]}: ${stored(cfg)} layers stored, ${run(cfg)} layers run.`)}">
	<div class="lt-arch__stage">${big}</div>
	<div class="lt-arch__id">
		<div><p class="lt-fig__name">${esc(T.design[k])}</p><p class="lt-fig__tags"><span class="lt-tagm">residual ×${Number(cfg.regions.core.residual.toFixed(2))}</span></p>${ev(cfg.recipes[0])}</div>
		<dl class="lt-count"><div><dt>Layers stored</dt><dd>${stored(cfg)}</dd></div><div><dt>Layers run</dt><dd class="is-loop">${run(cfg)}</dd></div></dl>
	</div>
	<div class="lt-orders">
		${order('phase1/m1s', 'Stack order', 'A → B → A → B')}
		${order('phase1/m1', 'Layer order', 'A → A → B → B')}
	</div>
	${key()}
</figure>`;
}

/* ---------------------------------------------------------------- recipes: the nine design decisions */
function recipes() {
	const MF = mechanismDrawings();
	const cells = T.decisions
		.map((d) => `<li class="lt-mech"><div class="lt-mech__fig"><svg class="mf" viewBox="0 0 240 92" aria-hidden="true">${MF[d.id]()}</svg></div><p class="lt-mech__k">${esc(d.k)}</p><p class="lt-mech__d">${esc(d.d)}</p><p class="lt-mech__f">${esc(d.f)}</p></li>`)
		.join('');
	return `<figure class="not-content lt-fig lt-fig--grid"><ol class="lt-mechs">${cells}</ol><figcaption class="lt-cap">Each design decision is a field of one model; a recipe sets these fields. The <a href="${BASE}docs/method-map/">method map</a> draws each recipe.</figcaption></figure>`;
}

/* ---------------------------------------------------------------- comparison: match first, then compare */
function gauge(b) {
	const DOM = 0.08;
	const clamp = (v) => Math.max(-DOM, Math.min(DOM, v));
	const pos = (v) => `${((clamp(v) / DOM) * 50 + 50).toFixed(2)}%`;
	const out = Math.abs(b.rel) > b.tol;
	const off = Math.abs(b.rel) > DOM;
	return `<div class="lt-gauge"><span class="lt-gauge__v${out ? ' is-out' : ''}">${pct(b.rel)}</span><span class="lt-gtrack"><span class="lt-gtrack__band" style="left:${pos(-b.tol)};width:${((b.tol / DOM) * 100).toFixed(2)}%"></span><span class="lt-gtrack__z"></span><span class="lt-gtrack__dot${out ? ' is-out' : ''}" style="left:${pos(b.rel)}"></span>${off ? '<svg class="lt-gtrack__off" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 2.5l4 3.5-4 3.5M7 2.5l4 3.5-4 3.5"/></svg>' : ''}</span></div>`;
}
function comparison() {
	const rows = ['phase1/m0', 'phase1/m3', 'phase1/m4', 'methods/m0-loop'];
	const body = rows
		.map((k) => {
			const cfg = C[k];
			const ref = k === 'phase1/m0';
			const rels = BKEYS.map((b) => cfg.budgets[b].rel);
			const out = !ref && BKEYS.some((b) => Math.abs(cfg.budgets[b].rel) > cfg.budgets[b].tol);
			const state = ref ? 'reference' : out ? 'outside ±5%' : 'within ±5%';
			return `<li class="lt-cmp${ref ? ' is-ref' : ''}${out ? ' is-out' : ''}">
				<div class="lt-cmp__fig">${modelSvg(cfg, { slots: 18, pitch: 10, bw: 8.6, hU: 46, top: 20, label: true })}</div>
				<div class="lt-cmp__id"><p class="lt-cmp__n">${esc(T.design[k])}</p><p class="lt-cmp__m"><span class="lt-okline${out ? ' is-out' : ''}${ref ? ' is-ref' : ''}">${ref ? '' : bandSvg(rels)}<span>${state}</span></span><span class="lt-cmp__c">${stored(cfg)} stored · ${run(cfg)} run</span></p></div>
				<div class="lt-cmp__g">${BKEYS.map((b) => gauge(cfg.budgets[b])).join('')}</div>
			</li>`;
		})
		.join('');
	return `<figure class="not-content lt-fig lt-fig--cmp">
	<div class="lt-fig__head"><p class="lt-fig__title">Match first, then compare</p><p class="lt-fig__hint">Against ${ev(C['phase1/m0'].reference)}</p></div>
	<ol class="lt-cmps"><li class="lt-cmp lt-cmp--head" aria-hidden="true"><span></span><span></span><span class="lt-cmp__g">${BKEYS.map((b) => `<span>${T.budget[b]}</span>`).join('')}</span></li>${body}</ol>
	<figcaption class="lt-cap">Unique non-embedding parameters, training FLOPs per token and logical KV cache from <code>compare_recipes</code>, on a CPU; the band is ±5%. The unrolled twin writes the loop out as distinct layers. A loop added without matching leaves the band.</figcaption>
</figure>`;
}

/* ---------------------------------------------------------------- method map: every recipe as a drawing */
function methodMap() {
	const list = ['phase1/m3', 'methods/meta', 'methods/foil', 'methods/expert-tying', 'methods/huginn', 'methods/loom', 'methods/ouro', 'methods/sandwich', 'methods/relaxed', 'methods/latentloop-residual'];
	const cells = list
		.map((k) => {
			const cfg = C[k];
			const tags = mechs(cfg).map((t) => `<span class="lt-tagm">${esc(t)}</span>`).join('');
			return `<li class="lt-des"><div class="lt-des__fig">${modelSvg(cfg, { slots: 13, pitch: 10, bw: 8.6, hU: 50, top: 22, label: true })}</div><div class="lt-des__t"><p class="lt-des__n">${esc(T.design[k])}</p><p class="lt-fig__tags">${tags || '<span class="lt-tagm lt-tagm--n">every part shared</span>'}</p><p class="lt-des__f">${ev(cfg.recipes[0])}<span>${stored(cfg)} stored · ${run(cfg)} run</span></p></div></li>`;
		})
		.join('');
	return `<figure class="not-content lt-fig lt-fig--grid"><ol class="lt-dess">${cells}</ol>${key()}</figure>`;
}

/* ---------------------------------------------------------------- results: the one-change chain */
function chainSvg(W, small) {
	const BPB = P.fig1_decomposition.test_bpb;
	const CH = T.chain;
	const steps = CH.slice(1).map(([k, n]) => ({ n, b: k, loop: k === 'M3' }));
	steps.forEach((s, i) => (s.a = CH[i][0]));
	const cum = (o, k) => (BPB[k][o] - BPB.M0[o]) * 1000;
	const LW = small ? 0 : 200;
	const RH = small ? 56 : 46;
	const top = 8;
	const rows = [{ n: 'Ordinary MoE', base: true }, ...steps, { n: 'Looped MoE, all five', total: true, a: 'M0', b: 'M3' }];
	const H = top + rows.length * RH + 44;
	const LO = -10;
	const HI = 2;
	const P0 = LW + 18;
	const P1 = W - (small ? 30 : 40);
	const x = (v) => P0 + ((P1 - P0) * (v - LO)) / (HI - LO);
	const yAx = top + rows.length * RH + 6;
	let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="The looped recipe’s gain over the ordinary MoE, built one decision at a time, under AdamW and Muon. The loop step itself is close to zero.">`;
	for (let v = LO; v <= HI; v += 2) s += `<line class="${v === 0 ? 'c-zero' : 'c-grid'}" x1="${x(v)}" x2="${x(v)}" y1="${top}" y2="${yAx}"/>`;
	rows.forEach((r, i) => {
		const y0 = top + i * RH;
		let g = `<g class="c-row${r.loop ? ' is-loop' : ''}${r.total ? ' is-total' : ''}"><rect class="c-hit" x="-6" y="${y0 + 2}" width="${W + 12}" height="${RH - 4}" rx="8"/>`;
		if (r.total) s += `<line class="c-sep" x1="0" x2="${W}" y1="${y0 - 1}" y2="${y0 - 1}"/>`;
		g += `<text class="c-name" x="4" y="${small ? y0 + 17 : y0 + RH / 2 + 5}">${esc((r.base || r.total ? '' : '+ ') + r.n)}</text>`;
		const tip = [];
		if (r.base) {
			g += `<rect class="c-base" x="${x(0) - 1}" y="${y0 + 11}" width="2" height="${RH - 22}" rx="1"/>`;
			g += `<text class="c-note" x="${x(0) + 8}" y="${y0 + RH / 2 + 4}">reference</text>`;
			tip.push(`AdamW ${BPB.M0.adamw.toFixed(4)} · Muon ${BPB.M0.muon.toFixed(4)} test BPB`);
		} else {
			[['adamw', 0], ['muon', 1]].forEach(([o, j]) => {
				const a = r.total ? 0 : cum(o, r.a);
				const b = cum(o, r.b);
				const yy = (small ? y0 + 26 : y0 + RH / 2 - 9) + j * 10;
				const xa = x(a);
				const xb = x(b);
				g += `<rect class="c-bar c-bar--${o}" x="${Math.min(xa, xb).toFixed(2)}" y="${yy}" width="${Math.max(1.5, Math.abs(xb - xa)).toFixed(2)}" height="8" rx="2"/>`;
				if (!r.total && i > 1) s += `<line class="c-carry" x1="${xa}" x2="${xa}" y1="${yy - RH + 10}" y2="${yy + 8}"/>`;
				const d = b - a;
				g += `<text class="c-num${r.total ? ' is-strong' : ''}" x="${(d < 0 ? Math.min(xa, xb) - 6 : Math.max(xa, xb) + 6).toFixed(2)}" y="${yy + 7.5}" text-anchor="${d < 0 ? 'end' : 'start'}">${sgn(d, 1)}</text>`;
				tip.push(`${o === 'adamw' ? 'AdamW' : 'Muon'} ${BPB[r.b][o].toFixed(4)} test BPB (${sgn(d, 2)} mBPB${r.total ? '' : ' this step'})`);
			});
		}
		g += `<title>${esc(`${r.n}: ${tip.join('; ')}`)}</title></g>`;
		s += g;
	});
	s += `<line class="c-axis" x1="${P0}" x2="${P1}" y1="${yAx}" y2="${yAx}"/>`;
	for (let v = LO; v <= HI; v += 2) s += `<text class="c-tick" x="${x(v)}" y="${yAx + 17}" text-anchor="middle">${v < 0 ? `−${-v}` : v}</text>`;
	s += `<text class="c-axt" x="${P1}" y="${yAx + 36}" text-anchor="end">← better · test mBPB against the ordinary MoE</text>`;
	return s + '</svg>';
}
function results() {
	const TR = P.fig3_instrument.training;
	const lr = (v) => v.toExponential(1).replace(/\.0e/, 'e').replace('-', '−');
	const tokens = Math.round((TR.horizon_5B_updates * TR.tokens_per_update) / 1e9);
	const B = (n) => `${(n / 1e9).toFixed(1)}B`;
	const sc = P.scale;
	return `<figure class="not-content lt-fig lt-fig--chain">
	<div class="lt-fig__head"><div><p class="lt-fig__eyebrow"><span class="lt-mlabel">Phase 1</span>${B(sc.stored_non_embedding)} stored · ${B(sc.active_non_embedding)} active · ${tokens}B tokens</p><p class="lt-fig__title">The looped recipe, built one decision at a time</p></div><p class="lt-legend"><span><span class="lt-legend__k"></span>AdamW</span><span><span class="lt-legend__k lt-legend__k--m"></span>Muon</span></p></div>
	<div class="lt-chain lt-chain--wide">${chainSvg(720, false)}</div>
	<div class="lt-chain lt-chain--small">${chainSvg(360, true)}</div>
	<figcaption class="lt-cap">Test mBPB against the ordinary MoE trained with the same optimizer; lower is better. Each step adds one decision at matched parameters, FLOPs and KV cache, as in the first table of each optimizer below. Learning rate tuned once per optimizer on the ordinary MoE (AdamW ${lr(TR.lr_adamw)}, Muon ${lr(TR.lr_muon)}), seed ${TR.seed}.</figcaption>
</figure>`;
}

const BY_SLUG = {
	'docs/architecture': architecture,
	'docs/recipes': recipes,
	'docs/comparison': comparison,
	'docs/method-map': methodMap,
	'docs/results/phase1': results,
};
const cache = new Map();
/** The figure for a page, or ''. */
export function figureFor(slug) {
	if (!BY_SLUG[slug]) return '';
	if (!cache.has(slug)) cache.set(slug, BY_SLUG[slug]());
	return cache.get(slug);
}
