// The homepage's data, read at build time from the synced copy in public/ (the same files the homepage loads),
// so that every figure in the documentation draws the same values with the same drawing rules.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const PUBLIC = path.resolve(process.cwd(), 'public');
const ctx = { window: {} };
vm.createContext(ctx);
for (const f of ['data/configs.js', 'data/compare.js', 'data/checks.js', 'data/data.js', 'data/record.js', 'data/refs.js', 'content.js']) {
	vm.runInContext(fs.readFileSync(path.join(PUBLIC, f), 'utf8'), ctx, { filename: f });
}

export const C = ctx.window.CONFIGS;
export const P = ctx.window.PAPER;
export const CMP = ctx.window.COMPARE;
export const REFS = ctx.window.REFS;
export const T = ctx.window.CONTENT.en;
export const BKEYS = ['unique_non_embedding_parameters', 'training_flops_per_token', 'logical_kv_bytes'];

/* ---------------------------------------------------------------- the homepage's drawing rules (app.js) */
export function layers(cfg) {
	const out = [];
	const add = (n, r) => {
		for (let i = 0; i < n; i++) out.push({ region: r, P: cfg.regions[r].expertParamsM, E: cfg.regions[r].experts });
	};
	add(cfg.pre, 'prefix');
	add(cfg.core, 'core');
	add(cfg.suf, 'suffix');
	return out;
}
const pOf = (cfg) => Math.max(...layers(cfg).map((l) => l.P));
/** One height scale for every drawing: block area = expert parameters, against the largest layer of any configuration. */
export const PMAX = Math.max(...Object.values(C).map(pOf));

export const arcPath = (l0, r0, top, a) => `M${r0} ${top}C${r0 + 3} ${top - a * 1.32} ${l0 - 3} ${top - a * 1.32} ${l0} ${top}`;
export const headPath = (l0, top, k = 1) => `M${l0 - 3.6 * k} ${top - 4.4 * k}L${l0} ${top + 0.4}L${l0 + 3.2 * k} ${top - 4.6 * k}`;
const f = (v) => Number(v.toFixed(2));

/**
 * A model drawing as an SVG string: 1 block = 1 layer, block area = its expert parameters, indigo = the looped layers,
 * the arc = the loop with its visit count. Geometry follows the homepage's Model and miniModel.
 * o: { slots, pitch, bw, hU, top, sw, label, glow, regions, rx, pad }
 */
export function modelSvg(cfg, o = {}) {
	const slots = o.slots ?? 18;
	const pitch = o.pitch ?? 10;
	const bw = o.bw ?? 8.6;
	const hU = o.hU ?? 60;
	const top = o.top ?? 22;
	const bottom = o.bottom ?? 4;
	const sw = o.sw ?? 1.3;
	const W = slots * pitch + 8;
	const H = top + hU + bottom;
	const L = layers(cfg);
	const n = L.length;
	const cy = top + hU / 2;
	const x0 = W / 2 - (pitch * (n - 1)) / 2;
	const loop = cfg.visits > 1;
	const id = `g${Math.random().toString(36).slice(2, 8)}`;
	let s = `<svg class="model" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" aria-hidden="true" focusable="false">`;
	if (loop && o.glow) {
		const c0 = x0 + pitch * cfg.pre;
		const c1 = x0 + pitch * (cfg.pre + cfg.core - 1);
		s += `<defs><radialGradient id="${id}"><stop offset="0" stop-color="var(--fig-glow)" stop-opacity=".20"/><stop offset=".55" stop-color="var(--fig-glow)" stop-opacity=".07"/><stop offset="1" stop-color="var(--fig-glow)" stop-opacity="0"/></radialGradient></defs>`;
		s += `<ellipse cx="${f((c0 + c1) / 2)}" cy="${f(cy)}" rx="${f((c1 - c0) / 2 + pitch * 1.6)}" ry="${f(hU * 0.62)}" fill="url(#${id})"/>`;
	}
	s += `<line class="m-shaft" x1="${f(x0 - pitch * 0.7)}" x2="${f(x0 + pitch * (n - 0.3))}" y1="${f(cy)}" y2="${f(cy)}"/>`;
	let coreTop = cy;
	L.forEach((l, i) => {
		const k = Math.sqrt(l.P / PMAX);
		const hh = Math.max(1.5, hU * k);
		const ww = Math.max(1.2, bw * k);
		const acc = loop && l.region === 'core';
		if (l.region === 'core') coreTop = Math.min(coreTop, cy - hh / 2);
		const rx = Math.min(o.rx ?? 1.6, hh / 2, ww / 2);
		if (o.gap === i) {
			s += `<rect class="m-gap" x="${f(x0 + pitch * i - ww / 2)}" y="${f(cy - hh / 2)}" width="${f(ww)}" height="${f(hh)}" rx="${f(rx)}"/>`;
			return;
		}
		s += `<rect class="${acc ? 'm-acc' : 'm-ink'}" x="${f(x0 + pitch * i - ww / 2)}" y="${f(cy - hh / 2)}" width="${f(ww)}" height="${f(hh)}" rx="${f(rx)}"/>`;
	});
	if (loop) {
		const t = coreTop - 3;
		if (cfg.order === 'layer') {
			for (let i = 0; i < cfg.core; i++) {
				const cx = x0 + pitch * (cfg.pre + i);
				const r = pitch * 0.32;
				s += `<path class="m-arc" stroke-width="${sw}" d="M${f(cx + r)} ${f(t)}C${f(cx + r)} ${f(t - pitch)} ${f(cx - r)} ${f(t - pitch)} ${f(cx - r)} ${f(t)}"/>`;
			}
			if (o.label) s += `<text class="m-lab" x="${f(x0 + pitch * (cfg.pre + cfg.core - 1) + pitch * 0.9)}" y="${f(t - pitch * 0.55)}">×${cfg.visits}</text>`;
		} else {
			const c0 = x0 + pitch * cfg.pre;
			const c1 = x0 + pitch * (cfg.pre + cfg.core - 1);
			const one = c1 - c0 < 1;
			const l0 = c0 - (one ? 4 : 0);
			const r0 = c1 + (one ? 4 : 0);
			const a = Math.min(top - 6, 6 + (r0 - l0) * 0.22);
			s += `<path class="m-arc" stroke-width="${sw}" d="${arcPath(l0, r0, t, a)}"/>`;
			s += `<path class="m-arc" stroke-width="${sw}" stroke-linejoin="round" d="${headPath(l0, t, (o.hk ?? 0.7))}"/>`;
			if (o.label) s += `<text class="m-lab" x="${f((l0 + r0) / 2)}" y="${f(t - a - 4)}" text-anchor="middle">×${cfg.visits}</text>`;
		}
	}
	if (o.regions && loop) {
		const yb = top + hU + 4;
		const seg = (i0, i1, txt, cls) => {
			const a = x0 + pitch * i0 - pitch * 0.38;
			const b = x0 + pitch * i1 + pitch * 0.38;
			return `<path class="m-brace ${cls}" d="M${f(a)} ${f(yb)}v3H${f(b)}v-3"/><text class="m-reg ${cls}" x="${f((a + b) / 2)}" y="${f(yb + 12)}" text-anchor="middle">${txt}</text>`;
		};
		if (cfg.pre) s += seg(0, cfg.pre - 1, o.regions[0], '');
		s += seg(cfg.pre, cfg.pre + cfg.core - 1, o.regions[1], 'is-acc');
		if (cfg.suf) s += seg(cfg.pre + cfg.core, n - 1, o.regions[2], '');
	}
	return s + '</svg>';
}

/* ---------------------------------------------------------------- numbers */
export const minus = (str) => String(str).replace(/-/g, '−');
export const sgn = (v, d = 2) => {
	const r = Number(Math.abs(v).toFixed(d));
	return (r === 0 ? '' : v > 0 ? '+' : '−') + r.toFixed(d);
};
export const pct = (v) => `${sgn(v * 100, Math.abs(v) >= 0.1 ? 1 : 2)}%`;
