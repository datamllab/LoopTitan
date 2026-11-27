/* LoopTitan homepage, round 6 (b).
   One drawing grammar for every model: 1 block = 1 layer, block area = its expert parameters (one scale for the
   whole page, so equal parameters look like equal ink), indigo = the looped layers, the arc = the loop and its
   visits. A light pulse walks the token's path through the layers, so "stored once, run twice" is seen, not read.
   On load the spiral mark unrolls into the first model. Every value comes from data/*.js, generated from the
   repository. Strings: content.js. */
(() => {
	'use strict';
	const C = window.CONFIGS;
	const CMP = window.COMPARE;
	const CHK = window.CHECKS;
	const P = window.PAPER;
	const REC = window.RECORD;
	const REFS = window.REFS;
	const T = window.CONTENT.en;
	const root = document.documentElement;
	const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
	const REPO = 'https://github.com/datamllab/LoopTitan/';
	const BLOB = REPO + 'blob/main/';
	const ISSUE = REPO + 'issues/';
	const BKEYS = ['unique_non_embedding_parameters', 'training_flops_per_token', 'logical_kv_bytes'];

	const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
	const lerp = (a, b, t) => a + (b - a) * t;
	/* the motion system's easings, solved exactly from the same cubic-bezier tokens as style.css */
	const bezier = (x1, y1, x2, y2) => {
		const f = (a, b, t) => ((1 - 3 * b + 3 * a) * t + (3 * b - 6 * a)) * t * t + 3 * a * t;
		const d = (a, b, t) => 3 * (1 - 3 * b + 3 * a) * t * t + 2 * (3 * b - 6 * a) * t + 3 * a;
		return (x) => {
			if (x <= 0) return 0;
			if (x >= 1) return 1;
			let t = x;
			for (let i = 0; i < 6; i++) { const e = f(x1, x2, t) - x, s_ = d(x1, x2, t); if (Math.abs(e) < 1e-5 || !s_) break; t -= e / s_; }
			return f(y1, y2, clamp(t));
		};
	};
	const easeOut = bezier(0.16, 1, 0.3, 1);
	const easeIO = bezier(0.65, 0, 0.35, 1);
	const easeIn = bezier(0.7, 0, 0.84, 0);
	const minus = (str) => String(str).replace(/-/g, '−');
	const sgn = (v, d = 2) => {
		const r = Number(Math.abs(v).toFixed(d));
		return (r === 0 ? '' : v > 0 ? '+' : '−') + r.toFixed(d);
	};
	const pct = (v) => `${sgn(v * 100, Math.abs(v) >= 0.1 ? 1 : 2)}%`;
	const NS = 'http://www.w3.org/2000/svg';
	function s(tag, attrs, parent) {
		const n = document.createElementNS(NS, tag);
		for (const k in attrs) n.setAttribute(k, attrs[k]);
		if (parent) parent.appendChild(n);
		return n;
	}
	function h(tag, cls, parent, text) {
		const n = document.createElement(tag);
		if (cls) n.className = cls;
		if (text != null) n.textContent = text;
		if (parent) parent.appendChild(n);
		return n;
	}
	const css = getComputedStyle(root);
	const remK = () => parseFloat(getComputedStyle(root).fontSize) / 16 || 1;
	const hex = (name) => {
		const v = css.getPropertyValue(name).trim();
		return [1, 3, 5].map((i) => parseInt(v.slice(i, i + 2), 16));
	};
	/* durations from the motion tokens in style.css: press, t1 … t5, stagger (ms) */
	const MO = Object.fromEntries(['press', '1', '2', '3', '4', '5'].map((n) => [n === 'press' ? 'press' : 't' + n, parseFloat(css.getPropertyValue(`--t-${n}`)) || 0]));
	MO.stagger = parseFloat(css.getPropertyValue('--stagger')) || 40;
	/* colour is mixed in OKLab, so ink and indigo never pass through a dull navy */
	const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
	const gam = (c) => 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(Math.max(0, c), 1 / 2.4) - 0.055);
	const toLab = (c) => {
		const [r, g, b] = c.map(lin);
		const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b), m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b), q = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
		return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * q, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * q, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * q];
	};
	const fromLab = ([L, a, b]) => {
		const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, q = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
		return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * q, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * q, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * q].map((v) => clamp(gam(v), 0, 255));
	};
	const lerpL = (a, b, t) => a.map((x, i) => lerp(x, b[i], t));
	const INK = toLab(hex('--ink'));
	const ACC = toLab(hex('--acc'));
	const ACC2 = toLab(hex('--acc-2'));
	const INKHI = lerpL(INK, ACC, 0.32);
	const rgb = (lab) => `rgb(${fromLab(lab).map((x) => Math.round(x)).join(',')})`;

	/* one spring for every drawn thing: critically damped, ω = 11 (settles in about --t-4); leaving things ω = 16 */
	const W_IN = 11, W_OUT = 16;
	class Spring {
		constructor(v, w = W_IN, z = 1) { this.v = v; this.t = v; this.vel = 0; this.w = w; this.z = z; }
		set(v) { this.v = this.t = v; this.vel = 0; }
		step(dt) {
			if (this.v === this.t && this.vel === 0) return false;
			const n = Math.max(1, Math.ceil(dt * 240));
			const d = dt / n;
			for (let i = 0; i < n; i++) {
				const a = -this.w * this.w * (this.v - this.t) - 2 * this.z * this.w * this.vel;
				this.vel += a * d;
				this.v += this.vel * d;
			}
			if (Math.abs(this.v - this.t) < 1e-3 && Math.abs(this.vel) < 1e-2) { this.v = this.t; this.vel = 0; }
			return true;
		}
	}
	function animate(fn) {
		let last = 0;
		const loop = () => {
			const now = performance.now();
			const dt = Math.min(0.05, (now - (last || now)) / 1000);
			last = now;
			if (fn(dt, now)) requestAnimationFrame(loop);
		};
		requestAnimationFrame(loop);
	}
	function onVisible(el, fn, threshold = 0.25) {
		const io = new IntersectionObserver((es) => {
			if (!es[0].isIntersecting) return;
			io.disconnect();
			fn();
		}, { threshold });
		io.observe(el);
	}
	const onResize = (el, fn) => {
		let w = el.clientWidth, hh = el.clientHeight;
		new ResizeObserver(() => {
			if (el.clientWidth === w && el.clientHeight === hh) return;
			w = el.clientWidth; hh = el.clientHeight;
			fn();
		}).observe(el);
	};

	/* ---------------------------------------------------------------- references */
	const refNo = {};
	REFS.forEach((r, i) => (refNo[r.id] = i + 1));
	const arx = (url) => (url && (url.match(/(\d{4}\.\d{5})/) || [])[1]) || null;
	function refMark(parent, ids, plain) {
		ids = [...new Set(ids.filter(Boolean))].sort((a, b) => refNo[a] - refNo[b]);
		if (!ids.length) return null;
		const sup = h('sup', 'ref-k', parent);
		if (plain) {
			sup.textContent = ids.map((id) => refNo[id]).join(',');
			sup.setAttribute('aria-label', `reference ${sup.textContent}`);
			return sup;
		}
		ids.forEach((id, i) => {
			if (i) sup.append(',');
			const a = h('a', '', sup, String(refNo[id]));
			a.href = `#ref-${refNo[id]}`;
			a.setAttribute('aria-label', `Reference ${refNo[id]}`);
		});
		return sup;
	}
	function refs() {
		const ol = document.querySelector('[data-refs-list]');
		REFS.forEach((r, i) => {
			const li = h('li', '', ol);
			li.id = `ref-${i + 1}`;
			h('span', 'refs__n', li, `${i + 1}`);
			const t = h('span', '', li);
			const a = h('a', '', t, r.title.replace(/ -- /g, ' – '));
			a.href = r.url;
			a.rel = 'noopener';
			t.append(` arXiv:${r.id}, ${r.year}.`);
		});
		document.querySelectorAll('[data-refs]').forEach((n) => {
			const p = n.parentNode;
			const m = refMark(p, n.dataset.refs.split(' '));
			if (m) p.replaceChild(m, n);
		});
	}

	/* ---------------------------------------------------------------- configurations */
	function layers(cfg) {
		const out = [];
		const add = (n, r) => { for (let i = 0; i < n; i++) out.push({ region: r, P: cfg.regions[r].expertParamsM, E: cfg.regions[r].experts }); };
		add(cfg.pre, 'prefix');
		add(cfg.core, 'core');
		add(cfg.suf, 'suffix');
		return out;
	}
	const pOf = (cfg) => Math.max(...layers(cfg).map((l) => l.P));
	const PMAX = Math.max(...Object.values(C).map(pOf));
	const fileOf = (k) => C[k].recipes[0];
	const num = (r) => `×${Number(r.toFixed(2))}`;
	function mechs(cfg) {
		const M = T.mech;
		const out = [];
		const r = cfg.regions.core.residual;
		if (cfg.visitResidualScales) out.push(M.visitResidual);
		else if (r !== 1) out.push(`residual ${num(r)}`);
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
	const fmtVal = (v) => (v === null || v === undefined ? '—' : typeof v === 'number' ? minus(String(Number(v.toPrecision(4)))) : Array.isArray(v) ? `[${v.map(fmtVal).join(', ')}]` : String(v));
	function diffRows(changes) {
		const rows = [];
		for (const f in changes) {
			const { old: a, new: b } = changes[f];
			if ((a && typeof a === 'object' && !Array.isArray(a)) || (b && typeof b === 'object' && !Array.isArray(b))) {
				const A = a || {}, B = b || {};
				for (const k of new Set([...Object.keys(A), ...Object.keys(B)])) {
					const dflt = k === 'residual_scale' ? 1 : null;
					if (JSON.stringify(A[k] ?? dflt) !== JSON.stringify(B[k] ?? dflt)) rows.push([`${f}.${k}`, fmtVal(A[k]), fmtVal(B[k])]);
				}
			} else rows.push([f, fmtVal(a), fmtVal(b)]);
		}
		return rows;
	}
	const fieldsOf = (cfg) => new Set(diffRows(cfg.changes).map((r) => r[0]));
	function added(k) {
		const cfg = C[k];
		const par = cfg.parent ? C[cfg.parent] : null;
		const F = fieldsOf(cfg);
		const out = new Set();
		const loops = cfg.visits > 1 || (par && par.visits > 1);
		const has = (...f) => f.some((x) => F.has(x));
		if (has('prefix_cells', 'num_cells', 'suffix_cells')) {
			if (loops) out.add('region');
			else if (par && cfg.pre + cfg.core + cfg.suf === par.pre + par.core + par.suf) out.add('middle');
			else out.add('layers');
		}
		if (has('num_visits')) out.add('visits');
		if (has('attention_per_visit', 'router_per_visit', 'norm_per_visit')) out.add('owners');
		if (has('core.residual_scale', 'visit_residual_scales')) out.add('residual');
		if (has('input_injection', 'injection_dt')) out.add('input');
		if (has('core.output_lr_scale')) out.add('outlr');
		if (has('core.output_init_scale')) out.add('outinit');
		if (has('initial_state', 'state_update')) out.add('state');
		if (has('exit_distribution')) out.add('depth');
		if (has('lora_attention_rank')) out.add('lora');
		if (has('order')) out.add('tying');
		if (has('dim', 'hidden_dim', 'num_heads')) out.add('width');
		if ([...F].some((f) => /num_experts|top_k|\.hidden_dim/.test(f))) out.add('experts');
		if (!F.size && cfg.parent) out.add('lr');
		if (par && par.visits > 1 && cfg.visits === 1) { out.delete('region'); out.delete('visits'); out.add('unroll'); }
		const order = ['unroll', 'region', 'visits', 'owners', 'residual', 'input', 'state', 'depth', 'lora', 'tying', 'outlr', 'outinit', 'layers', 'middle', 'width', 'experts', 'lr'];
		return order.filter((x) => out.has(x));
	}

	/* ---------------------------------------------------------------- the model drawing */
	const arcPath = (l0, r0, top, a) => `M${r0} ${top}C${r0 + 3} ${top - a * 1.32} ${l0 - 3} ${top - a * 1.32} ${l0} ${top}`;
	const headPath = (l0, top, k = 1) => `M${l0 - 3.6 * k} ${top - 4.4 * k}L${l0} ${top + 0.4}L${l0 + 3.2 * k} ${top - 4.6 * k}`;
	const bez = (p0, p1, p2, p3, t) => {
		const u = 1 - t;
		return [u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0], u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]];
	};
	let uid = 0;
	const DPR = () => window.devicePixelRatio || 1;
	const snapPx = (v) => Math.round(v * DPR()) / DPR();
	class Model {
		constructor(svg, o = {}) {
			this.svg = svg;
			this.o = o;
			this.id = ++uid;
			const defs = s('defs', {}, svg);
			const rg = s('radialGradient', { id: `glow${this.id}` }, defs);
			s('stop', { offset: '0', 'stop-color': 'var(--acc)', 'stop-opacity': '.20' }, rg);
			s('stop', { offset: '.55', 'stop-color': 'var(--acc-2)', 'stop-opacity': '.07' }, rg);
			s('stop', { offset: '1', 'stop-color': 'var(--acc-2)', 'stop-opacity': '0' }, rg);
			this.glow = s('ellipse', { fill: `url(#glow${this.id})`, opacity: 0 }, svg);
			this.shaft = s('line', { stroke: 'var(--hair)', 'stroke-width': 1, 'stroke-linecap': 'round' }, svg);
			this.barG = s('g', {}, svg);
			this.slots = o.slots || 18;
			/* block side = sqrt of its expert parameters on one page-wide scale: block area = expert parameters */
			// a layer is an ink block and an indigo block over it; the loop reaches a layer by filling it from the shaft outward
			this.bars = Array.from({ length: this.slots + 2 }, () => {
				const g = s('g', { opacity: 0 }, this.barG);
				return { g, el: s('rect', {}, g), ov: s('rect', {}, g), x: new Spring(0), h: new Spring(0), a: new Spring(0), c: new Spring(0), hit: 0, hov: 0, L: null };
			});
			this.arcG = s('g', { opacity: 0 }, svg);
			const sw = o.sw || 1.5;
			this.arc = s('path', { fill: 'none', stroke: 'var(--acc)', 'stroke-width': sw, 'stroke-linecap': 'round', pathLength: 1, 'stroke-dasharray': '1 1' }, this.arcG);
			this.head = s('path', { fill: 'none', stroke: 'var(--acc)', 'stroke-width': sw, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', pathLength: 1, 'stroke-dasharray': '1 1' }, this.arcG);
			this.lab = s('text', { fill: 'var(--acc)', 'text-anchor': 'middle', 'font-weight': 600 }, this.arcG);
			this.small = s('g', { opacity: 0 }, svg);
			this.smallEls = [];
			this.smallLab = s('text', { fill: 'var(--acc)', 'font-weight': 600 }, this.small);
			this.trailG = s('g', {}, svg);
			this.trail = [];
			this.trailEls = Array.from({ length: 16 }, () => s('line', { stroke: 'var(--acc-2)', 'stroke-linecap': 'round', opacity: 0 }, this.trailG));
			// the mechanisms a design adds, drawn on the model itself: input re-entry, the start state, the exit rule, a tied bank
			this.mechG = s('g', { opacity: 0, class: 'mechs-on' }, svg);
			this.mk = '';
			this.pulseG = s('g', { opacity: 0 }, svg);
			this.halo = s('circle', { fill: 'var(--acc)', opacity: 0.18 }, this.pulseG);
			this.dot = s('circle', { fill: '#fff', stroke: 'var(--acc)' }, this.pulseG);
			this.SP = { s0: new Spring(0), s1: new Spring(0), x0: new Spring(0), x1: new Spring(0), top: new Spring(0), aA: new Spring(0), aS: new Spring(0), hA: new Spring(1, W_OUT), dA: new Spring(1), mA: new Spring(0), gx: new Spring(0), grx: new Spring(0), gA: new Spring(0), sh: new Spring(1) };
			this.want = { aA: 0, aS: 0 };
			this.cfg = null;
			this.layerOrder = null;
			this.running = false;
			this.pulse = null;
			this.labHold = null;
			this.measure();
		}
		measure() {
			const W = this.svg.clientWidth || 600, H = this.svg.clientHeight || 300;
			const o = this.o;
			const k = (this.k = remK());
			const pitch = Math.min((o.maxPitch || 52) * k, (W - 24 * k) / (this.slots + 0.6));
			const bw = clamp(pitch * (o.bwk || 0.84), 4, (o.maxBw || 36) * k);
			const top = o.top != null ? o.top * k : clamp(H * (o.topF || 0.25), 34 * k, (o.topMax || 96) * k);
			const bottom = (o.bottom != null ? o.bottom : 8) * k;
			this.fs = (o.fs || 13) * k;
			if (o.fitP) {
				// a drawing that only ever shows a known set: its largest block fills the room, and the drawing sits on its floor
				const hFit = (H - top - bottom) * (o.fill || 0.72);
				const cy = H - bottom - hFit / 2;
				this.geo = { W, H, pitch, bw, top: cy - hFit / 2, hU: hFit / Math.sqrt(o.fitP / PMAX), cy };
			} else {
				const hU = H - top - bottom;
				this.geo = { W, H, pitch, bw, top, hU, cy: top + hU / 2 };
			}
			const sw = (o.sw || 1.5) * k;
			for (const e of [this.arc, this.head]) e.setAttribute('stroke-width', sw);
			this.lab.setAttribute('font-size', this.fs);
			this.smallLab.setAttribute('font-size', this.fs);
			this.dot.setAttribute('r', 4.5 * k);
			this.dot.setAttribute('stroke-width', 2.25 * k);
			this.halo.setAttribute('r', 12 * k);
		}
		/* never two loop drawings at once: the incoming one waits until the outgoing one has gone */
		gate() {
			const SP = this.SP;
			SP.aA.t = this.want.aA && SP.aS.v < 0.05 ? 1 : 0;
			SP.aS.t = this.want.aS && SP.aA.v < 0.05 ? 1 : 0;
			SP.aA.w = SP.aA.t ? W_IN : W_OUT;
			SP.aS.w = SP.aS.t ? W_IN : W_OUT;
		}
		set(cfg, snap) {
			this.cfg = cfg;
			const L = layers(cfg);
			const n = L.length;
			const { W, hU, cy } = this.geo;
			let pitch = this.geo.pitch;
			if (this.o.spread && n <= 6) pitch = Math.min(pitch * 1.9, (W * 0.62) / Math.max(1, n - 1));
			this.pitchN = pitch;
			const x0 = W / 2 - (pitch * (n - 1)) / 2;
			const xLast = x0 + pitch * (n - 1);
			const loop = cfg.visits > 1;
			let coreTop = cy;
			// one change in three beats, never two states at once: leaving layers sink into the shaft, the layers that
			// stay slide to their places, then new layers rise out of the shaft where they belong, left to right
			const now = performance.now();
			let moves = false, leaves = false;
			this.bars.forEach((b, i) => {
				if (b.a.v <= 0.02 || b.h.t === 0) return;
				if (i >= n) leaves = true;
				else if (Math.abs(b.x.v - (x0 + pitch * i)) > 2) moves = true;
			});
			// the layers that stay slide and resize at once, so the drawing stays centred through the change
			const tMove = 0;
			const tRise = tMove + (moves ? MO.t3 : leaves ? MO.t1 : 0);
			let born = 0;
			this.bars.forEach((b, i) => {
				if (i < n) {
					const hh = Math.max(2, hU * Math.sqrt(L[i].P / PMAX));
					b.a.w = W_IN;
					b.h.w = W_IN;
					b.leaving = false;
					b.at = 0;
					b.atX = 0;
					if (b.a.v < 0.02 || b.h.t === 0) {
						b.x.t = x0 + pitch * i;
						if (snap) { b.x.set(b.x.t); b.h.set(hh); }
						else {
							b.x.set(b.x.t); b.h.set(0); b.c.set(loop && L[i].region === 'core' ? 1 : 0); b.a.set(1);
							b.at = now + tRise + born++ * (MO.stagger / 2);
							b.hT = hh;
						}
					} else if (tMove) { b.atX = now + tMove; b.xT = x0 + pitch * i; }
					else b.x.t = x0 + pitch * i;
					if (!b.at) b.h.t = hh;
					b.a.t = 1;
					b.c.t = loop && L[i].region === 'core' ? 1 : 0;
					b.L = L[i];
					b.i = i;
					if (L[i].region === 'core') coreTop = Math.min(coreTop, cy - hh / 2);
				} else {
					b.L = null;
					b.at = 0;
					if (b.a.v > 0.02 && !snap) { b.leaving = true; b.h0 = b.h.v; b.h.w = 24; b.h.t = 0; b.c.t = b.c.v; }
					else { b.leaving = false; b.a.t = 0; b.a.set(0); }
				}
			});
			const SP = this.SP;
			SP.s0.t = x0 - pitch * 0.7;
			SP.s1.t = xLast + pitch * 0.7;
			if (loop) {
				const c0 = x0 + pitch * cfg.pre, c1 = x0 + pitch * (cfg.pre + cfg.core - 1);
				const one = c1 - c0 < 1;
				SP.x0.t = c0 - (one ? 8 : 0);
				SP.x1.t = c1 + (one ? 8 : 0);
				SP.top.t = coreTop - 7 * this.k;
				SP.gx.t = (c0 + c1) / 2;
				SP.grx.t = (c1 - c0) / 2 + pitch * 1.6;
				if (SP.aA.v < 0.02 && SP.aS.v < 0.02) { SP.x0.set(SP.x0.t); SP.x1.set(SP.x1.t); SP.top.set(SP.top.t); }
				if (SP.gA.v < 0.02) { SP.gx.set(SP.gx.t); SP.grx.set(SP.grx.t); }
				if (cfg.order === 'layer') this.Vs = cfg.visits;
				else {
					// a loop that changes its count: the old label and head leave, the new ones draw in once the arc has moved
					if (!snap && this.Va && this.Va !== cfg.visits && SP.aA.v > 0.5) { SP.hA.set(0); this.hAat = now + tMove + MO.t3; }
					this.Va = cfg.visits;
				}
			}
			if (cfg.order === 'layer' && loop) this.layerOrder = { pre: cfg.pre, core: cfg.core, v: cfg.visits };
			this.want.aA = loop && cfg.order !== 'layer' ? 1 : 0;
			this.want.aS = loop && cfg.order === 'layer' ? 1 : 0;
			this.gate();
			SP.gA.t = loop ? 1 : 0;
			this.x0 = x0;
			this.n = n;
			this.fin = { cy, l0: SP.x0.t, r0: SP.x1.t, top: SP.top.t, a: this.arcA(SP.x0.t, SP.x1.t) };
			this.labHold = null;
			// when the drawing has settled enough for the light to start
			this.settle = snap ? 0 : tRise + born * (MO.stagger / 2) + MO.t3;
			if (snap) {
				for (const b of this.bars) { b.x.set(b.x.t); b.h.set(b.h.t); b.a.set(b.a.t); b.c.set(b.c.t); }
				SP.aA.t = this.want.aA; SP.aS.t = this.want.aS;
				for (const q in SP) SP[q].set(SP[q].t);
			}
			this.stopPulse();
		}
		/* the arc's height: grows with its span, and always leaves room for its label above it */
		arcA(l0, r0) { return Math.max(8 * this.k, Math.min((this.geo.top - 9 * this.k - 0.73 * this.fs) / 0.99, 12 * this.k + (r0 - l0) * 0.2)); }
		label(V) {
			const p = this.pulse;
			if (p && p.seg && p.seg.lab) return p.seg.lab;
			if (this.labHold && performance.now() < this.labHold.until) return this.labHold.text;
			return `×${V}`;
		}
		frame() {
			const { cy, pitch, bw, hU } = this.geo;
			const k = this.k;
			const rx = Math.min(bw / 2, this.o.rx != null ? this.o.rx : 5);
			this.bars.forEach((b) => {
				const a = clamp(b.a.v);
				if (a < 0.01 || b.h.v < 0.3) { if (b.vis !== false) { b.g.setAttribute('opacity', 0); b.vis = false; } return; }
				if (b.wasLeaving !== !!b.leaving) {
					// leaving layers are drawn under the others
					b.wasLeaving = !!b.leaving;
					for (const x of this.bars) if (!x.leaving) this.barG.append(x.g);
					for (const x of this.bars) if (x.leaving) this.barG.prepend(x.g);
				}
				b.vis = true;
				const hh = b.h.v;
				const ww = Math.max(2, (bw * hh) / hU);
				let yo = 0, po = 1;
				if (b.peel) {
					const u = clamp((b.x.v - b.peel.xs) / (b.peel.xd - b.peel.xs || 1));
					yo = -Math.min(b.peel.A, cy - hh / 2 - 2) * Math.sin(Math.PI * u);
					// a copy comes out of its layer as it lifts, and goes back into it as it lands there
					po = b.fold ? clamp((0.6 - u) * 4) : clamp((u - 0.08) * 8);
				}
				let x = b.x.v - ww / 2, y = cy - hh / 2 + yo, w = ww, h = hh;
				if (b.x.v === b.x.t && b.h.v === b.h.t && !yo) { x = snapPx(x); y = snapPx(y); w = snapPx(w); h = snapPx(h); }
				b.g.setAttribute('opacity', ((b.leaving ? a * clamp(b.h.v / (b.h0 || b.h.v)) : a) * po).toFixed(3));
				// the fill reaches the block's ends a little before the spring settles, so no ink rim lingers
				const c = clamp(b.c.v / 0.96), hc = h * c;
				const lit = Math.max(b.hit * b.hit, b.hov);
				b.el.setAttribute('x', x.toFixed(2));
				b.el.setAttribute('y', y.toFixed(2));
				b.el.setAttribute('width', w.toFixed(2));
				b.el.setAttribute('height', h.toFixed(2));
				b.el.setAttribute('rx', Math.min(rx, h / 2, w / 2).toFixed(2));
				b.el.setAttribute('opacity', c > 0.999 ? 0 : 1);
				b.el.setAttribute('fill', rgb(lerpL(INK, INKHI, lit)));
				b.ov.setAttribute('opacity', c < 0.001 ? 0 : 1);
				b.ov.setAttribute('x', x.toFixed(2));
				b.ov.setAttribute('y', (cy - hc / 2 + yo).toFixed(2));
				b.ov.setAttribute('width', w.toFixed(2));
				b.ov.setAttribute('height', hc.toFixed(2));
				b.ov.setAttribute('rx', Math.min(rx, hc / 2, w / 2).toFixed(2));
				b.ov.setAttribute('fill', rgb(lerpL(ACC, ACC2, lit)));
			});
			const SP = this.SP;
			const ys = Math.round(cy) + 0.5;
			this.shaft.setAttribute('x1', SP.s0.v);
			this.shaft.setAttribute('x2', SP.s1.v);
			this.shaft.setAttribute('y1', ys);
			this.shaft.setAttribute('y2', ys);
			this.shaft.setAttribute('opacity', clamp(SP.sh.v));
			const l0 = SP.x0.v, r0 = SP.x1.v, top = SP.top.v;
			const a = this.arcA(l0, r0);
			const hA = clamp(SP.hA.v);
			this.arc.setAttribute('d', arcPath(l0, r0, top, a));
			this.arc.setAttribute('stroke-dashoffset', (1 - clamp(SP.dA.v)).toFixed(4));
			this.head.setAttribute('d', headPath(l0, top, (this.o.hk || 1.5) * k));
			this.head.setAttribute('stroke-dashoffset', (1 - hA).toFixed(3));
			this.lab.setAttribute('x', (l0 + r0) / 2);
			this.lab.setAttribute('y', top - 0.99 * a - 7 * k + (1 - hA) * 4 * k);
			this.lab.setAttribute('opacity', hA.toFixed(3));
			if (this.Va) this.lab.textContent = this.want.aA ? this.label(this.Va) : `×${this.Va}`;
			this.arcG.setAttribute('opacity', clamp(SP.aA.v).toFixed(3));
			this.glow.setAttribute('cx', SP.gx.v);
			this.glow.setAttribute('cy', cy);
			this.glow.setAttribute('rx', Math.max(1, SP.grx.v));
			this.glow.setAttribute('ry', hU * 0.62);
			this.glow.setAttribute('opacity', clamp(SP.gA.v) * (this.o.glow != null ? this.o.glow : 1));
			const lo = this.layerOrder;
			this.small.setAttribute('opacity', clamp(SP.aS.v).toFixed(3));
			if (SP.aS.v > 0.01 && lo) {
				let topC = cy;
				for (let i = 0; i < lo.core; i++) topC = Math.min(topC, cy - this.bars[lo.pre + i].h.v / 2);
				const r = Math.min(13 * k, pitch * 0.27), t = topC - 7 * k, sw = (this.o.sw || 1.5) * k;
				while (this.smallEls.length < lo.core) this.smallEls.push([0, 1].map(() => s('path', { fill: 'none', stroke: 'var(--acc)', 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, this.small)));
				this.smallEls.forEach(([ap, hp], i) => {
					const on = i < lo.core;
					ap.style.display = hp.style.display = on ? '' : 'none';
					if (!on) return;
					const cx = this.bars[lo.pre + i].x.v;
					ap.setAttribute('d', `M${cx + r} ${t}C${cx + r} ${t - 26 * k} ${cx - r} ${t - 26 * k} ${cx - r} ${t}`);
					hp.setAttribute('d', headPath(cx - r, t, (this.o.hk || 1.5) * 0.85 * k));
					ap.setAttribute('stroke-width', sw);
					hp.setAttribute('stroke-width', sw);
				});
				this.smallLab.setAttribute('x', this.bars[lo.pre + lo.core - 1].x.v + r + 8 * k);
				this.smallLab.setAttribute('y', topC - 26 * k);
				this.smallLab.textContent = this.want.aS ? this.label(lo.v) : `×${lo.v}`;
				this.smallTop = t;
				this.smallR = r;
			}
			if (this.o.mechs) this.paintMechs();
			this.paintPulse();
		}
		mechKey(cfg) {
			if (!cfg) return '';
			const m = [];
			if (cfg.inputInjection && cfg.inputInjection !== 'none') m.push('in');
			if (cfg.initialState && cfg.initialState !== 'input') m.push('st');
			if (cfg.exitDistribution) m.push('ex');
			if (cfg.order === 'layer' && cfg.visits > 1) m.push('tie');
			return m.join(' ');
		}
		/* one mechanism drawing at a time: the old one leaves before the new one is drawn */
		paintMechs() {
			const SP = this.SP, want = this.mechKey(this.cfg);
			if (want !== this.mk) { SP.mA.t = 0; if (SP.mA.v < 0.04) { this.mk = want; this.mechG.textContent = ''; this.mechEls = null; } }
			else SP.mA.t = want ? 1 : 0;
			this.mechG.setAttribute('opacity', clamp(SP.mA.v).toFixed(3));
			if (!this.mk || SP.mA.v < 0.01 && SP.mA.t === 0) return;
			const { cy, bw } = this.geo, k = this.k, cfg = this.cfg, pitch = this.pitchN || this.geo.pitch;
			const g = this.mechG, B = this.bars;
			const c0 = cfg.pre, c1 = cfg.pre + cfg.core - 1;
			let hc = 0;
			for (let i = c0; i <= c1; i++) hc = Math.max(hc, B[i].h.v);
			const xe = B[c0].x.v - pitch / 2, xr = B[c1].x.v + pitch / 2;
			const low = cy + hc / 2;
			const fs = 11 * k;
			if (!this.mechEls) {
				const E = (this.mechEls = {});
				const T_ = (cls, anchor) => { const t = s('text', { class: cls, 'text-anchor': anchor || 'start', 'font-size': fs }, g); return t; };
				if (this.mk.includes('in')) {
					E.dash = s('path', { class: 'mk-dash' }, g);
					E.plus = s('circle', { class: 'mk-plus' }, g);
					E.plusX = s('path', { class: 'mk-line' }, g);
					E.inT = T_('mk-t');
					E.inT.textContent = 'input, at every visit';
				}
				if (this.mk.includes('st')) {
					E.st = s('circle', { class: 'mk-st' }, g);
					E.stDots = Array.from({ length: 7 }, () => s('circle', { class: 'mk-dot' }, g));
					E.stL = s('path', { class: 'mk-hair' }, g);
					E.stT = T_('mk-t', 'end');
					E.stT.textContent = `start: ${cfg.initialState}`;
				}
				if (this.mk.includes('ex')) {
					E.ex = s('path', { class: 'mk-line' }, g);
					E.exH = s('path', { class: 'mk-line' }, g);
					E.exT = T_('mk-a');
					E.exT.textContent = 'learned exit';
					E.exT2 = T_('mk-t');
					E.exT2.textContent = `after visit ${Array.from({ length: cfg.visits }, (_, i) => i + 1).join(', ').replace(/, (\d+)$/, ' or $1')}`;
				}
				if (this.mk.includes('tie')) {
					E.stems = Array.from({ length: cfg.core }, () => s('path', { class: 'mk-stem' }, g));
					E.bank = s('rect', { class: 'mk-bank' }, g);
					E.bankT = T_('mk-w', 'middle');
					E.bankT.textContent = 'one expert bank';
				}
			}
			const E = this.mechEls;
			if (E.plus) {
				const r = 5 * k, yb = Math.min(this.geo.H - 4 * k - fs, low + 12 * k);
				E.plus.setAttribute('cx', xe); E.plus.setAttribute('cy', cy); E.plus.setAttribute('r', r);
				E.plusX.setAttribute('d', `M${xe - r * 0.55} ${cy}h${r * 1.1}M${xe} ${cy - r * 0.55}v${r * 1.1}`);
				E.dash.setAttribute('d', `M${xe} ${cy + r + 2 * k}V${yb}`);
				E.inT.setAttribute('x', xe + 6 * k); E.inT.setAttribute('y', yb + 0.35 * fs);
			}
			if (E.st) {
				const r = 11 * k, yc = cy - hc / 2 - r - 4 * k, x = xe - pitch * 0.9;
				E.st.setAttribute('cx', x); E.st.setAttribute('cy', yc); E.st.setAttribute('r', r);
				const P_ = [[-.4, -.35], [.2, -.5], [-.1, .1], [.45, .15], [-.5, .35], [.1, .55], [.5, -.2]];
				E.stDots.forEach((d, i) => { d.setAttribute('cx', x + P_[i][0] * r); d.setAttribute('cy', yc + P_[i][1] * r); d.setAttribute('r', 1.1 * k); });
				E.stL.setAttribute('d', `M${x + r * 0.8} ${yc + r * 0.6}Q${xe} ${yc + r} ${xe} ${cy - 6 * k}`);
				E.stT.setAttribute('x', x - r - 5 * k); E.stT.setAttribute('y', yc + 0.35 * fs);
			}
			if (E.ex) {
				// the token may leave the loop after any visit: an exit from the end of the loop, beside the arc's way back
				const x1 = xr + pitch * 1.1, hk = 4.5 * k;
				E.ex.setAttribute('d', `M${xr - pitch * 0.5 + bw * 0.6} ${cy}H${x1}`);
				E.exH.setAttribute('d', `M${x1 - hk} ${cy - hk}L${x1} ${cy}L${x1 - hk} ${cy + hk}`);
				E.exT.setAttribute('x', xr + 4 * k); E.exT.setAttribute('y', cy - 8 * k);
				E.exT2.setAttribute('x', xr + 4 * k); E.exT2.setAttribute('y', cy + 8 * k + fs * 0.75);
			}
			if (E.bank) {
				const hb = 14 * k, yb = Math.min(this.geo.H - hb - 2 * k, low + 12 * k);
				const xa = B[c0].x.v - bw, xb = B[c1].x.v + bw;
				E.bank.setAttribute('x', xa); E.bank.setAttribute('y', yb); E.bank.setAttribute('width', xb - xa); E.bank.setAttribute('height', hb); E.bank.setAttribute('rx', 4 * k);
				E.bankT.setAttribute('x', (xa + xb) / 2); E.bankT.setAttribute('y', yb + hb / 2 + 0.35 * fs);
				E.stems.forEach((p, i) => { const x = B[c0 + i].x.v, yt = cy + B[c0 + i].h.v / 2 + 1.5 * k; p.setAttribute('d', `M${x} ${yt}V${yb}`); });
			}
		}
		kick() {
			if (this.running) return;
			this.running = true;
			animate((dt, now) => {
				let mv = false;
				this.gate();
				for (const b of this.bars) {
					if (b.leaving && b.h.v < 0.5) { b.leaving = false; b.a.set(0); b.h.set(0); mv = true; }
					if (b.at) { mv = true; if (now >= b.at) { b.at = 0; b.h.t = b.hT; } }
					if (b.atX) { mv = true; if (now >= b.atX) { b.atX = 0; b.x.t = b.xT; } }
					if (b.atC) { mv = true; if (now >= b.atC) { b.atC = 0; b.c.t = 1; } }
					if (b.peel) { mv = true; if (!b.atX && Math.abs(b.x.v - b.peel.xd) < 0.5) { b.peel = null; if (b.fold) { b.fold = false; b.a.set(0); b.h.set(0); b.a.t = 0; b.h.t = 0; } } }
					mv = b.x.step(dt) | mv; mv = b.h.step(dt) | mv; mv = b.a.step(dt) | mv; mv = b.c.step(dt) | mv;
					if (b.hit > 0) { b.hit = Math.max(0, b.hit - dt * 2); mv = true; }
				}
				if (this.hAat) { mv = true; if (now >= this.hAat) { this.hAat = 0; this.SP.hA.t = 1; } }
				if (this.drawing && this.SP.dA.v > 0.85) { this.drawing = false; this.SP.hA.t = 1; }
				for (const q in this.SP) mv = this.SP[q].step(dt) | mv;
				if (this.pulse) { this.stepPulse(now); mv = true; }
				if (this.labHold && now < this.labHold.until + 40) mv = true;
				this.frame();
				if (!mv) this.running = false;
				return mv;
			});
		}
		/* the token's path: forward through every stored layer, back along the arc for every further visit */
		route() {
			const cfg = this.cfg;
			const seq = [];
			for (let i = 0; i < cfg.pre; i++) seq.push(i);
			if (cfg.visits > 1 && cfg.order === 'layer') {
				for (let i = 0; i < cfg.core; i++) for (let v = 0; v < cfg.visits; v++) seq.push(cfg.pre + i);
			} else {
				for (let v = 0; v < cfg.visits; v++) for (let i = 0; i < cfg.core; i++) seq.push(cfg.pre + i);
			}
			for (let i = 0; i < cfg.suf; i++) seq.push(cfg.pre + cfg.core + i);
			return seq;
		}
		startPulse(delay = 0) {
			if (reduced || !this.cfg) return;
			const cfg = this.cfg;
			const seq = this.route();
			const segs = [];
			const fwd = this.o.fwd || 70;
			const V = cfg.visits;
			const inCore = (i) => V > 1 && i >= cfg.pre && i < cfg.pre + cfg.core;
			// the arc label counts the visit the light is on: true at every frame
			let v = 0;
			const lab = (kind, a, b) => {
				if (!inCore(b)) { v = 0; return null; }
				if (cfg.order === 'layer') v = kind === 'self' ? v + 1 : 1;
				else v = kind === 'ret' ? v + 1 : inCore(a) ? v : 1;
				return `${v} / ${V}`;
			};
			let t = delay;
			segs.push({ kind: 'in', a: -1, b: seq[0], t0: t, d: fwd * 1.6, lab: lab('in', -1, seq[0]) });
			t += fwd * 1.6;
			for (let i = 1; i < seq.length; i++) {
				const a = seq[i - 1], b = seq[i];
				const kind = b > a ? 'fwd' : b === a ? 'self' : 'ret';
				const d = kind === 'fwd' ? fwd * (b - a) : kind === 'self' ? 360 : 540;
				segs.push({ kind, a, b, t0: t, d, lab: lab(kind, a, b) });
				t += d;
			}
			segs.push({ kind: 'out', a: seq[seq.length - 1], b: -2, t0: t, d: fwd * 1.6, lab: null });
			t += fwd * 1.6;
			this.pulse = { segs, start: performance.now(), end: t, k: -1 };
			this.kick();
		}
		stopPulse() {
			if (!this.pulse) return;
			this.pulse = null;
			this.pulseG.setAttribute('opacity', 0);
			this.trail.length = 0;
			this.paintTrail(0, 0);
		}
		stepPulse(now) {
			const p = this.pulse;
			const ms = now - p.start;
			if (ms > p.end + 200) { this.stopPulse(); if (this.o.onPulseEnd) this.o.onPulseEnd(); return; }
			if (ms < p.segs[0].t0) { p.seg = null; return; }
			let seg = p.segs[0];
			for (const sg of p.segs) if (ms >= sg.t0) seg = sg;
			const idx = p.segs.indexOf(seg);
			if (idx !== p.k) {
				// arriving at a layer: it lights
				for (let j = p.k + 1; j <= idx; j++) {
					const sg = p.segs[j];
					if (sg.kind !== 'in' && sg.a >= 0) this.bars[sg.a].hit = 1;
				}
				const prev = p.segs[p.k];
				if (prev && prev.lab && !seg.lab) this.labHold = { text: prev.lab, until: now + MO.t2 };
				p.k = idx;
			}
			p.seg = seg;
			p.u = clamp((ms - seg.t0) / seg.d);
		}
		paintTrail(x, y) {
			const now = performance.now();
			const tr = this.trail;
			const span = MO.t2;
			while (tr.length && now - tr[0][2] > span) tr.shift();
			const i0 = Math.max(1, tr.length - this.trailEls.length);
			this.trailEls.forEach((el, j) => {
				const i = i0 + j;
				if (i >= tr.length) { el.setAttribute('opacity', 0); return; }
				const age = (now - tr[i][2]) / span;
				el.setAttribute('x1', tr[i - 1][0]); el.setAttribute('y1', tr[i - 1][1]);
				el.setAttribute('x2', tr[i][0]); el.setAttribute('y2', tr[i][1]);
				el.setAttribute('stroke-width', (4 * (1 - age) + 1) * this.k);
				el.setAttribute('opacity', (0.7 * (1 - age) * (this.pulseOp || 1)).toFixed(3));
			});
		}
		paintPulse() {
			const p = this.pulse;
			if (!p || !p.seg) { this.pulseG.setAttribute('opacity', 0); return; }
			const { cy, pitch } = this.geo;
			const xs = (i) => (i === -1 ? this.SP.s0.v : i === -2 ? this.SP.s1.v : this.bars[i].x.v);
			const sg = p.seg;
			const u = p.u;
			let x, y, op = 1;
			if (sg.kind === 'in' || sg.kind === 'out' || sg.kind === 'fwd') {
				const e = sg.kind === 'fwd' ? easeIO(u) : u;
				x = lerp(xs(sg.a), xs(sg.b), e);
				y = cy;
				if (sg.kind === 'in') op = clamp(u * 3);
				if (sg.kind === 'out') op = clamp((1 - u) * 3);
			} else if (sg.kind === 'ret') {
				const l0 = this.SP.x0.v, r0 = this.SP.x1.v, top = this.SP.top.v, a = this.arcA(l0, r0);
				const xr = xs(sg.a), xl = xs(sg.b);
				const e = easeIO(u);
				if (e < 0.2) { const v = e / 0.2; x = lerp(xr, r0, v); y = lerp(cy, top, v); }
				else if (e < 0.8) { const v = (e - 0.2) / 0.6; [x, y] = bez([r0, top], [r0 + 3, top - a * 1.32], [l0 - 3, top - a * 1.32], [l0, top], v); }
				else { const v = (e - 0.8) / 0.2; x = lerp(l0, xl, v); y = lerp(top, cy, v); }
			} else {
				const cx = xs(sg.a), r = this.smallR || pitch * 0.3, t = this.smallTop != null ? this.smallTop : cy - 40;
				const e = easeIO(u);
				if (e < 0.2) { const v = e / 0.2; x = lerp(cx, cx + r, v); y = lerp(cy, t, v); }
				else if (e < 0.8) { const v = (e - 0.2) / 0.6; [x, y] = bez([cx + r, t], [cx + r, t - 26 * this.k], [cx - r, t - 26 * this.k], [cx - r, t], v); }
				else { const v = (e - 0.8) / 0.2; x = lerp(cx - r, cx, v); y = lerp(t, cy, v); }
			}
			this.pulseG.setAttribute('opacity', op);
			this.pulseOp = op;
			// the light leaves a short fading trail along the path it took
			this.trail.push([x, y, performance.now()]);
			this.paintTrail(x, y);
			this.dot.setAttribute('cx', x);
			this.dot.setAttribute('cy', y);
			this.halo.setAttribute('cx', x);
			this.halo.setAttribute('cy', y);
		}
		hitTest(px, py) {
			const { cy, pitch } = this.geo;
			let best = null;
			for (let i = 0; i < this.n; i++) {
				const b = this.bars[i];
				if (Math.abs(px - b.x.v) <= pitch / 2 && Math.abs(py - cy) <= Math.max(b.h.v / 2, 14) + 6) best = b;
			}
			return best;
		}
	}
	/* A static model in its own viewBox, for small figures. All on the page-wide height scale. */
	function miniModel(cfg, o = {}) {
		const slots = o.slots || 18;
		const pitch = 10, bw = o.bw || 8.6, hU = o.hU || 60, top = o.top || 22;
		const W = slots * pitch + 8, H = top + hU + 4;
		const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'model model--mini', 'aria-hidden': 'true', preserveAspectRatio: 'xMidYMid meet' });
		const L = layers(cfg), n = L.length;
		const cy = top + hU / 2;
		const x0 = W / 2 - (pitch * (n - 1)) / 2;
		const loop = cfg.visits > 1;
		s('line', { x1: x0 - pitch * 0.7, x2: x0 + pitch * (n - 0.3), y1: cy, y2: cy, stroke: 'var(--hair)', 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke' }, svg);
		let coreTop = cy;
		L.forEach((l, i) => {
			const k = Math.sqrt(l.P / PMAX);
			const hh = Math.max(1.5, hU * k), ww = Math.max(1.2, bw * k);
			const acc = loop && l.region === 'core';
			if (l.region === 'core') coreTop = Math.min(coreTop, cy - hh / 2);
			s('rect', { x: x0 + pitch * i - ww / 2, y: cy - hh / 2, width: ww, height: hh, rx: Math.min(1.6, hh / 2, ww / 2), fill: acc ? 'var(--acc)' : 'var(--ink)' }, svg);
		});
		if (loop) {
			const t = coreTop - 3;
			if (cfg.order === 'layer') {
				for (let i = 0; i < cfg.core; i++) {
					const cx = x0 + pitch * (cfg.pre + i), r = 3.2;
					s('path', { d: `M${cx + r} ${t}C${cx + r} ${t - 10} ${cx - r} ${t - 10} ${cx - r} ${t}`, fill: 'none', stroke: 'var(--acc)', 'stroke-width': 1.25, 'vector-effect': 'non-scaling-stroke' }, svg);
				}
			} else {
				const c0 = x0 + pitch * cfg.pre, c1 = x0 + pitch * (cfg.pre + cfg.core - 1);
				const one = c1 - c0 < 1;
				const l0 = c0 - (one ? 4 : 0), r0 = c1 + (one ? 4 : 0);
				const a = Math.min(top - 6, 6 + (r0 - l0) * 0.22);
				s('path', { d: arcPath(l0, r0, t, a), fill: 'none', stroke: 'var(--acc)', 'stroke-width': 1.25, 'stroke-linecap': 'round', 'vector-effect': 'non-scaling-stroke' }, svg);
				s('path', { d: headPath(l0, t, 0.7), fill: 'none', stroke: 'var(--acc)', 'stroke-width': 1.25, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'vector-effect': 'non-scaling-stroke' }, svg);
			}
		}
		return svg;
	}

	/* ---------------------------------------------------------------- a segmented control with a sliding highlight */
	function seg(box, labels, pick) {
		const hl = h('span', 'seg__hl', box);
		hl.setAttribute('aria-hidden', 'true');
		let cur = 0;
		const bs = labels.map((lab, i) => {
			const b = h('button', 'seg__b', box);
			b.type = 'button';
			b.setAttribute('role', 'radio');
			b.setAttribute('aria-checked', 'false');
			h('span', '', b, lab);
			b.addEventListener('click', () => pick(i, true));
			b.addEventListener('keydown', (e) => {
				let j = -1;
				if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = (i + 1) % labels.length;
				if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = (i - 1 + labels.length) % labels.length;
				if (e.key === 'Home') j = 0;
				if (e.key === 'End') j = labels.length - 1;
				if (j < 0) return;
				e.preventDefault();
				pick(j, true);
				bs[j].focus();
			});
			return b;
		});
		// only a choice slides the highlight; a first placement, a resize or a font load puts it in place
		let placed = false;
		const place = (slide) => {
			const b = bs[cur];
			if (!slide || !placed) hl.style.transition = 'none';
			hl.style.width = `${b.offsetWidth}px`;
			hl.style.transform = `translate(${b.offsetLeft}px, ${b.offsetTop}px)`;
			if (!slide || !placed) { void hl.offsetWidth; hl.style.transition = ''; }
			placed = true;
			box.classList.toggle('is-over', box.scrollWidth > box.clientWidth + 1);
		};
		onResize(box, () => place(false));
		if (document.fonts) document.fonts.ready.then(() => place(false));
		return {
			bs,
			set(i, slide) {
				cur = i;
				bs.forEach((b, j) => { b.setAttribute('aria-checked', j === i ? 'true' : 'false'); b.tabIndex = j === i ? 0 : -1; });
				place(slide);
				const b = bs[i];
				if (box.scrollWidth > box.clientWidth + 1) box.scrollTo({ left: b.offsetLeft - (box.clientWidth - b.offsetWidth) / 2, behavior: reduced ? 'auto' : 'smooth' });
			}
		};
	}

	/* ---------------------------------------------------------------- gauges: budget against the standard MoE */
	/* one tolerance band with the design's three dots: the budget state, read at a glance */
	function bandSvg(host) {
		const svg = s('svg', { viewBox: '0 0 72 12', preserveAspectRatio: 'none' }, host);
		s('line', { x1: 0, x2: 72, y1: 6, y2: 6, stroke: 'var(--line)', 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke' }, svg);
		s('rect', { class: 'ob__band', x: 36 - 22.5, y: 1, width: 45, height: 10, rx: 5 }, svg);
		s('line', { class: 'ob__z', x1: 36, x2: 36, y1: 1, y2: 11 }, svg);
		return BKEYS.map(() => s('circle', { class: 'ob__d', cx: 36, cy: 6, r: 2.5 }, svg));
	}
	function gauges(box, o = {}) {
		const DOM = o.dom || 0.08;
		const pos = (v) => `${(clamp(v, -DOM, DOM) / DOM) * 50 + 50}%`;
		const sum = o.sum ? bandSvg(o.sum) : null;
		const rows = BKEYS.map((k, i) => {
			const g = h('div', 'gauge', box);
			h('span', 'gauge__n', g, (o.long ? T.budgetLong : T.budget)[k]);
			const v = h('span', 'gauge__v', g, '0%');
			const tr = h('div', 'gtrack', g);
			const band = h('span', 'gtrack__band', tr);
			band.style.left = pos(-0.05);
			band.style.width = `${(0.05 / DOM) * 100}%`;
			h('span', 'gtrack__z', tr).style.left = '50%';
			const dot = h('span', 'gtrack__dot', tr);
			// past the end of the axis: a chevron at the track end, the dot stays a dot
			const off = s('svg', { class: 'gtrack__off', viewBox: '0 0 12 12', 'aria-hidden': 'true' }, tr);
			s('path', { d: 'M4.5 3L7.5 6l-3 3' }, off);
			let note = null;
			if (o.axis) {
				const ax = h('div', 'gtrack__ax', g);
				for (const t of [-0.05, 0, 0.05]) h('span', '', ax, t ? `${t > 0 ? '+' : '−'}5%` : '0').style.left = pos(t);
				note = h('span', 'gtrack__note', ax, 'outside the band');
			}
			return { k, dot, v, off, note, sp: new Spring(0), tol: 0.05 };
		});
		// the number always states the true value; only the dot travels
		const paint = () => rows.forEach((r) => {
			r.dot.style.left = pos(r.sp.v);
			const out = Math.abs(r.sp.t) > r.tol;
			r.dot.classList.toggle('is-out', out);
			r.v.textContent = pct(r.sp.t);
			r.v.classList.toggle('is-out', out);
			r.off.classList.toggle('is-on', Math.abs(r.sp.v) > DOM * 0.98);
			if (r.note) r.note.classList.toggle('is-on', out);
			if (sum) { const d = sum[rows.indexOf(r)]; d.setAttribute('cx', (36 + (clamp(r.sp.v, -DOM, DOM) / DOM) * 36).toFixed(2)); d.classList.toggle('is-out', out); }
		});
		const setT = (cfg, snap) => rows.forEach((r) => {
			const d = cfg.budgets[r.k];
			r.sp.t = d.rel;
			r.tol = d.tol;
			if (snap) r.sp.set(d.rel);
		});
		let running = false;
		const kick = () => {
			if (running) return;
			running = true;
			animate((dt) => {
				let mv = false;
				for (const r of rows) mv = r.sp.step(dt) | mv;
				paint();
				if (!mv) running = false;
				return mv;
			});
		};
		return {
			rows, paint, setT, kick,
			go(cfg, snap) { setT(cfg, snap || reduced); if (snap || reduced) paint(); else kick(); },
			from0(cfg) { setT(cfg); rows.forEach((r) => { r.sp.v = 0; r.sp.vel = 0; }); paint(); }
		};
	}
	const within = (cfg) => BKEYS.every((k) => Math.abs(cfg.budgets[k].rel) <= cfg.budgets[k].tol);

	/* ---------------------------------------------------------------- the mark: the shell
	   The rule of the identity files (landing/mark2b/final-shell/shell.py): one spiral r = 2^(t / 2pi), every turn twice
	   the one before; the ink is the band between the spiral scaled by 2^gap and by 2, from `turns` turns back to the
	   aperture ray at 45 degrees. Growing it means drawing the band from the pole out to the parameter T: every frame is
	   the true shell up to T, in the finished mark's own frame, so it grows in place, turn on turn. */
	const SHELL = { medium: { turns: 2.0, gap: 0.24 }, small: { turns: 1.35, gap: 0.26 }, display: { turns: 3.0, gap: 0.18 } };
	function shellGrower(svg, master) {
		const { turns, gap } = SHELL[master];
		const TAU = 2 * Math.PI, ROT = Math.PI / 4, cIn = 2 ** gap, cOut = 2;
		const sp = (t) => { const r = 2 ** (t / TAU); return [r * Math.cos(ROT + t), r * Math.sin(ROT + t)]; };
		const T0 = -turns * TAU, T1 = 0;
		const band = (T, n) => {
			const pts = [];
			for (let i = 0; i <= n; i++) pts.push(sp(T0 + ((T - T0) * i) / n));
			return pts.map(([x, y]) => [x * cOut, y * cOut]).concat(pts.reverse().map(([x, y]) => [x * cIn, y * cIn]));
		};
		const full = band(T1, 360);
		const xs = full.map((p) => p[0]), ys = full.map((p) => p[1]);
		const x0 = Math.min(...xs), y1 = Math.max(...ys), sc = 100 / (y1 - Math.min(...ys));
		// the growing shell is drawn as its own path over the mark; the mark itself (brand/mark.svg) shows again at the end
		const use = svg.querySelector('use');
		const path = svg.appendChild(document.createElementNS('http://www.w3.org/2000/svg', 'path'));
		if (use) use.style.visibility = 'hidden';
		// the first frame that shows is already a curl (seven tenths of the innermost turn), never a loose arc
		const r0 = 2 ** ((T0 + 0.7 * TAU) / TAU), r1 = 2 ** (T1 / TAU);
		return (p) => {
			if (p >= 1) { path.remove(); if (use) use.style.visibility = ''; return; }
			if (p <= 0) { path.setAttribute('d', ''); return; }
			// p is the share of the spiral's length drawn: the length of an equiangular spiral grows with its radius
			const T = TAU * Math.log2(r0 + p * (r1 - r0));
			const pts = band(T, Math.max(24, Math.round(240 * p)));
			path.setAttribute('d', 'M' + pts.map(([x, y]) => `${((x - x0) * sc).toFixed(2)} ${((y1 - y) * sc).toFixed(2)}`).join('L') + 'Z');
		};
	}

	/* Every five to seven seconds (a little irregular, so it surprises rather than ticks) the header shell winds back to
	   its pole and grows again, as at the opening; never while the tab is hidden, and not at all for reduced motion. */
	function markAgain(svg) {
		const BACK = 600;
		const again = () => setTimeout(once, 5000 + Math.random() * 2000);
		const once = () => {
			if (document.hidden) return again();
			const grow = shellGrower(svg, svg.dataset.mark), t0 = performance.now();
			animate((dt, now) => {
				const ms = now - t0;
				if (ms < BACK) { grow(Math.max(0.001, 1 - easeIn(ms / BACK))); return true; }
				const p = clamp((ms - BACK) / MO.t5);
				grow(p >= 1 ? 1 : easeOut(p));
				if (p >= 1) { again(); return false; }
				return true;
			});
		};
		again();
	}

	/* ---------------------------------------------------------------- the opening
	   Two things, one clock, about 1.2 s. The shell in the header grows from its pole along its own spiral (760 ms).
	   Beside it the card comes in; its layers rise out of the shaft in one wave from the loop outward; as the loop's
	   layers stand, its arc draws from foot to head and the indigo fills them from the shaft; the readouts arrive with
	   the layers. Every frame is a true state of both drawings. */
	function opening(model, slots, ctl) {
		// the header mark grows if the mark knows how (data-grow="shell"); any other mark simply fades in with the card
		const hm = document.querySelector('[data-mark][data-grow="shell"]');
		const grow = hm ? shellGrower(hm, hm.dataset.mark) : (p) => { const m = document.querySelector('.lockup__mark'); m.style.opacity = p >= 1 ? '' : p.toFixed(3); };
		grow(0);
		ctl.chrome(0);
		document.documentElement.classList.remove('is-opening');
		const n = slots.length, mid = (n - 1) / 2, STAND = MO.t1;
		ctl.stand = STAND + MO.stagger;
		const t0 = performance.now();
		model.bars.slice(0, n).forEach((b, i) => { b.hT = slots[i].h; b.at = t0 + STAND + Math.abs(i - mid) * (MO.stagger / 2); });
		const coreLast = Math.max(...slots.map((s_, i) => (s_.acc ? Math.abs(i - mid) : 0)));
		const T_ARC = STAND + coreLast * (MO.stagger / 2) + MO.t2, T_END = T_ARC + MO.t5;
		let shown = false, arced = false;
		model.kick();
		animate((dt, now) => {
			const ms = now - t0;
			grow(easeOut(clamp(ms / MO.t5)));
			ctl.chrome(ms);
			if (!shown) { shown = true; ctl.show(); }
			if (!arced && ms > T_ARC) { arced = true; ctl.arc(); }
			if (ms > T_END && ms > MO.t5) { grow(1); ctl.chrome(Infinity); ctl.end(); if (hm && !reduced) markAgain(hm); return false; }
			return true;
		});
	}

	/* ---------------------------------------------------------------- 0. the instrument on the first screen */
	function hero() {
		const fig = document.querySelector('[data-hero]');
		const keys = T.hero;
		const chipsBox = fig.querySelector('[data-chips]');
		const stage = fig.querySelector('[data-stage]');
		const svg = fig.querySelector('[data-svg]');
		const tip = fig.querySelector('[data-tip]');
		const idBox = fig.querySelector('.inst__id');
		const elName = fig.querySelector('[data-name]');
		const elTags = fig.querySelector('[data-tags]');
		const elStored = fig.querySelector('[data-stored]');
		const elRun = fig.querySelector('[data-run]');
		const ok = fig.querySelector('[data-ok]');
		const okT = fig.querySelector('[data-ok-t]');
		const autoB = fig.querySelector('[data-auto]');
		const G = gauges(fig.querySelector('[data-gauges]'), { sum: fig.querySelector('[data-ok-b]') });
		const model = new Model(svg, {
			slots: Math.max(...keys.map((k) => layers(C[k]).length)),
			topF: 0.2, topMax: 80, fs: 15, bottom: 44, fill: 0.74, fitP: Math.max(...keys.map((k) => pOf(C[k]))), mechs: true, spread: true,
			onPulseEnd() { if (auto && armed) schedule(); }
		});
		const S = seg(chipsBox, keys.map((k) => T.heroShort[k] || T.short[k]), (i) => { stop(); go(i); });
		const chips = S.bs;
		chips.forEach((b, i) => { b.title = T.design[keys[i]]; b.prog = h('span', 'seg__prog', b); });
		let cur = keys.indexOf(T.heroStart);
		function text(k = keys[cur], quiet) {
			const cfg = C[k];
			elName.textContent = T.design[k];
			refMark(elName, [arx(cfg.source)]);
			elTags.textContent = '';
			const tags = mechs(cfg);
			if (!cfg.parent) h('span', 'tagm tagm--n', elTags, 'the reference').style.setProperty('--i', 0);
			if (cfg.visits === 1 && cfg.parent) h('span', 'tagm tagm--n', elTags, 'no loop').style.setProperty('--i', 0);
			tags.forEach((t, i) => h('span', 'tagm', elTags, t).style.setProperty('--i', i));
			if (quiet) elTags.querySelectorAll('.tagm').forEach((t) => (t.style.animation = 'none'));
			const stored = cfg.pre + cfg.core + cfg.suf, run = cfg.pre + cfg.core * cfg.visits + cfg.suf;
			elStored.textContent = stored;
			elRun.textContent = run;
			elRun.classList.toggle('is-loop', run > stored);
			const ref = !cfg.parent;
			ok.classList.toggle('is-ref', ref);
			ok.classList.toggle('is-out', !ref && !within(cfg));
			okT.textContent = ref ? 'reference' : within(cfg) ? 'within ±5%' : 'outside ±5%';
		}
		/* the id block takes the height of the tallest design, so the card never changes size on a switch */
		function lockId() {
			idBox.style.minHeight = '';
			let hMax = 0;
			for (const k of keys) { text(k, true); hMax = Math.max(hMax, idBox.offsetHeight); }
			idBox.style.minHeight = `${hMax}px`;
			text(keys[cur], true);
		}
		function paintChips(slide) {
			S.set(cur, slide);
			chips.forEach((c) => {
				c.prog.style.transition = 'none';
				c.prog.style.transform = 'scaleX(0)';
			});
		}
		function go(i, snap) {
			cur = i;
			text();
			paintChips(!snap);
			model.set(C[keys[cur]], reduced || snap);
			G.go(C[keys[cur]], snap);
			if (reduced || snap) model.frame(); else { model.kick(); model.startPulse(Math.max(MO.t4, model.settle)); }
		}
		// the tour: each design shows its pulse, then the next one; stopped for good by a choice, paused on hover or focus
		let auto = !reduced, armed = false, hover = false, visible = true, timer = 0;
		const DWELL = 1600;
		function schedule() {
			clearTimeout(timer);
			const c = chips[(cur + 1) % keys.length];
			c.prog.style.transition = 'none';
			c.prog.style.transform = 'scaleX(0)';
			if (!auto || !armed || hover || !visible || document.hidden) return;
			void c.prog.offsetWidth;
			c.prog.style.transition = `transform ${DWELL}ms linear`;
			c.prog.style.transform = 'scaleX(1)';
			timer = setTimeout(() => go((cur + 1) % keys.length), DWELL);
		}
		function stop() {
			idBox.setAttribute('aria-live', 'polite');
			auto = false;
			clearTimeout(timer);
			autoB.classList.add('is-paused');
			autoB.setAttribute('aria-label', 'Play the tour');
			autoB.title = 'Play the tour';
			chips.forEach((c) => { c.prog.style.transition = 'none'; c.prog.style.transform = 'scaleX(0)'; });
		}
		autoB.addEventListener('click', () => {
			if (auto) { stop(); return; }
			auto = true;
			armed = true;
			autoB.classList.remove('is-paused');
			autoB.setAttribute('aria-label', 'Pause the tour');
			autoB.title = 'Pause the tour';
			go((cur + 1) % keys.length);
		});
		if (reduced) { autoB.hidden = true; }
		fig.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') { hover = true; schedule(); } });
		fig.addEventListener('pointerleave', () => { hover = fig.contains(document.activeElement) && document.activeElement.matches(':focus-visible'); if (!model.pulse) schedule(); });
		fig.addEventListener('focusin', () => { hover = true; schedule(); });
		fig.addEventListener('focusout', (e) => { if (fig.contains(e.relatedTarget)) return; hover = false; if (!model.pulse) schedule(); });
		new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (!model.pulse) schedule(); }, { threshold: 0.3 }).observe(fig);
		document.addEventListener('visibilitychange', () => { if (!model.pulse) schedule(); });
		// a layer under the pointer says what it is: above the layer, or below it when the arc is overhead
		let hovB = null;
		const unhover = () => { if (hovB) { hovB.hov = 0; hovB = null; model.kick(); } tip.classList.remove('is-on'); };
		stage.addEventListener('pointermove', (e) => {
			const r = svg.getBoundingClientRect();
			const z = r.width / (svg.clientWidth || r.width) || 1;
			const b = model.hitTest((e.clientX - r.left) / z, (e.clientY - r.top) / z);
			if (!b || !b.L) { unhover(); return; }
			const cfg = model.cfg;
			const n = layers(cfg).length;
			const loop = cfg.visits > 1 && b.L.region === 'core';
			const role = cfg.visits > 1 ? (loop ? `looped, ${cfg.visits} visits` : b.L.region === 'prefix' ? 'before the loop' : 'after the loop') : 'run once';
			tip.innerHTML = `<b>Layer ${b.i + 1} of ${n}</b> · ${role}<br><span>${b.L.E} experts · top-${cfg.topK} · ${b.L.P.toFixed(0)}M expert parameters</span>`;
			tip.classList.toggle('is-below', loop);
			tip.classList.add('is-on');
			tip.style.left = `${b.x.v}px`;
			tip.style.top = `${loop ? model.geo.cy + b.h.v / 2 : model.geo.cy - b.h.v / 2}px`;
			if (hovB !== b) { if (hovB) hovB.hov = 0; hovB = b; b.hov = 0.35; model.kick(); }
		});
		stage.addEventListener('pointerleave', unhover);
		go(cur, true);
		let entering = !reduced;
		const relock = () => { if (document.fonts) document.fonts.ready.then(lockId); else lockId(); };
		relock();
		let lw = fig.clientWidth;
		new ResizeObserver(() => { if (fig.clientWidth !== lw) { lw = fig.clientWidth; lockId(); } }).observe(fig);
		onResize(stage, () => { model.measure(); if (entering) return; model.set(C[keys[cur]], true); model.frame(); });
		if (entering) {
			fig.classList.add('is-wait');
			model.bars.forEach((b) => b.h.set(0));
			model.SP.sh.set(0); model.SP.aA.set(0); model.SP.gA.set(0);
			G.from0(C[keys[cur]]);
			model.frame();
		}
		hero.start = () => {
			if (reduced) return;
			model.measure();
			model.set(C[keys[cur]], true);
			const n = layers(C[keys[cur]]).length;
			const slots = model.bars.slice(0, n).map((b) => ({ x: b.x.t, h: b.h.t, w: Math.max(2, (model.geo.bw * b.h.t) / model.geo.hU), acc: b.c.t > 0.5 }));
			const SP = model.SP;
			const want = { ...model.want };
			const s0t = SP.s0.t, s1t = SP.s1.t, mid = (s0t + s1t) / 2;
			model.bars.forEach((b) => { b.h.set(0); b.c.set(0); });
			model.want.aA = model.want.aS = 0;
			SP.s0.set(mid); SP.s1.set(mid); SP.sh.set(0); SP.aA.set(0); SP.aS.set(0); SP.gA.set(0); SP.hA.set(0);
			G.from0(C[keys[cur]]);
			model.frame();
			// one sequence, driven by one clock: the card arrives with the mark; its controls and readouts arrive
			// with the layers as they stand, one after another
			const chrome = [...fig.children].filter((c) => !c.classList.contains('inst__stage'));
			let gk = false;
			fig.classList.remove('is-wait');
			fig.style.transition = 'none';
			chrome.forEach((c) => (c.style.transition = 'none'));
			const ctl = {
				chrome(ms) {
					if (ms === Infinity) { [fig, ...chrome].forEach((c) => ['opacity', 'translate', 'transition'].forEach((q) => c.style.removeProperty(q))); return; }
					fig.style.opacity = easeOut(clamp(ms / MO.t4)).toFixed(3);
					chrome.forEach((c, i) => {
						const k = easeOut(clamp((ms - ctl.stand - i * MO.stagger) / MO.t3));
						c.style.opacity = k.toFixed(3);
						c.style.translate = `0 ${((1 - k) * 6).toFixed(2)}px`;
					});
					if (!gk && ms > ctl.stand + 2 * MO.stagger) { gk = true; G.kick(); }
				},
				show() { SP.sh.set(1); SP.s0.t = s0t; SP.s1.t = s1t; model.kick(); },
				land(i) { model.bars[i].h.set(slots[i].h); if (!ctl.arced) model.bars[i].c.set(0); model.frame(); },
				arc() {
					ctl.arced = true;
					Object.assign(model.want, want);
					SP.aA.set(want.aA); SP.aS.set(want.aS);
					SP.dA.set(0); SP.dA.t = 1; model.drawing = true; SP.gA.t = 1;
					const now = performance.now();
					slots.map((s_, i) => i).filter((i) => slots[i].acc).reverse().forEach((i, r) => (model.bars[i].atC = now + r * (MO.stagger / 2)));
					model.kick();
				},
				end() {
					entering = false;
					model.kick();
					armed = true;
					model.startPulse(MO.t4);
				}
			};
			opening(model, slots, ctl);
		};
	}

	/* ---------------------------------------------------------------- 1. designs: the mechanisms */
	const MF = {
		region: () => `
			<rect class="ink" x="55" y="37" width="10" height="34" rx="3"/><rect class="ink" x="75" y="37" width="10" height="34" rx="3"/>
			<rect class="acc" x="105" y="37" width="10" height="34" rx="3"/><rect class="acc" x="125" y="37" width="10" height="34" rx="3"/><rect class="acc" x="145" y="37" width="10" height="34" rx="3"/>
			<rect class="ink" x="175" y="37" width="10" height="34" rx="3"/><rect class="ink" x="195" y="37" width="10" height="34" rx="3"/>
			<path class="la" d="M150 31C151 12 109 12 110 31"/><path class="la" d="M106.6 27.4L110 31.4l3-4"/>
			<text x="70" y="86" text-anchor="middle">prelude</text><text x="130" y="86" text-anchor="middle" class="a">loop</text><text x="190" y="86" text-anchor="middle">coda</text>`,
		visits: () => `
			<rect class="ink" x="65" y="44" width="10" height="26" rx="3"/>
			<rect class="acc" x="100" y="44" width="10" height="26" rx="3"/><rect class="acc" x="115" y="44" width="10" height="26" rx="3"/><rect class="acc" x="130" y="44" width="10" height="26" rx="3"/>
			<rect class="ink" x="165" y="44" width="10" height="26" rx="3"/>
			<path class="la" d="M135 39C136 29 104 29 105 39" opacity=".45"/><path class="la" d="M135 39C137 21 103 21 105 39" opacity=".7"/><path class="la" d="M135 39C138 13 102 13 105 39"/>
			<path class="la" d="M101.6 35.4L105 39.4l3-4"/>
			<text x="150" y="20" class="a m">×4</text>`,
		owners: () => `
			<text x="30" y="34.5" text-anchor="middle">visit 1</text><text x="30" y="65.5" text-anchor="middle">visit 2</text>
			<rect class="tint" x="56" y="20" width="58" height="22" rx="5"/><rect class="tint" x="120" y="20" width="46" height="22" rx="5"/>
			<rect class="tint" x="56" y="51" width="58" height="22" rx="5"/><rect class="tint" x="120" y="51" width="46" height="22" rx="5"/>
			<text x="85" y="34.5" text-anchor="middle" class="a">attention</text><text x="143" y="34.5" text-anchor="middle" class="a">router</text>
			<text x="85" y="65.5" text-anchor="middle" class="a">attention</text><text x="143" y="65.5" text-anchor="middle" class="a">router</text>
			<rect class="acc" x="172" y="20" width="52" height="53" rx="6"/><text x="198" y="50" text-anchor="middle" class="w">experts</text>
			<text x="198" y="87" text-anchor="middle">shared</text><text x="111" y="87" text-anchor="middle">owned per visit</text>`,
		residual: () => `
			<path d="M20 70H220" stroke="var(--hair)" stroke-width="4" stroke-linecap="round"/>
			<text x="120" y="88" text-anchor="middle">the stream</text>
			<rect class="ink" x="38" y="12" width="84" height="20" rx="10"/><text x="80" y="25.5" text-anchor="middle" class="w">layer</text>
			<path d="M80 36V60" stroke="var(--ink)" stroke-width="4.2" stroke-linecap="round"/><path class="lk" d="M74.5 55.5L80 61.5l5.5-6" stroke-width="2" stroke-linejoin="round"/>
			<text x="90" y="52" class="k m">×1</text>
			<rect class="acc" x="128" y="12" width="84" height="20" rx="10"/><text x="170" y="25.5" text-anchor="middle" class="w">looped layer</text>
			<path d="M170 36V60" stroke="var(--acc)" stroke-width="2.1" stroke-linecap="round"/><path class="la" d="M164.5 55.5L170 61.5l5.5-6" stroke-width="2"/>
			<text x="180" y="52" class="a m">×0.5</text>`,
		input: () => `
			<path class="ln" d="M28 46H212"/>
			<rect class="ink" x="40" y="31" width="10" height="30" rx="3"/>
			<circle cx="80" cy="46" r="7" fill="#fff" stroke="var(--acc)" stroke-width="1.5" vector-effect="non-scaling-stroke"/><path class="la" d="M76.5 46h7M80 42.5v7"/>
			<rect class="acc" x="100" y="31" width="10" height="30" rx="3"/><rect class="acc" x="118" y="31" width="10" height="30" rx="3"/><rect class="acc" x="136" y="31" width="10" height="30" rx="3"/>
			<rect class="ink" x="180" y="31" width="10" height="30" rx="3"/>
			<path class="la" d="M141 26C142 6 79 6 80 36"/><path class="la" d="M76.6 32.4L80 36.4l3-4"/>
			<path class="lad" d="M80 82V56"/><text x="88" y="84">input, at every visit</text>`,
		state: () => `
			<path class="ln" d="M55 44H185"/>
			<circle cx="40" cy="44" r="15" fill="#fff" stroke="var(--hair)" stroke-width="1.25" vector-effect="non-scaling-stroke"/>
			<g class="acc2"><circle cx="34" cy="38" r="1.8"/><circle cx="43" cy="35" r="1.5"/><circle cx="38" cy="46" r="1.7"/><circle cx="47" cy="47" r="1.4"/><circle cx="32" cy="50" r="1.3"/><circle cx="42" cy="54" r="1.6"/><circle cx="49" cy="40" r="1.2"/></g>
						<rect class="acc" x="98" y="28" width="12" height="32" rx="4"/><rect class="acc" x="118" y="28" width="12" height="32" rx="4"/>
			<path class="la" d="M124 24C125 8 103 8 104 24"/><path class="la" d="M100.6 20.4L104 24.4l3-4"/>
						<circle cx="200" cy="44" r="15" fill="#fff" stroke="var(--acc)" stroke-width="1.5" vector-effect="non-scaling-stroke"/><path d="M200 29a15 15 0 0 1 0 30z" class="acc3"/><path d="M200 29v30" stroke="var(--acc)" stroke-width="1" vector-effect="non-scaling-stroke"/>
			<text x="40" y="78" text-anchor="middle">start:</text><text x="40" y="89" text-anchor="middle" class="a">random</text>
			<text x="200" y="78" text-anchor="middle">update:</text><text x="200" y="89" text-anchor="middle" class="a">average</text>`,
		depth: () => `
			<rect class="acc" x="62" y="52" width="10" height="20" rx="3"/><rect class="acc" x="102" y="52" width="10" height="20" rx="3"/><rect class="acc" x="142" y="52" width="10" height="20" rx="3"/><rect class="acc" x="182" y="52" width="10" height="20" rx="3"/>
			<rect class="acc3" data-x x="60" y="38" width="14" height="6" rx="2"/><rect class="acc3" data-x x="100" y="28" width="14" height="16" rx="2"/><rect class="acc3" data-x x="140" y="18" width="14" height="26" rx="2"/><rect class="acc3" data-x x="180" y="30" width="14" height="14" rx="2"/>
			<path class="ln" d="M52 44H202"/>
			<text x="67" y="86" text-anchor="middle" class="m">1</text><text x="107" y="86" text-anchor="middle" class="m">2</text><text x="147" y="86" text-anchor="middle" class="m">3</text><text x="187" y="86" text-anchor="middle" class="m">4</text>
			<text x="14" y="34" class="m">P(exit)</text><text x="24" y="66">visit</text>`,
		lora: () => `
			<rect class="acc" x="40" y="28" width="16" height="42" rx="4"/><text x="48" y="86" text-anchor="middle">shared</text>
			<text x="80" y="54" text-anchor="middle" class="m">+</text>
			<rect class="acc" x="104" y="28" width="16" height="42" rx="4" opacity=".3"/><rect class="acc2" x="124" y="28" width="5" height="42" rx="2"/>
			<rect class="acc" x="160" y="28" width="16" height="42" rx="4" opacity=".3"/><rect class="acc2" x="180" y="28" width="5" height="42" rx="2"/>
			<text x="116" y="86" text-anchor="middle">visit 1</text><text x="172" y="86" text-anchor="middle">visit 2</text>
			<text x="127" y="20" text-anchor="middle" class="a m">Δ</text><text x="183" y="20" text-anchor="middle" class="a m">Δ</text>`,
		tying: () => `
			<rect class="ink" x="66" y="16" width="10" height="24" rx="3"/><rect class="ink" x="96" y="16" width="10" height="24" rx="3"/><rect class="ink" x="126" y="16" width="10" height="24" rx="3"/><rect class="ink" x="156" y="16" width="10" height="24" rx="3"/>
			<path class="la" d="M71 42V58M101 42V58M131 42V58M161 42V58" opacity=".7"/>
			<rect class="acc" x="58" y="58" width="116" height="16" rx="5"/><text x="116" y="69.5" text-anchor="middle" class="w">one expert bank</text>
			<text x="190" y="32">4 layers</text>`
	};
	/* each tile plays its own mechanism in place: a token runs the tile's route, and every part it passes answers.
	   Routes are drawn on the tile's own geometry; a route may be a branch timed to arrive where the main token passes. */
	const ROUTE = {
		region: [{ d: 'M45 54H150V31C151 12 109 12 110 31V54H215' }],
		visits: [{ d: 'M55 57H135V39C136 29 104 29 105 39V57H135V39C137 21 103 21 105 39V57H135V39C138 13 102 13 105 39V57H185' }],
		owners: [{ d: 'M48 31H226', t0: 0, t1: 0.46 }, { d: 'M48 62H226', t0: 0.54, t1: 1 }],
		residual: [{ d: 'M80 36V62' }, { d: 'M170 36V62', r: 0.5 }],
		input: [{ d: 'M28 46H141V26C142 6 79 6 80 36V46H212', feed: { d: 'M80 82V53', at: [80, 46] } }],
		state: [{ d: 'M40 44H124V24C125 8 103 8 104 24V44H186' }],
		depth: [{ d: 'M45 62H202' }],
		lora: [{ d: 'M30 49H200' }],
		tying: [0, 1, 2, 3].map((i) => ({ d: `M${71 + 30 * i} 66V40`, t0: i / 4, t1: (i + 0.8) / 4 }))
	};
	function tilePlayer(svg, id) {
		const routes = ROUTE[id].map((r) => ({ ...r, el: s('path', { d: r.d, fill: 'none', stroke: 'none' }, svg) }));
		const parts = [...svg.querySelectorAll('rect.ink, rect.acc, rect.acc2, rect.acc3, rect.tint, circle[stroke]')].map((el) => {
			el.style.transformBox = 'fill-box';
			el.style.transformOrigin = 'center';
			return { el, b: el.getBBox(), xOnly: el.hasAttribute('data-x'), hit: 0 };
		});
		const dots = [];
		const dot = (r) => {
			const g = s('g', { opacity: 0 }, svg);
			s('circle', { r: 9 * r, fill: 'var(--acc)', opacity: 0.16 }, g);
			s('circle', { r: 3.6 * r, fill: '#fff', stroke: 'var(--acc)', 'stroke-width': 1.8 * r }, g);
			dots.push(g);
			return g;
		};
		routes.forEach((r) => {
			r.len = r.el.getTotalLength();
			r.dot = dot(r.r || 1);
			if (r.feed) {
				r.feed.el = s('path', { d: r.feed.d, fill: 'none', stroke: 'none' }, svg);
				r.feed.len = r.feed.el.getTotalLength();
				r.feed.dot = dot(0.8);
				// the fractions of the main route at which its token passes the feed's target
				const N = 240, hits = [];
				for (let i = 0; i <= N; i++) { const p = r.el.getPointAtLength((r.len * i) / N); if (Math.hypot(p.x - r.feed.at[0], p.y - r.feed.at[1]) < 1.5 && (!hits.length || i / N - hits[hits.length - 1] > 0.1)) hits.push(i / N); }
				r.feed.hits = hits;
			}
		});
		const total = Math.max(...routes.map((r) => r.len / ((r.t1 || 1) - (r.t0 || 0)))) / 0.16; // 160 units per second
		let t0 = 0, on = false, run = false;
		const put = (g, path, len, u, op) => {
			const p = path.getPointAtLength(len * clamp(u));
			g.setAttribute('transform', `translate(${p.x.toFixed(2)} ${p.y.toFixed(2)})`);
			g.setAttribute('opacity', op.toFixed(3));
			return p;
		};
		function step(dt, now) {
			const T = (now - t0) / total, u = T % 1.25; // a rest of a quarter turn between plays
			const pts = [];
			routes.forEach((r) => {
				const a = r.t0 || 0, b = r.t1 || 1, v = (u - a) / (b - a);
				const op = on && v >= 0 && v <= 1 ? Math.min(1, v * 8, (1 - v) * 8) : 0;
				const p = put(r.dot, r.el, r.len, easeIO(clamp(v)), op);
				if (op > 0.5) pts.push(p);
				if (r.feed) {
					let fo = 0, fu = 0;
					for (const f of r.feed.hits) { const w = (v - (f - 0.14)) / 0.14; if (w >= 0 && w <= 1) { fo = Math.min(1, w * 6, (1 - w) * 6); fu = w; } }
					const q = put(r.feed.dot, r.feed.el, r.feed.len, easeOut(fu), on ? fo : 0);
					if (fo > 0.5) pts.push(q);
				}
			});
			let busy = on;
			for (const pt of parts) {
				const inX = pts.some((p) => p.x >= pt.b.x - 3 && p.x <= pt.b.x + pt.b.width + 3 && (pt.xOnly || (p.y >= pt.b.y - 3 && p.y <= pt.b.y + pt.b.height + 3)));
				pt.hit = inX ? 1 : Math.max(0, pt.hit - dt * 3.2);
				if (pt.hit > 0) busy = true;
				pt.el.style.transform = pt.hit > 0.001 ? `scale(${(1 + 0.14 * easeOut(pt.hit)).toFixed(3)})` : '';
			}
			if (!busy) { dots.forEach((g) => g.setAttribute('opacity', 0)); run = false; }
			return busy;
		}
		return {
			play() { if (reduced || on) return; on = true; t0 = performance.now(); if (!run) { run = true; animate(step); } },
			stop() { on = false; }
		};
	}
	function designs(onPick) {
		const ol = document.querySelector('[data-mechs]');
		const nodes = Object.keys(C);
		const A = Object.fromEntries(nodes.map((k) => [k, added(k)]));
		const compact = (ns) => {
			const out = [];
			for (let i = 0; i < ns.length; i++) {
				let j = i;
				while (j + 1 < ns.length && ns[j + 1] === ns[j] + 1) j++;
				out.push(j - i >= 2 ? `${ns[i]}–${ns[j]}` : j > i ? `${ns[i]},${ns[j]}` : `${ns[i]}`);
				i = j;
			}
			return out.join(',');
		};
		const buttons = [];
		T.decisions.forEach((d, i) => {
			const li = h('li', '', ol);
			li.setAttribute('data-rv', '');
			li.style.setProperty('--i', i % 3);
			const b = h('button', 'mech', li);
			b.type = 'button';
			b.setAttribute('aria-pressed', 'false');
			const fig = h('div', 'mech__fig', b);
			fig.innerHTML = `<svg class="mf" viewBox="0 0 240 92" aria-hidden="true">${MF[d.id]()}</svg>`;
			let player = null;
			const play = () => { player = player || tilePlayer(fig.firstChild, d.id); player.play(); };
			const stop = () => player && player.stop();
			b.addEventListener('pointerenter', play);
			b.addEventListener('pointerleave', stop);
			b.addEventListener('focus', () => { if (b.matches(':focus-visible')) play(); });
			b.addEventListener('blur', stop);
			const k = h('span', 'mech__k', b);
			const kt = h('span', '', k, d.k);
			const set = nodes.filter((x) => A[x].includes(d.id) || (x === d.show));
			const ids = [...new Set(set.map((x) => arx(C[x].source)).filter(Boolean))].sort((a, c) => refNo[a] - refNo[c]);
			if (ids.length) {
				const sup = h('sup', 'ref-k', kt, compact(ids.map((id) => refNo[id])));
				sup.title = ids.map((id) => `${refNo[id]}. ${REFS[refNo[id] - 1].title}`).join('\n');
			}
			h('span', 'mech__n', k, String(i + 1).padStart(2, '0'));
			h('span', 'mech__d', b, d.d);
			const f = h('span', 'mech__f', b);
			h('code', '', f, d.f);
			const go = h('span', 'mech__go', f, 'Example');
			go.insertAdjacentHTML('beforeend', '<svg class="i" aria-hidden="true"><use href="#g-head"/></svg>');
			b.setAttribute('aria-label', `${d.k}. ${d.d} Field: ${d.f}. Show an example configuration.`);
			b.addEventListener('click', () => {
				buttons.forEach((x) => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
				onPick(d.show, true);
			});
			buttons.push(b);
		});
		return {
			clear() { buttons.forEach((x) => x.setAttribute('aria-pressed', 'false')); },
			press(i) { buttons.forEach((x, j) => x.setAttribute('aria-pressed', j === i ? 'true' : 'false')); }
		};
	}

	/* ---------------------------------------------------------------- 1. designs: the explorer */
	function explorer() {
		const box = document.querySelector('[data-explorer]');
		const tree = box.querySelector('[data-tree]');
		const svg = box.querySelector('[data-isvg]');
		const elName = box.querySelector('[data-iname]');
		const elFrom = box.querySelector('[data-from]');
		const elFile = box.querySelector('[data-ifile]');
		const elDiff = box.querySelector('[data-diff]');
		const elDL = box.querySelector('[data-dlabel]');
		const filter = box.querySelector('[data-filter]');
		const fn = box.querySelector('[data-filter-n]');
		const nodes = Object.keys(C);
		box.querySelector('[data-n-configs]').textContent = nodes.length;
		const G = gauges(box.querySelector('[data-igauges]'), { long: true });
		const model = new Model(svg, { slots: 36, maxPitch: 30, maxBw: 16, top: 52, fs: 13 });
		const kids = {};
		nodes.forEach((k) => { const p = C[k].parent; if (p) (kids[p] = kids[p] || []).push(k); });
		const famOrder = ['phase1', 'shape', 'methods', 'untie', 'p05', 'p1'];
		const fo = (k) => famOrder.indexOf(k.split('/')[0]);
		const rootK = nodes.find((k) => !C[k].parent);
		const order = [];
		const depth = {};
		(function dfs(k, d) {
			order.push(k);
			depth[k] = d;
			(kids[k] || []).slice().sort((a, b) => (kids[b] ? 1 : 0) - (kids[a] ? 1 : 0) || fo(a) - fo(b) || nodes.indexOf(a) - nodes.indexOf(b)).forEach((c) => dfs(c, d + 1));
		})(rootK, 0);
		const btn = {};
		order.forEach((k) => {
			const li = h('li', 'tree__i', tree);
			const b = h('button', 'tree__b', li);
			b.type = 'button';
			b.setAttribute('role', 'option');
			b.style.setProperty('--d', Math.min(depth[k], 3));
			const t = h('span', 'tree__t', b, T.short[k]);
			refMark(t, [arx(C[k].source)], true);
			b.title = T.design[k];
			b.addEventListener('click', () => pick(k));
			b.addEventListener('keydown', (e) => {
				const vis = order.filter((x) => !btn[x].parentNode.classList.contains('is-hide'));
				const i = vis.indexOf(k);
				let j = -1;
				if (e.key === 'ArrowDown') j = Math.min(vis.length - 1, i + 1);
				if (e.key === 'ArrowUp') j = Math.max(0, i - 1);
				if (e.key === 'Home') j = 0;
				if (e.key === 'End') j = vis.length - 1;
				if (j < 0) return;
				e.preventDefault();
				pick(vis[j]);
				btn[vis[j]].focus();
			});
			btn[k] = b;
		});
		const empty = h('li', 'tree__empty', tree, 'No configuration matches.');
		empty.hidden = true;
		const hl = h('li', 'tree__hl', tree);
		hl.setAttribute('role', 'presentation');
		tree.prepend(hl);
		const placeHl = (slide) => {
			const b = btn[cur];
			if (!b || b.parentNode.classList.contains('is-hide')) { hl.style.opacity = 0; return; }
			hl.style.opacity = 1;
			if (!slide) hl.style.transition = 'none';
			hl.style.transform = `translateY(${b.parentNode.offsetTop}px)`;
			if (!slide) { void hl.offsetWidth; hl.style.transition = ''; }
		};
		// the lineage drawn as a lineage: an elbow from each parent to its children; a bar per configuration,
		// indigo with its arc when it loops; the path from the root to the chosen one in ink
		const gut = s('svg', { class: 'tree__g', 'aria-hidden': 'true' });
		tree.prepend(gut);
		const PITCH = 12, X = (d) => 14 + d * PITCH;
		function drawLineage() {
			gut.textContent = '';
			const q = filter.value.trim();
			const rowY = {};
			order.forEach((k) => { const li = btn[k].parentNode; if (!li.classList.contains('is-hide')) rowY[k] = li.offsetTop + li.offsetHeight / 2; });
			gut.setAttribute('width', tree.clientWidth);
			gut.setAttribute('height', tree.scrollHeight);
			const onPath = new Set();
			for (let k = cur; k; k = C[k].parent) onPath.add(k);
			if (!q) order.forEach((k) => {
				const p = C[k].parent;
				if (!p || rowY[k] == null || rowY[p] == null) return;
				s('path', { class: 'ln' + (onPath.has(k) ? ' is-on' : ''), d: `M${X(depth[p])} ${rowY[p] + 6}V${rowY[k] - 3}Q${X(depth[p])} ${rowY[k]} ${X(depth[p]) + 3} ${rowY[k]}H${X(depth[k]) - 3}` }, gut);
			});
			order.forEach((k) => {
				if (rowY[k] == null) return;
				const x = q ? X(0) : X(depth[k]), y = rowY[k], loop = C[k].visits > 1;
				s('rect', { class: 'mk' + (loop ? ' is-loop' : ''), x: x - 1.5, y: y - 5, width: 3, height: 10, rx: 1 }, gut);
				if (loop) s('path', { class: 'ar', d: `M${x + 5} ${y - 2}C${x + 5} ${y - 10} ${x - 5} ${y - 10} ${x - 5} ${y - 2}` }, gut);
			});
		}
		const xf = elName.parentNode;
		xf.classList.add('insp__x');
		let cur = null, pulseT = 0;
		function pick(k, fromMech) {
			const first = cur === null;
			cur = k;
			const cfg = C[k];
			order.forEach((x) => { const on = x === k; btn[x].setAttribute('aria-selected', on ? 'true' : 'false'); btn[x].tabIndex = on ? 0 : -1; });
			placeHl(!first && !reduced);
			drawLineage();
			if (fromMech) {
				const b = btn[k], tr = tree.getBoundingClientRect(), br = b.getBoundingClientRect();
				if (br.top < tr.top || br.bottom > tr.bottom) tree.scrollTo({ top: b.offsetTop - tree.clientHeight / 2, behavior: reduced ? 'auto' : 'smooth' });
			}
			if (!first && !reduced) {
				xf.classList.add('is-out');
				clearTimeout(xf.t);
				xf.t = setTimeout(() => { fill(k); xf.classList.remove('is-out'); }, 90);
			} else fill(k);
			const rows = diffRows(cfg.changes);
			paintDiff(k, cfg, rows);
			model.set(cfg, reduced || !shown);
			G.go(cfg, !shown);
			if (reduced || !shown) model.frame();
			else {
				model.kick();
				clearTimeout(pulseT);
				pulseT = setTimeout(() => model.startPulse(Math.max(0, model.settle - 160)), 160);
			}
		}
		function fill(k) {
			const cfg = C[k];
			elFrom.textContent = '';
			if (cfg.parent) { elFrom.append('From '); h('b', '', elFrom, T.short[cfg.parent]); } else elFrom.textContent = 'The root of the lineage';
			elName.textContent = T.design[k];
			refMark(elName, [arx(cfg.source)]);
			elFile.href = BLOB + fileOf(k);
			elFile.querySelector('span').textContent = fileOf(k);
		}
		function paintDiff(k, cfg, rows) {
			elDiff.textContent = '';
			if (!cfg.parent) {
				elDL.textContent = 'The reference';
				[['dim', cfg.dim], ['num_cells', cfg.core], ['num_experts', cfg.regions.core.experts], ['top_k', cfg.topK]].forEach(([f, v], i) => {
					const r = h('div', 'diff__r', elDiff);
					r.style.setProperty('--i', i);
					h('b', '', r, f);
					const vv = h('span', 'diff__v', r);
					h('em', '', vv, String(v));
				});
			} else if (!rows.length) {
				elDL.textContent = 'Changed against its parent';
				h('p', 'diff__lr', elDiff, 'No model field changed: the learning rate only.');
			} else {
				elDL.textContent = `Changed against ${T.short[cfg.parent]}`;
				const MAX = 6;
				rows.slice(0, rows.length > MAX ? MAX - 1 : MAX).forEach(([f, a, b], i) => {
					const r = h('div', 'diff__r', elDiff);
					r.style.setProperty('--i', i);
					h('b', '', r, f).title = f;
					const v = h('span', 'diff__v', r);
					h('s', '', v, a);
					h('span', 'diff__arrow', v, '→');
					h('em', '', v, b);
				});
				if (rows.length > MAX) h('p', 'diff__more', elDiff, `+${rows.length - MAX + 1} more fields in the file`);
			}
		}
		filter.addEventListener('input', () => {
			const q = filter.value.trim().toLowerCase();
			let n = 0;
			order.forEach((k) => {
				const hit = !q || T.short[k].toLowerCase().includes(q) || T.design[k].toLowerCase().includes(q) || k.includes(q);
				btn[k].parentNode.classList.toggle('is-hide', !hit);
				if (hit) n++;
			});
			empty.hidden = n > 0;
			fn.textContent = q ? `${n} / ${order.length}` : '';
			placeHl(false);
			drawLineage();
		});
		addEventListener('keydown', (e) => {
			if (e.key !== '/' || /input|textarea/i.test(document.activeElement.tagName)) return;
			const r = box.getBoundingClientRect();
			if (r.bottom < 0 || r.top > innerHeight) return;
			e.preventDefault();
			filter.focus();
		});
		let shown = false;
		pick(T.decisions[0].show);
		onResize(svg, () => { model.measure(); model.set(C[cur], true); model.frame(); });
		onVisible(box, () => { shown = true; model.startPulse(MO.t3); }, 0.35);
		if (document.fonts) document.fonts.ready.then(() => { placeHl(false); drawLineage(); });
		onResize(tree, drawLineage);
		return pick;
	}

	/* the cost of one token, layer by layer: one cell per layer run, its width that layer's share of the training FLOPs.
	   The strip ends where the configuration's FLOPs end, against the standard MoE's (100 %) and its ±5 % band. */
	function runStrip(host) {
		const W = 300, H = 30, X0 = 4, U = 190; // U = the standard MoE's FLOPs per token
		const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'runs' }, host);
		s('rect', { class: 'runs__band', x: X0 + U * 0.95, y: 2, width: U * 0.1, height: 16, rx: 2 }, svg);
		s('line', { class: 'runs__ref', x1: X0 + U, x2: X0 + U, y1: 0, y2: 20 }, svg);
		const lab = s('text', { class: 'runs__t', x: X0 + U, y: H - 1, 'text-anchor': 'middle' }, svg);
		lab.textContent = 'standard MoE';
		const end = s('text', { class: 'runs__v', x: 0, y: 14 }, svg);
		const cells = Array.from({ length: 24 }, () => s('rect', { class: 'runs__c', x: 0, y: 4, width: 1, height: 12 }, svg));
		const st = cells.map(() => ({ x: new Spring(X0), w: new Spring(0), o: new Spring(0), k: 0 }));
		const tot = new Spring(0);
		let run = false;
		function paint() {
			st.forEach((q, i) => {
				const c = cells[i];
				c.setAttribute('transform', `translate(${q.x.v.toFixed(2)} 0) scale(${Math.max(0, q.w.v - 0.8).toFixed(3)} 1)`);
				c.setAttribute('opacity', clamp(q.o.v).toFixed(3));
				c.setAttribute('class', `runs__c runs__c--${q.k}`);
			});
			end.setAttribute('x', (X0 + tot.v + 6).toFixed(1));
		}
		return {
			go(cfg, snap) {
				const f = cfg.budgets.training_flops_per_token;
				const L = [];
				for (let i = 0; i < cfg.pre; i++) L.push(0);
				for (let v = 0; v < cfg.visits; v++) for (let i = 0; i < cfg.core; i++) L.push(cfg.visits > 1 ? 1 + Math.min(v, 1) : 0);
				for (let i = 0; i < cfg.suf; i++) L.push(0);
				const total = (U * f.candidate) / f.reference, w = total / L.length;
				st.forEach((q, i) => {
					const on = i < L.length;
					q.x.t = X0 + (on ? i : L.length) * w; q.w.t = on ? w : 0; q.o.t = on ? 1 : 0; if (on) q.k = L[i];
					if (snap || reduced) { q.x.set(q.x.t); q.w.set(q.w.t); q.o.set(q.o.t); }
				});
				tot.t = total;
				if (snap || reduced) tot.set(total);
				end.textContent = `${Math.round((100 * f.candidate) / f.reference)}%`;
				end.classList.toggle('is-out', !f.within);
				if (snap || reduced) { paint(); return; }
				if (run) return;
				run = true;
				animate((dt) => {
					let mv = tot.step(dt);
					st.forEach((q) => { mv = q.x.step(dt) | mv; mv = q.w.step(dt) | mv; mv = q.o.step(dt) | mv; });
					paint();
					if (!mv) run = false;
					return mv;
				});
			}
		};
	}

	/* ---------------------------------------------------------------- 2. compare */
	function compare() {
		const term = document.querySelector('[data-term]');
		const cmd = term.querySelector('[data-cmd]');
		const out = term.querySelector('[data-out]');
		const full = cmd.textContent.replace(/^\$ /, '');
		const M = CMP.matched;
		// the printed relative differences, each on its own ±5 % band, at the foot of the terminal
		const tg = (() => {
			const host = h('div', 'tgauges', term);
			host.setAttribute('aria-hidden', 'true');
			const DOM = 0.08;
			const R = {};
			const NM = { unique_non_embedding_parameters: 'parameters', training_flops_per_token: 'training FLOPs', logical_kv_bytes: 'KV cache' };
			Object.keys(M).map((q) => [q, NM[q] || q]).forEach(([q, n]) => {
				const row = h('div', 'tg', host);
				h('span', 'tg__n', row, n);
				const tr = h('span', 'tg__b', row);
				const tol = M[q].relative_tolerance;
				const band = h('i', 'tg__band', tr);
				band.style.left = `${50 - (tol / DOM) * 50}%`; band.style.width = `${(tol / DOM) * 100}%`;
				h('i', 'tg__z', tr);
				const dot = h('i', 'tg__d', tr);
				dot.tr = tr;
				const v = h('span', 'tg__v', row, '');
				R[q] = { dot, v, sp: new Spring(0), t: M[q].relative_difference };
			});
			let run = false;
			const paint = () => Object.values(R).forEach((r) => (r.dot.style.transform = `translateX(${((clamp(r.sp.v, -DOM, DOM) / DOM) * r.dot.tr.clientWidth / 2).toFixed(2)}px)`));
			return {
				reset() { Object.values(R).forEach((r) => { r.sp.set(0); r.sp.t = 0; r.v.textContent = ''; r.dot.style.opacity = 0; }); paint(); },
				set(q, snap) {
					const r = R[q];
					r.sp.t = r.t; r.dot.style.opacity = 1;
					r.v.textContent = pct(r.t);
					if (snap || reduced) { r.sp.set(r.t); paint(); return; }
					if (run) return;
					run = true;
					animate((dt) => { let mv = false; Object.values(R).forEach((x) => (mv = x.sp.step(dt) | mv)); paint(); if (!mv) run = false; return mv; });
				}
			};
		})();
		// the real output, the matched block (basis lines left out)
		const L = [], rdAt = {};
		const k = (t) => `<span class="k">"${t}"</span>`;
		const nv = (v) => `<span class="n">${v}</span>`;
		const sv = (v) => `<span class="s">"${v}"</span>`;
		const pu = (t) => `<span class="p">${t}</span>`;
		L.push(pu('{'));
		L.push(`  ${k('reference')}${pu(':')} ${sv(CMP.reference)}${pu(',')}`);
		L.push(`  ${k('candidate')}${pu(':')} ${sv(CMP.candidate)}${pu(',')}`);
		L.push(`  ${k('matched')}${pu(': {')}`);
		const mk = Object.keys(M);
		mk.forEach((q, i) => {
			const d = M[q];
			L.push(`    ${k(q)}${pu(': {')}`);
			L.push(`      ${k('reference')}${pu(':')} ${nv(d.reference)}${pu(',')}`);
			L.push(`      ${k('candidate')}${pu(':')} ${nv(d.candidate)}${pu(',')}`);
			rdAt[q] = L.length;
			L.push(`      ${k('relative_difference')}${pu(':')} ${nv(Number(d.relative_difference.toPrecision(4)))}${pu(',')}`);
			L.push(`      ${k('relative_tolerance')}${pu(':')} ${nv(d.relative_tolerance)}`);
			L.push(`    ${pu(i < mk.length - 1 ? '},' : '}')}`);
		});
		L.push(`  ${pu('},')}`);
		L.push(`  ${k('differences')}${pu(':')} ${pu('{')} <span class="p">… ${Object.keys(CMP.differences).length} fields</span> ${pu('}')}`);
		L.push(pu('}'));
		// the command types itself once the terminal is in view: 9 ms a character, a short pause at each line break
		let playing = 0;
		const prompt = () => { cmd.innerHTML = '<span class="term__p" aria-hidden="true">$ </span>'; };
		const at = [];
		for (let i = 0, t = 0; i <= full.length; i++) { at.push(t); t += full[i] === '\n' ? 120 : 9; }
		function play() {
			const id = ++playing;
			if (reduced) { out.innerHTML = L.join('\n'); Object.keys(rdAt).forEach((q) => tg.set(q, true)); return; }
			tg.reset();
			out.innerHTML = '';
			prompt();
			const span = document.createElement('span');
			cmd.appendChild(span);
			const caret = h('span', 'caret', cmd);
			let i = 0;
			const t0 = performance.now();
			animate((dt, now) => {
				if (id !== playing) return false;
				let n = i;
				while (n < full.length && at[n + 1] <= now - t0) n++;
				if (n !== i) { i = n; span.textContent = full.slice(0, i); }
				if (i < full.length) return true;
				caret.remove();
				setTimeout(() => lines(id), MO.t2);
				return false;
			});
		}
		function lines(id) {
			const t0 = performance.now();
			let shownN = 0;
			animate((dt, now) => {
				if (id !== playing) return false;
				const n = Math.min(L.length, Math.floor((now - t0) / MO.stagger) + 1);
				if (n !== shownN) {
					// as a relative difference prints, its gauge below takes that value
					for (const q in rdAt) if (rdAt[q] >= shownN && rdAt[q] < n) tg.set(q);
					shownN = n; out.innerHTML = L.slice(0, n).join('\n');
				}
				return n < L.length;
			});
		}
		out.innerHTML = '';
		if (!reduced) prompt();
		tg.reset();
		term.querySelector('[data-copy]').dataset.text = full;
		onVisible(term, play, 0.4);
		term.querySelector('[data-replay]').addEventListener('click', play);
		// match first: the looped MoE, its unrolled twin, and the same loop added without matching
		const mf = document.querySelector('[data-match]');
		const MK = [['phase1/m3', 'Looped MoE'], ['phase1/m4', 'Unrolled twin'], ['methods/m0-loop', 'Loop, not matched']];
		const mcap = mf.querySelector('[data-mcap]');
		const mok = mf.querySelector('[data-mok]');
		const mokT = mf.querySelector('[data-mok-t]');
		const MG = gauges(mf.querySelector('[data-mgauges]'), { axis: true, sum: mf.querySelector('[data-mok-b]') });
		const mm = new Model(mf.querySelector('[data-msvg]'), { slots: 18, maxPitch: 34, maxBw: 24, top: 48, bottom: 0, fs: 13 });
		const MS = seg(mf.querySelector('[data-mchips]'), MK.map((x) => x[1]), (i) => mgo(i));
		const run = runStrip(mf.querySelector('[data-mrun]'));
		/* the twin writes the loop out: the loop's second-visit copies lift off together along an arc to their own slots,
		   and the layers after the loop move right to make room; going back, the copies fold back onto the loop.
		   All copies move as one block on parallel arcs, so no two ever cross. */
		let mi = 0;
		function morph(from, to) {
			const unrolled = (c) => c.visits === 1 && c.core > 6;
			const a = C[MK[from][0]], b = C[MK[to][0]];
			if (reduced || unrolled(a) === unrolled(b)) { mm.set(b); return; }
			const B = mm.bars, P = a.pre, K = 6, now = performance.now();
			const src = Array.from({ length: K }, (_, j) => B[P + j].x.v);
			if (unrolled(b)) {
				// [pre, core, suf] -> [pre, core, copies, suf]: the suffix objects move to the end, the copies take the gap
				const suf = B.splice(P + K, a.suf), copies = B.splice(P + K, K);
				B.splice(P + K, 0, ...copies, ...suf);
				mm.set(b);
				const A = mm.geo.hU * 0.6 + 18 * mm.k;
				copies.forEach((c, j) => {
					c.at = 0; c.a.set(1); c.h.set(c.hT || c.h.t); c.h.t = c.h.v; c.c.set(1); c.c.t = 0;
					c.x.set(src[j]);
					c.atX = now + MO.t1; c.xT = mm.x0 + mm.geo.pitch * (P + K + j);
					c.peel = { xs: src[j], xd: c.xT, A };
				});
			} else {
				// [pre, core, copies, suf] -> [pre, core, suf]: the copies ride back onto the loop and merge into it
				const copies = B.splice(P + K, K), suf = B.splice(P + K, a.suf);
				B.splice(P + K, 0, ...suf, ...copies);
				const xs = copies.map((c) => c.x.v);
				mm.set(b);
				const A = mm.geo.hU * 0.6 + 18 * mm.k;
				copies.forEach((c, j) => {
					c.leaving = false; c.L = null; c.a.set(1); c.a.t = 1; c.h.set(c.h0 || c.h.v); c.h.t = c.h.v; c.c.t = 1;
					const xd = mm.x0 + mm.geo.pitch * (P + j);
					c.x.set(xs[j]); c.atX = now + 1; c.xT = xd;
					c.peel = { xs: xs[j], xd, A }; c.fold = true;
				});
			}
		}
		function mgo(i, snap) {
			const cfg = C[MK[i][0]];
			const from = mi;
			mi = i;
			MS.set(i, !snap);
			mcap.textContent = `${T.design[MK[i][0]]} · ${cfg.pre + cfg.core + cfg.suf} stored, ${cfg.pre + cfg.core * cfg.visits + cfg.suf} run`;
			const inb = within(cfg);
			mok.classList.toggle('is-out', !inb);
			mokT.textContent = inb ? 'within ±5%' : 'outside ±5%';
			if (reduced || snap) mm.set(cfg, true); else morph(from, i);
			run.go(cfg, snap);
			MG.go(cfg, snap);
			if (reduced || snap) mm.frame(); else { mm.kick(); mm.startPulse(Math.max(MO.t3, mm.settle)); }
		}
		mgo(0, true);
		onResize(mf.querySelector('.match__stage'), () => { mm.measure(); mm.set(mm.cfg, true); mm.frame(); });
		onVisible(mf, () => mm.startPulse(MO.t3), 0.4);
		// the chain
		const ch = document.querySelector('[data-chainrow]');
		T.chain.forEach(([, n, k]) => {
			const li = h('li', '', ch);
			li.appendChild(miniModel(C[k], { slots: 18, hU: 60 }));
			h('span', '', li, n);
		});
	}

	/* ---------------------------------------------------------------- 3. check: two pictures */
	function checkPictures(cases, Q, tolOf) {
		const X = T.checks;
		// 1. the loop calls the official layers, in the loop's order, with the weights shared (structure of m3)
		const cfg = C['phase1/m3'];
		const seq = [];
		for (let i = 0; i < cfg.pre; i++) seq.push(i);
		for (let v = 0; v < cfg.visits; v++) for (let i = 0; i < cfg.core; i++) seq.push(cfg.pre + i);
		for (let i = 0; i < cfg.suf; i++) seq.push(cfg.pre + cfg.core + i);
		const nB = cfg.pre + cfg.core + cfg.suf, nC = seq.length;
		const host = document.querySelector('[data-calls]');
		const plot = document.querySelector('[data-tplot]');
		const loopB = (i) => i >= cfg.pre && i < cfg.pre + cfg.core;
		let ran = false, plotOn = false;
		function drawCalls() {
			host.textContent = '';
			const k = remK(), VW = Math.max(300, host.clientWidth / k), narrow = VW < 520;
			const pC = Math.min(31, (VW - 16) / nC), pB = Math.min(26, pC * 1.1), bw = Math.min(10, pB * 0.45), cw = Math.min(16, pC * 0.62);
			const yB = narrow ? 44 : 30, yC = yB + 74, H = yC + 44;
			const xB = (i) => VW / 2 - (pB * (nB - 1)) / 2 + i * pB, xC = (j) => VW / 2 - (pC * (nC - 1)) / 2 + j * pC;
			const svg = s('svg', { viewBox: `0 0 ${VW} ${H}`, height: H * k, class: 'calls__svg' + (ran ? ' is-done' : '') }, host);
			const links = seq.map((b, j) => s('path', { class: 'calls__l' + (loopB(b) ? ' is-loop' : ''), d: `M${xC(j)} ${yC - 2}C${xC(j)} ${yC - 34} ${xB(b)} ${yB + 52} ${xB(b)} ${yB + 22}` }, svg));
			for (let i = 0; i < nB; i++) s('rect', { class: loopB(i) ? 'calls__b is-loop' : 'calls__b', x: xB(i) - bw / 2, y: yB - 20, width: bw, height: 40, rx: Math.min(3, bw / 3) }, svg);
			seq.forEach((b, j) => s('rect', { class: 'calls__c', x: xC(j) - cw / 2, y: yC, width: cw, height: 24, rx: 3 }, svg));
			const calls = seq.map((b, j) => s('rect', { class: 'calls__h' + (loopB(b) ? ' is-loop' : ''), x: xC(j) - cw / 2, y: yC, width: cw, height: 24, rx: 3 }, svg));
			const tx = (x, y, t, a) => { const e = s('text', { x, y, class: 'calls__t', 'text-anchor': a }, svg); e.textContent = t; };
			if (narrow) tx(VW / 2, 12, `LoopTitan, ${nB} layers stored`, 'middle');
			else tx(xB(0) - 14, yB + 4, `LoopTitan, ${nB} layers stored`, 'end');
			tx(VW / 2, H - 4, `official Qwen3-MoE layer calls, in the loop's order: ${nC}`, 'middle');
			calls.forEach((c, j) => { c.style.animationDelay = links[j].style.animationDelay = `${j * 60}ms`; });
			svg.run = () => svg.classList.add(reduced ? 'is-done' : 'is-run');
			if (ran) svg.classList.add('is-done');
			return svg;
		}
		let callsSvg = drawCalls();
		onVisible(host, () => { ran = true; callsSvg.run(); }, 0.4);
		// 2. thirty differences, each against its own tolerance: every dot sits left of the line
		function drawPlot() {
			plot.textContent = '';
			const k = remK(), PW = Math.max(300, plot.clientWidth / k), narrow = PW < 520;
			const rowH = 22, top = 18, H = top + rowH * Q.length + 26;
			const L0 = narrow ? 92 : 150, Lx = PW - (narrow ? 34 : 70), lo = -3, G0 = narrow ? 36 : 52;
			const xL = (r) => L0 + G0 + ((Math.log10(r) - lo) / -lo) * (Lx - L0 - G0);
			const ps = s('svg', { viewBox: `0 0 ${PW} ${H}`, height: H * k, class: 'tplot__svg' + (plotOn ? ' is-on is-done' : '') }, plot);
			for (let e = lo; e <= 0; e++) {
				s('line', { class: e === 0 ? 'tplot__tol' : 'tplot__g', x1: xL(10 ** e), x2: xL(10 ** e), y1: top - 6, y2: H - 22 }, ps);
				if (narrow && e === -2) continue;
				const t = s('text', { class: e === 0 ? 'tplot__k' : 'tplot__a', x: xL(10 ** e), y: H - 8, 'text-anchor': e === 0 && narrow ? 'end' : 'middle' }, ps);
				t.textContent = e === 0 ? 'tolerance' : e === -1 ? '1/10' : `1/${10 ** -e}`;
				if (e === 0 && narrow) t.setAttribute('x', xL(1) + 30);
			}
			const tz = s('text', { class: 'tplot__a', x: L0 + 12, y: H - 8, 'text-anchor': 'middle' }, ps);
			tz.textContent = 'exact';
			let n = 0;
			Q.forEach((q, i) => {
				const y = top + rowH * i + rowH / 2;
				s('line', { class: 'tplot__row', x1: L0, x2: Lx, y1: y, y2: y }, ps);
				const t = s('text', { class: 'tplot__q', x: L0 - 10, y: y + 3.5, 'text-anchor': 'end' }, ps);
				t.textContent = X.quantities[q];
				const placed = [];
				cases.forEach((c) => {
					const v = CHK.cases[c].errors[q], tol = CHK.tolerances[tolOf[q]].atol, r = v / tol;
					const cx = v === 0 ? L0 + 12 : xL(r);
					// two values that would cover each other stand one above the other, so all thirty can be counted
					const near = placed.filter((p) => Math.abs(p - cx) < 7.5).length;
					placed.push(cx);
					const d = s('circle', { class: 'tplot__d', cx, cy: y + (near ? (near % 2 ? -5.5 : 5.5) : 0), r: 3.6 }, ps);
					d.style.setProperty('--i', n++);
					s('title', {}, d).textContent = `${X.cases[c]} · ${X.quantities[q]}: ${v === 0 ? 'exact' : v.toExponential(1)} against ${tol.toExponential(1)}`;
				});
			});
			return ps;
		}
		let plotSvg = drawPlot();
		onVisible(plot, () => { plotOn = true; plotSvg.classList.add('is-on'); }, 0.4);
		onResize(host, () => { callsSvg = drawCalls(); });
		onResize(plot, () => { plotSvg = drawPlot(); });
	}

	/* ---------------------------------------------------------------- 3. check */
	function check() {
		const box = document.querySelector('[data-matrix]');
		const X = T.checks;
		const Q = ['logits', 'loss', 'gradient', 'updated_parameter', 'cached_logits'];
		const tolOf = { logits: 'logits_and_cached_logits', cached_logits: 'logits_and_cached_logits', loss: 'loss', gradient: 'gradient', updated_parameter: 'updated_parameter' };
		const cases = Object.keys(CHK.cases);
		// mantissa first; the exponent in a lighter grey
		const ex = (v, html) => {
			if (v === 0) return '0';
			const [m, e] = v.toExponential(1).split('e');
			const ee = `e${minus(e).replace('+', '')}`;
			return html ? `${m}<span class="mx__e">${ee}</span>` : m + ee;
		};
		h('div', 'mx__h', box, 'Case');
		Q.forEach((q) => {
			const c = h('div', 'mx__h', box, X.quantities[q]);
			h('small', '', c).innerHTML = `tol ${ex(CHK.tolerances[tolOf[q]].atol, true)}`;
		});
		let n = 0, pass = 0;
		cases.forEach((c, i) => {
			h('div', 'mx__r', box, X.cases[c]);
			Q.forEach((q, j) => {
				const v = CHK.cases[c].errors[q];
				const tol = CHK.tolerances[tolOf[q]].atol;
				const cell = h('div', 'mx__c', box);
				cell.title = `${X.cases[c]} · ${X.quantities[q]}: ${v === 0 ? 'exactly 0' : minus(v.toExponential(2))} (absolute tolerance ${ex(tol)})`;
				h('span', 'mx__v', cell).innerHTML = v === 0 ? '0<span class="mx__e"> exact</span>' : ex(v, true);
				const bar = h('span', 'mx__bar', cell);
				const fill = h('i', '', bar);
				fill.style.setProperty('--f', Math.max(v === 0 ? 0 : 0.02, v / tol).toFixed(3));
				fill.style.setProperty('--dl', `${(i + j) * MO.stagger}ms`);
				n++;
				if (v <= tol) pass++;
			});
		});
		document.querySelector('[data-mx-ok]').textContent = `${pass} of ${n} within tolerance`;
		checkPictures(cases, Q, tolOf);
		const tog = document.querySelector('[data-mx-tog]'), mxw = document.getElementById('mx-values');
		tog.addEventListener('click', () => {
			const open = mxw.hidden;
			mxw.hidden = !open;
			tog.setAttribute('aria-expanded', String(open));
			tog.classList.toggle('is-open', open);
			if (open) box.classList.add('is-on');
		});
		const cells = s('svg', { viewBox: `0 0 ${n * 4 - 1} 12` }, document.querySelector('[data-mx-cells]'));
		cases.forEach((c, i) => Q.forEach((q, j) => {
			const ok = CHK.cases[c].errors[q] <= CHK.tolerances[tolOf[q]].atol;
			s('rect', { x: (i * Q.length + j) * 4, y: 1, width: 3, height: 10, rx: 1, fill: ok ? 'var(--ink)' : 'none', stroke: ok ? 'none' : 'var(--ink)' }, cells);
		}));

		const cl = document.querySelector('[data-checks]');
		const names = { ownership: 'Parameter ownership', shared_gradients: 'Shared gradients', cost_calls: 'Cost model calls', cache: 'Cache', hf_roundtrip: 'Hugging Face round trip', identity: 'Identity' };
		CHK.checks.forEach((c) => {
			const li = h('li', '', cl);
			li.innerHTML = '<svg class="i" aria-hidden="true"><use href="#g-in"/></svg>';
			li.append(names[c] || c);
		});
		document.querySelector('[data-tests]').textContent = CHK.tests.toLocaleString('en-US');
	}

	/* ---------------------------------------------------------------- 5. Phase 1: the controlled chain */
	const BPB = P.fig1_decomposition.test_bpb;
	const TR = P.fig3_instrument.training;
	const lrS = (v) => minus(v.toExponential(1).replace(/\.0e/, 'e'));
	function study() {
		document.querySelector('[data-scale]').textContent = window.V.scale;
		const protocol = `Test mBPB against the standard MoE trained with the same optimizer; lower is better. Each step adds one decision at matched parameters, FLOPs and KV cache. ${window.TYPED.data}, ${window.TYPED.tokenizer}, ${window.V.context}, seed ${TR.seed}. Learning rate: tuned once per optimizer on the standard MoE (AdamW ${lrS(TR.lr_adamw)}, Muon ${lrS(TR.lr_muon)}). The looped MoE at the end of the chain selects the same value. Every other configuration inherits it.`;
		document.querySelector('[data-protocol]').textContent = protocol;
		const box = document.querySelector('[data-fall]');
		const CH = T.chain;
		const steps = CH.slice(1).map(([k, n, c], i) => ({ n, c, a: CH[i][0], b: k, loop: k === 'M3' }));
		const cum = (o, k) => (BPB[k][o] - BPB.M0[o]) * 1000;
		// the chart is built once per width; the build only moves transforms and opacity
		const EX = P.fig6_exchange;
		const ex3 = [['M0-Loop', 'methods/m0-loop'], ['M0-2E', 'phase1/m0-2e']].map(([q, c]) => ({ n: T.short[c], d: (EX[q].muon - BPB.M0.muon) * 1000 }));
		const finds = [...document.querySelectorAll('.finds li')];
		let anim = [], hold = { el: null }, svgEl = null, built = false;
		const STEP = MO.t2 + MO.stagger, HOLD = MO.t4;
		const FR = [{ n: 'Standard MoE', c: 'phase1/m0', base: true }, ...steps, { n: 'Looped MoE, all five', c: 'phase1/m3', total: true, a: 'M0', b: 'M3' }];
		const loopI = FR.findIndex((r) => r.loop);
		// when each row starts: one step after another, a held beat on the loop, then the total
		const rowT = (i) => i * STEP + (i > loopI ? HOLD : 0);
		const END = rowT(FR.length) + MO.t4 + 2 * STEP;
		function build() {
			if (box.clientWidth < 120) return; // not laid out yet; the resize observer builds it
			box.textContent = '';
			const k = remK();
			const W = box.clientWidth / k;
			const small = W < 560;
			const LW = small ? 0 : 200, GW = small ? 0 : 64;
			const RH = small ? 58 : 48, top = 10;
			const H0 = top + FR.length * RH;
			const IH = small ? 118 : 96; // the inset for the third finding, with its own axis
			const H = H0 + 48 + IH;
			const LO = -10, HI = 2;
			const P0 = LW + GW + 22, P1 = W - (small ? 34 : 44);
			const x = (v) => P0 + ((P1 - P0) * (v - LO)) / (HI - LO);
			const svg = (svgEl = s('svg', { viewBox: `0 0 ${W} ${H}`, height: H * k, role: 'img', 'aria-label': 'The looped recipe’s gain over the standard MoE, built one decision at a time, under AdamW and Muon. The loop step itself is close to zero.' }, box));
			anim = [];
			const bars = [], labs = [];
			let mark2 = null;
			const yAx = H0 + 6;
			for (let v = LO; v <= HI; v += 2) s('line', { x1: x(v), x2: x(v), y1: top, y2: yAx, stroke: v === 0 ? 'var(--hair)' : 'var(--faint)', 'stroke-width': 1 }, svg);
			const mark = (n, mx, my) => {
				const g = s('g', { class: 'fmark', 'data-f': n, transform: `translate(${mx} ${my})` }, svg);
				s('circle', { r: 8.5 }, g);
				const t = s('text', { y: 3.6, 'text-anchor': 'middle' }, g);
				t.textContent = n;
				return g;
			};
			FR.forEach((r, i) => {
				const y0 = top + i * RH, t0 = rowT(i);
				const g = s('g', { class: 'frow' + (r.loop ? ' is-loop' : '') + (r.total ? ' is-total' : ''), 'data-f': r.loop ? '1' : r.total ? '2' : '' }, svg);
				const hit = s('rect', { class: 'hit', x: -12, y: y0 + 2, width: W + 24, height: RH - 4, rx: 12 }, g);
				if (r.loop) anim.push({ el: hit, t0, d: MO.t3, kind: 'op' });
				if (r.total) s('line', { x1: 0, x2: W, y1: y0 - 1, y2: y0 - 1, stroke: 'var(--line)' }, svg);
				const lab = s('g', {}, g);
				anim.push({ el: lab, t0, d: MO.t2, kind: 'op' });
				const tn = s('text', { x: 4, y: small ? y0 + 17 : y0 + RH / 2 + 4.5, 'font-size': 13, 'font-weight': r.loop || r.total ? 600 : 500, fill: r.loop ? 'var(--acc)' : 'var(--ink)' }, lab);
				tn.textContent = (r.base || r.total ? '' : '+ ') + r.n;
				if (GW) {
					// the configuration this step reaches, on the page-wide height scale
					const m = miniModel(C[r.c], { slots: 18, hU: 60 });
					m.setAttribute('x', LW); m.setAttribute('y', y0 + 7); m.setAttribute('width', GW); m.setAttribute('height', RH - 14);
					lab.appendChild(m);
				}
				const tip = [];
				if (r.base) {
					s('rect', { x: x(0) - 1, y: y0 + 12, width: 2, height: RH - 24, rx: 1, fill: 'var(--ink)' }, lab);
					const t = s('text', { x: x(0) - 8, y: y0 + RH / 2 + 4, 'font-size': 12, fill: 'var(--soft-t)', 'text-anchor': 'end' }, lab);
					t.textContent = 'reference';
					tip.push(`AdamW ${BPB.M0.adamw.toFixed(4)} · Muon ${BPB.M0.muon.toFixed(4)} test BPB`);
				} else {
					[['adamw', 0], ['muon', 1]].forEach(([o, j]) => {
						const a = r.total ? 0 : cum(o, r.a), b = cum(o, r.b);
						const yy = (small ? y0 + 26 : y0 + RH / 2 - 9) + j * 10;
						const col = r.loop || r.total ? (j ? 'var(--acc-3)' : 'var(--acc)') : j ? 'var(--soft)' : 'var(--ink)';
						const xa = x(a), xb = x(b), d = b - a;
						const tiny = Math.abs(d) < 0.15;
						const tj = t0 + j * MO.stagger;
						// a step too small to draw as a bar is a tick at its start, labelled to its right
						const bb = tiny ? { x: xa - 0.75, y: yy - 1, w: 1.5, h: 10 } : { x: Math.min(xa, xb), y: yy, w: Math.max(1.5, Math.abs(xb - xa)), h: 8 };
						bars.push(bb);
						if (tiny) anim.push({ el: s('rect', { x: bb.x, y: bb.y, width: bb.w, height: bb.h, rx: 0.75, fill: col }, g), t0: tj, d: MO.t3, kind: 'op' });
						else anim.push({ el: s('rect', { x: bb.x, y: bb.y, width: bb.w, height: bb.h, rx: 2, fill: col }, g), t0: tj, d: MO.t4, kind: 'grow', o: xa });
						if (!r.total && i > 1 && j === 0) anim.push({ el: s('line', { x1: xa, x2: xa, y1: small ? Math.max(yy - RH + 10, y0 + 21) : yy - RH + 10, y2: yy + 18, stroke: 'var(--hair)', 'stroke-width': 1, 'stroke-dasharray': '2 2' }, svg), t0: tj, d: MO.t2, kind: 'op' });
						const right = tiny || d >= 0;
						const vt = s('text', { class: 'num', y: yy + 7.5, 'font-size': 12, fill: r.loop || r.total ? 'var(--acc)' : 'var(--mid)', 'font-weight': r.total ? 600 : 400 }, g);
						vt.textContent = sgn(d, 1) || '0.0';
						// the label's place is chosen after every bar is drawn, so it never covers one (see below)
						labs.push({ el: vt, bar: bb, right, y: yy, total: r.total && j === 1, dark: j === 0 });
						anim.push({ el: vt, t0: tj + MO.t4 * 0.6, d: MO.t2, kind: 'op' });
						tip.push(`${o === 'adamw' ? 'AdamW' : 'Muon'} ${BPB[r.b][o].toFixed(4)} test BPB (${sgn(d, 2)} mBPB${r.total ? '' : ' this step'})`);
						// the findings, at the places that show them: 1 on the loop step, 2 on the total under Muon
						if (r.loop && j === 1) anim.push({ el: mark('1', P1 + 18, y0 + RH / 2), t0: tj + MO.t3, d: MO.t2, kind: 'op' });
						if (r.total && j === 1) mark2 = { t0: tj + MO.t4, y: yy + 4 };
					});
				}
				s('title', {}, g).textContent = `${r.n}: ${tip.join('; ')}`;
			});
			// value labels: outside the bar's end by default, else outside its other end, else inside it; never over a bar,
			// a label or the finding marks
			const boxes = [], hit = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
			for (const L_ of labs) {
				const w = L_.el.getComputedTextLength() || L_.el.textContent.length * 7.3;
				const yT = L_.y - 2.5, hT = 12.5;
				const b = L_.bar, cands = [];
				const left = { x: b.x - 6 - w, anchor: 'end', tx: b.x - 6 }, rightC = { x: b.x + b.w + 6, anchor: 'start', tx: b.x + b.w + 6 };
				if (L_.right) cands.push(rightC, left); else cands.push(left, rightC);
				if (b.w > w + 12) cands.push({ x: b.x + 4, anchor: 'start', tx: b.x + 4, inside: true });
				const free = (c) => { const r_ = { x: c.x, y: yT, w, h: hT }; return c.x >= 0 && c.x + w <= W && !bars.some((q) => q !== b && hit(r_, q)) && !boxes.some((q) => hit(r_, q)); };
				const c = cands.find(free) || cands[0];
				L_.el.setAttribute('x', c.tx);
				L_.el.setAttribute('text-anchor', c.anchor);
				if (c.inside) L_.el.setAttribute('fill', L_.dark ? '#fff' : 'var(--ink)');
				boxes.push({ x: c.x, y: yT, w, h: hT });
				if (L_.total) mark2.x = Math.min(c.x, b.x) - 14;
			}
			if (mark2) anim.push({ el: mark('2', mark2.x, mark2.y), t0: mark2.t0, d: MO.t2, kind: 'op' });
			s('line', { x1: P0, x2: P1, y1: yAx, y2: yAx, stroke: 'var(--hair)' }, svg);
			for (let v = LO; v <= HI; v += 2) {
				const t = s('text', { class: 'num', x: x(v), y: yAx + 18, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--soft-t)' }, svg);
				t.textContent = v < 0 ? `−${-v}` : String(v);
			}
			const at = s('text', { x: P1, y: yAx + 40, 'text-anchor': 'end', 'font-size': 12, fill: 'var(--mid)' }, svg);
			at.textContent = 'better · test mBPB against the standard MoE';
			const aw = at.getComputedTextLength() || 260;
			const ax0 = P1 - aw - 8, ay = yAx + 36;
			s('path', { d: `M${ax0} ${ay}H${ax0 - 16}M${ax0 - 12} ${ay - 3.5}L${ax0 - 16} ${ay}L${ax0 - 12} ${ay + 3.5}`, fill: 'none', stroke: 'var(--mid)', 'stroke-width': 1.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, svg);
			// the third finding: beyond matched budgets, under Muon, one more visit against twice the experts (own scale)
			const iy = H0 + 58, tEnd = rowT(FR.length);
			const ig = s('g', { class: 'frow is-inset', 'data-f': '3' }, svg);
			s('rect', { class: 'hit', x: -12, y: iy - 6, width: W + 24, height: IH - 2, rx: 12 }, ig);
			const it = s('text', { x: 4, y: iy + 10, 'font-size': 12, 'font-weight': 600, fill: 'var(--ink)' }, ig);
			it.textContent = 'Beyond matched budgets, Muon · on its own axis';
			anim.push({ el: it, t0: tEnd, d: MO.t2, kind: 'op' });
			const LO3 = Math.floor(Math.min(...ex3.map((e) => e.d)) / 2) * 2;
			const x3 = (v) => P0 + ((P1 - P0) * (v - LO3)) / (0 - LO3);
			ex3.forEach((e, j) => {
				const yy = iy + (small ? 30 : 22) + j * (small ? 28 : 18);
				const tl = small ? s('text', { x: 4, y: yy - 4, 'font-size': 12, fill: 'var(--mid)' }, ig) : s('text', { x: P0 - 10, y: yy + 7.5, 'text-anchor': 'end', 'font-size': 12, fill: 'var(--mid)' }, ig);
				tl.textContent = e.n;
				anim.push({ el: tl, t0: tEnd + j * MO.stagger, d: MO.t2, kind: 'op' });
				anim.push({ el: s('rect', { x: x3(e.d), y: yy, width: x3(0) - x3(e.d), height: 8, rx: 2, fill: j ? 'var(--soft)' : 'var(--ink)' }, ig), t0: tEnd + j * MO.stagger, d: MO.t4, kind: 'grow', o: x3(0) });
				const vt = s('text', { class: 'num', x: x3(e.d) - 8, y: yy + 7.5, 'text-anchor': 'end', 'font-size': 12, fill: 'var(--mid)' }, ig);
				vt.textContent = sgn(e.d, 1);
				anim.push({ el: vt, t0: tEnd + j * MO.stagger + MO.t3, d: MO.t2, kind: 'op' });
			});
			s('line', { x1: x3(0), x2: x3(0), y1: iy + 16, y2: iy + (small ? 78 : 50), stroke: 'var(--hair)' }, ig);
			for (let v = 0; v >= LO3; v -= LO3 <= -12 ? 4 : 2) {
				const ya = iy + (small ? 80 : 54);
				s('line', { x1: x3(v), x2: x3(v), y1: ya, y2: ya + 4, stroke: 'var(--hair)' }, ig);
				const tt = s('text', { class: 'num', x: x3(v), y: ya + 15, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--soft-t)' }, ig);
				tt.textContent = v < 0 ? `−${-v}` : '0';
			}
			s('line', { x1: x3(LO3), x2: x3(0), y1: iy + (small ? 80 : 54), y2: iy + (small ? 80 : 54), stroke: 'var(--hair)' }, ig);
			anim.push({ el: mark('3', P1 + 18, iy + 31), t0: tEnd + MO.t4, d: MO.t2, kind: 'op' });
			anim.forEach((a) => a.el.style.willChange = 'opacity, transform');
			built = true;
			paint(reduced ? 1e9 : started ? performance.now() - t0 : 0);
		}
		let started = reduced, t0 = 0;
		function paint(ms) {
			for (const a of anim) {
				const e = easeOut(clamp((ms - a.t0) / a.d));
				if (a.kind === 'op') a.el.setAttribute('opacity', e.toFixed(3));
				else a.el.setAttribute('transform', e >= 1 ? '' : `translate(${a.o} 0) scale(${Math.max(0.001, e).toFixed(4)} 1) translate(${-a.o} 0)`);
			}
		}
		build();
		onResize(box, build);
		onVisible(box, () => {
			if (reduced) return;
			started = true;
			t0 = performance.now();
			animate((dt, now) => { paint(now - t0); return now - t0 < END; });
		}, 0.2);
		// a finding and its place in the chart point at each other
		const link = (n, on) => {
			finds[n - 1] && finds[n - 1].classList.toggle('is-hl', on);
			if (svgEl) svgEl.querySelectorAll(`[data-f="${n}"]`).forEach((e) => e.classList.toggle('is-hl', on));
		};
		finds.forEach((li, i) => {
			li.tabIndex = 0;
			['pointerenter', 'focus'].forEach((ev) => li.addEventListener(ev, () => link(i + 1, true)));
			['pointerleave', 'blur'].forEach((ev) => li.addEventListener(ev, () => link(i + 1, false)));
		});
		box.addEventListener('pointerover', (e) => { const g = e.target.closest('[data-f]'); if (g && g.dataset.f) link(+g.dataset.f, true); });
		box.addEventListener('pointerout', (e) => { const g = e.target.closest('[data-f]'); if (g && g.dataset.f) link(+g.dataset.f, false); });
		// published configurations, one layer down, all neutral
		const R8 = P.fig8_recipes.recipes;
		const PUB = T.pub;
		const rows = [{ name: T.design['phase1/m3'], adamw: BPB.M3.adamw, muon: BPB.M3.muon, ref: arx(C['phase1/m3'].source) }]
			.concat(PUB.map(([k, c]) => ({ name: T.design[c], adamw: R8[k].adamw, muon: R8[k].muon, ref: arx(C[c].source), note: k === 'Huginn' ? 'a' : null })));
		const t = document.querySelector('[data-pub]');
		const thead = h('thead', '', t);
		const hr = h('tr', '', thead);
		['Configuration', 'AdamW test BPB', 'mBPB', 'Muon test BPB', 'mBPB'].forEach((x) => h('th', '', hr, x));
		const tb = h('tbody', '', t);
		const base = h('tr', 'rt__base', tb);
		h('td', '', base, T.design['phase1/m0']);
		h('td', '', base, BPB.M0.adamw.toFixed(4));
		h('td', 'rt__d', base, '—');
		h('td', '', base, BPB.M0.muon.toFixed(4));
		h('td', 'rt__d', base, '—');
		rows.forEach((r) => {
			const tr = h('tr', '', tb);
			const nc = h('td', '', tr, r.name);
			refMark(nc, [r.ref]);
			for (const o of ['adamw', 'muon']) {
				const vc = h('td', '', tr, r[o].toFixed(4));
				if (r.note && o === 'adamw') h('sup', 'rt__nm', vc, r.note);
				h('td', 'rt__d', tr, sgn((r[o] - BPB.M0[o]) * 1000, 1));
			}
		});
		const notes = document.querySelector('[data-pub-notes]');
		h('p', '', notes, `Protocol: ${window.TYPED.data}, ${window.TYPED.tokenizer}, ${window.V.context}, ${window.V.scaleLong}, ${window.V.tokens}, seed ${TR.seed}. Learning rate: tuned once per optimizer on the standard MoE (AdamW ${lrS(TR.lr_adamw)}, Muon ${lrS(TR.lr_muon)}). The looped MoE at the end of the chain selects the same value. Every other configuration inherits it. Test BPB on the complete held-out split; mBPB against the standard MoE with the same optimizer.`);
		const m = h('p', '', notes, 'Each configuration follows the shape of its source at matched budgets; what is kept and what is adapted is listed per design in the ');
		const a = h('a', 'ev ev--doc', m, 'method map');
		a.href = BLOB + 'docs/method-map.md';
		a.rel = 'noopener';
		m.append('.');
		const n = h('p', '', notes);
		h('sup', 'rt__nm', n, 'a');
		const RR = P.reruns['methods/huginn'];
		n.append(` At the protocol learning rate the loss of this configuration fell slowly under AdamW; a rerun at learning rate ${lrS(RR.lr)} gave test BPB ${RR.test_bpb.toFixed(6)} and validation BPB ${RR.validation_bpb.toFixed(6)} (`);
		const i2 = h('a', 'ev ev--doc', n, `#${RR.issue}`);
		i2.href = ISSUE + RR.issue;
		i2.rel = 'noopener';
		n.append(').');
		const lk = h('p', 'ev-row', notes);
		const a1 = h('a', 'ev', lk, 'docs/results/phase1.md');
		a1.href = 'docs/results/phase1/';
		a1.rel = 'noopener';
	}

	/* ---------------------------------------------------------------- 6. the record */
	function record() {
		const th = REC.threads;
		const kinds = T.record.kinds;
		const colors = { study: '#16181d', run: '#3b404b', plan: '#5b616e', code: '#9aa0ab', pr: '#c9cdd4' };
		const stack = ['pr', 'code', 'plan', 'run', 'study'];
		const tally = document.querySelector('[data-tally]');
		kinds.forEach(([k, n]) => {
			const d = h('div', '', tally);
			const dt = h('dt', '', d, n);
			dt.style.setProperty('--c', colors[k]);
			const cnt = th.filter((x) => x.kind === k).length;
			const dd = h('dd', '', d, String(cnt));
			const bar = h('i', 'tally__b', dd);
			bar.style.background = colors[k];
			bar.style.width = `${(cnt / Math.max(...kinds.map(([q]) => th.filter((x) => x.kind === q).length))) * 100}%`;
		});
		// the record, one square per public thread, stacked on the day it was opened. The counts above double as the key.
		const box = document.querySelector('[data-act]');
		const day = (s_) => Date.parse(s_ + 'T00:00:00Z') / 86400000;
		const D0 = Math.min(...th.map((t) => day(t.opened))), D1 = Math.max(...th.map((t) => day(t.opened)));
		const nD = D1 - D0 + 1;
		const fmtD = (d) => new Date(d * 86400000).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' });
		const order = ['study', 'run', 'plan', 'code', 'pr'];
		const kindName = Object.fromEntries(kinds.map(([k, n]) => [k, n.replace(/s$/, '').toLowerCase()]));
		const byDay = Array.from({ length: nD }, () => []);
		th.forEach((t) => byDay[day(t.opened) - D0].push(t));
		byDay.forEach((a) => a.sort((x, y) => order.indexOf(x.kind) - order.indexOf(y.kind) || x.n - y.n));
		const CASE = 230;
		function draw() {
			box.textContent = '';
			const k = remK();
			const W = box.clientWidth / k;
			const per = W < 520 ? 2 : 3; // squares across one day
			const colW = W / nD, q = Math.min(7, (colW - 2) / per - 1.5), step = q + 1.5;
			const rows = Math.max(...byDay.map((a) => Math.ceil(a.length / per)));
			const top = 26, base = top + rows * step, H = base + 30;
			const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, height: H * k, class: 'tl', role: 'img', 'aria-label': `${th.length} public threads, one square each, on the day each was opened, ${fmtD(D0)} to ${fmtD(D1)}. Issue #${CASE}, the real case, is marked.` }, box);
			s('line', { class: 'tl__base', x1: 0, x2: W, y1: base + 1.5, y2: base + 1.5 }, svg);
			let cx = 0, cy = 0;
			byDay.forEach((a, d) => {
				const x0 = d * colW + (colW - per * step + 1.5) / 2;
				a.forEach((t, i) => {
					const x = x0 + (i % per) * step, y = base - (Math.floor(i / per) + 1) * step;
					const r = s('rect', { class: `tl__q tl__q--${t.kind}` + (t.n === CASE ? ' is-case' : ''), x, y, width: q, height: q, rx: 1 }, svg);
					s('title', {}, r).textContent = `#${t.n} · ${kindName[t.kind]} · opened ${fmtD(day(t.opened))}`;
					if (t.n === CASE) { cx = x + q / 2; cy = y; }
				});
				if (!a.length) s('line', { class: 'tl__tick', x1: d * colW + colW / 2, x2: d * colW + colW / 2, y1: base + 1.5, y2: base + 4.5 }, svg);
			});
			// the case beside: its square, and a label above the stacks
			if (cx) {
				const g = s('g', { class: 'tl__case', 'data-case': '' }, svg);
				s('path', { class: 'tl__lead', d: `M${cx} ${cy - 2}V${12}` }, g);
				const t = s('text', { x: cx + 4, y: 10 }, g);
				t.textContent = `#${CASE}, the real case`;
			}
			[D0, D1].forEach((d, j) => {
				const t = s('text', { class: 'tl__d', x: j ? W : 0, y: H - 8, 'text-anchor': j ? 'end' : 'start' }, svg);
				t.textContent = fmtD(d);
			});
			const mid = s('text', { class: 'tl__d', x: W / 2, y: H - 8, 'text-anchor': 'middle' }, svg);
			mid.textContent = 'day each thread was opened';
		}
		draw();
		const key = h('p', 'tl__key', box.parentNode);
		key.textContent = 'Each square is one public thread: a study, a run task, a plan, a code task or a pull request.';
		onResize(box, draw);
		// the record unrolls by day as it comes into view (a clip, nothing redrawn)
		if (!reduced) { box.classList.add('is-wait'); onVisible(box, () => box.classList.replace('is-wait', 'is-on'), 0.3); }
		// the case card and its square point at each other
		const caseCard = document.querySelector('.card--case');
		const hl = (on) => { caseCard.classList.toggle('is-hl', on); box.querySelectorAll('.is-case, .tl__case').forEach((e) => e.classList.toggle('is-hl', on)); };
		caseCard.addEventListener('pointerenter', () => hl(true));
		caseCard.addEventListener('pointerleave', () => hl(false));
		box.addEventListener('pointerover', (e) => { if (e.target.closest('.tl__case, .is-case')) hl(true); });
		box.addEventListener('pointerout', (e) => { if (e.target.closest('.tl__case, .is-case')) hl(false); });
		// how an outside result enters: one folder by pull request, the bench check, a merged record
		const en = document.querySelector('[data-entry]');
		if (en) en.innerHTML = `<svg viewBox="0 0 200 52" class="entry__svg"><g class="entry__f"><path d="M4 6h13l4 4h21v24H4z"/></g><path class="entry__l" d="M48 20H70"/><g class="entry__c"><path d="M74 20h44"/><path d="M84 13v14M108 13v14"/><circle cx="97" cy="20" r="3.4"/></g><path class="entry__l" d="M122 20H146"/><g class="entry__m"><path d="M150 20h38"/><path d="M188 14v12"/></g><text class="entry__t" x="23" y="49" text-anchor="middle">folder</text><text class="entry__t" x="96" y="49" text-anchor="middle">check</text><text class="entry__t" x="169" y="49" text-anchor="middle">merged</text></svg>`;
	}


	/* ---------------------------------------------------------------- the page's own small figures */
	function vocabulary() {
		const m3 = C['phase1/m3'];
		// the release line carries the configuration it reports on
		const rel = document.querySelector('[data-release]');
		if (rel) rel.innerHTML = '<svg class="i" aria-hidden="true"><use href="#g-loop"/></svg>';
		// the three proof numbers, computed from the data, each with a small true figure
		const tolOf = { logits: 'logits_and_cached_logits', cached_logits: 'logits_and_cached_logits', loss: 'loss', gradient: 'gradient', updated_parameter: 'updated_parameter' };
		const Q = Object.keys(tolOf), cases = Object.keys(CHK.cases);
		const vals = cases.flatMap((c) => Q.map((q) => CHK.cases[c].errors[q] / CHK.tolerances[tolOf[q]].atol));
		const nb = BKEYS.filter((k) => Math.abs(m3.budgets[k].rel) <= m3.budgets[k].tol).length;
		const num = { configs: String(Object.keys(C).length), budgets: `${nb} of ${BKEYS.length}`, checks: `${vals.filter((v) => v <= 1).length} of ${vals.length}` };
		document.querySelectorAll('[data-pn]').forEach((n) => (n.textContent = num[n.dataset.pn]));
		const fig = (k) => document.querySelector(`[data-pf="${k}"]`);
		{
			// the lineage in miniature: the root, its trunk, and its first four children (indigo when they loop)
			const svg = s('svg', { viewBox: '0 0 56 32' }, fig('model'));
			const root = Object.keys(C).find((k) => !C[k].parent);
			const kids = Object.keys(C).filter((k) => C[k].parent === root).slice(0, 4);
			s('rect', { x: 2, y: 1, width: 4, height: 8, rx: 1.2, fill: 'var(--ink)' }, svg);
			s('path', { d: `M4 10V${8 + kids.length * 6 - 1}`, stroke: 'var(--hair)', 'stroke-width': 1.75, fill: 'none', 'stroke-linecap': 'round' }, svg);
			kids.forEach((k, i) => {
				const y = 13 + i * 6;
				s('path', { d: `M4 ${y - 3}V${y - 1}Q4 ${y} 6 ${y}H14`, stroke: 'var(--hair)', 'stroke-width': 1.75, fill: 'none', 'stroke-linecap': 'round' }, svg);
				s('rect', { x: 16, y: y - 2.5, width: 4, height: 5, rx: 1, fill: C[k].visits > 1 ? 'var(--acc)' : 'var(--ink)' }, svg);
				s('rect', { x: 23, y: y - 1, width: Math.min(32, T.short[k].length * 1.6), height: 2, rx: 1, fill: 'var(--hair)' }, svg);
			});
		}
		{
			const svg = s('svg', { viewBox: '0 0 56 32' }, fig('budget'));
			const x = (v) => 28 + (clamp(v, -0.08, 0.08) / 0.08) * 26;
			s('line', { x1: 1, x2: 55, y1: 16, y2: 16, stroke: 'var(--hair)', 'stroke-width': 1.5 }, svg);
			s('rect', { x: x(-0.05), y: 9, width: x(0.05) - x(-0.05), height: 14, rx: 7, fill: 'var(--surface-3)', stroke: 'var(--soft)', 'stroke-width': 1.5 }, svg);
			BKEYS.forEach((k) => s('circle', { cx: x(m3.budgets[k].rel), cy: 16, r: 3.2, fill: 'var(--ink)', stroke: 'var(--surface-3)', 'stroke-width': 1 }, svg));
		}
		{
			// the tolerance plot in miniature: per quantity, the span of its six differences, all left of the tolerance line
			const svg = s('svg', { viewBox: '0 0 56 32' }, fig('check'));
			const xq = (r) => 4 + ((Math.log10(r) + 3) / 3) * 44;
			s('line', { x1: 51, x2: 51, y1: 1, y2: 31, stroke: 'var(--ink)', 'stroke-width': 2, 'stroke-linecap': 'round' }, svg);
			Q.forEach((q, j) => {
				const y = 4 + j * 6;
				const rs = cases.map((c, i) => vals[i * Q.length + j]).filter((v) => v > 0);
				s('line', { x1: xq(Math.min(...rs)), x2: xq(Math.max(...rs)), y1: y, y2: y, stroke: 'var(--ink)', 'stroke-width': 3, 'stroke-linecap': 'round' }, svg);
			});
		}
		// each eyebrow shows where its section sits in the page, as a layer sits in the model
		const ebs = [...document.querySelectorAll('.eyebrow')];
		ebs.forEach((e) => {
			const idx = parseInt(e.querySelector('span').textContent, 10) - 1;
			const svg = s('svg', { class: 'eyebrow__s', viewBox: '0 0 56 12', 'aria-hidden': 'true' });
			s('line', { x1: 0, x2: 56, y1: 6, y2: 6, stroke: 'var(--line)', 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke' }, svg);
			for (let i = 0; i < ebs.length; i++) {
				const on = i === idx;
				s('rect', { x: 4 + i * 8 - (on ? 1.5 : 1), y: on ? 0 : 2, width: on ? 3 : 2, height: on ? 12 : 8, rx: 1, fill: on ? 'var(--ink)' : 'var(--hair)' }, svg);
			}
			e.prepend(svg);
		});
		// the converted checkpoints' error, on a rail of decades
		const st = document.querySelector('.card--stat .stat');
		if (st) {
			const box = h('div', 'statrail');
			box.setAttribute('aria-hidden', 'true');
			st.after(box);
			const svg = s('svg', { viewBox: '0 0 240 32' }, box);
			const X = (e) => 6 + ((e + 8) / 6) * 228;
			s('line', { x1: X(-8), x2: X(-2), y1: 8, y2: 8, stroke: 'var(--line)', 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke' }, svg);
			for (let e = -8; e <= -2; e++) {
				s('line', { x1: X(e), x2: X(e), y1: 4, y2: 12, stroke: 'var(--hair)', 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke' }, svg);
				if (e % 2 === 0) s('text', { x: X(e), y: 28, 'text-anchor': 'middle' }, svg).textContent = `1e${minus(String(e))}`;
			}
			s('circle', { cx: X(Math.log10(window.TYPED.stat.mantissa * 10 ** window.TYPED.stat.exponent)), cy: 8, r: 3.5, fill: 'var(--ink)' }, svg);
		}
		// adding a configuration is one step in the lineage: a parent, its child, the field it changes
		const af = document.querySelector('[data-addfig]');
		if (af) {
			const P_ = 'phase1/m0', K_ = 'phase1/m0-2e';
			const [f, a, b] = diffRows(C[K_].changes)[0];
			const svg = s('svg', { viewBox: '0 0 300 72' }, af);
			const mp = miniModel(C[P_], { slots: 12 }), mc = miniModel(C[K_], { slots: 12 });
			[[mp, 0], [mc, 196]].forEach(([m, x]) => { m.setAttribute('x', x); m.setAttribute('y', 6); m.setAttribute('width', 104); m.setAttribute('height', 52); svg.appendChild(m); });
			s('path', { d: 'M104 34H150V34H196', fill: 'none', stroke: 'var(--line)', 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke' }, svg);
			s('circle', { cx: 150, cy: 34, r: 2.5, fill: 'var(--ink)' }, svg);
			const t = s('text', { x: 150, y: 24, 'text-anchor': 'middle', fill: 'var(--ink)' }, svg);
			t.textContent = f;
			const t2 = s('text', { x: 150, y: 52, 'text-anchor': 'middle', fill: 'var(--mid)' }, svg);
			t2.textContent = `${a} → ${b}`;
			s('text', { x: 52, y: 70, 'text-anchor': 'middle', fill: 'var(--soft-t)' }, svg).textContent = 'parent';
			s('text', { x: 248, y: 70, 'text-anchor': 'middle', fill: 'var(--soft-t)' }, svg).textContent = 'your configuration';
		}
	}
	/* the published table: its head and first rows always in view; the rest on demand */
	function disclosure() {
		const disc = document.querySelector('[data-disc]');
		const tog = disc.querySelector('[data-disc-tog]');
		const lab = tog.querySelector('[data-disc-l]');
		const notes = disc.querySelector('[data-pub-notes]');
		const more = disc.querySelectorAll('tbody tr').length - 3;
		const set = (open) => {
			disc.classList.toggle('is-open', open);
			tog.setAttribute('aria-expanded', String(open));
			lab.textContent = open ? `− ${more} rows` : `+ ${more} rows`;
			notes.inert = !open;
			disc.querySelectorAll('tbody tr').forEach((r, i) => { r.inert = !open && i >= 3; });
		};
		tog.addEventListener('click', () => set(!disc.classList.contains('is-open')));
		set(false);
	}

	/* ---------------------------------------------------------------- the film */
	/* Muted, it plays while it is in view and stops when it leaves, the tab hides or the visitor holds it; nothing of it
	   loads until it is near the view. With reduced motion it waits on its poster for a press. Sound and full screen
	   are one press each; native controls only in full screen. The chapters below are its timeline. */
	function film() {
		const fig = document.querySelector('[data-reel]');
		if (!fig) return;
		const L = T.film;
		const v = fig.querySelector('[data-reel-v]');
		const big = fig.querySelector('[data-reel-big]');
		const playB = fig.querySelector('[data-reel-play]');
		const sndB = fig.querySelector('[data-reel-snd]');
		const fullB = fig.querySelector('[data-reel-full]');
		v.controls = false;
		v.muted = true;
		// the last stretch, the end card, is a rest: a line without a label
		const chs = [...fig.querySelectorAll('[data-reel-ch] li')].map((li) => ({ li, b: li.querySelector('button'), a: +li.dataset.a, z: +li.dataset.z }));
		let held = reduced; // the visitor holds it still: reduced motion, a pause, or a refused autoplay
		let seen = false, loaded = false, raf = 0, cur = -1;
		const isFull = () => document.fullscreenElement === v || document.webkitFullscreenElement === v || v.webkitDisplayingFullscreen === true;
		const load = () => { if (loaded) return; loaded = true; v.preload = 'auto'; v.load(); };
		const paint = () => {
			const t = v.currentTime;
			let k = -1;
			chs.forEach((c, i) => {
				c.li.style.setProperty('--f', clamp((t - c.a) / (c.z - c.a)).toFixed(4));
				if (t >= c.a && t < c.z) k = i;
			});
			if (k === cur) return;
			cur = k;
			chs.forEach((c, i) => { c.li.classList.toggle('is-on', i === k); if (!c.b) return; if (i === k) c.b.setAttribute('aria-current', 'time'); else c.b.removeAttribute('aria-current'); });
		};
		const tick = () => { paint(); raf = v.paused ? 0 : requestAnimationFrame(tick); };
		const ui = () => {
			const playing = !v.paused;
			playB.classList.toggle('is-paused', !playing);
			playB.setAttribute('aria-label', playing ? L.pause : L.play);
			playB.title = playing ? L.pause : L.play;
			big.hidden = !(held && v.paused) || isFull();
			sndB.setAttribute('aria-pressed', String(!v.muted));
			sndB.title = v.muted ? L.soundOn : L.soundOff;
		};
		// the events our own calls cause are expected, so a play or pause from the browser's controls reads as the visitor's
		const mine = { play: 0, pause: 0 };
		const play = () => {
			load();
			mine.play++;
			const p = v.play();
			if (p) p.catch(() => { mine.play = 0; if (v.paused) { held = true; ui(); } });
		};
		const stop = () => { mine.pause++; v.pause(); };
		const sync = () => {
			if (!held && !document.hidden && (seen || isFull())) { if (v.paused) play(); } else if (!v.paused) stop();
			ui();
		};
		v.addEventListener('play', () => { if (mine.play) mine.play--; else held = false; if (!raf) raf = requestAnimationFrame(tick); ui(); });
		v.addEventListener('pause', () => { if (mine.pause) mine.pause--; else held = true; paint(); ui(); });
		v.addEventListener('seeked', paint);
		v.addEventListener('volumechange', ui);
		const hold = (on) => { held = on; sync(); };
		v.addEventListener('click', () => { if (!isFull()) hold(!v.paused); });
		playB.addEventListener('click', () => hold(!v.paused));
		big.addEventListener('click', () => hold(false));
		sndB.addEventListener('click', () => { v.muted = !v.muted; if (!v.muted) held = false; sync(); });
		chs.forEach((c) => c.b && c.b.addEventListener('click', () => { load(); v.currentTime = c.a; paint(); hold(false); }));
		// full screen: the video alone, with the browser's own controls; the phone turns it sideways when it may
		const enter = v.requestFullscreen || v.webkitRequestFullscreen;
		if (!enter && !v.webkitEnterFullscreen) fullB.hidden = true;
		fullB.addEventListener('click', () => {
			held = false;
			load();
			if (enter) {
				const p = enter.call(v);
				const turn = () => { try { const o = screen.orientation; if (o && o.lock) o.lock('landscape').catch(() => {}); } catch (e) { /* no lock here */ } };
				if (p && p.then) p.then(turn, () => {}); else turn();
			} else v.webkitEnterFullscreen();
			sync();
		});
		const onFull = () => { v.controls = isFull(); sync(); };
		document.addEventListener('fullscreenchange', onFull);
		document.addEventListener('webkitfullscreenchange', onFull);
		v.addEventListener('webkitendfullscreen', onFull);
		// as it enters the view it starts to load (not with reduced motion: then only on a press); half in view, it plays
		new IntersectionObserver((es) => { if (es[0].isIntersecting && !reduced) load(); }, { rootMargin: '0px' }).observe(fig);
		new IntersectionObserver((es) => {
			const r = es[0].intersectionRatio;
			if (r >= 0.5) seen = true; else if (r < 0.15) seen = false; else return;
			sync();
		}, { threshold: [0, 0.15, 0.5] }).observe(fig.querySelector('.reel__stage'));
		document.addEventListener('visibilitychange', sync);
		paint();
		ui();
	}

	/* ---------------------------------------------------------------- page */
	function page() {
		const nav = document.querySelector('[data-nav]');
		const onScroll = () => nav.classList.toggle('is-on', scrollY > 8);
		addEventListener('scroll', onScroll, { passive: true });
		onScroll();
		const io = new IntersectionObserver((es) => {
			for (const e of es) {
				if (!e.isIntersecting) continue;
				e.target.classList.add('is-in');
				io.unobserve(e.target);
			}
		}, { rootMargin: '0px 0px -24px 0px', threshold: 0 });
		document.querySelectorAll('[data-rv]').forEach((n) => io.observe(n));
		// cards in one grid arrive one after another
		document.querySelectorAll('.bento').forEach((b) => [...b.children].forEach((c, i) => c.style.setProperty('--i', i)));
		// the section you are in, as a pill that slides between links
		const links = [...document.querySelectorAll('[data-navlinks] a[href^="#"]')];
		const pill = document.querySelector('[data-navpill]');
		const secs = links.map((a) => document.querySelector(a.getAttribute('href')));
		let here = -1;
		// the pill appears in place; it slides only between two links that are both shown
		const place = () => {
			const a = links[here];
			if (!a) { pill.classList.remove('is-on', 'is-move'); return; }
			const was = pill.classList.contains('is-on');
			pill.classList.toggle('is-move', was);
			if (!was) { void pill.offsetWidth; }
			pill.style.left = `${a.offsetLeft}px`;
			pill.style.width = `${a.offsetWidth}px`;
			if (!was) { void pill.offsetWidth; pill.classList.add('is-on'); }
		};
		const spy = () => {
			const y = innerHeight * 0.4;
			let k = -1;
			secs.forEach((sc, i) => { if (sc.getBoundingClientRect().top < y) k = i; });
			if (k >= 0 && secs[k].getBoundingClientRect().bottom < y * 0.5 && k === secs.length - 1) k = -1;
			if (k !== here) { here = k; links.forEach((a, i) => a.classList.toggle('is-here', i === k)); place(); }
		};
		addEventListener('scroll', spy, { passive: true });
		addEventListener('resize', place);
		spy();
		document.querySelectorAll('[aria-disabled="true"]').forEach((a) => {
			a.addEventListener('click', (e) => e.preventDefault());
			a.tabIndex = -1;
		});
		document.querySelectorAll('[data-tip]').forEach((b) => (b.title = b.dataset.tip));
		document.querySelectorAll('[data-copy]').forEach((b) => {
			b.addEventListener('click', async () => {
				const el = document.getElementById(b.dataset.copy);
				const txt = (b.dataset.text || el.textContent).replace(/^(\$|\d)\s+/gm, '').trim();
				try {
					await navigator.clipboard.writeText(txt);
					b.classList.add('is-done');
					b.title = T.copy.done;
				} catch (e) { /* the clipboard is not available here */ }
				setTimeout(() => { b.classList.remove('is-done'); b.title = T.copy.idle; }, 1600);
			});
		});
		const src = document.querySelector('[data-src]');
		src.append('Every value on this page is generated from the repository at commit ');
		const a = h('a', '', src, window.REPO_COMMIT);
		a.href = `${REPO}commit/${window.REPO_COMMIT}`;
		a.rel = 'noopener';
		src.append('.');
	}

	refs();
	hero();
	const pick = explorer();
	const exStage = document.querySelector('.explorer__main');
	const mechUI = designs((k) => {
		// a change never happens off screen: the explorer comes into view first, then shows the example
		const r = exStage.getBoundingClientRect();
		const seen = r.top >= 0 && r.top < innerHeight * 0.5 && r.bottom > innerHeight * 0.4;
		if (seen || reduced) { pick(k, true); return; }
		exStage.scrollIntoView({ behavior: 'smooth', block: 'center' });
		let done = false;
		const run = () => { if (done) return; done = true; removeEventListener('scrollend', run); pick(k, true); };
		addEventListener('scrollend', run, { once: true });
		setTimeout(run, 900);
	});
	mechUI.press(0);
	document.querySelector('[data-tree]').addEventListener('click', () => mechUI.clear());
	compare();
	check();
	study();
	record();
	vocabulary();
	disclosure();
	film();
	page();
	// the mark is sampled now, so its first frames never wait on the sampling
	// the copy reveals in one frame; the mark starts in a later one, so the two never share a frame
	const go = () => requestAnimationFrame(() => {
		root.classList.add('is-in');
		requestAnimationFrame(() => setTimeout(() => hero.start(), reduced ? 0 : 120));
	});
	if (document.fonts && document.fonts.ready) Promise.race([document.fonts.ready, new Promise((r) => setTimeout(r, 700))]).then(go);
	else go();
	// a change of the motion setting while the page is open is honoured
	matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', () => location.reload());
})();
