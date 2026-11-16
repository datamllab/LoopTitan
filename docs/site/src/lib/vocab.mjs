// Small drawings of the homepage's vocabulary, as static SVG strings (the homepage draws them in app.js).

/** The eyebrow: a shaft with one tick per section, the current one raised (app.js, "each eyebrow shows where its section sits"). */
export function eyebrowSvg(idx, n) {
	let s = `<svg class="lt-eyebrow__s" viewBox="0 0 ${8 * n} 12" aria-hidden="true"><line x1="0" x2="${8 * n}" y1="6" y2="6" stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
	for (let i = 0; i < n; i++) {
		const on = i === idx;
		s += `<rect x="${4 + i * 8 - (on ? 1.5 : 1)}" y="${on ? 0 : 2}" width="${on ? 3 : 2}" height="${on ? 12 : 8}" rx="1" fill="${on ? 'var(--ink)' : 'var(--hair)'}"/>`;
	}
	return s + '</svg>';
}

/** The tolerance band with its dots: one dot per budget on a ±8% line, the ±tol band behind (app.js bandSvg). */
export function bandSvg(rels, tol = 0.05, dom = 0.08) {
	const x = (v) => (36 + (Math.max(-dom, Math.min(dom, v)) / dom) * 36).toFixed(2);
	const w = (tol / dom) * 36;
	let s = `<svg class="lt-band" viewBox="0 0 72 12" preserveAspectRatio="none" aria-hidden="true"><line x1="0" x2="72" y1="6" y2="6" stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
	s += `<rect class="lt-band__b" x="${36 - w}" y="1" width="${2 * w}" height="10" rx="5"/><line class="lt-band__z" x1="36" x2="36" y1="1" y2="11"/>`;
	for (const r of rels) s += `<circle class="lt-band__d${Math.abs(r) > tol ? ' is-out' : ''}" cx="${x(r)}" cy="6" r="2.5"/>`;
	return s + '</svg>';
}
