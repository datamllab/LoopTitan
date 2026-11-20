// Small, exact changes to the rendered HTML of a document (never to its text):
//  - every table sits in a scroll container with rounded edges;
//  - right-aligned (numeric) cells get tabular figures and a true minus sign;
//  - on the results page, each "Δ test" cell gets a small diverging bar of its own value, and the headers of the
//    ten-column tables take short forms (the full name stays in the title) so that the table fits the column.

const MINUS = (s) => s.replace(/(^|[\s(>])-(?=\d)/g, '$1−');
const SHORT = {
	'Δ test (mBPB)': ['Δ mBPB', 'Test BPB minus the standard MoE’s with the same optimizer, in mBPB (0.001 BPB)'],
	Validation: ['Val.', 'Validation BPB'],
	Parameters: ['Params', 'Unique non-embedding parameters against the standard MoE'],
	FLOPs: ['FLOPs', 'Training FLOPs per token against the standard MoE'],
	KV: ['KV', 'Logical KV cache against the standard MoE'],
	LR: ['LR', 'Learning rate'],
	Source: ['Src', 'Published designs it follows, numbered in Sources'],
	Test: ['Test', 'Test BPB on the complete held-out split'],
};
const isRight = (attrs) => /align="right"|text-align:\s*right/.test(attrs);

function deltaCell(attrs, inner) {
	const text = inner.replace(/<[^>]+>/g, '').trim();
	const v = Number(text.replace('−', '-'));
	if (!Number.isFinite(v)) return null;
	const R = 16; // mBPB at the end of the track
	const w = Math.min(Math.abs(v), R) / R;
	const cls = v < 0 ? 'is-neg' : v > 0 ? 'is-pos' : 'is-zero';
	const over = Math.abs(v) > R ? ' is-over' : '';
	return `<td${attrs} class="lt-delta ${cls}${over}"><span class="lt-delta__v">${MINUS(text)}</span><span class="lt-delta__track" aria-hidden="true"><span class="lt-delta__bar" style="--w:${w.toFixed(3)}"></span></span></td>`;
}

function table(html, slug) {
	const head = html.match(/<thead>([\s\S]*?)<\/thead>/);
	const heads = head ? [...head[1].matchAll(/<th([^>]*)>([\s\S]*?)<\/th>/g)].map((m) => m[2].replace(/<[^>]+>/g, '').trim()) : [];
	const deltaCol = slug === 'docs/results/phase1' ? heads.findIndex((h) => h.startsWith('Δ')) : -1;
	const cols = heads.length;
	const body = html.replace(/<tr>([\s\S]*?)<\/tr>/g, (row, cells) => {
		let i = -1;
		const out = cells.replace(/<t([dh])([^>]*)>([\s\S]*?)<\/t\1>/g, (cell, kind, attrs, inner) => {
			i++;
			if (kind === 'h' && slug === 'docs/results/phase1' && SHORT[inner.trim()]) {
				const [t, title] = SHORT[inner.trim()];
				return `<th${attrs}><abbr title="${title}">${t}</abbr></th>`;
			}
			if (kind === 'd' && i === deltaCol) {
				const d = deltaCell(attrs, inner);
				if (d) return d;
			}
			if (kind === 'd' && isRight(attrs)) return `<td${attrs} class="lt-num">${MINUS(inner)}</td>`;
			return cell;
		});
		return `<tr>${out}</tr>`;
	});
	const cls = ['lt-tbl', cols >= 7 ? 'lt-tbl--wide' : '', slug === 'docs/results/phase1' ? 'lt-tbl--results' : ''].filter(Boolean).join(' ');
	return `<div class="${cls}" tabindex="0" role="region" aria-label="Table, scrolls sideways">${body}</div>`;
}

export function postprocess(html, slug) {
	return html.replace(/<table>[\s\S]*?<\/table>/g, (t) => table(t, slug));
}
