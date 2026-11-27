/* ================================================================== TYPED BY HAND
   Values that no repository file holds as data, each with its source. Everything else on the page comes from
   data/*.js, which scripts/gen_data.py generates from the checkout at every build. */
window.TYPED = {
	// the largest maximum absolute logit difference in the conversion tables of the official checkpoints:
	// #199 (Ouro-1.4B and Ouro-2.6B, eager and SDPA, depths 1-4; largest 4.01e-5, Ouro-2.6B eager) and
	// #201 (Huginn, depths 1-32; largest 1.96e-5); argmax agrees in every case
	stat: { mantissa: 4, exponent: -5, pulls: [199, 201] },
	// the run blocked before launch in #230: the D/N 3 cool-down fork records decay_ratio = 60/298 = 0.2013,
	// study #223 declared run.schedule.decay_ratio=0.2 as Controlled (the Decision moved it to Adaptable)
	case: { issue: 230, field: 'decay_ratio', recorded: '0.2013', declared: '0.2' },
	// optimizer.expert_lr_multiplier of recipes/phase1/m7-elr-adamw.json and m4-elr-adamw.json: sqrt(2) = 1.4142
	// (tests/setup/test_site.py checks it)
	expertLr: '×1.41',
	// the Phase 1 studies whose arms this card reports, as docs/results/phase1.json names them ("the studies are #177,
	// #216 and #225"): #177 (the decomposition chain), #216 (its shape steps G, U, R) and #225 (the published designs)
	studies: [177, 216, 225],
	// published looped designs in the results table: nine by the counting rule of docs/method-map.md ("Counting rule");
	// the tenth source, the residual split, is a variant of the recipe it modifies and is not counted as a design
	designs: 9,
	// the training data and tokenizer of the Phase 1 track: eval/bench/tracks.json (dataset
	// mlfoundations/dclm-baseline-1.0-parquet; the Qwen/Qwen3-30B-A3B tokenizer) and docs/training.md
	data: 'DCLM baseline data',
	tokenizer: 'Qwen3 tokenizer'
};

/* Every string drawn by script lives here, so a Chinese version can be added beside it.
   Prose lives in index.html. Values come from data/*.js, generated from the repository; a name that states a count
   is a function of its configuration, so it always says what the data says. Loaded after data/*.js, before the body. */
(() => {
const C = window.CONFIGS;
const TYPED = window.TYPED;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];
const word = (n) => WORDS[n] || String(n);
const Word = (n) => { const w = word(n); return w[0].toUpperCase() + w.slice(1); };
const times = (v) => (v === 2 ? 'twice' : `${word(v)} times`);
const layersOf = (c) => c.pre + c.core + c.suf;
const experts = (c) => c.regions.core.experts;
const res = (c) => `×${+c.regions.core.residual.toFixed(2)}`;
const ratio = (a, b) => { const r = a / b; return r === 2 ? 'twice' : `${+r.toFixed(2)}×`; };
const cap = (w) => w[0].toUpperCase() + w.slice(1);
window.CONTENT = {
	en: {
		/* Each configuration is named by what it does, never by a paper or method name. */
		design: {
			'phase1/m0': (c) => `Standard MoE: ${layersOf(c)} layers, ${experts(c)} experts`,
			'phase1/m3': (c) => `Loop the middle ${c.core} layers ${times(c.visits)}, residual ${res(c)}`,
			'phase1/m4': (c) => `Unrolled twin: ${c.core} middle layers, run once`,
			'phase1/m7': 'Middle loop sharing only the experts',
			'phase1/m5': 'Middle loop sharing experts and router',
			'phase1/m1': (c) => `One expert bank for each ${c.visits} adjacent layers`,
			'phase1/m1s': (c) => `${Word(c.core)} expert banks reused in loop order`,
			'phase1/m2': (c) => `One expert bank for each ${c.visits} adjacent layers`,
			'phase1/m6': (c) => `${experts(c)} narrow experts, top-${c.topK}, nothing shared`,
			'phase1/m0-2e': (c) => `${cap(ratio(experts(c), experts(C[c.parent])))} the experts`,
			'phase1/m7-elr': () => `Experts-only loop, expert learning rate ${TYPED.expertLr}`,
			'phase1/m4-elr': () => `Unrolled twin, expert learning rate ${TYPED.expertLr}`,
			'shape/g': (c) => `Finer experts: ${experts(c)} per layer, top-${c.topK}`,
			'shape/u': (c) => `Deeper and narrower: ${layersOf(c)} uniform layers`,
			'shape/r': 'Looped expert layout, residual not scaled',
			'shape/m0r': (c) => `Middle residual ${res(c)} only`,
			'shape/m0lr': 'Middle output learning rate scaled only',
			'shape/m0i': 'Middle output initialisation scaled only',
			'shape/d24': (c) => `Deeper and narrower: ${layersOf(c)} layers`,
			'shape/d36': (c) => `Deeper and narrower: ${layersOf(c)} layers`,
			'p05/m0': (c) => `Standard MoE, ${layersOf(c)} layers`,
			'p05/m3': (c) => `Loop the middle ${c.core} of ${layersOf(c)} layers ${times(c.visits)}`,
			'p05/u': (c) => `The loop unrolled into ${layersOf(c)} layers`,
			'untie/u0': 'Middle loop untied: each visit owns its layers',
			'p1/m0': (c) => `Standard MoE, ${layersOf(c)} layers`,
			'p1/m3': (c) => `Loop the middle ${c.core} of ${layersOf(c)} layers ${times(c.visits)}`,
			'methods/smelt-v3': (c) => `Loop the middle ${c.core} layers ${times(c.visits)}, residual ${res(c)}`,
			'methods/smelt-v4': (c) => `Loop the middle ${c.core} layers ${times(c.visits)}, residual ${res(c)}`,
			'methods/latentloop-residual': 'Middle loop with a residual scale per visit',
			'methods/m0-loop': (c) => `Standard MoE, middle ${c.core} layers run ${times(c.visits)}`,
			'methods/meta': 'Shared middle stack between its own prelude and coda',
			'methods/foil': (c) => `Flattened core: ${ratio(experts(c), c.regions.prefix.experts)} the experts, attention per visit`,
			'methods/expert-tying': (c) => `One expert bank for ${c.visits} consecutive layers`,
			'methods/huginn': 'Recurrent core: input re‑enters each visit, random start',
			'methods/loom': 'Whole stack looped, router per visit, moving-average state',
			'methods/ouro': 'Whole stack looped with a learned exit',
			'methods/sandwich': 'Recurrent core with a decaying input re‑entry',
			'methods/relaxed': 'Whole stack in a cycle, low-rank delta per visit'
		},
		short: {
			'phase1/m0': 'Standard MoE', 'phase1/m3': (c) => `Middle loop ×${c.visits}`, 'phase1/m4': 'Unrolled twin', 'phase1/m7': 'Share experts only',
			'phase1/m5': 'Share experts and router', 'phase1/m1': (c) => `Bank per ${c.visits} layers`, 'phase1/m1s': 'Banks in loop order', 'phase1/m2': (c) => `Bank per ${c.visits} layers`,
			'phase1/m6': (c) => `${experts(c)} narrow experts`, 'phase1/m0-2e': (c) => `${cap(ratio(experts(c), experts(C[c.parent])))} the experts`, 'phase1/m7-elr': () => `Experts only, LR ${TYPED.expertLr}`, 'phase1/m4-elr': () => `Twin, expert LR ${TYPED.expertLr}`,
			'shape/g': 'Finer experts', 'shape/u': (c) => `Deeper, narrower: ${layersOf(c)}`, 'shape/r': 'Looped expert layout', 'shape/m0r': (c) => `Middle residual ${res(c)}`,
			'shape/m0lr': 'Middle output LR', 'shape/m0i': 'Middle output init', 'shape/d24': (c) => `Deeper, narrower: ${layersOf(c)}`, 'shape/d36': (c) => `Deeper, narrower: ${layersOf(c)}`,
			'p05/m0': (c) => `Standard MoE, ${layersOf(c)} layers`, 'p05/m3': (c) => `Middle loop, ${layersOf(c)} layers`, 'p05/u': (c) => `Unrolled, ${layersOf(c)} layers`, 'untie/u0': 'Loop untied',
			'p1/m0': (c) => `Standard MoE, ${layersOf(c)} layers`, 'p1/m3': (c) => `Middle loop, ${layersOf(c)} layers`, 'methods/smelt-v3': (c) => `Middle loop ×${c.visits}`, 'methods/smelt-v4': (c) => `Middle loop ×${c.visits}`,
			'methods/latentloop-residual': 'Residual per visit', 'methods/m0-loop': 'Loop without matching', 'methods/meta': 'Prelude, loop, coda', 'methods/foil': 'Flattened core',
			'methods/expert-tying': 'Tied expert banks', 'methods/huginn': 'Recurrent core', 'methods/sandwich': 'Decaying re-entry', 'methods/loom': 'Whole-stack loop',
			'methods/ouro': 'Learned exit', 'methods/relaxed': 'Per-visit low rank'
		},
		/* the chips on the first screen: every one inside the tolerance */
		hero: ['phase1/m0', 'phase1/m3', 'methods/huginn', 'methods/ouro', 'methods/expert-tying'],
		heroShort: { 'methods/ouro': 'Whole stack', 'methods/expert-tying': 'Tied experts' },
		heroStart: 'phase1/m3',
		budget: {
			unique_non_embedding_parameters: 'Parameters',
			training_flops_per_token: 'Training FLOPs',
			logical_kv_bytes: 'KV cache'
		},
		budgetLong: {
			unique_non_embedding_parameters: 'Unique parameters',
			training_flops_per_token: 'Training FLOPs per token',
			logical_kv_bytes: 'KV cache'
		},
		mech: {
			visitResidual: 'residual per visit',
			attentionPerVisit: 'attention per visit',
			routerPerVisit: 'router per visit',
			normPerVisit: 'norms per visit',
			lora: 'low-rank delta per visit',
			injection: 'input re‑enters',
			random: 'random start',
			ema: 'moving-average state',
			exit: 'learned exit',
			layerOrder: 'each layer repeats in place'
		},
		decisions: [
			{ k: 'Where the loop sits', d: 'A prelude, the looped block, a coda.', f: 'prefix_cells · num_cells · suffix_cells', id: 'region', show: 'methods/meta' },
			{ k: 'Number of visits', d: 'How many times the looped block runs.', f: 'num_visits', id: 'visits', show: 'methods/smelt-v4' },
			{ k: 'What a visit shares', d: 'Attention, router and norms: shared, or owned by each\u00a0visit.', f: 'attention_per_visit · router_per_visit · norm_per_visit', id: 'owners', show: 'phase1/m7' },
			{ k: 'Residual scale', d: 'How much a looped layer adds to the stream.', f: 'core.residual_scale · visit_residual_scales', id: 'residual', show: 'methods/latentloop-residual' },
			{ k: 'Input re-entry', d: 'The input is added again at every visit.', f: 'input_injection · injection_dt', id: 'input', show: 'methods/huginn' },
			{ k: 'Start and update', d: 'Where the state starts, and how a visit updates it.', f: 'initial_state · state_update', id: 'state', show: 'methods/loom' },
			{ k: 'Depth rule', d: 'A fixed number of visits, or a learned exit.', f: 'exit_distribution', id: 'depth', show: 'methods/ouro' },
			{ k: 'Low-rank delta per visit', d: 'A small correction that each visit owns.', f: 'lora_attention_rank', id: 'lora', show: 'methods/relaxed' },
			{ k: 'Expert tying', d: 'One expert bank serves adjacent layers.', f: 'order', id: 'tying', show: 'methods/expert-tying' }
		],
		mlabel: {
			region: 'loop region', visits: 'visits', owners: 'per-visit owners', residual: 'residual scale', input: 'input re-entry',
			state: 'start and update', depth: 'depth rule', lora: 'low-rank delta', tying: 'expert tying',
			width: 'width', experts: 'experts', layers: 'layers', middle: 'middle region', lr: 'learning rate', outlr: 'middle output learning rate', outinit: 'middle output initialisation', unroll: 'the loop unrolled'
		},
		groups: { phase1: 'Phase 1', methods: 'Published designs', shape: 'Shape steps', untie: 'Untied', p05: 'Larger scale', p1: 'Smaller scale' },
		checks: {
			quantities: { logits: 'Logits', loss: 'Loss', gradient: 'Gradients', updated_parameter: 'Optimizer step', cached_logits: 'Cached logits' },
			cases: {
				ordinary_moe: 'Standard MoE', shared_stack: 'Shared stack', shared_layer: 'Shared layer',
				partial_attention: 'Attention per visit', partial_router: 'Router per visit', expert_tying: 'Expert tying'
			}
		},
		chain: [
			['M0', 'Standard MoE', 'phase1/m0'],
			['G', 'Finer experts', 'shape/g'],
			['U', 'Deeper and narrower', 'shape/u'],
			['R', 'Looped expert layout', 'shape/r'],
			['M4', (c) => `Residual ${res(c)}`, 'phase1/m4'],
			['M3', 'The loop', 'phase1/m3']
		],
		record: {
			kinds: [['study', 'Studies'], ['run', 'Run tasks'], ['plan', 'Plans'], ['code', 'Code tasks'], ['pr', 'Pull requests']],
			// the label of the branches past the present
			next: ['Branch from any result.'],
			axis: 'every public issue, in the order opened',
			aria: (n, studies, runs, d) => `The public record as a tree, to ${d}: ${n} GitHub issues and pull requests, among them ${studies} studies and ${runs} run tasks. Each study or plan is a branch with its tasks on it; every issue stays. New branches start from results across the record.`
		},
		copy: { idle: 'Copy', done: 'Copied' },
		// the entry of an outside result, as .github/CONTRIBUTING.md describes it
		entry: ['folder', 'pull request', 'bench check', 'merged'],
		film: { pause: 'Pause the film', play: 'Play the film', soundOn: 'Turn the music on', soundOff: 'Turn the music off' },
		/* the published configurations of the results table: name in the results data, configuration */
		pub: [['V3', 'methods/smelt-v3'], ['V4', 'methods/smelt-v4'], ['LatentLoop-Residual', 'methods/latentloop-residual'], ['Meta', 'methods/meta'], ['Foil', 'methods/foil'], ['Tying', 'methods/expert-tying'], ['Huginn', 'methods/huginn'], ['Sandwich', 'methods/sandwich'], ['LOOM', 'methods/loom'], ['Ouro', 'methods/ouro'], ['Relaxed', 'methods/relaxed']]
	}
};
const T = window.CONTENT.en;
for (const g of ['design', 'short']) for (const k in T[g]) if (typeof T[g][k] === 'function') T[g][k] = T[g][k](C[k]);
T.chain.forEach((r) => { if (typeof r[1] === 'function') r[1] = r[1](C[r[2]]); });

/* ------------------------------------------------------------------ the first-paint values
   Each [data-v] element is filled from the data before it is shown: fillV() runs inline right after its markup. */
const K = window.CHECKS, P = window.PAPER, R = window.RECORD, CMP = window.COMPARE;
const tolOf = { logits: 'logits_and_cached_logits', cached_logits: 'logits_and_cached_logits', loss: 'loss', gradient: 'gradient', updated_parameter: 'updated_parameter' };
const cases = Object.keys(K.cases);
const ok = cases.flatMap((c) => Object.keys(tolOf).map((q) => K.cases[c].errors[q] <= K.tolerances[tolOf[q]].atol));
const m3 = C['phase1/m3'], BK = ['unique_non_embedding_parameters', 'training_flops_per_token', 'logical_kv_bytes'];
const upd = [...new Set(cases.map((c) => K.cases[c].updates))];
// the published table: the looped MoE and every published configuration; the number of designs is typed (TYPED.designs)
const pubKeys = ['phase1/m3', ...T.pub.map((x) => x[1])];
const nDesigns = TYPED.designs;
const B = (n) => `${(n / 1e9).toFixed(1)}B`;
const TR = P.fig3_instrument.training;
const V = (window.V = {
	configs: String(Object.keys(C).length),
	budgets: `${BK.filter((b) => m3.budgets[b].within).length} of ${BK.length}`,
	checks: `${ok.filter(Boolean).length} of ${ok.length}`,
	threads: String(R.threads.length),
	checksWithin: `${ok.filter(Boolean).length} of ${ok.length} within tolerance`,
	checksAll: `the ${ok.length} values`,
	cases: Word(cases.length),
	casesLabel: `${Word(cases.length)} configurations against `,
	updates: upd.length === 1 ? word(upd[0]) : upd.join(' or '),
	optimizer: K.optimizer,
	dtype: K.dtype,
	device: K.device === 'cpu' ? 'a CPU' : `a ${K.device.toUpperCase()}`,
	transformers: K.transformers,
	tests: K.tests.toLocaleString('en-US'),
	pubTitle: `${Word(pubKeys.length)} configurations of ${word(nDesigns)} published looped designs ran under the same protocol`,
	scaleShort: `${B(P.scale.stored_non_embedding)} stored · ${B(P.scale.active_non_embedding)} active`,
	scaleLong: `${B(P.scale.stored_non_embedding)} stored and ${B(P.scale.active_non_embedding)} active parameters`,
	tokens: `${Math.round((TR.horizon_5B_updates * TR.tokens_per_update) / 1e9)}B tokens`,
	context: `${TR.seq_len.toLocaleString('en-US')}-token context`,
	cmd: CMP.command.replace(/ (recipes\/)/g, ' \\\n    $1'),
	studies: TYPED.studies.filter((n) => R.threads.some((t) => t.n === n && t.kind === 'study')),
	commit: window.REPO_COMMIT
});
V.scale = `${V.scaleShort} · ${V.tokens}`;
window.PUB_KEYS = pubKeys;
const ISSUE = 'https://github.com/datamllab/LoopTitan/issues/', PULL = 'https://github.com/datamllab/LoopTitan/pull/';
const link = (cls, href, text) => { const a = document.createElement('a'); a.className = cls; a.href = href; a.rel = 'noopener'; a.textContent = text; return a; };
window.fillV = () => {
	for (const e of document.querySelectorAll('[data-v]:not([data-vd])')) {
		e.setAttribute('data-vd', '');
		const k = e.dataset.v;
		if (k === 'studies') { e.textContent = ''; V.studies.forEach((n) => e.append(link('ev ev--doc', ISSUE + n, `Study #${n}`))); }
		else if (k === 'stat') {
			const t = TYPED.stat;
			e.setAttribute('aria-label', `${t.mantissa} times 10 to the minus ${-t.exponent}`);
			e.innerHTML = `${t.mantissa}<span class="stat__x" aria-hidden="true">×</span><span aria-hidden="true">10</span><span class="stat__e" aria-hidden="true">−${-t.exponent}</span>`;
		} else if (k === 'statPulls') { e.textContent = ''; TYPED.stat.pulls.forEach((n) => e.append(link('ev ev--doc', PULL + n, `#${n}`))); }
		else if (k === 'caseIssue') { e.href = ISSUE + TYPED.case.issue; e.textContent = `#${TYPED.case.issue}`; }
		else if (k === 'caseRecorded') e.textContent = `${TYPED.case.field} = ${TYPED.case.recorded}`;
		else if (k === 'caseDeclared') e.textContent = TYPED.case.declared;
		else if (k === 'caseField') e.textContent = TYPED.case.field;
		else if (k === 'cmd') { const p = e.querySelector('.term__p'); e.textContent = V.cmd; if (p) e.prepend(p); }
		else e.textContent = V[k];
	}
};
})();

