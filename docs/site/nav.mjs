// The one list of documentation pages: the sync step reads it to know what to copy and where,
// astro.config.mjs reads it to build the sidebar. A page is [source path in the repository, slug, sidebar label].
// The source path is null for the one page this site owns (the documentation landing page).

export const REPO_URL = 'https://github.com/datamllab/LoopTitan';

export const GROUPS = [
	{
		label: 'Start',
		pages: [
			[null, '', 'Overview'],
			['docs/development.md', 'development', 'Development and checks'],
		],
	},
	{
		label: 'Concepts',
		pages: [
			['docs/architecture.md', 'architecture', 'Architecture'],
			['docs/recipes.md', 'recipes', 'Recipes'],
			['docs/extensions.md', 'extensions', 'Extensions'],
		],
	},
	{
		label: 'Compare',
		pages: [
			['docs/comparison.md', 'comparison', 'Cost and comparison'],
			['docs/baselines.md', 'baselines', 'Baseline check'],
		],
	},
	{
		label: 'Train',
		pages: [
			['docs/training.md', 'training', 'Training and data'],
			['docs/validation.md', 'validation', 'Validation record'],
		],
	},
	{
		label: 'Results',
		pages: [['docs/results/phase1.md', 'results/phase1', 'Phase 1']],
	},
	{
		label: 'Protocol',
		pages: [
			['docs/protocol.md', 'protocol', 'Experiment protocol'],
			['.github/CONTRIBUTING.md', 'contributing', 'Enter a result'],
			['docs/torchtitan.md', 'torchtitan', 'TorchTitan record'],
		],
	},
	{
		label: 'Reference',
		pages: [
			['docs/method-map.md', 'method-map', 'Method map'],
			['docs/cvsm-adoption.md', 'cvsm-adoption', 'CVSM source'],
		],
	},
];

// One sentence per page for its description (meta description, link previews, llms.txt), by slug. A page without
// one takes the first sentence of its document.
export const DESCRIPTIONS = {
	development: 'Set up the LoopTitan development environment, run the tests and the conformance check of a looped MoE configuration, and read the per-visit diagnostics.',
	architecture: 'How LoopTitan builds every looped MoE design from one plain PyTorch model: physical blocks separate from logical calls, TorchTitan training, exact resume and Hugging Face export.',
	recipes: 'The published looped MoE recipes as LoopTitan configurations (SMELT, Foil, Tying, Huginn, LOOM, Ouro and others), their fields, and where to start reading the code.',
	extensions: 'Extend LoopTitan: write a new recurrent state update or forward pass, replace attention, add per-visit LoRA, and add a configuration with its conformance check.',
	comparison: 'The cost contract for comparing looped and standard MoE: unique parameters, training FLOPs and KV cache from the configuration, matched within 5% before training, and the Looped MoE Bench.',
	baselines: "LoopTitan's model checked against the official Hugging Face Transformers Qwen3-MoE (Qwen3MoeForCausalLM) in logits, gradients, optimizer steps and KV cache.",
	training: 'Train looped MoE language models on TorchTitan with AdamW or Muon: data preparation, distributed training, full-state resume, Hugging Face export, evaluation and generation.',
	validation: 'What LoopTitan has run on GPU, and the scope of each result: distributed training, full-state resume, Hugging Face delivery and downstream evaluation.',
	'results/phase1': 'Phase 1 results: standard and looped MoE configurations trained on 5.0B DCLM tokens with AdamW and with Muon, with test bits per byte, the three budgets and the run of every row.',
	protocol: 'How LoopTitan records its studies and runs: states and decisions, from recipe screen to pretraining, the run record interface and the limits of resume.',
	contributing: 'Enter a result in the Looped MoE Bench: train a frozen track and submit one folder by pull request; the bench workflow checks it, and failed runs and negative results are kept.',
	torchtitan: 'The optional experiment record of a TorchTitan run: where it enters the trainer, the batch and token definitions, and the MLflow recorder.',
	'method-map': 'From looped-model papers to LoopTitan mechanisms: what each recipe keeps from its source and adapts (SMELT, LatentLoop, Huginn, Ouro, LOOM and others), and a survey of related methods.',
	'cvsm-adoption': 'The source record of the CVSM protocol in LoopTitan: the commit it was copied from, what was copied, and the boundary of the adaptation.',
};

/** Every synced page: { source, slug, label, group, groupIndex }. */
export const PAGES = GROUPS.flatMap((g, gi) =>
	g.pages.filter(([source]) => source).map(([source, slug, label]) => ({ source, slug, label, group: g.label, groupIndex: gi }))
);

/** The base path the site is served under: SITE_BASE, '/' by default ('/LoopTitan/' on GitHub Pages). Always ends in '/'. */
export const BASE = `/${(process.env.SITE_BASE ?? '/').replace(/^\/+|\/+$/g, '')}/`.replace(/^\/\/$/, '/');

/** Site URL of a page slug, under the base. */
export const pageUrl = (slug) => `${BASE}docs/${slug ? `${slug}/` : ''}`;
