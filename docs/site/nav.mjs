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

/** Every synced page: { source, slug, label, group, groupIndex }. */
export const PAGES = GROUPS.flatMap((g, gi) =>
	g.pages.filter(([source]) => source).map(([source, slug, label]) => ({ source, slug, label, group: g.label, groupIndex: gi }))
);

/** The base path the site is served under: SITE_BASE, '/' by default ('/LoopTitan/' on GitHub Pages). Always ends in '/'. */
export const BASE = `/${(process.env.SITE_BASE ?? '/').replace(/^\/+|\/+$/g, '')}/`.replace(/^\/\/$/, '/');

/** Site URL of a page slug, under the base. */
export const pageUrl = (slug) => `${BASE}docs/${slug ? `${slug}/` : ''}`;
