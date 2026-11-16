// @ts-check
// LoopTitan site: the homepage at the base (a static folder copied into public/ by scripts/sync.mjs) and the
// documentation under docs/ (Starlight; pages synced from the repository's docs/). Pages and groups: nav.mjs.
// SITE_BASE sets the base path: '/' (default, previews) or '/LoopTitan/' (GitHub Pages); nav.mjs reads it too.
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import { BASE, GROUPS, REPO_URL } from './nav.mjs';

const sidebar = GROUPS.map((g) => ({
	label: g.label,
	items: g.pages.map(([, slug, label]) => ({ label, slug: slug ? `docs/${slug}` : 'docs' })),
}));

export default defineConfig({
	site: 'https://datamllab.github.io',
	base: BASE,
	trailingSlash: 'ignore',
	integrations: [
		starlight({
			title: 'LoopTitan',
			description: 'LoopTitan documentation: the design space of looped MoE in one model, compared on equal terms.',
			favicon: '/brand/favicon.svg',
			sidebar,
			social: [{ icon: 'github', label: 'GitHub', href: REPO_URL }],
			// tokens.css is the homepage's own file, linked by Head.astro; these map Starlight onto it.
			customCss: ['./src/styles/theme.css', './src/styles/docs.css', './src/styles/figures.css'],
			components: {
				Head: './src/components/starlight/Head.astro',
				Header: './src/components/starlight/Header.astro',
				ThemeSelect: './src/components/starlight/ThemeSelect.astro',
				PageTitle: './src/components/starlight/PageTitle.astro',
				MobileMenuToggle: './src/components/starlight/MobileMenuToggle.astro',
				MarkdownContent: './src/components/starlight/MarkdownContent.astro',
				Footer: './src/components/starlight/Footer.astro',
			},
			tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
			titleDelimiter: '·',
			credits: false,
			lastUpdated: false,
			pagination: true,
			// Every internal link, anchors included, is checked at build; a broken one fails the build.
			plugins: [starlightLinksValidator({ errorOnLocalLinks: false })],
		}),
	],
	vite: {
		build: { assetsInlineLimit: 0 },
	},
});
