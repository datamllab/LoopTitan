// @ts-check
// LoopTitan site: the homepage at the base (a static folder copied into public/ by scripts/sync.mjs) and the
// documentation under docs/ (Starlight; pages synced from the repository's docs/). Pages and groups: nav.mjs.
// SITE_BASE sets the base path: '/' (default, previews) or '/LoopTitan/' (GitHub Pages); nav.mjs reads it too.
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import { BASE, GROUPS, REPO_URL } from './nav.mjs';

const sidebar = GROUPS.map((g) => ({
	label: g.label,
	items: g.pages.map(([, slug, label]) => ({ label, slug: slug ? `docs/${slug}` : 'docs' })),
}));

const SITE = 'https://datamllab.github.io';
// The share image of the homepage (home/brand/og.png, 1200 × 630), for every documentation page too.
const ogImage = new URL(`${BASE}brand/og.png`, SITE).href;

export default defineConfig({
	site: SITE,
	base: BASE,
	trailingSlash: 'ignore',
	integrations: [
		starlight({
			title: 'LoopTitan',
			description: 'LoopTitan documentation: the design space of looped mixture-of-experts (MoE) in one model, compared on equal terms.',
			head: [
				{ tag: 'meta', attrs: { property: 'og:image', content: ogImage } },
				{ tag: 'meta', attrs: { property: 'og:image:width', content: '1200' } },
				{ tag: 'meta', attrs: { property: 'og:image:height', content: '630' } },
				{ tag: 'meta', attrs: { property: 'og:image:alt', content: 'LoopTitan. A common ground for looped MoE research.' } },
			],
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
		// Starlight's sitemap lists the documentation; the homepage is a static file, so it is added here.
		sitemap({ customPages: [new URL(BASE, SITE).href] }),
	],
	vite: {
		build: { assetsInlineLimit: 0 },
	},
});
