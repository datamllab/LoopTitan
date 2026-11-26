// Sync step: copies the two sources of the site into place. Nothing it writes is edited by hand.
//
//   1. The homepage: every file of HOME_SRC (default home/, a zero-build folder whose links and assets are all
//      relative) into public/, unchanged. public/ belongs to this step and is replaced on every run. The homepage's
//      data/*.js are generated into home/data/ by scripts/gen_data.py before this step (build.sh runs both).
//   2. The documents: each page of nav.mjs from the repository (REPO, default ../.., the checkout this folder sits in)
//      into src/content/docs/docs/<slug>.md, from the working tree, or from the git ref DOCS_REF if it is set.
//      The first heading becomes the page title; the description is the page's sentence in nav.mjs (DESCRIPTIONS),
//      or else the document's first sentence. Links are rewritten so that they resolve:
//        a link to another synced document  -> its page on this site (/docs/<slug>/#anchor)
//        a link to any other repository path -> the file or folder on GitHub (blob/main, tree/main)
//        a bare issue reference (#123)       -> the issue on GitHub
//      Stray angle brackets outside code (`<s>` in a table) are escaped so they read as text, as intended.
//   SITE_BASE (default '/', '/LoopTitan/' on GitHub Pages) prefixes every site link, as astro.config.mjs does.
//
// Usage: node scripts/sync.mjs        (npm run sync)
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fromMarkdown } from 'mdast-util-from-markdown';
import { gfmFromMarkdown } from 'mdast-util-gfm';
import { gfm } from 'micromark-extension-gfm';
import { BASE, DESCRIPTIONS, PAGES, REPO_URL, pageUrl } from '../nav.mjs';

const SITE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const HOME_SRC = path.resolve(SITE, process.env.HOME_SRC ?? 'home');
const REPO = path.resolve(SITE, process.env.REPO ?? '../..');
const REF = process.env.DOCS_REF ?? '';
const OUT_DOCS = path.join(SITE, 'src/content/docs/docs');
const PUBLIC = path.join(SITE, 'public');

/* ------------------------------------------------------------------ 1. homepage */
function syncHome() {
	if (!fs.existsSync(path.join(HOME_SRC, 'index.html'))) throw new Error(`no index.html in HOME_SRC ${HOME_SRC}`);
	fs.rmSync(PUBLIC, { recursive: true, force: true });
	fs.cpSync(HOME_SRC, PUBLIC, {
		recursive: true,
		// screenshots and design notes in the source folder are not part of the page
		filter: (src) => !/(^|\/)(shots|\.DS_Store)(\/|$)/.test(path.relative(HOME_SRC, src)) && !/\.(md|py)$/.test(src),
	});
	for (const need of ['tokens.css', 'fonts/inter.woff2', 'fonts/geist-mono.woff2', 'brand/mark.svg', 'brand/favicon.svg', 'data/configs.js', 'data/data.js', 'content.js']) {
		if (!fs.existsSync(path.join(PUBLIC, need))) throw new Error(`the homepage folder has no ${need}; the documentation uses it`);
	}
	const n = fs.readdirSync(PUBLIC, { recursive: true }).length;
	console.log(`homepage: ${path.relative(SITE, HOME_SRC)} -> public/ (${n} entries)`);
}

/* ------------------------------------------------------------------ 2. documents */
const git = (...args) => execFileSync('git', ['-C', REPO, ...args], { encoding: 'utf8', maxBuffer: 64 << 20 });
const read = (p) => (REF ? git('show', `${REF}:${p}`) : fs.readFileSync(path.join(REPO, p), 'utf8'));
const files = new Set(
	REF ? git('ls-tree', '-r', '--name-only', REF).split('\n').filter(Boolean) : git('ls-files', '--cached', '--others', '--exclude-standard').split('\n').filter(Boolean)
);
const dirs = new Set();
for (const f of files) for (let d = path.posix.dirname(f); d !== '.'; d = path.posix.dirname(d)) dirs.add(d);
const bySource = new Map(PAGES.map((p) => [p.source, p]));
const sha = git('rev-parse', '--short', REF || 'HEAD').trim();

const problems = [];

function resolveUrl(url, from) {
	if (/^[a-z][a-z0-9+.-]*:/i.test(url) || url.startsWith('#') || url.startsWith('/') || url.startsWith('//')) return null;
	const [p, hash = ''] = url.split(/#(.*)/s, 2);
	const target = path.posix.normalize(path.posix.join(path.posix.dirname(from), decodeURI(p))).replace(/\/$/, '');
	const anchor = hash ? `#${hash}` : '';
	if (bySource.has(target)) return pageUrl(bySource.get(target).slug) + anchor;
	if (target.startsWith('..')) {
		problems.push(`${from}: link leaves the repository: ${url}`);
		return null;
	}
	if (files.has(target)) return `${REPO_URL}/blob/main/${target}${anchor}`;
	if (dirs.has(target) || target === '.') return `${REPO_URL}/tree/main/${target === '.' ? '' : target}`;
	problems.push(`${from}: link to a path that is not in the repository at ${REF || 'the working tree'}: ${url}`);
	return null;
}

const plain = (n) => (n.value != null && n.type !== 'html' ? n.value : (n.children || []).map(plain).join(''));

function transform(src, from) {
	const tree = fromMarkdown(src, { extensions: [gfm()], mdastExtensions: [gfmFromMarkdown()] });
	const edits = [];
	let title = null;
	let description = '';
	const walk = (node, inLink) => {
		const s = node.position?.start.offset;
		const e = node.position?.end.offset;
		if (node.type === 'heading' && node.depth === 1 && title == null) {
			title = plain(node).trim();
			let end = e;
			while (src[end] === '\n') end++;
			edits.push([s, end, '']);
			return;
		}
		if (node.type === 'paragraph' && title != null && !description) {
			const text = plain(node).replace(/\s+/g, ' ').trim();
			const cut = text.search(/(?<=[.!?])\s/);
			description = (cut > 30 ? text.slice(0, cut) : text).slice(0, 220);
		}
		if (node.type === 'link' || node.type === 'image' || node.type === 'definition') {
			let to = resolveUrl(node.url, from);
			if (to && node.type === 'image' && to.startsWith(REPO_URL + '/blob/')) to = to.replace('/blob/', '/raw/');
			if (to) {
				const raw = src.slice(s, e);
				const at = node.type === 'definition' ? raw.indexOf(node.url) : raw.lastIndexOf('](' + node.url) + 2;
				if (at < (node.type === 'definition' ? 0 : 2)) problems.push(`${from}: cannot place link ${node.url}`);
				else edits.push([s + at, s + at + node.url.length, to]);
			}
			for (const c of node.children || []) walk(c, true);
			return;
		}
		if (node.type === 'text' && !inLink) {
			const raw = src.slice(s, e);
			for (const m of raw.matchAll(/(^|[^\w&/#\]])#(\d{1,4})\b/g)) {
				const at = s + m.index + m[1].length;
				edits.push([at, at + 1 + m[2].length, `[#${m[2]}](${REPO_URL}/issues/${m[2]})`]);
			}
			return;
		}
		if (node.type === 'html') {
			const raw = src.slice(s, e);
			if (!/^<!--[\s\S]*-->$/.test(raw.trim())) edits.push([s, e, raw.replace(/</g, '&lt;')]);
			return;
		}
		for (const c of node.children || []) walk(c, inLink);
	};
	walk(tree, false);
	if (title == null) problems.push(`${from}: no first-level heading for the title`);
	edits.sort((a, b) => b[0] - a[0]);
	let out = src;
	let last = Infinity;
	for (const [a, b, r] of edits) {
		if (b > last) throw new Error(`${from}: overlapping edits at ${a}`);
		out = out.slice(0, a) + r + out.slice(b);
		last = a;
	}
	return { title, description, body: out.trimStart() };
}

function syncDocs() {
	fs.mkdirSync(OUT_DOCS, { recursive: true });
	for (const f of fs.readdirSync(OUT_DOCS, { recursive: true })) {
		if (String(f).endsWith('.md')) fs.rmSync(path.join(OUT_DOCS, String(f)));
	}
	for (const page of PAGES) {
		const { title, description, body } = transform(read(page.source), page.source);
		const front = [
			'---',
			`title: ${JSON.stringify(title)}`,
			`description: ${JSON.stringify(DESCRIPTIONS[page.slug] ?? description)}`,
			`editUrl: ${JSON.stringify(`${REPO_URL}/blob/main/${page.source}`)}`,
			'---',
			`<!-- Generated by docs/site/scripts/sync.mjs from ${page.source} at ${sha}. Edit the source, not this file. -->`,
			'',
		].join('\n');
		const out = path.join(OUT_DOCS, `${page.slug}.md`);
		fs.mkdirSync(path.dirname(out), { recursive: true });
		fs.writeFileSync(out, front + body);
	}
	console.log(`documents: ${PAGES.length} pages from ${path.relative(SITE, REPO)} at ${REF || 'working tree'} (${sha}) -> src/content/docs/docs/, base ${BASE}`);
}

syncHome();
syncDocs();
if (problems.length) {
	console.error(problems.map((p) => `  ${p}`).join('\n'));
	throw new Error(`${problems.length} link problem(s) in the documents`);
}
