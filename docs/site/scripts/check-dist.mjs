// After the build: the homepage arrived (as synced), every documentation page exists, and every root-relative
// link or asset in the built HTML (homepage included) lies under the base and resolves to a file in dist/.
// Relative links in the homepage are checked too. Starlight's link validator has already checked the links
// inside the documents, anchors included. dist/ is what is served at the base (SITE_BASE).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BASE, PAGES, pageUrl } from '../nav.mjs';

const SITE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(SITE, 'dist');
const bad = [];

const home = fs.readFileSync(path.join(DIST, 'index.html'), 'utf8');
if (home !== fs.readFileSync(path.join(SITE, 'public/index.html'), 'utf8')) bad.push('dist/index.html differs from the synced homepage');

const file = (u) => path.join(DIST, decodeURI(u.split(/[?#]/)[0]).slice(BASE.length));
const exists = (u) => {
	const f = file(u);
	return fs.existsSync(f) && (fs.statSync(f).isFile() || fs.existsSync(path.join(f, 'index.html')));
};
for (const u of [pageUrl(''), ...PAGES.map((p) => pageUrl(p.slug))]) if (!exists(u)) bad.push(`missing page ${u}`);
if (!fs.existsSync(path.join(DIST, '404.html'))) bad.push('missing 404.html');

let n = 0;
for (const f of fs.readdirSync(DIST, { recursive: true })) {
	if (!String(f).endsWith('.html')) continue;
	const html = fs.readFileSync(path.join(DIST, String(f)), 'utf8');
	for (const m of html.matchAll(/\s(?:href|src)="(\/[^"/][^"]*|\/)"/g)) {
		n++;
		if (!m[1].startsWith(BASE)) bad.push(`${f}: ${m[1]} is outside the base ${BASE}`);
		else if (!exists(m[1])) bad.push(`${f}: ${m[1]}`);
	}
}
// the homepage's own relative links and assets
for (const m of home.matchAll(/\s(?:href|src)="(?![a-z]+:|#|\/)([^"]+)"/g)) {
	n++;
	if (!exists(BASE + m[1])) bad.push(`index.html: ${m[1]}`);
}
if (bad.length) {
	console.error(bad.map((b) => `  ${b}`).join('\n'));
	process.exit(1);
}
console.log(`dist (base ${BASE}): homepage as synced, ${PAGES.length + 1} documentation pages and 404, ${n} links resolve`);
