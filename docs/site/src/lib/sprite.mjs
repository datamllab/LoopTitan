// The glyphs: the homepage's own sprite (read from its index.html, so the two never drift apart) and the few
// glyphs only the documentation needs, drawn in the same grammar (16-unit grid, 1.5 stroke, round joins).
import fs from 'node:fs';
import path from 'node:path';

const html = fs.readFileSync(path.resolve(process.cwd(), 'public/index.html'), 'utf8');
const home = html.match(/<svg class="sprite"[^>]*>([\s\S]*?)<\/svg>/);
if (!home) throw new Error('sprite: the homepage has no <svg class="sprite">');
const homeSymbols = home[1].replace(/<!--[\s\S]*?-->/g, '').trim();

const S = 'fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"';
const docs = [
	// search: the homepage's lens of the filter field
	`<symbol id="g-search" viewBox="0 0 16 16"><g ${S}><circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5L14 14"/></g></symbol>`,
	// light and dark: a disc on the shaft, and the same disc half dark
	`<symbol id="g-light" viewBox="0 0 16 16"><path d="M.5 8h2.5M13 8h2.5" stroke="currentColor" stroke-width="1" opacity=".45"/><circle cx="8" cy="8" r="4.25" ${S}/><path d="M8 1v1.5M8 13.5V15" ${S}/></symbol>`,
	`<symbol id="g-dark" viewBox="0 0 16 16"><path d="M.5 8h2.5M13 8h2.5" stroke="currentColor" stroke-width="1" opacity=".45"/><circle cx="8" cy="8" r="4.25" ${S}/><path d="M8 3.75a4.25 4.25 0 0 1 0 8.5z" fill="currentColor"/></symbol>`,
	// menu: three layers; close: two crossing strokes
	`<symbol id="g-menu" viewBox="0 0 16 16"><path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h7" ${S}/></symbol>`,
	`<symbol id="g-close" viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8" ${S}/></symbol>`,
	// a source file
	`<symbol id="g-file" viewBox="0 0 16 16"><g ${S}><path d="M4 1.75h5l3 3v9.5H4z"/><path d="M9 1.75v3h3"/></g></symbol>`,
].join('');

/** The inline sprite every page carries once. */
export const sprite = `<svg class="sprite" aria-hidden="true" focusable="false" style="position:absolute;width:0;height:0;overflow:hidden">${homeSymbols}${docs}</svg>`;

/** One glyph. */
export const g = (id, cls = 'i') => `<svg class="${cls}" aria-hidden="true" focusable="false"><use href="#g-${id}"/></svg>`;
