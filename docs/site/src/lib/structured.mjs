// The structured data (schema.org JSON-LD) of the homepage: the software, the paper and the models. Read from its
// index.html, so the documentation carries the same graph and the two never drift apart.
import fs from 'node:fs';
import path from 'node:path';

const html = fs.readFileSync(path.resolve(process.cwd(), 'public/index.html'), 'utf8');
const block = html.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/);
if (!block) throw new Error('structured: the homepage has no <script type="application/ld+json">');

/** The parsed graph (it must parse: a broken block fails the build). */
export const graph = JSON.parse(block[1]);

/** The graph as one line, ready for a <script type="application/ld+json">. */
export const jsonLd = JSON.stringify(graph).replace(/</g, '\\u003c');

const byType = (t) => graph['@graph'].find((n) => n['@type'] === t);
/** The paper and the software, for the text files built from them. */
export const paper = byType('ScholarlyArticle');
export const software = byType('SoftwareSourceCode');
export const models = byType('Collection');
