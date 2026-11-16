// The mark, read from the homepage's brand file (never redrawn): the medium master of the shell, one colour,
// filled with the accent as on the homepage.
import fs from 'node:fs';
import path from 'node:path';

const raw = fs.readFileSync(path.resolve(process.cwd(), 'public/brand/mark.svg'), 'utf8');
const medium = raw.match(/<symbol id="medium" viewBox="([^"]+)">([\s\S]*?)<\/symbol>/);
if (!medium) throw new Error('brand/mark.svg has no symbol "medium"');

export const markSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${medium[1]}" focusable="false" aria-hidden="true" fill="var(--acc)">${medium[2]}</svg>`;
