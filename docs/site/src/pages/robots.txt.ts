// robots.txt is read only at the root of a host; under a base path (/LoopTitan/ on GitHub Pages) this copy is for
// reference, and the sitemap it names is the one to use.
import type { APIRoute } from 'astro';
import { BASE } from '../../nav.mjs';

export const GET: APIRoute = ({ site }) =>
	new Response(`User-agent: *\nAllow: /\n\nSitemap: ${new URL(`${BASE}sitemap-index.xml`, site).href}\n`, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
