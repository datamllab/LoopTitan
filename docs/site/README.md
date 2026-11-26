# LoopTitan site

One static site built from this repository: the homepage at the base path and the documentation under `docs/`.
It is published at https://datamllab.github.io/LoopTitan/ by `.github/workflows/pages.yml`.

```sh
./docs/site/build.sh                          # base /: previews; output in docs/site/dist/
SITE_BASE=/LoopTitan/ ./docs/site/build.sh    # base /LoopTitan/: GitHub Pages
```

Requirements: Node 22.23.4 or later (`.nvmrc` selects 24.22.3; `build.sh` uses nvm when it is installed) and the
repository's model environment for the data step (below). `build.sh` runs four steps:

1. `scripts/gen_data.py` writes the homepage's data files (`home/data/*.js`) from this checkout.
2. `scripts/sync.mjs` copies the two sources into place; nothing it writes is edited by hand.
   - The homepage: every file of `home/` into `public/`, unchanged. Its links and assets are relative, so it needs
     no build and works under any base.
   - The documents: each page listed in `nav.mjs` from the working tree (`DOCS_REF=<ref>` reads a git ref instead)
     into `src/content/docs/docs/`. Links to other synced documents become site links under the base, links to any
     other repository path become GitHub links, bare `#123` becomes a link to the issue. A link to a path that is not
     in the repository fails the sync.
3. `astro build` (Astro 7.4.12, Starlight 0.42.11) with `base` from `SITE_BASE`. `starlight-links-validator` checks
   every internal link and anchor in the documents; Pagefind indexes the documentation for search. The build also
   writes `sitemap-index.xml` (the homepage and every page), `llms.txt` and `llms-full.txt` (below) and `robots.txt`
   at the base.
4. `scripts/check-dist.mjs`: the homepage arrived as synced, every page and `404.html` exist, every root-relative
   link and asset in the built HTML lies under the base and resolves, and so does every relative link of the homepage.

`npm ci` runs on first use (`node_modules/`). `npm run dev` syncs and serves with live reload after one build (search
works only in a build). `node_modules/`, `.astro/`, `dist/`, `public/`, `home/data/` and the synced documents are
ignored by git.

## The base path

`SITE_BASE` is the one setting: `/` by default, `/LoopTitan/` for GitHub Pages. `nav.mjs` reads it (`BASE`,
`pageUrl`) and both `astro.config.mjs` and `scripts/sync.mjs` take it from there. Serve `dist/` under the same base
to preview it, with `404.html` for missing pages.

## The homepage's data

No value on the homepage is kept current by hand. `scripts/gen_data.py` runs at every build and writes
`home/data/*.js` from the checkout: the configurations and their budgets (`scripts/lineage.py`, the recipe files,
`compare_recipes`), the command output of the Compare section, the baseline check record
(`docs/evidence/qwen3-cpu.json`, `moeloop.testing.CHECKS`) and the test count (`pytest --collect-only`), the
Phase 1 results (`docs/results/phase1.json`) and the training setup of the frozen recipes, the reference list of
`README.md`, and the commit.

It imports the repository and collects its tests, so it runs in the model environment: torch and the `data`, `hf`,
`train` and `report` extras, as the model job of `.github/workflows/test.yml`. By default `build.sh` runs it through
`uv run --frozen` with those extras; `PYTHON=/path/to/python` uses an existing environment instead.

`record.js` comes from `data/record-snapshot.json`, a committed snapshot of the repository's issues and merged pull
requests, dated by its `as_of`. The build needs no token. The pages workflow refreshes the snapshot with its own
read token before it publishes; to refresh the committed copy, run
`GH_TOKEN=... python docs/site/scripts/gen_data.py --refresh-record` (read-only, `gh api`) and commit it.

The few values that no repository file holds as data (a figure from a pull request, a case from an issue, the
studies a card links) are the `TYPED` block at the top of `home/content.js`, each with a comment naming its source.

## Where things are

| Path | What |
|---|---|
| `home/` | the homepage: `index.html`, `style.css`, `tokens.css` (the shared tokens), `app.js`, `content.js` (every string drawn by script, and `TYPED`), `brand/`, `fonts/` (Inter and Geist Mono, SIL Open Font License, licence files beside them) |
| `nav.mjs` | the pages, their slugs, sidebar labels and groups; the base path |
| `src/styles/theme.css` | the dark values of the homepage's tokens; Starlight's variables mapped onto them |
| `src/styles/docs.css`, `figures.css`, `ec.config.mjs` | the documentation in the homepage's vocabulary; code blocks |
| `src/components/starlight/` | overrides: Head, Header (the homepage's header), ThemeSelect, MobileMenuToggle, PageTitle (eyebrow), MarkdownContent, Footer |
| `src/components/DocsHome.astro`, `NotFound.astro` | the documentation landing page; the 404 page |
| `src/lib/sprite.mjs`, `mark.mjs` | the glyph sprite: the homepage's own symbols (read from its `index.html`) and five more in the same grammar; the mark |
| `src/lib/home.mjs`, `figures.mjs`, `vocab.mjs` | build-time figures from the homepage's `data/*.js`, `content.js` and the mechanism drawings in `app.js` |
| `src/lib/postprocess.mjs` | rendered-HTML refinements: tables, numeric cells, result bars, short result headers |
| `scripts/gen_data.py`, `data/record-snapshot.json` | the homepage's data; the record snapshot |
| `home/index.html` `<head>` | title, description, canonical URL, link previews (`brand/og.png`, 1200 × 630) and the structured data (schema.org JSON-LD: the software, the paper, the models); `src/lib/structured.mjs` gives the documentation the same graph |
| `nav.mjs` `DESCRIPTIONS` | the description of each documentation page |
| `src/llms.md`, `src/lib/llms.mjs`, `src/pages/` | `llms.txt` (after llmstxt.org: the summary, the paper, the findings, the models, the citation, then every page), `llms-full.txt` (the same, then every page in full), `robots.txt` |
| `scripts/sync.mjs`, `scripts/check-dist.mjs` | the sync step; the check of the built site |
| `tests/setup/test_site.py` (repository root) | the navigation names existing documents, every document is in the navigation, the homepage stays under the base |
