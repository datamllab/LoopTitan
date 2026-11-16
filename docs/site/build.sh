#!/usr/bin/env bash
# One command: generate the homepage's data from this checkout, sync the homepage and the documents, build and
# validate. Output: dist/ (static), served at the base path.
#   ./build.sh                          base / (previews)
#   SITE_BASE=/LoopTitan/ ./build.sh    base /LoopTitan/ (GitHub Pages)
#   PYTHON=/path/to/python ./build.sh   run scripts/gen_data.py with this Python, one with the repository's model
#                                       environment (as CI's model job); by default `uv run --frozen` with the data,
#                                       hf, train and report extras
set -euo pipefail
cd "$(dirname "$0")"
if [ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]; then
	set +u; . "${NVM_DIR:-$HOME/.nvm}/nvm.sh"; nvm use --silent >/dev/null || nvm install; set -u
fi
node -e 'const [a,b,c]=process.versions.node.split(".").map(Number); if (a<22||(a===22&&(b<23||(b===23&&c<4)))) { console.error("Node >= 22.23.4 needed, found "+process.version); process.exit(1) }'
export PYTHONDONTWRITEBYTECODE=1
if [ -n "${PYTHON:-}" ]; then
	"$PYTHON" scripts/gen_data.py
else
	uv run --frozen --project ../.. --extra data --extra hf --extra train --extra report python scripts/gen_data.py
fi
[ -d node_modules ] || npm ci --no-fund --no-audit
npm run build
