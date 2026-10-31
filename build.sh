#!/bin/sh
# Rebuild the submission from scratch. Resumable: cached features and finished folds are reused.
# Needs the challenge bundle in ./data (or MTAH_DATA_DIR); downloads the open training matches.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
caffeinate -i "$PY" -m kora.external          # 7 open Bundesliga matches -> data/cache
caffeinate -i "$PY" -m kora.train             # classifier + decoder by leave-one-match-out CV
caffeinate -i "$PY" -m kora.stack             # stage 2, kept only if CV says it helps
"$PY" -m kora.run --out "${1:-Kora Analytics.csv}"
"$PY" -m kora.checks "${1:-Kora Analytics.csv}"
