#!/bin/sh
# Rebuild the submission from scratch. Resumable: cached features and finished folds are reused.
# Needs the challenge bundle in ./data (or MTAH_DATA_DIR); downloads the open training matches.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -x "$PY" ]; then
    python3 -m venv .venv
    # Linux: the CPU torch wheel first, so the pin below doesn't pull the CUDA build.
    [ "$(uname -s)" = Linux ] && .venv/bin/pip install -q torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
    .venv/bin/pip install -q -r requirements.txt
fi
"$PY" -c "import torch" 2>/dev/null || { echo "torch missing from .venv: see README (Setup)" >&2; exit 1; }
# Keep macOS awake during training; Linux runs the same commands directly.
run_awake() {
    if [ "$(uname -s)" = Darwin ] && command -v caffeinate >/dev/null 2>&1; then
        caffeinate -i "$@"
    else
        "$@"
    fi
}
run_awake "$PY" -m kora.external             # 7 open Bundesliga matches -> data/cache
run_awake "$PY" -m kora.train                # classifier + decoder by leave-one-match-out CV
run_awake "$PY" -m kora.ensemble             # Y16: average stage 1 over the seed members, re-pick the decoder
run_awake "$PY" -m kora.boundary             # change detector + duration decoder, kept only if CV says it helps
"$PY" -m kora.run --out "${1:-Kora Analytics.csv}"
"$PY" -m kora.checks "${1:-Kora Analytics.csv}"
want=$(cut -d' ' -f1 models/submission.sha256)
got=$(shasum -a 256 "${1:-Kora Analytics.csv}" | cut -d' ' -f1)
if [ "$want" = "$got" ]; then
    echo "reproduced: identical to the committed submission"
else
    echo "DIFFERENT from the committed submission: $got" >&2
    exit 1
fi
