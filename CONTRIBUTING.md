# Contributing

## Rules that would disqualify us
Organiser clarification of 3 Nov 2026 overrides earlier answers:
- Train, fit, calibrate and tune **only on external data**. Never on the ground truth (including the public
  5 minutes) or the provided video. Every number in `kora/degrade.py` comes from `models/degrade_external.json`
  (`python -m kora.missingness`, external data only).
- No hand labels or hand corrections on the challenge match, for any purpose. No checking output against the video.
- Do not upload provided data to AI services: AI assistants work on code and external-data results only;
  challenge-side checks (`kora.checks`, `kora.profile`, `kora.shift`) are run and read by a person.
- Challenge data is read only by `kora/run.py` (inference) and the checks; `tests/test_compliance.py` enforces it.
- Never commit match data or anything derived from it (`data/`, `*.npz` are ignored).

## How a change gets in
1. Work on your own branch. Changes inside your own files (see below) and `RESEARCH.md` findings can go
   straight to `main`; open a PR only for files outside your area or `README.md`.
2. Add the change as a variant in `kora/variant.py` and pre-register it in `RESEARCH.md` (candidates and
   decision rule) **before** running outer folds.
3. Score it: `python -m kora.nested --variant <name>` (caches are keyed by code hash, so nothing stale is
   reused), then `python -m kora.compare <production> <name>`.
4. It goes in only if `kora.compare` prints KEEP: mean spellF1 up and better on at least 5 of the 7 matches.
   Then set `PRODUCTION` in `kora/variant.py`.
5. Tests: `.venv/bin/python -m unittest discover -s tests`. New code gets a test on synthetic data.
6. If the submission changes, update the hash: `shasum -a 256 "Kora Analytics.csv" > models/submission.sha256`,
   and commit the new `models/`. Anyone can then check `shasum -a 256 -c models/submission.sha256`.

## Who works where
Each track keeps to its own files so branches don't conflict; anything else, say so in the chat first.
- Younis, clean evaluation and features: `kora/nested.py`, `kora/compare.py`, `kora/variant.py`, `kora/features.py`,
  `kora/touch.py`, `kora/train.py`, `kora/missingness.py`, `kora/skillcorner.py`, `build.sh`, `docs/`.
- Vladymyr, error metrics and decoder: `kora/errors.py`, `kora/decode.py`, `kora/boundary.py`.
- José, missing-ball robustness and extra data: `kora/degrade.py`, `kora/external.py`.
- `RESEARCH.md`: edit only your own section; push findings straight to `main`, even for unfinished work.

## Commits
[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/): `feat:`, `fix:`, `docs:`, `refactor:`,
`perf:`, `test:`, `chore:`, optional scope, e.g. `feat(decode): duration prior per team`.

Commit with your GitHub noreply address (Settings > Emails); the repo will be public at submission.
