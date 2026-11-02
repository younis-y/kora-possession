# Contributing

## Rules that would disqualify us
- No hand labels or hand corrections on the challenge match, for any purpose.
- The public 5-minute ground truth is a format check only: never train, tune or choose between versions on it.
- Never commit match data or anything derived from it (`data/`, `*.npz` are ignored).

## How a change gets in
1. Branch from `main`, open a PR. Nobody pushes to `main` directly.
2. Score it and paste the leave-one-match-out spellF1 (mean, nested, per match) in the PR:
   - decoder or detector changes: put the shared `oof.npz` (training-match predictions, from the team
     OneDrive) in `models/` and run `python -m kora.boundary`. Minutes, no retraining.
   - feature, degradation or training changes: run `./build.sh`.
3. Merge only if it beats `main` on the mean and on at least 5 of the 7 matches.
4. If the submission changes, update the hash: `shasum -a 256 "Kora Analytics.csv" > models/submission.sha256`,
   and commit the new `models/`. Anyone can then check `shasum -a 256 -c models/submission.sha256`.

## Who works where
Each track keeps to its own files so branches don't conflict; anything else, say so in the chat first.
- Younis, clean evaluation: `kora/nested.py` (new), `build.sh`.
- Vladymyr, error metrics and decoder: `kora/errors.py`, `kora/decode.py`, `kora/boundary.py`.
- José, missing-ball robustness and extra data: `kora/degrade.py`, `kora/external.py`.
- `RESEARCH.md`: edit only your own section.

## Commits
[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/): `feat:`, `fix:`, `docs:`, `refactor:`,
`perf:`, `test:`, `chore:`, optional scope, e.g. `feat(decode): duration prior per team`.

Commit with your GitHub noreply address (Settings > Emails); the repo will be public at submission.
