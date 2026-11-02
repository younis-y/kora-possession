# Contributing

## Rules that would disqualify us
- No hand labels or hand corrections on the challenge match, for any purpose.
- The public 5-minute ground truth is a format check only: never train, tune or choose between versions on it.
- Never commit match data or anything derived from it (`data/`, `*.npz` are ignored).

## How a change gets in
1. Branch from `main`, open a PR.
2. Run `./build.sh`. Paste the leave-one-match-out spellF1 it prints (mean, nested, per match).
3. Merge only if it beats `main` on the mean and on at least 5 of the 7 matches.
4. If the submission changes, update the hash: `shasum -a 256 "Kora Analytics.csv" > models/submission.sha256`,
   and commit the new `models/`. Anyone can then check `shasum -a 256 -c models/submission.sha256`.

## Commits
[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/): `feat:`, `fix:`, `docs:`, `refactor:`,
`perf:`, `test:`, `chore:`, optional scope, e.g. `feat(decode): duration prior per team`.

Commit with your GitHub noreply address (Settings > Emails); the repo will be public at submission.
