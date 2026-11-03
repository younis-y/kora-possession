# Kora Analytics: possession from automatic tracking, learned on other matches

## 1. Approach

A frame classifier learns p(H, A, D | tracking around this frame) on **other matches**. A hybrid
hidden-Markov decoder then turns those probabilities into a label path with realistic possessions.
Nothing in the pipeline is fitted to, tuned on, or selected with labels from the challenge match.

Why learned rather than hand rules: every hand rule has a threshold, and with no labels on this
match the only honest way to set thresholds is on other matches where the answer is known. Once that
is the protocol, a model that learns the thresholds jointly from about a million labelled frames
does it better than a person picking them one at a time.

## 2. Training data (all from other games)

| dataset | what | labels | who labelled | size |
|---|---|---|---|---|
| IDSSE open data (Bassek, Weber, Rein, Memmert, *Scientific Data* 2025, doi:10.1038/s41597-025-04505-y), CC BY 4.0 | 7 full Bundesliga / 2. Bundesliga 2022/23 matches, DFL positional data, 25 Hz | per-frame `BallStatus` (alive/dead) and `BallPossession` (home/guest) from the official tracking feed | DFL / Sportec Solutions, official match data | about 1.0 M labelled frames |
| SkillCorner open data (github.com/SkillCorner/opendata, commit 4340d27), MIT licence, credit SkillCorner | 20 A-League 2024/25 matches of automatic broadcast tracking, 10 fps, with `is_detected` for the ball and every player | none used as training labels; only its detection gaps are measured (section 3) | n/a | about 1.2 M frames |

These labels are the same kind the challenge ground truth is derived from: corrected tracking that
marks the ball dead or names the team that owns it. No event data, possession data or annotation
of the challenge match is used anywhere.

## 3. Making the training data look like automatic tracking (external measurements only)

The training tracking is corrected and complete; the challenge tracking is automatic and loses the
ball and players. A model trained on clean data leans on signals that are missing at test time, so
each training copy is degraded at random (domain randomisation). Every number in the degradation
comes from **external data** (`kora/missingness.py` -> `models/degrade_external.json`):

| process | measured on | how |
|---|---|---|
| P(ball missing \| dead), P(ball missing \| alive) | SkillCorner | every frame of both periods (camera cuts included); a frame with no possession owner stands in for a stoppage: 0.79 and 0.077 |
| live ball-gap lengths | SkillCorner | gaps flanked by a known possession on both sides |
| blackout offsets inside a stoppage | SkillCorner | ball gaps inside runs with no possession owner |
| player visible / lost stretches, close-ups | SkillCorner | `is_detected` per player; stoppages where fewer than half the usual players are detected |
| positional noise | SkillCorner vs IDSSE | excess variance of the second difference of positions |

Every quantity is measured on every frame of the external stream. A camera-on-only estimate of ball
loss (0.14 in stoppages) was also computed and rejected because it drops the stoppage frames where
tracking loses the ball. The organisers' documentation reports the challenge ball missing on 41% of
graded frames; we knew that when choosing the rule, and the chosen external estimate's overall share
is 40%. No parameter was fitted to the challenge.

SkillCorner is broadcast tracking, so its missingness is heavier than a fixed tactical camera's. Each
training copy therefore draws a severity s ~ U(0, 1) that scales every process, from clean IDSSE
(s = 0) to SkillCorner broadcast (s = 1). The model sees the whole range and is tuned on none of
the challenge tracking.

## 4. Features (permutation-invariant, orientation-canonical, missing-aware)

Track slots carry no identity, so every feature is a team-level aggregate: nearest and second-nearest
player to the ball per team, counts within 2 m and 5 m, ball-to-player relative velocity (a carried
ball moves with its player), ball speed, turn and out-of-pitch distance, team centroids, spread,
length, forward velocity, running share, median player speed. Within each half the home side is
rotated to attack +x. Ball blackouts get explicit context: seconds since and until the ball is seen,
the ball and nearest-team state at either end of the gap, and which team was last and next within
1.5 m of it. Everything is also summarised over centred, past-only and future-only windows from
0.5 s to 16 s (the submission is offline, so the model may look both ways). Training copies are also
mirrored (teams swapped, pitch rotated) so the model has no home/away bias.

## 5. Model and decoding

- **Classifier:** gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`, 3 classes).
- **Decoder:** Viterbi over H, A, D with the classifier log-posteriors as emissions (optionally
  divided by the class prior, the hybrid-HMM correction) and two switching costs: one for a change of
  possession, one for entering or leaving dead ball. This is what makes a lone deflection cheaper to
  ignore than to report, matching the challenge's "controlled possession" rule.
- **Model selection:** leave-one-match-out cross-validation over the seven training matches, scored
  with the challenge's own `possession_metrics.py` (spellF1, vendored under its MIT licence).

## 6. Compliance with the organisers' rules (clarification of 3 November 2026)

- **No labels, video or ground truth of the provided game** are used to train, fit, calibrate, tune or
  select anything. The public 5-minute ground truth is not used at all; the organisers' validator
  checks the file format.
- **Rebased on 3 Nov.** Before the clarification, the degradation ranges were set from unlabelled
  statistics of the challenge tracking (ball-missing rate by player speed, close-up frequency,
  positional jitter), and a domain classifier on that tracking informed the design rule that the
  model must not see team-identity features (goalkeeper depth). Under the clarified rules we
  re-derived every degradation value from external data (section 3) and re-ran the evaluation
  (baseline B0, RESEARCH.md Y5). The no-team-identity rule stays as a design principle: track slots
  carry no identity and every training copy is mirrored, so team identity must not matter.
- **AI assistance.** Code was written with an AI coding assistant (Claude). On 30 October, before the
  clarification, raw rows of the provided tracking files and one frame of the broadcast video were
  displayed to it while inspecting the data formats. Since 3 November it has worked only on code and
  on external-data results; challenge-side checks are run and read by a team member, and for our own
  submission file it reads only the validator verdict, row count and hash.
- **Challenge data is read only for inference and automatic checks** (`kora/run.py`, `kora/checks.py`;
  `kora/profile.py` and `kora/shift.py` are diagnostics). A unit test (`tests/test_compliance.py`)
  fails if any other module imports the challenge loaders.
