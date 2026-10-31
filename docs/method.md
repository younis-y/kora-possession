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

These labels are the same kind the challenge ground truth is derived from: corrected tracking that
marks the ball dead or names the team that owns it. No event data, possession data or annotation
of the challenge match is used anywhere.

## 3. Making the training data look like the challenge tracking

The training tracking is corrected and complete; the challenge tracking is automatic. A model
trained on clean data leans on signals that are missing at test time, so each training match is
degraded at random before training (domain randomisation). The degradation was **measured on the
challenge tracking without labels**:

- **The missing ball is mostly a stopped ball.** Within bands of median player speed, the
  challenge's ball-missing rate tracks the training matches' dead-ball rate. A two-component
  mixture fit gives P(missing | dead) ~ 0.80 and P(missing | alive) ~ 0.04. Training copies hide the
  ball for most of each stoppage and only briefly in play.
- **Few tracked players means a camera close-up**, almost only at walking pace: training copies
  drop far players in some stoppages, plus random track drop-outs.
- **No positional noise:** the challenge positions are as smooth as the corrected data.

A first, naive degradation (random blackouts, added jitter) was caught by **adversarial
validation**: a classifier separated training and challenge frames on player speed alone. After the
fix, speed and ball features no longer separate the domains; what remains is team style
(goalkeeper depth), and those features were removed so the model cannot key on team identity.

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
