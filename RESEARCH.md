# Research log

**Last updated:** 2026-11-03

**Statuses**

- **[pending]** Awaiting work
- **[verified]** Confirmed
- **[deferred]** Postponed
- **[dropped]** Tried but not worth keeping
- **[inconclusive]** Unclear evidence

An outcome of **—** means the experiment has not run.
Model comparisons depend on Y1.

## Younis — evaluation and reproducibility

### [verified] Y1: Is the held-out match excluded from the whole fitted pipeline?

**Hypothesis:** Boundary-training inputs can come from first-stage models that saw the outer
held-out match, creating indirect leakage.

**Experiment:**

1. For outer match `m`, generate boundary-training predictions for match `k` without training
   the first stage on either match.
2. Repeat this isolation inside model selection. Fit priors, durations, and boundary rates on
   the training partition.
3. Include excluded match IDs, configuration versions, and seeds in cache keys.
4. Recompute `main` with full nested CV.

**Outcome (2026-11-03, `python -m kora.nested`, commit 3742788, seed 2027, features v3):** clean
spellF1 **0.7169 ± 0.0304**. Per fold, stage 1 (42 inner fits on 5 matches), detector, prior, rate,
duration prior and decoder selection use only the 6 inner matches; the 5-of-6 keep rule picks the
decoder. The old 0.7194 was slightly optimistic (−0.0025).

| Held-out | Decoder chosen | spellF1 | plain | HSMM | IoU 0.3 / 0.5 / 0.7 | true / predicted spells |
| --- | --- | ---: | ---: | ---: | --- | --- |
| J03WPY | plain | 0.679 | 0.679 | 0.716 | 0.758 / 0.686 / 0.589 | 421 / 299 |
| J03WN1 | HSMM | 0.688 | 0.684 | 0.688 | 0.792 / 0.693 / 0.563 | 500 / 371 |
| J03WMX | HSMM | 0.690 | 0.667 | 0.690 | 0.770 / 0.707 / 0.580 | 511 / 369 |
| J03WOH | HSMM | 0.767 | 0.747 | 0.767 | 0.844 / 0.770 / 0.672 | 402 / 330 |
| J03WQQ | HSMM | 0.728 | 0.722 | 0.728 | 0.816 / 0.747 / 0.600 | 401 / 349 |
| J03WOY | HSMM | 0.719 | 0.692 | 0.719 | 0.809 / 0.730 / 0.616 | 456 / 385 |
| J03WR9 | HSMM | 0.747 | 0.737 | 0.747 | 0.826 / 0.761 / 0.640 | 370 / 342 |

- HSMM beats plain on all 7 held-out matches (mean 0.7221 vs 0.7040). The keep rule rejected it on
  the J03WPY fold, which costs 0.005 on the mean; with 6 inner matches the 5-of-6 rule is strict.
- Every fold predicts 8-29% fewer possessions than the truth: the pipeline merges possessions.
  This bears on the 150-per-45-min count on the challenge match (see V1 for merge/split counts).
- Settings chosen per fold are stable: beta 4 everywhere, cost_ha 4-8, cost_d 8-16, w_dur 2-4.

**Artifacts:** `models/nested.json` (settings and scores per fold, committed);
`nested_predictions.npz`, schema-1 bundle for `kora.errors --predictions`, on the
[nested-v1 release](https://github.com/younis-y/kora-possession/releases/tag/nested-v1)
(`gh release download nested-v1 -D models`). Reuse: the 42 inner stage-1 folds are cached as
`data/cache/oof_v3_{k}_x{m}.npy`, so a decoder-only rerun takes about 15 minutes.

### [pending] Y2: How much does repeated selection overfit the seven matches?

**Hypothesis:** Reusing outer results for development can bias selection after pipeline leakage
is fixed.

**Experiment:** Freeze experiment batches before inspecting outer scores. Explore within inner
folds. Schedule full evaluations for selected candidates. Save predictions and all seven paired
score changes, including mean, median, range, and positive-match count. Match seeds and conditions.

**Outcome:** —

Five wins out of seven is a merge rule, not a significance test. The organisers' full-match noise
band is not a CV threshold. See [selection bias](https://www.jmlr.org/papers/v11/cawley10a.html).

### [verified] Y3: Is saved-model inference reproducible on Linux?

**Hypothesis:** Platform differences do not change the generated submission.

**Experiment:** Regenerate the CSV on Linux, compare its committed SHA-256, and validate its format.

**Outcome:** Exact hash matched. Official validation passed. Full retraining remains unverified.

### [pending] Y4: Does a fresh environment support the full rebuild?

**Hypothesis:** Fresh installation and retraining may expose failures absent from saved-model inference.

**Experiment:** Run the full build in fresh Linux and macOS environments. Check `shasum`
availability. Use `hashlib` if portability requires it. Record environment and artifact hashes.

**Outcome:** —

### [pending] Y5: Compliance rebase (organiser rules, 3 Nov 2026), baseline B0

**Why:** the organisers' clarification of 3 Nov requires every trained, fitted, calibrated or tuned
parameter to come from external data. Several degradation values and one feature-removal decision
were measured on the unlabelled challenge tracking (old `degrade.py` docstring, `shift.py`), so the
Y1 score of 0.7169 is not a compliant baseline.

**Change:** `kora/missingness.py` measures every degradation value on external data only
(SkillCorner open data, 20 matches, for real tracking gaps; IDSSE for positional smoothness) and
writes `models/degrade_external.json`.

**Measurement rule (owner chose every frame on 2026-11-03; replaced the same day by the pre-registered
check below):** a frame with no SkillCorner possession owner stands in for a stoppage. Estimates of
ball loss seen before any choice:

| estimate | P(ball missing \| stoppage) | P(ball missing \| live) | overall share |
| --- | ---: | ---: | ---: |
| camera-on frames, speed-band mixture with IDSSE dead rate (first draft) | 0.14 | 0.024 | |
| every frame of both periods, cuts and replays included (`degrade_external_all.json`) | 0.79 | 0.077 | 0.40 |
| **frames with a detected player (`degrade_external.json`, in use)** | **0.23** | **0.015** | **0.057** |

The organisers' documentation states the challenge ball is missing on 41% of graded frames. We knew
this when the owner first chose every frame (overall share 40%), which is why the choice was handed to
the external-only check. In-use values: close-up probability 0.026 per long stoppage, player visible
16.8 s mean, positional noise 0 (SkillCorner is smoother than IDSSE). `kora/degrade.py` reads only that file. Each training copy draws a
severity s ~ U(0, 1) that scales every process from clean IDSSE (0) to SkillCorner broadcast (1).
The duration-matched stress test on José's branch still samples challenge gap lengths; it must move to `gap_s_quantiles` / `missing_share` in the external JSON before it is used again. Caches are keyed by a hash of
the feature and degradation code plus the external parameters (`kora/variant.py`).

**Which estimate: external-only rule (pre-registered 2026-11-03, before running; raised by Vladymyr).**
Nested CV cannot choose between the estimates: each variant is graded on evaluation copies degraded its
own way. `python -m kora.missingness --check` degrades the seven IDSSE matches at full severity with each
candidate (`degrade_external.json` every frame, `degrade_external_camera.json` frames with a detected
player, both with the no-owner stoppage proxy), builds the missingness features (`b_seen`, `t_since`,
`t_until`, `nH`, `nA` and every `b_seen_*` window, one frame per second) and trains a classifier to tell
them from real SkillCorner tracking. Seven folds each hold out one IDSSE match and every 7th SkillCorner
match. **Lower mean AUC wins; a difference under 0.01 keeps the every-frame rule.** If the camera-on
candidate wins, R0 is rerun in full with it.

**Check result (2026-11-03, commit b32a9d1):** camera on wins. Mean AUC **0.874** (folds 0.846 0.894
0.892 0.875 0.848 0.880 0.881) vs every frame **0.960** (0.957 0.980 0.958 0.960 0.940 0.959 0.963),
lower on all 7 folds. Camera on is now `models/degrade_external.json`; every frame is kept as
`degrade_external_all.json`. Both still separate easily from real tracking (AUC well above 0.5).

**Fix after the check (bug, not a choice):** the camera-on file had positional noise 0.32 m because
dropping cut frames joined positions across each cut, and the joins were scored as jitter. Without the
cross-join terms the variance equals the full stream's (0.00021), so noise is now measured before any
join and is 0, as in the every-frame file. The check's features do not use positions, so the decision
is unchanged. No other value moved.

**Inner HSMM rule (corrected 2026-11-03, raised by Vladymyr).** The change to 4 of 6 was motivated by
the J03WPY outer result in Y1, which is selection on an outer score. The reference stays at **5 of 6**;
4 of 6 is the exploratory variant `b0_w4`. Grids and the rule now come from `kora.variant.DECODER`.

**Run R0 (2026-11-03, 13:58-16:34, commit 1cd8d1e code, features f971ad3ea2e):** ran with 4 of 6 before the
correction, so it is the `b0_w4` result: clean spellF1 **0.6889 ± 0.0224**, per match 0.710 0.661 0.669
0.676 0.674 0.714 0.718 (HSMM on all 7 folds). It used the every-frame degradation, so after the check
it is a historical result, not comparable with B0. B0 (5 of 6, camera-on degradation) is a full rerun,
and R1/R2 follow on the same degradation.

**Outcome (2026-11-03, 20:18-22:14, commit e46b803, features fb1066a3834):** B0 = clean spellF1
**0.7147 ± 0.0234** (plain 0.7046), per match 0.693 0.695 0.679 0.738 0.736 0.722 0.741
(J03WPY, J03WN1, J03WMX, J03WOH, J03WQQ, J03WOY, J03WR9). This is the compliant baseline; release `nested-b0` holds the json and
predictions. It is not comparable with Y1 (0.7169) or `b0_w4` (0.6889): each was graded on evaluation
copies degraded differently.

### [verified] Y6: Do touch and handover features recover short possessions? (R1, R2)

**Hypothesis:** oracle headroom on Y1 predictions (`notes/2026-11-03-headroom.py`): inserting missed
possessions under 2 s is worth +0.129, far more than any other error type. Those possessions start
with a touch, which the v3 features see only through ±4-frame averages.

**Candidates (pre-registered 2026-11-03, frozen before any outer score is seen):**

- R1 `touch`: B0 + `kora/touch.py` (ball velocity change over ±2 frames, per-team touch strength,
  touch events, last/next touch team, touch counts in 1-4 s windows, same-team touch chain, 5- and
  9-frame windows). Touch threshold 2.7 m/s = 95th percentile of free-flight velocity change in
  IDSSE live play, label-free.
- R2 `touch_bal`: R1 + possession-balanced sample weights, λ = 0.5 (each possession weighs the
  same, as in spellF1).

**Decision rule:** each candidate against B0 with `python -m kora.compare b0 <variant>`: keep if
mean spellF1 is higher and at least 5 of 7 matches improve. R2 is compared with B0 and with R1;
it is kept only if it also passes against R1. Secondary, reported but not deciding: recall under
2 s, count ratio, spellF1 per IoU.

**Outcome (2026-11-04, commit e46b803 code, run from a checkout without PR #7):** both kept;
`PRODUCTION` is now `touch_bal`.

| variant | clean spellF1 | per match (J03WPY, J03WN1, J03WMX, J03WOH, J03WQQ, J03WOY, J03WR9) | vs B0 | vs R1 |
| --- | ---: | --- | --- | --- |
| B0 | 0.7147 ± 0.0234 | 0.693 0.695 0.679 0.738 0.736 0.722 0.741 | | |
| R1 `touch` | 0.7432 ± 0.0325 | 0.742 0.743 0.696 0.786 0.739 0.708 0.789 | +0.0284, 6/7 | |
| R2 `touch_bal` | 0.7528 ± 0.0279 | 0.752 0.744 0.703 0.775 0.744 0.751 0.801 | +0.0381, 7/7 | +0.0097, 6/7 |

Secondary: recall at IoU 0.5 for true possessions under 1 s 0.208 (B0), 0.271 (R1), 0.268 (R2);
1-2 s 0.439, 0.514, 0.538; count ratio 0.834, 0.858, 0.870; spellF1 at IoU 0.7 0.611, 0.644, 0.662.
Following J6's registration, `mixed`, `b0_rand` and `mixed_rand` now sit on top of `touch_bal`, and
J6 rule 2 reads `python -m kora.compare touch_bal mixed`. The committed submission was rebuilt with
`touch_bal` on 2026-11-05 (commit 7007d78). The all-match CV kept the plain decoder there, because the
detector improved fewer than 5 of 7 matches.

### [pending] Y7: Do a stage-2 refiner, a wider grid, dwell features or stronger balancing recover short possessions? (Y7a, Y7c, Y8a, Y8b)

**Hypothesis:** on R2's held-out predictions, missed possessions under 2 s are still the largest
error (oracle headroom +0.11). In 46% of them stage 1 has the right team on most frames but the decoder
smooths them away. In the other 54% stage 1 is undecided (mean p(true team) 0.49-0.52). Team swaps are
6% of misses (V1, PR #11).

**Candidates (pre-registered 2026-11-05, before any outer score of these variants).** Each is
`touch_bal` plus one change, in `kora/variant.py`:

- `y7a_refine` (decoder smoothing; the MS-TCN idea, Abu Farha & Gall 2019). A second gradient-boosted
  model reads the detector's context windows (`boundary.context`: stage-1 probabilities over
  0.5-32 s, plus raw signals) and outputs H/A/D probabilities. These replace stage 1's in both decoders.
  It is fitted on exactly the detector's folds (`nested.fit_refiner`), so it inherits the detector's
  small leak (J1). The detector itself still reads stage 1. Stage 1 is reused.
- `y7c_wide`: V5's 180-setting HSMM grid on production instead of B0. R2's sharper
  probabilities may choose lower switching costs. Stage 1 is reused.
- `y8a_dwell` (undecided stage 1; Link & Hoernig 2017). Nine columns from `kora/dwell.py`, derived
  from existing features:
  - seconds the close side has held, backward and forward;
  - changes of the close side in the past and next 1 s and 2 s;
  - the far team's distance to the ball, its 0.5 s mean and its ±0.5 s change.
  It uses the 1.5 m radius that `features.py` already applies. Nothing is tuned on labels. Stage 1 is refitted.
- `y8b_bal1`: possession-balanced weights at λ = 1.0 instead of 0.5. Stage 1 is refitted.

**Decision rule:**

1. Each candidate is run with `python -m kora.nested --variant <v>` and compared with
   `python -m kora.compare touch_bal <v>`. Keep it if the mean spellF1 is higher and at least 5 of 7
   matches improve strictly.
2. Every kept candidate goes into one combined variant (Y9) on top of the production variant at
   that time. V7 or J6 may change production first. Y9 is kept by the same rule against production.
3. Four candidates against one baseline can pass by luck. So Y9 and production are both rerun with
   stage-1 seed 2028, and Y9's mean must still be higher there.
   (Amended 2026-11-05, before any Y7/Y8 outer score: the seed switch is `"seed": 2028` in a variant.
   It changes the degraded copies and stage 1 together, for example `mixed_s2028`.)

Secondary, reported but not deciding: recall at IoU 0.5 under 1 s and 1-2 s, the count ratio, and
`kora.errors` causes on the bundles.

**Deployment:** Y7a and Y8a are wired into `kora.run` and `kora.train` only if they pass. Y7c and
Y8b are configuration only.

**Outcome:** —

## Vladymyr — error metrics and decoder

### [verified] V1: Are possessions merged, split, or miscounted?

**Hypothesis:** Merges and splits can cancel in total counts. Approximate matching can undercount
false predictions.

**Experiment:**

- Trace the predictions consumed by official one-to-one matching.
- Per held-out half, report true/predicted counts per 45 minutes of total half time, count error,
  mean/median duration, and dead-ball share.
- Count merges and splits from interval overlaps under a declared rule. Separate wrong-team
  fragments, false D interruptions, and complex cases.

**Outcome:** Metric implemented in `kora/errors.py`. Result on B0 below.

- Input contract: `analyse(y_true, y_pred, half, ball_seen, fps=25)` on frame-aligned codes
  (0 H, 1 A, 2 D, 3 X). Each half is one contiguous block, and all spell and transition work
  stays inside a half. Temporal metrics fill predicted X as `pm.evaluate` does. Frame metrics
  count X as wrong.
- `python -m kora.errors --predictions BUNDLE.npz` reads a local schema-v1 bundle of decoded
  labels and never decodes, trains, or rebuilds features. A bundle is clean only if its stated
  provenance is accurate. The loader checks schema and alignment, not fold exclusion.
- The no-argument path needs `models/oof.npz` and exits with a clear message when it is absent.
  Its output is labelled development OOF because the decoder settings were chosen on all
  matches. The `oof-v3` release supplies these development predictions. They do not answer V1–V4.
- Exact greedy pairs at each IoU replace the approximate `used` set. Matched counts equal the
  official scorer's counts.
- Counts per 45 minutes use total half time, D included. Signed count error is predicted minus true.
- Associations: same-team spells linked when the intersection covers at least half of the
  shorter spell. Graph components are 1:1, split, merge, complex, or isolated. They are not
  official matches. Splits are typed by the predicted gap labels (wrong-team fragment, false
  D interruption, mixed). Merges are typed by the true gap labels (missed opposite possession,
  missed stoppage, mixed). Discarded weak overlaps are reported. These are same-team structural
  diagnostics, not exhaustive counts of erased possessions. Truth H then A predicted all H gives
  one association and one isolated true A, not a merge.
- Checks: `tests/test_errors.py`, synthetic sequences only, compared with the vendored official
  metrics. They cover perfect predictions, tied IoUs, collisions, each split and merge type,
  many-to-many, weak overlaps, erased opposite possessions, adjacent halves, X handling,
  normalisation, and bundle validation.

**Result on B0 predictions** (`nested-b0` release, SHA-256 `9efc7fbe…`, which reproduces spellF1
0.7147 ± 0.0234). B0 is the compliant baseline (Y5). These are descriptive diagnostics on the seven
external matches. They replace the provisional Y1 figures recorded earlier.

- 3,061 true and 2,538 predicted possessions, 17% too few (206 against 171 per 45 minutes).
- At IoU 0.5, 2,014 are matched, 1,047 true spells are missed, and 524 predicted spells are
  unmatched.
- Under the declared association rule, 211 merge components cover 448 true spells, all with
  opposite-team gaps, versus 47 split components, 46 with wrong-team gaps and 1 with a false D gap.
  Another 502 true spells and 166 predicted spells have no retained same-team overlap.
- The shortfall of 523 decomposes exactly under the association rule: merges contribute 237 (448
  true spells in 211 predictions), isolated true spells exceed isolated predicted spells by 336 (502
  against 166), and splits offset the shortfall by 50 (47 true spells in 97 predictions). The 1:1
  and complex components each have equal true and predicted counts.

**Production check on `touch_bal`** (`nested-touch-bal` release, SHA-256 `9a9d20f2…`, spellF1
0.7528). Descriptive, same seven external matches.

- 3,061 true and 2,647 predicted possessions, 13.5% too few. The shortfall of 414 decomposes under the
  association rule into merges 209 (395 true spells in 186 predictions), isolated true over isolated
  predicted 253 (417 against 164), splits −49 (44 true spells in 93 predictions), and complex
  components +1 (13 against 12).
- At IoU 0.5, 2,170 are matched, 891 true spells are missed, and 477 predicted spells are unmatched.
  Of the 891 missed, 426 (48%) are covered on at least half their frames by opposite-team
  predictions, 411 (46%) by same-team predictions despite failing IoU matching, 51 (6%) have a
  wrong-team prediction with IoU ≥ 0.5, and
  3 are mostly predicted D. These ordered categories come from the cause split in PR #11 and do not
  establish causes.
- Recall at IoU 0.5: `<1` s 0.268, `1–2` s 0.538, `15+` s 0.982. At 25 frames, H↔A recall 0.618 and
  precision 0.749, stoppages 0.725 and 0.728, restarts 0.967 and 0.973. Of 195 stoppages missed under
  exact pairs, 190 have a predicted stoppage within 1 s with the wrong preceding team.
- False D during live play: 505 of 553,105 frames. H/A predicted on true D: 388 of 449,539 frames.

### [verified] V2: Are short possessions missing or mistimed?

**Hypothesis:** IoU 0.5 alone hides the difference between missed possessions and inaccurate boundaries.

**Experiment:** Report recall for `<1`, `1–2`, `2–4`, `4–8`, `8–15`, and `15+` seconds at IoUs
**0.3, 0.4, 0.5, 0.6, 0.7**. Match all spells before grouping by true duration. Include support,
precision, and false predicted spells by duration. Recheck the reported one-fifth recall below
one second on clean predictions.

**Outcome:** Metric implemented in `kora/errors.py`. Result on B0 below.

- Spells are matched per half at each IoU before grouping. Recall uses true duration groups.
  Precision and false predicted spells use predicted duration groups, so one pair can fall in
  different groups on each side. Counts are pooled, and ratios with a zero denominator print `n/a`.
- Checks: group edges, cross-group pairs, empty groups, totals against the official scorer,
  and extra fragments lowering precision without changing recall.

**Result on B0 predictions** (`nested-b0` release, SHA-256 `9efc7fbe…`, which reproduces spellF1
0.7147 ± 0.0234). B0 is the compliant baseline (Y5). These are descriptive diagnostics on the seven
external matches. They replace the provisional Y1 figures recorded earlier.

- Recall at IoU 0.5 by true duration: `<1` s 0.208 (384 spells), `1–2` s 0.439 (535), `2–4` s
  0.642 (702), `4–8` s 0.769 (620), `8–15` s 0.909 (438), `15+` s 0.976 (382). About one fifth of
  possessions under one second are recovered.
- Precision at IoU 0.5 by predicted duration: `<1` s 0.524 (166 spells), `1–2` s 0.611 (360),
  `15+` s 0.957 (419). 166 spells are predicted below one second against 384 true.

### [verified] V3: Which transition types fail, and by how much?

**Hypothesis:** The reported 68% of wrong frames near transitions mixes handovers, stoppages, and restarts.

**Experiment:** Separate H↔A, H/A→D, and D→H/A. Report missed/invented transitions, precision,
recall, early/late bias, and median/p90 absolute timing errors. Match exact transition types
within each half at the official tolerances. Report errors within one second of each type,
their share of all wrong frames, and overlapping windows.

**Outcome:** Metric implemented in `kora/errors.py`. Result on B0 below.

- Transitions are matched per half by exact (from, to) pair at 12 and 25 frames. Results are
  grouped as H↔A, H/A→D, and D→H/A, with the six pairs as a breakdown. Latency is predicted
  minus true frame (positive is late), pooled before computing median and p90.
- Both ambiguity directions are reported: true events with several feasible predictions, and
  predicted events with several feasible truths.
- The official default (`typed="to"`) is printed separately as a reference.
- Error windows are `[c−25, c+25)` around true transitions, clipped to the half and unioned
  within each category. Exclusive rows, an overlapping row, and an outside row partition all
  frames and wrong frames. Inclusive rows give the errors within one second of each category.
  They overlap, so their shares can sum above 1. The union is a subtotal.
- Checks: timing offsets at both tolerances, type confusion, transitions across adjacent
  halves, both ambiguity cases, window overlap and clipping, and pooled quantiles.

**Result on B0 predictions** (`nested-b0` release, SHA-256 `9efc7fbe…`, which reproduces spellF1
0.7147 ± 0.0234). B0 is the compliant baseline (Y5). These are descriptive diagnostics on the seven
external matches. They replace the provisional Y1 figures recorded earlier.

- At 25 frames: H↔A recall 0.549 and precision 0.706 (2,351 true, 1,829 predicted). H/A→D recall
  0.696 and precision 0.698 (710 and 708). D→H/A recall 0.967 and precision 0.971 (699 and 696).
- Among handovers matched within 25 frames, median absolute error is 2 frames, p90 is 16 frames,
  and mean signed error is −0.3 frames. These statistics describe only matched events. One true and
  13 predicted handovers had more than one feasible partner.
- Of 216 stoppages missed under exact pairs at tolerance 25 frames, 211 have a predicted stoppage
  within 1 s whose preceding team is wrong. This identifies a preceding-team mismatch near the
  stoppage.
- Wrong frames: 58.6% lie within 1 s of a handover, 11.7% within 1 s of a stoppage, and 2.1%
  within 1 s of a restart (inclusive rows). 66.7% lie in any window and 33.3% outside all windows.
- Restart timing (p90 1 frame) may be flattered by the simulated degradation, which still hides the
  ball more often in dead ball than in play.

### [verified] V4: How often is D predicted during live play?

**Hypothesis:** The reported 2% of errors on true dead ball excludes false D during possession.

**Experiment:** Report `false D frames / true live-play frames`, split by H/A and ball visibility.
Measure the reverse error: H/A predicted on true D. Include numerators and denominators.

**Outcome:** Metric implemented in `kora/errors.py`. Result on B0 below.

- A per-half count matrix `[true H/A/D, ball visibility, raw prediction H/A/D/X]` gives
  `false D / true live` by team and visibility, and `predicted H/A / true D` by visibility.
  Each row shows numerator, denominator, and the remaining outcomes, which sum to the denominator.
- Visibility must come from the copy the predictions were made on. In the training copies it is
  simulated blackout, not challenge tracking.
- Checks: hand-computed rows, X on live frames, zero denominators, and agreement with
  `pm.confusion`.

**Result on B0 predictions** (`nested-b0` release, SHA-256 `9efc7fbe…`, which reproduces spellF1
0.7147 ± 0.0234). B0 is the compliant baseline (Y5). These are descriptive diagnostics on the seven
external matches. They replace the provisional Y1 figures recorded earlier.

- False D during live play: 761 of 553,105 true live frames (0.14%). Ball seen: 687 of 550,026
  (0.12%). Ball missing: 74 of 3,079 (2.4%).
- H/A predicted on true D: 568 of 449,539 frames (0.13%).
- The ball is missing on 3.9% of frames in B0's evaluation copies, so the missing-ball rows rest on
  few live frames. Visibility here is simulated, so these rates may not transfer to the challenge
  tracking.

### [dropped] V5: Which decoder weights improve possession recovery?

**Hypothesis:** Inner selection sits on edges of the HSMM grid `cost_ha (2, 4, 8) × cost_d (8, 16) ×
w_dur (2, 3, 4) × beta (2, 3, 4)`. Across the 14 folds of Y1 and `b0_w4`, under earlier degradation
estimates, beta 4 was chosen 12 times, cost_d 16 eight times, and cost_ha 2 in three `b0_w4` folds.
A wider grid may let inner selection reach settings that recover more short possessions. Lower
cost_ha makes handovers cheaper. Higher beta raises the bonus at every live segment start where the
detector exceeds its base rate, including restarts and same-state continuations, and lowers it where
the detector is below the rate. Its effect is broader than cheaper handovers.

**Pre-registration, round 1 (2026-11-03, before any outer run of this candidate).** Only
inner-selected settings informed the design. No outer score of a V5 candidate exists.

- Candidate `v5_wide` in `kora/variant.py`: HSMM grid `cost_ha (1, 2, 4, 8) × cost_d (8, 16, 32) ×
  w_dur (2, 3, 4) × beta (2, 3, 4, 6, 8)`, 180 settings. B0's 54 settings come first in their
  existing order, so a B0 setting wins an exact tie at the maximum inner mean whenever that tie
  includes a B0 setting. The plain grid and `inner_wins` 5 are unchanged. Features and stage 1
  are unchanged, so stage-1 caches are reused and detectors are refitted.
- Base: freeze `PRODUCTION` and its full configuration before either outer run. Explicitly inherit
  that configuration, changing only `hsmm_grid`. If its plain grid, HSMM grid, or `inner_wins`
  differs from B0, amend this registration before running. Record the base name, resolved
  configuration, commit, and any uncommitted diff hash.
- Run: `python -m kora.nested --variant v5_wide`, then `python -m kora.compare <base> v5_wide`.
  Seed 2027, with the same code state and evaluation copies for both variants.
- Decision: keep if the unrounded mean spellF1 difference is positive and at least 5 of 7 matches
  improve strictly. Ties are not wins. This is a selection rule, not a significance test. One
  candidate in this round.
- Pairing check before reading the result: seven matches in the same order, the same versions and
  seed, and identical truth, half, frame, and visibility arrays in both bundles. `kora.compare`
  checks match names only.
- Secondary, reported but not deciding: recall at IoU 0.5 for `<1` s and `1–2` s, precision for
  predicted spells under 2 s from pooled counts, the predicted-to-true count ratio, spellF1 per IoU,
  plain or HSMM per fold, and chosen settings on each edge of the new grid, including settings
  rejected by the gate. Use `kora.errors --predictions` on both bundles for duration and count
  diagnostics. Use the nested JSON for per-IoU scores, decoder choices, and edge counts.
- Known limits: the best inner-mean HSMM setting is chosen before the 5-of-6 gate, so a wider grid
  can move a fold to the plain decoder. A superset grid does not guarantee improvement. Chosen
  settings on a new edge are recorded and do not trigger another candidate this round.
- Deployment: `kora.boundary` and `kora.train` keep their own grids for final fitting. A kept result
  needs a separate change before it reaches the submission.
- Cost: 7,560 inner `decode_hsmm` calls against 2,268 for B0 (3.3×). Wall time is not yet measured.

**Outcome:** Dropped under the registered rule. Run by Younis on B0 (`nested-touch-bal` release,
SHA-256 `2567c74f…`).

- Pairing check passed: the same seven matches in the same order, identical truth, half, frame, and
  visibility arrays, seed 2027, and the same feature and stage-1 versions as B0. The bundles record
  commits `e46b803` for B0 and `b15b72d` for V5 and V6. The commits also differ in decoder and
  evaluation code.
- spellF1 0.7143 against 0.7147, a mean difference of −0.0004. No match improves, one is worse
  (J03WOH −0.0029), and six are exactly tied.
- Inner selection kept the semi-Markov decoder in every fold and chose B0's own setting in six of
  seven. Predictions differ from B0 on 192 frames, all in J03WOH. Chosen settings lay on an edge of
  the new grid for cost_ha in 4 folds, cost_d in 6, w_dur in 5, and beta in none. As registered, this
  triggers no further candidate.
- The wider grid improved the selected inner mean in one fold but did not improve outer spellF1
  under the registered rule. This result applies to the tested grid on B0.

### [dropped] V6: Should transition costs and boundary bonuses depend on direction?

**Hypothesis:** The decoder uses only the detector's H↔A output. Currently, `decode_hsmm` adds its
log-ratio bonus at every live segment start, including restarts and same-state continuations. These
starts are penalised when the detector is below its base rate. Restricting the bonus to H↔A
transitions may make handovers cheaper where the detector fires without rewarding or penalising
restarts.

Separate costs for entering and leaving dead ball were dropped. Stoppages and restarts alternate
within a half. Writing their costs as `m − d` and `m + d` gives a total of
`m(N_in + N_out) + d(N_out − N_in)`. With `m`, endpoint states, and same-state continuation
penalties fixed, the directional term is constant, so it cannot change the preferred internal
boundaries. Otherwise endpoint preferences or continuation penalties can change. The plain decoder
has the same property.

**Pre-registration, round 1 (2026-11-03, before any outer run of this candidate).**

- Candidate `v6_ha_only` in `kora/variant.py`: B0's 54 HSMM settings in their existing order, each
  with `ha_bonus_only` set. The bonus then applies only to H→A and A→H transitions. Initial
  segments, restarts, stoppages, and same-state continuations get no bonus. The plain grid and
  `inner_wins` 5 are unchanged. Stage-1 caches are reused and detectors are refitted.
- Default unchanged: with `ha_bonus_only` off, `decode_hsmm` matches the previous implementation
  exactly in synthetic tests (`tests/test_v6_decoder.py`), and `kora.run` reproduces the committed
  submission hash.
- Independent of V5: frozen on B0 before any V5 result exists, and not combined with a V5 winner in
  this round.
- Base, run, decision, and pairing check: as in V5 round 1, with `v6_ha_only` in place of
  `v5_wide`. Keep if the unrounded mean spellF1 difference is positive and at least 5 of 7 matches
  improve strictly. Ties are not wins. One candidate in this round.
- Secondary, reported but not deciding: as in V5, plus handover and restart recall and precision at
  12 and 25 frames from `kora.errors --predictions`, to monitor possible indirect effects on restart
  predictions.
- Known limits: the best inner-mean HSMM setting is chosen before the 5-of-6 gate. The beta values
  were set for a bonus on every live start and may not suit a handover-only bonus. A joint grid is
  not part of this round.
- Deployment: `kora.boundary` and `kora.run` keep the default. A kept result needs a separate change
  before it reaches the submission.
- Cost: 2,268 inner `decode_hsmm` calls, the same as B0.

**Outcome:** Dropped under the registered rule. Run by Younis on B0 (`nested-touch-bal` release,
SHA-256 `433960d0…`). The pairing check passed as for V5.

- spellF1 0.7146 against 0.7147, a mean difference of −0.0001. One match improves (J03WPY +0.0013),
  one is worse (J03WR9 −0.0022), and five are exactly tied.
- Predictions differ from B0 on 51 frames across all seven matches. Restricting the bonus to H↔A
  transitions barely changes the decoded paths.

### [dropped] V7: Does boundary labeling suppress short possessions?

**Hypothesis:** Overlapping ±5-frame labels let `np.maximum` replace handover targets with
dead-ball targets. Wide feature windows may blur nearby transitions.

**Experiment:** Compare narrower or soft targets, separate transition outputs, event-balanced
weights, and extra weight near short spells. Measure short-spell recall and false fragments
at all five IoUs.

**Outcome:** Read-only label audit and round-1 target experiment completed. The round-1 result is
below.

- Scope: `boundary.target` applied to the labels of the seven IDSSE training matches. No
  features, models, or challenge data were used. The audit counts target labels only.
- Of 2,351 handovers, 41 lose some of their 11 class-1 window frames to class 2 (1.7%). None is
  fully erased. Two more are shortened only by clipping at the array end. Overwriting removes 187
  of 25,341 handover-target frames (0.7%). After the `[::2]` subsampling in training, 90 of
  12,635 are removed.
- The affected handovers are concentrated near stoppages. 36 occur within 10 frames before a
  stoppage and 5 within 10 frames after a restart. 38 of the 41 start a true possession shorter
  than 1 s.
- There are 85 unordered same-half handover pairs separated by 1–10 frames. Their windows
  overlap and would form one class-1 block without class-2 overwriting. Windows 11 frames apart
  also touch, without overlapping.
- Transitions across the half boundary are excluded, but nearby target windows are not clipped
  there. Two dead-ball windows spill into the other half, totaling nine frame incidences. No
  handover window spills in these matches.
- Synthetic cases confirm target overwriting. A handover d frames after a restart loses 11 − d
  target frames for d = 1–10. At separations of at least 11 frames, the two windows do not
  overlap. With two short possessions bracketed by stoppages, all 11 frames of the intervening
  handover target can be overwritten.
- The label audit alone does not establish an effect on detector quality, short-spell recall, or
  spellF1. The round-1 comparison below measures the pipeline change, including the recomputed rate
  and decoder selection. Blurring by wide feature windows remains unaudited.


**Pre-registration, round 1 (2026-11-05, before any outer run of this candidate).** Only the label
audit above informed the design. No outer score of a V7 candidate exists.

- Candidate `v7_ha_priority` in `kora/variant.py`: production `touch_bal` with
  `detector.ha_priority`. A frame inside any H↔A ±5-frame window gets class 1 even when a live/dead
  window also covers it. Radius, merged handover windows, and the unclipped half boundary are
  unchanged. `rate` is recomputed from the new targets. Features and stage 1 are identical to
  `touch_bal` (same feature and stage-1 versions), so stage-1 caches are reused. The 49 detector
  fits and the full decoder selection rerun.
- Base: `touch_bal`, run from the same commit and code state as the candidate. A `touch_bal` run
  made on current code for J6 serves if its commit, configuration, and versions match. Record the
  base name, resolved configuration, commit, and any uncommitted diff hash before running.
- Run: `python -m kora.nested --variant v7_ha_priority`, then
  `python -m kora.compare touch_bal v7_ha_priority`. Seed 2027.
- Decision: keep if the unrounded mean spellF1 difference is positive and at least 5 of 7 matches
  improve strictly. Ties are not wins. This is a selection rule, not a significance test. One
  candidate in this round.
- Pairing check before reading the result: seven matches in the same order, the same versions and
  seed, and identical truth, half, frame, and visibility arrays in both bundles.
- Secondary, reported but not deciding: H↔A and stoppage recall and precision at 12 and 25 frames,
  missed stoppages split by cause, recall at IoU 0.5 for `<1` s and `1–2` s, the predicted-to-true
  count ratio, `rate` for both variants, and plain or HSMM per fold.
- Known limits: the audit found overwriting in 41 of 2,351 handover windows, 0.7% of handover-target
  frames, so a small effect is expected. A changed `rate` rescales the boundary bonus and can change
  decoder selection, so a difference is not attributable to the recovered frames alone.
- Deployment: `kora.boundary` reads the same switch for final fitting. A kept result becomes
  production through `PRODUCTION` and a rebuild.

**Round 1 result (2026-11-05).** Dropped under the registered rule. Run by Younis (`nested-v7`
release, SHA-256 `48816cfb…` for the candidate and `de9dad7c…` for the `touch_bal` base).

- Pairing check passed: the same commit `7914131` with a clean tree, seed 2027, the same feature and
  stage-1 versions, and identical truth, half, frame, and visibility arrays. Only the detector switch
  differs.
- spellF1 0.7574 against 0.7528, a mean difference of +0.0046. Four matches improve (J03WOH +0.0164,
  J03WQQ +0.0107, J03WOY +0.0054, J03WMX +0.0024), one is worse (J03WPY −0.0029), and two are
  exactly tied. Four of seven fails the 5-of-7 rule.
- The two ties are J03WN1 and J03WR9, where both runs chose the plain decoder, which does not use the
  change detector. Decoder choices also moved: J03WOH went from plain to HSMM and J03WQQ from HSMM to
  plain. As registered, the difference is not attributable to the recovered target frames alone.
- `rate` rose by about 0.0002 in every fold. Predictions differ on 3,073 frames.
- Secondary diagnostics moved in the hypothesised direction but are not deciding: stoppage recall at
  25 frames 0.725 to 0.735, missed stoppages with the wrong preceding team 190 to 183, handover recall
  0.618 to 0.623, recall at IoU 0.5 for `<1` s 0.268 to 0.276 and `1–2` s 0.538 to 0.548, and missed
  possessions 891 to 878.



**Pre-registration, round 2 (2026-11-05, before any outer run of this candidate).** Proposed after
round 1's result, so it is not independent of it (Y2).

- Candidate `v7r2`: `variant.with_detector(<base>, ha_priority=True)`, a deep copy of the base
  variant with handover-priority detector targets. The base is production after Y9's complete
  registered decision, including its seed-2028 confirmation. Pin the base name and configuration in
  `kora/variant.py` before the run. Record the resolved configuration, run commit, and clean-tree
  state here before running. The candidate uses an explicit base name and does not follow later
  changes of `PRODUCTION`.
- Base run: the pinned production variant, evaluated with seed 2027 on the same clean code commit
  as the candidate. Reuse Y9's production run only if its seed, resolved base configuration, feature
  and stage-1 versions, decoder and detector settings, and evaluation copies match. Otherwise rerun
  the base.
- Decision: the standard rule, unchanged. Keep if the unrounded mean spellF1 difference is positive
  and at least 5 of 7 matches improve strictly. Ties are not wins. The inner 5-of-6 gate is
  unchanged. Round 1's ties on plain-decoder folds are a known limit and do not change the rule.
- Pairing check before reading the result, as in round 1.
- Secondary, reported but not deciding: the round-1 diagnostics, plus
  `kora.errors --predictions <v7r2> --compare <base>` (paired possession and match bootstrap).
- Known limits: the same seven matches as round 1. Round 1's positive bootstrap interval does not
  make this round confirmatory.
- Run: Friday 6 November, after Y9, by Younis.
### [pending] V8: How does production hold up when the ball is missing far more often?

**Why:** Nested evaluation copies are degraded with the camera-on estimate at a random severity,
with 3.9% missing-ball frames recorded for B0 in V4. The tech doc lists limited exposure to ball
gaps as a validation limitation. This item reports how production behaves under much heavier ball
loss.

**Pre-registration, round 1 (2026-11-05, before any run). Report only.**

- Variant `touch_bal_stress` in `kora/variant.py`: production `touch_bal`, but every evaluation copy
  is degraded at severity 1 with the every-frame SkillCorner estimate
  (`models/degrade_external_all.json`, observed external missing share 0.4038, not a guaranteed
  simulated share). Training copies and their RNG draws are unchanged. Evaluation player gaps and
  close-ups also use the full-severity every-frame profile. The detector is refitted and decoder
  settings are reselected using stressed copies. Because every evaluation copy is stressed, the detector, inner selection,
  and the held-out score all see the heavier gaps, so this measures the whole pipeline under them.
- Run: `python -m kora.nested --variant touch_bal_stress`, then `kora.errors --predictions` on its
  bundle. Context: the `touch_bal` result on normal evaluation copies (`nested-v7` release).
- Reported: spellF1 mean, per match, and per IoU, the realised missing-ball share, and the
  `kora.errors` diagnostics (counts, cause split, duration recall, transitions, false D by ball
  visibility).
- Never used to select, tune, or change production, and not an estimate of the challenge score.
  Every value comes from external data. The every-frame share is close to the organisers' published
  41% for the challenge match. That similarity is why this condition is reported, and it sets no
  parameter.

**Amendment after J6, before any V8 run (2026-11-05). Report only.** J6 passed its registered rules
and production is changing to `mixed`. This round runs only `mixed_stress`
(`mixed` plus `eval_stress`), replacing the scheduled `touch_bal_stress` run.

- Training copies and their RNG draws remain identical to `mixed`. Every evaluation copy uses
  `degrade` at severity 1 with `degrade_external_all.json`, including player gaps and close-ups. The
  detector is refitted and decoder settings are reselected using stressed copies.
- J6's random-gap test uses the same `random_ball` generator as `mixed`'s odd training copies. V8's
  gaps are label-conditioned instead, with separate live gap lengths and stoppage offsets, so this
  checks `mixed` under a heavier profile of the mechanism its other training copies already use.
- Run `python -m kora.nested --variant mixed_stress`, followed by the registered diagnostics. Use
  J6's normal-condition `mixed` result as descriptive context. The registered reporting measures and
  external parameters are unchanged. This report will not select, tune, or change production and
  does not estimate the challenge score. J6 informed this amendment, so it is not independent
  confirmation.

**Outcome:** —

## José — missing-ball robustness and extra data

### [verified] J1: Does the score depend on one blackout simulation?

**Hypothesis:** Label-conditioned gaps may reward an unrealistic association between missing ball
and dead ball.

**Experiment:** Compare clean tracking, current degradation, label-independent gaps, heavier
live-play blackouts, and player-loss/close-up patterns. Use at least three fixed seeds per
stochastic condition and identical masks for competing models. Report gap lengths and paired
per-match/per-condition scores. Fix condition weights before evaluation.

**Outcome:** Exploratory held-out tests on three German matches and seeds 11, 22, 33 show
sensitivity to simulated ball gaps. The frame classifier was trained on the other six matches. The
fixed plain-decoder settings were historically selected using all seven matches, so these are not Y1
scores and cannot select a model.

| Held-out match | Current gaps | Independent gaps | Heavier live gaps |
| --- | --- | --- | --- |
| J03WPY | 0.7085 / 0.6784 / 0.7021 | 0.6472 / 0.6249 / 0.6302 | 0.6595 / 0.6579 / 0.6656 |
| J03WN1 | 0.6789 / 0.6778 / 0.6737 | 0.6207 / 0.5993 / 0.5803 | 0.6498 / 0.6242 / 0.5969 |
| J03WMX | 0.6520 / 0.6727 / 0.6635 | 0.5772 / 0.6057 / 0.5753 | 0.5951 / 0.6213 / 0.5964 |

Each entry is spellF1 for seeds 11 / 22 / 33. Player dropout is shared across these conditions but
is label-conditioned. Independent and current conditions change both gap placement and rate. On
J03WMX, a separate rearrangement preserved the number and lengths of ball gaps within each half:
current 0.6520 / 0.6727 / 0.6635 versus rearranged 0.5289 / 0.5368 / 0.4925. The rearrangement
changes alignment with player dropouts and retains label-conditioned gap lengths. Separate random
generators for gap masks and degradation use the same seed; statistical independence between
their streams has not been established. On the unlabelled challenge tracking, the two
scored halves contain 150 missing-ball stretches, with a pooled median of 10.72 s and p90 of
33.81 s. The independent-gap simulation has a median of 1.32–1.44 s across the seven matches and
three seeds. The observed target duration distribution does not reveal whether each gap is live
or dead and is not labelled validation. Test a label-independent duration-matched scenario before
drawing conclusions from the independent-gap experiment.

A duration-matched stress test sampled the unlabelled challenge gap lengths and fixed the pooled
missing share per simulated half. In J03WPY, with seeds 11 / 22 / 33, current-gap spellF1 was
0.7085 / 0.6784 / 0.7021 and duration-matched spellF1 was 0.5411 / 0.5347 / 0.5630. The simulated
median gaps were 11.48 / 11.00 / 10.84 s. The generator independently places gaps, shortens the
last sampled gap if needed, and does not reproduce any relationship with live/dead phases or
player visibility. The classifier was trained on the other six matches, but the fixed plain-decoder
settings were historically chosen using all seven. This is one exploratory held-out match, not
Y1 or an estimate of the challenge score. Experiment code is on `soda/missing-ball-robustness` at
`c4ce758`; exact input and code hashes were recorded locally.

A follow-up refitted the six-match classifier and used the Y1-selected J03WPY plain decoder
(alpha 0, H/A cost 16, D cost 32), with settings chosen without J03WPY. Before testing changed
gaps, the baseline decoded path, labels, halves, frames and visibility matched the saved Y1
bundle exactly. This is baseline-path parity, not proof that probabilities under other masks are
identical. The `oof-v3` release probabilities also decode to that saved path; Y1 did not record
a hash proving which probability bytes it used. On seeds 11 / 22 / 33, current-gap spellF1 was
0.7133 / 0.6911 / 0.6963 and duration-matched spellF1 was 0.5520 / 0.5407 / 0.5612. This is
one previously examined outer match, not a seven-match nested estimate or challenge score. The
simulated placements ignore live/dead state, and player loss remains label-conditioned. Run code
was committed as `a3749f5`; input, artifact and source hashes were recorded locally.

One provisional J03WPY retraining run used independent ball gaps during training. Its
independent-gap scores were 0.6809 / 0.6516 / 0.6645, versus 0.6472 / 0.6249 / 0.6302 for the
current training simulation. Its current-gap scores were 0.6909 / 0.6618 / 0.6951, versus 0.7085 /
0.6784 / 0.7021. This is one match, one training seed, and a fixed historical decoder; it is not a
merge decision. Seven-match clean nested comparisons and player-loss sensitivity remain pending.
Y1 now supplies held-out predictions, matching visibility and provenance; its decoder settings
exclude the outer match. Source inspection found an inner-selection dependency: for outer match m
and inner validation match k, detector-training features for another match j can come from a
stage-1 model trained on k. Inner prior and boundary rate also include k. This may bias inner
selection scores, but does not by itself put outer match m into training. Its effect is unmeasured.

**Run J1-mixed (pre-registered 2026-11-03, before any score):** training `current` vs `mixed` (even
copies current gaps, odd copies label-independent gaps), all seven German matches held out in turn,
fixed plain decoder (alpha 0, H/A 8, D 32) for both arms. Evaluation conditions: current,
independent, live, shuffled_current; seeds 11, 22, 33; no challenge-derived condition. Primary
score per match: equal-weight mean spellF1 over the 12 condition-seed pairs. Mixed passes if the
seven-match mean is higher and at least 5 of 7 matches improve. Reported, not deciding: current
condition alone. This uses the pre-Y5 degradation; a pass makes `mixed` a candidate variant
against B0, not a merge. `--duration-matched` now reads `models/degrade_external.json`, never
the challenge match.

**Outcome (2026-11-03, commit e0c22fd):** mixed passes the pre-registered rule: primary score
**0.6480 vs 0.6337 (+0.0144), better on 7 of 7 matches** (+0.006 to +0.028). It costs a little
on the current simulation alone: 0.6993 vs 0.7031 (−0.004, 4 of 7).

| condition (mean of 7 matches × 3 seeds) | current training | mixed training |
| --- | ---: | ---: |
| current | 0.7031 | 0.6993 |
| independent | 0.6352 | 0.6589 |
| live | 0.6534 | 0.6679 |
| shuffled_current | 0.5430 | 0.5661 |

The plain decoder is fixed and its settings were chosen on all seven matches in the old pipeline,
so these are paired comparisons, not nested scores. Degradation is pre-Y5. Next: register `mixed`
as a variant on B0 (label-independent gaps on odd training copies) and score it with
`kora.nested` + `kora.compare` after R0. Logs: `data/j1-mixed/` (local).

### [pending] J6: Does mixed training beat B0 when the ball goes missing at random?

**Hypothesis:** J1's gain carries over to the external degradation (Y5): training copies whose ball
gaps ignore the labels make the model less dependent on "ball missing means stoppage".

**Candidates (pre-registered 2026-11-03, before any code or outer score):**

- `mixed`: production variant + odd training copies get random ball gaps instead of the degraded
  ball mask. Players stay degraded as in B0.
- Random gaps: `kora.degrade.random_ball`, lengths drawn from `gap_s_quantiles`, placed uniformly
  without labels in each half until `missing_share` of its frames is spent (overlaps allowed, so the
  realised share is a little lower); both values from `models/degrade_external.json` (SkillCorner,
  external). Own seed per match and copy, so the B0 draws stay identical. J1 used 2 s geometric
  gaps instead; this run uses the external profile. (Amended before any code or run: first draft
  named `kora.blackout.duration_matched_mask`, which is not on `main`.) (Amended 2026-11-04, before
  any J6 run: when Y5's check moved `degrade_external.json` to the camera-on estimate, the registered
  values (every frame: 40% missing share and its gap lengths) moved to `degrade_external_all.json`;
  `random_ball` now reads that file, so J6 runs with the values it was registered on.)
- Random-gap test: `b0_rand` and `mixed_rand` are `b0` and `mixed` with the held-out (and inner)
  evaluation copy masked the same way. Decoder selection then also sees random gaps.
- If R1/R2 change `PRODUCTION`, the same switches go on top of the new production variant.

**Decision rule (agreed with Younis):** keep `mixed` if both hold:

1. `python -m kora.compare b0_rand mixed_rand`: better on at least 5 of 7 matches.
2. `python -m kora.compare b0 mixed`: mean spellF1 drops by no more than 0.01. (After Y6 set
   `PRODUCTION = touch_bal`, this reads `compare touch_bal mixed`; see Y6.)

Secondary, reported but not deciding: means, short-possession recall, count ratio.

**Outcome:** —

### [dropped] J2: Can player movement identify possession during ball gaps?

**Hypothesis:** Player geometry and movement can provide evidence without reconstructing ball coordinates.

**Experiment:** Train a player-only H/A model without ball-derived features or carried ball
context. Compare main, auxiliary, and fused models on identical gaps: `<0.5`, `0.5–1`, `1–2`,
`2–4`, and `4+` seconds. Tune visibility/gap-dependent fusion within training folds. Report gap
errors and overall spellF1.

**Outcome:** An exploratory player-only H/A model was trained with leave-one-match-out splits and
tested using simulated gaps on all seven German matches, with seeds 11, 22, 33. Pooled over
current-simulation live frames inside gaps, it got 58,612 / 69,479 frames correct (84.4%); the
always-home baseline got 37,023 / 69,479 (53.3%). Counts repeat frames across seeds and are not
independent samples. Both figures condition on the true frame being live. Gap scenarios select
different frames; player-only predictions do not change when only the ball mask changes. Fusion,
decoded spellF1, and clean nested comparison remain pending. The saved run manifest
records a blackout evaluator hash different from this branch; the earlier source snapshot was not
saved. Exact rerun reproducibility of these results has not been established.

**Decision (2026-11-03):** dropped. On the same 69,479 live gap frames (current simulation, seven
matches, seeds 11/22/33), the main classifier gets 59,625 right (85.8%) and the player-only model
58,612 (84.4%); with mixed training the main model gets 86.2%. The main model already uses the
player signal, so a fusion has little to add. Fusion itself was not tested.

### [verified] J3: Does the domain comparison survive time- and match-separated splits?

**Hypothesis:** Random frame splitting in `kora.shift` gives correlated neighboring frames to
training and evaluation.

**Experiment:** Use separated challenge time blocks and match-aware source splits. Compare AUC
and feature importance across splits before changing degradation or removing features.

**Outcome:** On three source matches and one degradation seed, random-frame domain AUC was 1.000.
Holding out one source match and one separately built challenge half gave 0.926–0.979 across six
splits (mean 0.944). Both source and target features were built separately per half. This shows the
samples are distinguishable, not which differences matter for possession. Match style, camera
coverage, and other domain differences may contribute. The target data was unlabelled.

**Decision (2026-11-03):** closed. Random frame splits overstate the shift (1.000 vs 0.944), but the
domains stay separable under separated splits. Under the 3 Nov rules no degradation value or feature
choice may come from this comparison; Y5 re-derives degradation from external data instead.

### [dropped] J4: Does Metrica improve training despite different possession labels?

**Hypothesis:** Extra tracking can help without treating event-derived ownership as equivalent ground truth.

**Experiment:** Verify coordinates, frame rate, synchronization, and label definitions for each
[sample game](https://github.com/metrica-sports/sample-data). Compare auxiliary pretraining or
high-confidence event anchors followed by Sportec training against Sportec-only training.
Evaluate on the same held-out matches. Avoid treating full event-derived paths as hard labels
without validating the mapping.

**Outcome:** Metrica's three sample games were published before the 2024 challenge fixture. The
pinned [2021 snapshot](https://github.com/metrica-sports/sample-data/tree/e706dd506b360d69d9d123d5b8026e7294b13996)
contains synchronized Game 3 events and tracking. Of 1,395 CARRY events, 14 opposing-team pairs
overlap in their interiors, or 23 if touching endpoints count. An inclusive-endpoint audit found 75
conflicted frames. Endpoint semantics and the mapping to the challenge possession definition remain
unverified. The source invites community use and requests attribution but has no explicit standard
licence; clarify reuse terms before depending on it for released training artifacts.

A local, predeclared player-only feasibility comparison used J03WPY as the held-out match,
mask seed 11, and six German training matches. It added 14,336 mirrored and unmirrored weak
CARRY rows at weight 0.25 to 314,392 German live rows. Unknown and conflicting anchors were
excluded. For current simulated gaps on true live frames, Sportec-only was 2,802 / 3,313
correct (84.58%); Sportec plus Metrica was 2,820 / 3,313 (85.12%). With clean players and
the same ball mask, the scores were 2,853 / 3,313 (86.12%) and 2,825 / 3,313 (85.27%).
The result is mixed and does not establish an improvement. This is one match and one mask seed,
with no fusion or spellF1 measurement. Both models used the same fixed model seed, but adding
auxiliary rows changed the row-level internal early-stopping split and validation objective.
A comparison with fixed German-only validation or fixed iterations and the Y1 evaluation is
needed. No Metrica-trained model has been released.

A follow-up used a fixed whole German match (J03WR9) for both models' early-stopping
validation, five other German matches for fitting, and J03WPY as the outer test. J03WPY was
excluded from fitting and early stopping, but has been examined in earlier experiments; repeated
research decisions can overfit it.
The Metrica anchors were used only in fitting. On the same seed-11 current simulated gaps,
Sportec-only was 2,812 / 3,313 correct (84.88%); Sportec plus Metrica was 2,792 / 3,313
(84.27%). The fits stopped at 128 and 107 iterations, respectively. This removes the differing
validation-set confound in the earlier comparison but remains a single match and seed, with no
fusion or spellF1 result. No change is selected; the full Y1 comparison remains necessary.

Research tools on this branch need the pinned Game 3 files in the ignored data directory:

```bash
revision=e706dd506b360d69d9d123d5b8026e7294b13996
base="https://raw.githubusercontent.com/metrica-sports/sample-data/$revision/data/Sample_Game_3"
mkdir -p "data/external/metrica/$revision"
for file in Sample_Game_3_events.json Sample_Game_3_metadata.xml Sample_Game_3_tracking.txt; do
  curl -fL "$base/$file" -o "data/external/metrica/$revision/$file"
done
.venv/bin/python -m kora.metrica_anchors
.venv/bin/python -m kora.metrica_tracking
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m kora.metrica_eval
```

The audit and comparison verify pinned file hashes. The comparison also requires the seven
German matches cached by `python -m kora.external`. These commands are research only, not the
submission build or the clean nested evaluation. Metrica reuse terms remain unresolved.

**Decision (2026-11-03):** dropped. Two matched comparisons gave +0.54 and −0.61 points of gap-frame
team accuracy, which is no reliable gain. The sample data has no standard licence, and the rules
require rights to every dataset. SkillCorner open data (MIT, Y5) now covers external automatic
tracking. The research code stays on `soda/missing-ball-robustness` and is not merged.

### [dropped] J5: Are useful tracking measurements discarded?

**Hypothesis:** Retaining only ball x/y may omit useful measurements.

**Experiment:** Confirm unused columns, units, and missing-value rules from provider metadata.
Evaluate supported signals individually. Do not assume a column means confidence or speed.

**Outcome:** The raw player and ball records have a third `s` field that is not retained by the
model. Player metadata describes it as km/h, while the organiser guide describes player speed in
m/s. The ball field is finite in 10.5% of all frames, versus 56.9% for ball x. Its meaning and units
remain uncertain, and the external-match loader does not retain a directly equivalent player-speed
field. No new signal was added to inference.

**Decision (2026-11-03):** dropped. The IDSSE training data has no equivalent field, so a model could
learn its meaning only from the challenge match, which the 3 Nov rules forbid. Player speed is
already derived from positions.

## Shared follow-ups

### [deferred] S1: Do separate dead/live and H/A models help?

**Hypothesis:** Stoppage detection and live-play team ownership benefit from separate models.

**Experiment:** Compare the three-class classifier with a dead/live model plus an H/A model
trained on live frames. Combine probabilities and retain the decoder. Measure false D,
conditional team accuracy, transitions, and spellF1.

**Outcome:** —

### [deferred] S2: Can Smart Tagging improve predictions under a credible validation setup?

**Hypothesis:** Phase/team tags add evidence, but tag agreement alone cannot validate a gain.

**Experiment:** Build automatic phase proxies on external matches. Test small additive priors
for attacking, defending, and set-piece evidence, including sensitivity to tag errors. Check
whether proxy errors resemble the actual tagger before relying on the comparison.

**Outcome:** —

### [deferred] S3: Does a small temporal model recover ordering lost by averages?

**Hypothesis:** Event order around changes contains information missing from temporal averages.

**Experiment:** Compare a shallow temporal convolutional network over existing features with
the boosted-tree model, retaining the decoder. Start with short windows and few parameters.

**Outcome:** —

### [deferred] S4: Does confidence-filtered self-training help target adaptation?

**Hypothesis:** Confident pseudo-labels can improve adaptation or reinforce existing errors.

**Experiment:** Hide each external held-out match's labels, generate pseudo-labels, retain
confident segments, adapt, then score. Select confidence thresholds and adaptation settings
inside inner validation. Compare with no adaptation.

**Outcome:** —

### [deferred] S5: Which larger integrations warrant revisiting?

**Hypothesis:** Integration cost may outweigh gains from replacing the current pipeline.

**Experiment:** Revisit after the smaller experiments identify a specific limitation:

- [PathCRF](https://github.com/hyunsungkim-ds/pathcrf): structured player-only inference.
  Different player/event representations complicate a full port.
- [Ball Radar](https://github.com/hyunsungkim-ds/ballradar): possession inference is relevant.
  Full trajectory reconstruction adds a separate task.
- Broadcast/video processing: detection and synchronization add substantial work.
- Domain-adversarial networks: forcing similarity may remove useful football/style information.

**Outcome:** —

## Run records

Record code/evaluation versions, configuration, seeds, artifact locations, five-IoU scores,
seven paired match results, and relevant error metrics. Under Outcome, record the result and
keep/drop/inconclusive decision with its reason. Update the status and date.

Project rules and merge requirements: [CONTRIBUTING.md](CONTRIBUTING.md).
