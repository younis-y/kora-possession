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

## Vladymyr ([research/vladymyr-decoder](https://github.com/younis-y/kora-possession/tree/research/vladymyr-decoder)) — error metrics and decoder

### [pending] V1: Are possessions merged, split, or miscounted?

**Hypothesis:** Merges and splits can cancel in total counts. Approximate matching can undercount
false predictions.

**Experiment:**

- Trace the predictions consumed by official one-to-one matching.
- Per held-out half, report true/predicted counts per 45 minutes of total half time, count error,
  mean/median duration, and dead-ball share.
- Count merges and splits from interval overlaps under a declared rule. Separate wrong-team
  fragments, false D interruptions, and complex cases.

**Outcome:** Metric implemented in `kora/errors.py`. Empirical result pending.

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

### [pending] V2: Are short possessions missing or mistimed?

**Hypothesis:** IoU 0.5 alone hides the difference between missed possessions and inaccurate boundaries.

**Experiment:** Report recall for `<1`, `1–2`, `2–4`, `4–8`, `8–15`, and `15+` seconds at IoUs
**0.3, 0.4, 0.5, 0.6, 0.7**. Match all spells before grouping by true duration. Include support,
precision, and false predicted spells by duration. Recheck the reported one-fifth recall below
one second on clean predictions.

**Outcome:** Metric implemented in `kora/errors.py`. Empirical result pending.

- Spells are matched per half at each IoU before grouping. Recall uses true duration groups.
  Precision and false predicted spells use predicted duration groups, so one pair can fall in
  different groups on each side. Counts are pooled, and ratios with a zero denominator print `n/a`.
- Checks: group edges, cross-group pairs, empty groups, totals against the official scorer,
  and extra fragments lowering precision without changing recall.
- The one-fifth recall below one second still needs clean predictions.

### [pending] V3: Which transition types fail, and by how much?

**Hypothesis:** The reported 68% of wrong frames near transitions mixes handovers, stoppages, and restarts.

**Experiment:** Separate H↔A, H/A→D, and D→H/A. Report missed/invented transitions, precision,
recall, early/late bias, and median/p90 absolute timing errors. Match exact transition types
within each half at the official tolerances. Report errors within one second of each type,
their share of all wrong frames, and overlapping windows.

**Outcome:** Metric implemented in `kora/errors.py`. Empirical result pending.

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

### [pending] V4: How often is D predicted during live play?

**Hypothesis:** The reported 2% of errors on true dead ball excludes false D during possession.

**Experiment:** Report `false D frames / true live-play frames`, split by H/A and ball visibility.
Measure the reverse error: H/A predicted on true D. Include numerators and denominators.

**Outcome:** Metric implemented in `kora/errors.py`. Empirical result pending.

- A per-half count matrix `[true H/A/D, ball visibility, raw prediction H/A/D/X]` gives
  `false D / true live` by team and visibility, and `predicted H/A / true D` by visibility.
  Each row shows numerator, denominator, and the remaining outcomes, which sum to the denominator.
- Visibility must come from the copy the predictions were made on. In the training copies it is
  simulated blackout, not challenge tracking.
- Checks: hand-computed rows, X on live frames, zero denominators, and agreement with
  `pm.confusion`.

### [pending] V5: Which decoder weights improve possession recovery?

**Hypothesis:** Switching costs, duration weights, and boundary bonuses trade missed possessions
against false fragments.

**Experiment:** Vary one group at a time, then freeze a small combined grid. Select settings
inside inner validation.

| Setting | Current HSMM | Proposed values |
| --- | ---: | --- |
| `cost_ha` | 8 | 4, 8, 16 |
| `cost_d` | 8 | 4, 8, 16 |
| `w_dur` | 2 | 0, 1, 2, 3 |
| `beta` | 4 | 0, 2, 4, 6 |

**Outcome:** —

### [pending] V6: Should transition costs and boundary bonuses depend on direction?

**Hypothesis:** Entering D differs from leaving D. H→A evidence should not reward unrelated
transitions or every new live segment.

**Experiment:** Separate live→D and D→live costs while retaining H/A symmetry. Then compare
transition-specific outputs and bonuses for H→A, A→H, live→D, and D→live. Track timing and
merge/split changes.

**Outcome:** —

### [pending] V7: Does boundary labeling suppress short possessions?

**Hypothesis:** Overlapping ±5-frame labels let `np.maximum` replace handover targets with
dead-ball targets. Wide feature windows may blur nearby transitions.

**Experiment:** Compare narrower or soft targets, separate transition outputs, event-balanced
weights, and extra weight near short spells. Measure short-spell recall and false fragments
at all five IoUs.

**Outcome:** Read-only label audit done. The target experiment has not run.

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
- Not shown: whether different targets change detector quality, short-spell recall, or spellF1.
  That needs the planned comparison on clean evaluation. Not yet audited: blurring by wide
  feature windows.

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
  named `kora.blackout.duration_matched_mask`, which is not on `main`.)
- Random-gap test: `b0_rand` and `mixed_rand` are `b0` and `mixed` with the held-out (and inner)
  evaluation copy masked the same way. Decoder selection then also sees random gaps.
- If R1/R2 change `PRODUCTION`, the same switches go on top of the new production variant.

**Decision rule (agreed with Younis):** keep `mixed` if both hold:

1. `python -m kora.compare b0_rand mixed_rand`: better on at least 5 of 7 matches.
2. `python -m kora.compare b0 mixed`: mean spellF1 drops by no more than 0.01.

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
