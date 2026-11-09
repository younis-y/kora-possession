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
The duration-matched stress test on José's branch reads `gap_s_quantiles` / `missing_share` from the external JSON (`891d61d`). Caches are keyed by a hash of
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

### [dropped] Y7: Do a stage-2 refiner, a wider grid, dwell features or stronger balancing recover short possessions? (Y7a, Y7c, Y8a, Y8b)

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

**Outcome so far:** `y8a_dwell` did not meet the keep rule: mean spellF1 was 0.7492 versus
0.7528 for `touch_bal`, with 3/7 matches improving. At IoU 0.5, recall for true possessions
under 1 s rose from 103/384 (0.268) to 124/384 (0.323), while precision for predicted
possessions under 1 s fell from 0.621 (169 predicted) to 0.565 (216 predicted). For predicted
possessions of 1–2 s, precision fell from 0.670 to 0.615. The 1,000-draw paired-possession
bootstrap (seed 0) gave a 5th–95th interval of −0.0104 to +0.0035 for spellF1 difference.
The interval is descriptive and does not establish equivalence. Future work on short-possession
recall should also preserve precision.

**Outcome (2026-11-05, recorded 2026-11-07 from `~/kora-runs/compare_*.log`):** all dropped against
`touch_bal` (0.7528). `y7a_refine` 0.7415 (-0.0114), better on 0 of 7, recall under 1 s 0.268 -> 0.245,
count ratio 0.870 -> 0.846 (the refiner smooths). `y7c_wide` 0.7547 (+0.0019, median 0), better on 3 of 7.
`y8b_bal1` 0.7500 (-0.0028), better on 2 of 7, recall under 1 s 0.268 -> 0.305 but count ratio 0.902.

### [dropped] Y10: Do cuts at detector peaks recover swallowed short possessions?

**Hypothesis:** every decoder pays the same switching cost for a 0.5 s possession as for a 30 s one,
so short possessions inside the other team's spell are swallowed. 48% of R2's misses are covered
mostly by the other team (V1, PR #11), and Y7's refiner and wider grid made this worse or did not
help. ASRF (Ishikawa et al., WACV 2021, arXiv 2007.06866) recovers short segments by cutting the
decoded path where a boundary detector peaks and relabelling each piece by its frame
probabilities. On 50Salads that took F1@50 from 66.7 to 77.3 over the same model without it.

**Candidate (pre-registered 2026-11-05, before any outer score):** `y10_cuts` is production (`mixed`)
plus `kora/cuts.py`:

- Peaks are local maxima of the existing detector's team-change probability at or above θ, at least
  `gap` frames apart.
- Every H/A run of the selected decoder's path is cut at the peaks inside it.
- Each piece takes H or A by its summed stage-1 probabilities. Dead-ball runs are untouched.
- θ ∈ {0.2, 0.3, 0.4, 0.5, 0.6} and gap ∈ {6, 12, 25} frames are chosen on the six inner matches.
- Cuts are used on the outer match only if they beat no cuts on the inner mean and on at least 5 of
  6 inner matches, the same rule as the HSMM.
- Stage 1 and the detector are reused from `mixed`.
- The JSON records `spell_f1_nocut`, which must equal `mixed` on every fold. That is the pairing check.

**Decision rule:** `python -m kora.compare mixed y10_cuts`. Keep if the mean spellF1 is higher and at
least 5 of 7 matches improve strictly.

Secondary, reported but not deciding: recall and precision for possessions under 2 s, which must
both rise for a follow-up; the count ratio; and `kora.errors --compare` intervals.

**Deployment** (`kora.run` with the all-7 detector) is added only if Y10 passes.

**Outcome (2026-11-05, recorded 2026-11-07 from `~/kora-runs/compare_y10_cuts.log`):** dropped.
`kora.compare mixed y10_cuts`: 0.7511 -> 0.7549 (+0.0038, median +0.0017), better on 4 of 7 (min
-0.0028, max +0.0155); spellF1 up at every IoU 0.3-0.7; recall under 1 s 0.263 -> 0.305, count ratio
0.879 -> 0.897. The closest to passing at the time, and the reason for the ceiling diagnosis below.

### [dropped] Y11, Y12: Do mirror averaging and weak-touch peaks help on top of production?

**Candidates (pre-registered 2026-11-05, before any outer score).** Each is `mixed` plus one change:

- `y11_mirror` (stage-1 variance). Each stage-1 classifier also predicts on the mirrored evaluation copy:
  - The copy is the same degraded copy with the teams swapped, rebuilt by replaying `train._copies`.
    The run fails unless the replay equals the cached copy exactly.
  - Its H/A-swapped probabilities are averaged with the plain ones.
  - Training already uses mirrored copies, so this is the matching test-time averaging. Ensembling
    raised F1@50 by 3.9-5.7 points in C2F-TCN (arXiv 2105.10859).
  - Stage-1 keys are unchanged. The mirrored predictions are cached beside the plain ones, so the 49
    classifiers are refitted once. Seed bagging is not included: the classifier's only randomness is
    its early-stopping split.
- `y12_peaks` (short possessions; ballradar, KDD 2023; Link & Hoernig 2017). New columns from `kora/peaks.py`:
  - Weak touches are local maxima of `touch.py`'s touch strength in [0.2, 1) × 2.7 m/s, below the
    strong-touch threshold.
  - The columns are the side of and time since the last and next weak touch, and the home minus away
    weak touches in the past and next 1 s and 2 s.
  - On one IDSSE match there are 18 per minute of live play, and 12% lie within 0.5 s of a team change
    against 5.2% of frames.
  - The planned gain-of-possession rule (Vidal-Codina 2022) is not a candidate, because `touch.py`'s
    |Δv| > 2.7 m/s already covers both its speed and turn conditions.
  - Features are reused and stage 1 is refitted.

**Decision rule:** `python -m kora.compare mixed <v>`. Keep if the mean spellF1 is higher and at least
5 of 7 matches improve strictly. Winners from Y10-Y12 are combined in Y9 and confirmed with seed 2028
(Y7's rule).

Secondary: recall and precision under 2 s, the count ratio, and `kora.errors --compare` intervals.

**Outcome (2026-11-06):** both dropped, so Y9 has nothing to combine and the V7 round 2 base stays `mixed`.
`y11_mirror`: mean 0.7511 -> 0.7553 (+0.0042), better on 4 of 7, recall under 1 s 0.263 -> 0.294.
`y12_peaks`: 0.7511 -> 0.7497 (-0.0014), better on 3 of 7. Seed check: `mixed_s2028` scored 0.7334;
its evaluation copies are degraded with seed 2028 too, so this is not a paired comparison (see Y16).

### [dropped] Y13-Y15: Can a touch-event model find the possessions the frame model swallows?

**Diagnosis (2026-11-05, `python -m kora.ceiling mixed mixed_rand`, held-out predictions of production).**
spellF1 if one error type were fixed perfectly:

| Fixed perfectly | `mixed` | `mixed_rand` (heavy gaps) |
|---|---|---|
| nothing | 0.7511 | 0.6607 |
| swallowed spells (true H/A spell covered on < half its frames) | 0.9183 | 0.8881 |
| spells under 2 s | 0.8608 | 0.8046 |
| edges within 0.5 s of a predicted change | 0.7663 | 0.6827 |

Inside the 470 swallowed spells (ball visible 99% of the time) stage 1 prefers the true team on at
least half the frames in 16%; the true team reaches P >= 0.5 in 56%, but the other team does so in
59% of long spells production holds correctly. So the frame classifier's signal for these spells is
no stronger than its noise, which is why decoder changes (V5-V7, Y7, Y10) moved the score by < 0.005,
and timing is worth at most +0.015.

**Y13 gate (pre-registered 2026-11-05, before it ran).** `kora/events.py`: candidates are local maxima
of the ball's velocity change `b_acc` (+-2 frames, >= 0.2 x 2.7 m/s); each gets the feature row at
the candidate, the change of 15 kinematic columns 3 and 6 frames either side, and the gap to and
strength of the neighbouring candidates; label = an H<->A handover into H / A within 5 frames. A
gradient-boosted model per held-out match m is trained on the other six matches' training copies
(ballradar, KDD 2023; ELASTIC, arXiv 2508.09238). `python -m kora.events` fits it and m's outer
change detector exactly as `kora.nested` does, then counts swallowed spells of production with a
peak within 5 frames of both edges, at the detector's number of false peaks for θ = 0.3, 0.4, 0.5.
**Passes** if the event model cuts out more swallowed spells at all three levels. If it fails,
Y14 is not run.

**Y13 gate result (2026-11-05):** PASSED. Of 470 swallowed spells, peaks within 5 frames of both
edges at the detector's false-peak counts for θ = 0.3 / 0.4 / 0.5: detector 62 / 47 / 32, event model
80 / 69 / 57 (event θ 0.10 / 0.15 / 0.21). The event model names the right team at 0.79 of the
swallowed starts it scores. `models/gate_y13_mixed.json`.

**Y14 (amended after the gate, before any outer score):** `y14_events` = `mixed` plus closed event
pairs: a peak handing the ball to the other team, followed within `max_len` frames inside the same
decoded run by a peak handing it back, relabels the frames between to the opening event's team.
θ ∈ {0.1, 0.15, 0.2, 0.3, 0.4} and max_len ∈ {50, 100, 200} are chosen on the six inner matches (event
models trained without m and k) and used on m only under the HSMM rule (inner mean up, >= 5 of 6).
Two changes from the registration, both before any score: pieces take the event's team, not
`kora/cuts.py`'s summed stage-1 probabilities, because the diagnosis shows stage 1 prefers the true
team in only 16% of these spells; and the event columns are not fed to stage 1 here, since clean
columns need a stage-1 refit per excluded set. `spell_f1_noevent` must equal `mixed` on every fold.
**Y15:** ball gaps under 1 s with both ends seen are
filled by straight lines, with a flag column, before features. Both are decided by
`kora.compare mixed <v>` (mean up and >= 5 of 7 matches better) and checked on heavy-gap copies.

**Outcome (Y14, 2026-11-05):** dropped. `kora.compare mixed y14_events`: mean 0.7511 -> 0.7521
(+0.0010), better on 1 of 7 (4 ties, 1 loss of -0.0039); recall under 1 s 0.263 -> 0.273, count ratio
0.879 -> 0.891. Events beat no events on the inner mean in all 7 folds (+0.004 to +0.010) but passed
the 5-of-6 inner rule in only 2, so 5 folds fell back to production. The event model is better than
the detector but at usable false-peak rates still cuts out only 12-17% of swallowed spells; its
features are the frame columns at the kink.

**Outcome (Y15, 2026-11-06):** dropped. `kora.compare mixed y15_fill`: 0.7511 -> 0.7576 (+0.0065), better
on 4 of 7, recall under 1 s 0.263 -> 0.286. On heavy gaps, `kora.compare mixed_rand y15_fill_rand`:
0.6607 -> 0.6633 (+0.0026), better on 4 of 7, recall under 1 s 0.167 -> 0.211. José's V8 diagnostics
agree: only 31 of the 346 short possessions `mixed_stress` misses have any missing-ball frame.

### [verified] Y16: Does averaging stage 1 over three seeds help?

**Why (2026-11-06).** `mixed_s2028` (Y11 outcome) scored 0.7334 against 0.7511, but the seed there also
drives the evaluation copies, so it mixes test-set and model variance. Here the seeds are split: the
evaluation copies stay those of `mixed` (and `mixed_rand`), and only training changes.

**Candidate (pre-registered 2026-11-06, before any member is fitted).** `y16_seeds` = `mixed` with
stage 1 averaged, equal weights, over three members. Member s degrades its training copies with seed s
(`train._copies`) and seeds its classifier with s, for s = 2027, 2028, 2029, fixed now; no other seeds
or weights will be tried. Member 2027 is production itself (same cache). The classifier's own
randomness is only its early-stopping split (Y11), so most of the members' diversity comes from the
degraded copies, which makes this bagging over simulated gaps. The average replaces stage 1 in both the
outer and the inner predictions; the detector, prior, decoder grids and selection rules are rerun on it
unchanged (`kora/nested.py`, `member_stage1`). A pair's classifier predicts on both its matches, so
each new member needs 28 fits, not 49, and every fit also predicts the heavy-gap evaluation copies.

**Decision rule:** `kora.compare mixed y16_seeds` (mean up and at least 5 of 7 matches better). It
ships only if `kora.compare mixed_rand y16_seeds_rand` does not lose more than 0.01 on the mean (V8).
Secondary: recall and precision under 2 s, `kora.errors --compare`, and the seed-aware interval
(#18) once merged. If kept, `kora.train`/`kora.run` fit the three members for the submission.

**Outcome (2026-11-06, 6fe6389, clean checkout):** kept, the first candidate to pass since J6.
`kora.compare mixed y16_seeds`: 0.7511 -> 0.7566 (+0.0055, median +0.0089), better on 5 of 7 (min
-0.0090, max +0.0164); spellF1 rises at every IoU from 0.3 to 0.7; recall under 1 s 0.263 -> 0.320, 1-2 s
0.536 -> 0.557; count ratio 0.879 -> 0.898. Heavy gaps, `kora.compare mixed_rand y16_seeds_rand`:
0.6607 -> 0.6783 (+0.0176), better on 6 of 7; recall under 1 s 0.167 -> 0.195. Both well inside the V8
condition. Next: fit the three members in `kora.train`/`kora.run` for the submission.

### [dropped] Y18: Does the short ball-gap fill help on top of the seed ensemble?

**Why (2026-11-07).** Y15's fill raised the mean (+0.0065, heavy gaps +0.0026) but was better on only
4 of 7 matches. Single-seed stage 1 is noisy enough to flip a match by ~0.01 (Y16's members), so a
small real gain can fail the 5-of-7 rule; averaging three members lowers that noise. The fill also
targets the challenge match's 41% missing ball, which Y16 does not. Y11's mirror averaging, the other
near-passer, is left out: it is the same kind of variance reduction the members already give.

**Candidate (pre-registered 2026-11-07, before any filled member is fitted).** `y18_fill` = `y16_seeds`
with `kora/fill.py` (unchanged since Y15) applied to every copy of every member: gaps under 25 frames
inside a half, both ends seen, filled linearly, plus `b_filled`. Seeds 2027 / 2028 / 2029 and equal
weights as in Y16; member 2027 is `y15_fill`'s own stage-1 cache. Nothing else changes and nothing
else will be tried.

**Decision rule:** `kora.compare y16_seeds y18_fill` (mean up and at least 5 of 7 matches better). It
ships only if `kora.compare y16_seeds_rand y18_fill_rand` does not lose more than 0.01 on the mean (V8).
Secondary: recall under 1 s and 1-2 s, count ratio. If kept, `kora.train`, `kora.ensemble` and
`kora.run` apply the fill for the submission.

**Outcome (2026-11-07, cdc2ae0, clean checkout):** dropped. `kora.compare y16_seeds y18_fill`: 0.7566 ->
0.7582 (+0.0016, median -0.0005), better on 3 of 7 (min -0.0075, max +0.0139); recall under 1 s 0.320 ->
0.326, 1-2 s 0.557 -> 0.548. Heavy gaps, `kora.compare y16_seeds_rand y18_fill_rand`: 0.6783 -> 0.6782,
better on 3 of 7, though recall under 1 s rises 0.195 -> 0.221 and the count ratio 0.819 -> 0.850. With
the members averaged, Y15's +0.0065 shrinks to +0.0016: most of it was the seed noise Y16 removes.

### [dropped] Y19: Do SkillCorner's possession events give usable H/A/D labels?

**Why (2026-11-07).** The seven IDSSE matches are both the training set and the only test set, and one
seed swing (~0.018, Y11) is larger than most candidate deltas, so the keep rule has been deciding on
noise (Y10, Y11, Y15, Y18). SkillCorner's open data (MIT, used since Y5 for missingness) also ships
dynamic events for its 20 A-League matches: about 960 timed player possessions per match, with start
and end types and game-interruption flags. If labels derived from them fit the possession the challenge
scores, the 20 matches give 27 test matches, and real rather than simulated tracking gaps for training.
Risks: the events are SkillCorner's own detection on broadcast tracking, possession runs touch to
touch, dead time has to be inferred, and the source is 10 fps.

**Labels (fixed 2026-11-07, before any statistic of them was computed).** `kora/skillcorner.py`
`events_label`: each player possession [frame_start, frame_end] belongs to its team. The gap to the next
possession stays with the team that had the ball last, unless a game interruption is flagged after the
first or before the second; then it is dead, except its first `OUT_S` = 0.5 s (the ball still travelling,
last team) and its last `RESTART_S` = 1.0 s (the restart in the air: the receiving team, or the other
team when the start type is an `*_interception`). A live gap longer than `FLIGHT_S` = 4 s is unknown
(X, not scored), as are the frames before a period's first and after its last possession. Frames are
aligned by the `match_periods` start frames and checked against the tracking rows per period. The
constants are football timing (pass flight, restart), not fitted.

**Amended 2026-11-07 13:20, after the first gate run stopped at its alignment check, before any
verdict.** 1899585's tracking ends at frame 60529, one frame before its `match_periods` end frame 60530,
so the per-period row check raised. Rows are now placed by their own tracking frame number (`frames`,
read once from the JSONL); for the other two gate matches the mapping is unchanged. That run had already
printed the G1-G3 statistics of 1874553 and 1886347. No label rule or threshold changed.

**Gate (pre-registered 2026-11-07, `python -m kora.scgate`, writes `models/gate_y19.json`).** It reads
only `GATE` = 1874553, 1886347, 1899585 (the first three in `kora.skillcorner` order); the other 17 are
not scored until it passes.
- G1, labels against IDSSE (no model): dead share of labelled frames within 0.30-0.60; H/A spells per
  45 minutes and median spell length each within 0.67-1.5 times the IDSSE mean; unknown at most 0.15 of
  frames.
- G2, alignment: on frames both call H or A, the event label agrees with SkillCorner's per-frame
  `possession.group` on at least 0.90.
- G3, the shipped model (`y16_seeds`, de873a7 models, its plain decoder, trained on IDSSE only): mean
  spellF1 at least 0.45 on the three, and team accuracy at least 0.80 on frames both call H or A.
- Pass = G1, G2 and G3. Reported, not gated (G4): member 2027 alone (`mixed`) against `touch_bal`
  fitted on all seven, same decoder. Heavy simulated gaps ranked mixed far above (J6: +0.088).

**If it passes**, each step pre-registered on its own: (1) the 17 others become extra test matches:
each candidate fitted on all seven IDSSE matches, decoded with its IDSSE-chosen decoder, scored on the
20, paired, beside the 7 nested deltas; (2) then SkillCorner as training data (real gaps), scored on the
nested IDSSE folds and on SkillCorner held out by match. If it fails, the reason is recorded and
SkillCorner stays missingness-only.

**Outcome (2026-11-07, dc8cf28, clean checkout, `models/gate_y19.json`):** fails G1 and G2, so
SkillCorner stays missingness-only and neither follow-up runs. The event labels are coarser than
IDSSE's possession: 136 H/A spells per 45 minutes against 206 (ratio 0.66, bound 0.67) and a median
spell of 5.7 s against 3.8 s (1.52, bound 1.5); the dead share (0.50 against 0.45) and the unknown
share (0.06) are within bounds. They agree with SkillCorner's own per-frame possession group on 0.856
of the frames where both name a team (bound 0.90). G3 passes: the shipped model scores 0.503 with team
accuracy 0.916, but predicts 1,339 spells against 848 labelled (1.58 times; 0.90 on IDSSE). So most of
the shortfall is the labels missing short possessions that IDSSE marks, which is the error we are trying
to fix; as a test set they would reward swallowing them. G4 (reported only): `y16_seeds` 0.503, `mixed`
0.452, `touch_bal` 0.458. Mixed is not ahead of touch_bal on these real gaps, and the ensemble's lead
comes partly from predicting fewer spells (1,339 against 1,491), which these labels reward.

### [verified] Y20: Does another stage-1 classifier, or more capacity, help?

**Why (2026-11-07).** Stage 1 still uses the day-one HistGradientBoosting settings (250 trees, learning
rate 0.1, 63 leaves, at least 200 rows per leaf), and every fit runs all 250 iterations, so its early
stopping never fires: it may be capacity-limited. A timing check (no scores) on one six-match fold: HGB
60 s, LightGBM with the same settings 28 s, train-set argmax agreement 0.988. A faster stage 1 also
halves the cost of every later run.

**Candidates (pre-registered 2026-11-07, before any candidate is fitted).** Each alone on top of
`mixed`, through `kora/engine.py` (LightGBM 4.7.0, deterministic, column-wise); nothing else will be
tried in this round.
- `y20_lgbm`: LightGBM with the same settings (250 trees, 0.1, 63 leaves, min 200, L2 1.0).
- `y20_long`: 1000 trees at 0.05.
- `y20_deep`: 500 trees at 0.05, 255 leaves, min 100.
- `y20_bag`: 500 trees at 0.05, half the features and 80% of the rows per tree (the seed now matters).

Each `_rand` twin is the same fits scored on the heavy-gap evaluation copies; `kora.nested` caches both
from one run (the training copies are the same), so the V8 check costs no fits.

**Decision rule.**
1. Screen, `kora.nested --variant <c> --screen` (the seven outer fits, reused by step 2): kora.train's
   plain-decoder pick on the outer folds, against `mixed`'s cached folds scored the same way. A
   candidate goes on if the mean delta is at least +0.002 (`y20_lgbm`: at least -0.003).
2. Stage A, full nested: `kora.compare mixed <c>` keep rule (mean up, at least 5 of 7), and
   `kora.compare mixed_rand <c>_rand` not lower by more than 0.01 (V8).
3. Stage B, the second-seed check: the Stage A winner (largest mean delta if several) as a three-member
   ensemble exactly as Y16, `y21` = winner + seeds 2027 / 2028 / 2029: `kora.compare y16_seeds y21` keep
   rule, and `y16_seeds_rand y21_rand` not lower by more than 0.01. Promoted only if Stage A and Stage B
   both pass; the build then fits the same classifier.
4. `y20_lgbm` with no gain: if it passes the screen and Stage A is within -0.003 on the mean, LightGBM
   becomes the engine for later experiments (twice as fast), not production by itself.

**Amended 2026-11-07 14:42, for compute, before `y20_deep`'s screen result.** Under the current load a
`y20_deep` fit takes about 14 minutes, so its Stage A (42 more fits) would take about 7 hours; its screen
was stopped after 4 of 7 fits, before any score, and it gets no Stage A in this round. `y20_lgbm`'s
Stage A is not run either: no later experiment in this round needs a faster tree engine (Y22 and Y23
reuse the cached trees). Stage A runs for `y20_long` only (`y20_bag` failed its screen).

**Outcome so far (2026-11-07, 63ea21d, clean checkout).** Screens against `mixed`'s outer folds (0.7516):
`y20_lgbm` 0.7488 (-0.0028, 2/7: inside its -0.003 bar), `y20_bag` 0.7521 (+0.0004, 3/7: out),
`y20_long` 0.7555 (+0.0038, 5/7); `y20_deep` stopped before any score (amendment). Stage A for
`y20_long`: `kora.compare mixed y20_long` 0.7511 -> 0.7576 (+0.0065, median +0.0050), better on 5 of 7
(min -0.0052, max +0.0213); recall under 1 s 0.263 -> 0.307; heavy gaps `mixed_rand` -> `y20_long_rand`
0.6607 -> 0.6670 (+0.0063, 5/7). So it passes Stage A; Stage B (`y21_long`) is running.

**Stage B (2026-11-07, 8ffce8f, clean checkout).** `kora.compare y16_seeds y21_long` 0.7566 -> 0.7595
(+0.0029, median +0.0061), better on 5 of 7 (min -0.0173, max +0.0150); heavy gaps `y16_seeds_rand` ->
`y21_long_rand` 0.6783 -> 0.6818 (+0.0035, 4/7), inside V8. Y20 passes; `y21_long` is a Y24 parent.

### [dropped] Y22: Does a temporal network over the frame features help as a stage-1 member? (S3)

**Why (2026-11-07).** Swallowed short possessions are worth +0.17 (ceiling diagnosis, Y13), and inside
them the trees' evidence is no stronger than their noise; decoder and feature changes since have moved
the score by less than 0.007. The trees see each frame through window averages, while a short
possession is an ordered pattern (a touch by one team between touches by the other). S3 proposed a
shallow temporal convolutional network over the existing features; Y19 closed the SkillCorner route,
so it runs now. A timing check (no scores): 250 steps on one six-match fold take 6 s on the laptop's
GPU (MPS), a prediction of one evaluation copy 0.3 s.

**Model (fixed 2026-11-07, before any fit is scored), `kora/seq.py`.** Input: the 204 frame columns,
standardised on the training copies, NaN set to 0, clipped to +-10. A 1x1 convolution to 64 channels,
8 residual blocks (GroupNorm, dilated non-causal convolution with kernel 3 and dilation 2^i, GELU,
dropout 0.1, 1x1 convolution), receptive field +-255 frames (about 10 s), then a 1x1 convolution to H/A/D
logits. Trained on the variant's own training copies (those of `mixed`, seed 2027: degraded, randomly
gapped and mirrored as for the trees) with the trees' balance weights (lambda 0.5): random 1024-frame
windows inside one half, batch 16, 3000 steps, AdamW (one-cycle learning rate to 2e-3, weight decay
0.01). Each half of an evaluation copy is predicted as one sequence. torch 2.15.0, a research
dependency (not in `requirements.txt` unless promoted; its test is skipped without it).

**Amended 2026-11-07 13:28, after one network fit and before any score.** The first queue was stopped
because GroupNorm normalised over the whole window, so a 1024-frame training window and a whole half at
prediction would see different statistics. Now each frame is normalised over its channels (LayerNorm),
the loss counts only frames whose whole +-255-frame receptive field lies inside the window, and windows
are 2048 frames. Nothing else changed.

**Candidates (pre-registered 2026-11-07).** `y16_seeds` with stage 1 = (1 - w) x the three tree members'
average + w x the network: w = 0.25 (`y22_quarter`, the network as a fourth equal member), 0.5
(`y22_half`), 1.0 (`y22_only`). The detector, prior and decoder selection are rerun on the blend
unchanged. One set of 28 network fits (7 singles and 21 pairs, each pair predicting both its matches, as
Y16's members) serves all three weights and both evaluation sets. Nothing else is tried in this round.

**Decision rule.**
1. Screen, `kora.nested --variant y22_<w> --screen` (the 7 single fits): kora.train's plain-decoder pick
   on the seven outer folds, the blend against the same run's tree-only outer folds (that is,
   `y16_seeds`). A candidate goes on if the mean delta is at least +0.002.
2. Stage A, full nested: `kora.compare y16_seeds y22_<w>` keep rule (mean up, at least 5 of 7), and
   `kora.compare y16_seeds_rand y22_<w>_rand` not lower by more than 0.01 (V8).
3. Stage B, the second seed: the Stage A winner (largest mean delta if several) refitted with network
   seed 2028 (`y22_<w>_s2028`) must pass the same two checks against the same bases. Promoted only if
   Stage A and Stage B both pass; the build then fits the network on all seven matches and `kora.run`
   blends it.

**Outcome so far (2026-11-07, f4844d5, clean checkout).** Screens against the tree-only folds (0.7575):
`y22_quarter` 0.7642 (+0.0068, 7/7), `y22_half` 0.7632 (+0.0057, 5/7), `y22_only` 0.7375 (-0.0199, 0/7).
The network alone is worse than the trees, but blended in it helps: its errors differ. Stage A:
`kora.compare y16_seeds y22_quarter` 0.7566 -> 0.7673 (+0.0107, median +0.0122), better on 7 of 7 (min
+0.0016, max +0.0186); spellF1 up at every IoU 0.3-0.7 (0.833 -> 0.842 at 0.3, 0.668 -> 0.677 at 0.7);
recall under 1 s 0.320 -> 0.349, 1-2 s 0.557 -> 0.579; count ratio 0.898 -> 0.908. Heavy gaps,
`y16_seeds_rand` -> `y22_quarter_rand`: 0.6783 -> 0.6836 (+0.0053, 4/7), inside V8. `y22_half` 0.7653
(+0.0087, 5/7); heavy gaps 0.6773 (-0.0011, 3/7), inside V8. So `y22_quarter` is the Stage A winner.

**Stage B (2026-11-07 15:34): fails, so Y22 is not promoted.** With network seed 2028 the same blend,
`y22_quarter_s2028`, scores 0.7577 against `y16_seeds` 0.7566 (+0.0011, median +0.0021), better on only 4
of 7 (min -0.0076, max +0.0088); recall under 1 s 0.294. Against the seed-2027 run it is lower on all 7
(-0.0096). Most of Stage A's +0.0107 was the luck of one network's seed: a single network on six matches
is very seed-dependent. Y23's seed averages test whether a seed-robust gain remains (against
`y16_seeds`, as its rule says when Stage B fails).

### [verified] Y23: Does averaging the temporal network over seeds, or a larger one, add to Y22?

**Why (2026-11-07).** Alone the network is weaker than the trees (Y22 screen 0.7375 against 0.7575) and,
like any single fit, noisy; Y16 showed that averaging seeds helps the trees. A larger receptive field
(+-20 s instead of +-10 s) may also let it see more of a possession.

**Candidates (pre-registered 2026-11-07 14:45, after Y22's Stage A and before its Stage B result).**
Each is `y22_quarter` (network weight 0.25) with one change; nothing else is tried in this round.
- `y23_seeds2`: the network averaged over seeds 2027 and 2028 (both fitted by Y22, no new fits).
- `y23_seeds3`: seeds 2027, 2028 and 2029.
- `y23_big`: one larger network, 96 channels and 9 blocks (receptive field +-511 frames), 3072-frame
  windows, batch 12, seed 2027; otherwise as Y22.

**Decision rule.** Against `y22_quarter`: `kora.compare` keep rule (mean up, at least 5 of 7), and the
`_rand` twin not lower than `y22_quarter_rand` by more than 0.01 (V8). `y23_big` also needs its second
seed, `y23_big_s2028`, to pass the same checks. The winner is the passer with the largest mean delta. If
Y22's Stage B fails, the same rules are applied against `y16_seeds` / `y16_seeds_rand` instead.

**Amended 2026-11-07 16:13, after `y23_seeds2`'s result and before any other Y23 result: one more,
stricter check.** Y22 showed one network seed can carry the whole gain, and both seed averages include
seed 2027. So the winning seed average must also pass the same rule with a disjoint set of network
seeds of the same size, starting at 2030 (`y23_seeds2r`: 2030, 2031; `y23_seeds3r`: 2030-2032). This can
only block a promotion. `y23_seeds2` so far: 0.7566 -> 0.7646 (+0.0080, 6/7), heavy gaps 0.6783 ->
0.6859 (+0.0076, 5/7).

**Amended 2026-11-07 18:28, after `y23_big`'s normal-copy result (0.7669, +0.0103, 7/7) and before its
heavy-gap and second-seed results: big cannot ship as a single network.** If `y23_big` and
`y23_big_s2028` both pass, the candidate becomes their average, `y23_big2` (seeds 2027 and 2028, no new
fits), which must pass the same rule; the winner is then the largest mean delta among the passing
`y23_seeds2`, `y23_seeds3` and `y23_big2`. `y23_big2` needs no disjoint replicate (`y23_big_s2028` is
its independent seed). Also recorded: `y23_seeds3` 0.7641 (+0.0075, 7/7, min +0.0026), heavy gaps 0.6808
(+0.0025, 5/7): passes.

**Outcome (2026-11-07, clean checkouts at 4f839b4 and c757ffb).** Y22's Stage B failed, so all are judged
against `y16_seeds` (0.7566; heavy gaps 0.6783). `y23_seeds2` +0.0080, 6/7 (heavy gaps +0.0076, 5/7);
`y23_seeds3` +0.0075, 7/7 (+0.0025, 5/7); `y23_big` 0.7669 +0.0103, 7/7 (+0.0021, 5/7); `y23_big_s2028`
0.7635 +0.0069, 6/7 (heavy gaps -0.0007, 3/7, inside V8). Both big seeds pass, so `y23_big2` was run:
0.7566 -> 0.7665 (+0.0099, median +0.0087), better on all 7 (min +0.0019, max +0.0201); heavy gaps
0.6783 -> 0.6822 (+0.0039, 4/7), inside V8. **Winner: `y23_big2`** (largest mean delta among the passers;
no disjoint replicate, as amended 18:28). Unlike Y22's small network, the larger one holds across seeds.

### [dropped] Y24: Do the Y20 trees and the Y23 network add up?

**Why (2026-11-07).** Y20 (`y20_long`, more capacity in the trees) and Y23 (a temporal network blended
into stage 1) change different parts of stage 1, so if both pass, their gains may add.

**Candidate (pre-registered 2026-11-07 19:35, before Y21's result and before Y23's winner is known).**
Only if `y21_long` passes Y20's Stage B and Y23 has a winner: `y24` = `y21_long`'s three tree members
with Y23's winning network(s) blended exactly as in Y23 (weight 0.25). No new fits: both parents'
caches are reused (the network's cache key leaves out the tree classifier, which does not change its
training copies).

**Decision rule.** Against the better parent (the larger mean delta against `y16_seeds`): keep rule
(mean up, at least 5 of 7) and the `_rand` twin not lower by more than 0.01 (V8). If it passes, `y24` is
production; if not, the better parent is. If only one parent passes, that parent is production.

**Status (2026-11-07 23:59).** Both parents pass (`y23_big2` +0.0099, `y21_long` +0.0029; the better parent
is `y23_big2`). Not run: the team stopped new model runs at 23:59. Until it runs, the rule's fallback holds:
`y23_big2` is the production candidate. **Shipped 2026-11-08** (production `y23_big2`, clean-checkout build of
95bb677): validator valid, 199,320 lines, sha256 117c95d0...; the build kept the semi-Markov decoder
(0.7675 vs plain 0.7640 over the seven matches) and the tree members are byte-identical to Y16's.

**Run (2026-11-08, owner: "run some more models").** `y24` defined as registered and run with Y25 below.

### [dropped] Y25: Does a third network seed, or more network weight, add to y23_big2?

**Why (2026-11-08).** Y23 showed seed averaging steadies the network (small network: one seed +0.0107,
the other +0.0011), and Y22 picked weight 0.25 for the small network only; the larger one is stronger
alone, so it may deserve more weight.

**Candidates (pre-registered 2026-11-08 13:30, before any run).** Each is `y23_big2` with one change:
- `y25_big3`: the larger network averaged over seeds 2027, 2028 and 2029 (28 new fits for 2029).
- `y25_half`: network weight 0.5 instead of 0.25 (no new fits).
`y24` (Y24) runs in the same round.

**Decision rule.** Against `y23_big2`: keep rule (mean up, at least 5 of 7) and the `_rand` twin not
lower than `y23_big2_rand` by more than 0.01 (V8). Winner = largest mean delta among `y24`, `y25_big3`,
`y25_half`. Second-seed check: if `y25_half` wins, `y25_half_r` (weight 0.5, network seeds 2028 and 2029)
must pass the same checks; `y25_big3` is itself a seed average and `y24`'s parents each passed alone.
Promoted only then; deploying `y24` would also have needed the LightGBM+torch workaround below.

**LightGBM + torch on macOS (found 7 Nov).** Once LightGBM has loaded its OpenMP runtime, torch's CPU
convolutions in the same process can hang at 0% CPU (two OpenMP runtimes). MPS (GPU) fits are not
affected, and the shipped build never loads LightGBM. A build that needs both should run the network's
CPU inference in a subprocess (as `tests/test_y19_y20.py` does) or with `torch.set_num_threads(1)`.

**Outcome so far (2026-11-08, 81bcce9, clean checkout).** `y24` 0.7634 against `y23_big2` 0.7665
(-0.0031, median -0.0044, better on 3 of 7): fails; its heavy-gap twin was not finished (battery).
`y25_half` 0.7630 (-0.0034, 2 of 7), heavy gaps 0.6822 -> 0.6848 (+0.0025, 4/7): fails. `y25_big3` 0.7671
(+0.0006, median -0.0011, 3/7), heavy gaps 0.6798 (-0.0024, 2/7): fails. No winner; production stays `y23_big2`. Bundles for `y16_seeds` and `y23_big2` (+ `_rand`) are in release `nested-y23`.

### [dropped] Y26: Is the decoder we ship tuned for the wrong amount of missing ball?

**Outcome (2026-11-09):** no. What ships, scored under heavy gaps (`y23_big2_xn`), is **0.6860**,
above the 0.6822 that re-chooses the decoder on heavy-gap copies (2 of 7 matches better for the
re-chosen one). No candidate is kept and production stays `y23_big2`. All four pairing checks passed
(xn and heavysel choices equal `y23_big2` and `y23_big2_rand`; cuts' no-cut scores equal the base on
both conditions). Clean tree `ffc1b81`, `~/kora-runs/y26_pass.py`.

| Candidate | Heavy gaps vs `y23_big2_rand` (0.6822) | Normal vs `y23_big2` (0.7665) | Rule |
|---|---|---|---|
| `y26_heavysel` | 0.6822 vs xn 0.6860 (−0.0038, 2/7) | 0.7657 (−0.0008, 3/7) | fail |
| `y26_viscost` | 0.6817 (−0.0006, 1/7) | 0.7666 (+0.0002, 1/7) | fail |
| `y26_cuts` | 0.6815 (−0.0008, 0/7) | 0.7672 (+0.0007, 2/7) | fail |
| `y26_visw` | 0.6831 (+0.0009, 5/7) | 0.7642 (−0.0022, 1/7) | pass, then fails the second seed: `y23_big_s2028_rand` 0.6776 → 0.6739 (−0.0037, 4/7) |

Per match (J03WMX, WN1, WOH, WOY, WPY, WQQ, WR9), `y23_big2_xn` 0.653 0.688 0.709 0.675 0.687 0.697
0.693 against `y23_big2_rand` 0.648 0.676 0.708 0.667 0.688 0.692 0.698. Choosing on heavy-gap inner
folds raises cost_d but does not score better on held-out heavy-gap matches: with seven matches the
heavy-gap choice is noisier than the normal one, and the phantom-stoppage effect in the motivation was
not large enough to show. The honest heavy-gap figure for what ships is 0.6860, not 0.6822;
`docs/method.md` should cite it. No candidate's heavy-gap score reaches 0.6860, so reading the rule
against xn instead of `_rand` gives the same answer.


**Why (2026-11-08).** The challenge match has the ball missing on 41% of graded frames (CHALLENGE.md);
the normal evaluation copies have 3.9% (V4) and the heavy-gap copies (`_rand`, the external every-frame
profile) about 40%. Production's detector, prior and decoder settings are chosen on normal copies
(`kora.ensemble.pick_decoder`, `kora.boundary`), and that is what `kora.run` applies to the challenge. The
heavy-gap figures so far (0.6822 for `y23_big2_rand`) re-choose all three on heavy-gap copies, so they
score a pipeline we do not ship. The choices differ in every fold (`models/nested_y23_big2*.json`): plain
cost_d 2-16 on normal copies against 16-128 on heavy-gap copies, HSMM cost_d 8 against 8-16. Missing
ball reads as a stoppage to stage 1, and a decoder tuned without gaps lets those phantom stoppages split
possessions. The Y13 ceiling also shows stage-1 signal, not decoding, limits the normal copies; under
heavy gaps the decoder has more to gain.

**Measurement (report only).** `y23_big2_xn`: `y23_big2_rand`'s heavy-gap evaluation copies, with the
detector, prior and decoder settings chosen on the normal inner folds ("cross" in `kora.nested`), i.e.
what ships, scored under heavy gaps. Pairing check: its per-fold choices and inner means equal
`y23_big2`'s.

**Candidates (pre-registered 2026-11-08, before any run).** All reuse `y23_big2`'s cached stage 1 (no fits
other than the detectors), each scored on both conditions:
- `y26_heavysel`: detector, prior and decoder chosen on the heavy-gap inner folds and scored on normal
  copies. Its heavy-gap side is `y23_big2_rand` itself. Pairing check: its per-fold choices and inner
  means equal `y23_big2_rand`'s.
- `y26_viscost`: plain grid times a factor in {1, 2, 4, 8} on cost_d for switches on frames without the
  ball (`kora/visdecode.py`; factor 1 is the plain decoder exactly; `decode.py` unchanged).
- `y26_cuts`: Y10's cuts (`kora/cuts.py`, same theta/gap grid and inner rule) on `y23_big2`. Pairing
  check: `spell_f1_nocut` equals `y23_big2` on every fold.
- `y26_visw`: network weight 0.5 on frames without the ball, 0.25 elsewhere (fixed, no grid).

**Decision rule (heavy-gap primary, proposed to Vladymyr and José; J6's robustness form).**
- `y26_heavysel` is kept if `kora.compare y23_big2_xn y23_big2_rand` has the mean up and at least 5 of 7
  matches better, and `kora.compare y23_big2 y26_heavysel` loses no more than 0.01.
- Each other candidate is kept if `kora.compare y23_big2_rand <v>_rand` has the mean up and at least 5 of 7
  better, and `kora.compare y23_big2 <v>` loses no more than 0.01.
- Second-seed check for each kept candidate: the same change on `y23_big_s2028` (the larger network,
  seed 2028 only, cached) must also raise the heavy-gap mean: `kora.compare y26_heavysel_r_xn
  y23_big_s2028_rand` for heavysel, `kora.compare y23_big_s2028_rand y26_<v>_r_rand` for the others.
- If the team keeps the normal copies primary instead, the standard keep rule plus V8 decides, and
  both readings are recorded.

Kept candidates are combined (Y28) and deployed only after Vladymyr signs off the decoder change.
Deploying `y26_heavysel` is a configuration change (amended before any run, 2026-11-08): the build
variant sets `eval_rand`, so `kora.train`, `kora.ensemble` and `kora.boundary` make their leave-one-out
choices on the heavy-gap evaluation copies (`train._copies`). Training copies, and so every fitted
model, are unchanged; nothing in `decode.py` or `boundary.py` changes.

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
- Base pin, final (2026-11-06, before any outer run of `v7r2`): Younis reports that Y11 (0.7553, 4/7)
  and Y12 (0.7497, 3/7) were dropped. Their JSON has not been checked here. With Y7a, Y7c, Y8a, Y8b,
  Y10 and Y14 already out, Y9 has nothing to combine, its seed-2028 confirmation does not apply, and
  the base is `mixed`. Pin: `VARIANTS["v7r2"] = with_detector("mixed", ha_priority=True)` (PR #17),
  resolving to `{"touch": True, "balance": 0.5, "mixed": True, "detector": {"ha_priority": True}}`,
  with the same feature and stage-1 versions as `mixed`. Both `mixed` and `v7r2` run with seed 2027
  on commit `2f8b46f` (PR #17's head), because the J6 `mixed` run used an earlier commit. Before
  either starts, record Younis's confirmation here that the whole working tree, including `models/`,
  is clean. The run's `dirty` flag checks only `kora/`. Confirmation: pending.

**Round 2 status (2026-11-08): deferred, not run.** No round-2 run was recorded before the team's run
stop on 7 November (23:59), and no clean-tree confirmation was recorded. Production has since moved from
`mixed` to `y16_seeds` and then `y23_big2` (shipped 8 November, semi-Markov decoder with the detector),
so a pass on `mixed` would not apply to either. The `mixed` pin is retired (removed from PR #17). Round 1's
dropped result stands; any retry needs a new registration against the production of that time.
### [verified] V8: How does production hold up when the ball is missing far more often?

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

**Outcome (2026-11-06, `mixed_stress`, commit e7019e9, seed 2027):** report only.
At run start the tracked code was clean, but the whole tree was not: earlier result JSON files
`nested_touch_bal_stress.json` and `nested_y8a_dwell.json` were untracked in `models/`.
The metadata `dirty: false` checks only `kora/` and does not establish a clean whole tree.
Mean spellF1 was 0.6301 ± 0.0402 across seven matches. Per match, in J03WPY, J03WN1,
J03WMX, J03WOH, J03WQQ, J03WOY, J03WR9 order: 0.6450 / 0.6430 / 0.5508 / 0.6876 /
0.6061 / 0.6213 / 0.6567. Mean spellF1 at IoU 0.3–0.7 was 0.7449 / 0.6986 / 0.6372 /
0.5730 / 0.4967. The model reported 2,626 possessions versus 3,061 true (0.858 ratio).

The realized ball-missing share was 328,976 / 1,002,644 frames (32.81%). At IoU 0.5,
recall was 38/384 (0.099) for true possessions under 1 s and 207/535 (0.387) for 1–2 s.
The association diagnostic reported 228 merge components and 99 split components; 226 of
those merges involved a missed opposite-team possession. At a 25-frame tolerance, exact H↔A
handover recall was 1,125/2,351 (0.479). False D on true live frames was 745/518,257 (0.0014)
when the ball was seen and 1,740/34,848 (0.0499) when missing. Of 346 missed true
possessions under 1 s, 31 had any missing-ball frames and 28 had a missing-ball majority.
Those misses contained 404 missing frames out of 4,621 total (8.7%). These counts describe
the simulated evaluation copies, not the challenge match, and do not establish a cause.

Normal-condition `mixed` scored 0.7511 as descriptive context. This is not a controlled
estimate of the isolated cost of missing balls: the evaluation profile changes player gaps and
close-ups as well, and the detector is refitted and decoder settings reselected on stressed
copies. The gaps remain label-conditioned and are not an estimate of the challenge score.
The summary is `models/nested_mixed_stress.json`; regenerating V1–V4 diagnostics requires its
separate prediction bundle. No production decision follows from V8.

**Check and remaining diagnostics (2026-11-06, Vladymyr).** The release `nested-v8` bundle
`nested_mixed_stress_predictions.npz` has SHA-256 `8ebd5a9348adfd6ac159b265f7f30ba93399ae963fd6ce07bb71a3e210d9151e`
(GitHub records no digest for it). `kora.errors --predictions` on it (PR #18 code, `2da0ac9`)
reproduces every diagnostic count above. The JSON matches the mean, per-match and per-IoU scores and
the possession counts. The two untracked JSONs are outputs; `kora.nested` at `e7019e9` reads no
result JSON, so they could not affect the run. The registered diagnostics not listed above:

- Missed true possessions at IoU 0.5: 1,256. Absorbed by the other team 573 (45.6%), same team with
  boundaries off 590 (47.0%), team swap 74, dead ball instead 19. Under 1 s: 206 of 346 absorbed by the
  other team, 124 boundaries off, 13 dead ball, 3 team swap.
- Invented predicted possessions: 821. Inside the other team's possession 317, same team with
  boundaries off 356, team swap 74, on dead ball 64, mixed 10.
- Missed live→D transitions at 25 frames: 295. A predicted stoppage lay within 25 frames but the
  preceding team was wrong for 238 (80.7%); no predicted stoppage for 57.
- Ball visibility in true possessions of 1–2 s: missed 328 with 1,044 of 11,897 frames missing
  (8.8%), found 207 with 259 of 7,964 (3.3%). Under 1 s, found possessions had no missing frames.

In these stressed copies, 206 of 346 missed true possessions under 1 s were classified as absorbed by
the other team; 31 had any missing-ball frames. These descriptive counts do not establish why the
possessions were missed. Report only.

### [verified] V9: Where does `y23_big2` gain over `y16_seeds`?

**Why (2026-11-08).** This item reports `kora.errors --compare` and the V1–V4 diagnostics for the shipped
`y23_big2` against `y16_seeds`. Report only: it selects nothing and changes no rule.

**Inputs.** Release `nested-y23`: `nested_y16_seeds`, `nested_y16_seeds_rand`, `nested_y23_big2` and
`nested_y23_big2_rand` prediction bundles. All four match the SHA-256s in the release notes, and their
means reproduce the JSONs (0.7566, 0.6783, 0.7665, 0.6822). Both pairs pass the pairing check (match
order, labels, half, frame, ball visibility). `kora.errors` as of `bd098f9`, 1,000 draws, seed 0.

**Normal copies** (`y16_seeds` → `y23_big2`): +0.0099, better on 7 of 7. Paired possession bootstrap
+0.0051 to +0.0149 (5th–95th), match bootstrap +0.0066 to +0.0134. Most of the gain is precision:
invented possessions at IoU 0.5 fall from 519 to 475 and predicted possessions from 2,732 to 2,710
(3,061 true). Missed possessions fall from 848 to 826, mostly at 1 s and longer (1–2 s: 237 → 229;
2–4 s: 197 → 187); "absorbed by the other team" falls from 399 to 382. Under 1 s, recall at IoU 0.5 is
almost unchanged: 123 → 124 of 384.

**Heavy-gap copies** (`y16_seeds_rand` → `y23_big2_rand`): +0.0039, better on 4 of 7 (−0.0155 to
+0.0239). Paired possession bootstrap −0.0023 to +0.0103, with 85.1% of draws above zero; match
bootstrap −0.0039 to +0.0118. Recall under 1 s at IoU 0.5 rises from 75 to 87 of 384 and for 1–2 s
from 210 to 217 of 535, but invented possessions rise from 590 to 625. Under these random gaps, the
missed possessions under 1 s have 43.0% of their frames without the ball (1,659 of 3,855) against 0.9%
for the found ones (14 of 1,491); on normal copies the shares are 0.5% and 1.4%.

On normal copies `y23_big2` has fewer split components (62 → 54) and more merge components (169 → 180).
Its higher score accompanies fewer invented possessions and fewer missed possessions lasting at least
1 s; recovery under 1 s barely changes. These describe seven matches already used for
development; the intervals are descriptive, not a significance test.

### [deferred] V10: How does PathCRF compare with our models on matched players-only inputs?

**Outcome (2026-11-08):** considered, not run. A matched, report-only trial (S5's PathCRF) was drafted but
not registered. PathCRF is trained on event-derived possession edges, so it would need its supervision
adapted to our team-possession labels, and the paper reports about 565 s per epoch at 5 fps on an RTX
4090. On 8 November the team agreed to skip it because seven-fold training was judged infeasible
before the deadline.

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

### [verified] J6: Does mixed training beat B0 when the ball goes missing at random?

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

**Outcome (2026-11-05, `python -m kora.compare`, seed 2027, all four runs clean; `touch_bal` and
`mixed_rand` at `7914131`, `mixed` at `222f776`, `b0_rand` at `ad0319e`):** both rules hold.

| comparison | means | per match (J03WPY, J03WN1, J03WMX, J03WOH, J03WQQ, J03WOY, J03WR9) | wins |
| --- | --- | --- | --- |
| 1. `b0_rand` → `mixed_rand` | 0.5726 → 0.6607 (+0.0881) | +0.070 +0.075 +0.084 +0.106 +0.096 +0.075 +0.111 | 7/7 |
| 2. `touch_bal` → `mixed` | 0.7528 → 0.7511 (−0.0017) | −0.002 +0.016 +0.003 +0.014 −0.003 −0.022 −0.019 | 3/7 |

Secondary: count ratio 0.695 → 0.826 under random gaps and 0.870 → 0.879 on the normal copies;
spellF1 at IoU 0.7 0.466 → 0.539 and 0.662 → 0.657. At IoU 0.5, recall for true possessions
under 1 s was 81/384 → 64/384 under random gaps and 103/384 → 101/384 on normal copies. Recall
for 1–2 s was 182/535 → 207/535 under random gaps and 288/535 → 287/535 on normal copies.

Paired possession bootstrap on the released bundles (1,000 draws, seed 0, 5th–95th percentiles):
random gaps +0.0758 to +0.1004; normal copies −0.0078 to +0.0039. Both pairs passed checks for
match order, labels, half, frame and ball visibility. These intervals describe resampling on seven
matches and do not establish equivalence, replace the registered rules, or verify provenance.

**Decision (2026-11-05):** kept under the J6 rule. Under label-free random gaps the production model
loses 0.180 (0.7528 → 0.5726) and predicts 30% too few possessions; mixed training recovers about
half of that on every match, at a cost of 0.0017 on the normal copies, inside the 0.01 limit. The
general keep rule (mean up and 5/7) is not met on the normal copies. J6 was registered as a
robustness rule, not a score rule. The team selected `mixed`, but the production switch was
reverted pending an update to `test_v7_targets`; `PRODUCTION` remains `touch_bal` at this commit.

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
