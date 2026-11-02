# Research log

**Last updated:** 2026-11-02

**Statuses**

- **[pending]** Awaiting work
- **[verified]** Confirmed
- **[deferred]** Postponed
- **[dropped]** Tried but not worth keeping
- **[inconclusive]** Unclear evidence

An outcome of **—** means the experiment has not run.
Model comparisons depend on Y1.

## Younis — evaluation and reproducibility

### [pending] Y1: Is the held-out match excluded from the whole fitted pipeline?

**Hypothesis:** Boundary-training inputs can come from first-stage models that saw the outer
held-out match, creating indirect leakage.

**Experiment:**

1. For outer match `m`, generate boundary-training predictions for match `k` without training
   the first stage on either match.
2. Repeat this isolation inside model selection. Fit priors, durations, and boundary rates on
   the training partition.
3. Include excluded match IDs, configuration versions, and seeds in cache keys.
4. Recompute `main` with full nested CV.

**Outcome:** Clean baseline pending. Recorded scores `0.7067` (plain), `0.7194` (HSMM), and
`0.7187` (nested settings) come from the old evaluation.

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

**Outcome:** —

### [pending] V2: Are short possessions missing or mistimed?

**Hypothesis:** IoU 0.5 alone hides the difference between missed possessions and inaccurate boundaries.

**Experiment:** Report recall for `<1`, `1–2`, `2–4`, `4–8`, `8–15`, and `15+` seconds at IoUs
**0.3, 0.4, 0.5, 0.6, 0.7**. Match all spells before grouping by true duration. Include support,
precision, and false predicted spells by duration. Recheck the reported one-fifth recall below
one second on clean predictions.

**Outcome:** —

### [pending] V3: Which transition types fail, and by how much?

**Hypothesis:** The reported 68% of wrong frames near transitions mixes handovers, stoppages, and restarts.

**Experiment:** Separate H↔A, H/A→D, and D→H/A. Report missed/invented transitions, precision,
recall, early/late bias, and median/p90 absolute timing errors. Match exact transition types
within each half at the official tolerances. Report errors within one second of each type,
their share of all wrong frames, and overlapping windows.

**Outcome:** —

### [pending] V4: How often is D predicted during live play?

**Hypothesis:** The reported 2% of errors on true dead ball excludes false D during possession.

**Experiment:** Report `false D frames / true live-play frames`, split by H/A and ball visibility.
Measure the reverse error: H/A predicted on true D. Include numerators and denominators.

**Outcome:** —

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

**Outcome:** —

## José — missing-ball robustness and extra data

### [pending] J1: Does the score depend on one blackout simulation?

**Hypothesis:** Label-conditioned gaps may reward an unrealistic association between missing ball
and dead ball.

**Experiment:** Compare clean tracking, current degradation, label-independent gaps, heavier
live-play blackouts, and player-loss/close-up patterns. Use at least three fixed seeds per
stochastic condition and identical masks for competing models. Report gap lengths and paired
per-match/per-condition scores. Fix condition weights before evaluation.

**Outcome:** —

### [pending] J2: Can player movement identify possession during ball gaps?

**Hypothesis:** Player geometry and movement can provide evidence without reconstructing ball coordinates.

**Experiment:** Train a player-only H/A model without ball-derived features or carried ball
context. Compare main, auxiliary, and fused models on identical gaps: `<0.5`, `0.5–1`, `1–2`,
`2–4`, and `4+` seconds. Tune visibility/gap-dependent fusion within training folds. Report gap
errors and overall spellF1.

**Outcome:** —

### [pending] J3: Does the domain comparison survive time- and match-separated splits?

**Hypothesis:** Random frame splitting in `kora.shift` gives correlated neighboring frames to
training and evaluation.

**Experiment:** Use separated challenge time blocks and match-aware source splits. Compare AUC
and feature importance across splits before changing degradation or removing features.

**Outcome:** —

### [pending] J4: Does Metrica improve training despite different possession labels?

**Hypothesis:** Extra tracking can help without treating event-derived ownership as equivalent ground truth.

**Experiment:** Verify coordinates, frame rate, synchronization, and label definitions for each
[sample game](https://github.com/metrica-sports/sample-data). Compare auxiliary pretraining or
high-confidence event anchors followed by Sportec training against Sportec-only training.
Evaluate on the same held-out matches. Avoid treating full event-derived paths as hard labels
without validating the mapping.

**Outcome:** —

### [pending] J5: Are useful tracking measurements discarded?

**Hypothesis:** Retaining only ball x/y may omit useful measurements.

**Experiment:** Confirm unused columns, units, and missing-value rules from provider metadata.
Evaluate supported signals individually. Do not assume a column means confidence or speed.

**Outcome:** —

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
