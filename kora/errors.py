"""Possession diagnostics from aligned predictions, without fitting or decoding.

    python -m kora.errors --predictions BUNDLE.npz

Schema v1 contains scalar JSON ``meta`` (schema, fps, matches, provenance) and,
for each match M, M__y_true, M__y_pred, M__half, M__ball_seen, M__frame. Frame IDs
increase by one. Visibility must come from the exact prediction copy. Legacy
blackouts are simulated, not challenge tracking. Bundles must stay out of Git.

A bundle is clean only if its stated provenance is accurate. The loader checks
schema and alignment, not whether each held-out match was excluded from every
fitted stage and from decoder selection.

Minimal schema-v1 bundle (synthetic data, 0 H, 1 A, 2 D, 3 X):

    import json
    import numpy as np

    y_true = np.array([0, 0, 2, 1, 1, 2], dtype=np.int8)
    y_pred = np.array([0, 2, 2, 1, 1, 3], dtype=np.int8)
    meta = {"schema": 1, "fps": 25, "matches": ["M"],
            "provenance": "Synthetic example only. No match performance measured."}
    np.savez("predictions.npz", meta=json.dumps(meta),
             M__y_true=y_true, M__y_pred=y_pred,
             M__half=np.array([1, 1, 1, 2, 2, 2]),
             M__ball_seen=np.array([1, 0, 0, 1, 1, 0]),
             M__frame=np.arange(len(y_true)))

Associations are same-team structural diagnostics, not exhaustive counts of
erased possessions. A merge needs two or more same-team true spells. Truth H, A
predicted all H gives one 1:1 association and one isolated true A, not a merge.
Truth H, A, H predicted all H gives a merge (missed opposite possession) plus
an isolated true A.

The no-argument compatibility path uses development OOF probabilities and shared
decoder settings. Its results are not a clean estimate.
"""
import argparse
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .vendor import possession_metrics as pm

# Diagnostic association, not an official IoU match. Contained slivers qualify;
# weak positive overlaps are counted separately. This is not a tuned threshold.
ASSOC_MIN_COVER = 0.5
ASSOC_TYPES = ("association", "split", "merge", "complex", "isolated true", "isolated predicted")
BUNDLE_ARRAYS = ("y_true", "y_pred", "half", "ball_seen", "frame")
# Exact edges belong to the next group: searchsorted(..., side="right").
DURATION_EDGES_S = (1, 2, 4, 8, 15)
DURATION_GROUPS = ("<1", "1–2", "2–4", "4–8", "8–15", "15+")
DURATION_COUNT_FIELDS = ("support", "matched", "missed", "predicted", "matched_predicted", "false_predicted")
TRANSITION_CATEGORIES = {
    "H↔A": ((pm.H, pm.A), (pm.A, pm.H)),
    "H/A→D": ((pm.H, pm.D), (pm.A, pm.D)),
    "D→H/A": ((pm.D, pm.H), (pm.D, pm.A)),
}
TRANSITION_PAIRS = tuple(pair for pairs in TRANSITION_CATEGORIES.values() for pair in pairs)
TRANSITION_COUNT_FIELDS = ("n_true", "n_pred", "matched", "missed", "invented", "ambiguous_true", "ambiguous_pred")
WINDOW_RADIUS = 25  # [c - 25, c + 25), clipped to the half; 1 second at 25 fps.
WINDOW_ROWS = tuple(f"{category} only" for category in TRANSITION_CATEGORIES) + ("overlapping", "outside all windows")
WINDOW_INCLUSIVE_ROWS = tuple(f"{category} (inclusive)" for category in TRANSITION_CATEGORIES)
WINDOW_UNION = "any window (union)"


def _validate_inputs(y_true, y_pred, half, ball_seen, fps):
    if (not isinstance(fps, (int, float, np.integer, np.floating))
            or isinstance(fps, (bool, np.bool_)) or fps != 25):
        raise ValueError("fps must equal 25")
    arrays = [np.asarray(x) for x in (y_true, y_pred, half, ball_seen)]
    names = ("y_true", "y_pred", "half", "ball_seen")
    for name, array in zip(names, arrays):
        if array.ndim != 1:
            raise ValueError(f"{name} must be 1-D")
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("input arrays must have equal length")
    for name, array, maximum in zip(names[:2], arrays[:2], (pm.D, pm.X)):
        if array.dtype.kind not in "iu" or np.any((array < 0) | (array > maximum)):
            raise ValueError(f"{name} must contain integer codes in 0..{maximum}")
    half, seen = arrays[2:]
    if seen.dtype.kind not in "biuf" or not np.all((seen == 0) | (seen == 1)):
        raise ValueError("ball_seen must be binary (0 or 1)")
    if (half.dtype.kind not in "biufUS"
            or (half.dtype.kind == "f" and not np.isfinite(half).all())):
        raise ValueError("half IDs must be finite numbers or strings")
    edges = np.flatnonzero(half[1:] != half[:-1]) + 1
    bounds = [0, *edges.tolist(), len(half)] if len(half) else [0]
    runs = list(zip(bounds[:-1], bounds[1:]))
    ids = [half[a].item() for a, _ in runs]
    if len(set(ids)) != len(ids):
        raise ValueError("each half ID must form one contiguous block")
    return (*arrays, runs)


def match_spell_pairs(overlaps, n_gt, threshold):
    """Return (gt index, prediction index, IoU) using the official greedy order.

    overlaps must come from pm.spell_overlaps, including its stable tie order.
    Indices refer to the half's spell lists; both sides are consumed once.
    """
    flags = np.zeros(n_gt, dtype=bool)
    used_pred, pairs = set(), []
    for iou, i, j in overlaps:
        if iou < threshold:
            break
        if flags[i] or j in used_pred:
            continue
        flags[i] = True
        used_pred.add(j)
        pairs.append((i, j, iou))
    assert np.array_equal(flags, pm.matched_gt_flags(overlaps, n_gt, threshold))
    return pairs


def _gap_kind(pieces, other, team, bounds):
    """Classify gaps between consecutive pieces, clipped to their container."""
    labels = set()
    pieces = sorted(pieces)
    for left, right in zip(pieces, pieces[1:]):
        a, b = max(bounds[0], left[1]), min(bounds[1], right[0])
        if a < b:
            labels.update(np.unique(other[a:b]).tolist())
    if labels == {1 - team}:
        return "opposite"
    if labels == {pm.D}:
        return "dead"
    return "mixed"  # Includes residual X, ignored same-team pieces, empty gaps.


def _associations(gt, pred, y_true, filled, overlaps):
    counts = {kind: {"components": 0, "true_spells": 0, "pred_spells": 0}
              for kind in ASSOC_TYPES}
    subtypes = {"split": {}, "merge": {}}
    graph = [set() for _ in range(len(gt) + len(pred))]
    discarded = 0
    for _, i, j in overlaps:
        a, b, _ = gt[i]
        c, d, _ = pred[j]
        intersection = min(b, d) - max(a, c)
        if intersection >= ASSOC_MIN_COVER * min(b - a, d - c):
            graph[i].add(len(gt) + j)
            graph[len(gt) + j].add(i)
        else:
            discarded += 1
    visited = set()
    for node in range(len(graph)):
        if node in visited:
            continue
        stack, gs, ps = [node], [], []
        visited.add(node)
        while stack:
            current = stack.pop()
            if current < len(gt):
                gs.append(current)
            else:
                ps.append(current - len(gt))
            for neighbour in graph[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        ng, npred = len(gs), len(ps)
        if not npred:
            kind = "isolated true"
        elif not ng:
            kind = "isolated predicted"
        elif ng == npred == 1:
            kind = "association"
        elif ng == 1:
            kind = "split"
        elif npred == 1:
            kind = "merge"
        else:
            kind = "complex"
        for key, value in (("components", 1), ("true_spells", ng), ("pred_spells", npred)):
            counts[kind][key] += value
        if kind == "split":
            container = gt[gs[0]]
            gap = _gap_kind([pred[j] for j in ps], filled, container[2], container[:2])
            subtype = {"opposite": "wrong-team fragment", "dead": "false D interruption",
                       "mixed": "mixed"}[gap]
        elif kind == "merge":
            container = pred[ps[0]]
            gap = _gap_kind([gt[i] for i in gs], y_true, container[2], container[:2])
            subtype = {"opposite": "missed opposite possession", "dead": "missed stoppage",
                       "mixed": "mixed"}[gap]
        else:
            continue
        row = subtypes[kind].setdefault(subtype, {"components": 0, "true_spells": 0, "pred_spells": 0})
        row["components"] += 1
        row["true_spells"] += ng
        row["pred_spells"] += npred
    return {"counts": counts, "subtypes": subtypes, "positive_overlaps": len(overlaps),
            "discarded_overlaps": discarded, "retained_overlaps": len(overlaps) - discarded}


def _duration_stats(lengths):
    return ((float(np.mean(lengths)), float(np.median(lengths))) if len(lengths)
            else (None, None))


def _duration_counts(true_lengths, pred_lengths, matches):
    """Bucket both sides independently, after matching all spells of the half."""
    true_groups = np.searchsorted(DURATION_EDGES_S, true_lengths, side="right")
    pred_groups = np.searchsorted(DURATION_EDGES_S, pred_lengths, side="right")
    n_groups = len(DURATION_GROUPS)
    support = np.bincount(true_groups, minlength=n_groups)
    predicted = np.bincount(pred_groups, minlength=n_groups)
    counts = {}
    for iou, pairs in matches.items():
        matched = np.zeros(n_groups, dtype=int)
        matched_predicted = np.zeros(n_groups, dtype=int)
        for i, j, _ in pairs:
            matched[true_groups[i]] += 1
            matched_predicted[pred_groups[j]] += 1
        counts[iou] = {
            group: {"support": int(support[k]), "matched": int(matched[k]),
                    "missed": int(support[k] - matched[k]), "predicted": int(predicted[k]),
                    "matched_predicted": int(matched_predicted[k]),
                    "false_predicted": int(predicted[k] - matched_predicted[k])}
            for k, group in enumerate(DURATION_GROUPS)
        }
    return counts


def _frame_count_matrix(y_true, y_pred, ball_seen):
    """Counts [true H/A/D, visibility missing/seen, raw prediction H/A/D/X]."""
    index = y_true.astype(np.intp) * 8 + ball_seen.astype(np.intp) * 4 + y_pred.astype(np.intp)
    return np.bincount(index, minlength=24).reshape(3, 2, 4)


def _frame_error_rows(counts):
    """Derive V4 numerators and exhaustive outcomes from a pooled count matrix."""
    teams = (("H", (pm.H,)), ("A", (pm.A,)))
    visibility = (("seen", (1,)), ("missing", (0,)), ("all", (0, 1)))
    live_groups = [(name, codes, vis, seen) for name, codes in teams for vis, seen in visibility[:2]]
    live_groups += [(name, codes, "all", (0, 1)) for name, codes in teams]
    live_groups += [("live", (pm.H, pm.A), vis, seen) for vis, seen in visibility]
    live = []
    for name, codes, vis, seen in live_groups:
        pred = counts[list(codes)][:, list(seen), :].sum(axis=(0, 1))
        correct = sum(int(counts[team, list(seen), team].sum()) for team in codes)
        wrong = sum(int(counts[team, list(seen), 1 - team].sum()) for team in codes)
        false_d, pred_x, denominator = int(pred[pm.D]), int(pred[pm.X]), int(pred.sum())
        assert false_d + pred_x + wrong + correct == denominator
        live.append({"true": name, "visibility": vis, "false_d": false_d,
                     "denominator": denominator, "pred_x": pred_x,
                     "wrong_team": wrong, "correct_team": correct})
    dead = []
    for vis, seen in visibility:
        pred = counts[pm.D, list(seen), :].sum(axis=0)
        pred_h, pred_a, correct_d, pred_x = map(int, pred)
        denominator = int(pred.sum())
        assert pred_h + pred_a + pred_x + correct_d == denominator
        dead.append({"visibility": vis, "pred_live": pred_h + pred_a,
                     "denominator": denominator, "pred_h": pred_h, "pred_a": pred_a,
                     "pred_x": pred_x, "correct_d": correct_d})
    return {"live": live, "dead": dead}


def _transition_windows(y_true, y_pred, gt, run):
    a, b = run
    cover = np.zeros((len(TRANSITION_CATEGORIES), b - a), dtype=bool)
    for k, pairs in enumerate(TRANSITION_CATEGORIES.values()):
        for c, origin, destination in gt:
            if (origin, destination) in pairs:
                cover[k, max(a, c - WINDOW_RADIUS) - a:min(b, c + WINDOW_RADIUS) - a] = True
    n_cover = cover.sum(axis=0)
    masks = [category & (n_cover == 1) for category in cover] + [n_cover >= 2, n_cover == 0]
    wrong = y_true[a:b] != y_pred[a:b]  # Raw X counts as wrong.
    rows = {name: {"frames": int(mask.sum()), "wrong": int(wrong[mask].sum())}
            for name, mask in zip(WINDOW_ROWS, masks)}
    assert sum(row["frames"] for row in rows.values()) == b - a
    assert sum(row["wrong"] for row in rows.values()) == int(wrong.sum())
    rows.update({name: {"frames": int(mask.sum()), "wrong": int(wrong[mask].sum())}
                 for name, mask in zip(WINDOW_INCLUSIVE_ROWS, cover)})
    rows[WINDOW_UNION] = {"frames": int((n_cover > 0).sum()), "wrong": int(wrong[n_cover > 0].sum())}
    return rows


def _transition_diagnostics(y_true, y_pred, filled, run):
    """Match only this half; retain counts and latency deltas, not event frames."""
    gt = pm.transitions(y_true, [run])
    pred = pm.transitions(filled, [run])
    exact, official = {}, {}
    for tol in pm.TAUS:
        matched, _, _ = pm.match_transitions(gt, pred, tol, typed="pair")
        exact[tol] = {}
        for key in TRANSITION_PAIRS:
            gs = [t for t in gt if t[1:] == key]
            ps = [t for t in pred if t[1:] == key]
            latencies = [p[0] - g[0] for g, p in matched if g[1:] == key]
            n = len(latencies)
            exact[tol][key] = {
                "n_true": len(gs), "n_pred": len(ps), "matched": n,
                "missed": len(gs) - n, "invented": len(ps) - n,
                "ambiguous_true": pm._count_ambiguous(gs, ps, tol, typed="pair"),
                "ambiguous_pred": pm._count_ambiguous(ps, gs, tol, typed="pair"),
                "latencies": latencies,
            }
        reference = pm.transition_scores(gt, pred, tol, typed="to")
        official[tol] = {key: reference[source] for key, source in
                         (("n_true", "n_gt"), ("n_pred", "n_pred"), ("matched", "n_matched"),
                          ("missed", "n_missed"), ("invented", "n_spurious"))}
    return {"pairs": exact, "official": official, "windows": _transition_windows(y_true, y_pred, gt, run)}


def _pool_transition_rows(rows):
    pooled = {key: sum(row[key] for row in rows) for key in TRANSITION_COUNT_FIELDS}
    pooled["latencies"] = [latency for row in rows for latency in row["latencies"]]
    assert pooled["matched"] == len(pooled["latencies"])
    assert pooled["n_true"] == pooled["matched"] + pooled["missed"]
    assert pooled["n_pred"] == pooled["matched"] + pooled["invented"]
    return pooled


def _latency_stats(latencies):
    values = np.asarray(latencies, dtype=float)
    return {"early": int(np.sum(values < 0)), "exact": int(np.sum(values == 0)),
            "late": int(np.sum(values > 0)),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "median_abs": float(np.median(np.abs(values))) if len(values) else None,
            "p90_abs": float(np.percentile(np.abs(values), 90)) if len(values) else None}


def analyse(y_true, y_pred, half, ball_seen, fps=25):
    """Pure, half-local V1–V4 diagnostics for one frame-aligned match at 25 fps.

    Temporal/spell metrics fill predicted X inside each half, like pm.evaluate.
    train.spell_f1 does not fill X. An entirely abstaining half remains X and has
    no predicted spells. Frame counts use raw predictions: X is wrong, remains
    in every denominator, and is reported separately. No inputs are modified.
    Each half's v4 matrix has axes [true H/A/D, visibility 0/1, raw pred H/A/D/X].
    """
    y_true, y_pred, half, ball_seen, runs = _validate_inputs(y_true, y_pred, half, ball_seen, fps)
    filled = pm.fill_abstentions(y_pred, runs)
    halves = []
    for a, b in runs:
        gt = pm.possession_spells(y_true, [(a, b)])
        pred = pm.possession_spells(filled, [(a, b)])
        overlaps = pm.spell_overlaps(gt, pred)
        true_lengths = np.array([(z - s) / fps for s, z, _ in gt])
        pred_lengths = np.array([(z - s) / fps for s, z, _ in pred])
        true_mean, true_median = _duration_stats(true_lengths)
        pred_mean, pred_median = _duration_stats(pred_lengths)
        n, ng, npred = b - a, len(gt), len(pred)
        true_d = int(np.sum(y_true[a:b] == pm.D))
        pred_d = int(np.sum(y_pred[a:b] == pm.D))
        pred_x = int(np.sum(y_pred[a:b] == pm.X))
        scale = 2700 * fps / n  # Total half time, including D and abstentions.
        matches = {iou: match_spell_pairs(overlaps, ng, iou) for iou in pm.SPELL_IOUS}
        halves.append({
            "half": half[a].item(), "n_frames": n, "n_true": ng, "n_pred": npred,
            "count_error": npred - ng, "true_per_45": ng * scale,
            "pred_per_45": npred * scale, "count_error_per_45": (npred - ng) * scale,
            "true_lengths_s": true_lengths, "pred_lengths_s": pred_lengths,
            "true_mean_s": true_mean, "true_median_s": true_median,
            "pred_mean_s": pred_mean, "pred_median_s": pred_median,
            "true_d_frames": true_d, "pred_d_frames": pred_d, "pred_x_frames": pred_x,
            "true_d_share": true_d / n, "pred_d_share": pred_d / n, "pred_x_share": pred_x / n,
            "wrong_frames": int(np.sum(y_true[a:b] != y_pred[a:b])),
            "true_spells": gt, "pred_spells": pred,
            "matches": matches,
            "v2": _duration_counts(true_lengths, pred_lengths, matches),
            "v3": _transition_diagnostics(y_true, y_pred, filled, (a, b)),
            "v4": _frame_count_matrix(y_true[a:b], y_pred[a:b], ball_seen[a:b]),
            "associations": _associations(gt, pred, y_true, filled, overlaps),
        })
    return {"fps": fps, "runs": runs, "halves": halves}


def load_bundle(path):
    """Check bundle schema and alignment; do not verify its stated provenance."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("prediction bundle must be an .npz file")
    with np.load(path, allow_pickle=False) as bundle:
        if "meta" not in bundle.files:
            raise ValueError("bundle is missing meta")
        raw_meta = bundle["meta"]
        if raw_meta.shape != () or raw_meta.dtype.kind != "U":
            raise ValueError("bundle meta must be a scalar JSON string")
        try:
            meta = json.loads(raw_meta.item())
        except json.JSONDecodeError as exc:
            raise ValueError("bundle meta is not valid JSON") from exc
        if not isinstance(meta, dict) or type(meta.get("schema")) is not int or meta["schema"] != 1:
            raise ValueError("bundle meta.schema must equal 1")
        matches = meta.get("matches")
        if (not isinstance(matches, list) or not matches
                or any(not isinstance(m, str) or not m.strip() for m in matches)
                or len(set(matches)) != len(matches)):
            raise ValueError("bundle meta.matches must be a nonempty list of unique match IDs")
        if not isinstance(meta.get("provenance"), str) or not meta["provenance"].strip():
            raise ValueError("bundle meta.provenance must be nonempty text")
        if "fps" not in meta:
            raise ValueError("bundle meta is missing fps")
        expected = {"meta"} | {f"{m}__{key}" for m in matches for key in BUNDLE_ARRAYS}
        if expected != set(bundle.files):
            missing = sorted(expected - set(bundle.files))
            extra = sorted(set(bundle.files) - expected)
            raise ValueError(f"bundle keys do not match schema (missing: {', '.join(missing) or 'none'}; "
                             f"unexpected: {', '.join(extra) or 'none'})")
        data = {}
        for m in matches:
            arrays = tuple(bundle[f"{m}__{key}"] for key in BUNDLE_ARRAYS)
            try:
                _validate_inputs(*arrays[:4], meta["fps"])
                frame = arrays[4]
                if frame.ndim != 1 or len(frame) != len(arrays[0]) or frame.dtype.kind not in "iu":
                    raise ValueError("frame must be a 1-D integer array aligned with the labels")
                if np.any(frame[1:] <= frame[:-1]) or np.any(np.diff(frame) != 1):
                    raise ValueError("frame must be strictly increasing by 1")
            except ValueError as exc:
                raise ValueError(f"match {m}: {exc}") from exc
            data[m] = arrays[:4]
    return meta, data


def check_legacy_artifacts(models):
    """Fail before imports/copies if legacy prediction artifacts are unavailable."""
    models = Path(models)
    config = models / "decoder.json"
    cfg = json.loads(config.read_text()) if config.is_file() else None
    if config.is_file() and not isinstance(cfg, dict):
        raise ValueError("legacy decoder.json must contain a JSON object")
    required = [models / "oof.npz", config]
    if cfg is not None and cfg.get("hsmm"):
        required.append(models / "change_oof.npz")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing legacy artifacts: " + ", ".join(missing)
                         + ". Use --predictions BUNDLE.npz for aligned held-out labels; no training was run.")
    return cfg


def _load_legacy():
    models = Path(__file__).resolve().parent.parent / "models"
    cfg = check_legacy_artifacts(models)
    from .boundary import runs_of
    from .decode import decode, decode_hsmm, log_duration
    from .external import MATCHES
    from .train import copies, segments_of

    with np.load(models / "oof.npz", allow_pickle=False) as bundle:
        oof = {m: bundle[m] for m in MATCHES}
    h = cfg.get("hsmm")
    if h:
        with np.load(models / "change_oof.npz", allow_pickle=False) as bundle:
            change = {m: bundle[m] for m in MATCHES}
    prior = np.array(cfg["prior"])
    copies_by_match = {m: copies(m)[1] for m in MATCHES}
    data = {}
    for m, (Xe, y, half) in copies_by_match.items():
        if h:
            ld = log_duration([r for k in MATCHES if k != m for r in runs_of(*copies_by_match[k][1:])])
            pred = decode_hsmm(oof[m], change[m], segments_of(half), prior, ld,
                               h["cost_ha"], h["cost_d"], h["w_dur"], h["beta"], h["rate"])
        else:
            pred = decode(oof[m], segments_of(half), prior, cfg["alpha"], cfg["cost_ha"], cfg["cost_d"])
        data[m] = (y, pred, half, Xe["b_seen"].to_numpy())
    return {"fps": 25, "provenance": "development out-of-fold, decoder settings chosen on all matches: "
            "not a clean estimate"}, data


def _table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(cell(x) for x in row) + " |")
    print()


def _number(value):
    return "n/a" if value is None else f"{value:.3f}"


def _ratio(numerator, denominator):
    return _number(numerator / denominator if denominator else None)


def _print_duration_report(halves):
    # Only cached counts are pooled; no spell matching or duration binning here.
    counts = {
        iou: {group: {key: sum(h["v2"][iou][group][key] for h in halves)
                      for key in DURATION_COUNT_FIELDS}
              for group in DURATION_GROUPS}
        for iou in pm.SPELL_IOUS
    }
    print("V2 duration recovery: true and predicted durations are grouped independently.\n")
    _table(("IoU", "Duration s", "True support", "Matched true", "Missed true", "Recall",
            "Predicted spells", "Matched predicted", "False predicted", "Precision"),
           [(iou, group, row["support"], row["matched"], row["missed"],
             _ratio(row["matched"], row["support"]), row["predicted"], row["matched_predicted"],
             row["false_predicted"], _ratio(row["matched_predicted"], row["predicted"]))
            for iou in pm.SPELL_IOUS for group, row in counts[iou].items()])
    iou_headers = tuple(f"IoU {iou:g}" for iou in pm.SPELL_IOUS)
    first_iou = pm.SPELL_IOUS[0]  # Support is the same at every IoU; do not sum it across IoUs.
    print("V2 recall by true duration (pooled counts).\n")
    _table(("True duration s", "True support", *iou_headers),
           [(group, counts[first_iou][group]["support"],
             *[_ratio(counts[iou][group]["matched"], counts[iou][group]["support"])
               for iou in pm.SPELL_IOUS]) for group in DURATION_GROUPS])
    print("V2 precision by predicted duration (pooled counts).\n")
    _table(("Pred duration s", "Predicted spells", *iou_headers),
           [(group, counts[first_iou][group]["predicted"],
             *[_ratio(counts[iou][group]["matched_predicted"], counts[iou][group]["predicted"])
               for iou in pm.SPELL_IOUS]) for group in DURATION_GROUPS])


def _print_frame_report(halves):
    counts = np.zeros((3, 2, 4), dtype=np.int64)
    for h in halves:
        counts += h["v4"]
    rows = _frame_error_rows(counts)
    visibility_header = "Ball visibility (legacy: simulated blackout)"
    print("V4 false D during live play (raw predictions, pooled counts).\n")
    _table(("True label", visibility_header, "False D frames", "Live frames", "False D rate",
            "Pred X", "Wrong team", "Correct team"),
           [(r["true"], r["visibility"], r["false_d"], r["denominator"],
             _ratio(r["false_d"], r["denominator"]), r["pred_x"], r["wrong_team"], r["correct_team"])
            for r in rows["live"]])
    print("V4 reverse error on true D (raw predictions, pooled counts).\n")
    _table((visibility_header, "Pred H/A frames (subtotal)", "True D frames", "H/A rate",
            "Pred H", "Pred A", "Pred X", "Pred D correct"),
           [(r["visibility"], r["pred_live"], r["denominator"],
             _ratio(r["pred_live"], r["denominator"]), r["pred_h"], r["pred_a"], r["pred_x"], r["correct_d"])
            for r in rows["dead"]])
    print("Ball visibility comes from the supplied evaluation copy. Legacy b_seen is simulated blackout, "
          "not challenge tracking. Pred H/A is a subtotal of Pred H and Pred A.\n")
    print('The existing "2% of errors on true dead ball" figure is a different quantity: '
          'the share of all wrong frames whose true label is D.\n')


def _timing(value):
    return "n/a" if value is None else f"{value:.3f} / {value / pm.FPS:.3f}"


def _print_transition_report(halves):
    display = []
    for tol in pm.TAUS:
        pooled = {key: _pool_transition_rows([h["v3"]["pairs"][tol][key] for h in halves])
                  for key in TRANSITION_PAIRS}
        for category, keys in TRANSITION_CATEGORIES.items():
            display.append((tol, category, "category subtotal", _pool_transition_rows([pooled[key] for key in keys])))
            display.extend((tol, category, f"{pm.LABELS[key[0]]}→{pm.LABELS[key[1]]}", pooled[key]) for key in keys)
    print('V3 exact-pair transition diagnostics (typed="pair").\n')
    print("The categorised table uses exact pairs and is a diagnostic, not the official default metric. "
          "Category subtotals include their two pair rows.\n")
    _table(("Tol frames", "Category", "Pair", "n_true", "n_pred", "Matched", "Missed", "Invented",
            "Precision", "Recall", "Ambiguous true", "Ambiguous pred"),
           [(tol, category, pair, r["n_true"], r["n_pred"], r["matched"], r["missed"], r["invented"],
             _ratio(r["matched"], r["n_pred"]), _ratio(r["matched"], r["n_true"]),
             r["ambiguous_true"], r["ambiguous_pred"]) for tol, category, pair, r in display])
    print("V3 matched-transition timing. Positive latency means late; values are frames / seconds.\n")
    _table(("Tol frames", "Category", "Pair", "Early", "Exact", "Late", "Mean signed frames / s",
            "Median signed frames / s", "Median abs frames / s", "p90 abs frames / s"),
           [(tol, category, pair, stats["early"], stats["exact"], stats["late"],
             *[_timing(stats[key]) for key in ("mean", "median", "median_abs", "p90_abs")])
            for tol, category, pair, r in display for stats in [_latency_stats(r["latencies"])]])
    print("Ambiguity counts flag where matching was a choice: true events with multiple feasible predictions "
          "and predicted events with multiple feasible truths, within the half and exact pair. "
          "Latency figures are indicative when either count is nonzero.\n")
    print('V3 official default reference (typed="to", matched per half, pooled by count).\n')
    reference = {tol: {key: sum(h["v3"]["official"][tol][key] for h in halves)
                      for key in TRANSITION_COUNT_FIELDS[:5]} for tol in pm.TAUS}
    _table(("Tol frames", "n_true", "n_pred", "Matched", "Missed", "Invented", "Precision", "Recall"),
           [(tol, r["n_true"], r["n_pred"], r["matched"], r["missed"], r["invented"],
             _ratio(r["matched"], r["n_pred"]), _ratio(r["matched"], r["n_true"])) for tol, r in reference.items()])
    windows = {name: {key: sum(h["v3"]["windows"][name][key] for h in halves) for key in ("frames", "wrong")}
               for name in (*WINDOW_ROWS, *WINDOW_INCLUSIVE_ROWS, WINDOW_UNION)}
    all_wrong = sum(windows[name]["wrong"] for name in WINDOW_ROWS)
    headers = ("Window region", "Frames", "Wrong frames", "Wrong / frames", "Wrong / all wrong frames")

    def window_row(name, label=None):
        r = windows[name]
        return (label or name, r["frames"], r["wrong"], _ratio(r["wrong"], r["frames"]), _ratio(r["wrong"], all_wrong))

    print(f"V3 true-transition error windows: radius {WINDOW_RADIUS} frames, half-open and clipped to each half.\n")
    _table(headers, [window_row(name) for name in WINDOW_ROWS])
    print("---\n\nWindow subtotals (already included in the exclusive rows):\n")
    _table(headers, [window_row(name) for name in WINDOW_INCLUSIVE_ROWS]
           + [window_row(WINDOW_UNION, WINDOW_UNION + " — subtotal")])
    print("Inclusive rows overlap, so their shares can sum above 1; "
          "the exclusive rows partition all frames and wrong frames.\n")


def print_report(meta, results):
    """Print aggregates only: no frame IDs, spell intervals, or matched-pair lists."""
    halves = [(m, h) for m, result in results for h in result["halves"]]
    fields = ("n_frames", "n_true", "n_pred", "true_d_frames", "pred_d_frames", "pred_x_frames", "wrong_frames")
    total = {key: sum(h[key] for _, h in halves) for key in fields}
    n = total["n_frames"]
    total["count_error"] = total["n_pred"] - total["n_true"]
    for key, count in (("true_per_45", total["n_true"]), ("pred_per_45", total["n_pred"]),
                       ("count_error_per_45", total["count_error"])):
        total[key] = count * 2700 * meta["fps"] / n if n else None
    for side in ("true", "pred"):
        lengths = np.concatenate([h[f"{side}_lengths_s"] for _, h in halves]) if halves else []
        total[f"{side}_mean_s"], total[f"{side}_median_s"] = _duration_stats(lengths)
    print("# Possession error analysis — V1–V4\n")
    print("Provenance: " + meta["provenance"]
          + " (producer-stated; a bundle is clean only if this provenance is accurate. "
          "The loader checks schema and alignment, not whether each held-out match was excluded "
          "from every fitted stage and from decoder selection.)\n")
    rows = [(m, h["half"], h) for m, h in halves] + [("ALL", "all", total)]
    _table(("Match", "Half", "Frames", "True", "Predicted", "Pred − true", "True/45 min", "Pred/45 min", "Error/45 min"),
           [(m, half, h["n_frames"], h["n_true"], h["n_pred"], h["count_error"],
             _number(h["true_per_45"]), _number(h["pred_per_45"]), _number(h["count_error_per_45"]))
            for m, half, h in rows])
    print("Rates use total half time, including D. ALL rates use pooled exposure.\n")
    _table(("Match", "Half", "True mean s", "True median s", "Pred mean s", "Pred median s",
            "True D frames/share", "Raw pred D frames/share", "Raw pred X frames/share", "Wrong frames"),
           [(m, half, *[_number(h[key]) for key in ("true_mean_s", "true_median_s", "pred_mean_s", "pred_median_s")],
             *[f"{h[key]} / {_number(h[key] / h['n_frames'] if h['n_frames'] else None)}"
               for key in ("true_d_frames", "pred_d_frames", "pred_x_frames")], h["wrong_frames"])
            for m, half, h in rows])
    _table(("IoU", "Matched", "Missed true", "Invented predicted"),
           [(iou, tp, total["n_true"] - tp, total["n_pred"] - tp)
            for iou in pm.SPELL_IOUS
            for tp in [sum(len(h["matches"][iou]) for _, h in halves)]])
    print(f"Associations retain intersection >= {ASSOC_MIN_COVER:g} × shorter spell length. "
          "A 1:1 association is not an official IoU match; isolated spells have no retained overlap. "
          "These are same-team structural diagnostics, not exhaustive counts of erased possessions. "
          "A merge needs two or more same-team true spells. Truth H, A predicted all H gives one "
          "1:1 association and one isolated true A, not a merge. Truth H, A, H predicted all H gives "
          "a merge (missed opposite possession) plus an isolated true A.\n")
    keys = ("components", "true_spells", "pred_spells")
    _table(("Association type", "Components", "True spells", "Predicted spells"),
           [(kind, *[sum(h["associations"]["counts"][kind][key] for _, h in halves) for key in keys])
            for kind in ASSOC_TYPES])
    _table(("Type", "Gap subtype", "Components", "True spells", "Predicted spells"),
           [(kind, subtype, *[sum(h["associations"]["subtypes"][kind].get(subtype, {}).get(key, 0)
                                 for _, h in halves) for key in keys])
            for kind, subtypes in (("split", ("wrong-team fragment", "false D interruption", "mixed")),
                                   ("merge", ("missed opposite possession", "missed stoppage", "mixed")))
            for subtype in subtypes])
    _table(("Same-team positive overlaps", "Retained", "Discarded by association rule"),
           [[sum(h["associations"][key] for _, h in halves)
             for key in ("positive_overlaps", "retained_overlaps", "discarded_overlaps")]])
    _print_duration_report([h for _, h in halves])
    _print_frame_report([h for _, h in halves])
    _print_transition_report([h for _, h in halves])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path,
                        help="local schema-v1 bundle of aligned decoded labels (provenance is producer-stated)")
    args = parser.parse_args()
    try:
        meta, data = load_bundle(args.predictions) if args.predictions else _load_legacy()
        results = [(m, analyse(*arrays, fps=meta["fps"])) for m, arrays in data.items()]
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
        parser.exit(2, f"kora.errors: {exc}\n")
    print_report(meta, results)


if __name__ == "__main__":
    main()
