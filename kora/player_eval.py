"""Exploratory player-only H/A leave-one-match-out evaluation on seven matches.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m kora.player_eval
    python -m kora.player_eval --held-out J03WPY --seeds 11 22 33

Train on the other matches' existing degraded/mirrored copies, with fixed model
settings and no held-out fitting or selection. Report live-frame team accuracy
and simulated missing-live accuracy, including missing-gap duration buckets.
The always_H baseline is fixed in advance, without held-out team selection.
Buckets use full missing runs (including dead frames), split at half boundaries;
only true H/A frames contribute to accuracy. Pool counts across matches/seeds,
so repeated seeds count repeated frames, not independent observations.

This is neither nested spellF1 evaluation nor a fusion experiment. Degradation
is label-conditioned, and the extractor uses future player context and can mix
context across the concatenated halves. No model,
feature cache or submission is written; missing source data may be downloaded
and cached by the existing loader.
"""
import argparse
import csv
import sys
import time

import numpy as np
import pandas as pd

from .blackout_eval import conditions
from .degrade import FPS, degrade, stretches
from .external import MATCHES, load
from .features import mirror
from .player_only import fit_player_model, player_features
from .train import SEED, SUBSAMPLE, TRAIN_COPIES, segments_of


GAP_BINS = ('gap_<0.5s', 'gap_0.5-1s', 'gap_1-2s', 'gap_2-4s', 'gap_4+s')
DEFAULT_SEEDS = (11, 22, 33)


def training_data(match):
    """Build only player features, using the main pipeline's fixed augmentation."""
    data = load(match)
    rng = np.random.default_rng([SEED, list(MATCHES).index(match)])
    xs, ys = [], []
    for _ in range(TRAIN_COPIES):
        players, ball = degrade(data['players'], data['ball'], data['label'], rng)
        normal = player_features(players, data['team'], data['half'])
        xs.append(normal.iloc[::SUBSAMPLE].copy())
        ys.append(data['label'][::SUBSAMPLE].copy())
        mp, mt, _ = mirror(players, data['team'], ball)
        mirrored = player_features(mp, mt, data['half'])
        xs.append(mirrored.iloc[::SUBSAMPLE].copy())
        y = data['label'][::SUBSAMPLE]
        ys.append(np.where(y < 2, 1 - y, y))
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def fit_fold(training, held_out):
    """Fit from explicitly partitioned per-match arrays, excluding the held-out ID."""
    if held_out not in MATCHES and held_out not in training:
        raise ValueError('unknown held-out match')
    others = [arrays for match, arrays in training.items() if match != held_out]
    if not others:
        raise ValueError('no other training matches')
    features = pd.concat([x for x, _ in others], ignore_index=True)
    labels = np.concatenate([y for _, y in others])
    return fit_player_model(features, labels)


def gap_scores(labels, pred, half, missing):
    """Return correct/total live-frame counts, preserving empty bucket support."""
    labels, pred, half = np.asarray(labels), np.asarray(pred), np.asarray(half)
    missing = np.asarray(missing, dtype=bool)
    if labels.ndim != 1 or any(a.shape != labels.shape for a in (pred, half, missing)):
        raise ValueError('labels, predictions, half and mask must be equally sized vectors')
    if not np.isin(labels, (0, 1, 2)).all() or not np.isin(pred, (0, 1)).all():
        raise ValueError('expected H/A/D labels and H/A predictions')
    live, correct = labels < 2, labels == pred

    def count(mask):
        scored = mask & live
        return {'correct': int((correct & scored).sum()), 'total': int(scored.sum())}

    result = {'live': count(live), 'missing_live': count(missing)}
    # Each frame belongs to its full ball-gap duration, even across H/A/D changes.
    bucket = np.full(len(labels), -1, dtype=np.int8)
    for start, end in segments_of(half):
        for a, b in stretches(missing[start:end]):
            duration = (b - a) / FPS
            bucket[start + a:start + b] = np.searchsorted((0.5, 1, 2, 4), duration, side='right')
    return {**result, **{name: count(bucket == i) for i, name in enumerate(GAP_BINS)}}


def evaluate_fold(model, data, seed):
    """Reuse blackout_eval's exact masks and shared degraded players for each seed."""
    scenarios = conditions(data, seed)
    degraded_players = scenarios['current'][0]
    if any(players is not data['players'] and players is not degraded_players
           for players, _ in scenarios.values()):
        raise ValueError('new player condition requires its own predictions')
    clean = model.predict(player_features(data['players'], data['team'], data['half']))
    degraded = model.predict(player_features(degraded_players, data['team'], data['half']))
    return {name: gap_scores(data['label'], clean if players is data['players'] else degraded,
                             data['half'], ~np.isfinite(ball[:, :2]).all(axis=1))
            for name, (players, ball) in scenarios.items()}


def baseline_scores(data, seed):
    """Predeclared always-home baseline on exactly the same simulated gap masks."""
    pred = np.zeros(len(data['label']), dtype=np.int8)
    return {name: gap_scores(data['label'], pred, data['half'],
                             ~np.isfinite(ball[:, :2]).all(axis=1))
            for name, (_, ball) in conditions(data, seed).items()}


def pool_scores(rows):
    """Pool numerators/denominators rather than averaging match accuracies."""
    pooled = {}
    for row in rows:
        key = row['predictor'], row['condition'], row['scope']
        previous = pooled.get(key, {'correct': 0, 'total': 0})
        pooled[key] = {name: previous[name] + row[name] for name in ('correct', 'total')}
    return pooled


def report_scores(writer, match, seed, predictor, scores):
    rows = [{'predictor': predictor, 'condition': condition, 'scope': scope, **count}
            for condition, values in scores.items() for scope, count in values.items()]
    for row in rows:
        accuracy = row['correct'] / row['total'] if row['total'] else ''
        writer.writerow((match, seed, predictor, row['condition'], row['scope'],
                         row['correct'], row['total'], accuracy))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held-out', nargs='+', choices=MATCHES, default=list(MATCHES))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    start = time.monotonic()
    held_out = list(dict.fromkeys(args.held_out))
    seeds = list(dict.fromkeys(args.seeds))
    required = [m for m in MATCHES if any(m != held for held in held_out)]
    training = {}
    for match in required:
        training[match] = training_data(match)
        print(f'training features ready: {match} ({time.monotonic() - start:.0f}s)', file=sys.stderr, flush=True)
    print('EXPLORATORY simulated blackout H/A accuracy; no fusion or spellF1; '
          f'model_seed={SEED} mask_seeds={seeds}; pooled counts include repeated seeds',
          file=sys.stderr, flush=True)
    writer = csv.writer(sys.stdout)
    writer.writerow(('match', 'seed', 'predictor', 'condition', 'scope', 'correct', 'total', 'accuracy'))
    rows = []
    for match in held_out:
        model = fit_fold(training, match)
        data = load(match)
        for seed in seeds:
            rows.extend(report_scores(writer, match, seed, 'player_only', evaluate_fold(model, data, seed)))
            rows.extend(report_scores(writer, match, seed, 'always_H', baseline_scores(data, seed)))
            sys.stdout.flush()
        print(f'fold {match} finished ({time.monotonic() - start:.0f}s)', file=sys.stderr, flush=True)
    for (predictor, condition, scope), count in pool_scores(rows).items():
        accuracy = count['correct'] / count['total'] if count['total'] else ''
        writer.writerow(('pooled', 'all', predictor, condition, scope, count['correct'], count['total'], accuracy))


if __name__ == '__main__':
    main()
