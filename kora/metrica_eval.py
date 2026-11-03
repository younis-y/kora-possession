"""Local research: sparse Game 3 CARRY supervision for the player-only H/A model.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m kora.metrica_eval

Predeclared comparison: outer test J03WPY, validation J03WR9 (the complete
German match), mask seed 11, classifier seed 2027,
German H/A row weight 1 and auxiliary H/A row weight 0.25. German training uses
player_eval's fixed degradation/mirroring; Metrica uses clean players and mirrored
copies, with the same original-frame stride before filtering sparse anchors.
Unknown/conflicted anchors never become targets. Inclusive event endpoints are
an unverified convention, not corrected-tracking possession ground truth.

Only pinned local Metrica inputs and existing German caches are read. No model,
feature cache, submission or challenge label is written or read. Metrica reuse
terms remain unresolved: local research only; no released training artifacts.
The existing extractor uses future context and can mix context across halves;
training degradation is label-conditioned. Both fits receive identical explicit
German-only validation rows and unit weights; Metrica rows enter training only.
Five German matches train, one validates, and one remains the outer test.
This is a one-match feasibility check, not fusion, nested selection or spellF1.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .blackout_eval import conditions
from .data import data_dir
from .external import MATCHES, load
from .features import mirror
from .metrica_anchors import DEFAULT_EVENTS, EVENTS_SHA256, HALVES, REVISION, carry_anchors
from .metrica_tracking import DEFAULT_METADATA, DEFAULT_TRACKING, load_metrica_tracking
from .player_eval import gap_scores, report_scores, training_data
from .player_only import PLAYER_FEATURES, player_features
from .train import SEED, SUBSAMPLE, classifier

AUXILIARY_WEIGHT = 0.25
VALIDATION_MATCH = 'J03WR9'


def auxiliary_training(tracking, events):
    """Align 1-based frame IDs before stride/filtering; return fresh H/A rows only."""
    n = len(tracking.players)
    if (tracking.fps != 25 or not np.array_equal(tracking.frame, np.arange(1, n + 1))
            or not np.array_equal(tracking.section, np.where(tracking.frame <= HALVES[1][1], 1, 2))):
        raise ValueError('Expected contiguous 1-based 25 FPS frames with Game 3 half alignment')
    anchors, counts = carry_anchors(events, n_frames=n)
    indices = np.arange(0, n, SUBSAMPLE)
    selected = indices[np.isin(anchors[indices], ('H', 'A'))]
    labels = (anchors[selected] == 'A').astype(np.int8)
    normal = player_features(tracking.players, tracking.team, tracking.section)
    players, team, _ = mirror(tracking.players, tracking.team, np.full((n, 2), np.nan))
    mirrored = player_features(players, team, tracking.section)
    features = pd.concat([normal.iloc[selected], mirrored.iloc[selected]], ignore_index=True)
    return (features, np.r_[labels, 1 - labels],
            {**counts, 'sampled_rows_with_mirror': len(features)})


def load_auxiliary(events=DEFAULT_EVENTS, metadata=DEFAULT_METADATA, tracking=DEFAULT_TRACKING):
    """Verify all three pinned local sources before constructing auxiliary targets."""
    raw = Path(events).read_bytes()
    if hashlib.sha256(raw).hexdigest() != EVENTS_SHA256:
        raise ValueError('SHA-256 does not match the pinned Game 3 events')
    match = load_metrica_tracking(metadata, tracking)
    return auxiliary_training(match, json.loads(raw)['data'])


def training_arrays(training, held_out, auxiliary, auxiliary_weight):
    """Partition explicitly, drop German D rows and apply fixed auxiliary weights."""
    if held_out not in MATCHES and held_out not in training:
        raise ValueError('unknown held-out match')
    if not np.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ValueError('auxiliary weight must be finite and nonnegative')
    others = [arrays for match, arrays in training.items() if match != held_out]
    if not others:
        raise ValueError('no other training matches')
    features = pd.concat([x for x, _ in others], ignore_index=True)
    labels = np.concatenate([y for _, y in others])
    if len(features) != len(labels) or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError('Expected aligned German H/A/D training labels')
    live = labels < 2
    if not np.array_equal(np.unique(labels[live]), [0, 1]):
        raise ValueError('German training must contain both teams')
    features, labels = features.loc[live, PLAYER_FEATURES].copy(), labels[live].copy()
    weights = np.ones(len(labels))
    ax, ay = auxiliary
    if len(ax) != len(ay) or not np.isin(ay, (0, 1)).all():
        raise ValueError('Expected aligned auxiliary H/A labels only')
    if auxiliary_weight:
        features = pd.concat([features, ax.loc[:, PLAYER_FEATURES]], ignore_index=True)
        labels = np.r_[labels, ay]
        weights = np.r_[weights, np.full(len(ay), auxiliary_weight)]
    return features, labels, weights


def validation_partition(training, held_out):
    """Reserve the predeclared whole German match before auxiliary rows enter."""
    if held_out == VALIDATION_MATCH or VALIDATION_MATCH not in training:
        raise ValueError('Expected a separate J03WR9 validation match')
    remaining = {match: arrays for match, arrays in training.items()
                 if match not in (held_out, VALIDATION_MATCH)}
    if not remaining:
        raise ValueError('no German fitting matches after validation exclusion')
    empty = (pd.DataFrame(columns=PLAYER_FEATURES), np.array([], dtype=np.int8))
    validation = training_arrays({VALIDATION_MATCH: training[VALIDATION_MATCH]}, held_out, empty, 0.)
    return remaining, validation


def fit_fold(training, held_out, auxiliary, auxiliary_weight):
    """Use the same explicit German validation match with either auxiliary weight."""
    fitting, (x_val, y_val, w_val) = validation_partition(training, held_out)
    features, labels, weights = training_arrays(fitting, held_out, auxiliary, auxiliary_weight)
    return classifier().fit(features, labels, sample_weight=weights,
                            X_val=x_val, y_val=y_val, sample_weight_val=w_val)


def evaluate_models(models, data, seed):
    """Compute conditions and player features once for the paired comparison."""
    scenarios = conditions(data, seed)
    degraded = scenarios['current'][0]
    if any(players is not data['players'] and players is not degraded
           for players, _ in scenarios.values()):
        raise ValueError('new player condition requires its own predictions')
    clean_features = player_features(data['players'], data['team'], data['half'])
    degraded_features = player_features(degraded, data['team'], data['half'])
    result = {}
    for name, model in models.items():
        clean_pred, degraded_pred = model.predict(clean_features), model.predict(degraded_features)
        result[name] = {condition: gap_scores(
            data['label'], clean_pred if players is data['players'] else degraded_pred,
            data['half'], ~np.isfinite(ball[:, :2]).all(axis=1))
            for condition, (players, ball) in scenarios.items()}
    return result


def require_cached_matches():
    """Prevent the existing German loader from downloading or creating caches."""
    missing = [match for match in MATCHES
               if not (data_dir() / 'cache' / f'idsse_{match}.npz').is_file()]
    if missing:
        raise ValueError(f'Existing local German caches required: {", ".join(missing)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held-out', choices=[m for m in MATCHES if m != VALIDATION_MATCH], default='J03WPY')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--events', type=Path, default=DEFAULT_EVENTS)
    parser.add_argument('--metadata', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--tracking', type=Path, default=DEFAULT_TRACKING)
    args = parser.parse_args()
    start = time.monotonic()
    try:
        require_cached_matches()
        ax, ay, counts = load_auxiliary(args.events, args.metadata, args.tracking)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({'research_only': True, 'source_revision': REVISION,
                      'held_out': args.held_out, 'mask_seed': args.seed, 'model_seed': SEED,
                      'validation_match': VALIDATION_MATCH, 'validation_row_weight': 1.,
                      'auxiliary_row_weight': AUXILIARY_WEIGHT, 'counts': counts}),
          file=sys.stderr, flush=True)
    training = {}
    for match in MATCHES:
        if match != args.held_out:
            training[match] = training_data(match)
            print(f'training features ready: {match} ({time.monotonic() - start:.0f}s)',
                  file=sys.stderr, flush=True)
    german_rows = sum(int((y < 2).sum()) for match, (_, y) in training.items() if match != VALIDATION_MATCH)
    validation_rows = int((training[VALIDATION_MATCH][1] < 2).sum())
    print(f'German fitting live rows={german_rows}; validation {VALIDATION_MATCH} rows={validation_rows}; '
          f'auxiliary rows={len(ay)}; '
          f'auxiliary weight mass={len(ay) * AUXILIARY_WEIGHT:g}', file=sys.stderr, flush=True)
    models = {}
    for name, weight in (('sportec_only', 0.), ('sportec_metrica_0.25', AUXILIARY_WEIGHT)):
        models[name] = fit_fold(training, args.held_out, (ax, ay), weight)
        print(f'{name}: {models[name].n_iter_} iterations ({time.monotonic() - start:.0f}s)',
              file=sys.stderr, flush=True)
    scores = evaluate_models(models, load(args.held_out), args.seed)
    writer = csv.writer(sys.stdout)
    writer.writerow(('match', 'seed', 'predictor', 'condition', 'scope', 'correct', 'total', 'accuracy'))
    for name, values in scores.items():
        report_scores(writer, args.held_out, args.seed, name, values)
    print(f'finished ({time.monotonic() - start:.0f}s); no model saved', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
