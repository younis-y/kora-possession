"""Provisional blackout stress test of one held-out German match.

    OMP_NUM_THREADS=1 python -m kora.blackout_eval --held-out J03WPY --seeds 11 22 33
    OMP_NUM_THREADS=1 python -m kora.blackout_eval --held-out J03WPY --training-condition independent
    OMP_NUM_THREADS=1 python -m kora.blackout_eval --held-out J03WPY --duration-matched
    OMP_NUM_THREADS=1 python -m kora.blackout_eval --held-out J03WPY --nested-plain --duration-matched

The classifier is trained on six other matches with the existing degradation and
mirroring. The plain decoder uses fixed historical costs (H/A 8, D 32, alpha 0),
which were previously selected using all seven matches. Clean uses untouched
players; clean_ball, current, independent and live share degraded players. The
player degradation is label-conditioned. Scores are exploratory,
not the nested Y1 score and not an estimate for the challenge game. No model,
feature cache, submission or challenge data is written. Loading a missing source
match can download and cache its tracking data.

Independent training changes the ball mask but keeps label-conditioned player degradation.
Mixed training uses the current mask on even copies and the independent mask on odd copies.
The optional duration-matched scenario samples external (SkillCorner) gap-length quantiles and
missing share from models/degrade_external.json; it never reads the challenge match. Its placements ignore labels. The last sampled gap can be shortened; retained
source gaps may increase the realized missing rate or join gaps.

Nested mode accepts only the pinned Y1 summary and bundle and folds that selected the plain
decoder. It requires baseline decoded-path and alignment parity, which does not prove model
probability identity. Its stress scores are not a fresh seven-match nested evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import warnings

import numpy as np

from .blackout import apply_mask, duration_matched_mask, make_mask, shuffle_gaps, summarize
from .decode import decode
from .degrade import degrade
from .external import MATCHES, load
from .features import build, mirror
from .train import (FEATURE_VERSION, GRID_ALPHA, GRID_D, GRID_HA, MODELS, SEED,
                    SUBSAMPLE, TRAIN_COPIES, classifier, segments_of, spell_f1)
from .vendor import possession_metrics as pm

warnings.filterwarnings('ignore', category=RuntimeWarning)

CONDITION_NAMES = ('clean', 'clean_ball', 'current', 'independent', 'live', 'live_crossing',
                   'clean_players_current', 'clean_players_independent', 'shuffled_current',
                   'clean_players_shuffled_current', 'duration_matched',
                   'clean_players_duration_matched')
DURATION_CONDITIONS = ('duration_matched', 'clean_players_duration_matched')
TRAINING_CONDITIONS = ('current', 'independent', 'mixed')
Y1_SUMMARY_SHA256 = '65203af20938c8824ddd0a7ea3707508864db02b0860ef1ff9d0d137f24f39ff'
Y1_BUNDLE_SHA256 = 'd54d02b2dccf7b56f986747f913db88722286394d313155cdd81710fafa44535'


def nested_plain_settings(summary, meta, held_out):
    """Validate Y1 provenance and return its already-selected plain decoder settings."""
    expected = {'schema': 1, 'fps': 25, 'matches': list(MATCHES), 'dirty': False,
                'seed': SEED, 'feature_version': FEATURE_VERSION,
                'train_copies': TRAIN_COPIES, 'subsample': SUBSAMPLE}
    if any(summary.get(k) != v or meta.get(k) != v for k, v in expected.items()):
        raise ValueError('Y1 metadata is incompatible with this experiment')
    commit = summary.get('commit')
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit) or commit != meta.get('commit'):
        raise ValueError('Y1 metadata commits do not match')
    fold = summary.get('folds', {}).get(held_out, {})
    inner = fold.get('inner_matches', [])
    if len(inner) != len(MATCHES) - 1 or set(inner) != set(MATCHES) - {held_out}:
        raise ValueError('Y1 inner matches do not exclude the held-out match')
    if fold.get('decoder') != 'plain':
        raise ValueError('Y1 selected decoder must be plain; HSMM is not supported')
    plain = fold.get('plain', {})
    setting = (plain.get('alpha'), plain.get('cost_ha'), plain.get('cost_d'))
    if setting[0] not in GRID_ALPHA or setting[1] not in GRID_HA or setting[2] not in GRID_D:
        raise ValueError('invalid Y1 plain decoder settings')
    prior = np.asarray(fold.get('prior', []), dtype=float)
    if prior.shape != (3,) or not np.isfinite(prior).all() or (prior <= 0).any() or not np.isclose(prior.sum(), 1):
        raise ValueError('invalid Y1 prior')
    return prior, setting


def verify_reference_arrays(reference, actual):
    """Require exact held-out predictions and source alignment before stress scoring."""
    for key in ('y_true', 'y_pred', 'half', 'frame', 'ball_seen'):
        if key not in reference or key not in actual or not np.array_equal(reference[key], actual[key]):
            raise ValueError(f'Y1 parity failed for {key}')


def verify_y1_digest(path, expected):
    with path.open('rb') as source:
        actual = hashlib.file_digest(source, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError(f'Y1 artifact digest mismatch: {path.name}')


def load_y1(held_out):
    summary_path, bundle_path = MODELS / 'nested.json', MODELS / 'nested_predictions.npz'
    verify_y1_digest(summary_path, Y1_SUMMARY_SHA256)
    verify_y1_digest(bundle_path, Y1_BUNDLE_SHA256)
    summary = json.loads(summary_path.read_text())
    with np.load(bundle_path, allow_pickle=False) as bundle:
        meta = json.loads(str(bundle['meta']))
        prior, setting = nested_plain_settings(summary, meta, held_out)
        reference = {k: bundle[f'{held_out}__{k}'].copy()
                     for k in ('y_true', 'y_pred', 'half', 'frame', 'ball_seen')}
    sources = ['kora/train.py', 'kora/degrade.py', 'kora/features.py', 'kora/external.py',
               'kora/decode.py', 'kora/nested.py', 'kora/boundary.py', 'kora/data.py',
               'kora/vendor/possession_metrics.py']
    root = Path(__file__).resolve().parent.parent
    check = subprocess.run(['git', 'diff', '--quiet', meta['commit'], '--', *sources], cwd=root)
    if check.returncode:
        raise ValueError('Y1 source differs from recorded commit or commit is unavailable')
    return prior, setting, reference, meta['commit']


def verify_y1_model(model, data, held_out, prior, setting, reference):
    """Regenerate the exact evaluation copy used by Y1, including its RNG history."""
    rng = np.random.default_rng([SEED, list(MATCHES).index(held_out)])
    for _ in range(TRAIN_COPIES):
        degrade(data['players'], data['ball'], data['label'], rng)
    players, ball = degrade(data['players'], data['ball'], data['label'], rng)
    features = build(players, data['team'], ball, data['half'])
    pred = decode(model.predict_proba(features.to_numpy()), segments_of(data['half']), prior, *setting)
    actual = {'y_true': data['label'], 'y_pred': pred, 'half': data['half'],
              'frame': np.arange(len(pred)), 'ball_seen': features['b_seen'].to_numpy().astype(np.int8)}
    verify_reference_arrays(reference, actual)


def training_matches(held_out, matches):
    if held_out not in matches:
        raise ValueError('held-out match is not in the training set')
    return [match for match in matches if match != held_out]


def conditions(data, seed, duration_profile=None):
    labels, half, ball = data['label'], data['half'], data['ball']
    if not np.isfinite(ball[:, :2]).all():
        raise ValueError('shuffled gap control requires complete ball tracking')
    players, current_ball = degrade(data['players'], ball, labels, np.random.default_rng(seed))
    baseline = ~np.isfinite(current_ball).all(axis=1)
    original = ~np.isfinite(ball[:, :2]).all(axis=1)
    independent = original | make_mask(labels, half, 'independent', seed)
    live = make_mask(labels, half, 'live', seed, baseline)
    crossing = make_mask(labels, half, 'live_crossing', seed, baseline)
    shuffled_ball = apply_mask(ball[:, :2], shuffle_gaps(baseline, half, seed))
    result = {'clean': (data['players'], ball[:, :2].copy()),
            'clean_ball': (players, ball[:, :2].copy()),
            'current': (players, current_ball),
            'independent': (players, apply_mask(ball[:, :2], independent)),
            'live': (players, apply_mask(ball[:, :2], live)),
            'live_crossing': (players, apply_mask(ball[:, :2], crossing)),
            'clean_players_current': (data['players'], current_ball),
            'clean_players_independent': (data['players'], apply_mask(ball[:, :2], independent)),
            'shuffled_current': (players, shuffled_ball),
            'clean_players_shuffled_current': (data['players'], shuffled_ball)}
    if duration_profile is not None:
        lengths, share = duration_profile
        mask = duration_matched_mask(half, lengths, share, seed, original)
        duration_ball = apply_mask(ball[:, :2], mask)
        result['duration_matched'] = (players, duration_ball)
        result['clean_players_duration_matched'] = (data['players'], duration_ball)
    return result


def scores(y, pred, half):
    spell, macro = spell_f1(y, pred, half)
    cuts = np.flatnonzero(half[1:] != half[:-1]) + 1
    gt, pr = np.insert(y, cuts, pm.X), np.insert(pred, cuts, pm.X)
    per_iou = pm.spell_scores(gt, pr, pm.valid_runs(gt))['per_iou']
    return {'accuracy': float(np.mean(y == pred)), 'macro_f1': float(macro),
            'spell_f1': float(spell), 'spell_f1_by_iou': [v['f1'] for v in per_iou.values()]}


def training_data(held_out, condition='current', training_seed=SEED):
    if condition not in TRAINING_CONDITIONS:
        raise ValueError('unknown training condition')
    xs, ys = [], []
    for match in training_matches(held_out, MATCHES):
        data = load(match)
        rng = np.random.default_rng([training_seed, list(MATCHES).index(match)])
        for copy in range(TRAIN_COPIES):
            players, ball = degrade(data['players'], data['ball'], data['label'], rng)
            if condition == 'independent' or (condition == 'mixed' and copy % 2):
                mask = make_mask(data['label'], data['half'], 'independent',
                                 [training_seed, list(MATCHES).index(match), copy])
                ball = apply_mask(data['ball'][:, :2], mask)
            normal = build(players, data['team'], ball, data['half'])
            xs.append(normal.to_numpy()[::SUBSAMPLE].copy())
            ys.append(data['label'][::SUBSAMPLE].copy())
            del normal
            mp, mt, mb = mirror(players, data['team'], ball)
            mirrored = build(mp, mt, mb, data['half'])
            y = data['label'].copy()
            y[y < 2] = 1 - y[y < 2]
            xs.append(mirrored.to_numpy()[::SUBSAMPLE].copy())
            ys.append(y[::SUBSAMPLE].copy())
            del mirrored
        print(f'training features ready: {match}', flush=True)
    return np.concatenate(xs), np.concatenate(ys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held-out', choices=MATCHES, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    parser.add_argument('--training-condition', choices=TRAINING_CONDITIONS, default='current')
    parser.add_argument('--training-seed', type=int, default=SEED)
    parser.add_argument('--nested-plain', action='store_true',
                        help='use Y1 settings for a plain fold and require exact reference parity')
    parser.add_argument('--duration-matched', action='store_true',
                        help='sample gap lengths and missing share from models/degrade_external.json')
    parser.add_argument('--conditions', nargs='+', choices=CONDITION_NAMES,
                        help='evaluate only the named conditions')
    args = parser.parse_args()
    if args.conditions and any(c in DURATION_CONDITIONS for c in args.conditions) and not args.duration_matched:
        parser.error('duration-matched condition requires --duration-matched')
    if args.nested_plain and (args.training_condition != 'current' or args.training_seed != SEED):
        parser.error('--nested-plain requires current training and the Y1 training seed')
    nested = None
    if args.nested_plain:
        try:
            nested = load_y1(args.held_out)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            parser.error(str(error))
    start = time.monotonic()
    profile = None
    if args.duration_matched:
        ext = json.loads((MODELS / 'degrade_external.json').read_text())
        lengths = np.maximum(np.rint(np.asarray(ext['gap_s_quantiles']) * 25), 1).astype(np.int64)
        share = float(ext['missing_share'])
        profile = lengths, share
        print(f'external gap profile (SkillCorner quantiles): missing share {share:.4f}; '
              f'median {np.median(lengths) / 25:.2f}s; p90 {np.percentile(lengths, 90) / 25:.2f}s', flush=True)
    print(f'training condition: {args.training_condition}; training seed: {args.training_seed}', flush=True)
    X, y = training_data(args.held_out, args.training_condition, args.training_seed)
    prior = np.bincount(y, minlength=3) / len(y)
    model = classifier().fit(X, y)
    del X, y
    print(f'classifier fitted: {time.monotonic() - start:.0f}s; prior {prior.tolist()}', flush=True)
    data = load(args.held_out)
    labels, half = data['label'], data['half']
    if nested:
        prior, setting, reference, commit = nested
        verify_y1_model(model, data, args.held_out, prior, setting, reference)
        print(f'Y1 baseline decoded path and alignment parity verified; fold {args.held_out}; commit {commit}; '
              f'plain settings alpha={setting[0]} cost_ha={setting[1]} cost_d={setting[2]}', flush=True)
    else:
        setting = (0., 8., 32.)
        print('PROVISIONAL fixed plain decoder alpha=0 cost_ha=8 cost_d=32; historical selection used all matches')
    print('match seed condition spell_f1 macro_f1 accuracy live_missing/total dead_missing/total gap_median_s gap_p90_s f1_at_0.3_0.4_0.5_0.6_0.7')
    for seed in args.seeds:
        for name, (players, ball) in conditions(data, seed, profile).items():
            if args.conditions and name not in args.conditions:
                continue
            X = build(players, data['team'], ball, half)
            proba = model.predict_proba(X.to_numpy())
            pred = decode(proba, segments_of(half), prior, *setting)
            result = scores(labels, pred, half)
            missing = ~np.isfinite(ball[:, :2]).all(axis=1)
            s = summarize(missing, labels, half)
            lengths = np.asarray(s['gap_frames']) / 25
            median, p90 = np.percentile(lengths, [50, 90]) if len(lengths) else (0, 0)
            print(f"{args.held_out} {seed} {name} {result['spell_f1']:.4f} "
                  f"{result['macro_f1']:.4f} {result['accuracy']:.4f} "
                  f"{s['live_missing']}/{s['live_total']} {s['dead_missing']}/{s['dead_total']} "
                  f"{median:.2f} {p90:.2f} " + ','.join(f'{x:.4f}' for x in result['spell_f1_by_iou']), flush=True)


if __name__ == '__main__':
    main()
