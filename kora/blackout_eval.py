"""Provisional blackout stress test of one held-out German match.

    OMP_NUM_THREADS=1 python -m kora.blackout_eval --held-out J03WPY --seeds 11 22 33
    OMP_NUM_THREADS=1 python -m kora.blackout_eval --held-out J03WPY --training-condition independent
    OMP_NUM_THREADS=1 python -m kora.blackout_eval --held-out J03WPY --duration-matched

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
missing share from models/degrade_external.json; it never reads the challenge match. Its
placements ignore labels. The last sampled gap can be shortened; retained source gaps may
increase the realized missing rate or join gaps.
"""
import argparse
import json
import time
import warnings

import numpy as np

from .blackout import apply_mask, duration_matched_mask, make_mask, shuffle_gaps, summarize
from .decode import decode
from .degrade import degrade
from .external import MATCHES, load
from .features import build, mirror
from .train import MODELS, SEED, SUBSAMPLE, TRAIN_COPIES, classifier, segments_of, spell_f1
from .vendor import possession_metrics as pm

warnings.filterwarnings('ignore', category=RuntimeWarning)

CONDITION_NAMES = ('clean', 'clean_ball', 'current', 'independent', 'live', 'live_crossing',
                   'clean_players_current', 'clean_players_independent', 'shuffled_current',
                   'clean_players_shuffled_current', 'duration_matched',
                   'clean_players_duration_matched')
DURATION_CONDITIONS = ('duration_matched', 'clean_players_duration_matched')
TRAINING_CONDITIONS = ('current', 'independent', 'mixed')
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
    parser.add_argument('--duration-matched', action='store_true',
                        help='sample gap lengths and missing share from models/degrade_external.json')
    parser.add_argument('--conditions', nargs='+', choices=CONDITION_NAMES,
                        help='evaluate only the named conditions')
    args = parser.parse_args()
    if args.conditions and any(c in DURATION_CONDITIONS for c in args.conditions) and not args.duration_matched:
        parser.error('duration-matched condition requires --duration-matched')
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
    setting = (0., 8., 32.)
    print('PROVISIONAL fixed plain decoder alpha=0 cost_ha=8 cost_d=32; historical selection used all matches')
    print('match seed condition spell_f1 macro_f1 accuracy live_missing/total dead_missing/total gap_median_s gap_p90_s f1_at_0.3_0.4_0.5_0.6_0.7 gap_team_correct/total')
    for seed in args.seeds:
        for name, (players, ball) in conditions(data, seed, profile).items():
            if args.conditions and name not in args.conditions:
                continue
            X = build(players, data['team'], ball, half)
            proba = model.predict_proba(X.to_numpy())
            pred = decode(proba, segments_of(half), prior, *setting)
            result = scores(labels, pred, half)
            missing = ~np.isfinite(ball[:, :2]).all(axis=1)
            gap = missing & (labels < 2)
            s = summarize(missing, labels, half)
            lengths = np.asarray(s['gap_frames']) / 25
            median, p90 = np.percentile(lengths, [50, 90]) if len(lengths) else (0, 0)
            print(f"{args.held_out} {seed} {name} {result['spell_f1']:.4f} "
                  f"{result['macro_f1']:.4f} {result['accuracy']:.4f} "
                  f"{s['live_missing']}/{s['live_total']} {s['dead_missing']}/{s['dead_total']} "
                  f"{median:.2f} {p90:.2f} " + ','.join(f'{x:.4f}' for x in result['spell_f1_by_iou']) +
                  f" {int((proba[gap, :2].argmax(1) == labels[gap]).sum())}/{int(gap.sum())}", flush=True)


if __name__ == '__main__':
    main()
