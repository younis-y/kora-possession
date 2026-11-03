"""Explore ball blackouts on labelled training matches, without changing the model.

    python -m kora.blackout --matches J03WPY --seeds 11 22 33

Independent gaps cover about 40% of frames; live stress adds gaps on about 25% of
live frames to the current degradation. These are stress scenarios, not estimates
of missingness in the challenge game. Use the same seed and mask for comparisons.
"""
import argparse

import numpy as np

from .degrade import FPS, degrade, stretches
from .external import MATCHES, load


def _random_gaps(n, share, rng):
    """Alternating geometric gaps and visible spans, with mean gap of 2 seconds."""
    mean_gap = 2 * FPS
    mean_visible = mean_gap * (1 - share) / share
    missing = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        i += rng.geometric(1 / mean_visible)
        end = min(i + rng.geometric(1 / mean_gap), n)
        missing[i:end] = True
        i = end
    return missing


def gap_profile(ball, blocks):
    """Pool gap lengths and missing share from selected, separate tracking blocks."""
    ball = np.asarray(ball)
    if ball.ndim != 2 or ball.shape[1] < 2:
        raise ValueError('ball must have at least two coordinates')
    lengths, missing, total = [], 0, 0
    last = 0
    for start, end in blocks:
        if not (0 <= start < end <= len(ball)) or start < last:
            raise ValueError('blocks must be ordered, non-overlapping and in range')
        mask = ~np.isfinite(ball[start:end, :2]).all(axis=1)
        lengths.extend(b - a for a, b in stretches(mask))
        missing += int(mask.sum())
        total += end - start
        last = end
    if not total or not 0 < missing < total:
        raise ValueError('selected blocks require both visible and missing ball frames')
    return np.asarray(lengths, dtype=np.int64), missing / total


def duration_matched_mask(half, lengths, share, seed, baseline=None):
    """Sample gap lengths without labels; round the share per half and retain baseline gaps.

    The last sampled gap may be shortened to meet the missing-frame budget. Baseline gaps
    can increase the realized share or join simulated gaps.
    """
    half, lengths = np.asarray(half), np.asarray(lengths)
    if half.ndim != 1 or lengths.ndim != 1 or not len(lengths) or not np.issubdtype(lengths.dtype, np.integer) or (lengths <= 0).any():
        raise ValueError('half must be a vector and lengths must be positive integer frames')
    if not np.isfinite(share) or not 0 < share < 1:
        raise ValueError('missing share must be between zero and one')
    if baseline is not None and np.asarray(baseline).shape != half.shape:
        raise ValueError('baseline mask must match half')
    rng = np.random.default_rng([seed, 422027])
    mask = np.zeros(len(half), dtype=bool)
    edges = np.r_[0, np.flatnonzero(half[1:] != half[:-1]) + 1, len(half)]
    for start, end in zip(edges[:-1], edges[1:]):
        missing = int(round((end - start) * share))
        visible = end - start - missing
        remaining, gaps = missing, []
        while remaining and len(gaps) <= visible:
            length = min(int(rng.choice(lengths)), remaining)
            gaps.append(length)
            remaining -= length
        if remaining:
            raise ValueError('not enough visible frames to separate sampled gaps')
        if not gaps:
            continue
        minimum = np.r_[0, np.ones(len(gaps) - 1, dtype=int), 0]
        extra = visible - len(gaps) + 1
        spacings = minimum + rng.multinomial(extra, rng.dirichlet(np.ones(len(gaps) + 1)))
        pos = start + spacings[0]
        for length, spacing in zip(gaps, spacings[1:]):
            mask[pos:pos + length] = True
            pos += length + spacing
    return mask if baseline is None else mask | np.asarray(baseline, dtype=bool)


def make_mask(labels, half, condition, seed, baseline=None):
    """Return a frame mask; 'live' adds gaps to the supplied current mask."""
    labels = np.asarray(labels)
    half = np.asarray(half)
    if labels.ndim != 1 or half.shape != labels.shape:
        raise ValueError('labels and half must be one-dimensional and equally sized')
    if condition not in ('independent', 'live', 'live_crossing'):
        raise ValueError(f'unknown blackout condition: {condition}')
    if condition in ('live', 'live_crossing') and baseline is None:
        raise ValueError('live stress requires a baseline mask')
    if baseline is not None and np.asarray(baseline).shape != labels.shape:
        raise ValueError('baseline mask must match labels')
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    edges = np.r_[0, np.flatnonzero(half[1:] != half[:-1]) + 1, len(half)]
    for start, end in zip(edges[:-1], edges[1:]):
        mask[start:end] = _random_gaps(end - start, 0.4 if condition == 'independent' else 0.25, rng)
        if condition == 'live_crossing':
            for a, b in stretches(mask[start:end]):
                if labels[start + a] == 2:
                    mask[start + a:start + b] = False
    if condition == 'live':
        mask = (mask & (labels != 2)) | np.asarray(baseline, dtype=bool)
    elif condition == 'live_crossing':
        mask = mask | np.asarray(baseline, dtype=bool)
    return mask


def apply_mask(ball, mask):
    result = np.asarray(ball).copy()
    result[np.asarray(mask, dtype=bool)] = np.nan
    return result


def shuffle_gaps(mask, half, seed):
    """Move gaps within each half while preserving each gap and visible-run length."""
    mask, half = np.asarray(mask, dtype=bool), np.asarray(half)
    if mask.ndim != 1 or half.shape != mask.shape:
        raise ValueError('mask and half must be equally sized vectors')
    rng = np.random.default_rng(seed)
    result = np.zeros(len(mask), dtype=bool)
    edges = np.r_[0, np.flatnonzero(half[1:] != half[:-1]) + 1, len(half)]
    for start, end in zip(edges[:-1], edges[1:]):
        gaps = stretches(mask[start:end])
        if not gaps:
            continue
        lengths = rng.permutation([b - a for a, b in gaps])
        visible = rng.permutation([gaps[i + 1][0] - gaps[i][1] for i in range(len(gaps) - 1)])
        margins = rng.permutation([gaps[0][0], end - start - gaps[-1][1]])
        pos = start + margins[0]
        for i, length in enumerate(lengths):
            result[pos:pos + length] = True
            pos += length + (visible[i] if i < len(visible) else 0)
    return result


def summarize(mask, labels, half):
    """Count missing live/dead frames and gaps, splitting gaps at half boundaries."""
    mask, labels, half = np.asarray(mask, dtype=bool), np.asarray(labels), np.asarray(half)
    if mask.ndim != 1 or labels.shape != mask.shape or half.shape != mask.shape:
        raise ValueError('mask, labels and half must be one-dimensional and equally sized')
    live = labels != 2
    edges = np.r_[0, np.flatnonzero(half[1:] != half[:-1]) + 1, len(half)]
    lengths = [end - start for a, b in zip(edges[:-1], edges[1:])
               for start, end in stretches(mask[a:b])]
    return {'live_missing': int((mask & live).sum()), 'live_total': int(live.sum()),
            'dead_missing': int((mask & ~live).sum()), 'dead_total': int((~live).sum()),
            'gap_frames': lengths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', nargs='+', choices=MATCHES, default=list(MATCHES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    args = parser.parse_args()
    print('match seed condition missing/total live missing/total dead gaps median_s p90_s')
    for match in args.matches:
        data = load(match)
        labels, half, ball = data['label'], data['half'], data['ball']
        originally_missing = ~np.isfinite(ball[:, :2]).all(axis=1)
        for seed in args.seeds:
            _, degraded_ball = degrade(data['players'], ball, labels, np.random.default_rng(seed))
            baseline = ~np.isfinite(degraded_ball).all(axis=1)
            conditions = {
                'clean': originally_missing,
                'current': baseline,
                'independent': originally_missing | make_mask(labels, half, 'independent', seed),
                'live': make_mask(labels, half, 'live', seed, baseline),
                'live_crossing': make_mask(labels, half, 'live_crossing', seed, baseline),
            }
            for name, mask in conditions.items():
                s = summarize(mask, labels, half)
                lengths = np.array(s['gap_frames']) / FPS
                median, p90 = np.percentile(lengths, [50, 90]) if len(lengths) else (0, 0)
                print(f"{match} {seed} {name} {s['live_missing']}/{s['live_total']} "
                      f"{s['dead_missing']}/{s['dead_total']} {len(lengths)} {median:.2f} {p90:.2f}")


if __name__ == '__main__':
    main()
