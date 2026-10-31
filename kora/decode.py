"""Hybrid HMM decoding: classifier posteriors as emissions, switching costs as transitions.

A frame classifier gives p(state | features) for H, A, D. Consecutive frames are far from
independent, so a raw per-frame argmax flickers and invents possessions; the ranking metric charges
for every invented one. Viterbi finds the single label path that maximises

    sum_t [ log p(s_t | x_t) - alpha * log prior(s_t) ]  -  sum of switching costs,

with one cost for a change of possession (H<->A) and one for entering or leaving dead ball. The
costs and alpha are chosen by leave-one-match-out cross-validation on the training matches.
"""
import numpy as np


def viterbi(logp, cost_ha, cost_d):
    n = len(logp)
    C = np.array([[0.0, cost_ha, cost_d],
                  [cost_ha, 0.0, cost_d],
                  [cost_d, cost_d, 0.0]])
    score = logp[0].astype(np.float64).copy()
    back = np.empty((n, 3), np.int8)
    back[0] = np.arange(3)
    for t in range(1, n):
        cand = score[:, None] - C
        prev = cand.argmax(0)
        back[t] = prev
        score = cand[prev, (0, 1, 2)] + logp[t]
    path = np.empty(n, np.int8)
    s = int(score.argmax())
    for t in range(n - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return path


def emissions(proba, prior, alpha):
    p = np.clip(proba, 1e-6, 1.0)
    return np.log(p) - alpha * np.log(prior)[None, :]


def decode(proba, segments, prior, alpha, cost_ha, cost_d):
    """Decode each [start, stop) segment (e.g. a half) independently."""
    out = np.full(len(proba), -1, np.int8)
    e = emissions(proba, prior, alpha)
    for a, b in segments:
        out[a:b] = viterbi(e[a:b], cost_ha, cost_d)
    return out
