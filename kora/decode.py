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


STRIDE = 3                      # the segment decoder works on 3-frame blocks (0.12 s)
DMAX = 600                      # longest single segment, in blocks (72 s); longer ones chain


def log_duration(runs):
    """log P(duration) per state, in blocks, from (label, frames) runs; H and A share one prior.

    Kernel density in log-duration, because possession lengths span 0.04 s to minutes."""
    d = np.arange(1, DMAX + 1)
    out = np.zeros((3, DMAX))
    for k, keep in ((0, (0, 1)), (2, (2,))):
        x = np.log(np.maximum([n / STRIDE for lab, n in runs if lab in keep], 0.2))
        dens = np.exp(-0.5 * ((np.log(d)[:, None] - x[None, :]) / 0.25) ** 2).sum(1) / d
        out[k] = np.log(dens / dens.sum())
    out[1] = out[0]
    return out


def hsmm(e, ld, cost_ha, cost_d, w, bonus):
    """Semi-Markov Viterbi over blocks. A segment of state k on blocks [j, t) scores
    sum(e[j:t, k]) + w * ld[k, t-j] + bonus[j, k], minus the switching cost from the previous
    segment's state. Same-state segments may follow each other (paying the cost) so possessions
    longer than DMAX blocks stay possible."""
    T = len(e)
    S = np.vstack([np.zeros(3), np.cumsum(e, 0)])
    C = np.array([[cost_ha, cost_ha, cost_d], [cost_ha, cost_ha, cost_d], [cost_d, cost_d, cost_d]], float)
    ldw = (w * ld).T
    Vs = np.full((T + 1, 3), -np.inf); Vs[0] = 0          # best score of a path ending at j, entering state k
    prev = np.zeros((T + 1, 3), np.int64)
    dur = np.zeros((T + 1, 3), np.int64)
    V = np.full((T + 1, 3), -np.inf)
    for t in range(1, T + 1):
        D = min(DMAX, t)
        j = t - np.arange(1, D + 1)
        cand = Vs[j] - S[j] + ldw[:D] + bonus[j]
        i = cand.argmax(0)
        V[t] = cand[i, (0, 1, 2)] + S[t]
        dur[t] = i + 1
        if t < T:
            M = V[t][:, None] - C
            pk = M.argmax(0)
            Vs[t] = M[pk, (0, 1, 2)]
            prev[t] = pk
    path = np.empty(T, np.int8)
    t, k = T, int(V[T].argmax())
    while t > 0:
        d = dur[t, k]
        path[t - d:t] = k
        t -= d
        if t > 0:
            k = prev[t, k]
    return path


def decode_hsmm(proba, change, segments, prior, ld, cost_ha, cost_d, w, beta, rate):
    """Segment decoding of each [start, stop); change = detector P(H<->A change nearby) per frame,
    rate = its mean on the training matches. Its log-ratio is a bonus for starting a possession."""
    out = np.full(len(proba), -1, np.int8)
    e = emissions(proba, prior, 0.0)
    for a, z in segments:
        n = (z - a) // STRIDE * STRIDE
        eb = e[a:a + n].reshape(-1, STRIDE, 3).sum(1)
        cb = change[a:a + n].reshape(-1, STRIDE).max(1)
        bonus = np.zeros((len(eb) + 1, 3))
        bonus[:-1, :2] = beta * np.log(np.clip(cb, 1e-4, 1.0) / rate)[:, None]
        p = hsmm(eb, ld, cost_ha, cost_d, w, bonus)
        out[a:a + n] = np.repeat(p, STRIDE)
        out[a + n:z] = p[-1]
    return out
