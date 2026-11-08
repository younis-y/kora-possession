"""Y26: decode.py's plain Viterbi with a dearer dead-ball switch on frames where the ball is missing.

Under heavy ball gaps stage 1 reads a missing ball as a stoppage, and the decoder picked on such copies
raises cost_d everywhere (RESEARCH.md Y26: 8-16 on normal copies, 32-128 on heavy-gap copies). Here only a
switch into or out of D on a frame without the ball pays `factor` x cost_d, so a stoppage with the ball
seen keeps its usual cost. factor 1 is decode.decode exactly. decode.py is left unchanged.
"""
import numpy as np

from .decode import emissions


def viterbi(logp, cost_ha, cost_d, missing, factor):
    def costs(cd):
        return np.array([[0.0, cost_ha, cd], [cost_ha, 0.0, cd], [cd, cd, 0.0]])
    C = (costs(cost_d), costs(cost_d * factor))
    n = len(logp)
    score = logp[0].astype(np.float64).copy()
    back = np.empty((n, 3), np.int8)
    back[0] = np.arange(3)
    for t in range(1, n):
        cand = score[:, None] - C[bool(missing[t])]
        prev = cand.argmax(0)
        back[t] = prev
        score = cand[prev, (0, 1, 2)] + logp[t]
    path = np.empty(n, np.int8)
    s = int(score.argmax())
    for t in range(n - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return path


def decode(proba, segments, prior, alpha, cost_ha, cost_d, factor, missing):
    """decode.decode with `factor` x cost_d on the frames where `missing` is true."""
    out = np.full(len(proba), -1, np.int8)
    e = emissions(proba, prior, alpha)
    for a, b in segments:
        out[a:b] = viterbi(e[a:b], cost_ha, cost_d, missing[a:b], factor)
    return out
