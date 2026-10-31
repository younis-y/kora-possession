"""The challenge match in the same representation as the training matches."""
import numpy as np

from .data import HALVES, teams, tracking

PITCH = (105.3, 68.0)        # games.py: measured from the corrected tracking
HALFTIME_SPLIT = 100730      # between the two graded blocks; orientation is fitted per side of it


def load_target():
    home, away, side = teams()
    _, P, B = tracking()
    players = np.empty(P.shape[:2] + (2,), np.float32)
    players[..., 0] = (P[..., 0] - 0.5) * PITCH[0]
    players[..., 1] = (P[..., 1] - 0.5) * PITCH[1]
    ball = np.empty((len(B), 2), np.float32)
    ball[:, 0] = (B[:, 0] - 0.5) * PITCH[0]
    ball[:, 1] = (B[:, 1] - 0.5) * PITCH[1]
    section = np.where(np.arange(len(B)) < HALFTIME_SPLIT, 1, 2).astype(np.int8)
    graded = [(a, b + 1) for a, b in HALVES.values()]
    return players, side.astype(np.int8), ball, section, graded
