"""Experimental H/A model using player tracking only.

Train and select any fusion rule inside the training partition of each evaluation fold.
"""
import numpy as np

from .features import build
from .train import classifier


PLAYER_FEATURES = (
    'nH', 'nA', 'cxH', 'cxA', 'sxH', 'sxA', 'syH', 'syA',
    'lenH', 'lenA', 'vxH', 'vxA', 'spH', 'spA', 'runH', 'runA',
    'sp_med', 'sp_max', 'cx_diff', 'vx_diff', 'sp_diff', 'sx_diff', 'sy_diff',
    *(f'{name}_c{window}' for name in ('sp_med', 'vx_diff', 'cx_diff', 'sp_diff', 'sx_diff')
      for window in (13, 25, 51, 101, 201, 401)),
    *(f'{name}_{mode}{window}' for name in ('sp_med', 'vx_diff', 'cx_diff')
      for window in (25, 75) for mode in ('p', 'f', 'j')),
)


def player_features(players, team, section):
    """Extract player-only features; the extractor never receives ball tracking."""
    empty_ball = np.full((len(players), 2), np.nan, dtype=np.float32)
    return build(players, team, empty_ball, section).loc[:, PLAYER_FEATURES]


def fit_player_model(features, labels):
    """Fit on H/A frames from training matches supplied by the caller."""
    labels = np.asarray(labels)
    if len(features) != len(labels) or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError('Expected one H/A/D label per feature row')
    live = labels < 2
    if not np.array_equal(np.unique(labels[live]), [0, 1]):
        raise ValueError('Training data must contain both teams')
    return classifier().fit(features.loc[live, PLAYER_FEATURES], labels[live])
