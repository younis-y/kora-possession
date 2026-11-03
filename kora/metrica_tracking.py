"""Research-only player tracking adapter for pinned Metrica Sample Game 3.

Run ``python -m kora.metrica_tracking`` after obtaining the two pinned files locally.
No download, cache, export, training, inference, or possession labels are produced.
The source requests acknowledgement; a standard reuse licence is unresolved.

This reads the register schema embedded in Metrica's EPTS XML, not a general EPTS
XSD implementation. Each time-dependent layout resolves channels to stable roster
identities. Coordinates keep the source top-left axis directions, translated to
centre-origin metres as expected by ``player_only.player_features``. Finite off-pitch
coordinates are valid and never clipped; missing coordinates remain NaN.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .metrica_anchors import DEFAULT_EVENTS, REVISION

BASE_URL = (f'https://raw.githubusercontent.com/metrica-sports/sample-data/{REVISION}/'
            'data/Sample_Game_3/')
METADATA_SHA256 = '009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1'
TRACKING_SHA256 = '570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658'
DEFAULT_METADATA = DEFAULT_EVENTS.with_name('Sample_Game_3_metadata.xml')
DEFAULT_TRACKING = DEFAULT_EVENTS.with_name('Sample_Game_3_tracking.txt')


@dataclass(frozen=True)
class Layout:
    first: int
    last: int
    player_indices: tuple
    axes: tuple


@dataclass(frozen=True)
class Metadata:
    player_ids: tuple
    team: tuple
    pitch: tuple
    fps: int
    halves: tuple
    layouts: tuple


@dataclass(frozen=True)
class PlayerTracking:
    """Read-only arrays: row i is source frame i+1; halves are inclusive, 1-based.

    ``players[n, roster, 2]``, ``team[roster]`` (0 local/home, 1 visiting/away),
    and ``section[n]`` can be passed directly to ``player_features``. Inactive
    substitutes have NaN coordinates. Ball channels are validated but not returned.
    """
    frame: np.ndarray
    players: np.ndarray
    team: np.ndarray
    section: np.ndarray
    player_ids: tuple
    halves: tuple
    pitch: tuple
    fps: int


def _integer(text):
    if not isinstance(text, str) or not text.isascii() or not text.isdecimal():
        raise ValueError(f'Expected a nonnegative integer, got {text!r}')
    return int(text)


def _periods(root):
    parameters = root.findall('Metadata/GlobalConfig/ProviderGlobalParameters/ProviderParameter')
    values = {p.findtext('Name'): p.findtext('Value') for p in parameters}
    if len(values) != len(parameters):
        raise ValueError('Duplicate provider parameters')
    halves = tuple((_integer(values[f'{half}_half_start']),
                    _integer(values[f'{half}_half_end'])) for half in ('first', 'second'))
    if (halves[0][0] != 1 or halves[0][1] < 1
            or halves[1][0] != halves[0][1] + 1 or halves[1][1] < halves[1][0]):
        raise ValueError('Half ranges must be positive, ordered and contiguous')
    if any(value for key, value in values.items() if key and 'extra_half' in key):
        raise ValueError('Extra periods are unsupported for Game 3')
    return halves


def _roster(root):
    scores = root.findall('Metadata/Sessions/Session/MatchParameters/Score')
    if len(scores) != 1:
        raise ValueError('Expected one match score with local and visiting team IDs')
    teams = (scores[0].attrib['idLocalTeam'], scores[0].attrib['idVisitingTeam'])
    listed = tuple(t.attrib['id'] for t in root.findall('Metadata/Teams/Team'))
    if len(set(teams)) != 2 or len(listed) != 2 or set(teams) != set(listed):
        raise ValueError('Expected two distinct local/visiting teams')
    players = root.findall('Metadata/Players/Player')
    ids = tuple(p.attrib['id'] for p in players)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Player IDs must be nonempty and unique')
    side = tuple(teams.index(p.attrib['teamId']) for p in players)
    if set(side) != {0, 1}:
        raise ValueError('Both teams must have roster players')
    return ids, side


def _channels(root, player_ids):
    units = root.findall('Metadata/Devices/Device/Sensors/Sensor/Channels/Channel')
    if (len(units) != 2 or {u.attrib['id'] for u in units} != {'x', 'y'}
            or any(u.findtext('Unit') != 'normalized' for u in units)):
        raise ValueError('Expected normalized x/y position channels')
    refs = root.findall('Metadata/PlayerChannels/PlayerChannel')
    mapping = {r.attrib['id']: (player_ids.index(r.attrib['playerId']),
                               ('x', 'y').index(r.attrib['channelId'])) for r in refs}
    if len(mapping) != len(refs) or len(set(mapping.values())) != len(refs):
        raise ValueError('Player channels must be unique')
    if set(mapping.values()) != {(i, axis) for i in range(len(player_ids)) for axis in (0, 1)}:
        raise ValueError('Every roster player must have x/y channels')
    return mapping


def _layout(spec, channels):
    if (spec.attrib['separator'] != ':' or len(spec) != 3
            or spec[0].tag != 'StringRegister' or spec[0].attrib.get('name') != 'frameCount'
            or spec[1].tag != 'SplitRegister' or spec[1].attrib.get('separator') != ';'
            or spec[2].tag != 'SplitRegister' or spec[2].attrib.get('separator') != ','
            or tuple((r.tag, r.attrib.get('channelId')) for r in spec[2])
            != (('BallChannelRef', 'x'), ('BallChannelRef', 'y'))):
        raise ValueError('Unsupported frame/player/ball register schema')
    pairs = tuple(tuple(channels[r.attrib['playerChannelId']] for r in pair)
                  for pair in spec[1])
    for node, pair in zip(spec[1], pairs):
        if (node.tag != 'SplitRegister' or node.attrib.get('separator') != ','
                or any(r.tag != 'PlayerChannelRef' for r in node) or len(pair) != 2
                or pair[0][0] != pair[1][0] or {p[1] for p in pair} != {0, 1}):
            raise ValueError('Expected one x/y channel pair per player register')
    ids = tuple(pair[0][0] for pair in pairs)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Layout players must be nonempty and unique')
    axes = tuple(tuple(next(i for i, p in enumerate(pair) if p[1] == axis)
                       for axis in (0, 1)) for pair in pairs)
    return Layout(_integer(spec.attrib['startFrame']), _integer(spec.attrib['endFrame']), ids, axes)


def parse_metadata(xml):
    """Parse the supported Game 3 metadata structure; reject ambiguous layouts."""
    try:
        root = ET.fromstring(xml)
        halves = _periods(root)
        player_ids, team = _roster(root)
        fps = _integer(root.findtext('Metadata/GlobalConfig/FrameRate'))
        if fps != 25:
            raise ValueError('Player feature extraction requires 25 FPS')
        pitch = tuple(float(root.findtext(f'Metadata/Sessions/Session/FieldSize/{axis}'))
                      for axis in ('Width', 'Height'))
        if not all(np.isfinite(v) and v > 0 for v in pitch):
            raise ValueError('Pitch dimensions must be positive finite metres')
        channels = _channels(root, player_ids)
        layouts = tuple(_layout(spec, channels) for spec in
                        root.findall('DataFormatSpecifications/DataFormatSpecification'))
        previous = 0
        for layout in layouts:
            if layout.first != previous + 1 or layout.last < layout.first:
                raise ValueError('Layout ranges must be ordered, contiguous and nonoverlapping')
            previous = layout.last
        if not layouts or previous != halves[-1][1]:
            raise ValueError('Layouts must cover exactly the half ranges')
        return Metadata(player_ids, team, pitch, fps, halves, layouts)
    except (ET.ParseError, KeyError, TypeError, IndexError) as error:
        raise ValueError(f'Invalid Metrica metadata: {error}') from error


def _coordinates(register, count):
    try:
        values = np.asarray([pair.split(',') for pair in register.split(';')], dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Invalid coordinate register') from error
    if values.shape != (count, 2) or np.isinf(values).any():
        raise ValueError('Expected finite or NaN x/y coordinates per register')
    return values


def parse_tracking(lines, metadata):
    """Read every row in order; missing, repeated or out-of-range frames fail.

    The lower-level parser accepts synthetic data; ``load_metrica_tracking`` is the
    research entry point and verifies both complete source files against their pins.
    """
    n_frames = metadata.halves[-1][1]
    players = np.full((n_frames, len(metadata.player_ids), 2), np.nan, dtype=np.float32)
    section = np.concatenate(tuple(np.full(last - first + 1, half, dtype=np.int8)
                                   for half, (first, last) in enumerate(metadata.halves, 1)))
    layout_index = 0
    seen = 0
    for row, line in enumerate(lines):
        fields = line.rstrip('\r\n').split(':')
        if len(fields) != 3 or _integer(fields[0]) != row + 1 or row >= n_frames:
            raise ValueError(f'Invalid frame sequence or row schema at row {row + 1}')
        if row + 1 > metadata.layouts[layout_index].last:
            layout_index += 1
        layout = metadata.layouts[layout_index]
        values = _coordinates(fields[1], len(layout.player_indices))
        _coordinates(fields[2], 1)
        xy = np.take_along_axis(values, np.asarray(layout.axes), axis=1)
        players[row, layout.player_indices] = (xy - 0.5) * metadata.pitch
        seen += 1
    if seen != n_frames:
        raise ValueError(f'Expected {n_frames} frames, got {seen}')
    frame = np.arange(1, n_frames + 1, dtype=np.int32)
    team = np.asarray(metadata.team, dtype=np.int8)
    for array in (frame, players, team, section):
        array.setflags(write=False)
    return PlayerTracking(frame, players, team, section, metadata.player_ids,
                          metadata.halves, metadata.pitch, metadata.fps)


def _verify(path, expected):
    with Path(path).open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError(f'SHA-256 does not match the pinned Game 3 file: {Path(path).name}')


def load_metrica_tracking(metadata=DEFAULT_METADATA, tracking=DEFAULT_TRACKING):
    """Load only the pinned local files; no network requests or derived artifacts."""
    _verify(metadata, METADATA_SHA256)
    _verify(tracking, TRACKING_SHA256)
    parsed = parse_metadata(Path(metadata).read_bytes())
    with Path(tracking).open(encoding='ascii') as handle:
        return parse_tracking(handle, parsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--tracking', type=Path, default=DEFAULT_TRACKING)
    args = parser.parse_args()
    try:
        match = load_metrica_tracking(args.metadata, args.tracking)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        'source_revision': REVISION, 'metadata_url': BASE_URL + DEFAULT_METADATA.name,
        'tracking_url': BASE_URL + DEFAULT_TRACKING.name,
        'metadata_sha256': METADATA_SHA256, 'tracking_sha256': TRACKING_SHA256,
        'frames': len(match.frame), 'player_shape': match.players.shape,
        'home_roster': int((match.team == 0).sum()),
        'away_roster': int((match.team == 1).sum()), 'fps': match.fps,
        'pitch_metres': match.pitch, 'halves_inclusive': match.halves,
        'missing_coordinate_values': int(np.isnan(match.players).sum()),
        'limitation': 'Research only; reuse terms unresolved; no possession labels defined.',
    }, indent=2))


if __name__ == '__main__':
    main()
