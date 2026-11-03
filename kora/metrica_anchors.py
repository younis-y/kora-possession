"""Research-only CARRY anchor audit for pinned Metrica Sample Game 3.

Run ``python -m kora.metrica_anchors [path/to/Sample_Game_3_events.json]``.
The input must match the pinned SHA-256. No download, training, inference, ground-truth
comparison, or derived-data export occurs. Unknown and conflicting frames stay unlabeled.

Intervals use inclusive event endpoints as an audit convention, not a verified possession
boundary definition. These weak anchors differ from corrected-tracking possession labels.
The source requests acknowledgement; a standard reuse licence has not been established.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

REVISION = "e706dd506b360d69d9d123d5b8026e7294b13996"
SOURCE_URL = (f"https://raw.githubusercontent.com/metrica-sports/sample-data/{REVISION}/"
              "data/Sample_Game_3/Sample_Game_3_events.json")
EVENTS_SHA256 = "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0"
N_FRAMES = 143761
HALVES = {1: (1, 69661), 2: (69662, N_FRAMES)}
TEAM_LABELS = {"FIFATMA": "H", "FIFATMB": "A"}
DEFAULT_EVENTS = (Path(__file__).resolve().parent.parent / "data" / "external" /
                  "metrica" / REVISION / "Sample_Game_3_events.json")


def _interval(event, n_frames):
    """Return rejection category or a validated inclusive (start, end, label)."""
    start, end = event.get("start"), event.get("end")
    period = event.get("period")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return "rejected_invalid", None
    a, b = start.get("frame"), end.get("frame")
    if (type(a) is not int or type(b) is not int or type(period) is not int
            or period not in HALVES or not 1 <= a <= b <= n_frames):
        return "rejected_invalid", None
    first, last = HALVES[period]
    if not first <= a <= b <= last:
        return "rejected_invalid", None
    team = event.get("team")
    team_id = team.get("id") if isinstance(team, dict) else None
    if not isinstance(team_id, str) or team_id not in TEAM_LABELS:
        return "rejected_unknown_team", None
    return "accepted_carries", (a, b, TEAM_LABELS[team_id])


def _coverage(intervals, label, n_frames):
    starts = [a - 1 for a, b, code in intervals if code == label]
    stops = [b for a, b, code in intervals if code == label]
    delta = (np.bincount(starts, minlength=n_frames + 1)
             - np.bincount(stops, minlength=n_frames + 1))
    return np.cumsum(delta)[:-1] > 0


def carry_anchors(events, n_frames=N_FRAMES):
    """Return sparse labels and support counts; labels[i] describes source frame i+1.

    H/A use the metadata's local/visiting team IDs. Empty string means unknown, including
    every opposing-team overlap. Invalid intervals are rejected whole, never clipped.
    ``carry_frame_sum`` counts inclusive event lengths before same-team deduplication.
    """
    if type(n_frames) is not int or not 1 <= n_frames <= N_FRAMES:
        raise ValueError(f"n_frames must be an integer in 1..{N_FRAMES}")
    if not isinstance(events, (list, tuple)):
        raise ValueError("events must be a list or tuple")
    carries = tuple(e for e in events if isinstance(e, dict)
                    and isinstance(e.get("type"), dict)
                    and e["type"].get("name") == "CARRY")
    parsed = tuple(_interval(e, n_frames) for e in carries)
    rejected = Counter(status for status, interval in parsed)
    intervals = tuple(interval for status, interval in parsed if interval is not None)
    home, away = (_coverage(intervals, code, n_frames) for code in ("H", "A"))
    conflict = home & away
    labels = np.where(home & ~away, "H", np.where(away & ~home, "A", ""))
    pair_count = sum(a[2] != b[2] and max(a[0], b[0]) <= min(a[1], b[1])
                     for i, a in enumerate(intervals) for b in intervals[i + 1:])
    counts = {
        "total_events": len(events), "carry_events": len(carries),
        "accepted_carries": rejected["accepted_carries"],
        "rejected_invalid": rejected["rejected_invalid"],
        "rejected_unknown_team": rejected["rejected_unknown_team"],
        "carry_frame_sum": sum(b - a + 1 for a, b, code in intervals),
        "home_covered_frames": int(home.sum()), "away_covered_frames": int(away.sum()),
        "conflicting_carry_pairs": pair_count, "conflict_frames": int(conflict.sum()),
        "labeled_home_frames": int(np.count_nonzero(labels == "H")),
        "labeled_away_frames": int(np.count_nonzero(labels == "A")),
        "unknown_frames": int(np.count_nonzero(labels == "")),
    }
    return labels, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, nargs="?", default=DEFAULT_EVENTS)
    args = parser.parse_args()
    try:
        raw = args.events.read_bytes()
        if hashlib.sha256(raw).hexdigest() != EVENTS_SHA256:
            raise ValueError("SHA-256 does not match the pinned Game 3 events")
        _, counts = carry_anchors(json.loads(raw)["data"])
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({
        "source_revision": REVISION, "source_url": SOURCE_URL,
        "events_sha256": EVENTS_SHA256, "n_frames": N_FRAMES,
        "frame_convention": "1-based inclusive endpoints; array index = frame - 1",
        "limitation": "Endpoint semantics unverified; weak anchors, not possession ground truth.",
        "unknown_label": "", "counts": counts,
    }, indent=2))


if __name__ == "__main__":
    main()
