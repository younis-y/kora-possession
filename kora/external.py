"""Training data from other matches: the IDSSE open dataset (Bassek et al., Scientific Data 2025).

Seven full Bundesliga 2022/23 matches of DFL positional data, CC BY 4.0, from figshare:
    https://doi.org/10.1038/s41597-025-04505-y
Each ball frame carries the official BallStatus (alive/dead) and BallPossession (home/guest), the
same kind of per-frame label the challenge ground truth is derived from (corrected tracking that
marks the ball dead or names the owning team). No data about the challenge match is used.

    python -m kora.external          # download (if missing) and cache all seven matches
"""
import re
import subprocess
from pathlib import Path

import numpy as np

from .data import cache_dir, data_dir

MATCHES = {   # match id: (meta, events, tracking) figshare file ids, from kloppy's IDSSE table
    "J03WPY": (51643487, 51643505, 51643526),
    "J03WN1": (51643472, 51643496, 51643517),
    "J03WMX": (51643475, 51643493, 51643514),
    "J03WOH": (51643478, 51643499, 51643520),
    "J03WQQ": (51643484, 51643508, 51643529),
    "J03WOY": (51643481, 51643502, 51643523),
    "J03WR9": (51643490, 51643511, 51643532),
}
URL = "https://ndownloader.figshare.com/files/{}"

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")?')
BALLATTR = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def ext_dir():
    d = data_dir() / "external" / "idsse"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(match):
    for kind, fid in zip(("meta", "events", "tracking"), MATCHES[match]):
        p = ext_dir() / f"{match}_{kind}.xml"
        if not p.exists() or p.stat().st_size == 0:
            subprocess.run(["curl", "-sfL", "-A", "Mozilla/5.0", "-o", str(p), URL.format(fid)], check=True)


def home_team(match):
    meta = (ext_dir() / f"{match}_meta.xml").read_text(encoding="utf-8")
    return re.search(r'HomeTeamId="([^"]+)"', meta).group(1)


def load(match):
    """Dict of arrays over all frames of both halves, 25 fps:
    half[n] (1/2), players[n, P, 2] metres (NaN when off pitch), team[P] (0 home, 1 away),
    ball[n, 3] x, y, z metres, label[n] 0 H / 1 A / 2 D.  Origin is the centre spot."""
    cache = cache_dir() / f"idsse_{match}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    download(match)
    home = home_team(match)
    series = {}       # (section, person) -> (team, frames list, xy list)
    ball = {}         # section -> (frames, xyz, poss, status)
    cur = None
    with open(ext_dir() / f"{match}_tracking.xml", encoding="utf-8") as f:
        for line in f:
            if line.startswith("<Frame "):
                if cur[1] == "SKIP":
                    continue
                m = FRAME.match(line)
                n, x, y, zz = int(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)
                if cur[1] == "BALL":
                    b = BALLATTR.search(line)
                    ball[cur[0]].append((n, x, y, float(zz or 0), int(b.group(1)), int(b.group(2))))
                else:
                    series[(cur[0], cur[2])][1].append((n, x, y))
            elif line.startswith("<FrameSet"):
                sec, team, person = FRAMESET.match(line).groups()
                cur = (sec, team, person)
                if team == "BALL":
                    ball.setdefault(sec, [])
                elif team.startswith("DFL-CLU"):          # players only; some files also track referees
                    series.setdefault((sec, person), (team, []))
                else:
                    cur = (sec, "SKIP", person)
    persons = sorted({p for _, p in series})
    pidx = {p: i for i, p in enumerate(persons)}
    teams = {p: t for (_, p), (t, _) in series.items()}
    team = np.array([0 if teams[p] == home else 1 for p in persons], dtype=np.int8)
    halves, P, B, L = [], [], [], []
    for h, sec in ((1, "firstHalf"), (2, "secondHalf")):
        arr = np.array(ball[sec], dtype=np.float64)
        f0, f1 = int(arr[:, 0].min()), int(arr[:, 0].max())
        n = f1 - f0 + 1
        bb = np.full((n, 3), np.nan, np.float32)
        lab = np.full(n, 2, np.int8)
        i = arr[:, 0].astype(int) - f0
        bb[i] = arr[:, 1:4]
        poss, status = arr[:, 4].astype(int), arr[:, 5].astype(int)
        lab[i] = np.where(status == 0, 2, np.where(poss == 1, 0, 1))
        pp = np.full((n, len(persons), 2), np.nan, np.float32)
        for (s, person), (_, rows) in series.items():
            if s != sec or not rows:
                continue
            r = np.array(rows, dtype=np.float64)
            j = r[:, 0].astype(int) - f0
            ok = (j >= 0) & (j < n)
            pp[j[ok], pidx[person]] = r[ok, 1:3]
        halves.append(np.full(n, h, np.int8)); P.append(pp); B.append(bb); L.append(lab)
    out = {"half": np.concatenate(halves), "players": np.concatenate(P), "team": team,
           "ball": np.concatenate(B), "label": np.concatenate(L)}
    np.savez_compressed(cache, **out)
    return out


if __name__ == "__main__":
    for m in MATCHES:
        d = load(m)
        lab = d["label"]
        print(m, d["players"].shape, f"H {np.mean(lab == 0):.2f} A {np.mean(lab == 1):.2f} D {np.mean(lab == 2):.2f}",
              f"ball seen {np.isfinite(d['ball'][:, 0]).mean():.2f}",
              f"on pitch per frame {np.isfinite(d['players'][:, :, 0]).sum(1).mean():.1f}")
