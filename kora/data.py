"""Load the match bundle once; everything downstream reads the cached arrays.

Layout is the one the organisers prescribe (data/tactical/..., data/broadcast/...), rooted at
MTAH_DATA_DIR or ./data. Caches live in <data>/cache and are safe to delete.
"""
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

GAME = "BAR-ATH"
FPS = 25
PITCH_M = (105.0, 68.0)          # x, y scale for the normalised 0-1 coordinates
HALVES = {1: (11185, 90651), 2: (110808, 190197)}   # graded video-frame blocks, inclusive
N_FRAMES = {"tactical": 199319, "broadcast": 164488}

ROOT = Path(__file__).resolve().parent.parent


def data_dir():
    return Path(os.environ.get("MTAH_DATA_DIR") or ROOT / "data")


def paths(source="tactical"):
    d = data_dir() / source
    p = {
        "video": d / f"{GAME}_{source}.mp4",
        "atd_meta": d / f"{GAME}_{source}_atd.xml",
        "atd_raw": d / f"{GAME}_{source}_atd.txt",
        "homography": d / f"{GAME}_{source}_homography.csv",
    }
    if source == "tactical":
        p["smart_tagging"] = d / f"{GAME}_{source}_smart_tagging.xml"
    return p


def cache_dir():
    c = data_dir() / "cache"
    c.mkdir(parents=True, exist_ok=True)
    return c


def teams(source="tactical"):
    """(home_name, away_name, side_of_track) with side 0 = home, 1 = away per track slot.

    First team in the EPTS metadata is home (FIFA convention, and Barcelona here).
    """
    xml = paths(source)["atd_meta"].read_text()
    team_ids = re.findall(r'<Team id="(\w+)">\s*<Name>([^<]+)</Name>', xml)
    home_id = team_ids[0][0]
    slots = re.findall(r'<Player id="(\d+)" teamId="(\w+)"', xml)
    slots.sort(key=lambda s: int(s[0]))
    side = np.array([0 if t == home_id else 1 for _, t in slots], dtype=np.int8)
    return team_ids[0][1], team_ids[1][1], side


def tracking(source="tactical"):
    """frame_id (1-based), players[n, slots, 3] (x, y, speed), ball[n, 3]. NaN = not seen."""
    cache = cache_dir() / f"{source}_tracking.npz"
    if cache.exists():
        z = np.load(cache)
        return z["frame"], z["players"], z["ball"]
    frames, players, ball = [], [], []
    with open(paths(source)["atd_raw"]) as f:
        for line in f:
            fr, pl, bl = line.rstrip("\n").split(":")
            frames.append(int(fr))
            players.append([[float(v) for v in p.split(",")] for p in pl.split(";")])
            ball.append([float(v) for v in bl.split(",")])
    frame = np.array(frames)
    players = np.array(players, dtype=np.float32)
    ball = np.array(ball, dtype=np.float32)
    np.savez_compressed(cache, frame=frame, players=players, ball=ball)
    return frame, players, ball


def homography(source="tactical"):
    """H[n, 3, 3], reprojection_error[n], valid[n] (bool), indexed by video frame."""
    cache = cache_dir() / f"{source}_homography.npz"
    if cache.exists():
        z = np.load(cache)
        return z["H"], z["err"], z["valid"]
    df = pd.read_csv(paths(source)["homography"])
    H = df.iloc[:, 1:10].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    err = df["reprojection_error"].to_numpy(dtype=np.float32)
    valid = df["valid"].to_numpy().astype(bool)
    np.savez_compressed(cache, H=H, err=err, valid=valid)
    return H, err, valid


def smart_tagging():
    """One row per instance: code, team, half, type, side, start_frame, end_frame (half-open)."""
    cache = cache_dir() / "smart_tagging.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for inst in ET.parse(paths()["smart_tagging"]).iter("instance"):
        tags = {l.findtext("group"): l.findtext("text") for l in inst.findall("label")}
        s, e = float(inst.findtext("start")), float(inst.findtext("end"))
        rows.append({
            "id": int(inst.findtext("ID")),
            "code": inst.findtext("code"),
            "team": tags.get("Team"),
            "half": tags.get("Half"),
            "type": tags.get("Type"),
            "side": tags.get("Side"),
            "start_s": s, "end_s": e,
            "start_frame": int(round(s * FPS)),
            "end_frame": int(round(e * FPS)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="parse the bundle into data/cache and print shapes")
    ap.add_argument("--source", default="tactical", choices=["tactical", "broadcast"])
    a = ap.parse_args()
    home, away, side = teams(a.source)
    print(f"{a.source}: H = {home} ({(side == 0).sum()} slots), A = {away} ({(side == 1).sum()} slots)")
    fr, P, B = tracking(a.source)
    print("tracking", fr.shape, P.shape, B.shape, f"ball seen {np.isfinite(B[:, 0]).mean():.1%}")
    H, err, valid = homography(a.source)
    print("homography", H.shape, f"valid {valid.mean():.1%}")
    if a.source == "tactical":
        st = smart_tagging()
        print("smart tagging", len(st), "instances;", st.code.value_counts().to_dict())
