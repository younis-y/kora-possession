"""External automatic tracking: the SkillCorner open data (A-League 2024/25, 10 fps broadcast tracking).

Twenty matches from https://github.com/SkillCorner/opendata, pinned to one commit, each file checked
against the SHA-256 of its git LFS object. The repository is MIT licensed and asks users to credit
SkillCorner. Used for what IDSSE cannot show: how automatic tracking really loses the ball and the
players (`is_detected`). Its per-frame `possession.group` is SkillCorner's own output, with unknown
provenance, so it is an H/A label at most and never a dead-ball label.

    python -m kora.skillcorner        # download (if missing) and verify every match
"""
import json
import hashlib
import subprocess

import numpy as np

from .data import cache_dir, data_dir

COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/{}/data/matches/{}/{}"
LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/{}/data/matches/{}/{}"
FPS_SOURCE = 10
UNKNOWN = 3   # label code for frames with no possession group (pm.X)

MATCHES = {   # match id: SHA-256 of {id}_tracking_extrapolated.jsonl (git LFS oid at COMMIT)
    "1874553": "ea97f58f8eaad925feaeacc6395ec24860dd80027af8a89450276adebd29d265",
    "1886347": "3577e2803da95390f8b2f85d47829eb55fbbfee6211c94adc65db0dc78e46b41",
    "1899585": "90b6289ec2181192d1793ec6240fbbe8225ff0219c7956710c6278c31a661100",
    "1925299": "48528be40ad82fa3e2b232e1af8d311705f7d6d119925ef199d2b59ea36fd3c9",
    "1927964": "858c7a756ca3d33017f864775af5999d2c0f364e58ab60ae040c8b9ebab17827",
    "1953632": "1c2014f97c3eff1ce2475203512c9d126930d7599c036105c9ac442d1de1667e",
    "1959846": "5ba2d87f6fd1ba3138301b369bd9266be69416a2d799a3fba5b5ec8a52e1284b",
    "1986691": "630e946950fe459f5e48872b01a36e6dfc319cf9027e7b98156150bb6e2c30f6",
    "1996435": "f42764d8bde290e8f738fa21141cadbed75408314b0b337bbf1d5a3444540a02",
    "1996436": "fab9d331e06f3042c83ac17798901b49bb259c99a8f3b319cd0146263cde65e7",
    "2006229": "3ee193ba01f1fe2cbce087eca8a881c21b8b0ea3485ae6c3d7e32e8b2f51e738",
    "2006363": "61dfdbd83f8e654127b249b8503f062338c14ec0c9e65404bcf70fb7c1346bd4",
    "2007448": "12d00d6e4257218812df6e4c4263c49387ba0313a83ff8d826c4416828f9c798",
    "2007721": "fea913b8d4cbde8d85c6bdf5f1a4b3fd203b78227e976a171c8ab0946705d378",
    "2010085": "04562d9754b2977c80430535e9b5733426bc4ce910db56fe7a3760876211ac5a",
    "2011166": "8ec2937050930d87e4f52b2d65515f70ef8edd0ae2814bf40155c4448133390e",
    "2013725": "13eb83329814ed5f5d73da8f8ab5c5979d6c8a7858e89e28a7fa8072f91fb250",
    "2015213": "413990d0024d84856ce1f2a16866b8887472c0cea22c61caef0d1f46505a609e",
    "2016236": "d78555f9dfd7d1ce9de38066d3c98e6d54417c6d45d73b1f1ce7e26f07263d8f",
    "2017461": "59cfa5588efee041da439f44df5b5b92f2b7305571dc831164e3d05f7415787a",
}


def ext_dir():
    d = data_dir() / "external" / "skillcorner"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(match):
    for name, url in ((f"{match}_match.json", RAW), (f"{match}_tracking_extrapolated.jsonl", LFS)):
        p = ext_dir() / name
        if not p.exists() or p.stat().st_size == 0:
            subprocess.run(["curl", "-sfL", "--retry", "3", "-o", str(p),
                            url.format(COMMIT, match, name)], check=True)
    got = sha256(ext_dir() / f"{match}_tracking_extrapolated.jsonl")
    if got != MATCHES[match]:
        raise ValueError(f"SkillCorner {match}: SHA-256 {got} does not match the pinned commit")


def parse(meta, lines):
    """Raw 10 fps arrays from the match JSON and the tracking JSONL lines.

    Undetected (extrapolated) players and balls become NaN: the point of this source is its real
    gaps. Frames outside the two periods are dropped."""
    home = meta["home_team"]["id"]
    ids = sorted(p["id"] for p in meta["players"])
    col = {pid: j for j, pid in enumerate(ids)}
    team = np.array([0 if p["team_id"] == home else 1
                     for p in sorted(meta["players"], key=lambda p: p["id"])], np.int8)
    half, players, ball, label = [], [], [], []
    for line in lines:
        r = json.loads(line)
        if r["period"] not in (1, 2):
            continue
        pp = np.full((len(ids), 2), np.nan, np.float32)
        for p in r["player_data"]:
            if p["is_detected"] and p["player_id"] in col:
                pp[col[p["player_id"]]] = (p["x"], p["y"])
        b = r["ball_data"]
        bb = (b["x"], b["y"]) if b["is_detected"] and b["x"] is not None else (np.nan, np.nan)
        g = r["possession"]["group"]
        half.append(r["period"]); players.append(pp); ball.append(bb)
        label.append({"home team": 0, "away team": 1}.get(g, UNKNOWN))
    return {"half": np.array(half, np.int8), "players": np.stack(players), "team": team,
            "ball": np.array(ball, np.float32), "label": np.array(label, np.int8)}


def raw(match):
    """10 fps arrays, cached: half, players[n, P, 2], team[P], ball[n, 2], label (0 H, 1 A, 3 unknown)."""
    cache = cache_dir() / f"skillcorner_{match}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    download(match)
    meta = json.loads((ext_dir() / f"{match}_match.json").read_text(encoding="utf-8"))
    with open(ext_dir() / f"{match}_tracking_extrapolated.jsonl", encoding="utf-8") as f:
        out = parse(meta, f)
    np.savez_compressed(cache, **out)
    return out


def resample(d, fps):
    """Linear resample of positions to `fps` within each half; labels take the nearest source frame.
    A target frame is NaN unless both bracketing source frames are observed."""
    out = {k: [] for k in ("half", "players", "ball", "label")}
    for h in (1, 2):
        idx = np.flatnonzero(d["half"] == h)
        if not len(idx):
            continue
        t = np.arange(int((len(idx) - 1) * fps / FPS_SOURCE) + 1) * FPS_SOURCE / fps
        i0 = np.floor(t).astype(int)
        i1 = np.minimum(i0 + 1, len(idx) - 1)
        w = (t - i0)[:, None]
        for key in ("players", "ball"):
            x = d[key][idx]
            a, b = x[i0], x[i1]
            w_ = w[:, :, None] if x.ndim == 3 else w
            out[key].append((a * (1 - w_) + b * w_).astype(np.float32))
        out["label"].append(d["label"][idx][np.rint(t).astype(int).clip(0, len(idx) - 1)])
        out["half"].append(np.full(len(t), h, np.int8))
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["team"] = d["team"]
    return res


def load(match, fps=25):
    """The external.load() contract at `fps`: half, players (metres, centre origin), team, ball[n, 2], label."""
    return resample(raw(match), fps)


if __name__ == "__main__":
    for m in MATCHES:
        d = raw(m)
        print(m, len(d["half"]), "frames at 10 fps", flush=True)
