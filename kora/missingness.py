"""How automatic tracking loses the ball and the players, measured on EXTERNAL data only.

The organisers' rules (3 Nov 2026) require every fitted or tuned parameter to come from external
data. kora/degrade.py therefore takes all its numbers from models/degrade_external.json, written
here from two open sources and nothing else:

- SkillCorner open data (kora/skillcorner.py): automatic broadcast tracking with real gaps
  (`is_detected` for the ball and every player).
- IDSSE (kora/external.py): corrected tracking, the clean end of the range (positional smoothness).

One rule for every quantity: measure it on **every frame of the two periods** of the external stream,
camera cuts and replays included, because that is what automatic tracking delivers. SkillCorner has
no alive/dead flag, so a frame with no possession owner stands in for a stoppage (it also includes
loose balls). An earlier draft measured the ball only while the camera showed players, through a
speed-band mixture with IDSSE's dead rate; that censors exactly the stoppage frames where the ball is
lost (P(missing | dead) 0.14 vs 0.79 here). Both estimates were seen before this rule was chosen;
RESEARCH.md Y5 records that.

Each training copy draws a severity s ~ U(0, 1) and scales every process by it, from clean corrected
tracking (s = 0) to SkillCorner automatic tracking (s = 1). Nothing here reads the challenge match.

    python -m kora.missingness       # writes models/degrade_external.json
"""
import json
from pathlib import Path

import numpy as np

from . import skillcorner
from .external import MATCHES as IDSSE, load as load_idsse

OUT = Path(__file__).resolve().parent.parent / "models" / "degrade_external.json"
SC_FPS = skillcorner.FPS_SOURCE
CLOSEUP_MIN_S = 4.0                         # only stoppages at least this long can be close-ups (design)
QUANTILES = np.linspace(0, 1, 101)


def runs(mask):
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def second_diff_var(players):
    """Variance of the second difference of observed positions (motion plus measurement noise)."""
    d2 = players[2:] - 2 * players[1:-1] + players[:-2]
    d2 = d2[np.isfinite(d2)]
    return float(np.var(d2))


def resample_to(players, fps_from, fps_to):
    t = np.arange(0, len(players) - 1, fps_from / fps_to)
    i0 = np.floor(t).astype(int)
    w = (t - i0)[:, None, None]
    return players[i0] * (1 - w) + players[np.minimum(i0 + 1, len(players) - 1)] * w


def idsse_second_diff_var():
    var = []
    for m in IDSSE:
        d = load_idsse(m)
        for h in (1, 2):
            var.append(second_diff_var(resample_to(d["players"][d["half"] == h], 25, SC_FPS)))
    return float(np.mean(var))


def measure_skillcorner():
    """Pooled over the twenty matches, per half, every frame of the period, 10 fps."""
    n = dict(miss_owner=0, owner=0, miss_free=0, free=0)
    live_gaps, all_gaps, start_off, end_off = [], [], [], []
    vis_runs, gap_runs, closeup, stoppages, radius, var = [], [], 0, 0, [], []
    for m in skillcorner.MATCHES:
        d = skillcorner.raw(m)
        for h in (1, 2):
            i = d["half"] == h
            P, B, L = d["players"][i], d["ball"][i], d["label"][i]
            seen_p = np.isfinite(P[:, :, 0])
            miss = ~np.isfinite(B[:, 0])
            known = L != skillcorner.UNKNOWN
            n["miss_owner"] += int((miss & known).sum()); n["owner"] += int(known.sum())
            n["miss_free"] += int((miss & ~known).sum()); n["free"] += int((~known).sum())
            var.append(second_diff_var(P))
            for s, e in zip(*runs(miss)):
                all_gaps.append((e - s) / SC_FPS)
                if s > 0 and e < len(L) and known[s - 1] and known[e]:      # flanked by an owner: live play
                    live_gaps.append((e - s) / SC_FPS)
            ref = np.median(seen_p[known].sum(1))
            for s, e in zip(*runs(~known)):                                 # stoppage proxy: no owner
                g = np.flatnonzero(miss[s:e])
                if len(g):
                    start_off.append(g[0] / SC_FPS)
                    end_off.append((e - s - 1 - g[-1]) / SC_FPS)
                if (e - s) >= CLOSEUP_MIN_S * SC_FPS:
                    stoppages += 1
                    if np.median(seen_p[s:e].sum(1)) < ref / 2:
                        closeup += 1
                        with np.errstate(all="ignore"):
                            c = np.nanmean(B[s:e], 0)
                            if np.isfinite(c).all():
                                dist = np.linalg.norm(P[s:e] - c[None, None], axis=2)
                                if np.isfinite(dist).any():
                                    radius.append(float(np.nanpercentile(dist, 90)))
            for j in range(P.shape[1]):                                     # each player while in this half
                v = seen_p[:, j]
                if not v.any():
                    continue
                first, last = np.flatnonzero(v)[[0, -1]]
                v = v[first:last + 1]
                vis_runs += [(e - s) / SC_FPS for s, e in zip(*runs(v))]
                gap_runs += [(e - s) / SC_FPS for s, e in zip(*runs(~v))]
    return dict(n=n, var=float(np.mean(var)), live_gaps=np.array(live_gaps), all_gaps=np.array(all_gaps),
                start_off=np.array(start_off), end_off=np.array(end_off), vis=np.array(vis_runs),
                gap=np.array(gap_runs), closeup=closeup, stoppages=stoppages, radius=np.array(radius))


def quantiles(x):
    return [round(float(v), 3) for v in np.quantile(x, QUANTILES)]


def main():
    sc = measure_skillcorner()
    var_id = idsse_second_diff_var()
    n = sc["n"]
    noise = float(np.sqrt(max(0.0, (sc["var"] - var_id) / 6)))      # white noise adds 6 sigma^2
    out = {
        "_source": "kora/missingness.py on SkillCorner open data (commit " + skillcorner.COMMIT + ", every "
                   "frame of both periods) and the seven IDSSE matches; no challenge data. Each copy draws "
                   "severity s ~ U(0,1) and scales every process by it (s = 0 clean IDSSE, s = 1 SkillCorner).",
        "p_miss_dead": n["miss_free"] / n["free"],          # stoppage proxy: no possession owner
        "p_miss_alive": n["miss_owner"] / n["owner"],
        "missing_share": (n["miss_free"] + n["miss_owner"]) / (n["free"] + n["owner"]),
        "live_gap_s_quantiles": quantiles(sc["live_gaps"]),
        "gap_s_quantiles": quantiles(sc["all_gaps"]),       # every ball gap, for label-free stress tests
        "dead_start_offset_s_quantiles": quantiles(sc["start_off"]),
        "dead_end_offset_s_quantiles": quantiles(sc["end_off"]),
        "player_visible_s_mean": float(sc["vis"].mean()),
        "player_gap_s_quantiles": quantiles(sc["gap"]),
        "closeup_prob": sc["closeup"] / max(sc["stoppages"], 1),
        "closeup_min_s": CLOSEUP_MIN_S,
        "closeup_radius_m_quantiles": quantiles(sc["radius"]) if len(sc["radius"]) else [],
        "position_noise_m": noise,
        "_diagnostics": {
            "frames": n, "n_ball_gaps": int(len(sc["all_gaps"])), "n_live_gaps": int(len(sc["live_gaps"])),
            "n_stoppages": int(sc["stoppages"]), "n_closeups": int(sc["closeup"]),
            "second_diff_var": {"idsse": var_id, "skillcorner": sc["var"]},
        },
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("quantiles")}, indent=1))
    for name, x in (("live ball gaps", sc["live_gaps"]), ("all ball gaps", sc["all_gaps"]),
                    ("player gaps", sc["gap"]), ("stoppage start offset", sc["start_off"]),
                    ("stoppage end offset", sc["end_off"])):
        print(f"{name}: median {np.median(x):.2f}s p90 {np.percentile(x, 90):.2f}s (n {len(x)})")


if __name__ == "__main__":
    main()
