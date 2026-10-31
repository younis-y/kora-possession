"""Descriptive profile of the tactical data. Writes markdown to stdout; no ground truth is read.

    python -m kora.profile > notes/YYYY-MM-DD-profile.md
"""
import numpy as np
import pandas as pd

from .data import FPS, HALVES, PITCH_M, homography, smart_tagging, teams, tracking


def runs(mask):
    """(start, length) of each run of True in a boolean array."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return starts, ends - starts


def graded_mask(n):
    g = np.zeros(n, dtype=bool)
    for a, b in HALVES.values():
        g[a:b + 1] = True
    return g


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    home, away, side = teams()
    frame, P, B = tracking()
    n = len(frame)
    g = graded_mask(n)
    print("# Tactical data profile\n")
    print(f"{n:,} frames, graded {g.sum():,}. H = {home}, A = {away}.\n")

    # ball visibility
    seen = np.isfinite(B[:, 0])
    print("## Ball visibility\n")
    print(f"- seen on {pct(seen.mean())} of all frames, {pct(seen[g].mean())} of graded frames")
    _, L = runs(~seen & g)
    sec = L / FPS
    print(f"- {len(L):,} missing-ball gaps inside graded time; median {np.median(sec):.2f} s, "
          f"p90 {np.percentile(sec, 90):.1f} s, max {sec.max():.1f} s")
    for t in (1, 2, 5, 10):
        print(f"  - gaps longer than {t} s: {(sec > t).sum():,} covering {sec[sec > t].sum() / 60:.1f} min")
    _, L2 = runs(seen & g)
    print(f"- {len(L2):,} seen stretches; median {np.median(L2) / FPS:.2f} s, p90 {np.percentile(L2, 90) / FPS:.1f} s\n")

    # players
    npl = np.isfinite(P[:, :, 0]).sum(1)
    print("## Tracked players per frame (graded)\n")
    q = np.percentile(npl[g], [5, 25, 50, 75, 95])
    print(f"- mean {npl[g].mean():.1f}; p5/p25/p50/p75/p95 = {'/'.join(str(int(v)) for v in q)}; "
          f"frames with >22: {pct((npl[g] > 22).mean())}, with <15: {pct((npl[g] < 15).mean())}")
    hn = np.isfinite(P[:, side == 0, 0]).sum(1)
    an = np.isfinite(P[:, side == 1, 0]).sum(1)
    print(f"- home slots visible mean {hn[g].mean():.1f}, away {an[g].mean():.1f}\n")

    # ball position / speed
    print("## Ball position and movement (frames where seen)\n")
    bx, by = B[:, 0], B[:, 1]
    out = seen & ((bx < 0) | (bx > 1) | (by < 0) | (by > 1))
    print(f"- outside the 0-1 pitch box: {pct(out[g & seen].sum() / max(seen[g].sum(), 1))} of seen graded frames, "
          f"{len(runs(out & g)[0])} stretches")
    bs = B[:, 2]
    print(f"- ball speed column finite on {pct(np.isfinite(bs[seen]).mean())} of seen frames"
          + (f"; median {np.nanmedian(bs[seen]):.2f} m/s" if np.isfinite(bs[seen]).any() else ""))
    # displacement over 1 s where both ends seen
    k = FPS
    dx = (bx[k:] - bx[:-k]) * PITCH_M[0]
    dy = (by[k:] - by[:-k]) * PITCH_M[1]
    both = seen[k:] & seen[:-k] & g[k:]
    disp = np.hypot(dx, dy)[both]
    print(f"- 1-s displacement (both ends seen, graded): median {np.median(disp):.2f} m, p10 {np.percentile(disp, 10):.2f} m, "
          f"share under 0.5 m: {pct((disp < 0.5).mean())}")
    still = np.zeros(n, dtype=bool)
    still[k:][both & (np.hypot(dx, dy)[both] < 0.5) if False else np.zeros(n - k, bool)] = False
    s_idx = np.flatnonzero(both)[np.hypot(dx, dy)[both] < 0.5] + k
    still[s_idx] = True
    S, L3 = runs(still & g)
    print(f"- near-stationary ball (<0.5 m per s) stretches >= 3 s in graded time: {(L3 >= 3 * FPS).sum()} "
          f"covering {L3[L3 >= 3 * FPS].sum() / FPS / 60:.1f} min\n")

    # nearest player distance
    d = np.linalg.norm((P[:, :, :2] - B[:, None, :2]) * np.array(PITCH_M, dtype=np.float32), axis=2)
    d[np.isnan(d)] = np.inf
    nd = d.min(1)
    ok = g & np.isfinite(nd)
    print("## Nearest tracked player to the ball (graded, ball seen)\n")
    q = np.percentile(nd[ok], [10, 25, 50, 75, 90])
    print(f"- distance p10/p25/p50/p75/p90 = {'/'.join(f'{v:.1f}' for v in q)} m; within 1.5 m on {pct((nd[ok] < 1.5).mean())}, "
          f"within 3 m on {pct((nd[ok] < 3).mean())}")
    team_near = side[d.argmin(1)]
    raw = np.where(ok, team_near, -1)
    chg = (raw[1:] != raw[:-1]) & (raw[1:] >= 0) & (raw[:-1] >= 0)
    print(f"- raw nearest-team flips between consecutive seen frames: {chg.sum():,} (a possession count would be a few hundred)\n")

    # smart tagging
    st = smart_tagging()
    print("## Smart tagging\n")
    print(f"- {len(st)} instances; team tags: {st.team.value_counts(dropna=False).to_dict()}")
    cov = np.zeros(n, dtype=np.int8)   # 0 none, 1 H only, 2 A only, 3 both/NA mixed
    hm = np.zeros(n, bool); am = np.zeros(n, bool); na = np.zeros(n, bool)
    for r in st.itertuples():
        sl = slice(max(r.start_frame, 0), min(r.end_frame, n))
        if r.team == home: hm[sl] = True
        elif r.team == away: am[sl] = True
        else: na[sl] = True
    print(f"- graded frames inside a {home}-tagged phase {pct(hm[g].mean())}, {away}-tagged {pct(am[g].mean())}, "
          f"both at once {pct((hm & am)[g].mean())}, neither {pct((~hm & ~am)[g].mean())}, N/A-tagged {pct(na[g].mean())}")
    sp = st[st.code.str.upper() == "SET PIECES"]
    print(f"- set pieces: {sp.type.value_counts().to_dict()}")
    print(f"- set-piece team tags: {sp.team.value_counts(dropna=False).to_dict()}")
    ko = sp[sp.type.str.contains("Kick Off", case=False, na=False)][["type", "half", "team", "start_frame", "end_frame"]]
    print("\nKick-offs and goals (video frames):\n")
    print(ko.to_markdown(index=False))
    print()
    print(st[st.code.str.upper() == "GOALS"][["team", "half", "start_frame", "end_frame"]].to_markdown(index=False))
    print(f"\nGraded blocks: {HALVES}. Smart-tagging first/last frame per half: "
          + ", ".join(f"{h}: {grp.start_frame.min()}-{grp.end_frame.max()}" for h, grp in st.groupby('half')))

    # homography error
    H, err, valid = homography()
    print("\n## Homography\n")
    print(f"- valid {pct(valid.mean())}; reprojection error median {np.median(err[g]):.2f} (graded), "
          f"{np.median(err[~g]):.2f} (ungraded); p99 graded {np.percentile(err[g], 99):.2f}")


if __name__ == "__main__":
    main()
