# %%
# PRC 2026 — ADS-B v3 events: one row per departure with measured off-block evidence.
#
# Reads the extract_v3.py outputs (adsb3_stays_<TAG>.parquet, adsb3_points_<TAG>.parquet) from
# /kaggle/input (the k3 month kernels attached as kernel sources) or $DATA_ROOT, downloads
# nothing, and writes adsb3_events_<TAG>.parquet per month plus adsb3_events_summary.json.
#
# 1. Stitch: a stay that opens a day's trace (open_start, no landing seen) continues the same
#    hex's stay of the previous UTC day at the same airport if that one ran to the end of its
#    trace (open_end). The pieces form one chain: landing .. takeoff across midnight.
# 2. A departure is a chain whose last piece ended on airborne evidence (t_next_air with an
#    altitude). takeoff_seen = the climb followed the last ground point within 600 s.
# 3. Per departure, from the chain's ground points up to takeoff:
#    - dwells: runs of points reporting gs < 1 kt (positions may jump up to JUMP_M: surface
#      positions of a parked aircraft do), carried through "parked-through" gaps (same spot,
#      slow on both sides); reappearing already moving after such a gap ends the run there;
#    - separating gaps: any other gap >= 300 s; one >= 20 min hides a turnaround, so nothing
#      before it can time this departure (the evidence starts after it);
#    - park (the stand): with the arrival in the chain, the first dwell >= 300 s after the
#      landing; otherwise the dwell the evidence starts with (first seen parked) or one lasting
#      >= 30 min; then walked forward across tows (moves <= 16 kt to another dwell >= 30 min);
#    - line-up: the last point below 40 kt joined to the takeoff roll (else the roll's first);
#    - motion from the observed start: path length, speed, stops near the runway (<= 800 m of
#      line-up) and elsewhere, gaps with the positions either side, continuity;
#    - turnaround: touchdown of the same hex in the chain (landing seen) -> t_obs.
# 4. Grade: A = park seen, then seen leaving it within 60 s: t_obs = end of the park (the
#    pushback). B / C = first seen moving (after the park when leaving it was unseen, after the
#    last 20-min gap, or at the chain start when the arrival is not in it): t_obs is a lower
#    bound on off-block; B more than 800 m from line-up, C within. D = no departure evidence
#    (only the arrival, or silence of 20 min or more up to takeoff). obs_taxi_s = t_takeoff -
#    t_obs. Other candidates are kept as columns (t_leave, t_longest_dwell_end,
#    t_last_dwell_end, t_first, gap_* before/after positions).
#
# Data: derived from adsb.lol (ODbL 1.0); share-alike applies.

import glob
import json
import math
import multiprocessing as mp
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R_STAND_M = 15.0
MIN_PARK_S = 300.0
MIN_DWELL_S = 60.0
PARK_FROM_LINEUP_M = 300.0
TOW_TO_LINEUP_M = 800.0
NEAR_RWY_M = 800.0
GAP_S = 300.0
SEP_M = 30.0          # a gap across which the aircraft moved this far separates the evidence
LEAVE_GAP_S = 60.0    # grade A needs the first point after the dwell within this
BIG_SEP_S = 1200.0    # a separating gap this long hides the turnaround
MAX_EARLY_PUSH_S = 3600.0  # a stand before a big gap counts only if left within this of takeoff
FIRST_PARK_S = 300.0  # "first seen parked": the dwell starts within this of the evidence start
LONG_PARK_S = 1800.0  # a dwell this long is a stand even when the aircraft was seen moving first
JUMP_M = 100.0        # position jitter tolerated for a point that reports gs < 1 kt
TOW_KT = 16.0         # a move between two parks no faster than this is a tow or reposition
SLOW_KT = 5.0         # both sides of a gap below this (and the same spot): parked through it
LINEUP_KT = 40.0
MAX_STITCH_S = 36 * 3600
DWELL_KEEP = 12       # dwells written per departure (the last ones before takeoff)
DWELL_COLS = ["d_t0", "d_t1", "d_lat0", "d_lon0", "d_lat", "d_lon", "d_to_lineup_m", "d_leave_gap_s", "d_gs_leave",
              "d_gs_max_after", "d_before_big_gap", "d_is_park"]
NPROC = int(os.environ.get("NPROC", os.cpu_count() or 1))
T0 = time.time()


def roots():
    if os.environ.get("DATA_ROOT"):
        return [os.environ["DATA_ROOT"]]
    return [r for r in ("/kaggle/input", ".") if os.path.isdir(r)]


def files(prefix):
    out = {}
    for r in roots():
        for f in glob.glob(os.path.join(r, "**", f"{prefix}_*.parquet"), recursive=True):
            tag = os.path.basename(f)[len(prefix) + 1:-len(".parquet")]
            out.setdefault(tag, f)
    return dict(sorted(out.items()))


def dist_m(lat1, lon1, lat2, lon2):
    c = math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot((lat2 - lat1) * 110574.0, (lon2 - lon1) * 111320.0 * c)


def dist_v(lat1, lon1, lat2, lon2):
    c = np.cos(np.radians((lat1 + lat2) / 2))
    return np.hypot((lat2 - lat1) * 110574.0, (lon2 - lon1) * 111320.0 * c)


# %% stitching
def stitch(st: pd.DataFrame) -> pd.DataFrame:
    st = st.sort_values(["hex", "t_first", "k"]).reset_index(drop=True)
    prev = st.groupby("hex", sort=False)[["apt", "open_end", "day", "t_last_any"]].shift(1)
    d_cur = pd.to_datetime(st["day"].astype(str), format="%Y%m%d")
    d_prev = pd.to_datetime(prev["day"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    link = ((st["open_start"] == 1) & (st["landed"] == 0) & (prev["apt"] == st["apt"]) & (prev["open_end"] == 1)
            & ((d_cur - d_prev).dt.days == 1) & ((st["t_first"] - prev["t_last_any"]) < MAX_STITCH_S))
    st["stitched_prev"] = link.fillna(False).astype("int8")
    st["chain"] = (~link.fillna(False)).cumsum().astype("int64")
    return st


# %% per-departure measurement
def parked_through(tg, la, lo, v, i):
    """Gap i -> i+1 (>= GAP_S) with the aircraft on the same spot, slow on both sides."""
    return (tg[i + 1] - tg[i] >= GAP_S and dist_m(la[i], lo[i], la[i + 1], lo[i + 1]) <= SEP_M
            and (np.isnan(v[i]) or v[i] < SLOW_KT) and (np.isnan(v[i + 1]) or v[i + 1] < SLOW_KT))


def dwells(tg, la, lo, v):
    """Runs of stationary points: [(i0, i1)].

    Stationary is read from the reported ground speed first: surface positions of a parked
    aircraft often jump 20-80 m while gs stays 0.0, so a point with gs < 1 kt belongs to the run
    if it lies within JUMP_M of the run's first point; a point without gs needs R_STAND_M.
    A parked-through gap (same spot within SEP_M, below SLOW_KT on both sides) keeps the run
    going, and can start one: an aircraft that vanished rolling onto its stand sat there. If it
    reappears moving and keeps moving, the push began at that point, which ends the run; a
    single moving reading followed by a stationary one is noise. Otherwise a run ends at the
    first of two consecutive points that are moving or away.
    """
    runs, n, i = [], len(tg), 0
    known = ~np.isnan(v)
    slow = ~known | (v < 1.0)
    while i < n:
        if not (slow[i] or (i + 1 < n and parked_through(tg, la, lo, v, i))):
            i += 1
            continue
        a_lat, a_lon = la[i], lo[i]
        last_in, out, k = i, 0, i + 1
        while k < n:
            d = dist_m(a_lat, a_lon, la[k], lo[k])
            reach = JUMP_M if known[k] else R_STAND_M
            if k - 1 == last_in and parked_through(tg, la, lo, v, k - 1):
                last_in = k
                if slow[k]:
                    out = 0
                elif k + 1 < n and slow[k + 1] and dist_m(a_lat, a_lon, la[k + 1], lo[k + 1]) <= JUMP_M:
                    out = 0  # a moving blip on reappearance
                else:
                    break  # reappeared moving: the push began here
            elif slow[k] and d <= reach:
                last_in, out = k, 0
            else:
                out += 1
                if out >= 2 or d > 3 * JUMP_M or (known[k] and v[k] >= 3.0):
                    break
            k += 1
        if tg[last_in] - tg[i] >= MIN_DWELL_S:
            runs.append((i, last_in))
        i = last_in + 1
    return runs


def event_row(P, info):
    """P: dict of numpy arrays (t, lat, lon, gs, alt, gnd, cls) for the chain, sorted by t."""
    t, lat, lon, gs, alt, gnd, cls = (P[c] for c in ("t", "lat", "lon", "gs", "alt", "gnd", "cls"))
    ref = info["ref_alt"]
    row = {}
    # Takeoff time: the last ground-flag point, else the last ground-ish point (roll without the flag).
    if info["took_off"]:
        tail = t <= info["t_next_air"]
        if (gnd[tail] == 1).any():
            t_to = t[tail & (gnd == 1)].max()
        else:
            gish = tail & ((cls == 0) | ((cls == 1) & (np.nan_to_num(alt, nan=1e9) <= ref + 100)))
            t_to = t[gish].max() if gish.any() else info["t_last"]
    else:
        climb = max(0.0, (info["alt_next_air"] or ref) - ref)
        t_to = info["t_next_air"] - 20.0 - climb / 2500.0 * 60.0
    row["t_takeoff"] = float(t_to)
    g = (cls == 0) & (t <= t_to)
    tg, la, lo, v = t[g], lat[g].astype(float), lon[g].astype(float), gs[g].astype(float)
    n = len(tg)
    row["n_gnd"] = int(n)
    if n == 0:
        return row
    # Line-up: the last ground point below 40 kt, if the takeoff roll follows it without a gap;
    # otherwise (unseen from the stand to the roll) the first point of the roll.
    slow = np.where(np.isnan(v) | (v < LINEUP_KT))[0]
    li = int(slow[-1]) if len(slow) else n - 1
    if li + 1 < n and tg[li + 1] - tg[li] > 120.0:
        li += 1
    row["lineup_from_roll"] = int(len(slow) == 0 or li != int(slow[-1]))
    lu_lat, lu_lon = la[li], lo[li]
    row.update(t_lineup=float(tg[li]), lat_lineup=float(lu_lat), lon_lineup=float(lu_lon))
    d_lu = dist_v(la, lo, lu_lat, lu_lon)
    # Gaps (any ground points, up to takeoff). A gap >= 300 s across which the aircraft moved
    # more than SEP_M separates the evidence: whatever happened before it (the arrival, a dwell
    # that ended unseen) cannot time this departure's off-block. Silence from the last ground
    # point to takeoff counts as such a gap too.
    dt = np.diff(tg)
    gap_idx = np.where(dt >= GAP_S)[0]
    # A gap is "parked through" when the aircraft was slow before it and reappears within SEP_M;
    # anything else (vanishing at taxi speed at a coverage edge, reappearing elsewhere) separates
    # the evidence. A separating gap of BIG_SEP_S or more hides a turnaround: the stand time
    # sits inside it, so nothing before it can time this departure.
    sep = [int(i) + 1 for i in gap_idx if not parked_through(tg, la, lo, v, i)]
    big = [s for s in sep if dt[s - 1] >= BIG_SEP_S]
    Lb = big[-1] if big else 0
    tail_gap = float(t_to - tg[-1])
    landed = bool(info.get("landed"))
    # Keep the dimensionless ground-point index separate from UTC epoch seconds.
    row.update(tail_gap_s=tail_gap, n_sep_gaps=len(sep), n_big_sep_gaps=len(big),
               dep_evidence_start_idx=int(Lb), t_dep_evidence_start=float(tg[Lb]))
    runs = dwells(tg, la, lo, v)
    row["n_dwells"] = len(runs)

    def away(r):
        return d_lu[r[0]] > PARK_FROM_LINEUP_M and tg[r[1]] < t_to - 60

    def dur(r):
        return tg[r[1]] - tg[r[0]]

    # Park (the stand). The chain is cut into segments at the big separating gaps; each segment
    # may hold a stand: in the first segment of a chain that holds its arrival, the first dwell
    # >= 300 s after the landing; elsewhere the dwell the segment starts with (first seen
    # parked) or one lasting >= 30 min (towed or taxied in, then parked). A shorter dwell entered
    # after taxiing is a de-icing or runway hold, never the stand. The latest segment with a
    # stand supplies the park: a big gap after it hid part of the taxi, not the turnaround. A
    # stand in an earlier segment counts only when it ends within MAX_EARLY_PUSH_S of takeoff
    # (otherwise it was the arrival's, and the push is inside the gap).
    bounds = list(zip([0] + big, big + [n]))

    def stand_in(a, b, arrival):
        reg = [r for r in runs if a <= r[0] < b and away(r)]
        if arrival:
            return next((r for r in reg if r[0] >= 1 and dur(r) >= MIN_PARK_S), None)
        return next((r for r in reg
                     if (tg[r[0]] - tg[a] <= FIRST_PARK_S
                         and bool(np.all(np.isnan(v[a:r[0] + 1]) | (v[a:r[0] + 1] < 1.0))))
                     or dur(r) >= LONG_PARK_S), None)

    park, seg_end = None, n
    for si in range(len(bounds) - 1, -1, -1):
        a, b = bounds[si]
        c = stand_in(a, b, landed and si == 0)
        if c is not None:
            if si == len(bounds) - 1 or t_to - tg[c[1]] <= MAX_EARLY_PUSH_S:
                park, seg_end = c, b
            break
    region = [r for r in runs if r[0] < seg_end and away(r)]
    # A tow or reposition: from the park, walk through the following dwells of its segment while
    # every move between them stays at or below TOW_KT; the last of them lasting LONG_PARK_S or
    # more (and away from the runway) is the stand this departure leaves from (a de-icing or
    # runway hold is reached by taxiing, faster, and is shorter).
    n_tows = 0
    if park is not None:
        cur, hops = park, 0
        for nxt in [r for r in region if r[0] > park[1]]:
            transit = v[cur[1] + 1:nxt[0] + 1]
            if len(transit) and np.nanmax(np.append(transit, 0.0)) > TOW_KT:
                break
            cur, hops = nxt, hops + 1
            if dur(nxt) >= LONG_PARK_S and d_lu[nxt[0]] > TOW_TO_LINEUP_M:
                park, n_tows = nxt, hops
    row["n_tows"] = n_tows
    # Every dwell of the departure side (fs7b): the stand-constrained park choice happens downstream,
    # where the matched flight's recorded stand is known (adsb3_dwells_<TAG>.parquet).
    dw = []
    for r in runs[-DWELL_KEEP:]:
        a_, b_ = r
        dw.append((float(tg[a_]), float(tg[b_]), float(la[a_]), float(lo[a_]), float(np.nanmedian(la[a_:b_ + 1])),
                   float(np.nanmedian(lo[a_:b_ + 1])), float(d_lu[a_]),
                   float(tg[b_ + 1] - tg[b_]) if b_ + 1 < n else np.nan,
                   float(v[b_ + 1]) if b_ + 1 < n else np.nan,
                   float(np.nanmax(np.append(v[b_ + 1:], 0.0))) if b_ + 1 < n else np.nan,
                   int(a_ < Lb), int(park is not None and r == park)))
    row["_dwells"] = dw
    long_runs = [r for r in runs if away(r) and dur(r) >= MIN_PARK_S]
    if long_runs:
        lr = max(long_runs, key=dur)
        row.update(t_longest_dwell_end=float(tg[lr[1]]), longest_dwell_s=float(dur(lr)),
                   longest_dwell_to_lineup_m=float(d_lu[lr[0]]))
    if park is not None:
        i0, i1 = park
        row.update(park_short=int(dur(park) < MIN_PARK_S), t_park_start=float(tg[i0]), t_push=float(tg[i1]),
                   park_s=float(dur(park)), lat_park=float(la[i0]), lon_park=float(lo[i0]),
                   park_to_lineup_m=float(d_lu[i0]), park_gaps=int(((gap_idx >= i0) & (gap_idx < i1)).sum()),
                   gs_push=float(v[i1]))
        if i1 + 1 < n:
            row.update(t_leave=float(tg[i1 + 1]), gs_leave=float(v[i1 + 1]),
                       leave_gap_s=float(tg[i1 + 1] - tg[i1]))
    # Other views: the last dwell before takeoff away from the runway (an upper bound on
    # off-block when it ends unseen) and the first one.
    anyc = [r for r in runs if away(r)]
    if anyc:
        a, b = anyc[-1]
        row.update(t_last_dwell_end=float(tg[b]), last_dwell_s=float(tg[b] - tg[a]),
                   last_dwell_to_lineup_m=float(d_lu[a]))
    if runs and runs[0][0] == 0:
        row.update(t_first_dwell_end=float(tg[runs[0][1]]), first_dwell_s=float(tg[runs[0][1]] - tg[0]))
    # Grade and observed off-block.
    #   A: the park, then seen leaving it (next point within LEAVE_GAP_S); t_obs = end of the park.
    #   B/C: first seen moving (after the park when its end was unseen, else after the last
    #        separating gap, else the chain start when the arrival is not in it): a lower bound
    #        on taxi-out. B more than NEAR_RWY_M from the line-up point, C within it.
    #   D: no departure ground evidence (only the arrival, then silence up to takeoff).
    if park is not None and row.get("leave_gap_s", 1e9) <= LEAVE_GAP_S:
        grade, s0 = "A", park[1]
    else:
        if park is not None:
            s0 = park[1] + 1 if park[1] + 1 < n else None
        elif Lb > 0 or not landed:
            s0 = Lb  # a shorter separating gap is a coverage hole during the taxi, not a reset
        else:
            s0 = None  # only the arrival is visible before the silence that hid the stand
        if s0 is not None and tail_gap >= BIG_SEP_S:
            s0 = None  # the aircraft vanished long before takeoff: what was seen is not this taxi
        if s0 is None:
            grade = "D"
        else:
            grade = "B" if d_lu[s0] > NEAR_RWY_M else "C"
    row["grade"] = grade
    if s0 is not None:
        row.update(t_dep_first=float(tg[s0]), lat_dep_first=float(la[s0]), lon_dep_first=float(lo[s0]),
                   gs_dep_first=float(v[s0]), dep_first_to_lineup_m=float(d_lu[s0]))
        t_obs = float(tg[s0])
        row["t_obs"] = t_obs
        row["obs_taxi_s"] = float(t_to - t_obs)
        # Motion from the observed start to takeoff.
        ts, ls, os_, vs = tg[s0:], la[s0:], lo[s0:], v[s0:]
        if len(ts) >= 2:
            step = dist_v(ls[:-1], os_[:-1], ls[1:], os_[1:])
            dts = np.diff(ts)
            stopped = (np.nan_to_num(vs[:-1], nan=0) < 1.0) & (np.nan_to_num(vs[1:], nan=0) < 1.0) & (step < 10.0)
            near = d_lu[s0 + 1:n] <= NEAR_RWY_M
            mov = vs[~np.isnan(vs) & (vs >= 3.0)]
            row.update(move_dist_m=float(step.sum()), move_time_s=float(t_to - ts[0]),
                       gs_moving_med=float(np.median(mov)) if len(mov) else np.nan,
                       stopped_s=float(dts[stopped].sum()), stopped_near_rwy_s=float(dts[stopped & near].sum()),
                       stopped_apron_s=float(dts[stopped & ~near].sum()),
                       n_stops=int(((~stopped[:-1]) & stopped[1:]).sum()) if len(stopped) > 1 else 0,
                       max_gap_after_obs_s=float(max(dts.max(), tail_gap)))
        seg_dt = np.diff(tg[s0:])
        row["seen_continuous"] = int(tail_gap < 120.0 and (len(seg_dt) == 0 or seg_dt.max() < 120.0))
        span = t_to - tg[s0]
        row["coverage"] = float(seg_dt[seg_dt <= 60.0].sum() / span) if span > 0 else np.nan
    # Chain first seen and gaps.
    row.update(t_first=float(tg[0]), lat_first=float(la[0]), lon_first=float(lo[0]), gs_first=float(v[0]),
               first_to_lineup_m=float(d_lu[0]), max_gap_s=float(max(dt.max() if len(dt) else 0.0, tail_gap)),
               n_gaps_300=int(len(gap_idx)))
    if len(gap_idx):
        gb = int(gap_idx[np.argmax(dt[gap_idx])])
        gl = int(gap_idx[-1])
        for name, gi in (("big", gb), ("last", gl)):
            row.update({f"gap_{name}_t0": float(tg[gi]), f"gap_{name}_t1": float(tg[gi + 1]),
                        f"gap_{name}_lat0": float(la[gi]), f"gap_{name}_lon0": float(lo[gi]),
                        f"gap_{name}_lat1": float(la[gi + 1]), f"gap_{name}_lon1": float(lo[gi + 1]),
                        f"gap_{name}_disp_m": float(dist_m(la[gi], lo[gi], la[gi + 1], lo[gi + 1])),
                        f"gap_{name}_to_lineup_m": float(d_lu[gi + 1])})
        row["gaps"] = ";".join(f"{t_to - tg[i]:.0f}/{dt[i]:.0f}/{dist_m(la[i], lo[i], la[i + 1], lo[i + 1]):.0f}"
                               for i in gap_idx[-8:])
    return row


def arrival_row(P, info):
    """In-block of one landing: the stand the aircraft parks at after touchdown."""
    t, lat, lon, gs, cls = (P[c] for c in ("t", "lat", "lon", "gs", "cls"))
    g = cls == 0
    tg, la, lo, v = t[g], lat[g].astype(float), lon[g].astype(float), gs[g].astype(float)
    n = len(tg)
    row = {"n_gnd_arr": int(n)}
    if n < 2:
        row["grade_arr"] = "D"
        return row
    runs = dwells(tg, la, lo, v)
    # The stand: the first dwell >= 300 s after touchdown, or one that lasts to the end of the
    # piece (still parked when the day's trace ends). Shorter stops are runway crossings or
    # waiting for the stand.
    stand = next((r for r in runs if r[0] >= 1 and (tg[r[1]] - tg[r[0]] >= MIN_PARK_S or r[1] == n - 1)), None)
    dt = np.diff(tg)
    sep = [int(i) for i in np.where(dt >= GAP_S)[0] if not parked_through(tg, la, lo, v, int(i))]
    if stand is not None and not any(i < stand[0] for i in sep):
        i0, i1 = stand
        row.update(grade_arr="A", t_inblock=float(tg[i0]), lat_stand=float(la[i0]), lon_stand=float(lo[i0]),
                   stand_dwell_s=float(tg[i1] - tg[i0]), t_stand_end=float(tg[i1]), stand_to_end=int(i1 == n - 1))
    elif sep:
        # Vanished during the taxi-in: the last point before the first separating gap bounds
        # in-block from below.
        i = sep[0]
        row.update(grade_arr="B", t_inblock=float(tg[i]), lat_stand=float(la[i]), lon_stand=float(lo[i]),
                   gs_vanish=float(v[i]))
    else:
        row.update(grade_arr="D", t_last_seen=float(tg[-1]))
    row["taxi_in_s"] = row.get("t_inblock", np.nan) - float(tg[0])
    return row


# %% driver
PTS = None
CHAINS = None
ARRS = None


def work_arr(idx_list):
    out = []
    for ci in idx_list:
        info, piece = ARRS[ci]
        sel = PTS["_rows"].get(piece)
        if sel is None or len(sel) == 0:
            out.append({**info, "grade_arr": "D", "n_gnd_arr": 0})
            continue
        sel = sel[np.argsort(PTS["t"][sel], kind="stable")]
        P = {c: PTS[c][sel] for c in ("t", "lat", "lon", "gs", "cls")}
        try:
            row = arrival_row(P, info)
        except Exception as exc:  # noqa: BLE001
            row = {"error": str(exc)[:200]}
        out.append({**info, **row})
    return out


def work(idx_list):
    out = []
    for ci in idx_list:
        info, pieces = CHAINS[ci]
        sel = np.concatenate([PTS["_rows"][p] for p in pieces if p in PTS["_rows"]]) if pieces else np.array([], int)
        if len(sel) == 0:
            out.append({**info["keys"], "n_gnd": 0})
            continue
        sel = sel[np.argsort(PTS["t"][sel], kind="stable")]
        P = {c: PTS[c][sel] for c in ("t", "lat", "lon", "gs", "alt", "gnd", "cls")}
        try:
            row = event_row(P, info)
        except Exception as exc:  # noqa: BLE001
            row = {"error": str(exc)[:200]}
        out.append({**info["keys"], **row})
    return out


def load_points(month, prev_month, first_days):
    """The month's points (main file plus any fix-up files) and the previous month's last day."""
    cols = ["day", "hex", "k", "t", "lat", "lon", "gs", "alt", "gnd", "cls"]
    fs = files("adsb3_points")
    parts = [pq.read_table(f, columns=cols) for tag, f in fs.items() if tag[:6] == month or tag == month]
    if prev_month and first_days:
        parts += [pq.read_table(f, columns=cols, filters=[("day", "in", sorted(first_days))])
                  for tag, f in fs.items() if tag[:6] == prev_month]
    tbl = pd.concat([p.to_pandas() for p in parts], ignore_index=True)
    return tbl.drop_duplicates(["day", "hex", "k", "t"])


def run_month(tag, st, deps, arrs, prev_tag):
    global PTS, CHAINS
    t_m = time.time()
    need = st[st["chain"].isin(deps["chain"])]
    first_days = set(need.loc[(need["day"] // 100) != int(tag), "day"].astype(int)) if tag.isdigit() else set()
    pts = load_points(tag, prev_tag, first_days)
    grp = pts.groupby(["day", "hex", "k"], sort=False).indices
    PTS = {c: pts[c].to_numpy() for c in ("t", "lat", "lon", "gs", "alt", "gnd", "cls")}
    PTS["alt"] = PTS["alt"].astype(float)
    PTS["_rows"] = grp
    pieces = need.groupby("chain")[["day", "hex", "k"]].apply(
        lambda g: [(int(d), h, int(k)) for d, h, k in zip(g["day"], g["hex"], g["k"])]).to_dict()
    CHAINS = []
    for r in deps.itertuples(index=False):
        info = {"keys": {"hex": r.hex, "apt": r.apt, "day": int(r.day), "k": int(r.k)},
                "ref_alt": float(r.ref_alt), "took_off": int(r.took_off), "t_next_air": float(r.t_next_air),
                "alt_next_air": None if pd.isna(r.alt_next_air) else float(r.alt_next_air),
                "t_last": float(r.t_last), "landed": int(r.chain_landed)}
        CHAINS.append((info, pieces.get(r.chain, [])))
    idx = list(range(len(CHAINS)))
    chunks = [idx[i:i + 2000] for i in range(0, len(idx), 2000)]
    rows = []
    if NPROC > 1 and len(chunks) > 1:
        with mp.get_context("fork").Pool(NPROC) as pool:
            for part in pool.imap(work, chunks):
                rows.extend(part)
    else:
        for c in chunks:
            rows.extend(work(c))
    dws = []
    for r in rows:
        for k_, d_ in enumerate(r.pop("_dwells", None) or []):
            dws.append((r["hex"], r["apt"], r["day"], r["k"], k_) + d_)
    globals()["DWELLS"] = pd.DataFrame(dws, columns=["hex", "apt", "day", "k", "d_idx"] + DWELL_COLS)
    ev = pd.DataFrame(rows)
    # Arrivals: the landing piece of every chain that starts with a landing in this month.
    global ARRS
    ARRS = [({"hex": r.hex, "apt": r.apt, "day": int(r.day), "k": int(r.k), "chain": int(r.chain)},
             (int(r.day), r.hex, int(r.k))) for r in arrs.itertuples(index=False)]
    aidx = list(range(len(ARRS)))
    achunks = [aidx[i:i + 2000] for i in range(0, len(aidx), 2000)]
    arows = []
    if NPROC > 1 and len(achunks) > 1:
        with mp.get_context("fork").Pool(NPROC) as pool:
            for part in pool.imap(work_arr, achunks):
                arows.extend(part)
    else:
        for c in achunks:
            arows.extend(work_arr(c))
    ar = pd.DataFrame(arows)
    print(f"{tag}: {len(ev):,} departures, {len(ar):,} arrivals, {len(pts):,} points, {time.time() - t_m:.0f}s",
          flush=True)
    return ev, ar


def main():
    sf = files("adsb3_stays")
    if not sf:
        raise SystemExit("no adsb3_stays_*.parquet found under " + ", ".join(roots()))
    st = pd.concat([pd.read_parquet(f) for f in sf.values()], ignore_index=True)
    st = st.drop_duplicates(["day", "hex", "k"])
    print(f"stays: {len(st):,} from {list(sf)}", flush=True)
    st = stitch(st)
    last = st.groupby("chain").tail(1)
    first = st.groupby("chain").head(1).set_index("chain")
    deps = last[last["t_next_air"].notna() & last["alt_next_air"].notna()].copy()
    # Chain-level context from the first piece and the whole chain.
    agg = st.groupby("chain").agg(n_pieces=("k", "size"), chain_n_raw=("n_raw", "sum"),
                                  chain_gnd_flag_frac=("gnd_flag_frac", "mean"), chain_mlat_frac=("mlat_frac", "mean"),
                                  chain_callsigns=("callsigns", lambda s: "|".join(x for x in s if x)))
    ctx = first[["landed", "t_first", "t_prev_air", "alt_prev_air", "dist_prev_air_m", "cs_prev_air",
                 "n_air_before", "day"]].rename(columns=lambda c: f"chain_{c}").join(agg)
    deps = deps.join(ctx, on="chain")
    deps["month"] = (deps["day"] // 100).astype(str)
    arrs = st[(st["landed"] == 1)].copy()  # a landing opens its chain (stitching never links a landed piece)
    arrs["month"] = (arrs["day"] // 100).astype(str)
    tags = sorted({t[:6] for t in sf if t[:6].isdigit()}) or list(sf)
    summary = {}
    out_all = []
    for i, tag in enumerate(tags):
        d = deps[deps["month"] == tag] if tag.isdigit() else deps
        if not len(d):
            continue
        prev_tag = tags[i - 1] if i > 0 else None
        a = arrs[arrs["month"] == tag] if tag.isdigit() else arrs
        ev, ar = run_month(tag, st, d, a, prev_tag)
        if len(ar):
            ar = a[["hex", "apt", "day", "k", "reg", "actype", "t_first", "t_prev_air", "alt_prev_air",
                    "dist_prev_air_m", "cs_prev_air", "cs_first", "callsigns", "open_end"]].merge(
                ar, on=["hex", "apt", "day", "k"], how="left")
            ar = ar.rename(columns={"t_first": "t_touchdown"})
            ar["callsign_arr"] = ar["cs_prev_air"].fillna(ar["cs_first"]).fillna("").astype(str).str.strip().str.upper()
            dep_of_chain = deps.set_index("chain")[["day", "k"]].rename(columns={"day": "dep_day", "k": "dep_k"})
            ar = ar.join(dep_of_chain, on="chain")
            ar.to_parquet(f"adsb3_arrivals_{tag}.parquet", index=False)
            summary.setdefault("arrivals", {})[tag] = ar.groupby(["apt", "grade_arr"]).size().unstack(
                fill_value=0).to_dict(orient="index")
        keep = ["hex", "apt", "day", "k", "chain", "reg", "actype", "category", "took_off", "t_next_air",
                "alt_next_air", "dist_next_air_m", "cs_next_air", "cs_last", "callsigns", "n_pieces",
                "chain_landed", "chain_t_first", "chain_t_prev_air", "chain_alt_prev_air", "chain_dist_prev_air_m",
                "chain_cs_prev_air", "chain_n_air_before", "chain_day", "chain_callsigns", "chain_n_raw",
                "chain_gnd_flag_frac", "chain_mlat_frac"]
        ev = d[keep].merge(ev, on=["hex", "apt", "day", "k"], how="left")
        ev["callsign"] = ev["cs_next_air"].fillna(ev["cs_last"]).fillna("").astype(str).str.strip().str.upper()
        ev["t_touchdown"] = np.where(ev["chain_landed"] == 1, ev["chain_t_first"], np.nan)
        ev["turnaround_s"] = ev["t_obs"] - ev["t_touchdown"]
        ev["takeoff_seen"] = ev["took_off"]
        ev.to_parquet(f"adsb3_events_{tag}.parquet", index=False)
        dw = globals().get("DWELLS")
        if dw is not None and len(dw):
            for c in DWELL_COLS:
                if c not in ("d_t0", "d_t1"):
                    dw[c] = dw[c].astype("float32")
            dw.to_parquet(f"adsb3_dwells_{tag}.parquet", index=False)
            summary.setdefault("dwells", {})[tag] = int(len(dw))
        cnt = ev.groupby(["apt", "grade"]).size().unstack(fill_value=0)
        summary[tag] = {"departures": int(len(ev)), "by_apt_grade": cnt.to_dict(orient="index")}
        print(cnt.to_string(), flush=True)
        out_all.append(ev[["apt", "grade"]])
    if out_all:
        allc = pd.concat(out_all).groupby(["apt", "grade"]).size().unstack(fill_value=0)
        summary["all"] = allc.to_dict(orient="index")
        print("all months\n" + allc.to_string(), flush=True)
    summary["hours"] = round((time.time() - T0) / 3600, 3)
    with open("adsb3_events_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
