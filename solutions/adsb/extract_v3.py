# %%
# PRC 2026 — ADS-B v3: whole ground stays at the 10 challenge airports, from adsb.lol (ODbL-1.0).
#
# Streams each day's readsb globe_history release from GitHub
# (adsblol/globe_history_YYYY, tag vYYYY.MM.DD-planes-readsb-prod-0) and keeps, for every
# aircraft (hex) with ground points inside a box around a challenge airport, its WHOLE ground
# presence there: no reset on gaps, arrivals included, stays with no observed climb included.
#
# Two outputs per calendar month in DATES (a kernel may carry several months; the test kernel
# names its pair "test"):
#   adsb3_points_<YYYYMM>.parquet  thinned ground points, one row per kept point
#   adsb3_stays_<YYYYMM>.parquet   one row per ground stay (hex, airport, landing .. takeoff)
# plus adsb3_log_<TAG>.json (per-day counts, failed or skipped days) for the kernel.
#
# A stay is the time between arriving on the ground at one airport and leaving it. It is closed
# only by evidence that the aircraft left: an airborne point (baro altitude > ground reference +
# 300 ft, or geometric altitude > elevation + 500 ft), or any point outside the airport's box.
# Gaps never split a stay: a transponder switched off at the stand and on again 3 h later at the
# same airport, with no airborne point between, is one stay.
#
# Midnight: each day's release holds that UTC day's trace, so a stay that crosses 00:00Z is
# emitted as two per-day pieces. `open_start` (nothing before the stay in the day's trace but
# ground points here) and `open_end` (the trace ends inside the stay) mark the pieces;
# events_v3.py stitches day D's open_start stay to day D-1's open_end stay of the same hex at
# the same airport. Keys: (day, hex, k), k = stay index within that hex's day.
#
# Data: https://github.com/adsblol (Open Database License 1.0), openly accessible, as the
# challenge's external-data rule requires. Derived tables are ODbL share-alike.

import gzip
import io
import json
import math
import multiprocessing as mp
import os
import re
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

DATES = [date(2025, 1, 15)]  # replaced per kernel
TAG = "test"                 # replaced per kernel
COMPARE_PREFILTER = True     # replaced per kernel (test only): count what the old token prefilter parses
MAX_HOURS = 11.0             # no new day starts past this; Kaggle kills at 12 h and keeps nothing

AIRPORTS = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4706, -0.4619),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LTFM": (41.2753, 28.7519),
    "LSZH": (47.4647, 8.5492),
}
ELEV_FT = {"EDDF": 364, "EDDM": 1487, "EGLL": 83, "EHAM": -11, "LEBL": 12, "LEMD": 2000,
           "LFPG": 392, "LIRF": 13, "LTFM": 325, "LSZH": 1416}
DLAT, DLON = 0.08, 0.12      # the airport box (same as extract.py)
MARGIN_LAT, MARGIN_LON = 0.05, 0.07  # a "ground" point outside box+margin means the aircraft is elsewhere
CLIMB_GAP_S = 600            # last ground point to first airborne point for a takeoff to count as seen
LAND_GAP_S = 900             # last airborne point to first ground point for a landing to count as seen
LOW_SLOW_KT = 50             # without the ground flag: in the box, below this speed and low = on the ground
THIN_MOVE_M = 5.0            # thinning: keep a point that moved this far from the last kept one
THIN_EVERY_S = 60.0          #           or this long after the last kept one
THIN_GS_KT = 2.0             #           or whose speed changed this much
THIN_TRK_DEG = 10.0          #           or whose track/heading turned this much
GAP_KEEP_S = 30.0            #           and both ends of any gap longer than this
NPROC = os.cpu_count() or 1
DAY_PAR = 2                  # days streamed concurrently (download-bound: one stream ran ~5 MB/s)
BATCH = 256
T0 = time.time()

SRC_CODES = {"adsb_icao": 1, "adsb_icao_nt": 2, "adsr_icao": 3, "tisb_icao": 4, "adsc": 5, "mlat": 6,
             "other": 7, "mode_s": 8, "adsb_other": 9, "adsr_other": 10, "tisb_trackfile": 11,
             "tisb_other": 12, "mode_ac": 13}
BOX = {a: (lat - DLAT, lat + DLAT, lon - DLON, lon + DLON) for a, (lat, lon) in AIRPORTS.items()}
BOXM = {a: (lat - DLAT - MARGIN_LAT, lat + DLAT + MARGIN_LAT, lon - DLON - MARGIN_LON, lon + DLON + MARGIN_LON)
        for a, (lat, lon) in AIRPORTS.items()}


def _prefixes(lo, hi):
    out, x = [], math.floor(lo * 10) / 10
    while x <= hi + 1e-9:
        out.append(f"{x:.1f}".replace(".", r"\."))
        x = round(x + 0.1, 1)
    return out


# Exact prefilter: a trace entry is "[t,lat,lon,..." so a point inside a box shows up as
# ",<lat>,<lon>" with both numbers inside the box's 0.1-degree prefixes. Speeds and altitudes
# never sit in that pair position. COMPARE_PREFILTER checks it against the old token filter.
_PAT = b"|".join(
    (r",\s?(?:" + "|".join(_prefixes(b[0], b[1])) + r")\d*,\s?(?:" + "|".join(_prefixes(b[2], b[3])) + r")\d*").encode()
    for b in BOX.values())
PREFILTER = re.compile(_PAT)
OLD_TOKENS = sorted({f"{lat:.1f}".encode() for lat, _ in AIRPORTS.values()}
                    | {f"{lat - 0.1:.1f}".encode() for lat, _ in AIRPORTS.values()}
                    | {f"{lat + 0.1:.1f}".encode() for lat, _ in AIRPORTS.values()})


class ChainedStream(io.RawIOBase):
    """Sequential read over the split tar parts, streamed over HTTP."""

    def __init__(self, urls):
        self.urls = list(urls)
        self.resp = None
        self.it = None
        self.buf = b""

    def _next(self):
        if self.resp is not None:
            self.resp.close()
        if not self.urls:
            return False
        url = self.urls.pop(0)
        for attempt in range(5):
            try:
                self.resp = requests.get(url, stream=True, timeout=120)
                self.resp.raise_for_status()
                self.it = self.resp.iter_content(chunk_size=8 << 20)
                return True
            except Exception as exc:
                print(f"retry {attempt} {url}: {exc}", flush=True)
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"could not fetch {url}")

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            if self.it is None and not self._next():
                return 0
            try:
                self.buf = next(self.it)
            except StopIteration:
                self.it = None
                if not self._next():
                    return 0
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        return n


def part_urls(day):
    repo = f"adsblol/globe_history_{day.year}"
    tag = f"v{day:%Y.%m.%d}-planes-readsb-prod-0"
    for attempt in range(5):
        r = requests.get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", timeout=60)
        if r.status_code == 200:
            return sorted(a["browser_download_url"] for a in r.json()["assets"] if ".tar" in a["name"])
        if r.status_code == 404:
            return []
        print(f"release lookup {tag}: HTTP {r.status_code}", flush=True)
        time.sleep(30 * (attempt + 1))
    return []


def box_of(lat, lon, boxes=BOX):
    for code, (a, b, c, d) in boxes.items():
        if a <= lat <= b and c <= lon <= d:
            return code
    return None


def dist_m(lat1, lon1, lat2, lon2):
    """Equirectangular metres; exact enough inside one airport box."""
    c = math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot((lat2 - lat1) * 110574.0, (lon2 - lon1) * 111320.0 * c)


def ang(a, b):
    if a is None or b is None:
        return 0.0
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


# Point tuple layout inside a stay (raw, before thinning):
#   0 t, 1 lat, 2 lon, 3 alt (baro ft or None), 4 gnd (1 = readsb "ground"), 5 gs, 6 trk, 7 geom,
#   8 flags, 9 src code, 10 callsign, 11 cls (0 ground, 1 low non-ground in box: roll/hover)
def thin(pts):
    n = len(pts)
    keep = [False] * n
    keep[0] = keep[-1] = True
    last = 0
    for i in range(1, n):
        p, q, r = pts[i], pts[last], pts[i - 1]
        change = False
        if p[0] - r[0] >= GAP_KEEP_S:
            keep[i - 1] = True  # both ends of a gap
            change = True
        elif dist_m(q[1], q[2], p[1], p[2]) >= THIN_MOVE_M:
            change = True
        elif abs((p[5] or 0.0) - (q[5] or 0.0)) >= THIN_GS_KT:
            change = True
        elif ang(p[6], q[6]) >= THIN_TRK_DEG and max(p[5] or 0.0, q[5] or 0.0) >= 1.0:
            change = True
        elif p[10] != q[10] or p[11] != q[11] or p[4] != q[4]:
            change = True
        if change:
            # The last point of a quiet stretch is kept too, so the end of a dwell is exact.
            if not keep[i - 1] and r[0] - q[0] >= 10.0:
                keep[i - 1] = True
            keep[i] = True
            last = i
        elif p[0] - q[0] >= THIN_EVERY_S:
            keep[i] = True
            last = i
    return [p for p, k in zip(pts, keep) if k]


def stays_from_trace(j, day_start, day_end):
    """All ground stays of one aircraft-day at the challenge airports: (stay rows, point rows)."""
    base = j.get("timestamp", 0)
    trace = j.get("trace") or []
    hexid = j.get("icao")
    cs = None
    cat = None
    stays = []
    cur = None                 # open stay
    last_air = None            # (t, lat, lon, alt, cs) last airborne/elsewhere point with no stay open
    n_air_since = 0            # airborne/elsewhere points since the previous stay closed (or day start)
    seen_before = False        # any point (of any kind) before the current stay in this day's trace
    n_clipped = 0
    for p in trace:
        t = base + p[0]
        if t < day_start or t >= day_end:
            n_clipped += 1
            continue
        info = p[8] if len(p) > 8 else None
        if isinstance(info, dict):
            if info.get("flight"):
                cs = info["flight"].strip()
            if info.get("category") and cat is None:
                cat = info["category"]
        lat, lon = p[1], p[2]
        if lat is None or lon is None:
            continue
        alt_raw = p[3]
        gnd = 1 if alt_raw == "ground" else 0
        alt = num(alt_raw)
        gs = num(p[4])
        trk = num(p[5])
        flags = p[6] if isinstance(p[6], int) else 0
        src = SRC_CODES.get(p[9], 0) if len(p) > 9 and isinstance(p[9], str) else 0
        geom = num(p[10]) if len(p) > 10 else None
        apt = box_of(lat, lon)
        if apt is not None:
            elev = ELEV_FT[apt]
            if gnd:
                cls = 0
            elif gs is not None and gs <= LOW_SLOW_KT and (
                    (alt is not None and alt <= elev + 1000) or (geom is not None and geom <= elev + 400)):
                cls = 0
            elif gs is None and alt is not None and alt <= elev + 150:
                cls = 0
            else:
                cls = 1  # low and fast (takeoff/landing roll without the ground flag) or airborne; decided below
        else:
            cls = 2
        if cur is not None:
            # Does this point say the aircraft has left the airport of the open stay?
            left = False
            if apt != cur["apt"]:
                if gnd:
                    left = box_of(lat, lon, BOXM) != cur["apt"]  # tolerate box-edge flapping on the ground
                else:
                    left = True
            elif cls == 1:
                ref = cur["ref_alt"]
                if (alt is not None and alt > ref + 300) or (geom is not None and geom > ELEV_FT[apt] + 500):
                    left = True
            if left:
                close_stay(cur, t, lat, lon, alt, cs, departed=True)
                stays.append(cur)
                cur = None
                n_air_since = 1
                last_air = (t, lat, lon, alt, cs)
                seen_before = True
                continue
            cur["pts"].append((t, lat, lon, alt, gnd, gs, trk, geom, flags, src, cs, cls))
            if cls == 0:
                cur["t_last_gnd"] = t
                if alt is not None and not gnd:
                    cur["gnd_alts"].append(alt)
                    if len(cur["gnd_alts"]) in (3, 10, 30):
                        cur["ref_alt"] = float(np.median(cur["gnd_alts"]))
            continue
        # No stay open: a ground point here opens one.
        if apt is not None and cls == 0:
            cur = {"apt": apt, "pts": [], "gnd_alts": [], "ref_alt": float(ELEV_FT[apt]),
                   "t_first": t, "t_last_gnd": t, "k": len(stays), "open_start": int(not seen_before),
                   "n_air_before": n_air_since, "prev_air": last_air}
            cur["pts"].append((t, lat, lon, alt, gnd, gs, trk, geom, flags, src, cs, cls))
            if alt is not None and not gnd:
                cur["gnd_alts"].append(alt)
            continue
        n_air_since += 1
        last_air = (t, lat, lon, alt, cs)
        seen_before = True
    if cur is not None:
        close_stay(cur, None, None, None, None, cs, departed=False)
        stays.append(cur)
    # Airborne points after each stay (before the next one or the day's end).
    for a, b in zip(stays, stays[1:]):
        a["n_air_after"] = b["n_air_before"]
    if stays:
        stays[-1]["n_air_after"] = n_air_since if stays[-1]["open_end"] == 0 else 0
    return finish(stays, j, hexid, cat, n_clipped)


def close_stay(s, t, lat, lon, alt, cs, departed):
    s["open_end"] = int(not departed)
    s["t_next_air"] = t
    s["lat_next_air"], s["lon_next_air"], s["alt_next_air"] = lat, lon, alt
    s["cs_next_air"] = cs


def finish(stays, j, hexid, cat, n_clipped):
    srows, prows = [], []
    for s in stays:
        pts = s["pts"]
        apt = s["apt"]
        alat, alon = AIRPORTS[apt]
        gpts = [p for p in pts if p[11] == 0]
        if not gpts:
            continue
        kept = thin(pts)
        prev_air = s["prev_air"]
        t_first, t_last = gpts[0][0], gpts[-1][0]
        # Landing seen: an airborne point shortly before the first ground point, near the airport.
        landed = 0
        if prev_air is not None and t_first - prev_air[0] <= LAND_GAP_S and \
                dist_m(prev_air[1], prev_air[2], alat, alon) <= 30000:
            landed = 1
        took_off = 0
        if s["t_next_air"] is not None and s["t_next_air"] - t_last <= CLIMB_GAP_S and \
                dist_m(s["lat_next_air"], s["lon_next_air"], alat, alon) <= 30000 and s["alt_next_air"] is not None:
            took_off = 1
        gaps = [b[0] - a[0] for a, b in zip(pts, pts[1:])]
        cs_list = []
        for p in pts:
            if p[10] and (not cs_list or cs_list[-1] != p[10]):
                cs_list.append(p[10])
        if s["cs_next_air"] and (not cs_list or cs_list[-1] != s["cs_next_air"]):
            cs_list.append(s["cs_next_air"])
        srows.append({
            "hex": hexid, "k": s["k"], "apt": apt, "reg": j.get("r"), "actype": j.get("t"),
            "desc": j.get("desc"), "category": cat,
            "t_first": t_first, "t_last": t_last, "t_last_any": pts[-1][0],
            "lat_first": gpts[0][1], "lon_first": gpts[0][2], "lat_last": gpts[-1][1], "lon_last": gpts[-1][2],
            "gs_first": gpts[0][5], "gs_last": gpts[-1][5],
            "landed": landed,
            "t_prev_air": prev_air[0] if prev_air else None,
            "alt_prev_air": prev_air[3] if prev_air else None,
            "dist_prev_air_m": dist_m(prev_air[1], prev_air[2], alat, alon) if prev_air else None,
            "cs_prev_air": prev_air[4] if prev_air else None,
            "took_off": took_off,
            "t_next_air": s["t_next_air"], "alt_next_air": s["alt_next_air"],
            "dist_next_air_m": (dist_m(s["lat_next_air"], s["lon_next_air"], alat, alon)
                                if s["t_next_air"] is not None else None),
            "cs_next_air": s["cs_next_air"],
            "cs_first": gpts[0][10], "cs_last": gpts[-1][10], "callsigns": "|".join(cs_list),
            "open_start": s["open_start"], "open_end": s["open_end"],
            "n_air_before": s["n_air_before"], "n_air_after": s.get("n_air_after", 0),
            "n_raw": len(pts), "n_gnd": len(gpts), "n_kept": len(kept),
            "gnd_flag_frac": sum(p[4] for p in gpts) / len(gpts),
            "mlat_frac": sum(p[9] == 6 for p in gpts) / len(gpts),
            "max_gap_s": max(gaps) if gaps else 0.0,
            "n_gaps_300": sum(g >= 300 for g in gaps),
            "ref_alt": s["ref_alt"], "n_clipped": n_clipped,
        })
        for p in kept:
            prows.append((hexid, s["k"], apt, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                          p[10], p[11]))
    return srows, prows


POINT_COLS = ["hex", "k", "apt", "t", "lat", "lon", "alt", "gnd", "gs", "trk", "geom", "flags", "src", "cs", "cls"]


def trace_rows(raw, day_start, day_end):
    """Decompress, prefilter and extract one trace; runs in a worker process."""
    stat = {"traces": 1, "pre_new": 0, "pre_old": 0, "old_only_box": 0, "vehicles": 0}
    try:
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        new = PREFILTER.search(raw) is not None
        stat["pre_new"] = int(new)
        if COMPARE_PREFILTER:
            old = any(tok in raw for tok in OLD_TOKENS)
            stat["pre_old"] = int(old)
            if old and not new:
                j = json.loads(raw)
                base = j.get("timestamp", 0)
                if any(p[1] is not None and p[2] is not None and box_of(p[1], p[2])
                       and day_start <= base + p[0] < day_end for p in (j.get("trace") or [])):
                    stat["old_only_box"] = 1
        if not new:
            return [], [], stat
        j = json.loads(raw)
        srows, prows = stays_from_trace(j, day_start, day_end)
        # Surface vehicles and obstacles (emitter category C*) are not aircraft.
        if srows and (srows[0]["category"] or "").startswith("C"):
            stat["vehicles"] = 1
            return [], [], stat
        return srows, prows, stat
    except Exception as exc:  # a malformed trace must not kill the day
        stat["error"] = 1
        stat["error_msg"] = str(exc)[:200]
        return [], [], stat


def trace_batch(args):
    raws, day_start, day_end = args
    S, P, st = [], [], {}
    for raw in raws:
        s, p, stat = trace_rows(raw, day_start, day_end)
        S.extend(s)
        P.extend(p)
        for k, v in stat.items():
            if k == "error_msg":
                st.setdefault("error_msgs", []).append(v)
            else:
                st[k] = st.get(k, 0) + v
    pdf = pd.DataFrame(P, columns=POINT_COLS)
    return S, pdf, st


POINT_SCHEMA = pa.schema([("day", pa.int32()), ("hex", pa.string()), ("k", pa.int16()), ("apt", pa.string()),
                          ("t", pa.float64()), ("lat", pa.float32()), ("lon", pa.float32()), ("alt", pa.float32()),
                          ("gnd", pa.int8()), ("gs", pa.float32()), ("trk", pa.float32()), ("geom", pa.float32()),
                          ("flags", pa.int16()), ("src", pa.int8()), ("cs", pa.string()), ("cls", pa.int8())])


def typed_points(pdf, day_int):
    pdf = pdf.copy()
    pdf.insert(0, "day", np.int32(day_int))
    for c, dt in (("k", "int16"), ("t", "float64"), ("lat", "float32"), ("lon", "float32"), ("alt", "float32"),
                  ("gnd", "int8"), ("gs", "float32"), ("trk", "float32"), ("geom", "float32"), ("flags", "int16"),
                  ("src", "int8"), ("cls", "int8")):
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce").astype(dt)
    for c in ("hex", "apt", "cs"):
        pdf[c] = pdf[c].astype("object").where(pdf[c].notna(), None)
    return pa.Table.from_pandas(pdf, schema=POINT_SCHEMA, preserve_index=False)


def process_day(day, pool):
    urls = part_urls(day)
    if not urls:
        print(f"{day}: no release", flush=True)
        return None, None, {"day": str(day), "status": "no_release"}
    day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
    day_end = day_start + 86400
    S, P, stats = [], [], {}
    n_bytes = 0
    t_day = time.time()
    pending, batch = [], []

    def collect(res):
        s, p, st = res
        S.extend(s)
        if len(p):
            P.append(p)
        for k, v in st.items():
            if k == "error_msgs":
                stats.setdefault("error_msgs", []).extend(v[:3])
            else:
                stats[k] = stats.get(k, 0) + v

    with tarfile.open(fileobj=io.BufferedReader(ChainedStream(urls), buffer_size=16 << 20), mode="r|") as tar:
        for m in tar:
            if not (m.isfile() and "trace_full_" in m.name):
                continue
            raw = tar.extractfile(m).read()
            n_bytes += len(raw)
            batch.append(raw)
            if len(batch) >= BATCH:
                pending.append(pool.apply_async(trace_batch, ((batch, day_start, day_end),)))
                batch = []
                while len(pending) > 4 * NPROC:
                    collect(pending.pop(0).get())
    if batch:
        pending.append(pool.apply_async(trace_batch, ((batch, day_start, day_end),)))
    for r in pending:
        collect(r.get())
    day_int = int(f"{day:%Y%m%d}")
    sdf = pd.DataFrame(S)
    if len(sdf):
        sdf.insert(0, "day", np.int32(day_int))
    pdf = typed_points(pd.concat(P, ignore_index=True) if P else pd.DataFrame(columns=POINT_COLS), day_int)
    n_pts = pdf.num_rows
    stats.update({"day": str(day), "status": "ok", "bytes": n_bytes, "stays": len(sdf), "points": n_pts,
                  "seconds": round(time.time() - t_day, 1)})
    if len(sdf):
        stats["departures_seen"] = int(sdf["took_off"].sum())
        stats["landings_seen"] = int(sdf["landed"].sum())
        stats["stays_by_apt"] = sdf["apt"].value_counts().to_dict()
        stats["deps_by_apt"] = sdf.loc[sdf["took_off"] == 1, "apt"].value_counts().to_dict()
    print(f"{day}: {stats.get('traces', 0):,} traces, {n_bytes / 1e9:.2f} GB, prefilter {stats.get('pre_new', 0):,}"
          f" (old {stats.get('pre_old', 0):,}, old-only with a box point {stats.get('old_only_box', 0)}),"
          f" {len(sdf):,} stays, {stats.get('departures_seen', 0):,} takeoffs, {stats.get('landings_seen', 0):,}"
          f" landings, {n_pts:,} points, {stats['seconds']:.0f}s, total {(time.time() - T0) / 3600:.2f}h",
          flush=True)
    return sdf, pdf, stats


STAY_TYPES = {"k": "int16", "landed": "int8", "took_off": "int8", "open_start": "int8", "open_end": "int8",
              "n_air_before": "int32", "n_air_after": "int32", "n_raw": "int32", "n_gnd": "int32", "n_kept": "int32",
              "n_gaps_300": "int16", "n_clipped": "int32"}
STAY_FLOATS = ["t_first", "t_last", "t_last_any", "lat_first", "lon_first", "lat_last", "lon_last", "gs_first",
               "gs_last", "t_prev_air", "alt_prev_air", "dist_prev_air_m", "t_next_air", "alt_next_air",
               "dist_next_air_m", "gnd_flag_frac", "mlat_frac", "max_gap_s", "ref_alt"]
STAY_STRS = ["hex", "apt", "reg", "actype", "desc", "category", "cs_prev_air", "cs_next_air", "cs_first", "cs_last",
             "callsigns"]


STAY_SCHEMA = pa.schema([("day", pa.int32())]
                        + [(c, pa.string()) for c in STAY_STRS]
                        + [(c, getattr(pa, t)()) for c, t in STAY_TYPES.items()]
                        + [(c, pa.float64()) for c in STAY_FLOATS])


def typed_stays(sdf):
    for c, dt in STAY_TYPES.items():
        sdf[c] = pd.to_numeric(sdf[c], errors="coerce").fillna(0).astype(dt)
    for c in STAY_FLOATS:
        sdf[c] = pd.to_numeric(sdf[c], errors="coerce").astype("float64")
    for c in STAY_STRS:
        sdf[c] = sdf[c].astype("object").where(sdf[c].notna(), None)
    return pa.Table.from_pandas(sdf[STAY_SCHEMA.names], schema=STAY_SCHEMA, preserve_index=False)


def merge_parts(parts, name):
    """Stream per-day part files into one parquet file (row group per day), then drop the parts."""
    writer = None
    n = 0
    for f in parts:
        tbl = pq.read_table(f)
        if writer is None:
            writer = pq.ParquetWriter(name, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        n += tbl.num_rows
    if writer is not None:
        writer.close()
    for f in parts:
        os.remove(f)
    return n


# %%
if __name__ == "__main__":
    os.makedirs("parts", exist_ok=True)
    pool = mp.Pool(NPROC)
    log = {"tag": TAG, "dates": [str(d) for d in DATES], "days": [], "skipped": []}
    day_secs = []
    lock = threading.Lock()

    def run_day(day):
        with lock:
            used_h = (time.time() - T0) / 3600
            proj_h = (1.3 * DAY_PAR * max(day_secs) / 3600) if day_secs else 0.5
        if used_h + proj_h > MAX_HOURS:
            print(f"RUNTIME_BUDGET: stage={day} used_h={used_h:.2f} projected_h={used_h + proj_h:.2f} -> skipped",
                  flush=True)
            with lock:
                log["skipped"].append(str(day))
            return
        stats = {"day": str(day), "status": "failed"}
        for attempt in range(3):  # a stream broken mid-tar restarts the whole day
            try:
                t_try = time.time()
                sdf, pdf, stats = process_day(day, pool)
                if sdf is not None:
                    if len(sdf):
                        pq.write_table(typed_stays(sdf), f"parts/stays_{day:%Y%m%d}.parquet", compression="zstd")
                        pq.write_table(pdf, f"parts/points_{day:%Y%m%d}.parquet", compression="zstd")
                    with lock:
                        day_secs.append((time.time() - t_try) / DAY_PAR)
                break
            except Exception as exc:
                print(f"{day}: FAILED attempt {attempt + 1}: {exc}", flush=True)
                stats = {"day": str(day), "status": "failed", "error": str(exc)[:300]}
                time.sleep(60)
        stats["attempts"] = attempt + 1
        with lock:
            log["days"].append(stats)
            with open(f"adsb3_log_{TAG}.json", "w") as fh:
                json.dump(log, fh, indent=1, default=str)

    # DAY_PAR days stream at once: each its own tar stream, all feeding one worker pool.
    with ThreadPoolExecutor(max_workers=DAY_PAR) as ex:
        list(ex.map(run_day, DATES))
    log["days"].sort(key=lambda s: s["day"])
    pool.close()
    # One output pair per calendar month (a kernel may carry several months); the test keeps its TAG.
    months = sorted({f[len("stays_"):len("stays_") + 6] for f in os.listdir("parts") if f.startswith("stays_")})
    log["outputs"] = {}
    for mo in months:
        # A fix-up kernel (TAG "fix...") re-runs failed days; its files carry the TAG so they
        # sit beside the month's main files instead of replacing them.
        out_tag = TAG if TAG == "test" else (f"{mo}{TAG}" if TAG.startswith("fix") else mo)
        sp = sorted(f"parts/{f}" for f in os.listdir("parts") if f.startswith(f"stays_{mo}"))
        pp = sorted(f"parts/{f}" for f in os.listdir("parts") if f.startswith(f"points_{mo}"))
        ns = merge_parts(sp, f"adsb3_stays_{out_tag}.parquet")
        npt = merge_parts(pp, f"adsb3_points_{out_tag}.parquet")
        log["outputs"][out_tag] = {"stays": ns, "points": npt}
    log["hours"] = round((time.time() - T0) / 3600, 2)
    with open(f"adsb3_log_{TAG}.json", "w") as fh:
        json.dump(log, fh, indent=1, default=str)
    size = {f: os.path.getsize(f) for f in os.listdir(".") if f.startswith("adsb3_")}
    print(f"wrote {log['outputs']}; files {size}; {log['hours']} h", flush=True)
