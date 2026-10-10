# %%
# PRC 2026 — ADS-B departure ground segments from adsb.lol history (ODbL-1.0).
#
# Streams each day's readsb globe_history release from GitHub
# (adsblol/globe_history_YYYY, tag vYYYY.MM.DD-planes-readsb-prod-0), keeps
# aircraft seen on the ground inside a box around the 10 challenge airports,
# and reduces every departure to one row: first ground point, first movement,
# first taxi speed, takeoff, stopped time and ground distance.  Output:
# adsb_departures_<first>_<last>.parquet in the working directory.
#
# Data: https://github.com/adsblol (Open Database License 1.0), openly
# accessible, as the challenge's external-data rule requires.

import gzip
import io
import json
import multiprocessing as mp
import math
import os
import tarfile
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

DATES = [date(2025, 1, 15)]  # replaced per kernel
AIRPORTS = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4706, -0.4619),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LTFM": (41.2753, 28.7519),
    "LSZH": (47.4647, 8.5492),
}
ELEV_FT = {"EDDF": 364, "EDDM": 1487, "EGLL": 83, "EHAM": -11, "LEBL": 12, "LEMD": 2000,
           "LFPG": 392, "LIRF": 13, "LTFM": 325, "LSZH": 1416}
DLAT, DLON = 0.08, 0.12
DWELL_S = 300          # stationary this long, away from the runway, reads as a stand dwell
RUNWAY_RADIUS_M = 1500 # holds within this distance of the last ground point are queue/runway holds
GAP_S = 900
NPROC = os.cpu_count() or 1  # traces decompress and parse in parallel; the tar stream stays serial
BATCH = 256
CLIMB_GAP_S = 600    # last ground point to first point above ELEV+300 ft (180 lost 10% of EGLL)
T0 = time.time()


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
            assets = sorted(a["browser_download_url"] for a in r.json()["assets"] if ".tar" in a["name"])
            return assets
        print(f"release lookup {tag}: HTTP {r.status_code}", flush=True)
        time.sleep(30 * (attempt + 1))
    return []


AIRPORT_TOKENS = sorted({f"{lat:.1f}".encode() for lat, _ in AIRPORTS.values()}
                        | {f"{lat - 0.1:.1f}".encode() for lat, _ in AIRPORTS.values()}
                        | {f"{lat + 0.1:.1f}".encode() for lat, _ in AIRPORTS.values()})


def haversine_m(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742000 * math.asin(math.sqrt(a))


def airport_of(lat, lon):
    for code, (alat, alon) in AIRPORTS.items():
        if abs(lat - alat) <= DLAT and abs(lon - alon) <= DLON:
            return code
    return None


def is_ground(alt, gs, apt):
    if alt == "ground":
        return True
    if isinstance(alt, (int, float)) and apt is not None:
        return alt <= ELEV_FT[apt] + 150 and (gs is None or gs <= 60)
    return False


def summarise(pts, takeoff_t, apt, j, callsign):
    """pts: [(t, lat, lon, gs)] ground points at one airport ending at takeoff."""
    t_last, lat_l, lon_l = pts[-1][0], pts[-1][1], pts[-1][2]
    # Stationary runs; the last long one away from the runway ends the stand dwell.
    runs, k = [], 0
    while k < len(pts):
        j0 = k
        while (k + 1 < len(pts) and (pts[k + 1][3] or 0) < 1.0
               and haversine_m(pts[j0][1], pts[j0][2], pts[k + 1][1], pts[k + 1][2]) < 25):
            k += 1
        if k > j0:
            runs.append((j0, k))
        k += 1
    stand_end = 0
    for a, b in runs:
        if (pts[b][0] - pts[a][0] >= DWELL_S
                and haversine_m(pts[a][1], pts[a][2], lat_l, lon_l) > RUNWAY_RADIUS_M):
            stand_end = b
    gap_start = 0
    for k in range(1, len(pts)):
        if pts[k][0] - pts[k - 1][0] >= 180:
            gap_start = k
    start = max(stand_end, gap_start)
    seg = pts[start:]
    lat0, lon0 = seg[0][1], seg[0][2]
    t_move = t_taxi = None
    dist = stopped = 0.0
    n_stops = 0
    moving = False
    for k, p in enumerate(seg):
        gs = p[3] or 0.0
        if t_move is None and (gs >= 2.5 or haversine_m(lat0, lon0, p[1], p[2]) > 30):
            t_move = p[0]
        if t_taxi is None and gs >= 8:
            t_taxi = p[0]
        if k:
            q = seg[k - 1]
            dist += haversine_m(q[1], q[2], p[1], p[2])
            if t_move is not None and gs < 1.0 and (q[3] or 0) < 1.0:
                stopped += p[0] - q[0]
            if moving and gs < 1.0:
                n_stops += 1
        moving = gs >= 3.0
    # Whole-track view: the segment above starts after the last gap or stand dwell, which on
    # long taxi-outs drops most of the ground time (median 268 s kept of 2,106 s).
    lat_f, lon_f = pts[0][1], pts[0][2]
    t_move_all = None
    dist_all = stopped_all = 0.0
    max_gap = 0.0
    n_gaps = 0
    for k, p in enumerate(pts):
        gs = p[3] or 0.0
        if t_move_all is None and (gs >= 2.5 or haversine_m(lat_f, lon_f, p[1], p[2]) > 30):
            t_move_all = p[0]
        if k:
            q = pts[k - 1]
            dt = p[0] - q[0]
            dist_all += haversine_m(q[1], q[2], p[1], p[2])
            max_gap = max(max_gap, dt)
            n_gaps += dt >= 180
            if t_move_all is not None and gs < 1.0 and (q[3] or 0) < 1.0:
                stopped_all += dt
    whole = {
        "t_move_all": t_move_all, "lat_first": lat_f, "lon_first": lon_f,
        "first_to_last_m": haversine_m(lat_f, lon_f, lat_l, lon_l), "dist_all_m": dist_all,
        "stopped_all_s": stopped_all, "max_gap_s": max_gap, "n_gaps": n_gaps,
        "dwell_first_s": (t_move_all - pts[0][0]) if t_move_all is not None else None,
    }
    return {
        **whole,
        "hex": j.get("icao"), "reg": j.get("r"), "actype": j.get("t"),
        "callsign": (callsign or "").strip(), "apt": apt,
        "t_first": pts[0][0], "t_gap_start": pts[gap_start][0], "t_stand_end": pts[stand_end][0],
        "t_start": seg[0][0], "t_move": t_move, "t_taxi": t_taxi, "t_takeoff": takeoff_t,
        "lat_start": lat0, "lon_start": lon0, "lat_last": lat_l, "lon_last": lon_l,
        "start_to_last_m": haversine_m(lat0, lon0, lat_l, lon_l),
        "ground_dist_m": dist, "stopped_s": stopped, "n_stops": n_stops,
        "n_points": len(pts), "n_points_seg": len(seg), "had_stand_dwell": int(stand_end > 0),
    }


def departures_from_trace(j):
    """One row per departure: ground points at an airport followed by a climb-out."""
    base = j.get("timestamp", 0)
    trace = j.get("trace") or []
    callsign = None
    pts, apt_cur, prev_t = [], None, None
    out = []
    for p in trace:
        t = base + p[0]
        lat, lon, alt, gs = p[1], p[2], p[3], p[4]
        info = p[8] if len(p) > 8 else None
        if isinstance(info, dict) and info.get("flight"):
            callsign = info["flight"]
        if lat is None or lon is None:
            continue
        apt = airport_of(lat, lon)
        if prev_t is not None and t - prev_t > GAP_S:
            pts, apt_cur = [], None
        prev_t = t
        if apt and is_ground(alt, gs, apt):
            if apt_cur != apt:
                pts, apt_cur = [], apt
            pts.append((t, lat, lon, gs))
        elif pts and isinstance(alt, (int, float)) and apt_cur is not None:
            if alt > ELEV_FT[apt_cur] + 300 and t - pts[-1][0] <= CLIMB_GAP_S:
                if len(pts) >= 2:
                    row = summarise(pts, pts[-1][0], apt_cur, j, callsign)
                    row["t_air"], row["air_gap_s"] = t, t - pts[-1][0]
                    out.append(row)
                pts, apt_cur = [], None
    return out


def trace_rows(raw):
    """Decompress, prefilter and summarise one trace; runs in a worker process."""
    try:
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        # Cheap prefilter: only traces that pass near a challenge airport.
        if not any(tok in raw for tok in AIRPORT_TOKENS):
            return []
        return departures_from_trace(json.loads(raw))
    except Exception:
        return []


def trace_batch(raws):
    out = []
    for raw in raws:
        out.extend(trace_rows(raw))
    return out


def process_day(day, pool=None):
    urls = part_urls(day)
    if not urls:
        print(f"{day}: no release", flush=True)
        return pd.DataFrame()
    rows, n_traces, n_bytes = [], 0, 0
    t_day = time.time()
    pending, batch = [], []
    with tarfile.open(fileobj=io.BufferedReader(ChainedStream(urls), buffer_size=16 << 20), mode="r|") as tar:
        for m in tar:
            if not (m.isfile() and "trace_full_" in m.name):
                continue
            raw = tar.extractfile(m).read()
            n_bytes += len(raw)
            n_traces += 1
            if pool is None:
                rows.extend(trace_rows(raw))
                continue
            batch.append(raw)
            if len(batch) >= BATCH:
                pending.append(pool.apply_async(trace_batch, (batch,)))
                batch = []
                # Bound memory: keep at most 4 batches per worker in flight.
                while len(pending) > 4 * NPROC:
                    rows.extend(pending.pop(0).get())
    if pool is not None:
        if batch:
            pending.append(pool.apply_async(trace_batch, (batch,)))
        for r in pending:
            rows.extend(r.get())
    df = pd.DataFrame(rows)
    print(f"{day}: {n_traces:,} traces, {n_bytes / 1e9:.2f} GB, {len(df):,} departures, "
          f"{time.time() - t_day:.0f}s, total {(time.time() - T0) / 3600:.2f}h", flush=True)
    return df


# %%
if __name__ == "__main__":
    frames = []
    pool = mp.Pool(NPROC) if NPROC > 1 else None
    name = f"adsb_departures_{DATES[0]:%Y%m%d}_{DATES[-1]:%Y%m%d}.parquet"
    for day in DATES:
        for attempt in range(3):  # a stream broken mid-tar restarts the whole day
            try:
                frames.append(process_day(day, pool))
                break
            except Exception as exc:
                print(f"{day}: FAILED attempt {attempt + 1}: {exc}", flush=True)
                time.sleep(60)
        done = [f for f in frames if len(f)]
        if done:  # checkpoint after every day
            pd.concat(done, ignore_index=True).to_parquet(name, index=False)
    out = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
    out.to_parquet(name, index=False)
    print(f"wrote {name}: {len(out):,} rows; by airport {out['apt'].value_counts().to_dict() if len(out) else {}}", flush=True)
