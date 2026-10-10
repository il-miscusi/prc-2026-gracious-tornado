# ==== features.py ====
# Shared feature code for the PRC 2026 taxi-out campaign (from v26's solution.py).
#
# build() makes one row per departure from fields the ranking file publishes; the
# encodings are leave-one-month-out for training rows and fitted on every fit row for
# applied rows.  Kernels embed this file (solutions/core/build_kernel.py).
from __future__ import annotations

import csv
import json
import os
import sys
import time

# %%
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
SEED = 2026
ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
VAL_MONTHS = (1, 7)
CLIP_TARGET = 7200.0
TE_CLIP = 3600.0
RARE_MIN = 30
MAX_ROUNDS = int(os.environ.get("COMPBOT_ROUNDS", "5000"))
RUNTIME_CAP_H = 12.0
OUTPUT_DIR = Path.cwd()
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
TZ = {
    "EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam", "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris", "LIRF": "Europe/Rome", "LTFM": "Europe/Istanbul",
    "LSZH": "Europe/Zurich",
}


def used_h() -> float:
    return (time.time() - T0) / 3600.0


def budget(stage: str, projected_stage_h: float) -> bool:
    projected = used_h() + projected_stage_h
    print(f"RUNTIME_BUDGET: stage={stage} used_h={used_h():.3f} projected_h={projected:.3f}", flush=True)
    return projected < RUNTIME_CAP_H * 0.85


def resolve_data_dir() -> Path:
    explicit = os.environ.get("DATA_ROOT")
    if explicit:
        # An explicit root is never silently replaced: a missing smoke slice
        # once fell through to the full data and trained on it locally.
        hits = [h.parent for h in sorted(Path(explicit).rglob("ranking.parquet"))
                if list(h.parent.glob("training_*.parquet"))]
        if not hits:
            raise FileNotFoundError(f"DATA_ROOT={explicit} holds no ranking.parquet beside training files")
        return hits[0]
    roots = []
    roots += [
        Path("/kaggle/input"),
        Path.cwd(),
        # <redacted: development machine data path>
    ]
    for root in roots:
        if root.exists():
            for hit in sorted(root.rglob("ranking.parquet")):
                if list(hit.parent.glob("training_*.parquet")):
                    return hit.parent
    raise FileNotFoundError("no directory holds ranking.parquet beside training_*.parquet")


# %%
# Event-window helpers.  Every count is over one key (airport or airport|runway)
# and uses searchsorted on that key's sorted event times.
def _by_key(q_key, r_key):
    q_codes, uniq = pd.factorize(pd.Series(q_key), sort=False)
    r_codes = pd.Categorical(r_key, categories=uniq).codes
    for code in range(len(uniq)):
        qi = np.flatnonzero(q_codes == code)
        ri = np.flatnonzero(r_codes == code)
        if len(qi):
            yield qi, ri


def count_between(q_key, q_lo, q_hi, r_key, r_t):
    """Events r with q_lo <= r_t < q_hi, same key.  NaN bounds give NaN."""
    out = np.full(len(q_key), np.nan)
    ok_r = ~np.isnan(r_t)
    r_key, r_t = np.asarray(r_key)[ok_r], r_t[ok_r]
    for qi, ri in _by_key(q_key, r_key):
        ts = np.sort(r_t[ri])
        lo, hi = q_lo[qi], q_hi[qi]
        good = ~(np.isnan(lo) | np.isnan(hi))
        res = np.full(len(qi), np.nan)
        res[good] = np.searchsorted(ts, hi[good], "left") - np.searchsorted(ts, lo[good], "left")
        out[qi] = res
    return out


def mean_between(q_key, q_lo, q_hi, r_key, r_t, r_v):
    out = np.full(len(q_key), np.nan)
    ok_r = ~(np.isnan(r_t) | np.isnan(r_v))
    r_key, r_t, r_v = np.asarray(r_key)[ok_r], r_t[ok_r], r_v[ok_r]
    for qi, ri in _by_key(q_key, r_key):
        order = np.argsort(r_t[ri], kind="stable")
        ts, vs = r_t[ri][order], r_v[ri][order]
        cs = np.concatenate([[0.0], np.cumsum(vs)])
        lo, hi = q_lo[qi], q_hi[qi]
        good = ~(np.isnan(lo) | np.isnan(hi))
        a = np.searchsorted(ts, lo[good], "left")
        b = np.searchsorted(ts, hi[good], "left")
        n = b - a
        res = np.full(len(qi), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            res[good] = np.where(n > 0, (cs[b] - cs[a]) / np.maximum(n, 1), np.nan)
        out[qi] = res
    return out


def stand_neighbours(q_key, q_t, r_key, r_t, r_attrs: dict, lags=(1, 2)):
    """For each query, the r events at the same key: last `lags` at or before q_t, and the next after."""
    n = len(q_key)
    out = {f"t_prev{l}": np.full(n, np.nan) for l in lags}
    out["t_next"] = np.full(n, np.nan)
    for name in r_attrs:
        for l in lags:
            out[f"{name}_prev{l}"] = np.full(n, None, dtype=object)
    ok_r = ~np.isnan(r_t)
    r_key, r_t = np.asarray(r_key)[ok_r], r_t[ok_r]
    attrs = {k: np.asarray(v, dtype=object)[ok_r] for k, v in r_attrs.items()}
    for qi, ri in _by_key(q_key, r_key):
        order = ri[np.argsort(r_t[ri], kind="stable")]
        ts = r_t[order]
        pos = np.searchsorted(ts, q_t[qi], "right")
        for l in lags:
            j = pos - l
            good = j >= 0
            out[f"t_prev{l}"][qi[good]] = ts[j[good]]
            for name, arr in attrs.items():
                out[f"{name}_prev{l}"][qi[good]] = arr[order][j[good]]
        good = pos < len(ts)
        out["t_next"][qi[good]] = ts[pos[good]]
    return out


WX_COLS = ["temp_c", "spread_c", "wind_kt", "gust_kt", "vis_km", "ceil_ft", "precip_mm", "wx_snow",
           "wx_freezing", "wx_rain", "wx_fog", "wx_thunder", "wx_shower", "deice_likely"]


def load_metar() -> pd.DataFrame | None:
    roots = [Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob("metar_prc2026.parquet"))
            if hits:
                m = pd.read_parquet(hits[0])
                m["t"] = (pd.to_datetime(m["valid_utc"], utc=True) - EPOCH).dt.total_seconds()
                out = []
                for st, g in m.sort_values("t").groupby("station"):
                    g = g.copy()
                    idx = pd.to_datetime(g["valid_utc"], utc=True)
                    roll = g.set_index(idx)
                    g["snow_3h"] = roll["wx_snow"].rolling("3h").max().to_numpy()
                    g["deice_3h"] = roll["deice_likely"].rolling("3h").max().to_numpy()
                    g["tmin_6h"] = roll["temp_c"].rolling("6h").min().to_numpy()
                    out.append(g)
                print(f"metar: {hits[0]} ({len(m):,} reports)", flush=True)
                return pd.concat(out, ignore_index=True)
    print("metar: not found; weather features skipped", flush=True)
    return None


METAR = None


def weather_features(apt: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    """As-of METAR (latest report at or before t) per airport."""
    cols = WX_COLS + ["snow_3h", "deice_3h", "tmin_6h"]
    out = pd.DataFrame(np.nan, index=np.arange(len(t)), columns=cols)
    if METAR is None:
        return out
    q = pd.DataFrame({"apt": apt, "t": t, "i": np.arange(len(t))}).dropna(subset=["t"]).sort_values("t")
    for st, g in METAR.groupby("station"):
        qq = q[q["apt"] == st]
        if qq.empty:
            continue
        m = pd.merge_asof(qq, g[["t"] + cols].sort_values("t"), on="t", direction="backward",
                          tolerance=3 * 3600.0)
        out.loc[m["i"].to_numpy(), cols] = m[cols].to_numpy()
    return out


ARP = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4706, -0.4619),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LTFM": (41.2753, 28.7519),
    "LSZH": (47.4647, 8.5492),
}
ADSB = None
ADSB_RAW = ["t_first", "t_gap_start", "t_stand_end", "t_start", "t_move", "t_taxi", "t_takeoff",
            "stopped_s", "n_stops", "ground_dist_m", "start_to_last_m", "n_points", "n_points_seg",
            "had_stand_dwell", "lat_start", "lon_start",
            "t_move_all", "lat_first", "lon_first", "first_to_last_m", "dist_all_m", "stopped_all_s",
            "max_gap_s", "n_gaps", "dwell_first_s", "air_gap_s"]


def load_adsb() -> pd.DataFrame | None:
    roots = [Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.exists():
            # Re-extracted files (_w600r richest, then _w600) carry more columns: read them first so the dedupe
            # below keeps their version of a departure both files hold.
            hits = sorted(root.rglob("adsb_departures_*.parquet"),
                          key=lambda h: (0 if "_w600r" in h.name else 1 if "_w600" in h.name else 2, h.name))
            if hits:
                a = pd.concat([pd.read_parquet(h) for h in hits], ignore_index=True)
                a = a.drop_duplicates(subset=["hex", "t_takeoff", "apt"]).reset_index(drop=True)
                for c in ADSB_RAW:
                    if c not in a:
                        a[c] = np.nan
                a["cs"] = a["callsign"].fillna("").astype(str).str.strip().str.upper()
                span = pd.to_datetime(a["t_takeoff"], unit="s", utc=True)
                print(f"adsb: {len(hits)} files, {len(a):,} departures, {span.min()} .. {span.max()}", flush=True)
                return a
    print("adsb: not found; ADS-B features skipped", flush=True)
    return None


def blank_adsb(X: pd.DataFrame, apt, airports, cols) -> None:
    """Blank ADS-B features in place at airports without 2025 ground coverage."""
    rows = np.isin(apt, airports)
    X.loc[rows, cols] = np.nan
    X.loc[rows, "adsb_found"] = 0.0


def adsb_features(apt, t, cs1, cs2) -> pd.DataFrame:
    n = len(t)
    raw = pd.DataFrame(np.nan, index=np.arange(n), columns=ADSB_RAW + ["cs_match"])
    if ADSB is None:
        return raw
    q = pd.DataFrame({"i": np.arange(n), "apt": apt, "t": t,
                      "cs1": pd.Series(cs1).fillna("").astype(str).str.strip().str.upper().to_numpy(),
                      "cs2": pd.Series(cs2).fillna("").astype(str).str.strip().str.upper().to_numpy()})
    a = ADSB[["apt", "cs"] + ADSB_RAW]
    # Nearest same-callsign takeoff within 5 min, per (airport, callsign). An exact join on
    # the callsign pairs each flight with every day it flew and runs out of memory at 7 months.
    aa = a.dropna(subset=["t_takeoff"]).sort_values("t_takeoff")
    parts = []
    for col in ("cs1", "cs2"):
        qq = q[(q[col].str.len() >= 3) & q["t"].notna()].sort_values("t")
        m = pd.merge_asof(qq, aa.rename(columns={"cs": col}), left_on="t", right_on="t_takeoff",
                          by=["apt", col], direction="nearest", tolerance=300.0)
        parts.append(m[m["t_takeoff"].notna()])
    m = pd.concat(parts, ignore_index=True)
    if len(m):
        m["adt"] = (m["t_takeoff"] - m["t"]).abs()
        m = m.sort_values("adt").drop_duplicates("i")
        raw.loc[m["i"].to_numpy(), ADSB_RAW] = m[ADSB_RAW].to_numpy()
        raw.loc[m["i"].to_numpy(), "cs_match"] = 1.0
    rest = q[~q["i"].isin(m["i"] if len(m) else [])].dropna(subset=["t"])
    for ap, g in rest.groupby("apt"):
        aa = a[a["apt"] == ap].sort_values("t_takeoff")
        if aa.empty:
            continue
        mm = pd.merge_asof(g.sort_values("t"), aa.drop(columns=["apt", "cs"]), left_on="t", right_on="t_takeoff",
                           direction="nearest", tolerance=60.0)
        mm = mm[mm["t_takeoff"].notna()]
        raw.loc[mm["i"].to_numpy(), ADSB_RAW] = mm[ADSB_RAW].to_numpy()
        raw.loc[mm["i"].to_numpy(), "cs_match"] = 0.0
    out = pd.DataFrame(index=raw.index)
    out["adsb_found"] = raw["t_takeoff"].notna().astype(float)
    out["adsb_cs_match"] = raw["cs_match"]
    out["adsb_dt_takeoff"] = raw["t_takeoff"] - t
    for c in ("t_first", "t_gap_start", "t_stand_end", "t_start", "t_move", "t_taxi", "t_move_all"):
        out[f"adsb_obs_{c[2:]}"] = t - raw[c]
    for c in ("stopped_s", "n_stops", "ground_dist_m", "start_to_last_m", "n_points", "n_points_seg", "had_stand_dwell",
              "first_to_last_m", "dist_all_m", "stopped_all_s", "max_gap_s", "n_gaps", "dwell_first_s", "air_gap_s"):
        out[f"adsb_{c}"] = raw[c]
    lat0 = np.array([ARP.get(x, (np.nan, np.nan))[0] for x in apt])
    lon0 = np.array([ARP.get(x, (np.nan, np.nan))[1] for x in apt])
    out["adsb_dy_km"] = (raw["lat_start"] - lat0) * 111.2
    out["adsb_dx_km"] = (raw["lon_start"] - lon0) * 111.2 * np.cos(np.radians(lat0))
    out["adsb_first_dy_km"] = (raw["lat_first"] - lat0) * 111.2
    out["adsb_first_dx_km"] = (raw["lon_first"] - lon0) * 111.2 * np.cos(np.radians(lat0))
    return out


GEOM = None


def load_geometry():
    """Stand / runway / stand x runway tables from research/geometry (OSM ODbL, X-Plane GPL,
    OurAirports public domain), published as usiadianimuwa/prc-taxi-2026-geometry."""
    roots = [Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob("stand_runway.parquet"))
            if hits:
                d = hits[0].parent
                g = {"sr": pd.read_parquet(d / "stand_runway.parquet"), "st": pd.read_parquet(d / "stands.parquet"),
                     "rw": pd.read_parquet(d / "runways.parquet")}
                print(f"geometry: {d} ({len(g['sr']):,} stand x runway pairs)", flush=True)
                return g
    print("geometry: not found; skipped", flush=True)
    return None


def geometry_features(apt, stand, rwy) -> pd.DataFrame:
    """Taxi path length, straight distance and runway crossings from the stand to the departure
    runway's end; stand position (km from the ARP), stand heading and ICAO size class; the
    bearing from the stand to the runway end relative to the stand's heading (push direction)."""
    n = len(apt)
    cols = ["geo_path_m", "geo_straight_m", "geo_crossings", "geo_graph_m", "geo_backtrack_m", "geo_path_ratio",
            "geo_stand_dx_km", "geo_stand_dy_km", "geo_stand_hdg_sin", "geo_stand_hdg_cos", "geo_stand_cat",
            "geo_rwy_len_m", "geo_rel_bearing_cos", "geo_rel_bearing_sin", "geo_exact"]
    out = pd.DataFrame(np.nan, index=np.arange(n), columns=cols)
    if GEOM is None:
        return out
    q = pd.DataFrame({"apt": apt, "stand": pd.Series(stand).astype(str).to_numpy(),
                      "runway": pd.Series(rwy).astype(str).to_numpy()})
    sr = GEOM["sr"][["apt", "stand", "runway", "path_m", "straight_m", "crossings", "graph_m", "backtrack_m",
                     "path_over_straight"]]
    m = q.merge(sr, on=["apt", "stand", "runway"], how="left")
    out["geo_path_m"] = m["path_m"].to_numpy()
    out["geo_straight_m"] = m["straight_m"].to_numpy()
    out["geo_crossings"] = m["crossings"].to_numpy()
    out["geo_graph_m"] = m["graph_m"].to_numpy()
    out["geo_backtrack_m"] = m["backtrack_m"].to_numpy()
    out["geo_path_ratio"] = m["path_over_straight"].to_numpy()
    st = GEOM["st"][["apt", "stand", "lat", "lon", "heading", "xp_icao_cat", "exact"]]
    s = q.merge(st, on=["apt", "stand"], how="left")
    lat0 = np.array([ARP.get(x, (np.nan, np.nan))[0] for x in apt])
    lon0 = np.array([ARP.get(x, (np.nan, np.nan))[1] for x in apt])
    out["geo_stand_dy_km"] = (s["lat"].to_numpy(dtype=float) - lat0) * 111.2
    out["geo_stand_dx_km"] = (s["lon"].to_numpy(dtype=float) - lon0) * 111.2 * np.cos(np.radians(lat0))
    h = np.radians(s["heading"].to_numpy(dtype=float))
    out["geo_stand_hdg_sin"], out["geo_stand_hdg_cos"] = np.sin(h), np.cos(h)
    out["geo_stand_cat"] = s["xp_icao_cat"].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}).to_numpy(dtype=float)
    out["geo_exact"] = s["exact"].astype(float).to_numpy()
    rw = GEOM["rw"][["apt", "runway", "thr_lat", "thr_lon", "length_m"]]
    r = q.merge(rw, on=["apt", "runway"], how="left")
    out["geo_rwy_len_m"] = r["length_m"].to_numpy(dtype=float)
    dy = (r["thr_lat"].to_numpy(dtype=float) - s["lat"].to_numpy(dtype=float)) * 111.2
    dx = (r["thr_lon"].to_numpy(dtype=float) - s["lon"].to_numpy(dtype=float)) * 111.2 * np.cos(np.radians(lat0))
    bearing = np.arctan2(dx, dy)
    rel = bearing - h
    out["geo_rel_bearing_cos"], out["geo_rel_bearing_sin"] = np.cos(rel), np.sin(rel)
    return out


def secs(values) -> np.ndarray:
    return (pd.to_datetime(values, utc=True, errors="coerce") - EPOCH).dt.total_seconds().to_numpy()


# %%
def build(df: pd.DataFrame) -> pd.DataFrame:
    """One row per departure, using only fields the ranking file publishes."""
    df = df.reset_index(drop=True)
    dep = df["PHASE_mvt"].eq("DEP").to_numpy()
    arr = df["PHASE_mvt"].eq("ARR").to_numpy()
    apt = np.where(dep, df["ADEP_mvt"].astype(str), df["ADES_mvt"].astype(str))
    rwy = pd.Series(apt) + "|" + df["RUNWAY_mvt"].astype(str)
    t = secs(df["MVT_TIME_UTC_mvt"])
    blk = secs(df["BLOCK_TIME_UTC_mvt"])
    blk[dep] = np.nan  # departure off-block is the answer; blank it in training too
    taxi_in = np.where(arr, blk - t, np.nan)
    aobt3 = secs(df["AOBT_3_flt"])

    d = df.loc[dep].reset_index(drop=True)
    q = np.flatnonzero(dep)
    qa, qt, qr = apt[q], t[q], rwy.to_numpy()[q]
    qa3 = aobt3[q]
    F = pd.DataFrame({ID: d[ID].to_numpy()})
    if TARGET in d and d[TARGET].notna().any():
        F["y"] = d[TARGET].astype("float64").to_numpy()

    # Per-flight timing.
    sched, eobt1, lobt, iobt = (secs(d[c]) for c in ("SCHED_TIME_UTC_mvt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt"))
    arvt1, arvt3 = secs(d["ARVT_1_flt"]), secs(d["ARVT_3_flt"])
    F["d_aobt3"] = qt - qa3
    F["d_eobt1"] = qt - eobt1
    F["d_lobt"] = qt - lobt
    F["d_iobt"] = qt - iobt
    F["d_sched"] = qt - sched
    F["dd_utc"] = np.floor(qt / 86400) - np.floor(sched / 86400)
    F["aobt3_m_eobt1"] = qa3 - eobt1
    F["aobt3_m_sched"] = qa3 - sched
    F["lobt_m_iobt"] = lobt - iobt
    F["eobt1_m_lobt"] = eobt1 - lobt
    F["arvt3_m_arvt1"] = arvt3 - arvt1
    F["air3"] = arvt3 - qt
    F["air1"] = arvt1 - eobt1
    F["unmatched"] = np.isnan(qa3).astype("float32")
    F["mvt_sec"] = np.mod(qt, 60)
    F["aobt3_sec"] = np.mod(qa3, 60)
    F["d_aobt3_mod60"] = np.mod(F["d_aobt3"].to_numpy(), 60)

    # Stand occupancy: block-ins at our stand around our takeoff.
    stand_all = (pd.Series(apt) + "|" + df["STAND_mvt"].astype(str)).to_numpy()
    has_stand = df["STAND_mvt"].notna().to_numpy()
    flight_all = df["FLIGHT_mvt"].fillna("").astype(str).str[:3].to_numpy()
    type_all = df["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()
    ra = arr & has_stand
    arr_sched_all = secs(df["SCHED_TIME_UTC_mvt"])
    # An arrival whose in-block equals its scheduled time was filled, not observed.
    arr_fill = arr & (np.abs(blk - arr_sched_all) <= 60)
    nb = stand_neighbours(stand_all[q], qt, stand_all[ra], blk[ra],
                          {"air": flight_all[ra], "typ": type_all[ra], "fill": arr_fill[ra].astype(float)})
    my_air, my_typ = flight_all[q], type_all[q]
    q_has_stand = has_stand[q]
    for l in (1, 2):
        gap = qt - nb[f"t_prev{l}"]
        gap[~q_has_stand] = np.nan
        F[f"stand_gap_prev{l}"] = gap
        F[f"stand_same_air_prev{l}"] = np.where(np.isnan(gap), np.nan, (nb[f"air_prev{l}"] == my_air).astype(float))
        F[f"stand_same_typ_prev{l}"] = np.where(np.isnan(gap), np.nan, (nb[f"typ_prev{l}"] == my_typ).astype(float))
    other = (F["stand_same_air_prev1"].to_numpy() == 0) | (F["stand_same_typ_prev1"].to_numpy() == 0)
    F["stand_lb_other"] = np.where(other, F["stand_gap_prev1"].to_numpy(), np.nan)
    nxt = nb["t_next"] - qt
    nxt[~q_has_stand] = np.nan
    F["stand_gap_next"] = nxt
    F["stand_prev1_fill"] = np.array([np.nan if v is None else float(v) for v in nb["fill_prev1"]])
    F["stand_prev1_fill"] = np.where(np.isnan(F["stand_gap_prev1"].to_numpy()), np.nan, F["stand_prev1_fill"].to_numpy())
    # Airport-wide share of filled arrival in-blocks around our scheduled off-block.
    apt_arr = np.where(arr, apt, "")
    for w in (3600, 10800):
        n_all = count_between(qa, sched - w, sched + w, apt_arr[arr], blk[arr])
        n_fill = count_between(qa, sched - w, sched + w, apt_arr[arr_fill], blk[arr_fill])
        F[f"apt_arr_fill_{w // 3600}h"] = np.where(n_all > 0, np.nan_to_num(n_fill) / np.maximum(n_all, 1), np.nan)
    F["stand_arr_2h"] = count_between(stand_all[q], qt - 7200, qt + 1, stand_all[ra], blk[ra])
    F["stand_arr_2h"] = np.where(q_has_stand, F["stand_arr_2h"].to_numpy(), np.nan)

    # Inbound aircraft: latest same-airline block-in at our stand before takeoff.
    stand_air = (pd.Series(stand_all) + "|" + pd.Series(flight_all)).to_numpy()
    ib = stand_neighbours(stand_air[q], qt, stand_air[ra], blk[ra], {"sch": arr_sched_all[ra]}, lags=(1,))
    inb = ib["t_prev1"]
    inb_sch = np.array([np.nan if v is None else float(v) for v in ib["sch_prev1"]])
    inb[~q_has_stand] = np.nan
    dsafe = np.where(F["d_sched"].to_numpy() > 60, F["d_sched"].to_numpy(), np.nan)
    F["inb_gap"] = qt - inb
    F["inb_delay"] = inb - inb_sch
    F["inb_share"] = np.clip((inb - sched) / dsafe, -5, 5)
    F["inb_gap_ratio"] = np.clip((qt - inb) / dsafe, -5, 5)
    F["inb_fill"] = np.where(np.isnan(F["inb_delay"].to_numpy()), np.nan, (np.abs(F["inb_delay"].to_numpy()) <= 60).astype(float))

    # Neighbour long-delay context (airport-wide taxiway holds).
    dsch = F["d_sched"].to_numpy()
    longq = dsch > 3600
    safe = np.where(longq, dsch, 1.0)
    c3 = np.where(longq, np.clip(F["d_aobt3"].to_numpy() / safe, 0, 1.5), np.nan)
    ce = np.where(longq, np.clip(F["d_eobt1"].to_numpy() / safe, 0, 1.5), np.nan)
    for w in (3600, 10800):
        F[f"nb_long_share_d3_{w}"] = mean_between(qa, qt - w, qt + w, qa, qt, c3)
        F[f"nb_long_share_e1_{w}"] = mean_between(qa, qt - w, qt + w, qa, qt, ce)
        F[f"nb_long_count_{w}"] = count_between(qa, qt - w, qt + w, qa[longq], qt[longq])

    # Weather at takeoff and around off-block.
    w0 = weather_features(qa, qt)
    w1 = weather_features(qa, qt - 1800)
    for c in w0.columns:
        F[f"wx_{c}_t"] = w0[c].to_numpy()
    for c in ["temp_c", "vis_km", "wind_kt", "deice_likely", "wx_snow", "precip_mm"]:
        F[f"wx_{c}_m30"] = w1[c].to_numpy()

    # ADS-B ground movement (adsb.lol), for the extracted months.
    ad = adsb_features(qa, qt, d["CALLSIGN_flt"].to_numpy(), d["FLIGHT_mvt"].to_numpy())
    for c in ad.columns:
        F[c] = ad[c].to_numpy()

    # ADS-B v3 events (fs4).
    a3 = adsb3_features(qa, qt, d["CALLSIGN_flt"].to_numpy(), d["FLIGHT_mvt"].to_numpy(), d["STAND_mvt"].to_numpy())
    for c in a3.columns:
        F[c] = a3[c].to_numpy()

    # Airport geometry (fs3).
    ge = geometry_features(qa, d["STAND_mvt"].to_numpy(), d["RUNWAY_mvt"].to_numpy())
    for c in ge.columns:
        F[c] = ge[c].to_numpy()

    # Local clock.
    local_hour = np.full(len(q), np.nan)
    local_dow = np.full(len(q), np.nan)
    local_day = np.empty(len(q), dtype=object)
    ts_utc = pd.to_datetime(qt, unit="s", utc=True)
    for code, tz in TZ.items():
        m = qa == code
        if m.any():
            lt = ts_utc[m].tz_convert(tz)
            local_hour[m] = lt.hour + lt.minute / 60.0
            local_dow[m] = lt.dayofweek
            local_day[m] = lt.strftime("%Y-%m-%d")
    other = pd.isna(local_day)
    local_day[other] = ts_utc[other].strftime("%Y-%m-%d")
    local_hour[other] = ts_utc[other].hour
    local_dow[other] = ts_utc[other].dayofweek
    F["local_hour"] = local_hour
    F["local_dow"] = local_dow
    F["month"] = ts_utc.month.to_numpy()
    F["mday"] = ts_utc.day.to_numpy()
    day_key = pd.Series(qa) + "|" + pd.Series(local_day)

    # Runway queue and airport flow, from published takeoff and landing times.
    dep_t, dep_k, dep_r = t[dep], apt[dep], rwy.to_numpy()[dep]
    arr_t, arr_k, arr_r = t[arr], apt[arr], rwy.to_numpy()[arr]
    for w in (300, 600, 900, 1800, 3600):
        F[f"dep_prev_{w}"] = count_between(qa, qt - w, qt, dep_k, dep_t)
        F[f"arr_prev_{w}"] = count_between(qa, qt - w, qt, arr_k, arr_t)
    for w in (300, 600, 1800):
        F[f"dep_next_{w}"] = count_between(qa, qt + 1, qt + w, dep_k, dep_t)
        F[f"arr_next_{w}"] = count_between(qa, qt + 1, qt + w, arr_k, arr_t)
    for w in (600, 1800):
        F[f"rwy_dep_prev_{w}"] = count_between(qr, qt - w, qt, dep_r, dep_t)
        F[f"rwy_dep_next_{w}"] = count_between(qr, qt + 1, qt + w, dep_r, dep_t)
        F[f"rwy_arr_prev_{w}"] = count_between(qr, qt - w, qt, arr_r, arr_t)
    lo = np.where(qa3 < qt, qa3, np.nan)
    F["queue_aobt3"] = count_between(qa, lo, qt, dep_k, dep_t)
    F["rwy_queue_aobt3"] = count_between(qr, lo, qt, dep_r, dep_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["queue_rate"] = F["queue_aobt3"] / np.maximum(qt - lo, 60) * 3600

    # Surface load: departures between AOBT_3 and takeoff, arrivals between
    # landing and in-block.
    a3_dep = aobt3[dep]
    ok3 = ~np.isnan(a3_dep)
    far = -1e12
    for name, at in (("at_t", qt), ("at_aobt3", qa3)):
        pushed = count_between(qa, np.full(len(q), far), at + 1, dep_k[ok3], a3_dep[ok3])
        gone = count_between(qa, np.full(len(q), far), at + 1, dep_k[ok3], dep_t[ok3])
        F[f"dep_surface_{name}"] = pushed - gone
        landed = count_between(qa, np.full(len(q), far), at + 1, arr_k, arr_t)
        blocked = count_between(qa, np.full(len(q), far), at + 1, arr_k, blk[arr])
        F[f"arr_surface_{name}"] = landed - blocked

    # Live conditions: arrival taxi-in and neighbours' MVT - AOBT_3.
    tin = np.clip(taxi_in[arr], 0, TE_CLIP)
    for w in (1800, 3600):
        F[f"taxiin_prev_{w}"] = mean_between(qa, qt - w, qt, arr_k, arr_t, tin)
    F["taxiin_pm_1800"] = mean_between(qa, qt - 1800, qt + 1800, arr_k, arr_t, tin)
    d3_all = np.clip(t - aobt3, 0, 5400)
    d3_dep = np.where(dep, d3_all, np.nan)[dep]
    for w in (1800, 3600):
        F[f"nb_d3_pm_{w}"] = mean_between(qa, qt - w, qt + w, dep_k, dep_t, d3_dep)
    F["rwy_nb_d3_pm_1800"] = mean_between(qr, qt - 1800, qt + 1800, dep_r, dep_t, d3_dep)

    # Airport-day conditions (de-icing, weather, closures).
    arr_day = np.empty(arr.sum(), dtype=object)
    at_utc = pd.to_datetime(arr_t, unit="s", utc=True)
    for code, tz in TZ.items():
        m = arr_k == code
        if m.any():
            arr_day[m] = at_utc[m].tz_convert(tz).strftime("%Y-%m-%d")
    arr_day[pd.isna(arr_day)] = at_utc[pd.isna(arr_day)].strftime("%Y-%m-%d")
    arr_day_key = pd.Series(arr_k) + "|" + pd.Series(arr_day)
    day_tin = pd.Series(tin).groupby(arr_day_key.to_numpy()).mean()
    F["day_taxiin"] = day_key.map(day_tin).to_numpy()
    F["day_d3"] = pd.Series(d3_dep).groupby(day_key.to_numpy()).transform("mean").to_numpy()
    F["day_deps"] = day_key.map(day_key.value_counts()).to_numpy()
    F["day_unmatched"] = pd.Series(F["unmatched"].to_numpy()).groupby(day_key.to_numpy()).transform("mean").to_numpy()

    # Runway configuration per airport-hour (departure and arrival runways in use).
    hb = np.floor(t / 3600)
    ev = pd.DataFrame({"apt": apt, "hb": hb, "rwy": df["RUNWAY_mvt"].astype(str), "dep": dep})
    ev = ev.drop_duplicates()
    conf = (ev.sort_values("rwy").groupby(["apt", "hb", "dep"])["rwy"].agg("/".join).unstack("dep"))
    conf.columns = ["arr_rwys" if not c else "dep_rwys" for c in conf.columns]
    key = pd.MultiIndex.from_arrays([qa, hb[q]])
    conf = conf.reindex(key)
    F["rwy_conf"] = (pd.Series(qa) + "|" + conf["dep_rwys"].fillna("-").to_numpy()
                     + "|" + conf["arr_rwys"].fillna("-").to_numpy()).to_numpy()

    # Categoricals.
    F["apt"] = qa
    F["rwy"] = qr
    F["stand"] = (pd.Series(qa) + "|" + d["STAND_mvt"].astype(str)).to_numpy()
    F["stand_rwy"] = (F["stand"] + "|" + d["RUNWAY_mvt"].astype(str)).to_numpy()
    F["actype"] = d["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()
    F["wake"] = d["WK_TBL_CAT_flt"].astype(str).to_numpy()
    F["segment"] = d["MARKET_SEGMENT_flt"].astype(str).to_numpy()
    F["ftype"] = d["FLIGHT_TYPE_flt"].astype(str).to_numpy()
    F["frule"] = d["FLIGHT_RULE_mvt"].astype(str).to_numpy()
    F["operator"] = d["AIRCRAFT_OPERATOR_flt"].astype(str).to_numpy()
    F["ades"] = d["ADES_mvt"].astype(str).to_numpy()
    F["ades_region"] = d["ADES_mvt"].astype(str).str[:2].to_numpy()
    F["diverted_plan"] = (d["ADES_FILED_flt"].astype(str) != d["ADES_flt"].astype(str)).astype("float32").to_numpy()
    flight = d["FLIGHT_mvt"].fillna("").astype(str)
    F["flight_len"] = flight.str.len().to_numpy()
    F["flight_dbl"] = ((flight.str.len() >= 5) & (flight.str[1:3] == flight.str[3:5])).astype("float32").to_numpy()
    F["flight_digits"] = flight.str.count(r"\d").to_numpy()
    F["no_actype"] = d["AIRCRAFT_TYPE_mvt"].isna().astype("float32").to_numpy()
    F["no_stand"] = d["STAND_mvt"].isna().astype("float32").to_numpy()
    F["callsign_eq_flight"] = (d["CALLSIGN_flt"].astype(str) == flight).astype("float32").to_numpy()
    F["airline3"] = (pd.Series(qa) + "|" + flight.str[:3]).to_numpy()
    F["flight_key"] = (pd.Series(qa) + "|" + flight).to_numpy()
    F["apt_actype"] = (F["apt"] + "|" + F["actype"]).to_numpy()
    F["apt_operator"] = (F["apt"] + "|" + F["operator"]).to_numpy()
    F["rwy_hour"] = (F["rwy"] + "|" + np.floor(local_hour).astype(int).astype(str)).to_numpy()
    F["t_utc"] = qt
    return F


CATS = ["apt", "rwy", "stand", "stand_rwy", "actype", "wake", "segment", "ftype", "frule", "airline3",
        "operator", "ades", "ades_region", "rwy_conf", "apt_actype", "apt_operator"]
TE_KEYS = ["stand_rwy", "stand", "rwy", "apt_actype", "apt_operator", "rwy_hour", "rwy_conf"]
DROP = {ID, "y", "month", "mday", "rwy_hour", "flight_key", "t_utc"}
LONG_KEYS = ["airline3", "flight_key", "apt_operator", "stand_rwy", "apt", "ades"]


# %%
def target_encode(train: dict, month: np.ndarray, y: np.ndarray, fit: np.ndarray,
                  apply: list[dict], m: float = 30.0):
    """Smoothed mean of clipped y over raw (uncollapsed) keys.

    Training rows leave their own month out; `apply` rows use every fit row.
    """
    y = np.clip(y, 0, TE_CLIP)
    yf = np.where(fit, y, np.nan)
    prior_tab = pd.Series(yf).groupby(train["apt"]).mean()
    enc_train = {}
    enc_apply = [dict() for _ in apply]
    for key in TE_KEYS:
        k = train[key]
        frame = pd.DataFrame({"k": k, "mo": month, "y": yf, "n": fit.astype(float)})
        tot = frame.groupby("k")[["y", "n"]].sum()
        km = frame.groupby(["k", "mo"])[["y", "n"]].sum()
        idx = pd.MultiIndex.from_arrays([k, frame["mo"].to_numpy()])
        s = pd.Series(k).map(tot["y"]).to_numpy() - km["y"].reindex(idx).to_numpy()
        n = pd.Series(k).map(tot["n"]).to_numpy() - km["n"].reindex(idx).to_numpy()
        prior = pd.Series(train["apt"]).map(prior_tab).to_numpy()
        enc_train[f"te_{key}"] = (s + m * prior) / (n + m)
        enc_train[f"te_n_{key}"] = n
        for j, keys_apply in enumerate(apply):
            ka = keys_apply[key]
            s_a = pd.Series(ka).map(tot["y"]).fillna(0).to_numpy()
            n_a = pd.Series(ka).map(tot["n"]).fillna(0).to_numpy()
            p_a = pd.Series(keys_apply["apt"]).map(prior_tab).to_numpy()
            enc_apply[j][f"te_{key}"] = (s_a + m * p_a) / (n_a + m)
            enc_apply[j][f"te_n_{key}"] = n_a
    return pd.DataFrame(enc_train), [pd.DataFrame(e) for e in enc_apply]


def categorize(train: pd.DataFrame, others: list[pd.DataFrame], ref_mask: np.ndarray):
    """Rare and unseen levels collapse to one bucket; codes shared across frames."""
    for c in CATS:
        vc = pd.Series(train.loc[ref_mask, c]).value_counts()
        keep = vc.index[vc >= RARE_MIN]
        cats = pd.Index(list(keep) + ["__RARE__"])
        for frame in [train] + others:
            v = frame[c].where(frame[c].isin(keep), "__RARE__")
            frame[c] = pd.Categorical(v, categories=cats)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


QUANTILE_KEYS = ["stand_rwy", "stand"]
QUANTILES = (0.1, 0.5, 0.9)
PATH_DS = 3600.0   # schedule-fill test applies when takeoff is > 1 h past schedule
PATH_TOL = 120.0


def quantile_encode(train: dict, month: np.ndarray, y: np.ndarray, fit: np.ndarray,
                    apply: list[dict], min_n: int = 20):
    """p10/p50/p90 of clipped y per key; training rows leave their own month out."""
    y = np.clip(y, 0, TE_CLIP)
    n = len(y)
    enc_train = {f"tq{int(q * 100)}_{k}": np.full(n, np.nan) for k in QUANTILE_KEYS for q in QUANTILES}
    enc_apply = [{c: np.full(len(a["apt"]), np.nan) for c in enc_train} for a in apply]

    def table(src: np.ndarray, key: str) -> pd.DataFrame:
        frame = pd.DataFrame({"k": train[key][src], "y": y[src]})
        grouped = frame.groupby("k")["y"]
        tab = grouped.quantile(list(QUANTILES)).unstack()
        return tab[grouped.size().reindex(tab.index) >= min_n]

    for key in QUANTILE_KEYS:
        for m in np.unique(month):
            dst = month == m
            tab = table(fit & ~dst, key)
            keys = pd.Series(train[key][dst])
            for q in QUANTILES:
                enc_train[f"tq{int(q * 100)}_{key}"][dst] = keys.map(tab[q]).to_numpy()
        tab = table(fit, key)
        for j, a in enumerate(apply):
            keys = pd.Series(a[key])
            for q in QUANTILES:
                enc_apply[j][f"tq{int(q * 100)}_{key}"] = keys.map(tab[q]).to_numpy()
    return pd.DataFrame(enc_train), [pd.DataFrame(e) for e in enc_apply]



LONG_DS = 3600.0
TWIN_TEST_LIKE = 344.601  # v14 and v18 twin on the days-16-31 hold-out
ADSB_MIN_TRAIN_RATE = 0.0  # v16: ADS-B transfers to airports unseen with it (198.97 -> 193.73), so mask nothing
PATH_TOL = 120.0
R_CLIP = 1.5
W_CAP = 43200.0


def loo_encode(train: dict, month: np.ndarray, values: np.ndarray, fit: np.ndarray, apply: dict,
               keys: list[str], prefix: str, m: float = 20.0):
    """Smoothed mean of `values` per key; training rows leave their own month out."""
    vf = np.where(fit, values, np.nan)
    prior_tab = pd.Series(vf).groupby(train["apt"]).mean()
    prior_all = np.nanmean(vf)
    enc_tr, enc_ap = {}, {}
    prior_tr = pd.Series(train["apt"]).map(prior_tab).fillna(prior_all).to_numpy()
    prior_ap = pd.Series(apply["apt"]).map(prior_tab).fillna(prior_all).to_numpy()
    for key in keys:
        k = train[key]
        frame = pd.DataFrame({"k": k, "mo": month, "v": vf, "n": fit.astype(float)})
        tot = frame.groupby("k")[["v", "n"]].sum()
        km = frame.groupby(["k", "mo"])[["v", "n"]].sum()
        idx = pd.MultiIndex.from_arrays([k, month])
        s = pd.Series(k).map(tot["v"]).to_numpy() - km["v"].reindex(idx).to_numpy()
        n = pd.Series(k).map(tot["n"]).to_numpy() - km["n"].reindex(idx).to_numpy()
        enc_tr[f"{prefix}_{key}"] = (s + m * prior_tr) / (n + m)
        enc_tr[f"{prefix}_n_{key}"] = n
        ka = pd.Series(apply[key])
        s_a = ka.map(tot["v"]).fillna(0).to_numpy()
        n_a = ka.map(tot["n"]).fillna(0).to_numpy()
        enc_ap[f"{prefix}_{key}"] = (s_a + m * prior_ap) / (n_a + m)
        enc_ap[f"{prefix}_n_{key}"] = n_a
    return pd.DataFrame(enc_tr), pd.DataFrame(enc_ap)


def all_encodings(keys_tr, month, y, ds, r, long, path, base_fit, keys_rk):
    te_tr, (te_rk,) = target_encode(keys_tr, month, y, base_fit & ~path, [keys_rk])
    tq_tr, (tq_rk,) = quantile_encode(keys_tr, month, y, base_fit & ~path, [keys_rk])
    lr_tr, lr_rk = loo_encode(keys_tr, month, r, base_fit & long, keys_rk, LONG_KEYS, "ter")
    lp_tr, lp_rk = loo_encode(keys_tr, month, path.astype(float), base_fit & long, keys_rk, LONG_KEYS, "tep")
    return (pd.concat([te_tr, tq_tr, lr_tr, lp_tr], axis=1),
            pd.concat([te_rk, tq_rk, lr_rk, lp_rk], axis=1))



RATIO_SOURCES = ["adsb_obs_move_all", "adsb_stopped_all_s", "adsb_obs_start", "adsb_obs_move", "adsb_obs_first", "adsb_obs_taxi", "adsb_stopped_s", "d_aobt3", "d_eobt1", "d_lobt", "d_iobt", "aobt3_m_sched", "aobt3_m_eobt1",
                 "stand_gap_prev1", "stand_gap_prev2", "stand_lb_other", "nb_d3_pm_1800", "rwy_nb_d3_pm_1800",
                 "te_stand_rwy", "tq10_stand_rwy", "tq50_stand_rwy", "tq90_stand_rwy", "day_taxiin"]


def add_ratios(X: pd.DataFrame) -> pd.DataFrame:
    d = X["d_sched"].to_numpy(dtype=float)
    d = np.where(d > 60, d, np.nan)
    extra = {f"ratio_{c}": np.clip(X[c].to_numpy(dtype=float) / d, -5, 5) for c in RATIO_SOURCES if c in X}
    extra["log_d_sched"] = np.log1p(np.clip(X["d_sched"].to_numpy(dtype=float), 0, None))
    extra = {k: v.astype(np.float32) for k, v in extra.items()}
    return pd.concat([X, pd.DataFrame(extra, index=X.index)], axis=1)




# %%
# Label-free congestion measured by ADS-B (fs2).  A departure's excess is its observed taxi
# (takeoff - first movement) minus the typical observed taxi for its stand x runway, where
# "typical" is the median over every credible ADS-B departure of train AND ranking rows (no
# target involved).  The neighbour excess is the mean excess of the other ADS-B departures
# at the same airport (or runway) within +-w of our takeoff.  On 2025 hold-out rows without
# their own ADS-B it tracks y - te_stand_rwy almost 1:1 above +400 s, and on disrupted days
# (EHAM 3-7 Jan 2026 de-icing) it lies far outside 2025's range, where a tree cannot follow
# but an anchor can.
NB_OBS = "adsb_obs_move_all"
NB_MAX = 3600.0


def _credible(F: pd.DataFrame) -> np.ndarray:
    if NB_OBS not in F:  # no ADS-B mounted (smoke slices)
        F[NB_OBS] = np.nan
        F["adsb_found"] = 0.0
    o = F[NB_OBS].to_numpy(dtype=float)
    return (F["adsb_found"].to_numpy() > 0) & (o > 30) & (o < NB_MAX)


def add_nb_excess(frames: list) -> None:
    for F in frames:
        _credible(F)  # creates empty ADS-B columns when none is mounted
    pool = pd.concat([pd.DataFrame({"k": F["stand_rwy"].astype(str).to_numpy(),
                                    "o": F[NB_OBS].to_numpy(dtype=float)})[_credible(F)] for F in frames])
    typ = pool.groupby("k")["o"].median()
    cnt = pool.groupby("k").size()
    typ = typ[cnt >= 10]
    for F in frames:
        ok = _credible(F)
        # period-specific typical taxi (this frame's own months: 2025 for training rows, Jan/Jul
        # 2026 for ranking rows), per stand x runway and per stand x runway x month, leaving the
        # row's own observation out; shows 2026 route changes that 2025 target encodings cannot
        o = np.where(ok, F[NB_OBS].to_numpy(dtype=float), np.nan)
        for name, key in (("sr", F["stand_rwy"].astype(str)),
                          ("srm", F["stand_rwy"].astype(str) + "|" + F["month"].astype(str))):
            fr = pd.DataFrame({"k": key.to_numpy(), "o": o})
            s = fr.groupby("k")["o"].transform("sum").to_numpy() - np.nan_to_num(o)
            n = fr.groupby("k")["o"].transform("count").to_numpy() - ok
            F[f"adsb_per_{name}"] = np.where(n >= 5, s / np.maximum(n, 1), np.nan)
            F[f"adsb_per_n_{name}"] = n
        t = F["t_utc"].to_numpy(dtype=float)
        tp = F["stand_rwy"].astype(str).map(typ).to_numpy(dtype=float)
        exc = np.where(ok, F[NB_OBS].to_numpy(dtype=float) - tp, np.nan)
        F["adsb_typ_sr"] = tp
        F["adsb_exc"] = exc
        for key in ("apt", "rwy"):
            k = F[key].astype(str).to_numpy()
            for w in (1800, 3600):
                s = mean_between(k, t - w, t + w, k, t, exc)
                n = count_between(k, t - w, t + w, k[~np.isnan(exc)], t[~np.isnan(exc)])
                # leave the row's own excess out
                own = np.where(np.isnan(exc), 0.0, exc)
                own_n = (~np.isnan(exc)).astype(float)
                tot = np.nan_to_num(s) * np.nan_to_num(n)
                n2 = np.nan_to_num(n) - own_n
                F[f"nbx_{key}_{w}"] = np.where(n2 > 0, (tot - own) / np.maximum(n2, 1), np.nan)
                F[f"nbn_{key}_{w}"] = n2


# %%
# ADS-B v3 events (solutions/adsb/extract_v3.py + events_v3.py, adsb.lol ODbL): one row per
# departure with graded off-block evidence.  Grade A = seen parked, then leaving (t_obs = the
# pushback, median error -7 s on Jan 2025 labels); B/C = first seen moving (a lower bound) with
# alternative candidates (t_push, t_leave, dwell ends).  Times become "seconds before takeoff".
ADSB3 = None
A3_TIMES = ["t_obs", "t_push", "t_leave", "t_last_dwell_end", "t_longest_dwell_end", "t_dep_first", "t_first",
            "t_park_start", "t_touchdown", "t_first_dwell_end", "dep_evidence_start", "gap_big_t0", "gap_big_t1",
            "gap_last_t0", "gap_last_t1", "t_lineup"]
A3_VALUES = ["turnaround_s", "park_s", "park_to_lineup_m", "park_short", "park_gaps", "gs_push", "gs_leave",
             "leave_gap_s", "last_dwell_s", "last_dwell_to_lineup_m", "longest_dwell_s", "longest_dwell_to_lineup_m",
             "dep_first_to_lineup_m", "gs_dep_first", "first_to_lineup_m", "gs_first", "move_dist_m", "move_time_s",
             "gs_moving_med", "stopped_s", "stopped_near_rwy_s", "stopped_apron_s", "n_stops", "max_gap_after_obs_s",
             "seen_continuous", "coverage", "n_sep_gaps", "n_big_sep_gaps", "max_gap_s", "n_gaps_300", "n_dwells",
             "n_tows", "gap_big_disp_m", "gap_big_to_lineup_m", "gap_last_disp_m", "gap_last_to_lineup_m",
             "first_dwell_s", "tail_gap_s", "takeoff_seen", "chain_landed", "chain_gnd_flag_frac", "chain_mlat_frac"]
A3_POS = ["lat_park", "lon_park", "lat_dep_first", "lon_dep_first", "lat_first", "lon_first"]


def load_adsb3() -> pd.DataFrame | None:
    # events are a mounted kernel output even when the other inputs were fetched into DATA_ROOT
    roots = ([Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else []) + [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob("adsb3_events_*.parquet")) if root.exists() else []
            if hits:
                cols = ["apt", "callsign", "t_takeoff", "grade"] + A3_TIMES + A3_VALUES + A3_POS
                a = pd.concat([pd.read_parquet(h, columns=[c for c in cols if c in pq_columns(h)]) for h in hits],
                              ignore_index=True)
                for c in cols:
                    if c not in a:
                        a[c] = np.nan
                a = a.dropna(subset=["t_takeoff"]).drop_duplicates(subset=["apt", "t_takeoff", "callsign"])
                a["cs"] = a["callsign"].fillna("").astype(str).str.strip().str.upper()
                print(f"adsb3: {len(hits)} files, {len(a):,} departures", flush=True)
                return a
    print("adsb3: not found; v3 features skipped", flush=True)
    return None


def pq_columns(path) -> list:
    import pyarrow.parquet as _pq
    return _pq.ParquetFile(path).schema_arrow.names


def adsb3_features(apt, t, cs1, cs2, stand) -> pd.DataFrame:
    n = len(t)
    names = (["a3_found", "a3_grade", "a3_cs_match", "a3_dt_takeoff"] + [f"a3_obs_{c[2:]}" for c in A3_TIMES]
             + [f"a3_{c}" for c in A3_VALUES] + ["a3_park_dx_km", "a3_park_dy_km", "a3_first_dx_km", "a3_first_dy_km",
                                                  "a3_park_to_stand_m", "a3_first_to_stand_m"])
    out = pd.DataFrame(np.nan, index=np.arange(n), columns=names)
    if ADSB3 is None:
        return out
    q = pd.DataFrame({"i": np.arange(n), "apt": apt, "t": t,
                      "cs1": pd.Series(cs1).fillna("").astype(str).str.strip().str.upper().to_numpy(),
                      "cs2": pd.Series(cs2).fillna("").astype(str).str.strip().str.upper().to_numpy()})
    cols = ["t_takeoff", "grade"] + A3_TIMES + A3_VALUES + A3_POS
    a = ADSB3[["apt", "cs"] + cols].sort_values("t_takeoff")
    parts = []
    for col in ("cs1", "cs2"):
        qq = q[(q[col].str.len() >= 3) & q["t"].notna()].sort_values("t")
        m = pd.merge_asof(qq, a.rename(columns={"cs": col}), left_on="t", right_on="t_takeoff",
                          by=["apt", col], direction="nearest", tolerance=300.0)
        parts.append(m[m["t_takeoff"].notna()].assign(csm=1.0))
    m = pd.concat(parts, ignore_index=True)
    rest = q[~q["i"].isin(m["i"])].dropna(subset=["t"])
    for ap, g in rest.groupby("apt"):
        aa = a[a["apt"] == ap]
        if aa.empty:
            continue
        mm = pd.merge_asof(g.sort_values("t"), aa.drop(columns=["apt", "cs"]), left_on="t", right_on="t_takeoff",
                           direction="nearest", tolerance=60.0)
        parts.append(mm[mm["t_takeoff"].notna()].assign(csm=0.0))
    m = pd.concat(parts, ignore_index=True)
    if not len(m):
        return out
    m["adt"] = (m["t_takeoff"] - m["t"]).abs()
    m = m.sort_values(["csm", "adt"], ascending=[False, True]).drop_duplicates("i")
    i = m["i"].to_numpy()
    tq = m["t"].to_numpy()
    out.loc[i, "a3_found"] = 1.0
    out.loc[i, "a3_grade"] = m["grade"].map({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}).to_numpy()
    out.loc[i, "a3_cs_match"] = m["csm"].to_numpy()
    out.loc[i, "a3_dt_takeoff"] = m["t_takeoff"].to_numpy() - tq
    for c in A3_TIMES:
        out.loc[i, f"a3_obs_{c[2:]}"] = tq - m[c].to_numpy(dtype=float)
    for c in A3_VALUES:
        out.loc[i, f"a3_{c}"] = m[c].to_numpy(dtype=float)
    ap_i = np.asarray(apt)[i]
    lat0 = np.array([ARP.get(x, (np.nan, np.nan))[0] for x in ap_i])
    lon0 = np.array([ARP.get(x, (np.nan, np.nan))[1] for x in ap_i])
    kx = 111.2 * np.cos(np.radians(lat0))
    out.loc[i, "a3_park_dy_km"] = (m["lat_park"].to_numpy(dtype=float) - lat0) * 111.2
    out.loc[i, "a3_park_dx_km"] = (m["lon_park"].to_numpy(dtype=float) - lon0) * kx
    out.loc[i, "a3_first_dy_km"] = (m["lat_first"].to_numpy(dtype=float) - lat0) * 111.2
    out.loc[i, "a3_first_dx_km"] = (m["lon_first"].to_numpy(dtype=float) - lon0) * kx
    if GEOM is not None:
        st = GEOM["st"][["apt", "stand", "lat", "lon"]]
        s = pd.DataFrame({"apt": ap_i, "stand": pd.Series(np.asarray(stand)[i]).astype(str).to_numpy()}).merge(
            st, on=["apt", "stand"], how="left")
        for src, name in (("park", "a3_park_to_stand_m"), ("first", "a3_first_to_stand_m")):
            dy = (m[f"lat_{src}"].to_numpy(dtype=float) - s["lat"].to_numpy(dtype=float)) * 111200.0
            dx = (m[f"lon_{src}"].to_numpy(dtype=float) - s["lon"].to_numpy(dtype=float)) * kx * 1000.0
            out.loc[i, name] = np.hypot(dx, dy)
    return out


# ==== remote_inputs.py ====
# %%
# Inputs over URLs, for a kernel on an account that cannot mount the primary's private
# datasets and kernel outputs (miscusi).  build_kernel.py mints the URLs: signed GCS URLs for
# dataset files, tokenised kernel-output URLs; neither needs a credential on the VM.

INPUT_URLS = {'prc-taxi-2026-data/ranking.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/submitting.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-01-01_2025-02-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-02-01_2025-03-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-03-01_2025-04-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-04-01_2025-05-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-05-01_2025-06-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-06-01_2025-07-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-07-01_2025-08-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-08-01_2025-09-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-09-01_2025-10-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-10-01_2025-11-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-11-01_2025-12-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-12-01_2026-01-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-metar/metar_prc2026.parquet': '<redacted signed URL>', 'prc-taxi-2026-geometry/README.md': '<redacted signed URL>', 'prc-taxi-2026-geometry/build_geometry.py': '<redacted signed URL>', 'prc-taxi-2026-geometry/check_value.py': '<redacted signed URL>', 'prc-taxi-2026-geometry/coverage.csv': '<redacted signed URL>', 'prc-taxi-2026-geometry/graphs.csv': '<redacted signed URL>', 'prc-taxi-2026-geometry/runways.parquet': '<redacted signed URL>', 'prc-taxi-2026-geometry/stand_features_raw.parquet': '<redacted signed URL>', 'prc-taxi-2026-geometry/stand_runway.parquet': '<redacted signed URL>', 'prc-taxi-2026-geometry/stands.parquet': '<redacted signed URL>', 'prc-taxi-2026-geometry/value_check.csv': '<redacted signed URL>', 'adsb/prc-adsb-extract-202501w/adsb_departures_20250101_20250131_w600.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202502r/adsb_departures_20250201_20250228_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202503r/adsb_departures_20250301_20250331_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202504r/adsb_departures_20250401_20250430_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202505r/adsb_departures_20250501_20250531_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202506r/adsb_departures_20250601_20250630_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202507w/adsb_departures_20250701_20250731_w600.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202508r/adsb_departures_20250801_20250831_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202509r/adsb_departures_20250901_20250930_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202510r/adsb_departures_20251001_20251031_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202511r/adsb_departures_20251101_20251130_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202512r/adsb_departures_20251201_20251231_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202601r/adsb_departures_20260101_20260131_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202607r/adsb_departures_20260701_20260731_w600r.parquet': '<redacted signed URL>'}


def fetch_inputs(root: str = "/tmp/in") -> None:
    import urllib.request
    base = Path(root)
    for rel, url in INPUT_URLS.items():
        out = base / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=600) as resp, open(out, "wb") as fh:
                    while True:
                        chunk = resp.read(8 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"input retry {attempt} {rel}: {exc}", flush=True)
                time.sleep(10 * (attempt + 1))
    os.environ["DATA_ROOT"] = str(base)
    print(f"inputs: {len(INPUT_URLS)} files under {base}  used_h={used_h():.3f}", flush=True)


# ==== fs_main.py ====
# %%
# Feature store: build every departure's features once and write them with the encodings
# for each fold, so trainer kernels load matrices instead of rebuilding them.
#
# Folds: A = Jan + Jul 2025 days 16-31 (the v14-v27 hold-out), B = Jan + Jul days 1-15.
# Together they give out-of-fold predictions for all of Jan + Jul 2025.  "full" fits the
# encodings on every 2025 row for the ranking refit.  Encodings are leave-one-month-out
# for training rows, so each fold's file covers every training row.
#
# Outputs (float32 features, raw strings for categoricals):
#   tr_base.parquet  rk_base.parquet  enc_A.parquet  enc_B.parquet  enc_full.parquet
#   enc_full_rk.parquet  fs_meta.json


def fold_masks(month: np.ndarray, mday: np.ndarray) -> dict:
    jj = np.isin(month, VAL_MONTHS)
    # F: a forward winter fold (Nov + Dec 2025), fitted on Jan-Oct only (see fs_main)
    return {"A": jj & (mday >= 16), "B": jj & (mday <= 15), "F": np.isin(month, (11, 12))}


def _f32(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in out.columns:
        if out[c].dtype == np.float64 and c not in ("y", ID):  # ids exceed float32's exact range
            out[c] = out[c].astype(np.float32)
    return out


def fs_main():
    global METAR, ADSB
    if globals().get("INPUT_URLS"):
        fetch_inputs()
    data_dir = resolve_data_dir()
    print(f"data: {data_dir}", flush=True)
    METAR = load_metar()
    ADSB = load_adsb()
    globals()["METAR"], globals()["ADSB"] = METAR, ADSB
    globals()["GEOM"] = load_geometry()
    globals()["ADSB3"] = load_adsb3()
    raw = pd.concat((pd.read_parquet(p) for p in sorted(data_dir.glob("training_*.parquet"))), ignore_index=True)
    ranking = pd.read_parquet(data_dir / "ranking.parquet")
    tr = build(raw)
    del raw
    rk = build(ranking)
    tr = tr[tr["y"].notna()].reset_index(drop=True)
    print(f"built: train deps {len(tr):,}  ranking deps {len(rk):,}  used_h={used_h():.3f}", flush=True)
    add_nb_excess([tr, rk])
    print(f"nb excess: rk nbx_apt_1800 p50/p99 {np.nanpercentile(rk['nbx_apt_1800'], [50, 99])}  used_h={used_h():.3f}", flush=True)

    month = tr["month"].to_numpy()
    y = tr["y"].to_numpy()
    ds = tr["d_sched"].to_numpy()
    long = ds > LONG_DS
    path = long & (np.abs(y - ds) <= PATH_TOL)
    r = np.where(long, np.clip(y / np.where(long, ds, 1.0), 0, R_CLIP), np.nan)
    dd = tr["dd_utc"].to_numpy()
    artifact = (dd >= 1) & ((y - 86400 * dd) >= -900) & ((y - 86400 * dd) <= 3600)
    tr["artifact"] = artifact
    key_names = sorted(set(TE_KEYS + LONG_KEYS + ["apt"]))
    keys_tr = {k: tr[k].astype(str).to_numpy() for k in key_names}
    keys_rk = {k: rk[k].astype(str).to_numpy() for k in key_names}

    _f32(tr).to_parquet(OUTPUT_DIR / "tr_base.parquet", index=False)
    _f32(rk).to_parquet(OUTPUT_DIR / "rk_base.parquet", index=False)
    folds = fold_masks(month, tr["mday"].to_numpy())
    for name, val in folds.items():
        fit = ~val & ~artifact
        if name == "F":
            fit &= month <= 10  # forward: nothing after the validated months
        enc_tr, _ = all_encodings(keys_tr, month, y, ds, r, long, path, fit, keys_rk)
        _f32(enc_tr).to_parquet(OUTPUT_DIR / f"enc_{name}.parquet", index=False)
        print(f"enc_{name}: {enc_tr.shape}  val rows {int(val.sum()):,}  used_h={used_h():.3f}", flush=True)
    enc_all, enc_rk = all_encodings(keys_tr, month, y, ds, r, long, path, ~artifact, keys_rk)
    _f32(enc_all).to_parquet(OUTPUT_DIR / "enc_full.parquet", index=False)
    _f32(enc_rk).to_parquet(OUTPUT_DIR / "enc_full_rk.parquet", index=False)
    meta = {"n_tr": len(tr), "n_rk": len(rk), "folds": {k: int(v.sum()) for k, v in folds.items()},
            "artifact": int(artifact.sum()), "columns": list(tr.columns), "enc_columns": list(enc_all.columns),
            "used_h": used_h()}
    (OUTPUT_DIR / "fs_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"FS_DONE used_h={used_h():.3f}", flush=True)


if __name__ == "__main__":
    fs_main()
