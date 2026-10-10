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


# ==== train_main.py ====
# %%
# Trainer: load a feature store, fit the short/long pair per fold and on all of 2025.
#
# Short rows (takeoff <= 1 h after schedule): regression on min(y, 2 h).
# Long rows: regression on r = y / d (clipped to [0, 1.5]) weighted by min(d, 12 h)^2,
# so the loss is squared error in seconds; y = r * d.
#
# SPEC (filled by build_kernel.py) picks the library and parameters.  Folds A and B give
# out-of-fold predictions for all of Jan + Jul 2025; "full" refits on every 2025 row with
# 1.1 x the folds' mean best iteration and predicts the ranking file.
#
# Outputs: oof.csv (id, fold, prediction), submission.parquet, results.json.

import importlib
import subprocess

SPEC = {'lib': 'cat', 'gpus': 1, 'parallel': False, 'folds': ['B'], 'refit': False, 'max_rounds': 8000, 'split_by': 'apt', 'params': {'task_type': 'GPU', 'depth': 8, 'learning_rate': 0.08, 'l2_leaf_reg': 3, 'border_count': 254, 'one_hot_max_size': 4, 'max_ctr_complexity': 1, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'random_seed': 2311, 'gpu_ram_part': 0.9}, 'params_long': {'depth': 6}, 'fs': 'prc-fs5', 'fs_urls': {'enc_A.parquet': '<redacted signed URL>', 'enc_B.parquet': '<redacted signed URL>', 'enc_F.parquet': '<redacted signed URL>', 'enc_full.parquet': '<redacted signed URL>', 'enc_full_rk.parquet': '<redacted signed URL>', 'fs_meta.json': '<redacted signed URL>', 'rk_base.parquet': '<redacted signed URL>', 'tr_base.parquet': '<redacted signed URL>'}}
MAX_ROUNDS = int(SPEC.get("max_rounds", MAX_ROUNDS))


def ensure(pkg: str, wheel_glob: str | None = None):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        import glob as _glob
        whl = sorted(_glob.glob(f"/kaggle/input/**/{wheel_glob or pkg}-*.whl", recursive=True))
        if whl:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", whl[-1]], check=True)
        else:  # internet on (TPU hosts ship without these)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
        return importlib.import_module(pkg)


def ncpu() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


def fetch_urls(urls: dict, dest: Path) -> Path:
    import urllib.request
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in urls.items():
        out = dest / name
        if out.exists() and out.stat().st_size > 0:
            continue
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
                print(f"fetch retry {attempt} {name}: {exc}", flush=True)
                time.sleep(10 * (attempt + 1))
        print(f"fetched {name} {out.stat().st_size / 1e6:.1f} MB  used_h={used_h():.3f}", flush=True)
    return dest


def find_fs() -> Path:
    if os.environ.get("FS_URLS_FILE"):
        return fetch_urls(json.load(open(os.environ["FS_URLS_FILE"])), Path.cwd() / "fs")
    if SPEC.get("fs_urls"):
        return fetch_urls(SPEC["fs_urls"], Path("/tmp/fs") if Path("/kaggle").exists() else Path.cwd() / "fs")
    roots = [Path(os.environ["FS_ROOT"])] if os.environ.get("FS_ROOT") else [Path("/kaggle/input")]
    for root in roots:
        hits = sorted(root.rglob("fs_meta.json"))
        want = None if os.environ.get("FS_ROOT") else SPEC.get("fs")
        if want:
            hits = [h for h in hits if want in str(h)]
        if hits:
            return hits[0].parent
    raise FileNotFoundError("no feature store (fs_meta.json) mounted")


# ---- libraries -------------------------------------------------------------------------
def _norm_w(w):
    return None if w is None else w / np.mean(w)


class LGB:
    def __init__(self, params, threads):
        self.lgb = ensure("lightgbm")
        self.p = {**params, "num_threads": threads, "verbose": -1}

    def fit(self, X, y, w, Xv, yv, wv, rounds=None):
        lgb = self.lgb
        if rounds is None and (Xv is None or len(Xv) == 0):
            rounds = 300  # nothing to early-stop on (a small group in a fold)
        dtr = lgb.Dataset(X, y, weight=_norm_w(w), categorical_feature=self.cats)
        if rounds is None:
            dva = lgb.Dataset(Xv, yv, weight=_norm_w(wv), categorical_feature=self.cats, reference=dtr)
            m = lgb.train(self.p, dtr, MAX_ROUNDS, valid_sets=[dva], valid_names=["val"],
                          callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(1000)])
            return m, int(m.best_iteration)
        return lgb.train(self.p, dtr, rounds), rounds

    def predict(self, m, X, best=None):
        return m.predict(X, num_iteration=best)

    def importance(self, m, names):
        return pd.Series(m.feature_importance("gain"), index=names)


class XGB:
    def __init__(self, params, threads):
        self.xgb = ensure("xgboost")
        self.p = {"tree_method": "hist", "objective": "reg:squarederror", "nthread": threads, **params}

    def fit(self, X, y, w, Xv, yv, wv, rounds=None):
        xgb = self.xgb
        if rounds is None and (Xv is None or len(Xv) == 0):
            rounds = 300
        dtr = xgb.QuantileDMatrix(X, y, weight=_norm_w(w), enable_categorical=True,
                                  max_bin=self.p.get("max_bin", 256))
        if rounds is None:
            dva = xgb.QuantileDMatrix(Xv, yv, weight=_norm_w(wv), enable_categorical=True, ref=dtr)
            m = xgb.train(self.p, dtr, MAX_ROUNDS, evals=[(dva, "val")], early_stopping_rounds=300,
                          verbose_eval=1000)
            return m, int(m.best_iteration) + 1
        return xgb.train(self.p, dtr, rounds), rounds

    def predict(self, m, X, best=None):
        d = self.xgb.DMatrix(X, enable_categorical=True)
        return m.predict(d, iteration_range=(0, best) if best else (0, 0))

    def importance(self, m, names):
        g = m.get_score(importance_type="total_gain")
        return pd.Series({n: g.get(n, 0.0) for n in names})


class CAT:
    def __init__(self, params, threads):
        self.cb = ensure("catboost")
        self.p = {"loss_function": "RMSE", "thread_count": threads, "verbose": 1000, **params}

    def _pool(self, X, y=None, w=None):
        X = X.copy()
        for c in self.cats:
            X[c] = X[c].astype(str)
        return self.cb.Pool(X, y, weight=_norm_w(w), cat_features=self.cats)

    def fit(self, X, y, w, Xv, yv, wv, rounds=None):
        if rounds is None and (Xv is None or len(Xv) == 0):
            rounds = 300
        if rounds is None:
            m = self.cb.CatBoostRegressor(iterations=MAX_ROUNDS, od_type="Iter", od_wait=300,
                                          use_best_model=True, **self.p)
            m.fit(self._pool(X, y, w), eval_set=self._pool(Xv, yv, wv))
            return m, int(m.get_best_iteration()) + 1
        m = self.cb.CatBoostRegressor(iterations=rounds, **self.p)
        m.fit(self._pool(X, y, w))
        return m, rounds

    def predict(self, m, X, best=None):
        return m.predict(self._pool(X))

    def importance(self, m, names):
        return pd.Series(m.get_feature_importance(), index=names)


LIBS = {"lgb": LGB, "xgb": XGB, "cat": CAT}


# ---- training ----------------------------------------------------------------------------
def _anchor_parts(X: pd.DataFrame):
    spec = SPEC["anchor"]
    fb = X[spec.get("fallback", "te_stand_rwy")].to_numpy(dtype=float)
    nb = spec.get("nb")
    if nb and nb["col"] in X:
        # congestion measured on the neighbours' ADS-B, added linearly so it extrapolates
        x = np.clip(X[nb["col"]].to_numpy(dtype=float), *nb.get("clip", (-300.0, 3000.0)))
        n = X[nb["col"].replace("nbx_", "nbn_")].to_numpy(dtype=float)
        fb = fb + np.where((n >= nb.get("min_n", 3)) & np.isfinite(x), nb.get("k", 0.5) * x, 0.0)
    obs = X[spec["obs"]].to_numpy(dtype=float) if spec["obs"] in X else np.full(len(X), np.nan)
    ok = np.isfinite(obs) & (obs > spec.get("min", 30)) & (obs < spec.get("max", 7200))
    apt = X["apt"].astype(str).to_numpy()
    return fb, obs, ok, apt


def short_anchor(X: pd.DataFrame, y: np.ndarray, fit: np.ndarray) -> np.ndarray | None:
    """Anchor (SPEC["anchor"]): the measured taxi (ADS-B first movement + a per-airport
    offset) where it is plausible, else the stand x runway mean (+ k x neighbour excess).
    The models learn y - anchor.  The offsets are fitted on `fit` rows only."""
    if not SPEC.get("anchor"):
        return None
    fb, obs, ok, apt = _anchor_parts(X)
    off = pd.Series(y - obs).where(ok & fit & (y < CLIP_TARGET)).groupby(apt).median()
    ANCHOR["off"] = off
    ANCHOR["fb_med"] = float(np.nanmedian(fb))
    return anchor_apply(X)


def anchor_apply(X: pd.DataFrame) -> np.ndarray:
    fb, obs, ok, apt = _anchor_parts(X)
    a = np.where(ok, obs + pd.Series(apt).map(ANCHOR["off"]).fillna(0).to_numpy(), fb)
    return np.where(np.isfinite(a), a, ANCHOR["fb_med"])


def fit_pair(libs, X, y, ds, r, long, fit, valid, rounds=None):
    """One short/long pair, or one per group of SPEC["split_by"] (per-airport models); in the
    split case `rounds` is {group: rounds} and the result is {"groups": {group: pair}}."""
    if SPEC.get("split_by"):
        grp = X[SPEC["split_by"]].astype(str).to_numpy()
        out = {}
        for g in np.unique(grp):
            m = grp == g
            if not (fit & m).any():
                continue
            rg = None if rounds is None else rounds.get(g, rounds.get("*"))
            out[g] = fit_one(libs, X.loc[m], y[m], ds[m], r[m], long[m], fit[m],
                             None if valid is None else valid[m], rg)
        return {"groups": out}
    return fit_one(libs, X, y, ds, r, long, fit, valid, rounds)


def fit_one(libs, X, y, ds, r, long, fit, valid, rounds=None):
    ls, ll = libs
    w = np.minimum(ds, W_CAP) ** 2
    ys = np.minimum(y, CLIP_TARGET)
    anc = short_anchor(X, y, fit)
    ANCHOR["a"] = anc
    if anc is not None:
        ys = ys - anc
        if SPEC["anchor"].get("long"):
            # long model: residual on the anchor in ratio space, r - a/d, still weighted by d^2,
            # so its loss stays squared error in seconds of y - (a + f d)
            r = r - np.where(long, anc / np.where(long, ds, 1.0), 0.0)
    if SPEC.get("long_mode") == "mix":
        return fit_mix(libs, X, y, ds, long, fit, valid, rounds)
    sf, lf = fit & ~long, fit & long
    if rounds is None:
        sv, lv = valid & ~long, valid & long
        ms, bs = ls.fit(X.loc[sf], ys[sf], None, X.loc[sv], ys[sv], None)
        ml, bl = ll.fit(X.loc[lf], r[lf], w[lf], X.loc[lv], r[lv], w[lv])
    else:
        ms, bs = ls.fit(X.loc[sf], ys[sf], None, None, None, None, rounds[0])
        ml, bl = ll.fit(X.loc[lf], r[lf], w[lf], None, None, None, rounds[1])
    return (ms, bs), (ml, bl)


MIX_HELD_R = 0.6


def mix_classes(y, ds, long):
    """0 gate-held (taxied normally after a gate delay), 1 held after pushback, 2 schedule fill."""
    fill = long & (np.abs(y - ds) <= PATH_TOL)
    held = long & ~fill & (y >= MIX_HELD_R * ds)
    return fill, held, np.where(fill, 2, np.where(held, 1, 0))


def fit_mix(libs, X, y, ds, long, fit, valid, rounds=None):
    """Mixture for long rows: a taxi regressor in seconds trained on short and gate-held rows,
    a 3-class classifier {gate, held, fill} on long rows, and a ratio regressor on held rows.
    Prediction for a long row: p_gate * taxi + p_held * r_held * d + p_fill * d."""
    ls, ll = libs
    assert SPEC["lib"] == "lgb", "mixture mode is LightGBM-only"
    fill, held, cls = mix_classes(y, ds, long)
    gate = long & ~fill & ~held
    ys = np.minimum(y, CLIP_TARGET)
    w = np.minimum(ds, W_CAP) ** 2
    r = np.clip(y / np.where(long, ds, 1.0), 0, R_CLIP)
    mix = SPEC.get("mix", {})
    lc = LIBS[SPEC["lib"]]({**ll.p, **mix.get("clf_params", {}), "objective": "multiclass", "num_class": 3,
                             "metric": "multi_logloss"}, ll.p.get("num_threads", 4))
    lc.cats = ll.cats
    wc = w if mix.get("clf_weight") == "d2" else None
    tg, vg = fit & (~long | gate), (valid & (~long | gate)) if valid is not None else None
    tl = fit & long
    th = fit & held
    few_held = int(th.sum()) < 30  # too few held rows: a constant held ratio instead of a model
    if rounds is None:
        vl, vh = valid & long, valid & held
        mt, bt = ls.fit(X.loc[tg], ys[tg], None, X.loc[vg], ys[vg], None)
        mc, bc = lc.fit(X.loc[tl], cls[tl], None if wc is None else wc[tl], X.loc[vl], cls[vl],
                        None if wc is None else wc[vl])
        if few_held:
            mh, bh = float(np.median(r[th])) if th.any() else 0.75, 0
        else:
            mh, bh = ll.fit(X.loc[th], r[th], w[th], X.loc[vh], r[vh], w[vh])
    else:
        mt, bt = ls.fit(X.loc[tg], ys[tg], None, None, None, None, rounds[0])
        mc, bc = lc.fit(X.loc[tl], cls[tl], None if wc is None else wc[tl], None, None, None, rounds[1])
        if few_held:
            mh, bh = float(np.median(r[th])) if th.any() else 0.75, 0
        else:
            mh, bh = ll.fit(X.loc[th], r[th], w[th], None, None, None, max(rounds[2], 50))
    MIXLIB["clf"] = lc
    return [(mt, bt), (mc, bc), (mh, bh)]


MIXLIB = {}


def pred_mix(libs, pair, X, ds, long, use_best=True):
    ls, ll = libs
    (mt, bt), (mc, bc), (mh, bh) = pair
    out = ls.predict(mt, X, bt if use_best else None).copy()
    if long.any():
        XL = X.loc[long]
        p = np.asarray(MIXLIB["clf"].predict(mc, XL, bc if use_best else None)).reshape(-1, 3)
        rh = np.full(len(XL), mh) if isinstance(mh, float) else ll.predict(mh, XL, bh if use_best else None)
        rh = np.clip(rh, MIX_HELD_R * 0.5, R_CLIP)
        d = ds[long]
        out[long] = p[:, 0] * out[long] + p[:, 1] * rh * d + p[:, 2] * d
    return out


ANCHOR = {}


def pred_pair(libs, pair, X, ds, long, use_best=True):
    if isinstance(pair, dict):
        grp = X[SPEC["split_by"]].astype(str).to_numpy()
        out = np.full(len(X), np.nan)
        for g, pg in pair["groups"].items():
            m = grp == g
            if m.any():
                out[m] = pred_one(libs, pg, X.loc[m], ds[m], long[m], use_best)
        return out
    return pred_one(libs, pair, X, ds, long, use_best)


def pred_one(libs, pair, X, ds, long, use_best=True):
    if SPEC.get("long_mode") == "mix":
        return pred_mix(libs, pair, X, ds, long, use_best)
    ls, ll = libs
    (ms, bs), (ml, bl) = pair
    out = ls.predict(ms, X, bs if use_best else None)
    out_anchor = anchor_apply(X) if SPEC.get("anchor") else None
    if out_anchor is not None:
        # the offset table comes from the fit rows of this pair (stored at fit time)
        out = out + out_anchor
    if long.any():
        pr = ll.predict(ml, X.loc[long], bl if use_best else None)
        if SPEC.get("anchor", {}).get("long"):
            pr = pr + (out_anchor[long] / ds[long])
        pr = np.clip(pr, 0, R_CLIP)
        out = out.copy()
        out[long] = pr * ds[long]
    return out


def load_matrices(fs: Path, lib_name: str):
    tr = pd.read_parquet(fs / "tr_base.parquet")
    rk = pd.read_parquet(fs / "rk_base.parquet")
    tr["y"] = tr["y"].astype(np.float64)
    if lib_name != "cat":
        categorize(tr, [rk], ref_mask=np.ones(len(tr), bool))
    drop = set(DROP) | {"artifact"} | set(SPEC.get("drop_features", []))
    feats = [c for c in tr.columns if c not in drop]
    return tr, rk, feats


def assemble(base, enc, feats):
    enc = enc.astype(np.float32)
    X = add_ratios(pd.concat([base[feats].reset_index(drop=True), enc.reset_index(drop=True)], axis=1))
    drop = [c for c in SPEC.get("drop_features", []) if c in X]
    return X.drop(columns=drop)


def make_libs(lib_name, params, params_long, threads, cats, seed_offset=0, gpu=None):
    out = []
    for p in (params, {**params, **(params_long or {})}):
        p = dict(p)
        if gpu is not None and lib_name == "xgb":
            p["device"] = f"cuda:{gpu}"
        if gpu is not None and lib_name == "cat" and p.get("task_type") == "GPU":
            p["devices"] = str(gpu)
        for key in ("seed", "random_seed", "random_state"):
            if key in p:
                p[key] = int(p[key]) + seed_offset
        lib = LIBS[lib_name](p, threads)
        lib.cats = cats
        out.append(lib)
    return tuple(out)


def run_fold(name, lib, X, tr, fold_val, threads):
    y = tr["y"].to_numpy()
    ds = tr["d_sched"].to_numpy()
    long = ds > LONG_DS
    r = np.where(long, np.clip(y / np.where(long, ds, 1.0), 0, R_CLIP), np.nan)
    art = tr["artifact"].to_numpy().astype(bool)
    t0 = time.time()
    pair = fit_pair(lib, X, y, ds, r, long, ~fold_val & ~art, fold_val & ~art)
    pv = pred_pair(lib, pair, X.loc[fold_val].reset_index(drop=True), ds[fold_val], long[fold_val])
    keep = ~art[fold_val]
    score = rmse(y[fold_val][keep], pv[keep])
    first = next(iter(pair["groups"].values())) if isinstance(pair, dict) else pair
    print(f"[fold {name}] test-like {score:.3f}  short {first[0][1]}  long {first[1][1]}  "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if isinstance(pair, dict):
        best = {g: tuple(int(m[1]) for m in p) for g, p in pair["groups"].items()}
    else:
        best = tuple(int(m[1]) for m in pair)
    return pv, best, score, pair


STATE = {}


def _targets(tr):
    y = tr["y"].to_numpy()
    ds = tr["d_sched"].to_numpy()
    long = ds > LONG_DS
    r = np.where(long, np.clip(y / np.where(long, ds, 1.0), 0, R_CLIP), np.nan)
    art = tr["artifact"].to_numpy().astype(bool)
    return y, ds, long, r, art


def fold_job(name: str, threads: int) -> None:
    g = STATE
    gpus = int(SPEC.get("gpus", 0))
    gpu = (sorted(g["folds"]).index(name) % gpus) if gpus else None
    lib = make_libs(g["lib_name"], g["params"], g["params_long"], threads, g["cats"], gpu=gpu)
    X = assemble(g["tr"], pd.read_parquet(g["fs"] / f"enc_{name}.parquet"), g["feats"])
    val = g["folds"][name]
    pv, best, score, pair = run_fold(name, lib, X, g["tr"], val, threads)
    pd.DataFrame({"id": g["tr"].loc[val, ID].to_numpy(), "fold": name, "prediction": pv}).to_csv(
        OUTPUT_DIR / f"oof_{name}.csv", index=False)
    json.dump({"best": best if isinstance(best, dict) else list(best), "score": score},
              open(OUTPUT_DIR / f"best_{name}.json", "w"))
    first = next(iter(pair["groups"].values())) if isinstance(pair, dict) else pair
    lib[0].importance(first[0][0], list(X.columns)).sort_values(ascending=False).to_csv(OUTPUT_DIR / f"imp_{name}.csv")


def refit_job(k: int, threads: int) -> None:
    g = STATE
    gpus = int(SPEC.get("gpus", 0))
    lib = make_libs(g["lib_name"], g["params"], g["params_long"], threads, g["cats"], seed_offset=1000 * k,
                    gpu=(k % gpus) if gpus else None)
    y, ds, long, r, art = _targets(g["tr"])
    pair = fit_pair(lib, g["X_all"], y, ds, r, long, ~art, None, g["rounds"])
    rk_ds = g["rk"]["d_sched"].to_numpy()
    pred = pred_pair(lib, pair, g["X_rk"], rk_ds, rk_ds > LONG_DS, use_best=False)
    np.save(OUTPUT_DIR / f"refit_{k}.npy", pred)
    print(f"refit {k}: rounds {g['rounds']}  used_h={used_h():.3f}", flush=True)


def _run(jobs, parallel: bool, total_threads: int):
    if not parallel or len(jobs) == 1:
        for fn, arg in jobs:
            fn(arg, total_threads)
        return
    from multiprocessing import get_context
    ctx = get_context("fork")
    per = max(total_threads // len(jobs), 1)
    procs = [ctx.Process(target=fn, args=(arg, per)) for fn, arg in jobs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode]
    if bad:
        raise RuntimeError(f"worker exit codes {bad}")


def train_main():
    spec = SPEC
    lib_name = spec["lib"]
    threads = int(spec.get("threads", ncpu()))
    parallel = bool(spec.get("parallel", threads >= 32))
    fs = find_fs()
    print(f"spec {json.dumps(spec)}\nfeature store {fs}  cores {threads}  parallel {parallel}", flush=True)
    tr, rk, feats = load_matrices(fs, lib_name)
    month, mday = tr["month"].to_numpy(), tr["mday"].to_numpy()
    jj = np.isin(month, VAL_MONTHS)
    STATE.update(dict(lib_name=lib_name, params=spec["params"], params_long=spec.get("params_long"), cats=[c for c in CATS if c in feats], tr=tr,
                      rk=rk, feats=feats, fs=fs, folds={"A": jj & (mday >= 16), "B": jj & (mday <= 15),
                                                        "F": np.isin(month, (11, 12))}))
    print(f"loaded {len(tr):,} x {len(feats)}  used_h={used_h():.3f}", flush=True)

    names = spec.get("folds", ["A", "B"])
    info = {}
    if names:
        _run([(fold_job, n) for n in names], parallel, threads)
        oof = pd.concat([pd.read_csv(OUTPUT_DIR / f"oof_{n}.csv") for n in names])
        oof.to_csv(OUTPUT_DIR / "oof.csv", index=False)
        oof.to_parquet(OUTPUT_DIR / "oof.parquet", index=False)
        info = {n: json.load(open(OUTPUT_DIR / f"best_{n}.json")) for n in names}
        imp = pd.read_csv(OUTPUT_DIR / f"imp_{names[0]}.csv", index_col=0).iloc[:, 0]
        print(f"top short features:\n{imp.head(25).to_string()}", flush=True)
    else:
        # refit-only: rounds come from an earlier run's folds (SPEC["rounds"] = [short, long] best iters)
        given = spec["rounds"]
        info = {"given": {"best": given if isinstance(given, dict) else list(given), "score": None}}
    results = {"spec": spec, "folds": info}
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=1))

    if spec.get("refit", True):
        X_all = assemble(tr, pd.read_parquet(fs / "enc_full.parquet"), feats)
        X_rk = assemble(rk, pd.read_parquet(fs / "enc_full_rk.parquet"), feats)[list(X_all.columns)]
        bests = [v["best"] for v in info.values()]
        if isinstance(bests[0], dict):  # per-group rounds, averaged over the folds that have the group
            groups = set().union(*bests)
            rounds = {g: tuple(max(int(x * 1.1), 50) for x in
                               np.mean([b[g] for b in bests if g in b], axis=0)) for g in groups}
        else:
            b = np.array(bests, dtype=float).mean(axis=0)
            rounds = tuple(max(int(x * 1.1), 50) for x in b)
        STATE.update(dict(X_all=X_all, X_rk=X_rk, rounds=rounds))
        seeds = int(spec.get("refit_seeds", 1))
        _run([(refit_job, k) for k in range(seeds)], parallel, threads)
        pred = np.mean([np.load(OUTPUT_DIR / f"refit_{k}.npy") for k in range(seeds)], axis=0)
        sub = pd.DataFrame({ID: rk[ID].to_numpy(), TARGET: np.clip(pred, 60, None)})
        sub.to_parquet(OUTPUT_DIR / "submission.parquet", index=False)
        results.update({"refit_rounds": rounds, "test_mean": float(pred.mean()), "test_max": float(pred.max())})
    results["used_h"] = used_h()
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=1))
    print(f"TRAIN_DONE used_h={used_h():.3f}", flush=True)


if __name__ == "__main__":
    train_main()
