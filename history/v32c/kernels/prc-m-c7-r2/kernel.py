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


# fs7 options (empty = the fs6 store, unchanged).  build_kernel.py --opts fills them:
#   match: "v7"     one-to-one ADS-B event <-> flight assignment with callsign aliases (IATA -> ICAO),
#                   a 900 s window on takeoff / first airborne point; time-only matches feed own-flight
#                   columns only when the aircraft type agrees and the first airborne point lines up
#                   (match conf 1.5); the rest are demoted to identity-free event-pool columns (evp_*)
#   airline: "icao" airline key from the callsign / mapped IATA designator (not FLIGHT_mvt[:3]);
#                   stand_lb_other, the inbound link and a next-occupant bound use it
#   pru: true       EUROCONTROL PRU published airport-month taxi-out levels (PRU_TAB)
#   demean: true    airport-month-demeaned NM lags (label-free, per frame)
#   ext: "glob"     merge an external per-movement table keyed by MVT_ID_mvt (e.g. opdi_*.parquet)
#   openflights: true  fill IATA designators missing from the data-derived map from OpenFlights
#                   airlines.dat (ODbL 1.0, openflights.org/data), fetched in the kernel
FS_OPTS = {}  # replaced by build_kernel.py: fs7 options
PRU_TAB = {}  # replaced by build_kernel.py: {"APT|YYYY|M": {pru_pub, pru_ref, pru_deice_sh, pru_noref_sh, pru_pub_dev}}
IATA_MAP: dict = {}


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


def mean_between_ex(q_key, q_lo, q_hi, r_key, r_t, r_v, own_v):
    """Sum and count of r values in [q_lo, q_hi) per query (nb_delay_features takes the
    query's own value out when the query is itself one of the r events)."""
    s = np.full(len(q_key), np.nan)
    n = np.zeros(len(q_key))
    ok_r = ~(np.isnan(r_t) | np.isnan(r_v))
    r_key, r_t, r_v = np.asarray(r_key)[ok_r], r_t[ok_r], r_v[ok_r]
    for qi, ri in _by_key(q_key, r_key):
        order = np.argsort(r_t[ri], kind="stable")
        ts, vs = r_t[ri][order], r_v[ri][order]
        cs = np.concatenate([[0.0], np.cumsum(vs)])
        lo, hi = q_lo[qi], q_hi[qi]
        a = np.searchsorted(ts, lo, "left")
        b = np.searchsorted(ts, hi, "left")
        s[qi], n[qi] = cs[b] - cs[a], b - a
    return s, n


def nb_delay_features(F, qk, qt, own, rk, rt, rv, name, windows):
    """Mean of neighbours' values around our takeoff (self excluded), per window spec
    (label, lo offset, hi offset, own-in-window)."""
    for lab, lo, hi, incl in windows:
        s, n = mean_between_ex(qk, qt + lo, qt + hi, rk, rt, rv, own)
        if incl:
            has = ~np.isnan(own)
            s = np.where(has, s - np.nan_to_num(own), s)
            n = np.where(has, n - 1, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            F[f"{name}_{lab}"] = np.where(n >= 2, s / np.maximum(n, 1), np.nan)
        F[f"{name}_n_{lab}"] = n


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


def canon(s) -> np.ndarray:
    """Canonical callsign: upper case, alphanumerics only, leading zeros of an ICAO flight number
    dropped (DLH0796 -> DLH796)."""
    s = pd.Series(s).fillna("").astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    return s.str.replace(r"^([A-Z]{3})0+(?=[0-9A-Z])", r"\1", regex=True).to_numpy()


def _is_iata(f: pd.Series) -> pd.Series:
    return f.str.match(r"^[A-Z0-9]{2}\d") & ~f.str.match(r"^[A-Z]{3}\d")


def build_iata_map(flight, cs) -> dict:
    """IATA designator (2 chars) -> ICAO prefix (3 letters), learned from rows that carry an IATA flight
    number and an ICAO callsign; a designator needs >= 5 rows and a >= 50% majority."""
    f = pd.Series(flight).fillna("").astype(str).str.upper().str.replace(" ", "", regex=False)
    c = pd.Series(cs).fillna("").astype(str).str.upper().str.strip()
    ok = _is_iata(f) & c.str.match(r"^[A-Z]{3}")
    tab = pd.DataFrame({"p": f[ok].str[:2].to_numpy(), "q": c[ok].str[:3].to_numpy()}).value_counts().rename("n").reset_index()
    tot = tab.groupby("p")["n"].transform("sum")
    tab = tab[(tab["n"] >= 5) & (tab["n"] >= 0.5 * tot)].drop_duplicates("p")
    return dict(zip(tab["p"], tab["q"]))


def openflights_map() -> dict:
    """Active airlines' IATA -> ICAO from OpenFlights airlines.dat (ODbL 1.0), unique designators only."""
    import io
    import urllib.request
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read().decode("utf-8", "replace")
        a = pd.read_csv(io.StringIO(raw), header=None, na_values=["\\N"], keep_default_na=False,
                        names=["id", "name", "alias", "iata", "icao", "callsign", "country", "active"])
        a = a[(a["active"] == "Y") & a["iata"].astype(str).str.match(r"^[A-Z0-9]{2}$")
              & a["icao"].astype(str).str.match(r"^[A-Z]{3}$")]
        a = a[~a["iata"].duplicated(keep=False)]
        return dict(zip(a["iata"], a["icao"]))
    except Exception as exc:  # noqa: BLE001
        print(f"openflights: unavailable ({exc}); data-derived IATA map only", flush=True)
        return {}


def icao_alias(flight) -> np.ndarray:
    f = pd.Series(flight).fillna("").astype(str).str.upper().str.replace(" ", "", regex=False)
    pre = f.str[:2].map(IATA_MAP)
    return canon(np.where(_is_iata(f) & pre.notna(), pre.fillna("") + f.str[2:], ""))


def airline_key(cs, flight) -> np.ndarray:
    """ICAO airline: the callsign prefix when it is three letters, else the flight number's ICAO prefix
    or its mapped IATA designator; unknown designators keep a '?'-marked IATA prefix."""
    c = pd.Series(cs).fillna("").astype(str).str.upper().str.strip()
    f = pd.Series(flight).fillna("").astype(str).str.upper().str.replace(" ", "", regex=False)
    a = np.where(c.str.match(r"^[A-Z]{3}"), c.str[:3], "")
    fi = np.where(f.str.match(r"^[A-Z]{3}\d"), f.str[:3], f.str[:2].map(IATA_MAP).fillna("").to_numpy())
    a = np.where(a == "", fi, a)
    return np.where(a == "", "?" + f.str[:2].to_numpy(), a).astype(object)


def _mutual(p: pd.DataFrame, cost: str = "cost") -> pd.DataFrame:
    """Greedy one-to-one assignment: repeatedly keep pairs that are each other's cheapest."""
    p = p.sort_values([cost, "i", "j"], kind="stable")
    keep = []
    while len(p):
        bf = p.drop_duplicates("i")
        be = p.drop_duplicates("j")
        m = bf.merge(be[["i", "j"]], on=["i", "j"])
        if not len(m):
            break
        keep.append(m)
        p = p[~p["i"].isin(m["i"]) & ~p["j"].isin(m["j"])]
    return pd.concat(keep, ignore_index=True) if keep else p.iloc[:0].copy()


MATCH_WIN = 900.0     # callsign matches: |takeoff or first-airborne offset| within this
TONLY_WIN = 60.0      # time-only candidates
TONLY_NA_TOL = 30.0   # a time-only match is credible when the first airborne point lines up this well


def match_events(q_apt, q_t, q_alias: list, q_type, e_apt, e_to, e_na, e_keys: list, e_keys2: list, e_type,
                 tag: str = "") -> pd.DataFrame:
    """One-to-one flight <-> ADS-B event assignment per airport.  Returns i (flight), j (event), conf:
    3 = primary callsign, 2 = another callsign the chain broadcast, 1.5 = time-only with the aircraft type
    agreeing and the first airborne point lined up, 1 = other time-only (not this flight's own evidence)."""
    q_apt = np.asarray(q_apt).astype(str)
    e_apt = np.asarray(e_apt).astype(str)
    q_alias = [np.asarray(a_).astype(str) for a_ in q_alias]
    q_t = np.asarray(q_t, dtype=float)
    e_to = np.asarray(e_to, dtype=float)
    e_na = np.asarray(e_na, dtype=float)
    nq, ne = len(q_t), len(e_to)
    cols = ["i", "j", "conf", "dto", "dna"]
    if nq == 0 or ne == 0:
        return pd.DataFrame(columns=cols)
    # Candidate pairs share airport, a key and a 30-min time bucket (either of the event's anchors,
    # flights probe the neighbouring buckets too): joining on the callsign alone pairs a flight with
    # every day its callsign flew and does not fit in memory over 14 months.
    B = 1800.0
    parts = []
    for ap in np.unique(q_apt):
        qi = np.flatnonzero(q_apt == ap)
        ej = np.flatnonzero(e_apt == ap)
        if not len(ej):
            continue
        Q = pd.concat([pd.DataFrame({"i": qi, "k": np.asarray(a_)[qi]}) for a_ in q_alias])
        Q = Q[(Q["k"].str.len() >= 3).to_numpy() & ~np.isnan(q_t[Q["i"].to_numpy()])].drop_duplicates(["i", "k"])
        qb = np.floor(q_t[Q["i"].to_numpy()] / B)
        Q = pd.concat([Q.assign(b=qb + o) for o in (-1, 0, 1)])
        E = pd.concat([pd.DataFrame({"j": ej, "k": np.asarray(k_)[ej], "lvl": lvl})
                       for lvl, keys in ((0, e_keys), (1, e_keys2)) for k_ in keys])
        E = E[(E["k"].str.len() >= 3).to_numpy()].sort_values("lvl", kind="stable").drop_duplicates(["j", "k"])
        jj_ = E["j"].to_numpy()
        E = pd.concat([E.assign(b=np.floor(e_to[jj_] / B)), E.assign(b=np.floor(e_na[jj_] / B))])
        E = E[~np.isnan(E["b"].to_numpy())].drop_duplicates(["j", "k", "b"])
        mm = Q.merge(E, on=["k", "b"])[["i", "j", "lvl"]]
        parts.append(mm.sort_values("lvl", kind="stable").drop_duplicates(["i", "j"]))
    m = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["i", "j", "lvl"])
    m = m.sort_values("lvl", kind="stable").drop_duplicates(["i", "j"])
    ii, jj = m["i"].to_numpy(dtype=int), m["j"].to_numpy(dtype=int)
    dto = e_to[jj] - q_t[ii]
    dna = e_na[jj] - q_t[ii]
    near = (m["lvl"].to_numpy() == 0) & (np.abs(dto) <= 300)
    o_to = float(np.nanmedian(dto[near])) if near.sum() >= 50 else 0.0
    o_na = float(np.nanmedian(dna[near])) if np.isfinite(dna[near]).sum() >= 50 else np.nan
    c = np.fmin(np.abs(dto - o_to), np.abs(dna - o_na))
    m = m.assign(dto=dto, dna=dna, cost=c + 300.0 * m["lvl"].to_numpy())
    m = m[c <= MATCH_WIN]
    cs = _mutual(m[["i", "j", "cost", "lvl", "dto", "dna"]])
    cs["conf"] = np.where(cs["lvl"].to_numpy() == 0, 3.0, 2.0)
    # time-only: flights and events left over, takeoff within TONLY_WIN, one-to-one
    qi = np.setdiff1d(np.arange(nq), cs["i"].to_numpy())
    ej = np.setdiff1d(np.arange(ne), cs["j"].to_numpy())
    qi = qi[~np.isnan(q_t[qi])]
    ej = ej[~np.isnan(e_to[ej])]
    tq = pd.DataFrame({"i": qi, "apt": q_apt[qi], "b": np.floor(q_t[qi] / TONLY_WIN)})
    te = pd.DataFrame({"j": ej, "apt": e_apt[ej], "b0": np.floor((e_to[ej] - o_to) / TONLY_WIN)})
    te = pd.concat([te.assign(b=te["b0"] + o) for o in (-1, 0, 1)])
    mt = tq.merge(te[["j", "apt", "b"]], on=["apt", "b"])
    mt["dto"] = e_to[mt["j"].to_numpy()] - q_t[mt["i"].to_numpy()]
    mt["dna"] = e_na[mt["j"].to_numpy()] - q_t[mt["i"].to_numpy()]
    mt["cost"] = np.abs(mt["dto"].to_numpy() - o_to)
    mt = mt[mt["cost"] <= TONLY_WIN]
    to = _mutual(mt[["i", "j", "cost", "dto", "dna"]])
    qt_ = pd.Series(np.asarray(q_type)[to["i"].to_numpy()]).astype(str).str.upper().to_numpy()
    et_ = pd.Series(np.asarray(e_type)[to["j"].to_numpy()]).astype(str).str.upper().to_numpy()
    lined = np.abs(to["dna"].to_numpy() - o_na) <= TONLY_NA_TOL if np.isfinite(o_na) else np.zeros(len(to), bool)
    to["conf"] = np.where((qt_ == et_) & lined, 1.5, 1.0)
    out = pd.concat([cs[cols], to[cols]], ignore_index=True)
    vc = out["conf"].value_counts().to_dict()
    print(f"match {tag}: flights {nq:,} events {ne:,} offsets to {o_to:.1f} na {o_na:.1f}  conf counts "
          f"{ {k: vc[k] for k in sorted(vc)} }", flush=True)
    return out


def _event_keys(ev: pd.DataFrame, primary: list, secondary: list):
    k1 = [canon(ev[c]) for c in primary if c in ev]
    k2 = []
    for c in secondary:
        if c in ev:
            parts = ev[c].fillna("").astype(str).str.split("|")
            for p in range(int(min(parts.str.len().max() or 0, 8))):
                k2.append(canon(parts.str[p].fillna("").replace("@@@@@@@@", "")))
    return k1, k2


def blank_adsb(X: pd.DataFrame, apt, airports, cols) -> None:
    """Blank ADS-B features in place at airports without 2025 ground coverage."""
    rows = np.isin(apt, airports)
    X.loc[rows, cols] = np.nan
    X.loc[rows, "adsb_found"] = 0.0


def adsb_features(apt, t, cs1, cs2, actype=None) -> pd.DataFrame:
    n = len(t)
    raw = pd.DataFrame(np.nan, index=np.arange(n), columns=ADSB_RAW + ["cs_match"])
    if ADSB is None:
        return raw
    if FS_OPTS.get("match") == "v7":
        a = ADSB.dropna(subset=["t_takeoff"]).reset_index(drop=True)
        k1, k2 = _event_keys(a, ["callsign"], [])
        na = a["t_air"].to_numpy(dtype=float) if "t_air" in a else np.full(len(a), np.nan)
        mm = match_events(apt, t, [canon(cs1), canon(cs2), icao_alias(cs2)],
                          actype if actype is not None else np.full(n, ""), a["apt"].to_numpy(),
                          a["t_takeoff"].to_numpy(dtype=float), na, k1, k2,
                          a["actype"].to_numpy() if "actype" in a else np.full(len(a), ""), tag="adsb v1")
        own = mm[mm["conf"] >= 1.5]
        i = own["i"].to_numpy(dtype=int)
        raw.loc[i, ADSB_RAW] = a.loc[own["j"].to_numpy(dtype=int), ADSB_RAW].to_numpy()
        raw.loc[i, "cs_match"] = own["conf"].to_numpy()
        conf = np.zeros(n)
        conf[mm["i"].to_numpy(dtype=int)] = mm["conf"].to_numpy()
    q = pd.DataFrame({"i": np.arange(n), "apt": apt, "t": t,
                      "cs1": pd.Series(cs1).fillna("").astype(str).str.strip().str.upper().to_numpy(),
                      "cs2": pd.Series(cs2).fillna("").astype(str).str.strip().str.upper().to_numpy()})
    if FS_OPTS.get("match") != "v7":
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
    if FS_OPTS.get("match") == "v7":
        out["adsb_cs_match"] = (raw["cs_match"] >= 2).astype(float).where(raw["cs_match"].notna())
        out["adsb_match_conf"] = conf
        out["adsb_tonly"] = (conf == 1.0).astype(float)
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
    if FS_OPTS.get("airline") == "icao":
        flight_all = airline_key(df["CALLSIGN_flt"], df["FLIGHT_mvt"])
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
    if FS_OPTS.get("airline") == "icao":
        # Next occupant: the first in-block at our stand after our (same-airline) inbound's in-block and
        # before our takeoff belongs to another aircraft, so our off-block precedes it (a lower bound on
        # the taxi), unless the stand is shared; how many arrived in between says which.
        inb_d = np.where(qt - inb <= 86400.0, inb, np.nan)  # an inbound older than a day is not ours
        nxo = stand_neighbours(stand_all[q], inb_d + 1.0, stand_all[ra], blk[ra], {}, lags=(1,))["t_next"]
        nxo = np.where(nxo < qt, nxo, np.nan)
        F["stand_next_occ_gap"] = qt - nxo
        F["stand_n_after_inb"] = count_between(stand_all[q], inb_d + 1.0, qt, stand_all[ra], blk[ra])
        F["inb_found"] = (~np.isnan(inb)).astype("float32")

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
    ad = adsb_features(qa, qt, d["CALLSIGN_flt"].to_numpy(), d["FLIGHT_mvt"].to_numpy(), d["AIRCRAFT_TYPE_mvt"].to_numpy())
    for c in ad.columns:
        F[c] = ad[c].to_numpy()

    # ADS-B v3 events (fs4).
    a3 = adsb3_features(qa, qt, d["CALLSIGN_flt"].to_numpy(), d["FLIGHT_mvt"].to_numpy(), d["STAND_mvt"].to_numpy(),
                        d["AIRCRAFT_TYPE_mvt"].to_numpy())
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
    # Neighbours' takeoff - EOBT_1 (how late the traffic around us leaves against its plan), self
    # excluded; "x" subtracts each flight's stand x runway median (label-free: both times are
    # published).  At LTFM, with no ADS-B, the same-runway +-15 min mean correlates ~0.2 with the
    # v16 stack's residual.
    e1_dep = np.where(np.abs(qt - eobt1) < 7200, qt - eobt1, np.nan)
    sr = (pd.Series(qr) + "|" + d["STAND_mvt"].astype(str)).to_numpy()
    typ = pd.Series(e1_dep).groupby(sr).transform("median").to_numpy()
    typ = np.where(np.isnan(typ), pd.Series(e1_dep).groupby(qr).transform("median").to_numpy(), typ)
    x1_dep = e1_dep - typ
    wins = [("r600", -600, 600, True), ("r900", -900, 900, True), ("rb900", -900, 0, False)]
    nb_delay_features(F, qr, qt, e1_dep, qr, qt, e1_dep, "nbe1", wins)
    nb_delay_features(F, qr, qt, x1_dep, qr, qt, x1_dep, "nbx1", wins)
    nb_delay_features(F, qa, qt, x1_dep, qa, qt, x1_dep, "nbx1a", [("a900", -900, 900, True)])

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
    if FS_OPTS.get("airline") == "icao":
        F["airline3"] = (pd.Series(qa) + "|" + pd.Series(flight_all[q])).to_numpy()
    F["flight_key"] = (pd.Series(qa) + "|" + flight).to_numpy()
    F["apt_actype"] = (F["apt"] + "|" + F["actype"]).to_numpy()
    F["apt_operator"] = (F["apt"] + "|" + F["operator"]).to_numpy()
    F["rwy_hour"] = (F["rwy"] + "|" + np.floor(local_hour).astype(int).astype(str)).to_numpy()
    F["t_utc"] = qt
    if FS_OPTS.get("pru"):
        ym = pd.Series(qa) + "|" + pd.Series(ts_utc.year.astype(str)) + "|" + pd.Series(ts_utc.month.astype(str))
        for c in ("pru_pub", "pru_ref", "pru_deice_sh", "pru_noref_sh", "pru_pub_dev"):
            F[c] = ym.map({k: v.get(c, np.nan) for k, v in PRU_TAB.items()}).to_numpy(dtype=float)
    if FS_OPTS.get("demean"):
        # Airport-month-demeaned NM lags: NM clock conventions drift between years (LTFM Jan MVT - EOBT_1
        # +186 s while MVT - AOBT_3 stayed flat; LFPG Jul EOBT_1 changed meaning).  The median is over the
        # frame's own departures of that airport-month with d_sched <= 1 h, so no label is involved.
        apm = pd.Series(qa) + "|" + pd.Series(ts_utc.strftime("%Y-%m"))
        ok = F["d_sched"].to_numpy() <= 3600
        for c in ("d_eobt1", "d_aobt3", "d_lobt", "d_iobt", "aobt3_m_eobt1"):
            v = F[c].to_numpy(dtype=float)
            vv = np.where(ok & (np.abs(v) < 7200), v, np.nan)
            med = pd.Series(vv).groupby(apm.to_numpy()).transform("median").to_numpy()
            F[f"apm_{c}"] = med
            F[f"dm_{c}"] = v - med
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
A3_ID = ["t_next_air", "cs_next_air", "cs_last", "callsigns", "actype", "obs_taxi_s", "hex", "day", "k"]
DWELLS3 = None
MIN_STAND_S = 300.0
STAND_R_M = 60.0    # a dwell of >= 300 s within this of the recorded stand's coordinates is the stand (fs7b)


def load_dwells3() -> pd.DataFrame | None:
    """Per-dwell table of the v3 departures (events_v3.py adsb3_dwells_<TAG>.parquet, fs7b)."""
    roots = ([Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else []) + [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob("adsb3_dwells_*.parquet"))
            if hits:
                d = pd.concat([pd.read_parquet(h) for h in hits], ignore_index=True)
                d = d.drop_duplicates(["hex", "apt", "day", "k", "d_idx"])
                print(f"dwells3: {len(hits)} files, {len(d):,} dwells", flush=True)
                return d
    print("dwells3: not found; stand-constrained park skipped", flush=True)
    return None


def stand_park_features(i_rows, ev: pd.DataFrame, t, apt, stand, n) -> pd.DataFrame:
    """fs7b: the park chosen with the recorded stand.  Among the matched departure's dwells, the last one
    whose position lies within STAND_R_M of the stand's coordinates is the stand; its end is the
    pushback (grade A when the next point follows within 60 s).  Dwells after it and before takeoff are
    holds (remote, de-icing, runway): their total duration and count.  ev: matched event keys per i_rows."""
    cols = ["a3s_found", "a3s_obs_end", "a3s_obs_start", "a3s_dur", "a3s_dist_m", "a3s_leave_gap", "a3s_gs_leave",
            "a3s_before_gap", "a3s_is_park", "a3s_gradeA", "a3s_hold_s", "a3s_n_hold", "a3s_min_dist_m",
            "a3s_n_dwells"]
    out = pd.DataFrame(np.nan, index=np.arange(n), columns=cols)
    if DWELLS3 is None or GEOM is None or not len(i_rows):
        return out
    st = GEOM["st"][["apt", "stand", "lat", "lon"]].drop_duplicates(["apt", "stand"])
    q = pd.DataFrame({"i": i_rows, "hex": ev["hex"].to_numpy(), "apt": np.asarray(apt)[i_rows],
                      "day": ev["day"].to_numpy(), "k": ev["k"].to_numpy(),
                      "stand": pd.Series(np.asarray(stand)[i_rows]).astype(str).to_numpy(),
                      "t": np.asarray(t, dtype=float)[i_rows]})
    q = q.merge(st, on=["apt", "stand"], how="left")
    d = q.merge(DWELLS3, on=["hex", "apt", "day", "k"], how="inner")
    if not len(d):
        return out
    kx = np.cos(np.radians(d["lat"].to_numpy(dtype=float)))
    d["dist"] = np.hypot((d["d_lat"].to_numpy(dtype=float) - d["lat"].to_numpy(dtype=float)) * 111200.0,
                         (d["d_lon"].to_numpy(dtype=float) - d["lon"].to_numpy(dtype=float)) * 111200.0 * kx)
    d = d[d["d_t1"] < d["t"]]
    g = d.groupby("i")
    out.loc[g.size().index.to_numpy(), "a3s_n_dwells"] = g.size().to_numpy()
    md = g["dist"].min()
    out.loc[md.index.to_numpy(), "a3s_min_dist_m"] = md.to_numpy()
    # the last stay of >= 300 s on the stand: shorter stops within the radius are the post-push stop
    # (2025 Jan+Jul, callsign matches: R 60 m + 300 s -> grade A' RMSE 134 s, 84% within 60 s, against
    # the extractor's grade A 314 s / 80%; the last dwell within 100 m of any length was the push stop)
    near = d[(d["dist"] <= STAND_R_M) & ((d["d_t1"] - d["d_t0"]) >= MIN_STAND_S)]
    near = near.sort_values(["i", "d_t1"]).drop_duplicates("i", keep="last")
    if not len(near):
        return out
    i = near["i"].to_numpy()
    tt = near["t"].to_numpy()
    out.loc[i, "a3s_found"] = 1.0
    out.loc[i, "a3s_obs_end"] = tt - near["d_t1"].to_numpy()
    out.loc[i, "a3s_obs_start"] = tt - near["d_t0"].to_numpy()
    out.loc[i, "a3s_dur"] = (near["d_t1"] - near["d_t0"]).to_numpy()
    out.loc[i, "a3s_dist_m"] = near["dist"].to_numpy()
    out.loc[i, "a3s_leave_gap"] = near["d_leave_gap_s"].to_numpy(dtype=float)
    out.loc[i, "a3s_gs_leave"] = near["d_gs_leave"].to_numpy(dtype=float)
    out.loc[i, "a3s_before_gap"] = near["d_before_big_gap"].to_numpy(dtype=float)
    out.loc[i, "a3s_is_park"] = near["d_is_park"].to_numpy(dtype=float)
    out.loc[i, "a3s_gradeA"] = ((near["d_leave_gap_s"].to_numpy(dtype=float) <= 60)
                                & (near["d_before_big_gap"].to_numpy() == 0)).astype(float)
    # holds after leaving the stand: later dwells of >= 60 s more than 300 m from line-up
    later = d.merge(near[["i", "d_t1"]].rename(columns={"d_t1": "s_t1"}), on="i")
    later = later[(later["d_t0"] > later["s_t1"]) & (later["d_to_lineup_m"] > 300) & (later["dist"] > 150)]
    h = later.assign(dur=later["d_t1"] - later["d_t0"]).groupby("i")["dur"].agg(["sum", "size"])
    out.loc[i, "a3s_hold_s"] = 0.0
    out.loc[i, "a3s_n_hold"] = 0.0
    out.loc[h.index.to_numpy(), "a3s_hold_s"] = h["sum"].to_numpy()
    out.loc[h.index.to_numpy(), "a3s_n_hold"] = h["size"].to_numpy()
    return out


def load_adsb3() -> pd.DataFrame | None:
    # events are a mounted kernel output even when the other inputs were fetched into DATA_ROOT
    roots = ([Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else []) + [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob("adsb3_events_*.parquet")) if root.exists() else []
            if hits:
                cols = ["apt", "callsign", "t_takeoff", "grade"] + A3_TIMES + A3_VALUES + A3_POS + A3_ID
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


EVP_W = (900, 1800)


def event_pool_features(apt, t, own_j) -> pd.DataFrame:
    """Identity-free surface state from every v3 event at the airport (fs7): mean observed taxi of the
    departures ADS-B saw taking off within +-w of our takeoff (all grades, and grade A only), their count,
    and the share seen at the stand.  The row's own matched event (own_j, -1 = none) is left out."""
    n = len(t)
    out = {}
    a = ADSB3
    ea = a["apt"].astype(str).to_numpy()
    et = a["t_takeoff"].to_numpy(dtype=float)
    ob = np.clip(a["obs_taxi_s"].to_numpy(dtype=float), 0, NB_MAX)
    gA = (a["grade"] == "A").to_numpy().astype(float)
    obA = np.where(gA > 0, ob, np.nan)
    has = own_j >= 0
    jj = np.where(has, own_j, 0)
    own_ob = np.where(has, ob[jj], np.nan)
    own_obA = np.where(has, obA[jj], np.nan)
    own_t = np.where(has, et[jj], np.nan)
    qa = np.asarray(apt).astype(str)
    for w in EVP_W:
        for name, v, ov in (("obs", ob, own_ob), ("obsA", obA, own_obA)):
            s, c = mean_between_ex(qa, t - w, t + w, ea, et, v, ov)
            inside = has & ~np.isnan(ov) & (np.abs(own_t - t) < w)
            s = np.where(inside, s - np.nan_to_num(ov), s)
            c = np.where(inside, c - 1, c)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"evp_{name}_{w}"] = np.where(c >= 1, s / np.maximum(c, 1), np.nan)
            out[f"evp_n{name}_{w}"] = c
    return pd.DataFrame(out)


def adsb3_features(apt, t, cs1, cs2, stand, actype=None) -> pd.DataFrame:
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
    if FS_OPTS.get("match") == "v7":
        a = ADSB3.reset_index(drop=True)
        k1, k2 = _event_keys(a, ["callsign", "cs_next_air", "cs_last"], ["callsigns"])
        mm = match_events(apt, t, [canon(cs1), canon(cs2), icao_alias(cs2)],
                          actype if actype is not None else np.full(n, ""), a["apt"].to_numpy(),
                          a["t_takeoff"].to_numpy(dtype=float), a["t_next_air"].to_numpy(dtype=float), k1, k2,
                          a["actype"].to_numpy(), tag="adsb v3")
        conf = np.zeros(n)
        conf[mm["i"].to_numpy(dtype=int)] = mm["conf"].to_numpy()
        own_j = np.full(n, -1)
        own = mm[mm["conf"] >= 1.5]
        own_j[own["i"].to_numpy(dtype=int)] = own["j"].to_numpy(dtype=int)
        ev = event_pool_features(apt, np.asarray(t, dtype=float), own_j)
        m = a.loc[own["j"].to_numpy(dtype=int), cols].reset_index(drop=True)
        m["i"] = own["i"].to_numpy(dtype=int)
        m["t"] = np.asarray(t, dtype=float)[m["i"].to_numpy()]
        m["csm"] = (own["conf"].to_numpy() >= 2).astype(float)
        out = _a3_fill(out, m, apt, stand)
        out["a3_match_conf"] = conf
        out["a3_tonly"] = (conf == 1.0).astype(float)
        out["a3_dt_next_air"] = np.nan
        out.loc[own["i"].to_numpy(dtype=int), "a3_dt_next_air"] = own["dna"].to_numpy()
        for c in ev.columns:
            out[c] = ev[c].to_numpy()
        if FS_OPTS.get("park") == "stand":
            keys = a.loc[own["j"].to_numpy(dtype=int), ["hex", "day", "k"]].reset_index(drop=True)
            sp = stand_park_features(own["i"].to_numpy(dtype=int), keys, t, apt, stand, n)
            for c in sp.columns:
                out[c] = sp[c].to_numpy()
        return out
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
    return _a3_fill(out, m, apt, stand)


def _a3_fill(out: pd.DataFrame, m: pd.DataFrame, apt, stand) -> pd.DataFrame:
    """Write matched v3 events (m: i, t, csm + event columns) into the a3_* frame."""
    if not len(m):
        return out
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
# so the loss is squared error in seconds; y = r * d.  SPEC["long_target"] = "sec" regresses
# y itself (capped at 1.5 d) with unit weights instead.
#
# SPEC (filled by build_kernel.py) picks the library and parameters.  Folds A and B give
# out-of-fold predictions for all of Jan + Jul 2025; "full" refits on every 2025 row with
# 1.1 x the folds' mean best iteration and predicts the ranking file.
#
# Outputs: oof.csv (id, fold, prediction), submission.parquet, results.json.
#
# Validation options (WS-E, 2026-10-01; every one is off unless set, so old SPECs run unchanged):
#   folds: [..., "M1".."M12", "MJ", "MY"]  whole-month outer folds: train on 2025 minus that
#        month, score all of it (MJ = M1, MY = M7).  Their encodings are not in the store; they
#        are computed here with the store's own recipe (all_encodings, fit = rest of 2025).
#   enc_check: true   recompute enc_A in-kernel and print its max difference from the store's.
#   fold_rounds: [s, l] | {fold: [s, l] or {group: [s, l]}}   fixed rounds, no early stopping
#        on the scored rows (review B defect 6).
#   es_inner: {"days": [8..14], "refit": false, "mult": 1.1}   early-stop on an inner time
#        block of the TRAINING side (those days of every month); with refit, refit on all the
#        training side at mult x the inner best.
#   long_w: "cap" (min(d, w_cap_h)^2, default) | "d2" (exact seconds SSE) | "none";
#   w_cap_h: hours (12);  r_clip: ratio clip of labels and predictions (1.5).
#   pred_cap_d: T   predictions capped at d_sched where d_sched > T (review B defect 4).
#   only: "long" | "short"   fit one half; the other half's predictions are NaN (fill from a
#        base run with research/eval_full.py --fill).  Use with refit: false.
#   eval_mask: {"mode": "cov26"}   coverage stress: in Jan/Jul rows, mask ADS-B on a random share
#        max(0, 1 - cov2026/cov2025) of the covered rows of each airport x month, re-predict, and
#        write oof_cov26.csv (all fold rows, masked ones replaced).
#   max_parallel: k   run fold (and refit) jobs in waves of k processes, GPU = slot % gpus
#        (one job per GPU at a time: two CatBoost GPU jobs on one card exhaust its memory).

import importlib
import subprocess

SPEC = {'lib': 'cat', 'gpus': 1, 'folds': [], 'rounds': [9600, 2900], 'refit_seeds': 2, 'params': {'task_type': 'GPU', 'depth': 10, 'learning_rate': 0.06, 'l2_leaf_reg': 3, 'border_count': 254, 'one_hot_max_size': 4, 'max_ctr_complexity': 1, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'random_seed': 6101, 'gpu_ram_part': 0.9}, 'params_long': {'depth': 8}, 'drop_features': ['pru_pub', 'pru_ref', 'pru_deice_sh', 'pru_noref_sh', 'pru_pub_dev', 'apm_d_eobt1', 'dm_d_eobt1', 'apm_d_aobt3', 'dm_d_aobt3', 'apm_d_lobt', 'dm_d_lobt', 'apm_d_iobt', 'dm_d_iobt', 'apm_aobt3_m_eobt1', 'dm_aobt3_m_eobt1'], 'fs': 'prc-fs7', 'fs_urls': {'enc_A.parquet': '<redacted signed URL>', 'enc_B.parquet': '<redacted signed URL>', 'enc_F.parquet': '<redacted signed URL>', 'enc_full.parquet': '<redacted signed URL>', 'enc_full_rk.parquet': '<redacted signed URL>', 'fs_meta.json': '<redacted signed URL>', 'rk_base.parquet': '<redacted signed URL>', 'tr_base.parquet': '<redacted signed URL>'}, 'parallel': False, 'threads': 4}
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


def mem_gb() -> str:
    """Host memory (GB) available / total, for sizing parallel jobs; '' off Linux."""
    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        kb = lambda k: int(info[k].split()[0]) / 1e6  # noqa: E731
        return f"mem {kb('MemAvailable'):.1f}/{kb('MemTotal'):.1f} GB"
    except Exception:  # noqa: BLE001
        return ""


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


class MLP:
    """Feed-forward net: rank-gauss numerics + missing flags, embedded categoricals.

    The learning rate follows one cosine over params["epochs"], so a refit that stops at the
    fold's best epoch replays the same schedule.  `rounds` counts tenths of an epoch, which keeps
    the trainer's refit rule (1.1 x mean best, at least 50) meaningful."""

    def __init__(self, params, threads):
        self.torch = ensure("torch")
        self.p = {"hidden": [512, 256, 128], "dropout": 0.1, "lr": 2e-3, "wd": 1e-5, "batch": 2048,
                  "epochs": 14, "patience": 3, "seed": 0, **params}
        self.threads = threads

    def _prep_fit(self, X):
        from scipy.special import erfinv
        cats = [c for c in self.cats if c in X]
        nums = [c for c in X.columns if c not in cats]
        A = X[nums].to_numpy(np.float32)
        rng = np.random.default_rng(0)
        s = A[rng.choice(len(A), min(len(A), 200_000), replace=False)]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q = np.nanpercentile(s, np.linspace(0, 100, 101), axis=0).T  # (cols, 101)
        q = np.nan_to_num(q)
        u = np.linspace(0.0, 1.0, 101)
        grid = np.sqrt(2) * erfinv(2 * np.clip(u, 0.002, 0.998) - 1)
        miss = np.where(np.isnan(A).any(axis=0))[0]
        card = [len(X[c].cat.categories) + 1 for c in cats]
        return dict(cats=cats, nums=nums, q=q, grid=grid, miss=miss, card=card)

    def _mat(self, prep, X):
        A = X[prep["nums"]].to_numpy(np.float32)
        M = np.isnan(A[:, prep["miss"]]).astype(np.float32)
        Z = np.empty((len(A), A.shape[1] + M.shape[1]), np.float32)
        for j in range(A.shape[1]):
            xp, col = prep["q"][j], A[:, j]
            xu, first = np.unique(xp, return_index=True)
            if len(xu) < 3:  # (near-)constant or binary: centred raw value
                Z[:, j] = np.nan_to_num(col - xp[50])
                continue
            # ties in the quantile table: map to the middle of each tied run
            last = np.r_[first[1:] - 1, len(xp) - 1]
            fp = (prep["grid"][first] + prep["grid"][last]) / 2
            Z[:, j] = np.nan_to_num(np.interp(col, xu, fp), nan=0.0)
        Z[:, A.shape[1]:] = M
        C = (np.stack([X[c].cat.codes.to_numpy().astype(np.int64) + 1 for c in prep["cats"]], 1)
             if prep["cats"] else np.zeros((len(X), 0), np.int64))
        return Z, C

    def _net(self, nz, card):
        nn = self.torch.nn
        torch = self.torch
        hidden, dropout = self.p["hidden"], self.p["dropout"]

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.ModuleList([nn.Embedding(k, min(24, int(1.6 * k ** 0.56) + 1)) for k in card])
                d = nz + sum(e.embedding_dim for e in s.emb)
                layers = []
                for h in hidden:
                    layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
                    d = h
                layers.append(nn.Linear(d, 1))
                s.mlp = nn.Sequential(*layers)

            def forward(s, z, c):
                return s.mlp(torch.cat([z] + [e(c[:, i]) for i, e in enumerate(s.emb)], 1)).squeeze(1)

        return Net()

    def fit(self, X, y, w, Xv, yv, wv, rounds=None):
        import copy
        import math
        torch = self.torch
        torch.set_num_threads(self.threads)
        torch.manual_seed(int(self.p["seed"]))
        rng = np.random.default_rng(int(self.p["seed"]))
        prep = self._prep_fit(X)
        y = np.asarray(y, np.float64)
        prep["ymu"], prep["ysd"] = float(np.mean(y)), float(np.std(y) or 1.0)
        Z, C = self._mat(prep, X)
        zt, ct = torch.from_numpy(Z), torch.from_numpy(C)
        yt = torch.from_numpy(((y - prep["ymu"]) / prep["ysd"]).astype(np.float32))
        wt = torch.from_numpy((np.ones(len(y)) if w is None else _norm_w(np.asarray(w, float))).astype(np.float32))
        val = rounds is None and Xv is not None and len(Xv) > 0
        if val:
            Zv, Cv = self._mat(prep, Xv)
            yv_ = (np.asarray(yv, np.float64) - prep["ymu"]) / prep["ysd"]
            wv_ = np.ones(len(yv_)) if wv is None else _norm_w(np.asarray(wv, float))
        E, B, n = int(self.p["epochs"]), int(self.p["batch"]), len(y)
        spe = math.ceil(n / B)
        run_steps = E * spe if rounds is None and val else int(round((rounds or 30) / 10 * spe))
        net = self._net(Z.shape[1], prep["card"])
        opt = torch.optim.AdamW(net.parameters(), lr=self.p["lr"], weight_decay=self.p["wd"])
        total = E * spe
        best, best_state, best_ep, bad, t, ep = np.inf, None, 0, 0, 0, 0
        while t < run_steps:
            net.train()
            perm = torch.from_numpy(rng.permutation(n))
            for i in range(0, n, B):
                if t >= run_steps:
                    break
                idx = perm[i:i + B]
                if len(idx) < 2:
                    continue
                for g in opt.param_groups:
                    g["lr"] = self.p["lr"] * 0.5 * (1 + math.cos(math.pi * min(t, total) / total))
                loss = (wt[idx] * (net(zt[idx], ct[idx]) - yt[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                t += 1
            ep += 1
            if val:
                pv = self._predict_std(net, Zv, Cv)
                v = float(np.mean(wv_ * (pv - yv_) ** 2))
                print(f"  mlp epoch {ep}: val {v:.5f}", flush=True)
                if v < best - 1e-6:
                    best, best_state, best_ep, bad = v, copy.deepcopy(net.state_dict()), ep, 0
                else:
                    bad += 1
                    if bad >= int(self.p["patience"]):
                        break
        if val:
            net.load_state_dict(best_state)
            return (net, prep), best_ep * 10
        return (net, prep), int(round(run_steps / spe * 10))

    def _predict_std(self, net, Z, C):
        torch = self.torch
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Z), 65536):
                out.append(net(torch.from_numpy(Z[i:i + 65536]), torch.from_numpy(C[i:i + 65536])).numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)

    def predict(self, m, X, best=None):
        net, prep = m
        Z, C = self._mat(prep, X)
        return self._predict_std(net, Z, C) * prep["ysd"] + prep["ymu"]

    def importance(self, m, names):
        return pd.Series(0.0, index=names)


LIBS = {"lgb": LGB, "xgb": XGB, "cat": CAT, "mlp": MLP}


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


ADSB_PREFIXES = ("adsb_", "a3_", "nbx_", "ratio_adsb_")
ADSB_KEEP = ("adsb_found", "adsb_per_", "adsb_typ_sr")


def mask_adsb(X: pd.DataFrame) -> pd.DataFrame:
    """The rows as they look when adsb.lol saw nothing (LIRF in Feb 2025, and in July 2026):
    adsb_found 0, neighbour ADS-B counts 0, every other ADS-B value missing; the period-typical
    stand x runway columns stay."""
    X = X.copy()
    for c in X.columns:
        if c.startswith(ADSB_PREFIXES) and not c.startswith(ADSB_KEEP):
            X[c] = np.nan
        elif c.startswith("nbn_"):
            X[c] = 0.0
    if "adsb_found" in X:
        X["adsb_found"] = 0.0
    return X


def _dropout_augment(X, y, ds, r, long, fit, valid):
    """SPEC["adsb_dropout"] = {"p": share, "apt": [...] or None}: append ADS-B-masked copies of a
    random share of the fit rows (optionally one set of airports), so every airport x season
    also has examples without ADS-B."""
    spec = SPEC["adsb_dropout"]
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    cand = fit.copy()
    if spec.get("apt"):
        cand &= X["apt"].astype(str).isin(spec["apt"]).to_numpy()
    pick = np.flatnonzero(cand & (rng.random(len(cand)) < float(spec["p"])))
    Xm = mask_adsb(X.iloc[pick])
    Xm.index = np.arange(len(X), len(X) + len(pick)) + (int(X.index.max()) + 1 - len(X))
    cat = lambda a: np.concatenate([a, a[pick]])  # noqa: E731
    return (pd.concat([X, Xm]), cat(y), cat(ds), cat(r), cat(long),
            np.concatenate([fit, np.ones(len(pick), bool)]),
            None if valid is None else np.concatenate([valid, np.zeros(len(pick), bool)]))


def fit_pair(libs, X, y, ds, r, long, fit, valid, rounds=None):
    """One short/long pair, or one per group of SPEC["split_by"] (per-airport models); in the
    split case `rounds` is {group: rounds} and the result is {"groups": {group: pair}}."""
    if SPEC.get("adsb_dropout"):
        X, y, ds, r, long, fit, valid = _dropout_augment(X, y, ds, r, long, fit, valid)
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


ROWW = {}


def cell_weights(tr: pd.DataFrame, rk: pd.DataFrame, lo: float = 0.25, hi: float = 4.0) -> np.ndarray:
    """Covariate-shift weights for training rows: p_2026(cell) / p_2025(cell), where a cell is
    airport x ADS-B v3 grade (-1 = no event) x delay regime, clipped to [lo, hi], mean 1."""
    def key(f):
        g = np.where(f["a3_found"].to_numpy(dtype=float) > 0, f["a3_grade"].to_numpy(dtype=float), -1.0)
        g = np.nan_to_num(g, nan=-2.0).astype(int)
        lg = (f["d_sched"].to_numpy(dtype=float) > LONG_DS).astype(int)
        return pd.Series(f["apt"].astype(str).to_numpy()) + "|" + pd.Series(g).astype(str) + "|" + pd.Series(lg).astype(str)
    k_tr, k_rk = key(tr), key(rk)
    ratio = (k_rk.value_counts(normalize=True) / k_tr.value_counts(normalize=True)).clip(lo, hi)
    w = k_tr.map(ratio).fillna(lo).to_numpy(dtype=float)
    return w / w.mean()


def rclip() -> float:
    return float(SPEC.get("r_clip", R_CLIP))


def long_weight(ds: np.ndarray) -> np.ndarray:
    """Weight of a long row in the ratio regression.  (d r_hat - y)^2 = d^2 (r_hat - y/d)^2, so
    exact seconds SSE needs d^2; the default caps d at 12 h (review A §4, review B defect 10)."""
    mode = SPEC.get("long_w", "cap")
    if mode == "d2":
        return np.asarray(ds, float) ** 2
    if mode == "none":
        return np.ones_like(ds, dtype=float)
    cap = float(SPEC["w_cap_h"]) * 3600.0 if "w_cap_h" in SPEC else W_CAP
    return np.minimum(ds, cap) ** 2


def fit_one(libs, X, y, ds, r, long, fit, valid, rounds=None):
    ls, ll = libs
    w = long_weight(ds)
    rw = ROWW["w"][X.index.to_numpy()] if "w" in ROWW else None  # covariate-shift row weights
    if rw is not None:
        w = w * rw
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
    if SPEC.get("long_target") == "sec":
        # long rows regressed on y in seconds (capped at R_CLIP x d) with unit weights: the same
        # squared-error loss as the ratio model, but leaves are constant in y rather than in y/d
        r = np.where(long, np.minimum(y, rclip() * ds), np.nan)
        w = np.ones_like(ds)
    sf, lf = fit & ~long, fit & long
    only = SPEC.get("only")
    ms = bs = ml = bl = None
    if rounds is None:
        sv, lv = valid & ~long, valid & long
        if only != "long":
            ms, bs = ls.fit(X.loc[sf], ys[sf], None if rw is None else rw[sf], X.loc[sv], ys[sv],
                            None if rw is None else rw[sv])
        if only != "short":
            ml, bl = ll.fit(X.loc[lf], r[lf], w[lf], X.loc[lv], r[lv], w[lv])
    else:
        if only != "long":
            ms, bs = ls.fit(X.loc[sf], ys[sf], None if rw is None else rw[sf], None, None, None, rounds[0])
        if only != "short":
            ml, bl = ll.fit(X.loc[lf], r[lf], w[lf], None, None, None, rounds[1])
    return (ms, bs or 0), (ml, bl or 0)


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
    w = long_weight(ds)
    r = np.clip(y / np.where(long, ds, 1.0), 0, rclip())
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
        rh = np.clip(rh, MIX_HELD_R * 0.5, rclip())
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
    if ms is None:  # only: "long"
        out = np.full(len(X), np.nan)
    else:
        out = ls.predict(ms, X, bs if use_best else None)
    out_anchor = anchor_apply(X) if SPEC.get("anchor") else None
    if out_anchor is not None:
        # the offset table comes from the fit rows of this pair (stored at fit time)
        out = out + out_anchor
    if long.any():
        out = out.copy()
        if ml is None:  # only: "short"
            out[long] = np.nan
        else:
            pr = ll.predict(ml, X.loc[long], bl if use_best else None)
            if SPEC.get("anchor", {}).get("long"):
                pr = pr + (out_anchor[long] / ds[long])
            if SPEC.get("long_target") == "sec":
                out[long] = np.clip(pr, 0, rclip() * ds[long])
            else:
                out[long] = np.clip(pr, 0, rclip()) * ds[long]
    if SPEC.get("pred_cap_d") is not None:
        big = ds > float(SPEC["pred_cap_d"])
        out = np.where(big, np.minimum(out, ds), out)
    return out


PRELOAD = {}  # variants mode: the store loaded once in the parent, shared copy-on-write by the forks


def _load_store(fs: Path, lib_name: str, want_keys: bool):
    key = (str(fs), lib_name == "cat")
    if key in PRELOAD and (PRELOAD[key][2] is not None or not want_keys):
        return PRELOAD[key]
    tr = pd.read_parquet(fs / "tr_base.parquet")
    rk = pd.read_parquet(fs / "rk_base.parquet")
    tr["y"] = tr["y"].astype(np.float64)
    keys = _enc_keys(tr) if want_keys else None  # before categorize() collapses rare levels
    if lib_name != "cat":
        categorize(tr, [rk], ref_mask=np.ones(len(tr), bool))
    return tr, rk, keys


def load_matrices(fs: Path, lib_name: str):
    want = any(month_fold(n) for n in SPEC.get("folds", [])) or bool(SPEC.get("enc_check"))
    tr, rk, keys = _load_store(fs, lib_name, want)
    if want:
        STATE["enc_keys"] = keys
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


def _fold_rounds(name):
    fr = SPEC.get("fold_rounds")
    if fr is None:
        return None
    if isinstance(fr, dict) and name in fr:
        fr = fr[name]
    elif isinstance(fr, dict) and any(k in STATE["folds"] or month_fold(k) for k in fr):
        return None  # rounds given for other folds only: this one early-stops as before
    if isinstance(fr, dict):  # per-group rounds {group: [s, l]}
        return {g: tuple(int(x) for x in v) for g, v in fr.items()}
    return tuple(int(x) for x in fr)


def _best_of(pair):
    if isinstance(pair, dict):
        return {g: tuple(int(m[1]) for m in p) for g, p in pair["groups"].items()}
    return tuple(int(m[1]) for m in pair)


def _cov26_mask(tr, sel, seed):
    """Rows of `sel` whose ADS-B is masked so that each airport x month (Jan, Jul) has the 2026
    ranking file's ADS-B v3 coverage: a random share max(0, 1 - cov26 / cov25) of the covered
    rows.  Months other than Jan/Jul have no 2026 counterpart and stay unmasked."""
    rk = STATE["rk"]
    f_rk = rk["a3_found"].fillna(0).to_numpy() > 0
    cov26 = pd.Series(f_rk).groupby([rk["apt"].astype(str).to_numpy(), rk["month"].to_numpy()]).mean()
    found = tr["a3_found"].fillna(0).to_numpy() > 0
    apt, month = tr["apt"].astype(str).to_numpy(), tr["month"].to_numpy()
    rng = np.random.default_rng(seed)
    u = rng.random(len(tr))
    out = np.zeros(len(tr), bool)
    for (a, mo), c26 in cov26.items():
        cell = sel & (apt == a) & (month == mo)
        if not cell.any():
            continue
        c25 = found[cell].mean()
        p = max(0.0, 1.0 - c26 / c25) if c25 > 0 else 0.0
        out |= cell & found & (u < p)
        print(f"  cov26 {a} m{mo}: 2025 {c25:.3f} 2026 {c26:.3f} -> mask {p:.3f} of covered rows", flush=True)
    return out


def run_fold(name, lib, X, tr, fold_val, threads):
    y = tr["y"].to_numpy()
    ds = tr["d_sched"].to_numpy()
    long = ds > LONG_DS
    r = np.where(long, np.clip(y / np.where(long, ds, 1.0), 0, rclip()), np.nan)
    art = tr["artifact"].to_numpy().astype(bool)
    t0 = time.time()
    fit, val = ~fold_val & ~art, fold_val & ~art
    fixed = _fold_rounds(name)
    inner = SPEC.get("es_inner")
    if fixed is not None:
        # fixed rounds: nothing is early-stopped, least of all on the scored rows
        pair = fit_pair(lib, X, y, ds, r, long, fit, None, fixed)
    elif inner:
        days = inner.get("days", list(range(8, 15)))
        iv = fit & np.isin(tr["mday"].to_numpy(), days)
        print(f"[fold {name}] inner early stopping on {int(iv.sum()):,} training-side rows (days {days})", flush=True)
        pair = fit_pair(lib, X, y, ds, r, long, fit & ~iv, iv)
        if inner.get("refit"):
            mult = float(inner.get("mult", 1.1))
            b = _best_of(pair)
            rr = ({g: tuple(max(int(x * mult), 50) for x in v) for g, v in b.items()} if isinstance(b, dict)
                  else tuple(max(int(x * mult), 50) for x in b))
            print(f"[fold {name}] inner best {b} -> refit on the whole training side at {rr}", flush=True)
            pair = fit_pair(lib, X, y, ds, r, long, fit, None, rr)
    else:
        pair = fit_pair(lib, X, y, ds, r, long, fit, val)
    pv = pred_pair(lib, pair, X.loc[fold_val].reset_index(drop=True), ds[fold_val], long[fold_val])
    keep = ~art[fold_val] & np.isfinite(pv)
    score = rmse(y[fold_val][keep], pv[keep])
    em = SPEC.get("eval_mask")
    if em and em.get("mode") == "cov26":
        msk = _cov26_mask(tr, fold_val, int(em.get("seed", 0)))
        pc = pv.copy()
        inval = msk[fold_val]  # masked rows, in fold-row order
        if msk.any():
            pm = pred_pair(lib, pair, mask_adsb(X.loc[msk]).reset_index(drop=True), ds[msk], long[msk])
            pc[inval] = pm
            ok = np.isfinite(pm)
            print(f"[fold {name}] cov26 stress: {int(msk.sum()):,} rows masked, their RMSE masked "
                  f"{rmse(y[msk][ok], pm[ok]):.3f} vs unmasked {rmse(y[msk][ok], pv[inval][ok]):.3f}", flush=True)
        pd.DataFrame({"id": tr.loc[fold_val, ID].to_numpy(), "fold": name, "prediction": pc}).to_csv(
            OUTPUT_DIR / f"oof_{name}_cov26.csv", index=False)
    elif em:  # the same fold rows re-predicted with ADS-B masked (e.g. LIRF in July, as in 2026)
        sel = fold_val & ~art
        if em.get("apt"):
            sel &= tr["apt"].astype(str).isin(em["apt"]).to_numpy()
        if em.get("months"):
            sel &= np.isin(tr["month"].to_numpy(), em["months"])
        pm = pred_pair(lib, pair, mask_adsb(X.loc[sel]).reset_index(drop=True), ds[sel], long[sel])
        pd.DataFrame({"id": tr.loc[sel, ID].to_numpy(), "fold": name, "prediction": pm}).to_csv(
            OUTPUT_DIR / f"oof_{name}_masked.csv", index=False)
        print(f"[fold {name}] masked eval on {int(sel.sum())} rows: {rmse(y[sel], pm):.3f} "
              f"(unmasked {rmse(y[sel], pred_pair(lib, pair, X.loc[sel].reset_index(drop=True), ds[sel], long[sel])):.3f})",
              flush=True)
    first = next(iter(pair["groups"].values())) if isinstance(pair, dict) else pair
    print(f"[fold {name}] test-like {score:.3f}  short {first[0][1]}  long {first[1][1]}  "
          f"{(time.time() - t0) / 60:.1f} min  {mem_gb()}", flush=True)
    return pv, _best_of(pair), score, pair


STATE = {}


def _targets(tr):
    y = tr["y"].to_numpy()
    ds = tr["d_sched"].to_numpy()
    long = ds > LONG_DS
    r = np.where(long, np.clip(y / np.where(long, ds, 1.0), 0, rclip()), np.nan)
    art = tr["artifact"].to_numpy().astype(bool)
    return y, ds, long, r, art


def fold_job(name: str, threads: int) -> None:
    g = STATE
    gpus = int(SPEC.get("gpus", 0))
    if gpus and "gpu_id" in SPEC:  # a variant process pinned to one GPU
        gpu = int(SPEC["gpu_id"]) % gpus
    elif gpus and "slot" in g:  # waves (max_parallel): one job per GPU at a time
        gpu = g["slot"].get(name, 0) % gpus
    else:
        gpu = (sorted(g["folds"]).index(name) % gpus) if gpus else None
    lib = make_libs(g["lib_name"], g["params"], g["params_long"], threads, g["cats"], gpu=gpu)
    X = assemble(g["tr"], fold_encodings(name), g["feats"])
    val = g["folds"][name]
    pv, best, score, pair = run_fold(name, lib, X, g["tr"], val, threads)
    pd.DataFrame({"id": g["tr"].loc[val, ID].to_numpy(), "fold": name, "prediction": pv}).to_csv(
        OUTPUT_DIR / f"oof_{name}.csv", index=False)
    json.dump({"best": best if isinstance(best, dict) else list(best), "score": score},
              open(OUTPUT_DIR / f"best_{name}.json", "w"))
    first = next(iter(pair["groups"].values())) if isinstance(pair, dict) else pair
    lib_i, m_i = (lib[0], first[0][0]) if first[0][0] is not None else (lib[1], first[1][0])
    lib_i.importance(m_i, list(X.columns)).sort_values(ascending=False).to_csv(OUTPUT_DIR / f"imp_{name}.csv")


def refit_job(k: int, threads: int) -> None:
    g = STATE
    gpus = int(SPEC.get("gpus", 0))
    slot = int(SPEC["gpu_id"]) if "gpu_id" in SPEC else (g["slot"].get(k, k) if "slot" in g else k)
    lib = make_libs(g["lib_name"], g["params"], g["params_long"], threads, g["cats"], seed_offset=1000 * k,
                    gpu=(slot % gpus) if gpus else None)
    y, ds, long, r, art = _targets(g["tr"])
    pair = fit_pair(lib, g["X_all"], y, ds, r, long, ~art, None, g["rounds"])
    rk_ds = g["rk"]["d_sched"].to_numpy()
    pred = pred_pair(lib, pair, g["X_rk"], rk_ds, rk_ds > LONG_DS, use_best=False)
    np.save(OUTPUT_DIR / f"refit_{k}.npy", pred)
    print(f"refit {k}: rounds {g['rounds']}  used_h={used_h():.3f}", flush=True)


MONTH_ALIASES = {"MJ": 1, "MY": 7}


def month_fold(name: str):
    """Month number of a whole-month outer fold name (M1..M12, MJ, MY), else None."""
    if name in MONTH_ALIASES:
        return MONTH_ALIASES[name]
    if len(name) > 1 and name[0] == "M" and name[1:].isdigit() and 1 <= int(name[1:]) <= 12:
        return int(name[1:])
    return None


def fold_encodings(name: str) -> pd.DataFrame:
    """The store's enc_<fold>.parquet, or, for a fold the store lacks (month-out folds), the
    same encodings computed here with fs_main's recipe: fit = every non-artifact row outside the
    fold, training rows leave their own month out."""
    g = STATE
    p = g["fs"] / f"enc_{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    cache = Path("/tmp/enc_cache") / f"enc_{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    t0 = time.time()
    enc = _encode(g["folds"][name])
    cache.parent.mkdir(parents=True, exist_ok=True)
    enc.to_parquet(cache, index=False)
    print(f"enc_{name}: computed in-kernel {enc.shape} in {time.time() - t0:.0f} s", flush=True)
    return enc


def _encode(val: np.ndarray) -> pd.DataFrame:
    g = STATE
    tr = g["tr"]
    month = tr["month"].to_numpy()
    y = tr["y"].to_numpy()
    ds = tr["d_sched"].to_numpy()
    long = ds > LONG_DS
    path = long & (np.abs(y - ds) <= PATH_TOL)
    r = np.where(long, np.clip(y / np.where(long, ds, 1.0), 0, R_CLIP), np.nan)  # the store's clip, not SPEC's
    art = tr["artifact"].to_numpy().astype(bool)
    keys = g["enc_keys"]
    dummy = {k: v[:1] for k, v in keys.items()}
    enc, _ = all_encodings(keys, month, y, ds, r, long, path, ~val & ~art, dummy)
    return enc.astype(np.float32)


def _enc_keys(tr: pd.DataFrame) -> dict:
    """Raw (uncollapsed) encoding keys as integer codes, taken before categorize() collapses rare
    levels, so in-kernel encodings match the store's."""
    names = sorted(set(TE_KEYS + LONG_KEYS + ["apt"]))
    return {k: pd.factorize(tr[k].astype(str).to_numpy())[0] for k in names}


def _run(jobs, parallel: bool, total_threads: int):
    k = int(SPEC.get("max_parallel", 0) or 0)
    if parallel and k and len(jobs) > k:
        # waves of k processes; slot i of a wave gets GPU i % gpus
        for i in range(0, len(jobs), k):
            wave = jobs[i:i + k]
            STATE["slot"] = {arg: j for j, (_, arg) in enumerate(wave)}
            _run(wave, True, total_threads)
        STATE.pop("slot", None)
        return
    if parallel and k and "slot" not in STATE:
        STATE["slot"] = {arg: j for j, (_, arg) in enumerate(jobs)}
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
    if spec.get("row_weight") == "cell26":
        ROWW["w"] = cell_weights(tr, rk, *spec.get("row_weight_clip", (0.25, 4.0)))
        q = np.percentile(ROWW["w"], [1, 25, 50, 75, 99])
        print(f"row weights (2026 cell mix): p1/25/50/75/99 {np.round(q, 2).tolist()}", flush=True)
    month, mday = tr["month"].to_numpy(), tr["mday"].to_numpy()
    jj = np.isin(month, VAL_MONTHS)
    folds = {"A": jj & (mday >= 16), "B": jj & (mday <= 15), "F": np.isin(month, (11, 12))}
    for n in spec.get("folds", []) or []:
        if month_fold(n):
            folds[n] = month == month_fold(n)
    STATE.update(dict(lib_name=lib_name, params=spec["params"], params_long=spec.get("params_long"), cats=[c for c in CATS if c in feats], tr=tr,
                      rk=rk, feats=feats, fs=fs, folds=folds))
    print(f"loaded {len(tr):,} x {len(feats)}  used_h={used_h():.3f}  {mem_gb()}", flush=True)
    if spec.get("enc_check"):
        mine, store = _encode(folds["A"]), pd.read_parquet(fs / "enc_A.parquet")
        diff = (mine[store.columns].astype(np.float64) - store.astype(np.float64)).abs()
        print(f"enc_check A: max |diff| {np.nanmax(diff.to_numpy()):.6f}  NaN pattern equal "
              f"{bool((mine[store.columns].isna().to_numpy() == store.isna().to_numpy()).all())}", flush=True)

    names = spec.get("folds", ["A", "B"])
    info = {}
    if names:
        _run([(fold_job, n) for n in names], parallel, threads)
        oof = pd.concat([pd.read_csv(OUTPUT_DIR / f"oof_{n}.csv") for n in names])
        oof.to_csv(OUTPUT_DIR / "oof.csv", index=False)
        oof.to_parquet(OUTPUT_DIR / "oof.parquet", index=False)
        cov = [OUTPUT_DIR / f"oof_{n}_cov26.csv" for n in names]
        if all(p.exists() for p in cov):
            pd.concat([pd.read_csv(p) for p in cov]).to_csv(OUTPUT_DIR / "oof_cov26.csv", index=False)
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


def _variant_job(vname: str, threads: int) -> None:
    g = globals()
    base, slot = VARIANTS["base"], VARIANTS["slot"].get(vname, 0)
    spec = {**base, **VARIANTS["specs"][vname], "threads": threads, "parallel": False, "gpu_id": slot}
    spec.pop("variants", None)
    spec.pop("max_parallel", None)
    g["SPEC"] = spec
    g["MAX_ROUNDS"] = int(spec.get("max_rounds", MAX_ROUNDS))
    out = VARIANTS["root"] / vname
    out.mkdir(parents=True, exist_ok=True)
    g["OUTPUT_DIR"] = out
    STATE.clear()
    t0 = time.time()
    try:
        train_main()
    except Exception:
        import traceback
        (out / "error.txt").write_text(traceback.format_exc())
        raise
    print(f"[variant] {vname} done in {(time.time() - t0) / 60:.1f} min  used_h={used_h():.3f}", flush=True)


VARIANTS = {}


def variants_main():
    """SPEC["variants"] = {name: overrides}: run several specs on one host (e.g. the two T4s of a
    Kaggle GPU session), each in its own process with SPEC = base | overrides, in waves of
    SPEC["max_parallel"] (default: number of GPUs, else 1); variant i of a wave gets GPU i.
    Each writes OUTPUT_DIR/<name>/ exactly what a single trainer writes."""
    base = dict(SPEC)
    names = list(base["variants"])
    k = int(base.get("max_parallel") or base.get("gpus") or 1)
    threads = int(base.get("threads", ncpu()))
    VARIANTS.update(base=base, specs=base["variants"], root=OUTPUT_DIR)
    # load the store once here: each variant process loading its own copy ran a 33 GB GPU host out
    # of memory (one variant per wave killed, exit -9, prc-m-wse-seeds / prc-m-wse-longgpu)
    libs = {base["variants"][n].get("lib", base["lib"]) for n in names}
    want = any(month_fold(f) or f == "enc_check" for n in names
               for f in base["variants"][n].get("folds", base.get("folds", ["A", "B"])))
    fs = find_fs()
    for lib in libs:
        PRELOAD[(str(fs), lib == "cat")] = _load_store(fs, lib, want)
    print(f"[variant] store preloaded for {sorted(libs)}  {mem_gb()}", flush=True)
    print(f"[variant] {len(names)} variants, {k} at a time on {threads} cores: {names}", flush=True)
    from multiprocessing import get_context
    ctx = get_context("fork")
    codes = {}
    for i in range(0, len(names), k):
        wave = names[i:i + k]
        VARIANTS["slot"] = {n: j for j, n in enumerate(wave)}
        procs = {n: ctx.Process(target=_variant_job, args=(n, max(threads // len(wave), 1))) for n in wave}
        for p in procs.values():
            p.start()
        for n, p in procs.items():
            p.join()
            codes[n] = p.exitcode
        print(f"[variant] wave {wave} exit codes {[codes[n] for n in wave]}  used_h={used_h():.3f}", flush=True)
    (OUTPUT_DIR / "variants.json").write_text(json.dumps({"codes": codes, "used_h": used_h()}))


if __name__ == "__main__":
    if SPEC.get("variants"):
        variants_main()
    else:
        train_main()
