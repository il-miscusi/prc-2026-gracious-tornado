"""Shared loading and scoring for the LEVERS lane (post-layers on top of the stack).

Two frames, built from the organisers' files only:

* ``load_2025()`` - every 2025 departure with its label ``y`` and the 12-month-out
  OOF prediction of member c6 (``research/cache/oof_mo12_c6.parquet``, column ``p``).
* ``load_final()`` - every Jan/Feb/Jun/Jul 2026 departure of ``final_ranking.parquet``
  (no labels).

Both carry the same derived columns, so a lever's feature code runs unchanged on either:

    id, apt, ades, month, day (takeoff UTC date), d = MVT - SCHED (s), nm (NM-matched),
    ml = MVT - LOBT (s), a3 = MVT - AOBT_3 (s), lobt_sched = LOBT - SCHED (s),
    stand, runway, callsign, icao (3-letter callsign prefix), flight, hour, wd

``score()`` is the evaluation contract of B section 8.4: full, untrimmed, all-row
squared error; delta SSE divided by the 2,085,047 departures of 2025; per month; per
airport; per scheduled-delay band; airport-day block bootstrap 95% CI.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

C = Path(__file__).resolve().parents[3]          # competition directory
TRAIN_GLOB = str(C / "data/09-22-2026-08-13-30_files_list/training_2025-*.parquet")
FINAL = C / "data/final_20261008/final_ranking.parquet"
OOF = C / "research/cache/oof_mo12_c6.parquet"
PRU = C / "research/pru"
ART = Path(__file__).resolve().parent / "artifacts"
CACHE = Path(os.environ.get("LEVERS_CACHE", str(Path(__file__).resolve().parent / ".cache")))
N2025 = 2_085_047

RAW = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
       "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt",
       "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt", "LOBT_flt", "CALLSIGN_flt", "IOBT_flt",
       "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "MARKET_SEGMENT_flt",
       "FLIGHT_TYPE_flt", "AIRCRAFT_OPERATOR_flt", "ADES_flt", "WK_TBL_CAT_flt"]
TS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt",
      "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]
BANDS = [(-np.inf, 3600, "<=1h"), (3600, 14400, "1-4h"), (14400, 28800, "4-8h"), (28800, np.inf, ">8h")]


def _secs(a, b):
    return (a - b).dt.total_seconds().to_numpy(float)


def derive(raw: pd.DataFrame) -> pd.DataFrame:
    """Derived columns shared by every lever (departures or arrivals)."""
    r = raw.copy()
    for c in TS:
        if c in r:
            r[c] = pd.to_datetime(r[c], utc=True, errors="coerce")
    out = pd.DataFrame(index=r.index)
    out["id"] = r.MVT_ID_mvt.astype("int64")
    out["phase"] = r.PHASE_mvt.to_numpy()
    out["apt"] = np.where(r.PHASE_mvt == "DEP", r.ADEP_mvt, r.ADES_mvt)
    out["adep"] = r.ADEP_mvt.to_numpy()
    out["ades"] = r.ADES_mvt.to_numpy()
    mvt = r.MVT_TIME_UTC_mvt
    out["mvt"] = mvt
    out["sched"] = r.SCHED_TIME_UTC_mvt
    out["block"] = r.BLOCK_TIME_UTC_mvt
    out["month"] = mvt.dt.month.to_numpy()
    out["year"] = mvt.dt.year.to_numpy()
    out["day"] = mvt.dt.strftime("%Y-%m-%d").to_numpy()
    out["hour"] = (mvt.dt.hour + mvt.dt.minute / 60).to_numpy()
    out["wd"] = mvt.dt.weekday.to_numpy()
    out["d"] = _secs(mvt, r.SCHED_TIME_UTC_mvt)
    out["nm"] = r.FLIGHT_ID_mvt.notna().to_numpy()
    out["ml"] = _secs(mvt, r.LOBT_flt)
    out["a3"] = _secs(mvt, r.AOBT_3_flt)
    out["me1"] = _secs(mvt, r.EOBT_1_flt)
    out["mi"] = _secs(mvt, r.IOBT_flt)
    out["lobt_sched"] = _secs(r.LOBT_flt, r.SCHED_TIME_UTC_mvt)
    out["stand"] = r.STAND_mvt.astype("string").fillna("").to_numpy()
    out["runway"] = r.RUNWAY_mvt.astype("string").fillna("").to_numpy()
    cs = r.CALLSIGN_flt.astype("string").fillna("")
    out["callsign"] = cs.to_numpy()
    out["icao"] = cs.str[:3].where(cs.str.len() >= 3, "").to_numpy()
    fl = r.FLIGHT_mvt.astype("string").fillna("")
    out["flight"] = fl.to_numpy()
    out["iata"] = fl.str[:2].to_numpy()
    out["actype"] = r.AIRCRAFT_TYPE_mvt.astype("string").fillna("").to_numpy()
    out["mseg"] = r.MARKET_SEGMENT_flt.astype("string").fillna("").to_numpy()
    out["op"] = r.AIRCRAFT_OPERATOR_flt.astype("string").fillna("").to_numpy()
    out["y"] = pd.to_numeric(r.TAXITIME_SEC_mvt, errors="coerce").to_numpy(float)
    return out


def _cache(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name


def load_2025(arrivals: bool = False) -> pd.DataFrame:
    """2025 departures (+ OOF p, fold) or, with arrivals=True, 2025 arrivals."""
    tag = "arr2025.parquet" if arrivals else "dep2025.parquet"
    f = _cache(tag)
    if f.exists():
        out = pd.read_parquet(f)
        out.attrs["levers"] = True
        return out
    raw = pd.concat([pd.read_parquet(x, columns=RAW) for x in sorted(glob.glob(TRAIN_GLOB))],
                    ignore_index=True)
    dep = derive(raw[raw.PHASE_mvt == "DEP"].reset_index(drop=True))
    arr = derive(raw[raw.PHASE_mvt == "ARR"].reset_index(drop=True))
    oof = pd.read_parquet(OOF)
    oof["id"] = oof["id"].astype("int64")
    dep = dep.merge(oof.rename(columns={"prediction": "p"}), on="id", how="inner", validate="1:1")
    assert len(dep) == N2025, len(dep)
    dep.to_parquet(_cache("dep2025.parquet"))
    arr.to_parquet(_cache("arr2025.parquet"))
    return arr if arrivals else dep


def load_final(arrivals: bool = False) -> pd.DataFrame:
    tag = "arr2026.parquet" if arrivals else "dep2026.parquet"
    f = _cache(tag)
    if f.exists():
        out = pd.read_parquet(f)
        out.attrs["levers"] = True
        return out
    raw = pd.read_parquet(FINAL, columns=RAW)
    dep = derive(raw[raw.PHASE_mvt == "DEP"].reset_index(drop=True))
    arr = derive(raw[raw.PHASE_mvt == "ARR"].reset_index(drop=True))
    dep.to_parquet(_cache("dep2026.parquet"))
    arr.to_parquet(_cache("arr2026.parquet"))
    return arr if arrivals else dep


def ensure_columns(frame: pd.DataFrame, need: list[str]) -> pd.DataFrame:
    """Return a frame in the caller's row order whose ``need`` columns come from the levers'
    own derivation (joined by id).

    The caller's same-named columns are NOT trusted: the assembler's ``frame.py`` uses e.g.
    ``day`` = "%m%d" where the levers use "%Y-%m-%d", and ``d_lobt`` where the levers use
    ``ml``. A frame produced by ``load_2025``/``load_final`` (attrs["levers"] set) passes through.
    """
    if frame.attrs.get("levers") and all(c in frame.columns for c in need):
        return frame
    ids = (frame["id"] if "id" in frame else frame["MVT_ID_mvt"]).round().astype("int64").to_numpy()
    src = load_final()
    inf = np.isin(ids, src.id.to_numpy())
    if not inf.any():
        src = load_2025().drop(columns=["p", "fold", "y", "block"])     # a 2025 (pseudo-final) frame
    elif not inf.all():
        src = pd.concat([src, load_2025().drop(columns=["p", "fold", "y", "block"])], ignore_index=True)
    s = src.set_index("id")
    assert np.isin(ids, s.index.to_numpy()).all(), "ids not found in the organisers' files"
    out = s.loc[ids, [c for c in need if c != "id"]].reset_index()
    out.attrs["levers"] = True
    return out


def band_of(d: np.ndarray) -> np.ndarray:
    out = np.empty(len(d), dtype=object)
    for lo, hi, name in BANDS:
        out[(d > lo) & (d <= hi)] = name
    return out


def score(name: str, frame: pd.DataFrame, p_new: np.ndarray, p_base: np.ndarray | None = None,
          n_boot: int = 2000, seed: int = 0, verbose: bool = True) -> dict:
    """Delta of full all-row MSE (SSE/2,085,047) of ``p_new`` against ``p_base``.

    CI: airport-day block bootstrap of the summed delta (blocks resampled with replacement).
    """
    y = frame["y"].to_numpy(float)
    pb = frame["p"].to_numpy(float) if p_base is None else np.asarray(p_base, float)
    pn = np.asarray(p_new, float)
    assert np.isfinite(pn).all(), "non-finite prediction"
    de = (pn - y) ** 2 - (pb - y) ** 2
    N = N2025
    base_mse = ((pb - y) ** 2).sum() / N
    blocks = pd.Series(de).groupby((frame["apt"] + "|" + frame["day"]).to_numpy()).sum()
    rng = np.random.default_rng(seed)
    bv = blocks.to_numpy()
    idx = rng.integers(0, len(bv), size=(n_boot, len(bv)))
    boot = bv[idx].sum(1) / N
    lo, hi = np.quantile(boot, [0.025, 0.975])
    mon = pd.Series(de).groupby(frame["month"].to_numpy()).sum() / N
    apt = pd.Series(de).groupby(frame["apt"].to_numpy()).sum() / N
    bnd = pd.Series(de).groupby(band_of(frame["d"].to_numpy(float))).sum() / N
    touched = int((np.abs(pn - pb) > 1e-9).sum())
    res = {
        "name": name, "dmse": de.sum() / N, "ci": (float(lo), float(hi)),
        "base_mse": base_mse, "new_rmse": float(np.sqrt(base_mse + de.sum() / N)),
        "by_month": {int(k): float(v) for k, v in mon.items()},
        "by_apt": {k: float(v) for k, v in apt.items()},
        "by_band": {k: float(v) for k, v in bnd.items()},
        "touched": touched,
        "months_worse": int((mon > 0.5).sum()),
    }
    if verbose:
        print(fmt(res))
    return res


def fmt(r: dict) -> str:
    m = r["by_month"]
    s = (f"{r['name']}: dMSE {r['dmse']:+.1f} [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}]  "
         f"rmse {np.sqrt(r['base_mse']):.2f}->{r['new_rmse']:.2f}  touched {r['touched']}  "
         f"months worse {r['months_worse']}/12\n"
         f"   M1 {m.get(1, 0):+.1f}  M2 {m.get(2, 0):+.1f}  M6 {m.get(6, 0):+.1f}  M7 {m.get(7, 0):+.1f} | "
         + " ".join(f"M{k}:{v:+.0f}" for k, v in sorted(m.items())) + "\n"
         "   apt " + " ".join(f"{k}:{v:+.0f}" for k, v in sorted(r["by_apt"].items()) if abs(v) >= 0.5) + "\n"
         "   band " + " ".join(f"{k}:{v:+.0f}" for k, v in r["by_band"].items()))
    return s


def md_row(r: dict) -> str:
    m = r["by_month"]
    return (f"| {r['name']} | {r['dmse']:+.1f} | [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] | "
            + " | ".join(f"{m.get(k, m.get(str(k), 0)):+.1f}" for k in range(1, 13))
            + f" | {r['months_worse']}/12 | {r['touched']} |")


MD_HEAD = ("| layer | dMSE | 95% CI | " + " | ".join(f"M{k}" for k in range(1, 13))
           + " | months worse | rows touched |\n|---|---:|---|" + "---:|" * 12 + "---:|---:|")


def touched_2026(frame: pd.DataFrame, p_before: np.ndarray, p_after: np.ndarray) -> pd.DataFrame:
    """Rows changed per 2026 month and the distribution of the change (s)."""
    dlt = np.asarray(p_after, float) - np.asarray(p_before, float)
    ch = np.abs(dlt) > 1e-9
    g = pd.DataFrame({"month": frame["month"].to_numpy(), "dlt": dlt, "ch": ch})
    rows = []
    for mo, q in g.groupby("month"):
        x = q.dlt[q.ch]
        rows.append({"month": int(mo), "rows": int(q.ch.sum()),
                     "mean": float(x.mean()) if len(x) else 0.0,
                     "p10": float(x.quantile(.1)) if len(x) else 0.0,
                     "median": float(x.median()) if len(x) else 0.0,
                     "p90": float(x.quantile(.9)) if len(x) else 0.0,
                     "min": float(x.min()) if len(x) else 0.0,
                     "max": float(x.max()) if len(x) else 0.0})
    return pd.DataFrame(rows)


def resolve(p) -> Path:
    """A path as the assembler's configs write it: absolute, or relative to the competition directory
    (falls back to the current directory)."""
    p = Path(p)
    if p.is_absolute():
        return p
    return C / p if (C / p).exists() or not p.exists() else p.resolve()


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(C)) if p.is_relative_to(C) else str(p)


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
