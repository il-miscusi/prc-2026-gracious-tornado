"""Prediction frame: one row per departure to predict, in submission-template order.

    from frame import load_frame
    f = load_frame("final")   # final_ranking.parquet departures, final_submitting.parquet order
    f = load_frame("board")   # old ranking.parquet departures, old submitting.parquet order

Both frames come from organiser files at fixed paths (no directory search). The frame keeps every
raw departure column and adds the derived columns the assembly steps use:

    id           MVT_ID_mvt as int64
    apt          ADEP_mvt (departure airport)
    year, month, mday, day   takeoff (MVT_TIME_UTC_mvt) calendar fields; day is "%m%d"
    d            MVT_TIME - SCHED_TIME in seconds (supplied delay to takeoff; y <= d unless the
                 off-block preceded the schedule)
    d_sched      alias of d (name used by the long-row helpers in research/tail/)
    d_lobt       MVT_TIME - LOBT_flt (NaN when no NM flight)
    d_aobt3      MVT_TIME - AOBT_3_flt (NaN when no NM flight)
    nm           FLIGHT_ID_mvt notna: the movement was joined to an NM flight
    unmatched    1.0 when AOBT_3_flt is missing, else 0.0 (feature-store definition; equals
                 ~nm plus the few NM-joined rows without LOBT/AOBT_3)
    lirf_orphan  apt == LIRF and not nm
    sr           stand and runway both present (the PRU-comparable selection uses it)

The final file's January/July rows carry the organisers' +-12 s takeoff dither; d, d_lobt and
d_aobt3 are computed from the dithered takeoff, as every model input is.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

C = Path(__file__).resolve().parents[2]
FRAMES = {
    "final": {"ranking": "data/final_20261008/final_ranking.parquet",
              "template": "data/final_20261008/final_submitting.parquet",
              "rows": 670_790},
    "board": {"ranking": "data/09-22-2026-08-13-30_files_list/ranking.parquet",
              "template": "data/09-22-2026-08-13-30_files_list/submitting.parquet",
              "rows": 344_841},
}
ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def paths(kind: str) -> dict[str, Path]:
    spec = FRAMES[kind]
    return {"ranking": C / spec["ranking"], "template": C / spec["template"]}


def secs(a: pd.Series, b: pd.Series) -> np.ndarray:
    return (a - b).dt.total_seconds().to_numpy(dtype=float)


TRAIN_DIR = "data/09-22-2026-08-13-30_files_list"


def training_file(month: int) -> Path:
    """2025 training file of one month; the files are split by takeoff month."""
    nxt = f"{2025 + (month == 12)}-{month % 12 + 1:02d}-01"
    return C / TRAIN_DIR / f"training_2025-{month:02d}-01_{nxt}.parquet"


def load_frame(kind: str = "final") -> pd.DataFrame:
    p = paths(kind)
    tpl = pd.read_parquet(p["template"], columns=[ID])
    rk = pd.read_parquet(p["ranking"])
    dep = rk[rk["PHASE_mvt"] == "DEP"].drop(columns=[TARGET])
    assert len(tpl) == FRAMES[kind]["rows"], f"{kind} template has {len(tpl)} rows"
    assert tpl[ID].is_unique and dep[ID].is_unique, "duplicate ids"
    assert set(tpl[ID].astype("int64")) == set(dep[ID].astype("int64")), "template ids != ranking departures"
    f = tpl.merge(dep, on=ID, how="left", validate="one_to_one")
    assert len(f) == len(tpl) and (f[ID].to_numpy() == tpl[ID].to_numpy()).all()
    f = derive(f)
    f.attrs["kind"] = kind
    f.attrs["inputs"] = {str(v.relative_to(C)): sha256(v) for v in p.values()}
    return f


def load_training_frame(months=(1, 2, 6, 7)) -> tuple[pd.DataFrame, np.ndarray]:
    """2025 departures of the given takeoff months as a prediction frame, and their labels y.

    The frame has the ranking files' schema: the departure's label and off-block time are removed
    (TAXITIME_SEC_mvt dropped, BLOCK_TIME_UTC_mvt blanked), so no step can read them."""
    files = [training_file(m) for m in months]
    t = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dep = t[t["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    assert dep[ID].is_unique
    y = dep[TARGET].to_numpy(dtype=float)
    assert np.isfinite(y).all(), "2025 departures without a label"
    f = derive(dep.drop(columns=[TARGET]).assign(BLOCK_TIME_UTC_mvt=pd.NaT))
    assert set(f["month"].unique()) == set(months), "takeoff months differ from the files' months"
    f.attrs["kind"] = "train2025"
    f.attrs["inputs"] = {str(v.relative_to(C)): sha256(v) for v in files}
    return f, y


def derive(f: pd.DataFrame) -> pd.DataFrame:
    assert f["MVT_TIME_UTC_mvt"].notna().all() and f["SCHED_TIME_UTC_mvt"].notna().all()
    mvt = f["MVT_TIME_UTC_mvt"]
    f["id"] = f[ID].round().astype("int64")
    f["apt"] = f["ADEP_mvt"].astype(str)
    f["year"] = mvt.dt.year.astype(int)
    f["month"] = mvt.dt.month.astype(int)
    f["mday"] = mvt.dt.day.astype(int)
    f["day"] = mvt.dt.strftime("%m%d")
    f["d"] = secs(mvt, f["SCHED_TIME_UTC_mvt"])
    f["d_sched"] = f["d"]
    f["d_lobt"] = secs(mvt, f["LOBT_flt"])
    f["d_aobt3"] = secs(mvt, f["AOBT_3_flt"])
    f["nm"] = f["FLIGHT_ID_mvt"].notna().to_numpy()
    f["unmatched"] = f["AOBT_3_flt"].isna().astype(float).to_numpy()
    f["lirf_orphan"] = (f["apt"] == "LIRF").to_numpy() & ~f["nm"].to_numpy()
    f["sr"] = (f["STAND_mvt"].notna() & f["RUNWAY_mvt"].notna()).to_numpy()
    return f


def summary(f: pd.DataFrame) -> str:
    g = f.groupby(["year", "month"]).agg(rows=("id", "size"), nm=("nm", "mean"), unmatched=("unmatched", "mean"),
                                         gt4h=("d", lambda s: int((s > 4 * 3600).sum())))
    return g.round(4).to_string()


if __name__ == "__main__":
    import sys
    for k in sys.argv[1:] or ["board", "final"]:
        fr = load_frame(k)
        print(f"[{k}] {len(fr):,} rows; inputs {fr.attrs['inputs']}")
        print(summary(fr))
