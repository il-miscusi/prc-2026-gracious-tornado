"""Airport-month level calibration to EUROCONTROL's published monthly taxi-out means.

Generalises research/pru/level_cal.py (same arithmetic) from the Jan/Jul 2026 board to any frame:
every (year, month, airport) cell present in the frame that the monthly file covers is calibrated.

Per cell, over the PRU-comparable selection (stand and runway present, prediction <= 2 h):
    x     = published mean - our mean prediction
    shift = x + c,  c = composition offset (airport intercept + b_deice * de-icing share +
                        b_noref * NO_REF share), fitted on 2025 cells against the 12-month-out c6 OOF
    applied shift = lam * shift, added to every selected row of the cell; 0 when the cell's
                    de-icing-excluded share is >= cap (the published mean then omits flights de-iced
                    after off-block, e.g. EDDM Jan/Feb 2026).
The 2025 fit uses only cells under the same de-icing cap. v32c used lam 0.75, cap 0.05.

Inputs (fixed paths): research/pru/taxi_out_additional_time_{2025,2026}.csv, the twelve 2025
training files, research/cache/oof_mo12_c6.parquet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

C = Path(__file__).resolve().parents[2]
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
PRU = {y: C / "research" / "pru" / f"taxi_out_additional_time_{y}.csv" for y in (2025, 2026)}
TRAIN_DIR = C / "data" / "09-22-2026-08-13-30_files_list"
TRAIN = [TRAIN_DIR / f"training_2025-{m:02d}-01_{2025 + (m == 12)}-{m % 12 + 1:02d}-01.parquet" for m in range(1, 13)]
C6_OOF = C / "research" / "cache" / "oof_mo12_c6.parquet"
KEY = ["year", "month", "apt"]


def inputs() -> list[Path]:
    return [PRU[2025], PRU[2026], C6_OOF] + TRAIN


def pru() -> pd.DataFrame:
    p = pd.concat([pd.read_csv(PRU[y]) for y in (2025, 2026)])
    p = p[p.APT_ICAO.isin(APTS)].copy()
    p["pub"] = (p.TOTAL_REF_TIME_MIN + p.TOTAL_ADD_TIME_MIN) * 60 / p.TOTAL_REF_NB_FL
    p["deice"] = (p.TF - p.VALID_FL) / p.TF
    p["noref"] = p.NO_REF / p.TF
    return p.rename(columns={"APT_ICAO": "apt", "YEAR": "year", "MONTH_NUM": "month"})[
        KEY + ["pub", "deice", "noref"]]


def train_rows() -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt",
            "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
    t = pd.concat([pd.read_parquet(f, columns=cols) for f in TRAIN])
    t = t[t.PHASE_mvt == "DEP"]
    d = pd.DataFrame({"id": t.MVT_ID_mvt.astype("int64").to_numpy(), "apt": t.ADEP_mvt.to_numpy(),
                      "year": t.MVT_TIME_UTC_mvt.dt.year.to_numpy(), "month": t.MVT_TIME_UTC_mvt.dt.month.to_numpy(),
                      "sr": (t.STAND_mvt.notna() & t.RUNWAY_mvt.notna()).to_numpy(),
                      "ds": (t.MVT_TIME_UTC_mvt - t.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy(),
                      "y": t.TAXITIME_SEC_mvt.astype(float).to_numpy()})
    oof = pd.read_parquet(C6_OOF, columns=["id", "prediction"]).rename(columns={"prediction": "pred"})
    oof["id"] = oof["id"].astype("int64")
    return d.merge(oof, on="id")


def select(d: pd.DataFrame) -> pd.DataFrame:
    d["sel_x"] = d.sr & (d.pred <= 7200)
    d["sel"] = d["sel_x"]
    return d


def cells(d: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    g = d[d.sel_x].groupby(KEY).agg(pm=("pred", "mean")).reset_index()
    s = d[d.sel]
    g = g.merge(s.groupby(KEY).agg(n=("pred", "size")).reset_index())
    if "y" in s:
        g = g.merge(s.assign(res=s.y - s.pred).groupby(KEY).res.mean().reset_index())
    g = g.merge(p, on=KEY)
    g["x"] = g.pub - g.pm
    return g


def design(g: pd.DataFrame) -> np.ndarray:
    return np.column_stack([(g.apt == a).astype(float) for a in APTS] + [g.deice, g.noref])


def fit(g: pd.DataFrame, ridge: float = 1e-3) -> np.ndarray:
    X = design(g)
    w = g.n.to_numpy() / g.n.mean()
    A = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ ((g.res - g.x).to_numpy() * w))


_CELLS25 = None


def fit_coef(cap: float, exclude: list[tuple[int, int]] = ()) -> np.ndarray:
    """Composition-offset coefficients from 2025 cells under the de-icing cap, leaving out the
    (year, month) cells in `exclude` (the 2025 pseudo-final fits without its scored month)."""
    global _CELLS25
    if _CELLS25 is None:
        _CELLS25 = cells(select(train_rows()), pru())
    g = _CELLS25
    keep = (g.deice < cap) & ~pd.Series(list(zip(g.year, g.month)), index=g.index).isin(list(exclude))
    return fit(g[keep])


def frame_rows(fr: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"apt": fr.apt.to_numpy(), "year": fr.year.to_numpy(), "month": fr.month.to_numpy(),
                         "sr": fr.sr.to_numpy(), "ds": fr.d.to_numpy(), "pred": np.asarray(pred, float)})


def apply(fr: pd.DataFrame, pred: np.ndarray, lam: float, cap: float, coef: np.ndarray | None = None):
    """Returns (calibrated prediction clipped at 60, cell table, coef)."""
    coef = fit_coef(cap) if coef is None else coef
    p = pru()
    r = select(frame_rows(fr, pred))
    g = cells(r, p)
    g["comp"] = design(g) @ coef
    g["sh"] = np.where(g.deice < cap, lam * (g.x + g.comp), 0.0)
    missing = r[r.sel].groupby(KEY).size().index.difference(g.set_index(KEY).index)
    if len(missing):
        print(f"[level] WARNING: {len(missing)} frame cells have no PRU row and are not calibrated: {list(missing)}")
    sh = r.merge(g[KEY + ["sh"]], on=KEY, how="left").sh.fillna(0).to_numpy()
    new = np.clip(r.pred.to_numpy() + np.where(r.sel.to_numpy(), sh, 0.0), 60, None)
    return new, g, coef


def composition_check(fr: pd.DataFrame, pred: np.ndarray, coef: np.ndarray, tol: float = 60.0) -> pd.DataFrame:
    """Checklist item 2: per airport-month, published mean + 2025 composition offset - our mean over
    the PRU-comparable selection of `pred`. Cells outside +-tol are flagged (not failed)."""
    r = select(frame_rows(fr, pred))
    g = cells(r, pru())
    g["comp"] = design(g) @ coef
    g["gap"] = g.pub + g.comp - g.pm
    g["flag"] = np.where(g.gap.abs() > tol, "CHECK", "")
    return g[KEY + ["n", "pub", "comp", "pm", "gap", "deice", "flag"]]
