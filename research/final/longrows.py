"""Rows with d > 4 h: member stack blended with a physical estimate (WS-D construction).

For a departure more than 4 h late the learners extrapolate a ratio of d, while in 2025 almost every
such flight that was not a LIRF schedule fill taxied normally. The construction (research/tail/
longrows_v30.py, used by v30-v32c) is, on rows with d > 4 h that are not LIRF-unmatched:

    M = members stacked with NNLS weights fitted on board-like, non-fill 1-4 h OOF rows
    E = physical estimate (research/tail/tailkit.phys): MVT - AOBT_3 minus the airport's median
        lag when 0 <= MVT - AOBT_3 <= d (kind N); else grade-A ADS-B stand departure under 1 h plus
        the airport offset (kind A); else the airport's median y (kind M). Tables come from 2025
        departures in the same band (4-8 h, > 8 h), LIRF-unmatched and date-shift rows excluded.
    y_hat = M + w(d) (E - M),  w(d) = clip(a + b log(d / 4 h), 0, 1)   ("plog")

(a, b) is fitted by least squares on the A+B OOF rows past 4 h, with E built from tables that
exclude the scored fold's half-months. v30-v32c used M = 0.5 M_new + 0.5 M_old, where M_old was the
v21 submission (members no longer retained); `alpha` and `m_old` exist only to reproduce that.
The final pipeline uses alpha = 1.

Two row groups past 4 h are not blended here; the caller decides their value:
  - LIRF rows without AOBT_3 ("fill" candidates: 80-98% were recorded at the schedule time in 2025);
  - rows whose AOBT_3 lies more than `guard_h` (4) hours before takeoff (recording-mode rows: the
    blend toward MVT - AOBT_3 cost one 2025 OOF row +4,087 fold MSE). v30-v32c kept v29 there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

C = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(C / "research" / "tail"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stacking import load_oof  # noqa: E402
from tailkit import apply_wlog, fill_prob, fit_wlog, is_fill, kind, nnls_w, phys  # noqa: E402,F401

H = 3600.0
LONG_TR = C / "research" / "tail" / "cache" / "long_tr.parquet"
PHYS_COLS = ["apt", "d_sched", "unmatched", "d_aobt3", "a3_grade", "a3_obs_leave"]


def long_tr() -> pd.DataFrame:
    """2025 departures with d > 1 h and the estimator inputs (fs5 store), from research/tail/build_cache.py."""
    return pd.read_parquet(LONG_TR)


def fold_held(tr: pd.DataFrame, f: str) -> np.ndarray:
    """2025 rows inside fold f's scored half-months (A = Jan/Jul days 16-31, B = days 1-15)."""
    return (tr.month.isin([1, 7]) & ((tr.mday >= 16) if f == "A" else (tr.mday <= 15))).to_numpy()


def long_oof(members: list[str], specs: dict, m: pd.DataFrame, tr: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per A/B OOF departure with d > 1 h: labels, folds, estimator inputs, p_<member>."""
    tr = (long_tr() if tr is None else tr).set_index("MVT_ID_mvt")
    oofs = {r: load_oof(specs[r]["oof"]) for r in members}
    ids = m.index[m.ds > H]
    for o in oofs.values():
        ids = ids.intersection(o.index)
    ids = ids.sort_values()
    df = m.loc[ids, ["y", "ds", "day", "artifact", "boardlike"]].copy()
    df["fold"] = oofs[members[0]].loc[ids, "fold"].to_numpy()
    t = tr.reindex(ids)
    assert t.apt.notna().all(), "long OOF rows missing from long_tr"
    assert np.allclose(t.d_sched.to_numpy(), df.ds.to_numpy(), atol=1.0)
    df = df.join(t[["apt", "d_sched", "unmatched", "d_aobt3", "a3_grade", "a3_obs_leave", "month", "mday"]])
    for r, o in oofs.items():
        assert (o.loc[ids, "fold"].to_numpy() == df.fold.to_numpy()).all(), r
        df[f"p_{r}"] = o.loc[ids, "p"].to_numpy()
    return df


def stack_rows(df: pd.DataFrame) -> np.ndarray:
    """Rows the long-stack weights are fitted on: board-like, 1 h < d <= 4 h, not LIRF-unmatched."""
    ds = df.ds.to_numpy()
    return df.boardlike.to_numpy() & (ds > H) & (ds <= 4 * H) & ~is_fill(df)


def stack_weights(df: pd.DataFrame, members: list[str]) -> np.ndarray:
    P = df[[f"p_{r}" for r in members]].to_numpy()
    s = stack_rows(df)
    return nnls_w(P[s], df.y.to_numpy()[s])


def oof_phys(df: pd.DataFrame, tr: pd.DataFrame) -> np.ndarray:
    """E on OOF rows, each fold's tables built from 2025 rows outside its scored half-months."""
    trx = tr.set_index("MVT_ID_mvt") if "MVT_ID_mvt" in tr else tr
    E = np.full(len(df), np.nan)
    for f in "AB":
        s = (df.fold == f).to_numpy()
        E[s] = phys(df[s], trx[~fold_held(trx, f)])
    return E


def fit_plog(df: pd.DataFrame, M_oof: np.ndarray, tr: pd.DataFrame) -> dict:
    """w(d) = a + b log(d / 4 h), pooled over estimator kinds, on board-like non-fill OOF rows past 4 h."""
    ds, y = df.ds.to_numpy(), df.y.to_numpy()
    fr = df.boardlike.to_numpy() & (ds > 4 * H) & ~is_fill(df)
    E = oof_phys(df, tr)
    coef = fit_wlog(y[fr], M_oof[fr], E[fr], ds[fr], np.full(int(fr.sum()), "N"))
    return {k: [float(x) for x in v] for k, v in coef.items()}


def guard_rows(fr: pd.DataFrame, guard_h: float | None) -> np.ndarray:
    if guard_h is None:
        return np.zeros(len(fr), bool)
    return (fr.d_aobt3 > guard_h * H).to_numpy()


def construct(fr: pd.DataFrame, M: np.ndarray, coef: dict, tr: pd.DataFrame, lo_h: float = 4.0,
              guard_h: float | None = 4.0):
    """Blend M with E on the frame's rows past lo_h that are neither fill nor guard rows.
    Returns (mask, y_hat, E, w, kind)."""
    d = fr.d_sched.to_numpy()
    nf = (d > lo_h * H) & ~is_fill(fr) & ~guard_rows(fr, guard_h)
    sub = fr.loc[nf, PHYS_COLS]
    E = phys(sub, tr)
    q, w = apply_wlog({k: np.asarray(v) for k, v in coef.items()}, M[nf], E, d[nf], np.full(int(nf.sum()), "N"))
    return nf, np.clip(q, 60, None), E, w, kind(sub)
