"""Building blocks for the > 4 h long-row rebuild (WS-D stage 1).

Everything here is fitted on one set of rows and applied to another, so the same functions serve the
cross-fitted A<->B evaluation (exp_stage1.py) and the 2026 candidate (longrows_v30.py).

Physical estimator (extreme_blend.py's, with tables passed in so they can come from training folds
only): kind N = MVT - AOBT_3 minus the airport's median lag (0 < d_aobt3 < d), kind A = grade-A ADS-B
stand departure under 1 h plus the airport offset, kind M = airport median y.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

C = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(C / "research"))
from extreme_blend import estimate, tables  # noqa: E402

KINDS = ("N", "A", "M")
H = 3600.0


def is_fill(df: pd.DataFrame) -> np.ndarray:
    return ((df.apt == "LIRF") & (df.unmatched == 1)).to_numpy()


def kind(df: pd.DataFrame) -> np.ndarray:
    n = df.d_aobt3.between(0, df.d_sched).to_numpy()
    a = ((df.a3_grade == 3) & df.a3_obs_leave.between(0, 3600)).to_numpy()
    return np.where(n, "N", np.where(a, "A", "M"))


def phys(df: pd.DataFrame, tr: pd.DataFrame, cuts=(4, 8)) -> np.ndarray:
    """Physical estimate with band tables (lo h, next lo h) built from `tr` only."""
    e = np.full(len(df), np.nan)
    d = df.d_sched.to_numpy()
    edges = list(cuts) + [1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (d > lo * H) & (d <= hi * H)
        if s.any():
            e[s] = estimate(df[s], tables(tr, lo, hi))
    return e


def nnls_w(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, _ = nnls(P, y)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


def fit_wlog(y, m, e, d, k, lo_h=4.0, ridge=4.0, slope=True):
    """Per-kind weight w_k(d) = a_k + b_k * log(d / lo_h), least squares on y - m = w (e - m).

    Kind coefficients are shrunk toward the pooled fit with a ridge whose strength is `ridge` x
    the mean squared (e - m) of the kind, i.e. in units of rows. Returns {kind: (a, b)}.
    """
    t = np.log(d / (lo_h * H))
    g = e - m
    r = y - m

    def solve(sel, prior=None, lam=0.0):
        X = np.column_stack([g[sel], g[sel] * t[sel]]) if slope else g[sel][:, None]
        A = X.T @ X
        b = X.T @ r[sel]
        p0 = np.zeros(X.shape[1]) if prior is None else prior
        L = lam * np.mean(g[sel] ** 2) * np.eye(X.shape[1]) if sel.any() else np.zeros_like(A)
        return np.linalg.solve(A + L + 1e-9 * np.eye(X.shape[1]), b + L @ p0)

    allr = np.ones(len(y), bool)
    pooled = solve(allr)
    out = {"pooled": pooled}
    for kk in KINDS:
        s = k == kk
        out[kk] = solve(s, pooled, ridge) if s.sum() >= 3 else pooled
    return out


def apply_wlog(coef, m, e, d, k, lo_h=4.0, slope=True):
    t = np.log(d / (lo_h * H))
    w = np.zeros(len(m))
    for kk in KINDS:
        s = k == kk
        c = coef.get(kk, coef["pooled"])
        w[s] = c[0] + (c[1] * t[s] if slope else 0.0)
    w = np.clip(w, 0.0, 1.0)
    return m + w * (e - m), w


def fill_prob(tr: pd.DataFrame, d: np.ndarray, lo_h=4.0) -> tuple[np.ndarray, float]:
    """P(schedule fill | LIRF unmatched, d) from `tr`'s LIRF unmatched rows past lo_h, logistic in
    log d (unweighted), and the non-fill median y. Fill = |y - d| <= 120 s."""
    x = tr[(tr.apt == "LIRF") & (tr.unmatched == 1) & ~tr.artifact.astype(bool) & (tr.d_sched > lo_h * H)]
    f = ((x.y - x.d_sched).abs() <= 120).to_numpy().astype(float)
    t = np.log(x.d_sched.to_numpy() / (lo_h * H))
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0).fit(t[:, None], f)
    p = lr.predict_proba(np.log(d / (lo_h * H))[:, None])[:, 1]
    e0 = float(np.median(x.y.to_numpy()[f == 0])) if (f == 0).any() else 1100.0
    return p, e0
