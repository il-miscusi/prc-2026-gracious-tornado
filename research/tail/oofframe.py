"""Long-row OOF frame shared by the WS-D scripts.

load() returns one row per A/B OOF departure with d_sched > 1 h: labels, meta, the physical-estimator
inputs and every member's OOF prediction (columns p_<run>), plus `v29`, the v29 construction replayed
on OOF exactly as research/review_audit_20261001.py does it (v18 long stack cross-fitted, 1-4 h band
re-stacked, 4-8 h / 8 h+ physical blend 0.11 / 0.695 with tables from rows outside the scored fold).
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

K = C / "solutions" / "kernels"
TAIL = C / "research" / "tail" / "cache"
BAND = ["prc-c6s", "prc-c5s", "prc-x5", "prc-c6", "prc-cs5", "prc-split-fs4"]
OLD = ["prc-c5d10", "prc-cat-fs4", "prc-cs5", "prc-xgb-fs4", "prc-x5", "prc-split-fs4"]
EXTRA = ["prc-cs6", "prc-mlp6fs", "prc-c6x2", "prc-c6sx2", "prc-c5sx2", "prc-cs6x2"]
MEMBERS = list(dict.fromkeys(BAND + OLD + EXTRA))


def oof(run: str) -> pd.Series:
    o = pd.read_csv(K / run / "output" / "oof.csv")
    o["id"] = o.id.round().astype("int64")
    assert o.id.is_unique, run
    return o.set_index("id")


def load(members=MEMBERS) -> pd.DataFrame:
    m = pd.read_parquet(C / "research" / "cache" / "oof_meta_v2.parquet")
    m["id"] = m.id.round().astype("int64")
    m = m.set_index("id")
    assert m.index.is_unique
    tr = pd.read_parquet(TAIL / "long_tr.parquet").set_index("MVT_ID_mvt")
    base = oof(BAND[0])
    ids = base.index[base.index.isin(m.index[m.ds > 3600])]
    df = m.loc[ids, ["y", "ds", "day", "artifact", "boardlike", "seg"]].copy()
    df["fold"] = base.fold.reindex(ids).to_numpy()
    t = tr.reindex(ids)
    assert t.apt.notna().all(), "long OOF rows missing from long_tr"
    assert np.allclose(t.d_sched.to_numpy(), df.ds.to_numpy(), atol=1.0)
    assert np.allclose(t.y.to_numpy(), df.y.to_numpy(), atol=0.5)
    df = df.join(t.drop(columns=["y", "artifact"]))
    for r in members:
        o = oof(r)
        assert ids.isin(o.index).all(), r
        assert (o.fold.reindex(ids).to_numpy() == df.fold.to_numpy()).all(), r
        df[f"p_{r}"] = o.prediction.reindex(ids).to_numpy()
    df["v29"] = replay_v29(df, tr)
    return df


def nnls_w(P, y):
    w, _ = nnls(P, y)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


def replay_v29(df: pd.DataFrame, tr: pd.DataFrame) -> np.ndarray:
    """review_audit_20261001.py's v29 replay restricted to long rows (identical arithmetic)."""
    ds, y, valid, fold = df.ds.to_numpy(), df.y.to_numpy(), df.boardlike.to_numpy(), df.fold.to_numpy()
    pred = np.zeros(len(df))
    fill = ((df.apt == "LIRF") & (df.unmatched == 1)).to_numpy()
    for f in ("A", "B"):
        test, train = fold == f, (fold != f) & valid
        for runs, s in ((OLD, ds > 3600), (BAND, (ds > 3600) & (ds <= 14400))):
            P = df[[f"p_{r}" for r in runs]].to_numpy()
            w = nnls_w(P[train & s], y[train & s])
            pred[test & s] = P[test & s] @ w
        fit = ~(tr.month.isin([1, 7]) & ((tr.mday >= 16) if f == "A" else (tr.mday <= 15)))
        for lo, hi, w in [(4, 8, 0.11), (8, 1e9, 0.695)]:
            s = test & (ds > lo * 3600) & (ds <= hi * 3600) & ~fill
            e = estimate(df.loc[s], tables(tr.loc[fit], lo, hi))
            pred[s] = w * e + (1 - w) * pred[s]
    return pred


def fold_held(tr: pd.DataFrame, f: str) -> pd.Series:
    return tr.month.isin([1, 7]) & ((tr.mday >= 16) if f == "A" else (tr.mday <= 15))
