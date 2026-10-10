"""Member predictions and NNLS segment weights.

A member is declared in the config as

    "members": {"<name>": {"pred": "<run dir or .parquet>", "oof": "<run dir with oof.csv>"}}

`pred` holds submission.parquet (MVT_ID_mvt, TAXITIME_SEC_mvt) covering every id of the frame.
`oof` is the A/B out-of-fold run of the same specification (2025 Jan + Jul, folds A = days 16-31,
B = days 1-15). For refits on the final store, `oof` points at the OLD run of the same spec: the
refits have no A/B folds, and the weights are a property of the specification.

Weights per delay segment are fitted by non-negative least squares on board-like OOF rows of the
segment (research/evalb.py's definition: no LIRF date-shift artifacts, no short-delay y > 3 h rows),
normalised to sum to 1 and rounded to 6 decimals; the rounded weights are the ones applied and
recorded. This is research/assemble.py's arithmetic, restricted to the members present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

C = Path(__file__).resolve().parents[2]
META = C / "research" / "cache" / "oof_meta_v2.parquet"
H = 3600.0


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else C / q


def pred_file(spec: str) -> Path:
    """A .parquet path as given; anything else is a run directory holding submission.parquet."""
    p = resolve(spec)
    return p if p.suffix == ".parquet" else p / "submission.parquet"


def oof_file(spec: str) -> Path:
    """A .csv/.parquet path as given; anything else is a run directory holding oof.csv."""
    p = resolve(spec)
    return p if p.suffix in (".csv", ".parquet") else p / "oof.csv"


def oof_meta() -> pd.DataFrame:
    m = pd.read_parquet(META)
    m["id"] = m["id"].round().astype("int64")
    m = m.set_index("id")
    assert m.index.is_unique
    return m


def load_oof(spec: str) -> pd.DataFrame:
    """id -> fold, p (mean over duplicate ids, as research/assemble.py)."""
    p = oof_file(spec)
    o = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    o["id"] = o["id"].round().astype("int64")
    return o.groupby("id").agg(fold=("fold", "first"), p=("prediction", "mean"))


def fit_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, _ = nnls(P, y)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


def oof_matrix(members: list[str], specs: dict, m: pd.DataFrame, rows: str = "boardlike"):
    """Member OOF matrix on the ids every member shares (board-like by default)."""
    oofs = [load_oof(specs[r]["oof"]) for r in members]
    ids = m.index[m["boardlike"]] if rows == "boardlike" else m.index
    for o in oofs:
        ids = ids.intersection(o.index)
    P = np.column_stack([o.loc[ids, "p"].to_numpy() for o in oofs])
    fold = oofs[0].loc[ids, "fold"].to_numpy()
    for o in oofs[1:]:
        assert (o.loc[ids, "fold"].to_numpy() == fold).all(), "members disagree on folds"
    return P, ids, fold


def in_segment(ds: np.ndarray, lo_h: float, hi_h: float) -> np.ndarray:
    return (ds > lo_h * H) & (ds <= hi_h * H)


def segment_weights(seg: dict, specs: dict, m: pd.DataFrame) -> dict[str, float]:
    P, ids, _ = oof_matrix(seg["members"], specs, m)
    s = in_segment(m.loc[ids, "ds"].to_numpy(), seg["lo_h"], seg["hi_h"])
    w = fit_weights(P[s], m.loc[ids, "y"].to_numpy()[s])
    return dict(zip(seg["members"], np.round(w, 6).tolist()))


def crossfit(seg: dict, specs: dict, m: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Cross-fitted A<->B segment prediction on its board-like OOF rows (weights from the other fold).

    With `weights`, those fixed weights are applied instead (no refit), for comparing a frozen
    weight vector against a refit on the same rows."""
    P, ids, fold = oof_matrix(seg["members"], specs, m)
    y = m.loc[ids, "y"].to_numpy()
    s = in_segment(m.loc[ids, "ds"].to_numpy(), seg["lo_h"], seg["hi_h"])
    out = np.full(len(ids), np.nan)
    for f in ("A", "B"):
        k = s & (fold == f)
        if weights is None:
            w = fit_weights(P[s & (fold != f)], y[s & (fold != f)])
        else:
            w = np.array([weights[r] for r in seg["members"]])
        out[k] = P[k] @ w
    return pd.Series(out, index=ids)[s]


def member_pred(spec: str, ids: np.ndarray) -> np.ndarray:
    s = pd.read_parquet(pred_file(spec))
    s.index = s["MVT_ID_mvt"].round().astype("int64")
    assert s.index.is_unique, spec
    v = s["TAXITIME_SEC_mvt"].reindex(ids).to_numpy(dtype=float)
    assert np.isfinite(v).all(), f"{spec}: {int((~np.isfinite(v)).sum())} frame ids missing or not finite"
    return v
