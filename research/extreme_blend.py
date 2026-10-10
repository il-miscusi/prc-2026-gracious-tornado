"""Physical estimate for genuine extreme delays, blended into a submission's long rows.

    python3 research/extreme_blend.py IN.parquet OUT.parquet [--hours 8] [--w 0.5]
    python3 research/extreme_blend.py IN.parquet OUT.parquet --bands 4:0.11,8:0.695

Departures more than --hours late (excluding LIRF rows without a flight-plan match, which are
schedule fills: y = d) almost always taxi normally in 2025: median ~1,000 s, 6 of 1,212 rows over
an hour past 8 h. The ratio model predicts y = r x d there and extrapolates badly (hold-out mean
1,828 s vs 1,096 at d > 24 h; some 2026 rows are predicted at 4-14k s). The estimate E is, in
order of preference: MVT - AOBT_3 minus the airport's median lag (when 0 < d_aobt3 < d); grade-A
ADS-B stand departure plus the airport's median offset (only under 1 h: longer ADS-B leave times
are tows in 2025, labels median 903 s); else the airport's median y. Tables come from all 2025
rows past the threshold. Hold-out test (findings_20260930.md): w 0.5 at 8 h gains -30 (fold A) and
-64 (fold B) board-MSE units on the cross-fitted v21 stack.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

C = Path(__file__).resolve().parents[1]
FS = C / "research" / "cache" / "fs5"
COLS = ["apt", "d_sched", "unmatched", "d_aobt3", "a3_grade", "a3_obs_leave"]


def tables(tr: pd.DataFrame, hours: float, hi_hours: float = 1e9):
    X = tr[(tr.d_sched > hours * 3600) & (tr.d_sched <= hi_hours * 3600) & ~tr.artifact.astype(bool) & ~((tr.apt == "LIRF") & (tr.unmatched == 1))]
    A = X[(X.a3_grade == 3) & X.a3_obs_leave.between(0, 3600)]
    N = X[X.d_aobt3.between(0, X.d_sched)]
    return (A.y - A.a3_obs_leave).groupby(A.apt).median(), (N.d_aobt3 - N.y).groupby(N.apt).median(), X.y.groupby(X.apt).median()


def estimate(df: pd.DataFrame, tabs) -> np.ndarray:
    off, gap, med = tabs
    e = df.apt.map(med).astype(float).to_numpy()
    ok_a = ((df.a3_grade == 3) & df.a3_obs_leave.between(0, 3600)).to_numpy()
    e = np.where(ok_a, df.a3_obs_leave + df.apt.map(off).fillna(0), e)
    ok_n = df.d_aobt3.between(0, df.d_sched).to_numpy()
    e = np.where(ok_n, df.d_aobt3 - df.apt.map(gap).fillna(0), e)
    return np.clip(e, 60, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--bands", default="", help="lower-hours:weight,... (per-band weights fitted on the "
                    "cross-fitted hold-out, research/findings_20260930.md); overrides --hours/--w")
    a = ap.parse_args()
    tr = pd.read_parquet(FS / "tr_base.parquet", columns=COLS + ["y", "artifact"])
    rk = pd.read_parquet(FS / "rk_base.parquet", columns=["MVT_ID_mvt"] + COLS).set_index("MVT_ID_mvt")
    sub = pd.read_parquet(a.inp)
    df = rk.reindex(sub.MVT_ID_mvt.to_numpy())
    p = sub.TAXITIME_SEC_mvt.to_numpy()
    bands = sorted((float(h), float(w)) for h, w in (b.split(":") for b in a.bands.split(","))) if a.bands \
        else [(a.hours, a.w)]
    fill = ((df.apt == "LIRF") & (df.unmatched == 1)).to_numpy()
    q, sel = p.copy(), np.zeros(len(p), bool)
    for i, (lo, w) in enumerate(bands):
        hi = bands[i + 1][0] if i + 1 < len(bands) else 1e9
        s = (df.d_sched > lo * 3600).to_numpy() & (df.d_sched <= hi * 3600).to_numpy() & ~fill
        e = estimate(df[s], tables(tr, lo, hi))
        q[s] = w * e + (1 - w) * p[s]
        sel |= s
    out = sub.assign(TAXITIME_SEC_mvt=np.clip(q, 60, None))
    assert len(out) == 344841 and out.TAXITIME_SEC_mvt.notna().all()
    out.to_parquet(a.out, index=False)
    ch = q - p
    print(f"rows changed {int(sel.sum())}  mean change {ch[sel].mean():+.0f} s  sum sq change / N {float((ch ** 2).sum() / len(q)):.0f}"
          f"  -> {a.out}")


if __name__ == "__main__":
    main()
