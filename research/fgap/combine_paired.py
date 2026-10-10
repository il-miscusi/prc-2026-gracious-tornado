"""Combine paired variants runs into one OOF per arm (feature-gap lane, 2026-10-09).

    python3 research/fgap/combine_paired.py OUTPUT_DIR [--arms base,new]
    python3 research/fgap/combine_paired.py OUT_DIR --runs RUN1_DIR,RUN2_DIR [--arms base,new]

For each arm, over every run directory given (default: OUTPUT_DIR alone): short rows (d <= 1 h, where the
long-only runs are NaN) are the mean of the <arm>/oof.csv runs (full short+long fits, one per seed); long rows
are the mean of those and every <arm>_lK/oof.csv (long-only seeds).  Writes <OUT>/<arm>_comb/oof.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--arms", default="base,new")
ap.add_argument("--runs", default="", help="comma list of run directories (default: OUT itself)")
a = ap.parse_args()
root = Path(a.out)
runs = [Path(r) for r in a.runs.split(",")] if a.runs else [root]
for arm in a.arms.split(","):
    mains = [pd.read_csv(r / arm / "oof.csv").set_index(["id", "fold"])["prediction"] for r in runs]
    idx = mains[0].index
    mains = [m.reindex(idx) for m in mains]
    longs = [pd.read_csv(p).set_index(["id", "fold"])["prediction"].reindex(idx)
             for r in runs for p in sorted(r.glob(f"{arm}_l*/oof.csv"))]
    M = np.column_stack([m.to_numpy() for m in mains])
    short_pred = np.nanmean(M, axis=1)
    if longs:
        L = np.column_stack([s.to_numpy() for s in longs])
        long_rows = ~np.isnan(L).all(axis=1)
        allp = np.column_stack([M, L])
        comb = np.where(long_rows, np.nanmean(allp, axis=1), short_pred)
    else:
        long_rows = np.zeros(len(idx), bool)
        comb = short_pred
    out = root / f"{arm}_comb"
    out.mkdir(parents=True, exist_ok=True)
    df = idx.to_frame(index=False)
    df["prediction"] = comb
    df.to_csv(out / "oof.csv", index=False)
    print(f"{arm}: {len(df):,} rows; short rows mean of {M.shape[1]} full fits; {int(long_rows.sum()):,} long rows "
          f"mean of {M.shape[1] + len(longs)} seeds")
