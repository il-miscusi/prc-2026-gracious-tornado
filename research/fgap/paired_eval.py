"""Paired month-out comparison of two member fold runs (feature-gap lane, 2026-10-09).

    python3 research/fgap/paired_eval.py BASE_OOF NEW_OOF [--folds M2,M7] [--reps 2000] [--json out.json]

BASE_OOF / NEW_OOF: oof.csv files (id, fold, prediction), or kernel names under solutions/kernels/<name>/output,
or <kernel>/<variant> for a variants run. Rows are every 2025 departure of the held month (untrimmed, artifact
rows included), joined on id; only ids present in both runs are scored (the script says how many dropped).

For each fold: full SSE/N of both runs, dMSE = new - base, and an airport-day block bootstrap CI of dMSE
(resampling airport x UTC-day blocks of the fold with replacement). The same per band (d = takeoff - schedule:
<=1 h, 1-4 h, >4 h) and per airport, where dMSE is reported both per cell row (the cell's own MSE change, with
its own block-bootstrap CI) and as the cell's contribution to the all-row dMSE (cell dSSE / fold N).

"nw" (no date-wrap): the same all-row dMSE with the 2025 date-wrap rows left unchanged (dSSE set to 0 on rows with
y > d + 600 and y > 80,000, research/final/levers/evaluate.py), the figure to expect in 2026, as in the levers table.

Gate (lane brief): all-row dMSE <= -300 on every fold, and no band or airport whose own CI lies wholly above 0.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

C = Path(__file__).resolve().parents[2]
D = C / "data" / "09-22-2026-08-13-30_files_list"
BANDS = [(-1e9, 3600, "<=1h"), (3600, 14400, "1-4h"), (14400, 1e9, ">4h")]


def labels(months: list[int]) -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
    files = [f for m in months for f in sorted(glob.glob(str(D / f"training_2025-{m:02d}-*.parquet")))]
    lab = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    lab = lab[lab.PHASE_mvt == "DEP"].copy()
    lab["id"] = lab.MVT_ID_mvt.astype("int64")
    lab["d"] = (lab.MVT_TIME_UTC_mvt - lab.SCHED_TIME_UTC_mvt).dt.total_seconds()
    lab["apt"] = lab.ADEP_mvt.astype(str)
    lab["day"] = lab.MVT_TIME_UTC_mvt.dt.strftime("%m-%d")
    lab["y"] = lab.TAXITIME_SEC_mvt.astype(float)
    return lab.set_index("id")[["y", "d", "apt", "day"]]


def load(ref: str) -> pd.DataFrame:
    p = Path(ref)
    if not p.suffix:
        base = C / "solutions" / "kernels"
        parts = ref.split("/")
        p = base / parts[0] / "output" / ("/".join(parts[1:]) + "/oof.csv" if len(parts) > 1 else "oof.csv")
    o = pd.read_csv(p)
    o["id"] = o["id"].astype("int64")
    return o.set_index("id")


def block_ci(blocks: np.ndarray, dsse: np.ndarray, n: np.ndarray, reps: int, rng) -> tuple[float, float]:
    """CI of sum(dsse)/sum(n) by resampling blocks."""
    codes, uniq = pd.factorize(blocks)
    bs = np.bincount(codes, weights=dsse)
    bn = np.bincount(codes, weights=n)
    k = len(uniq)
    idx = rng.integers(0, k, size=(reps, k))
    stats = bs[idx].sum(1) / np.maximum(bn[idx].sum(1), 1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def compare(base: pd.DataFrame, new: pd.DataFrame, lab: pd.DataFrame, folds: list[str], reps: int, seed: int = 0):
    out = {}
    rng = np.random.default_rng(seed)
    for f in folds:
        b, n_ = base[base.fold == f], new[new.fold == f]
        j = lab.join(b["prediction"].rename("pb"), how="inner").join(n_["prediction"].rename("pn"), how="inner")
        held = lab.index.isin(b.index) | lab.index.isin(n_.index)
        dropped = int(held.sum() - len(j))
        eb, en = (j.pb - j.y) ** 2, (j.pn - j.y) ** 2
        dsse = (en - eb).to_numpy()
        blocks = (j.apt + "|" + j.day).to_numpy()
        N = len(j)
        lo, hi = block_ci(blocks, dsse, np.ones(N), reps, rng)
        wrap = ((j.y > j.d + 600) & (j.y > 80000)).to_numpy()
        dnw = np.where(wrap, 0.0, dsse)
        nlo, nhi = block_ci(blocks, dnw, np.ones(N), reps, rng)
        res = {"n": N, "dropped": dropped, "rmse_base": float(np.sqrt(eb.mean())), "rmse_new": float(np.sqrt(en.mean())),
               "dmse": float(dsse.mean()), "ci": [lo, hi], "n_wrap": int(wrap.sum()), "dmse_nw": float(dnw.mean()),
               "ci_nw": [nlo, nhi], "cells": {}}
        cells = [(nm, ((j.d > a) & (j.d <= z)).to_numpy()) for a, z, nm in BANDS]
        cells += [(a, (j.apt == a).to_numpy()) for a in sorted(j.apt.unique())]
        for nm, k in cells:
            if not k.any():
                continue
            clo, chi = block_ci(blocks[k], dsse[k], np.ones(k.sum()), reps, rng)
            res["cells"][nm] = {"n": int(k.sum()), "rmse_base": float(np.sqrt(eb[k].mean())),
                                "rmse_new": float(np.sqrt(en[k].mean())), "dmse_cell": float(dsse[k].mean()),
                                "ci_cell": [clo, chi], "contrib": float(dsse[k].sum() / N),
                                "worse": bool(clo > 0)}
        out[f] = res
    return out


def report(res: dict, gate: float = -300.0) -> bool:
    ok = True
    for f, r in res.items():
        flag = r["dmse"] <= gate
        ok &= flag
        print(f"{f}: n={r['n']:,} (dropped {r['dropped']})  RMSE {r['rmse_base']:.2f} -> {r['rmse_new']:.2f}  "
              f"dMSE {r['dmse']:+.0f} [{r['ci'][0]:+.0f}, {r['ci'][1]:+.0f}]  nw {r['dmse_nw']:+.0f} [{r['ci_nw'][0]:+.0f}, "
              f"{r['ci_nw'][1]:+.0f}] ({r['n_wrap']} wrap rows)  {'<= ' + str(gate) if flag else 'FAILS ' + str(gate)}")
        for nm, c in r["cells"].items():
            ok &= not c["worse"]
            print(f"   {nm:6s} n={c['n']:>7,}  RMSE {c['rmse_base']:8.2f} -> {c['rmse_new']:8.2f}  cell dMSE {c['dmse_cell']:+9.0f} "
                  f"[{c['ci_cell'][0]:+9.0f}, {c['ci_cell'][1]:+9.0f}]  contrib {c['contrib']:+7.0f}{'  WORSE' if c['worse'] else ''}")
    print(f"GATE {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--folds", default="M2,M7")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--json", default="")
    ap.add_argument("--alias", default="", help="rename folds of a run, e.g. MJ=M1,MY=M7")
    a = ap.parse_args()
    folds = a.folds.split(",")
    base, new = load(a.base), load(a.new)
    for pair in filter(None, a.alias.split(",")):
        k, v = pair.split("=")
        base.loc[base.fold == k, "fold"] = v
        new.loc[new.fold == k, "fold"] = v
    lab = labels(sorted({int(f[1:]) for f in folds}))
    res = compare(base, new, lab, folds, a.reps)
    report(res)
    if a.json:
        Path(a.json).write_text(json.dumps({"base": a.base, "new": a.new, "folds": res}, indent=1))
