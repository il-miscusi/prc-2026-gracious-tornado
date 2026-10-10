"""Regenerate the 2026 short-band meta predictions from the kept booster, without training.

    python3 research/final/meta/reproduce.py [--arm G1] [--run solutions/kernels/prc-meta-gbm/output]
    python3 research/final/meta/reproduce.py --run solutions/kernels/prc-meta-gbm-f4/output --frames submissions/final/meta/f4

Reads the kernel's report.json (base weights, alpha, rounds), the saved booster <arm>_short.txt and
submissions/final/meta/meta2026.parquet (research/final/meta/build_frame.py --frame final), recomputes
base + alpha * booster(features) on the d <= 1 h rows with the kernel's own feature code, and compares
with the kernel's final_<arm>.parquet. Prints the max absolute difference. The base weights are refitted
from meta2025.parquet exactly as the kernel does (the report keeps them rounded) and checked against it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

C = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="G1")
    ap.add_argument("--run", default="solutions/kernels/prc-meta-gbm/output")
    ap.add_argument("--frames", default="submissions/final/meta", help="directory of meta2026.parquet (f4 re-fit: submissions/final/meta/f4)")
    a = ap.parse_args()
    run = C / a.run
    rep = json.loads((run / "report.json").read_text())
    # the kernel module reads meta_spec.json (recursively) under META_INPUT: give it this run's spec only
    stage = Path(tempfile.mkdtemp(prefix="meta-repro-"))
    if (C / a.frames / "meta_spec.json").exists():
        shutil.copy2(C / a.frames / "meta_spec.json", stage / "meta_spec.json")
    os.environ["META_INPUT"] = str(stage)
    spec = importlib.util.spec_from_file_location("kmeta", C / "research/final/meta/kernels/prc-meta-gbm/kernel.py")
    km = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(km)
    f = pd.read_parquet(C / a.frames / "meta2026.parquet")
    fin = rep[a.arm]["short|final"]
    rows = km.seg_rows(f, "short")
    # base weights: the kernel's own NNLS on all four months (report.json stores them rounded to 6 decimals,
    # which can flip a split on `base`/`base_d`)
    t = pd.read_parquet(C / a.frames / "meta2025.parquet")
    w, _ = km.fit_base(t, "short", km.seg_rows(t, "short") & t.boardlike.to_numpy())
    assert np.abs(w - np.array(fin["base_w"])).max() < 1e-6, (w, fin["base_w"])
    base = f[[f"p_{m}" for m in km.SEGS["short"][2]]].to_numpy(float) @ w
    bst = lgb.Booster(model_file=str(run / f"{a.arm}_short.txt"))
    mine = base[rows] + fin["alpha"] * bst.predict(km.features(f, a.arm, "short", base)[rows])
    ref = pd.read_parquet(run / f"final_{a.arm}.parquet")
    assert (ref.MVT_ID_mvt.round().astype("int64").to_numpy() == f.id.to_numpy()).all()
    diff = np.abs(mine - ref.TAXITIME_SEC_mvt.to_numpy()[rows])
    print(f"{a.arm}: {int(rows.sum()):,} d <= 1 h rows, rounds {fin['rounds']}, alpha {fin['alpha']}, "
          f"trees {bst.num_trees()}, max |diff| {diff.max():.2e} s (lightgbm kernel {rep['lightgbm']}, local {lgb.__version__})")


if __name__ == "__main__":
    main()
