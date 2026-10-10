"""One 4-fold month-out OOF for the fs8 c7 member (seed 3101, the prc-f-c7a-mo spec on the fs8 store).

    python3 research/fgap/combine_c7_mo.py

Folds and their sources (all c7(3101) via build_members.fold_spec, fixed rounds 9600/2900, one T4 each):
  M1, M6  miscusi/prc-f8-c7a-mo                 (store prc-f-fs8 v2)
  M2      miscusi/prc-g8-c7-mo, variant new     (store prc-f-fs8 v1; that fold finished before the variant's OOM)
  M7      miscusi/prc-g8-c7-mo-new, variant new (store prc-f-fs8 v2)
v1 and v2 differ only in four stand-traffic columns, v2 = v1 / 12 on the 2025 frame (12 months at every airport),
a constant rescaling that leaves CatBoost's quantile borders, and so the model, unchanged.
Writes solutions/kernels/prc-f8-c7a-mo/output/oof.csv (id, fold, prediction: train_main's schema) + oof.parquet
and oof_sources.json; the kernel's own two-fold oof.csv/parquet are kept as oof_kernel_M1M6.*.
Checks: each fold's id set equals prc-f-c7a-mo's, ids unique, predictions finite.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

C = Path(__file__).resolve().parents[2]
K = C / "solutions" / "kernels"
OUT = K / "prc-f8-c7a-mo" / "output"
SOURCES = {"M1": K / "prc-f8-c7a-mo" / "output" / "oof_M1.csv",
           "M6": K / "prc-f8-c7a-mo" / "output" / "oof_M6.csv",
           "M2": K / "prc-g8-c7-mo" / "output" / "new" / "oof_M2.csv",
           "M7": K / "prc-g8-c7-mo-new" / "output" / "new" / "oof_M7.csv"}
STORE = {"M1": "prc-f-fs8 v2", "M6": "prc-f-fs8 v2", "M2": "prc-f-fs8 v1", "M7": "prc-f-fs8 v2"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ref = pd.read_csv(K / "prc-f-c7a-mo" / "output" / "oof.csv")
parts = []
for f in ("M1", "M2", "M6", "M7"):
    o = pd.read_csv(SOURCES[f])
    assert set(o.fold) == {f}, (f, set(o.fold))
    assert o.id.is_unique and np.isfinite(o.prediction).all(), f
    want = set(ref.loc[ref.fold == f, "id"])
    assert set(o.id) == want, f"{f}: id set differs from prc-f-c7a-mo ({len(set(o.id) ^ want)} ids)"
    parts.append(o[["id", "fold", "prediction"]])
oof = pd.concat(parts, ignore_index=True)
for name in ("oof.csv", "oof.parquet"):
    p = OUT / name
    kept = OUT / name.replace("oof.", "oof_kernel_M1M6.")
    if p.exists() and not kept.exists():
        old = pd.read_csv(p) if name.endswith(".csv") else pd.read_parquet(p)
        if set(old.fold) == {"M1", "M6"}:
            p.rename(kept)
oof.to_csv(OUT / "oof.csv", index=False)
oof.to_parquet(OUT / "oof.parquet", index=False)
(OUT / "oof_sources.json").write_text(json.dumps(
    {"member": "c7 fs8, seed 3101", "folds": {f: {"file": str(SOURCES[f].relative_to(C)), "sha256": sha(SOURCES[f]),
                                                  "store": STORE[f], "rows": int((oof.fold == f).sum())} for f in SOURCES},
     "oof_sha256": sha(OUT / "oof.csv")}, indent=1))
print(oof.fold.value_counts().to_dict(), "->", OUT / "oof.csv")
