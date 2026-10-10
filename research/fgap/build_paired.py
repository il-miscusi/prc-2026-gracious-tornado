"""Paired c7 month-out test of the fs8 feature group (feature-gap lane, 2026-10-09).

    python3 research/fgap/build_paired.py --meta FS_META_JSON [--slug prc-g8-c7-mo] [--folds M2,M7]
    python3 research/fgap/build_paired.py --meta FS_META_JSON --slug prc-g8-c7-mo-new --arms new --gpus 1 --folds M7

--arms keeps only the named variants (one at a time when --gpus 1): prc-g8-c7-mo v1 lost "new" to a host OOM
(exit -9 at 32.9 GB) while "base" ran beside it on the other T4, after new's M2 fold had finished.

One Kaggle GPU session (T4 x2) runs variants two at a time, one per GPU:
  base / new        the c7a month-out fold spec (build_members.fold_spec(c7(3101))): same seed, same fixed
                    fold rounds, same store files; base drops every g8_* column (so its matrix is exactly the
                    fs7 c7 matrix), new keeps them (or only the groups named by --groups)
  base_lK / new_lK  long head only (SPEC only=long), seeds 3101+K: the long rows (d > 1 h) are a seed lottery
                    (one seed + rounds swap moved M7 by -3,655 MSE, -3,050 of it from 873 rows past 4 h), so
                    the long head is compared on a 3-seed average, as the deployed c7a+c7b average it
Evaluate with research/fgap/combine_paired.py, then research/fgap/paired_eval.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parents[1]
sys.path.insert(0, str(C / "solutions" / "final"))
import build_members as bm  # noqa: E402

BUILD = C / "solutions" / "core" / "build_kernel.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="the fs8 store's fs_meta.json (fs8_columns)")
    ap.add_argument("--slug", default="prc-g8-c7-mo")
    ap.add_argument("--fs", default="prc-f-fs8")
    ap.add_argument("--folds", default="M2,M7")
    ap.add_argument("--long-seeds", type=int, default=2, help="extra long-only seeds per arm")
    ap.add_argument("--groups", default="", help="comma list of g8 groups to keep in 'new' (p,l,n,q,a,s); default all")
    ap.add_argument("--arms", default="", help="comma list of variants to run (e.g. new); default all")
    ap.add_argument("--gpus", type=int, default=2, help="1: one variant at a time on one T4 (host memory)")
    ap.add_argument("--seed", type=int, default=3101, help="seed of base/new; long-only seeds are seed+1..seed+K")
    a = ap.parse_args()
    g8 = json.loads(Path(a.meta).read_text())["fs8_columns"]
    keep = g8 if not a.groups else [c for c in g8 if c.split("_")[1] in a.groups.split(",")]
    folds = a.folds.split(",")
    spec = bm.fold_spec(bm.c7(a.seed), folds)
    base_drop = spec["drop_features"] + g8
    new_drop = spec["drop_features"] + [c for c in g8 if c not in keep]
    params = spec["params"]
    variants = {"base": {"drop_features": base_drop}, "new": {"drop_features": new_drop}}
    for k in range(1, a.long_seeds + 1):
        p = {**params, "random_seed": params["random_seed"] + k}
        variants[f"base_l{k}"] = {"drop_features": base_drop, "only": "long", "params": p}
        variants[f"new_l{k}"] = {"drop_features": new_drop, "only": "long", "params": p}
    if a.arms:
        variants = {k: v for k, v in variants.items() if k in a.arms.split(",")}
        assert variants, "no variant left"
    spec = {**spec, "gpus": a.gpus, "max_parallel": a.gpus, "variants": variants}
    cmd = [sys.executable, str(BUILD), "train", a.slug, "--fs", a.fs, "--fs-owner", bm.PRIMARY, "--owner", bm.FALLBACK,
           "--accel", "gpu", "--spec", json.dumps(spec)]
    out = subprocess.run(cmd, cwd=C, capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(out.stderr)
    folder = C / "solutions" / "kernels" / a.slug
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    meta["enable_internet"] = True
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    (folder / "member.json").write_text(json.dumps({"test": "fs8 paired month-out", "kept_g8": keep,
                                                    "spec": {k: v for k, v in spec.items() if k != "fs_urls"}}, indent=1))
    print(folder, f"{len(keep)} of {len(g8)} g8 columns kept; variants {list(variants)}")


if __name__ == "__main__":
    main()
