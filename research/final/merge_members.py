"""Seed-average refit kernels of one member specification into a single member prediction.

    python3 research/final/merge_members.py c7       # prc-f-c7a + prc-f-c7b       -> prc-f-c7
    python3 research/final/merge_members.py mlp7     # prc-f-mlp7a + prc-f-mlp7b   -> prc-f-mlp7
    python3 research/final/merge_members.py --out solutions/kernels/NAME/output SRC [SRC ...]

A source is a kernel slug under --root (default solutions/kernels; its output/ directory is used)
or a directory. A source whose output holds submission.parquet is one unit; one whose output holds
variant subdirectories (output/r2911/submission.parquet, ...) contributes each subdirectory as a
unit. Within a unit, the per-seed arrays refit_<k>.npy (unclipped, in the order of the unit's
submission.parquet ids, as solutions/core/train_main.py writes them) are used when present, so the
merge equals one run with all the seeds: mean over every seed, then the 60 s floor. A unit without
seed arrays enters as its submission.parquet weighted by its seed count (results.json
spec.refit_seeds, else 1).

Writes OUT/submission.parquet (MVT_ID_mvt float64 in template order, TAXITIME_SEC_mvt float64) and
OUT/merge_manifest.json (units, seeds, sha256 of every input, per-unit RMS from the merged mean,
output sha256). Refuses to overwrite a different existing output without --force.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from frame import C, FRAMES, ID, TARGET, sha256  # noqa: E402

PRESETS = {"c7": (["prc-f-c7a", "prc-f-c7b"], "prc-f-c7"),
           "mlp7": (["prc-f-mlp7a", "prc-f-mlp7b"], "prc-f-mlp7")}


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(C))
    except ValueError:
        return str(p)


def units_of(src: str, root: Path) -> list[Path]:
    p = Path(src)
    base = p if p.is_dir() else root / src / "output"
    if not base.is_dir():
        raise SystemExit(f"source {src}: no directory {base}")
    subs = sorted(q.parent for q in base.glob("*/submission.parquet"))
    if subs:
        if (base / "submission.parquet").exists():
            print(f"[{src}] has variant subdirectories AND a top-level submission.parquet; using the subdirectories")
        return subs
    if (base / "submission.parquet").exists():
        return [base]
    raise SystemExit(f"source {src}: no submission.parquet in {base} or its subdirectories")


def read_unit(u: Path) -> dict:
    sub = pd.read_parquet(u / "submission.parquet")
    ids = sub[ID].round().astype("int64").to_numpy()
    assert pd.Index(ids).is_unique, f"{u}: duplicate ids"
    npys = sorted(u.glob("refit_*.npy"), key=lambda q: int(q.stem.split("_")[1]))
    files = [u / "submission.parquet"] + npys + ([u / "results.json"] if (u / "results.json").exists() else [])
    if npys:
        raw = [np.load(q) for q in npys]
        arrs = [a.astype(float) for a in raw]
        for q, a in zip(npys, arrs):
            assert a.shape == (len(ids),), f"{q}: {a.shape} vs {len(ids)} ids"
        mean = np.mean(arrs, axis=0)
        ref = sub[TARGET].to_numpy(float)
        # the kernel averaged in the arrays' own precision: float32 seeds (xgboost) round at ~1 ulp
        tol = 1e-6 + (4 * np.finfo(np.float32).eps * np.abs(ref) if any(a.dtype == np.float32 for a in raw) else 0.0)
        gap = np.abs(np.clip(mean, 60, None) - ref)
        assert (gap <= tol).all(), f"{u}: submission.parquet is not the clipped mean of its refit_*.npy (max gap {gap.max()})"
        return {"dir": u, "ids": ids, "seeds": arrs, "weight": len(arrs), "level": "seed", "files": files}
    seeds = 1
    if (u / "results.json").exists():
        seeds = int(json.loads((u / "results.json").read_text()).get("spec", {}).get("refit_seeds", 1))
    return {"dir": u, "ids": ids, "seeds": [sub[TARGET].to_numpy(float)], "weight": seeds, "level": "submission",
            "files": files}


def seed_spread(units: list[dict], frame: str, merged: np.ndarray) -> dict:
    """Sanity of the seeds against each other (single seed arrays when the units have them, else the
    units): pairwise RMS, and each one's mean difference from the mean of the others by delay band
    (a seed that is off in level or in one band shows here)."""
    from frame import load_frame
    d = load_frame(frame).d.to_numpy()
    bands = {"d <= 1 h": d <= 3600, "1-4 h": (d > 3600) & (d <= 14400), "4-8 h": (d > 14400) & (d <= 28800),
             "> 8 h": d > 28800}
    if any(len(u["seeds"]) > 1 and u["level"] == "seed" for u in units):  # compare single seeds, not unit means
        vals, names = [], []
        for u in units:
            for k, a in enumerate(u["aligned_seeds"]):
                vals.append(np.clip(a, 60, None))
                names.append(f"{rel(u['dir'])}#seed{k}")
    else:
        vals = [np.clip(u["aligned_mean"], 60, None) for u in units]
        names = [rel(u["dir"]) for u in units]
    pair = {f"{names[i]} | {names[j]}": float(np.sqrt(np.mean((vals[i] - vals[j]) ** 2)))
            for i in range(len(vals)) for j in range(i + 1, len(vals))}
    by_band = {}
    for i, v in enumerate(vals):
        others = np.mean([w for j, w in enumerate(vals) if j != i], axis=0) if len(vals) > 1 else v
        by_band[names[i]] = {b: {"mean_diff": float((v - others)[m].mean()), "rms_diff": float(np.sqrt(((v - others)[m] ** 2).mean()))}
                             for b, m in bands.items()}
    print("pairwise RMS between units (s): min %.1f median %.1f max %.1f" % (min(pair.values()), np.median(list(pair.values())), max(pair.values()))
          if pair else "one unit")
    for nme, bb in by_band.items():
        print(f"   {nme}: mean diff vs the others by band " + "  ".join(f"{b} {x['mean_diff']:+.1f} (RMS {x['rms_diff']:.0f})" for b, x in bb.items()))
    return {"pairwise_rms": pair, "vs_others_by_band": by_band}


def merge(sources: list[str], out: Path, root: Path, frame: str = "final", force: bool = False) -> dict:
    tpl = pd.read_parquet(C / FRAMES[frame]["template"], columns=[ID])
    tid = tpl[ID].round().astype("int64").to_numpy()
    units = [read_unit(u) for s in sources for u in units_of(s, root)]
    total, acc, per_unit = 0.0, np.zeros(len(tid)), []
    for u in units:
        pos = pd.Series(np.arange(len(u["ids"])), index=u["ids"])
        missing = ~np.isin(tid, u["ids"])
        assert not missing.any(), f"{u['dir']}: {int(missing.sum())} template ids missing"
        extra = len(u["ids"]) - len(tid)
        assert extra == 0, f"{u['dir']}: {extra} ids beyond the {frame} template"
        idx = pos.reindex(tid).to_numpy()
        vals = [a[idx] for a in u["seeds"]]
        assert all(np.isfinite(v).all() for v in vals), f"{u['dir']}: non-finite predictions"
        unit_mean = np.mean(vals, axis=0)
        u["aligned_mean"] = unit_mean
        u["aligned_seeds"] = vals
        acc += unit_mean * u["weight"]
        total += u["weight"]
    merged = np.clip(acc / total, 60, None)
    spread = seed_spread(units, frame, merged)
    for u in units:
        per_unit.append({"dir": rel(u["dir"]), "level": u["level"], "seeds": len(u["seeds"]), "weight": u["weight"],
                         "rms_from_merged": float(np.sqrt(np.mean((np.clip(u["aligned_mean"], 60, None) - merged) ** 2))),
                         "inputs": {rel(f): sha256(f) for f in u["files"]}})
    out.mkdir(parents=True, exist_ok=True)
    target = out / "submission.parquet"
    df = pd.DataFrame({ID: tpl[ID].astype("float64").to_numpy(), TARGET: merged.astype("float64")})
    if target.exists() and not force:
        old = pd.read_parquet(target)
        if not (len(old) == len(df) and np.allclose(old[TARGET].to_numpy(), merged, atol=1e-9)):
            raise SystemExit(f"{target} exists with different values; use --force to replace it")
    df.to_parquet(target, index=False)
    man = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "command": " ".join([Path(sys.executable).name] + sys.argv), "frame": frame, "rows": len(df),
           "units": per_unit, "total_seeds": total, "seed_spread": spread, "output": {rel(target): sha256(target)},
           "mean": float(merged.mean()), "max": float(merged.max())}
    (out / "merge_manifest.json").write_text(json.dumps(man, indent=1))
    print(f"-> {rel(target)}: {len(df):,} rows from {len(units)} units / {total:g} seeds; mean {merged.mean():.1f} s")
    for u in per_unit:
        print(f"   {u['dir']}: {u['seeds']} {u['level']} arrays, weight {u['weight']:g}, RMS from merged {u['rms_from_merged']:.2f} s")
    return man


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sources", nargs="+", help="a preset (c7, mlp7) or kernel slugs / directories")
    ap.add_argument("--out", help="output directory (required unless a preset is given)")
    ap.add_argument("--root", default=str(C / "solutions" / "kernels"))
    ap.add_argument("--frame", default="final", choices=["final", "board"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if len(a.sources) == 1 and a.sources[0] in PRESETS:
        srcs, name = PRESETS[a.sources[0]]
        out = Path(a.out) if a.out else root / name / "output"
    else:
        if not a.out:
            ap.error("--out is required without a preset")
        srcs, out = a.sources, Path(a.out)
    merge(srcs, out, root, a.frame, a.force)


if __name__ == "__main__":
    main()
