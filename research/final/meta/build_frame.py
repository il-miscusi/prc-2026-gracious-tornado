"""Meta-learner training frame: 2025 Jan/Feb/Jun/Jul departures, month-out member predictions, context.

    python3 research/final/meta/build_frame.py [--frame 2025|final] [--spec research/final/meta/specs/NAME.json]

One row per departure of the four pseudo-final months (671,292 rows). Columns:
  id, month, mday, day, apt, y, d, boardlike, wrap (2025 date-wrap rows, research/final/pseudo_final.row_flags)
  p_<member>   month-out prediction: row of month m predicted by the fold trained on 2025 without m
               (same files and same loader as research/final/configs/f2x5_pseudo2025.json)
  context      nm, unmatched, lirf_orphan, d_lobt, d_aobt3, hour (takeoff UTC hour, fractional),
               sched_hour, wday, a3_found, adsb_found (research/final/inputs/2025/cover.parquet),
               a3_grade, a3_obs_leave (inputs/2025/phys.parquet)
Labels are read only for `y`; every context column is computed from the label-free frame
(research/final/frame.load_training_frame blanks the off-block time).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
sys.path.insert(0, str(FINAL))
from frame import C, load_training_frame, sha256  # noqa: E402
from pseudo_final import boardlike, load_mo, row_flags  # noqa: E402

MONTHS = [1, 2, 6, 7]
MEMBERS = {
    "c7": "solutions/kernels/prc-f-c7a-mo/output",
    "cs7": "solutions/kernels/prc-f-cs7-mo/output",
    "mlp7": "solutions/kernels/prc-f-mlp7a-mo/output",
    "c6s": "solutions/kernels/prc-f-c6s-mo/output",
    "c5s": "solutions/kernels/prc-f-c5s-mo/output",
    "x5": "solutions/kernels/prc-f-x5-mo/output",
    "k4": "solutions/kernels/prc-f-k4-mo/output",
}
COVER = "research/final/inputs/2025/cover.parquet"
PHYS = "research/final/inputs/2025/phys.parquet"


def by_id(path: str, ids: np.ndarray, cols: list[str]) -> pd.DataFrame:
    t = pd.read_parquet(C / path)
    t.index = t["MVT_ID_mvt"].round().astype("int64")
    assert t.index.is_unique, path
    miss = ~np.isin(ids, t.index.to_numpy())
    assert not miss.any(), f"{path}: {int(miss.sum())} ids missing"
    return t.reindex(ids)[cols]


def context(fr: pd.DataFrame) -> pd.DataFrame:
    """Context columns from a frame.py frame (2025 training frame or the 2026 final frame)."""
    to, sc = fr.MVT_TIME_UTC_mvt, fr.SCHED_TIME_UTC_mvt
    return pd.DataFrame({
        "nm": fr.nm.to_numpy().astype(np.int8),
        "unmatched": fr.unmatched.to_numpy().astype(np.int8),
        "lirf_orphan": fr.lirf_orphan.to_numpy().astype(np.int8),
        "d_lobt": fr.d_lobt.to_numpy(dtype=float),
        "d_aobt3": fr.d_aobt3.to_numpy(dtype=float),
        "hour": (to.dt.hour + to.dt.minute / 60).to_numpy(dtype=float),
        "sched_hour": (sc.dt.hour + sc.dt.minute / 60).to_numpy(dtype=float),
        "wday": to.dt.weekday.to_numpy().astype(np.int8),
    })


def build(out: Path) -> dict:
    fr, y = load_training_frame(MONTHS)
    ids = fr.id.to_numpy()
    t = pd.DataFrame({"id": ids, "month": fr.month.to_numpy(), "mday": fr.mday.to_numpy(), "day": fr.day.to_numpy(),
                      "apt": fr.apt.to_numpy(), "y": y, "d": fr.d.to_numpy(dtype=float),
                      "boardlike": boardlike(fr, y)})
    t["wrap"] = row_flags(MONTHS).reindex(ids)["wrap"].to_numpy().astype(bool)
    inputs = dict(fr.attrs["inputs"])
    for name, spec in MEMBERS.items():
        p, files = load_mo(spec, fr)
        t[f"p_{name}"] = p
        inputs.update({str(f.relative_to(C)): sha256(f) for f in files})
    t = pd.concat([t, context(fr)], axis=1)
    cov = by_id(COVER, ids, ["a3_found", "adsb_found"])
    t["a3_found"] = cov.a3_found.fillna(0).to_numpy().astype(np.int8)
    t["adsb_found"] = cov.adsb_found.fillna(0).to_numpy().astype(np.int8)
    ph = by_id(PHYS, ids, ["a3_grade", "a3_obs_leave"])
    t["a3_grade"] = ph.a3_grade.to_numpy(dtype=float)
    t["a3_obs_leave"] = ph.a3_obs_leave.to_numpy(dtype=float)
    for p in (COVER, PHYS):
        inputs[p] = sha256(C / p)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(out, index=False)
    man = {"rows": len(t), "members": MEMBERS, "inputs": inputs, "out": str(out.relative_to(C)), "sha256": sha256(out)}
    out.with_suffix(".json").write_text(json.dumps(man, indent=1))
    return man


REFITS = {name: f"solutions/kernels/prc-f-{name}/output" for name in MEMBERS}
COVER_FINAL = "research/final/inputs/final/cover.parquet"
PHYS_FINAL = "research/final/inputs/final/phys.parquet"


def build_final(out: Path) -> dict:
    """Same columns on the 2026 final frame (670,790 departures, final_submitting order), with the
    refit members' predictions (configs/f2_final.json's `pred` runs; x5 = the pinned prc-f-x5)."""
    from frame import load_frame
    from stacking import member_pred, pred_file
    fr = load_frame("final")
    ids = fr.id.to_numpy()
    t = pd.DataFrame({"id": ids, "month": fr.month.to_numpy(), "mday": fr.mday.to_numpy(), "day": fr.day.to_numpy(),
                      "apt": fr.apt.to_numpy(), "d": fr.d.to_numpy(dtype=float)})
    inputs = dict(fr.attrs["inputs"])
    for name, spec in REFITS.items():
        t[f"p_{name}"] = member_pred(spec, ids)
        inputs[str(pred_file(spec).relative_to(C))] = sha256(pred_file(spec))
    t = pd.concat([t, context(fr)], axis=1)
    cov = by_id(COVER_FINAL, ids, ["a3_found", "adsb_found"])
    t["a3_found"] = cov.a3_found.fillna(0).to_numpy().astype(np.int8)
    t["adsb_found"] = cov.adsb_found.fillna(0).to_numpy().astype(np.int8)
    ph = by_id(PHYS_FINAL, ids, ["a3_grade", "a3_obs_leave"])
    t["a3_grade"] = ph.a3_grade.to_numpy(dtype=float)
    t["a3_obs_leave"] = ph.a3_obs_leave.to_numpy(dtype=float)
    for p in (COVER_FINAL, PHYS_FINAL):
        inputs[p] = sha256(C / p)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(out, index=False)
    man = {"rows": len(t), "members": REFITS, "inputs": inputs, "out": str(out.relative_to(C)), "sha256": sha256(out)}
    out.with_suffix(".json").write_text(json.dumps(man, indent=1))
    return man


def use_spec(path: str, out_dir: str = "submissions/final/meta") -> None:
    """Swap the member set for a re-fit (e.g. fs8 refits): research/final/meta/specs/*.json with
    "mo" {name: month-out dir or list of dirs}, "refit" {name: refit dir}, and the kernel keys
    ("segments", "meta_features", "fit_segments", "arms"). The spec is copied next to the frames as
    meta_spec.json, which publish_dataset.py ships and the kernel reads."""
    spec = json.loads((C / path).read_text())
    MEMBERS.clear()
    MEMBERS.update(spec["mo"])
    REFITS.clear()
    REFITS.update(spec["refit"])
    dst = C / out_dir / "meta_spec.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(spec, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frame", choices=["2025", "final"], default="2025")
    ap.add_argument("--out")
    ap.add_argument("--spec", help="member spec JSON (research/final/meta/specs/); default: the v1 members above")
    ap.add_argument("--out-dir", default="submissions/final/meta",
                    help="directory for meta2025/meta2026.parquet and meta_spec.json (one per member set)")
    a = ap.parse_args()
    if a.spec:
        use_spec(a.spec, a.out_dir)
    if a.frame == "2025":
        man = build(C / (a.out or f"{a.out_dir}/meta2025.parquet"))
    else:
        man = build_final(C / (a.out or f"{a.out_dir}/meta2026.parquet"))
    print(f"{man['rows']:,} rows -> {man['out']} ({man['sha256'][:12]})")


if __name__ == "__main__":
    main()
