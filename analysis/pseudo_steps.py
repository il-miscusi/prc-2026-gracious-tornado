"""Per-step scores of one 2025 pseudo-final run: what each assembly step adds, in shipping order.

    python3 analysis/pseudo_steps.py ROOT NAME_MODE [--out analysis/results/pseudo_steps_NAME_MODE.json]

ROOT is a tree with the organiser files in data/ and the pseudo-final rows file
submissions/final/pseudo2025/NAME_MODE.parquet, written by
`python3 research/final/pseudo_final.py run research/final/configs/NAME.json --weights MODE`.
The rows file holds, per 2025 departure of January, February, June and July, the label y and the
prediction after every step (fill, overlay, cap, level, post layers, rules); its sha256 is recorded
in the run's JSON summary (research/final/pseudo/NAME_MODE.json, key rows_sha256).

For each step the script prints the pooled MSE over all rows and over the rows that are not 2025
date-wrap records ("nw": the right time of day stamped on the scheduled date, a recording mode the
2026 files do not show; research/final/pseudo_final.row_flags), and the change from the previous
step with a 95% airport-day block bootstrap interval (2,000 draws, seed 0: pseudo_final.boot).
Month columns are in month units (change in SSE / rows of that month).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STEPS = ["fill", "overlay", "cap", "level", "post1_lv_rules", "post2_recovered", "post3_daily",
         "post4_recovered", "pred"]
LABEL = {"fill": "member stack, rows past 4 h, LIRF fill", "overlay": "LIRF no-ADS-B overlay",
         "cap": "cap at d past 2 h", "level": "monthly level (L1c)", "post1_lv_rules": "rules (a), (d)",
         "post2_recovered": "recovered-orphan window (L0e)", "post3_daily": "daily reconciliation (L1a-v2)",
         "post4_recovered": "L0e again", "pred": "final rules and 60 s floor"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("run", help="NAME_MODE, e.g. f2x5_l1av2_pseudo2025_frozen")
    ap.add_argument("--out")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    sys.path.insert(0, str(root / "research" / "final"))
    sys.path.insert(0, str(root / "research" / "tail"))
    import pseudo_final as pf  # noqa: E402

    rows = root / "submissions" / "final" / "pseudo2025" / f"{a.run}.parquet"
    t = pd.read_parquet(rows)
    flags = pf.row_flags(sorted(int(m) for m in t.month.unique())).reindex(t.id.to_numpy())
    wrap = flags.wrap.to_numpy().astype(bool)
    y = t.y.to_numpy(float)
    block = (t.apt + "|" + t.day).to_numpy()
    month = t.month.to_numpy()
    out = {"run": a.run, "rows_file": f"submissions/final/pseudo2025/{a.run}.parquet",
           "rows_sha256": hashlib.sha256(rows.read_bytes()).hexdigest(), "rows": int(len(t)),
           "date_wrap_rows": int(wrap.sum()), "steps": []}
    prev = None
    nw = ~wrap
    grp = t.group.to_numpy()
    steps = STEPS[:-1] + (["post5_events"] if "post5_events" in t else []) + STEPS[-1:]
    for s in steps:
        e = (y - t[s].to_numpy(float)) ** 2
        rec = {"step": s, "label": LABEL.get(s, "grade-A pushback blend"), "mse_all": float(e.mean()), "mse_nw": float(e[nw].mean()),
               "mse_nw_month": {int(m): float(e[nw & (month == m)].mean()) for m in sorted(set(month))}}
        if prev is not None:
            dd = (e - prev)[nw]
            lo, hi = pf.boot(dd, block[nw], int(nw.sum()))
            rec["delta_all"] = float((e - prev).mean())
            rec["delta_nw"] = float(dd.mean())
            rec["delta_nw_ci95"] = [lo, hi]
            rec["delta_nw_month"] = {int(m): float((e - prev)[nw & (month == m)].mean()) for m in sorted(set(month))}
            rec["rows_changed"] = int((np.abs(t[s].to_numpy(float) - t[steps[steps.index(s) - 1]].to_numpy(float)) > 1e-9).sum())
        if prev is not None:  # where the step acts: rows moved and change per row group (nw, pooled units)
            before = t[steps[steps.index(s) - 1]].to_numpy(float)
            mv = np.abs(t[s].to_numpy(float) - before)
            rec["by_group"] = {str(g): {"rows": int((grp == g).sum()), "rows_changed": int(((grp == g) & (mv > 1e-9)).sum()),
                                        "mean_abs_change_s": float(mv[grp == g].mean()),
                                        "delta_nw": float(((e - prev) * ((grp == g) & nw)).sum() / nw.sum())}
                               for g in sorted(set(grp))}
        out["steps"].append(rec)
        prev = e
        d = rec.get("delta_nw")
        print(f"{s:16s} MSE all {rec['mse_all']:10.1f}  nw {rec['mse_nw']:10.1f}"
              + (f"  step nw {d:+9.1f} [{rec['delta_nw_ci95'][0]:+.1f}, {rec['delta_nw_ci95'][1]:+.1f}]"
                 f"  all {rec['delta_all']:+9.1f}  rows {rec['rows_changed']:,}" if d is not None else ""))
    # anatomy of the final prediction's nw error: share by delay band, and how much of each month's SSE
    # the largest single rows carry (aggregates only; no row is listed)
    e = (y - t["pred"].to_numpy(float)) ** 2
    band = t.band.to_numpy()
    anat = {}
    for m in sorted(set(month)):
        k = nw & (month == m)
        em = np.sort(e[k])[::-1]
        anat[int(m)] = {"rows": int(k.sum()), "mse": float(em.mean()),
                        "band_mse_contrib": {str(b): float(e[k & (band == b)].sum() / k.sum()) for b in sorted(set(band))},
                        "top10_share": float(em[:10].sum() / em.sum()), "top100_share": float(em[:100].sum() / em.sum()),
                        "mean_residual_1_4h_by_airport": {str(ap): [int(n), float(r)] for ap, n, r in (
                            (ap, ((k & (band == "1-4 h")) & (t.apt.to_numpy() == ap)).sum(),
                             (y - t["pred"].to_numpy(float))[(k & (band == "1-4 h")) & (t.apt.to_numpy() == ap)].mean())
                            for ap in sorted(set(t.apt)))}}
        print(f"M{m} nw MSE {anat[int(m)]['mse']:9.1f}  top-10 rows {anat[int(m)]['top10_share']:.1%}  top-100 {anat[int(m)]['top100_share']:.1%}  "
              + "  ".join(f"{b} {v:,.0f}" for b, v in anat[int(m)]["band_mse_contrib"].items()))
    out["anatomy_nw"] = anat
    path = Path(a.out) if a.out else Path(__file__).resolve().parent / "results" / f"pseudo_steps_{a.run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
