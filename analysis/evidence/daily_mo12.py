"""Monthly level (L1c) and daily reconciliation (L1a) on the 2025 12-month-out predictions of c6,
in the shipped order: c6 -> L1c -> rules (a), (d) -> recovered-orphan window (L0e) -> L1a -> rules
(a), (d) and L0e again.

    python3 analysis/evidence/daily_mo12.py ROOT [--out analysis/results/daily_mo12.json]

ROOT holds the organiser files in data/, research/cache/oof_mo12_c6.parquet and the EUROCONTROL
files in research/pru/ (or the ten-airport extracts in research/final/levers/artifacts/). Every
fitted quantity is fitted without the scored month: the level's composition offset
(daily.level_calibrated_c6) and L1a's parameters (daily.month_out refits them on the other eleven
months of the same chain). Units: change in full, untrimmed MSE over all 2,085,047 departures of
2025 (change in SSE / N), 95% airport-day block bootstrap interval, months worse = months whose
change in SSE / N is above +0.5 (research/final/levers/lv_common.score). "nw" leaves the 13 date-wrap
rows of 2025 unchanged (y > d + 600 and y > 80,000 s). Takes about ten minutes on one core.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--out")
a = ap.parse_args()
root = Path(a.root).resolve()
sys.path.insert(0, str(root / "research" / "final" / "levers"))
import lv_common as common  # noqa: E402
import daily  # noqa: E402
import recovered  # noqa: E402
import lv_rules  # noqa: E402

dep = common.load_2025()
arr = common.load_2025(arrivals=True)
y, d = dep["y"].to_numpy(float), dep["d"].to_numpy(float)
c6 = dep["p"].to_numpy(float)
wrap = (y > d + 600) & (y > 80000)
mlr = recovered.recovered_ml(dep, arr)
sel = np.isin(dep["apt"].to_numpy(), list(recovered.AIRPORTS)) & np.isfinite(mlr)
forbid = (np.where(sel, mlr - recovered.JOIN, np.nan), np.where(sel, mlr + recovered.JOIN, np.nan))


def bounds(p):
    return recovered.rule(dep, lv_rules.apply(dep, p, rules="ad"), mlr)


lev = daily.level_calibrated_c6(dep)
base = bounds(lev)
pru = daily.load_pru(2025)
l1a = bounds(daily.month_out(dep, base, pru, daily.GATE, forbid=forbid))

res = {}
res["L1c vs c6"] = common.score("monthly level (L1c) vs c6", dep, lev, c6)
res["L1c vs c6, nw"] = common.score("monthly level (L1c) vs c6, date-wrap rows unchanged", dep, np.where(wrap, c6, lev), c6)
res["rules (a), (d) + L0e vs L1c, nw"] = common.score("rules (a), (d) and L0e after L1c, date-wrap rows unchanged", dep,
                                                      np.where(wrap, lev, base), lev)
res["L1a vs base"] = common.score("daily layer (L1a) after L1c, rules (a), (d), L0e", dep, l1a, base)
res["L1a vs base, nw"] = common.score("daily layer (L1a), date-wrap rows unchanged", dep, np.where(wrap, base, l1a), base)
apt = dep["apt"].to_numpy()
for x in sorted(set(apt)):
    r = common.score(f"   L1a {x} only, nw", dep, np.where((apt == x) & ~wrap, l1a, base), base, verbose=False)
    res[f"L1a {x}, nw"] = r
    print(f"   L1a {x} nw: {r['dmse']:+.1f} [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] months worse {r['months_worse']}/12")
out = Path(a.out) if a.out else Path(__file__).resolve().parents[1] / "results" / "daily_mo12.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=1, default=float) + "\n")
print(f"-> {out}")
