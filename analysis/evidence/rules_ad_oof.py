"""The cap, rules (a) and (d), and the rejected rule (b), on the 2025 12-month-out predictions of c6.

    python3 analysis/evidence/rules_ad_oof.py ROOT

ROOT holds the organiser training files in data/ and research/cache/oof_mo12_c6.parquet (c6's
month-out predictions; sha256 in docs/REPRODUCE.md). Units: change in full, untrimmed MSE over all
2,085,047 departures of 2025 (change in SSE / N), 95% airport-day block bootstrap interval (2,000
draws, seed 0), "months worse" = months whose change in SSE / N is above +0.5. "nw" leaves the 2025
date-wrap rows unchanged (y > d + 600 and y > 80,000 s: the right time of day stamped on the
scheduled date, a mode the 2026 files do not show).

cap: y_hat <= d where d > 2 h (applied to c6 directly).
(a) NM join window: a matched departure's y lies in takeoff - LOBT +- 3,606 s (no exception in 2025).
(d) LEMD and LEBL: y <= 3,604 s (the 2025 maxima were 3,598 and 3,604 s).
(b) rejected: y >= takeoff - AOBT_3 - 3,600 where takeoff - AOBT_3 > 1,800 s.
"""
import glob
import sys

import numpy as np
import pandas as pd

C = sys.argv[1]
D = f"{C}/data/09-22-2026-08-13-30_files_list"
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "FLIGHT_ID_mvt", "LOBT_flt",
        "AOBT_3_flt", "TAXITIME_SEC_mvt"]
d = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{D}/training_2025-*.parquet"))])
d = d[d.PHASE_mvt == "DEP"].copy()
for c in ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]:
    d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
d["id"] = d.MVT_ID_mvt.astype("int64")
oof = pd.read_parquet(f"{C}/research/cache/oof_mo12_c6.parquet")
oof["id"] = oof["id"].astype("int64")
d = d.merge(oof, on="id")
y = d.TAXITIME_SEC_mvt.to_numpy(float)
p = d.prediction.to_numpy(float)
ds = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
a3 = (d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds().to_numpy()
ml = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds().to_numpy()
nm = d.FLIGHT_ID_mvt.notna().to_numpy()
apt = d.ADEP_mvt.to_numpy()
mon = d.MVT_TIME_UTC_mvt.dt.month.to_numpy()
block = apt + "|" + d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy()
wrap = (y > ds + 600) & (y > 80000)
N = len(y)
print(f"2025 departures with a c6 month-out prediction: {N:,}; c6 MSE {((p - y) ** 2).mean():.1f}; "
      f"date-wrap rows: {int(wrap.sum())}")


def boot(dd, n, draws=2000):
    codes, uniq = pd.factorize(block)
    sums = np.bincount(codes, weights=dd, minlength=len(uniq))
    rng = np.random.default_rng(0)
    b = sums[rng.integers(0, len(uniq), size=(draws, len(uniq)))].sum(1) / n
    return np.percentile(b, [2.5, 97.5])


def rep(name, q, base):
    for tag, keep in (("all", np.ones(N, bool)), ("nw", ~wrap)):
        qq = np.where(keep, q, base)
        dd = (qq - y) ** 2 - (base - y) ** 2
        lo, hi = boot(dd, N)
        bym = pd.Series(dd).groupby(mon).sum() / N
        sel = {m: round(float(bym.get(m, 0.0)), 1) for m in (1, 2, 6, 7)}
        print(f"{name:42s} {tag:3s} dMSE {dd.sum() / N:+8.1f} [{lo:+8.1f}, {hi:+8.1f}]  months worse "
              f"{int((bym > 0.5).sum())}/12  Jan/Feb/Jun/Jul {sel}  rows moved {int((qq != base).sum())}")


long2 = ds > 7200
print(f"departures with d > 2 h: {int(long2.sum()):,}; with y > d + 120 s: {int((long2 & (y > ds + 120)).sum())} "
      f"(of which date-wrap: {int((long2 & (y > ds + 120) & wrap).sum())})")
qcap = np.where(long2, np.minimum(p, ds), p)
rep("cap y_hat <= d where d > 2 h", qcap, p)
ok = nm & np.isfinite(ml)
hi_a, lo_a = ml + 3606, ml - 3606
qa = p.copy()
qa[ok & (qa > hi_a)] = hi_a[ok & (qa > hi_a)]
qa[ok & (qa < lo_a)] = np.maximum(lo_a[ok & (qa < lo_a)], 60)
rep("(a) NM join window", qa, p)
cp = np.isin(apt, ["LEMD", "LEBL"])
qd = qa.copy()
qd[cp] = np.minimum(qd[cp], 3604)
rep("(d) LEMD/LEBL cap, on top of (a)", qd, qa)
rep("(a)+(d)", qd, p)
fl = nm & np.isfinite(a3) & (a3 > 1800)
qb = qa.copy()
qb[fl] = np.maximum(qb[fl], a3[fl] - 3600)
rep("(b) rejected AOBT_3 floor, on top of (a)", qb, qa)
print("matched departures outside the NM window:", int((ok & ((y > hi_a) | (y < lo_a))).sum()), "of", int(ok.sum()))
print("LEMD / LEBL 2025 maxima:", {a: float(y[apt == a].max()) for a in ("LEMD", "LEBL")})
