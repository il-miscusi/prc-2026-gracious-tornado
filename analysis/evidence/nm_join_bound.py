import sys, glob, numpy as np, pandas as pd
C = sys.argv[1]; D = f"{C}/data/09-22-2026-08-13-30_files_list"
cols = ["MVT_ID_mvt","PHASE_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","TAXITIME_SEC_mvt","FLIGHT_ID_mvt","LOBT_flt","AOBT_3_flt","EOBT_1_flt","IOBT_flt"]
d = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{D}/training_2025-*.parquet"))]); d = d[d.PHASE_mvt=="DEP"].copy()
for c in ["MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","LOBT_flt","AOBT_3_flt","EOBT_1_flt","IOBT_flt"]:
    d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
m = d.FLIGHT_ID_mvt.notna()
bl = (d.BLOCK_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds()
print("matched rows", m.sum(), "unmatched", (~m).sum())
x = bl[m].dropna()
print("BLOCK-LOBT on matched rows: min %.0f max %.0f; share |.|>3606: %.6f; quantiles:" % (x.min(), x.max(), (x.abs()>3606).mean()), x.quantile([.0001,.001,.01,.5,.99,.999,.9999]).round(0).to_dict())
for c in ["AOBT_3_flt","EOBT_1_flt","IOBT_flt"]:
    z = (d.BLOCK_TIME_UTC_mvt - d[c]).dt.total_seconds()[m].dropna()
    print(f"BLOCK-{c}: min {z.min():.0f} max {z.max():.0f} share |.|>3606 {(z.abs()>3606).mean():.5f}")
# implied bound on y for matched rows: y = MVT - BLOCK in [MVT-LOBT-3606, MVT-LOBT+3606]
oof = pd.read_parquet(f"{C}/research/cache/oof_mo12_c6.parquet"); oof["id"] = oof["id"].astype("int64")
d["id"] = d.MVT_ID_mvt.astype("int64"); d = d.merge(oof, on="id")
m = d.FLIGHT_ID_mvt.notna().to_numpy()
ml = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds().to_numpy()
y = d.TAXITIME_SEC_mvt.to_numpy(float); p = d.prediction.to_numpy(float)
lo = ml - 3606; hi = ml + 3606
ok = m & np.isfinite(ml)
viol_hi = ok & (p > hi); viol_lo = ok & (p < lo)
q = p.copy(); q[viol_hi] = hi[viol_hi]; q[viol_lo] = np.maximum(lo[viol_lo], 60)
print("c6 OOF predictions above the bound:", viol_hi.sum(), " below:", viol_lo.sum())
print("truth outside bound on matched:", (ok & ((y > hi + 30) | (y < lo - 30))).sum())
base = np.mean((p-y)**2); new = np.mean((q-y)**2)
print(f"clip to NM-join bound: MSE {base:.1f} -> {new:.1f} (delta {new-base:+.1f})")
ds = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
for lo_, hi_ in [(3600,14400),(14400,1e9)]:
    sel = ok & (ds>lo_) & (ds<=hi_)
    print(f" ds in ({lo_/3600:.0f}h,{hi_/3600:.0f}h]: rows {sel.sum()}, preds above bound {(sel & viol_hi).sum()}, dSSE/N {(((q-y)**2)[sel].sum()-((p-y)**2)[sel].sum())/len(y):+.1f}")
