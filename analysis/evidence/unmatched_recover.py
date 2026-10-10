import sys, glob, numpy as np, pandas as pd
C = sys.argv[1]; D = f"{C}/data/09-22-2026-08-13-30_files_list"
cols = ["MVT_ID_mvt","FLIGHT_ID_mvt","FLIGHT_mvt","PHASE_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","TAXITIME_SEC_mvt","LOBT_flt","AOBT_3_flt","ADEP_flt","CALLSIGN_flt"]
d = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{D}/training_2025-*.parquet"))])
for c in ["MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","LOBT_flt","AOBT_3_flt"]: d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
dep = d[(d.PHASE_mvt=="DEP") & d.FLIGHT_ID_mvt.isna()].copy()
arr = d[(d.PHASE_mvt=="ARR") & d.FLIGHT_ID_mvt.notna()].copy()
print("unmatched departures", len(dep), "| to one of the 10 airports:", dep.ADES_mvt.isin(d.ADEP_mvt.dropna().unique()).sum())
# candidate: same flight number string, arrival at dep's ADES from dep's ADEP, landing 10 min..20 h after takeoff
arr = arr.rename(columns={c: c+"_a" for c in arr.columns})
m = dep.merge(arr, left_on=["FLIGHT_mvt","ADEP_mvt","ADES_mvt"], right_on=["FLIGHT_mvt_a","ADEP_mvt_a","ADES_mvt_a"], how="inner")
dt = (m.MVT_TIME_UTC_mvt_a - m.MVT_TIME_UTC_mvt).dt.total_seconds()
m = m[(dt > 600) & (dt < 20*3600)].copy()
m["dt"] = (m.MVT_TIME_UTC_mvt_a - m.MVT_TIME_UTC_mvt).dt.total_seconds()
m = m.sort_values("dt").drop_duplicates("MVT_ID_mvt")
print("recovered NM flight via destination arrival:", len(m), "of", len(dep))
gap = (m.BLOCK_TIME_UTC_mvt - m.LOBT_flt_a).dt.total_seconds()
print("BLOCK - recovered LOBT: share |.|>3606 = %.3f ; share > +3606 = %.3f ; share < -3606 = %.3f" % ((gap.abs()>3606).mean(), (gap>3606).mean(), (gap<-3606).mean()))
m["y"] = m.TAXITIME_SEC_mvt; m["ds"] = (m.MVT_TIME_UTC_mvt - m.SCHED_TIME_UTC_mvt).dt.total_seconds()
m["lb"] = (m.MVT_TIME_UTC_mvt - m.LOBT_flt_a).dt.total_seconds() + 3606
print("rows where y > MVT-LOBT+3606 (implied lower bound holds):", round(float((m.y > m.lb - 30).mean()),3))
print("by origin airport (recovered rows, share y>lb, share |y-ds|<=600):")
print(m.groupby("ADEP_mvt").apply(lambda q: pd.Series({"n": len(q), "y_gt_lb": (q.y > q.lb - 30).mean(), "ontime": ((q.ds - q.y) < 600).mean(), "ds>1h": (q.ds>3600).mean()}), include_groups=False).round(3).to_string())
