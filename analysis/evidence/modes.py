import sys, glob, numpy as np, pandas as pd
C = sys.argv[1]
D = f"{C}/data/09-22-2026-08-13-30_files_list"
oof = pd.read_parquet(f"{C}/research/cache/oof_mo12_c6.parquet"); oof["id"] = oof["id"].astype("int64")
cols = ["MVT_ID_mvt","PHASE_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","TAXITIME_SEC_mvt","FLIGHT_ID_mvt"]
d = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{D}/training_2025-*.parquet"))]); d = d[d.PHASE_mvt=="DEP"].copy()
d["id"] = d.MVT_ID_mvt.astype("int64"); d = d.merge(oof, on="id")
mvt = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True); sch = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True)
d["ds"] = (mvt - sch).dt.total_seconds(); d["y"] = d.TAXITIME_SEC_mvt; d["obd"] = d.ds - d.y   # AOBT - SOBT
d["nm"] = d.FLIGHT_ID_mvt.notna(); d["se"] = (d.prediction - d.y)**2
tot = d.se.sum()
for lo, hi in [(3600, 14400), (14400, 28800), (28800, 1e9)]:
    x = d[(d.ds > lo) & (d.ds <= hi)]
    t = x.groupby("ADEP_mvt").agg(n=("y","size"), ontime_share=("obd", lambda s: (s < 600).mean()), near_sched=("obd", lambda s: (s.abs() <= 120).mean()),
        sse_share=("se", lambda s: s.sum()/tot), rmse=("se", lambda s: np.sqrt(s.mean())))
    u = x.groupby(["ADEP_mvt","nm"]).agg(n=("y","size"), ontime=("obd", lambda s: (s < 600).mean())).unstack()
    print(f"==== ds in ({lo/3600:.0f}h, {hi/3600:.0f}h]"); print(t.round(4).T.to_string()); print(u.round(3).T.to_string())
