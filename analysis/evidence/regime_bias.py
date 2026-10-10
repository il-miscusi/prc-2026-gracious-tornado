import sys, glob, numpy as np, pandas as pd
C = sys.argv[1]; D = f"{C}/data/09-22-2026-08-13-30_files_list"
oof = pd.read_parquet(f"{C}/research/cache/oof_mo12_c6.parquet"); oof["id"] = oof["id"].astype("int64")
cols = ["MVT_ID_mvt","PHASE_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","TAXITIME_SEC_mvt","AOBT_3_flt"]
d = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{D}/training_2025-*.parquet"))]); d = d[d.PHASE_mvt=="DEP"].copy()
d["id"] = d.MVT_ID_mvt.astype("int64"); d = d.merge(oof, on="id")
mvt = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True); sch = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True); a3 = pd.to_datetime(d.AOBT_3_flt, utc=True)
d["ds"] = (mvt - sch).dt.total_seconds(); d["a3"] = (mvt - a3).dt.total_seconds(); d["mon"] = mvt.dt.month
d = d[d.ds <= 3600]
d["res"] = d.TAXITIME_SEC_mvt - d.prediction
t = d.groupby(["ADEP_mvt","mon"]).agg(a3_med=("a3","median"), mean_res=("res","mean"), rmse=("res", lambda s: np.sqrt((s**2).mean()))).round(1)
pd.set_option("display.width", 250)
for apt in ["LTFM","LFPG","EGLL","EDDF","EHAM"]:
    print(apt); print(t.loc[apt].T.to_string())
