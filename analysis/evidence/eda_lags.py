import sys, numpy as np, pandas as pd
C = sys.argv[1]
D = f"{C}/data/09-22-2026-08-13-30_files_list"; F = f"{C}/data/final_20261008"
cols = ["MVT_ID_mvt","FLIGHT_ID_mvt","PHASE_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","TAXITIME_SEC_mvt","AOBT_3_flt","EOBT_1_flt","LOBT_flt","IOBT_flt"]
months = {"01":"2025-02-01","02":"2025-03-01","03":"2025-04-01","04":"2025-05-01","05":"2025-06-01","06":"2025-07-01","07":"2025-08-01","08":"2025-09-01","09":"2025-10-01","10":"2025-11-01","11":"2025-12-01","12":"2026-01-01"}
fr = [pd.read_parquet(f"{D}/training_2025-{m}-01_{n}.parquet", columns=cols) for m,n in months.items()]
fr.append(pd.read_parquet(f"{F}/final_ranking.parquet", columns=cols))
d = pd.concat(fr); d = d[d.PHASE_mvt=="DEP"].copy()
for c in ["MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","AOBT_3_flt","EOBT_1_flt","LOBT_flt","IOBT_flt","BLOCK_TIME_UTC_mvt"]:
    d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
d["ym"] = d.MVT_TIME_UTC_mvt.dt.strftime("%y%m")
d["a3"] = (d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
d["e1"] = (d.MVT_TIME_UTC_mvt - d.EOBT_1_flt).dt.total_seconds()
d["lo"] = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds()
d["y"] = d.TAXITIME_SEC_mvt
d["ds"] = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for apt in ["LTFM","LFPG","EHAM","EDDF","LIRF","EGLL"]:
    x = d[d.ADEP_mvt==apt]
    g = x.groupby("ym")
    t = pd.DataFrame({"a3_med": g.a3.median(), "a3_mode_share": g.a3.agg(lambda s: s.round(-1).value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan).round(3),
                      "a3_iqr": g.a3.quantile(.75)-g.a3.quantile(.25), "e1_med": g.e1.median(), "lo_med": g.lo.median(), "y_med": g.y.median(), "y_mean_le2h": g.apply(lambda q: q.y[q.y<=7200].mean()).round(0),
                      "corr_y_a3": g.apply(lambda q: q[["y","a3"]].dropna().query("y<=7200 and a3<=7200").corr().iloc[0,1]).round(3),
                      "rmse_y_vs_a3": g.apply(lambda q: np.sqrt(((q.y-q.a3)[(q.y<=7200)&(q.a3<=7200)]**2).mean())).round(0)}).T
    print("=====", apt); print(t.to_string())
