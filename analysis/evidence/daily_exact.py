import sys, glob, numpy as np, pandas as pd
C = sys.argv[1]
D = f"{C}/data/09-22-2026-08-13-30_files_list"
cols = ["PHASE_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt"]
d = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{D}/training_2025-*.parquet"))]); d = d[d.PHASE_mvt=="DEP"].copy()
mvt = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True); sch = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True); blk = pd.to_datetime(d.BLOCK_TIME_UTC_mvt, utc=True)
d["day"] = mvt.dt.strftime("%Y-%m-%d"); d["apt"] = d.ADEP_mvt; d["c"] = ((blk - sch).dt.total_seconds()/60).clip(lower=0)
g = d.groupby(["apt","day"]).agg(n=("c","size"), S=("c","sum")).reset_index()
pr = pd.read_csv(f"{C}/research/pru/all_pre_departure_delays_2025.csv"); pr["day"] = pd.to_datetime(pr.FLT_DATE).dt.strftime("%Y-%m-%d")
m = g.merge(pr, left_on=["apt","day"], right_on=["APT_ICAO","day"])
m["cnt_exact"] = m.FLT_DEP_IFR_2 == m.n; m["sum_err_min"] = m.DLY_ALL_PRE_2 - m.S
print(m.groupby("apt").agg(days=("n","size"), cnt_exact=("cnt_exact","mean"), cnt_diff_med=("n", lambda s: 0), abs_err_med=("sum_err_min", lambda s: s.abs().median()), abs_err_p90=("sum_err_min", lambda s: s.abs().quantile(.9)), S_med=("S","median")).round(3).T.to_string())
print("days with exact count: sum err abs median", m[m.cnt_exact].sum_err_min.abs().median(), " p90", m[m.cnt_exact].sum_err_min.abs().quantile(.9))
print("days with inexact count: sum err abs median", m[~m.cnt_exact].sum_err_min.abs().median(), " p90", m[~m.cnt_exact].sum_err_min.abs().quantile(.9))
m["mon"] = m.day.str[:7]
print(m.groupby("mon").cnt_exact.mean().round(3).to_dict())
