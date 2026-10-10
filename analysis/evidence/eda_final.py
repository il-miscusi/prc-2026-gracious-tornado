import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
C = sys.argv[1]
D = f"{C}/data/09-22-2026-08-13-30_files_list"
F = f"{C}/data/final_20261008"
sch = pq.read_schema(f"{F}/final_ranking.parquet")
print("final_ranking columns:", sch.names)
cols = ["MVT_ID_mvt","FLIGHT_ID_mvt","PHASE_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","TAXITIME_SEC_mvt","RUNWAY_mvt","STAND_mvt","CALLSIGN_flt","AOBT_3_flt","EOBT_1_flt","LOBT_flt","IOBT_flt"]
fr = pd.read_parquet(f"{F}/final_ranking.parquet", columns=cols)
print("final_ranking rows", len(fr))
def prep(df):
    df = df.copy()
    for c in ["MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","AOBT_3_flt","EOBT_1_flt","LOBT_flt","IOBT_flt"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df["apt"] = np.where(df.PHASE_mvt=="DEP", df.ADEP_mvt, df.ADES_mvt)
    df["ym"] = df.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    return df
fr = prep(fr)
print(fr.groupby(["ym","PHASE_mvt"]).size().unstack())
sub = pd.read_parquet(f"{F}/final_submitting.parquet")
dep = fr[fr.PHASE_mvt=="DEP"]
print("deps", len(dep), "sub ids in deps", sub.MVT_ID_mvt.isin(dep.MVT_ID_mvt).sum(), "/", len(sub))
print("dep block null share", dep.BLOCK_TIME_UTC_mvt.isna().mean(), "taxi null", dep.TAXITIME_SEC_mvt.isna().mean())
# old ranking comparison on overlapping IDs
old = prep(pd.read_parquet(f"{D}/ranking.parquet", columns=cols))
m = fr.merge(old, on="MVT_ID_mvt", suffixes=("","_o"))
print("overlap rows", len(m), "old rows", len(old))
dt = (m.MVT_TIME_UTC_mvt - m.MVT_TIME_UTC_mvt_o).dt.total_seconds()
print("MVT dither: share nonzero", (dt!=0).mean().round(3), "min/max", dt.min(), dt.max(), "mean", round(dt.mean(),3))
for c in ["SCHED_TIME_UTC_mvt","AOBT_3_flt","EOBT_1_flt","LOBT_flt","IOBT_flt"]:
    d = (m[c]-m[c+"_o"]).dt.total_seconds()
    print(f"  {c}: share differs {(d.fillna(0)!=0).mean():.3f} range {d.min()}..{d.max()}")
for c in ["RUNWAY_mvt","STAND_mvt","CALLSIGN_flt","FLIGHT_ID_mvt"]:
    print(f"  {c}: share differs {(m[c].astype(str)!=m[c+'_o'].astype(str)).mean():.4f}")
arr = m[m.PHASE_mvt=="ARR"]
print("  arr taxi-in diff share", ((arr.TAXITIME_SEC_mvt-arr.TAXITIME_SEC_mvt_o).fillna(0)!=0).mean().round(3), "range", (arr.TAXITIME_SEC_mvt-arr.TAXITIME_SEC_mvt_o).min(), (arr.TAXITIME_SEC_mvt-arr.TAXITIME_SEC_mvt_o).max())
# training months 2025 Jan/Feb/Jun/Jul for comparison
tr = []
for mth in ["01","02","06","07"]:
    nxt = {"01":"2025-02-01","02":"2025-03-01","06":"2025-07-01","07":"2025-08-01"}[mth]
    tr.append(prep(pd.read_parquet(f"{D}/training_2025-{mth}-01_{nxt}.parquet", columns=cols)))
tr = pd.concat(tr)
allm = pd.concat([tr.assign(src="train"), fr.assign(src="final")])
d = allm[allm.PHASE_mvt=="DEP"].copy()
d["ds"] = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
d["a3"] = (d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
d["e1"] = (d.MVT_TIME_UTC_mvt - d.EOBT_1_flt).dt.total_seconds()
d["nm"] = d.FLIGHT_ID_mvt.notna()
g = d.groupby(["apt","ym"])
t = pd.DataFrame({"n": g.size(), "nm_match": g.nm.mean().round(3), "a3_avail": g.a3.apply(lambda s: s.notna().mean()).round(3),
    "a3_med": g.a3.median().round(0), "e1_med": g.e1.median().round(0),
    "ds>1h": g.ds.apply(lambda s: (s>3600).mean()*100).round(2), "ds>4h": g.ds.apply(lambda s: (s>14400).mean()*100).round(3), "ds>8h": g.ds.apply(lambda s: (s>28800).sum()),
    "y_med": g.TAXITIME_SEC_mvt.median()})
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(t)
a = allm[allm.PHASE_mvt=="ARR"]
print("arrival taxi-in mean by apt x month:")
print(a.groupby(["apt","ym"]).TAXITIME_SEC_mvt.mean().round(0).unstack())
# callsign novelty vs all 2025? use the 4 training months as proxy
cs25 = set(tr.loc[tr.PHASE_mvt=="DEP","CALLSIGN_flt"].dropna())
fd = fr[fr.PHASE_mvt=="DEP"]
print("callsign unseen in 2025 Jan/Feb/Jun/Jul share by month:", fd.groupby("ym").CALLSIGN_flt.apply(lambda s: (~s.dropna().isin(cs25)).mean()).round(3).to_dict())
