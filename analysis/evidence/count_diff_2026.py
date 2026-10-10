import sys, numpy as np, pandas as pd
C = sys.argv[1]
fr = pd.read_parquet(f"{C}/data/final_20261008/final_ranking.parquet", columns=["PHASE_mvt","ADEP_mvt","MVT_TIME_UTC_mvt"])
fr = fr[fr.PHASE_mvt=="DEP"]; mvt = pd.to_datetime(fr.MVT_TIME_UTC_mvt, utc=True)
n = fr.assign(day=mvt.dt.strftime("%Y-%m-%d")).groupby(["ADEP_mvt","day"]).size().rename("n").reset_index()
pr = pd.read_csv(f"{C}/research/pru/all_pre_departure_delays_2026.csv"); pr["day"] = pd.to_datetime(pr.FLT_DATE).dt.strftime("%Y-%m-%d")
m = n.merge(pr, left_on=["ADEP_mvt","day"], right_on=["APT_ICAO","day"])
m["diff"] = m.FLT_DEP_IFR_2 - m.n
print(m.groupby("ADEP_mvt")["diff"].describe(percentiles=[.1,.5,.9]).round(1).to_string())
