import sys, glob, numpy as np, pandas as pd
C = sys.argv[1]; D = f"{C}/data/09-22-2026-08-13-30_files_list"
cols = ["PHASE_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","FLIGHT_ID_mvt","LOBT_flt","ARVT_3_flt","ARVT_1_flt","AOBT_3_flt","EOBT_1_flt"]
def chk(df, tag):
    for c in ["MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","LOBT_flt","ARVT_3_flt","ARVT_1_flt","AOBT_3_flt","EOBT_1_flt"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    a = df[(df.PHASE_mvt=="ARR") & df.FLIGHT_ID_mvt.notna()]
    out = {}
    for x in ["BLOCK_TIME_UTC_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt"]:
        for yv in ["ARVT_3_flt","ARVT_1_flt"]:
            z = (a[x]-a[yv]).dt.total_seconds().dropna()
            out[f"{x[:5]}-{yv[:6]}"] = (round(z.min()), round(z.max()), round(float((z.abs()>3606).mean()),5))
    print(tag, "matched arrivals", len(a)); [print("   ", k, v) for k, v in out.items()]
tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{D}/training_2025-0[17]*.parquet"))])
chk(tr, "2025 Jan+Jul")
fr = pd.read_parquet(f"{C}/data/final_20261008/final_ranking.parquet", columns=cols)
chk(fr, "2026 final (4 months)")
