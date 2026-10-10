# %%
# Small per-row inputs of the final assembler (research/final/NEEDS.md), cut from the feature stores:
#   phys_final.parquet   prc-f-fs6 ranking rows: MVT_ID_mvt, a3_grade, a3_obs_leave, d_sched, d_aobt3, unmatched
#   cover_final.parquet  prc-f-fs7 ranking rows: MVT_ID_mvt, a3_found, adsb_found
#   cover_board.parquet  prc-fs7 (the store behind v31) ranking rows: same columns, for the board check
#   phys_2025.parquet / cover_2025.parquet   the same columns for 2025 Jan, Feb, Jun, Jul training rows
#     (the 2025 pseudo-final of the four final months)
from pathlib import Path

import pandas as pd

ROOT = Path("/kaggle/input")
PHYS = ["MVT_ID_mvt", "a3_grade", "a3_obs_leave", "d_sched", "d_aobt3", "unmatched"]
COVER = ["MVT_ID_mvt", "a3_found", "adsb_found"]


def store(name):
    hits = [h.parent for h in sorted(ROOT.rglob("fs_meta.json")) if h.parent.name == name or f"/{name}/" in str(h)]
    assert len(hits) == 1, (name, hits)
    return hits[0]


def cut(d, f, cols, months=None):
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(d / f).schema_arrow.names)
    missing = [c for c in cols if c not in have]
    if missing:
        print(f"{d.name}/{f}: missing {missing}", flush=True)
    cols = [c for c in cols if c in have]
    df = pd.read_parquet(d / f, columns=cols + (["month"] if months else []))
    if months:
        df = df[df["month"].isin(months)].drop(columns="month")
    df["MVT_ID_mvt"] = df["MVT_ID_mvt"].astype("int64")
    assert df["MVT_ID_mvt"].is_unique
    return df


f6, f7, o7 = store("prc-f-fs6"), store("prc-f-fs7"), store("prc-fs7")
for out, d, f, cols, months in (("phys_final", f6, "rk_base.parquet", PHYS, None),
                                ("cover_final", f7, "rk_base.parquet", COVER, None),
                                ("cover_board", o7, "rk_base.parquet", COVER, None),
                                ("phys_2025", f6, "tr_base.parquet", PHYS, [1, 2, 6, 7]),
                                ("cover_2025", f7, "tr_base.parquet", COVER, [1, 2, 6, 7])):
    df = cut(d, f, cols, months)
    df.to_parquet(f"{out}.parquet", index=False)
    print(out, d.name, len(df), {c: round(float(df[c].notna().mean()), 4) for c in df.columns[1:]}, flush=True)
