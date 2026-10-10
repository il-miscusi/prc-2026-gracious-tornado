"""Small cached frames for the long-row workstream (WS-D).

    python3 research/tail/build_cache.py

Writes research/tail/cache/long_tr.parquet (every 2025 departure with d_sched > 1 h, the columns
the physical estimators and the recording-mode audit need) and long_rk.parquet (the same columns
for 2026 ranking rows with d_sched > 1 h). Reads only these columns of the fs5 store.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

C = Path(__file__).resolve().parents[2]
FS = C / "research" / "cache" / "fs5"
OUT = C / "research" / "tail" / "cache"
COLS = ["MVT_ID_mvt", "apt", "d_sched", "unmatched", "d_aobt3", "d_eobt1", "d_lobt", "d_iobt",
        "aobt3_m_eobt1", "aobt3_m_sched", "eobt1_m_lobt", "lobt_m_iobt", "dd_utc", "a3_found", "a3_grade",
        "a3_cs_match", "a3_obs_leave", "a3_obs_push", "a3_obs_first", "adsb_found", "adsb_obs_move_all",
        "month", "mday", "local_hour", "inb_fill", "stand_prev1_fill", "apt_arr_fill_1h", "day_unmatched",
        "d_aobt3_mod60", "aobt3_sec", "mvt_sec"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(FS / "tr_base.parquet", columns=COLS + ["y", "artifact"])
    tr = tr[tr.d_sched > 3600].reset_index(drop=True)
    tr["MVT_ID_mvt"] = tr.MVT_ID_mvt.astype("int64")
    assert tr.MVT_ID_mvt.is_unique
    tr.to_parquet(OUT / "long_tr.parquet", index=False)
    rk = pd.read_parquet(FS / "rk_base.parquet", columns=COLS)
    rk["MVT_ID_mvt"] = rk.MVT_ID_mvt.astype("int64")
    assert rk.MVT_ID_mvt.is_unique
    rk.to_parquet(OUT / "rk_all.parquet", index=False)
    print("train long rows", len(tr), "ranking rows", len(rk), "ranking long", int((rk.d_sched > 3600).sum()))


if __name__ == "__main__":
    main()
