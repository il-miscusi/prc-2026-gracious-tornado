#!/usr/bin/env python3
"""Sanity check of the geometry tables against Jan + Jul 2025 departures (local, seconds of CPU).

Per airport: Pearson and Spearman correlation of path length (and straight-line distance) with
y = TAXITIME_SEC_mvt for y <= 1800 s; the same after removing the per-runway mean (does geometry
say anything the runway alone does not?); and how many (stand, runway) pairs are rare in 2025
(< 30 departures over the whole year) yet have geometry, with the 2026 ranking rows they carry.

    python3 check_value.py            # prints the tables, writes value_check.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "09-22-2026-08-13-30_files_list"
RARE = 30


def load(files):
    cols = ["ADEP_mvt", "PHASE_mvt", "STAND_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt"]
    d = pd.concat([pd.read_parquet(DATA / f, columns=cols) for f in files], ignore_index=True)
    d = d[d["PHASE_mvt"] == "DEP"].copy()
    d["STAND_mvt"] = d["STAND_mvt"].astype(str)
    d["RUNWAY_mvt"] = d["RUNWAY_mvt"].astype(str)
    return d


def main():
    pairs = pd.read_parquet(HERE / "stand_runway.parquet")
    key = ["apt", "stand", "runway"]
    geo = pairs[key + ["path_m", "straight_m", "crossings", "path_source", "n_dep_2025", "n_dep_2026"]]
    jj = load(["training_2025-01-01_2025-02-01.parquet", "training_2025-07-01_2025-08-01.parquet"])
    jj = jj.rename(columns={"ADEP_mvt": "apt", "STAND_mvt": "stand", "RUNWAY_mvt": "runway",
                            "TAXITIME_SEC_mvt": "y"})
    jj = jj.merge(geo, on=key, how="left")
    s = jj[(jj["y"] > 0) & (jj["y"] <= 1800)].copy()
    s["y_rw"] = s["y"] - s.groupby(["apt", "runway"])["y"].transform("mean")
    rows = []
    for apt, g in s.groupby("apt"):
        h = g[g["path_m"].notna()]
        hs = g[g["straight_m"].notna()]
        h_rw = h.assign(p_rw=h["path_m"] - h.groupby("runway")["path_m"].transform("mean"))
        gp = pairs[pairs["apt"] == apt]
        rare = gp[(gp["n_dep_2025"] < RARE)]
        rare_geo = rare[rare["path_m"].notna()]
        unseen = gp[(gp["n_dep_2025"] == 0)]
        rows.append({
            "apt": apt, "rows_jj": len(g), "path_cov": len(h) / len(g),
            "r_path": h["path_m"].corr(h["y"]), "rho_path": h["path_m"].corr(h["y"], method="spearman"),
            "r_straight": hs["straight_m"].corr(hs["y"]),
            "r_path_within_rwy": h_rw["p_rw"].corr(h_rw["y_rw"]),
            "r_crossings": h["crossings"].corr(h["y"]) if h["crossings"].nunique() > 1 else np.nan,
            "rows_rare_jj": int((h["n_dep_2025"] < RARE).sum()),
            "r_path_rare": h.loc[h["n_dep_2025"] < RARE, "path_m"].corr(h.loc[h["n_dep_2025"] < RARE, "y"]),
            "pairs": len(gp), "rare_pairs": len(rare), "rare_pairs_geo": len(rare_geo),
            "rare_geo_rows_2026": int(rare_geo["n_dep_2026"].sum()),
            "rank_rows_2026": int(gp["n_dep_2026"].sum()),
            "unseen_2025_pairs_geo": int(unseen["path_m"].notna().sum()),
            "unseen_geo_rows_2026": int(unseen.loc[unseen["path_m"].notna(), "n_dep_2026"].sum()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "value_check.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
