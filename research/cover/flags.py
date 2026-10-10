"""Flags for the LIRF no-ADS-B overlay (research/assemble.py `overlays`).

    python3 research/cover/flags.py FS_RK_BASE.parquet OUT.parquet [--suffix _fs7]

A ranking departure is uncovered when it has neither an adsb.lol v3 event nor a v1 departure match
(a3_found <= 0 and adsb_found <= 0, missing read as 0).  The overlay rule is LIRF, uncovered and
d_sched <= 4 h: there the candidate becomes 0.7 x candidate + 0.3 x the no-ADS-B LightGBM.
Run on the fs7 store's rk_base.parquet with --suffix _fs7 for the column v32c reads (lirf_rule_fs7).
Checked 2026-10-01 against the flags v32c was built with: the rule column matches on 344,841 of
344,841 rows (fs5 store).
"""
import sys

import numpy as np
import pandas as pd


def main():
    src, out = sys.argv[1], sys.argv[2]
    suffix = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ""
    rk = pd.read_parquet(src, columns=["MVT_ID_mvt", "apt", "month", "d_sched", "a3_found", "adsb_found"])
    unc = (rk.a3_found.fillna(0) <= 0) & (rk.adsb_found.fillna(0) <= 0)
    f = pd.DataFrame({"id": rk.MVT_ID_mvt.astype(np.int64), "apt": rk.apt.astype(str), "month": rk.month,
                      "d_sched": rk.d_sched,
                      f"uncovered{suffix}": unc & (rk.d_sched <= 4 * 3600),
                      f"lirf_rule{suffix}": unc & (rk.apt.astype(str) == "LIRF") & (rk.d_sched <= 4 * 3600)})
    assert f.id.is_unique
    f.to_parquet(out, index=False)
    print(f"{out}: {len(f):,} rows, rule rows {int(f[f'lirf_rule{suffix}'].sum()):,}")


if __name__ == "__main__":
    main()
