"""ADS-B coverage of the departures the final file predicts, and of the same months of 2025.

    python3 analysis/coverage.py ROOT

ROOT holds the organiser files in data/ and the assembler inputs research/final/inputs/{final,2025}/
cover.parquet, written by the prc-f-inputs kernel (kernels/prc-f-inputs/kernel.py) from the fs7
store. Per departure: a3_found = an adsb.lol v3 ground event was matched to the flight (store fs7),
adsb_found = the v1 extraction saw the departure. Printed: the share of departures with each, by
airport and takeoff month.
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "research" / "final"))
from frame import load_frame, load_training_frame  # noqa: E402

for kind in ("final", "2025"):
    fr = load_frame("final") if kind == "final" else load_training_frame((1, 2, 6, 7))[0]
    cv = pd.read_parquet(root / "research" / "final" / "inputs" / kind / "cover.parquet")
    cv.index = cv.MVT_ID_mvt.round().astype("int64")
    t = fr[["id", "apt", "year", "month"]].join(cv[["a3_found", "adsb_found"]], on="id")
    for col in ("a3_found", "adsb_found"):
        g = (t[col].fillna(0) > 0).groupby([t.apt, t.year.astype(str) + "-" + t.month.map("{:02d}".format)]).mean()
        print(f"\n{kind}: share of departures with {col}")
        print(g.unstack().mul(100).round(1).to_string())
