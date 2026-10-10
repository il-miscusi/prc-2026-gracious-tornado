"""Build the METAR feature file used from v13 on (external, openly accessible data).

Source: Iowa Environmental Mesonet ASOS/METAR archive,
https://mesonet.agron.iastate.edu/request/download.phtml (public, no login).
Half-hourly routine + special reports for the 10 challenge airports,
2025-01-01 to 2026-07-31 UTC. Published privately on Kaggle as
usiadianimuwa/prc-taxi-2026-metar for the kernels.

    python3 research/build_metar.py OUT_DIR
"""

from __future__ import annotations

import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

STATIONS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={s}"
       "&data=tmpf&data=dwpf&data=sknt&data=gust&data=vsby&data=wxcodes&data=skyl1&data=skyc1"
       "&data=mslp&data=p01m&year1=2025&month1=1&day1=1&year2=2026&month2=8&day2=1"
       "&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=empty&trace=0.0001"
       "&direct=no&report_type=3&report_type=4")


def fetch(station: str) -> pd.DataFrame:
    with urllib.request.urlopen(URL.format(s=station), timeout=180) as resp:
        return pd.read_csv(io.BytesIO(resp.read()))


def features(m: pd.DataFrame) -> pd.DataFrame:
    m["valid"] = pd.to_datetime(m["valid"], utc=True)
    wx = m["wxcodes"].fillna("").astype(str)
    out = pd.DataFrame({
        "station": m["station"], "valid_utc": m["valid"],
        "temp_c": (m["tmpf"] - 32) / 1.8, "dew_c": (m["dwpf"] - 32) / 1.8,
        "wind_kt": m["sknt"], "gust_kt": m["gust"], "vis_km": m["vsby"] * 1.609,
        "ceil_ft": np.where(m["skyc1"].isin(["BKN", "OVC", "VV"]), m["skyl1"], np.nan),
        "precip_mm": m["p01m"],
        "wx_snow": wx.str.contains("SN|SG|PL|GS").astype("int8"),
        "wx_freezing": wx.str.contains("FZ").astype("int8"),
        "wx_rain": wx.str.contains("RA|DZ").astype("int8"),
        "wx_fog": wx.str.contains("FG|BR").astype("int8"),
        "wx_thunder": wx.str.contains("TS").astype("int8"),
        "wx_shower": wx.str.contains("SH").astype("int8"),
    })
    out["spread_c"] = out["temp_c"] - out["dew_c"]
    wet = out["wx_snow"] + out["wx_freezing"] + out["wx_rain"] + out["wx_fog"] > 0
    out["deice_likely"] = ((out["temp_c"] <= 3) & (wet | (out["spread_c"] <= 1.5))).astype("int8")
    return out.sort_values(["station", "valid_utc"]).reset_index(drop=True)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for s in STATIONS:
        frames.append(fetch(s))
        time.sleep(2)
    features(pd.concat(frames, ignore_index=True)).to_parquet(out_dir / "metar_prc2026.parquet", index=False)
