"""Bounds that the organisers' data construction or 2025 labels impose on y, and the delivery checks.

Applied after level calibration and the post layers, so the bounds hold in the delivered file:

  (a) NM join window. The organisers joined an NM flight to a movement only when
      |off-block - LOBT| <= 3,606 s (zero exceptions in 2,062,623 matched 2025 departures). So for
      NM-matched rows y lies in [MVT - LOBT - 3,606, MVT - LOBT + 3,606]. The final file dithers
      takeoff (and, for arrivals, block) by up to +-12 s, so the window is widened by 25 s: 3,631.
  (d) LEMD and LEBL: y_hat <= 3,604 s (2025 maxima 3,598 and 3,604).
  Floor 60 s everywhere, last.

Optional LIRF orphan band rule (`lirf_orphan_floor_d`, off unless a config enables it): LIRF
departures without an NM flight and lo <= d < hi get y_hat >= d (`lirf_orphan_band_mode`
"floor", which also exempts the band from the assembler's cap) or y_hat = d ("equal").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT = {"nm_window_s": 3606, "nm_dither_s": 25, "lirf_orphan_floor_d": None,
           "lirf_orphan_band_mode": "floor", "cap_airports": ["LEMD", "LEBL"], "cap_s": 3604, "floor_s": 60}
BAND_KEYS = ("lirf_orphan_floor_d", "lirf_orphan_band_mode")


def full(cfg: dict | None) -> dict:
    return {**DEFAULT, **(cfg or {})}


def band_on(cfg: dict) -> bool:
    return full(cfg).get("lirf_orphan_floor_d") is not None


def effective(cfg: dict | None) -> dict:
    """The settings in force, for the manifest (the band keys only when the band rule is on)."""
    c = full(cfg)
    return c if band_on(c) else {k: v for k, v in c.items() if k not in BAND_KEYS}


def nm_bounds(fr: pd.DataFrame, cfg: dict):
    B = cfg["nm_window_s"] + cfg["nm_dither_s"]
    dl = fr.d_lobt.to_numpy()
    ok = fr.nm.to_numpy() & np.isfinite(dl)
    return ok, dl - B, dl + B


def orphan_band(fr: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Rows the optional LIRF orphan band rule acts on (none when it is off)."""
    if not band_on(cfg):
        return np.zeros(len(fr), bool)
    lo, hi = full(cfg)["lirf_orphan_floor_d"]
    d = fr.d.to_numpy()
    return fr.lirf_orphan.to_numpy() & (d >= lo) & (d < hi)


def capped_airports(fr: pd.DataFrame, cfg: dict) -> np.ndarray:
    return fr.apt.isin(cfg["cap_airports"]).to_numpy()


def apply(fr: pd.DataFrame, pred: np.ndarray, cfg: dict | None = None):
    cfg = full(cfg)
    q = np.asarray(pred, float).copy()
    counts = {}
    ok, lo, hi = nm_bounds(fr, cfg)
    above, below = ok & (q > hi), ok & (q < lo)
    counts["a_nm_above"], counts["a_nm_below"] = int(above.sum()), int(below.sum())
    q[above] = hi[above]
    q[below] = np.maximum(lo[below], cfg["floor_s"])
    if band_on(cfg):
        band = orphan_band(fr, cfg)
        d = fr.d.to_numpy()
        up = band & (q < d)
        counts["band_raised_to_d"] = int(up.sum())
        q[up] = d[up]
        if cfg["lirf_orphan_band_mode"] == "equal":
            down = band & (q > d)
            counts["band_lowered_to_d"] = int(down.sum())
            q[down] = d[down]
        else:
            assert cfg["lirf_orphan_band_mode"] == "floor", cfg["lirf_orphan_band_mode"]
    cp = capped_airports(fr, cfg) & (q > cfg["cap_s"])
    counts["d_cap_lowered"] = int(cp.sum())
    q[cp] = cfg["cap_s"]
    lowf = q < cfg["floor_s"]
    counts["floor_raised"] = int(lowf.sum())
    q = np.maximum(q, cfg["floor_s"])
    # rows where two bounds cannot both hold (the later rule wins); reported, expected to be zero
    counts["conflict_a_vs_d"] = int((ok & capped_airports(fr, cfg) & (lo > cfg["cap_s"])).sum())
    counts["conflict_a_vs_floor"] = int((ok & (hi < cfg["floor_s"])).sum())
    return q, counts


def check(fr: pd.DataFrame, pred: np.ndarray, cfg: dict | None = None) -> dict:
    """Checklist §8.6 item 1 bounds on the delivered values (tolerance 1e-6 s)."""
    cfg = full(cfg)
    p = np.asarray(pred, float)
    ok, lo, hi = nm_bounds(fr, cfg)
    exempt = (capped_airports(fr, cfg) & (lo > cfg["cap_s"])) | (hi < cfg["floor_s"])
    out = {
        "finite": int((~np.isfinite(p)).sum()) == 0,
        "nonneg": int((p < 0).sum()) == 0,
        "floor": int((p < cfg["floor_s"] - 1e-6).sum()),
        "nm_outside_window": int((ok & ~exempt & ((p > hi + 1e-6) | (p < lo - 1e-6))).sum()),
        "nm_window_exempt_conflicts": int((ok & exempt).sum()),
        "lemd_lebl_above_cap": int((capped_airports(fr, cfg) & (p > cfg["cap_s"] + 1e-6)).sum()),
    }
    bad = 0
    if band_on(cfg):
        band, d = orphan_band(fr, cfg), fr.d.to_numpy()
        out["band_below_d"] = int((band & (p < d - 1e-6)).sum())
        out["band_above_d_in_equal_mode"] = int((band & (p > d + 1e-6)).sum()) if cfg["lirf_orphan_band_mode"] == "equal" else 0
        bad = out["band_below_d"] + out["band_above_d_in_equal_mode"]
    out["pass"] = (out["finite"] and out["nonneg"] and out["floor"] == 0 and out["nm_outside_window"] == 0
                   and out["lemd_lebl_above_cap"] == 0 and bad == 0)
    return out
