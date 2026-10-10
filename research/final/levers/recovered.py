"""L0e: the NM join window applied to orphans whose NM flight is recoverable.

The organisers joined an NM flight to a departure only when |BLOCK - LOBT| <= 3,606 s
(review B section 5.5). An NM-unmatched departure whose flight lands at one of the ten
airports appears there as an NM-matched ARRIVAL (same flight number and route, landing
10 min - 20 h after our takeoff) that carries the flight's LOBT. If the failed join was
the window, BLOCK lies outside LOBT +- 3,606, i.e. y = MVT - BLOCK lies OUTSIDE the
interval (MVT - LOBT - 3,606, MVT - LOBT + 3,606).

2025 support for "the truth is outside that interval" (recovered orphans whose base
prediction falls inside it): LIRF 44/44, LTFM 21/21, EGLL 8/8, but EHAM 1/9, LEBL 2/11,
EDDF 3/5, LFPG 7/9 - there the join failed for other reasons. So the rule is gated to
LIRF and LTFM (EGLL measured but too small).

Mode feasibility (y = d for an on-time/fill record; y ~ tau for a held flight):
    fill allowed  iff d is outside the interval
    held allowed  iff the interval's lower end is above the held level (lo >= TAU_HI)
    "edge" (adopted): a prediction inside the interval moves to the nearer edge.
    "modes" (rejected): only fill -> d; only held -> min(p, lo); else nearer edge. Its whole
    2025 gain (-611) came from the 2025-only date-wrap rows; without them it is +17 [-116, +161].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import lv_common as common
except ImportError:  # run as a script from this folder
    import lv_common as common  # type: ignore

JOIN = 3606.0
TAU_HI = 1500.0
AIRPORTS = ("LIRF", "LTFM")


def recovered_ml(dep: pd.DataFrame, arr: pd.DataFrame) -> np.ndarray:
    """MVT - LOBT for orphans, LOBT taken from the destination's NM-matched arrival (NaN if none)."""
    sec = lambda t: (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    o = pd.DataFrame({"i": np.arange(len(dep)), "flight": dep["flight"].to_numpy(), "apt": dep["apt"].to_numpy(),
                      "ades": dep["ades"].to_numpy(), "tm": sec(dep["mvt"]).to_numpy()})
    o = o[~dep["nm"].to_numpy(bool) & (o.flight != "")]
    am = arr[arr["nm"].to_numpy(bool) & (arr["flight"] != "")]
    am = pd.DataFrame({"flight": am["flight"].to_numpy(), "adep": am["adep"].to_numpy(), "ades": am["ades"].to_numpy(),
                       "tland": sec(am["mvt"]).to_numpy(), "ml_a": am["ml"].to_numpy()})
    am["lobt"] = am.tland - am.ml_a
    m = o.merge(am, left_on=["flight", "apt", "ades"], right_on=["flight", "adep", "ades"], how="inner")
    m = m[(m.tland - m.tm > 600) & (m.tland - m.tm < 72000) & np.isfinite(m.lobt)]
    m = m.sort_values("tland").drop_duplicates("i")
    out = np.full(len(dep), np.nan)
    out[m.i.to_numpy()] = (m.tm - m.lobt).to_numpy()
    return out


def rule(dep: pd.DataFrame, pred: np.ndarray, mlr: np.ndarray, airports=AIRPORTS, variant: str = "edge") -> np.ndarray:
    q = np.asarray(pred, float).copy()
    d = dep["d"].to_numpy(float)
    sel = np.isin(dep["apt"].to_numpy(), list(airports)) & np.isfinite(mlr)
    lo, hi = mlr - JOIN, mlr + JOIN
    inside = sel & (q > lo) & (q < hi)
    if variant == "edge":
        q[inside] = np.where(q[inside] - lo[inside] < hi[inside] - q[inside], np.maximum(lo[inside], 60), hi[inside])
        return q
    fill_ok = (d <= lo) | (d >= hi)
    held_ok = lo >= TAU_HI
    only_fill = sel & fill_ok & ~held_ok
    only_held = sel & held_ok & ~fill_ok
    q[only_fill] = d[only_fill]
    q[only_held] = np.minimum(q[only_held], lo[only_held])
    rest = inside & ~only_fill & ~only_held
    q[rest] = np.where(q[rest] - lo[rest] < hi[rest] - q[rest], np.maximum(lo[rest], 60), hi[rest])
    return q


def apply(frame: pd.DataFrame, pred: np.ndarray, airports=AIRPORTS, variant: str = "edge", held_month=None) -> np.ndarray:
    """L0e on any frame (2026: the final file's arrivals; 2025: the training arrivals). Nothing is
    fitted on departure labels; the airport gate (LIRF, LTFM) was chosen on all 2025 rows, so in the
    pseudo-final ``held_month`` is accepted and ignored (disclosed: the gate saw the held month;
    LIRF 44/44 and LTFM 21/21 hold without any single month too)."""
    fr = common.ensure_columns(frame, ["apt", "nm", "d", "flight", "ades", "mvt", "year"])
    years = set(fr["year"].astype(int))
    arr = pd.concat([common.load_final(arrivals=True)] if min(years) >= 2026 else
                    [common.load_2025(arrivals=True)], ignore_index=True)
    return rule(fr, pred, recovered_ml(fr, arr), airports, variant)


if __name__ == "__main__":
    import lv_rules as rules
    dep = common.load_2025()
    arr = common.load_2025(arrivals=True)
    p0 = rules.apply(dep, dep["p"].to_numpy(float))
    mlr = recovered_ml(dep, arr)
    print("recovered orphans:", int(np.isfinite(mlr).sum()))
    for ap in [("LIRF",), ("LIRF", "LTFM"), ("LIRF", "LTFM", "EGLL")]:
        for v in ["edge", "modes"]:
            common.score(f"L0e recovered-orphan window {'+'.join(ap)} {v} (on L0)", dep, rule(dep, p0, mlr, ap, v), p0)
    fin = common.load_final()
    mlr26 = recovered_ml(fin, common.load_final(arrivals=True))
    s = np.isin(fin.apt.to_numpy(), list(AIRPORTS)) & np.isfinite(mlr26)
    print("2026 recovered LIRF/LTFM orphans by month:", pd.Series(fin.month.to_numpy()[s]).value_counts().sort_index().to_dict())
