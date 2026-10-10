"""L0 rule package: exact or 2025-supported bounds on y.

    (a) NM join window. The organisers joined an NM flight to a movement only when
        |BLOCK - LOBT| <= 3,606 s (0 exceptions in 2,062,623 matched 2025 departures).
        For a matched departure y = MVT - BLOCK therefore lies in
        [MVT - LOBT - 3,606, MVT - LOBT + 3,606]. Clip into it (lower end floored at 60).
        In the final file MVT is dithered by up to +-12 s, so the 2026 window is widened
        by ``dither`` (default 25 s; B section 8.6 uses 3,631).
    (c) Optional LIRF orphan band: LIRF departures without an NM flight and
        24,000 <= d < 86,400 get y_hat >= d (c_mode "floor") or y_hat = d ("equal").
        Applied when ``rules`` contains "c" (the default string is "acd"; configs that
        leave it out pass rules="ad").
    (d) LEMD and LEBL never exceed 3,604 s in 2025. Cap there.

There is no rule (b).

Pure function; no fitted parameters. ``apply(frame, pred)`` needs the columns
apt, nm, ml, d (computed from the raw final_ranking columns when absent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import lv_common as common
except ImportError:  # run as a script from this folder
    import lv_common as common  # type: ignore

JOIN = 3606.0
NEED = ["apt", "nm", "ml", "d"]


def rule_a(frame: pd.DataFrame, pred: np.ndarray, dither: float = 0.0) -> np.ndarray:
    q = np.asarray(pred, float).copy()
    ml = frame["ml"].to_numpy(float)
    ok = frame["nm"].to_numpy(bool) & np.isfinite(ml)
    hi = ml + JOIN + dither
    lo = ml - JOIN - dither
    up = ok & (q > hi)
    dn = ok & (q < lo)
    q[up] = hi[up]
    q[dn] = np.maximum(lo[dn], 60.0)
    return q


def rule_c(frame: pd.DataFrame, pred: np.ndarray, mode: str = "floor") -> np.ndarray:
    """mode "floor": y_hat >= d; "equal": y_hat = d."""
    q = np.asarray(pred, float).copy()
    d = frame["d"].to_numpy(float)
    m = (frame["apt"].to_numpy() == "LIRF") & ~frame["nm"].to_numpy(bool) & (d >= 24000) & (d < 86400)
    q[m] = d[m] if mode == "equal" else np.maximum(q[m], d[m])
    return q


def rule_d(frame: pd.DataFrame, pred: np.ndarray, cap: float = 3604.0) -> np.ndarray:
    q = np.asarray(pred, float).copy()
    m = np.isin(frame["apt"].to_numpy(), ["LEMD", "LEBL"])
    q[m] = np.minimum(q[m], cap)
    return q


def apply(frame: pd.DataFrame, pred: np.ndarray, dither: float | None = None,
          rules: str = "acd", c_mode: str = "floor", held_month=None) -> np.ndarray:
    """Apply rules in the order a, c, d. ``dither`` defaults to 25 s on 2026 rows, 0 on 2025.
    No fitted quantity, so ``held_month`` (pseudo-final) is accepted and ignored."""
    frame = common.ensure_columns(frame, NEED + (["year"] if dither is None else []))
    if dither is None:
        dither = np.where(frame["year"].to_numpy() >= 2026, 25.0, 0.0)
    q = np.asarray(pred, float)
    if "a" in rules:
        q = rule_a(frame, q, dither)
    if "c" in rules:
        q = rule_c(frame, q, c_mode)
    if "d" in rules:
        q = rule_d(frame, q)
    return q


def violations(frame: pd.DataFrame, pred: np.ndarray, dither: float = 25.0) -> dict:
    """Counts of rows that break each rule (the final-file checklist of B section 8.6)."""
    frame = common.ensure_columns(frame, NEED)
    q = np.asarray(pred, float)
    ml = frame["ml"].to_numpy(float)
    ok = frame["nm"].to_numpy(bool) & np.isfinite(ml)
    d = frame["d"].to_numpy(float)
    apt = frame["apt"].to_numpy()
    lf = (apt == "LIRF") & ~frame["nm"].to_numpy(bool) & (d >= 24000) & (d < 86400)
    return {
        "a_above": int((ok & (q > ml + JOIN + dither + 1e-6)).sum()),
        "a_below": int((ok & (q < ml - JOIN - dither - 1e-6) & (q > 60 + 1e-6)).sum()),
        "c_below_d": int((lf & (q < d - 1e-6)).sum()),
        "d_above": int((np.isin(apt, ["LEMD", "LEBL"]) & (q > 3604 + 1e-6)).sum()),
    }


if __name__ == "__main__":
    import json
    import sys

    dep = common.load_2025()
    p = dep["p"].to_numpy(float)
    res = []
    res.append(common.score("(a) NM-join clip", dep, rule_a(dep, p)))
    res.append(common.score("(a) NM-join clip, 2026 margin 3631", dep, rule_a(dep, p, 25.0)))
    qa = rule_a(dep, p)
    res.append(common.score("(c) LIRF orphan floor, on top of (a)", dep, rule_c(dep, qa), qa))
    qc = rule_c(dep, qa)
    res.append(common.score("(d) LEMD/LEBL cap, on top of (a)+(c)", dep, rule_d(dep, qc), qc))
    qall = apply(dep, p)
    res.append(common.score("L0 (a)+(c)+(d)", dep, qall))
    res.append(common.score("L0 (a)+(c)+(d), floor 60 everywhere", dep, np.maximum(qall, 60)))
    print("2025 violations of the base:", violations(dep, p, 0.0))
    fin = common.load_final()
    zero = np.zeros(len(fin))
    print("2026 rows exposed (rule could bind): a", int((fin.nm & np.isfinite(fin.ml)).sum()))
    if len(sys.argv) > 1:
        json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)
