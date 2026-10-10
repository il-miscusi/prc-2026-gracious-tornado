"""Grade-A ADS-B pushback blend, the last post layer (lever 1 of research/final/headroom/RESULTS.md).

A grade-A ADS-B v3 event means the aircraft was seen parked and then seen leaving the stand within
60 s. ``a3_obs_leave`` is that leave time as seconds before takeoff, so it is already in y units
(y = takeoff - off-block; the event estimates off-block). The layer moves f5's prediction part of
the way toward it, by an amount that depends on how far apart the two are.

Specification (fixed from the headroom script before the rehearsal gate was run):

  rows     a3_grade == 3 and 0 <= a3_obs_leave <= max(d, 3,600 s). Grades B, C and D never move
           (as blends they added nothing in 2025).
  groups   d <= 3,600 s and d > 3,600 s, fitted separately.
  event    y_e = a3_obs_leave + offset[group, airport]; offset = median(y - a3_obs_leave) over the
           fit rows of that group and airport when there are at least 50 of them, else the group's
           median over all airports. (The headroom script had no minimum; it binds only at LFPG, which
           has 12 short and 4 long grade-A rows in 2025, long median 945 s, and 0.1% grade-A rows in
           2026. Set before the rehearsal gate was run.)
  bins     |y_e - y_hat| cut at 60 / 180 / 600 / 1,800 s (five bins).
  weight   w[group, bin] = sum (y_e - y_hat)(y - y_hat) / sum (y_e - y_hat)^2 on the fit rows,
           clipped to [0, 1].
  blend    y_hat' = y_hat + w (y_e - y_hat).
  bounds   on the moved rows only, in f5's order after its level step: the assembler's cap
           y_hat' <= d where d > 2 h; lv_rules (a) + (d); the recovered-orphan window (L0e); floor
           60 s. The assembler's rules step (NM window +-3,631 s, LEMD/LEBL 3,604 s, floor 60 s)
           runs after this layer, as in f5. Rows that do not move keep their value bit for bit.
  fit rows the base pipeline's own 2025 rehearsal rows (pseudo_final.py run output) with
           ``boardlike`` set (no date-wrap rows, no short-delay y > 3 h artefacts), the column that
           enters this layer (``post4_recovered`` for f5: placed after the second recovered layer,
           so the L1a-v2 loadings need no refit), and the 2025 phys file.

Modes:
  final   ``artifact``: offsets and weights fitted on all four 2025 months (``events.py artifact``).
  pseudo  ``held_month=(2025, m)`` (the config's "pass_held_month"): fitted on the rows file's
          other months; "rows" comes from the pseudo config's post_layer_pseudo kwargs. The month's
          own base values must equal the rows file's column (``strict``), so the fit rows describe
          the same pipeline.

    python3 research/final/levers/events.py artifact --rows submissions/final/pseudo2025/f5_pseudo2025_frozen.parquet \\
        --out research/final/levers/artifacts/events_f5.json
    python3 research/final/levers/events.py share      # 2026 grade-A rows by airport x month
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import lv_common as common
    from . import lv_rules as lvr
    from . import recovered as rec
except ImportError:  # loaded by file path (assembler) or run as a script from this folder
    import lv_common as common  # type: ignore
    import lv_rules as lvr  # type: ignore
    import recovered as rec  # type: ignore

H = 3600.0
EDGES = (60.0, 180.0, 600.0, 1800.0)
GROUPS = ("short", "long")  # d <= 1 h, d > 1 h
GRADE_A = 3.0
MIN_OFFSET_ROWS = 50  # an airport offset needs this many fit rows in its group; else the group median
PHYS = {2025: "research/final/inputs/2025/phys.parquet", 2026: "research/final/inputs/final/phys.parquet"}
COL = "post4_recovered"


def read_phys(ids: np.ndarray, phys: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """a3_grade and a3_obs_leave for ``ids``; every id must be in the file."""
    p = pd.read_parquet(common.resolve(phys), columns=["MVT_ID_mvt", "a3_grade", "a3_obs_leave"])
    p.index = p["MVT_ID_mvt"].round().astype("int64")
    assert p.index.is_unique, phys
    miss = ~np.isin(ids, p.index.to_numpy())
    assert not miss.any(), f"{phys}: {int(miss.sum())} frame ids missing"
    p = p.reindex(ids)
    return p["a3_grade"].to_numpy(float), p["a3_obs_leave"].to_numpy(float)


def eligible(d: np.ndarray, grade: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows the layer may move, and their group (0 = d <= 1 h, 1 = d > 1 h)."""
    with np.errstate(invalid="ignore"):
        ok = (grade == GRADE_A) & np.isfinite(obs) & (obs >= 0) & (obs <= np.maximum(d, H))
    return ok, (d > H).astype(int)


def bins_of(gap: np.ndarray) -> np.ndarray:
    return np.digitize(np.abs(gap), EDGES)


def event_y(apt: np.ndarray, obs: np.ndarray, grp: np.ndarray, params: dict) -> np.ndarray:
    out = np.full(len(obs), np.nan)
    for g, name in enumerate(GROUPS):
        k = grp == g
        if not k.any():
            continue
        tab = params["groups"][name]
        off = pd.Series(tab["offsets"], dtype=float).reindex(apt[k]).to_numpy(float)
        out[k] = obs[k] + np.where(np.isfinite(off), off, tab["offset_fallback"])
    return out


def fit(d, apt, y, base, grade, obs) -> dict:
    """Offsets and weights on fit rows (callers pass boardlike rows of the training months only)."""
    ok, grp = eligible(d, grade, obs)
    out = {"edges": list(EDGES), "groups": {}}
    for g, name in enumerate(GROUPS):
        k = ok & (grp == g)
        res = y[k] - obs[k]
        cnt = pd.Series(apt[k]).value_counts().sort_index()
        off = pd.Series(res).groupby(apt[k]).median()
        tab = {"rows": int(k.sum()), "min_offset_rows": MIN_OFFSET_ROWS,
               "offsets": {a: float(v) for a, v in off.sort_index().items() if cnt[a] >= MIN_OFFSET_ROWS},
               "offset_rows": {a: int(v) for a, v in cnt.items()},
               "offset_fallback": float(np.median(res)) if k.any() else 0.0}
        out["groups"][name] = tab
    ye = event_y(apt, obs, grp, out)
    for g, name in enumerate(GROUPS):
        k = ok & (grp == g)
        dx, r = ye[k] - base[k], y[k] - base[k]
        b = bins_of(dx)
        w_raw, n = [], []
        for j in range(len(EDGES) + 1):
            s = b == j
            w_raw.append(float((dx[s] * r[s]).sum() / max(float((dx[s] ** 2).sum()), 1e-9)))
            n.append(int(s.sum()))
        out["groups"][name].update({"weights": [float(np.clip(w, 0.0, 1.0)) for w in w_raw], "weights_raw": w_raw,
                                    "bin_rows": n})
    return out


def blend(d, apt, base, grade, obs, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """y_hat + w (y_e - y_hat) on eligible rows; the rows whose value changes."""
    assert tuple(params["edges"]) == EDGES, "artifact bins differ from the layer's"
    ok, grp = eligible(d, grade, obs)
    q = np.asarray(base, float).copy()
    ye = event_y(apt, obs, grp, params)
    for g, name in enumerate(GROUPS):
        k = ok & (grp == g)
        if not k.any():
            continue
        w = np.asarray(params["groups"][name]["weights"], float)[bins_of(ye[k] - q[k])]
        q[k] = q[k] + w * (ye[k] - q[k])
    return q, ok & (q != base)


def bound(frame: pd.DataFrame, q: np.ndarray, cap_d_over_h: float = 2.0, floor_s: float = 60.0) -> np.ndarray:
    """The bounds f5 places after its level step, in its order: cap d (assembler step 5), lv_rules
    (a) + (d), the recovered-orphan window, floor."""
    d = frame["d"].to_numpy(float)
    q = np.where(d > cap_d_over_h * H, np.minimum(q, d), q)
    q = lvr.apply(frame, q, rules="ad")
    q = rec.apply(frame, q)
    return np.maximum(q, floor_s)


def fit_rows(rows: str | Path, col: str, phys: str | Path, months=None) -> dict:
    """Fit on a pseudo-final rows file (boardlike rows of ``months``, default all)."""
    rows = common.resolve(rows)
    t = pd.read_parquet(rows, columns=["id", "month", "apt", "d", "y", "boardlike", col])
    if months is not None:
        t = t[t.month.isin(list(months))]
    t = t[t.boardlike.to_numpy(bool)]
    grade, obs = read_phys(t.id.to_numpy("int64"), phys)
    p = fit(t.d.to_numpy(float), t.apt.to_numpy(), t.y.to_numpy(float), t[col].to_numpy(float), grade, obs)
    p["fitted_on"] = {"rows": common.rel(rows), "rows_sha256": common.sha256(rows), "col": col,
                      "phys": common.rel(common.resolve(phys)), "phys_sha256": common.sha256(common.resolve(phys)),
                      "months": sorted(int(m) for m in t.month.unique()), "fit_rows": int(len(t))}
    return p


def apply(frame: pd.DataFrame, pred: np.ndarray, artifact: str | Path | None = None, phys: str | Path | None = None,
          held_month: tuple | None = None, rows: str | Path | None = None, col: str = COL,
          cap_d_over_h: float = 2.0, strict: bool = True) -> np.ndarray:
    """Final mode: weights and offsets from ``artifact``. Pseudo-final mode (``held_month=(2025, m)``):
    refitted on ``rows`` without month m. ``phys`` defaults to the frame year's file."""
    ids = (frame["id"] if "id" in frame else frame["MVT_ID_mvt"]).round().astype("int64").to_numpy()
    years = set(frame["year"].astype(int)) if "year" in frame else set()
    base = np.asarray(pred, float)
    if held_month is not None:
        assert int(held_month[0]) == 2025 and years == {2025}, "held_month is for 2025 frames"
        assert rows is not None, "pseudo-final: the events layer is refitted without the held month (rows)"
        m = int(held_month[1])
        phys = phys or PHYS[2025]
        t = pd.read_parquet(common.resolve(rows), columns=["id", "month", "apt", "d", "y", "boardlike", col])
        params = fit_rows(rows, col, phys, months=[k for k in sorted(set(t.month.astype(int))) if k != m])
        if strict:  # the fit rows must come from this pipeline: the month's base equals the file's column
            own = t[t.month == m].set_index("id")[col].reindex(ids).to_numpy(float)
            assert np.isfinite(own).all() and np.allclose(own, base, rtol=0, atol=1e-6), \
                f"{rows}[{col}] month {m} differs from the prediction entering the events layer: rerun the base rehearsal"
    else:
        assert artifact is not None, "final mode needs the fitted artifact"
        params = json.loads(common.resolve(artifact).read_text())
        if phys is None:
            assert len(years) == 1, "frame spans several years: pass phys"
            phys = PHYS[years.pop()]
    grade, obs = read_phys(ids, phys)
    d, apt = frame["d"].to_numpy(float), frame["apt"].astype(str).to_numpy()
    q, moved = blend(d, apt, base, grade, obs, params)
    if not moved.any():
        return base.copy()
    qb = bound(frame, q, cap_d_over_h)
    out = base.copy()
    out[moved] = qb[moved]
    return out


# ------------------------------------------------------------------------------------------- CLI
def build_artifact(rows: str, col: str, phys: str, out: str) -> dict:
    art = fit_rows(rows, col, phys)
    t = pd.read_parquet(common.resolve(rows), columns=["month"])
    ms = sorted(int(m) for m in t.month.unique())
    # for reference: the weights each held month of the rehearsal uses (fitted without that month)
    art["month_out_weights"] = {f"M{m}": {g: fit_rows(rows, col, phys, [k for k in ms if k != m])["groups"][g]["weights"]
                                          for g in GROUPS} for m in ms}
    art["spec"] = ("rows a3_grade == 3 and 0 <= a3_obs_leave <= max(d, 3600); groups d <= 3600 / d > 3600; "
                   "y_e = a3_obs_leave + median(y - a3_obs_leave) by group x airport (fallback group median); "
                   "bins of |y_e - y_hat| at 60/180/600/1800 s; w = sum(dx r)/sum(dx^2) clipped to [0, 1]; "
                   "y_hat' = y_hat + w (y_e - y_hat); then cap d (d > 2 h), lv_rules (a)+(d), recovered, floor 60")
    art["script"] = f"python3 research/final/levers/events.py artifact --rows {rows} --col {col} --phys {phys} --out {out}"
    p = common.resolve(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(art, indent=1, sort_keys=True))
    return art


def share_2026(phys: str = PHYS[2026]) -> pd.DataFrame:
    """Share of 2026 final-frame rows with a usable grade-A event, by airport x month (and d group)."""
    fr = common.load_final()
    grade, obs = read_phys(fr.id.to_numpy("int64"), phys)
    d = fr.d.to_numpy(float)
    ok, grp = eligible(d, grade, obs)
    t = pd.DataFrame({"apt": fr.apt.to_numpy(), "month": fr.month.to_numpy(), "ok": ok, "A": grade == GRADE_A,
                      "okL": ok & (grp == 1), "L": d > H})
    g = t.groupby(["apt", "month"])
    return pd.DataFrame({"rows": g.size(), "grade_A": g.A.mean(), "eligible": g.ok.mean(),
                         "eligible_d_gt_1h_of_d_gt_1h": g.okL.sum() / g.L.sum().clip(lower=1)}).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("artifact", help="fit on all months of a rehearsal rows file and write the 2026 artifact")
    a.add_argument("--rows", default="submissions/final/pseudo2025/f5_pseudo2025_frozen.parquet")
    a.add_argument("--col", default=COL)
    a.add_argument("--phys", default=PHYS[2025])
    a.add_argument("--out", default="research/final/levers/artifacts/events_f5.json")
    s = sub.add_parser("share", help="2026 rows with a usable grade-A event, by airport x month")
    s.add_argument("--phys", default=PHYS[2026])
    x = ap.parse_args()
    if x.cmd == "artifact":
        art = build_artifact(x.rows, x.col, x.phys, x.out)
        p = common.resolve(x.out)
        print("wrote", common.rel(p), common.sha256(p))
        for g in GROUPS:
            t = art["groups"][g]
            print(f"  {g}: {t['rows']:,} rows, weights {np.round(t['weights'], 3).tolist()} (raw "
                  f"{np.round(t['weights_raw'], 3).tolist()}, bin rows {t['bin_rows']}), offset fallback "
                  f"{t['offset_fallback']:.1f}, offsets {({k: round(v, 1) for k, v in t['offsets'].items()})}")
        print("  month-out weights:", json.dumps({k: {g: np.round(v, 3).tolist() for g, v in w.items()}
                                                 for k, w in art["month_out_weights"].items()}))
    else:
        t = share_2026(x.phys)
        print(t.round(4).to_string(index=False))
        print(t.assign(n=t.rows * t.eligible).groupby("month").n.sum() / t.groupby("month").rows.sum())


if __name__ == "__main__":
    main()
