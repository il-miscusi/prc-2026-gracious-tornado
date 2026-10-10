"""Paired pseudo-final comparison of every meta run against the current stack (f2) in the same mode.

    python3 research/final/meta/compare_all.py NAME [NAME ...] [--base f2_pseudo2025]

For each NAME and mode (crossfit, frozen) present under submissions/final/meta/pseudo2025/, runs
research/final/pseudo_final.compare(NAME_mode, BASE_mode) (airport-day block bootstrap, 2,000 draws)
and writes research/final/meta/pseudo/compare_<NAME>_<mode>__<BASE>_<mode>.json; prints one line per
pair: all rows, nw (2025 date-wrap rows excluded), months (month units), bands (all-row units), the
nw split d <= 4 h / d > 4 h, nw without LTFM February and without February, and the gate verdict on
--scope (default nw): dMSE < 0 with the CI below 0, no month's CI entirely above 0.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pseudo_final  # noqa: E402
from frame import C  # noqa: E402

OUT = C / "submissions/final/meta/pseudo2025"
SCOPE = "nw"  # gate scope: "nw" (without the 2025 date-wrap rows) or "all"; --scope


def fmt(v):
    return f"{v['dmse']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"


def seg_rows(fa: Path, fb: Path, r: dict) -> dict:
    """dMSE restricted to the rows the meta member replaces (d <= 4 h) and to d > 4 h, all-row units,
    without the 2025 date-wrap rows: the long rows move only through L1a's day sums."""
    a, b = pd.read_parquet(fa), pd.read_parquet(fb)
    flags = pseudo_final.row_flags(sorted(int(m) for m in a.month.unique())).reindex(a.id.to_numpy())
    nw = ~flags.wrap.to_numpy().astype(bool)
    dd = ((a.y - a.pred) ** 2 - (b.y - b.pred) ** 2).to_numpy()[nw]
    blocks = (a.apt + "|" + a.day).to_numpy()[nw]
    d, mon, n = a.d.to_numpy()[nw], a.month.to_numpy()[nw], int(nw.sum())
    out = {}
    noltfm2 = ~((a.apt.to_numpy()[nw] == "LTFM") & (mon == 2))
    for name, k in (("nw d<=4h", d <= 4 * 3600), ("nw d>4h", d > 4 * 3600)):
        lo, hi = pseudo_final.boot(np.where(k, dd, 0.0), blocks, n)
        out[name] = {"dmse": float(dd[k].sum() / n), "ci95": [lo, hi], "month": {}}
        for m in sorted(set(mon)):
            km = mon == m
            lo2, hi2 = pseudo_final.boot(np.where(k[km], dd[km], 0.0), blocks[km], int(km.sum()))
            out[name]["month"][int(m)] = {"dmse": float(dd[km & k].sum() / km.sum()), "ci95": [lo2, hi2]}
    for name, k in (("nw without LTFM Feb", noltfm2), ("nw without Feb", mon != 2)):
        lo, hi = pseudo_final.boot(dd[k], blocks[k], int(k.sum()))
        out[name] = {"dmse": float(dd[k].sum() / k.sum()), "ci95": [lo, hi], "month": {}}
        for m in sorted(set(mon[k])):
            km = k & (mon == m)
            lo2, hi2 = pseudo_final.boot(dd[km], blocks[km], int(km.sum()))
            out[name]["month"][int(m)] = {"dmse": float(dd[km].sum() / km.sum()), "ci95": [lo2, hi2]}
    return out


def gate(r: dict) -> str:
    p = r[SCOPE]["pooled"]
    ok = p["dmse"] < 0 and p["ci95"][1] < 0
    worse = [m for m, v in r[SCOPE]["month"].items() if v["ci95"][0] > 0]
    return ("PASS" if ok and not worse else "FAIL") + (f" (months worse: {worse})" if worse else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="+")
    ap.add_argument("--base", default="f2_pseudo2025")
    ap.add_argument("--scope", choices=["nw", "all"], default="nw")
    ap.add_argument("--modes", default="crossfit,frozen")
    a = ap.parse_args()
    global SCOPE
    SCOPE = a.scope
    for name in a.names:
        for mode in a.modes.split(","):
            fa, fb = OUT / f"{name}_{mode}.parquet", OUT / f"{a.base}_{mode}.parquet"
            if not (fa.exists() and fb.exists()):
                continue
            js = HERE / "pseudo" / f"compare_{name}_{mode}__{a.base}_{mode}.json"
            with contextlib.redirect_stdout(io.StringIO()):
                r = pseudo_final.compare(str(fa.relative_to(C)), str(fb.relative_to(C)), str(js))
            al, nw = r["all"], r["nw"]
            print(f"{name} [{mode}] vs {a.base}: all {fmt(al['pooled'])}  nw {fmt(nw['pooled'])}  gate[{SCOPE}] {gate(r)}")
            print("    nw months: " + "  ".join(f"M{k} {fmt(v)}" for k, v in nw["month"].items()))
            print("    months: " + "  ".join(f"M{k} {fmt(v)}" for k, v in al["month"].items()))
            print("    bands:  " + "  ".join(f"{k} {fmt(v)}" for k, v in al["band"].items()))
            print("    airports: " + "  ".join(f"{k} {v:+.1f}" for k, v in al["airport"].items()))
            r["segment_split"] = seg_rows(fa, fb, r)
            for k, v in r["segment_split"].items():
                print(f"    {k}: {fmt(v)}   months: " + "  ".join(f"M{m} {fmt(x)}" for m, x in v["month"].items()))
            js.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
