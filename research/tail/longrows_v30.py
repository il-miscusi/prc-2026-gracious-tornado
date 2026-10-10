"""v30 long-row rebuild (WS-D stage 1): > 4 h rows from old + current members, cap at d past 2 h.

    python3 research/tail/longrows_v30.py BASE.parquet OUT.parquet [--alpha 0.5] [--phys v29|plog|klog]
    python3 research/tail/longrows_v30.py --eval [--alpha 0.5] [--phys v29|plog|klog]

BASE is v29 (submissions/gracious-tornado_v29.parquet). Only rows with d_sched > 2 h can change:

1. Rows with d_sched > 4 h that are not LIRF-unmatched (schedule-fill candidates):
   M = alpha * M_new + (1 - alpha) * M_old, where M_old is v29's own pre-blend long stack (v18's
   members; v21's value on these rows, byte-identical to v18) and M_new is the current members
   (research/oofframe.py BAND: c6s, c5s, x5, c6, cs5, split-fs4; seed-averaged refits) blended with
   NNLS weights fitted on board-like, NON-FILL 1-4 h OOF rows (folds A+B pooled). Then the physical
   estimate is blended in: --phys v29 keeps v29's board-tested band weights (0.11 at 4-8 h, 0.695
   past 8 h); plog / klog fit w(d) = a + b log(d / 4 h) pooled / per estimator kind (AOBT_3 lag,
   ADS-B, airport median) on the OOF rows (cross-fitted in --eval, pooled A+B for the candidate).
2. LIRF-unmatched rows past 4 h keep v29's value (their recording mode is WS-D stage 2's job).
3. Every row with d_sched > 2 h is capped at d (y > d + 120 s on 13 of 39,797 such 2025 rows).

--eval replays the same construction on A/B OOF against the v29 replay (research/tail/oofframe.py):
weights fitted on the other fold, estimator tables from 2025 rows outside the scored fold's
half-months (fixes extreme_blend.py's in-sample medians), deltas in all-row MSE units of the scored
fold, full risk and board-like, with a paired airport-day bootstrap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oofframe import BAND, OLD, fold_held  # noqa: E402
from tailkit import H, apply_wlog, fit_wlog, is_fill, kind, nnls_w, phys  # noqa: E402

C = Path(__file__).resolve().parents[2]
K = C / "solutions" / "kernels"
TAIL = C / "research" / "tail" / "cache"
SUB = C / "submissions"
NF = {"A": 177527, "B": 166892}  # 2025 fold sizes (A/B OOF rows), fixed by the folds
TEMPLATE = C / "data" / "09-22-2026-08-13-30_files_list" / "submitting.parquet"


def n_rows() -> int:
    """Rows of the ranking template (344,841 for Jan + Jul 2026; a new month has its own count)."""
    return len(pd.read_parquet(TEMPLATE, columns=["MVT_ID_mvt"]))
V29W = ((4, 8, 0.11), (8, 1e9, 0.695))


def v29w(d):
    return np.where(d > 8 * H, 0.695, np.where(d > 4 * H, 0.11, 0.0))


def new_weights(df, rows):
    P = df.loc[rows, [f"p_{r}" for r in BAND]].to_numpy()
    return nnls_w(P, df.y.to_numpy()[rows])


def blend_phys(mode, M, E, d, k, fit=None):
    """fit = (y, M, E, d, k) rows used for plog / klog."""
    if mode == "v29":
        w = v29w(d)
        return w * E + (1 - w) * M, w
    kk = k if mode == "klog" else np.full(len(k), "N")
    fy, fM, fE, fd, fk = fit
    fk = fk if mode == "klog" else np.full(len(fk), "N")
    coef = fit_wlog(fy, fM, fE, fd, fk)
    return apply_wlog(coef, M, E, d, kk)


# --------------------------------------------------------------------------------------------- eval
def evaluate(alpha, mode, keep_over=1e9, quiet=False):
    df = pd.read_parquet(TAIL / "long_oof.parquet")
    tr = pd.read_parquet(TAIL / "long_tr.parquet").set_index("MVT_ID_mvt")
    d, y, bl = df.ds.to_numpy(), df.y.to_numpy(), df.boardlike.to_numpy()
    fill, k = is_fill(df), kind(df)
    guard = (df.d_aobt3 > keep_over * H).to_numpy()
    E = np.full(len(df), np.nan)
    for f in "AB":
        s = (df.fold == f).to_numpy()
        E[s] = phys(df[s], tr[~fold_held(tr, f)])
    pred = df.v29.to_numpy().copy()
    for f in "AB":
        trn, tst = (df.fold != f).to_numpy() & bl, (df.fold == f).to_numpy()
        Po = df[[f"p_{r}" for r in OLD]].to_numpy()
        Mo = Po @ nnls_w(Po[trn & (d > H)], y[trn & (d > H)])
        Mn = df[[f"p_{r}" for r in BAND]].to_numpy() @ new_weights(df, trn & (d > H) & (d <= 4 * H) & ~fill)
        M = alpha * Mn + (1 - alpha) * Mo
        nf = tst & (d > 4 * H) & ~fill & ~guard
        fr = trn & (d > 4 * H) & ~fill & ~guard
        q, _ = blend_phys(mode, M[nf], E[nf], d[nf], k[nf], (y[fr], M[fr], E[fr], d[fr], k[fr]))
        pred[nf] = np.clip(q, 60, None)
    cap = d > 2 * H
    pred[cap] = np.minimum(pred[cap], d[cap])
    e1, e0 = (y - pred) ** 2, (y - df.v29.to_numpy()) ** 2
    blocks = (df.apt + df.day).to_numpy()
    out = []
    for f in "AB":
        for seg, m in (("all >2h", d > 2 * H), ("2-4h", (d > 2 * H) & (d <= 4 * H)), (">4h", d > 4 * H),
                       (">4h non-fill", (d > 4 * H) & ~fill), ("4-8h non-fill", (d > 4 * H) & (d <= 8 * H) & ~fill),
                       (">8h non-fill", (d > 8 * H) & ~fill), (">4h LIRF-unmatched", (d > 4 * H) & fill)):
            s = (df.fold == f).to_numpy() & m
            sb = s & bl
            dd = (e1 - e0)
            # paired bootstrap over airport-days of the board-like delta, in fold all-row units
            codes, uniq = pd.factorize(blocks[sb])
            sums = np.bincount(codes, weights=dd[sb])
            rng = np.random.default_rng(0)
            boot = sums[rng.integers(0, len(uniq), size=(2000, len(uniq)))].sum(1) / NF[f] if len(uniq) else np.zeros(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out.append(dict(fold=f, seg=seg, rows=int(s.sum()), full=dd[s].sum() / NF[f], board=dd[sb].sum() / NF[f],
                            ci=f"[{lo:+.1f}, {hi:+.1f}]"))
    rk = pd.read_parquet(TAIL / "rk_all.parquet")
    rk = rk[(rk.d_sched > 4 * H) & ~is_fill(rk) & ~(rk.d_aobt3 > keep_over * H)]
    n26 = rk.groupby([rk.apt, np.where(rk.d_sched > 8 * H, ">8h", "4-8h")]).size()
    for f in "AB":
        s = (df.fold == f).to_numpy() & bl & (d > 4 * H) & ~fill
        cell = pd.Series((e1 - e0)[s]).groupby([df.apt.to_numpy()[s], np.where(d[s] > 8 * H, ">8h", "4-8h")]).mean()
        com = n26.index.intersection(cell.index)
        out.append(dict(fold=f, seg="2026-mix proj >4h non-fill", rows=int(n26[com].sum()),
                        full=np.nan, board=float((cell[com] * n26[com]).sum() / n_rows()), ci=""))
    t = pd.DataFrame(out)
    if not quiet:
        pd.set_option("display.width", 200)
        print(f"alpha {alpha}  phys {mode}: dMSE vs v29 replay (all-row units of the scored fold; negative = better)")
        print(t.round(1).to_string(index=False))
    return t, pd.Series(pred, index=df.index)


# -------------------------------------------------------------------------------------------- build
def sub(path):
    s = pd.read_parquet(path)
    s.index = s.MVT_ID_mvt.astype("int64")
    assert s.index.is_unique
    return s.TAXITIME_SEC_mvt


def build(base_path, out_path, alpha, mode, keep_over=1e9, old_path=None):
    df = pd.read_parquet(TAIL / "long_oof.parquet")
    tr = pd.read_parquet(TAIL / "long_tr.parquet")
    rk = pd.read_parquet(TAIL / "rk_all.parquet").set_index("MVT_ID_mvt")
    base_raw = pd.read_parquet(base_path)
    base = sub(base_path)
    N = n_rows()
    assert len(base) == N and set(base.index) == set(rk.index)
    rk = rk.reindex(base.index)
    d = rk.d_sched.to_numpy()
    fill, k = is_fill(rk), kind(rk)
    old = sub(old_path or SUB / "gracious-tornado_v21.parquet").reindex(base.index).to_numpy()
    bl, y, ds = df.boardlike.to_numpy(), df.y.to_numpy(), df.ds.to_numpy()
    w = new_weights(df, bl & (ds > H) & (ds <= 4 * H) & ~is_fill(df))
    S = np.column_stack([sub(K / r / "output" / "submission.parquet").reindex(base.index).to_numpy() for r in BAND])
    assert np.isfinite(S).all()
    M = alpha * (S @ w) + (1 - alpha) * old
    E = phys(rk, tr)
    nf = (d > 4 * H) & ~fill & ~(rk.d_aobt3 > keep_over * H).to_numpy()
    fit = None
    if mode != "v29":  # pooled A+B fit on OOF, E with fold-excluded tables, M with pooled weights
        trx = tr.set_index("MVT_ID_mvt")
        Eo = np.full(len(df), np.nan)
        for f in "AB":
            s = (df.fold == f).to_numpy()
            Eo[s] = phys(df[s], trx[~fold_held(trx, f)])
        Po = df[[f"p_{r}" for r in OLD]].to_numpy()
        Mo = Po @ nnls_w(Po[bl & (ds > H)], y[bl & (ds > H)])
        Mf = alpha * (df[[f"p_{r}" for r in BAND]].to_numpy() @ w) + (1 - alpha) * Mo
        fr = bl & (ds > 4 * H) & ~is_fill(df)
        fit = (y[fr], Mf[fr], Eo[fr], ds[fr], kind(df)[fr])
    q, wt = blend_phys(mode, M[nf], E[nf], d[nf], k[nf], fit)
    out = base.to_numpy().copy()
    out[nf] = np.clip(q, 60, None)
    cap = d > 2 * H
    out[cap] = np.minimum(out[cap], d[cap])
    res = pd.DataFrame({"MVT_ID_mvt": base_raw.MVT_ID_mvt.to_numpy(), "TAXITIME_SEC_mvt": out})
    assert len(res) == N and res.MVT_ID_mvt.is_unique and np.isfinite(res.TAXITIME_SEC_mvt).all()
    assert (res.TAXITIME_SEC_mvt >= 60).all()
    ch = out - base.to_numpy()
    changed = ch != 0
    assert not changed[d <= 2 * H].any(), "rows at or under 2 h changed"
    res.to_parquet(out_path, index=False)
    print("new-member weights (1-4 h non-fill NNLS)", dict(zip(BAND, np.round(w, 3))))
    print(f"rows changed {int(changed.sum())} (>4 h non-fill {int((changed & nf).sum())}, capped {int((changed & ~nf).sum())})"
          f"  RMS change on >4 h non-fill {np.sqrt((ch[nf] ** 2).mean()):.0f} s  mean {ch[nf].mean():+.0f} s"
          f"  sum sq change / N {float((ch ** 2).sum() / len(ch)):.0f}  -> {out_path}")
    t = pd.DataFrame({"apt": rk.apt.to_numpy()[nf], "band": np.where(d[nf] > 8 * H, ">8h", "4-8h"),
                      "v29": base.to_numpy()[nf], "new": out[nf]})
    print(t.groupby(["band", "apt"]).agg(n=("new", "size"), v29=("v29", "mean"), new=("new", "mean")).round(0).to_string())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--phys", default="v29", choices=["v29", "plog", "klog"])
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--keep-aobt3-over", type=float, default=1e9,
                    help="hours: rows whose MVT - AOBT_3 exceeds this keep the base value (recording-mode rows)")
    ap.add_argument("--old", help="v21 submission (the pre-blend long stack M_old); default submissions/gracious-tornado_v21.parquet")
    a = ap.parse_args()
    if a.eval:
        evaluate(a.alpha, a.phys, a.keep_aobt3_over)
    else:
        build(a.base, a.out, a.alpha, a.phys, a.keep_aobt3_over, a.old)


if __name__ == "__main__":
    main()
