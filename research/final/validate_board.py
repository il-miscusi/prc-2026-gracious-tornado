"""Validation of the final assembler against the v32c reproduction and on 2025 A/B OOF.

    python3 research/final/validate_board.py legacy-long     # longrows.py reproduces the v30 long-row artifact
    python3 research/final/validate_board.py components A B  # per-component RMS between two stage files
    python3 research/final/validate_board.py alpha           # alpha = 1 vs 0.5 long rows on A/B OOF
    python3 research/final/validate_board.py fill            # LIRF-unmatched > 4 h: which member stack
    python3 research/final/validate_board.py weights         # kept-member weights vs v32c weights on OOF

Light arithmetic on cached OOF and submission files; nothing is trained.
All OOF deltas are SSE differences divided by the fold's OOF row count (A 177,527, B 166,892), i.e.
all-row MSE units of the scored fold; negative = better than the reference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import longrows  # noqa: E402
from frame import C, ID, TARGET, load_frame  # noqa: E402
from stacking import crossfit, fit_weights, member_pred, oof_matrix, oof_meta  # noqa: E402
from tailkit import is_fill, nnls_w  # noqa: E402

H = 3600.0
NF = {"A": 177527, "B": 166892}
K = "solutions/kernels/{}/output"
BAND = ["prc-c6s", "prc-c5s", "prc-x5", "prc-c6", "prc-cs5", "prc-split-fs4"]
OLD = ["prc-c5d10", "prc-cat-fs4", "prc-cs5", "prc-xgb-fs4", "prc-x5", "prc-split-fs4"]
V32C = json.loads((HERE / "configs" / "v32c_repro_board.json").read_text())


def specs(names):
    return {n: {"pred": K.format(n), "oof": K.format(n)} for n in names}


def sub(path) -> pd.Series:
    s = pd.read_parquet(C / path)
    s.index = s[ID].round().astype("int64")
    return s[TARGET].astype(float)


def boot_ci(dd: np.ndarray, blocks: np.ndarray, n: int) -> str:
    codes, uniq = pd.factorize(blocks)
    sums = np.bincount(codes, weights=dd)
    rng = np.random.default_rng(0)
    b = sums[rng.integers(0, len(uniq), size=(2000, len(uniq)))].sum(1) / n
    lo, hi = np.percentile(b, [2.5, 97.5])
    return f"[{lo:+.1f}, {hi:+.1f}]"


# ------------------------------------------------------------------------------------------ legacy
def legacy_long():
    """Rebuild research/tail/out/v30_longrows_a05_plog.parquet (alpha 0.5, M_old = v21) with longrows.py."""
    m = oof_meta()
    members = list(dict.fromkeys(BAND + OLD))
    sp = specs(members)
    tr = longrows.long_tr()
    df = longrows.long_oof(members, sp, m, tr)
    w = longrows.stack_weights(df, BAND)
    bl, ds, y = df.boardlike.to_numpy(), df.ds.to_numpy(), df.y.to_numpy()
    Po = df[[f"p_{r}" for r in OLD]].to_numpy()
    Mo = Po @ nnls_w(Po[bl & (ds > H)], y[bl & (ds > H)])
    Mf = 0.5 * (df[[f"p_{r}" for r in BAND]].to_numpy() @ w) + 0.5 * Mo
    coef = longrows.fit_plog(df, Mf, tr)
    fr = load_frame("board")
    ids = fr.id.to_numpy()
    S = np.column_stack([member_pred(sp[r]["pred"], ids) for r in BAND])
    pf = fr[["apt", "d_sched", "unmatched", "d_aobt3"]].copy()
    if "--f32" in sys.argv:  # the v30 script read d and MVT - AOBT_3 as float32 from the fs5 store
        pf["d_sched"], pf["d_aobt3"] = pf.d_sched.astype("float32"), pf.d_aobt3.astype("float32")
    rk = pd.read_parquet(C / "research/tail/cache/rk_all.parquet").set_index("MVT_ID_mvt").reindex(ids)
    pf["a3_grade"], pf["a3_obs_leave"] = rk.a3_grade.to_numpy(float), rk.a3_obs_leave.to_numpy(float)
    M = 0.5 * (S @ w) + 0.5 * sub("submissions/gracious-tornado_v21.parquet").reindex(ids).to_numpy()
    nf, q, E, wt, kinds = longrows.construct(pf, M, coef, tr, guard_h=4.0)
    d = fr.d.to_numpy()
    out = sub("submissions/gracious-tornado_v29.parquet").reindex(ids).to_numpy()
    out[nf] = q
    cap = d > 2 * H
    out[cap] = np.minimum(out[cap], d[cap])
    ref = sub("research/tail/out/v30_longrows_a05_plog.parquet").reindex(ids).to_numpy()
    L = d > 4 * H
    diff = out - ref
    print(f"long-member weights {dict(zip(BAND, np.round(w, 4)))}; plog a,b {np.round(coef['N'], 5).tolist()}")
    print(f"v30 artifact vs longrows.py (alpha 0.5, M_old = v21, guard 4 h): rows > 4 h {int(L.sum())}, blended {int(nf.sum())}; "
          f"max |diff| {np.abs(diff[L]).max():.2e} s on > 4 h rows, {np.abs(diff).max():.2e} on all rows")


# -------------------------------------------------------------------------------------- components
def components(a: str, b: str):
    """RMS difference between two assemblies' stage files, split by the step that touched each row."""
    sa, sb = pd.read_parquet(C / a), pd.read_parquet(C / b)
    assert (sa.id.to_numpy() == sb.id.to_numpy()).all()
    fr = load_frame("board")
    d = sa.d.to_numpy()
    fill = is_fill(fr) & (d > 4 * H)
    ovl = (d <= 4 * H) & (np.abs(sb["overlay"].to_numpy() - sb["fill"].to_numpy()) > 1e-9)
    groups = {"segments d <= 4 h (no overlay)": (d <= 4 * H) & ~ovl, "overlay rows": ovl,
              "long > 4 h non-fill": (d > 4 * H) & ~fill, "long > 4 h LIRF-unmatched": fill}
    print(f"{'stage':10s} " + " ".join(f"{g:>32s}" for g in groups) + f" {'all rows':>12s}")
    for st in ["segments", "long", "fill", "overlay", "cap", "level", "rules"]:
        if st not in sa or st not in sb:
            continue
        dv = np.nan_to_num(sa[st].to_numpy() - sb[st].to_numpy())  # rows not yet set in either stage: 0
        cells = [f"{np.sqrt(np.mean(dv[k] ** 2)):9.3f} s ({int((np.abs(dv[k]) > 1e-6).sum()):>6,} rows)" for k in groups.values()]
        print(f"{st:10s} " + " ".join(f"{c:>32s}" for c in cells) + f" {np.sqrt(np.mean(dv ** 2)):10.3f} s")
    lv_a = sa["level"].to_numpy() - sa["cap"].to_numpy()
    lv_b = sb["level"].to_numpy() - sb["cap"].to_numpy()
    print(f"level shift itself: RMS of (shift_a - shift_b) {np.sqrt(np.mean((lv_a - lv_b) ** 2)):.4f} s")
    rr = sa["rules"].to_numpy() - sa["post"].to_numpy() if "post" in sa else None
    if rr is not None:
        k = np.abs(rr) > 1e-9
        print(f"rule package in {a}: {int(k.sum())} rows moved, RMS move on them {np.sqrt(np.mean(rr[k] ** 2)) if k.any() else 0:.1f} s, "
              f"sum sq / N {float((rr ** 2).sum() / len(rr)):.1f}")


# --------------------------------------------------------------------------------------------- alpha
def eval_long(members, alpha_old=0.0, keep_fill_v29=True):
    """Cross-fitted A<->B replay of the long-row construction on OOF rows past 2 h (as
    research/tail/longrows_v30.py --eval), returning predictions on the long OOF frame."""
    m = oof_meta()
    names = list(dict.fromkeys(members + OLD))
    sp = specs(names)
    tr = longrows.long_tr()
    df = longrows.long_oof(names, sp, m, tr)
    df["v29"] = v29_replay().reindex(df.index).to_numpy()
    assert df.v29.notna().all()
    d, y, bl = df.ds.to_numpy(), df.y.to_numpy(), df.boardlike.to_numpy()
    fill = is_fill(df)
    E = longrows.oof_phys(df, tr)
    pred = df.v29.to_numpy().copy()
    from tailkit import apply_wlog, fit_wlog
    for f in "AB":
        trn, tst = (df.fold != f).to_numpy() & bl, (df.fold == f).to_numpy()
        P = df[[f"p_{r}" for r in members]].to_numpy()
        w = nnls_w(P[trn & (d > H) & (d <= 4 * H) & ~fill], y[trn & (d > H) & (d <= 4 * H) & ~fill])
        M = P @ w
        if alpha_old:
            Po = df[[f"p_{r}" for r in OLD]].to_numpy()
            M = (1 - alpha_old) * M + alpha_old * (Po @ nnls_w(Po[trn & (d > H)], y[trn & (d > H)]))
        guard = tst & (d > 4 * H) & ~fill & (df.d_aobt3 > 4 * H).to_numpy()
        nf = tst & (d > 4 * H) & ~fill & ~guard
        fr = trn & (d > 4 * H) & ~fill
        n = np.full(int(fr.sum()), "N")
        coef = fit_wlog(y[fr], M[fr], E[fr], d[fr], n)
        q, _ = apply_wlog(coef, M[nf], E[nf], d[nf], np.full(int(nf.sum()), "N"))
        pred[nf] = np.clip(q, 60, None)
        if not alpha_old:  # final pipeline: guard rows take the long-member stack (v30 kept v29 there)
            pred[guard] = np.clip(M[guard], 60, None)
    cap = d > 2 * H
    pred[cap] = np.minimum(pred[cap], d[cap])
    return df, pred


def v29_replay() -> pd.Series:
    """v29 construction replayed on the long OOF rows (research/tail/oofframe.py, cached, indexed by id)."""
    return pd.read_parquet(C / "research/tail/cache/long_oof.parquet", columns=["v29"])["v29"]


def report(df, preds: dict, ref: str):
    d, y, bl = df.ds.to_numpy(), df.y.to_numpy(), df.boardlike.to_numpy()
    fill = is_fill(df)
    blocks = (df.apt + df.day).to_numpy()
    e0 = (y - preds[ref]) ** 2
    rows = []
    for name, p in preds.items():
        if name == ref:
            continue
        dd = (y - p) ** 2 - e0
        for f in "AB":
            for seg, k in ((">4h non-fill", (d > 4 * H) & ~fill), ("4-8h non-fill", (d > 4 * H) & (d <= 8 * H) & ~fill),
                           (">8h non-fill", (d > 8 * H) & ~fill)):
                s = (df.fold == f).to_numpy() & k
                rows.append(dict(variant=name, fold=f, seg=seg, rows=int(s.sum()), full=dd[s].sum() / NF[f],
                                 board=dd[s & bl].sum() / NF[f], ci_board=boot_ci(dd[s & bl], blocks[s & bl], NF[f])))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"dMSE vs {ref} (all-row units of the scored fold):")
    print(t.round(1).to_string(index=False))


def alpha():
    kept = ["prc-c6s", "prc-c5s", "prc-x5", "prc-c7x2"]
    df, p05 = eval_long(BAND, alpha_old=0.5)
    _, p10 = eval_long(BAND)
    df2, p10k = eval_long(kept)
    assert (df2.index == df.index).all()
    report(df, {"v30 (alpha .5, BAND)": p05, "alpha 1, BAND": p10, "alpha 1, kept (c6s c5s x5 c7x2)": p10k,
                "v29 replay": df.v29.to_numpy()}, "v30 (alpha .5, BAND)")


# ---------------------------------------------------------------------------------------------- fill
def fill():
    """LIRF-unmatched OOF rows past 4 h: v29 replay (OLD stack) vs the 1-4 h segment stack vs the long stack."""
    m = oof_meta()
    seg = [s for s in V32C["segments"] if s["name"] == "1-4h"][0]["members"]
    names = list(dict.fromkeys(BAND + OLD + seg))
    sp = specs(names)
    df = longrows.long_oof(names, sp, m)
    df["v29"] = v29_replay().reindex(df.index).to_numpy()
    d, y, bl = df.ds.to_numpy(), df.y.to_numpy(), df.boardlike.to_numpy()
    fl = is_fill(df) & (d > 4 * H)
    out = {"v29 (OLD stack, all long rows)": df.v29.to_numpy().copy()}
    for name, mem, rows in (("1-4h segment stack (all board-like 1-4 h)", seg, lambda tr: tr & (d > H) & (d <= 4 * H)),
                            ("long stack (non-fill 1-4 h)", BAND, lambda tr: tr & (d > H) & (d <= 4 * H) & ~is_fill(df)),
                            ("1-4h segment members fitted on LIRF-unmatched > 1 h", seg, lambda tr: tr & (d > H) & is_fill(df))):
        P = df[[f"p_{r}" for r in mem]].to_numpy()
        p = np.full(len(df), np.nan)
        for f in "AB":
            trn, tst = (df.fold != f).to_numpy() & bl, (df.fold == f).to_numpy()
            p[tst] = P[tst] @ nnls_w(P[rows(trn)], y[rows(trn)])
        out[name] = p
    lirf_orphan_band = fl & (d >= 24000) & (d < 86400)  # unmatched == AOBT_3 missing; the orphan band uses FLIGHT_ID
    print(f"LIRF-unmatched OOF rows past 4 h: {int(fl.sum())} (board-like {int((fl & bl).sum())}); "
          f"in the 24,000-86,400 s band {int(lirf_orphan_band.sum())}; y >= d - 60 on {int((y[fl] >= d[fl] - 60).sum())}")
    out["y_hat = d (reference)"] = d.copy()
    from tailkit import fill_prob  # p(fill | LIRF-unmatched, d) logistic in log d; tables outside the scored fold
    tr = longrows.long_tr()
    pm = np.full(len(df), np.nan)
    for f in "AB":
        s = (df.fold == f).to_numpy() & fl
        p_f, e0 = fill_prob(tr[~longrows.fold_held(tr, f)], d[s])
        pm[s] = p_f * d[s] + (1 - p_f) * e0
    out["p(d) d + (1 - p(d)) median non-fill y (2025, fold-excluded)"] = np.where(fl, pm, np.nan)
    for name, p in out.items():
        for band_rule in ("max(q, d)", "= d"):
            q = np.clip(p, 60, None)
            q = np.where(lirf_orphan_band, np.maximum(q, d) if band_rule == "max(q, d)" else d, np.minimum(q, d))
            e = (y - q) ** 2
            cells = [f"{f}: all {e[fl & (df.fold == f).to_numpy()].sum() / NF[f]:8.1f} board-like "
                     f"{e[fl & bl & (df.fold == f).to_numpy()].sum() / NF[f]:7.1f}" for f in "AB"]
            print(f"  {name:52s} band {band_rule:9s} {'  '.join(cells)}")


# ------------------------------------------------------------------------------------------- weights
def weights():
    m = oof_meta()
    v32c = {s["name"]: s for s in V32C["segments"]}
    variants = {
        "short": {"v32c": v32c["short"]["members"],
                  "kept (c7 cs7 mlp7)": ["prc-c7x2", "prc-cs7", "prc-mlp7"],
                  "kept + cs6 cs5 mlp6fs": ["prc-c7x2", "prc-cs7", "prc-mlp7", "prc-cs6", "prc-cs5", "prc-mlp6fs"]},
        "1-4h": {"v32c": v32c["1-4h"]["members"],
                 "kept (c6s c5s x5 c7)": ["prc-c6s", "prc-c5s", "prc-x5", "prc-c7x2"],
                 "kept + cs5": ["prc-c6s", "prc-c5s", "prc-x5", "prc-c7x2", "prc-cs5"]},
    }
    for segname, vs in variants.items():
        seg0 = v32c[segname]
        sp = specs(list(dict.fromkeys(sum(vs.values(), []))))
        ref_seg = {**seg0}
        p_ref = crossfit(ref_seg, sp, m)  # cross-fitted v32c recipe (weights from the other fold)
        P, ids, _ = oof_matrix(seg0["members"], sp, m)
        s = (m.loc[ids, "ds"].to_numpy() > seg0["lo_h"] * H) & (m.loc[ids, "ds"].to_numpy() <= seg0["hi_h"] * H)
        y_all = m.loc[ids, "y"].to_numpy()
        w_ref = np.array([seg0["weights"][r] for r in seg0["members"]])
        mse_ref_in = np.mean((P[s] @ w_ref - y_all[s]) ** 2)
        print(f"== segment {segname}: {int(s.sum()):,} board-like OOF rows; v32c weights in-sample MSE {mse_ref_in:.1f}")
        for name, mem in vs.items():
            seg = {**seg0, "members": mem}
            Pk, idk, _ = oof_matrix(mem, sp, m)
            sk = (m.loc[idk, "ds"].to_numpy() > seg["lo_h"] * H) & (m.loc[idk, "ds"].to_numpy() <= seg["hi_h"] * H)
            wk = fit_weights(Pk[sk], m.loc[idk, "y"].to_numpy()[sk])
            ins = np.mean((Pk[sk] @ np.round(wk, 6) - m.loc[idk, "y"].to_numpy()[sk]) ** 2)
            p1 = crossfit(seg, sp, m)
            com = p1.index.intersection(p_ref.index)
            y = m.loc[com, "y"].to_numpy()
            dd = (y - p1.loc[com].to_numpy()) ** 2 - (y - p_ref.loc[com].to_numpy()) ** 2
            fold = np.where(m.loc[com, "day"].str[2:].astype(int).to_numpy() >= 16, "A", "B")
            blocks = (m.loc[com, "apt"] + m.loc[com, "day"]).to_numpy()
            cf = "  ".join(f"{f} {dd[fold == f].sum() / NF[f]:+7.1f} {boot_ci(dd[fold == f], blocks[fold == f], NF[f])}" for f in "AB")
            print(f"  {name:24s} weights {dict(zip(mem, np.round(wk, 4)))}")
            print(f"  {'':24s} in-sample dMSE vs v32c weights {(ins - mse_ref_in) * s.sum() / sum(NF.values()):+7.1f} (all-row units); "
                  f"cross-fitted A<->B dMSE vs cross-fitted v32c: {cf}")


# ---------------------------------------------------------------------------------------------- drop
def drop(member: str = "prc-x5"):
    """F0's kept recipe (old A/B runs) with and without one 1-4 h / long-stack member: new weights,
    cross-fitted A<->B dMSE on the 1-4 h segment and on the long rows (alpha 1, as F0)."""
    m = oof_meta()
    band = ["prc-c6s", "prc-c5s", "prc-x5", "prc-c7x2"]
    kept = [r for r in band if r != member]
    sp = specs(band)
    seg_with = {"name": "1-4h", "lo_h": 1, "hi_h": 4, "members": band}
    seg_wo = {**seg_with, "members": kept}
    from stacking import segment_weights
    print(f"1-4 h weights without {member}: {segment_weights(seg_wo, sp, m)}")
    p1, p0 = crossfit(seg_wo, sp, m), crossfit(seg_with, sp, m)
    com = p1.index.intersection(p0.index)
    y = m.loc[com, "y"].to_numpy()
    dd = (y - p1.loc[com].to_numpy()) ** 2 - (y - p0.loc[com].to_numpy()) ** 2
    fold = np.where(m.loc[com, "day"].str[2:].astype(int).to_numpy() >= 16, "A", "B")
    blocks = (m.loc[com, "apt"] + m.loc[com, "day"]).to_numpy()
    for f in "AB":
        k = fold == f
        print(f"  1-4 h segment, fold {f}: dMSE {dd[k].sum() / NF[f]:+.1f} {boot_ci(dd[k], blocks[k], NF[f])} "
              f"({int(k.sum()):,} board-like rows; all-row units of the fold)")
    df, p_with = eval_long(band)
    df2, p_wo = eval_long(kept)
    assert (df.index == df2.index).all()
    report(df, {f"long rows without {member}": p_wo, "with": p_with}, "with")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "drop":
        drop(*sys.argv[2:3])
        sys.exit()
    {"legacy-long": legacy_long, "alpha": alpha, "fill": fill, "weights": weights}.get(cmd, lambda: components(*sys.argv[2:4]))()
