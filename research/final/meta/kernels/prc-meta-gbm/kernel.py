# Candidate 3 of the stacking lane: LightGBM meta-model on the residual of the segment-linear stack.
#
# Data (private dataset usiadianimuwa/prc-taxi-2026-meta, built by research/final/meta/build_frame.py):
#   meta2025.parquet  2025 Jan/Feb/Jun/Jul departures, month-out member predictions (row of month m
#                     predicted by the member fold trained on 2025 without m), labels y, context
#   meta2026.parquet  the 670,790 final departures with the refit members' predictions, same context
#
# Per segment (short: d <= 1 h, members c7 cs7 mlp7; 1-4 h: c6s c5s c7 - the current stack's) and
# per held month m in {1, 2, 6, 7}:
#   base  = NNLS stack of the segment's members (normalised to sum 1) fitted on the other three
#           months' board-like rows (= research/final/pseudo_final.py crossfit weights)
#   g     = LightGBM on r = y - base over the other three months' board-like segment rows
#   pred  = base + alpha * g(rounds)
#   (rounds, alpha) chosen INSIDE the three training months by leave-one-month-out on all segment
#   rows (the metric's rows); month m is never used to fit or select anything.
# For 2026: the same procedure with all four months as training (selection by leave-one-month-out
# over the four), applied to the refit members' predictions.
#
# Arms: G1 = members + base + spread + d + context; G1m = members + base + spread + d + airport only;
# G1x = G1 + x5 as an extra input (the pinned prc-f-x5 refit exists for 2026).
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls

T0 = time.time()
ROOT = os.environ.get("META_INPUT", "/kaggle/input")
OUT = os.environ.get("META_OUT", "/kaggle/working")
SMOKE = int(os.environ.get("META_SMOKE", "0"))
H = 3600.0
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
SEGS = {"short": (-1e9, 1 * H, ["c7", "cs7", "mlp7"]), "1-4h": (1 * H, 4 * H, ["c6s", "c5s", "c7"])}
ALL_MEMBERS = ["c7", "cs7", "mlp7", "c6s", "c5s", "k4"]
CONTEXT = ["d_lobt", "d_aobt3", "hour", "sched_hour", "wday", "nm", "unmatched", "a3_found", "adsb_found",
           "a3_grade", "a3_obs_leave"]
ARMS = {
    "G1": {"members": ALL_MEMBERS, "context": CONTEXT},
    "G1m": {"members": ALL_MEMBERS, "context": []},
    "G1x": {"members": ALL_MEMBERS + ["x5"], "context": CONTEXT},
}
# Optional meta_spec.json in the dataset (research/final/meta/specs/*.json, copied by build_frame.py):
# re-fits the same procedure on another member set without editing this file. Keys used here:
#   "segments": {"short": [...], "1-4h": [...]}   members of each segment's base stack
#   "meta_features": [...]                          member predictions given to the trees
#   "arms": ["G1", ...]                              arms to run (default: all three below)
#   "fit_segments": ["short"]                        segments to model (others keep the base stack)
SPEC = {}
_spec_hits = sorted(glob.glob(f"{ROOT}/**/meta_spec.json", recursive=True))
if _spec_hits:
    SPEC = json.load(open(_spec_hits[0]))
    for _seg, _mem in SPEC.get("segments", {}).items():
        SEGS[_seg] = (SEGS[_seg][0], SEGS[_seg][1], list(_mem))
    if "meta_features" in SPEC:
        ALL_MEMBERS = list(SPEC["meta_features"])
        ARMS = {"G1": {"members": ALL_MEMBERS, "context": CONTEXT},
                "G1m": {"members": ALL_MEMBERS, "context": []},
                "G1x": {"members": ALL_MEMBERS + ["x5"], "context": CONTEXT}}
    if "arms" in SPEC:
        ARMS = {k: v for k, v in ARMS.items() if k in SPEC["arms"]}
FIT_SEGMENTS = SPEC.get("fit_segments", list(SEGS))
ROUNDS = [25, 50, 100, 200, 400, 600] if not SMOKE else [2, 4]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
PARAMS = dict(objective="l2", learning_rate=0.05, num_leaves=15, min_data_in_leaf=2000, lambda_l2=50.0,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, max_bin=255, verbose=-1, seed=7,
              num_threads=os.cpu_count() or 4, deterministic=True, force_row_wise=True)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def find(name):
    hits = sorted(glob.glob(f"{ROOT}/**/{name}", recursive=True))
    assert hits, f"missing input {name}"
    return hits[0]


def nnls_norm(P, y):
    w, _ = nnls(P, y)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


def features(t, arm, seg, base):
    mem = ARMS[arm]["members"]
    P = t[[f"p_{m}" for m in mem]].to_numpy(float)
    S = t[[f"p_{m}" for m in SEGS[seg][2]]].to_numpy(float)
    X = pd.DataFrame({f"p_{m}": P[:, i] for i, m in enumerate(mem)})
    X["base"] = base
    X["spread"] = S.std(1)
    X["range"] = S.max(1) - S.min(1)
    X["d"] = t.d.to_numpy(float)
    X["base_d"] = base / np.maximum(np.abs(t.d.to_numpy(float)), 60)
    X["apt"] = pd.Categorical(t.apt.to_numpy(), categories=APTS)
    for c in ARMS[arm]["context"]:
        X[c] = t[c].to_numpy(float)
    return X


def seg_rows(t, seg):
    lo, hi, _ = SEGS[seg]
    d = t.d.to_numpy()
    return (d > lo) & (d <= hi)


def fit_base(t, seg, fit):
    P = t[[f"p_{m}" for m in SEGS[seg][2]]].to_numpy(float)
    w = nnls_norm(P[fit], t.y.to_numpy()[fit])
    return w, P @ w


_CACHE = {}


def train(t, arm, seg, months):
    """base weights on the board-like segment rows of `months`, booster (max(ROUNDS) rounds) on their
    residuals. Cached by (arm, seg, months): a model trained on a month set is the same whichever
    outer month is held, and evaluating it at num_iteration=k equals training k rounds."""
    key = (arm, seg, tuple(sorted(months)))
    if key in _CACHE:
        return _CACHE[key]
    rows = seg_rows(t, seg)
    fit = rows & np.isin(t.month.to_numpy(), months) & t.boardlike.to_numpy()
    w, base = fit_base(t, seg, fit)
    X = features(t, arm, seg, base)
    r = t.y.to_numpy() - base
    ds = lgb.Dataset(X[fit], r[fit], categorical_feature=["apt"], free_raw_data=True)
    bst = lgb.train(PARAMS, ds, num_boost_round=max(ROUNDS))
    log(f"  trained {arm} {seg} on months {key[2]}: {int(fit.sum()):,} rows")
    _CACHE[key] = (w, bst)
    return w, bst


def predict(t, arm, seg, w, bst, rows, rounds_list):
    P = t[[f"p_{m}" for m in SEGS[seg][2]]].to_numpy(float)
    base = P @ w
    X = features(t, arm, seg, base)[rows]
    return base[rows], {k: bst.predict(X, num_iteration=k) for k in rounds_list}


def select(t, arm, seg, months):
    """(rounds, alpha) by leave-one-month-out over `months` on all segment rows."""
    rows = seg_rows(t, seg)
    mon, y = t.month.to_numpy(), t.y.to_numpy()
    sse = {(k, a): 0.0 for k in ROUNDS for a in ALPHAS}
    for v in months:
        w, bst = train(t, arm, seg, [x for x in months if x != v])
        te = rows & (mon == v)
        base, g = predict(t, arm, seg, w, bst, te, ROUNDS)
        for k in ROUNDS:
            for a in ALPHAS:
                sse[(k, a)] += float(((y[te] - base - a * g[k]) ** 2).sum())
    best = min(sse, key=sse.get)
    return best, {f"{k}|{a}": v for (k, a), v in sse.items()}


def main():
    t = pd.read_parquet(find("meta2025.parquet"))
    f = pd.read_parquet(find("meta2026.parquet"))
    if SMOKE:
        t = t.groupby("month", group_keys=False).apply(lambda g: g.sample(3000, random_state=0)).reset_index(drop=True)
        f = f.sample(5000, random_state=0).reset_index(drop=True)
    log(f"2025 rows {len(t):,}, 2026 rows {len(f):,}")
    months = sorted(int(m) for m in t.month.unique())
    report = {"params": PARAMS, "rounds": ROUNDS, "alphas": ALPHAS, "arms": ARMS, "segments": SEGS,
              "lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    for arm in ARMS:
        mo = t.p_c7.to_numpy(float).copy()  # rows past 4 h keep c7; the segments never read them
        rep = report.setdefault(arm, {})
        for seg in FIT_SEGMENTS:
            rows = seg_rows(t, seg)
            for m in months:
                tr = [x for x in months if x != m]
                (k, a), sse = select(t, arm, seg, tr)
                w, bst = train(t, arm, seg, tr)
                te = rows & (t.month.to_numpy() == m)
                base, g = predict(t, arm, seg, w, bst, te, [k])
                mo[te] = base + a * g[k]
                yt = t.y.to_numpy()[te]
                rep[f"{seg}|M{m}"] = {"rounds": k, "alpha": a, "base_w": w.round(6).tolist(), "inner_sse": sse,
                                      "mse_base": float(((yt - base) ** 2).mean()),
                                      "mse_meta": float(((yt - mo[te]) ** 2).mean())}
                log(f"{arm} {seg} M{m}: rounds {k} alpha {a} base w {np.round(w, 3)} "
                    f"MSE base {rep[f'{seg}|M{m}']['mse_base']:,.0f} meta {rep[f'{seg}|M{m}']['mse_meta']:,.0f}")
            # 2026: selection over all four months, fit on all four
            (k, a), sse = select(t, arm, seg, months)
            w, bst = train(t, arm, seg, months)
            frows = seg_rows(f, seg)
            P = f[[f"p_{x}" for x in SEGS[seg][2]]].to_numpy(float)
            fb = P @ w
            fg = bst.predict(features(f, arm, seg, fb)[frows], num_iteration=k)
            rep[f"{seg}|final"] = {"rounds": k, "alpha": a, "base_w": w.round(6).tolist(), "inner_sse": sse,
                                   "importance_gain": dict(zip(bst.feature_name(), bst.feature_importance("gain").round(1).tolist()))}
            col = f"final_{seg}"
            f[col] = np.nan
            f.loc[frows, col] = fb[frows] + a * fg
            bst.save_model(f"{OUT}/{arm}_{seg}.txt", num_iteration=k)
            log(f"{arm} {seg} final: rounds {k} alpha {a} base w {np.round(w, 3)}")
        fin = f.p_c7.to_numpy(float).copy()
        for seg in SEGS:
            if seg in FIT_SEGMENTS:
                k = f[f"final_{seg}"].notna().to_numpy()
                fin[k] = f[f"final_{seg}"].to_numpy()[k]
                continue
            # a segment that is not modelled keeps its base stack: NNLS month-out for 2025, all four months for 2026
            trows, frows = seg_rows(t, seg), seg_rows(f, seg)
            w, _ = fit_base(t, seg, trows & t.boardlike.to_numpy())
            fin[frows] = f.loc[frows, [f"p_{x}" for x in SEGS[seg][2]]].to_numpy(float) @ w
            for m in months:
                te = trows & (t.month.to_numpy() == m)
                wm, _ = fit_base(t, seg, trows & t.boardlike.to_numpy() & (t.month.to_numpy() != m))
                mo[te] = t.loc[te, [f"p_{x}" for x in SEGS[seg][2]]].to_numpy(float) @ wm
            rep[f"{seg}|final"] = {"base_only": True, "base_w": w.round(6).tolist()}
        pd.DataFrame({"id": t.id.to_numpy(), "fold": "M" + t.month.astype(str), "prediction": mo}).to_csv(
            f"{OUT}/mo_{arm}.csv", index=False)
        pd.DataFrame({"MVT_ID_mvt": f.id.to_numpy().astype("float64"), "TAXITIME_SEC_mvt": fin}).to_parquet(
            f"{OUT}/final_{arm}.parquet", index=False)
        f = f.drop(columns=[c for c in f if c.startswith("final_")])
        with open(f"{OUT}/report.json", "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
