"""2025 pseudo-final: the final assembly run on January, February, June and July 2025, scored on labels.

    python3 research/final/pseudo_final.py run CONFIG [--weights frozen|crossfit] [--out-dir DIR]
    python3 research/final/pseudo_final.py compare A.parquet B.parquet
    python3 research/final/pseudo_final.py fit-weights CONFIG [--out JSON]   # segment weights on all months

`run` executes the steps of research/final/assemble_final.py (segments, long rows, fill, overlay,
cap, level, post layers, rules) once per held month m, on the 2025 departures of that month, with
MONTH-OUT member predictions (each member's fold M<m> was trained on 2025 without month m). The
frame has the ranking files' schema: labels and departure off-block times are removed before any
step runs; y is used only for fitting on the other months and for scoring.

Everything fitted from labels is fitted without month m:
  segment / long-stack weights   crossfit mode: NNLS on the other three months' month-out
                                 predictions (rows without date-shift artifacts or short-delay
                                 y > 3 h, as the recipe's weights);
                                 frozen mode: the recipe's weights (fitted on Jan/Jul 2025 A/B OOF,
                                 so January and July were consulted; disclose)
  plog coefficients              same two modes; crossfit E for a row of month k uses tables built
                                 without months k and m
  physical-estimate tables, fill model   2025 rows outside month m
  level composition offset       2025 cells outside month m; lam and cap fixed by the recipe (chosen
                                 on 2025 month-out in research/pru/level_cal.py, all months: disclose)
Post layers are called once per held month with that month's frame; a layer that fits on 2025
labels must exclude the held month itself (`"pass_held_month": true` adds held_month=(2025, m) to
its kwargs; `"pseudo_kwargs": {...}` are merged in only here, "{mode}" in a string value becoming
frozen/crossfit). Rehearsal-only settings of a layer belong in the pseudo config, not the 2026
recipe: "post_layer_pseudo": {<module stem>: {"pseudo_kwargs": {...}, "inputs": [...]}} adds them to
the recipe's layer(s) of that module ("{mode}" substituted in the inputs too).

CONFIG: {"name", "recipe": <assembly config>, "recipe_overrides": {...} (deep-merged),
         "post_layer_pseudo": {...} (optional, above), "months": [1, 2, 6, 7],
         "members": {<recipe member>: {"mo": <oof.csv or run dir>}},
         "phys_inputs": <2025 a3_grade/a3_obs_leave parquet>, "overlay_flags": {<overlay>: <parquet>}}
Month-out files: id, fold, prediction (train_main's oof.csv); a run directory without oof.csv but
with variant subdirectories (r*/oof.csv) is averaged over them. Fold names M1..M12 (MJ/MY aliases).

Scores are full and untrimmed: SSE / N over every departure of the month, pooled over the four
months, by airport and by delay band, and after each step. `compare` pairs two runs row by row:
dMSE pooled, per month and per airport, with a 95% airport-day block bootstrap interval.
Outputs: OUT/NAME_MODE.parquet (per row: id, month, apt, d, y, step predictions) and
NAME_MODE.json (scores, per-month fits, inputs with sha256), copied to research/final/pseudo/.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assemble_final as af  # noqa: E402
import level  # noqa: E402
import longrows  # noqa: E402
from frame import C, load_training_frame, sha256  # noqa: E402
from stacking import fit_weights, in_segment, resolve  # noqa: E402
from tailkit import fit_wlog, is_fill, nnls_w, phys  # noqa: E402

H = 3600.0
BANDS = [("d <= 1 h", -np.inf, 1), ("1-4 h", 1, 4), ("4-8 h", 4, 8), ("> 8 h", 8, np.inf)]
STEPS = ["fill", "overlay", "cap", "level", "post", "rules"]
ALIASES = {"MJ": 1, "MY": 7}


def fold_month(name: str) -> int:
    if name in ALIASES:
        return ALIASES[name]
    assert name.startswith("M") and name[1:].isdigit(), f"not a month-out fold: {name}"
    return int(name[1:])


def mo_file(spec: str) -> list[Path]:
    p = resolve(spec)
    if p.suffix in (".csv", ".parquet"):
        return [p]
    if (p / "oof.csv").exists():
        return [p / "oof.csv"]
    subs = sorted(p.glob("*/oof.csv"))
    assert subs, f"{spec}: no oof.csv"
    return subs


def read_oof(f: Path) -> pd.DataFrame:
    return pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)


def load_mo(spec, fr: pd.DataFrame) -> tuple[np.ndarray, list[Path]]:
    """Month-out prediction for every frame row; the fold must be the row's own month.

    `spec` is one oof file / run directory, or a LIST of them whose folds are disjoint parts of the
    same member (e.g. months M2/M7 from one run and M1/M6 from another); a list is concatenated."""
    if isinstance(spec, list):
        parts = [(f, read_oof(f)) for s in spec for f in mo_file(s)]
        o = pd.concat([q for _, q in parts], ignore_index=True)
        o["id"] = o["id"].round().astype("int64")
        assert o["id"].is_unique, f"{spec}: the parts overlap (an id appears in more than one file)"
        files, frames = [f for f, _ in parts], [o]
    else:
        files = mo_file(spec)
        frames = [read_oof(f) for f in files]
    preds = []
    for f, o in zip(files if not isinstance(spec, list) else [files], frames):
        o["id"] = o["id"].round().astype("int64")
        o["fm"] = o["fold"].astype(str).map(fold_month)
        o = o.set_index("id")
        assert o.index.is_unique, f"{f}: duplicate ids"
        sub = o.reindex(fr.id.to_numpy())
        miss = sub["prediction"].isna().to_numpy()
        assert not miss.any(), f"{f}: {int(miss.sum())} departures of the scored months have no month-out prediction"
        wrong = sub["fm"].to_numpy() != fr.month.to_numpy()
        assert not wrong.any(), f"{f}: {int(wrong.sum())} rows predicted by a fold that is not their own month"
        preds.append(sub["prediction"].to_numpy(dtype=float))
    return np.mean(preds, axis=0), files


def boardlike(fr: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Rows without date-shift artifacts or short-delay y > 3 h (research/evalb.py's definition)."""
    dd = (fr.MVT_TIME_UTC_mvt.dt.floor("D") - fr.SCHED_TIME_UTC_mvt.dt.floor("D")).dt.days.to_numpy()
    art = (dd >= 1) & (y - 86400 * dd >= -900) & (y - 86400 * dd <= 3600)
    return ~art & ~((fr.d.to_numpy() <= 3600) & (y > 10800))


class Context:
    """The four months' frame, labels and month-out member predictions."""

    def __init__(self, cfg: dict, recipe: dict):
        self.F, self.y = load_training_frame(cfg.get("months", [1, 2, 6, 7]))
        self.d = self.F.d.to_numpy()
        self.month = self.F.month.to_numpy()
        self.bl = boardlike(self.F, self.y)
        self.fill = is_fill(self.F)
        self.inputs = dict(self.F.attrs["inputs"])
        self.preds = {}
        for name in recipe["members"]:
            spec = cfg["members"].get(name)
            assert spec and "mo" in spec, f"pseudo config: no month-out predictions for recipe member {name}"
            self.preds[name], files = load_mo(spec["mo"], self.F)
            self.inputs.update({af.rel(f): sha256(f) for f in files})
        self.tr = longrows.long_tr()
        self.inputs[af.rel(longrows.LONG_TR)] = sha256(longrows.LONG_TR)
        allrun = af.Run(recipe, "train2025", fr=self.F, verbose=False)
        self.phys = af.phys_frame(allrun, recipe["long"]["phys_inputs"])
        self.inputs.update(allrun.inputs)


class PseudoRun(af.Run):
    def __init__(self, recipe: dict, ctx: Context, m: int):
        self.ctx, self.m = ctx, m
        self.mask = ctx.month == m
        fr = ctx.F[self.mask].reset_index(drop=True)
        fr.attrs = {"kind": "train2025", "inputs": {}}
        super().__init__(recipe, "train2025", fr=fr, verbose=False)
        self.T = ~self.mask

    def member(self, name: str) -> np.ndarray:
        return self.ctx.preds[name][self.mask]

    def seg_fit(self, seg: dict) -> dict:
        c = self.ctx
        P = np.column_stack([c.preds[n] for n in seg["members"]])
        s = self.T & c.bl & in_segment(c.d, seg["lo_h"], seg["hi_h"])
        return dict(zip(seg["members"], np.round(fit_weights(P[s], c.y[s]), 6).tolist()))

    def long_fit(self, members: list[str], w_frozen: np.ndarray | None):
        c = self.ctx
        P = np.column_stack([c.preds[n] for n in members])
        s = self.T & c.bl & (c.d > H) & (c.d <= 4 * H) & ~c.fill
        fitted = nnls_w(P[s], c.y[s])
        M = P @ (fitted if w_frozen is None else w_frozen)
        fr = self.T & c.bl & (c.d > 4 * H) & ~c.fill
        E = np.full(len(c.d), np.nan)
        for k in np.unique(c.month[fr]):
            rows = fr & (c.month == k)
            E[rows] = phys(c.phys[rows], c.tr[~c.tr.month.isin([k, self.m])])
        coef = fit_wlog(c.y[fr], M[fr], E[fr], c.d[fr], np.full(int(fr.sum()), "N"))
        return fitted, {k: [float(x) for x in v] for k, v in coef.items()}, {
            "fit_rows_stack": int(s.sum()), "fit_rows_plog": int(fr.sum())}

    def tables_tr(self) -> pd.DataFrame:
        return self.ctx.tr[self.ctx.tr.month != self.m]

    def level_coef(self, cap: float) -> np.ndarray:
        return level.fit_coef(cap, exclude=[(2025, self.m)])

    def post_kwargs(self, pl: dict) -> dict:
        # a string pseudo kwarg may name the run's weights mode as "{mode}" (e.g. a base run's rows file)
        pk = {k: v.replace("{mode}", self.ctx.mode) if isinstance(v, str) else v for k, v in pl.get("pseudo_kwargs", {}).items()}
        kw = {**pl.get("kwargs", {}), **pk}
        if pl.get("pass_held_month"):
            kw["held_month"] = (2025, self.m)
        return kw


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def recipe_for(cfg: dict, mode: str) -> dict:
    recipe = json.loads(resolve(cfg["recipe"]).read_text()) if "recipe" in cfg else copy.deepcopy(cfg)
    recipe = deep_merge(recipe, cfg.get("recipe_overrides", {}))
    recipe["name"] = cfg["name"]
    recipe["long"]["phys_inputs"] = cfg["phys_inputs"]
    for ov in recipe.get("overlays", []):
        ov.pop("rows", None)
        ov.pop("col", None)
        ov["flags"] = cfg["overlay_flags"][ov["name"]]
    for stem, extra in cfg.get("post_layer_pseudo", {}).items():
        layers = [pl for pl in recipe.get("post_layers", []) if Path(pl["module"]).stem == stem]
        assert layers, f"post_layer_pseudo: the recipe has no post layer {stem}"
        for pl in layers:
            pl["pseudo_kwargs"] = {**pl.get("pseudo_kwargs", {}), **extra.get("pseudo_kwargs", {})}
            pl["inputs"] = pl.get("inputs", []) + [x.replace("{mode}", mode) for x in extra.get("inputs", [])]
    if mode == "crossfit":
        for seg in recipe["segments"]:
            seg.pop("weights", None)
        recipe["long"].pop("weights", None)
        recipe["long"].pop("plog_coef", None)
    return recipe


def band_of(d: np.ndarray) -> np.ndarray:
    out = np.empty(len(d), dtype=object)
    for name, lo, hi in BANDS:
        out[(d > lo * H) & (d <= hi * H)] = name
    return out


def group_of(t: pd.DataFrame, ctx: Context) -> np.ndarray:
    """Row groups the long-row decisions act on."""
    pos = pd.Series(np.arange(len(ctx.F)), index=ctx.F.id.to_numpy()).reindex(t.id.to_numpy()).to_numpy()
    d, fill = t.d.to_numpy(), ctx.fill[pos]
    guard = (ctx.F.d_aobt3.to_numpy()[pos] > 4 * H) & ~fill
    return np.select([d <= 4 * H, fill, guard], ["d <= 4 h", "> 4 h LIRF-unmatched", "> 4 h AOBT_3 guard"],
                     "> 4 h blended")


def scores(t: pd.DataFrame, col: str = "pred") -> dict:
    e = (t.y - t[col]) ** 2
    n = len(t)
    out = {"pooled": {"rows": n, "mse": float(e.mean()), "rmse": float(np.sqrt(e.mean()))}}
    out["month"] = {int(k): {"rows": int(len(g)), "mse": float(g.mean()), "rmse": float(np.sqrt(g.mean()))}
                    for k, g in e.groupby(t.month)}
    out["airport"] = {k: {"rows": int(len(g)), "mse_contrib": float(g.sum() / n), "rmse": float(np.sqrt(g.mean()))}
                      for k, g in e.groupby(t.apt)}
    out["band"] = {k: {"rows": int(len(g)), "mse_contrib": float(g.sum() / n), "rmse": float(np.sqrt(g.mean()))}
                   for k, g in e.groupby(t.band)}
    out["group"] = {k: {"rows": int(len(g)), "mse_contrib": float(g.sum() / n), "rmse": float(np.sqrt(g.mean()))}
                    for k, g in e.groupby(t.group)}
    return out


def run(cfg_path: str, mode: str, out_dir: str) -> dict:
    cfg = json.loads(Path(cfg_path).read_text())
    recipe = recipe_for(cfg, mode)
    missing = []
    for name in recipe["members"]:
        mo = cfg["members"].get(name, {}).get("mo", "<none>")
        for one in (mo if isinstance(mo, list) else [mo]):
            spec = resolve(one)
            if not (spec.is_file() or (spec / "oof.csv").exists() or any(spec.glob("*/oof.csv"))):
                missing.append(f"  member {name} month-out: {af.rel(spec)}")
    for what, p in (("phys_inputs", cfg["phys_inputs"]), *[(f"overlay {k}", v) for k, v in cfg["overlay_flags"].items()]):
        if not resolve(p).exists():
            missing.append(f"  {what}: {p}")
    if missing:
        sys.exit("missing inputs:\n" + "\n".join(missing))
    af.preflight({**recipe, "members": {}})
    ctx = Context(cfg, recipe)
    ctx.mode = mode
    print(f"== {cfg['name']} [{mode}] on 2025 months {sorted(int(x) for x in set(ctx.month))}: {len(ctx.F):,} departures")
    parts, fits = [], {}
    for m in sorted(int(x) for x in set(ctx.month)):
        r = PseudoRun(recipe, ctx, m)
        af.run_steps(r)
        part = pd.DataFrame({"id": r.ids, "month": m, "day": r.fr.day.to_numpy(), "apt": r.fr.apt.to_numpy(), "d": r.d,
                             "y": ctx.y[r.mask], "boardlike": ctx.bl[r.mask]})
        for st in STEPS + [k for k in r.stages if k.startswith("post") and k != "post"]:
            part[st] = r.stages[st]
        part["pred"] = r.pred
        parts.append(part)
        for p in r.inputs:
            ctx.inputs.setdefault(p, r.inputs[p])
        fits[m] = {k: r.man[k] for k in ("segments", "long", "overlays", "cap", "level", "post_layers", "rules")}
        e = (part.y - part.pred) ** 2
        print(f"  M{m}: {len(part):,} rows  MSE {e.mean():,.0f}  RMSE {np.sqrt(e.mean()):.2f}  "
              f"seg weights {[r.man['segments'][s]['weights'] for s in r.man['segments']]}")
    t = pd.concat(parts, ignore_index=True)
    t["band"] = band_of(t.d.to_numpy())
    t["group"] = group_of(t, ctx)
    res = {"name": cfg["name"], "mode": mode, "scores": scores(t)}
    steps = ["fill", "overlay", "cap", "level"] + [c for c in t.columns if c.startswith("post") and c != "post"] + ["post", "rules"]
    res["steps_month_mse"] = {st: {int(k): float(g.mean()) for k, g in ((t.y - t[st]) ** 2).groupby(t.month)} for st in steps}
    res["steps_pooled_mse"] = {st: float(((t.y - t[st]) ** 2).mean()) for st in steps}
    # mean residual y - prediction per month x delay band after each step (level bias diagnostics)
    res["bias_month_band"] = {st: {f"M{int(mo)} {b}": float(g.mean()) for (mo, b), g in (t.y - t[st]).groupby([t.month, t.band])}
                              for st in steps}
    print("mean residual y - pred on 1-4 h rows, per month, after each step:")
    for st in steps:
        print(f"  {st:24s} " + "  ".join(f"{k.split()[0]} {v:+7.1f}" for k, v in res["bias_month_band"][st].items() if k.endswith("1-4 h")))
    sp = res["scores"]["pooled"]
    print(f"pooled: {sp['rows']:,} rows  MSE {sp['mse']:,.1f}  RMSE {sp['rmse']:.3f}")
    print("after each step (pooled MSE): " + "  ".join(f"{st} {v:,.1f}" for st, v in res["steps_pooled_mse"].items()))
    print("after each step (MSE by month): " + "  ".join(f"{st} " + "/".join(f"{v:,.0f}" for v in res["steps_month_mse"][st].values())
                                                       for st in steps))
    print("by band (MSE contribution / RMSE): " + "  ".join(
        f"{k}: {v['mse_contrib']:,.0f} / {v['rmse']:.0f} ({v['rows']:,})" for k, v in res["scores"]["band"].items()))
    print("by row group (MSE contribution / RMSE): " + "  ".join(
        f"{k}: {v['mse_contrib']:,.0f} / {v['rmse']:.0f} ({v['rows']:,})" for k, v in res["scores"]["group"].items()))
    print("by airport (MSE contribution): " + "  ".join(f"{k} {v['mse_contrib']:,.0f}" for k, v in res["scores"]["airport"].items()))
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"{cfg['name']}_{mode}.parquet"
    t.to_parquet(rows_path, index=False)
    for p in af.CODE + [Path(__file__).resolve()]:
        ctx.inputs.setdefault(af.rel(p), sha256(p))
    res.update({"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "command": " ".join([Path(sys.executable).name] + sys.argv), "git": af.git_state(),
                "config": cfg, "recipe": recipe, "fits": fits, "rows_file": af.rel(rows_path),
                "rows_sha256": sha256(rows_path), "inputs": ctx.inputs})
    copies = [out / f"{cfg['name']}_{mode}.json"]
    if out.resolve().is_relative_to((C / "submissions").resolve()):
        copies.append(HERE / "pseudo" / f"{cfg['name']}_{mode}.json")
    for p in copies:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1, default=str))
    print(f"-> {af.rel(rows_path)}, {af.rel(copies[0])}")
    return res


def fit_weights_all_months(cfg_path: str) -> dict:
    """Segment weights by NNLS on ALL the config's months' month-out predictions (rows without
    date-shift artifacts or short-delay y > 3 h), rounded to 6 decimals: what a 2026 recipe would
    freeze for a member that has month-out folds but no A/B OOF. Used inside the rehearsal they are
    in-sample for every scored month."""
    cfg = json.loads(Path(cfg_path).read_text())
    recipe = recipe_for(cfg, "crossfit")
    ctx = Context(cfg, recipe)
    out = {}
    for seg in recipe["segments"]:
        P = np.column_stack([ctx.preds[n] for n in seg["members"]])
        s = ctx.bl & in_segment(ctx.d, seg["lo_h"], seg["hi_h"])
        out[seg["name"]] = dict(zip(seg["members"], np.round(fit_weights(P[s], ctx.y[s]), 6).tolist()))
        print(f"[{seg['name']}] {int(s.sum()):,} rows: {out[seg['name']]}")
    return out


def boot(dd: np.ndarray, blocks: np.ndarray, n: int, draws: int = 2000) -> tuple[float, float]:
    codes, uniq = pd.factorize(blocks)
    sums = np.bincount(codes, weights=dd, minlength=len(uniq))
    rng = np.random.default_rng(0)
    b = sums[rng.integers(0, len(uniq), size=(draws, len(uniq)))].sum(1) / n
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def row_flags(months) -> pd.DataFrame:
    """Per id: `wrap`, the 2025 date-wrap rows (the right time of day stamped on the scheduled date:
    y within [-900, +3,600] s of a whole number of days >= 1; 2026 does not show this mode), and
    `set`, disjoint row sets defined structurally: c = LIRF without NM flight, 24,000 <= d < 86,400
    (the LIRF orphan band); fill = LIRF without AOBT_3 past 4 h (long.fill), minus c; long = every
    other row past 4 h; short = the rest."""
    fr, y = load_training_frame(months)
    dd = (fr.MVT_TIME_UTC_mvt.dt.floor("D") - fr.SCHED_TIME_UTC_mvt.dt.floor("D")).dt.days.to_numpy()
    d = fr.d.to_numpy()
    c = fr.lirf_orphan.to_numpy() & (d >= 24000) & (d < 86400)
    fill = is_fill(fr) & (d > 4 * H) & ~c
    long = (d > 4 * H) & ~c & ~fill
    return pd.DataFrame({"wrap": (dd >= 1) & (y - 86400 * dd >= -900) & (y - 86400 * dd <= 3600),
                         "set": np.select([c, fill, long], ["c", "fill", "long"], "short")}, index=fr.id.to_numpy())


def compare(a_path: str, b_path: str, out_json: str | None = None) -> dict:
    """Paired dMSE of run A minus run B (negative = A better): all rows and without the 2025
    date-wrap rows ("nw"), pooled / per month / per band / per row group / per airport, with 95%
    airport-day block bootstrap intervals (per-month rows in month units, the rest in pooled units)."""
    a, b = pd.read_parquet(resolve(a_path)), pd.read_parquet(resolve(b_path))
    assert (a.id.to_numpy() == b.id.to_numpy()).all() and np.allclose(a.y, b.y), "runs are not on the same rows"
    t = a[["id", "month", "apt", "d"]].copy()
    t["dd"] = ((a.y - a.pred) ** 2 - (b.y - b.pred) ** 2).to_numpy()
    t["block"] = (a.apt + "|" + a.day).to_numpy()  # airport-day blocks
    t["band"] = band_of(t.d.to_numpy())
    t["group"] = a.group.to_numpy() if "group" in a else ""
    flags = row_flags(sorted(int(m) for m in t.month.unique())).reindex(t.id.to_numpy())
    t["wrap"], t["set"] = flags.wrap.to_numpy().astype(bool), flags["set"].to_numpy()
    out = {"a": a_path, "b": b_path, "rows": len(t), "date_wrap_rows": int(t.wrap.sum())}
    print(f"{a_path}\n  minus {b_path}  (negative = first better; date-wrap rows: {int(t.wrap.sum())})")
    for scope, tt in (("all", t), ("nw", t[~t.wrap])):
        n = len(tt)
        res = {}
        lo, hi = boot(tt.dd.to_numpy(), tt.block.to_numpy(), n)
        res["pooled"] = {"rows": n, "dmse": float(tt.dd.sum() / n), "ci95": [lo, hi]}
        res["month"] = {}
        for k, g in tt.groupby("month"):
            lo2, hi2 = boot(g.dd.to_numpy(), g.block.to_numpy(), len(g))
            res["month"][int(k)] = {"rows": int(len(g)), "dmse": float(g.dd.sum() / len(g)), "ci95": [lo2, hi2]}
        for key in ("band", "group", "set"):
            res[key] = {}
            for k, g in tt.groupby(key):
                lo2, hi2 = boot(np.where(tt[key] == k, tt.dd, 0.0), tt.block.to_numpy(), n)
                res[key][str(k)] = {"rows": int(len(g)), "dmse": float(g.dd.sum() / n), "ci95": [lo2, hi2]}
        res["airport"] = {k: float(g.dd.sum() / n) for k, g in tt.groupby("apt")}
        out[scope] = res
        print(f" [{scope}] pooled {res['pooled']['dmse']:+.1f} [{lo:+.1f}, {hi:+.1f}] ({n:,} rows)")
        print("   month (month units): " + "  ".join(f"M{k} {v['dmse']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"
                                                    for k, v in res["month"].items()))
        print("   band (pooled units): " + "  ".join(f"{k} {v['dmse']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"
                                                   for k, v in res["band"].items()))
        print("   row set (pooled units): " + "  ".join(f"{k} {v['dmse']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] ({v['rows']:,})"
                                                     for k, v in res["set"].items()))
        if "group" in a:
            print("   group (pooled units): " + "  ".join(f"{k} {v['dmse']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"
                                                        for k, v in res["group"].items()))
        print("   airport (pooled units): " + "  ".join(f"{k} {v:+.1f}" for k, v in res["airport"].items()))
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--weights", choices=["frozen", "crossfit"], default="frozen")
    r.add_argument("--out-dir", default="submissions/final/pseudo2025")
    w = sub.add_parser("fit-weights", help="segment weights on all months' month-out predictions")
    w.add_argument("config")
    w.add_argument("--out")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json", help="write the comparison here (default research/final/pseudo/compare_A__B.json)")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.config, a.weights, a.out_dir)
    elif a.cmd == "fit-weights":
        wts = fit_weights_all_months(a.config)
        if a.out:
            Path(a.out).write_text(json.dumps(wts, indent=1))
    else:
        default = HERE / "pseudo" / f"compare_{Path(a.a).stem}__{Path(a.b).stem}.json"
        compare(a.a, a.b, a.json or str(default))


if __name__ == "__main__":
    main()
