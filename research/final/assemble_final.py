"""Assemble a submission for the final (four months) or the board (Jan/Jul) with one command.

    python3 research/final/assemble_final.py research/final/configs/NAME.json [--frame final|board]
            [--ref-board FILE] [--compare FILE] [--out-dir submissions/final]

The recipe generalises v32c's assembly (research/assemble.py) to any frame and adds the rule
package. On every departure of the frame, in this order:

  1 segments  per delay band (d = takeoff - schedule), members stacked with NNLS weights fitted on
              their A/B OOF (research/final/stacking.py); clip 60.
  2 long      d > 4 h, not LIRF-unmatched: long-member stack blended with the physical estimate,
              plog w(d) (research/final/longrows.py), alpha = 1. Rows whose AOBT_3 is more than
              4 h before takeoff take the long-member stack unblended.
  3 fill      d > 4 h, LIRF-unmatched: the configured member stack (`long.fill`).
  4 overlay   flagged rows: pred = (1 - w) pred + w member (LIRF rows with neither an ADS-B v3
              event nor ADS-B coverage and d <= 4 h, no-ADS-B LightGBM k4, w = 0.3).
  5 cap       pred <= d where d > 2 h (the optional LIRF orphan band rule in "floor" mode exempts
              its band).
  6 clip 60
  7 level     PRU airport-month calibration (research/final/level.py), lam/cap from the config.
  8 post      optional post_layers, each fn(frame, pred, **kwargs) -> pred. A layer entry with
              "freeze_nm_bound_rows": true cannot move rows that sit on a rule-(a) window bound.
  9 rules     NM join window, LEMD/LEBL cap, floor 60, and the optional LIRF orphan band rule
              (lirf_orphan_floor_d), off unless a config enables it (research/final/rules.py).

Checks (§8.6 of research/ADVERSARIAL_REVIEW_20261008_B.md): item 1 (ids, dtypes, finite, >= 0, rule
bounds) fails the run and the final file is not written under its delivery name; item 2 (airport-
month mean vs PRU after the 2025 composition offset) is printed and flagged; item 3 (Jan/Jul RMS
against --ref-board) fails above 10 s.

Outputs in --out-dir: NAME_final.parquet (final frame: all 670,790 ids in final_submitting.parquet
order; MVT_ID_mvt and TAXITIME_SEC_mvt both float64), NAME_board.parquet
(Jan/Jul rows keyed and ordered as the old board template), NAME_stages.parquet (the prediction
after each step, for diagnosis) and NAME.manifest.json (config, weights, coefficients, row counts,
sha256 of every input and output, git commit; a member may list upstream files under "inputs",
e.g. the dataset and code behind a meta-model's predictions, and they are hashed too). The manifest
is also copied to
research/final/manifests/NAME.json.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import level  # noqa: E402
import longrows  # noqa: E402
import rules  # noqa: E402
from frame import FRAMES, ID, TARGET, C, load_frame, sha256  # noqa: E402
from stacking import META, in_segment, member_pred, oof_file, oof_meta, pred_file, resolve, segment_weights  # noqa: E402

H = 3600.0
CODE = [HERE / f for f in ("frame.py", "stacking.py", "longrows.py", "level.py", "rules.py", "assemble_final.py")] + [
    C / "research" / "tail" / "tailkit.py", C / "research" / "extreme_blend.py"]


def rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(C))
    except ValueError:
        return str(p)


def git_state() -> dict:
    def run(*a):
        r = subprocess.run(["git", "-C", str(C), *a], capture_output=True, text=True)
        return r.stdout.strip()
    paths = [rel(p) for p in CODE]
    return {"commit": run("rev-parse", "HEAD"), "dirty": run("status", "--porcelain", "--", *paths).splitlines()}


def rms(a) -> float:
    a = np.asarray(a, float)
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0


class Run:
    """Holds the frame, the evolving prediction, the stage snapshots and the manifest.

    The methods below are where the 2026 assembly and the 2025 pseudo-final differ (member
    predictions, where weights and coefficients are fitted, which 2025 rows the tables may use);
    research/final/pseudo_final.py overrides them and runs the same steps."""

    def __init__(self, cfg: dict, kind: str, fr: pd.DataFrame | None = None, verbose: bool = True):
        self.cfg, self.kind, self.verbose = cfg, kind, verbose
        self.fr = load_frame(kind) if fr is None else fr
        self.ids = self.fr.id.to_numpy()
        self.d = self.fr.d.to_numpy()
        self.pred = np.full(len(self.fr), np.nan)
        self.stages: dict[str, np.ndarray] = {}
        self.inputs: dict[str, str] = dict(self.fr.attrs.get("inputs", {}))
        self.man: dict = {"name": cfg["name"], "frame": kind}
        self._preds: dict[str, np.ndarray] = {}
        self._meta = None

    def log(self, *a):
        if self.verbose:
            print(*a)

    @property
    def meta(self) -> pd.DataFrame:
        if self._meta is None:
            self.track(META)
            self._meta = oof_meta()
        return self._meta

    def has_oof(self, members) -> bool:
        return all("oof" in self.cfg["members"][n] for n in members)

    def seg_fit(self, seg: dict) -> dict | None:
        """Weights fitted on the members' A/B OOF (None when a member has no OOF)."""
        if not self.has_oof(seg["members"]):
            return None
        for name in seg["members"]:
            self.track(oof_file(self.cfg["members"][name]["oof"]))
        return segment_weights(seg, self.cfg["members"], self.meta)

    def long_fit(self, members: list[str], w_frozen: np.ndarray | None):
        """(stack weights, plog coefficients, info) fitted on the long members' A/B OOF."""
        if not self.has_oof(members):
            return None, None, {}
        for name in members:
            self.track(oof_file(self.cfg["members"][name]["oof"]))
        tr = longrows.long_tr()
        df = longrows.long_oof(members, self.cfg["members"], self.meta, tr)
        fitted = longrows.stack_weights(df, members)
        w = fitted if w_frozen is None else w_frozen
        coef = longrows.fit_plog(df, df[[f"p_{n}" for n in members]].to_numpy() @ w, tr)
        return fitted, coef, {"oof_rows": int(len(df)), "oof_stack_rows": int(longrows.stack_rows(df).sum())}

    def tables_tr(self) -> pd.DataFrame:
        """2025 rows the physical-estimate tables and the fill model may use."""
        self.track(longrows.LONG_TR)
        return longrows.long_tr()

    def level_coef(self, cap: float) -> np.ndarray:
        return level.fit_coef(cap)

    def post_kwargs(self, pl: dict) -> dict:
        return dict(pl.get("kwargs", {}))

    def track(self, *paths):
        for p in paths:
            p = resolve(str(p))
            k = rel(p)
            if k not in self.inputs:
                self.inputs[k] = sha256(p)

    def member(self, name: str) -> np.ndarray:
        if name not in self._preds:
            spec = self.cfg["members"][name]["pred"]
            self.track(pred_file(spec))
            self._preds[name] = member_pred(spec, self.ids)
        return self._preds[name]

    def snap(self, stage: str):
        self.stages[stage] = self.pred.copy()


def step_segments(r: Run):
    r.man["segments"] = {}
    for seg in r.cfg["segments"]:
        fitted = r.seg_fit(seg)
        frozen = seg.get("weights")
        if frozen:
            assert list(frozen) == seg["members"], f"{seg['name']}: frozen weights must list the members in order"
            if fitted:
                gap = max(abs(frozen[k] - fitted[k]) for k in fitted)
                r.log(f"[{seg['name']}] frozen weights; max |frozen - refit on OOF| = {gap:.2e}")
        w = frozen or fitted
        assert w, f"{seg['name']}: no frozen weights and no OOF to fit them"
        S = np.column_stack([r.member(name) for name in seg["members"]])
        k = in_segment(r.d, seg["lo_h"], seg["hi_h"])
        r.pred[k] = np.clip(S[k] @ np.array([w[n] for n in seg["members"]]), 60, None)
        r.man["segments"][seg["name"]] = {"lo_h": seg["lo_h"], "hi_h": seg["hi_h"], "rows": int(k.sum()),
                                          "weights": w, "weights_refit_on_oof": fitted, "frozen": bool(frozen)}
        r.log(f"[{seg['name']}] {int(k.sum()):,} rows  weights {w}")
    r.snap("segments")


def phys_frame(r: Run, path: str) -> pd.DataFrame:
    """Frame columns the physical estimate needs; ADS-B fields from the supplied store extract."""
    r.track(path)
    px = pd.read_parquet(resolve(path))
    key = "MVT_ID_mvt" if "MVT_ID_mvt" in px else "id"
    px.index = px[key].round().astype("int64")
    assert px.index.is_unique, path
    need = (r.d > r.cfg["long"].get("lo_h", 4) * H)
    miss = ~np.isin(r.ids[need], px.index.to_numpy())
    assert not miss.any(), f"{path}: {int(miss.sum())} long-row ids missing"
    out = r.fr[["apt", "d_sched", "unmatched", "d_aobt3"]].copy()
    for c in ("a3_grade", "a3_obs_leave"):
        out[c] = px[c].reindex(r.ids).to_numpy(dtype=float)
    sub = px.reindex(r.ids[need])  # store rows must describe the same frame (dithered takeoff included)
    for c in ("d_sched", "d_aobt3", "unmatched"):
        if c in px.columns:
            a, b = sub[c].to_numpy(dtype=float), out[c].to_numpy()[need]
            assert np.allclose(np.nan_to_num(a, nan=-1e9), np.nan_to_num(b, nan=-1e9), atol=0.5), \
                f"{path}: {c} disagrees with the frame (store built on another ranking file?)"
    return out


def step_long(r: Run):
    L = r.cfg["long"]
    lo_h = L.get("lo_h", 4)
    k_long = r.d > lo_h * H
    if "legacy_source" in L:  # v32c reproduction only: rows taken from a fixed artifact
        r.track(L["legacy_source"])
        src = pd.read_parquet(resolve(L["legacy_source"]))
        src.index = src[ID].round().astype("int64")
        r.pred[k_long] = src[TARGET].reindex(r.ids[k_long]).to_numpy(dtype=float)
        r.man["long"] = {"legacy_source": L["legacy_source"], "rows": int(k_long.sum())}
        r.log(f"[long] {int(k_long.sum()):,} rows from legacy artifact {L['legacy_source']}")
        r.snap("long")
        r.snap("fill")
        return
    seg_by_name = {s["name"]: s for s in r.cfg["segments"]}
    members = L.get("members") or seg_by_name["1-4h"]["members"]
    w_frozen = np.array([L["weights"][n] for n in members]) if L.get("weights") else None
    fitted, coef_fit, info = r.long_fit(members, w_frozen)
    w = w_frozen if w_frozen is not None else fitted
    coef = L.get("plog_coef") or coef_fit
    assert w is not None and coef is not None, "long rows: no frozen weights/coefficients and no OOF to fit them"
    tr = r.tables_tr()
    S = np.column_stack([r.member(n) for n in members])
    M = S @ w
    pf = phys_frame(r, L["phys_inputs"])
    guard_h = L.get("guard_aobt3_h", 4.0)
    nf, q, E, wt, kinds = longrows.construct(pf, M, coef, tr, lo_h, guard_h)
    r.pred[nf] = q
    guard = k_long & longrows.guard_rows(pf, guard_h) & ~longrows.is_fill(pf)
    r.pred[guard] = np.clip(M[guard], 60, None)
    r.snap("long")

    def stack_of(src: str) -> np.ndarray:
        if src == "long_stack":
            return M
        if src.startswith("segment:"):
            seg = seg_by_name[src.split(":", 1)[1]]
            sw = r.man["segments"][seg["name"]]["weights"]
            return np.column_stack([r.member(n) for n in seg["members"]]) @ np.array([sw[n] for n in seg["members"]])
        raise ValueError(f"long.fill: unknown source {src}")
    fill = k_long & ~nf & ~guard
    src = L.get("fill", "long_stack")
    if src == "fill_prob":  # 2025 mixture: p(fill | d) d + (1 - p) median non-fill y, logistic in log d
        p_fill, y0 = longrows.fill_prob(tr, r.d[fill]) if fill.any() else (np.zeros(0), 0.0)
        r.pred[fill] = np.clip(p_fill * r.d[fill] + (1 - p_fill) * y0, 60, None)
    else:
        r.pred[fill] = np.clip(stack_of(src)[fill], 60, None)
    r.snap("fill")
    r.man["long"] = {"lo_h": lo_h, "members": members, "weights": dict(zip(members, w.tolist())),
                     "weights_refit_on_oof": dict(zip(members, fitted.tolist())) if fitted is not None else None,
                     "frozen": bool(L.get("weights")), "plog_coef_frozen": bool(L.get("plog_coef")),
                     "plog_coef": coef, "rows": int(nf.sum()), "kinds": pd.Series(kinds).value_counts().to_dict(),
                     "w_mean": float(wt.mean()) if len(wt) else None, "fill_rows": int(fill.sum()), "fill_source": src,
                     "guard_aobt3_h": guard_h, "guard_rows": int(guard.sum()), "guard_source": "long_stack", **info}
    r.log(f"[long] {int(nf.sum()):,} rows (kinds {r.man['long']['kinds']}), mean w {r.man['long']['w_mean']:.3f}; "
          f"weights {dict(zip(members, np.round(w, 4)))}; plog a,b {np.round(coef['N'], 4).tolist()}")
    r.log(f"[guard] {int(guard.sum())} rows with AOBT_3 more than {guard_h} h before takeoff: long-member stack, no blend")
    r.log(f"[fill] {int(fill.sum()):,} LIRF-unmatched rows past {lo_h} h from {src}")


def overlay_flag(r: Run, ov: dict) -> np.ndarray:
    if "rows" in ov:  # precomputed boolean column
        r.track(ov["rows"])
        t = pd.read_parquet(resolve(ov["rows"]))
        key = "id" if "id" in t else "MVT_ID_mvt"
        t.index = t[key].round().astype("int64")
        miss = ~np.isin(r.ids, t.index.to_numpy())
        assert not miss.any(), f"{ov['rows']}: {int(miss.sum())} frame ids missing"
        return t[ov["col"]].reindex(r.ids).astype(bool).to_numpy()
    r.track(ov["flags"])  # a3_found / adsb_found per id (fs7 store): uncovered = neither found
    t = pd.read_parquet(resolve(ov["flags"]))
    key = "id" if "id" in t else "MVT_ID_mvt"
    t.index = t[key].round().astype("int64")
    miss = ~np.isin(r.ids, t.index.to_numpy())
    assert not miss.any(), f"{ov['flags']}: {int(miss.sum())} frame ids missing"
    t = t.reindex(r.ids)
    unc = (t.a3_found.fillna(0).to_numpy() <= 0) & (t.adsb_found.fillna(0).to_numpy() <= 0)
    return unc & (r.d <= ov.get("max_d_h", 4) * H) & r.fr.apt.isin(ov.get("airports", ["LIRF"])).to_numpy()


def step_overlays(r: Run):
    r.man["overlays"] = {}
    for ov in r.cfg.get("overlays", []):
        k = overlay_flag(r, ov)
        mv = r.member(ov["member"])
        r.pred[k] = (1 - ov["weight"]) * r.pred[k] + ov["weight"] * mv[k]
        r.man["overlays"][ov["name"]] = {"rows": int(k.sum()), "weight": ov["weight"], "member": ov["member"]}
        r.log(f"[overlay {ov['name']}] {int(k.sum()):,} rows, weight {ov['weight']}")
    r.snap("overlay")


def step_cap(r: Run):
    rc = r.cfg.get("rules")
    k = (r.d > r.cfg["cap_d_over_h"] * H) & (r.pred > r.d)
    exempt = np.zeros(len(k), bool)
    band = rc is not None and rules.band_on(rc)
    if band and rules.full(rc)["lirf_orphan_band_mode"] == "floor":
        exempt = rules.orphan_band(r.fr, rules.full(rc))  # the band's floor y_hat >= d replaces the cap
    k &= ~exempt
    r.pred[k] = r.d[k]
    r.pred = np.clip(r.pred, 60, None)
    r.man["cap"] = {"d_over_h": r.cfg["cap_d_over_h"], "rows": int(k.sum())}
    if band:
        r.man["cap"]["band_exempt_rows"] = int(exempt.sum())
    r.log(f"[cap] {int(k.sum())} rows capped at d" + (f" (orphan band exempt: {int(exempt.sum())} rows)" if band else ""))
    r.snap("cap")


def step_level(r: Run):
    lv = r.cfg.get("level")
    for p in level.inputs():
        r.track(p)
    if not lv:
        r.man["level"] = None
        r.coef_level = r.level_coef(0.05)
        r.snap("level")
        return
    r.coef_level = r.level_coef(lv["cap"])
    new, g, coef = level.apply(r.fr, r.pred, lam=lv["lam"], cap=lv["cap"], coef=r.coef_level)
    ch = new - r.pred
    r.log(g[["year", "month", "apt", "n", "pub", "pm", "x", "comp", "deice", "sh"]].round(1).to_string(index=False))
    r.log(f"[level] lam {lv['lam']} cap {lv['cap']}: {int((ch != 0).sum()):,} rows shifted, RMS change {rms(ch):.2f} s")
    r.pred = new
    r.man["level"] = {**lv, "coef": coef.tolist(), "rows_changed": int((ch != 0).sum()),
                      "cells": g[["year", "month", "apt", "n", "pub", "pm", "x", "comp", "deice", "sh"]].round(3).to_dict("records")}
    r.snap("level")


@contextlib.contextmanager
def isolated_imports(directory: Path):
    """Let a post-layer module import its own siblings (e.g. a `rules.py` next to it) even where this
    directory has a module of the same name: shadowing entries leave sys.modules while the layer is
    loaded and called, and come back afterwards."""
    local = {p.stem for p in directory.glob("*.py")}
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k in local}
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        for k in local:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
        sys.path.remove(str(directory))


def step_post(r: Run):
    r.man["post_layers"] = []
    for i, pl in enumerate(r.cfg.get("post_layers", [])):
        path = resolve(pl["module"])
        r.track(path)
        r.track(*[resolve(x) for x in pl.get("inputs", [])])  # files the layer reads (sibling modules, artifacts, CSVs)
        before = r.pred.copy()
        frozen = np.zeros(len(before), bool)
        if pl.get("freeze_nm_bound_rows"):  # rows rule (a) put on an NM-window bound keep that value
            rc = rules.full(r.cfg.get("rules"))
            for dither in (rc["nm_dither_s"], 0.0):  # the levers' rule (a) uses no dither on 2025 frames
                ok, lo, hi = rules.nm_bounds(r.fr, {**rc, "nm_dither_s": dither})
                frozen |= ok & ((np.abs(before - lo) < 1e-6) | (np.abs(before - hi) < 1e-6))
        with isolated_imports(path.parent):
            spec = importlib.util.spec_from_file_location(f"post_layer_{i}_{path.stem}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            out = np.asarray(getattr(mod, pl.get("fn", "apply"))(r.fr.copy(), before.copy(), **r.post_kwargs(pl)), float)
        assert out.shape == before.shape and np.isfinite(out).all(), f"post layer {pl['module']} returned a bad array"
        held = frozen & (np.abs(out - before) > 1e-9)
        out[frozen] = before[frozen]
        ch = out - before
        r.pred = out
        r.stages[f"post{i + 1}_{path.stem}"] = out.copy()
        r.man["post_layers"].append({**pl, "rows_changed": int((ch != 0).sum()), "rms_change": rms(ch),
                                     "rows_frozen_on_nm_bound": int(frozen.sum()), "moves_undone": int(held.sum())})
        r.log(f"[post {path.name}] {int((ch != 0).sum()):,} rows changed, RMS change {rms(ch):.2f} s"
              + (f"; {int(frozen.sum())} rows on an NM-window bound held, {int(held.sum())} moves undone" if frozen.any() else ""))
    r.snap("post")


def step_rules(r: Run):
    rc = r.cfg.get("rules")
    if rc is None:
        r.pred = np.clip(r.pred, 60, None)
        r.man["rules"] = None
        r.log("[rules] disabled")
    else:
        r.pred, counts = rules.apply(r.fr, r.pred, rc)
        r.man["rules"] = {"config": rules.effective(rc), "counts": counts}
        r.log(f"[rules] {counts}")
    r.snap("rules")


def write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def submission(ids_float: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({ID: np.asarray(ids_float, dtype="float64"), TARGET: np.asarray(pred, dtype="float64")})


def check_file(path: Path, template: Path) -> dict:
    """Checklist item 1, file level: columns, dtypes, ids, order, values."""
    s, t = pd.read_parquet(path), pd.read_parquet(template, columns=[ID])
    out = {"columns": list(s.columns) == [ID, TARGET],
           "dtypes": str(s[ID].dtype) == "float64" and str(s[TARGET].dtype) == "float64",
           "rows": len(s) == len(t), "unique": bool(s[ID].is_unique),
           "id_set": set(s[ID].astype("int64")) == set(t[ID].astype("int64")),
           "order": bool(len(s) == len(t) and (s[ID].to_numpy() == t[ID].to_numpy()).all()),
           "finite_nonneg": bool(np.isfinite(s[TARGET]).all() and (s[TARGET] >= 0).all())}
    out["pass"] = all(out.values())
    return out


def preflight(cfg: dict) -> None:
    """Every file the config names must exist before anything runs; list all that do not."""
    need = []
    for name, spec in cfg["members"].items():
        need.append((f"member {name} pred", pred_file(spec["pred"])))
        if "oof" in spec:
            need.append((f"member {name} oof", oof_file(spec["oof"])))
        need += [(f"member {name} input", resolve(x)) for x in spec.get("inputs", [])]
    L = cfg.get("long", {})
    for key in ("phys_inputs", "legacy_source"):
        if key in L:
            need.append((f"long.{key}", resolve(L[key])))
    for ov in cfg.get("overlays", []):
        need.append((f"overlay {ov['name']}", resolve(ov.get("rows") or ov["flags"])))
    for pl in cfg.get("post_layers", []):
        need.append(("post layer", resolve(pl["module"])))
        need += [("post layer input", resolve(x)) for x in pl.get("inputs", [])]
    missing = [f"  {what}: {rel(p)}" for what, p in need if not p.exists()]
    if missing:
        sys.exit("missing inputs (see research/final/NEEDS.md):\n" + "\n".join(missing))


def run_steps(r: Run) -> np.ndarray:
    """Steps 1-9 on r's frame; the prediction after each step is kept in r.stages."""
    step_segments(r)
    step_long(r)
    assert np.isfinite(r.pred).all(), f"{int((~np.isfinite(r.pred)).sum())} rows not covered by segments/long"
    step_overlays(r)
    step_cap(r)
    step_level(r)
    step_post(r)
    step_rules(r)
    return r.pred


def build(cfg: dict, kind: str, out_dir: Path, ref_board: str | None = None, compare: str | None = None) -> dict:
    preflight(cfg)
    r = Run(cfg, kind)
    print(f"== {cfg['name']} on the {kind} frame: {len(r.fr):,} departures")
    pred = run_steps(r)
    for spec in cfg["members"].values():  # every member's prediction and declared upstream files
        r.track(pred_file(spec["pred"]), *[resolve(x) for x in spec.get("inputs", [])])

    # ---- checks
    checks = {}
    rc = rules.full(cfg.get("rules"))
    checks["bounds"] = rules.check(r.fr, pred, rc)
    comp = level.composition_check(r.fr, pred, r.coef_level)
    flagged = comp[comp.flag != ""]
    print("[check 2] airport-month mean vs PRU + 2025 composition offset (gap = pub + comp - our mean):")
    print(comp.round(1).to_string(index=False))
    checks["pru_cells_flagged"] = flagged[["year", "month", "apt", "gap", "deice"]].round(1).to_dict("records")
    bad1 = not checks["bounds"]["pass"]
    if cfg.get("rules") is None:
        print(f"[check 1] rules disabled: bound violations are informational: {checks['bounds']}")
        bad1 = not (checks["bounds"]["finite"] and checks["bounds"]["nonneg"])
    else:
        print(f"[check 1] bounds {checks['bounds']}")

    # ---- outputs
    out_dir = resolve(str(out_dir))
    name = cfg["name"]
    tpl_ids = pd.read_parquet(resolve(FRAMES[kind]["template"]), columns=[ID])[ID].to_numpy()
    outputs = {}
    if kind == "final":
        target = out_dir / (f"{name}_final.parquet" if not bad1 else f"{name}_final.FAILED.parquet")
        write(submission(tpl_ids, pred), target)
        checks["file_final"] = check_file(target, resolve(FRAMES["final"]["template"]))
        outputs[rel(target)] = sha256(target)
        bad1 |= not checks["file_final"]["pass"]
    old = pd.read_parquet(resolve(FRAMES["board"]["template"]), columns=[ID])
    old_ids = old[ID].round().astype("int64").to_numpy()
    pos = pd.Series(np.arange(len(r.ids)), index=r.ids)
    present = np.isin(old_ids, r.ids)
    print(f"[board] old template ids present in the {kind} frame: {int(present.sum()):,}/{len(old_ids):,}")
    assert present.all()
    bpred = pred[pos.reindex(old_ids).to_numpy()]
    btarget = out_dir / (f"{name}_board.parquet" if not bad1 else f"{name}_board.FAILED.parquet")
    write(submission(old[ID].to_numpy(), bpred), btarget)
    checks["file_board"] = check_file(btarget, resolve(FRAMES["board"]["template"]))
    outputs[rel(btarget)] = sha256(btarget)
    bad1 |= not checks["file_board"]["pass"]
    for key, other in (("ref_board", ref_board), ("compare", compare)):
        if not other:
            continue
        ref = pd.read_parquet(resolve(other))
        ref.index = ref[ID].round().astype("int64")
        dv = bpred - ref[TARGET].reindex(old_ids).to_numpy(dtype=float)
        checks[key] = {"file": other, "sha256": sha256(resolve(other)), "rms": rms(dv), "sum_sq_over_n": float((dv ** 2).mean()),
                       "max_abs": float(np.abs(dv).max()), "rows_diff_gt_1s": int((np.abs(dv) > 1).sum())}
        tag = "[check 3]" if key == "ref_board" else "[compare]"
        print(f"{tag} Jan/Jul vs {other}: RMS {rms(dv):.2f} s, sum sq / N {(dv ** 2).mean():.1f}, max |diff| {np.abs(dv).max():.0f} s")
    stages = pd.DataFrame({"id": r.ids, "d": r.d, **{k: v for k, v in r.stages.items()}})
    spath = out_dir / f"{name}_stages.parquet"
    write(stages, spath)

    fail = []
    if bad1:
        fail.append(f"checklist item 1 failed: {checks['bounds']} {checks.get('file_final')} {checks.get('file_board')}")
    if ref_board and checks["ref_board"]["rms"] > 10:
        fail.append(f"checklist item 3: Jan/Jul RMS {checks['ref_board']['rms']:.2f} s > 10 s vs {ref_board}")

    # ---- manifest
    for p in CODE:
        r.track(p)
    man = {**r.man,
           "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "command": " ".join([Path(sys.executable).name] + sys.argv),
           "git": git_state(), "notes": cfg.get("notes", []), "config": cfg, "rows": len(r.fr),
           "checks": checks, "failed": fail, "inputs": r.inputs, "outputs": outputs, "stages_file": rel(spath),
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                           "scipy": __import__("scipy").__version__, "pyarrow": __import__("pyarrow").__version__}}
    copies = [out_dir / f"{name}.manifest.json"]
    if out_dir.resolve().is_relative_to((C / "submissions").resolve()):  # scratch runs leave no copy in git
        copies.append(HERE / "manifests" / f"{name}.json")
    for path in copies:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(man, indent=1, default=str))
    print(f"-> {', '.join(outputs)}; manifest {rel(out_dir / f'{name}.manifest.json')}")
    if fail:
        print("\n!!! CHECKS FAILED !!!\n" + "\n".join(fail))
    return man


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--frame", choices=["final", "board"])
    ap.add_argument("--ref-board", help="last complete board candidate from the same final store: item 3, fails above 10 s RMS")
    ap.add_argument("--compare", help="any board-keyed submission: print the Jan/Jul RMS difference (informational)")
    ap.add_argument("--out-dir", default="submissions/final")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    kind = a.frame or cfg.get("frame", "final")
    man = build(cfg, kind, Path(a.out_dir), a.ref_board, a.compare)
    sys.exit(1 if man.get("failed") else 0)


if __name__ == "__main__":
    main()
