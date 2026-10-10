"""L1a: reconcile flight-level predictions with EUROCONTROL's daily pre-departure delay.

Observation (B section 5.3). For airport a and takeoff UTC day g, EUROCONTROL publishes
DLY_ALL_PRE_2 (minutes); on 2025 labels  K_g = 60 * DLY_ALL_PRE_2 = sum_i max(0, delta_i) + eps_g
over the competition's own departures, where delta_i = off-block - schedule = d_i - y_i
(d = takeoff - schedule is supplied). eps_g is a small definitional term (the +-12 s dither,
and a few flights more or fewer in EUROCONTROL's population on some days).

Model, per airport-day:

* ordinary rows (d <= 1 h). Their predicted contributions max(0, d - p) are summed. The
  error of that sum is a common day effect: e_g = sum_ord [max(0,d-y) - max(0,d-p)]
  = -n_eff * s_g, where n_eff counts ordinary rows with p < d and s_g is the day's mean
  residual y - p over them (corr(s, mean residual of those rows) 0.85-0.998 in 2025).
  Prior s_g ~ t_5(mu_s, sigma_s) per airport.
* long rows (d > 1 h). Each has a discrete prior over y: an on-time/fill point y = d
  (delta = 0, contributes 0) with probability pi_i, and a held component (the aircraft left
  the stand late and taxied normally) given by the airport's empirical held-taxi spread,
  located so that the prior mean equals the incoming prediction p_i (so the layer is the
  identity when the day carries no information). pi_i comes from the L1b mode model when
  given, else it is implied by p_i at LIRF (pi = (p - tau)/(d - tau)) and set to the
  airport's 2025 on-time frequency elsewhere. NM-matched rows have their support
  truncated to the NM join window (|BLOCK - LOBT| <= 3,606 s).
* likelihood: R_g = K_g - sum_ord max(0, d - p) = A_g - n_eff * s_g + eps_g,
  A_g = sum_long max(0, d - y), eps_g ~ t_3(mu_eps, sqrt(s0^2 + kappa^2 dn^2)) where
  dn = FLT_DEP_IFR_2 - (our departure count that day).

The posterior is exact on a 60 s grid: forward-backward over the long rows' discrete
contributions, and a grid over s. Outputs are posterior means: long rows E[y | K], delayed
ordinary rows p + E[s | K], other ordinary rows p + rho * E[s | K] (rho = regression of
their day-mean residual on s, fitted on 2025).

Every parameter is fitted on 2025 departures and the incoming predictions; the month-out
evaluation fits on the other 11 months. ``apply`` uses the all-2025 fit stored in
``artifacts/daily_params.json`` and EUROCONTROL's 2026 file. Airports are gated by
``GATE`` (decided by the month-out evaluation, see RESULTS.md).

L1a-v2 (``apply(..., v2=..., v2_loadings=True)``; off by default, so earlier manifests replay
unchanged). The posterior over s is unchanged; an ordinary row moves by b * (E[s|K] - s_typical),
with b looked up by (delayed: d - p > 0) x d sub-band (<= 15, 15-30, 30-45, 45-60 min) instead of
1 / rho. b is the through-origin slope of y - p on that day shift over the stack's own 2025
pseudo-final rows (``daily.py v2 --rows``; in the pseudo-final refitted without the held month),
shrunk to its delayed/other parent. The adopted fit: delayed 0.35 / 0.52 / 0.98 / 1.33, other
0.18 / 0.38 / 0.22 / 0.21. Evidence classes (ADS-B event, a3 grade, NM match), airport and hour
added nothing on top of the d sub-band. ``v2_spreads`` (held spreads by airport x ADS-B evidence)
is implemented for the record and rejected (RESULTS.md, L1a-v2).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special

try:
    from . import lv_common as common, lv_rules as rules
except ImportError:  # run as a script from this folder
    import lv_common as common  # type: ignore
    import lv_rules as rules  # type: ignore

H = 60.0                 # contribution grid (s)
LONG = 3600.0
LONG_BY_APT: dict[str, float] = {}     # per-airport override of the long-row threshold (s)


def long_thr(apt) -> np.ndarray:
    a = np.asarray(apt)
    return np.array([LONG_BY_APT.get(x, LONG) for x in a]) if LONG_BY_APT else np.full(len(a), LONG)
NU_S, NU_E = 5.0, 3.0
N_S = 81                 # grid points for s
HELD_Q = np.linspace(0.025, 0.975, 20)
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
PARAMS = common.ART / "daily_params.json"
DEMEAN_MONTH = False   # True: remove each airport-month's mean ordinary-row day shift (leave the monthly
                       # level to the PRU monthly calibration that runs before the post-layers)
GATE: dict[str, str] = {a: "full" for a in ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]}
# decided by the month-out evaluation (RESULTS.md): every airport passes in "full" mode with the
# held-only long-row prior outside LIRF; "level" = ordinary-row day shift only; absent = off

# L1a-v2 ordinary-row loadings (off unless ``apply`` gets ``loadings`` / ``loadings_rows``). An
# ordinary row moves by b * (E[s|K] - s_typical), b looked up by (delayed: d - p > 0) x d sub-band,
# instead of 1 (delayed) / rho (other). b is the through-origin slope of the row residual y - p on
# the day's posterior shift, fitted on the stack's own month-out rows; each sub-band slope is shrunk
# to its delayed/other parent (DerSimonian-Laird) and clipped.
LOAD_EDGES = (900.0, 1800.0, 2700.0)    # sub-bands of d (s): <= 15, 15-30, 30-45, 45-60 min
LOAD_CLIP = (-0.5, 2.5)
LOAD_FIT_MAX_Y = 10800.0                # the fit leaves out ordinary rows with y > 3 h (stacking's board-like rule)
# L1a-v2 held spreads (off unless ``v2_spreads``): a long row's held component takes the spread of
# y - p over held long rows of its airport x evidence class (a3_found), fitted on the stack's own
# month-out rows, instead of the airport's spread of held y. LIRF keeps its spread: its held
# component sits at tau with the on-time probability implied by p, not at p.
SPREAD_MIN_ROWS = 200
SPREAD_SKIP = ("LIRF",)


def load_pru(year: int) -> pd.DataFrame:
    """EUROCONTROL daily file; falls back to the ten-airport extract kept in artifacts/ (same rows,
    all columns, written from the original by ``python3 daily.py extract``)."""
    src = common.PRU / f"all_pre_departure_delays_{year}.csv"
    if not src.exists():
        src = common.ART / f"pru_daily_10apt_{year}.csv"
    pr = pd.read_csv(src)
    pr["day"] = pd.to_datetime(pr.FLT_DATE).dt.strftime("%Y-%m-%d")
    pr = pr[pr.APT_ICAO.isin(APTS)]
    return pr[["APT_ICAO", "day", "DLY_ALL_PRE_2", "FLT_DEP_IFR_2"]].rename(columns={"APT_ICAO": "apt"})


def _t_logpdf(z, nu):
    return (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log(nu * math.pi)
            - (nu + 1) / 2 * np.log1p(z * z / nu))


def _rs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


# ----------------------------------------------------------------------------- fitting
def day_table(fr: pd.DataFrame, p: np.ndarray, pru: pd.DataFrame) -> pd.DataFrame:
    """Per airport-day aggregates on labelled rows (fitting only)."""
    y = fr["y"].to_numpy(float)
    d = fr["d"].to_numpy(float)
    lg = d > long_thr(fr["apt"].to_numpy())
    cp = np.maximum(0, d - p)
    ct = np.maximum(0, d - y)
    key = (fr["apt"] + "|" + fr["day"]).to_numpy()
    res = y - p
    deld = ~lg & (d - p > 0)
    nd = ~lg & ~(d - p > 0)
    t = pd.DataFrame({"key": key, "apt": fr["apt"].to_numpy(), "day": fr["day"].to_numpy(),
                      "cp_o": np.where(lg, 0, cp), "ct_o": np.where(lg, 0, ct), "ct_l": np.where(lg, ct, 0),
                      "ct": ct, "deld": deld.astype(float), "res_nd": np.where(nd, res, 0.0), "nd": nd.astype(float)})
    g = t.groupby(["apt", "day"]).agg(n=("ct", "size"), cp_o=("cp_o", "sum"), ct_o=("ct_o", "sum"),
                                      ct_l=("ct_l", "sum"), ct=("ct", "sum"), neff=("deld", "sum"),
                                      res_nd=("res_nd", "sum"), nnd=("nd", "sum")).reset_index()
    g = g.merge(pru, on=["apt", "day"], how="left")
    g["K"] = g.DLY_ALL_PRE_2 * 60.0
    g["dn"] = g.FLT_DEP_IFR_2 - g.n
    g["eps"] = g.K - g.ct
    g["s"] = -(g.ct_o - g.cp_o) / g.neff.clip(lower=1)
    g["b_nd"] = g.res_nd / g.nnd.clip(lower=1)
    return g


def fit(fr: pd.DataFrame, p: np.ndarray, pru: pd.DataFrame) -> dict:
    """Per-airport parameters from labelled rows ``fr`` with incoming predictions ``p``."""
    g = day_table(fr, p, pru)
    g = g[g.K.notna()]
    y = fr["y"].to_numpy(float)
    d = fr["d"].to_numpy(float)
    apt = fr["apt"].to_numpy()
    nm = fr["nm"].to_numpy(bool)
    out = {}
    for a in APTS:
        q = g[g.apt == a]
        if len(q) < 20:
            continue
        s_mu = float(np.median(q.s))
        s_sc = max(_rs(q.s), 3.0)
        ok = q.dn == 0
        e_mu = float(np.median(q.eps[ok])) if ok.sum() >= 10 else float(np.median(q.eps))
        s0 = max(_rs(q.eps[ok]) if ok.sum() >= 10 else _rs(q.eps), 60.0)
        bad = q[~ok]
        # a flight EUROCONTROL counts and we do not (or vice versa) carries an unknown
        # contribution; floor its scale at the airport's mean contribution per departure
        per_flight = float(np.median(q.K / q.n))
        if len(bad) >= 10:
            k2 = np.median(np.maximum((bad.eps - e_mu) ** 2 - s0 ** 2, 0) / bad.dn ** 2)
            kappa = float(max(np.sqrt(k2), per_flight, 60.0))
        else:
            kappa = max(per_flight, 600.0)
        has_nd = q.nnd > 20
        rho = float(np.clip(np.polyfit(q.s[has_nd], q.b_nd[has_nd], 1)[0], 0, 1)) if has_nd.sum() > 20 else 0.0
        # long rows: on-time frequency by NM status and band; held-taxi spread
        thr = LONG_BY_APT.get(a, LONG)
        m = (apt == a) & (d > thr)
        ontime = np.abs(d - y) <= 600
        held = m & (y < d - 600)
        freq = {}
        for tag, sel in (("nm", m & nm), ("orphan", m & ~nm)):
            for lo, hi, bname in ((thr, 14400, "1-4h"), (14400, 1e12, ">4h")):
                ss = sel & (d > lo) & (d <= hi)
                freq[f"{tag}|{bname}"] = float((ontime[ss].sum() + 0.5) / (ss.sum() + 25))
        hy = y[held]
        tau = float(np.mean(hy)) if len(hy) > 50 else 1200.0
        spread = (np.quantile(hy, HELD_Q) - tau).tolist() if len(hy) > 50 else (np.linspace(-500, 900, len(HELD_Q))).tolist()
        out[a] = {"s_mu": s_mu, "s_sc": s_sc, "e_mu": e_mu, "s0": s0, "kappa": kappa, "rho": rho,
                  "tau": tau, "spread": spread, "freq": freq, "days": int(len(q)), "long": thr}
    return out


# --------------------------------------------------------------------------- posterior
OMEGA = 0.1              # weight of the broad component (any y inside the feasible window)


def _tilt(ys, ws, target):
    """Exponentially tilt discrete weights so that their mean equals ``target``
    (minimum-KL change); no-op when target is outside the support."""
    if not (ys.min() < target < ys.max()):
        return ws
    z = (ys - target) / max(ys.max() - ys.min(), 1.0)
    lo, hi = -60.0, 60.0
    for _ in range(60):
        th = 0.5 * (lo + hi)
        w = ws * np.exp(th * z - np.max(th * z))
        m = (w * z).sum() / w.sum()
        if m > 0:
            hi = th
        else:
            lo = th
    w = ws * np.exp(th * z - np.max(th * z))
    return w / w.sum()


def _long_prior(d, p, nm, ml, pi, tau, spread, join=3606.0, dither=0.0, omega=OMEGA, flo=np.nan, fhi=np.nan):
    """Discrete prior over y for one long row: (y support, weights), mean = p.

    Components: on-time/fill point y = d (weight pi), held taxi (the airport's held spread
    located so the mixture mean is p), and a broad component spread over the feasible
    window. NM-matched rows are truncated to the join window; the weights are then tilted
    back to mean p."""
    sp = np.asarray(spread, float)
    if nm and np.isfinite(ml):
        lo_w, hi_w = ml - join - dither, ml + join + dither
    else:
        lo_w, hi_w = 30.0, np.inf
    if p >= d:                       # fill / floor-at-d rows: no held mass left
        return np.array([p]), np.array([1.0])
    pi = float(np.clip(pi, 0.0, 0.999))
    loc = (p - pi * d) / (1 - pi)
    if loc < 60:
        pi, loc = 0.0, p
    held = np.maximum(loc + sp - np.mean(sp), 30.0)
    broad = np.linspace(max(30.0, lo_w), min(d, hi_w), 9) if min(d, hi_w) > max(30.0, lo_w) else np.array([p])
    ys = np.concatenate([[d], held, broad])
    ws = np.concatenate([[pi * (1 - omega)], np.full(len(held), (1 - pi) * (1 - omega) / len(held)),
                         np.full(len(broad), omega / len(broad))])
    keep = (ys >= lo_w - 1e-6) & (ys <= hi_w + 1e-6) & (ws > 0)
    if np.isfinite(flo) and np.isfinite(fhi):        # L0e: recovered-orphan NM window excluded
        keep2 = keep & ~((ys > flo) & (ys < fhi))
        keep = keep2 if keep2.any() else keep
    if not keep.any():
        return np.array([float(np.clip(p, lo_w, hi_w))]), np.array([1.0])
    ys, ws = ys[keep], ws[keep] / ws[keep].sum()
    ws = _tilt(ys, ws, float(np.clip(p, lo_w, hi_w)))
    return ys, ws


def load_of(d, p, load: dict) -> np.ndarray:
    """L1a-v2 loading of each ordinary row: (delayed, d sub-band) lookup in ``load``."""
    band = np.searchsorted(np.asarray(load["edges"], float), d, side="left")
    return np.where(d - p > 0, np.asarray(load["del"], float)[band], np.asarray(load["nd"], float)[band])


def _day(K, dn, d, p, nm, ml, pi, prm, level_only=False, dither=0.0, omega=OMEGA, flo=None, fhi=None, load=None,
         spread=None):
    """Posterior means for one airport-day. Returns (new p, E[s]). ``spread``: per-row held spreads
    (L1a-v2; None or a None entry = the airport's ``prm["spread"]``, a number = that spread times it)."""
    lg = d > prm.get("long", LONG)
    out = p.copy()
    cp = np.maximum(0, d - p)
    R = K - cp[~lg].sum()
    deld = ~lg & (d - p > 0)
    neff = float(deld.sum())
    # s grid and prior
    s_grid = prm["s_mu"] + prm["s_sc"] * np.linspace(-6, 6, N_S)
    lp_s = _t_logpdf((s_grid - prm["s_mu"]) / prm["s_sc"], NU_S)
    # extra takeoff dither of the final file (+-12 s uniform per flight, variance 48 s^2) on every
    # row that contributes to the day sum
    n_contrib = neff + float(lg.sum())
    se = math.sqrt(prm["s0"] ** 2 + (prm["kappa"] * dn) ** 2 + (48.0 * n_contrib if dither > 0 else 0.0))
    # long-row priors on the contribution grid
    rows = np.flatnonzero(lg)
    sup = []
    tot = 0
    for i in rows:
        sp_i = prm["spread"] if spread is None or spread[i] is None else spread[i]
        if np.isscalar(sp_i):                       # L1a-v2 "scale": the airport's spread times a factor
            sp_i = np.asarray(prm["spread"], float) * float(sp_i)
        ys, ws = _long_prior(d[i], p[i], nm[i], ml[i], pi[i], prm["tau"], sp_i, dither=dither, omega=omega,
                             flo=np.nan if flo is None else flo[i], fhi=np.nan if fhi is None else fhi[i])
        cb = np.rint(np.maximum(0, d[i] - ys) / H).astype(int)
        sup.append((ys, ws, cb))
        tot += int(cb.max())
    B = tot + 1
    x = np.arange(B) * H
    # F(x) = sum_s prior(s) * lik(R - x + neff*s)
    z = (R - x[:, None] + neff * s_grid[None, :] - prm["e_mu"]) / se
    lw = lp_s[None, :] + _t_logpdf(z, NU_E)
    mx = lw.max()
    Fx = np.exp(lw - mx).sum(1)
    # forward
    alphas = []
    a = np.zeros(B)
    a[0] = 1.0
    for ys, ws, cb in sup:
        alphas.append(a)
        nxt = np.zeros(B)
        for w, c in zip(ws, cb):
            nxt[c:] += w * a[:B - c]
        a = nxt / nxt.sum()
    a_all = a
    # E[s | R]
    ps = (a_all[:, None] * np.exp(lw - mx)).sum(0)
    if ps.sum() <= 0 or not np.isfinite(ps.sum()):
        return out, 0.0
    Es = float((ps * s_grid).sum() / ps.sum())
    shift = Es - prm["s_mu"]          # prior mean of s is the airport's typical bias: shift relative to it
    if load is None:
        out[deld] = p[deld] + shift
        nd = ~lg & ~deld
        out[nd] = p[nd] + prm["rho"] * shift
    else:                             # L1a-v2: per-class loadings of the ordinary-row shift
        out[~lg] = p[~lg] + load_of(d[~lg], p[~lg], load) * shift
    if level_only or not len(rows):
        return out, Es
    # backward messages and per-row posteriors
    beta = Fx / Fx.max()
    for j in range(len(rows) - 1, -1, -1):
        ys, ws, cb = sup[j]
        al = alphas[j]
        f = np.array([(al[:B - c] * beta[c:]).sum() for c in cb])
        post = ws * f
        if post.sum() > 0 and np.isfinite(post.sum()):
            out[rows[j]] = float((post * ys).sum() / post.sum())
        nb = np.zeros(B)
        for w, c in zip(ws, cb):
            nb[:B - c] += w * beta[c:]
        beta = nb / max(nb.max(), 1e-300)
    return out, Es


def posterior(fr: pd.DataFrame, p: np.ndarray, pru: pd.DataFrame, params: dict, pi: np.ndarray | None = None,
              airports: dict | None = None, dither: float = 0.0, held_only: bool = True,
              forbid: tuple | None = None, load: dict | None = None, record: list | None = None,
              spread_rows: np.ndarray | None = None) -> np.ndarray:
    """Posterior-mean predictions. ``airports`` maps airport -> "full" | "level".

    ``held_only`` (default): outside LIRF a delayed departure's long-row prior has no on-time
    point and no broad component, only the held-taxi spread around p (B section 5.4: the
    on-time off-block mode is a LIRF phenomenon; with a fill point EDDF lost +693 month-out).
    ``load``: L1a-v2 ordinary-row loadings (None = 1 / rho). ``record``: a list that receives
    (airport, day, E[s|K] - s_typical) for every airport-day (the loading fit's regressor).
    ``spread_rows``: L1a-v2 per-row held spreads (object array aligned with ``fr``; None entries keep
    the airport's spread)."""
    assert load is None or not DEMEAN_MONTH, "DEMEAN_MONTH assumes the 1 / rho loadings"
    p = np.asarray(p, float)
    out = p.copy()
    d = fr["d"].to_numpy(float)
    apt = fr["apt"].to_numpy()
    nm = fr["nm"].to_numpy(bool)
    ml = fr["ml"].to_numpy(float)
    day = fr["day"].to_numpy()
    if pi is None:
        pi = implied_pi(fr, p, params)
    K = pru.set_index(["apt", "day"])
    for a, mode in (airports or {x: "full" for x in APTS}).items():
        if a not in params:
            continue
        prm = params[a]
        ia = np.flatnonzero(apt == a)
        if not len(ia):
            continue
        n_by_day = pd.Series(day[ia]).value_counts()
        shifts = []
        for g, idx in pd.Series(ia).groupby(day[ia]):
            idx = idx.to_numpy()
            if (a, g) not in K.index:
                continue
            row = K.loc[(a, g)]
            if not np.isfinite(row.DLY_ALL_PRE_2):
                continue
            dn = float(row.FLT_DEP_IFR_2 - n_by_day[g]) if np.isfinite(row.FLT_DEP_IFR_2) else 10.0
            lirf_like = a == "LIRF" or not held_only
            new, Es = _day(60.0 * row.DLY_ALL_PRE_2, dn, d[idx], p[idx], nm[idx], ml[idx],
                           pi[idx] if lirf_like else np.zeros(len(idx)), prm,
                           level_only=(mode == "level"), dither=dither, omega=OMEGA if lirf_like else 0.0,
                           flo=None if forbid is None else forbid[0][idx], fhi=None if forbid is None else forbid[1][idx],
                           load=load, spread=None if spread_rows is None else spread_rows[idx])
            out[idx] = new
            shifts.append((g[:7], idx, Es - prm["s_mu"]))
            if record is not None:
                record.append((a, g, Es - prm["s_mu"]))
        if DEMEAN_MONTH and shifts:
            sh = pd.DataFrame({"m": [x[0] for x in shifts], "v": [x[2] for x in shifts]})
            mm = sh.groupby("m").v.mean()
            for mon, idx, _v in shifts:
                lg = d[idx] > prm.get("long", LONG)
                fac = np.where(lg, 0.0, np.where(d[idx] - p[idx] > 0, 1.0, prm["rho"]))
                out[idx] = out[idx] - fac * mm[mon]
    return out


def implied_pi(fr: pd.DataFrame, p: np.ndarray, params: dict) -> np.ndarray:
    """On-time probability of long rows without a mode model: implied by p at LIRF,
    the airport's 2025 frequency elsewhere."""
    d = fr["d"].to_numpy(float)
    apt = fr["apt"].to_numpy()
    nm = fr["nm"].to_numpy(bool)
    pi = np.zeros(len(d))
    for a, prm in params.items():
        m = apt == a
        if a == "LIRF":
            pi[m] = np.clip((p[m] - prm["tau"]) / np.maximum(d[m] - prm["tau"], 1.0), 0, 1)
        else:
            for tag, sel in (("nm", nm), ("orphan", ~nm)):
                pi[m & sel & (d <= 14400)] = prm["freq"][f"{tag}|1-4h"]
                pi[m & sel & (d > 14400)] = prm["freq"][f"{tag}|>4h"]
    return pi


# -------------------------------------------------------------------------- evaluation
def month_out(dep: pd.DataFrame, p: np.ndarray, pru: pd.DataFrame, airports: dict, pi_fn=None,
              verbose=False, forbid: tuple | None = None) -> np.ndarray:
    out = np.asarray(p, float).copy()
    mon = dep["month"].to_numpy()
    for mo in range(1, 13):
        tr = mon != mo
        prm = fit(dep[tr], p[tr], pru)
        te = np.flatnonzero(~tr)
        sub = dep.iloc[te]
        pi = None if pi_fn is None else pi_fn(sub, p[te], mo)
        fb = None if forbid is None else (forbid[0][te], forbid[1][te])
        out[te] = posterior(sub, p[te], pru, prm, pi=pi, airports=airports, forbid=fb)
        if verbose:
            print(f"  month {mo} done", flush=True)
    return out


def level_calibrated_c6(dep: pd.DataFrame, also_exclude: int | None = None) -> np.ndarray:
    """c6 2025 OOF after the assembler's airport-month level calibration (research/final/level.py,
    lam 0.75, cap 0.05 as v32c), with composition coefficients fitted month-out (each month's
    coefficients exclude that month, and ``also_exclude`` too when given) - the base L1a sees in the
    shipped order (level, then post-layers)."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import level  # research/final/level.py (assembler lane), used read-only
    fr = pd.DataFrame({"apt": dep["apt"].to_numpy(), "year": dep["year"].to_numpy(), "month": dep["month"].to_numpy(),
                       "sr": ((dep["stand"] != "") & (dep["runway"] != "")).to_numpy(), "d": dep["d"].to_numpy()})
    b0 = dep["p"].to_numpy(float)
    out = b0.copy()
    for m in sorted(set(int(x) for x in dep["month"])):
        s = dep["month"].to_numpy() == m
        exc = [(2025, m)] + ([(2025, also_exclude)] if also_exclude and also_exclude != m else [])
        out[s] = level.apply(fr[s].reset_index(drop=True), b0[s], lam=0.75, cap=0.05,
                             coef=level.fit_coef(0.05, exclude=exc))[0]
    return out


def fit_base(oof: Path | None = None, col: str = "prediction", after_level: bool = True,
             exclude_month: int | None = None) -> tuple[pd.DataFrame, np.ndarray, str]:
    """2025 rows and the base L1a's parameters are fitted on: the incoming prediction after the L0
    rules and L0e, on every 2025 departure the source covers, minus ``exclude_month``.

    Sources: the c6 12-month-out OOF (default; after the level calibration when ``after_level``,
    its coefficients also excluding ``exclude_month``), or ``oof`` = any parquet with ``id`` and
    column ``col`` - e.g. the pseudo-final's per-row output (submissions/final/pseudo2025/
    NAME_MODE.parquet, col "level" = the prediction entering the post layers)."""
    try:
        from . import recovered
    except ImportError:
        import recovered  # type: ignore
    dep = common.load_2025()
    if oof is not None:
        o = pd.read_parquet(oof, columns=["id", col])
        o["id"] = o["id"].round().astype("int64")
        assert o["id"].is_unique, f"{oof}: duplicate ids"
        dep = dep.drop(columns=["p"]).merge(o.rename(columns={col: "p"}), on="id", how="inner", validate="1:1")
        assert dep["p"].notna().all() and len(dep), f"{oof}: no usable rows"
        what = f"{Path(oof).name}[{col}]"
    elif after_level:
        dep = dep.assign(p=level_calibrated_c6(dep, also_exclude=exclude_month))
        what = "c6 12-month-out OOF + level calibration (L1c, coefficients month-out)"
    else:
        what = "c6 12-month-out OOF"
    if exclude_month is not None:
        dep = dep[dep["month"].to_numpy() != exclude_month].reset_index(drop=True)
    p0 = rules.apply(dep, dep["p"].to_numpy(float))
    p0 = recovered.rule(dep, p0, recovered.recovered_ml(dep, common.load_2025(arrivals=True)))
    return dep, p0, what + " + L0 rules (a)+(c)+(d) + L0e"


def nested_params(held_month: int, oof: Path | None = None, col: str = "prediction", after_level: bool = True) -> dict:
    """Parameters fitted without 2025 month ``held_month`` (pseudo-final mode), cached on disk."""
    import hashlib
    key = f"{held_month}|{oof}|{col}|{after_level}|{common.sha256(Path(oof)) if oof else ''}"
    f = common.CACHE / f"daily_params_m{held_month}_{hashlib.sha1(key.encode()).hexdigest()[:10]}.json"
    if f.exists():
        return json.loads(f.read_text())
    dep, p0, _ = fit_base(oof, col, after_level, exclude_month=held_month)
    params = fit(dep, p0, load_pru(2025))
    common.CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(params))
    return params


def build_artifact(gate: dict, path: Path = PARAMS, oof: Path | None = None, after_level: bool = True,
                   col: str = "prediction") -> dict:
    """Fit the 2026 parameters on 2025 and write the artifact.

    Default: c6 + level calibration (L1c) + L0 + L0e on all of 2025. With ``oof`` (and ``col``), fit
    on that file's rows instead - for the shipped stack, the pseudo-final's per-row output with
    col "level" (four months of 2025, the stack's own month-out predictions)."""
    dep, p0, what = fit_base(oof, col, after_level and oof is None)
    params = fit(dep, p0, load_pru(2025))
    src = Path(oof) if oof else common.OOF
    art = {"params": params, "gate": gate, "grid_s": H, "long_s": LONG, "long_by_apt": LONG_BY_APT, "omega": OMEGA,
           "fitted_on": f"2025 departures ({len(dep):,} rows, months {sorted(set(int(x) for x in dep.month))}), base = {what}",
           "inputs": {"oof": common.rel(src), "oof_col": col if oof else "prediction",
                      "oof_sha256": common.sha256(src), "after_level": bool(after_level and oof is None),
                      "all_pre_departure_delays_2025.csv": common.sha256(common.PRU / "all_pre_departure_delays_2025.csv"),
                      "all_pre_departure_delays_2026.csv": common.sha256(common.PRU / "all_pre_departure_delays_2026.csv")},
           "script": "python3 research/final/levers/daily.py artifact" + (f" --oof {src} --col {col}" if oof else "")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(art, indent=1, sort_keys=True))
    return art


def _recovered_mod():
    try:
        from . import recovered
    except ImportError:
        import recovered  # type: ignore
    return recovered


def _frame_inputs(frame: pd.DataFrame) -> tuple:
    """The levers frame, the EUROCONTROL rows of its year(s), the recovered-orphan MVT - LOBT and
    the L0e forbidden intervals - what ``apply`` hands to ``posterior``."""
    recovered = _recovered_mod()
    fr = common.ensure_columns(frame, ["apt", "day", "d", "nm", "ml", "year"])
    years = sorted(set(fr["year"].astype(int)))
    pru = pd.concat([load_pru(y) for y in years], ignore_index=True)
    fr2 = common.ensure_columns(frame, ["apt", "day", "d", "nm", "ml", "year", "flight", "ades", "mvt"])
    arr = common.load_final(arrivals=True) if min(years) >= 2026 else common.load_2025(arrivals=True)
    mlr = recovered.recovered_ml(fr2, arr)
    sel = np.isin(fr2["apt"].to_numpy(), list(recovered.AIRPORTS)) & np.isfinite(mlr)
    forbid = (np.where(sel, mlr - recovered.JOIN, np.nan), np.where(sel, mlr + recovered.JOIN, np.nan))
    return fr, fr2, years, pru, mlr, forbid


# ------------------------------------------------------------------ L1a-v2: loadings, spreads
def _slope(x, r, blk) -> tuple[float, float]:
    """Through-origin slope of r on x and its airport-day clustered standard error."""
    sxx = float((x * x).sum())
    if len(x) < 2 or sxx <= 0:
        return float("nan"), float("inf")
    b = float((x * r).sum() / sxx)
    s = pd.Series((r - b * x) * x).groupby(blk).sum().to_numpy()
    return b, float(np.sqrt((s ** 2).sum()) / sxx)


def fit_loadings(apt, d, p, y, xhat, blk, edges=LOAD_EDGES) -> dict:
    """Loadings of the ordinary-row day shift by (delayed, d sub-band), from labelled rows whose
    day shift ``xhat`` = E[s|K] - s_typical was computed month-out. Each sub-band slope is shrunk to
    its parent (all delayed / all other rows) with a DerSimonian-Laird between-band variance."""
    d, p, y, xhat = (np.asarray(v, float) for v in (d, p, y, xhat))
    blk = np.asarray(blk)
    ok = (d <= long_thr(apt)) & (y <= LOAD_FIT_MAX_Y) & np.isfinite(xhat)
    band = np.searchsorted(np.asarray(edges, float), d, side="left")
    r = y - p
    out = {"edges": list(edges), "clip": list(LOAD_CLIP), "fit": {}}
    for tag, sel in (("del", ok & (d - p > 0)), ("nd", ok & ~(d - p > 0))):
        bp, sp = _slope(xhat[sel], r[sel], blk[sel])
        bs, ses, ns = [], [], []
        for k in range(len(edges) + 1):
            s = sel & (band == k)
            b, se = _slope(xhat[s], r[s], blk[s])
            bs.append(b), ses.append(se), ns.append(int(s.sum()))
        bs, ses = np.array(bs), np.array(ses)
        good = np.isfinite(bs) & np.isfinite(ses) & (ses > 0)
        dev = np.where(good, bs - bp, 0.0)
        w = np.where(good, 1 / np.where(good, ses, 1.0) ** 2, 0.0)
        k_ = int(good.sum())
        den = w.sum() - (w ** 2).sum() / max(w.sum(), 1e-12)
        tau2 = max(0.0, float(((w * dev ** 2).sum() - (k_ - 1)) / den)) if k_ > 1 and den > 0 else 0.0
        keep = np.where(good, tau2 / (tau2 + np.where(good, ses, 1.0) ** 2), 0.0)
        out[tag] = np.clip(bp + keep * dev, *LOAD_CLIP).round(6).tolist()
        out["fit"][tag] = {"parent": round(bp, 6), "parent_se": round(sp, 6), "raw": [round(float(v), 6) for v in bs],
                           "se": [round(float(v), 6) for v in ses], "rows": ns, "tau2": round(tau2, 6)}
    return out


def evidence(ids, cover: str | Path) -> np.ndarray:
    """True where the flight has an adsb.lol v3 departure event (a3_found, the fs7 store's column;
    research/final/inputs/<frame>/cover.parquet). Absent ids count as no evidence."""
    c = pd.read_parquet(common.resolve(cover), columns=["MVT_ID_mvt", "a3_found"])
    c = c.assign(id=c["MVT_ID_mvt"].round().astype("int64")).drop_duplicates("id").set_index("id")
    ids = np.asarray(ids).round().astype("int64")
    return (c["a3_found"].reindex(ids).fillna(0).to_numpy(float) > 0)


def fit_spreads(apt, d, p, y, ev, kind: str = "quantiles") -> dict:
    """Held-taxi spreads around the incoming prediction, per airport x evidence class, from held long
    rows (d past the long threshold, y < d - 600) of the stack's month-out predictions.
    ``kind`` "quantiles": the HELD_Q quantiles of y - p; "scale": a factor on the airport's base
    spread, the class's robust SD of y - p over the airport's (keeps the airport's average width).
    A class with fewer than SPREAD_MIN_ROWS rows takes the airport's pooled value; an airport with
    fewer keeps the base spread (no entry)."""
    apt, d, p, y, ev = (np.asarray(v) for v in (apt, d, p, y, ev))
    d, p, y = d.astype(float), p.astype(float), y.astype(float)
    held = (d > long_thr(apt)) & (y < d - 600)
    r = y - p
    out = {}
    for a in APTS:
        m = held & (apt == a)
        if m.sum() < SPREAD_MIN_ROWS:
            continue
        pooled = np.quantile(r[m], HELD_Q)
        out[a] = {}
        for tag, sel in (("a3", m & ev.astype(bool)), ("none", m & ~ev.astype(bool))):
            if kind == "scale":
                out[a][tag] = round(_rs(r[sel]) / _rs(r[m]), 4) if sel.sum() >= SPREAD_MIN_ROWS else 1.0
            else:
                q = np.quantile(r[sel], HELD_Q) if sel.sum() >= SPREAD_MIN_ROWS else pooled
                out[a][tag] = [round(float(v), 3) for v in q]
        out[a]["rows"] = {"a3": int((m & ev.astype(bool)).sum()), "none": int((m & ~ev.astype(bool)).sum())}
    return out


def spread_rows(apt, ev, spreads: dict, skip=SPREAD_SKIP) -> np.ndarray:
    """Per-row held spread (object array; None = the airport's base spread)."""
    apt, ev = np.asarray(apt), np.asarray(ev, bool)
    out = np.full(len(apt), None, dtype=object)
    for a, tab in spreads.items():
        if a in skip:
            continue
        for tag, sel in (("a3", ev), ("none", ~ev)):
            for i in np.flatnonzero((apt == a) & sel):
                out[i] = tab[tag]
    return out


def _rows_table(rows_path: Path, col: str) -> pd.DataFrame:
    rows = pd.read_parquet(rows_path, columns=["id", "month", "d", "y", col])
    rows["id"] = rows["id"].round().astype("int64")
    lv = common.ensure_columns(rows[["id"]], ["apt", "day"])
    rows["apt"], rows["day"] = lv["apt"].to_numpy(), lv["day"].to_numpy()
    return rows


def nested_spreads(exclude: list[int], rows_path: Path, col: str, cover: str | Path, kind: str = "quantiles") -> dict:
    """Spreads fitted on a pseudo-final rows file without the 2025 months in ``exclude``."""
    rows = _rows_table(rows_path, col)
    rows = rows[~rows["month"].isin(exclude)]
    return fit_spreads(rows["apt"], rows["d"], rows[col], rows["y"], evidence(rows["id"], cover), kind)


def day_shifts(rows: pd.DataFrame, col: str, month: int, params: dict, gate: dict,
               spreads: dict | None = None, cover: str | Path | None = None) -> pd.DataFrame:
    """E[s|K] - s_typical per airport-day of 2025 month ``month`` for the prediction ``rows``[``col``]
    (rows: id, month, ``col``), with the same inputs ``apply`` uses."""
    sub = rows[rows["month"].to_numpy() == month].reset_index(drop=True)
    fr, _fr2, _years, pru, _mlr, forbid = _frame_inputs(sub[["id"]])
    sr = None if spreads is None else spread_rows(fr["apt"], evidence(sub["id"], cover), spreads)
    rec: list = []
    posterior(fr, sub[col].to_numpy(float), pru, params, airports=gate, forbid=forbid, record=rec, spread_rows=sr)
    return pd.DataFrame(rec, columns=["apt", "day", "xhat"])


def _loading_table(rows_path: Path, col: str, months: list[int], oof, fit_col: str, after_level: bool,
                   gate: dict, held: int | None = None, spreads_rows: Path | None = None,
                   cover: str | Path | None = None, kind: str = "quantiles") -> pd.DataFrame:
    """Ordinary-row fit table (apt, d, p, y, xhat, block) for ``months`` of a pseudo-final rows file;
    each month's day shifts use the daily parameters fitted without that month (cached on disk) and,
    with ``spreads_rows``, the held spreads fitted without that month and ``held``."""
    import hashlib
    rows = _rows_table(rows_path, col)
    rsha = common.sha256(rows_path)
    ssha = common.sha256(spreads_rows) if spreads_rows is not None else ""
    parts = []
    for k in months:
        exc = sorted({k} | ({held} if held is not None else set()))
        key = (f"{rsha}|{col}|{k}|{oof}|{fit_col}|{after_level}|{json.dumps(gate, sort_keys=True)}"
               + (f"|spreads {ssha} {exc} {cover}" + ("" if kind == "quantiles" else f" {kind}")
                  if spreads_rows is not None else ""))
        f = common.CACHE / f"daily_xhat_m{k}_{hashlib.sha1(key.encode()).hexdigest()[:10]}.parquet"
        if f.exists():
            xs = pd.read_parquet(f)
        else:
            spr = None if spreads_rows is None else nested_spreads(exc, spreads_rows, col, cover, kind)
            xs = day_shifts(rows, col, k, nested_params(k, oof, fit_col, after_level), gate, spr, cover)
            common.CACHE.mkdir(parents=True, exist_ok=True)
            xs.to_parquet(f, index=False)
        sub = rows[rows["month"].to_numpy() == k]
        parts.append(sub.merge(xs, on=["apt", "day"], how="left", validate="m:1"))
    t = pd.concat(parts, ignore_index=True)
    t["blk"] = t["apt"] + "|" + t["day"]
    return t


def nested_loadings(held_month: int, rows_path: Path, col: str, oof, fit_col: str, after_level: bool,
                    gate: dict, spreads_rows: Path | None = None, cover: str | Path | None = None,
                    kind: str = "quantiles") -> dict:
    """Loadings fitted on a pseudo-final rows file without 2025 month ``held_month``."""
    rows_m = sorted(set(int(x) for x in pd.read_parquet(rows_path, columns=["month"])["month"]))
    t = _loading_table(rows_path, col, [k for k in rows_m if k != held_month], oof, fit_col, after_level, gate,
                       held_month, spreads_rows, cover, kind)
    return fit_loadings(t["apt"], t["d"], t[col], t["y"], t["xhat"], t["blk"])


def build_v2(rows_path: Path, col: str, out_path: Path, params_path: Path = PARAMS, loadings: bool = True,
             spreads: bool = False, cover: str | Path | None = None, kind: str = "quantiles") -> dict:
    """The final-mode L1a-v2 artifact, fitted on every month of a pseudo-final rows file: ``spreads``
    (held spreads by airport x evidence) and/or ``loadings`` (each month's day shifts month-out, the
    daily parameters refitted without that month on the c6 chain, as in the pseudo-final)."""
    gate = json.loads(Path(params_path).read_text()).get("gate", GATE)
    rows_m = sorted(set(int(x) for x in pd.read_parquet(rows_path, columns=["month"])["month"]))
    art: dict = {}
    inputs = {common.rel(rows_path): common.sha256(rows_path), common.rel(params_path): common.sha256(params_path),
              "research/cache/oof_mo12_c6.parquet": common.sha256(common.OOF)}
    what = []
    if spreads:
        art["spreads"] = nested_spreads([], rows_path, col, cover, kind)
        art["spread_kind"] = kind
        art["spreads_skip"] = list(SPREAD_SKIP)
        inputs[common.rel(common.resolve(cover))] = common.sha256(common.resolve(cover))
        what.append("held spreads: quantiles of y - p over held long rows by airport x a3_found")
    if loadings:
        t = _loading_table(rows_path, col, rows_m, None, "prediction", True, gate, None,
                           rows_path if spreads else None, cover, kind)
        art["loadings"] = fit_loadings(t["apt"], t["d"], t[col], t["y"], t["xhat"], t["blk"])
        what.append(f"loadings: ordinary rows with y <= {LOAD_FIT_MAX_Y:.0f} s, day shifts month-out (daily parameters "
                    "refitted without each month on the c6 chain" + (", spreads without each month" if spreads else "") + ")")
    art.update({"fitted_on": f"{common.rel(rows_path)}[{col}], 2025 months {rows_m}; " + "; ".join(what),
                "inputs": inputs,
                "script": f"python3 research/final/levers/daily.py v2 --rows {common.rel(rows_path)} --col {col} "
                          f"--out {common.rel(out_path)}" + ("" if loadings else " --no-loadings")
                          + (f" --spreads {kind} --cover {cover}" if spreads else "")})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(art, indent=1, sort_keys=True))
    return art


def apply(frame: pd.DataFrame, pred: np.ndarray, pi: np.ndarray | None = None, gate: dict | None = None,
          params_path: str | Path = PARAMS, held_month: tuple | None = None, fit_oof: str | None = None,
          fit_col: str = "prediction", after_level: bool = True, use_recovered: bool = True,
          v2: str | Path | None = None, v2_rows: str | Path | None = None, v2_col: str = "post2_recovered",
          v2_loadings: bool = True, v2_spreads: bool | str = False, cover: str | Path | None = None) -> np.ndarray:
    """Posterior with the EUROCONTROL daily file of the frame's year, then rule (a) and L0e again
    (both guards; the assembler's rules + floor 60 follow). ``use_recovered=False`` drops L0e from
    inside the layer (the long-row priors keep the recovered-orphan interval otherwise) - for the
    "minus L0e" ablation only.

    Final mode: parameters from ``params_path`` (artifacts/daily_params.json, all 2025).
    Pseudo-final mode (``held_month=(2025, m)``, passed by research/final/pseudo_final.py when the
    config sets "pass_held_month"): parameters refitted without month m - on the c6 chain by
    default, or on ``fit_oof``[``fit_col``] (e.g. a previous pseudo-final's per-row output, col
    "level") - so no label of the scored month enters. The gate comes from the artifact.

    L1a-v2 (off by default; ``v2_loadings`` / ``v2_spreads`` choose the parts, ``v2_spreads`` True =
    "quantiles" or the kind's name): final mode reads
    ``v2`` = the artifact of ``daily.py v2``; pseudo-final mode refits on ``v2_rows`` = the base
    pseudo-final rows file (its ``v2_col`` = the prediction entering this layer) without month m.
    ``cover``: a3_found per id for the frame's rows (needed by the spreads)."""
    fr = common.ensure_columns(frame, ["apt", "day", "d", "nm", "ml", "year"])
    params_path = common.resolve(params_path)
    art = json.loads(params_path.read_text())
    oof = None
    if held_month is not None:
        assert int(held_month[0]) == 2025 and set(fr["year"].astype(int)) == {2025}, "held_month is for 2025 frames"
        oof = None if fit_oof is None else common.resolve(fit_oof)
        params = nested_params(int(held_month[1]), oof, fit_col, after_level)
    else:
        params = art["params"]
    gate = gate if gate is not None else art.get("gate", GATE)
    load, spr = None, None
    if held_month is not None and (v2 is not None or v2_rows is not None):
        assert v2_rows is not None, "pseudo-final: L1a-v2 must be refitted without the held month (v2_rows)"
        rows_path, m = common.resolve(v2_rows), int(held_month[1])
        kind = v2_spreads if isinstance(v2_spreads, str) else "quantiles"
        if v2_spreads:
            spr = nested_spreads([m], rows_path, v2_col, cover, kind)
        if v2_loadings:
            load = nested_loadings(m, rows_path, v2_col, oof, fit_col, after_level, gate,
                                   rows_path if v2_spreads else None, cover, kind)
    elif v2 is not None:
        a2 = json.loads(common.resolve(v2).read_text())
        load = a2["loadings"] if v2_loadings else None
        spr = a2["spreads"] if v2_spreads else None
    recovered = _recovered_mod()
    fr, fr2, years, pru, mlr, forbid = _frame_inputs(frame)
    sr = None
    if spr is not None:
        assert cover is not None, "L1a-v2 spreads need the frame's cover file (a3_found)"
        ids = (frame["id"] if "id" in frame else frame["MVT_ID_mvt"]).to_numpy()
        sr = spread_rows(fr["apt"], evidence(ids, cover), spr)
    out = posterior(fr, np.asarray(pred, float), pru, params, pi=pi, airports=gate,
                    dither=25.0 if min(years) >= 2026 else 0.0, forbid=forbid if use_recovered else None, load=load,
                    spread_rows=sr)
    out = rules.rule_a(fr, out, 25.0 if min(years) >= 2026 else 0.0)
    if not use_recovered:
        return out
    return recovered.rule(fr2, out, mlr)          # L0e again: a posterior mean can sit inside the interval


if __name__ == "__main__":
    import sys
    import time

    if sys.argv[1:2] == ["extract"]:
        for yr in (2025, 2026):
            src = pd.read_csv(common.PRU / f"all_pre_departure_delays_{yr}.csv")
            dst = common.ART / f"pru_daily_10apt_{yr}.csv"
            src[src.APT_ICAO.isin(APTS)].to_csv(dst, index=False)
            print(dst, common.sha256(dst))
        raise SystemExit
    if sys.argv[1:2] == ["artifact"]:
        import argparse
        ap = argparse.ArgumentParser(prog="daily.py artifact")
        ap.add_argument("cmd")
        ap.add_argument("--oof", help="parquet with id and --col (e.g. a pseudo-final NAME_MODE.parquet)")
        ap.add_argument("--col", default="prediction", help='prediction column of --oof ("level" for a pseudo-final)')
        ap.add_argument("--raw", action="store_true", help="c6 without the level calibration")
        ap.add_argument("--out", help="artifact path (default artifacts/daily_params.json; --raw: daily_params_c6raw.json)")
        ap.add_argument("--gate", help="JSON airport -> full|level (default: all ten full)")
        a = ap.parse_args()
        out_path = Path(a.out) if a.out else (PARAMS.with_name("daily_params_c6raw.json") if a.raw else PARAMS)
        art = build_artifact(json.loads(a.gate) if a.gate else GATE, path=out_path,
                             oof=common.resolve(a.oof) if a.oof else None, after_level=not a.raw, col=a.col)
        print("wrote", out_path, common.sha256(out_path), "gate", art["gate"], "|", art["fitted_on"])
        raise SystemExit
    if sys.argv[1:2] == ["v2"]:
        import argparse
        ap = argparse.ArgumentParser(prog="daily.py v2")
        ap.add_argument("cmd")
        ap.add_argument("--rows", required=True, help="pseudo-final rows parquet (pseudo_final.py run output)")
        ap.add_argument("--col", default="post2_recovered", help="the prediction entering the daily layer")
        ap.add_argument("--out", required=True, help="artifact path (never artifacts/daily_params.json)")
        ap.add_argument("--no-loadings", action="store_true")
        ap.add_argument("--spreads", nargs="?", const="quantiles", choices=["quantiles", "scale"])
        ap.add_argument("--cover", default="research/final/inputs/2025/cover.parquet")
        a = ap.parse_args()
        out_path = common.resolve(a.out)
        assert out_path.resolve() != PARAMS.resolve(), "the L1a-v2 artifact must not overwrite daily_params.json"
        art = build_v2(common.resolve(a.rows), a.col, out_path, loadings=not a.no_loadings, spreads=bool(a.spreads),
                       cover=a.cover, kind=a.spreads or "quantiles")
        print("wrote", common.rel(out_path), common.sha256(out_path))
        if "loadings" in art:
            print(json.dumps({k: art["loadings"][k] for k in ("edges", "del", "nd")}))
        raise SystemExit
    dep = common.load_2025()
    p0 = rules.apply(dep, dep["p"].to_numpy(float))       # the base after the L0 rules
    pru = load_pru(2025)
    which = sys.argv[1] if len(sys.argv) > 1 else "LIRF"
    mode = sys.argv[2] if len(sys.argv) > 2 else "full"
    apts = APTS if which == "all" else which.split(",")
    t0 = time.time()
    q = month_out(dep, p0, pru, {a: mode for a in apts}, verbose=True)
    q = rules.apply(dep, q)
    print(f"elapsed {time.time() - t0:.0f}s")
    r = common.score(f"L1a {which} {mode} (on L0 base)", dep, q, p0)
    np.save(common.CACHE / f"daily_{which.replace(',', '_')}_{mode}.npy", q)
