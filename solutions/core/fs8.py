# %%
# fs8 feature group (feature-gap lane, 2026-10-09): label-free features that competitors reported and fs7
# lacks.  Every column is named g8_* and is computed from fields the ranking file publishes; a departure's
# own off-block (BLOCK_TIME of a DEP row) is never read.  Arrival in-blocks are published and are used.
#
#   plan revision      g8_p_*   AOBT_3 - LOBT, AOBT_3 - IOBT, schedule gaps, route-normalised planned and
#                               NM airborne times (resilient-kiwi v14 -5.23, kind-mango -3.72/-1.22)
#   lateness nowcast   g8_l_*   others' AOBT_3 - EOBT_1 and AOBT_3 - LOBT in the past / next hour
#   neighbour taxi     g8_n_*   others' MVT - AOBT_3 in past / future windows by airport and runway,
#                               its spread, own minus it, and the flights that pushed with us
#   queue              g8_q_*   take-offs between EOBT_1 / LOBT and our take-off, flights that pushed
#                               after us and took off before us (and the reverse), take-off spacing and
#                               queue bursts on our runway
#   arrivals           g8_a_*   arrival taxi-in in the last 15 / next 30 min
#   stand context      g8_s_*   per-frame (label-free) stand / stand-group typical MVT - AOBT_3 and arrival
#                               taxi-in: the fallback for 2026 stands with no 2025 history (EDDF T3,
#                               EDDM T1 pier); traffic at the stand as a rate per month of the frame
#
# derive_main() turns a built store (fs7) into an fs8 store: the base store's files are copied unchanged
# and tr_base / rk_base get the g8_* columns appended (pyarrow append_column), so every fs7 column keeps
# its bytes and a trainer that drops g8_* sees exactly the fs7 matrix.

import re
import shutil

G8_PREFIX = "g8_"
BASE_STORE = ""  # replaced by build_kernel.py derive: the base store's kernel slug


def _sums(q_key, q_lo, q_hi, r_key, r_t, r_v):
    """Sum, sum of squares and count of r values with q_lo <= r_t < q_hi, same key."""
    n_q = len(q_key)
    s, s2, n = np.zeros(n_q), np.zeros(n_q), np.zeros(n_q)
    ok_r = ~(np.isnan(r_t) | np.isnan(r_v))
    r_key, r_t, r_v = np.asarray(r_key)[ok_r], r_t[ok_r], r_v[ok_r]
    for qi, ri in _by_key(q_key, r_key):
        order = np.argsort(r_t[ri], kind="stable")
        ts, vs = r_t[ri][order], r_v[ri][order]
        c1 = np.concatenate([[0.0], np.cumsum(vs)])
        c2 = np.concatenate([[0.0], np.cumsum(vs * vs)])
        lo, hi = q_lo[qi], q_hi[qi]
        good = ~(np.isnan(lo) | np.isnan(hi))
        a = np.searchsorted(ts, np.where(good, lo, 0), "left")
        b = np.searchsorted(ts, np.where(good, hi, 0), "left")
        b = np.maximum(a, b)
        s[qi], s2[qi], n[qi] = c1[b] - c1[a], c2[b] - c2[a], np.where(good, b - a, 0)
    return s, s2, n


def _mean_std(s, s2, n, own=None, min_n=2):
    """Mean and std from sums, taking the query's own value out when given (own NaN = not inside)."""
    if own is not None:
        has = ~np.isnan(own)
        s = np.where(has, s - np.nan_to_num(own), s)
        s2 = np.where(has, s2 - np.nan_to_num(own) ** 2, s2)
        n = np.where(has, n - 1, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n >= min_n, s / np.maximum(n, 1), np.nan)
        v = np.where(n >= min_n, s2 / np.maximum(n, 1) - m * m, np.nan)
    return m, np.sqrt(np.clip(v, 0, None)), n


def _pair_less(q_key, q_lo, q_hi, q_ref, r_key, r_t, r_v, chunk_pairs=40_000_000):
    """For each query: (count of r with q_lo <= r_t < q_hi and r_v < q_ref, count in the window).
    Pairs are enumerated per key in chunks (windows here are at most a few hours long)."""
    n_q = len(q_key)
    less, tot = np.full(n_q, np.nan), np.full(n_q, np.nan)
    ok_r = ~(np.isnan(r_t) | np.isnan(r_v))
    r_key, r_t, r_v = np.asarray(r_key)[ok_r], r_t[ok_r], r_v[ok_r]
    for qi, ri in _by_key(q_key, r_key):
        order = np.argsort(r_t[ri], kind="stable")
        ts, vs = r_t[ri][order], r_v[ri][order]
        lo, hi, ref = q_lo[qi], q_hi[qi], q_ref[qi]
        good = ~(np.isnan(lo) | np.isnan(hi) | np.isnan(ref))
        gi = np.flatnonzero(good)
        a = np.searchsorted(ts, lo[gi], "left")
        b = np.maximum(np.searchsorted(ts, hi[gi], "left"), a)
        L = b - a
        cl = np.zeros(len(gi))
        start = 0
        while start < len(gi):
            cum = np.cumsum(L[start:])
            stop = start + max(int(np.searchsorted(cum, chunk_pairs, "right")), 1)
            Lc, ac = L[start:stop], a[start:stop]
            tot_c = int(Lc.sum())
            if tot_c:
                owner = np.repeat(np.arange(len(Lc)), Lc)
                offs = np.arange(tot_c) - np.repeat(np.cumsum(Lc) - Lc, Lc)
                idx = np.repeat(ac, Lc) + offs
                hit = vs[idx] < np.repeat(ref[gi[start:stop]], Lc)
                cl[start:stop] = np.bincount(owner, weights=hit, minlength=len(Lc))
            start = stop
        res_l, res_t = np.full(len(qi), np.nan), np.full(len(qi), np.nan)
        res_l[gi], res_t[gi] = cl, L
        less[qi], tot[qi] = res_l, res_t
    return less, tot


def _runs(key, t, gap):
    """For events sorted by (key, t), key as integer codes: how many consecutive gaps <= gap end at each
    event (backward run length) and start at it (forward run length)."""
    n = len(t)
    idx = np.arange(n)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    brk = np.r_[True, (key[1:] != key[:-1]) | (np.diff(t) > gap)]       # a run starts here
    brk_f = np.r_[(key[1:] != key[:-1]) | (np.diff(t) > gap), True]     # a run ends here
    start = np.maximum.accumulate(np.where(brk, idx, 0))
    end = np.minimum.accumulate(np.where(brk_f, idx, n)[::-1])[::-1]
    return (idx - start).astype(float), (end - idx).astype(float)


def _stand_group(stand: pd.Series) -> np.ndarray:
    """A coarse stand group: the leading letters of the stand name, else its leading digits less the last
    two (EDDF 'V155' -> 'V', EGLL '501' -> '5')."""
    s = stand.fillna("").astype(str).str.strip().str.upper()
    lead = s.str.extract(r"^([A-Z]+)", expand=False)
    digits = s.str.extract(r"^(\d+)", expand=False)
    num = np.where(digits.fillna("").str.len() > 2, digits.fillna("").str[:-2], digits.fillna(""))
    return np.where(lead.notna(), lead.fillna(""), np.where(s == "", "-", pd.Series(num).radd("#").to_numpy()))


def _loo_mean(key, v, min_n=5):
    """Mean of v over the frame's rows with the same key, the row's own value left out."""
    fr = pd.DataFrame({"k": key, "v": v})
    s = fr.groupby("k")["v"].transform("sum").to_numpy() - np.nan_to_num(v)
    n = fr.groupby("k")["v"].transform("count").to_numpy() - (~np.isnan(v))
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan), n.astype(float)


def fs8_features(df: pd.DataFrame) -> pd.DataFrame:
    """The g8_* columns for every departure of a raw movement frame (training or ranking file)."""
    df = df.reset_index(drop=True)
    dep = df["PHASE_mvt"].eq("DEP").to_numpy()
    arr = df["PHASE_mvt"].eq("ARR").to_numpy()
    apt = np.where(dep, df["ADEP_mvt"].astype(str), df["ADES_mvt"].astype(str))
    rwy = (pd.Series(apt) + "|" + df["RUNWAY_mvt"].astype(str)).to_numpy()
    t = secs(df["MVT_TIME_UTC_mvt"])
    blk = secs(df["BLOCK_TIME_UTC_mvt"])
    blk[dep] = np.nan  # a departure's off-block is the answer: never read
    tin = np.where(arr, np.clip(blk - t, 0, TE_CLIP), np.nan)

    d = df.loc[dep].reset_index(drop=True)
    qa, qr, qt = apt[dep], rwy[dep], t[dep]
    sched, eobt1, lobt, iobt, aobt3 = (secs(d[c]) for c in ("SCHED_TIME_UTC_mvt", "EOBT_1_flt", "LOBT_flt",
                                                             "IOBT_flt", "AOBT_3_flt"))
    arvt1, arvt3 = secs(d["ARVT_1_flt"]), secs(d["ARVT_3_flt"])
    F = pd.DataFrame({ID: d[ID].to_numpy()})
    n = len(d)

    # ---- plan revision (per flight)
    F["g8_p_aobt3_m_lobt"] = aobt3 - lobt
    F["g8_p_aobt3_m_iobt"] = aobt3 - iobt
    F["g8_p_eobt1_m_sched"] = eobt1 - sched
    F["g8_p_lobt_m_sched"] = lobt - sched
    F["g8_p_iobt_m_sched"] = iobt - sched
    route = (pd.Series(qa) + "|" + d["ADES_mvt"].astype(str)).to_numpy()
    route_t = (pd.Series(route) + "|" + d["AIRCRAFT_TYPE_mvt"].astype(str)).to_numpy()
    air1 = arvt1 - eobt1
    air3 = arvt3 - qt
    air1 = np.where((air1 > 0) & (air1 < 86400), air1, np.nan)
    air3 = np.where((air3 > 0) & (air3 < 86400), air3, np.nan)

    def route_median(v):
        m_t = pd.Series(v).groupby(route_t).transform("median").to_numpy()
        c_t = pd.Series(v).groupby(route_t).transform("count").to_numpy()
        m_r = pd.Series(v).groupby(route).transform("median").to_numpy()
        return np.where(c_t >= 5, m_t, m_r)

    med1, med3 = route_median(air1), route_median(air3)
    F["g8_p_air1_res"] = air1 - med1
    F["g8_p_air3_res"] = air3 - med3
    F["g8_p_plan_taxi"] = air1 - med3  # planned block-to-arrival less the route's typical airborne time

    # ---- lateness nowcasts (others' AOBT_3 - EOBT_1 / - LOBT, by their take-off time)
    late_e1 = np.clip(aobt3 - eobt1, -7200, 7200)
    late_lo = np.clip(aobt3 - lobt, -7200, 7200)
    for nm, v in (("e1", late_e1), ("lobt", late_lo)):
        for lab, lo, hi in (("p60", qt - 3600, qt), ("f60", qt + 0.5, qt + 3600.5)):
            s, s2, c = _sums(qa, lo, hi, qa, qt, v)
            F[f"g8_l_{nm}_{lab}"] = _mean_std(s, s2, c)[0]

    # ---- neighbours' MVT - AOBT_3 (label-free proxy of taxi-out), past / future, airport / runway
    d3 = np.clip(qt - aobt3, 0, 5400)
    for kn, key in (("apt", qa), ("rwy", qr)):
        for lab, lo, hi in (("p15", qt - 900, qt), ("p60", qt - 3600, qt), ("f15", qt + 0.5, qt + 900.5),
                            ("f60", qt + 0.5, qt + 3600.5)):
            s, s2, c = _sums(key, lo, hi, key, qt, d3)
            m, sd, c = _mean_std(s, s2, c)
            F[f"g8_n_d3_{kn}_{lab}"] = m
            if lab in ("p15", "f15"):
                F[f"g8_n_cnt_{kn}_{lab}"] = c
        s, s2, c = _sums(key, qt - 1800, qt + 1800.5, key, qt, d3)
        m, sd, c = _mean_std(s, s2, c, own=d3, min_n=3)
        F[f"g8_n_d3sd_{kn}_pm30"] = sd
        F[f"g8_n_own_m_{kn}_pm30"] = d3 - m
        # flights that pushed (AOBT_3) within 10 min of our AOBT_3: their MVT - AOBT_3
        s, s2, c = _sums(key, aobt3 - 600, aobt3 + 600.5, key, aobt3, d3)
        m, sd, c = _mean_std(s, s2, c, own=np.where(np.isnan(aobt3), np.nan, d3))
        F[f"g8_n_push_d3_{kn}"] = m
        F[f"g8_n_push_cnt_{kn}"] = c

    # ---- queue
    valid_t = ~np.isnan(qt)
    for nm, anchor in (("e1", eobt1), ("lobt", lobt)):
        lo = np.where(anchor < qt, np.maximum(anchor, qt - 4 * 3600), np.nan)
        F[f"g8_q_{nm}_apt"] = count_between(qa, lo, qt, qa[valid_t], qt[valid_t])
        F[f"g8_q_{nm}_rwy"] = count_between(qr, lo, qt, qr[valid_t], qt[valid_t])
    # take-offs between our AOBT_3 (EOBT_1) and ours, split by whether they pushed (were planned) before us
    for nm, anc in (("a3", aobt3), ("e1", eobt1)):
        lo = np.where(anc < qt, np.maximum(anc, qt - 7200), np.nan)
        for kn, key in (("apt", qa), ("rwy", qr)):
            less, tot = _pair_less(key, lo, qt, anc, key, qt, anc)
            F[f"g8_q_ahead_{nm}_{kn}"] = less          # pushed / planned before us, took off before us
            F[f"g8_q_ovt_{nm}_{kn}"] = tot - less     # pushed / planned at or after us, took off before us
    for kn, key in (("apt", qa), ("rwy", qr)):
        # pushed before us, took off within the hour after us (we overtook them)
        lo = np.where(np.isnan(aobt3), np.nan, qt + 0.5)
        less, _ = _pair_less(key, lo, qt + 3600.5, aobt3, key, qt, aobt3)
        F[f"g8_q_behind_a3_{kn}"] = less
    # take-off spacing and queue bursts on our runway (departures only; then every movement)
    kq = pd.factorize(qr, sort=True)[0]
    order = np.lexsort((qt, kq))
    k_s, t_s = kq[order], qt[order]
    same_prev = np.r_[False, k_s[1:] == k_s[:-1]]
    same_next = np.r_[k_s[1:] == k_s[:-1], False]
    gp = np.where(same_prev, t_s - np.r_[np.nan, t_s[:-1]], np.nan)
    gn = np.where(same_next, np.r_[t_s[1:], np.nan] - t_s, np.nan)
    back, fwd = _runs(k_s, t_s, 120.0)
    for nm, v in (("g8_q_gap_prev_dep", gp), ("g8_q_gap_next_dep", gn), ("g8_q_burst_prev", back),
                  ("g8_q_burst_next", fwd)):
        out = np.empty(n)
        out[order] = v
        F[nm] = out
    mv = dep | arr
    all_k, all_t = rwy[mv], t[mv]
    o2 = np.lexsort((all_t, pd.factorize(all_k, sort=True)[0]))
    F["g8_q_gap_prev_any"] = _prev_gap(all_k[o2], all_t[o2], qr, qt)  # any movement on our runway before us

    # ---- arrivals' taxi-in, last 15 min and next 30 min (by landing time)
    ak, at_ = apt[arr], t[arr]
    for lab, lo, hi in (("p15", qt - 900, qt), ("f30", qt + 0.5, qt + 1800.5)):
        s, s2, c = _sums(qa, lo, hi, ak, at_, tin[arr])
        F[f"g8_a_tin_{lab}"] = _mean_std(s, s2, c)[0]

    # ---- stand context (per frame, label-free)
    stand = (pd.Series(qa) + "|" + d["STAND_mvt"].astype(str)).to_numpy()
    grp = (pd.Series(qa) + "|" + pd.Series(_stand_group(d["STAND_mvt"]))).to_numpy()
    month = pd.to_datetime(qt, unit="s", utc=True).month.to_numpy().astype(str)
    has_stand = d["STAND_mvt"].notna().to_numpy()
    d3v = np.where((qt - aobt3 > 0) & (qt - aobt3 < 5400), qt - aobt3, np.nan)
    # counts are per month of the frame at the airport: the training frame spans 12 months of 2025 and the
    # final frame 4 of 2026, so a raw count would differ threefold between training and prediction (v1 of
    # this store kept raw counts, g8_s_n_*; v2 replaces them with these rates, g8_s_pm_*)
    n_mon = pd.Series(month).groupby(qa).transform("nunique").to_numpy(dtype=float)
    for nm, key in (("sr", pd.Series(stand) + "|" + d["RUNWAY_mvt"].astype(str)), ("s", pd.Series(stand)),
                    ("gr", pd.Series(grp) + "|" + d["RUNWAY_mvt"].astype(str)),
                    ("srm", pd.Series(stand) + "|" + d["RUNWAY_mvt"].astype(str) + "|" + pd.Series(month))):
        m, c = _loo_mean(key.to_numpy(), d3v)
        ok = has_stand
        F[f"g8_s_d3_{nm}"] = np.where(ok, m, np.nan)
        F[f"g8_s_pm_{nm}"] = np.where(ok, c if nm == "srm" else c / n_mon, np.nan)
    a_stand = (pd.Series(apt[arr]) + "|" + df.loc[arr, "STAND_mvt"].astype(str).to_numpy()).to_numpy()
    a_grp = (pd.Series(apt[arr]) + "|" + pd.Series(_stand_group(df.loc[arr, "STAND_mvt"].reset_index(drop=True)))).to_numpy()
    a_has = df.loc[arr, "STAND_mvt"].notna().to_numpy()
    for nm, ak2, qk in (("s", a_stand, stand), ("g", a_grp, grp)):
        tv = pd.Series(tin[arr][a_has]).groupby(ak2[a_has])
        med, cnt = tv.median(), tv.count()
        F[f"g8_s_tin_{nm}"] = np.where(has_stand, pd.Series(qk).map(med[cnt >= 5]).to_numpy(dtype=float), np.nan)
    F["g8_s_tin_pm_s"] = np.where(has_stand, pd.Series(stand).map(
        pd.Series(tin[arr][a_has]).groupby(a_stand[a_has]).count()).fillna(0).to_numpy(dtype=float) / n_mon, np.nan)

    for c in F.columns:
        if c != ID:
            F[c] = F[c].astype(np.float32)
    return F


def _prev_gap(keys_sorted, t_sorted, q_key, q_t):
    """Seconds from the latest event on the same key strictly before q_t (NaN if none)."""
    out = np.full(len(q_key), np.nan)
    for qi, ri in _by_key(q_key, keys_sorted):
        ts = t_sorted[ri]  # already sorted within key
        j = np.searchsorted(ts, q_t[qi], "left") - 1
        ok = j >= 0
        res = np.full(len(qi), np.nan)
        res[ok] = q_t[qi][ok] - ts[j[ok]]
        out[qi] = res
    return out


def _find_store(slug: str) -> Path:
    hits = [h.parent for h in sorted(Path("/kaggle/input").rglob("fs_meta.json")) if slug in str(h)]
    if not hits:
        raise FileNotFoundError(f"base store {slug} not mounted")
    return hits[0]


def _append(src: Path, dst: Path, ext: pd.DataFrame) -> dict:
    """dst = src parquet with the ext columns appended in src's row order (joined on ID)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    tab = pq.read_table(src)
    ids = tab.column(ID).to_numpy().astype("int64")
    e = ext.copy()
    e[ID] = e[ID].astype("int64")
    assert e[ID].is_unique, "fs8: duplicate ids"
    pos = pd.Index(e[ID]).get_indexer(ids)
    miss = int((pos < 0).sum())
    assert miss == 0, f"fs8: {miss} store rows have no fs8 row"
    for c in [c for c in e.columns if c != ID]:
        tab = tab.append_column(c, pa.array(e[c].to_numpy()[pos], type=pa.float32()))
    pq.write_table(tab, dst)
    return {"rows": tab.num_rows, "columns": tab.num_columns}


def derive_main():
    data_dir = resolve_data_dir()
    base = _find_store(BASE_STORE)
    print(f"data {data_dir}  base store {base}", flush=True)
    raw = pd.concat((pd.read_parquet(p) for p in sorted(data_dir.glob("training_*.parquet"))), ignore_index=True)
    rk_path = resolve_rk_file(data_dir)
    t0 = time.time()
    f_tr = fs8_features(raw)
    del raw
    print(f"fs8 train: {f_tr.shape} in {time.time() - t0:.0f} s  used_h={used_h():.3f}", flush=True)
    f_rk = fs8_features(pd.read_parquet(rk_path))
    print(f"fs8 predict frame {rk_path.name}: {f_rk.shape}  used_h={used_h():.3f}", flush=True)
    info = {}
    for name, F in (("tr_base.parquet", f_tr), ("rk_base.parquet", f_rk)):
        info[name] = _append(base / name, OUTPUT_DIR / name, F)
        print(f"{name}: {info[name]}", flush=True)
    for p in sorted(base.glob("enc_*.parquet")):
        shutil.copyfile(p, OUTPUT_DIR / p.name)
    meta = json.loads((base / "fs_meta.json").read_text())
    g8 = [c for c in f_tr.columns if c != ID]
    meta.update({"derived_from": BASE_STORE, "fs8_columns": g8, "columns": meta["columns"] + g8,
                 "fs8_rows": info, "base_sha256": {p.name: file_sha256(p) for p in sorted(base.iterdir())
                                                   if p.suffix in (".parquet", ".json")}})
    (OUTPUT_DIR / "fs_meta.json").write_text(json.dumps(meta, indent=1))
    desc = f_tr[g8].describe(percentiles=[0.5]).T[["count", "mean", "50%"]]
    desc["rk_count"] = f_rk[g8].count().to_numpy()
    print(desc.to_string(), flush=True)
    print(f"FS8_DONE used_h={used_h():.3f}", flush=True)


def file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    derive_main()
