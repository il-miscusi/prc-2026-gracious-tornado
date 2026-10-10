# %%
# Feature store: build every departure's features once and write them with the encodings
# for each fold, so trainer kernels load matrices instead of rebuilding them.
#
# Folds: A = Jan + Jul 2025 days 16-31 (the v14-v27 hold-out), B = Jan + Jul days 1-15.
# Together they give out-of-fold predictions for all of Jan + Jul 2025.  "full" fits the
# encodings on every 2025 row for the ranking refit.  Encodings are leave-one-month-out
# for training rows, so each fold's file covers every training row.
#
# Outputs (float32 features, raw strings for categoricals):
#   tr_base.parquet  rk_base.parquet  enc_A.parquet  enc_B.parquet  enc_full.parquet
#   enc_full_rk.parquet  fs_meta.json


def fold_masks(month: np.ndarray, mday: np.ndarray) -> dict:
    jj = np.isin(month, VAL_MONTHS)
    # F: a forward winter fold (Nov + Dec 2025), fitted on Jan-Oct only (see fs_main)
    return {"A": jj & (mday >= 16), "B": jj & (mday <= 15), "F": np.isin(month, (11, 12))}


def _f32(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in out.columns:
        if out[c].dtype == np.float64 and c not in ("y", ID):  # ids exceed float32's exact range
            out[c] = out[c].astype(np.float32)
    return out


def fs_main():
    global METAR, ADSB
    if globals().get("INPUT_URLS"):
        fetch_inputs()
    data_dir = resolve_data_dir()
    print(f"data: {data_dir}", flush=True)
    METAR = load_metar()
    ADSB = load_adsb()
    globals()["METAR"], globals()["ADSB"] = METAR, ADSB
    globals()["GEOM"] = load_geometry()
    globals()["ADSB3"] = load_adsb3()
    if FS_OPTS.get("park") == "stand":
        globals()["DWELLS3"] = load_dwells3()
    raw = pd.concat((pd.read_parquet(p) for p in sorted(data_dir.glob("training_*.parquet"))), ignore_index=True)
    rk_path = resolve_rk_file(data_dir)
    ranking = pd.read_parquet(rk_path)
    print(f"predict frame: {rk_path} ({len(ranking):,} movements)", flush=True)
    if feature_opts():
        print(f"fs7 options: {FS_OPTS}", flush=True)
        mp = build_iata_map(pd.concat([raw["FLIGHT_mvt"], ranking["FLIGHT_mvt"]]),
                            pd.concat([raw["CALLSIGN_flt"], ranking["CALLSIGN_flt"]]))
        if FS_OPTS.get("openflights"):
            of = openflights_map()
            added = {k: v for k, v in of.items() if k not in mp}
            print(f"iata map: {len(mp)} from data, +{len(added)} from OpenFlights", flush=True)
            mp.update(added)
        IATA_MAP.clear()
        IATA_MAP.update(mp)
        print(f"iata map: {len(IATA_MAP)} designators", flush=True)
    tr = build(raw)
    del raw
    rk = build(ranking)
    if FS_OPTS.get("ext"):
        tr, rk = merge_external(tr, rk, FS_OPTS["ext"])
    if feature_opts():
        for name, F in (("tr", tr), ("rk", rk)):
            cols = [c for c in ("a3_match_conf", "adsb_match_conf") if c in F]
            for c in cols:
                print(f"{name} {c}: " + F.groupby("apt")[c].value_counts().unstack(fill_value=0).to_string(), flush=True)
            if "inb_found" in F:
                print(f"{name} inbound found by airport: {F.groupby('apt')['inb_found'].mean().round(3).to_dict()}", flush=True)
    tr = tr[tr["y"].notna()].reset_index(drop=True)
    print(f"built: train deps {len(tr):,}  ranking deps {len(rk):,}  used_h={used_h():.3f}", flush=True)
    add_nb_excess([tr, rk])
    print(f"nb excess: rk nbx_apt_1800 p50/p99 {np.nanpercentile(rk['nbx_apt_1800'], [50, 99])}  used_h={used_h():.3f}", flush=True)

    if FS_OPTS.get("pseudo_te"):
        pseudo_labels(tr, [tr, rk])
        g = globals()
        g["PS_AP"] = rk["ps_y"].to_numpy(dtype=float)
        g["PS_AP_MONTH"] = rk["month"].to_numpy()
        g["PS_TR"] = np.full(len(tr), np.nan)  # store training encodings stay 2025-only
    month = tr["month"].to_numpy()
    y = tr["y"].to_numpy()
    ds = tr["d_sched"].to_numpy()
    long = ds > LONG_DS
    path = long & (np.abs(y - ds) <= PATH_TOL)
    r = np.where(long, np.clip(y / np.where(long, ds, 1.0), 0, R_CLIP), np.nan)
    dd = tr["dd_utc"].to_numpy()
    artifact = (dd >= 1) & ((y - 86400 * dd) >= -900) & ((y - 86400 * dd) <= 3600)
    tr["artifact"] = artifact
    key_names = sorted(set(TE_KEYS + LONG_KEYS + ["apt"]))
    keys_tr = {k: tr[k].astype(str).to_numpy() for k in key_names}
    keys_rk = {k: rk[k].astype(str).to_numpy() for k in key_names}

    _f32(tr).to_parquet(OUTPUT_DIR / "tr_base.parquet", index=False)
    _f32(rk).to_parquet(OUTPUT_DIR / "rk_base.parquet", index=False)
    folds = fold_masks(month, tr["mday"].to_numpy())
    for name, val in folds.items():
        fit = ~val & ~artifact
        if name == "F":
            fit &= month <= 10  # forward: nothing after the validated months
        enc_tr, _ = all_encodings(keys_tr, month, y, ds, r, long, path, fit, keys_rk)
        _f32(enc_tr).to_parquet(OUTPUT_DIR / f"enc_{name}.parquet", index=False)
        print(f"enc_{name}: {enc_tr.shape}  val rows {int(val.sum()):,}  used_h={used_h():.3f}", flush=True)
    enc_all, enc_rk = all_encodings(keys_tr, month, y, ds, r, long, path, ~artifact, keys_rk)
    _f32(enc_all).to_parquet(OUTPUT_DIR / "enc_full.parquet", index=False)
    _f32(enc_rk).to_parquet(OUTPUT_DIR / "enc_full_rk.parquet", index=False)
    meta = {"fs_opts": FS_OPTS, "iata_map_n": len(IATA_MAP), "n_tr": len(tr), "n_rk": len(rk), "folds": {k: int(v.sum()) for k, v in folds.items()},
            "artifact": int(artifact.sum()), "columns": list(tr.columns), "enc_columns": list(enc_all.columns),
            "used_h": used_h()}
    (OUTPUT_DIR / "fs_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"FS_DONE used_h={used_h():.3f}", flush=True)


def merge_external(tr: pd.DataFrame, rk: pd.DataFrame, pattern: str):
    """Left-join an external per-movement table (numeric columns, keyed by MVT_ID_mvt) onto both frames."""
    roots = ([Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else []) + [Path("/kaggle/input"), Path.cwd()]
    hits = [h for r in roots if r.exists() for h in sorted(r.rglob(pattern))]
    if not hits:
        raise FileNotFoundError(f"ext: no file matches {pattern}")
    ext = pd.concat([pd.read_parquet(h) for h in hits], ignore_index=True).drop_duplicates()
    ext[ID] = ext[ID].astype("int64")
    assert ext[ID].is_unique, "ext: MVT_ID_mvt must be unique"
    keep = [c for c in ext.columns if c != ID and pd.api.types.is_numeric_dtype(ext[c])]
    ext = ext[[ID] + keep]
    out = []
    for F in (tr, rk):
        ids = F[ID].astype("int64")
        G = pd.DataFrame({ID: ids}).merge(ext, on=ID, how="left", validate="one_to_one")
        for c in keep:
            F[c] = G[c].to_numpy(dtype=float)
        out.append(F)
        print(f"ext {pattern}: {len(keep)} columns, coverage {G[keep[0]].notna().mean():.3f}" if keep else "ext: none", flush=True)
    return out


if __name__ == "__main__":
    fs_main()
