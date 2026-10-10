# %%
# Trainer: load a feature store, fit the short/long pair per fold and on all of 2025.
#
# Short rows (takeoff <= 1 h after schedule): regression on min(y, 2 h).
# Long rows: regression on r = y / d (clipped to [0, 1.5]) weighted by min(d, 12 h)^2,
# so the loss is squared error in seconds; y = r * d.  SPEC["long_target"] = "sec" regresses
# y itself (capped at 1.5 d) with unit weights instead.
#
# SPEC (filled by build_kernel.py) picks the library and parameters.  Folds A and B give
# out-of-fold predictions for all of Jan + Jul 2025; "full" refits on every 2025 row with
# 1.1 x the folds' mean best iteration and predicts the ranking file.
#
# Outputs: oof.csv (id, fold, prediction), submission.parquet, results.json.
#
# Validation options (WS-E, 2026-10-01; every one is off unless set, so old SPECs run unchanged):
#   folds: [..., "M1".."M12", "MJ", "MY"]  whole-month outer folds: train on 2025 minus that
#        month, score all of it (MJ = M1, MY = M7).  Their encodings are not in the store; they
#        are computed here with the store's own recipe (all_encodings, fit = rest of 2025).
#   enc_check: true   recompute enc_A in-kernel and print its max difference from the store's.
#   fold_rounds: [s, l] | {fold: [s, l] or {group: [s, l]}}   fixed rounds, no early stopping
#        on the scored rows (review B defect 6).
#   es_inner: {"days": [8..14], "refit": false, "mult": 1.1}   early-stop on an inner time
#        block of the TRAINING side (those days of every month); with refit, refit on all the
#        training side at mult x the inner best.
#   long_w: "cap" (min(d, w_cap_h)^2, default) | "d2" (exact seconds SSE) | "none";
#   w_cap_h: hours (12);  r_clip: ratio clip of labels and predictions (1.5).
#   pred_cap_d: T   predictions capped at d_sched where d_sched > T (review B defect 4).
#   only: "long" | "short"   fit one half; the other half's predictions are NaN (fill from a
#        base run with research/eval_full.py --fill).  Use with refit: false.
#   eval_mask: {"mode": "cov26"}   coverage stress: in Jan/Jul rows, mask ADS-B on a random share
#        max(0, 1 - cov2026/cov2025) of the covered rows of each airport x month, re-predict, and
#        write oof_cov26.csv (all fold rows, masked ones replaced).
#   save_models: true   each refit seed's fitted pair is pickled to output/models/refit_<k>.pkl with
#        what prediction needs (spec, lib, params, categorical columns, column order, anchor table).
#   predict_only: true   no training: load saved models (SPEC["models"]: directories or globs, default
#        every models/refit_*.pkl under /kaggle/input; "models_kernels": [slug, ...] to filter;
#        "model_urls": {name: url} to fetch them) and predict this store's ranking rows (a new month);
#        writes refit_<k>.npy and submission.parquet exactly like a refit run.
#   drop_adsb: true   drop every ADS-B column (prefixes adsb_ a3_ nbx_ nbn_ ratio_adsb_): the no-ADS-B
#        member of the LIRF uncovered overlay (WS-J, research/cover/NOTES.txt; was a kernel-local patch).
#   max_parallel: k   run fold (and refit) jobs in waves of k processes, GPU = slot % gpus
#        (one job per GPU at a time: two CatBoost GPU jobs on one card exhaust its memory).

import importlib
import hashlib
import subprocess

SPEC = {}  # replaced by build_kernel.py
MAX_ROUNDS = int(SPEC.get("max_rounds", MAX_ROUNDS))


_PACKAGE_OWNER_PID = os.getpid()  # only the original host process may change its environment


def ensure(pkg: str, wheel_glob: str | None = None):
    from importlib.metadata import version, PackageNotFoundError
    pinned = SPEC.get("package_versions", {}).get(pkg)
    if pinned:
        try:
            installed = version(pkg)
        except PackageNotFoundError:
            installed = None
        if installed != pinned:
            if os.getpid() != _PACKAGE_OWNER_PID:
                raise RuntimeError(f"worker requires {pkg}=={pinned}; parent prepared {installed}")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"{pkg}=={pinned}"], check=True)
    try:
        return importlib.import_module(pkg)
    except ImportError:
        if os.getpid() != _PACKAGE_OWNER_PID:
            raise RuntimeError(f"worker cannot import {pkg}; prepare packages before forking")
        import glob as _glob
        whl = sorted(_glob.glob(f"/kaggle/input/**/{wheel_glob or pkg}-*.whl", recursive=True))
        if whl:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", whl[-1]], check=True)
        else:  # internet on (TPU hosts ship without these)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
        return importlib.import_module(pkg)


def _prepare_variant_packages(base: dict) -> None:
    """Install the union once in the parent; reject incompatible pins on one shared host."""
    packages = {}
    for override in base["variants"].values():
        spec = {**base, **override}
        pkg = {"cat": "catboost", "lgb": "lightgbm", "xgb": "xgboost", "mlp": "torch"}.get(spec["lib"], spec["lib"])
        pins = dict(spec.get("package_versions", {}))
        pins.setdefault(pkg, None)
        for name, pin in pins.items():
            previous = packages.get(name)
            if previous and pin and previous != pin:
                raise ValueError(f"conflicting package pins: {name} {previous} / {pin}")
            packages[name] = pin or previous
    global SPEC
    original = SPEC
    try:
        SPEC = {**base, "package_versions": {name: pin for name, pin in packages.items() if pin}}
        for pkg in sorted(packages):
            ensure(pkg)
    finally:
        SPEC = original


def mem_gb() -> str:
    """Host memory (GB) available / total, for sizing parallel jobs; '' off Linux."""
    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        kb = lambda k: int(info[k].split()[0]) / 1e6  # noqa: E731
        return f"mem {kb('MemAvailable'):.1f}/{kb('MemTotal'):.1f} GB"
    except Exception:  # noqa: BLE001
        return ""


def ncpu() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


def normalized_ids(values) -> np.ndarray:
    """Recover exact integral legacy float IDs, never round or accept a lossy range."""
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iuf":
        raise ValueError("identifiers must be a one-dimensional numeric array")
    if not np.isfinite(values).all():
        raise ValueError("identifiers must be finite")
    if values.dtype.kind == "f":
        if np.any(np.abs(values) >= 2**53) or np.any(values != np.floor(values)):
            raise ValueError("floating identifiers must be integral within the exact integer range")
    elif values.size and (int(values.min()) < -(2**63) or int(values.max()) > 2**63 - 1):
        raise ValueError("identifiers exceed int64 range")
    result = values.astype(np.int64)
    if not np.array_equal(result.astype(values.dtype), values) or len(np.unique(result)) != len(result):
        raise ValueError("identifiers must round-trip exactly and be unique")
    return result


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _validate_download(path: Path, suffix: str, expected: dict) -> dict:
    size = path.stat().st_size
    if not size or (expected.get("size") is not None and size != int(expected["size"])):
        raise ValueError(f"download size mismatch: {path.name}")
    digest = file_sha256(path)
    if expected.get("sha256") and digest != expected["sha256"]:
        raise ValueError(f"download hash mismatch: {path.name}")
    if suffix == ".parquet":
        import pyarrow.parquet as pq
        pq.ParquetFile(path).schema_arrow  # verify the footer before publishing
    elif suffix == ".json":
        json.loads(path.read_text())
    return {"size": size, "sha256": digest}


def fetch_urls(urls: dict, dest: Path) -> Path:
    """Accept URL strings or {url, size, sha256}; publish only verified complete files."""
    import urllib.request
    dest.mkdir(parents=True, exist_ok=True)
    for name, item in urls.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"unsafe download path: {name}")
        expected = item if isinstance(item, dict) else {"url": item}
        url = expected["url"]
        out = dest / name
        out.parent.mkdir(parents=True, exist_ok=True)
        receipt = out.with_name(out.name + ".download.json")
        if out.exists() and receipt.exists():
            try:
                old = json.loads(receipt.read_text())
                _validate_download(out, out.suffix, {**old, **expected})
                continue
            except (ValueError, OSError):
                pass  # an unverified/changed cache is fetched again
        part = out.with_name(out.name + f".{os.getpid()}.part")
        last_error = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=600) as resp, open(part, "wb") as fh:
                    length = resp.headers.get("Content-Length")
                    while True:
                        chunk = resp.read(8 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                observed = _validate_download(part, out.suffix,
                                              {"size": int(length) if length else None, **expected})
                os.replace(part, out)
                receipt.write_text(json.dumps(observed))
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                part.unlink(missing_ok=True)
                if getattr(exc, "code", None) == 429:
                    raise  # do not shorten the provider's Retry-After window
                print(f"fetch retry {attempt} {name}: {exc}", flush=True)
                if attempt < 4:
                    time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"download failed after 5 attempts: {name}") from last_error
        print(f"fetched {name} {out.stat().st_size / 1e6:.1f} MB  used_h={used_h():.3f}", flush=True)
    return dest


def find_fs() -> Path:
    if os.environ.get("FS_URLS_FILE"):
        return fetch_urls(json.load(open(os.environ["FS_URLS_FILE"])), Path.cwd() / "fs")
    if SPEC.get("fs_urls"):
        return fetch_urls(SPEC["fs_urls"], Path("/tmp/fs") if Path("/kaggle").exists() else Path.cwd() / "fs")
    roots = [Path(os.environ["FS_ROOT"])] if os.environ.get("FS_ROOT") else [Path("/kaggle/input")]
    for root in roots:
        hits = sorted(root.rglob("fs_meta.json"))
        want = None if os.environ.get("FS_ROOT") else SPEC.get("fs")
        if want:
            hits = [h for h in hits if want in str(h)]
        if hits:
            return hits[0].parent
    raise FileNotFoundError("no feature store (fs_meta.json) mounted")


# ---- libraries -------------------------------------------------------------------------
def _norm_w(w):
    return None if w is None else w / np.mean(w)


class LGB:
    def __init__(self, params, threads):
        self.lgb = ensure("lightgbm")
        self.p = {**params, "num_threads": threads, "verbose": -1}

    def fit(self, X, y, w, Xv, yv, wv, rounds=None):
        lgb = self.lgb
        if rounds is None and (Xv is None or len(Xv) == 0):
            rounds = 300  # nothing to early-stop on (a small group in a fold)
        dtr = lgb.Dataset(X, y, weight=_norm_w(w), categorical_feature=self.cats)
        if rounds is None:
            dva = lgb.Dataset(Xv, yv, weight=_norm_w(wv), categorical_feature=self.cats, reference=dtr)
            m = lgb.train(self.p, dtr, MAX_ROUNDS, valid_sets=[dva], valid_names=["val"],
                          callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(1000)])
            return m, int(m.best_iteration)
        return lgb.train(self.p, dtr, rounds), rounds

    def predict(self, m, X, best=None):
        return m.predict(X, num_iteration=best)

    def importance(self, m, names):
        return pd.Series(m.feature_importance("gain"), index=names)


class XGB:
    def __init__(self, params, threads):
        self.xgb = ensure("xgboost")
        self.p = {"tree_method": "hist", "objective": "reg:squarederror", "nthread": threads, **params}

    def fit(self, X, y, w, Xv, yv, wv, rounds=None):
        xgb = self.xgb
        if rounds is None and (Xv is None or len(Xv) == 0):
            rounds = 300
        dtr = xgb.QuantileDMatrix(X, y, weight=_norm_w(w), enable_categorical=True,
                                  max_bin=self.p.get("max_bin", 256))
        if rounds is None:
            dva = xgb.QuantileDMatrix(Xv, yv, weight=_norm_w(wv), enable_categorical=True, ref=dtr)
            m = xgb.train(self.p, dtr, MAX_ROUNDS, evals=[(dva, "val")], early_stopping_rounds=300,
                          verbose_eval=1000)
            return m, int(m.best_iteration) + 1
        return xgb.train(self.p, dtr, rounds), rounds

    def predict(self, m, X, best=None):
        d = self.xgb.DMatrix(X, enable_categorical=True)
        return m.predict(d, iteration_range=(0, best) if best else (0, 0))

    def importance(self, m, names):
        g = m.get_score(importance_type="total_gain")
        return pd.Series({n: g.get(n, 0.0) for n in names})


class CAT:
    def __init__(self, params, threads):
        self.cb = ensure("catboost")
        self.p = {"loss_function": "RMSE", "thread_count": threads, "verbose": 1000, **params}

    def _pool(self, X, y=None, w=None):
        X = X.copy()
        for c in self.cats:
            X[c] = X[c].astype(str)
        return self.cb.Pool(X, y, weight=_norm_w(w), cat_features=self.cats)

    def fit(self, X, y, w, Xv, yv, wv, rounds=None):
        if rounds is None and (Xv is None or len(Xv) == 0):
            rounds = 300
        if rounds is None:
            m = self.cb.CatBoostRegressor(iterations=MAX_ROUNDS, od_type="Iter", od_wait=300,
                                          use_best_model=True, **self.p)
            m.fit(self._pool(X, y, w), eval_set=self._pool(Xv, yv, wv))
            return m, int(m.get_best_iteration()) + 1
        m = self.cb.CatBoostRegressor(iterations=rounds, **self.p)
        m.fit(self._pool(X, y, w))
        return m, rounds

    def predict(self, m, X, best=None):
        return m.predict(self._pool(X))

    def importance(self, m, names):
        return pd.Series(m.get_feature_importance(), index=names)


class MLP:
    """Feed-forward net: rank-gauss numerics + missing flags, embedded categoricals.

    The learning rate follows one cosine over params["epochs"], so a refit that stops at the
    fold's best epoch replays the same schedule.  `rounds` counts tenths of an epoch, which keeps
    the trainer's refit rule (1.1 x mean best, at least 50) meaningful."""

    def __init__(self, params, threads):
        self.torch = ensure("torch")
        self.p = {"hidden": [512, 256, 128], "dropout": 0.1, "lr": 2e-3, "wd": 1e-5, "batch": 2048,
                  "epochs": 14, "patience": 3, "seed": 0, **params}
        self.threads = threads

    def _training_device(self):
        device = self.torch.device(self.p.get("device", "cpu"))
        if device.type == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA MLP requested but no CUDA device is available")
        return device

    def _prep_fit(self, X):
        from scipy.special import erfinv
        cats = [c for c in self.cats if c in X]
        nums = [c for c in X.columns if c not in cats]
        A = X[nums].to_numpy(np.float32)
        rng = np.random.default_rng(0)
        s = A[rng.choice(len(A), min(len(A), 200_000), replace=False)]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q = np.nanpercentile(s, np.linspace(0, 100, 101), axis=0).T  # (cols, 101)
        q = np.nan_to_num(q)
        u = np.linspace(0.0, 1.0, 101)
        grid = np.sqrt(2) * erfinv(2 * np.clip(u, 0.002, 0.998) - 1)
        miss = np.where(np.isnan(A).any(axis=0))[0]
        card = [len(X[c].cat.categories) + 1 for c in cats]
        return dict(cats=cats, nums=nums, q=q, grid=grid, miss=miss, card=card)

    def _mat(self, prep, X):
        A = X[prep["nums"]].to_numpy(np.float32)
        M = np.isnan(A[:, prep["miss"]]).astype(np.float32)
        Z = np.empty((len(A), A.shape[1] + M.shape[1]), np.float32)
        for j in range(A.shape[1]):
            xp, col = prep["q"][j], A[:, j]
            xu, first = np.unique(xp, return_index=True)
            if len(xu) < 3:  # (near-)constant or binary: centred raw value
                Z[:, j] = np.nan_to_num(col - xp[50])
                continue
            # ties in the quantile table: map to the middle of each tied run
            last = np.r_[first[1:] - 1, len(xp) - 1]
            fp = (prep["grid"][first] + prep["grid"][last]) / 2
            Z[:, j] = np.nan_to_num(np.interp(col, xu, fp), nan=0.0)
        Z[:, A.shape[1]:] = M
        C = (np.stack([X[c].cat.codes.to_numpy().astype(np.int64) + 1 for c in prep["cats"]], 1)
             if prep["cats"] else np.zeros((len(X), 0), np.int64))
        return Z, C

    def _net(self, nz, card):
        nn = self.torch.nn
        torch = self.torch
        hidden, dropout = self.p["hidden"], self.p["dropout"]

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.ModuleList([nn.Embedding(k, min(24, int(1.6 * k ** 0.56) + 1)) for k in card])
                d = nz + sum(e.embedding_dim for e in s.emb)
                layers = []
                for h in hidden:
                    layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
                    d = h
                layers.append(nn.Linear(d, 1))
                s.mlp = nn.Sequential(*layers)

            def forward(s, z, c):
                return s.mlp(torch.cat([z] + [e(c[:, i]) for i, e in enumerate(s.emb)], 1)).squeeze(1)

        return Net()

    def fit(self, X, y, w, Xv, yv, wv, rounds=None):
        import copy
        import math
        torch = self.torch
        torch.set_num_threads(self.threads)
        torch.manual_seed(int(self.p["seed"]))
        rng = np.random.default_rng(int(self.p["seed"]))
        device = self._training_device()
        prep = self._prep_fit(X)
        y = np.asarray(y, np.float64)
        prep["ymu"], prep["ysd"] = float(np.mean(y)), float(np.std(y) or 1.0)
        Z, C = self._mat(prep, X)
        nz = Z.shape[1]
        zt, ct = torch.from_numpy(Z).to(device), torch.from_numpy(C).to(device)
        yt = torch.from_numpy(((y - prep["ymu"]) / prep["ysd"]).astype(np.float32)).to(device)
        wt = torch.from_numpy((np.ones(len(y)) if w is None else _norm_w(np.asarray(w, float))).astype(np.float32)).to(device)
        if device.type == "cuda":
            del Z, C  # GPU tensors own the data; release the large host copies before fitting
        val = rounds is None and Xv is not None and len(Xv) > 0
        if val:
            Zv, Cv = self._mat(prep, Xv)
            yv_ = (np.asarray(yv, np.float64) - prep["ymu"]) / prep["ysd"]
            wv_ = np.ones(len(yv_)) if wv is None else _norm_w(np.asarray(wv, float))
        E, B, n = int(self.p["epochs"]), int(self.p["batch"]), len(y)
        spe = math.ceil(n / B)
        run_steps = E * spe if rounds is None and val else int(round((rounds or 30) / 10 * spe))
        net = self._net(nz, prep["card"]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.p["lr"], weight_decay=self.p["wd"])
        total = E * spe
        best, best_state, best_ep, bad, t, ep = np.inf, None, 0, 0, 0, 0
        while t < run_steps:
            net.train()
            perm = torch.from_numpy(rng.permutation(n)).to(device)
            for i in range(0, n, B):
                if t >= run_steps:
                    break
                idx = perm[i:i + B]
                if len(idx) < 2:
                    continue
                for g in opt.param_groups:
                    g["lr"] = self.p["lr"] * 0.5 * (1 + math.cos(math.pi * min(t, total) / total))
                loss = (wt[idx] * (net(zt[idx], ct[idx]) - yt[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                t += 1
            ep += 1
            if val:
                pv = self._predict_std(net, Zv, Cv)
                net.to(device)  # validation/inference uses CPU; resume fitting on its requested device
                v = float(np.mean(wv_ * (pv - yv_) ** 2))
                print(f"  mlp epoch {ep}: val {v:.5f}", flush=True)
                if v < best - 1e-6:
                    best, best_state, best_ep, bad = v, copy.deepcopy(net.state_dict()), ep, 0
                else:
                    bad += 1
                    if bad >= int(self.p["patience"]):
                        break
        if val:
            net.load_state_dict(best_state)
            return (net.cpu(), prep), best_ep * 10
        return (net.cpu(), prep), int(round(run_steps / spe * 10))

    def _predict_std(self, net, Z, C):
        torch = self.torch
        torch.set_num_threads(self.threads)
        net = net.to("cpu")  # original predictions and saved-model replay share this exact route
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Z), 65536):
                out.append(net(torch.from_numpy(Z[i:i + 65536]), torch.from_numpy(C[i:i + 65536])).numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)

    def predict(self, m, X, best=None):
        net, prep = m
        Z, C = self._mat(prep, X)
        return self._predict_std(net, Z, C) * prep["ysd"] + prep["ymu"]

    def importance(self, m, names):
        return pd.Series(0.0, index=names)


LIBS = {"lgb": LGB, "xgb": XGB, "cat": CAT, "mlp": MLP}


# ---- training ----------------------------------------------------------------------------
def _anchor_parts(X: pd.DataFrame):
    spec = SPEC["anchor"]
    fb = X[spec.get("fallback", "te_stand_rwy")].to_numpy(dtype=float)
    nb = spec.get("nb")
    if nb and nb["col"] in X:
        # congestion measured on the neighbours' ADS-B, added linearly so it extrapolates
        x = np.clip(X[nb["col"]].to_numpy(dtype=float), *nb.get("clip", (-300.0, 3000.0)))
        n = X[nb["col"].replace("nbx_", "nbn_")].to_numpy(dtype=float)
        fb = fb + np.where((n >= nb.get("min_n", 3)) & np.isfinite(x), nb.get("k", 0.5) * x, 0.0)
    obs = X[spec["obs"]].to_numpy(dtype=float) if spec["obs"] in X else np.full(len(X), np.nan)
    ok = np.isfinite(obs) & (obs > spec.get("min", 30)) & (obs < spec.get("max", 7200))
    apt = X["apt"].astype(str).to_numpy()
    return fb, obs, ok, apt


def short_anchor(X: pd.DataFrame, y: np.ndarray, fit: np.ndarray) -> np.ndarray | None:
    """Anchor (SPEC["anchor"]): the measured taxi (ADS-B first movement + a per-airport
    offset) where it is plausible, else the stand x runway mean (+ k x neighbour excess).
    The models learn y - anchor.  The offsets are fitted on `fit` rows only."""
    if not SPEC.get("anchor"):
        return None
    fb, obs, ok, apt = _anchor_parts(X)
    off = pd.Series(y - obs).where(ok & fit & (y < CLIP_TARGET)).groupby(apt).median()
    ANCHOR["off"] = off
    ANCHOR["fb_med"] = float(np.nanmedian(fb))
    return anchor_apply(X)


def anchor_apply(X: pd.DataFrame) -> np.ndarray:
    fb, obs, ok, apt = _anchor_parts(X)
    a = np.where(ok, obs + pd.Series(apt).map(ANCHOR["off"]).fillna(0).to_numpy(), fb)
    return np.where(np.isfinite(a), a, ANCHOR["fb_med"])


ADSB_PREFIXES = ("adsb_", "a3_", "nbx_", "ratio_adsb_")
ADSB_KEEP = ("adsb_found", "adsb_per_", "adsb_typ_sr")


def mask_adsb(X: pd.DataFrame) -> pd.DataFrame:
    """The rows as they look when adsb.lol saw nothing (LIRF in Feb 2025, and in July 2026):
    adsb_found 0, neighbour ADS-B counts 0, every other ADS-B value missing; the period-typical
    stand x runway columns stay."""
    X = X.copy()
    for c in X.columns:
        if c.startswith(ADSB_PREFIXES) and not c.startswith(ADSB_KEEP):
            X[c] = np.nan
        elif c.startswith("nbn_"):
            X[c] = 0.0
    if "adsb_found" in X:
        X["adsb_found"] = 0.0
    return X


def _dropout_augment(X, y, ds, r, long, fit, valid):
    """SPEC["adsb_dropout"] = {"p": share, "apt": [...] or None}: append ADS-B-masked copies of a
    random share of the fit rows (optionally one set of airports), so every airport x season
    also has examples without ADS-B."""
    spec = SPEC["adsb_dropout"]
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    cand = fit.copy()
    if spec.get("apt"):
        cand &= X["apt"].astype(str).isin(spec["apt"]).to_numpy()
    pick = np.flatnonzero(cand & (rng.random(len(cand)) < float(spec["p"])))
    Xm = mask_adsb(X.iloc[pick])
    Xm.index = np.arange(len(X), len(X) + len(pick)) + (int(X.index.max()) + 1 - len(X))
    cat = lambda a: np.concatenate([a, a[pick]])  # noqa: E731
    return (pd.concat([X, Xm]), cat(y), cat(ds), cat(r), cat(long),
            np.concatenate([fit, np.ones(len(pick), bool)]),
            None if valid is None else np.concatenate([valid, np.zeros(len(pick), bool)]))


def fit_pair(libs, X, y, ds, r, long, fit, valid, rounds=None):
    """One short/long pair, or one per group of SPEC["split_by"] (per-airport models); in the
    split case `rounds` is {group: rounds} and the result is {"groups": {group: pair}}."""
    if SPEC.get("adsb_dropout"):
        X, y, ds, r, long, fit, valid = _dropout_augment(X, y, ds, r, long, fit, valid)
    if SPEC.get("split_by"):
        grp = X[SPEC["split_by"]].astype(str).to_numpy()
        out = {}
        for g in np.unique(grp):
            m = grp == g
            if not (fit & m).any():
                continue
            rg = None if rounds is None else rounds.get(g, rounds.get("*"))
            out[g] = fit_one(libs, X.loc[m], y[m], ds[m], r[m], long[m], fit[m],
                             None if valid is None else valid[m], rg)
        return {"groups": out}
    return fit_one(libs, X, y, ds, r, long, fit, valid, rounds)


ROWW = {}


def cell_weights(tr: pd.DataFrame, rk: pd.DataFrame, lo: float = 0.25, hi: float = 4.0) -> np.ndarray:
    """Covariate-shift weights for training rows: p_2026(cell) / p_2025(cell), where a cell is
    airport x ADS-B v3 grade (-1 = no event) x delay regime, clipped to [lo, hi], mean 1."""
    def key(f):
        g = np.where(f["a3_found"].to_numpy(dtype=float) > 0, f["a3_grade"].to_numpy(dtype=float), -1.0)
        g = np.nan_to_num(g, nan=-2.0).astype(int)
        lg = (f["d_sched"].to_numpy(dtype=float) > LONG_DS).astype(int)
        return pd.Series(f["apt"].astype(str).to_numpy()) + "|" + pd.Series(g).astype(str) + "|" + pd.Series(lg).astype(str)
    k_tr, k_rk = key(tr), key(rk)
    ratio = (k_rk.value_counts(normalize=True) / k_tr.value_counts(normalize=True)).clip(lo, hi)
    w = k_tr.map(ratio).fillna(lo).to_numpy(dtype=float)
    return w / w.mean()


def rclip() -> float:
    return float(SPEC.get("r_clip", R_CLIP))


def long_weight(ds: np.ndarray) -> np.ndarray:
    """Weight of a long row in the ratio regression.  (d r_hat - y)^2 = d^2 (r_hat - y/d)^2, so
    exact seconds SSE needs d^2; the default caps d at 12 h (review A §4, review B defect 10)."""
    mode = SPEC.get("long_w", "cap")
    if mode == "d2":
        return np.asarray(ds, float) ** 2
    if mode == "none":
        return np.ones_like(ds, dtype=float)
    cap = float(SPEC["w_cap_h"]) * 3600.0 if "w_cap_h" in SPEC else W_CAP
    return np.minimum(ds, cap) ** 2


def fit_one(libs, X, y, ds, r, long, fit, valid, rounds=None):
    ls, ll = libs
    w = long_weight(ds)
    rw = ROWW["w"][X.index.to_numpy()] if "w" in ROWW else None  # covariate-shift row weights
    if rw is not None:
        w = w * rw
    ys = np.minimum(y, CLIP_TARGET)
    anc = short_anchor(X, y, fit)
    ANCHOR["a"] = anc
    if anc is not None:
        ys = ys - anc
        if SPEC["anchor"].get("long"):
            # long model: residual on the anchor in ratio space, r - a/d, still weighted by d^2,
            # so its loss stays squared error in seconds of y - (a + f d)
            r = r - np.where(long, anc / np.where(long, ds, 1.0), 0.0)
    if SPEC.get("long_mode") == "mix":
        return fit_mix(libs, X, y, ds, long, fit, valid, rounds)
    if SPEC.get("long_target") == "sec":
        # long rows regressed on y in seconds (capped at R_CLIP x d) with unit weights: the same
        # squared-error loss as the ratio model, but leaves are constant in y rather than in y/d
        r = np.where(long, np.minimum(y, rclip() * ds), np.nan)
        w = np.ones_like(ds)
    sf, lf = fit & ~long, fit & long
    only = SPEC.get("only")
    ms = bs = ml = bl = None
    if rounds is None:
        sv, lv = valid & ~long, valid & long
        if only != "long":
            ms, bs = ls.fit(X.loc[sf], ys[sf], None if rw is None else rw[sf], X.loc[sv], ys[sv],
                            None if rw is None else rw[sv])
        if only != "short":
            ml, bl = ll.fit(X.loc[lf], r[lf], w[lf], X.loc[lv], r[lv], w[lv])
    else:
        if only != "long":
            ms, bs = ls.fit(X.loc[sf], ys[sf], None if rw is None else rw[sf], None, None, None, rounds[0])
        if only != "short":
            ml, bl = ll.fit(X.loc[lf], r[lf], w[lf], None, None, None, rounds[1])
    return (ms, bs or 0), (ml, bl or 0)


MIX_HELD_R = 0.6


def mix_classes(y, ds, long):
    """0 gate-held (taxied normally after a gate delay), 1 held after pushback, 2 schedule fill."""
    fill = long & (np.abs(y - ds) <= PATH_TOL)
    held = long & ~fill & (y >= MIX_HELD_R * ds)
    return fill, held, np.where(fill, 2, np.where(held, 1, 0))


def fit_mix(libs, X, y, ds, long, fit, valid, rounds=None):
    """Mixture for long rows: a taxi regressor in seconds trained on short and gate-held rows,
    a 3-class classifier {gate, held, fill} on long rows, and a ratio regressor on held rows.
    Prediction for a long row: p_gate * taxi + p_held * r_held * d + p_fill * d."""
    ls, ll = libs
    assert SPEC["lib"] == "lgb", "mixture mode is LightGBM-only"
    fill, held, cls = mix_classes(y, ds, long)
    gate = long & ~fill & ~held
    ys = np.minimum(y, CLIP_TARGET)
    w = long_weight(ds)
    r = np.clip(y / np.where(long, ds, 1.0), 0, rclip())
    mix = SPEC.get("mix", {})
    lc = LIBS[SPEC["lib"]]({**ll.p, **mix.get("clf_params", {}), "objective": "multiclass", "num_class": 3,
                             "metric": "multi_logloss"}, ll.p.get("num_threads", 4))
    lc.cats = ll.cats
    wc = w if mix.get("clf_weight") == "d2" else None
    tg, vg = fit & (~long | gate), (valid & (~long | gate)) if valid is not None else None
    tl = fit & long
    th = fit & held
    few_held = int(th.sum()) < 30  # too few held rows: a constant held ratio instead of a model
    if rounds is None:
        vl, vh = valid & long, valid & held
        mt, bt = ls.fit(X.loc[tg], ys[tg], None, X.loc[vg], ys[vg], None)
        mc, bc = lc.fit(X.loc[tl], cls[tl], None if wc is None else wc[tl], X.loc[vl], cls[vl],
                        None if wc is None else wc[vl])
        if few_held:
            mh, bh = float(np.median(r[th])) if th.any() else 0.75, 0
        else:
            mh, bh = ll.fit(X.loc[th], r[th], w[th], X.loc[vh], r[vh], w[vh])
    else:
        mt, bt = ls.fit(X.loc[tg], ys[tg], None, None, None, None, rounds[0])
        mc, bc = lc.fit(X.loc[tl], cls[tl], None if wc is None else wc[tl], None, None, None, rounds[1])
        if few_held:
            mh, bh = float(np.median(r[th])) if th.any() else 0.75, 0
        else:
            mh, bh = ll.fit(X.loc[th], r[th], w[th], None, None, None, max(rounds[2], 50))
    MIXLIB["clf"] = lc
    return [(mt, bt), (mc, bc), (mh, bh)]


MIXLIB = {}


def pred_mix(libs, pair, X, ds, long, use_best=True):
    ls, ll = libs
    (mt, bt), (mc, bc), (mh, bh) = pair
    out = ls.predict(mt, X, bt if use_best else None).copy()
    if long.any():
        XL = X.loc[long]
        p = np.asarray(MIXLIB["clf"].predict(mc, XL, bc if use_best else None)).reshape(-1, 3)
        rh = np.full(len(XL), mh) if isinstance(mh, float) else ll.predict(mh, XL, bh if use_best else None)
        rh = np.clip(rh, MIX_HELD_R * 0.5, rclip())
        d = ds[long]
        out[long] = p[:, 0] * out[long] + p[:, 1] * rh * d + p[:, 2] * d
    return out


ANCHOR = {}


def pred_pair(libs, pair, X, ds, long, use_best=True):
    if isinstance(pair, dict):
        grp = X[SPEC["split_by"]].astype(str).to_numpy()
        out = np.full(len(X), np.nan)
        for g, pg in pair["groups"].items():
            m = grp == g
            if m.any():
                out[m] = pred_one(libs, pg, X.loc[m], ds[m], long[m], use_best)
        return out
    return pred_one(libs, pair, X, ds, long, use_best)


def pred_one(libs, pair, X, ds, long, use_best=True):
    if SPEC.get("long_mode") == "mix":
        return pred_mix(libs, pair, X, ds, long, use_best)
    ls, ll = libs
    (ms, bs), (ml, bl) = pair
    if ms is None:  # only: "long"
        out = np.full(len(X), np.nan)
    else:
        out = ls.predict(ms, X, bs if use_best else None)
    out_anchor = anchor_apply(X) if SPEC.get("anchor") else None
    if out_anchor is not None:
        # the offset table comes from the fit rows of this pair (stored at fit time)
        out = out + out_anchor
    if long.any():
        out = out.copy()
        if ml is None:  # only: "short"
            out[long] = np.nan
        else:
            pr = ll.predict(ml, X.loc[long], bl if use_best else None)
            if SPEC.get("anchor", {}).get("long"):
                pr = pr + (out_anchor[long] / ds[long])
            if SPEC.get("long_target") == "sec":
                out[long] = np.clip(pr, 0, rclip() * ds[long])
            else:
                out[long] = np.clip(pr, 0, rclip()) * ds[long]
    if SPEC.get("pred_cap_d") is not None:
        big = ds > float(SPEC["pred_cap_d"])
        out = np.where(big, np.minimum(out, ds), out)
    return out


PRELOAD = {}  # variants mode: the store loaded once in the parent, shared copy-on-write by the forks


def _load_store(fs: Path, lib_name: str, want_keys: bool):
    key = (str(fs), lib_name == "cat")
    if key in PRELOAD and (PRELOAD[key][2] is not None or not want_keys):
        return PRELOAD[key]
    tr = pd.read_parquet(fs / "tr_base.parquet")
    rk = pd.read_parquet(fs / "rk_base.parquet")
    tr["y"] = tr["y"].astype(np.float64)
    keys = _enc_keys(tr) if want_keys else None  # before categorize() collapses rare levels
    if lib_name != "cat":
        categorize(tr, [rk], ref_mask=np.ones(len(tr), bool))
    return tr, rk, keys


def load_matrices(fs: Path, lib_name: str):
    want = any(month_fold(n) for n in SPEC.get("folds", [])) or bool(SPEC.get("enc_check"))
    tr, rk, keys = _load_store(fs, lib_name, want)
    if want:
        STATE["enc_keys"] = keys
    drop = set(DROP) | {"artifact"} | set(SPEC.get("drop_features", []))
    feats = [c for c in tr.columns if c not in drop]
    if SPEC.get("drop_adsb"):
        feats = [c for c in feats if not _is_adsb(c)]
        print(f"drop_adsb: {len(feats)} base features left", flush=True)
    return tr, rk, feats


def _is_adsb(c: str) -> bool:
    """ADS-B-derived columns (WS-J no-ADS-B member, research/cover/NOTES.txt)."""
    return c.startswith(("adsb_", "a3_", "nbx_", "nbn_", "ratio_adsb_"))


def assemble(base, enc, feats):
    enc = enc.astype(np.float32)
    if SPEC.get("drop_adsb"):
        enc = enc[[c for c in enc.columns if not _is_adsb(c)]]
    X = add_ratios(pd.concat([base[feats].reset_index(drop=True), enc.reset_index(drop=True)], axis=1))
    drop = [c for c in SPEC.get("drop_features", []) if c in X]
    if SPEC.get("drop_adsb"):
        drop += [c for c in X.columns if _is_adsb(c)]
    return X.drop(columns=drop)


SOURCE_HASHES = {}


def experiment_identity(fs: Path, columns: list, job: str) -> dict:
    """Hash the realized feature arm, immutable input bytes and effective spec."""
    key = str(fs.resolve())
    if key not in SOURCE_HASHES:
        SOURCE_HASHES[key] = {p.name: file_sha256(p) for p in sorted(fs.iterdir())
                              if p.suffix in (".parquet", ".json") and not p.name.endswith(".download.json")}
    spec = {k: v for k, v in SPEC.items() if k not in ("fs_urls", "model_urls", "variants")}
    script = Path(sys.argv[0])
    identity = {"spec": spec, "columns": columns, "job": job,
                "sources": SOURCE_HASHES[key], "code_sha256": file_sha256(script)}
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    (OUTPUT_DIR / f"identity_{job}.json").write_text(json.dumps(identity, indent=1))
    return identity


def make_libs(lib_name, params, params_long, threads, cats, seed_offset=0, gpu=None):
    out = []
    for p in (params, {**params, **(params_long or {})}):
        p = dict(p)
        if gpu is not None and lib_name == "mlp" and SPEC.get("mlp_device") == "cuda":
            p["device"] = f"cuda:{gpu}"
        if gpu is not None and lib_name == "xgb":
            p["device"] = f"cuda:{gpu}"
        if gpu is not None and lib_name == "cat" and p.get("task_type") == "GPU":
            p["devices"] = str(gpu)
        for key in ("seed", "random_seed", "random_state"):
            if key in p:
                p[key] = int(p[key]) + seed_offset
        lib = LIBS[lib_name](p, threads)
        lib.cats = cats
        out.append(lib)
    return tuple(out)


def _fold_rounds(name):
    fr = SPEC.get("fold_rounds")
    if fr is None:
        return None
    if isinstance(fr, dict) and name in fr:
        fr = fr[name]
    elif isinstance(fr, dict) and any(k in STATE["folds"] or month_fold(k) for k in fr):
        return None  # rounds given for other folds only: this one early-stops as before
    if isinstance(fr, dict):  # per-group rounds {group: [s, l]}
        return {g: tuple(int(x) for x in v) for g, v in fr.items()}
    return tuple(int(x) for x in fr)


def _best_of(pair):
    if isinstance(pair, dict):
        return {g: tuple(int(m[1]) for m in p) for g, p in pair["groups"].items()}
    return tuple(int(m[1]) for m in pair)


def _cov26_mask(tr, sel, seed):
    """Rows of `sel` whose ADS-B is masked so that each airport x month (Jan, Jul) has the 2026
    ranking file's ADS-B v3 coverage: a random share max(0, 1 - cov26 / cov25) of the covered
    rows.  Months other than Jan/Jul have no 2026 counterpart and stay unmasked."""
    rk = STATE["rk"]
    f_rk = rk["a3_found"].fillna(0).to_numpy() > 0
    cov26 = pd.Series(f_rk).groupby([rk["apt"].astype(str).to_numpy(), rk["month"].to_numpy()]).mean()
    found = tr["a3_found"].fillna(0).to_numpy() > 0
    apt, month = tr["apt"].astype(str).to_numpy(), tr["month"].to_numpy()
    rng = np.random.default_rng(seed)
    u = rng.random(len(tr))
    out = np.zeros(len(tr), bool)
    for (a, mo), c26 in cov26.items():
        cell = sel & (apt == a) & (month == mo)
        if not cell.any():
            continue
        c25 = found[cell].mean()
        p = max(0.0, 1.0 - c26 / c25) if c25 > 0 else 0.0
        out |= cell & found & (u < p)
        print(f"  cov26 {a} m{mo}: 2025 {c25:.3f} 2026 {c26:.3f} -> mask {p:.3f} of covered rows", flush=True)
    return out


def run_fold(name, lib, X, tr, fold_val, threads):
    y = tr["y"].to_numpy()
    ds = tr["d_sched"].to_numpy()
    long = ds > LONG_DS
    r = np.where(long, np.clip(y / np.where(long, ds, 1.0), 0, rclip()), np.nan)
    art = tr["artifact"].to_numpy().astype(bool)
    t0 = time.time()
    fit, val = ~fold_val & ~art, fold_val & ~art
    fixed = _fold_rounds(name)
    inner = SPEC.get("es_inner")
    if fixed is not None:
        # fixed rounds: nothing is early-stopped, least of all on the scored rows
        pair = fit_pair(lib, X, y, ds, r, long, fit, None, fixed)
    elif inner:
        days = inner.get("days", list(range(8, 15)))
        iv = fit & np.isin(tr["mday"].to_numpy(), days)
        print(f"[fold {name}] inner early stopping on {int(iv.sum()):,} training-side rows (days {days})", flush=True)
        pair = fit_pair(lib, X, y, ds, r, long, fit & ~iv, iv)
        if inner.get("refit"):
            mult = float(inner.get("mult", 1.1))
            b = _best_of(pair)
            rr = ({g: tuple(max(int(x * mult), 50) for x in v) for g, v in b.items()} if isinstance(b, dict)
                  else tuple(max(int(x * mult), 50) for x in b))
            print(f"[fold {name}] inner best {b} -> refit on the whole training side at {rr}", flush=True)
            pair = fit_pair(lib, X, y, ds, r, long, fit, None, rr)
    else:
        pair = fit_pair(lib, X, y, ds, r, long, fit, val)
    pv = pred_pair(lib, pair, X.loc[fold_val].reset_index(drop=True), ds[fold_val], long[fold_val])
    keep = ~art[fold_val] & np.isfinite(pv)
    score = rmse(y[fold_val][keep], pv[keep])
    em = SPEC.get("eval_mask")
    if em and em.get("mode") == "cov26":
        msk = _cov26_mask(tr, fold_val, int(em.get("seed", 0)))
        pc = pv.copy()
        inval = msk[fold_val]  # masked rows, in fold-row order
        if msk.any():
            pm = pred_pair(lib, pair, mask_adsb(X.loc[msk]).reset_index(drop=True), ds[msk], long[msk])
            pc[inval] = pm
            ok = np.isfinite(pm)
            print(f"[fold {name}] cov26 stress: {int(msk.sum()):,} rows masked, their RMSE masked "
                  f"{rmse(y[msk][ok], pm[ok]):.3f} vs unmasked {rmse(y[msk][ok], pv[inval][ok]):.3f}", flush=True)
        pd.DataFrame({"id": tr.loc[fold_val, ID].to_numpy(), "fold": name, "prediction": pc}).to_csv(
            OUTPUT_DIR / f"oof_{name}_cov26.csv", index=False)
    elif em:  # the same fold rows re-predicted with ADS-B masked (e.g. LIRF in July, as in 2026)
        sel = fold_val & ~art
        if em.get("apt"):
            sel &= tr["apt"].astype(str).isin(em["apt"]).to_numpy()
        if em.get("months"):
            sel &= np.isin(tr["month"].to_numpy(), em["months"])
        pm = pred_pair(lib, pair, mask_adsb(X.loc[sel]).reset_index(drop=True), ds[sel], long[sel])
        pd.DataFrame({"id": tr.loc[sel, ID].to_numpy(), "fold": name, "prediction": pm}).to_csv(
            OUTPUT_DIR / f"oof_{name}_masked.csv", index=False)
        print(f"[fold {name}] masked eval on {int(sel.sum())} rows: {rmse(y[sel], pm):.3f} "
              f"(unmasked {rmse(y[sel], pred_pair(lib, pair, X.loc[sel].reset_index(drop=True), ds[sel], long[sel])):.3f})",
              flush=True)
    first = next(iter(pair["groups"].values())) if isinstance(pair, dict) else pair
    print(f"[fold {name}] test-like {score:.3f}  short {first[0][1]}  long {first[1][1]}  "
          f"{(time.time() - t0) / 60:.1f} min  {mem_gb()}", flush=True)
    return pv, _best_of(pair), score, pair


STATE = {}


def _targets(tr):
    y = tr["y"].to_numpy()
    ds = tr["d_sched"].to_numpy()
    long = ds > LONG_DS
    r = np.where(long, np.clip(y / np.where(long, ds, 1.0), 0, rclip()), np.nan)
    art = tr["artifact"].to_numpy().astype(bool)
    return y, ds, long, r, art


def fold_job(name: str, threads: int) -> None:
    g = STATE
    gpus = int(SPEC.get("gpus", 0))
    if gpus and "gpu_id" in SPEC:  # a variant process pinned to one GPU
        gpu = int(SPEC["gpu_id"]) % gpus
    elif gpus and "slot" in g:  # waves (max_parallel): one job per GPU at a time
        gpu = g["slot"].get(name, 0) % gpus
    else:
        gpu = (sorted(g["folds"]).index(name) % gpus) if gpus else None
    lib = make_libs(g["lib_name"], g["params"], g["params_long"], threads, g["cats"], gpu=gpu)
    X = assemble(g["tr"], fold_encodings(name), g["feats"])
    experiment_identity(g["fs"], list(X.columns), name)
    val = g["folds"][name]
    pv, best, score, pair = run_fold(name, lib, X, g["tr"], val, threads)
    pd.DataFrame({"id": normalized_ids(g["tr"].loc[val, ID].to_numpy()), "fold": name, "prediction": pv}).to_csv(
        OUTPUT_DIR / f"oof_{name}.csv", index=False)
    json.dump({"best": best if isinstance(best, dict) else list(best), "score": score},
              open(OUTPUT_DIR / f"best_{name}.json", "w"))
    first = next(iter(pair["groups"].values())) if isinstance(pair, dict) else pair
    lib_i, m_i = (lib[0], first[0][0]) if first[0][0] is not None else (lib[1], first[1][0])
    lib_i.importance(m_i, list(X.columns)).sort_values(ascending=False).to_csv(OUTPUT_DIR / f"imp_{name}.csv")
    if SPEC.get("save_fold_models"):
        save_model(name, lib, pair, list(X.columns), prefix="fold")


def refit_job(k: int, threads: int) -> None:
    g = STATE
    gpus = int(SPEC.get("gpus", 0))
    slot = int(SPEC["gpu_id"]) if "gpu_id" in SPEC else (g["slot"].get(k, k) if "slot" in g else k)
    lib = make_libs(g["lib_name"], g["params"], g["params_long"], threads, g["cats"], seed_offset=1000 * k,
                    gpu=(slot % gpus) if gpus else None)
    y, ds, long, r, art = _targets(g["tr"])
    pair = fit_pair(lib, g["X_all"], y, ds, r, long, ~art, None, g["rounds"])
    rk_ds = g["rk"]["d_sched"].to_numpy()
    pred = pred_pair(lib, pair, g["X_rk"], rk_ds, rk_ds > LONG_DS, use_best=False)
    np.save(OUTPUT_DIR / f"refit_{k}.npy", pred)
    if SPEC.get("save_models"):
        save_model(k, lib, pair, list(g["X_all"].columns))
    print(f"refit {k}: rounds {g['rounds']}  used_h={used_h():.3f}", flush=True)


# ---- saved models (phase 2: predict a new month without refitting) ----------------------------
def _map_pair(fn_s, fn_l, pair):
    if isinstance(pair, dict):
        return {"groups": {g: _map_pair(fn_s, fn_l, p) for g, p in pair["groups"].items()}}
    (ms, bs), (ml, bl) = pair
    return (fn_s(ms), bs), (fn_l(ml), bl)


def _pack(lib, m):
    if m is not None and isinstance(lib, MLP):
        net, prep = m
        return {"mlp_state": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "prep": prep}
    return m


def _unpack(lib, m):
    if isinstance(m, dict) and "mlp_state" in m:
        prep = m["prep"]
        net = lib._net(len(prep["nums"]) + len(prep["miss"]), prep["card"])
        net.load_state_dict(m["mlp_state"])
        return net, prep
    if m is not None and type(m).__module__.startswith("xgboost"):
        m.set_param({"device": "cpu"})  # a GPU-trained booster predicts on any host
    return m


def save_model(k, libs, pair, columns, prefix="refit"):
    import pickle
    if SPEC.get("long_mode") == "mix":
        raise ValueError("save_models does not support long_mode mix")
    d = OUTPUT_DIR / "models"
    d.mkdir(parents=True, exist_ok=True)
    g = STATE
    bundle = {"spec": {k2: v for k2, v in SPEC.items() if k2 not in ("fs_urls", "variants")},
              "lib_name": g["lib_name"], "params": libs[0].p, "params_long": libs[1].p, "cats": g["cats"],
              "columns": columns, "anchor": {a: v for a, v in ANCHOR.items() if a != "a"},
              "pair": _map_pair(lambda m: _pack(libs[0], m), lambda m: _pack(libs[1], m), pair)}
    path = d / f"{prefix}_{k}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    path.with_suffix(".sha256").write_text(file_sha256(path) + "\n")
    print(f"saved model {path.name} {path.stat().st_size / 1e6:.1f} MB", flush=True)


def _model_files() -> list:
    import glob as _glob
    if SPEC.get("model_urls"):
        fetch_urls(SPEC["model_urls"], Path("/tmp/models"))
        return sorted(Path("/tmp/models").glob("*.pkl"))
    pats = SPEC.get("models") or ["/kaggle/input/**/models/refit_*.pkl"]
    out = []
    for pat in pats:
        p = Path(pat)
        out += sorted(p.glob("refit_*.pkl")) if p.is_dir() else [Path(x) for x in sorted(_glob.glob(pat, recursive=True))]
    if SPEC.get("models_kernels"):
        out = [p for p in out if any(s in str(p) for s in SPEC["models_kernels"])]
    if not out:
        raise FileNotFoundError(f"predict_only: no saved models under {pats}")
    return out


def predict_only_main():
    """Predict this store's ranking rows with saved refit models; no training."""
    import pickle
    g = globals()
    base_spec = dict(SPEC)
    files = _model_files()
    print(f"predict_only: {len(files)} saved models: {[str(f) for f in files]}", flush=True)
    fs = find_fs()
    first = pickle.load(open(files[0], "rb"))
    lib_name = first["lib_name"]
    g["SPEC"] = {**first["spec"], "fs": base_spec.get("fs", first["spec"].get("fs"))}
    tr, rk, feats = load_matrices(fs, lib_name)
    STATE.update(tr=tr, rk=rk, feats=feats, fs=fs)
    X_rk = assemble(rk, pd.read_parquet(fs / "enc_full_rk.parquet"), feats)
    ds = rk["d_sched"].to_numpy()
    threads = int(base_spec.get("threads", ncpu()))
    preds = []
    for k, f in enumerate(files):
        b = first if k == 0 else pickle.load(open(f, "rb"))
        assert b["lib_name"] == lib_name, f"{f}: mixed libraries in one predict_only run"
        g["SPEC"] = dict(b["spec"])
        libs = make_libs(b["lib_name"], b["params"], b["params_long"], threads, b["cats"])
        for lib, p in zip(libs, (b["params"], b["params_long"])):
            lib.p = p  # exactly the fitted parameters (seed offsets already applied)
        pair = _map_pair(lambda m: _unpack(libs[0], m), lambda m: _unpack(libs[1], m), b["pair"])
        ANCHOR.clear()
        ANCHOR.update(b["anchor"])
        missing = [c for c in b["columns"] if c not in X_rk]
        assert not missing, f"{f}: store lacks columns {missing[:10]}"
        pred = pred_pair(libs, pair, X_rk[b["columns"]], ds, ds > LONG_DS, use_best=False)
        np.save(OUTPUT_DIR / f"refit_{k}.npy", pred)
        preds.append(pred)
        print(f"predicted with {f.name}: mean {np.nanmean(pred):.1f}  used_h={used_h():.3f}", flush=True)
    g["SPEC"] = base_spec
    pred = np.mean(preds, axis=0)
    pd.DataFrame({ID: normalized_ids(rk[ID].to_numpy()), TARGET: np.clip(pred, 60, None)}).to_parquet(
        OUTPUT_DIR / "submission.parquet", index=False)
    (OUTPUT_DIR / "results.json").write_text(json.dumps(
        {"predict_only": [str(f) for f in files], "test_mean": float(pred.mean()), "used_h": used_h()}, indent=1))
    print(f"TRAIN_DONE used_h={used_h():.3f}", flush=True)


MONTH_ALIASES = {"MJ": 1, "MY": 7}


def month_fold(name: str):
    """Month number of a whole-month outer fold name (M1..M12, MJ, MY), else None."""
    if name in MONTH_ALIASES:
        return MONTH_ALIASES[name]
    if len(name) > 1 and name[0] == "M" and name[1:].isdigit() and 1 <= int(name[1:]) <= 12:
        return int(name[1:])
    return None


def fold_encodings(name: str) -> pd.DataFrame:
    """The store's enc_<fold>.parquet, or, for a fold the store lacks (month-out folds), the
    same encodings computed here with fs_main's recipe: fit = every non-artifact row outside the
    fold, training rows leave their own month out."""
    g = STATE
    p = g["fs"] / f"enc_{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    # Month name alone aliases encodings from different stores/recipes.
    cache_key = hashlib.sha256((file_sha256(g["fs"] / "tr_base.parquet")
                               + file_sha256(Path(sys.argv[0])) + name).encode()).hexdigest()
    cache = Path("/tmp/enc_cache") / cache_key / f"enc_{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    t0 = time.time()
    enc = _encode(g["folds"][name])
    cache.parent.mkdir(parents=True, exist_ok=True)
    part = cache.with_name(cache.name + f".{os.getpid()}.part")
    enc.to_parquet(part, index=False)
    os.replace(part, cache)
    print(f"enc_{name}: computed in-kernel {enc.shape} in {time.time() - t0:.0f} s", flush=True)
    return enc


def _encode(val: np.ndarray) -> pd.DataFrame:
    g = STATE
    tr = g["tr"]
    month = tr["month"].to_numpy()
    y = tr["y"].to_numpy()
    ds = tr["d_sched"].to_numpy()
    long = ds > LONG_DS
    path = long & (np.abs(y - ds) <= PATH_TOL)
    r = np.where(long, np.clip(y / np.where(long, ds, 1.0), 0, R_CLIP), np.nan)  # the store's clip, not SPEC's
    art = tr["artifact"].to_numpy().astype(bool)
    keys = g["enc_keys"]
    dummy = {k: v[:1] for k, v in keys.items()}
    enc, _ = all_encodings(keys, month, y, ds, r, long, path, ~val & ~art, dummy)
    return enc.astype(np.float32)


def _enc_keys(tr: pd.DataFrame) -> dict:
    """Raw (uncollapsed) encoding keys as integer codes, taken before categorize() collapses rare
    levels, so in-kernel encodings match the store's."""
    names = sorted(set(TE_KEYS + LONG_KEYS + ["apt"]))
    return {k: pd.factorize(tr[k].astype(str).to_numpy())[0] for k in names}


def _run(jobs, parallel: bool, total_threads: int):
    k = int(SPEC.get("max_parallel", 0) or 0)
    if parallel and k and len(jobs) > k:
        # waves of k processes; slot i of a wave gets GPU i % gpus
        for i in range(0, len(jobs), k):
            wave = jobs[i:i + k]
            STATE["slot"] = {arg: j for j, (_, arg) in enumerate(wave)}
            _run(wave, True, total_threads)
        STATE.pop("slot", None)
        return
    if parallel and k and "slot" not in STATE:
        STATE["slot"] = {arg: j for j, (_, arg) in enumerate(jobs)}
    if not parallel or len(jobs) == 1:
        for fn, arg in jobs:
            fn(arg, total_threads)
        return
    from multiprocessing import get_context
    ctx = get_context("fork")
    per = max(total_threads // len(jobs), 1)
    procs = [ctx.Process(target=fn, args=(arg, per)) for fn, arg in jobs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode]
    if bad:
        raise RuntimeError(f"worker exit codes {bad}")


def train_main():
    if SPEC.get("predict_only"):
        return predict_only_main()
    spec = SPEC
    from importlib.metadata import version, PackageNotFoundError
    import platform
    library = {"cat": "catboost", "lgb": "lightgbm", "xgb": "xgboost", "mlp": "torch"}.get(spec["lib"], spec["lib"])
    for pkg in sorted(set(spec.get("package_versions", {})) | {library}):
        ensure(pkg)
    environment = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for pkg in ("numpy", "pandas", "pyarrow", "scipy", "catboost", "lightgbm", "xgboost", "torch"):
        try:
            environment["packages"][pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    (OUTPUT_DIR / "environment.json").write_text(json.dumps(environment, indent=1))
    lib_name = spec["lib"]
    threads = int(spec.get("threads", ncpu()))
    parallel = bool(spec.get("parallel", threads >= 32))
    fs = find_fs()
    print(f"spec {json.dumps(spec)}\nfeature store {fs}  cores {threads}  parallel {parallel}", flush=True)
    tr, rk, feats = load_matrices(fs, lib_name)
    if spec.get("row_weight") == "cell26":
        ROWW["w"] = cell_weights(tr, rk, *spec.get("row_weight_clip", (0.25, 4.0)))
        q = np.percentile(ROWW["w"], [1, 25, 50, 75, 99])
        print(f"row weights (2026 cell mix): p1/25/50/75/99 {np.round(q, 2).tolist()}", flush=True)
    month, mday = tr["month"].to_numpy(), tr["mday"].to_numpy()
    jj = np.isin(month, VAL_MONTHS)
    folds = {"A": jj & (mday >= 16), "B": jj & (mday <= 15), "F": np.isin(month, (11, 12))}
    for n in spec.get("folds", []) or []:
        if month_fold(n):
            folds[n] = month == month_fold(n)
    STATE.update(dict(lib_name=lib_name, params=spec["params"], params_long=spec.get("params_long"), cats=[c for c in CATS if c in feats], tr=tr,
                      rk=rk, feats=feats, fs=fs, folds=folds))
    print(f"loaded {len(tr):,} x {len(feats)}  used_h={used_h():.3f}  {mem_gb()}", flush=True)
    if spec.get("enc_check"):
        mine, store = _encode(folds["A"]), pd.read_parquet(fs / "enc_A.parquet")
        diff = (mine[store.columns].astype(np.float64) - store.astype(np.float64)).abs()
        print(f"enc_check A: max |diff| {np.nanmax(diff.to_numpy()):.6f}  NaN pattern equal "
              f"{bool((mine[store.columns].isna().to_numpy() == store.isna().to_numpy()).all())}", flush=True)

    names = spec.get("folds", ["A", "B"])
    expected = ["environment.json"] + [f"{stem}_{n}.{ext}" for n in names
                for stem, ext in (("oof", "csv"), ("best", "json"), ("imp", "csv"), ("identity", "json"))]
    if spec.get("save_fold_models"):
        expected += [f"models/fold_{n}.pkl" for n in names]
    if spec.get("refit", True):
        expected += ["submission.parquet", "identity_refit.json"]
        expected += [f"refit_{k}.npy" for k in range(int(spec.get("refit_seeds", 1)))]
        if spec.get("save_models"):
            expected += [f"models/refit_{k}.pkl" for k in range(int(spec.get("refit_seeds", 1)))]
    (OUTPUT_DIR / "expected_outputs.json").write_text(json.dumps(expected, indent=1))
    info = {}
    if names:
        _run([(fold_job, n) for n in names], parallel, threads)
        oof = pd.concat([pd.read_csv(OUTPUT_DIR / f"oof_{n}.csv") for n in names])
        oof.to_csv(OUTPUT_DIR / "oof.csv", index=False)
        oof.to_parquet(OUTPUT_DIR / "oof.parquet", index=False)
        cov = [OUTPUT_DIR / f"oof_{n}_cov26.csv" for n in names]
        if all(p.exists() for p in cov):
            pd.concat([pd.read_csv(p) for p in cov]).to_csv(OUTPUT_DIR / "oof_cov26.csv", index=False)
        info = {n: json.load(open(OUTPUT_DIR / f"best_{n}.json")) for n in names}
        imp = pd.read_csv(OUTPUT_DIR / f"imp_{names[0]}.csv", index_col=0).iloc[:, 0]
        print(f"top short features:\n{imp.head(25).to_string()}", flush=True)
    else:
        # refit-only: rounds come from an earlier run's folds (SPEC["rounds"] = [short, long] best iters)
        given = spec["rounds"]
        info = {"given": {"best": given if isinstance(given, dict) else list(given), "score": None}}
    results = {"spec": spec, "folds": info}
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=1))

    if spec.get("refit", True):
        X_all = assemble(tr, pd.read_parquet(fs / "enc_full.parquet"), feats)
        X_rk = assemble(rk, pd.read_parquet(fs / "enc_full_rk.parquet"), feats)[list(X_all.columns)]
        experiment_identity(fs, list(X_all.columns), "refit")
        bests = [v["best"] for v in info.values()]
        if isinstance(bests[0], dict):  # per-group rounds, averaged over the folds that have the group
            groups = set().union(*bests)
            rounds = {g: tuple(max(int(x * 1.1), 50) for x in
                               np.mean([b[g] for b in bests if g in b], axis=0)) for g in groups}
        else:
            b = np.array(bests, dtype=float).mean(axis=0)
            rounds = tuple(max(int(x * 1.1), 50) for x in b)
        STATE.update(dict(X_all=X_all, X_rk=X_rk, rounds=rounds))
        seeds = int(spec.get("refit_seeds", 1))
        _run([(refit_job, k) for k in range(seeds)], parallel, threads)
        pred = np.mean([np.load(OUTPUT_DIR / f"refit_{k}.npy") for k in range(seeds)], axis=0)
        sub = pd.DataFrame({ID: normalized_ids(rk[ID].to_numpy()), TARGET: np.clip(pred, 60, None)})
        sub.to_parquet(OUTPUT_DIR / "submission.parquet", index=False)
        results.update({"refit_rounds": rounds, "test_mean": float(pred.mean()), "test_max": float(pred.max())})
    results["used_h"] = used_h()
    missing = [name for name in expected if not (OUTPUT_DIR / name).is_file() or not (OUTPUT_DIR / name).stat().st_size]
    if missing:
        raise RuntimeError(f"incomplete experiment outputs: {missing}")
    results["complete"] = True
    results["outputs_sha256"] = {name: file_sha256(OUTPUT_DIR / name) for name in expected}
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=1))
    print(f"TRAIN_DONE used_h={used_h():.3f}", flush=True)


def _variant_job(vname: str, threads: int) -> None:
    g = globals()
    base, slot = VARIANTS["base"], VARIANTS["slot"].get(vname, 0)
    spec = {**base, **VARIANTS["specs"][vname], "threads": threads, "parallel": False, "gpu_id": slot}
    spec.pop("variants", None)
    spec.pop("max_parallel", None)
    g["SPEC"] = spec
    g["MAX_ROUNDS"] = int(spec.get("max_rounds", MAX_ROUNDS))
    out = VARIANTS["root"] / vname
    out.mkdir(parents=True, exist_ok=True)
    g["OUTPUT_DIR"] = out
    STATE.clear()
    t0 = time.time()
    try:
        train_main()
    except Exception:
        import traceback
        (out / "error.txt").write_text(traceback.format_exc())
        raise
    print(f"[variant] {vname} done in {(time.time() - t0) / 60:.1f} min  used_h={used_h():.3f}", flush=True)


VARIANTS = {}


def variants_main():
    """SPEC["variants"] = {name: overrides}: run several specs on one host (e.g. the two T4s of a
    Kaggle GPU session), each in its own process with SPEC = base | overrides, in waves of
    SPEC["max_parallel"] (default: number of GPUs, else 1); variant i of a wave gets GPU i.
    Each writes OUTPUT_DIR/<name>/ exactly what a single trainer writes."""
    base = dict(SPEC)
    _prepare_variant_packages(base)  # pip must finish before any child can import the library
    names = list(base["variants"])
    k = int(base.get("max_parallel") or base.get("gpus") or 1)
    threads = int(base.get("threads", ncpu()))
    VARIANTS.update(base=base, specs=base["variants"], root=OUTPUT_DIR)
    # load the store once here: each variant process loading its own copy ran a 33 GB GPU host out
    # of memory (one variant per wave killed, exit -9, prc-m-wse-seeds / prc-m-wse-longgpu)
    libs = {base["variants"][n].get("lib", base["lib"]) for n in names}
    want = any(month_fold(f) or f == "enc_check" for n in names
               for f in base["variants"][n].get("folds", base.get("folds", ["A", "B"])))
    fs = find_fs()
    for lib in libs:
        PRELOAD[(str(fs), lib == "cat")] = _load_store(fs, lib, want)
    print(f"[variant] store preloaded for {sorted(libs)}  {mem_gb()}", flush=True)
    print(f"[variant] {len(names)} variants, {k} at a time on {threads} cores: {names}", flush=True)
    from multiprocessing import get_context
    ctx = get_context("fork")
    codes = {}
    for i in range(0, len(names), k):
        wave = names[i:i + k]
        VARIANTS["slot"] = {n: j for j, n in enumerate(wave)}
        procs = {n: ctx.Process(target=_variant_job, args=(n, max(threads // len(wave), 1))) for n in wave}
        for p in procs.values():
            p.start()
        for n, p in procs.items():
            p.join()
            codes[n] = p.exitcode
        print(f"[variant] wave {wave} exit codes {[codes[n] for n in wave]}  used_h={used_h():.3f}", flush=True)
    (OUTPUT_DIR / "variants.json").write_text(json.dumps({"codes": codes, "used_h": used_h()}))
    if any(code != 0 for code in codes.values()):
        raise RuntimeError(f"failed experiment variants: {codes}")


if __name__ == "__main__":
    if SPEC.get("variants"):
        variants_main()
    else:
        train_main()
