"""Assemble a Kaggle script kernel from the shared modules.

    python3 solutions/core/build_kernel.py fs  prc-fs1
    python3 solutions/core/build_kernel.py train prc-tr-lgb1 --fs prc-fs1 --accel tpu --spec '{"lib": "lgb", ...}'
    python3 solutions/core/build_kernel.py derive prc-f-fs8 --fs prc-f-fs7 --opts '{"rk_file": "final_ranking.parquet"}'

A feature-store kernel is features.py + fs_main.py (+ any --extra modules, which may
override build()).  A trainer is features.py + train_main.py with SPEC filled in; it mounts
the feature-store kernel's output.  A derived store (fs8.py) mounts a built store (--fs) and the raw data
and appends the g8_* columns to its tr_base / rk_base, copying everything else unchanged.  Kernels land in
solutions/kernels/<slug>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CORE = Path(__file__).resolve().parent
SOL = CORE.parent
OWNER = "usiadianimuwa"
DATA = ["usiadianimuwa/prc-taxi-2026-data", "usiadianimuwa/prc-taxi-2026-metar"]
WHEELS = "usiadianimuwa/sbrt-wheels"
GEOMETRY = "usiadianimuwa/prc-taxi-2026-geometry"
ADSB = json.loads((SOL / "adsb" / "sources.json").read_text())["kernel_sources"]
# final phase (--opts '{"rk_file": "final_ranking.parquet", ...}'): the organisers' final files and the
# v1 ADS-B extracts of its new months (solutions/adsb/build_final.py)
FINAL = "usiadianimuwa/prc-taxi-2026-final"
ADSB_FINAL = [f"usiadianimuwa/prc-adsb-extract-{p}" for p in ("202602ra", "202602rb", "202606ra", "202606rb")]


def data_sets() -> list[str]:
    return DATA + [GEOMETRY] + ([FINAL] if OPTS.get("rk_file") else [])


def adsb_kernels() -> list[str]:
    return ADSB + (ADSB_FINAL if OPTS.get("rk_file") else [])


def assemble(parts: list[Path], spec: dict | None, specs: dict | None = None) -> str:
    text = []
    for i, p in enumerate(parts):
        body = p.read_text()
        if i < len(parts) - 1 and 'if __name__ == "__main__":' in body:
            body = body[:body.rindex('if __name__ == "__main__":')]
        if specs is not None and p.name == "batch_main.py":
            body = body.replace("SPECS = {}  # replaced by build_kernel.py: {name: spec}", f"SPECS = {specs!r}", 1)
            body = body.replace("FS_URLS = {}  # replaced by build_kernel.py", f"FS_URLS = {FS_URLS!r}", 1)
        if p.name == "features.py" and OPTS:
            body = body.replace("FS_OPTS = {}  # replaced by build_kernel.py: fs7 options", f"FS_OPTS = {OPTS!r}", 1)
            if OPTS.get("pru"):
                body = body.replace("PRU_TAB = {}  # replaced by build_kernel.py", f"PRU_TAB = {pru_table()!r}  #", 1)
        if p.name == "remote_inputs.py" and INPUTS:
            body = body.replace("INPUT_URLS = {}  # replaced by build_kernel.py: {relative path: url}",
                                f"INPUT_URLS = {INPUTS!r}", 1)
        if spec is not None and p.name == "train_main.py":
            body = body.replace("SPEC = {}  # replaced by build_kernel.py", f"SPEC = {spec!r}", 1)
        text.append(f"# ==== {p.name} ====\n{body}")
    return "\n\n".join(text)


def write(slug: str, code: str, accel: str, datasets: list[str], kernels: list[str], internet=False,
          owner: str = OWNER) -> Path:
    d = SOL / "kernels" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "kernel.py").write_text(code)
    meta = {"id": f"{owner}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": accel == "gpu",
            "enable_tpu": accel == "tpu", "enable_internet": internet, "competition_sources": [],
            "dataset_sources": datasets, "kernel_sources": kernels}
    if accel == "gpu":
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    compile(code, str(d / "kernel.py"), "exec")
    return d


INPUTS: dict = {}
FS_URLS: dict = {}
OPTS: dict = {}
PRU_DIR = SOL.parent / "research" / "pru"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def pru_table() -> dict:
    """EUROCONTROL PRU taxi-out additional time (ansperformance.eu open data), per airport-month:
    published mean taxi-out (s), unimpeded reference (s), de-iced/invalid share, no-reference share, and
    the published mean minus the airport's 2025 annual mean."""
    import pandas as pd
    p = pd.concat([pd.read_csv(PRU_DIR / f"taxi_out_additional_time_{y}.csv") for y in (2025, 2026)])
    p = p[p["APT_ICAO"].isin(APTS)].copy()
    p["pru_pub"] = (p["TOTAL_REF_TIME_MIN"] + p["TOTAL_ADD_TIME_MIN"]) * 60 / p["TOTAL_REF_NB_FL"]
    p["pru_ref"] = p["TOTAL_REF_TIME_MIN"] * 60 / p["TOTAL_REF_NB_FL"]
    p["pru_deice_sh"] = (p["TF"] - p["VALID_FL"]) / p["TF"]
    p["pru_noref_sh"] = p["NO_REF"] / p["TF"]
    base = p[p["YEAR"] == 2025].groupby("APT_ICAO")["pru_pub"].mean()
    p["pru_pub_dev"] = p["pru_pub"] - p["APT_ICAO"].map(base)
    cols = ["pru_pub", "pru_ref", "pru_deice_sh", "pru_noref_sh", "pru_pub_dev"]
    return {f"{r.APT_ICAO}|{int(r.YEAR)}|{int(r.MONTH_NUM)}": {c: round(float(getattr(r, c)), 4) for c in cols}
            for r in p.itertuples()}


def input_urls() -> dict:
    """Signed URLs for every input of a feature-store kernel (data, METAR, ADS-B extracts)."""
    import sys
    sys.path.insert(0, str(SOL.parent / "tools"))  # tools/kaggle_env.py: Kaggle credentials
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
    from kaggle_env import get_api
    api = get_api("primary")
    out = {}
    with api.build_kaggle_client() as k:
        for ds in data_sets():
            owner, slug = ds.split("/")
            for f in api.dataset_list_files(ds).files:
                rq = ApiDownloadDatasetRequest()
                rq.owner_slug, rq.dataset_slug, rq.file_name = owner, slug, f.name
                resp = k.datasets.dataset_api_client.download_dataset(rq)
                out[f"{slug}/{f.name}"] = resp.history[0].headers["Location"]
    for ref in adsb_kernels():
        for name, url in fs_urls(ref.split("/")[1]).items():
            out[f"adsb/{ref.split('/')[1]}/{name}"] = url
    return out


def fs_urls(slug: str, owner: str = OWNER) -> dict:
    """Kernel-output URLs of a kernel; they need no credential (tokenised)."""
    import sys
    sys.path.insert(0, str(SOL.parent / "tools"))  # tools/kaggle_env.py: Kaggle credentials
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    from kaggle_env import account_for_owner, get_api
    api = get_api("primary" if owner == OWNER else account_for_owner(owner))
    with api.build_kaggle_client() as k:
        rq = ApiListKernelSessionOutputRequest()
        rq.user_name, rq.kernel_slug = owner, slug
        resp = k.kernels.kernels_api_client.list_kernel_session_output(rq)
    return {f.file_name: f.url for f in resp.files if f.file_name.endswith((".parquet", ".json"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["fs", "train", "batch", "fsbatch", "derive"])
    ap.add_argument("--specs", default="", help="batch: JSON file {name: spec}")
    ap.add_argument("slug")
    ap.add_argument("--fs", default="prc-fs1")
    ap.add_argument("--accel", default="cpu", choices=["cpu", "gpu", "tpu"])
    ap.add_argument("--spec", default="{}")
    ap.add_argument("--extra", nargs="*", default=[], help="extra modules appended after features.py")
    ap.add_argument("--extra-kernels", nargs="*", default=[])
    ap.add_argument("--owner", default=OWNER, help="miscusi: pull the feature store over URLs (internet on)")
    ap.add_argument("--fs-owner", default=OWNER, help="account that owns the feature-store kernel")
    ap.add_argument("--remote", action="store_true", help="fs: fetch every input over URLs even on the primary")
    ap.add_argument("--url-kernels", nargs="*", default=[], help="fs: OWNER/SLUG kernel outputs fetched over URLs")
    ap.add_argument("--opts", default="{}", help="fs/fsbatch: fs7 options JSON (see features.FS_OPTS)")
    a = ap.parse_args()
    OPTS.update(json.loads(a.opts))
    extra = [Path(p).resolve() for p in a.extra]
    if a.kind == "fs":
        remote = a.owner != OWNER or a.remote
        if remote:
            INPUTS.update(input_urls())
            for ref in a.url_kernels:
                own, slug = ref.split("/")
                for name, url in fs_urls(slug, own).items():
                    INPUTS[f"k/{slug}/{name}"] = url
        parts = [CORE / "features.py", *extra] + ([CORE / "remote_inputs.py"] if remote else []) + [CORE / "fs_main.py"]
        code = assemble(parts, None)
        datasets = [] if remote else data_sets() + ([WHEELS] if a.accel == "tpu" else [])
        d = write(a.slug, code, a.accel, datasets, a.extra_kernels if remote else adsb_kernels() + a.extra_kernels,
                  internet=remote, owner=a.owner)
    elif a.kind == "derive":
        code = assemble([CORE / "features.py", CORE / "fs8.py"], None)
        code = code.replace('BASE_STORE = ""  # replaced by build_kernel.py derive: the base store\'s kernel slug',
                            f"BASE_STORE = {a.fs!r}", 1)
        assert f"BASE_STORE = {a.fs!r}" in code
        d = write(a.slug, code, "cpu", data_sets(), [f"{a.fs_owner}/{a.fs}"], internet=False, owner=a.fs_owner)
    elif a.kind == "fsbatch":
        specs = json.loads(Path(a.specs).read_text())
        parts = [CORE / "features.py", *extra, CORE / "fs_main.py", CORE / "train_main.py", CORE / "batch_main.py",
                 CORE / "fsbatch_main.py"]
        code = assemble(parts, {}, specs)
        datasets = data_sets() + ([WHEELS] if a.accel == "tpu" else [])
        d = write(a.slug, code, a.accel, datasets, adsb_kernels() + a.extra_kernels, internet=True)
    elif a.kind == "batch":
        specs = json.loads(Path(a.specs).read_text())
        for s in specs.values():
            s.setdefault("fs", a.fs)
        remote = a.fs_owner != OWNER
        if remote:
            FS_URLS.update(fs_urls(a.fs, a.fs_owner))
        code = assemble([CORE / "features.py", *extra, CORE / "train_main.py", CORE / "batch_main.py"], {}, specs)
        datasets = [WHEELS] if a.accel == "tpu" else []
        kernels = ([] if remote else [f"{OWNER}/{a.fs}"]) + a.extra_kernels
        # TPU images lack xgboost/catboost: internet on lets ensure() pip-install them
        d = write(a.slug, code, a.accel, datasets, kernels, internet=remote or a.accel == "tpu")
    else:
        spec = json.loads(a.spec)
        spec.setdefault("fs", a.fs)
        # the store is mounted when the kernel runs on the account that built it, else fetched by URL
        remote = a.owner != a.fs_owner
        if remote:
            spec["fs_urls"] = fs_urls(a.fs, a.fs_owner)
        code = assemble([CORE / "features.py", *extra, CORE / "train_main.py"], spec)
        datasets = [WHEELS] if a.accel == "tpu" else []
        kernels = [] if remote else [f"{a.fs_owner}/{a.fs}"] + a.extra_kernels
        d = write(a.slug, code, a.accel, datasets, kernels, internet=remote or a.owner != OWNER, owner=a.owner)
    print(d)


if __name__ == "__main__":
    main()
