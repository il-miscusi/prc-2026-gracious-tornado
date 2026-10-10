"""Final-phase member refits: the v32c members, refitted on all of 2025 with banked weights, predicting
the final frame (final_ranking.parquet: Jan, Feb, Jun, Jul 2026).

    python3 solutions/final/build_members.py --list
    python3 solutions/final/build_members.py c7a c7b cs7 ...     # write kernel folders
    python3 research/kk.py [--account miscusi] push solutions/kernels/prc-f-<name>

Stores (solutions/core/build_kernel.py fs, opts rk_file=final_ranking.parquet):
  prc-f-fs6  plain features (the fs5/fs6 code path; fs5 members drop the fs6-only neighbour columns NB6)
  prc-f-fs7  fs7 options (one-to-one ADS-B matching, ICAO airline key, PRU, demeaned NM lags, OpenFlights)

Every spec below is the spec of the member that reached v32c (board v31, 234.12), copied from its refit
kernel (named in `src`), with four changes, the same for all members:
  * fs -> the final store;
  * save_models -> the fitted pair of every seed is pickled (output/models/refit_<k>.pkl);
  * a3_obs_p_evidence_start dropped: the store kernels that fed v32c stored an array index there
    (fixed in 62d3838a); in the final stores it is NaN in every row, dropped so no member depends on it;
  * seeds: more refit seeds where a member carries real weight (seed noise on the ranking side was
    17 s RMS with 3 seeds; findings_20261001).
Rounds are the fold-mean best iterations of the original A/B (or month-out) runs; the trainer refits at
1.1 x those, without early stopping.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parents[1]
BUILD = C / "solutions" / "core" / "build_kernel.py"
PRIMARY, FALLBACK = "usiadianimuwa", "miscusi"

NOPRU = ["pru_pub", "pru_ref", "pru_deice_sh", "pru_noref_sh", "pru_pub_dev", "apm_d_eobt1", "dm_d_eobt1",
         "apm_d_aobt3", "dm_d_aobt3", "apm_d_lobt", "dm_d_lobt", "apm_d_iobt", "dm_d_iobt", "apm_aobt3_m_eobt1",
         "dm_aobt3_m_eobt1"]
NB6 = [f"{n}_{s}{w}" for n, ws in (("nbe1", ("r600", "r900", "rb900")), ("nbx1", ("r600", "r900", "rb900")),
                                   ("nbx1a", ("a900",))) for w in ws for s in ("", "n_")]
EVID = ["a3_obs_p_evidence_start"]
CAT = {"task_type": "GPU", "l2_leaf_reg": 3, "border_count": 254, "one_hot_max_size": 4, "max_ctr_complexity": 1,
       "bootstrap_type": "Bernoulli", "subsample": 0.8, "gpu_ram_part": 0.9}
CAT_PIN = {"catboost": "1.2.8"}
MLP = {"hidden": [512, 256, 128], "dropout": 0.1, "lr": 0.002, "wd": 1e-05, "batch": 2048, "epochs": 14, "patience": 3}
CS7_ROUNDS = {"EDDF": [5316, 3014], "EDDM": [1860, 1316], "EGLL": [2518, 2434], "EHAM": [4330, 1242],
              "LEBL": [1002, 636], "LEMD": [1920, 2489], "LFPG": [1566, 746], "LIRF": [1670, 1054],
              "LSZH": [3146, 3199], "LTFM": [2082, 1410]}
CS6_ROUNDS = {"EDDF": [4504, 2808], "EDDM": [2772, 1489], "EGLL": [3052, 1674], "EHAM": [4400, 1426],
              "LEBL": [1176, 728], "LEMD": [2674, 1930], "LFPG": [1532, 896], "LIRF": [1856, 390],
              "LSZH": [3804, 642], "LTFM": [1872, 2110]}
CS5_ROUNDS = {"LSZH": [3620, 688], "EGLL": [2240, 1200], "LEMD": [2111, 2137], "LFPG": [1500, 1624],
              "LTFM": [1936, 3574], "EDDM": [2338, 1296], "EDDF": [3170, 2688], "LIRF": [1726, 326],
              "LEBL": [1299, 882], "EHAM": [3594, 4731]}


def c7(seed):
    return {"lib": "cat", "gpus": 1, "parallel": False, "folds": [], "rounds": [9600, 2900], "refit_seeds": 3,
            "params": {**CAT, "depth": 10, "learning_rate": 0.06, "random_seed": seed}, "params_long": {"depth": 8},
            "drop_features": NOPRU + EVID, "package_versions": CAT_PIN}


def mlp7(seeds):
    # one seed at a time: three parallel seeds ran out of host memory on the final frame (prc-f-mlp7a v1:
    # two of three killed, ~12 GB each; the final store predicts 670,790 rows, twice the old ranking)
    return {"lib": "mlp", "max_parallel": 1, "threads": 4, "drop_features": NOPRU + EVID,
            "variants": {f"r{s}": {"lib": "mlp", "refit": True, "folds": [], "rounds": [98, 55], "refit_seeds": 1,
                                   "save_models": True, "params": {**MLP, "seed": s},
                                   "params_long": {"batch": 1024, "epochs": 6}} for s in seeds}}


# name: (store, account, accelerator, spec, source kernel of the original spec, v32c role)
MEMBERS = {
    "c7a": ("prc-f-fs7", PRIMARY, "gpu", c7(3101), "prc-m-c7-r", "short .33, 1-4 h .26 (seeds 3101/4101/5101)"),
    "c7b": ("prc-f-fs7", PRIMARY, "gpu", c7(3151), "prc-m-c7-r", "c7 seeds 3151/4151/5151"),
    "cs7": ("prc-f-fs7", FALLBACK, "gpu",
            {"lib": "cat", "gpus": 2, "max_parallel": 2, "threads": 4, "parallel": True, "max_rounds": 8000,
             "split_by": "apt", "folds": [], "rounds": CS7_ROUNDS, "refit_seeds": 3,
             "params": {**CAT, "depth": 8, "learning_rate": 0.08, "random_seed": 3311}, "params_long": {"depth": 6},
             "drop_features": NOPRU + EVID, "package_versions": CAT_PIN},
            "prc-m-g7-r variant cs7r", "short .24 (per-airport CatBoost)"),
    "mlp7a": ("prc-f-fs7", FALLBACK, "gpu", mlp7([2911, 2912, 2913]), "prc-c-mlp7-r", "short .27, seeds 2911-2913"),
    "mlp7b": ("prc-f-fs7", FALLBACK, "gpu", mlp7([2914, 2915, 2916]), "prc-c-mlp7-r2", "mlp7 seeds 2914-2916"),
    "mlp7c": ("prc-f-fs7", FALLBACK, "gpu", mlp7([2911, 2913]), "prc-c-mlp7-r", "mlp7 seeds 2911/2913 (re-run of mlp7a's two OOM-killed seeds)"),
    "c6s": ("prc-f-fs6", PRIMARY, "gpu",
            {"lib": "cat", "gpus": 1, "parallel": False, "folds": [], "long_target": "sec", "rounds": [9379, 6612],
             "refit_seeds": 3, "params": {**CAT, "depth": 10, "learning_rate": 0.06, "random_seed": 2241},
             "params_long": {"depth": 8}, "drop_features": EVID, "package_versions": CAT_PIN},
            "prc-m-c6s-r", "1-4 h .25 (seconds-target long head)"),
    "c5s": ("prc-f-fs6", FALLBACK, "gpu",
            {"lib": "cat", "gpus": 1, "parallel": False, "folds": [], "long_target": "sec", "rounds": [9990, 4468],
             "refit_seeds": 3, "params": {**CAT, "depth": 10, "learning_rate": 0.06, "random_seed": 2233},
             "params_long": {"depth": 8}, "drop_features": NB6 + EVID, "package_versions": CAT_PIN},
            "prc-m-c5s-r3", "1-4 h .23 (fs5 = fs6 without the neighbour columns)"),
    "x5": ("prc-f-fs6", FALLBACK, "gpu",
           {"lib": "xgb", "gpus": 1, "parallel": False, "folds": [], "rounds": [1000, 1400], "refit_seeds": 3,
            "params": {"device": "cuda", "tree_method": "hist", "grow_policy": "lossguide", "max_leaves": 1023,
                       "max_depth": 0, "eta": 0.03, "min_child_weight": 30, "subsample": 0.8,
                       "colsample_bytree": 0.45, "lambda": 20, "max_bin": 256, "max_cat_to_onehot": 1,
                       "max_cat_threshold": 64, "seed": 2261},
            "params_long": {"max_leaves": 255}, "drop_features": NB6 + EVID,
            # xgboost 3.4.1 (Kaggle and Colab images, 10-08) breaks this member: 2025 month-out M2/M7 RMSE
            # 271.7/393.8 against 231.2/289.7 with 2.1.4 (c7: 229.6/277.1); findings_20261008.md
            "package_versions": {"xgboost": "2.1.4"}},
           "prc-p-x5-r", "1-4 h .19 (XGBoost)"),
    "k4": ("prc-f-fs6", PRIMARY, "cpu",
           {"lib": "lgb", "refit": True, "folds": [], "threads": 4, "rounds": [296, 195], "refit_seeds": 2,
            "drop_adsb": True, "drop_features": EVID,
            "params": {"objective": "regression", "learning_rate": 0.08, "num_leaves": 255, "min_data_in_leaf": 50,
                       "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
                       "cat_smooth": 20.0, "cat_l2": 10.0, "max_cat_threshold": 64, "min_data_per_group": 100,
                       "max_bin": 127, "seed": 7},
            "params_long": {"num_leaves": 127, "min_data_in_leaf": 30, "min_data_per_group": 30, "cat_smooth": 10.0}},
           "research/cover/spec_k4.json (miscusi/prc-cv-k4)", "LIRF uncovered overlay, weight .3"),
    "c6": ("prc-f-fs6", PRIMARY, "gpu",
           {"lib": "cat", "gpus": 1, "parallel": False, "folds": [], "rounds": [10000, 3500], "refit_seeds": 1,
            "params": {**CAT, "depth": 10, "learning_rate": 0.06, "random_seed": 2221}, "params_long": {"depth": 8},
            "drop_features": EVID, "package_versions": CAT_PIN},
           "prc-p-c6-r", "0 in short/1-4 h; in the >4 h long-row band"),
    "cs6": ("prc-f-fs6", FALLBACK, "gpu",
            {"lib": "cat", "gpus": 2, "max_parallel": 2, "threads": 4, "parallel": True, "max_rounds": 8000,
             "split_by": "apt", "folds": [], "rounds": CS6_ROUNDS, "refit_seeds": 3,
             "params": {**CAT, "depth": 8, "learning_rate": 0.08, "random_seed": 2611}, "params_long": {"depth": 6},
             "drop_features": EVID, "package_versions": CAT_PIN},
            "prc-p-cs6-r", "short .05"),
    "cs5": ("prc-f-fs6", FALLBACK, "gpu",
            {"lib": "cat", "gpus": 2, "max_parallel": 2, "threads": 4, "parallel": True, "max_rounds": 8000,
             "split_by": "apt", "folds": [], "rounds": CS5_ROUNDS, "refit_seeds": 2,
             "params": {**CAT, "depth": 8, "learning_rate": 0.08, "random_seed": 2311}, "params_long": {"depth": 6},
             "drop_features": NB6 + EVID, "package_versions": CAT_PIN},
            "prc-m-cs5-r", "short .05, 1-4 h .04, long band"),
    "mlp6fs": ("prc-f-fs6", PRIMARY, "cpu",
               {"lib": "mlp", "parallel": False, "folds": [], "rounds": [92, 57], "refit_seeds": 1, "threads": 4,
                "params": {**MLP, "seed": 2901}, "params_long": {"batch": 1024, "epochs": 6}, "drop_features": EVID},
               "prc-m-mlp6fs-r1", "short .04"),
}


def reseed(name: str, base: int) -> tuple:
    """A second seed set of a 1-4 h / long-stack member: the same spec and store with another base seed (the long
    rows are a seed lottery, so these members are averaged over six seeds as c7 is)."""
    store, owner, accel, spec, src, role = MEMBERS[name]
    key = "seed" if spec["lib"] == "xgb" else "random_seed"
    return (store, owner, accel, {**spec, "params": {**spec["params"], key: base}}, src,
            f"{name} seeds {base}/{base + 1000}/{base + 2000}")


MEMBERS["c6sb"] = reseed("c6s", 2251)
MEMBERS["c5sb"] = reseed("c5s", 2243)
MEMBERS["x5b"] = reseed("x5", 2271)


def fold_spec(spec: dict, folds: list[str]) -> dict:
    """Whole-month outer folds of a member (the 2025 pseudo-final of the four final months): train on
    2025 minus the month at the member's fixed rounds (no early stopping on the scored month), predict it.
    One seed; an MLP takes its first variant's seed."""
    if "variants" in spec:
        v = next(iter(spec["variants"].values()))
        spec = {**{k: x for k, x in spec.items() if k not in ("variants", "max_parallel")}, **v}
    rounds = spec["rounds"]
    spec = {k: x for k, x in spec.items() if k not in ("rounds", "refit_seeds", "save_models")}
    return {**spec, "folds": folds, "refit": False, "fold_rounds": {f: rounds for f in folds},
            "parallel": False, "max_parallel": 1, "gpus": 1 if spec.get("lib") != "lgb" else 0}


def build(name: str, folds: list[str] | None = None, owner_override: str | None = None) -> Path:
    store, owner, accel, spec, src, role = MEMBERS[name]
    owner = owner_override or owner  # where GPU sessions are free; the spec does not change
    if folds:
        spec = fold_spec(spec, folds)
        owner = owner_override or PRIMARY  # primary: run on Colab with FS_URLS_FILE (research/colab_run.py)
    else:
        spec = {**spec, "save_models": True}
    slug = f"prc-f-{name}" + ("-mo" if folds else "")
    cmd = [sys.executable, str(BUILD), "train", slug, "--fs", store, "--fs-owner", PRIMARY, "--owner", owner,
           "--accel", accel, "--spec", json.dumps(spec)]
    out = subprocess.run(cmd, cwd=C, capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(out.stderr)
    folder = C / "solutions" / "kernels" / slug
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    meta["enable_internet"] = True  # package pins (catboost 1.2.8) install over pip when the image differs
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    (folder / "member.json").write_text(json.dumps({"member": name, "store": store, "owner": owner, "accel": accel,
                                                    "folds": folds or [], "source_spec": src, "v32c_role": role,
                                                    "spec": spec}, indent=1))
    return folder


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--folds", default="", help="e.g. M1,M2,M6,M7: month-out fold kernel prc-f-<name>-mo")
    ap.add_argument("--owner", default=None, help="run on this Kaggle account instead of the table's")
    a = ap.parse_args()
    if a.list or not a.names:
        for n, (store, owner, accel, spec, src, role) in MEMBERS.items():
            print(f"{n:7s} {store:10s} {owner:14s} {accel:4s} seeds={spec.get('refit_seeds', len(spec.get('variants', {})))}  {role}")
    for n in a.names:
        print(build(n, [f for f in a.folds.split(",") if f] or None, a.owner))
