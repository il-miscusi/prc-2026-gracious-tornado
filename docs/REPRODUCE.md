# Reproduce

Commands run from the repository root. The heavy stages run as Kaggle script kernels: the builders
here write a kernel folder (`kernel.py` + `kernel-metadata.json`) under `solutions/kernels/` or
`solutions/adsb/`, and the folder is pushed with `python3 research/kk.py push FOLDER` or the
official `kaggle kernels push -p FOLDER`. Assembly, rehearsal and the checks run locally on one CPU.

The organisers do not re-run submissions. This document is for a reviewer who wants to check a
step, and for us to rebuild the final from stored inputs.

## What was checked

On 2026-10-10 the publication-tree code, with retained private inputs at the paths below,
rebuilt f6's final and board files byte for byte (`analysis/results/f6_published_replay.json`).
The earlier f5 rehearsal and meta-model checks were performed on 2026-10-09.
Assembly replay starts from member predictions; it does not certify a fresh retraining run.

| what | command | result |
|---|---|---|
| f6 final and board files | `python3 research/final/assemble_final.py research/final/configs/f6_final.json --out-dir OUT` | final sha256 `602c05a26e665b4136dfe08a712c8ce2f685a08303f32dc3334e132692b83bf3`; board `5785d8b84ce99528666d0fd22cadd25670cbc42388798be1066f60d27862c5ed`; both match `research/final/manifests/f6.json`; bounds and file checks pass |
| 2025 rehearsal | `python3 research/final/pseudo_final.py run research/final/configs/f5_pseudo2025.json --weights frozen` | per-row file sha256 `dfbb4925…`, as `research/final/pseudo/f5_pseudo2025_frozen.json` records; 5 min 31 s |
| meta-model rows | `python3 research/final/meta/reproduce.py --run solutions/kernels/prc-meta-gbm-f4/output --frames submissions/final/meta/f4` | the 596,663 d ≤ 1 h rows regenerated from the saved booster, largest difference 3.6 × 10⁻¹² s (`analysis/results/meta_reproduce_f4.txt`) |
| stack weights | `python3 research/final/pseudo_final.py fit-weights research/final/configs/f4_pseudo2025.json` | the d ≤ 1 h base and 1–4 h weights of `f5_final.json`, to all six decimals |

## Setup

- Python 3.13 and `pip install -r requirements.txt` for the local stages; LightGBM 4.6.0 for
  `research/final/meta/reproduce.py`.
- The organiser files in `data/` as listed in `docs/DATA.md`.
- The EUROCONTROL files: `research/pru/fetch_pru.sh` (downloads and checks their sha256).
- For the Kaggle stages: Kaggle credentials (`~/.kaggle`, or `KAGGLE_*` variables) for the
  account that owns the datasets. A second, compute-only account is optional; give it as a bearer
  token in a JSON file named by `$PRC_KAGGLE_ACCOUNTS` (`tools/kaggle_env.py`).
- Account names are constants in the builders (`OWNER`, `PRIMARY`, `FALLBACK` in
  `solutions/core/build_kernel.py`, `solutions/final/build_members.py`, `solutions/adsb/build_k3.py`
  and `build_final.py`: `usiadianimuwa` and `miscusi`). To run under another account, change those
  constants and publish the input datasets under that account.

## Stages

**Inputs and stores.**

| # | stage | command | runs on | outputs |
|---|---|---|---|---|
| 0a | EUROCONTROL files | `research/pru/fetch_pru.sh` | local | `research/pru/*.csv`, hashes checked |
| 0b | METAR | `python3 research/build_metar.py OUT` | local | METAR table → private dataset `prc-taxi-2026-metar` |
| 0c | geometry | `python3 research/geometry/build_geometry.py` | local | `research/geometry/*.parquet` → private dataset `prc-taxi-2026-geometry` |
| 0d | organiser data | `kaggle datasets create -p <dir>` | local | private datasets `prc-taxi-2026-data` (2025 and board files), `prc-taxi-2026-final` (final files) |
| 1a | ADS-B v3 stays | `python3 solutions/adsb/build_k3.py 202501 ... 202607`; `python3 solutions/adsb/build_final.py` | Kaggle CPU, internet | `adsb3_points_*`, `adsb3_stays_*` per month |
| 1b | ADS-B v3 events | `python3 solutions/adsb/build_k3.py events` (2025, Jan/Jul 2026); `k3_events_f` from `build_final.py` (Feb/Jun 2026) | Kaggle CPU | `adsb3_events_<YYYYMM>.parquet` |
| 1c | ADS-B v1 departures | `kf_*` folders from `build_final.py` (Feb/Jun 2026); the 2025 and Jan/Jul 2026 kernels in `solutions/adsb/sources.json` are `extract.py` with `DATES` set the same way (`build_final.build_v1`) | Kaggle CPU, internet | `adsb_departures_*.parquet` |
| 2a | stores fs6, fs7 | `python3 solutions/core/build_kernel.py fs prc-f-fs7 --opts '{"match": "v7", "airline": "icao", "pru": true, "demean": true, "openflights": true, "rk_file": "final_ranking.parquet"}'` and `python3 solutions/core/build_kernel.py fs prc-f-fs6 --opts '{"rk_file": "final_ranking.parquet"}'` | Kaggle CPU | `tr_base.parquet`, `rk_base.parquet`, encodings per fold |
| 2b | store fs8 | `python3 solutions/core/build_kernel.py derive prc-f-fs8 --fs prc-f-fs7 --opts '{"rk_file": "final_ranking.parquet"}'` | Kaggle CPU | fs7's files with 64 `g8_*` columns appended (`solutions/core/fs8.py`) |
| 3 | assembler inputs | push `kernels/prc-f-inputs/kernel.py` with its metadata | Kaggle CPU | `phys_*.parquet`, `cover_*.parquet` → `research/final/inputs/{final,board,2025}/` |

**Members.**

| # | stage | command | runs on | outputs |
|---|---|---|---|---|
| 4a | refits | `python3 solutions/final/build_members.py c7a c7b cs7 mlp7a mlp7b mlp7c c6s c6sb c5s c5sb x5 x5b k4` and `python3 research/fgap/build_fs8_members.py c7a c7b`, then push each folder | GPU (T4), k4 CPU | `submission.parquet`, `refit_<k>.npy`, `models/refit_<k>.pkl`, `environment.json` |
| 4b | month-out folds | `python3 solutions/final/build_members.py --folds M1,M2,M6,M7 c7a cs7 mlp7a c6s c5s x5 k4` and `python3 research/fgap/build_fs8_members.py c7a --folds M1,M2,M6,M7`, then push | GPU (T4), k4 CPU | `oof.csv` per member |
| 4c | fs8 c7 folds | `python3 research/fgap/combine_c7_mo.py` | local | `prc-f8-c7a-mo/output/oof.csv` from the four fold runs, with `oof_sources.json` |
| 5 | seed merges | `python3 research/final/merge_members.py c7`; `python3 research/final/merge_members.py --out solutions/kernels/prc-f-mlp7m6/output prc-f-mlp7a prc-f-mlp7c prc-f-mlp7b`; the same `--out` form for `prc-f-c6sm6` (`prc-f-c6s prc-f-c6sb`), `prc-f-c5sm6` (`prc-f-c5s prc-f-c5sb`), `prc-f-x5m6` (`prc-f-x5 prc-f-x5b`) and `prc-f8-c7` (`prc-f8-c7a prc-f8-c7b`) | local | merged `submission.parquet` + `merge_manifest.json` per member |

The February fold of fs8 c7 came from an earlier paired-test kernel that read version 1 of the
fs8 store; `combine_c7_mo.py` documents each fold's source. The two versions differ only in four
stand-traffic columns, which version 2 divides by 12 on the 2025 frame; that rescaling leaves
CatBoost's quantile borders, and so the model, unchanged.

**Fits, rehearsal passes and assembly.** Each recipe's L1a-v2 loadings are fitted on its own
rehearsal rows, so the rehearsals run in passes.

| # | stage | command | outputs |
|---|---|---|---|
| 6 | daily-layer parameters | `python3 research/final/levers/daily.py artifact` | `research/final/levers/artifacts/daily_params.json` |
| 7 | stack weights | `python3 research/final/pseudo_final.py fit-weights research/final/configs/f4_pseudo2025.json` | the weights frozen in `f4_final.json` and `f5_final.json` |
| 8a | rehearsal of f2x5 (rows for the first pass) | `python3 research/final/pseudo_final.py run research/final/configs/f2x5_pseudo2025.json --weights frozen` (and `crossfit`) | `submissions/final/pseudo2025/f2x5_pseudo2025_*.parquet` |
| 8b | f4, first pass | `pseudo_final.py run research/final/configs/f4p1_pseudo2025.json` (both modes) | f4's rows before the daily layer |
| 8c | f4's loadings | `python3 research/final/levers/daily.py v2 --rows submissions/final/pseudo2025/f4p1_pseudo2025_frozen.parquet --col post2_recovered --out research/final/artifacts/daily_v2_f4_loadings.json` | the artifact `f4_final.json` reads |
| 8d | f4 rehearsal | `pseudo_final.py run research/final/configs/f4_pseudo2025.json` (both modes) | f4's scores |
| 9a | meta-model frames | `python3 research/final/meta/build_frame.py --spec research/final/meta/specs/f4.json --out-dir submissions/final/meta/f4` and the same with `--frame final` | `meta2025.parquet`, `meta2026.parquet`, `meta_spec.json` |
| 9b | meta-model dataset | `python3 research/final/meta/publish_dataset.py create --slug prc-taxi-2026-meta-f4 --dir submissions/final/meta/f4` | private dataset `prc-taxi-2026-meta-f4`, file hashes in `research/final/meta/results/` |
| 9c | meta-model kernel | push `research/final/meta/kernels/prc-meta-gbm-f4`, copy its output to `solutions/kernels/prc-meta-gbm-f4/output/` | `final_G1.parquet` (2026 d ≤ 1 h rows), `mo_G1.csv` (month-out), `G1_short.txt` (booster), `report.json` |
| 9d | f5, first pass | `python3 research/final/meta/harness.py run research/final/meta/configs/meta_gbm-G1short-f4p1_pseudo2025.json --weights frozen` (and `crossfit`) | f5's rows before the daily layer, in `submissions/final/meta/pseudo2025/` |
| 9e | f5's loadings | `python3 research/final/levers/daily.py v2 --rows submissions/final/meta/pseudo2025/meta_gbm-G1short-f4p1_frozen.parquet --col post2_recovered --out research/final/meta/artifacts/daily_v2_f5_loadings.json` | the artifact `f5_final.json` reads |
| 10a | assembly | `python3 research/final/assemble_final.py research/final/configs/f5_final.json` | `submissions/final/f5_final.parquet`, `f5_board.parquet`, `f5_stages.parquet`, manifest |
| 10b | f5 rehearsal | `python3 research/final/pseudo_final.py run research/final/configs/f5_pseudo2025.json --weights crossfit` (and `frozen`) | per-row file and summary |
| 10c | ablations | `pseudo_final.py compare A B` for each pair in `docs/VALIDATION.md`; `python3 research/final/meta/compare_all.py meta_gbm-G1short-f4 --base f4_pseudo2025 --modes crossfit` (with f4's rows copied into `submissions/final/meta/pseudo2025/`); `python3 analysis/pseudo_steps.py . f5_pseudo2025_crossfit` | `research/final/pseudo/`, `research/final/meta/pseudo/`, `analysis/results/` |
| 11a | event-layer artifact | `python3 research/final/levers/events.py artifact` | airport offsets and blend weights, `research/final/levers/artifacts/events_f5.json` (from f5 frozen rehearsal rows) |
| 11b | f6 rehearsal | `python3 research/final/pseudo_final.py run research/final/configs/f6ev_pseudo2025.json --weights crossfit` (and `frozen`) | `f6ev_pseudo2025_*.parquet`; the events layer refits without the scored month |
| 11c | f6 assembly | `python3 research/final/assemble_final.py research/final/configs/f6_final.json` | `f6_final.parquet`, `f6_board.parquet`, stage file and manifest |
| 12 | file check | `python3 research/upload.py check-final submissions/final/f6_final.parquet` | pass/fail; the team uploads once with `put-final` |

The 2026 assembly reads only fitted artifacts; the rehearsal configs name the earlier pass's rows
under `post_layer_pseudo`. The meta member in `f5_final.json` lists its upstream files under
`inputs` (kernel output, the dataset's files, the frame and kernel code), and the assembler hashes
them into the manifest.

## Kernels of the final run

`kernels/<slug>/` holds, for every kernel the published configs read, its `kernel-metadata.json`
(account, accelerator, mounted datasets and kernels), `member.json` (the exact training spec) and
small JSON files of its output (`environment.json`, `identity_refit.json`, `merge_manifest.json`,
`oof_sources.json`). Generated `kernel.py` files are not committed: a kernel that runs on the
second account receives signed download URLs for its inputs inside `kernel.py`, so the generators
are published instead and rebuild the folder. The exceptions are `kernels/prc-f-inputs/kernel.py`
and the meta-model kernels in `research/final/meta/kernels/`, which no generator writes and which
hold no URL.

Some kernels ran on Google Colab instead of Kaggle when Kaggle GPU sessions were full. The kernel
folder is the same; our Colab runner is not part of this repository. `output/environment.json`
records the host.

| kernel | account | ran on | role |
|---|---|---|---|
| prc-f-fs6, prc-f-fs7, prc-f-fs8 | usiadianimuwa | Kaggle CPU | feature stores on `final_ranking.parquet` |
| prc-f-inputs | usiadianimuwa | Kaggle CPU | assembler inputs |
| prc-f-c7a | usiadianimuwa | Colab | c7, seeds 3101/4101/5101 |
| prc-f-c7b | miscusi | Kaggle T4 | c7, seeds 3151/4151/5151 |
| prc-f8-c7a, prc-f8-c7b | miscusi | Kaggle T4 | fs8 c7, same seeds as c7a and c7b |
| prc-f-cs7 | miscusi | Kaggle T4 | cs7, seeds 3311/4311/5311 |
| prc-f-mlp7a, prc-f-mlp7b | miscusi | Kaggle T4 | mlp7 seed 2912; seeds 2914–2916 |
| prc-f-mlp7c | usiadianimuwa | Kaggle T4 | mlp7 seeds 2911, 2913 |
| prc-f-c6s, prc-f-x5, prc-f-x5b | miscusi | Kaggle T4 | c6s seeds 2241/3241/4241; x5 (XGBoost 2.1.4) seeds 2261/3261/4261 and 2271/3271/4271 |
| prc-f-c6sb | usiadianimuwa | Colab | c6s seeds 2251/3251/4251 |
| prc-f-c5s | miscusi | Colab | c5s seeds 2233/3233/4233 |
| prc-f-c5sb | miscusi | Colab | c5s seeds 2243/3243/4243 |
| prc-f-k4 | usiadianimuwa | Kaggle CPU | no-ADS-B overlay member, 2 seeds |
| prc-f-c7a-mo, prc-f-cs7-mo, prc-f-mlp7a-mo, prc-f-c5s-mo, prc-f-x5-mo | usiadianimuwa | Colab | month-out folds for the rehearsal |
| prc-f-c6s-mo, prc-f8-c7a-mo | miscusi | Kaggle T4 | month-out folds |
| prc-f-k4-mo | usiadianimuwa | Kaggle CPU | month-out folds |
| prc-meta-gbm-f4 (code in `research/final/meta/kernels/`) | usiadianimuwa | Kaggle CPU | G1short meta-model, about 4 minutes |

## Trained weights and access

Every refit writes each seed's fitted model to `output/models/refit_<k>.pkl`
(`save_models` in `solutions/core/train_main.py`), together with its spec, parameters, column
order and categorical columns. A kernel with `predict_only` reloads them and predicts a store
without training.

Where the models are kept, until the organisers' review closes: **39 of
42 selected model files have verified private banks**.
The count includes the meta booster as one model. Full file checksums, archive checksums, dataset
versions and readback times are in
[retained_inputs_20261010.json](../analysis/results/retained_inputs_20261010.json).
Only files actually present in an archive count; a results file listing an absent seed does not.

| private dataset | selected models held | integrity |
|---|---|---|
| `usiadianimuwa/prc-f-c5s-bank-20261009` | 3 | verified |
| `usiadianimuwa/prc-f-c5sb-bank-20261009` | 3 | verified |
| `usiadianimuwa/prc-f-c6s-bank-20261010` | 3 | verified |
| `usiadianimuwa/prc-f-c6sb-bank-20261009` | 3 | verified |
| `usiadianimuwa/prc-f-c7a-bank-20261009` | 3 | verified |
| `usiadianimuwa/prc-f-c7b-bank-20261009` | 3 | verified |
| `usiadianimuwa/prc-f-cs7-bank-20261009` | 3 | verified |
| `usiadianimuwa/prc-f-k4-bank-20261010` | 2 | verified |
| `usiadianimuwa/prc-f-mlp7a-r2912-bank-20261009` | 1 | verified |
| `usiadianimuwa/prc-f-mlp7b-bank-20261009` | 3 | verified |
| `usiadianimuwa/prc-f-mlp7c-bank-20261009` | 2 | verified |
| `usiadianimuwa/prc-f-x5-seeds01-bank-20261010` | 2 | verified |
| `usiadianimuwa/prc-f-x5b-seed0-bank-20261010` | 1 | verified |
| `usiadianimuwa/prc-f8-c7a-bank-20261010` | 3 | verified |
| `usiadianimuwa/prc-f8-c7b-bank-20261010` | 3 | verified |
| `usiadianimuwa/prc-meta-gbm-f4-bank-20261010` | 1 | verified |

The x5 banks are partial: prc-f-x5 seeds 0 and 1 and prc-f-x5b seed 0 are held there.
prc-f-x5 seed 2 and prc-f-x5b seeds 1 and 2 still rely on their private kernel outputs.
Each weights bank holds an archive plus input_closure.json, with file checksums and the code
commit. A partial bank does not establish retention of the whole member.

The direct f6 assembly inputs, source code and golden files are retained together in the verified
private dataset usiadianimuwa/prc-f6-assembly-closure-20261010: 71 files, archive SHA256
122e2ed2eed033eaadfc4bf251458da103ae39fa124272b6af4836d84127e3b5.
This is enough to replay assembly from predictions; it does not establish the full training closure.

- `usiadianimuwa/prc-f-fs6-bank-20261010`: private version 1, 8 files, every file read back and SHA256 verified.
- `usiadianimuwa/prc-f-fs7-bank-20261010`: private version 1, 8 files, every file read back and SHA256 verified.
- `usiadianimuwa/prc-f-fs8-bank-20261010`: private version 1, 8 files, every file read back and SHA256 verified.
- `usiadianimuwa/prc-f-inputs-bank-20261010`: private version 1, 5 files, every file read back and SHA256 verified.

The fs6/fs7 refit identities also record Kaggle's generated __output__.json; the native
feature-store copies omit that metadata file, so their copies do not close that audit item.
Legacy A/B and month-out trained-model retention, original ADS-B daily release asset receipts,
extraction-output banks, and exact producing kernel versions remain incomplete. A fresh complete
training drill has not been certified. Publishing the code does not close these gaps.

The weights, feature stores, member predictions, meta-model frames and intermediate tables are
private because they contain the organisers' data and ODbL-derived tables. A reviewer who wants
them can open an issue in this repository or ask the organisers to forward the request; we share
the private Kaggle datasets and kernels listed here with the reviewer's Kaggle account for the
duration of the review.

## Environment

- Local stages: `requirements.txt` (Python 3.13.7; numpy 2.3.5, pandas 2.3.3, scipy 1.18.1,
  pyarrow 23.0.1, as recorded in every assembler manifest); LightGBM 4.6.0 for the meta-model
  rebuild.
- Kernels: the Kaggle Python image of 2026-10-08 (Python 3.13.15, numpy 2.1.3, pandas 2.3.3,
  scipy 1.16.3, pyarrow 23.0.1, LightGBM 4.6.0, torch 2.11.0). The member specs pin CatBoost 1.2.8
  and, for x5, XGBoost 2.1.4 (`package_versions`, installed over pip when the image differs; the
  image's own XGBoost, 3.4.1, gave a different and worse model). The Colab runs had pandas 2.2.3.
  The meta-model kernel runs LightGBM with `deterministic=True` and seed 7. Each kernel writes the
  versions it ran with to `output/environment.json`, copied under `kernels/`.

## Expected hashes

| artefact | sha256 |
|---|---|
| organiser files, EUROCONTROL files | `docs/DATA.md` |
| `research/cache/oof_mo12_c6.parquet` (c6 month-out predictions; level offset and daily-layer fit) | `e244b219c925740f89aeece51df3a6a6ef367d9c5b22a59aaa5be558763b9dc0` |
| `research/cache/oof_meta_v2.parquet` (A/B fold metadata) | `0010c39e9d2a978148159f1abb7d9a5f8148f6ad41083674e151660fca07cf0e` |
| `research/tail/cache/long_tr.parquet` (`research/tail/build_cache.py`) | `ef6829fa9f372f93f00eec2ef6a7210ee46216c8d281a9c8c052701478eb5228` |
| `research/final/inputs/final/phys.parquet` | `16a8b24d794b0fc3008c6d350aee65659292ea8924f9b66b363c5a06cf74d01e` |
| `research/final/inputs/final/cover.parquet` | `9f53d71f1c16770370eda792df296b8f1164f0deadf7d6ecc8b4c5466084f423` |
| member predictions (`submission.parquet` of each merged member, `final_G1.parquet`) | `inputs` of `research/final/manifests/f6.json` |
| `f6_final.parquet` | `602c05a26e665b4136dfe08a712c8ce2f685a08303f32dc3334e132692b83bf3` |
| `gracious-tornado_final.parquet` | Unique final upload pending; it must match the f6 hash above, checked by bucket readback |

`research/final/manifests/<name>.json` is the assembler's record of a run: the sha256 of every
input, output and code file, the config, the weights and coefficients used, the row counts of every
step and the package versions.

Some published files differ from the manifests' record. `tools/sync_from_compbot.py` made the
out-of-fold path recorded inside the daily-layer parameter files
(`research/final/levers/artifacts/daily_params.json`, `daily_params_c6raw.json`,
`research/final/artifacts/daily_params_f0x_pseudo.json`) relative to the repository root, which
`daily.py` does not read, and it points `research/final/meta/publish_dataset.py` (one of the meta
member's declared inputs) at `tools/kaggle_env.py` for Kaggle credentials. The rebuilds above used
the published files. `SYNC_MANIFEST.json` lists both sha256s of every synced file.

## Runtimes

| step | time |
|---|---|
| assembly of `f5_final.json` | 82 s on one CPU (2026-10-09) |
| assembly of `v32c_repro_board.json` (no post layers) | about 10 s |
| one rehearsal run of `f5_pseudo2025.json` (four months, one mode) | 5 min 31 s on one CPU (2026-10-09) |
| meta-model kernel | about 4 min (Kaggle CPU) |
| ADS-B extraction | about 5 min per day per kernel; a month split over 3–5 parallel CPU kernels takes 30–60 min |
| feature stores fs6, fs7 on the final file | about 0.33 h each (Kaggle CPU) |
| daily-layer fit | about 30 s per airport-year on one core |

Member refits took between about 15 minutes per seed (c7 on a Colab L4) and 1.4 hours for three
seeds (c6s on a Kaggle T4); each kernel's log in its output has the exact time.

## Limits met on Kaggle

- Per account at once: 2 GPU sessions, 5 CPU sessions, 1 TPU session. A refused push for session
  count is not a ban; wait for a free slot.
- An HTTP 429 on a kernel push is an account-level ban with a Retry-After of days. Never retry a
  push in a loop; `research/kk.py` records the ban and refuses later pushes from that account.
- Two CatBoost jobs on the fs7 store (282 columns) exceed a Kaggle GPU host's 32.9 GB, and three
  MLP seeds in parallel on the final frame ran out of memory: run one per session.
