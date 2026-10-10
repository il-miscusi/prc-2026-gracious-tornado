# PRC Data Challenge 2026: taxi-out time, team gracious-tornado

This repository holds the code and documentation behind team gracious-tornado's final submission
to the PRC Data Challenge 2026: the taxi-out time of every departure at ten European airports
(EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM) in January, February, June and July
2026, 670,790 rows, scored by RMSE. The code is licensed under the GNU GPL v3 (`LICENSE`); data
attributions are in `NOTICE.md`.

**Status.** The selected final config is `research/final/configs/f6_final.json`. It builds
`f6_final.parquet`, sha256 `602c05a26e665b4136dfe08a712c8ce2f685a08303f32dc3334e132692b83bf3`.
The file has passed assembly checks and byte replay; the unique final bucket upload is pending.
The public code and documentation are due by 2026-10-14 10:00 UTC.

| document | contents |
|---|---|
| this README | the problem, the model in brief, results, quick start |
| [`docs/METHOD.md`](docs/METHOD.md) | every step of the pipeline, the evidence for it, the results we chose not to ship |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | the 2025 rehearsal of the final, step and paired ablations, each number with its file |
| [`docs/DATA.md`](docs/DATA.md) | every input: what it is, how it is fetched, its hash, its licence |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | where each step and feature came from: own work, papers, or other teams' public repositories with their licences |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | commands from the stored inputs to the final file, kernels, weights, hashes, checked rebuilds |
| [`research/final/README.md`](research/final/README.md) | the assembler step by step, its checks and outputs |

## The problem

The target is y = takeoff − off-block, in seconds. Takeoff and the scheduled off-block time are
supplied, so the delay to takeoff, d = takeoff − schedule, is known for every row, and
y = d − δ with δ = off-block − schedule hidden. For most departures d is under an hour and y is
taxiing plus queueing. For a delayed departure y depends on how the off-block was recorded: a flight
held at the gate has an ordinary y, while one recorded at its schedule time has y close to d. Under
squared error these recording modes carry much of the error, and they are concentrated at LIRF
(`docs/METHOD.md`, section 1).

## The model

One run of `research/final/assemble_final.py` turns member predictions into the file. Each step is
a key in the config.

1. **Ordinary taxi model.** Gradient-boosted trees and an MLP refit on all 2025 labels, on three
   feature stores with schedule and NM timing, stand and runway encodings, geometry, METAR,
   traffic, graded adsb.lol off-block evidence, and (store fs8) plan revisions, live lateness,
   queues and stand context. For d ≤ 1 h a LightGBM model of the residual of a member stack
   (G1short); for 1–4 h a non-negative least-squares stack fitted on month-out predictions; past
   4 h a stack blended with a physical estimate. LIRF rows with d ≤ 4 h and no ADS-B take 30% of a
   member trained without ADS-B. Predictions past 2 h are capped at d.
2. **Bounds from how the data was built.** The organisers joined an NM flight only when
   |off-block − LOBT| ≤ 3,606 s, so a matched departure's y lies within takeoff − LOBT ± 3,606 s
   (±3,631 s with the final file's ±12 s dither). LEMD and LEBL never exceeded 3,604 s in 2025. The
   same window applies to departures whose NM flight is recovered from the destination's arrival
   record (LIRF, LTFM).
3. **Reconciliation with EUROCONTROL's aggregates.** Each airport-month is moved 0.75 of the way to
   EUROCONTROL's published mean taxi-out time, after a composition offset fitted on 2025. Then each
   airport-day is reconciled with EUROCONTROL's daily pre-departure delay, which on 2025 labels
   equals Σ max(0, d − y) over the day's departures: every prediction is replaced by its posterior
   mean given that day total.
4. **Grade-A pushback blend.** After reconciliation, blend toward the observed stand-leave time
   plus an airport offset. Offsets and blend weights by disagreement bin are fitted on 2025
   rehearsal rows. Only grade-A events move; large disagreements receive zero weight. Reapply
   the bounds and floor afterward.

## Results

The 2025 rehearsal runs the whole pipeline on January, February, June and July 2025 with members
trained without the scored month (`docs/VALIDATION.md`). RMSE in seconds; stack weights refitted
without the scored month (crossfit). "nw" leaves out 8 date-wrap rows, LIRF records whose y lies
near a whole number of days.

| | Jan | Feb | Jun | Jul | Jan + Jul | Feb + Jun | all four |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 rehearsal, all rows | 322.84 | 219.28 | 277.39 | 261.33 | 290.40 | 253.48 | 273.05 |
| 2025 rehearsal, nw | 315.37 | 219.28 | 207.33 | 242.28 | 277.29 | 212.67 | 247.94 |
| 2026, public board (Jan/Jul slice of f6) | | | | | 233.1717 | | |

Sources: `research/final/pseudo/f6ev_pseudo2025_crossfit.json` and
`analysis/results/pseudo_steps_f6ev_pseudo2025_crossfit.json`. In pooled MSE units (nw, crossfit),
the store fs8 member added −240.5 and the d ≤ 1 h meta-model −49.0 to the previous recipe; within
the pipeline, the cap adds −190.3, the monthly level −59.2, rules (a) and (d) −42.1, the
recovered-orphan window −160.6, the daily layer −3,095.7 and the grade-A blend −50.6
[−62.3, −38.2]. About two thirds of the meta-model's
gain is LTFM in February 2025 (without those rows: −17.6 [−45.3, +11.5]). The full ablation is in
`docs/VALIDATION.md`.

## Quick start

```bash
pip install -r requirements.txt
research/pru/fetch_pru.sh                     # EUROCONTROL files, hashes checked
# organiser files into data/ (docs/DATA.md); member outputs and assembler inputs at the paths of docs/REPRODUCE.md
python3 research/final/assemble_final.py research/final/configs/f6_final.json
python3 research/upload.py check-final submissions/final/f6_final.parquet
python3 research/final/pseudo_final.py run research/final/configs/f6ev_pseudo2025.json --weights crossfit
```

The member predictions, models, meta-model frames and feature stores are private Kaggle outputs,
because they contain the organisers' data and ODbL-derived tables; `docs/REPRODUCE.md` lists them
and says how a reviewer can get access. The evidence scripts in `analysis/` need only the organiser
files and the stored inputs named in their headers.

## Data and licences

Organiser files (not redistributed; no licence stated), adsb.lol ADS-B history (ODbL 1.0),
OpenStreetMap (ODbL 1.0), OpenFlights airline aliases (ODbL 1.0), OurAirports runways (public
domain), X-Plane Scenery Gateway layouts (GPL), Iowa Environmental Mesonet METAR (public domain),
and EUROCONTROL's monthly taxi-out and daily pre-departure delay files (EUROCONTROL terms: source
named, non-commercial, no modification). The organiser ruled the monthly EUROCONTROL file usable
for the scored months on 2026-10-08; no ruling names the daily file. Details, hashes and the
rulings are in `docs/DATA.md`. No OpenSky Trino data, no commercial flight data and no pretrained
model are used.

## Limitations

- **NM clock regimes.** At LTFM and LIRF, takeoff − AOBT_3 sits at a near-constant default, so
  AOBT_3 is not an off-block measurement there, and LTFM changed level between months in both
  2025 and 2026 (`analysis/evidence/eda_lags.py`). The monthly level step is the guard against a
  regime that moved.
- **ADS-B coverage.** Almost no LTFM departure has a matched adsb.lol event (0–0.1% in every
  month), and almost no LIRF departure in June and July 2026 (80% and 74% in January and February).
  LFPG has a matched event for 21–32% of departures (`analysis/results/coverage.txt`).
- **First-heard ADS-B features shift across years.** A diagnostic found substantial spatial
  distribution changes at several airports. Masking or quantile mapping those features worsened
  a 2026 NM timing-proxy test; that proxy is not the hidden taxi-out target. No such transformation
  is applied.
- **The meta-model's gain is narrow.** About two thirds of it is one month at one airport in the
  rehearsal, and it loses slightly at LFPG and LIRF.
- **Date-wrap rows.** 2025 has records whose y lies near a whole number of days. The cap at d rules
  out their recorded value; whether 2026 has them cannot be checked.
- **What reconciliation cannot resolve.** A daily total says how much off-block delay a day had,
  not which flight had it. At EHAM, LFPG and LTFM EUROCONTROL's daily departure counts differ from
  ours on most days, so the constraint there is weak.
- **The rehearsal is not an untouched test.** Some constants were fixed on all of 2025, and the
  meta-model's settings and band were chosen after an earlier round on the same months;
  `docs/VALIDATION.md` lists what each number saw.

## Repository layout

```
solutions/core/         feature stores (features.py, fs_main.py, fs8.py), trainer (train_main.py), kernel builder
solutions/adsb/         adsb.lol extraction (extract.py, extract_v3.py), ground events (events_v3.py), kernel builders
solutions/final/        build_members.py: the member refit table
research/final/         assembler, rules, monthly level, long rows, 2025 rehearsal, configs/, manifests/, pseudo/
research/final/levers/  post layers: rules pass, recovered orphans, daily reconciliation, grade-A blend; artifacts/
research/final/meta/    the d <= 1 h meta-model: frames, dataset, kernel, rehearsal harness, gate, rebuild
research/fgap/          store fs8 member refits and the paired fs8 tests (runs/)
research/tail/          long-row physical estimate and its 2025 tables; the v30 long-row build (rebuild of v31 only)
research/pru/           fetch_pru.sh: EUROCONTROL files with their expected hashes
research/geometry/      build_geometry.py: stand and runway geometry (OSM, X-Plane, OurAirports)
research/               build_metar.py, upload.py (file checks, bucket upload), lb_api.py, kk.py (Kaggle)
analysis/               coverage and rehearsal step scripts; evidence/ prints the 2025 tables quoted in the docs; results/
kernels/                Kaggle kernel specs of the final run (metadata, member spec, environment; no kernel.py)
history/v32c/           record of the earlier board upload v31: members, weights, seeds, kernels as they ran
tools/                  sync_from_compbot.py, check_publish.py, kaggle_env.py
```

The code was developed in a private working tree. `tools/sync_from_compbot.py` copies into this
repository the files the final submission depends on, with the comparison runs the docs cite, and
records every edit it makes in `SYNC_MANIFEST.json`. Some code comments and config notes cite the
team's internal working notes (`findings_*.md`, `ADVERSARIAL_REVIEW_*.md`, `RESULTS.md`,
`NOTES.md`). Those notes are not published; the numbers this documentation relies on are restated
in `docs/` with the file or script that holds them.
