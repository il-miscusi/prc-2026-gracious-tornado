# Final assembly

This directory turns member predictions into the submitted file. It runs on either of two frames:

- `final`: the 670,790 departures of `final_ranking.parquet` (January, February, June and July
  2026), in `final_submitting.parquet` order;
- `board`: the 344,841 departures of the original `ranking.parquet` (January and July 2026), in
  `submitting.parquet` order.

```bash
python3 research/final/assemble_final.py research/final/configs/f5_final.json            # final frame
python3 research/final/assemble_final.py research/final/configs/v32c_repro_board.json    # board frame
```

Nothing is trained here. The assembly reads member predictions, the members' out-of-fold
predictions, 2025 labels and EUROCONTROL's monthly and daily statistics, and does arithmetic on
them: non-negative least squares, a two-parameter least-squares fit, a 12-coefficient ridge fit
and, in the daily post layer, an exact posterior on a 60 s grid. The d ≤ 1 h meta-model is fitted
elsewhere (`meta/`, a Kaggle kernel) and enters as a member. `configs/f5_final.json` took 82 s on
one CPU (2026-10-09); `configs/v32c_repro_board.json`, which has no post layers, takes about 10 s.

## Target and notation

For a departure, the target is y = takeoff − off-block in seconds. The takeoff time and the
scheduled off-block time are supplied, so d = takeoff − schedule is known for every row; the
off-block time is not. Then y = d − δ, where δ = off-block − schedule. Most of the error sits in
the rows where d is large: the flight either left the gate late and taxied normally (y small), or
its off-block was recorded at the schedule time (y ≈ d).

## Steps

Every step works on all departures of the frame; the order is fixed. The config names every
switch. The steps below are those of `configs/f5_final.json`, the submitted recipe.

1. **Segments** (`stacking.py`). Rows are split by d: d ≤ 1 h and 1 h < d ≤ 4 h. In each segment
   the members' predictions are combined with non-negative weights that sum to one. In f5 the
   d ≤ 1 h segment is one member, the G1short meta-model (`meta/`): a non-negative least-squares
   stack of c7, cs7, mlp7 and fs8 c7 plus 0.5 times a LightGBM model of its residual, fitted on the
   2025 month-out predictions of January, February, June and July. The 1-4 h weights come from
   non-negative least squares on the members' month-out predictions of the same four months
   (`pseudo_final.py fit-weights`), on rows without the 2025 date-wrap records and without rows
   with d ≤ 1 h and y > 3 h. Earlier configs (f3 and before) fitted the segment weights on the
   members' A/B out-of-fold predictions (January and July 2025, two folds by day of month). The
   weights are rounded to six decimals, stored in the config and checked against a refit on every
   run where out-of-fold predictions exist. Predictions are floored at 60 s.
2. **Long rows** (`longrows.py`), d > 4 h. The learners extrapolate a ratio of d here, while in 2025
   almost every such flight outside LIRF taxied normally. The member stack M (weights fitted on
   1-4 h rows that are not LIRF-unmatched) is blended with a physical estimate E:
   E = takeoff − AOBT_3 minus the airport's median lag when that difference lies in [0, d];
   otherwise an ADS-B stand-departure time plus the airport's median offset (grade-A events under
   1 h); otherwise the airport's median y. The medians come from 2025 departures in the same delay
   band (4-8 h, more than 8 h). The blend is ŷ = M + w(d)(E − M) with
   w(d) = clip(a + b log(d / 4 h), 0, 1), where (a, b) is fitted by least squares on the
   out-of-fold rows past 4 h. Rows whose AOBT_3 is more than 4 h before takeoff are not blended
   and keep M.
3. **LIRF rows without AOBT_3 past 4 h** (`long.fill`). The shipped configs use `long_stack`: the
   long-member stack M, unblended. The alternatives `fill_prob` (p(d)·d + (1 − p(d))·ỹ, a logistic
   fit in log d of the 2025 share of such rows recorded at the schedule time, ỹ the median y of the
   others) and `segment:<name>` exist for comparison.
4. **Overlay.** LIRF rows with d ≤ 4 h that have neither an ADS-B v3 event nor ADS-B coverage get
   ŷ = 0.7 ŷ + 0.3 k4, where k4 is a LightGBM member trained without any ADS-B column.
5. **Cap.** ŷ ≤ d where d > 2 h. Floor 60 s.
6. **Monthly level** (`level.py`). For each airport and month, the mean prediction over the rows
   comparable with EUROCONTROL's published taxi-out mean (stand and runway known, ŷ ≤ 2 h) is
   compared with the published mean. The difference plus a composition offset fitted on 2025
   (airport intercepts, de-icing share, NO_REF share) gives a shift; 0.75 of it is added to those
   rows. Months whose de-icing-excluded share is 5% or more are not shifted, because the published
   mean then leaves out the de-iced flights.
7. **Post layers**, each a function `apply(frame, pred, **kwargs) -> pred` named in the config, in
   this order: `levers/lv_rules.py` with rules (a) and (d) (step 8's bounds, so the daily layer
   starts from bounded values); `levers/recovered.py` (the NM join window applied to departures
   whose NM flight is found through the destination's arrival record, LIRF and LTFM);
   `levers/daily.py` (reconciliation with EUROCONTROL's daily pre-departure delay per airport and
   day, with the L1a-v2 ordinary-row loadings fitted on f5's own rehearsal rows,
   `meta/artifacts/daily_v2_f5_loadings.json`);
   `levers/recovered.py` again. `levers/evaluate.py` prints their 2025 month-out evidence.
8. **Rules** (`rules.py`), applied last so the delivered values satisfy them:
   - (a) The organisers joined an NM flight to a movement only when |off-block − LOBT| ≤ 3,606 s
     (no exception in 2,062,623 matched 2025 departures). For a matched row, y therefore lies in
     takeoff − LOBT ± 3,606 s; the final file dithers takeoff by up to ±12 s, so the window used
     is ±3,631 s.
   - (d) LEMD and LEBL: ŷ ≤ 3,604 s (the 2025 maxima were 3,598 and 3,604).
   - Floor 60 s.

## Checks

Each run checks the delivered file and fails if any of these is violated: the ids equal the
template's (unique, same order), both columns are float64 like every scored upload, values are
finite and at least 60 s, and the rules the config switches on hold. It also prints, per airport
and month, the published mean plus the 2025 composition offset minus our mean, and flags cells off
by more than 60 s. With `--ref-board FILE` it fails when the January/July rows differ from FILE by
more than 10 s RMS.

## Outputs

In `submissions/final/` (not in git): `NAME_final.parquet` (final frame only), `NAME_board.parquet`
(the January/July rows on the original board template), `NAME_stages.parquet` (the prediction
after each step) and `NAME.manifest.json`. The manifest records the config, the weights and
coefficients used, the row counts of every step, the sha256 of every input file (organiser data,
member predictions and out-of-fold files, PRU files, code) and output, the git commit and the
package versions. A copy is kept in `research/final/manifests/`.

The delivery manifest of the final file is `submissions/manifests/final.json` (tracked copy
`research/final/manifests/final_delivery.json`), written by `final_manifest.py draft`. On top of
the build manifest it records, for every unit behind each member (each refit kernel output or
variant directory), the Kaggle kernel, the store files it read, its model checksums and the bank
receipt that holds it; the month-out runs, the meta-model's kernel and dataset, the store, inputs
and ADS-B kernels, the EUROCONTROL CSVs and METAR; where each build input is held; and a `missing`
list of everything not found locally. `final_manifest.py replay` checks every recorded input,
re-merges each seed-merged member from its units, rebuilds the final file in a temporary directory
and exits 1 unless the final and board sha256 match. `final_manifest.py platform` records the
uploaded object's name, etag and read-back sha256.

## Validation

`configs/v32c_repro_board.json` takes the long rows from the old artifact instead of step 2 and
turns the rules off. It reproduces the earlier board submission v31 byte for byte.
`validate_board.py` checks the remaining pieces against the old scripts and on 2025 out-of-fold
predictions; `docs/VALIDATION.md` summarises the results.

`pseudo_final.py run configs/f5_pseudo2025.json [--weights frozen|crossfit]` runs the same steps
on January, February, June and July 2025 with month-out member predictions (each month predicted
by models trained without it) and scores them on the labels, untrimmed: per month, pooled, by
airport, by delay band, by row group and after every step. Everything the steps fit from labels
(stack weights in `crossfit` mode, plog coefficients, the physical-estimate tables, the fill model,
the level offset, the daily layer's parameters and loadings) is fitted without the scored month.
In `frozen` mode the weights are the recipe's; f5's 1-4 h weights were fitted on these four
months, so its frozen numbers are in-sample and its crossfit numbers are the ones to read. The
meta-model's month-out predictions are cross-fitted in both modes. `pseudo_final.py compare A B` gives the paired
difference of two runs with airport-day block bootstrap intervals.

The four 2025 months contain 8 "date-wrap" rows (the right time of day stamped on the scheduled
date, y near a whole number of days). All eight are LIRF departures with d > 2 h and y above d, so
the cap ŷ ≤ d keeps the pipeline from predicting their recorded values. 2026 labels are hidden, so
whether 2026 has such rows cannot be checked; `compare` reports every delta with and without those
rows ("nw").

## Files

| file | purpose |
|---|---|
| `frame.py` | loads a frame from the organiser files and derives d, takeoff − LOBT, takeoff − AOBT_3, NM match flags |
| `stacking.py` | member predictions, out-of-fold matrices, segment weights |
| `longrows.py` | step 2 |
| `level.py` | step 6 |
| `rules.py` | step 8 and the bound checks |
| `assemble_final.py` | runs the steps, checks, writes outputs and the manifest |
| `validate_board.py` | comparisons with v32c and on 2025 out-of-fold predictions |
| `pseudo_final.py` | the 2025 four-month rehearsal with month-out members, and paired comparisons |
| `merge_members.py` | seed-averages refit kernels of one specification into one member |
| `final_manifest.py` | delivery manifest of the final file (`draft`), byte replay (`replay`), platform record (`platform`) |
| `levers/` | the post layers of step 7 and their month-out evaluation |
| `meta/` | the d ≤ 1 h meta-model: frames, dataset, Kaggle kernel, rehearsal harness, gate, rebuild |
| `configs/` | one JSON file per candidate |
