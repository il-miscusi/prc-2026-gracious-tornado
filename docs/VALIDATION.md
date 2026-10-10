# Validation

The final is one blind file over January, February, June and July 2026. Labels exist only for
2025. This document describes how the pipeline was validated on 2025, gives the results of the
final recipe f6, and lists every comparison behind the design with the file that holds it.

Every number below is either in a file of this repository (named next to it) or printed by a
script of this repository from the stored inputs (`docs/REPRODUCE.md`).

## Designs

| design | rows scored | members trained without | used for |
|---|---|---|---|
| A/B | January and July 2025, split by day (A = 16–31, B = 1–15) | the scored half-months | screening the fs6/fs7 members; fitting the long-row stack and coefficients |
| MO12 | all 2,085,047 departures of 2025, predicted by member c6 | the scored month (12 folds) | developing each post layer and bound |
| rehearsal | January, February, June and July 2025 (671,292 departures), the full pipeline | the scored month, for every member | choosing between complete pipelines; fitting the f4/f5 stack weights and the meta-model |

All scores are full and untrimmed: every row counts, including rows past 4 h and recording-mode
rows. Intervals are 95% airport-day block bootstrap intervals, 2,000 draws, seed 0. A change in
"pooled units" is the change in SSE divided by all rows of the evaluation set; a change in "month
units" is divided by that month's rows.

**Date-wrap rows ("nw").** Eight rehearsal rows, and 13 rows of 2025 as a whole, are LIRF
departures whose recorded y lies near a whole number of days (the right time of day stamped on the
scheduled date). All eight rehearsal rows have d > 2 h and y above d (one by 1 s), so the cap
ŷ ≤ d rules out their recorded value. Their squared errors are of the order of 10⁹ s² each: on its
own, this group of 8 rows lifts f6's pooled rehearsal MSE (crossfit) from 61,473 to 74,554. 2026
labels are hidden, so whether the final months contain such rows cannot be checked. Every
comparison is reported with and without them; "nw" is the number without them, and the one we read
first.

## The 2025 rehearsal

`research/final/pseudo_final.py run CONFIG --weights frozen|crossfit` runs every step of
`assemble_final.py` (segments, long rows, fill, overlay, cap, level, post layers, rules) once per
held month m ∈ {1, 2, 6, 7}, on the 2025 departures of that month:

- **Frame.** Built from the 2025 training file of the month with the label and the departure's
  off-block time removed before any step runs (`frame.load_training_frame`). y is read only for
  fitting on other months and for scoring.
- **Members.** Each member's fold M<m> is the member's specification trained on 2025 without
  month m, at the specification's fixed rounds, one model per fold (kernels `prc-f-<member>-mo`,
  `prc-f8-c7a-mo`; `kernels/<slug>/member.json` has each spec). x5's folds are the XGBoost 2.1.4
  runs `prc-f-x5-mo` (M1, M6) and `prc-f-x5p-mo` (M2, M7). fs8 c7's February fold was trained on
  version 1 of the fs8 store, its other three folds and its refits on version 2
  (`kernels/prc-f8-c7a-mo/output/oof_sources.json`). The 2026 refits average six seeds; the
  rehearsal does not measure seed averaging.
- **The meta-model** (f5's d ≤ 1 h segment) enters as a member whose prediction for month m comes
  from a base stack and a booster fitted on the other three months, with rounds and α chosen inside
  those three by leave-one-month-out (`mo_G1.csv` of kernel `prc-meta-gbm-f4`). It is cross-fitted
  in both modes.
- **Fitted without month m in both modes:** the physical-estimate tables and fill model of the long
  rows, the monthly level's composition offset, the daily layer's parameters (refitted on the c6
  chain without month m) and the L1a-v2 loadings. The loadings are fitted on the same recipe's own
  rows before the daily layer, so each recipe runs twice: a first pass (`*p1` configs) with the
  previous recipe's loadings provides those rows.
- **The f6 events layer** fits its airport offsets and disagreement-bin slopes on f5's rows
  from the other three months, in the same mode. Its entering prediction is checked against
  f5's `post4_recovered` column. The 2026 artifact uses all four frozen rehearsal months.
- **Crossfit mode** also refits the 1–4 h stack weights and the long-row stack and coefficients on
  the other three months' month-out predictions.
- **Frozen mode** uses the config's weights. For f4 and f5 the 1–4 h weights were fitted on these
  same four months, so frozen numbers for f4 and f5 are in-sample; the long-row stack was fitted on
  January and July 2025 (A/B). **For f5 and f6, read the crossfit numbers.**
- **Fixed on all of 2025** (and so seen in every held month): the level's λ = 0.75 and its 5%
  de-icing cap; the L0e airport gate (LIRF, LTFM); the bounds of rules (a) and (d), which come from
  the data's construction; the structure of the daily layer, including the held-only prior outside
  LIRF, chosen after an all-airport version failed month-out at EDDF; the meta-model's settings
  (leaves, minimum rows per leaf, penalty, learning rate), which are those of its first round, and
  its restriction to d ≤ 1 h, chosen after that round. February and June 2025 also informed earlier
  development through MO12. The rehearsal is the closest available analogue of the final, not an
  untouched test.

`pseudo_final.py compare A B` pairs two runs row by row and writes
`research/final/pseudo/compare_A__B.json`: the change (A minus B) pooled, per month, per delay
band, per row group and per airport, all rows and nw, with intervals. Each run writes a summary,
`research/final/pseudo/NAME_MODE.json`, with per-month scores, the MSE after every step, the fitted
parameters of every held month, the input hashes and the sha256 of its per-row file.

### Results of the submitted recipe

`f6ev_pseudo2025` (summaries `research/final/pseudo/f6ev_pseudo2025_{crossfit,frozen}.json`; nw values
from `analysis/results/pseudo_steps_f6ev_pseudo2025_{crossfit,frozen}.json`).

| months | rows | MSE crossfit | RMSE crossfit | nw MSE crossfit | nw RMSE crossfit | MSE frozen | nw MSE frozen |
|---|---:|---:|---:|---:|---:|---:|---:|
| January | 153,706 | 104,224.1 | 322.84 | 99,461.1 | 315.37 | 104,187.4 | 99,424.5 |
| February | 143,731 | 48,085.5 | 219.28 | 48,085.5 | 219.28 | 47,976.2 | 47,976.2 |
| June | 183,142 | 76,944.3 | 277.39 | 42,987.3 | 207.33 | 76,852.7 | 42,893.5 |
| July | 190,713 | 68,295.1 | 261.33 | 58,699.0 | 242.28 | 68,139.2 | 58,602.4 |
| January + July | 344,419 | 84,329.4 | 290.40 | 76,890.4 | 277.29 | 84,226.6 | 76,820.5 |
| February + June | 326,873 | 64,254.6 | 253.48 | 45,229.0 | 212.67 | 64,155.2 | 45,128.4 |
| all four | 671,292 | 74,554.4 | 273.05 | 61,473.4 | 247.94 | 74,453.2 | 61,388.6 |

The two-month rows are row-weighted from the month rows. The frozen numbers are lower by 85 nw
units pooled; for f6 the frozen 1–4 h weights were fitted on these months.

**Where the error is** (crossfit, nw, `anatomy_nw` in the same file). January's MSE is a handful of
rows: the ten largest errors carry 69.0% of the month's SSE, against 7–9% in the other three
months. Rows with d between 1 and 4 h are about 10% of the rows and carry 11,731 to 30,024 of each
month's MSE. In February, 1–4 h rows at LTFM were under-predicted by 438 s on average (2,141 rows);
at every other airport that month the 1–4 h mean residual is within ±64 s.

### What each step adds, in shipping order

`analysis/pseudo_steps.py` reads the rehearsal's per-row file, which holds the prediction after
every step, and scores each step against the one before it. This is a sequential ablation: each
step is measured on top of everything before it. Crossfit, pooled units; month columns nw in month
units.

| step | rows moved | nw change [95% CI] | frozen nw | all rows | Jan | Feb | Jun | Jul |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stacks, rows past 4 h, LIRF fill (start) | | MSE 65,094.0 | MSE 64,753.9 | MSE 88,127.3 | | | | |
| LIRF no-ADS-B overlay | 28,207 | −22.1 [−76.2, +32.3] | −19.8 | −22.1 | −82.9 | +9.4 | +52.1 | −68.1 |
| cap at d past 2 h | 25 | −190.3 [−432.2, −32.4] | −206.6 | −110.6 | −676.2 | −8.0 | −2.5 | −116.3 |
| monthly level (L1c) | 577,620 | −59.2 [−79.7, −37.8] | −58.4 | −59.3 | −0.2 | −62.1 | −39.8 | −123.3 |
| rules (a), (d) | 44 | −42.1 [−91.0, −5.4] | −36.4 | −4,039.3 | −12.4 | −79.7 | −0.7 | −77.5 |
| recovered orphans (L0e) | 29 | −160.6 [−304.8, −51.5] | −161.8 | −314.6 | −39.4 | −355.0 | −73.4 | −195.5 |
| daily reconciliation (L1a-v2) | 671,183 | −3,095.7 [−6,951.5, −838.9] | −2,831.2 | −8,976.4 | −753.6 | −8,699.6 | −1,588.8 | −2,207.0 |
| L0e again | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| grade-A pushback blend | 76,067 | −50.6 [−62.3, −38.2] | −50.9 | −50.6 | −96.2 | −53.4 | −48.0 | −14.4 |
| final rules and 60 s floor | 23 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **final** | | **MSE 61,473.4** | **MSE 61,388.6** | **MSE 74,554.4** | | | | |

Month columns are `delta_nw_month` of
`analysis/results/pseudo_steps_f6ev_pseudo2025_crossfit.json`. The all-row change of the rules step
is almost entirely one date-wrap row that rule (a) lifted into its window. The daily layer's gain
sits on two groups (`by_group` of that step): −2,256.2 on the 63 LIRF departures without AOBT_3
past 4 h (51 of them moved, by 2,914 s on average over the 63), and −833.0 on rows with d ≤ 4 h.

### Paired comparisons of complete pipelines

Each row switches one thing between two complete pipelines and scores both in the rehearsal.

Recipes: f2 is f3 without x5 and without the L1a-v2 loadings; f2x5 adds x5; f2x5_l1av2 adds the
loadings; f3 is f2x5_l1av2 with the six-seed mlp7 refit (same rehearsal rows); f4 and f5 are
described in `docs/METHOD.md`. f2_fillprob is f2 with the `fill_prob` long-row fill, and
f2_nopost is f2 without its three post layers (the assembler's own rules still apply (a), (d) and
the 60 s floor). Files are in `research/final/pseudo/`, named `compare_A_MODE__B_MODE.json`, except
where another directory is named.

**The main line.**

| switch | A − B | crossfit, nw | crossfit, all rows | frozen, nw | nw by month, crossfit (Jan / Feb / Jun / Jul) |
|---|---|---:|---:|---:|---|
| fs8 c7 added to the d ≤ 1 h and 1–4 h stacks, weights refitted month-out | f4 − f3 | −240.5 [−419.8, −99.9] | −39.4 [−380.0, +465.8] | −344.4 [−567.6, −162.0] (in-sample weights) | −217.4 / −491.2 / −261.1 / −50.5 |
| G1short meta-model replaces the d ≤ 1 h stack | f5 − f4 | −49.0 [−85.4, −12.2] | −40.5 [−80.8, +3.4] | −47.7 [−83.4, −12.3] | −2.0 / −163.4 / +14.7 / −62.0 |
| same, without LTFM in February 2025 | | −17.6 [−45.3, +11.5] | | | −2.0 / −16.2 / +14.7 / −62.0 |
| same, without February | | −17.9 [−51.4, +15.7] | | | |
| grade-A pushback blend, after the second recovered-window pass | f6 − f5 | −50.6 [−62.3, −38.2] | −50.6 [−62.3, −38.2] | −50.9 [−62.8, −38.5] | −96.2 / −53.4 / −48.0 / −14.4 |

The f6 − f5 files are `research/final/pseudo/compare_f6ev_pseudo2025_{crossfit,frozen}__f5_pseudo2025_*.json`.
The events specification was fixed before these comparisons. Every month's point estimate improves;
July's crossfit interval is [−42.4, +15.8]. No date-wrap row moves. The frozen layer moves
76,066 rows; the crossfit layer moves 76,067 at the analysis script's 10⁻⁹ s tolerance.

The f5 − f4 files are `analysis/results/compare_f5_pseudo2025_{crossfit,frozen}__f4_pseudo2025_*.json`.
The rows without LTFM February are `segment_split` of
`research/final/meta/pseudo/compare_meta_gbm-G1short-f4_crossfit__f4_pseudo2025_crossfit.json`,
written by `research/final/meta/compare_all.py` from the same per-row files (its crossfit nw
change equals the f5 − f4 file's, −49.0).

**Earlier switches that remain in the recipe.**

| switch | A − B | frozen, all rows | frozen, nw | crossfit, nw | nw by month, crossfit (Jan / Feb / Jun / Jul) |
|---|---|---:|---:|---:|---|
| L1a-v2 loadings | f2x5_l1av2 − f2x5 | −39.9 [−59.7, −23.8] | −39.9 [−59.7, −23.8] | −40.2 [−59.8, −24.2] | −15.8 / −27.3 / −49.5 / −60.8 |
| x5 in the 1–4 h and long stacks | f2x5 − f2 | −342.6 [−569.8, −148.3] | −283.7 [−468.9, −129.3] | −380.1 [−650.6, −145.8] | −164.9 / −215.9 / −62.2 / −982.7 |
| without the post layers (rules pass, L0e, daily layer, L0e) | f2_nopost − f2 | +9,358.1 [+3,239.5, +17,755.3] | +3,817.0 [+835.4, +9,157.4] | +4,668.6 [+915.9, +11,518.2] | +813.7 / +15,810.6 / +1,911.6 / +2,025.6 |
| fill `fill_prob` instead of `long_stack` | f2_fillprob − f2 | −423.1 [−2,044.3, +713.0] | +314.6 [−255.9, +983.2] | +252.2 [−340.9, +921.2] | +44.3 / +4.3 / +970.9 / −83.4 |

`long_stack` is ahead of `fill_prob` on nw in both modes, the intervals span zero, and on all rows
the sign flips because of the 2025 date-wrap rows (crossfit all rows −648.2 [−2,606.8, +665.5]).

**Not shipped** (`docs/METHOD.md`, section 7):

| switch | A − B | crossfit, nw | frozen, all rows | frozen, nw |
|---|---|---:|---:|---:|
| fs8 mlp7 as an extra 1–4 h member of f5 | f5fm7 − f5 | 0.0 (weight 0 in every held month) | not run | not run |
| daily layer may not move rows on a rule (a) bound | f2p − f2 | not run | +41.5 [+8.3, +93.2] | +22.2 [+4.2, +44.1] |
| held spreads by airport × ADS-B evidence (quantiles) | f2x5_l1av2sq − f2x5 | +81.7 [+49.8, +121.9] | +87.1 [+54.8, +128.9] | +87.1 [+54.8, +128.9] |
| held spreads, airport spread × class factor | f2x5_l1av2ss − f2x5 | +5.6 [−16.6, +25.9] | +3.5 [−19.2, +24.7] | +3.5 [−19.2, +24.7] |

The fs8 c7 replacement test was a paired month-out comparison of the member alone, not of the
full pipeline (`research/fgap/paired_eval.py`; gate: all-row change of −300 or better in every
fold, and no delay band or airport whose own interval lies wholly above zero):

| test | file | February (nw, month units) | July (nw, month units) |
|---|---|---:|---:|
| c7 on fs8 instead of fs7, two seeds below 1 h, six above | `research/fgap/runs/r3/paired_2s6l.json` | +117.5 [−1,220.0, +1,678.4] | −746.4 [−2,155.1, +219.4] |

and over all four months with one seed (`research/fgap/runs/c7_mo_fs8_vs_fs7.json`, nw, month
units): January −376.8 [−874.2, +138.0], February −1,167.9 [−4,634.8, +1,165.0], June −270.0
[−986.2, +511.7], July +601.0 [−1,297.6, +2,376.8].

The first-round meta-models were compared with f2; their files are
`research/final/meta/pseudo/compare_meta_<arm>_<mode>__f2_pseudo2025_<mode>.json`:

| arm | crossfit, all rows | crossfit, nw | frozen, nw | nw d ≤ 1 h band, crossfit |
|---|---:|---:|---:|---:|
| C1 `lin-apt`: weights per airport × band, ridge-shrunk | −269.3 [−1,012.0, +138.9] | +82.3 [+14.0, +161.7] | +64.0 [−33.6, +169.8] | −25.2 [−40.7, −11.4] |
| C2 `lin-vc`: weights varying with d, coverage, NM, hour | +34.6 [−101.9, +224.3] | −49.8 [−143.1, +38.5] | −66.7 [−173.2, +33.7] | −67.2 [−139.5, −6.6] |
| C3 `gbm-G1`: LightGBM residual model, both bands | +278.6 [−133.0, +988.0] | −19.6 [−144.8, +148.4] | −43.6 [−174.4, +96.0] | −75.4 [−110.1, −42.4] |
| C3 with x5 as an input (`gbm-G1x`) | +233.4 [−162.4, +914.1] | −51.1 [−174.4, +106.7] | −74.2 [−203.3, +59.5] | −78.6 [−113.4, −44.9] |
| C3 without row context (`gbm-G1m`) | +401.0 [−175.8, +1,299.4] | +69.5 [−193.0, +416.9] | +38.9 [−219.9, +344.3] | −63.1 [−88.3, −39.8] |
| `gbm-G1short`, d ≤ 1 h only (chosen after the rows above) | −38.8 [−99.4, +40.4] | −65.3 [−103.1, −25.6] | −69.9 [−105.6, −32.3] | −75.3 [−108.8, −42.7] |

`gbm-G1short` re-fitted on the f4 members is f5's meta-model (the main-line table above).

**The gate.** In the final phase a new component shipped only if its nw change was below zero with
the interval below zero (crossfit for f4 and f5, whose frozen weights are in-sample) and no month
got worse beyond noise (the meta lane's wording: no month's interval entirely above zero). fs8 c7
as an extra member, the grade-A blend, the L1a-v2 loadings and x5 pass it, and removing the post layers fails it in
the other direction (every month worse without them). G1short passes it formally; without LTFM
February its interval spans zero. Two kept components do not pass it: the LIRF overlay (its step
interval spans zero) and the `long_stack` fill (ahead on nw, interval spanning zero).
`docs/METHOD.md` gives the reasons each is kept.

## MO12 evidence for the bounds and post layers

Measured on c6's 12-month-out predictions, all 2025, in units of change in SSE / 2,085,047
(c6 MSE 55,101.4).

| step | change [95% CI] | nw | months worse | source |
|---|---:|---:|---:|---|
| cap ŷ ≤ d past 2 h | +45.8 [−149.3, +323.9] | −111.3 [−170.7, −63.4] | 0/12 nw | `analysis/evidence/rules_ad_oof.py` |
| rule (a) | −466.3 [−1,218.3, −43.1] | same | 0/12 | same |
| rule (d), on top of (a) | −8.5 [−17.9, −1.7] | same | 0/12 | same |
| rejected AOBT_3 floor, on top of (a) | +124.0 [+79.2, +177.9] | same | 12/12 | same |
| monthly level (L1c) | −177.0 [−201.5, −153.5] | −177.0 [−201.7, −153.7] | 0/12 | `analysis/evidence/daily_mo12.py` |
| rules (a), (d) and L0e, after L1c | | −607.7 [−1,515.1, −174.0] | 0/12 | same |
| daily layer, after L1c, rules (a), (d) and L0e | −1,725.1 [−2,946.3, −862.8] | −953.6 [−1,353.1, −614.3] | 0/12 | same |

`analysis/results/rules_ad_oof.txt` and `daily_mo12.txt` hold the two scripts' output.
`daily_mo12.py` refits the daily layer's parameters without each scored month and also prints the
layer by airport (nw); every airport's interval is below zero:

| | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| change | −23.1 | −9.1 | −40.4 | −9.1 | −18.8 | −10.4 | −46.0 | −724.8 | −4.6 | −67.2 |
| 95% CI | [−32.5, −15.4] | [−17.5, −2.3] | [−52.5, −30.5] | [−14.7, −4.8] | [−26.7, −11.4] | [−16.1, −5.6] | [−60.1, −30.8] | [−1,125.5, −388.3] | [−6.4, −3.0] | [−85.3, −49.1] |
| months worse | 0 | 1 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 |

## ADS-B transfer diagnostic

The label-free diagnostic `usiadianimuwa/prc-adsb-shift` v1 compared final fs7 features
in 2025 and 2026. All five store hashes matched the recorded inputs, and unmodified c7a/cs7
prediction means matched their saved seed arrays (`analysis/results/adsb_shift_summary.json`).
The first-heard spatial feature group shifted substantially at LIRF, LSZH and EGLL.

A LightGBM proxy trained on March–May and August–December 2025 predicted takeoff − AOBT_3,
which is observed in both years. On the four 2026 months, masking the five suspect columns
raised proxy MSE by 1,209.7 (day-clustered SE 91.9), and quantile mapping raised it by 227.0
(SE 23.3). In the retained taxi models, masking moved c7a by 58.4 s RMS and cs7 by 64.3 s RMS.
These are sensitivity and proxy results, not taxi-out target validation. They do not justify
changing the final pipeline, and neither transformation is applied.

## Board score of the submitted file

The organisers' public board scores the January/July 2026 rows. The submitted file's
January/July rows (`f6_board.parquet`, sha256 `5785d8b84ce99528666d0fd22cadd25670cbc42388798be1066f60d27862c5ed`),
uploaded as `gracious-tornado_v37.parquet`: RMSE **233.1717 s**, recorded on **2026-10-10** UTC.
This score covers January/July only; the blind four-month final is not scored on that board.

## Reproducing an earlier file byte for byte

The assembler also rebuilds the earlier board upload v31 (internal name v32c) from the stored
member predictions, which checks the assembler's arithmetic against a file built by the scripts
it replaced:

```bash
python3 research/tail/expand_artifact.py OUT
python3 research/tail/longrows_v30.py OUT/v29_base.parquet research/tail/out/v30_longrows_a05_plog.parquet \
    --alpha 0.5 --phys plog --keep-aobt3-over 4 --old OUT/v21_old.parquet
python3 research/final/assemble_final.py research/final/configs/v32c_repro_board.json --out-dir OUT
```

The board file it writes has sha256
`e878c18ebd19db569e304e0b92081ce698fe9c8caeeab09d85352ba503a8e9ac`, identical to the uploaded
`gracious-tornado_v31.parquet`. It needs the organiser files, the two EUROCONTROL monthly files,
v31's member predictions and out-of-fold files (private kernel outputs, `docs/REPRODUCE.md`) and
`research/tail/artifacts/longrows_inputs_over2h.parquet`. `history/v32c/` records v31's members,
seeds, rounds and kernels as they ran. The models behind v31's member predictions were not saved;
the final's members are new refits of the same specifications with their models saved.
