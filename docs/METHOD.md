# Method

This document describes the pipeline that builds the final file, step by step as
`research/final/assemble_final.py` runs it, and the evidence behind each step. The selected final
config is `research/final/configs/f6_final.json`; its 2025 rehearsal is
`research/final/configs/f6ev_pseudo2025.json`. `docs/VALIDATION.md` has the rehearsal protocol and
the full ablation.
`docs/PROVENANCE.md` says which ideas are our own and credits the ones other teams published.

The main line grew through paired rehearsal tests (section 6):

- **f3**: the base pipeline of this document with members on stores fs6 and fs7;
- **f4** = f3 + a CatBoost member on store fs8, segment weights refitted on month-out predictions,
  and six seeds for every tree member;
- **f5** = f4 with the d ≤ 1 h stack replaced by a LightGBM residual meta-model (G1short).
- **f6** = f5 with a final blend toward grade-A ADS-B pushback evidence.

## Evaluation sets used below

- **Rehearsal** (`research/final/pseudo_final.py`): the final pipeline run on January, February,
  June and July 2025, 671,292 departures, with month-out members (each month predicted by models
  trained on 2025 without that month). Changes are in pooled MSE units (change in SSE / 671,292)
  with a 95% airport-day block bootstrap interval. "Crossfit" refits every stack weight without
  the scored month. f5's frozen weights were fitted on the same four months, so for f5 the
  crossfit numbers are the ones to read.
- **nw**: the rehearsal without its 8 date-wrap rows. These are LIRF departures whose recorded y
  lies near a whole number of days (the right time of day stamped on the scheduled date). All
  eight have d > 2 h and y above d (one of them by 1 s), so the cap at d (step 5) rules out their
  recorded value. Each has a squared error of the order of 10⁹ s², so a step's effect on these 8
  rows can outweigh its effect on the other 671,284. 2026 labels are hidden, so whether 2026 has
  such rows cannot be checked. We read every comparison on nw first and give the all-row number
  beside it.
- **MO12**: the 12-month-out predictions of the earlier member c6 (CatBoost on store fs6) on all
  2,085,047 departures of 2025, MSE 55,101 (RMSE 234.74). The post layers were developed on this
  base, with every fitted quantity fitted on the other 11 months. Changes are in units of change
  in SSE / 2,085,047; "months worse" counts months whose change is above +0.5.
- **A/B**: January and July 2025 split by day of month (A = days 16–31, B = days 1–15), models
  trained on the rest of 2025. Used to screen the fs6/fs7 members and to fit the long-row stack.
  It is optimistic, because the other half of each scored month is in training.

## 1. The target

The target is y = takeoff − off-block, in seconds. Takeoff and the scheduled off-block time are
both supplied, so the delay to takeoff, d = takeoff − schedule, is known for every row. Writing
δ = off-block − schedule,

    y = d − δ.

For most departures d is under an hour and y is taxiing plus queueing. For a delayed departure y
depends on how the off-block was recorded. A flight held at the gate and pushed late has a large δ
and an ordinary y. A flight whose off-block was recorded at the schedule time, or that pushed on
time and then waited away from the stand, has δ near 0 and y close to d.

Squared error rewards the conditional mean of the recorded value. For the 89% of departures with
d ≤ 1 h, taxiing dominates. The rest of the error is in how delayed flights were recorded, and two
things in the data constrain that directly: the organisers' NM join rule and EUROCONTROL's
published airport aggregates. Hence three layers: an ordinary taxi model, bounds that follow from
how the data was built, and reconciliation with the aggregates.

The on-time off-block mode is almost only a LIRF phenomenon. Share of 2025 departures delayed by
more than an hour whose off-block was within 10 minutes of schedule
(`analysis/evidence/modes.py`):

| d band | LIRF all | LIRF, no NM match | LIRF, NM match | next highest airport |
|---|---:|---:|---:|---|
| 1–4 h | 9.9% | 46.9% | 6.9% | EGLL 2.1%, LTFM 1.8% (8.6% without NM match) |
| 4–8 h | 14.3% | 80.5% | 0% | LFPG 0.1% |
| > 8 h | 66.7% | 98% | 4% | none |

## 2. Layer 1: the ordinary taxi model

### Inputs and features

Features come from fields the ranking file publishes, from open external data, or from encodings
fitted on 2025 labels (leave-one-month-out for training rows). A training departure's off-block
time and target never become features of another row. The final stores are built from
`final_ranking.parquet` only, its dithered takeoff times included.

| family | what | why |
|---|---|---|
| schedule and NM timing | takeoff − EOBT_1, AOBT_3, LOBT, IOBT; their differences; ratios to d | the strongest label-free signal of when the aircraft left |
| stand and runway | target and quantile encodings of stand, runway, stand × runway, hour, configuration | taxi distance and routing |
| geometry | stand-to-runway path length, straight distance, crossings (`research/geometry`) | covers stand × runway pairs rare in 2025; EDDF Terminal 3 opened on 2026-04-23 |
| weather | IEM METAR: temperature, dew point, wind, gusts, visibility, cloud, pressure, precipitation, weather codes | de-icing and low-visibility procedures |
| traffic | departures and arrivals in moving windows; neighbours' takeoff − EOBT_1 on the same runway (fs6) | queues |
| turnaround | inbound arrival of the same aircraft, stand occupancy before and after | readiness |
| ADS-B | adsb.lol ground stays reduced to graded off-block evidence (below) | the only direct observation of off-block |
| fs8 additions | 64 `g8_*` columns (below) | plan revisions, live lateness and queues, stands unseen in 2025 |

**ADS-B events** (`solutions/adsb/`). `extract_v3.py` keeps every aircraft's whole ground stay at
the ten airports from each day's adsb.lol release, including stays that cross midnight.
`events_v3.py` reduces each departure to graded evidence: A = parked and seen leaving the stand
(the observed time is the pushback), B and C = first seen moving (a lower bound on off-block, far
from or near the runway), D = no departure evidence. Grade A's median error against the 2025
off-block is −0.6 s (January 2025). On A/B, adding the events (store fs4) moved XGBoost's fold B
MSE by −2,369 [−3,779, −1,324].

**Matching** (store fs7, `FS_OPTS["match"] = "v7"`). Events are assigned to flights one-to-one per
airport, with IATA-to-ICAO callsign aliases from OpenFlights and a 900 s window. A match on time
alone feeds the flight's own columns only when the aircraft type agrees; otherwise it goes to
columns that carry no identity. In fs6, 17.2% of time-only grade-A matches were within 60 s of the
true off-block, against 81.4% of callsign matches. CatBoost on fs7 (c7) against the same model on
fs6 (c6), A/B: −672 [−1,337, −81].

**Store fs8** (`solutions/core/fs8.py`, kernel `prc-f-fs8`, written by
`solutions/core/build_kernel.py derive`). fs8 is the fs7 store with 64 `g8_*` columns appended;
every fs7 column keeps its bytes. All are computed from published fields, and a departure's own
off-block is never read. The groups:

- plan revision (8 columns): AOBT_3, EOBT_1, LOBT and IOBT against the schedule and each other,
  and route-normalised planned airborne times;
- lateness (4): other flights' AOBT_3 − EOBT_1 and AOBT_3 − LOBT in the past and next hour;
- neighbour taxi (20): other flights' takeoff − AOBT_3 in past and future windows by airport and
  runway, its spread, and the flights that pushed with ours;
- queue (19): takeoffs between EOBT_1 or LOBT and our takeoff, flights that pushed after us and
  took off before us (and the reverse), takeoff spacing on our runway;
- arrivals (2): arrival taxi-in in the last 15 and next 30 minutes;
- stand context (11): typical takeoff − AOBT_3 and arrival taxi-in per stand and stand group,
  computed within the frame and so available for 2026 stands with no 2025 history (EDDF
  Terminal 3, a new EDDM pier), and traffic at the stand as a rate per month of the frame.

The store kernel ran twice. Version 2 counts stand traffic as a rate per month of the frame instead
of a raw count: the 2025 training frame spans 12 months and the 2026 frame 4, so a raw count
differs threefold between training and prediction (comment in `fs8.py`). The refits
and three of the four month-out folds of fs8 c7 read version 2; the February fold read version 1
(`kernels/prc-f8-c7a-mo/output/oof_sources.json`).

**Coverage moves between months** (`analysis/coverage.py`, output in
`analysis/results/coverage.txt`). Share of departures with a matched v3 event:

| airport | Jan 2026 | Feb 2026 | Jun 2026 | Jul 2026 | Jan 2025 | Jul 2025 |
|---|---:|---:|---:|---:|---:|---:|
| LIRF | 79.9% | 74.0% | 0.0% | 0.4% | 0.0% | 95.9% |
| LFPG | 26.6% | 20.9% | 32.1% | 27.3% | 0.6% | 15.1% |
| LTFM | 0.0% | 0.0% | 0.1% | 0.0% | 0.0% | 0.0% |
| other seven airports | 60–97% | 60–98% | 89–95% | 85–98% | 3–99% | 29–97% |

LIRF is seen in the winter months of 2026 and not in the summer months, the reverse of 2025. LTFM
is never seen. So the members learned LIRF in summer with ADS-B and must predict LIRF in summer
2026 without it; the overlay of step 4 is for that case.

**Columns computed but not used.** The fs7 store also writes EUROCONTROL monthly levels (`pru_*`)
and airport-month-demeaned NM lags (`apm_*`, `dm_*`). Every member drops them (`NOPRU` in
`solutions/final/build_members.py`): as features they were worse month-out (+677 [+54, +1,466]
and +1,035 [+321, +1,875], January and July held out). The monthly aggregates enter only through
layer 3. The column `a3_obs_p_evidence_start` is dropped too: earlier stores wrote an array index
into it instead of a timestamp.

### Members

Specs are in `solutions/final/build_members.py` (fs8 c7: `research/fgap/build_fs8_members.py`,
the c7 spec on store fs8); each kernel's exact spec is in `kernels/<slug>/member.json`. Every
member is refit on all of 2025 at fixed rounds (1.1 times the fold-mean best iteration of its
original cross-validated run, no early stopping), and every seed's model is saved.

| member | store | learner | target | seeds in the final | used in |
|---|---|---|---|---|---|
| c7 | fs7 | CatBoost 1.2.8, depth 10 (long head depth 8), GPU | d ≤ 1 h: min(y, 2 h); d > 1 h: y/d | 6: c7a 3101/4101/5101, c7b 3151/4151/5151 | meta-model; long stack |
| fs8 c7 | fs8 | same as c7 | same | 6: prc-f8-c7a, prc-f8-c7b (same seeds) | meta-model; 1–4 h |
| cs7 | fs7 | CatBoost 1.2.8, one model per airport | same | 3 | meta-model |
| mlp7 | fs7 | MLP | same | 6: 2911–2916 (`prc-f-mlp7m6`) | meta-model |
| c6s | fs6 | CatBoost 1.2.8, depth 10 | d > 1 h in seconds | 6 (`prc-f-c6sm6`) | meta-model inputs; 1–4 h; long |
| c5s | fs6 without neighbour columns | CatBoost 1.2.8, depth 10 | d > 1 h in seconds | 6 (`prc-f-c5sm6`) | meta-model inputs; 1–4 h; long |
| x5 | fs6 without neighbour columns | XGBoost 2.1.4, GPU | y/d | 6 (`prc-f-x5m6`) | 1–4 h; long |
| k4 | fs6, every ADS-B column dropped | LightGBM | y/d | 2 | meta-model inputs; LIRF overlay |

Rows with d > 1 h are fitted on r = y/d with weights min(d, 12 h)², because their squared error in
seconds scales with d². The seconds-target variants c6s and c5s earned their place in the 1–4 h
band: as 20% additions to an equal mix of c7, cs7 and mlp7 they changed A/B MSE by −525 (c5s) and
−492 (c6s); x5 by −391.

**x5 is pinned to XGBoost 2.1.4.** The first refit ran under 3.4.1, the default of the Kaggle and
Colab images on 2026-10-08, and its month-out folds were far worse than the same specification
under 2.1.4. Only the pinned runs are used. In the rehearsal, adding x5 to the 1–4 h and long stacks
(f2x5 − f2) changed nw MSE by −283.7 [−468.9, −129.3] with frozen weights and −380.1
[−650.6, −145.8] crossfit, every month better (`docs/VALIDATION.md`).

**fs8 c7 is an extra member, not a replacement.** Replacing c7 by the same model on fs8 failed its
gate. In a paired month-out test (two seeds for d ≤ 1 h, six for longer rows;
`research/fgap/runs/r3/paired_2s6l.json`) the change was +117.5 [−1,220.0, +1,678.4] in February
and −746.4 [−2,155.1, +219.4] in July (nw, month units), against a gate of −300 or better in every
fold. Over the four rehearsal months, single-seed c7 on fs8 against c7 on fs7
(`research/fgap/runs/c7_mo_fs8_vs_fs7.json`) gave nw −376.8, −1,167.9 and −270.0 in January,
February and June and +601.0 in July, every interval spanning zero. As an extra member of the
d ≤ 1 h and 1–4 h stacks (f4 against f3, crossfit) it changed nw MSE by −240.5 [−419.8, −99.9]:
d ≤ 1 h rows −116.0 [−143.1, −90.4], 1–4 h rows −153.4 [−320.8, −11.0]. January, February and June
improved with intervals below zero; July changed by −50.5 [−322.7, +233.7].

**Members dropped for the final.** cs6, cs5, mlp6fs and split-fs4 had about 5% weight or less in
the earlier stacks, and as 20% additions they made A/B MSE worse (+24.4, +32.5, +46.5, +60.4).
Refitting the stack without them, and without c6 and c5d10, cost nothing measurable out of fold.
mlp7 was also trained on fs8 and got weight 0 (section 7).

**Seed averaging.** `research/final/merge_members.py` averages every seed's refit predictions
(unclipped) and then floors at 60 s. Each merged member joins two three-seed kernels; each half is
11–16 s RMS from the merged prediction (`kernels/<slug>/output/merge_manifest.json`). mlp7 merges
six one-seed runs, each 47–51 s RMS from the merged prediction. The rehearsal's month-out members
are one model per held month, so seed averaging is not part of what the rehearsal measures.

### Step 1: the stacks

Rows are split by d: d ≤ 1 h, 1 h < d ≤ 4 h, and d > 4 h (step 2).

**d ≤ 1 h: the G1short meta-model** (`research/final/meta/`). Its prediction is

    ŷ = base + α · g(x),

where base is a non-negative least-squares stack of c7, cs7, mlp7 and fs8 c7 (weights .145542,
.252147, .247639, .354673 for 2026), g is a LightGBM model of the residual y − base, and α = 0.5.
g's inputs are the predictions of c7, cs7, mlp7, fs8 c7, c6s, c5s and k4, the base, the members'
spread and range, d, base/d, the airport, and row context: takeoff − LOBT, takeoff − AOBT_3,
takeoff and scheduled hour, weekday, NM match, a3/ADS-B coverage flags, the ADS-B evidence grade
and its observed leave time. Settings (`research/final/meta/results/prc-meta-gbm-f4_report.json`):
15 leaves, at least 2,000 rows per leaf, L2 penalty 50, learning rate 0.05, feature and bagging
fraction 0.8, seed 7, deterministic. It is fitted on the members' 2025 month-out predictions of
the four rehearsal months (the frame `research/final/meta/build_frame.py` builds), on rows without
the date-wrap records and without y > 3 h. The number of rounds (25 to 600) and α (0 to 1) are
chosen by leave-one-month-out inside the training months. For 2026 that gave 600 rounds and
α = 0.5, applied to the refit members' 2026 predictions in the Kaggle kernel `prc-meta-gbm-f4`
(Kaggle CPU, about 4 minutes). `research/final/meta/reproduce.py` regenerates the 2026 rows from
the saved booster: the largest difference from the kernel's file is 3.6 × 10⁻¹² s
(`analysis/results/meta_reproduce_f4.txt`).

In the rehearsal the meta-model is always cross-fitted: the prediction for month m comes from a
base and a booster fitted on the other three months, with rounds and α chosen inside those three.
f5 against f4, crossfit, nw: −49.0 [−85.4, −12.2]; by month −2.0, −163.4, +14.7 and −62.0
(January, February, June, July); d ≤ 1 h rows −56.8 [−89.3, −23.9], 1–4 h rows +7.5 [−1.7, +17.2].
Without the LTFM departures of February 2025 the change is −17.6 [−45.3, +11.5], and without
February −17.9 [−51.4, +15.7]
(`research/final/meta/pseudo/compare_meta_gbm-G1short-f4_crossfit__f4_pseudo2025_crossfit.json`,
key `segment_split`). By airport (nw, pooled units) the model gains most at LTFM (−39.3) and EGLL
(−8.2) and loses at LFPG (+9.6) and LIRF (+4.0). The restriction to d ≤ 1 h was chosen
after a first round of meta-models (section 7), so this re-fit on the f4 members is the
confirmatory run. It passed the gate fixed for that lane (change and interval below zero, no
month's interval above zero), but most of the gain is one month at one airport.

**1–4 h: non-negative least squares.** c6s .297598, c5s .021663, x5 .324981, c7 0, fs8 c7 .355759,
71,109 final rows. Since f4 the weights come from non-negative least squares on the members'
month-out predictions of the four rehearsal months (`research/final/pseudo_final.py fit-weights`),
on rows without the date-wrap records and without d ≤ 1 h rows with y > 3 h, normalised to sum to
one. fs8 c7 has no A/B out-of-fold predictions, so the A/B fit used before f4 was not possible; the
same four-month fit without fs8 c7 also moves away from the A/B weights. The weights are frozen in
the config and checked against a refit on every run. In the rehearsal, frozen mode uses these
weights on the months they were fitted on, so its numbers are in-sample; crossfit refits them on
the other three months.

**Long stack M, d > 4 h** (3,018 final rows): c6s .3083, c5s .1398, x5 .1646, c7 .3873, fitted on
A/B out-of-fold rows with 1 h < d ≤ 4 h, unchanged since f3.

### Step 2: rows with d > 4 h

Ratio models extrapolate badly past 4 h, while almost every 2025 flight outside LIRF delayed that
long taxied normally. `research/final/longrows.py` blends the long-member stack M with a physical
estimate E:

- E = takeoff − AOBT_3 minus the airport's median lag, when that difference lies in [0, d] (kind N,
  2,569 final rows);
- else a grade-A ADS-B stand departure under one hour plus the airport's median offset (kind A,
  88 rows);
- else the airport's median y in the delay band, 4–8 h or > 8 h (kind M, 297 rows).

    ŷ = M + w(d)·(E − M),   w(d) = clip(a + b·log(d / 4 h), 0, 1),   (a, b) = (−0.0476, 0.3798)

(a, b) is fitted by least squares on A/B out-of-fold rows past 4 h, with E built from tables that
exclude the scored fold. The mean weight on the final rows is 0.108. On A/B the smooth weight moved
MSE on rows past 2 h by A −16.9 and B −143.8. One final row whose AOBT_3 lies more than 4 h before
takeoff keeps M unblended: on one 2025 out-of-fold row, blending toward takeoff − AOBT_3 cost +4,087
fold MSE units.

### Step 3: LIRF rows without AOBT_3 past 4 h

These 63 final rows are LIRF departures without AOBT_3 (almost all without an NM flight), delayed
by more than 4 h. In 2025 most such rows were recorded at the schedule time (80.5% at 4–8 h, 98%
past 8 h; section 1). The shipped fill (`long.fill = "long_stack"`) is the long-member stack M,
unblended. The alternative, `fill_prob`, sets p(d)·d + (1 − p(d))·ỹ, with p(d) a logistic fit in
log d of the 2025 share of such rows recorded at the schedule time and ỹ the median y of the
others.

The evidence for `long_stack` is weak. In the rehearsal, `fill_prob` instead of `long_stack` in
recipe f2 changed nw MSE by +314.6 [−255.9, +983.2] with frozen weights and +252.2 [−340.9, +921.2]
crossfit: `long_stack` is ahead on nw, and the intervals span zero. Almost all of the difference is
on these rows (+303.6 of the +314.6, frozen) and in June (+1,018.1 month units). On all rows the
sign flips (−423.1 [−2,044.3, +713.0] frozen, −648.2 crossfit), because of the 2025 date-wrap
rows: six of them are fill rows (four in July), their recorded y lies above d, and `fill_prob` puts
them nearer to d. The daily
layer moves these rows a lot: in f5's rehearsal (crossfit) it changed 51 of the 63, by 2,914 s on
average over all 63, for −2,256.2 of its nw gain
(`analysis/results/pseudo_steps_f5_pseudo2025_crossfit.json`, step `post3_daily`, group
`> 4 h LIRF-unmatched`). We chose `long_stack` because it is ahead on nw in both modes. The data do
not settle it.

### Step 4: LIRF without ADS-B

Trained on 2025, the members lean on ADS-B columns that are empty for LIRF in June and July 2026.
On LIRF rows with d ≤ 4 h that have neither an event nor coverage, the prediction becomes
0.7·ŷ + 0.3·k4, where k4 is trained without any ADS-B column. 35,255 final rows qualify. On
January 2025 LIRF rows (11,300 rows, no LIRF coverage that month) the overlay changed MSE per row
by −1,734 on A/B (A −502, B −2,951). In f5's rehearsal it touched 28,207 rows and changed nw MSE by
−22.1 [−76.2, +32.3] (crossfit): the right sign, not significant. Two alternatives lost on A/B:
retraining the members on rows masked to look uncovered (+5,904 per row), and a LIRF-only
no-ADS-B specialist (+3,142 alone, folds disagreeing in a blend).

### Step 5: cap

ŷ ≤ d where d > 2 h, then a 60 s floor. In 2025, 23 of 39,809 departures with d > 2 h had
y > d + 120 s, and 11 of those 23 are date-wrap rows (`analysis/evidence/rules_ad_oof.py`). On MO12
the cap changed MSE by −111.3 [−170.7, −63.4] nw, no month worse (all rows +45.8, from the
date-wrap rows). In f5's rehearsal: −190.3 [−432.2, −32.4] nw (crossfit), on 25 rows. 34 final rows
are capped.

## 3. Layer 2: bounds from how the data was built

Two rules, applied twice: once as the first post layer (`research/final/levers/lv_rules.py` with
`rules: "ad"`), so the daily layer starts from bounded values, and once as the assembler's last
step (`research/final/rules.py`), so the delivered values satisfy them.

**(a) NM join window.** The organisers joined each movement to an EUROCONTROL Network Manager
flight. On all 2,062,623 NM-matched 2025 departures, off-block − LOBT lies in [−3,606 s, +3,606 s]
with no exception (`analysis/evidence/nm_join_bound.py`), and arrivals show the same window in
2025 and in all four 2026 months (`analysis/evidence/nm_join_2026.py`). So for a matched departure

    takeoff − LOBT − 3,606 ≤ y ≤ takeoff − LOBT + 3,606,

widened by 25 s on 2026 rows for the ±12 s takeoff dither of the final file (±3,631 s). The lower
end is floored at 60 s. A competitor described this window in a public repository
(github.com/joris-decombe/taxiout); we verified it on the data before using it.

**(d) LEMD and LEBL: ŷ ≤ 3,604 s.** The 2025 maxima were 3,598 s and 3,604 s.

| on MO12 | (a) | (d), on top of (a) |
|---|---:|---:|
| change [95% CI] | −466.3 [−1,218.3, −43.1] | −8.5 [−17.9, −1.7] |
| months worse | 0/12 | 0/12 |
| rows moved | 199 | 13 |

Source: `analysis/evidence/rules_ad_oof.py`, output in `analysis/results/rules_ad_oof.txt`. July
2025 carries −337.9 of (a)'s −466.3: the gain comes from a few rows with large errors. In f5's
rehearsal (crossfit), the post-layer pass of both rules changed nw MSE by −42.1 [−91.0, −5.4] on 44
rows. Its all-row change, −4,039, is almost all one date-wrap row that rule (a) lifted into its
window. On the final file the first pass moves 46 rows and the last pass lowers 3 LEMD/LEBL rows.

A rule from a competitor's public code, ŷ ≥ takeoff − AOBT_3 − 3,600 where
takeoff − AOBT_3 > 1,800 s, made MO12 worse (+124.0 [+79.2, +177.9] on top of (a), every month
worse) and is not used.

**Recovered orphans (L0e, `research/final/levers/recovered.py`).** A departure without an NM match
whose flight lands at one of the ten airports appears there as an NM-matched arrival with the same
flight number and route, landing 10 minutes to 20 hours after our takeoff. That arrival carries
the flight's LOBT. If the departure's join failed because of the window, its y lies outside
takeoff − LOBT ± 3,606 s, so a prediction inside that interval moves to the nearer edge. Among
2025 recovered orphans whose base prediction fell inside the interval, the truth was outside for
44 of 44 LIRF rows and 21 of 21 LTFM rows. At the other airports the join failed for other reasons
too often, so the step is limited to LIRF and LTFM; that gate was chosen on all of 2025, the
rehearsal months included. In f5's rehearsal: −160.6 [−304.8, −51.5] nw (crossfit), on 29 rows. On
the final file it moves 32 rows. It runs again after the daily layer and moved nothing there, in
the rehearsal or on the final file.

## 4. Layer 3: reconciliation with EUROCONTROL's aggregates

### Step 6: monthly level (L1c, `research/final/level.py`)

EUROCONTROL publishes, per airport and month, total reference and additional taxi-out minutes and
the number of flights they cover, so a mean taxi-out time. For each airport-month of the frame,

    shift = 0.75 · (published mean − our mean + c),

where our mean is over the comparable rows (stand and runway known, ŷ ≤ 2 h) and c is a
composition offset (airport intercepts, the de-icing share and the share without reference time)
fitted by ridge regression on 2025 airport-months against c6's month-out predictions. The shift is
added to those rows. The published mean leaves out flights de-iced after off-block, so an
airport-month whose de-icing exclusion share is 5% or more is not moved: in 2026 these are EDDF,
EDDM, LFPG and LSZH in January and EDDM and LSZH in February (`research/final/manifests/f5.json`,
`level.cells`). On MO12 the step changed MSE by −177.0 [−201.5, −153.5], no month worse
(`analysis/evidence/daily_mo12.py`); in f5's
rehearsal, −59.2 [−79.7, −37.8] nw (crossfit). Its purpose is monthly shifts in NM reporting
conventions, which `analysis/evidence/eda_lags.py` shows at LTFM, LFPG, EGLL and EDDF: a level bias
at one airport in one month is what this step removes. The organiser ruled the monthly file usable
for the scored months on 2026-10-08 (`docs/DATA.md`).

The step does not remove the largest monthly bias the rehearsal shows: LTFM in February 2025,
where 1–4 h rows were under-predicted by 438 s on average (2,141 rows), against at most ±64 s for
every other airport in that band and month (`analysis/results/pseudo_steps_f5_*.json`,
`anatomy_nw`). Each member's February fold was trained without February 2025; the final refit
trains on it.

### Step 7: daily reconciliation (L1a with L1a-v2 loadings, `research/final/levers/daily.py`)

EUROCONTROL's daily file gives, per airport and takeoff day, DLY_ALL_PRE_2 in minutes. On 2025
labels it equals Σ max(0, off-block − schedule) over the day's departures, keyed on takeoff date:
correlation 0.9997, median ratio 1.0001 (`analysis/evidence/daily_exact.py`). Keying on block or
scheduled date drops the correlation to 0.992 or 0.971, and unclipped sums give ratio 1.06. Since
d is known, the file fixes, per airport-day g,

    K_g = 60 · DLY_ALL_PRE_2 = Σ_i max(0, d_i − y_i) + ε_g .

A flight recorded at its schedule contributes about 0; a flight held at the gate contributes its
whole delay. That is the ambiguity layer 2 cannot resolve. The layer computes, per airport-day, an
exact posterior and replaces each prediction by its posterior mean:

- **Ordinary rows (d ≤ 1 h).** The error of their predicted contribution Σ max(0, d − p) is a
  common day effect, −n_eff·s_g, with n_eff the ordinary rows with p < d and s_g the day's mean
  residual over them. In 2025 this effect correlates with the day's mean residual of delayed
  ordinary rows at 0.85–0.998. Prior: s_g follows a Student-t with 5 degrees of freedom, location
  and scale from the airport's 2025 days.
- **Long rows (d > 1 h).** A discrete prior over y on a 60 s grid. At LIRF: a point y = d (recorded
  at schedule) with probability π = (p − τ)/(d − τ), implied by the incoming prediction p and the
  airport's held-taxi level τ; the airport's empirical held-taxi spread; and a 10% broad component
  over the feasible window. Elsewhere only the held spread, because the on-time mode is a LIRF
  phenomenon (section 1); with an on-time point at every airport, EDDF's MO12 change was +688.
  NM-matched rows are truncated to the rule (a) window, and recovered orphans lose the interval
  L0e forbids. The prior is then tilted (minimum KL divergence) so that its mean equals p, so a day
  that carries no information leaves the prediction unchanged.
- **Likelihood.** R_g = K_g − Σ_ordinary max(0, d − p) = A_g − n_eff·s_g + ε_g, where A_g is the
  long rows' contribution. ε_g is Student-t with 3 degrees of freedom and scale
  √(s0² + κ²·dn² + 48·n), where dn is EUROCONTROL's departure count minus ours, κ is floored at the
  airport's median contribution per departure (646–1,107 s), and 48 s² per contributing flight
  covers the ±12 s dither (2026 only).
- **Posterior.** Exact forward-backward over the long rows' contributions, with an 81-point grid
  over s_g. Long rows take E[y | K]. An ordinary row moves by b·(E[s | K] − s̄), where s̄ is the
  airport's typical 2025 day shift and b is the L1a-v2 loading below. Rule (a) is re-applied inside
  the layer.

The parameters (`research/final/levers/artifacts/daily_params.json`) are fitted on all of 2025 on
the c6 chain: c6's month-out predictions after the monthly level, the rules and L0e. In the
rehearsal they are refitted without the scored month.

**L1a-v2 loadings.** The first version moved every delayed ordinary row by the full day shift and
the others by a per-airport factor ρ. L1a-v2 looks the loading b up by whether the row is delayed
(d − p > 0) and by its d sub-band. Each b is the through-origin slope of y − p on the day shift
over the build's own 2025 rehearsal rows before the daily layer, shrunk to its delayed/other
parent (DerSimonian–Laird) and clipped to [−0.5, 2.5]; rows with y > 3 h are left out of the fit.
The loadings belong to the stack they were fitted on, so each build refits them on its own rows in
two passes: a first rehearsal with the previous build's loadings provides the rows, and the
loadings fitted on those rows are the build's. f5's values
(`research/final/meta/artifacts/daily_v2_f5_loadings.json`):

| loading b | d ≤ 15 min | 15–30 | 30–45 | 45–60 | parent |
|---|---:|---:|---:|---:|---:|
| delayed (d − p > 0) | 0.35 | 0.52 | 0.99 | 1.34 | 0.77 |
| other | 0.16 | 0.34 | 0.18 | 0.20 | 0.21 |

A row that took off within 15 minutes of schedule carries about a third of the day effect; a row
45–60 minutes late carries more than all of it. Finer classes (by ADS-B evidence, NM match,
airport, hour) were tried and none improved on this table by more than 0.5 MSE units. When the
loadings were introduced (f2x5_l1av2 against f2x5), they changed nw MSE by −39.9 [−59.7, −23.8]
with frozen weights and −40.2 [−59.8, −24.2] crossfit, every month better; only rows with
d ≤ 1 h moved. In the rehearsal the loadings are refitted without the scored month.

**Evidence for the layer.**

| measurement | change [95% CI] | notes |
|---|---:|---|
| f5 rehearsal, step `post3_daily`, nw, crossfit | −3,095.7 [−6,951.5, −838.9] | −2,256.2 on the 63 LIRF fill rows, −833.0 on rows with d ≤ 4 h |
| same step, all rows | −8,976.4 | includes the 8 date-wrap rows |
| recipe f2 without the three post layers (rules pass, L0e, daily layer), minus f2, nw | +3,817.0 [+835.4, +9,157.4] frozen; +4,668.6 [+915.9, +11,518.2] crossfit | every month worse without them (frozen: +806.2 / +12,475.7 / +1,611.9 / +1,835.2 month units); the assembler's own rules still apply (a), (d) and the floor |
| MO12, after the monthly level, rules (a), (d) and L0e, nw | −953.6 [−1,353.1, −614.3] | no month worse; every airport's interval below zero |
| same, all rows | −1,725.1 [−2,946.3, −862.8] | |

The MO12 numbers are printed by `analysis/evidence/daily_mo12.py` (output in
`analysis/results/daily_mo12.txt`), which refits the layer's parameters without each scored month.
On the 2026 final file the layer moves 670,721 rows by 45.0 s RMS.

What the layer cannot do: a day total says how much delay the day had, not which flight had it.
One long-delayed flight on a day resolves well; five 1–4 h flights on one day resolve only in
aggregate. Where EUROCONTROL's departure population differs from ours the constraint is weak:
daily counts agree exactly on 3–19% of 2025 days at EHAM, LFPG and LTFM
(`analysis/evidence/daily_exact.py`, `count_diff_2026.py`), and the likelihood widens with the
count gap on those days.

The daily file comes from the same EUROCONTROL portal and terms as the monthly file. No organiser
ruling names it (`docs/DATA.md`).

### Step 8: grade-A pushback blend (`research/final/levers/events.py`)

A grade-A event means the aircraft was seen parked and then leaving the stand within 60 s.
Its `a3_obs_leave` is seconds before takeoff, already in target units. This direct observation
can still improve the prediction after daily reconciliation. Only rows with `a3_grade == 3`
and 0 ≤ `a3_obs_leave` ≤ max(d, 3,600 s) are eligible; grades B, C and D never move.

The short (d ≤ 1 h) and long groups are fitted separately. For each group and airport the
event estimate is y_e = `a3_obs_leave` + median(y − `a3_obs_leave`). With fewer than 50 fit
rows the offset falls back to the group's pooled median. The blend is

    ŷ_new = ŷ + w · (y_e − ŷ),

where w is the through-origin least-squares slope of y − ŷ on y_e − ŷ within a bin of
|y_e − ŷ|, clipped to [0, 1]. Bin edges are 60, 180, 600 and 1,800 s. The specification was
fixed before its rehearsal gate. The 2026 artifact (`levers/artifacts/events_f5.json`) is fitted
on f5's frozen rehearsal rows, column `post4_recovered`: short weights 0.558 / 0.246 / 0.014 /
0 / 0; long weights 0.825 / 0.676 / 0.347 / 0 / 0. Disagreement above 600 s receives zero
weight, so an event far from the incoming estimate does not override it.

In the rehearsal each scored month's offsets and weights are fitted on the other three
months' f5 rows, in the same frozen or crossfit mode. The layer checks that its incoming
predictions equal f5's recorded values for that month. After blending, moved rows get the cap
at d past 2 h, rules (a) and (d), the recovered-orphan window and the 60 s floor again;
the assembler then applies its final rules. The daily-layer loadings stay f5's because this
blend runs after the daily layer and the second recovered-window pass.

The complete f6 − f5 comparison is −50.6 [−62.3, −38.2] pooled nw MSE crossfit, and
−50.9 [−62.8, −38.5] frozen. No date-wrap row moves. Every month has a lower point estimate;
July's interval spans zero. The frozen layer moves 76,066 rehearsal rows (crossfit: 76,067) and 118,366 final rows
(17.6% of the final, 10.65 s RMS over all rows). Almost no eligible evidence exists at LFPG
or LTFM; this is a coverage-dependent refinement (`docs/VALIDATION.md`).

## 5. What the final file looks like

On the 670,790 final departures (`research/final/manifests/f6.json`): 596,663 rows take the
meta-model, 71,109 the 1–4 h stack and 3,018 the rows past 4 h (2,954 blended, 63 fill, 1 guard);
the overlay touches 35,255 rows, the cap 34, the monthly level 595,912; the post layers move 46,
32, 670,721, 0 and 118,366 rows, the last being the events layer. The final rules lower 2 rows and raise 33 to the floor.
The
file is `f6_final.parquet`, sha256 `602c05a26e665b4136dfe08a712c8ce2f685a08303f32dc3334e132692b83bf3`.

## 6. How the main line grew

Each step is a paired rehearsal comparison, crossfit, nw (`docs/VALIDATION.md` names the files).

| step | change [95% CI] | by month (Jan / Feb / Jun / Jul) |
|---|---:|---|
| f4 − f3: fs8 c7 added to the d ≤ 1 h and 1–4 h stacks, weights refitted month-out | −240.5 [−419.8, −99.9] | −217.4 / −491.2 / −261.1 / −50.5 |
| f5 − f4: G1short meta-model replaces the d ≤ 1 h stack | −49.0 [−85.4, −12.2] | −2.0 / −163.4 / +14.7 / −62.0 |
| f6 − f5: grade-A pushback blend after daily reconciliation | −50.6 [−62.3, −38.2] | −96.2 / −53.4 / −48.0 / −14.4 |

The six-seed refits of c6s, c5s and x5 change only the 2026 predictions, because the rehearsal
uses one month-out model per member.

## 7. Results we chose not to ship

Each row is a paired rehearsal comparison of two complete pipelines unless noted;
`docs/VALIDATION.md` names the files.

| what | result | why not |
|---|---|---|
| fs8 c7 as a replacement for c7 (paired month-out test, not the full pipeline) | February +117.5 [−1,220.0, +1,678.4], July −746.4 [−2,155.1, +219.4] (nw, month units) | failed its gate (−300 or better in every fold); kept as an extra member instead |
| fs8 mlp7 as an extra member of f5's 1–4 h stack | weight 0 in every held month and on all four months; the arm equals f5 (change 0.0) | no weight |
| f2p: the daily layer may not move rows that rule (a) put on a window bound | +41.5 [+8.3, +93.2]; nw +22.2 [+4.2, +44.1] | worse |
| L1a-v2 held spreads for long rows from quantiles of y − p by airport × ADS-B evidence | nw +87.1 [+54.8, +128.9], every month worse | narrow priors stop the long rows absorbing the day's discrepancy |
| same, the airport's spread scaled by a per-class factor | nw +3.5 [−19.2, +24.7] | no gain |
| stack weights per airport × band, ridge-shrunk to the pooled weights (meta C1, first round) | nw +82.3 [+14.0, +161.7], crossfit | 1–4 h weights do not carry over between months (July 2025 +336) |
| weights varying with d, coverage, NM match and hour (meta C2, first round) | nw −49.8 [−143.1, +38.5], crossfit | interval spans zero |
| LightGBM residual model on both bands (meta C3, first round) | nw −19.6 [−144.8, +148.4], crossfit; 4–8 h rows +93.7 [+0.4, +252] | its 1–4 h changes move the daily layer's long-row posteriors |

The first-round meta-models were compared with recipe f2 (the f3 pipeline without x5 and without
the L1a-v2 loadings). The meta models themselves are always cross-fitted month-out; "crossfit"
refers to the rest of the recipe. The d ≤ 1 h-only variant of C3 passed in that round (nw −65.3
[−103.1, −25.6]) and became G1short after its re-fit on the f4 members (section 2).

Earlier experiments, measured on A/B or MO12 against the members of the day, before the final
pipeline existed:

| what | result | why not |
|---|---|---|
| OPDI surface events (EUROCONTROL/OpenSky) | pre-takeoff event coverage LIRF 0.0–0.2%, LFPG 0–1.2%, LTFM 0% (Jan/Jul 2025 and 2026) | no coverage where it is needed |
| TabM; TabICLv2 zero-shot | in the short stack: A −8.4, B +7.7; TabICL weight ≤ 0.014 | folds disagree; no gain |
| LIRF recording-mode classifier as a post layer, ŷ = P·d + (1 − P)·τ̂ | MO12 +577.0 [+65.3, +1,030.7], 11 of 12 months worse; guarded −134.0 [−345.6, +94.4] | the members already encode the mode; it now enters the daily layer's prior |
| non-LIRF orphans shrunk toward the airport median, d > 3 h | MO12 −112.4 [−312.0, +19.4] | interval spans zero; February worse |
| month-out residual corrector (LightGBM on c6's residual) | MO12 −232.5 on c6; +20.2 [−5.0, +47.5] on another stack's out-of-fold predictions | learns one member's residuals; does not transfer |
| ordered ground-track features; multi-event trajectory expert | +209.7 [−58.2, +649.4]; c6 + 0.1 × expert +58.8 [+34.7, +85.2] | no gain over the incumbent |
| LTFM queue and sequence features | January/July −395, March/September +233 [−107, +616] | failed a rule fixed before the run: no harm on March/September |
| exact d² long-row weights instead of min(d, 12 h)² | January/July held out +227 [−389, +879] | worse month-out |
| covariate-shift row weights (2026/2025 cell ratios) | fold A +513 [−513, +1,357] | amplifies thin cells |
