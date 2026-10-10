# Provenance of ideas

Where each step and feature group of the final recipe (`research/final/configs/f6_final.json`)
came from. "Own" means the team's own analysis of the organiser data. Where another team published
the idea first, or at the same time, it is credited with its repository and licence. Every idea
credited to another team was re-implemented from its written description (README, notes or
write-up); no code was copied. As a check, four of the credited source files
(gentle-octopus `pipeline/taxiout/train.py`, kind-mango `src_v3/postprocess.py`, zestful-fountain
`nm_neighbor_features.py` and `arrival_features.py`) were compared line by line with our rules,
post layers, feature stores and meta-model code on 2026-10-09: no non-trivial line is shared.

Licences were read from each repository's GitHub licence field on 2026-10-09. Rules and ideas are
not code; the credit is given because the eligibility page asks for originality and because the
other teams found them.

## Pipeline steps

| step | source of the idea | licence of the source | our implementation |
|---|---|---|---|
| segments by d (≤ 1 h, 1–4 h, > 4 h) with non-negative least-squares weights | own; NNLS stacking as in Breiman (1996), "Stacked regressions", Machine Learning 24 | paper | `research/final/stacking.py`, `research/final/pseudo_final.py` (`fit-weights`) |
| target y/d with weights min(d, 12 h)² for d > 1 h | own | – | `solutions/final/build_members.py`, `solutions/core/train_main.py` |
| long rows: member stack blended with a physical estimate, w(d) | own | – | `research/final/longrows.py` |
| LIRF rows without AOBT_3 past 4 h take the long-member stack | own | – | `research/final/longrows.py` |
| LIRF overlay: 30% of a member trained without ADS-B | own | – | `research/final/assemble_final.py` (overlays), member k4 in `solutions/final/build_members.py` |
| cap ŷ ≤ d where d > 2 h | own | – | `research/final/assemble_final.py` |
| 60 s floor | own; other teams use a similar floor (resilient-kiwi clips to [60, 90,000] s) | – | `research/final/rules.py`, `research/final/merge_members.py` |
| monthly level (L1c) to EUROCONTROL's published mean, de-icing cells left alone | own; that the published mean leaves out flights de-iced after off-block is also documented by vibrant-lollipop, github.com/ahmetabdullahgultekin/prc-taxiout-2026 | GPL-3.0 | `research/final/level.py` |
| rule (a): NM join window, y within takeoff − LOBT ± 3,606 s | gentle-octopus, github.com/joris-decombe/taxiout (`RESEARCH.md`); also documented by outgoing-boat, github.com/EnioAguiar/prc-taxiout-2026. Verified by us on all 2025 departures (`analysis/evidence/nm_join_bound.py`) | GPL-3.0 (both) | `research/final/rules.py`, `research/final/levers/lv_rules.py` |
| rule (d): LEMD and LEBL ŷ ≤ 3,604 s | kind-mango, github.com/skylinkapi/prc-data-challenge-2026-kind-mango (per-airport caps); joyful-lemon, github.com/fenil264/prc-2026-taxi-out (EDA note). Verified by us on 2025 (`analysis/evidence/rules_ad_oof.py`) | GPL-3.0; joyful-lemon states no licence | `research/final/rules.py`, `research/final/levers/lv_rules.py` |
| recovered NM window (L0e): LOBT found through the destination's arrival record | own | – | `research/final/levers/recovered.py` |
| daily reconciliation (L1a) with EUROCONTROL's daily pre-departure delay | own (the daily file as an exact constraint on Σ max(0, d − y)); the idea of reconciling forecasts with known aggregates follows hierarchical forecast reconciliation, Wickramasuriya, Athanasopoulos and Hyndman (2019), JASA 114. Other teams used the daily file as model features (outgoing-boat; quirky-honey, github.com/dhsdb010/OPDI) | paper; GPL-3.0 | `research/final/levers/daily.py` |
| L1a-v2 loadings by delayed/other and d sub-band | own; shrinkage by DerSimonian and Laird (1986), Controlled Clinical Trials 7 | paper | `research/final/levers/daily.py` |
| G1short: LightGBM model of the stack's residual, fitted month-out | a month-fold out-of-fold residual corrector on base predictions was described by resilient-kiwi, github.com/nurcombethorne-lgtm/prc-taxitime (`notes/FINDINGS.md`), quirky-honey, github.com/dhsdb010/OPDI, and outgoing-boat, github.com/EnioAguiar/prc-taxiout-2026 (also on the challenge Discord, 2026-09-28). The member-prediction inputs, the d ≤ 1 h restriction, the crossfit and the choice of α are ours | GPL-3.0 (all three) | `research/final/meta/build_frame.py`, `research/final/meta/kernels/prc-meta-gbm-f4/kernel.py` |

## Features

The grade-A pushback blend added in f6 is our own implementation
(`research/final/levers/events.py`). It uses the graded events credited below, with
airport offsets and disagreement-bin slopes fitted on 2025 rehearsal residuals;
`docs/METHOD.md`, step 8, gives its specification and validation.

| feature group | source of the idea | licence of the source | our implementation |
|---|---|---|---|
| schedule and NM timing: takeoff − EOBT_1, AOBT_3, LOBT, IOBT, their differences and ratios to d | own | – | `solutions/core/features.py` |
| stand and runway target and quantile encodings, configuration | own | – | `solutions/core/features.py` |
| neighbours' takeoff − EOBT_1 on the same runway (fs6) | own (residual analysis, 2026-09-28) | – | `solutions/core/features.py` |
| turnaround: inbound aircraft and stand occupancy | the latest arrival at the same stand was suggested by pdhl1965 on the challenge Discord (2026-09-22); the occupancy bound and inbound matching are ours | – | `solutions/core/features.py` |
| geometry: stand-to-runway path length and crossings | stand positions from X-Plane apt.dat were mentioned by SoK on the challenge Discord (2026-09-24); taxi-graph path lengths from OSM aeroways follow Olive, Waltert and Schultz (ATRDS 2025, code github.com/xoolive/airport_queues, not used) | paper | `research/geometry/build_geometry.py` |
| METAR weather, de-icing and snow flags with 3 h windows | own | – | `research/build_metar.py`, `solutions/core/features.py` |
| adsb.lol ground stays reduced to graded off-block evidence (A–D) | own grading; using adsb.lol history was also reported on the challenge Discord by SoK and GREKI (2026-09-24); that a transponder switched on at the stand shows the pushback is from Olive, Waltert and Schultz (ATRDS 2025) | paper | `solutions/adsb/extract_v3.py`, `solutions/adsb/events_v3.py` |
| one-to-one event matching with OpenFlights IATA-to-ICAO aliases (fs7) | own | – | `solutions/core/features.py` |
| fs8 plan revision (AOBT_3 − LOBT, AOBT_3 − IOBT, route-normalised planned times) | resilient-kiwi, github.com/nurcombethorne-lgtm/prc-taxitime; kind-mango, github.com/skylinkapi/prc-data-challenge-2026-kind-mango | GPL-3.0 | `solutions/core/fs8.py` (`g8_p_*`) |
| fs8 lateness of other flights in the past and next hour | quirky-honey, github.com/dhsdb010/OPDI; resilient-kiwi | GPL-3.0 | `solutions/core/fs8.py` (`g8_l_*`) |
| fs8 neighbours' takeoff − AOBT_3 in past and future windows | zestful-fountain, github.com/Phoenix-Ops-LTD/prc2026-taxiout; merry-mushroom, github.com/TAUIL-Abd-Elilah/prc-data-challenge-2026; outgoing-boat | GPL-3.0 | `solutions/core/fs8.py` (`g8_n_*`) |
| fs8 queue: take-offs between EOBT_1 or LOBT and ours, overtaking flights, runway spacing | unique-umbrella, github.com/kagervig/PRC-Challenge-2026; likable-river, github.com/sqcm9896/prc-2026-likable-river; merry-mushroom | GPL-3.0 | `solutions/core/fs8.py` (`g8_q_*`) |
| fs8 arrival taxi-in in the last 15 and next 30 minutes | outgoing-boat; zestful-fountain; merry-mushroom | GPL-3.0 | `solutions/core/fs8.py` (`g8_a_*`) |
| fs8 stand context for stands unseen in 2025 | the 2026 terminal changes (EDDF Terminal 3, EDDM pier) are documented by merry-quicksand, github.com/saurabhdxt-labs/prc-challenge; the per-frame stand statistics are ours | GPL-3.0 | `solutions/core/fs8.py` (`g8_s_*`) |

Team names are the challenge's team names; the Discord names are participants' handles on the
OpenSky Discord channel #prc-data-competition.
