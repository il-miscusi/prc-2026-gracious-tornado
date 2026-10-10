# Data

Every input of the final submission, how it is obtained, its hash where the file is fixed, and the
terms it is used under. No input data is stored in this repository: the organiser files go in
`data/` (git-ignored), the EUROCONTROL files in `research/pru/` (git-ignored), and everything
else is fetched by the kernels or scripts named below.

Licence statements were read on each source's own pages on 2026-10-08. Where a source does not
state its terms clearly, this document says so.

## Licences at a glance

One line per external source; the sections below give the details.

| source | licence or terms | where stated |
|---|---|---|
| organiser files (PRC Data Challenge 2026) | no licence stated; the organiser's answer on licensing was "ask the owner" (Discord, 2026-09-26). Not redistributed: `tools/check_publish.py` refuses any `.parquet`, `.csv` or other data file not on its allow list | https://prc-data-challenge-2026.netlify.app/data.html |
| adsb.lol globe_history 2025 and 2026 | ODbL 1.0 | https://github.com/adsblol/globe_history_2025, https://opendatacommons.org/licenses/odbl/1-0/ |
| OpenStreetMap (Overpass API) | ODbL 1.0, © OpenStreetMap contributors | https://www.openstreetmap.org/copyright |
| X-Plane Scenery Gateway apt.dat | GPL v2; official Gateway application notice (verified 2026-10-10) | https://gateway.x-plane.com/ |
| OurAirports `runways.csv` | public domain | https://ourairports.com/data/ |
| OpenFlights `airlines.dat` | ODbL 1.0, contents under the Database Contents License | https://openflights.org/data.php |
| Iowa Environmental Mesonet METAR | public domain; attribution requested (re-read 2026-10-09) | https://mesonet.agron.iastate.edu/disclaimer.php |
| EUROCONTROL `taxi_out_additional_time_{2025,2026}.csv` (monthly) | EUROCONTROL terms: source named, non-commercial, no modification; not an open licence. Organiser ruling 2026-10-08: "(1) no change. still ok" | https://ansperformance.eu/about/disclaimer/ |
| EUROCONTROL `all_pre_departure_delays_{2025,2026}.csv` (daily) | same EUROCONTROL terms; no organiser ruling names this file | https://ansperformance.eu/about/disclaimer/ |
| EUROCONTROL airport traffic files | not used | – |

## Organiser files

PRC Data Challenge 2026 (EUROCONTROL Performance Review Commission and the OpenSky Network),
downloaded from the challenge bucket. Place them as below; `research/final/frame.py` reads these
exact paths.

| file | path | rows | sha256 |
|---|---|---:|---|
| training, January–December 2025 (12 files) | `data/09-22-2026-08-13-30_files_list/training_2025-MM-01_*.parquet` | 2,085,047 departures | listed below |
| board ranking file (Jan/Jul 2026) | `data/09-22-2026-08-13-30_files_list/ranking.parquet` | 344,841 departures | `9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86` |
| board template | `data/09-22-2026-08-13-30_files_list/submitting.parquet` | 344,841 | `0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b` |
| final ranking file (Jan/Feb/Jun/Jul 2026) | `data/final_20261008/final_ranking.parquet` | 1,341,197 movements, 670,790 departures | `4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f` |
| final template | `data/final_20261008/final_submitting.parquet` | 670,790 | `b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924` |

Training files (sha256): 2025-01 `a4dc8f152c082c650cd7eb745fe9c641089f34f74df127e6944a188990a0ed94`,
02 `b6f9ff2cdb1cfe07c5442f4e1f11bbcdbeafe52852dcb0b8175f62ab84c6a066`,
03 `58fde7e5f8b539546af63b41d7f6ec0f2884fdc3d3dd9b6bfca7c59a420dcdd3`,
04 `9ffe0cf5741c4863799c41be81475dfaecedabe0ba687b8f8e4eacba992f9390`,
05 `5481c0642a8fbb4dc395f93bc713c57f4cc16854fb361d404967077e1fce2ff9`,
06 `c3ef5080f2cc636c3416baf1555f7d0ee3519f9046e8043c1c0c8f017a244025`,
07 `34d1c5bba7c8596a1c20c09edd3cb2f63277b5a684fa801911660fe0eaf63c9d`,
08 `6c8d2aa7a1b6c9dff9a0305a37eb02de1e87a68d07557264f41b77738272c540`,
09 `955308a4000947e32a00586d3ee13928949d2231b67f84f6dfb0631893f3754a`,
10 `4a31a35601c1bdcd74b8cd6379231ba516a48324e4c2478bf1c9360cd23e687f`,
11 `ac36b17db7a537e3baeb682fa4763dacaa1fcc5d9bfa6bf32b3ee5d9f91300b0`,
12 `2d357f2f403a86778a69e93f72d8754effe77f1ac0942399b3d1b7728241f724`.

Final departures per month: January 152,719; February 144,158; June 181,791; July 192,122. The
final template holds all 344,841 board ids plus 325,949 new ones. The January/July rows of
`final_ranking.parquet` equal `ranking.parquet` except for a deliberate ±12 s dither of takeoff on
about 92% of movements (arrival taxi-in up to ±24 s); the organiser asked teams to build the final
from the final files only, and every final feature is computed from `final_ranking.parquet`.

The challenge pages state no licence for these files, so they are not redistributed. The
eligibility page requires "All used external datasets are openly accessible/usable and
documented."

## adsb.lol ADS-B history

Daily readsb globe-history releases, `github.com/adsblol/globe_history_2025` and
`globe_history_2026`, tag `vYYYY.MM.DD-planes-readsb-prod-0`.

Kaggle CPU kernels with internet stream each day's release and keep only aircraft on
the ground near the ten airports. `solutions/adsb/extract.py` (v1: one row per departure) and
`extract_v3.py` (v3: whole ground stays), then `events_v3.py` (graded off-block evidence). The
kernel folders are written by `solutions/adsb/build_k3.py` (2025, January/July 2026) and
`build_final.py` (February/June 2026, each half-month padded by a day on both sides so stays
that cross midnight at a month edge are seen whole). The v1 kernels mounted by the feature stores
are listed in `solutions/adsb/sources.json`.

adsb.lol published nothing for 2025-05-28 to 2025-06-10; those days are skipped.
Coverage depends on volunteer receivers. LIRF had none from January to April 2025. In the final
months a matched event exists for 79.9% of LIRF departures in January 2026, 74.0% in February,
0.0% in June and 0.4% in July; LTFM has 0–0.1% in every month; the other airports are listed in
`analysis/results/coverage.txt` (`analysis/coverage.py`).

Licence: ODbL 1.0. The adsb.lol documentation page states "License: ODbL 1.0", and both
GitHub repositories state that the database "is made available under the Open Database
License". The repositories also ship a CC0 file; the README names only ODbL, and we treat the
releases as ODbL. Our stays, events and feature columns are derivative databases under the ODbL.
They live in private Kaggle kernel outputs, not in this repository.

Extraction uses the daily tag `vYYYY.MM.DD-planes-readsb-prod-0` of
`adsblol/globe_history_YYYY`. The original release-asset hashes and sizes were not retained;
the extraction logs record processed days, not a complete asset receipt. This is a raw-source
provenance gap. The certified assembly replay starts from retained feature-derived inputs
and member predictions, rather than downloading those releases again.

## OpenStreetMap

`aeroway` features (parking positions, gates, holding positions, runways, taxiways,
taxilanes) within 7 km of each airport reference point, through the Overpass API.

`research/geometry/build_geometry.py`, run once on 2026-09-27; the build rejects any
Overpass answer whose database snapshot is older than 72 h. Output tables are published as the
private Kaggle dataset `prc-taxi-2026-geometry`, which the feature-store kernels mount.
`research/geometry/README.md` lists the columns and snapshot times.

Licence: ODbL 1.0, © OpenStreetMap contributors (openstreetmap.org/copyright). The
OSM-derived columns are ODbL.

## X-Plane Scenery Gateway

For each airport, the recommended Gateway scenery pack's apt.dat: ramp starts (name,
heading, size class), the taxi route network and runway ends. Pack ids, approval dates and
artists are in `research/geometry/README.md`.

Same build as OSM, through `gateway.x-plane.com/apiv1`.

Licence: GPL v2. The official Gateway application's account-deletion notice states that
scenery remains on the Gateway because its use is governed by GNU GPL version 2.
Verified 2026-10-10 from the vendor's [airport-page bundle](https://gateway.x-plane.com/build/airports.show.js?version=4e80786c);
`research/geometry/gateway_license_20261010.json` records its checksum and short statement.
No or-later grant is asserted. X-Plane-derived geometry columns are not redistributed here.

## OurAirports

`runways.csv` (runway end coordinates, length, width, displaced thresholds).

`research/geometry/build_geometry.py`, from
`davidmegginson.github.io/ourairports-data/runways.csv`.

Licence: Public domain: ourairports.com/data states that all data is released to the public
domain.

## OpenFlights

`airlines.dat`, used to fill IATA airline designators missing from the data-derived
map, so that callsigns can be matched across IATA and ICAO forms (store fs7).

Fetched inside the fs7 feature-store kernel from
`raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat`.

Licence: openflights.org/data states the database is made available under the Open Database
License, with contents under the Database Contents License; it adds that airline data derived
from Wikipedia may be subject to the GNU FDL. The GitHub repository's LICENSE file is AGPL-3.0
and does not separate code from data.

## Iowa Environmental Mesonet METAR

Routine and special METAR reports for the ten airports, 2025-01-01 to 2026-07-31:
temperature, dew point, wind, gusts, visibility, weather codes, cloud, pressure, precipitation.

`python3 research/build_metar.py OUT_DIR` (IEM ASOS download service, no login). The
output is published as the private Kaggle dataset `prc-taxi-2026-metar`. Version-1 `metar_prc2026.parquet` SHA256:
`b63ce71fef8b71333738ef8bb922c5c46004786a089175e3e1d6957866698dc2`
(downloaded and hashed 2026-10-10; 2,048,918 bytes).

Licence: IEM's disclaimer page states its data is in the public domain and may be used
freely by anyone for any lawful purpose; attribution to the Iowa Environmental Mesonet of Iowa
State University is requested.

## EUROCONTROL performance data

| file | used by | rows | sha256 of the copy used |
|---|---|---|---|
| `taxi_out_additional_time_2025.csv` | `research/final/level.py` | airport × month | `4aedf19520d3f84bebc418b56835762c907b714a9c19e751bb8658f9e62ce598` |
| `taxi_out_additional_time_2026.csv` | same (January–July 2026 for all ten airports) | airport × month | `b72040e0987e6c7d5706279d645ef9f814a6506165c83f89d8509022154a1076` |
| `all_pre_departure_delays_2025.csv` | `research/final/levers/daily.py` (fit) | airport × day | `d742eeb626e5f9f7e0c756e2ae3cf90078f4627e989a2e31b8ed21ba8f5510be` |
| `all_pre_departure_delays_2026.csv` | same (apply; January–August 2026) | airport × day | `5952d20ef5f9c4245018dd6986064c5c4aa15d568a6c0c2dcdcbc73bf8a43fd2` |

`research/pru/fetch_pru.sh` downloads all four from
`https://www.eurocontrol.int/performance/data/download/csv/<name>.csv` and checks the hashes.
The server copies were last modified on 2026-09-19 and had the same sizes as ours on
2026-10-08. EUROCONTROL updates these files in place, so a later download can differ.

Terms: The ansperformance.eu disclaimer allows reuse "provided that EUROCONTROL is mentioned
as the source", for non-commercial purposes, and says the material "may not be modified without
prior written permission from EUROCONTROL". It is not an open licence in the usual sense. We use
the files as published, never redistribute them, and credit EUROCONTROL in `NOTICE.md`.

Monthly: the published mean taxi-out time per airport and month, and the share
of flights excluded for de-icing. Daily: DLY_ALL_PRE_2 and FLT_DEP_IFR_2 per airport and day.
The fs7 store also writes the monthly values as columns, and every member drops them
(`docs/METHOD.md`).

**Organiser rulings** (Discord #prc-data-competition, by the organiser):

| date (UTC) | ruling |
|---|---|
| 2026-10-01 | "the rule about open data applies: anything you think that helps, use it!" |
| 2026-10-01 | "If no open license is present it is not open" |
| 2026-10-08 14:03 | on the monthly EUROCONTROL taxi-out aggregates for the scored months: "(1) no change. still ok" |

No ruling names the daily file. It comes from the same portal and terms as the monthly file. Our reading:
the 10-08 ruling accepts EUROCONTROL's terms for the monthly file, and the daily file is closer to
the labels, so it is its own step with its effect reported separately. The submitted file includes
it: in the 2025 rehearsal it is the largest single gain (nw −3,095.7 [−6,951.5, −838.9] pooled
MSE units, crossfit), and on 2025 month-out predictions it improved every month and every airport
(`docs/METHOD.md`, step 7). Without it the rehearsal's nw MSE would be 64,619.7 instead of
61,524.0 (`analysis/results/pseudo_steps_f5_pseudo2025_crossfit.json`, step `post2_recovered`).

## OPDI (not used)

The Open Performance Data Initiative's flight events were tested and rejected on coverage
(`docs/METHOD.md`). The organiser confirmed its licence as open on 2026-10-04. No OPDI data
reaches the submission.

## Our own intermediate data

Feature stores, ADS-B extracts and events, member predictions, the meta-model's frames (private
dataset `prc-taxi-2026-meta-f4`) and trained models live in private Kaggle datasets and kernel
outputs owned by the team's Kaggle accounts (`usiadianimuwa`, `miscusi`). `docs/REPRODUCE.md` lists
them and explains how to request access. They contain organiser data and ODbL-derived tables, so
they are not public. Store fs8 and the meta-model add no external data: every `g8_*` column and
every meta-model input is computed from the organiser files and the inputs above.

`research/tail/artifacts/longrows_inputs_over2h.parquet` is the only data file in this repository:
two earlier submissions' predictions (v21, v29) for the 10,168 board rows with d > 2 h, ids and
seconds only. It is needed only to rebuild the board file v31 (`docs/VALIDATION.md`).
