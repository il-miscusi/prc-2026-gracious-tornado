# PRC 2026 airport geometry

Stand positions, runway ends and stand-to-runway taxi paths for the ten departure airports of the
PRC Data Challenge 2026 (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM, LSZH). Every input
is openly licensed, so the tables qualify under the challenge's external-data rule. Built
2026-09-27 by `build_geometry.py`; checked against Jan + Jul 2025 departures by `check_value.py`.

Kaggle (private): `usiadianimuwa/prc-taxi-2026-geometry`.

## Sources and licences

| source | what it gives | licence | retrieved |
|---|---|---|---|
| OpenStreetMap via Overpass API | `aeroway` = parking_position (with `ref`), gate, holding_position, runway, taxiway, taxilane, in a +-7 km box around each ARP | ODbL 1.0, (c) OpenStreetMap contributors | 2026-09-27; database snapshots 2026-09-27 10:27Z to 10:57Z (maps.mail.ru and overpass-api.de mirrors) |
| X-Plane Scenery Gateway (`gateway.x-plane.com/apiv1`), recommended scenery per airport | apt.dat ramp starts (rows 1300/1301: name, heading, ICAO size class), taxi route network (1201/1202), runway ends (100) | GPL v2 | 2026-09-27 |
| OurAirports `runways.csv` | runway end coordinates, length, width, displaced thresholds | public domain | 2026-09-27 |

X-Plane scenery packs used (Gateway scenery id, approval date, artist): EDDF 103368 (2024-08-16,
meikelm), EDDM 111124 (2026-06-10, Astix10), EGLL 111341 (2026-07-01, Litjan), EHAM 111943
(2026-09-09, KarsCD), LEBL 102255 (2024-05-13, pedromagueija), LEMD 99851 (2023-12-26,
bastienvdc), LFPG 111022 (2026-05-26, Julian), LIRF 110482 (2026-04-17, Astix10), LSZH 106102
(2025-03-29, Astix10), LTFM 105504 (2025-01-30, Julian).

The derived tables inherit the share-alike terms: OSM-derived columns (`osm_*`, and `lat`/`lon`
when `source` = osm) are ODbL; X-Plane-derived columns (`xp_*`, `xplane_*`) are GPL v2. No
OpenSky, FlightRadar or other non-open source is used anywhere.

Overpass mirrors differ: overpass.private.coffee and overpass.kumi.systems served database
snapshots from 2026-05-31 to 2026-07-24. The build rejects any answer whose `timestamp_osm_base`
is older than 72 h, so every airport comes from a same-day snapshot.

The Gateway's own application states that scenery use is governed by GNU GPL version 2
(account-deletion notice, checked 2026-10-10). See `gateway_license_20261010.json`
for the vendor bundle URL, checksum and short statement. An or-later grant is not asserted.

## Files

All parquet. `apt` is the ICAO code; `stand` and `runway` are the exact strings of `STAND_mvt`
and `RUNWAY_mvt` (so they join on `(ADEP_mvt, STAND_mvt, RUNWAY_mvt)` without any processing).

### `stands.parquet` (2,771 rows: one per airport and `STAND_mvt` value seen in 2025 training or 2026 ranking departures)

| column | meaning |
|---|---|
| apt, stand | join keys |
| key | canonical key used for matching (see normalisation) |
| n_dep_2025, n_dep_2026 | departures from this stand in training 2025 / ranking 2026 |
| lat, lon | chosen stand position (WGS84) |
| source | osm / xplane / osm_gate / none |
| match_rule | exact_osm, exact_xplane, base_*, children_*, neighbour_*, gate, none |
| exact | match_rule starts with exact |
| heading, heading_source | aircraft nose heading when parked, degrees true (xplane ramp heading; osm_leadin = direction of the OSM lead-in line towards the stop point; fallback_match) |
| osm_lat/lon/heading/label/id/places/resolved_by | the exact OSM match, if any (`places` = distinct places > 150 m apart that carry this key; `resolved_by` says how one was picked) |
| xp_lat/lon/heading/label/id/places/resolved_by, xp_icao_cat, xp_ramp_type | the exact X-Plane match (ICAO size class A-F, gate / tie_down / misc) |
| gate_* | an OSM `aeroway=gate` with the same ref (terminal door, not the stand) |
| osm_xp_diff_m | distance between the OSM and X-Plane positions of the same code |
| pos_conflict | osm_xp_diff_m > 150 m: the two sources place the code on different aprons |
| matched_label, matched_id | source label and feature id(s) behind `lat`/`lon` |

### `runways.parquet` (78 rows: one per runway end)

`runway` (designator as in `RUNWAY_mvt`), `opposite`, `thr_lat`/`thr_lon` (the runway end a
departure starts from), `end_lat`/`end_lon` (the far end), `heading_true`, `length_m`, `width_m`,
`centreline_m`, `displaced_thr_m`, `thr_source` (ourairports or xplane), plus the raw OurAirports
(`oa_*`) and X-Plane (`xp_*`) coordinates, their disagreement `oa_xp_diff_m`, and each one's
distance to the nearest OSM runway way (`oa_to_osm_runway_m`, `xp_to_osm_runway_m`,
`thr_to_osm_runway_m`).

The end coordinates are the physical runway ends; a displaced threshold is usable for take-off, so
the departure starts there. OurAirports is the source of record. X-Plane replaces it when
OurAirports has no coordinates, or when the two disagree by more than 30 m and X-Plane lies closer
to the OSM runway: that happened for EHAM 24 (48 m), LEMD 18R (181 m), LFPG 26R (74 m; the
OurAirports point is past the end of the paved runway) and LIRF 07 (113 m). Median distance of the
chosen ends to the OSM runway ways is 2 m.

`RUNWAY_mvt` values with no runway: `NA` (a few rows at 5 airports), EHAM `HEL` (40), LSZH `H`
(466), LIRF `01R` (6).

### `stand_runway.parquet` (12,458 rows: one per (apt, stand, runway) seen in 2025 training or 2026 ranking departures)

| column | meaning |
|---|---|
| n_dep_2025, n_dep_2026 | departures on this pair |
| straight_m | great-circle distance stand -> departure runway end |
| path_m | chosen taxi path length, metres = snap_m + graph_m + backtrack_m |
| graph_m | metres along the taxi network from the stand's snap node to the runway entry |
| backtrack_m | metres along the runway from that entry back to the departure end (0 when the network meets the runway at the end) |
| snap_m | straight leg from the stand position to the nearest node of the network |
| crossings, crossed | number and list of OTHER runways whose centreline the path crosses |
| path_source | osm or xplane (which network `path_m` comes from) |
| path_over_straight | path_m / straight_m |
| osm_*, xplane_* | the same quantities on each network separately (`*_runway_share` = share of path nodes on runway pavement) |

### Other files

`stand_features_raw.parquet` (every stand-like feature in both sources with its parsed key, for
re-matching without re-fetching), `coverage.csv` (coverage by departure rows, below), `graphs.csv`
(network sizes), `value_check.csv` (the check below). `build_geometry.py` rebuilds everything;
`publish_kaggle.py` uploads a new dataset version.

## Stand normalisation

Both sides reduce to one canonical key: upper case, alphanumerics only, and leading zeros removed
from every digit run (`B02` -> `B2`, `H01` -> `H1`, `021` -> `21`). This alone reconciles the
zero-padded codes (EHAM, LFPG, LSZH data write `H01`, `B02`, `A02`; some sources write `H1`).

Source labels (OSM `ref`, then `local_ref`, then `name`; X-Plane ramp names) are parsed first:

- words that are not part of a code are dropped: Stand, Gate, Ramp, Remote, Parking, Position,
  Apron, Cargo, Hangar and a few local equivalents (LEBL X-Plane "Gate 101", "Ramp 01"; LSZH
  "Stand D01");
- parenthesised text without digits is dropped: LTFM X-Plane names carry the ICAO size class
  (`133L-(C)`), EGLL has `323(constr)`;
- a letter prefix separated by a space is joined (`B 41` -> B41, `C 15a` -> C15A, LSZH
  `Gate E 19` -> E19), and so is a trailing L/R/A/B (`511 L` -> 511L);
- lists split into several stands: `A14;A14A`, `221L/221R`, `G10L/R` -> G10L + G10R, `A26 A28`,
  `X06/X07`;
- a label of 2-4 letters is kept as a code (LSZH `MFG`), lone letters are not.

When one key names places more than 150 m apart in the same source (LEBL: OSM carries two aprons
both numbered 95-129), the place is chosen by the other source's position for that key, then by
the numeric neighbours' positions, then by the larger cluster. Features of one key within 150 m
(MARS lead-in lines, a node and a way for the same stand) are averaged.

Each `STAND_mvt` code then takes the first rule that finds a position, OSM before X-Plane within
each rule:

1. `exact`: the keys are equal.
2. `base`: drop a trailing sub-stand letter (EDDF `D4A` -> D4, LEBL `151A` -> 151, LFPG `S6B` -> S6).
3. `children`: the sources only have the split stand (EDDM writes `210` where OSM and X-Plane have
   `210A`/`210B`; EDDM 7xx/8xx have A-D): mean of the children.
4. `neighbour`: same prefix, number +-1 or +-2 (EDDF `V153` -> V155, LIRF `801` -> 802). A guess,
   within one apron row; flagged by `match_rule`.
5. `gate`: an OSM `aeroway=gate` with that ref (used for one EDDM code with one departure).

What each airport needed (share of 2025 departure rows):

| apt | data convention | exact OSM | exact X-Plane | fallback | none | notes |
|---|---|---|---|---|---|---|
| EDDF | `A14`, `V108`, `D4A` | 96.4 | 2.0 | 0.8 | 0.8 | OSM writes some refs with a space (`V 134`); none: V151/V152, V333-V337, X1xxM |
| EDDM | `210`, `302` | 57.4 | 12.9 | 29.7 | 0.0 | MARS stands exist only as A/B(/X) in both sources (children); remote 5xx, 7xx, 8xx from X-Plane |
| EGLL | `511`, `221L` | 98.4 | 0.7 | 0.9 | 0.0 | 237/237L/237R -> neighbour 238 |
| EHAM | `H01`, `D54` | 94.6 | 4.7 | 0.0 | 0.7 | zero padding; none: HG01/HG03/HG32 (1.9k rows), K01/K02 |
| LEBL | `268`, `96`, `151A` | 89.7 | 9.0 | 0.5 | 0.9 | `UNKNOWN` (1,709 rows) has no position; duplicate 1xx aprons in OSM |
| LEMD | `352`, `T13`, `72` | 98.8 | 0.6 | 0.3 | 0.4 | none: 25, 26 |
| LFPG | `F32`, `B02` | 96.9 | 3.0 | 0.0 | 0.0 | X-Plane fills E10/E13/E44 and I35-I79 |
| LIRF | `308` | 99.0 | 0.1 | 0.7 | 0.2 | none: 108 |
| LSZH | `A02`, `E58`, `313` | 96.4 | 0.0 | 0.0 | 3.6 | none: GA6, MFG, REG, HAN, HPW (general-aviation and hangar areas, not stands in any source) |
| LTFM | `G11L`, `F4R`, `E1` | 99.2 | 0.7 | 0.0 | 0.1 | none: K25-K27 |

Position checks: where OSM and X-Plane both have a code, the median distance between them is 2-15 m
per airport and headings agree to a median 0.5 degrees (94% within 20 degrees). Codes the two
sources place > 150 m apart (`pos_conflict`) cover 2.5% of LSZH rows (101-105: OSM and X-Plane
are 1.2-1.4 km apart; OSM is kept), 0.4% of EDDM (103-105) and 0.2% of EHAM (Y73); nothing
elsewhere.

## Taxi paths

Two networks per airport, both undirected, densified to 15 m segments, in a local metric
projection around the ARP:

- **OSM**: taxiway + taxilane ways, runway ways (travel along them costs 4x per metre, so they are
  crossed rather than used), and parking_position lead-in lines. Dead ends within 12 m of another
  component are joined (OSM mapping gaps). The largest component holds 98-100% of nodes.
- **X-Plane**: the apt.dat taxi route network (runway edges treated the same way).

The departure end of runway R is reached through a virtual sink joined to every network node that
lies on R's centreline (within 35 m, from 150 m behind the runway end to the far end). The sink
edge costs 1.5x the along-runway distance from that entry back to the end, so the search prefers
the parallel taxiway to the end but still finds intersection-only layouts. `path_m` is the
full-length departure distance: snap leg + taxi metres + runway metres back to the end. One
Dijkstra per (airport, runway, network); the stand snaps to the nearest node of the largest
component.

`path_source` is OSM when the stand is within 60 m of the OSM network, otherwise whichever network
the stand is nearer to (up to 300 m). X-Plane supplies 0.6% of 2025 rows and 0.8% of 2026 rows,
mostly LEBL remote stands (6.3% of LEBL rows) and a few EDDM stands.

The two networks agree: median OSM/X-Plane path ratio 0.96-1.01 per airport; more than 25% apart
on 0-8% of pairs (EDDF 8%, mostly rarely used pairs).

**Crossings** count the other runways whose centreline (end to end, helicopter strips excluded)
the path crosses. Share of 2025+2026 departures whose path crosses at least one runway: EDDF 4%,
EDDM 0%, EGLL 3%, EHAM 47%, LEBL 79% (mostly runway 02/20, which is rarely active), LEMD 34%,
LFPG 12%, LIRF 7%, LSZH 25%, LTFM 37%. OSM and X-Plane give the same count on >= 99% of rows
except LEBL (83%) and LEMD (80%): there the paths pass near a runway END (LEMD 14R, LEBL 02) and one
network cuts the corner. `osm_crossed` and `xplane_crossed` are both kept.

## Coverage by departure rows

Share of departure ROWS (not distinct stands) with a stand position, with an exact code match,
and with a taxi path; last column is the share of 2026 rows whose path comes from X-Plane.

| apt | 2025 rows | stand % | exact % | path % | 2026 rows | stand % | exact % | path % | X-Plane path % (2026) |
|---|---|---|---|---|---|---|---|---|---|
| EDDF | 230,141 | 99.2 | 98.3 | 99.2 | 36,317 | 98.9 | 97.8 | 98.9 | 0.0 |
| EDDM | 167,334 | 100.0 | 70.3 | 100.0 | 25,966 | 99.9 | 70.7 | 99.9 | 2.8 |
| EGLL | 239,546 | 100.0 | 99.1 | 100.0 | 39,840 | 100.0 | 98.9 | 100.0 | 0.0 |
| EHAM | 247,951 | 99.3 | 99.3 | 99.3 | 38,182 | 99.4 | 99.4 | 99.4 | 0.2 |
| LEBL | 179,705 | 99.1 | 98.6 | 99.1 | 30,080 | 99.1 | 98.4 | 99.1 | 6.3 |
| LEMD | 212,242 | 99.6 | 99.3 | 99.6 | 36,954 | 99.6 | 99.3 | 99.6 | 0.0 |
| LFPG | 239,552 | 100.0 | 100.0 | 100.0 | 39,872 | 100.0 | 100.0 | 100.0 | 0.0 |
| LIRF | 160,704 | 99.8 | 99.0 | 99.8 | 26,899 | 99.8 | 98.7 | 99.8 | 0.0 |
| LSZH | 134,907 | 96.4 | 96.4 | 96.4 | 23,150 | 96.7 | 96.5 | 96.7 | 0.0 |
| LTFM | 272,965 | 99.9 | 99.9 | 99.9 | 47,581 | 99.7 | 99.7 | 99.7 | 0.0 |
| all | 2,085,047 | 99.5 | 96.7 | 99.5 | 344,841 | 99.4 | 96.8 | 99.4 | 0.8 |

EDDM's exact share is low only because its codes are the parents of split MARS stands; every EDDM
row has a position.

## Value check (Jan + Jul 2025 departures, y = TAXITIME_SEC_mvt, 0 < y <= 1800 s)

`r` = Pearson with y; `within rwy` = both path and y demeaned per runway (does geometry add
anything the runway alone does not?); `rare` = pairs with fewer than 30 departures in all of 2025,
where a stand x runway target encoding has little to learn from.

| apt | rows | r path | Spearman path | r straight | r path within rwy | r crossings | r path on rare pairs (rows) | rare pairs with geometry / rare pairs | 2026 rows on rare pairs with geometry | pairs unseen in 2025 with geometry / their 2026 rows |
|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 36,135 | 0.54 | 0.57 | 0.50 | 0.54 | 0.17 | 0.37 (235) | 443 / 474 | 2,623 (7.2%) | 57 / 760 |
| EDDM | 26,506 | 0.26 | 0.29 | 0.26 | 0.22 | - | 0.34 (201) | 122 / 175 | 1,352 (5.2%) | 25 / 1,178 |
| EGLL | 35,422 | 0.36 | 0.35 | 0.33 | 0.36 | 0.08 | 0.14 (116) | 222 / 239 | 224 (0.6%) | 61 / 71 |
| EHAM | 40,194 | 0.64 | 0.63 | 0.63 | 0.26 | 0.43 | 0.47 (368) | 569 / 645 | 1,354 (3.5%) | 10 / 18 |
| LEBL | 28,375 | 0.37 | 0.46 | 0.29 | 0.38 | 0.47 | 0.50 (293) | 312 / 324 | 490 (1.6%) | 39 / 61 |
| LEMD | 34,715 | 0.55 | 0.56 | 0.53 | 0.52 | 0.38 | 0.66 (923) | 552 / 557 | 1,389 (3.8%) | 35 / 62 |
| LFPG | 37,311 | 0.41 | 0.40 | 0.40 | 0.40 | 0.12 | 0.50 (1,596) | 1,186 / 1,202 | 1,869 (4.7%) | 38 / 45 |
| LIRF | 23,807 | 0.36 | 0.37 | 0.36 | 0.29 | 0.20 | 0.13 (512) | 307 / 341 | 536 (2.0%) | 16 / 24 |
| LSZH | 21,985 | 0.42 | 0.43 | 0.40 | 0.33 | 0.08 | 0.34 (460) | 425 / 472 | 741 (3.2%) | 31 / 76 |
| LTFM | 44,257 | 0.38 | 0.39 | 0.33 | 0.32 | 0.32 | 0.56 (1,272) | 1,576 / 1,593 | 2,057 (4.3%) | 50 / 55 |

In total 5,714 of the 6,022 rare pairs have geometry, carrying 12,635 ranking rows (3.7%); 362
pairs never seen in 2025 carry 2,350 ranking rows (EDDM alone 1,178). Path length beats
straight-line distance everywhere, and keeps most of its correlation after removing the runway
mean, so it is not just a runway proxy.

## Caveats

- Networks are undirected: one-way taxiways (X-Plane marks them) and actual ATC routings are
  ignored. `path_m` is a geometric lower bound on the usual route, full length to the runway end;
  intersection departures are not modelled.
- `neighbour` matches are guesses within an apron row; `pos_conflict` rows have one source wrong.
- Crossings are geometric: they say the path crosses the runway, not that the runway was active.
- OSM ways change; rebuilding later gives a different snapshot. The cache (`cache/`, not
  committed) holds the exact responses used here.

## Rebuild

```bash
cd competitions/independent-predict-aircraft-taxi-out-time-at-european-airports/research/geometry
python3 build_geometry.py            # fetch what is missing into cache/, then build (~15 s once cached)
python3 build_geometry.py --refresh  # re-download all sources
python3 check_value.py               # correlation table -> value_check.csv
python3 publish_kaggle.py --notes "..."   # new version of the private Kaggle dataset
```
