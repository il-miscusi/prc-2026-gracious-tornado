# Notices and attributions

The source code in this repository is licensed under the GNU General Public License v3 (`LICENSE`).
That licence covers the code only. The code downloads the external data below at run time; none of
that data, and no table derived from it, is stored in this repository. Anyone who runs the code and
publishes what it produces takes on the obligations listed here.

## adsb.lol historical ADS-B data

Source: adsb.lol, `github.com/adsblol/globe_history_2025` and `globe_history_2026` (daily releases
`vYYYY.MM.DD-planes-readsb-prod-0`). The data is made available under the Open Database License
(ODbL) 1.0: https://opendatacommons.org/licenses/odbl/1-0/

Used by `solutions/adsb/extract.py`, `extract_v3.py` and `events_v3.py`. The ground stays, events
and feature columns these scripts write are derivative databases under the ODbL. If you publish
them, publish them under the ODbL and credit adsb.lol.

## OpenStreetMap

Contains information from OpenStreetMap (© OpenStreetMap contributors), made available under the
Open Database License (ODbL) 1.0: https://www.openstreetmap.org/copyright

Used by `research/geometry/build_geometry.py` (stand positions, gates, holding points, taxiways,
runways, fetched through the Overpass API). The OSM-derived columns of the geometry tables are
ODbL.

## OpenFlights

Airline IATA-to-ICAO aliases from OpenFlights `airlines.dat`
(https://openflights.org/data.php), made available under the Open Database License (ODbL) 1.0,
with contents under the Database Contents License. OpenFlights notes that airline data taken from
Wikipedia may also fall under the GNU Free Documentation License. Fetched at run time by
`solutions/core/features.py` (feature store fs7).

## OurAirports

Runway ends and dimensions from OurAirports `runways.csv` (https://ourairports.com/data/),
released to the public domain. Used by `research/geometry/build_geometry.py`.

## X-Plane Scenery Gateway

Airport layouts (apt.dat ramp starts, taxi route network, runway ends) from the X-Plane Scenery
Gateway (https://gateway.x-plane.com/), under GNU GPL version 2 as stated in the Gateway's
own application notice (checked 2026-10-10; evidence in `docs/DATA.md`). Gateway artists of
the ten packs used are listed in `research/geometry/README.md`. No or-later grant is asserted.
X-Plane-derived geometry columns are not redistributed here.

## Iowa Environmental Mesonet

METAR observations from the Iowa Environmental Mesonet (IEM) of Iowa State University
(https://mesonet.agron.iastate.edu/), which states that its data is in the public domain. Fetched
by `research/build_metar.py`.

## EUROCONTROL

Source: EUROCONTROL, Aviation Intelligence Portal / ansperformance.eu (taxi-out additional time,
monthly; all pre-departure delays, daily). These files are used under EUROCONTROL's terms
(https://ansperformance.eu/about/disclaimer/): reuse with EUROCONTROL named as the source, for
non-commercial purposes, without modification. They are not redistributed here;
`research/pru/fetch_pru.sh` downloads them and checks their hashes.

## PRC Data Challenge 2026

The training and ranking files are the organisers' (EUROCONTROL Performance Review Commission and
the OpenSky Network). No licence for them is stated on the challenge pages, and they are not
redistributed here.
