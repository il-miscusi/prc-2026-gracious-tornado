#!/usr/bin/env python3
"""Airport geometry for the PRC Data Challenge 2026 (taxi-out time at 10 European airports).

Builds, for every departure airport in the challenge data:

* ``stands.parquet``            one row per (airport, stand code as written in ``STAND_mvt``)
                                with coordinates, heading, source and match flags;
* ``stand_features_raw.parquet`` every stand-like feature found in OSM and X-Plane, with the
                                normalised key used for matching;
* ``runways.parquet``           one row per (airport, runway end) with threshold, heading, length;
* ``stand_runway.parquet``      one row per (airport, stand, departure runway) pair present in
                                the 2025 training or 2026 ranking departures: straight-line
                                distance, taxi-graph path length (OSM and X-Plane networks),
                                runway crossings;
* ``coverage.csv``              coverage by departure rows, per airport and data split.

Sources, all openly licensed (see README.md):
  OpenStreetMap via Overpass (ODbL 1.0), X-Plane Scenery Gateway apt.dat (GPL v2 or later),
  OurAirports runways.csv (public domain).

Raw downloads are cached under ``cache/`` (not committed). Runs locally in well under a minute
once the cache is warm; the first run is dominated by Overpass.

    python3 build_geometry.py              # fetch what is missing, then build
    python3 build_geometry.py --refresh    # re-download everything
    python3 build_geometry.py --no-fetch   # build from the cache only
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

HERE = Path(__file__).resolve().parent
COMP = HERE.parents[1]
DATA = COMP / "data" / "09-22-2026-08-13-30_files_list"
CACHE = HERE / "cache"

# Aerodrome reference points (ICAO AIP).
AIRPORTS = {
    "EDDF": (50.0333, 8.5706),
    "EDDM": (48.3538, 11.7861),
    "EGLL": (51.4706, -0.4619),
    "EHAM": (52.3086, 4.7639),
    "LEBL": (41.2971, 2.0785),
    "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479),
    "LIRF": (41.8003, 12.2389),
    "LTFM": (41.2753, 28.7519),
    "LSZH": (47.4647, 8.5492),
}
RADIUS_M = 7000

UA = {"User-Agent": "prc-taxi-2026-geometry/1.0 (open-data airport geometry build; python-requests)"}
# Mirror order is by what answered on 2026-09-27: overpass-api.de and private.coffee returned
# HTTP 504 for every airport-sized query while maps.mail.ru served them in ~1-2 minutes.
OVERPASS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
MAX_SNAPSHOT_AGE_H = 72
XP_API = "https://gateway.x-plane.com/apiv1"
OURAIRPORTS = "https://davidmegginson.github.io/ourairports-data/runways.csv"

R_EARTH = 6371008.8
DENSIFY_M = 15.0          # max edge length after densification (snap error <= 7.5 m)
HEAL_M = 12.0             # join a dead-end taxi node to another component within this distance
RUNWAY_COST = 4.0         # cost multiplier for travelling along a runway edge (crossings stay cheap)
ENTRY_HALF_WIDTH_M = 35.0  # a taxi node this close to the departure centreline is a runway entry
ENTRY_OVERRUN_M = 150.0   # entries may sit this far behind the threshold (turn pads, blast pads)
BACKTRACK_COST = 1.5      # cost per metre of runway between the entry and the threshold (the length
                          # reported is unweighted); >1 prefers the parallel taxiway to the end


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------
# Fetch
# ----------------------------------------------------------------------------------------------

def overpass(query: str, rounds: int = 6) -> dict:
    last = None
    for rnd in range(rounds):
        for url in OVERPASS:
            try:
                r = requests.post(url, data={"data": query}, headers=UA, timeout=(20, 420))
            except requests.RequestException as e:
                last = f"{url}: {type(e).__name__}"
                log(f"  overpass {last}")
                continue
            if r.status_code == 200:
                try:
                    j = r.json()
                except ValueError:
                    last = f"{url}: non-JSON 200"
                    log(f"  overpass {last}")
                    continue
                if "remark" in j and "runtime error" in str(j.get("remark", "")):
                    last = f"{url}: {j['remark'][:120]}"
                    log(f"  overpass {last}")
                    continue
                base = j.get("osm3s", {}).get("timestamp_osm_base", "")
                age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(base.replace("Z", "+00:00"))
                         ).total_seconds() / 3600 if base else 1e9
                if age_h > MAX_SNAPSHOT_AGE_H:
                    # some mirrors serve months-old databases (private.coffee answered with
                    # 2026-05-31 and 2026-07-15 snapshots); mixing those with current data is not
                    # reproducible, so a stale answer counts as a failure
                    last = f"{url}: stale snapshot {base}"
                    log(f"  overpass {last}")
                    continue
                j["_endpoint"] = url
                return j
            last = f"{url}: HTTP {r.status_code}"
            log(f"  overpass {last}")
        wait = 20 * (rnd + 1)
        log(f"  overpass round {rnd + 1} failed; waiting {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"overpass failed: {last}")


def fetch_osm(icao: str, refresh: bool) -> Path:
    """Two Overpass queries per airport over a bounding box of +-RADIUS_M around the ARP.

    A bbox is far cheaper for the servers than ``around:`` and they were overloaded when this was
    built. Each query is cached on its own so an interrupted run resumes where it stopped.
    """
    out = CACHE / f"osm_{icao}.json"
    if out.exists() and not refresh:
        return out
    lat, lon = AIRPORTS[icao]
    dlat = RADIUS_M / (R_EARTH * math.radians(1))
    dlon = dlat / math.cos(math.radians(lat))
    bbox = f"({lat - dlat:.5f},{lon - dlon:.5f},{lat + dlat:.5f},{lon + dlon:.5f})"
    queries = {
        "points": f'[out:json][timeout:300];nwr["aeroway"~"^(parking_position|gate|holding_position|runway)$"]{bbox};out body geom;',
        "taxi": f'[out:json][timeout:300];way["aeroway"~"^(taxiway|taxilane)$"]{bbox};out body geom;',
    }
    elements, meta = [], []
    for name, q in queries.items():
        part = CACHE / f"osm_{icao}_{name}.json"
        if part.exists() and not refresh:
            j = json.loads(part.read_text())
        else:
            t0 = time.time()
            j = overpass(q)
            j["_seconds"] = round(time.time() - t0, 1)
            part.write_text(json.dumps(j))
            log(f"  {icao} OSM {name}: {len(j['elements']):,} elements in {j['_seconds']:.0f}s via {j['_endpoint']}")
        elements += j["elements"]
        meta.append({"query": q, "endpoint": j.get("_endpoint"),
                     "timestamp_osm_base": j.get("osm3s", {}).get("timestamp_osm_base"),
                     "n": len(j["elements"]), "seconds": j.get("_seconds")})
    doc = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "licence": "ODbL 1.0, (c) OpenStreetMap contributors", "requests": meta, "elements": elements}
    out.write_text(json.dumps(doc))
    for name in queries:
        (CACHE / f"osm_{icao}_{name}.json").unlink(missing_ok=True)
    return out


def fetch_xplane(icao: str, refresh: bool) -> Path:
    out = CACHE / f"xp_{icao}.dat"
    if out.exists() and not refresh:
        return out
    a = requests.get(f"{XP_API}/airport/{icao}", headers=UA, timeout=60)
    a.raise_for_status()
    ap = a.json().get("airport", {})
    sid = ap["recommendedSceneryId"]
    s = requests.get(f"{XP_API}/scenery/{sid}", headers=UA, timeout=180)
    s.raise_for_status()
    sc = s.json().get("scenery", {})
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(sc["masterZipBlob"])))
    name = next(n for n in z.namelist() if n.upper() == f"{icao}.DAT")
    out.write_bytes(z.read(name))
    meta = {k: sc.get(k) for k in ("sceneryId", "icao", "aptName", "userName", "dateUploaded",
                                   "dateApproved", "type", "artistComments")}
    meta["retrieved_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["licence"] = "GPL v2 (X-Plane Scenery Gateway)"
    (CACHE / f"xp_{icao}_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"  {icao} X-Plane scenery {sid} ({sc.get('dateApproved')}) {out.stat().st_size:,} bytes")
    return out


def fetch_ourairports(refresh: bool) -> Path:
    out = CACHE / "ourairports_runways.csv"
    if out.exists() and not refresh:
        return out
    r = requests.get(OURAIRPORTS, headers=UA, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df = df[df["airport_ident"].isin(AIRPORTS)]
    df.to_csv(out, index=False)
    (CACHE / "ourairports_meta.json").write_text(json.dumps({
        "url": OURAIRPORTS, "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "licence": "public domain (OurAirports)"}, indent=1))
    log(f"  OurAirports: {len(df)} runways at the 10 airports")
    return out


# ----------------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------------

class Proj:
    """Local equirectangular projection around the ARP (metres; error < 0.1% inside 8 km)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = R_EARTH * math.radians(1) * math.cos(math.radians(lat0))
        self.ky = R_EARTH * math.radians(1)

    def xy(self, lat, lon):
        return (np.asarray(lon, float) - self.lon0) * self.kx, (np.asarray(lat, float) - self.lat0) * self.ky

    def ll(self, x, y):
        return self.lat0 + np.asarray(y, float) / self.ky, self.lon0 + np.asarray(x, float) / self.kx


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


# ----------------------------------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------------------------------

def parse_osm(path: Path) -> dict:
    doc = json.loads(path.read_text())
    stands, gates, holds, runways, taxi = [], [], [], [], []
    for e in doc["elements"]:
        t = e.get("tags", {})
        aw = t.get("aeroway")
        if e["type"] == "node":
            rec = {"osm_id": f"n{e['id']}", "lat": e["lat"], "lon": e["lon"], "tags": t}
            if aw == "parking_position":
                stands.append({**rec, "kind": "node"})
            elif aw == "gate":
                gates.append(rec)
            elif aw == "holding_position":
                holds.append(rec)
            continue
        if e["type"] == "way":
            geom = e.get("geometry") or []
            if len(geom) < 2:
                continue
            lat = np.array([g["lat"] for g in geom])
            lon = np.array([g["lon"] for g in geom])
            rec = {"osm_id": f"w{e['id']}", "nodes": e.get("nodes", []), "lat": lat, "lon": lon, "tags": t}
            closed = len(rec["nodes"]) > 3 and rec["nodes"][0] == rec["nodes"][-1]
            area = closed and t.get("area") == "yes"
            if aw == "parking_position":
                stands.append({**rec, "kind": "area" if closed else "way"})
            elif aw == "gate":
                gates.append({**rec, "lat": float(lat.mean()), "lon": float(lon.mean())})
            elif aw == "holding_position":
                holds.append({**rec, "lat": float(lat.mean()), "lon": float(lon.mean())})
            elif aw == "runway" and not area:
                runways.append(rec)
            elif aw in ("taxiway", "taxilane") and not area:
                taxi.append({**rec, "kind": aw})
        # relations (multipolygon aprons/runway areas) carry no network information; skipped
    return {"stands": stands, "gates": gates, "holds": holds, "runways": runways, "taxi": taxi,
            "retrieved_utc": doc.get("retrieved_utc"),
            "osm_base": max((m.get("timestamp_osm_base") or "") for m in doc.get("requests", [{}]))}


def parse_xplane(path: Path) -> dict:
    ramps, nodes, edges, runways = [], {}, [], []
    last_ramp = None
    for line in path.read_text(errors="replace").splitlines():
        f = line.split()
        if not f:
            continue
        code = f[0]
        if code == "1300" and len(f) >= 6:
            last_ramp = {"lat": float(f[1]), "lon": float(f[2]), "heading": float(f[3]) % 360,
                         "ramp_type": f[4], "ac_types": f[5], "name": " ".join(f[6:]).strip(),
                         "icao_cat": None, "op_type": None, "airlines": None}
            ramps.append(last_ramp)
        elif code == "1301" and last_ramp is not None and len(f) >= 3:
            last_ramp["icao_cat"], last_ramp["op_type"] = f[1], f[2]
            last_ramp["airlines"] = " ".join(f[3:]) or None
        elif code == "1201" and len(f) >= 5:
            nodes[int(f[4])] = (float(f[1]), float(f[2]))
        elif code == "1202" and len(f) >= 4:
            kind = f[5] if len(f) > 5 else "taxiway"
            edges.append({"a": int(f[1]), "b": int(f[2]), "oneway": f[3], "kind": kind,
                          "name": " ".join(f[6:]) if len(f) > 6 else ""})
        elif code == "100" and len(f) >= 26:
            for off in (8, 17):
                runways.append({"ident": f[off], "lat": float(f[off + 1]), "lon": float(f[off + 2]),
                                "displaced_m": float(f[off + 3]), "width_m": float(f[1])})
            runways[-2]["other"], runways[-1]["other"] = runways[-1]["ident"], runways[-2]["ident"]
    return {"ramps": ramps, "nodes": nodes, "edges": edges, "runways": runways}


# ----------------------------------------------------------------------------------------------
# Stand code normalisation
# ----------------------------------------------------------------------------------------------
# A key is an upper-case alphanumeric string where every run of digits has its leading zeros
# removed ("B02" -> "B2", "007" -> "7"). This single canonical form makes zero padding
# irrelevant on both sides. Source labels are parsed into keys by ``source_keys``; the spelling
# conventions of all ten airports turned out to be covered by these general rules plus the
# fallbacks in ``match_stands`` (README.md lists what each airport needed).

_WORDS = re.compile(r"\b(STAND|GATE|PARKING|POSITION|POS|STANDPLATZ|PUERTA|PORTA|PISTE|REMOTE|"
                    r"RAMP|APRON|CARGO|HANGAR|PARK|PLATZ|POSTEGGIO|PLAZA|PIAZZALE|SPOT)\b\.?")


def canon(code) -> str:
    """Canonical key: upper case, alphanumerics only, leading zeros stripped from digit runs."""
    s = re.sub(r"[^0-9A-Z]", "", str(code).upper())
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", s)


def source_keys(icao: str, raw: str) -> list[str]:
    """All keys a source label (OSM ref/name, X-Plane ramp name) may stand for.

    A label can name one stand written with a space ("B 41", "C 15a", "511 L") or several
    stands ("A14;A14A", "221L/221R", "G10L/R", "A26 A28", "1 1A").
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    s = str(raw).upper().strip()
    s = _WORDS.sub(" ", s)
    s = re.sub(r"-?\([^()0-9]*\)", " ", s)            # "(C)" ICAO category, "(constr)", "(Cargo)"
    s = re.sub(r"[()\[\]]", " ", s)
    keys: list[str] = []
    for chunk in re.split(r"[;,/|+&]", s):
        toks = [t for t in re.split(r"\s+", chunk.strip()) if t and canon(t)]
        merged: list[str] = []
        for t in toks:
            if merged and re.fullmatch(r"[A-Z]{1,3}", merged[-1]) and t[0].isdigit():
                merged[-1] += t                      # "B 41" -> "B41"
            elif merged and re.fullmatch(r"[A-Z]", t) and merged[-1][-1].isdigit():
                merged[-1] += t                      # "511 L" -> "511L"
            else:
                merged.append(t)
        for t in merged:
            k = canon(t)
            if re.fullmatch(r"[A-Z]", k) and keys:
                prev = keys[-1]                      # "G10L/R" -> G10L, G10R
                k = (prev[:-1] if prev[-1].isalpha() else prev) + k
            if k and not re.fullmatch(r"[A-Z]+", k):
                keys.append(k)
    whole = canon(s)
    if whole and whole not in keys and not re.fullmatch(r"[A-Z]+", whole) and len(keys) <= 1:
        keys.append(whole)
    if not keys and re.fullmatch(r"[A-Z]{2,4}", whole):
        keys.append(whole)                           # letter-only areas ("MFG", "HGR")
    return list(dict.fromkeys(keys))


def data_key(icao: str, code: str) -> str:
    return canon(code)


# ----------------------------------------------------------------------------------------------
# Taxi graph
# ----------------------------------------------------------------------------------------------

class TaxiGraph:
    """Undirected graph of densified polylines in local metres.

    ``add_line`` takes a polyline (x, y arrays), the ids of its vertices (shared ids join lines),
    a kind (taxiway / taxilane / runway / lead_in) and a label.
    """

    def __init__(self):
        self.xy: list[tuple[float, float]] = []
        self.kind_of_node: list[str] = []
        self.idx: dict = {}
        self.ea, self.eb, self.elen, self.ecost = [], [], [], []

    def node(self, key, x, y, kind):
        if key is not None and key in self.idx:
            return self.idx[key]
        i = len(self.xy)
        self.xy.append((x, y))
        self.kind_of_node.append(kind)
        if key is not None:
            self.idx[key] = i
        return i

    def edge(self, a, b, length, kind):
        if a == b:
            return
        self.ea.append(a)
        self.eb.append(b)
        self.elen.append(length)
        self.ecost.append(length * (RUNWAY_COST if kind == "runway" else 1.0))

    def add_line(self, x, y, ids, kind):
        prev = self.node(ids[0], x[0], y[0], kind)
        for j in range(1, len(x)):
            nxt = self.node(ids[j], x[j], y[j], kind)
            seg = math.hypot(x[j] - x[j - 1], y[j] - y[j - 1])
            n = max(1, int(math.ceil(seg / DENSIFY_M)))
            a = prev
            for k in range(1, n):
                t = k / n
                b = self.node(None, x[j - 1] + t * (x[j] - x[j - 1]), y[j - 1] + t * (y[j] - y[j - 1]), kind)
                self.edge(a, b, seg / n, kind)
                a = b
            self.edge(a, nxt, seg / n, kind)
            prev = nxt

    def finish(self):
        self.XY = np.array(self.xy, float)
        n = len(self.XY)
        ea, eb = np.array(self.ea), np.array(self.eb)
        self.len_m = coo_matrix((np.r_[self.elen, self.elen], (np.r_[ea, eb], np.r_[eb, ea])), shape=(n, n)).tocsr()
        self.deg = np.bincount(np.r_[ea, eb], minlength=n)
        return self

    def heal(self, max_gap=HEAL_M):
        """Join dead ends to a different component within ``max_gap`` metres (OSM mapping gaps)."""
        self.finish()
        ncomp, lab = connected_components(self.len_m, directed=False)
        tree = cKDTree(self.XY)
        joined = 0
        for i in np.flatnonzero(self.deg == 1):
            for j in tree.query_ball_point(self.XY[i], max_gap):
                if lab[j] != lab[i]:
                    d = float(np.hypot(*(self.XY[i] - self.XY[j])))
                    self.edge(int(i), int(j), d, "taxiway")
                    joined += 1
                    break
        self.finish()
        return joined

    def build(self):
        """Make the cost matrix, keep the largest component, and index nodes for snapping."""
        self.finish()
        n = len(self.XY)
        ea, eb = np.array(self.ea), np.array(self.eb)
        c = np.array(self.ecost)
        # edges are de-duplicated beforehand (dedupe_edges): coo->csr would sum repeated pairs
        self.cost = coo_matrix((np.r_[c, c], (np.r_[ea, eb], np.r_[eb, ea])), shape=(n, n)).tocsr()
        ncomp, lab = connected_components(self.cost, directed=False)
        sizes = np.bincount(lab)
        big = int(np.argmax(sizes))
        self.in_lcc = lab == big
        self.lcc_share = float(sizes[big] / n) if n else 0.0
        self.tree = cKDTree(self.XY[self.in_lcc])
        self.lcc_nodes = np.flatnonzero(self.in_lcc)
        return self

    def snap(self, x, y):
        d, i = self.tree.query(np.c_[x, y])
        return self.lcc_nodes[i], d


def dedupe_edges(G: TaxiGraph):
    """coo->csr sums duplicate (a, b) entries; drop exact duplicates before building."""
    seen = {}
    ea, eb, el, ec = [], [], [], []
    for a, b, l, c in zip(G.ea, G.eb, G.elen, G.ecost):
        k = (a, b) if a < b else (b, a)
        if k in seen:
            j = seen[k]
            if c < ec[j]:
                el[j], ec[j] = l, c
            continue
        seen[k] = len(ea)
        ea.append(a); eb.append(b); el.append(l); ec.append(c)
    G.ea, G.eb, G.elen, G.ecost = ea, eb, el, ec


def osm_graph(osm: dict, P: Proj) -> TaxiGraph:
    G = TaxiGraph()
    for w in osm["taxi"]:
        x, y = P.xy(w["lat"], w["lon"])
        G.add_line(x, y, w["nodes"], w["kind"])
    for w in osm["runways"]:
        x, y = P.xy(w["lat"], w["lon"])
        G.add_line(x, y, w["nodes"], "runway")
    for s in osm["stands"]:
        if s["kind"] == "way":
            x, y = P.xy(s["lat"], s["lon"])
            G.add_line(x, y, s["nodes"], "lead_in")
    dedupe_edges(G)
    G.healed = G.heal()
    dedupe_edges(G)
    return G


def xplane_graph(xp: dict, P: Proj) -> TaxiGraph:
    G = TaxiGraph()
    for e in xp["edges"]:
        if e["a"] not in xp["nodes"] or e["b"] not in xp["nodes"]:
            continue
        (la, oa), (lb, ob) = xp["nodes"][e["a"]], xp["nodes"][e["b"]]
        x, y = P.xy([la, lb], [oa, ob])
        G.add_line(x, y, [("x", e["a"]), ("x", e["b"])], "runway" if e["kind"] == "runway" else "taxiway")
    dedupe_edges(G)
    G.healed = G.heal()
    dedupe_edges(G)
    return G


def seg_frame(x0, y0, x1, y1):
    """Unit vector and length of a runway centreline segment."""
    L = math.hypot(x1 - x0, y1 - y0)
    return (x1 - x0) / L, (y1 - y0) / L, L


def runway_paths(G: TaxiGraph, rw: pd.Series, P: Proj, others: list[tuple[str, LineString]],
                 stand_xy: dict) -> dict:
    """Shortest path from each stand to the departure end of runway ``rw``.

    A virtual sink joins every taxi-graph node lying on the departure runway's centreline
    (within ENTRY_HALF_WIDTH_M, between ENTRY_OVERRUN_M behind the threshold and the far end);
    the sink edge costs BACKTRACK_COST x the along-runway distance from that entry back to the
    threshold. The result is the full-length departure path: taxi to the entry, then (if the
    entry is not at the end) the distance back to the threshold, which approximates taxiing the
    parallel taxiway to the end. Runway edges other than crossings cost RUNWAY_COST per metre.
    ``path_m`` = stand snap leg + taxi-graph metres + backtrack metres, all unweighted.
    """
    x0, y0 = P.xy(rw["thr_lat"], rw["thr_lon"])
    x1, y1 = P.xy(rw["end_lat"], rw["end_lon"])
    ux, uy, L = seg_frame(float(x0), float(y0), float(x1), float(y1))
    rel = G.XY - [float(x0), float(y0)]
    s = rel[:, 0] * ux + rel[:, 1] * uy
    d = np.abs(rel[:, 0] * -uy + rel[:, 1] * ux)
    ent = np.flatnonzero(G.in_lcc & (d <= ENTRY_HALF_WIDTH_M) & (s >= -ENTRY_OVERRUN_M) & (s <= L))
    if len(ent) == 0:
        return {}
    back = np.abs(s[ent])          # along-runway distance from the entry to the threshold
    n = G.cost.shape[0]
    sink = n
    C = G.cost.tocoo()
    rows = np.r_[C.row, np.full(len(ent), sink), ent]
    cols = np.r_[C.col, ent, np.full(len(ent), sink)]
    vals = np.r_[C.data, BACKTRACK_COST * back + 1e-3, BACKTRACK_COST * back + 1e-3]
    M = coo_matrix((vals, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    dist, pred = dijkstra(M, directed=False, indices=sink, return_predecessors=True)
    back_of = dict(zip(ent.tolist(), back.tolist()))
    out = {}
    for key, (x, y) in stand_xy.items():
        node, snap_d = G.snap(x, y)
        node, snap_d = int(node[0]), float(snap_d[0])
        if not np.isfinite(dist[node]):
            continue
        path = [node]
        while pred[path[-1]] != sink and pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        entry = path[-1]
        pts = G.XY[path]
        graph_m = float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(path) > 1 else 0.0
        on_rwy = [G.kind_of_node[i] == "runway" for i in path]
        crossed = []
        if len(path) > 1:
            line = LineString(pts)
            for ident, rl in others:
                if line.intersects(rl):
                    crossed.append(ident)
        out[key] = {"graph_m": graph_m, "backtrack_m": float(back_of.get(entry, 0.0)),
                    "path_m": snap_d + graph_m + float(back_of.get(entry, 0.0)), "snap_m": snap_d,
                    "crossings": len(crossed), "crossed": ",".join(crossed),
                    "runway_share": float(np.mean(on_rwy))}
    return out


# ----------------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------------

def load_departures() -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "STAND_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt",
            "MVT_TIME_UTC_mvt"]
    frames = []
    for f in sorted(DATA.glob("training_*.parquet")):
        d = pd.read_parquet(f, columns=cols)
        d["split"] = "train2025"
        frames.append(d)
    d = pd.read_parquet(DATA / "ranking.parquet", columns=cols)
    d["split"] = "rank2026"
    frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(AIRPORTS)].copy()
    df["STAND_mvt"] = df["STAND_mvt"].astype(str)
    df["RUNWAY_mvt"] = df["RUNWAY_mvt"].astype(str)
    return df


def build_runways(icao: str, oa: pd.DataFrame, xp: dict, osm: dict) -> pd.DataFrame:
    rows = []
    xpr = {r["ident"]: r for r in xp["runways"]}
    for _, r in oa[oa["airport_ident"] == icao].iterrows():
        if int(r.get("closed", 0) or 0) == 1:
            continue
        for a, b in (("le", "he"), ("he", "le")):
            ident = str(r[f"{a}_ident"]).strip()
            other = str(r[f"{b}_ident"]).strip()
            rec = {"apt": icao, "runway": ident, "opposite": other,
                   "length_m": float(r["length_ft"]) * 0.3048 if pd.notna(r["length_ft"]) else np.nan,
                   "width_m": float(r["width_ft"]) * 0.3048 if pd.notna(r["width_ft"]) else np.nan,
                   "oa_lat": r[f"{a}_latitude_deg"], "oa_lon": r[f"{a}_longitude_deg"],
                   "oa_end_lat": r[f"{b}_latitude_deg"], "oa_end_lon": r[f"{b}_longitude_deg"],
                   "heading_true_oa": r[f"{a}_heading_degT"],
                   "displaced_thr_m": float(r[f"{a}_displaced_threshold_ft"]) * 0.3048
                   if pd.notna(r[f"{a}_displaced_threshold_ft"]) else 0.0}
            x = xpr.get(ident)
            if x is not None:
                rec.update(xp_lat=x["lat"], xp_lon=x["lon"])
                o = xpr.get(x["other"])
                rec.update(xp_end_lat=o["lat"] if o else np.nan, xp_end_lon=o["lon"] if o else np.nan,
                           xp_displaced_m=x["displaced_m"])
            rows.append(rec)
    # runway ends present in X-Plane but missing or coordinate-less in OurAirports
    have = {r["runway"] for r in rows if pd.notna(r.get("oa_lat"))}
    for ident, x in xpr.items():
        if ident in have:
            continue
        o = xpr.get(x["other"])
        rows = [r for r in rows if r["runway"] != ident]
        rows.append({"apt": icao, "runway": ident, "opposite": x["other"], "xp_lat": x["lat"], "xp_lon": x["lon"],
                     "xp_end_lat": o["lat"], "xp_end_lon": o["lon"], "width_m": x["width_m"],
                     "xp_displaced_m": x["displaced_m"]})
    df = pd.DataFrame(rows)
    for c in ("oa_lat", "oa_lon", "xp_lat", "xp_lon", "oa_end_lat", "oa_end_lon", "xp_end_lat", "xp_end_lon"):
        if c not in df:
            df[c] = np.nan
    df["oa_xp_diff_m"] = haversine(df["oa_lat"], df["oa_lon"], df["xp_lat"], df["xp_lon"])
    # OurAirports is the runway end of record. X-Plane replaces it when OurAirports has no
    # coordinates, or when the two disagree by more than 30 m and the X-Plane point lies closer to
    # the OSM runway ways (OurAirports' LFPG 26R sits 70 m past the end of the paved runway).
    lat0, lon0 = AIRPORTS[icao]
    P = Proj(lat0, lon0)
    lines = [LineString(np.c_[P.xy(w["lat"], w["lon"])]) for w in osm["runways"]]

    def to_osm(la, lo):
        out = []
        for a, b in zip(la, lo):
            if not lines or not np.isfinite(a):
                out.append(np.nan)
                continue
            q = Point(*P.xy(a, b))
            out.append(min(l.distance(q) for l in lines))
        return np.array(out)

    df["oa_to_osm_runway_m"] = to_osm(df["oa_lat"], df["oa_lon"])
    df["xp_to_osm_runway_m"] = to_osm(df["xp_lat"], df["xp_lon"])
    use_xp = df["xp_lat"].notna() & (df["oa_lat"].isna() | ((df["oa_xp_diff_m"] > 30) & (
        df["xp_to_osm_runway_m"].fillna(np.inf) < df["oa_to_osm_runway_m"].fillna(np.inf))))
    df["thr_source"] = np.where(use_xp, "xplane", "ourairports")
    df["thr_lat"] = np.where(use_xp, df["xp_lat"], df["oa_lat"])
    df["thr_lon"] = np.where(use_xp, df["xp_lon"], df["oa_lon"])
    df["thr_to_osm_runway_m"] = np.where(use_xp, df["xp_to_osm_runway_m"], df["oa_to_osm_runway_m"])
    # the far end is the opposite runway end's chosen point
    opp = df.set_index("runway")[["thr_lat", "thr_lon"]]
    df["end_lat"] = df["opposite"].map(opp["thr_lat"])
    df["end_lon"] = df["opposite"].map(opp["thr_lon"])
    df["centreline_m"] = haversine(df["thr_lat"], df["thr_lon"], df["end_lat"], df["end_lon"])
    df["length_m"] = df["length_m"].fillna(df["centreline_m"])
    df["heading_true"] = [bearing(a, b, c, d) if np.isfinite([a, b, c, d]).all() else np.nan
                          for a, b, c, d in zip(df["thr_lat"], df["thr_lon"], df["end_lat"], df["end_lon"])]
    return df


def stand_features(icao: str, osm: dict, xp: dict, P: Proj) -> pd.DataFrame:
    """Every stand-like feature with its candidate keys, position and heading."""
    rows = []
    taxi_lines = [LineString(np.c_[P.xy(w["lat"], w["lon"])]) for w in osm["taxi"]]
    tree = STRtree(taxi_lines) if taxi_lines else None
    for s in osm["stands"]:
        t = s["tags"]
        labels = [t.get("ref"), t.get("local_ref"), t.get("name")]
        lat, lon, head = None, None, np.nan
        if s["kind"] == "node":
            lat, lon = s["lat"], s["lon"]
        elif s["kind"] == "area":
            lat, lon = float(np.mean(s["lat"][:-1])), float(np.mean(s["lon"][:-1]))
        else:
            # the stop point is the end farther from the taxi network; nose heading points to it
            x, y = P.xy(s["lat"], s["lon"])
            ends = [(0, 1), (-1, -2)]
            dist = []
            for e, _ in ends:
                p = Point(x[e], y[e])
                dist.append(taxi_lines[tree.nearest(p)].distance(p) if tree is not None else 0.0)
            stop, prev = ends[int(np.argmax(dist))]
            other = -1 if stop == 0 else 0
            lat, lon = float(s["lat"][stop]), float(s["lon"][stop])
            head = bearing(s["lat"][other], s["lon"][other], lat, lon)
        for pri, lab in enumerate(labels):
            if not lab:
                continue
            for k in source_keys(icao, lab):
                rows.append({"apt": icao, "key": k, "label": lab, "label_field": ["ref", "local_ref", "name"][pri],
                             "source": "osm", "feature": f"parking_position_{s['kind']}", "feature_id": s["osm_id"],
                             "lat": lat, "lon": lon, "heading": head, "priority": pri})
    for g in osm["gates"]:
        t = g["tags"]
        for pri, lab in enumerate([t.get("ref"), t.get("name")]):
            if not lab:
                continue
            for k in source_keys(icao, lab):
                rows.append({"apt": icao, "key": k, "label": lab, "label_field": ["ref", "name"][pri],
                             "source": "osm_gate", "feature": "gate", "feature_id": g["osm_id"],
                             "lat": g["lat"], "lon": g["lon"], "heading": np.nan, "priority": 10 + pri})
    for i, r in enumerate(xp["ramps"]):
        for k in source_keys(icao, r["name"]):
            rows.append({"apt": icao, "key": k, "label": r["name"], "label_field": "name", "source": "xplane",
                         "feature": f"ramp_{r['ramp_type']}", "feature_id": f"xp{i}", "lat": r["lat"],
                         "lon": r["lon"], "heading": r["heading"], "priority": 0, "icao_cat": r["icao_cat"],
                         "op_type": r["op_type"], "airlines": r["airlines"]})
    return pd.DataFrame(rows)


CLUSTER_M = 150.0


def _clusters(g: pd.DataFrame) -> list[dict]:
    """Group one key's features (one source) into places more than CLUSTER_M apart."""
    cl: list[list] = []
    for _, r in g.sort_values("priority").iterrows():
        for c in cl:
            if haversine(c[0]["lat"], c[0]["lon"], r["lat"], r["lon"]) <= CLUSTER_M:
                c.append(r)
                break
        else:
            cl.append([r])
    out = []
    for c in cl:
        h = [r["heading"] for r in c if pd.notna(r["heading"])]
        out.append({"lat": float(np.mean([r["lat"] for r in c])), "lon": float(np.mean([r["lon"] for r in c])),
                    "heading": h[0] if h else np.nan, "label": str(c[0]["label"]),
                    "feature_id": "+".join(dict.fromkeys(str(r["feature_id"]) for r in c)), "n": len(c),
                    "icao_cat": c[0].get("icao_cat"), "feature": c[0].get("feature")})
    return out


def resolve_features(feats: pd.DataFrame) -> dict:
    """One position per (source, key).

    Features of one key within CLUSTER_M are averaged (MARS lead-in lines, node + way of the same
    stand). When a key names places further apart (LEBL has two aprons numbered 1xx in OSM), the
    place is chosen by, in order: the other source's position for that key, the positions of the
    numeric neighbours (key +-1..3) in the same source, the cluster with most features.
    """
    if not len(feats):
        return {}
    cl = {src: {k: _clusters(g) for k, g in gs.groupby("key")} for src, gs in feats.groupby("source")}
    other = {"osm": "xplane", "xplane": "osm", "osm_gate": "osm"}
    first = {}
    for src, keys in cl.items():
        rows = {}
        for k, cs in keys.items():
            pick, how = cs[0], "single"
            if len(cs) > 1:
                ref = cl.get(other[src], {}).get(k)
                if ref and len(ref) == 1:
                    pick = min(cs, key=lambda c: haversine(c["lat"], c["lon"], ref[0]["lat"], ref[0]["lon"]))
                    how = "other_source"
                else:
                    m = re.fullmatch(r"([A-Z]*)(\d+)([A-Z]*)", k)
                    nb = []
                    if m:
                        for dn in (-3, -2, -1, 1, 2, 3):
                            c2 = keys.get(f"{m.group(1)}{int(m.group(2)) + dn}{m.group(3)}")
                            if c2 and len(c2) == 1:
                                nb.append(c2[0])
                    if nb:
                        la, lo = np.mean([c["lat"] for c in nb]), np.mean([c["lon"] for c in nb])
                        pick = min(cs, key=lambda c: haversine(c["lat"], c["lon"], la, lo))
                        how = "neighbours"
                    else:
                        pick = max(cs, key=lambda c: c["n"])
                        how = "largest"
            rows[k] = {**pick, "n_places": len(cs), "resolved_by": how}
        first[src] = pd.DataFrame.from_dict(rows, orient="index")
    return first


def match_stands(icao: str, codes: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Attach a position to each STAND_mvt code of the airport.

    Rules, in order (``match_rule``):
      exact_osm / exact_xplane   the code's key equals a source key;
      base_osm / base_xplane     a MARS / sub-stand suffix is dropped ("D4A" -> "D4", "G11L" -> "G11");
      children_osm / _xplane     the sources only split the stand ("A66" -> mean of "A66A", "A66B");
      neighbour_osm / _xplane    same prefix, stand number within +-2 ("V151" -> "V150"), a guess;
      gate                       an OSM aeroway=gate with that ref (terminal door, not the stand);
      none.
    OSM wins over X-Plane at each rule; exact matches from both are kept for cross-checking.
    """
    first = resolve_features(feats)

    def look(src, k):
        f = first.get(src)
        if f is None or k not in f.index:
            return None
        return f.loc[k]

    def children(src, k):
        f = first.get(src)
        if f is None:
            return None
        kids = [c for c in f.index if re.fullmatch(re.escape(k) + r"[A-Z]", c)]
        if not kids:
            return None
        sub = f.loc[kids]
        return pd.Series({"lat": sub["lat"].mean(), "lon": sub["lon"].mean(), "heading": sub["heading"].iloc[0],
                          "label": "+".join(sub["label"].astype(str)), "feature_id": "+".join(sub["feature_id"])})

    def neighbour(src, k):
        m = re.fullmatch(r"([A-Z]*)(\d+)([A-Z]*)", k)
        f = first.get(src)
        if not m or f is None:
            return None
        pre, num = m.group(1), int(m.group(2))
        for dn in (1, -1, 2, -2):
            c = f"{pre}{num + dn}"
            if num + dn > 0 and c in f.index:
                return f.loc[c]
        return None

    out = []
    for _, c in codes.iterrows():
        k = data_key(icao, c["stand"])
        rec = {"apt": icao, "stand": c["stand"], "key": k, "n_dep_2025": int(c["n_dep_2025"]),
               "n_dep_2026": int(c["n_dep_2026"])}
        for src, tag in (("osm", "osm"), ("xplane", "xp"), ("osm_gate", "gate")):
            row = look(src, k)
            if row is not None:
                rec.update({f"{tag}_lat": row["lat"], f"{tag}_lon": row["lon"], f"{tag}_heading": row["heading"],
                            f"{tag}_label": str(row["label"]), f"{tag}_id": str(row["feature_id"]),
                            f"{tag}_places": int(row["n_places"]), f"{tag}_resolved_by": row["resolved_by"]})
                if tag == "xp":
                    rec["xp_icao_cat"] = row.get("icao_cat")
                    rec["xp_ramp_type"] = str(row.get("feature", "")).replace("ramp_", "")
        chosen, rule = None, "none"
        base = re.fullmatch(r"(.*\d)([A-Z]{1,2})", k)
        steps = [("exact", lambda s: look(s, k))]
        if base:
            steps.append(("base", lambda s: look(s, base.group(1))))
        steps += [("children", lambda s: children(s, k)), ("neighbour", lambda s: neighbour(s, k))]
        for name, fn in steps:
            for src, tag in (("osm", "osm"), ("xplane", "xplane")):
                row = fn(src)
                if row is not None:
                    chosen, rule = row, f"{name}_{tag}"
                    break
            if chosen is not None:
                break
        if chosen is None:
            row = look("osm_gate", k)
            if row is not None:
                chosen, rule = row, "gate"
        rec["match_rule"] = rule
        if chosen is not None:
            rec.update({"lat": float(chosen["lat"]), "lon": float(chosen["lon"]),
                        "matched_label": str(chosen["label"]), "matched_id": str(chosen["feature_id"]),
                        "matched_heading": chosen["heading"]})
        out.append(rec)
    df = pd.DataFrame(out)
    for tag in ("osm", "gate", "xp"):
        for col in ("lat", "lon", "heading", "label", "id"):
            if f"{tag}_{col}" not in df:
                df[f"{tag}_{col}"] = np.nan
    for col in ("lat", "lon", "matched_label", "matched_id"):
        if col not in df:
            df[col] = np.nan
    df["osm_match"] = df["osm_lat"].notna()
    df["xp_match"] = df["xp_lat"].notna()
    df["gate_match"] = df["gate_lat"].notna()
    df["source"] = df["match_rule"].map(lambda r: "none" if r == "none" else ("osm_gate" if r == "gate" else r.split("_")[-1]))
    df["exact"] = df["match_rule"].str.startswith("exact")
    df["osm_xp_diff_m"] = haversine(df["osm_lat"], df["osm_lon"], df["xp_lat"], df["xp_lon"])
    # OSM and X-Plane place the same code > CLUSTER_M apart: one of them names another stand
    df["pos_conflict"] = df["osm_xp_diff_m"] > CLUSTER_M
    for tag in ("osm", "xp", "gate"):
        for col in ("places", "resolved_by"):
            if f"{tag}_{col}" not in df:
                df[f"{tag}_{col}"] = np.nan
    # heading (aircraft nose, degrees true): X-Plane ramp heading when that ramp lies within 60 m of
    # the chosen position, else the direction of the OSM lead-in line
    d_xp = haversine(df["lat"], df["lon"], df["xp_lat"], df["xp_lon"])
    d_osm = haversine(df["lat"], df["lon"], df["osm_lat"], df["osm_lon"])
    use_xp = df["xp_heading"].notna() & (d_xp < 60)
    use_osm = ~use_xp & df["osm_heading"].notna() & (d_osm < 60)
    if "matched_heading" not in df:
        df["matched_heading"] = np.nan
    use_m = ~use_xp & ~use_osm & df["matched_heading"].notna()
    df["heading"] = np.where(use_xp, df["xp_heading"], np.where(use_osm, df["osm_heading"],
                                                                 df["matched_heading"].where(use_m)))
    df["heading_source"] = np.where(use_xp, "xplane", np.where(use_osm, "osm_leadin",
                                                               np.where(use_m, "fallback_match", "none")))
    df = df.drop(columns="matched_heading")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--airports", default=",".join(AIRPORTS))
    ap.add_argument("--fetch-only", action="store_true")
    args = ap.parse_args()
    apts = [a.strip().upper() for a in args.airports.split(",") if a.strip()]
    CACHE.mkdir(parents=True, exist_ok=True)

    if not args.no_fetch:
        log("fetch: OurAirports")
        fetch_ourairports(args.refresh)
        for icao in apts:
            log(f"fetch: {icao}")
            fetch_xplane(icao, args.refresh)
            fetch_osm(icao, args.refresh)
    if args.fetch_only:
        return

    dep = load_departures()
    log(f"departures: {len(dep):,} rows ({dep['split'].value_counts().to_dict()})")
    oa = pd.read_csv(CACHE / "ourairports_runways.csv")

    all_stands, all_feats, all_rw, all_pairs, graph_info = [], [], [], [], []
    for icao in apts:
        lat0, lon0 = AIRPORTS[icao]
        P = Proj(lat0, lon0)
        osm = parse_osm(CACHE / f"osm_{icao}.json")
        xp = parse_xplane(CACHE / f"xp_{icao}.dat")
        d = dep[dep["ADEP_mvt"] == icao]

        codes = (d.groupby(["STAND_mvt", "split"]).size().unstack(fill_value=0)
                 .reindex(columns=["train2025", "rank2026"], fill_value=0)
                 .rename(columns={"train2025": "n_dep_2025", "rank2026": "n_dep_2026"})
                 .reset_index().rename(columns={"STAND_mvt": "stand"}))
        feats = stand_features(icao, osm, xp, P)
        st = match_stands(icao, codes, feats)
        rw = build_runways(icao, oa, xp, osm)
        all_feats.append(feats)
        all_stands.append(st)
        all_rw.append(rw)

        # graphs
        graphs = {}
        for name, builder, src in (("osm", osm_graph, osm), ("xplane", xplane_graph, xp)):
            G = builder(src, P)
            G.build()
            graphs[name] = G
            graph_info.append({"apt": icao, "graph": name, "nodes": len(G.XY), "edges": len(G.ea),
                               "healed_joins": G.healed, "lcc_share": round(G.lcc_share, 4),
                               "length_km": round(sum(G.elen) / 1000, 1)})
        # runway centrelines for crossing tests (every runway of the airport except helicopter
        # strips such as LFPG 08H/26H and LSZH 01H/19H)
        cl = {}
        for _, r in rw.iterrows():
            if str(r["runway"]).endswith("H") or not r["centreline_m"] >= 800:
                continue
            if np.isfinite([r["thr_lat"], r["end_lat"]]).all():
                ident_pair = "/".join(sorted([r["runway"], r["opposite"]]))
                if ident_pair not in cl:
                    cl[ident_pair] = LineString(np.c_[P.xy([r["thr_lat"], r["end_lat"]], [r["thr_lon"], r["end_lon"]])])

        # both graphs start from the stand's chosen position (stands.parquet lat/lon)
        pairs = d.groupby(["STAND_mvt", "RUNWAY_mvt", "split"]).size().unstack(fill_value=0)
        pairs = pairs.reindex(columns=["train2025", "rank2026"], fill_value=0).reset_index()
        pairs.columns = ["stand", "runway", "n_dep_2025", "n_dep_2026"]
        stx = st.set_index("stand")
        sxy = {}
        for code, r in stx[stx["lat"].notna()].iterrows():
            x, y = P.xy(r["lat"], r["lon"])
            sxy[code] = (float(x), float(y))
        res = defaultdict(dict)
        for name, G in graphs.items():
            for rid in pairs["runway"].unique():
                rr = rw[rw["runway"] == rid]
                if rr.empty or not np.isfinite(rr.iloc[0][["thr_lat", "end_lat"]].astype(float)).all():
                    continue
                r0 = rr.iloc[0]
                own = "/".join(sorted([r0["runway"], r0["opposite"]]))
                others = [(k, v) for k, v in cl.items() if k != own]
                need = {c: sxy[c] for c in pairs.loc[pairs["runway"] == rid, "stand"] if c in sxy}
                for code, v in runway_paths(G, r0, P, others, need).items():
                    res[(code, rid)][name] = v
        rows = []
        for _, p in pairs.iterrows():
            rec = {"apt": icao, "stand": p["stand"], "runway": p["runway"],
                   "n_dep_2025": int(p["n_dep_2025"]), "n_dep_2026": int(p["n_dep_2026"])}
            s = stx.loc[p["stand"]] if p["stand"] in stx.index else None
            rr = rw[rw["runway"] == p["runway"]]
            if s is not None and pd.notna(s["lat"]) and not rr.empty and pd.notna(rr.iloc[0]["thr_lat"]):
                rec["straight_m"] = float(haversine(s["lat"], s["lon"], rr.iloc[0]["thr_lat"], rr.iloc[0]["thr_lon"]))
            for name in ("osm", "xplane"):
                v = res.get((p["stand"], p["runway"]), {}).get(name)
                if v:
                    for k2, val in v.items():
                        rec[f"{name}_{k2}"] = val
            rows.append(rec)
        pr = pd.DataFrame(rows)
        all_pairs.append(pr)
        log(f"{icao}: stands {st['source'].ne('none').mean():.1%} of codes; "
            f"pairs {len(pr)}; osm paths {pr.get('osm_path_m', pd.Series(dtype=float)).notna().mean():.1%}; "
            f"xp paths {pr.get('xplane_path_m', pd.Series(dtype=float)).notna().mean():.1%}")

    stands = pd.concat(all_stands, ignore_index=True)
    feats = pd.concat(all_feats, ignore_index=True)
    rws = pd.concat(all_rw, ignore_index=True)
    pairs = pd.concat(all_pairs, ignore_index=True)
    pairs = choose_path(pairs)
    gi = pd.DataFrame(graph_info)
    write_outputs(stands, feats, rws, pairs, gi, dep)


def choose_path(pr: pd.DataFrame) -> pd.DataFrame:
    """Pick one path per pair and record which network it came from (``path_source``)."""
    for c in ("osm_path_m", "osm_snap_m", "xplane_path_m", "xplane_snap_m", "osm_crossings", "xplane_crossings",
              "osm_graph_m", "xplane_graph_m", "osm_backtrack_m", "xplane_backtrack_m"):
        if c not in pr:
            pr[c] = np.nan
    # OSM when the stand sits on (or within 60 m of) the OSM network; otherwise whichever network
    # the stand snaps to more closely, up to 300 m (the straight snap leg is part of path_m)
    o_snap = pr["osm_snap_m"].where(pr["osm_path_m"].notna())
    x_snap = pr["xplane_snap_m"].where(pr["xplane_path_m"].notna())
    osm_ok = o_snap.le(60) | (o_snap.le(300) & ~(x_snap < o_snap))
    xp_ok = ~osm_ok & x_snap.le(300)
    pr["path_source"] = np.where(osm_ok, "osm", np.where(xp_ok, "xplane", "none"))
    for c in ("path_m", "graph_m", "backtrack_m", "snap_m", "crossings"):
        pr[c] = np.where(osm_ok, pr[f"osm_{c}"], np.where(xp_ok, pr[f"xplane_{c}"], np.nan))
    pr["crossed"] = np.where(osm_ok, pr.get("osm_crossed"), np.where(xp_ok, pr.get("xplane_crossed"), None))
    pr["path_over_straight"] = pr["path_m"] / pr["straight_m"]
    return pr


def coverage(dep: pd.DataFrame, stands: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Share of departure ROWS (not distinct stands) covered, per airport and split."""
    st = stands.set_index(["apt", "stand"])
    pk = pairs.set_index(["apt", "stand", "runway"])
    rows = []
    dep = dep[dep["ADEP_mvt"].isin(stands["apt"].unique())]
    for (icao, split), d in dep.groupby(["ADEP_mvt", "split"]):
        s = st.reindex(pd.MultiIndex.from_arrays([d["ADEP_mvt"], d["STAND_mvt"]]))
        p = pk.reindex(pd.MultiIndex.from_arrays([d["ADEP_mvt"], d["STAND_mvt"], d["RUNWAY_mvt"]]))
        rule = s["match_rule"].astype(object)
        src = p["path_source"].astype(object)
        rows.append({"apt": icao, "split": split, "rows": len(d),
                     "stand_any": float(s["lat"].notna().mean()),
                     "stand_exact": float(rule.astype(str).str.startswith("exact").mean()),
                     "stand_exact_osm": float((rule == "exact_osm").mean()),
                     "stand_exact_xp": float((rule == "exact_xplane").mean()),
                     "stand_fallback": float(rule.isin(["base_osm", "base_xplane", "children_osm", "children_xplane",
                                                        "neighbour_osm", "neighbour_xplane", "gate"]).mean()),
                     "osm_has_code": float(s["osm_match"].eq(True).mean()),
                     "xp_has_code": float(s["xp_match"].eq(True).mean()),
                     "straight": float(p["straight_m"].notna().mean()),
                     "path_any": float(p["path_m"].notna().mean()),
                     "path_osm": float((src == "osm").mean()),
                     "path_xp_fallback": float((src == "xplane").mean())})
    return pd.DataFrame(rows)


def write_outputs(stands, feats, rws, pairs, gi, dep):
    stands = stands.copy()
    for c in stands.columns:
        if stands[c].dtype == object:
            stands[c] = stands[c].astype("string")
    stands.to_parquet(HERE / "stands.parquet", index=False)
    feats = feats.copy()
    for c in feats.columns:
        if feats[c].dtype == object:
            feats[c] = feats[c].astype("string")
    feats.to_parquet(HERE / "stand_features_raw.parquet", index=False)
    rws.to_parquet(HERE / "runways.parquet", index=False)
    pairs = pairs.copy()
    for c in pairs.columns:
        if pairs[c].dtype == object:
            pairs[c] = pairs[c].astype("string")
    pairs.to_parquet(HERE / "stand_runway.parquet", index=False)
    gi.to_csv(HERE / "graphs.csv", index=False)
    cov = coverage(dep, stands, pairs)
    cov.to_csv(HERE / "coverage.csv", index=False)
    pd.set_option("display.width", 200)
    log(gi.to_string(index=False))
    log(cov.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    sys.exit(main())
