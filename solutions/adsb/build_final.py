"""Build the ADS-B kernels for the final phase's new months (Feb and Jun 2026).

    python3 solutions/adsb/build_final.py

Writes, under solutions/adsb/:
  kf_<part>/   v1 extract (extract.py, rich "_w600r" output) as usiadianimuwa/prc-adsb-extract-<part>
  k3f_<part>/  v3 extract (extract_v3.py) as miscusi/prc-adsb3-<part>
  k3_events_f/ events_v3.py over the k3f outputs as miscusi/prc-adsb3-events-f, writing events only
               for the final's new months (the adjacent days exist to stitch midnight stays)

Each half-month carries the neighbouring day before/after it, so a departure whose ground stay
crosses 00:00Z at a month edge is still seen whole.  Same code and parameters as the Jan/Jul 2026
kernels; only DATES and the output tag change.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIMARY, FALLBACK = "usiadianimuwa", "miscusi"


def span(a: date, b: date) -> list[date]:
    return [a + timedelta(days=i) for i in range((b - a).days + 1)]


PARTS = {
    "202602ra": span(date(2026, 1, 31), date(2026, 2, 14)),
    "202602rb": span(date(2026, 2, 15), date(2026, 3, 1)),
    "202606ra": span(date(2026, 5, 31), date(2026, 6, 15)),
    "202606rb": span(date(2026, 6, 16), date(2026, 7, 1)),
}
KEEP_TAGS = ["202602", "202606"]


def meta(owner, slug, code, sources=None, internet=True):
    return {"id": f"{owner}/{slug}", "title": slug.replace("-", " "), "code_file": code, "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": internet, "competition_sources": [], "dataset_sources": [],
            "kernel_sources": sources or []}


def dates_lit(days):
    return "[" + ", ".join(f"date({d.year}, {d.month}, {d.day})" for d in days) + "]"


def build_v1(part, days):
    src = (HERE / "extract.py").read_text()
    src, n1 = re.subn(r"^DATES = .*$", f"DATES = {dates_lit(days)}  # build_final.py", src, count=1, flags=re.M)
    src, n2 = re.subn(r'%Y%m%d}\.parquet"', '%Y%m%d}_w600r.parquet"', src, count=1)
    assert (n1, n2) == (1, 1)
    d = HERE / f"kf_{part}"
    d.mkdir(exist_ok=True)
    (d / "extract.py").write_text(src)
    (d / "kernel-metadata.json").write_text(json.dumps(meta(PRIMARY, f"prc-adsb-extract-{part}", "extract.py"), indent=2))
    return d


def build_v3(part, days):
    src = (HERE / "extract_v3.py").read_text()
    src, n1 = re.subn(r"^DATES = .*$", f"DATES = {dates_lit(days)}  # build_final.py", src, count=1, flags=re.M)
    src, n2 = re.subn(r'^TAG = .*$', f'TAG = "fix{part}"', src, count=1, flags=re.M)
    src, n3 = re.subn(r"^COMPARE_PREFILTER = .*$", "COMPARE_PREFILTER = False", src, count=1, flags=re.M)
    assert (n1, n2, n3) == (1, 1, 1)
    d = HERE / f"k3f_{part}"
    d.mkdir(exist_ok=True)
    (d / "extract_v3.py").write_text(src)
    (d / "kernel-metadata.json").write_text(json.dumps(meta(FALLBACK, f"prc-adsb3-f{part}", "extract_v3.py"), indent=2))
    return d


def build_events():
    src = (HERE / "events_v3.py").read_text()
    # write events only for the final's new months; the adjacent days' months stay inputs for stitching
    needle = "        d = deps[deps[\"month\"] == tag] if tag.isdigit() else deps\n"
    assert src.count(needle) == 1
    src = src.replace(needle, f"        if tag not in {KEEP_TAGS!r}:\n            continue\n" + needle, 1)
    # The 2025/Jan/Jul events (miscusi/prc-adsb3-events) predate the evidence-start fix (62d3838a) and
    # carry no usable evidence-start time, so the new months must not carry one either: a column that is
    # NaN in every training row and filled only in Feb/Jun would be a train/inference shift.
    write = '        ev.to_parquet(f"adsb3_events_{tag}.parquet", index=False)\n'
    assert src.count(write) == 1
    src = src.replace(write, '        ev = ev.drop(columns=["t_dep_evidence_start", "dep_evidence_start_idx"], errors="ignore")\n'
                      + write, 1)
    d = HERE / "k3_events_f"
    d.mkdir(exist_ok=True)
    (d / "events_v3.py").write_text(src)
    sources = [f"{FALLBACK}/prc-adsb3-f{p}" for p in PARTS]
    (d / "kernel-metadata.json").write_text(
        json.dumps(meta(FALLBACK, "prc-adsb3-events-f", "events_v3.py", sources, internet=False), indent=2))
    return d


if __name__ == "__main__":
    for p, days in PARTS.items():
        print(build_v1(p, days), len(days))
        print(build_v3(p, days), len(days))
    print(build_events())
