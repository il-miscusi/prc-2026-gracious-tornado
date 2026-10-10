"""Build the Kaggle kernel folders for the ADS-B v3 extraction (account: miscusi).

    python3 solutions/adsb/build_k3.py test            # k3_test: 2025-01-15, prefilter comparison on
    python3 solutions/adsb/build_k3.py test2           # k3_test: 2025-01-14..15 (concurrency, stitching)
    python3 solutions/adsb/build_k3.py 202501 202607   # one folder per month: k3_YYYYMM
    python3 solutions/adsb/build_k3.py 202510:1-15     # part of a month: k3_202510a (days 1-15)
    python3 solutions/adsb/build_k3.py w2a=202502,202503,202504   # several months in one kernel: k3_w2a
    python3 solutions/adsb/build_k3.py fix1=202510:3-5,202511:7-7  # re-run failed days
    python3 solutions/adsb/build_k3.py events          # k3_events: events_v3.py over every k3 month output

Each folder holds a copy of extract_v3.py with DATES / TAG / COMPARE_PREFILTER filled in and a
kernel-metadata.json for a private CPU script kernel with internet (adsb.lol is on GitHub).
adsb.lol published nothing for 2025-05-28..31 and 2025-06-01..10; those days are left out.
"""

from __future__ import annotations

import calendar
import json
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = "miscusi"
NO_RELEASE = {date(2025, 5, d) for d in range(28, 32)} | {date(2025, 6, d) for d in range(1, 11)}
MONTHS = ["202501", "202502", "202503", "202504", "202505", "202506", "202507", "202508", "202509",
          "202510", "202511", "202512", "202601", "202607"]


def metadata(slug: str, code_file: str, kernel_sources: list[str] | None = None, internet: bool = True) -> dict:
    return {
        "id": f"{OWNER}/{slug}", "title": slug.replace("-", " "), "code_file": code_file,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": internet,
        "competition_sources": [], "dataset_sources": [], "kernel_sources": kernel_sources or [],
    }


def parse_spec(spec: str) -> list[date]:
    days: list[date] = []
    for part in spec.split(","):
        if ":" in part:
            tag, rng = part.split(":")
            lo, hi = (int(x) for x in rng.split("-"))
            days += month_days(tag, lo, hi)
        else:
            days += month_days(part)
    return days


def build_extract(tag: str, days: list[date], compare: bool, suffix: str = "") -> Path:
    src = (HERE / "extract_v3.py").read_text()
    dates = "[" + ", ".join(f"date({d.year}, {d.month}, {d.day})" for d in days) + "]"
    src, n1 = re.subn(r"^DATES = .*$", f"DATES = {dates}  # built by build_k3.py", src, count=1, flags=re.M)
    src, n2 = re.subn(r'^TAG = .*$', f'TAG = "{tag}{suffix}"', src, count=1, flags=re.M)
    src, n3 = re.subn(r"^COMPARE_PREFILTER = .*$", f"COMPARE_PREFILTER = {compare}", src, count=1, flags=re.M)
    assert (n1, n2, n3) == (1, 1, 1), "extract_v3.py lost one of its per-kernel lines"
    folder = HERE / f"k3_{tag}{suffix}"
    folder.mkdir(exist_ok=True)
    code = f"extract_v3_{tag}{suffix}.py"
    (folder / code).write_text(src)
    slug = f"prc-adsb3-{tag}{suffix}"
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata(slug, code), indent=2) + "\n")
    return folder


def month_days(tag: str, lo: int = 1, hi: int = 31) -> list[date]:
    y, m = int(tag[:4]), int(tag[4:])
    last = calendar.monthrange(y, m)[1]
    return [date(y, m, d) for d in range(lo, min(hi, last) + 1) if date(y, m, d) not in NO_RELEASE]


def build_events() -> Path:
    folder = HERE / "k3_events"
    folder.mkdir(exist_ok=True)
    (folder / "events_v3.py").write_text((HERE / "events_v3.py").read_text())
    sources = []
    for d in sorted(HERE.glob("k3_*")):
        meta = d / "kernel-metadata.json"
        if meta.exists() and d.name not in ("k3_test", "k3_events"):
            sources.append(json.loads(meta.read_text())["id"])
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata("prc-adsb3-events", "events_v3.py", sources, internet=False), indent=2) + "\n")
    return folder


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if arg == "test":
            print(build_extract("test", [date(2025, 1, 15)], compare=True))
        elif arg == "test2":  # two consecutive days: stream concurrency timing + midnight stitching
            print(build_extract("test", [date(2025, 1, 14), date(2025, 1, 15)], compare=False))
        elif arg == "events":
            print(build_events())
        elif "=" in arg:
            name, spec = arg.split("=", 1)
            print(build_extract(name, parse_spec(spec), compare=False))
        elif ":" in arg:
            tag, rng = arg.split(":")
            lo, hi = (int(x) for x in rng.split("-"))
            print(build_extract(tag, month_days(tag, lo, hi), compare=False, suffix="a" if lo == 1 else "b"))
        else:
            print(build_extract(arg, month_days(arg), compare=False))
