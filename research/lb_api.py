"""Read the PRC 2026 board from the organisers' API.

    python3 research/lb_api.py            # our submissions + top 10 teams
    python3 research/lb_api.py --wait v2  # poll until gracious-tornado_v2 is scored

Every submission, every team, paginated by `nextCursor`. `usedPairs` is the
number of rows the scorer matched: 344,841 means the full ranking set counted.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

URL = "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"
TEAM = "gracious-tornado"


def fetch_all() -> list[dict]:
    items, cursor = [], None
    for _ in range(500):
        url = URL + (f"?cursor={urllib.parse.quote(cursor)}" if cursor else "")
        with urllib.request.urlopen(url, timeout=60) as resp:
            page = json.load(resp)
        items += page.get("items", [])
        cursor = page.get("nextCursor")
        if not cursor:
            break
    return items


def summary(items: list[dict]) -> None:
    ours = sorted((i for i in items if i["teamName"] == TEAM), key=lambda i: i["processedAt"])
    print(f"{len(items)} scored submissions")
    for i in ours:
        print(f"  {i['filename']:32s} score={i['score']}  usedPairs={i['usedPairs']}  {i['processedAt']}")
    best: dict[str, float] = {}
    for i in items:
        if i.get("score") is not None:
            best[i["teamName"]] = min(best.get(i["teamName"], float("inf")), i["score"])
    ranked = sorted(best.items(), key=lambda kv: kv[1])
    for rank, (team, score) in enumerate(ranked[:10], 1):
        print(f"  {rank:3d} {team:24s} {score:.4f}")
    if TEAM in best:
        rank = [t for t, _ in ranked].index(TEAM) + 1
        print(f"  ours: rank {rank}/{len(ranked)}  best {best[TEAM]:.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--wait":
        name = f"{TEAM}_{sys.argv[2]}.parquet"
        for _ in range(120):
            items = fetch_all()
            if any(i["filename"] == name for i in items):
                break
            time.sleep(60)
        summary(items)
    else:
        summary(fetch_all())
