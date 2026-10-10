"""Run research/final/pseudo_final.py on this lane's configs without writing into research/final/pseudo/.

    python3 research/final/meta/harness.py run CONFIG [--weights frozen|crossfit]
    python3 research/final/meta/harness.py compare A.parquet B.parquet [--json OUT]

pseudo_final.run copies its JSON summary to <pseudo_final.HERE>/pseudo/ when the out-dir is under
submissions/; here HERE points at research/final/meta, so the summaries land in
research/final/meta/pseudo/ and the row files in submissions/final/meta/pseudo2025/ (not in git).
Nothing else in the harness changes: same steps, same fits, same scoring.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pseudo_final  # noqa: E402

OUT = "submissions/final/meta/pseudo2025"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--weights", choices=["frozen", "crossfit"], default="frozen")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json")
    a = ap.parse_args()
    pseudo_final.HERE = HERE
    if a.cmd == "run":
        pseudo_final.run(a.config, a.weights, OUT)
    else:
        out = a.json or str(HERE / "pseudo" / f"compare_{Path(a.a).stem}__{Path(a.b).stem}.json")
        pseudo_final.compare(a.a, a.b, out)


if __name__ == "__main__":
    main()
