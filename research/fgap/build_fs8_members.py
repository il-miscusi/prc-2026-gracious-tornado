"""fs8 member refits (feature-gap lane): the fs7 members c7a/c7b, cs7, mlp7a/b on the fs8 store, refit on all
of 2025 with banked weights, predicting the final frame; plus their 2025 month-out folds.

    python3 research/fgap/build_fs8_members.py --list
    python3 research/fgap/build_fs8_members.py c7a c7b cs7 mlp7a mlp7b          # refit kernels prc-f8-<name>
    python3 research/fgap/build_fs8_members.py c7a cs7 mlp7a --folds M1,M2,M6,M7  # fold kernels prc-f8-<name>-mo
    python3 research/kkm.py push solutions/kernels/prc-f8-<name>

Each spec is the solutions/final/build_members.py spec of the fs7 member with two changes only:
  * store prc-f-fs8 (= prc-f-fs7's files unchanged, plus the g8_* columns appended; solutions/core/fs8.py);
  * owner miscusi (T4 x2).  Rounds, seeds, drop lists, package pins, parallelism are the fs7 member's: two c7
    CatBoost processes on one 32.9 GB host ran out of memory (prc-g8-c7-mo v1, exit -9), so c7 stays one seed /
    one fold at a time (~0.44 h per refit seed on a T4, prc-f-c7b).
Nothing here pushes; the lane brief requires a report before the refit is launched.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parents[1]
sys.path.insert(0, str(C / "solutions" / "final"))
import build_members as bm  # noqa: E402

BUILD = C / "solutions" / "core" / "build_kernel.py"
STORE = "prc-f-fs8"
NAMES = ["c7a", "c7b", "cs7", "mlp7a", "mlp7b"]


def spec_for(name: str, folds: list[str] | None) -> dict:
    store, owner, accel, spec, src, role = bm.MEMBERS[name]
    assert store == "prc-f-fs7", f"{name} is not an fs7 member"
    if folds:
        spec = bm.fold_spec(spec, folds)
    else:
        spec = {**spec, "save_models": True}
    return spec


def build(name: str, folds: list[str] | None) -> Path:
    spec = spec_for(name, folds)
    slug = f"prc-f8-{name}" + ("-mo" if folds else "")
    cmd = [sys.executable, str(BUILD), "train", slug, "--fs", STORE, "--fs-owner", bm.PRIMARY, "--owner", bm.FALLBACK,
           "--accel", "gpu", "--spec", json.dumps(spec)]
    out = subprocess.run(cmd, cwd=C, capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(out.stderr)
    folder = C / "solutions" / "kernels" / slug
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    meta["enable_internet"] = True
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    (folder / "member.json").write_text(json.dumps({"member": name, "store": STORE, "owner": bm.FALLBACK, "accel": "gpu",
                                                    "folds": folds or [], "source_spec": f"build_members {name} (fs7)",
                                                    "spec": {k: v for k, v in spec.items() if k != "fs_urls"}}, indent=1))
    return folder


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--folds", default="")
    a = ap.parse_args()
    folds = [f for f in a.folds.split(",") if f] or None
    if a.list or not a.names:
        for n in NAMES:
            s = spec_for(n, folds)
            print(n, {k: v for k, v in s.items() if k not in ("params", "drop_features", "variants")},
                  list(s.get("variants", {})))
    for n in a.names:
        print(build(n, folds))
