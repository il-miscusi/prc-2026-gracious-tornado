"""Create (once) or version the private Kaggle dataset the meta-learner kernel reads (primary account).

    python3 research/final/meta/publish_dataset.py create   # usiadianimuwa/prc-taxi-2026-meta, private
    python3 research/final/meta/publish_dataset.py version "message"
    python3 research/final/meta/publish_dataset.py create --slug prc-taxi-2026-meta-f4 --dir submissions/final/meta/f4

Files: submissions/final/meta/meta2025.parquet and meta2026.parquet (research/final/meta/build_frame.py)
with their build manifests. One request per call, never retried; the sha256 of every file is written
to research/final/meta/results/dataset_<action>.json.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from frame import C, sha256  # noqa: E402

sys.path.insert(0, str(C / "tools"))  # tools/kaggle_env.py: Kaggle credentials
SLUG = "prc-taxi-2026-meta"
FILES = ["submissions/final/meta/meta2025.parquet", "submissions/final/meta/meta2025.json",
         "submissions/final/meta/meta2026.parquet", "submissions/final/meta/meta2026.json"]


def main():
    from kaggle_env import get_api
    global SLUG, FILES
    argv = sys.argv[1:]
    for flag in ("--slug", "--dir"):
        if flag in argv:
            i = argv.index(flag)
            val = argv[i + 1]
            del argv[i:i + 2]
            if flag == "--slug":
                SLUG = val
            else:
                FILES = [f"{val}/{Path(f).name}" for f in FILES]
    sys.argv[1:] = argv
    action = sys.argv[1]
    api = get_api("primary")
    user = str(api.config_values.get("username"))
    rec = {"dataset": f"{user}/{SLUG}", "action": action, "private": True,
           "files": {f: sha256(C / f) for f in FILES}, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    spec = str(Path(FILES[0]).parent / "meta_spec.json")
    files = FILES + ([spec] if (C / spec).exists() else [])
    rec["files"] = {f: sha256(C / f) for f in files}
    with tempfile.TemporaryDirectory(prefix="prc-meta-") as d:
        stage = Path(d)
        for f in files:
            shutil.copy2(C / f, stage / Path(f).name)
        (stage / "dataset-metadata.json").write_text(json.dumps(
            {"title": SLUG, "id": rec["dataset"], "licenses": [{"name": "unknown"}]}))
        if action == "create":
            r = api.dataset_create_new(str(stage), public=False, quiet=True, convert_to_csv=False)
        else:
            r = api.dataset_create_version(str(stage), sys.argv[2], quiet=True, convert_to_csv=False)
    rec["error"] = str(getattr(r, "error", "") or "")
    rec["ref"] = str(getattr(r, "ref", "") or getattr(r, "url", ""))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"dataset_{SLUG}_{action}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    if rec["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
