"""Upload a candidate to the team bucket through the S3 API (MinIO), no browser.

    python3 research/upload.py check FILE            # validate a board file (Jan/Jul template) only
    python3 research/upload.py put FILE --as vN      # validate, then upload as gracious-tornado_vN.parquet
    python3 research/upload.py check-final FILE      # validate a final file (four-month template) only
    python3 research/upload.py put-final FILE        # validate, then upload ONCE as gracious-tornado_final.parquet
    python3 research/upload.py ls                    # list the bucket's objects (read-only)

Credentials live outside the repo, in the JSON file named by $PRC_MINIO_CREDENTIALS (0600):
{"endpoint", "access_key", "secret_key", "bucket"}.

Board uploads (`put`) are refused when: the ids differ from the board template in any way, a
prediction is not finite or is negative, the target name already exists in the bucket, the
version number is not the next one, or 5 uploads already landed in the current UTC day. Objects
whose names do not end in _vN.parquet (the final file) are ignored when numbering versions.

The final upload (`put-final`) is validated against data/final_20261008/final_submitting.parquet
(670,790 ids) and must have both columns float64, as every scored upload had (v31 included; the
template declares TAXITIME_SEC_mvt int32, but the organisers' scorer has been reported to ignore
non-float64 files). It is refused when any object with "final" in its name already exists: the
organisers accept one final file. After the upload the object is stat'ed (size, etag), downloaded
again, and its sha256 compared with the local file's.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

C = Path(__file__).resolve().parents[1]
TEMPLATE = C / "data" / "09-22-2026-08-13-30_files_list" / "submitting.parquet"
FINAL_TEMPLATE = C / "data" / "final_20261008" / "final_submitting.parquet"
FINAL_ROWS = 670_790
TEAM = "gracious-tornado"
FINAL_NAME = f"{TEAM}_final.parquet"
CRED = Path(os.environ.get("PRC_MINIO_CREDENTIALS", "minio-credentials.json"))
DAILY_CAP = 5
VERSION_RE = re.compile(r"_v(\d+)\.parquet$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(path: Path, template: Path = TEMPLATE, final: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(path)
    tpl = pd.read_parquet(template)
    assert list(df.columns) == list(tpl.columns), f"columns {list(df.columns)} != {list(tpl.columns)}"
    ids = df["MVT_ID_mvt"].to_numpy()
    assert df["MVT_ID_mvt"].dtype == tpl["MVT_ID_mvt"].dtype, f"id dtype {ids.dtype} != template"
    if final:
        assert str(df["TAXITIME_SEC_mvt"].dtype) == "float64", \
            f"TAXITIME_SEC_mvt dtype {df['TAXITIME_SEC_mvt'].dtype}: the final must be float64 like every scored upload"
        assert len(tpl) == FINAL_ROWS, f"final template has {len(tpl)} rows, expected {FINAL_ROWS}"
    assert (ids == np.round(ids)).all(), "non-integral ids"
    ids = ids.astype("int64")
    assert len(df) == len(tpl) and df["MVT_ID_mvt"].is_unique, "row count / duplicate ids"
    assert set(ids.tolist()) == set(tpl["MVT_ID_mvt"].to_numpy().astype("int64").tolist()), "id set differs"
    p = df["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    assert np.isfinite(p).all() and (p >= 0).all(), "non-finite or negative predictions"
    order = bool((df["MVT_ID_mvt"].to_numpy() == tpl["MVT_ID_mvt"].to_numpy()).all())
    print(f"{path.name}: {len(df)} rows ok (template {template.name}, same order: {order}); mean {p.mean():.1f} s, "
          f"max {p.max():.0f} s, dtypes {df['MVT_ID_mvt'].dtype}/{df['TAXITIME_SEC_mvt'].dtype}, sha256 {sha256(path)}")
    return df


def client():
    from minio import Minio
    c = json.loads(CRED.read_text())
    host = re.sub(r"^https?://", "", c["endpoint"])
    return Minio(host, access_key=c["access_key"], secret_key=c["secret_key"], secure=True), c["bucket"]


def versioned(objs) -> dict:
    """{N: object} for names ending in _vN.parquet; every other name (e.g. the final) is skipped."""
    out = {}
    for o in objs:
        mt = VERSION_RE.search(o.object_name)
        if mt:
            out[int(mt.group(1))] = o
    return out


BUILD_MANIFESTS = C / "research" / "final" / "manifests"
FORCED_LOG = BUILD_MANIFESTS / "forced_uploads.jsonl"


def built_by_pipeline(path: Path) -> str | None:
    """Name of the assemble_final.py manifest that lists this file's sha256 among its outputs, else None.

    The ranking page warns that "attempts to learn from or exploit the ranking process" are monitored, and the
    campaign rule is no label extraction and no differential probes (REQUIREMENTS.md). A board upload must be a
    whole pipeline build; a file combined by hand from other files has no build manifest.
    """
    digest = sha256(path)
    for mf in sorted(BUILD_MANIFESTS.glob("*.json")):
        try:
            man = json.loads(mf.read_text())
        except ValueError:
            continue
        if man.get("kind") in ("board_probe", "board_candidate"):
            continue
        if digest in (man.get("outputs") or {}).values():
            return mf.name
    return None


def put_board(path: Path, version: int, force: str | None = None) -> None:
    built = built_by_pipeline(path)
    if built is None and not force:
        sys.exit(f"refused: {path.name} is not an output of any assemble_final.py manifest in {BUILD_MANIFESTS}. "
                 "Board uploads are whole pipeline builds only (no hybrids or probes). --force REASON overrides "
                 "and is logged.")
    print(f"build manifest: {built}" if built else f"FORCED: {force}")
    m, bucket = client()
    objs = [o for o in m.list_objects(bucket) if o.object_name.endswith(".parquet")]
    versions = versioned(objs)
    nxt = max(versions) + 1
    if version != nxt:
        sys.exit(f"refused: next version is v{nxt}, asked for v{version}")
    today = dt.datetime.now(dt.timezone.utc).date()
    n_today = sum(o.last_modified.date() == today for o in versions.values())
    if n_today >= DAILY_CAP:
        sys.exit(f"refused: {n_today} uploads already on {today} UTC")
    name = f"{TEAM}_v{version}.parquet"
    if force:
        with open(FORCED_LOG, "a") as f:
            f.write(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "name": name,
                                "file": str(path), "sha256": sha256(path), "reason": force}) + "\n")
    m.fput_object(bucket, name, str(path), content_type="application/octet-stream")
    print(f"uploaded {name} ({n_today + 1}/{DAILY_CAP} today); score: python3 research/lb_api.py --wait v{version}")


def put_final(path: Path) -> None:
    from minio.error import S3Error
    m, bucket = client()
    finals = [o.object_name for o in m.list_objects(bucket) if "final" in o.object_name.lower()]
    if finals:
        sys.exit(f"refused: final object(s) already in the bucket: {finals}. One final submission only.")
    try:
        m.stat_object(bucket, FINAL_NAME)
        sys.exit(f"refused: {FINAL_NAME} already exists")
    except S3Error as e:
        if e.code not in ("NoSuchKey", "NoSuchObject"):
            raise
    local_sha, local_size = sha256(path), path.stat().st_size
    m.fput_object(bucket, FINAL_NAME, str(path), content_type="application/octet-stream")
    st = m.stat_object(bucket, FINAL_NAME)
    print(f"uploaded {FINAL_NAME}: remote size {st.size} (local {local_size}), etag {st.etag}, "
          f"last_modified {st.last_modified}")
    if st.size != local_size:
        sys.exit("READBACK FAILED: size differs")
    if "-" not in (st.etag or ""):  # single-part upload: etag is the content md5
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if st.etag.strip('"') != md5:
            sys.exit(f"READBACK FAILED: etag {st.etag} != local md5 {md5}")
    with tempfile.TemporaryDirectory() as td:
        back = Path(td) / "readback.parquet"
        m.fget_object(bucket, FINAL_NAME, str(back))
        remote_sha = sha256(back)
    print(f"readback sha256 {remote_sha}\nlocal    sha256 {local_sha}")
    if remote_sha != local_sha:
        sys.exit("READBACK FAILED: sha256 differs")
    others = [o.object_name for o in m.list_objects(bucket) if "final" in o.object_name.lower()]
    print(f"readback OK. final objects in the bucket: {others}")


def ls() -> None:
    m, bucket = client()
    objs = sorted(m.list_objects(bucket), key=lambda o: o.last_modified)
    for o in objs:
        print(f"{o.last_modified:%Y-%m-%d %H:%M:%S}Z  {o.size:>12,}  {o.object_name}")
    v = versioned(objs)
    print(f"{len(objs)} objects; highest version v{max(v) if v else 0}; "
          f"final objects: {[o.object_name for o in objs if 'final' in o.object_name.lower()]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "put", "check-final", "put-final", "ls"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--as", dest="version", type=int)
    ap.add_argument("--force", metavar="REASON", help="upload a board file no build manifest lists (logged)")
    a = ap.parse_args()
    if a.cmd == "ls":
        return ls()
    if not a.file:
        ap.error(f"{a.cmd} needs FILE")
    path = Path(a.file)
    if a.cmd in ("check", "put"):
        validate(path)
        if a.cmd == "put":
            put_board(path, a.version, a.force)
        return
    validate(path, FINAL_TEMPLATE, final=True)
    if a.cmd == "put-final":
        put_final(path)


if __name__ == "__main__":
    main()
