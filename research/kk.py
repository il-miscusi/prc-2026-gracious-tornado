"""Kaggle kernel helper for this campaign (primary account only).

    python3 research/kk.py push solutions/v28            # push a kernel folder (checks the push ban)
    python3 research/kk.py status REF [REF...]           # status of kernels (owner/slug or slug)
    python3 research/kk.py fetch REF DEST                # download a finished kernel's output
    python3 research/kk.py running                       # every kernel of ours still queued/running

Kaggle answers a push from an account it has rate-limited with HTTP 429 and a Retry-After of
days, so a 429 is recorded as a ban on that account (in the JSON file named by $PRC_PUSH_BANS,
default .kaggle_push_bans.json in the working directory) and later pushes from it are refused
before any request is sent. A push that does not answer in 10 minutes is reported as unknown,
never retried.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from kaggle_env import get_api, username  # noqa: E402

BAN_FILE = Path(os.environ.get("PRC_PUSH_BANS", ".kaggle_push_bans.json"))  # {account: "until ISO time"}

ACCOUNT = "primary"
if "--account" in sys.argv:
    i = sys.argv.index("--account")
    ACCOUNT = sys.argv[i + 1]
    del sys.argv[i:i + 2]


def _get_api():
    return get_api(ACCOUNT)


def _ref(s: str) -> str:
    return s if "/" in s else f"{username(ACCOUNT)}/{s}"


def _banned_until():
    import datetime as dt
    import json
    if not BAN_FILE.exists():
        return None
    until = json.loads(BAN_FILE.read_text()).get(ACCOUNT)
    return until if until and dt.datetime.fromisoformat(until) > dt.datetime.now(dt.timezone.utc) else None


def push(folder: str) -> None:
    banned = _banned_until()
    if banned:
        raise SystemExit(f"push refused: {ACCOUNT} is push-banned until {banned}")
    api = _get_api()
    out: dict = {}

    def run():
        try:
            out["r"] = api.kernels_push(folder)
        except Exception as exc:  # noqa: BLE001
            out["e"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(600)
    if t.is_alive():
        raise SystemExit("push outcome unknown after 10 min (not retried)")
    if "e" in out:
        msg = str(out["e"])
        if "429" in msg:  # an account-level push ban: record it, never retry
            import datetime as dt
            import json
            bans = json.loads(BAN_FILE.read_text()) if BAN_FILE.exists() else {}
            bans[ACCOUNT] = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=72)).isoformat()
            BAN_FILE.parent.mkdir(parents=True, exist_ok=True)
            BAN_FILE.write_text(json.dumps(bans))
        raise SystemExit(f"push failed: {msg}")
    r = out["r"]
    print(f"pushed {folder}: ref={getattr(r, 'ref', None)} version={getattr(r, 'version_number', None)} "
          f"error={getattr(r, 'error', None)}")


def status(refs: list[str]) -> None:
    api = _get_api()
    for ref in refs:
        try:
            s = api.kernels_status(_ref(ref))
            print(_ref(ref), str(s.status).split(".")[-1], (getattr(s, "failure_message", "") or "")[:200])
        except Exception as exc:  # noqa: BLE001
            print(_ref(ref), "ERR", exc)


def fetch(ref: str, dest: str) -> None:
    api = _get_api()
    os.makedirs(dest, exist_ok=True)
    api.kernels_output(_ref(ref), dest, force=True, quiet=True)
    print(sorted(os.listdir(dest)))


def running() -> None:
    api = _get_api()
    seen = set()
    for sort in ("dateRun", "dateCreated"):
        for k in api.kernels_list(mine=True, page_size=25, sort_by=sort):
            if k.ref in seen or not k.ref.startswith((username(ACCOUNT)) + "/"):
                continue
            seen.add(k.ref)
            try:
                st = str(api.kernels_status(k.ref).status).split(".")[-1]
            except Exception as exc:  # noqa: BLE001
                st = f"ERR {str(exc)[:40]}"
            if st in ("QUEUED", "RUNNING") or st.startswith("ERR"):
                print(k.ref, st, getattr(k, "last_run_time", ""))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"push": lambda: push(args[0]), "status": lambda: status(args),
     "fetch": lambda: fetch(args[0], args[1]), "running": running}[cmd]()
