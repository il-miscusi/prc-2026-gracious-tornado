"""Thin Kaggle adapter: the only place this repository touches Kaggle credentials.

The pipeline runs its heavy stages as Kaggle script kernels.  Two accounts were used:

  primary   the account that owns the private datasets and the feature stores (default
            ~/.kaggle credentials: OAuth credentials.json, kaggle.json or KAGGLE_* variables);
  any other a compute-only account given as a bearer token in the JSON file named by
            $PRC_KAGGLE_ACCOUNTS: {"<name>": {"username": "<kaggle user>", "token": "<API token>"}}
            (mode 0600, kept outside the repository).

A secondary account cannot mount the primary's private outputs, so build_kernel.py passes it
tokenised kernel-output URLs instead (remote_inputs.py).  Nothing here retries a push: a 429 on a
kernel push is an account-level ban (honour Retry-After, stop).

In the development tree, research/kk.py and build_kernel.py called the team's own account
modules; this module replaces them (tools/sync_from_compbot.py rewrites build_kernel.py to use it).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ACCOUNTS = Path(os.environ["PRC_KAGGLE_ACCOUNTS"]) if os.environ.get("PRC_KAGGLE_ACCOUNTS") else None
_CACHE: dict = {}


def accounts() -> dict:
    return json.loads(ACCOUNTS.read_text()) if ACCOUNTS and ACCOUNTS.exists() else {}


def username(name: str = "primary") -> str:
    if name == "primary":
        return str(get_api("primary").config_values.get("username"))
    return accounts()[name]["username"]


def get_api(name: str = "primary"):
    """Authenticated KaggleApi for `name` ("primary" or a key of the $PRC_KAGGLE_ACCOUNTS file)."""
    if name in _CACHE:
        return _CACHE[name]
    if name == "primary":
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = None
        if (Path.home() / ".kaggle" / "credentials.json").exists():
            try:
                api = KaggleApi(enable_oauth=True)
                api.authenticate()
            except Exception:  # noqa: BLE001
                api = None
        if api is None:
            api = KaggleApi()
            api.authenticate()
    else:
        acct = accounts()[name]
        # a bearer token is read only with OAuth off; otherwise ~/.kaggle/credentials.json wins
        os.environ["KAGGLE_API_TOKEN"] = acct["token"]
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        got = str(api.config_values.get("username") or "")
        if got and got != acct["username"]:
            raise RuntimeError(f"token for {name!r} authenticates as {got!r}, not {acct['username']!r}")
    _CACHE[name] = api
    return api


def account_for_owner(owner: str) -> str:
    """Adapter name whose Kaggle username is `owner` (the primary when none matches)."""
    for name, a in accounts().items():
        if a.get("username") == owner:
            return name
    return "primary"
