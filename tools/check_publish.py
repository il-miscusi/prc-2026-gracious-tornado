#!/usr/bin/env python3
"""Refuse to publish this repository while it holds a secret, a signed URL, a home-directory path,
an e-mail address, a stray data file or a reference to a writing tool.

    python3 tools/check_publish.py            # files git would publish (tracked + untracked, not ignored)
    python3 tools/check_publish.py --history  # also every blob and commit message reachable from any ref

Exit status 1 when anything fails. Commit author and committer e-mail addresses cannot be removed
from history without rewriting it, so they are listed as a warning for the person publishing.

The checks are patterns, not proof: they catch the shapes that have leaked before in this project
(signed Kaggle/GCS download URLs embedded in generated kernel.py files, a developer's absolute data
path, credential file locations) and the usual token formats.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CONTENT = {
    "signed URL": re.compile(
        rb"X-Amz-(?:Signature|Credential|Security-Token)|X-Goog-(?:Signature|Credential)|GoogleAccessId="
        rb"|[?&](?:Signature|sig|se|sv|token|Expires|KeyName)=[A-Za-z0-9%._~+/-]{6,}"
        rb"|storage\.googleapis\.com/[^\s'\"]*\?|kaggle-data-sets/[^\s'\"]+|kaggleusercontent\.com/[^\s'\"]+"
        rb"|\.blob\.core\.windows\.net/[^\s'\"]*\?"),
    "embedded URL dict": re.compile(
        rb"(?:INPUT_URLS|FS_URLS)\s*=\s*\{[^}]{0,400}https?://|['\"]fs_urls['\"]\s*:\s*\{[^}]{0,400}https?://"),
    "token": re.compile(
        rb"KGAT_[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
        rb"|xox[baprs]-[A-Za-z0-9-]{10,}|sk-[A-Za-z0-9]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
        rb"|\"(?:secret_key|access_key|password|api_key|token|key)\"\s*:\s*\"[^\"<$]{8,}\""
        rb"|\b(?:secret_key|access_key|password|api_key|token)\s*=\s*['\"][^'\"<$]{8,}['\"]"),
    "home path": re.compile(
        rb"/Users/[A-Za-z]|/home/[a-z][a-z0-9_-]*/|[A-Za-z]:\\\\?Users\\\\?|~/\.prc\b|\$HOME/\.prc\b"
        rb"|Documents-MacBook|Personal Projects"),
    "e-mail address": re.compile(rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"),
    "writing-tool mention": re.compile(
        rb"\b(?:Claude|CLAUDE\.md|Anthropic|Codex|ChatGPT|OpenAI|Copilot|Gemini)\b|AI-generated|Generated with"),
}
# e-mail-shaped strings that are not addresses (none needed so far)
EMAIL_ALLOW: set[bytes] = set()

DATA_SUFFIXES = {".parquet", ".csv", ".npy", ".npz", ".pkl", ".pickle", ".zip", ".gz", ".bz2", ".xz", ".feather",
                 ".arrow", ".cbm", ".bin", ".pt", ".pth", ".h5", ".joblib", ".db", ".sqlite", ".sqlite3"}
# data files that may be published, each with the reason
DATA_ALLOW = {
    "research/tail/artifacts/longrows_inputs_over2h.parquet":
        "two earlier submissions' own predictions for 10,168 board rows (ids and seconds, no organiser "
        "columns); needed only to rebuild the scored board file v31 (research/tail/expand_artifact.py)",
}
MAX_BYTES = 10 << 20
WARN_BYTES = 1 << 20
COMMIT_MSG = re.compile(rb"Co-Authored-By|\b(?:Claude|Anthropic|Codex|ChatGPT|OpenAI|Copilot|Gemini)\b"
                        rb"|AI-generated|Generated with", re.I)


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=True).stdout


def scan(name: str, data: bytes) -> list[str]:
    out = []
    for label, rx in CONTENT.items():
        for m in rx.finditer(data):
            hit = m.group(0)
            if label == "e-mail address" and hit in EMAIL_ALLOW:
                continue
            line = data.count(b"\n", 0, m.start()) + 1
            shown = hit[:80].decode("utf-8", "replace")
            out.append(f"{name}:{line}: {label}: {shown}")
    return out


def data_file(path: str) -> str | None:
    if Path(path).suffix.lower() in DATA_SUFFIXES and path not in DATA_ALLOW:
        return f"{path}: data file not in DATA_ALLOW (organiser data, EUROCONTROL files and model outputs stay out)"
    return None


def check_worktree() -> tuple[list[str], list[str]]:
    fails, warns = [], []
    names = [n for n in git("ls-files", "-co", "--exclude-standard", "-z").decode().split("\0") if n]
    for n in names:
        p = REPO / n
        if not p.is_file():
            continue
        if n == "tools/check_publish.py":
            continue  # holds the patterns themselves
        size = p.stat().st_size
        if size > MAX_BYTES:
            fails.append(f"{n}: {size:,} bytes (limit {MAX_BYTES:,})")
        elif size > WARN_BYTES:
            warns.append(f"{n}: {size:,} bytes")
        d = data_file(n)
        if d:
            fails.append(d)
        fails += scan(n, p.read_bytes())
    return fails, warns


def check_history() -> tuple[list[str], list[str]]:
    fails, warns = [], []
    seen: dict[str, str] = {}
    for line in git("rev-list", "--objects", "--all").decode().splitlines():
        sha, _, path = line.partition(" ")
        if path and sha not in seen:
            seen[sha] = path
    blobs = [s for s in seen if git("cat-file", "-t", s).strip() == b"blob"]
    for sha in blobs:
        path = seen[sha]
        if path == "tools/check_publish.py":
            continue
        d = data_file(path)
        if d:
            fails.append(f"history {sha[:10]} {d}")
        fails += [f"history {sha[:10]} {h}" for h in scan(path, git("cat-file", "-p", sha))]
    log = git("log", "--all", "--format=%H%x00%ae%x00%ce%x00%B%x01")
    emails = set()
    for rec in log.split(b"\x01"):
        parts = rec.strip(b"\n").split(b"\x00")
        if len(parts) < 4:
            continue
        h, ae, ce, body = parts[0], parts[1], parts[2], parts[3]
        emails |= {ae.decode(), ce.decode()}
        m = COMMIT_MSG.search(body)
        if m:
            fails.append(f"commit {h[:10].decode()}: message mentions {m.group(0).decode()!r}")
    warns.append("commit author/committer e-mail(s) in history (published with the commits): "
                 + ", ".join(sorted(emails)))
    return fails, warns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--history", action="store_true", help="also scan every reachable blob and commit message")
    a = ap.parse_args()
    fails, warns = check_worktree()
    if a.history:
        f2, w2 = check_history()
        fails += f2
        warns += w2
    for w in warns:
        print(f"WARN {w}")
    for f in fails:
        print(f"FAIL {f}")
    n_files = len([n for n in git("ls-files", "-co", "--exclude-standard", "-z").decode().split("\0") if n])
    print(f"{'FAILED' if fails else 'OK'}: {len(fails)} problem(s), {len(warns)} warning(s), "
          f"{n_files} file(s) checked{' plus history' if a.history else ''}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
