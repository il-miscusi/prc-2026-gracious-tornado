#!/usr/bin/env python3
"""Copy the code the final submission depends on from the development tree into this repository.

    python3 tools/sync_from_compbot.py --src DEV_DIR [--configs NAME.json,...] [--dry-run] [--prune]

DEV_DIR is the competition directory of the team's development tree (or set PRC_DEV_DIR). Both
trees share one layout (solutions/, research/, data/), so every module keeps its relative imports
and its paths relative to the repository root.

What is copied, and nothing else:

* PIPELINE: feature stores, trainer, ADS-B extraction, member refit table, final assembler, the
  2025 pseudo-final, upload validation, and the PRU / METAR / geometry builders.
* EVIDENCE: the short scripts that print the 2025 evidence tables quoted in the docs, published
  under analysis/evidence/.
* The assembly configs named by --configs (default DEFAULT_CONFIGS), plus any config they name as
  their "recipe", and the assembler's run manifest of each (research/final/manifests/NAME.json:
  sha256 of every input, output and code file) when that run exists.
* Post-layer modules named in those configs' "post_layers", with the sibling modules they import
  (followed recursively) and the JSON files a layer's kwargs or inputs name. A lever that no
  published config uses is not copied.
* PSEUDO_EVIDENCE: the 2025 pseudo-final summaries (research/final/pseudo/) of the published
  pseudo configs and the paired comparisons that docs/VALIDATION.md cites, and META_EVIDENCE: the
  stacking-lane comparisons (research/final/meta/pseudo/) that docs/METHOD.md cites.
* Kernel specs: for every run directory a shipped config reads (solutions/kernels/<slug>/output),
  the kernel's kernel-metadata.json, member.json (the spec written by solutions/final/build_members.py)
  and small JSON manifests of its output, into kernels/<slug>/. Never kernel.py: a kernel that runs
  on a second account carries signed download URLs for its inputs (INPUT_URLS, fs_urls), so the
  generator code (solutions/core/build_kernel.py, solutions/final/build_members.py) is published
  instead and rebuilds the folder.

Every copied file passes through REWRITES. Each rewrite is an exact string with an expected count;
when the development file has changed so that a rewrite no longer matches, the sync stops instead
of publishing an unedited file. Every rewrite applied is recorded in SYNC_MANIFEST.json together
with the source commit and both sha256s of each file.

Files that exist only in this repository (docs, tools/, research/kk.py, research/pru/fetch_pru.sh,
research/cover/flags.py, research/tail/expand_artifact.py and its artifact, history/) are never
touched. --prune deletes files that an earlier sync wrote and the current one no longer selects.

Run tools/check_publish.py after every sync.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "SYNC_MANIFEST.json"

PIPELINE = [
    # feature stores and trainer (assembled into Kaggle script kernels by build_kernel.py)
    "solutions/core/features.py",
    "solutions/core/fs_main.py",
    "solutions/core/train_main.py",
    "solutions/core/build_kernel.py",
    "solutions/core/batch_main.py",
    "solutions/core/fsbatch_main.py",
    "solutions/core/remote_inputs.py",
    # adsb.lol extraction and ground events
    "solutions/adsb/extract.py",
    "solutions/adsb/extract_v3.py",
    "solutions/adsb/events_v3.py",
    "solutions/adsb/build_k3.py",
    "solutions/adsb/build_final.py",
    # member refit table (one Kaggle kernel per member)
    "solutions/final/build_members.py",
    # store fs8 (fs7 + g8_* columns, derived by build_kernel.py derive) and its member refits and tests
    "solutions/core/fs8.py",
    "research/fgap/build_fs8_members.py",
    "research/fgap/build_paired.py",
    "research/fgap/combine_paired.py",
    "research/fgap/combine_c7_mo.py",
    "research/fgap/paired_eval.py",
    # the d <= 1 h meta-model (G1short): frames, private dataset, Kaggle kernel, rehearsal harness, gate,
    # and the rebuild of its 2026 rows from the kept booster
    "research/final/meta/build_frame.py",
    "research/final/meta/publish_dataset.py",
    "research/final/meta/harness.py",
    "research/final/meta/compare_all.py",
    "research/final/meta/reproduce.py",
    "research/final/meta/specs/f4.json",
    "research/final/meta/kernels/prc-meta-gbm/kernel.py",
    "research/final/meta/kernels/prc-meta-gbm/kernel-metadata.json",
    "research/final/meta/kernels/prc-meta-gbm-f4/kernel.py",
    "research/final/meta/kernels/prc-meta-gbm-f4/kernel-metadata.json",
    "research/final/meta/configs/meta_gbm-G1short-f4p1_pseudo2025.json",
    "research/final/meta/configs/meta_gbm-G1short-f4_pseudo2025.json",
    # final assembly, 2025 pseudo-final, checks
    "research/final/README.md",
    "research/final/frame.py",
    "research/final/stacking.py",
    "research/final/longrows.py",
    "research/final/level.py",
    "research/final/rules.py",
    "research/final/assemble_final.py",
    "research/final/merge_members.py",
    "research/final/pseudo_final.py",
    "research/final/validate_board.py",
    # long-row physical estimate (imported by research/final/longrows.py) and its 2025 tables
    "research/tail/tailkit.py",
    "research/tail/build_cache.py",
    "research/extreme_blend.py",
    # the v30 long-row build, needed only to reproduce the scored board file v31 byte for byte
    "research/tail/longrows_v30.py",
    "research/tail/oofframe.py",
    # the no-ADS-B overlay member's original spec (refit as prc-f-k4 by build_members.py)
    "research/cover/spec_k4.json",
    # external inputs
    "research/build_metar.py",
    "research/geometry/build_geometry.py",
    "research/geometry/check_value.py",
    "research/geometry/README.md",
    # delivery and board history
    "research/upload.py",
    "research/lb_api.py",
]

# short analysis scripts that print the evidence tables quoted in README.md and docs/; each takes
# the repository root as its only argument: python3 analysis/evidence/<name>.py "$PWD"
EVIDENCE_SRC = "research/review_20261008"
# (analysis/evidence/rules_ad_oof.py, which prints rules (a) and (d), exists only in this repository)
EVIDENCE = ["modes.py", "nm_join_bound.py", "nm_join_2026.py", "eda_lags.py", "regime_bias.py", "daily_exact.py",
            "count_diff_2026.py", "eda_final.py", "unmatched_recover.py"]
EVIDENCE_DST = "analysis/evidence"

DEFAULT_CONFIGS = [
    # the final recipe f6, its pre-event baseline f5, and the earlier paired comparisons
    "f6_final.json",
    "f6ev_pseudo2025.json",
    "f5_final.json",
    "f5_pseudo2025.json",
    "f4_pseudo2025.json",
    "f4p1_pseudo2025.json",
    # rejected: fs8 mlp7 as an extra 1-4 h member of f5 (two passes)
    "f5fm7_pseudo2025.json",
    "f5fm7p1_pseudo2025.json",
    # f3 is f2x5_l1av2 with the six-seed mlp7 refit (f2x5m6 is f2x5 with it); both run the same 2025 pseudo-final
    "f3_final.json",
    "f3_pseudo2025.json",
    "f2x5_l1av2_final.json",
    "f2x5_l1av2_pseudo2025.json",
    "f2x5m6_final.json",
    # the runs the main line is compared with in docs/VALIDATION.md (each names its recipe)
    "f2x5_pseudo2025.json",
    "f2_pseudo2025.json",
    "f2p_pseudo2025.json",
    "f2_fillprob_pseudo2025.json",
    "f2_nopost_pseudo2025.json",
    "f2x5_l1av2sq_pseudo2025.json",
    "f2x5_l1av2ss_pseudo2025.json",
    # byte-for-byte rebuild of the earlier board upload v31 (docs/VALIDATION.md)
    "v32c_repro_board.json",
    "v32c_a1_board.json",
]
PSEUDO_DIR = "research/final/pseudo"
# run summaries (research/final/pseudo/NAME_MODE.json) published: the ones docs/ cite. A summary embeds its
# recipe as it was on the day of the run, so older summaries carry superseded working notes; their paired
# comparisons below are published instead.
RUN_SUMMARIES = {"f6ev_pseudo2025", "f5_pseudo2025", "f4_pseudo2025", "f4p1_pseudo2025", "f5fm7_pseudo2025", "f5fm7p1_pseudo2025",
                 "f3_pseudo2025", "f2_pseudo2025", "f2_fillprob_pseudo2025", "f2_nopost_pseudo2025"}
# paired comparisons cited in docs/VALIDATION.md (pseudo_final.py compare A B: A minus B)
PSEUDO_EVIDENCE = [
    "compare_f6ev_pseudo2025_crossfit__f5_pseudo2025_crossfit.json",
    "compare_f6ev_pseudo2025_frozen__f5_pseudo2025_frozen.json",
    "compare_f4_pseudo2025_crossfit__f3_pseudo2025_crossfit.json",
    "compare_f4_pseudo2025_frozen__f3_pseudo2025_frozen.json",
    "compare_f5fm7_pseudo2025_crossfit__f5_pseudo2025_crossfit.json",
    "compare_f2x5_l1av2_pseudo2025_frozen__f2x5_pseudo2025_frozen.json",
    "compare_f2x5_l1av2_pseudo2025_crossfit__f2x5_pseudo2025_crossfit.json",
    "l1av2_report_f2x5_l1av2__f2x5_frozen.json",
    "l1av2_report_f2x5_l1av2__f2x5_crossfit.json",
    "compare_f2x5_pseudo2025_frozen__f2_pseudo2025_frozen.json",
    "compare_f2x5_pseudo2025_crossfit__f2_pseudo2025_crossfit.json",
    "compare_f2_fillprob_pseudo2025_frozen__f2_pseudo2025_frozen.json",
    "compare_f2_fillprob_pseudo2025_crossfit__f2_pseudo2025_crossfit.json",
    "compare_f2_nopost_pseudo2025_frozen__f2_pseudo2025_frozen.json",
    "compare_f2_nopost_pseudo2025_crossfit__f2_pseudo2025_crossfit.json",
    "compare_f2p_pseudo2025_frozen__f2_pseudo2025_frozen.json",
    "compare_f2x5_l1av2sq_pseudo2025_frozen__f2x5_pseudo2025_frozen.json",
    "compare_f2x5_l1av2sq_pseudo2025_crossfit__f2x5_pseudo2025_crossfit.json",
    "compare_f2x5_l1av2ss_pseudo2025_frozen__f2x5_pseudo2025_frozen.json",
    "compare_f2x5_l1av2ss_pseudo2025_crossfit__f2x5_pseudo2025_crossfit.json",
]
META_DIR = "research/final/meta/pseudo"
# stacking-lane meta-learners against f2 (docs/METHOD.md, results not shipped)
META_EVIDENCE = [f"compare_meta_{arm}_{mode}__f2_pseudo2025_{mode}.json"
                 for arm in ("lin-apt", "lin-vc", "gbm-G1", "gbm-G1x", "gbm-G1m", "gbm-G1short")
                 for mode in ("crossfit", "frozen")] + [
    # G1short re-fitted on f4's members (= f5), against f4, two passes of the L1a-v2 loadings
    "compare_meta_gbm-G1short-f4_crossfit__f4_pseudo2025_crossfit.json",
    "compare_meta_gbm-G1short-f4p1_crossfit__f4_pseudo2025_crossfit.json",
    "meta_gbm-G1short-f4_crossfit.json",
    "meta_gbm-G1short-f4_frozen.json",
    "meta_gbm-G1short-f4p1_crossfit.json",
    "meta_gbm-G1short-f4p1_frozen.json",
]
FGAP_DIR = "research/fgap/runs"
# store fs8 tests cited in docs/METHOD.md: c7 on fs8 against c7 on fs7, month-out folds (paired_eval.py)
FGAP_EVIDENCE = ["c7_mo_fs8_vs_fs7.json", "r3/paired_2s6l.json"]
CONFIG_DIR = "research/final/configs"
# the assembler's run manifest of each shipped config (inputs, outputs and code with sha256s)
MANIFEST_DIR = "research/final/manifests"

# kernel output files small enough to publish as evidence (never predictions, never kernel.py)
KERNEL_FILES = ["kernel-metadata.json", "member.json"]
# kernels the final file depends on that no config names as a member run directory: the two
# feature stores built on final_ranking.parquet and the kernel that cuts the assembler's per-row
# inputs from them. kernel.py is copied only for prc-f-inputs, which no generator writes; it is
# refused if it holds a URL.
KERNEL_EXTRAS = {"prc-f-fs6": ["kernel-metadata.json"], "prc-f-fs7": ["kernel-metadata.json"],
                 "prc-f-fs8": ["kernel-metadata.json"],
                 "prc-f-inputs": ["kernel-metadata.json", "kernel.py"]}
OUTPUT_JSON_MAX_BYTES = 1 << 20


@dataclass(frozen=True)
class Rewrite:
    old: str            # a regular expression (re.M), so this file never holds the text it removes
    new: str
    count: int
    why: str


# daily-layer parameter files record the out-of-fold file they were fitted on as an absolute path of the
# development machine; daily.py never reads that field, so it is made relative to the repository root
DAILY_PARAMS_PATH = Rewrite(
    r"/(?:Users|home)/[^\"\n]*?/independent-predict-aircraft-taxi-out-time-at-european-airports/", "", 1,
    "metadata only: the absolute development-tree prefix of the recorded out-of-fold path becomes repository-"
    "relative (daily.py does not read the field; the parameters are unchanged)")

REWRITES: dict[str, list[Rewrite]] = {
    "research/final/levers/artifacts/daily_params.json": [DAILY_PARAMS_PATH],
    "solutions/core/features.py": [
        Rewrite(
            r'^[ \t]+Path\("/(?:Users|home)/[^"\n]+"\),[ \t]*\n',
            "", 1,
            "drop a developer's absolute data path from the search list; DATA_ROOT, /kaggle/input and the "
            "working directory (the repository root, holding data/) remain"),
    ],
    "solutions/core/build_kernel.py": [
        Rewrite(
            r'^ADSB = json\.loads\(\(SOL / "v27" / "kernel-metadata\.json"\)\.read_text\(\)\)\["kernel_sources"\]$',
            'ADSB = json.loads((SOL / "adsb" / "sources.json").read_text())["kernel_sources"]', 1,
            "read the ADS-B extract kernel list from solutions/adsb/sources.json (generated by the sync from "
            "the development tree's solutions/v27/kernel-metadata.json, which is not published)"),
        Rewrite(
            r"^    sys\.path\.insert\(0, str\(SOL\.parents\[2\]\)\)\n",
            '    sys.path.insert(0, str(SOL.parent / "tools"))  # tools/kaggle_env.py: Kaggle credentials\n', 2,
            "Kaggle API access goes through tools/kaggle_env.py instead of the development tree's "
            "account modules"),
        Rewrite(
            r"^    from scout\.kaggle_fetcher import _get_api\n    api = _get_api\(\)\n",
            '    from kaggle_env import get_api\n    api = get_api("primary")\n', 1,
            "input_urls() authenticates through tools/kaggle_env.py"),
        Rewrite(
            r"^    if owner == OWNER:\n"
            r"        from scout\.kaggle_fetcher import _get_api\n"
            r"        api = _get_api\(\)\n"
            r"    else:\n"
            r"        from core import kaggle_accounts\n"
            r"        from solver\.kaggle_runner import _get_api\n"
            r"        acct = next\(a for a in kaggle_accounts\.load_accounts\(\) if a\.username == owner\)\n"
            r"        api = _get_api\(acct\)\n",
            "    from kaggle_env import account_for_owner, get_api\n"
            '    api = get_api("primary" if owner == OWNER else account_for_owner(owner))\n', 1,
            "fs_urls() authenticates through tools/kaggle_env.py"),
    ],
    "research/final/README.md": [
        Rewrite(r"`NOTES\.md` has the results\.", "`docs/VALIDATION.md` summarises the results.", 1,
                "the assembler's working log (NOTES.md) is not published; point to docs/VALIDATION.md"),
        Rewrite(r"^\| `NEEDS\.md` \| [^\n]*\n", "", 1,
                "NEEDS.md (the development tree's list of pending inputs) is not published"),
    ],
    "research/final/meta/publish_dataset.py": [
        Rewrite(r"^sys\.path\.insert\(0, str\(C\.parents\[1\]\)\)\n",
                'sys.path.insert(0, str(C / "tools"))  # tools/kaggle_env.py: Kaggle credentials\n', 1,
                "Kaggle API access goes through tools/kaggle_env.py instead of the development tree's account modules"),
        Rewrite(r"^    from core import kaggle_accounts\n    from solver\.kaggle_runner import _get_api\n",
                "    from kaggle_env import get_api\n", 1, "same"),
        Rewrite(r"^    acc = kaggle_accounts\.primary_account\(\)\n    api = _get_api\(acc\)\n"
                r"    assert api\.config_values\.get\(\"username\"\) == acc\.username == \"usiadianimuwa\", \"wrong account\"\n",
                '    api = get_api("primary")\n    user = str(api.config_values.get("username"))\n', 1,
                "the dataset owner is the authenticated primary account"),
        Rewrite(r'"dataset": f"\{acc\.username\}/\{SLUG\}"', '"dataset": f"{user}/{SLUG}"', 1, "same"),
    ],
    "research/upload.py": [
        Rewrite(
            r"^Credentials live outside the repo in ~/\.prc/minio\.json \(0600\):\n",
            "Credentials live outside the repo, in the JSON file named by $PRC_MINIO_CREDENTIALS (0600):\n", 1,
            "credential location is configured by environment, not a developer's home directory"),
        Rewrite(
            r'^CRED = Path\.home\(\) / "\.prc" / "minio\.json"\n',
            'CRED = Path(os.environ.get("PRC_MINIO_CREDENTIALS", "minio-credentials.json"))\n', 1,
            "the bucket credentials file is named by $PRC_MINIO_CREDENTIALS instead of a path in the "
            "developer's home directory"),
        Rewrite(r"^import json\nimport re\n", "import json\nimport os\nimport re\n", 1,
                "os is needed by the CRED line"),
        Rewrite(r"a hand-made hybrid of two board files \(10-09 v3\d-v3\d\) has no build manifest\.",
                "a file combined by hand from other files has no build manifest.", 1,
                "a docstring example named internal upload numbers; the rule it states is unchanged"),
    ],
}

SKIP_NAMES = {"__pycache__", ".DS_Store", ".cache"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def apply_rewrites(rel: str, text: str) -> tuple[str, list[dict]]:
    done = []
    for rw in REWRITES.get(rel, []):
        rx = re.compile(rw.old, re.M)
        n = len(rx.findall(text))
        if n != rw.count:
            raise SystemExit(f"{rel}: rewrite expects {rw.count} match(es), found {n}. The development file "
                             f"changed; update REWRITES in {Path(__file__).name} before syncing.\n"
                             f"  pattern: {rw.old[:120]!r}")
        text = rx.sub(lambda _m: rw.new, text)
        done.append({"pattern": rw.old, "replacement": rw.new, "count": rw.count, "why": rw.why})
    return text, done


def local_imports(path: Path) -> set[Path]:
    """Sibling modules a post-layer imports (import X / from . import X / from X import ...)."""
    tree = ast.parse(path.read_text(), str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                names.update(a.name for a in node.names)
            elif node.module:
                names.add(node.module.split(".")[0])
    return {path.parent / f"{n}.py" for n in names if (path.parent / f"{n}.py").exists()}


def module_closure(start: Path) -> set[Path]:
    seen: set[Path] = set()
    todo = [start]
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        todo.extend(local_imports(p) - seen)
    init = start.parent / "__init__.py"
    if init.exists():
        seen.add(init)
    return seen


def run_dirs(cfg: dict) -> set[str]:
    """Kernel slugs of every member run directory (pred / oof / mo) a config reads."""
    slugs = set()
    for m in cfg.get("members", {}).values():
        for key in ("pred", "oof", "mo"):
            v = m.get(key) if isinstance(m, dict) else None
            if isinstance(v, str) and v.startswith("solutions/kernels/"):
                slugs.add(v.split("/")[2])
    return slugs


def select(src: Path, config_names: list[str]) -> tuple[dict[str, Path], list[str], set[str], list[str]]:
    """{repo-relative destination: source file} plus the config list, kernel slugs and notes."""
    files: dict[str, Path] = {}
    notes: list[str] = []
    for rel in PIPELINE:
        p = src / rel
        if not p.exists():
            raise SystemExit(f"missing in the development tree: {rel}")
        files[rel] = p
    for name in EVIDENCE:
        p = src / EVIDENCE_SRC / name
        if not p.exists():
            raise SystemExit(f"missing in the development tree: {EVIDENCE_SRC}/{name}")
        files[f"{EVIDENCE_DST}/{name}"] = p
    configs: list[str] = []
    todo = list(config_names)
    while todo:
        name = todo.pop(0)
        rel = f"{CONFIG_DIR}/{Path(name).name}"
        if rel in files:
            continue
        p = src / rel
        if not p.exists():
            raise SystemExit(f"config not found: {rel}")
        files[rel] = p
        configs.append(rel)
        cfg = json.loads(p.read_text())
        recipe = cfg.get("recipe")
        if recipe:
            todo.append(Path(recipe).name)
    slugs: set[str] = set()
    for rel in configs:
        cfg = json.loads((src / rel).read_text())
        slugs |= run_dirs(cfg)
        man = src / MANIFEST_DIR / f"{cfg.get('name', Path(rel).stem)}.json"  # the assembler names it after cfg["name"]
        if man.exists():
            files[str(man.relative_to(src))] = man
        for pl in cfg.get("post_layers", []):
            mod = src / pl["module"]
            if not mod.exists():
                raise SystemExit(f"{rel}: post layer {pl['module']} not found")
            for q in module_closure(mod):
                files[str(q.relative_to(src))] = q
            # the JSON files the layer names (parameter and loading artifacts), not its whole directory
            named = [v for v in pl.get("kwargs", {}).values() if isinstance(v, str)] + list(pl.get("inputs", []))
            for v in named:
                q = src / v if v.endswith(".json") else None
                if q is not None and q.exists() and q.stat().st_size <= OUTPUT_JSON_MAX_BYTES:
                    files[str(q.relative_to(src))] = q
            notes.append(f"post layer {pl['module']} ships ({rel})")
    for name in sorted(p.name for p in (src / PSEUDO_DIR).glob("*.json")):
        stem = name.removesuffix(".json").removesuffix("_frozen").removesuffix("_crossfit")
        if stem in RUN_SUMMARIES:
            files[f"{PSEUDO_DIR}/{name}"] = src / PSEUDO_DIR / name
    for cfg_rel in configs:  # files a member declares as its upstream inputs (meta-model code, specs, records)
        for m in json.loads((src / cfg_rel).read_text()).get("members", {}).values():
            for v in m.get("inputs", []) if isinstance(m, dict) else []:
                q = src / v
                if v.startswith("research/") and q.exists() and q.stat().st_size <= OUTPUT_JSON_MAX_BYTES:
                    files[v] = q
    for d, names in ((PSEUDO_DIR, PSEUDO_EVIDENCE), (META_DIR, META_EVIDENCE), (FGAP_DIR, FGAP_EVIDENCE)):
        for name in names:
            p = src / d / name
            if not p.exists():
                raise SystemExit(f"evidence file missing in the development tree: {d}/{name}")
            files[f"{d}/{name}"] = p
    return files, configs, slugs, notes


def kernel_snapshot(src: Path, slugs: set[str]) -> tuple[dict[str, bytes], dict]:
    """kernels/<slug>/... bytes for every run directory the shipped configs read (and the kernels
    merged into it), without kernel.py and without any file that holds a URL."""
    out: dict[str, bytes] = {}
    report: dict = {}
    todo = sorted(slugs | set(KERNEL_EXTRAS))
    seen: set[str] = set()
    while todo:
        slug = todo.pop(0)
        if slug in seen:
            continue
        seen.add(slug)
        d = src / "solutions" / "kernels" / slug
        if not d.exists():
            report[slug] = {"status": "missing in the development tree (not built yet)"}
            continue
        got, skipped = [], []
        for name in KERNEL_EXTRAS.get(slug, KERNEL_FILES):
            if (d / name).exists():
                b = (d / name).read_bytes()
                if name == "kernel.py" and (b"http://" in b or b"https://" in b or b"_URLS = {'" in b
                                            or b"'fs_urls'" in b or b'"fs_urls"' in b):
                    skipped.append(name)
                    continue
                out[f"kernels/{slug}/{name}"] = b
                got.append(name)
        o = d / "output"
        if o.exists():
            for q in sorted(list(o.glob("*.json")) + list(o.glob("*/*.json"))):
                rel = q.relative_to(d).as_posix()
                b = q.read_bytes()
                if len(b) > OUTPUT_JSON_MAX_BYTES or b"http://" in b or b"https://" in b:
                    skipped.append(rel)
                    continue
                out[f"kernels/{slug}/{rel}"] = b
                got.append(rel)
                if q.name == "merge_manifest.json":
                    for s in merged_sources(json.loads(b)):
                        if s not in seen:
                            todo.append(s)
        report[slug] = {"status": "copied", "files": got, "skipped": skipped}
    return out, report


def merged_sources(man: dict) -> list[str]:
    """Kernel slugs named in a merge_manifest.json (research/final/merge_members.py)."""
    found = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x.startswith("solutions/kernels/"):
            found.append(x.split("/")[2])
    walk(man)
    return sorted(set(found))


def generated(src: Path) -> dict[str, bytes]:
    v27 = src / "solutions" / "v27" / "kernel-metadata.json"
    sources = json.loads(v27.read_text())["kernel_sources"]
    doc = {"kernel_sources": sources,
           "note": ("Kaggle kernels whose outputs hold the adsb.lol v1 ground extracts (solutions/adsb/extract.py) "
                    "of 2025 and of January/July 2026, mounted by every feature-store kernel "
                    "(solutions/core/build_kernel.py). The February/June 2026 extracts are added by "
                    "build_kernel.py's ADSB_FINAL list; solutions/adsb/build_final.py writes those kernels. "
                    "Generated by tools/sync_from_compbot.py.")}
    return {"solutions/adsb/sources.json": (json.dumps(doc, indent=1) + "\n").encode()}


def source_state(src: Path, rels: list[str]) -> dict:
    def git(*a):
        r = subprocess.run(["git", "-C", str(src), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    prefix = git("rev-parse", "--show-prefix") or ""
    dirty = []
    if commit:
        lines = (git("status", "--porcelain", "--", *rels) or "").splitlines()
        dirty = sorted(line[3:].removeprefix(prefix) for line in lines if line.strip())
    return {"commit": commit, "dirty_files": dirty}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default=os.environ.get("PRC_DEV_DIR"), help="development competition directory")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="comma-separated config file names in research/final/configs/ to publish")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prune", action="store_true", help="delete files an earlier sync wrote that are no longer selected")
    a = ap.parse_args()
    if not a.src:
        raise SystemExit("--src (or PRC_DEV_DIR) is required")
    src = Path(a.src).expanduser().resolve()
    names = [n.strip() for n in a.configs.split(",") if n.strip()]
    files, configs, slugs, notes = select(src, names)

    writes: dict[str, bytes] = {}
    entries: dict[str, dict] = {}
    for rel, p in sorted(files.items()):
        raw = p.read_bytes()
        if rel in REWRITES:
            text, done = apply_rewrites(rel, raw.decode())
            data = text.encode()
        else:
            data, done = raw, []
        writes[rel] = data
        source = str(p.relative_to(src))
        entries[rel] = {"source": source, "sha256_source": sha256_bytes(raw), "sha256": sha256_bytes(data),
                        "rewrites": done}
    for rel, data in generated(src).items():
        writes[rel] = data
        entries[rel] = {"generated_from": "solutions/v27/kernel-metadata.json (kernel_sources)",
                        "sha256": sha256_bytes(data), "rewrites": []}
    snap, snap_report = kernel_snapshot(src, slugs)
    for rel, data in snap.items():
        writes[rel] = data
        entries[rel] = {"source": f"solutions/kernels/{rel.removeprefix('kernels/')}",
                        "sha256_source": sha256_bytes(data), "sha256": sha256_bytes(data), "rewrites": []}
    unknown = set(REWRITES) - set(writes)
    if unknown:
        raise SystemExit(f"REWRITES name files that are not synced: {sorted(unknown)}")

    previous = json.loads(MANIFEST.read_text())["files"] if MANIFEST.exists() else {}
    stale = sorted(set(previous) - set(writes))
    changed = sorted(r for r in writes if not (REPO / r).exists() or (REPO / r).read_bytes() != writes[r])

    print(f"source: commit {source_state(src, list(files))['commit']}")
    print(f"configs: {', '.join(configs)}")
    for n in notes:
        print(f"  {n}")
    print(f"files selected: {len(writes)} ({len(snap)} kernel-spec files); changed or new: {len(changed)}")
    for r in changed:
        print(f"  write {r}")
    for slug, rep in sorted(snap_report.items()):
        if rep["status"] != "copied":
            print(f"  kernel {slug}: {rep['status']}")
        elif rep["skipped"]:
            print(f"  kernel {slug}: skipped (URL or size) {rep['skipped']}")
    for r in stale:
        print(f"  {'delete' if a.prune else 'stale (use --prune)'} {r}")
    rewrites = [(r, w["why"]) for r, e in entries.items() for w in e["rewrites"]]
    print(f"rewrites applied: {len(rewrites)}")
    for r, why in rewrites:
        print(f"  {r}: {why}")
    if a.dry_run:
        print("dry run: nothing written")
        return

    for rel in changed:
        dst = REPO / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(writes[rel])
        src_file = files.get(rel)
        if src_file is not None and os.access(src_file, os.X_OK):
            dst.chmod(0o755)
    if a.prune:
        for rel in stale:
            p = REPO / rel
            if p.exists():
                p.unlink()
            d = p.parent
            while d != REPO and d.exists() and not any(d.iterdir()):
                d.rmdir()
                d = d.parent
    state = source_state(src, list(files))
    MANIFEST.write_text(json.dumps({
        "synced_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source_commit": state["commit"],
        "source_dirty_files": state["dirty_files"],
        "configs": configs,
        "kernels": snap_report,
        "files": {r: entries[r] for r in sorted(entries)},
        "stale_not_pruned": [] if a.prune else stale,
    }, indent=1) + "\n")
    print(f"wrote {len(changed)} file(s) and {MANIFEST.name}. Next: python3 tools/check_publish.py")


if __name__ == "__main__":
    sys.exit(main())
