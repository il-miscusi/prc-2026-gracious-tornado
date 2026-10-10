"""Write kernels/<slug>/ snapshots and MANIFEST.json for a candidate, from the development tree.

    python3 tools/snapshot_kernels.py SRC_CAMPAIGN_DIR research/candidates/v32c.json

SRC_CAMPAIGN_DIR is the working directory the candidate was built in (solutions/kernels/<run>/...,
submissions/candidates/<name>.manifest.json).  Every kernel that produced a member's out-of-fold
predictions or ranking refits is copied as it ran (kernel.py + kernel-metadata.json) with signed and
tokenised URLs redacted; MANIFEST.json lists, per member: its stack weight, the kernels and seeds
behind each fold and each refit, rounds, feature store, library and accelerator, and the hashes of
the member files the candidate was assembled from.  The member -> kernel map (PROVENANCE) was
established on 2026-10-01 by matching each member's OOF and submission numerically to its kernels'
outputs (max |diff| < 0.01 s), except prc-cs7's OOF (prc-m-g7-ab, output not kept locally) and
prc-mlp7's refits (Colab prc-c-mlp7-r/-r2, seeds 2911-2916), which follow the run log.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# member -> {"oof": {fold: [kernel or kernel/variant, ...]}, "refit": [kernel or kernel/variant, ...]}
PROVENANCE = {
    "prc-mlp6fs": {"oof": {"A": ["prc-m-mlp6fs-s1-a", "prc-m-mlp6fs-s2-a", "prc-m-mlp6fs-s3-a"],
                           "B": ["prc-m-mlp6fs-s1-b", "prc-m-mlp6fs-s2-b", "prc-m-mlp6fs-s3-b"]},
                   "refit": [f"prc-m-mlp6fs-r{i}" for i in range(1, 7)]},
    "prc-cs6": {"oof": {"A": ["prc-m-cs6-a"], "B": ["prc-p-cs6-b"]}, "refit": ["prc-p-cs6-r"]},
    "prc-c6": {"oof": {"A": ["prc-m-c6-a"], "B": ["prc-p-c6-b"]}, "refit": ["prc-p-c6-r", "prc-m-c6-r2", "prc-m-c6-r3"]},
    "prc-cs5": {"oof": {"A": ["prc-m-cs5-a"], "B": ["prc-p-cs5-b"]}, "refit": ["prc-m-cs5-r", "prc-p-cs5-r2"]},
    "prc-c5d10": {"oof": {"A": ["prc-m-c5-a"], "B": ["prc-m-c5-b"]}, "refit": ["prc-m-c5-r", "prc-m-c5-r2", "prc-p-c5-r3"]},
    "prc-split-fs4": {"oof": {"A": ["prc-p-split-fs4-a"], "B": ["prc-p-split-fs4-b"]},
                      "refit": ["prc-p-split-fs4-r", "prc-p-split-fs4-r2"]},
    "prc-c7x2": {"oof": {"A": ["prc-m-c7-a", "prc-m-c7x2-a/foldA"], "B": ["prc-m-c7-b", "prc-m-c7x2-ab"]},
                 "refit": ["prc-m-c7-r", "prc-m-c7-r2", "prc-m-c7x2-a/r8101"]},
    "prc-cs7": {"oof": {"A": ["prc-m-g7-ab/cs7"], "B": ["prc-m-g7-ab/cs7"]}, "refit": ["prc-m-g7-r/cs7r"]},
    "prc-mlp7": {"oof": {"A": ["prc-c-mlp7-ab/mlp7s1", "prc-c-mlp7-ab/mlp7s2", "prc-c-mlp7-ab/mlp7s3"],
                         "B": ["prc-c-mlp7-ab/mlp7s1", "prc-c-mlp7-ab/mlp7s2", "prc-c-mlp7-ab/mlp7s3"]},
                 "refit": ["prc-c-mlp7-r", "prc-c-mlp7-r2"]},
    "prc-c6s": {"oof": {"A": ["prc-p-c6s-a"], "B": ["prc-p-c6s-b"]}, "refit": ["prc-m-c6s-r"]},
    "prc-c5s": {"oof": {"A": ["prc-m-c5s-a"], "B": ["prc-m-c5s-b"]}, "refit": ["prc-m-c5s-r", "prc-m-c5s-r2", "prc-m-c5s-r3"]},
    "prc-x5": {"oof": {"A": ["prc-p-x5-a"], "B": ["prc-p-x5-b"]}, "refit": ["prc-p-x5-r", "prc-m-x5-r2"]},
}
STORES = ["prc-fs4", "prc-fs5", "prc-fs6", "prc-fs7"]
EXTRA = {"prc-cv-k4": "solutions/cover/prc-cv-k4"}  # LIRF no-ADS-B overlay member
URL = re.compile(r"https://(?:www\.kaggleusercontent\.com|storage\.googleapis\.com|[^'\"\s]*\.googleusercontent\.com)[^'\"\s]*")


def sha16(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def spec_of(text: str) -> dict:
    m = re.search(r"^SPEC = (\{.*\})$", text, re.M)
    if not m:
        return {}
    try:
        return ast.literal_eval(m.group(1))
    except Exception:  # noqa: BLE001
        return {"unparsed": True}


def describe(src: Path, ref: str) -> dict:
    slug, _, variant = ref.partition("/")
    d = src / "solutions" / "kernels" / slug
    if not d.exists() and slug in EXTRA:
        d = src / EXTRA[slug]
    text = (d / "kernel.py").read_text() if (d / "kernel.py").exists() else ""
    meta = json.loads((d / "kernel-metadata.json").read_text()) if (d / "kernel-metadata.json").exists() else {}
    spec = spec_of(text)
    if variant and "variants" in spec:
        spec = {**{k: v for k, v in spec.items() if k != "variants"}, **spec["variants"].get(variant, {})}
    spec.pop("fs_urls", None)
    p = spec.get("params", {})
    out = {"kernel": meta.get("id", f"?/{slug}") + (f" (variant {variant})" if variant else ""),
           "accelerator": ("gpu" if meta.get("enable_gpu") else "tpu" if meta.get("enable_tpu") else
                           "colab" if slug.startswith("prc-c-") else "cpu"),
           "lib": spec.get("lib"), "store": spec.get("fs"),
           "seed": p.get("random_seed", p.get("seed", p.get("random_state"))),
           "folds": spec.get("folds"), "rounds": spec.get("rounds") or spec.get("fold_rounds"),
           "refit_seeds": spec.get("refit_seeds"),
           "options": {k: spec[k] for k in ("long_target", "split_by", "drop_features", "drop_adsb", "only")
                       if k in spec}}
    res = d / "output" / (variant or "") / "results.json"
    if res.exists():
        r = json.loads(res.read_text())
        if r.get("refit_rounds"):
            out["refit_rounds_used"] = r["refit_rounds"]
        if r.get("folds"):
            out["fold_best_rounds"] = {k: v.get("best") for k, v in r["folds"].items() if isinstance(v, dict)}
    return out


def snapshot(src: Path, slug: str) -> None:
    d = src / "solutions" / "kernels" / slug
    if not d.exists() and slug in EXTRA:
        d = src / EXTRA[slug]
    if not (d / "kernel.py").exists():
        return
    dst = REPO / "kernels" / slug
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "kernel.py").write_text(URL.sub("<redacted signed URL>", (d / "kernel.py").read_text()))
    if (d / "kernel-metadata.json").exists():
        (dst / "kernel-metadata.json").write_text((d / "kernel-metadata.json").read_text())


def main():
    src, cfg_path = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = json.loads((REPO / cfg_path).read_text())
    built = json.loads((src / "submissions" / "candidates" / f"{cfg['name']}.manifest.json").read_text())
    weights = {}
    for seg, v in built["segments"].items():
        for m, w in v["weights"].items():
            weights.setdefault(m, {})[seg] = w
    members = {}
    slugs = set(STORES) | set(EXTRA)
    for m, prov in PROVENANCE.items():
        if m not in weights:
            continue
        cur = sha16(src / "solutions" / "kernels" / m / "output" / "submission.parquet")
        members[m] = {"weights": weights[m],
                      "submission_sha16_at_build": built["inputs"].get(m), "submission_sha16_now": cur,
                      "oof": {f: [describe(src, r) for r in rs] for f, rs in prov["oof"].items()},
                      "refit": [describe(src, r) for r in prov["refit"]]}
        for rs in list(prov["oof"].values()) + [prov["refit"]]:
            slugs |= {r.split("/")[0] for r in rs}
    manifest = {
        "candidate": cfg["name"], "config": str(cfg_path), "output_sha16": built["output"][1],
        "segments": {s: {"rows": v["rows"]} for s, v in built["segments"].items()},
        "members": members,
        "stores": {s: describe(src, s) | {"kernel_sources": json.loads(
            (src / "solutions" / "kernels" / s / "kernel-metadata.json").read_text()).get("kernel_sources")}
            for s in STORES},
        "long_rows": {"source": built["long"][0], "sha16": built["long"][1], "rows": built["long"][2],
                      "built_by": "research/tail/longrows_v30.py <v29 base> OUT --alpha 0.5 --phys plog --keep-aobt3-over 4 --old <v21> (inputs: research/tail/expand_artifact.py)"},
        "overlays": built.get("overlays"), "overlay_kernels": {"prc-cv-k4": describe(src, "prc-cv-k4")},
        "capped_rows": built.get("capped_rows"), "level": built.get("level"),
    }
    (REPO / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str))
    for s in sorted(slugs):
        snapshot(src, s)
    print(f"MANIFEST.json: {len(members)} members, {len(slugs)} kernels snapshotted under kernels/")


if __name__ == "__main__":
    main()
