"""Rebuild the two historical inputs of longrows_v30.py from the committed artifact.

    python3 research/tail/expand_artifact.py OUT_DIR

longrows_v30.py (v32c's rows over 4 h, `--alpha 0.5`) reads two earlier candidates of this
pipeline: v29 (BASE: the value kept by LIRF rows with no NM match and by rows whose AOBT_3 lies
more than 4 h before takeoff) and v21 (`--old`: the pre-blend long stack M_old, weight 1 - alpha).
Only their rows with d_sched > 2 h can reach the output, so research/tail/artifacts/
longrows_inputs_over2h.parquet keeps exactly those (MVT_ID_mvt, v29, v21; 10,168 rows).  This
writes full-length BASE and OLD files (other rows 60 s, which never reach the > 4 h rows) so

    python3 research/tail/longrows_v30.py OUT_DIR/v29_base.parquet LONG.parquet \
        --alpha 0.5 --phys plog --keep-aobt3-over 4 --old OUT_DIR/v21_old.parquet

reproduces research/tail/out/v30_longrows_a05_plog.parquet on every row over 4 h (checked
2026-10-01: max |diff| 0 s, sha256[:16] 3e86fc2f7f7d037d for the full v29-based file).
For a new month there is no v21/v29: run longrows_v30.py with --alpha 1 and BASE = the
stack-only candidate (see PHASE2_DRILL.md).
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parents[1] / "data" / "09-22-2026-08-13-30_files_list" / "submitting.parquet"


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    a = pd.read_parquet(HERE / "artifacts" / "longrows_inputs_over2h.parquet").set_index("MVT_ID_mvt")
    ids = pd.read_parquet(TEMPLATE)["MVT_ID_mvt"]
    for col, name in (("v29", "v29_base.parquet"), ("v21", "v21_old.parquet")):
        v = a[col].reindex(ids.to_numpy()).fillna(60.0).to_numpy()
        pd.DataFrame({"MVT_ID_mvt": ids.to_numpy(), "TAXITIME_SEC_mvt": v}).to_parquet(out / name, index=False)
        print(out / name)


if __name__ == "__main__":
    main()
