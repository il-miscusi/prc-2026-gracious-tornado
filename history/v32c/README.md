# Board upload v31 (internal candidate v32c)

v31 was uploaded to the January/July 2026 board on 2026-10-02. This folder records how it was
built. The final submission does not read anything here.

- `v32c.json`: the candidate config of the old assembler (members, segment weights, overlay, cap,
  level calibration, long-row source).
- `MANIFEST.json`, `MANIFEST.txt`: per member, its stack weight and the Kaggle kernels, seeds and
  rounds behind each out-of-fold fold and each refit, with the hashes of the member files.
- `kernels/<slug>/`: each of those kernels as it ran (`kernel.py` and `kernel-metadata.json`), from
  the development tree. Two kinds of text were replaced: the signed download URLs that kernels on
  the second account received (`<redacted signed URL>`), and one absolute data path of the
  development machine in the input search list (`# <redacted: development machine data path>`).
  Nothing else differs from what ran.
- `snapshot_kernels.py`: the script that wrote this folder on 2026-10-01, with the member-to-kernel
  map it used.

The current assembler rebuilds v31 byte for byte from these members' predictions:
`research/final/configs/v32c_repro_board.json`, described in `docs/VALIDATION.md`. The scripts
that originally assembled v31 (`research/assemble.py`, `stack.py` and others) were replaced by
`research/final/` and remain in this repository's git history (commit 23869ee on branch `main`).

The models behind v31's member predictions were not saved, so v31 cannot predict a new month.
The final submission's members are new refits of the same specifications with their models saved.
