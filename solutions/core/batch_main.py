# %%
# Batch runner: train several SPECs on one large host (the TPU v5e VM's 96 cores) as
# parallel processes, each with its share of the cores, all reading the same feature
# store.  Each spec writes to OUTPUT_DIR/<name>/ exactly what a single trainer writes.

SPECS = {}  # replaced by build_kernel.py: {name: spec}
FS_URLS = {}  # replaced by build_kernel.py


def _run_spec(name: str, spec: dict, threads: int) -> None:
    g = globals()
    g["SPEC"] = {**spec, "threads": threads, "parallel": spec.get("parallel", False)}
    g["MAX_ROUNDS"] = int(os.environ.get("COMPBOT_ROUNDS", spec.get("max_rounds", 5000)))
    out = BATCH_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    g["OUTPUT_DIR"] = out
    STATE.clear()
    t0 = time.time()
    try:
        train_main()
        print(f"[batch] {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        (out / "error.txt").write_text(traceback.format_exc())
        print(f"[batch] {name} FAILED: {exc}", flush=True)
        raise


BATCH_ROOT = OUTPUT_DIR


def batch_main():
    from multiprocessing import get_context
    if FS_URLS:
        # fetch the store once, before forking (the spec processes would race on the files)
        fetch_urls(FS_URLS, Path("/tmp/fs"))
        os.environ["FS_ROOT"] = "/tmp/fs"
    total = ncpu()
    names = list(SPECS)
    wsum = sum(float(SPECS[n].get("weight", 1)) for n in names)
    per = {n: max(int(total * float(SPECS[n].get("weight", 1)) / wsum), 1) for n in names}
    print(f"[batch] {len(names)} specs on {total} cores: {per}", flush=True)
    ctx = get_context("fork")
    procs = {n: ctx.Process(target=_run_spec, args=(n, SPECS[n], per[n])) for n in names}
    for p in procs.values():
        p.start()
    for p in procs.values():
        p.join()
    codes = {n: p.exitcode for n, p in procs.items()}
    (BATCH_ROOT / "batch.json").write_text(json.dumps({"codes": codes, "used_h": used_h()}))
    print(f"[batch] exit codes {codes}  used_h={used_h():.3f}", flush=True)


if __name__ == "__main__":
    batch_main()
