# %%
# Build the feature store on this host, then train the batch on it (one TPU session does
# both).  The store lands in the kernel output under fs/, reusable by later kernels.


def fsbatch_main():
    g = globals()
    base = g["OUTPUT_DIR"]
    fsdir = base / "fs"
    fsdir.mkdir(parents=True, exist_ok=True)
    g["OUTPUT_DIR"] = fsdir
    fs_main()
    g["OUTPUT_DIR"] = base
    os.environ["FS_ROOT"] = str(fsdir)
    batch_main()


if __name__ == "__main__":
    fsbatch_main()
