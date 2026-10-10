# %%
# Inputs over URLs, for a kernel on an account that cannot mount the primary's private
# datasets and kernel outputs (miscusi).  build_kernel.py mints the URLs: signed GCS URLs for
# dataset files, tokenised kernel-output URLs; neither needs a credential on the VM.

INPUT_URLS = {}  # replaced by build_kernel.py: {relative path: url}


def fetch_inputs(root: str = "/tmp/in") -> None:
    import urllib.request
    base = Path(root)
    for rel, url in INPUT_URLS.items():
        out = base / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=600) as resp, open(out, "wb") as fh:
                    while True:
                        chunk = resp.read(8 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"input retry {attempt} {rel}: {exc}", flush=True)
                time.sleep(10 * (attempt + 1))
    os.environ["DATA_ROOT"] = str(base)
    print(f"inputs: {len(INPUT_URLS)} files under {base}  used_h={used_h():.3f}", flush=True)
