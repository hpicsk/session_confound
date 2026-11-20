"""
h4_build_image_cache.py — build the (40000, 256, 256, 3) uint8 image cache that the fusion
experiment (h4_fusion_collapse.py `run`) loads via load_images(). Preparation only, separate from
the CDD study's model (DaconModel) and from any training: it extracts ONLY the 40k
subsample's imagery from the AI Hub dataset's source zips, resizes each to 256x256, and writes SCRATCH/h4_img256.npy aligned to the
subsample row order (row j == the image for sub.iloc[j]['image'], the order load_metadata() and
PanelDS index by). Nothing is trained here; no GPU is touched.

Choices that follow the CDD study's training procedure (both models read the same cache, so they
also cancel in the gain delta):
  - RGB channel order. albumentations A.Normalize's default mean/std are RGB, so the pipeline
    expects RGB; cv2 decodes BGR, hence the explicit BGR->RGB.
  - Resize to 256x256 with INTER_LINEAR — exactly A.Resize(PX,PX)'s default interpolation. Pre-
    resizing is then a transparent caching step: the pipeline's own A.Resize(256,256) on an
    already-256 image is identity, so the cache equals what "load native JPG -> pipeline" would
    produce (and INTER_LINEAR is also torchvision's default, so the absolute macro-F1 stays
    comparable to a standard ResNet-50 baseline).
  - Aspect ratio is NOT preserved: A.Resize stretches to a square, and cv2.resize(.,(PX,PX))
    matches that. (The AI Hub dataset's source images are square anyway — 2832x2832.)

Fails loudly rather than writing a placeholder:
  - every one of the 40k rows must be claimed by exactly one source entry, checked as a bijection
    before any decoding — a missing image or a basename found in two zips aborts the plan;
  - an undecodable image raises (never a zero row written in its place);
  - the output is written to a .tmp and os.replace()'d to the final name only after all 40k rows
    are decoded, so a partial cache can never masquerade as complete (load_images() would then
    silently train on black frames).

    TROPICAL_SCRATCH=/path/to/extraction DATA=".../3.개방데이터/1.데이터" \
        python h4_build_image_cache.py
"""
import glob
import os
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# PX and the paths must match h4_fusion_collapse.py. Kept local (not imported) so this
# data-prep script never imports torch into a fork() parent. Drift is not silent: load_images()
# loads SCRATCH/f"h4_img{PX}.npy", so a PX mismatch surfaces as its fail-loud "cache missing".
PX = 256                                        # image side; equals h4_fusion_collapse.py's PX
SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])     # no fallback; same as h4_fusion_collapse
SUBSAMPLE = SCRATCH / "h4_subsample.parquet"    # the 40k membership (materializer)
CACHE = SCRATCH / f"h4_img{PX}.npy"             # EXACTLY what load_images() loads
TMP = SCRATCH / f"h4_img{PX}.npy.tmp"
DATA = Path(os.environ["DATA"])                 # AI Hub dataset root (README); no fallback
SRC_DIRS = [DATA / "Training" / "01.원천데이터", DATA / "Validation" / "01.원천데이터"]
N = 40000                                       # subsample size (h4_materialize_splits.py)
WORKERS = 48                                    # 128 cores available; I/O-friendly cap


def plan_worklists():
    """Map every subsample image to its source-zip entry. Returns {zippath: [(entry, j), ...]}
    where j is the row index the decoded image must land on. Asserts the map is a bijection onto
    0..N-1 (complete coverage, no duplicates) before any decoding happens."""
    sub = pd.read_parquet(SUBSAMPLE).reset_index(drop=True)  # identical order to load_metadata()
    assert len(sub) == N, f"subsample is {len(sub)} rows, expected {N}"
    needed = {}
    for j, img in enumerate(sub["image"]):
        assert img not in needed, f"duplicate image {img!r} in subsample"
        needed[img] = j

    zips = sorted({z for d in SRC_DIRS for z in glob.glob(f"{d}/*.zip")})
    assert zips, f"no source zips under {[str(d) for d in SRC_DIRS]}"
    plan, owner = {}, {}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            items = []
            for entry in zf.namelist():
                if entry.endswith("/"):
                    continue
                j = needed.get(os.path.basename(entry))
                if j is None:
                    continue
                bn = os.path.basename(entry)
                assert bn not in owner, f"{bn} appears in two zips: {owner[bn]} and {z}"
                owner[bn] = z
                items.append((entry, j))
        if items:
            plan[z] = items

    missing = [img for img in needed if img not in owner]
    assert not missing, f"{len(missing)} subsample images not found in any zip, e.g. {missing[:5]}"
    all_j = sorted(j for items in plan.values() for _, j in items)
    assert all_j == list(range(N)), "worklist is not a bijection onto 0..N-1 (coverage gap)"
    return plan


def build_zip(args):
    """Worker: decode every assigned entry of one zip into the shared memmap. Rows are disjoint
    across zips (bijection), so no coordination is needed. Fail loud on an undecodable image."""
    cv2.setNumThreads(1)  # one decode thread per process; the pool provides the parallelism
    zippath, items = args
    cache = np.load(TMP, mmap_mode="r+")
    t = time.time()
    with zipfile.ZipFile(zippath) as zf:
        for entry, j in items:
            arr = cv2.imdecode(np.frombuffer(zf.read(entry), np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                raise RuntimeError(f"undecodable image {entry!r} in {zippath}")
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            cache[j] = cv2.resize(arr, (PX, PX), interpolation=cv2.INTER_LINEAR)
    cache.flush()
    return os.path.basename(zippath), len(items), time.time() - t


def main():
    if CACHE.exists():
        print(f"cache already built: {CACHE} ({CACHE.stat().st_size / 1e9:.2f} GB) — nothing to do")
        return

    print("planning: mapping the 40k subsample images to source-zip entries ...", flush=True)
    t0 = time.time()
    plan = plan_worklists()
    total = sum(len(v) for v in plan.values())
    print(f"  {N} rows over {len(plan)} zips, {total} entries, bijection verified "
          f"({time.time() - t0:.1f}s)", flush=True)

    print(f"allocating {TMP} ({N * PX * PX * 3 / 1e9:.2f} GB) ...", flush=True)
    mm = np.lib.format.open_memmap(TMP, mode="w+", dtype=np.uint8, shape=(N, PX, PX, 3))
    del mm  # workers reopen it r+ and write disjoint rows

    print(f"decoding with {WORKERS} workers ...", flush=True)
    t0, done = time.time(), 0
    with Pool(WORKERS) as pool:
        for name, k, dt in pool.imap_unordered(build_zip, sorted(plan.items())):
            done += k
            print(f"  {name:<58s} {k:5d} imgs {dt:6.1f}s   [{done:5d}/{N}]", flush=True)
    assert done == N, f"decoded {done} rows != {N} (a worker dropped entries)"

    os.replace(TMP, CACHE)  # atomic publish; only reached when all N rows are decoded
    print(f"\nDONE: {CACHE} ({CACHE.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.1f}s")
    print("load_images() now returns this cache. The GPU run is a separate step "
          "(`python h4_fusion_collapse.py run`).")


if __name__ == "__main__":
    main()
