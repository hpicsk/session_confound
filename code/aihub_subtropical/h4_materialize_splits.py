"""
h4_materialize_splits.py — the subsample and the splits of the image–sensor fusion experiment.

Materializes the 40k subsample and the two splits the GPU run trains on, then
checks coverage on the *realized* subsample before any model is trained. Reads panel.parquet
and the 원천 source-zip *namelists* — to restrict the draw to images AI Hub actually
distributed as pixels (see `imaged_basenames`/`load_panel`); decodes no pixel, trains
nothing. The check keeps a stratified subsample and a date cut from silently stranding a
small cell in the test fold, which would make a drop in the gain that is really a failure on
classes absent from training.

Four conditions, checked on the partition the run uses: every class of the date-disjoint
test fold occurs in its training data (>= 0.99 of test images), >= POSITIVITY_FLOOR of its
test images' cells do, its leak rate is <= LEAK_REMOVED, and the in-distribution split's
leak rate is >= LEAK_FLOOR. The manuscript states the first (sec:intro); the two
leak-rate thresholds are leakage_collapse_gate.py's, and the manuscript prints the realized
leak rates instead (sec:fusion). If a condition fails, the subsample is enlarged along
ENLARGE_LADDER.

DEGREES OF FREEDOM:
  MEASURED  — every coverage and leak-rate number below, from panel.parquet.
  CHOSEN (structural) — the 40,000 subsample size, stratification by `cell` (48
    crop·part·class cells), the two splits {in-distribution, date},
    GroupKFold(5)/KFold(5) fold 0 as the single train/test partition per split (the
    run uses ONE partition per split; the 5 seeds supply the spread, not 5 CV folds), and
    SUBSAMPLE_SEED for the draw + the in-distribution shuffle (structural — NOT one of the 5
    training seeds).
  FITTED    — none. No threshold is tuned to any target; three thresholds are imported from
    leakage_collapse_gate.py so there is one source of truth.

WHAT THIS DOES NOT DO. It does not carve the checkpoint-selection validation set: that is a
date-disjoint sub-split of TRAIN, carved by the training script (h4_fusion_collapse.py). The
coverage conditions compare train with test only, so this stops there.

Panel-tuple identity, the 9 panel columns, and the three shared thresholds are imported from
leakage_collapse_gate.py so this check measures the SAME leak rate the candidate-split comparison
did; only CLS_POS_FLOOR is defined locally.
"""

import glob
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so the sibling import resolves
from leakage_collapse_gate import (  # noqa: E402  single source of truth
    _keyify, PANEL, LEAK_FLOOR, LEAK_REMOVED, POSITIVITY_FLOOR,
)

SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])  # no fallback; see README.md
HERE = Path(__file__).resolve().parent   # results land beside the script that
                                         # writes them: the path the paper reads
PANEL_PARQUET = SCRATCH / "panel.parquet"
DATA = Path(os.environ["DATA"])  # AI Hub dataset root; no fallback. IDENTICAL to h4_build_image_cache.py
SRC_DIRS = [DATA / "Training" / "01.원천데이터", DATA / "Validation" / "01.원천데이터"]  # source-image zips

# The subsample size; the ladder enlarges it if a coverage condition fails.
SUBSAMPLE_TARGET = 40_000
ENLARGE_LADDER = [40_000, 50_000, 60_000, 80_000, 120_000, 200_000, None]  # None == the whole dataset
SUBSAMPLE_SEED = 0          # structural — the draw + in-distribution shuffle; NOT a training seed
N_FOLDS = 5                 # the run's partition = fold 0; matches leakage_collapse_gate.py's GroupKFold(5)
N_CLS_EXPECTED = 18         # 17 pest and disease classes and a healthy class, all needed on both sides (cls_pos)

# Coverage thresholds. Three are imported from leakage_collapse_gate.py (single source);
# cls_pos is defined here.
CLS_POS_FLOOR = 0.99        # every class on both sides of the date-disjoint split (sec:intro)

MASTER_OUT = SCRATCH / "h4_subsample.parquet"        # the 40k membership + 2 split columns
RECHECK_OUT = HERE / "h4_positivity_recheck.csv"     # the coverage table the suite reads


def imaged_basenames():
    """The set of image basenames actually distributed as pixels in the 원천 (source-image) zips.

    panel.parquet is built by identifiability.py from Other.zip *sensor* metadata and is a
    superset of the imaged population: AI Hub shipped ~27k more sensor readings (274k) than image
    files (246.6k = the main-label count), so ~10% of panel rows have NO distributed pixels
    anywhere. An image+sensor dataset needs BOTH inputs, so the fusion experiment's draw is
    restricted to this set — otherwise the cache builder's fail-loud bijection aborts on the
    rows without pixels. Reads zip *namelists* only (no pixel decoded), using
    the SAME SRC_DIRS + exact-basename matching as h4_build_image_cache.py so the subsample this
    emits is exactly what the builder can find; the builder's bijection assert is the backstop if
    the two ever drift. The analyses that read no image keep the full 274k panel (they need no
    pixels) — only the fusion experiment restricts."""
    names = set()
    for d in SRC_DIRS:
        for z in sorted(glob.glob(f"{d}/*.zip")):
            with zipfile.ZipFile(z) as zf:
                names.update(os.path.basename(e) for e in zf.namelist() if not e.endswith("/"))
    assert names, f"no source-image zips under {[str(d) for d in SRC_DIRS]}"
    return names


def load_panel():
    assert PANEL_PARQUET.exists(), (
        f"panel.parquet missing at {PANEL_PARQUET}; regenerate via identifiability.py"
    )
    df = pd.read_parquet(PANEL_PARQUET).reset_index(drop=True)

    # Restrict to the imaged population BEFORE any draw: keep only rows whose image was actually
    # distributed as pixels (panel ∩ 원천 — see imaged_basenames). Without this
    # the proportional draw pulls in ~10% rows without pixels and the cache builder fail-loud aborts.
    src = imaged_basenames()
    n_before = len(df)
    df = df[df["image"].isin(src)].reset_index(drop=True)
    n_drop = n_before - len(df)
    print(f"imaged-population filter: {n_before:,} panel rows -> {len(df):,} with distributed "
          f"pixels ({n_drop:,} rows without pixels dropped, {100 * n_drop / n_before:.1f}%)")
    assert len(df) > 0, "no panel rows survived the imaged-population filter — check DATA/SRC_DIRS"

    df["pdate"] = pd.to_datetime(df["photo_dt"], errors="coerce").dt.normalize()
    df["panel_key"] = _keyify(df, PANEL)  # the model's sensor input — the leak rate is measured on THIS
    n_cls = df["cls"].nunique()
    assert n_cls == N_CLS_EXPECTED, (  # fail loud: the class count broke upstream
        f"expected {N_CLS_EXPECTED} classes, panel.parquet has {n_cls}"
    )
    assert int(df["pdate"].isna().sum()) == 0, "unexpected NaT photo_dt; date-split groups would break"
    return df


def draw_subsample(df, size, seed):
    """Proportional stratified draw by `cell` (preserves the natural class prior), exact `size`
    via largest-remainder apportionment, deterministic given `seed`. Returns a subsample frame."""
    counts = df["cell"].value_counts()
    exact = counts / counts.sum() * size
    quota = np.floor(exact).astype(int)
    remainder = size - int(quota.sum())
    if remainder:  # hand the leftover seats to the largest fractional parts (Hamilton method)
        frac_order = (exact - quota).sort_values(ascending=False).index[:remainder]
        quota.loc[frac_order] += 1
    assert (quota <= counts).all(), "a cell quota exceeds its available images — enlarge failed"
    parts = [df[df["cell"] == cell].sample(n=int(quota[cell]), random_state=seed)
             for cell in counts.index]
    sub = pd.concat(parts).sort_values("image").reset_index(drop=True)
    assert len(sub) == size and sub["cls"].nunique() == N_CLS_EXPECTED
    return sub


def _fold0_and_mean(sub, axis):
    """For one axis: fold-0 (the train/test partition the run uses) metrics + the 5-fold mean
    (comparable with leakage_collapse_gate.py). Returns (test_mask_fold0, fold0_metrics, mean_metrics)."""
    idx = np.arange(len(sub))
    if axis == "iid":
        gen = KFold(N_FOLDS, shuffle=True, random_state=SUBSAMPLE_SEED).split(idx)
    else:
        gcol = {"date": "pdate"}[axis]
        groups = sub[gcol].astype(str).to_numpy()
        gen = GroupKFold(N_FOLDS).split(idx, groups=groups)
    per_fold, fold0 = [], None
    for i, (tr, te) in enumerate(gen):
        train, test = sub.iloc[tr], sub.iloc[te]
        m = dict(
            panel_leak=float(test["panel_key"].isin(set(train["panel_key"])).mean()),
            cls_pos=float(test["cls"].isin(set(train["cls"])).mean()),
            cell_pos=float(test["cell"].isin(set(train["cell"])).mean()),
            farm_overlap=float(test["farm"].isin(set(train["farm"])).mean()),
        )
        per_fold.append(m)
        if i == 0:
            mask = np.zeros(len(sub), dtype=bool)
            mask[te] = True
            fold0 = (mask, m, len(te), len(tr))
    mean = {k: float(np.mean([f[k] for f in per_fold])) for k in per_fold[0]}
    return fold0[0], fold0[1], mean, fold0[2], fold0[3]


def check_coverage_conditions(m):
    """Evaluate the four conditions on the fold-0 metrics.
    Returns (checks: list of (name, passed, observed), failures: list[str])."""
    checks = [
        ("cls_pos(date)  >= 0.99", m["date"]["cls_pos"] >= CLS_POS_FLOOR,
         f"{m['date']['cls_pos']:.4f}"),
        ("cell_pos(date) >= 0.80", m["date"]["cell_pos"] >= POSITIVITY_FLOOR,
         f"{m['date']['cell_pos']:.4f}"),
        ("panel_leak(date) <= 0.10", m["date"]["panel_leak"] <= LEAK_REMOVED,
         f"{m['date']['panel_leak']:.4f}"),
        ("panel_leak(in-dist) >= 0.50", m["iid"]["panel_leak"] >= LEAK_FLOOR,
         f"{m['iid']['panel_leak']:.4f}"),
    ]
    failures = [f"{name} [{val}]" for name, ok, val in checks if not ok]
    return checks, failures


def main():
    df = load_panel()
    print("=" * 92)
    print("FUSION SUBSAMPLE + SPLITS — materialization & coverage check")
    print("=" * 92)
    print(f"dataset: {len(df):,} images | {df['cls'].nunique()} classes "
          f"{df['cell'].nunique()} cells {df['farm'].nunique()} farms {df['pdate'].nunique()} dates")

    chosen = None
    for size in ENLARGE_LADDER:
        eff = len(df) if size is None else size
        sub = df.copy() if size is None else draw_subsample(df, size, SUBSAMPLE_SEED)
        masks, fold0, mean = {}, {}, {}
        sizes = {}
        for axis in ("iid", "date"):
            mask, m0, mmean, n_te, n_tr = _fold0_and_mean(sub, axis)
            masks[axis], fold0[axis], mean[axis] = mask, m0, mmean
            sizes[axis] = (n_te, n_tr)
        checks, failures = check_coverage_conditions(fold0)

        tag = "whole dataset" if size is None else f"{eff:,}"
        print("\n" + "-" * 92)
        print(f"SUBSAMPLE = {tag}  (partition = fold 0 of {N_FOLDS}; "
              f"test/train sizes: in-distribution {sizes['iid']} date {sizes['date']})")
        print("-" * 92)
        hdr = f"{'axis':>5} | {'leak(f0)':>9} {'cls_pos':>8} {'cell_pos':>9} {'farm_ov':>8} "\
              f"|| {'leak(5f)':>9} {'cls_pos':>8} {'cell_pos':>9} {'farm_ov':>8}"
        print(hdr)
        for axis in ("iid", "date"):
            a, b = fold0[axis], mean[axis]
            print(f"{axis:>5} | {a['panel_leak']:>9.4f} {a['cls_pos']:>8.4f} {a['cell_pos']:>9.4f} "
                  f"{a['farm_overlap']:>8.4f} || {b['panel_leak']:>9.4f} {b['cls_pos']:>8.4f} "
                  f"{b['cell_pos']:>9.4f} {b['farm_overlap']:>8.4f}")
        print("  coverage conditions (on the fold-0 partition — what the run trains on):")
        for name, ok, val in checks:
            print(f"    [{'PASS' if ok else 'FAIL'}] {name}   observed {val}")

        if not failures:
            chosen = (eff, sub, masks, fold0, mean, sizes)
            if eff != SUBSAMPLE_TARGET:
                print(f"\n  NOTE: coverage required enlarging past {SUBSAMPLE_TARGET:,} "
                      f"to {eff:,}; this size is now the one the run uses.")
            break
        print(f"\n  coverage FAILED at {tag}: {failures} — enlarging.")
    else:
        raise AssertionError(
            "coverage check FAILED even on the whole dataset — this contradicts the "
            f"candidate-split comparison (leakage_collapse_gate.csv). Do NOT train. Failures: {failures}"
        )

    eff, sub, masks, fold0, mean, sizes = chosen

    # ---- write the master: 40k membership + the two split columns (one source of
    #      truth for membership, so the two splits cannot drift apart) ----
    out = sub[["image", "farm", "crop", "part", "cls", "cell", "pdate"]].copy()
    for axis, col in (("iid", "split_iid"), ("date", "split_date")):
        out[col] = np.where(masks[axis], "test", "train")
    out.to_parquet(MASTER_OUT, index=False)

    rows = []
    for axis in ("iid", "date"):
        r = {"axis": axis, "n_test": sizes[axis][0], "n_train": sizes[axis][1]}
        r.update({f"{k}_fold0": v for k, v in fold0[axis].items()})
        r.update({f"{k}_mean5": v for k, v in mean[axis].items()})
        rows.append(r)
    recheck = pd.DataFrame(rows)
    recheck.insert(0, "subsample_n", eff)
    recheck.to_csv(RECHECK_OUT, index=False)

    print("\n" + "=" * 92)
    print(f"coverage PASS at n={eff:,}. Wrote:")
    print(f"  {MASTER_OUT}   (image + cell/cls/farm/pdate + split_iid/split_date)")
    print(f"  {RECHECK_OUT}  (per-axis fold-0 + 5-fold-mean leak rate/coverage)")


if __name__ == "__main__":
    main()
