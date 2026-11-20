"""
h4_g4b_collapse.py — the fusion gain on the date-disjoint split, at the matched budget.

WHAT THIS MEASURES. At the CDD study's 300-epoch, batch-8 budget the in-distribution fusion gain
lies above zero (h4_fusion_matched_budget.py): delta_iid = +0.0084 [+0.0028, +0.0140] (paired 95%
CI, 5 seeds). This script measures that gain once a date-disjoint split lowers the leak rate (0.90 -> 0.073 on fold 0,
h4_positivity_recheck.csv).

The outcome that this script prints, evaluated at n=5, paired within seed. The manuscript
reports the gain and the paired reduction with their intervals, not this outcome (sec:fusion):
  COLLAPSE     : delta(date) CI includes 0  AND  drop = delta_iid - delta(date) CI excludes 0 (>0)
  SURVIVES     : delta(date) CI excludes 0 (the gain remains above zero)
  INCONCLUSIVE : else (the date run cannot separate a collapse from noise at n=5)
delta_iid(seed) is READ from the committed matched-budget CSV, NOT re-run. Both COLLAPSE
conditions are required: (i) alone can be a merely wider interval (an underpowered run), so (ii) — the
paired drop clearing seed noise — is what makes "collapse" the honest word.

DEGREES OF FREEDOM, identical to the matched-budget run except the split:
  MEASURED : the date partition (photo-date groups, h4_materialize_splits.py); its coverage
             (h4_positivity_recheck.csv).
  CHOSEN   : 300-epoch/batch-8 budget (= the CDD study's, carried from the matched-budget run);
             date-disjoint as the split that holds out time; seeds {0-4} (paired to the committed
             in-distribution gain).
  FITTED   : none — no knob tuned to any macro-F1 target.
BASELINE: ResNet-50 + the CDD study's late-fusion sensor branch + its 300-epoch/batch-8 training
  procedure, held identical across models AND across the in-distribution and date splits | obvious
  alternative a referee names: "the collapse is a split-difficulty artifact, not a leak effect" |
  discharge: the image-only model trains on the same date split, so generic date-split difficulty
  cancels in delta(date); the estimand is image+sensor - image.

INHERITED VERBATIM (imported, not copied — single source for the 300-epoch/batch-8 budget and the
train loop): from h4_fusion_matched_budget.py — train_arm, BS, EPOCHS, SEEDS, ARMS. From
h4_fusion_collapse.py — carve, ci, load_metadata, load_images. The ONLY differences from the
in-distribution run are the split (iid -> date) and the output file.

NOT re-run: h4_fusion_collapse.{py,csv} (40 epochs) and h4_fusion_matched_budget.{py,csv}
(300 epochs, in-distribution) are NOT touched. Output goes to its own file, h4_g4b_collapse.csv,
so the in-distribution table stays as committed and the checks that read it
(test_assertions.py::test_fusion_matched_budget_run_is_iid_only_and_complete) remain valid.

Run (each mode explicit; bare invocation prints usage and trains nothing):
  TROPICAL_SCRATCH=/path/to/extraction TROPICAL_GPU=0 python h4_g4b_collapse.py check  # no-GPU carve/coverage check
  TROPICAL_SCRATCH=/path/to/extraction TROPICAL_GPU=0 python h4_g4b_collapse.py date   # the ~3-GPU-day date run
"""

import argparse
import time

import numpy as np
import pandas as pd

# Single source: the matched-budget script owns the 300-epoch/batch-8 budget and the train loop; the
# 40-epoch script (imported by it) owns the model, data, metric and the carve. Importing (not copying)
# keeps train_arm / the budget constants / carve identical across the in-distribution and date runs, so the only
# differences here are the split and the output file (see header).
import h4_fusion_collapse as base
import h4_fusion_matched_budget as mb
from h4_fusion_collapse import carve, ci, load_images, load_metadata

IID_CSV = base.HERE / "h4_fusion_matched_budget.csv"     # committed in-distribution gain; the paired ref for drop (NOT re-run)
OUT = base.HERE / "h4_g4b_collapse.csv"                  # its own file; the other two tables stay as committed


def iid_delta_by_seed():
    """delta_iid(seed) = macroF1(all) - macroF1(img) from the COMMITTED matched-budget CSV — the
    paired reference for `drop`; read, never recomputed."""
    d = pd.read_csv(IID_CSV)
    w = d[d.split == "iid"].pivot_table(index="seed", columns="arm", values="test_f1")
    return w["all"] - w["img"]  # Series indexed by seed


def report(long, split):
    """This script's three-way printed outcome — COLLAPSE / SURVIVES / INCONCLUSIVE for the
    date split (header); the manuscript reports the gain and reduction instead."""
    w = long[long.split == split].pivot_table(index="seed", columns="arm", values="test_f1")
    dd = w["all"] - w["img"]  # delta(split) per seed
    md, ld, hd = ci(dd.values)
    print("\n" + "=" * 88)
    print(f"{split}-disjoint delta = all - img, {len(dd)} seeds paired")
    print("=" * 88)
    print(f"  {split} delta={md:+.4f} [{ld:+.4f}, {hd:+.4f}]   "
          f"img={w['img'].mean():.4f} all={w['all'].mean():.4f}")

    # paired drop vs the COMMITTED in-distribution gain, same seeds
    di = iid_delta_by_seed()
    common = sorted(set(dd.index) & set(di.index))
    drop = di.loc[common].to_numpy() - dd.loc[common].to_numpy()  # delta_iid - delta(date), within seed
    mdr, ldr, hdr = ci(drop)
    print(f"  drop = delta_iid - delta(date), paired on seeds {common} "
          f"(delta_iid mean {di.loc[common].mean():+.4f}):")
    print(f"  drop ={mdr:+.4f} [{ldr:+.4f}, {hdr:+.4f}]")

    date_ci_incl_0 = ld <= 0 <= hd
    date_ci_excl_0 = ld > 0
    drop_ci_excl_0 = ldr > 0
    print("\n" + "=" * 88)
    if date_ci_incl_0 and drop_ci_excl_0:
        verdict = "COLLAPSE"
        print("  COLLAPSE: delta(date) CI includes 0 AND the paired drop CI excludes 0. The\n"
              "  replicated in-distribution gain vanishes on the date-disjoint split and the\n"
              "  reduction clears seed noise.")
    elif date_ci_excl_0:
        verdict = "SURVIVES"
        print("  SURVIVES: delta(date) CI excludes 0. The fusion gain remains above zero once the\n"
              "  date-disjoint split lowers the leak rate.")
    else:
        verdict = "INCONCLUSIVE"
        print("  INCONCLUSIVE: delta(date) CI includes 0 but the drop CI also includes 0 — the\n"
              "  date run cannot separate a collapse from noise at n=5. No collapse claimed.")
    print("=" * 88)
    return dict(split=split, verdict=verdict, d=md, lo=ld, hi=hd, drop=mdr, drop_lo=ldr, drop_hi=hdr)


def run(split):
    """One split's 2 models x 5 seeds = 10 trainings at the matched 300-epoch/batch-8 budget (mb.train_arm).
    Resumable: completed (split, seed, arm) cells are skipped from OUT. Writes h4_g4b_collapse.csv."""
    print("=" * 88)
    print(f"FUSION RUN — {split}-disjoint, {len(mb.ARMS)} models x {len(mb.SEEDS)} seeds "
          f"({mb.EPOCHS}ep / batch{mb.BS}, no early-stop). Device {base.DEV}.")
    print("=" * 88)
    sub, y, env_raw = load_metadata()
    imgs = load_images(sub)  # fail-loud if the image cache is missing (see h4_build_image_cache.py)
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["split"], r["seed"], r["arm"]) for r in rows}
    n_this = len([d for d in done if d[0] == split])
    if n_this:
        print(f"resuming: {n_this}/{len(mb.SEEDS) * len(mb.ARMS)} {split} trainings already in {OUT}",
              flush=True)
    tr, va, te = carve(sub, split)
    print(f"{split} carve: tr/va/te = {len(tr)}/{len(va)}/{len(te)}  "
          f"classes tr/te = {len(np.unique(y[tr]))}/{len(np.unique(y[te]))}", flush=True)
    for seed in mb.SEEDS:
        for arm in mb.ARMS:
            if (split, seed, arm) in done:
                continue
            print(f"  --- {split} seed={seed} {arm} start ({time.strftime('%H:%M:%S')}) ---", flush=True)
            r = mb.train_arm(imgs, env_raw, y, tr, va, te, seed, arm)
            rows.append(dict(split=split, seed=seed, arm=arm, **r))
            print(f"  {split} seed={seed} {arm:3s} f1={r['test_f1']:.4f} val={r['val_f1']:.4f} "
                  f"best_ep={r['best_epoch']} {r['wall_s'] / 60:.1f}min", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
    report(pd.DataFrame(rows), split)
    print(f"\nwrote {OUT}")


def check():
    """No-GPU pre-flight: confirm the date carve keeps all 18 classes on both sides
    (its coverage) and echo the committed coverage table. Trains nothing, spends no GPU."""
    sub, y, env_raw = load_metadata()
    print(f"subsample: {len(sub)} rows, {len(np.unique(y))} classes")
    tr, va, te = carve(sub, "date")
    print(f"  date  carve: tr/va/te={len(tr)}/{len(va)}/{len(te)}  "
          f"classes tr/te={len(np.unique(y[tr]))}/{len(np.unique(y[te]))}  <-- needs all 18 both sides")
    recheck = base.HERE / "h4_positivity_recheck.csv"
    if recheck.exists():
        print(f"\ncommitted coverage check ({recheck.name}):")
        print(pd.read_csv(recheck).to_string(index=False))
    di = iid_delta_by_seed()
    print(f"\ncommitted in-distribution gain (paired ref for drop): delta_iid per seed = "
          f"{{{', '.join(f'{s}:{v:+.4f}' for s, v in di.items())}}}, mean {di.mean():+.4f}")


def main():
    ap = argparse.ArgumentParser(description="AI Hub fusion gain on the date-disjoint split (300 epochs, batch 8)")
    ap.add_argument("mode", nargs="?", choices=["check", "date"],
                    help="check: no-GPU carve/coverage pre-flight. "
                         "date: the ~3-GPU-day date-disjoint run.")
    args = ap.parse_args()
    if args.mode == "check":
        check()
    elif args.mode == "date":
        run("date")
    else:
        ap.print_help()
        print("\nNo mode given — nothing run. `check` is a no-GPU pre-flight; `date` spends GPU.")


if __name__ == "__main__":
    main()
