"""
CDD — the study's own committed splits, scored by classifiers that read no image.
Part of the analyses behind paper/main.tex (sec:reanalysis, sec:res-cdd).

------------------------------------------------------------------------------
WHAT THIS SCRIPT MEASURES
------------------------------------------------------------------------------
dacon_published_split.py scores the metadata under a split WE draw:
StratifiedKFold(7), test share 1/7. The CDD study's reported figures
(10.3390/app14104322, Table 2, ResNet-50 row: image 0.8779 -> image+environment
0.9037) are measured on a split THEY committed. Differencing the two is a
comparison across non-identical test sets.

This script removes the mismatch: it scores the same classifiers, the same
estimator and the same metric on the study's own committed identifiers, so the
resulting number and their 0.9037 are measured over the same test images.
Nothing here reads an image or runs a GPU.

Their training entry point hardcodes ONE of the two committed sets:

    src/main.py:48-49   train/val <- data/paper_work_seed777/{train,val}.csv
    src/main.py:153     test      <- data/paper_work_seed777/test.csv

so paper_work_seed777 is the split behind their Table 2 row, and it is the
split the outcome below is read on. paper_work_seed42 is
scored alongside it as the second committed set. The two are not nested, and
their test sets share 247 of their 1,154 identifiers, so both are reported. The
repository also ships data/5fold_seed42 (five training and validation lists, no
test partition), which src/main.py reads only in commented-out lines.

------------------------------------------------------------------------------
DEGREES OF FREEDOM
------------------------------------------------------------------------------
  split ............ MEASURED (read from their committed CSVs, not drawn here).
                     train = train.csv + val.csv, because nothing is tuned —
                     the same merge dacon_published_split.py makes and for the
                     same reason. Rows are used exactly as their files list
                     them, because their loader does; the lists hold no
                     duplicate and no identifier in both training and validation.
  label ............ CHOSEN (to match the CDD study): the 25-class
                     crop_disease_risk label their main.py predicts.
  metric ........... CHOSEN (to match the CDD study): macro F1. Balanced
                     accuracy is written beside it as a secondary metric; the
                     outcome mapping reads macro F1.
  env features ..... CHOSEN (structural): the 5-stat/channel cache, median-
                     imputed. Identical to dacon_published_split.py. Not
                     uniformly conservative against their LSTM — richer in
                     channel count, weaker in temporal modeling — and that
                     script's header states the comparison in full.
  timestamp features ... CHOSEN (structural): day_ord + hour, as in
                     dacon_published_split.py; not retuned here.
  estimator ........ CHOSEN: RandomForestClassifier(60, random_state=seed),
                     the same estimator as dacon_published_split.py.
  seeds ............ CHOSEN (structural): the project's 7-seed set, 0..6.
  nothing FITTED.

  BASELINE: RandomForestClassifier on timestamp + 5-stat env summaries |
    obvious alternative: "an LSTM over the raw 48 h series, as the CDD study
    used, would extract more from the channel" |
    tuning: none — every hyperparameter is carried over unchanged from
    dacon_published_split.py; no sweep was run on this split |
    discharge: sec:res-cdd, by scope — the manuscript states each comparison on
    this split for the random forests that sec:reanalysis names; the timestamp
    classifier, which reads no series, carries the headline on this split.

  WHAT THE INTERVAL IS. Their split files are fixed, so no split variance can
  enter any interval computed on them. The spread reported here is over the
  estimator seed alone. That is the same limitation sec:reanalysis states of the
  study's own intervals, which vary only the random seed of training, and it is a
  property of a fixed split rather than of either pipeline. No direction is
  claimed from it.

------------------------------------------------------------------------------
OUTCOME MAPPING
------------------------------------------------------------------------------
Read on timestamp+env macro F1 over paper_work_seed777, the split their code runs.
The thresholds are the study's own reported figures, quoted, not values we chose.

  C1  timestamp_env >= 0.9037 (their multimodal)
      => Metadata alone — timestamp plus the environment panel, no imagery —
         reaches the level the study reports for its fused image+environment
         model, on the study's own test images.

  C2  0.8779 <= timestamp_env < 0.9037
      => Metadata alone reaches the study's image branch but not its fused
         model, on the study's own test images. This is the outcome
         that sec:res-cdd reports.

  C3  timestamp_env < 0.8779 (their image-only)
      => Metadata alone reaches neither of the study's figures on its own
         test images.

Reported but outside the mapping: the same four classifiers on
paper_work_seed42, and the timestamp classifier alone on both. The leak already measured for each split
(dacon_audited_split_leakage.csv) is reprinted beside the scores so the number
sits next to the split property that explains it.

Run:  TROPICAL_SCRATCH=/path/to/cdd python dacon_their_split.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])   # hard lookup: no fallback
HERE = Path(__file__).resolve().parent           # written where the suite reads it
                                                 # (Paths that a checker reads)
META = SCRATCH / "dacon_meta.parquet"                  # per-image record, built by dacon_granularity.py
ENV_CACHE = SCRATCH / "dacon_env_feats_5stat.parquet"   # built by dacon_granularity.py
SPLIT_DIR = SCRATCH / "crop_disease" / "data"           # their pinned clone, @c825d9d
OUT = HERE / "dacon_their_split.csv"

SEEDS = [0, 1, 2, 3, 4, 5, 6]     # the project's variance sample; do NOT change post-hoc
N_TREES = 60
GROUP_KEY = "val_md5"             # timestamps-stripped env value hash

THEIR_SPLIT = "paper_work_seed777"   # src/main.py:48-49,153 hardcode this one
SECOND_SPLIT = "paper_work_seed42"

# The CDD study's reported figures (10.3390/app14104322, Table 2, ResNet-50).
# Their numbers, quoted as the thresholds of the outcome mapping above.
PAPER_IMAGE_ONLY_F1 = 0.8779
PAPER_MULTIMODAL_F1 = 0.9037


def load_split(name):
    """Their committed identifiers: train = train.csv + val.csv, nothing tuned."""
    d = SPLIT_DIR / name
    tr = pd.concat([pd.read_csv(d / "train.csv"), pd.read_csv(d / "val.csv")])
    te = pd.read_csv(d / "test.csv")
    return tr["image"].astype(str).to_numpy(), te["image"].astype(str).to_numpy()


def score(est, Xtr, ytr, Xte, yte):
    est.fit(Xtr, ytr)
    p = est.predict(Xte)
    return (f1_score(yte, p, average="macro", zero_division=0),
            balanced_accuracy_score(yte, p))


def main():
    assert META.exists(), f"missing {META}; run dacon_granularity.py"
    assert ENV_CACHE.exists(), f"missing {ENV_CACHE}; run dacon_granularity.py"
    assert SPLIT_DIR.exists(), f"missing {SPLIT_DIR}; clone the CDD study's repository"

    meta = pd.read_parquet(META)
    meta["image"] = meta["image"].astype(str)
    meta["day_ord"] = pd.to_datetime(meta["date"], format="%Y%m%d").map(pd.Timestamp.toordinal)
    t = meta["time"].astype(str).str.zfill(4)
    meta["hour"] = t.str[:2].astype(int) + t.str[2:].astype(int) / 60.0
    env = pd.read_parquet(ENV_CACHE)
    meta = meta.set_index("image")

    print("=" * 88)
    print("CDD — the study's own committed splits, scored without images")
    print(f"interrogating 10.3390/app14104322: image {PAPER_IMAGE_ONLY_F1} -> "
          f"multimodal {PAPER_MULTIMODAL_F1} (+2.58 %p), their split, their test images")
    print("=" * 88)

    def features(ids):
        m = meta.loc[ids]
        Xc = m[["day_ord", "hour"]].to_numpy(dtype=float)
        Xe = env.reindex(ids).to_numpy(dtype=float)
        return Xc, Xe, np.hstack([Xc, Xe]), m["label"].to_numpy(), m[GROUP_KEY].to_numpy()

    rows = []
    for split in [THEIR_SPLIT, SECOND_SPLIT]:
        tr_ids, te_ids = load_split(split)
        Xc_tr, Xe_tr, Xce_tr, y_tr, g_tr = features(tr_ids)
        Xc_te, Xe_te, Xce_te, y_te, g_te = features(te_ids)
        leak = np.isin(g_te, g_tr).mean()
        print(f"\n{split}: n_train={len(tr_ids)} n_test={len(te_ids)} "
              f"classes={len(set(y_tr) | set(y_te))} series leak={leak:.3f}")
        def fit_seed(seed):
            r = dict(split=split, seed=seed, n_train=len(tr_ids), n_test=len(te_ids),
                     group_leak=leak)
            arms = [("base", Xc_tr, Xc_te, False), ("timestamp", Xc_tr, Xc_te, False),
                    ("env", Xe_tr, Xe_te, True), ("timestamp_env", Xce_tr, Xce_te, True)]
            for arm, Xtr, Xte, imp in arms:
                if arm == "base":
                    est = DummyClassifier(strategy="most_frequent")
                else:
                    # One thread per forest, the seeds in parallel (below): a forest
                    # that predicts on several threads sums its trees' votes in no
                    # fixed order, so an exact tie resolves differently from run to run.
                    e = RandomForestClassifier(N_TREES, n_jobs=1, random_state=seed)
                    est = make_pipeline(SimpleImputer(strategy="median"), e) if imp else e
                r[f"{arm}_f1"], r[f"{arm}_ba"] = score(est, Xtr, y_tr, Xte, y_te)
            return r

        for r in Parallel(n_jobs=-1)(delayed(fit_seed)(seed) for seed in SEEDS):
            rows.append(r)
            print(f"  seed={r['seed']} | macroF1  base={r['base_f1']:.3f} "
                  f"timestamp={r['timestamp_f1']:.3f} env={r['env_f1']:.3f} "
                  f"timestamp+env={r['timestamp_env_f1']:.3f}")

    long = pd.DataFrame(rows)
    long.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")

    tcrit = stats.t.ppf(0.975, len(SEEDS) - 1)

    def ci(x):
        m, sd = x.mean(), x.std(ddof=1)
        h = tcrit * sd / np.sqrt(len(x))
        return m, m - h, m + h

    print("\n" + "=" * 88)
    print("MACRO F1 (mean [95% CI] over 7 estimator seeds; their split is fixed, so")
    print("no split variance enters these intervals)")
    print("=" * 88)
    summ = {}
    for split in [THEIR_SPLIT, SECOND_SPLIT]:
        print(f"\n{split}")
        for arm in ["base", "timestamp", "env", "timestamp_env"]:
            m, lo, hi = ci(long[long.split == split][f"{arm}_f1"])
            summ[(split, arm)] = m
            print(f"  {arm:10s} {m:.4f} [{lo:.4f}, {hi:.4f}]")

    ce = summ[(THEIR_SPLIT, "timestamp_env")]
    print("\n" + "=" * 88)
    print("OUTCOME")
    print("=" * 88)
    print(f"  timestamp+env on {THEIR_SPLIT} = {ce:.4f}")
    print(f"  their image-only {PAPER_IMAGE_ONLY_F1}   their multimodal {PAPER_MULTIMODAL_F1}")
    if ce >= PAPER_MULTIMODAL_F1:
        v = ("C1 — metadata alone reaches the level the study reports for its fused "
             "image+environment model, on the study's own test images.")
    elif ce >= PAPER_IMAGE_ONLY_F1:
        v = ("C2 — metadata alone reaches the study's image branch but not its fused "
             "model, on the study's own test images.")
    else:
        v = ("C3 — metadata alone reaches neither of the study's figures on its own "
             "test images.")
    print(f"  => {v}")


if __name__ == "__main__":
    main()
