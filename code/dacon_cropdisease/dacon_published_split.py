"""
CDD — a random split against a split grouped by environment series. Reads no image.
Part of the analyses behind paper/main.tex (sec:reanalysis, sec:res-cdd).

------------------------------------------------------------------------------
THE STUDY WHOSE CLAIM THIS SCRIPT EXAMINES
------------------------------------------------------------------------------
Lee et al., "A Deep Learning-Based Crop Disease Diagnosis Method Using Multimodal
Mixup Augmentation," Applied Sciences 14(10):4322, 2024. DOI 10.3390/app14104322.
The same dataset as ours (CDD, LG AI Research), n=4,675 labeled samples, joint
crop/disease/severity prediction, macro F1.

  reported: image-only 0.8779 +/- 0.0070 -> image+environment 0.9037 +/- 0.0059
            (Table 2, ResNet-50 row; +2.58 %p, 3 runs, 95 % CI)
            env-only, their LSTM on 9 env columns: 0.8084 +/- 0.0091
  split   : NOT STATED. The paper says only "We divided the dataset into a ratio
            of 5:3:2 for training, validation, and testing sets, respectively."
            The word "random" appears in the paper only for mixup pair sampling
            and for "random seed" — never for the split.

On CDD, 5,767 images carry only 780 distinct environment series, so under a
random split almost every test image has its byte-identical series in training
(dacon_published_split.csv). The question is not whether the environment series
"helps" — it is whether what it supplies is agronomy or a lookup of the group
identity that a split without shared series removes.

------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES, AND WHAT IT DOES NOT DO
------------------------------------------------------------------------------
It opens no image and trains no detector. It measures the environment series'
own predictive power at the published task (25-class crop_disease_risk, macro
F1) under two splits that differ in EXACTLY ONE respect:

    random : StratifiedKFold(7, shuffle, seed)        <- shares series between the sides
    group  : StratifiedGroupKFold(7, shuffle, seed) grouped by val_md5

`group` is the MINIMAL correction. It is not a farm split and not a time split;
it only forbids the same environment window from appearing on both sides. If the
series' predictive power is agronomic it should barely notice. If it is a
lookup, it falls here.

This script CANNOT by itself attribute the paper's +2.58 %p increment, because
that increment is measured on top of an image branch.

------------------------------------------------------------------------------
DEGREES OF FREEDOM
------------------------------------------------------------------------------
  SEEDS ............ MEASURED-over: the seed set 0..6 is the variance sample,
                     the same set as the AI Hub timestamp-baseline comparison.
  task/label ....... CHOSEN (to match the CDD study): the 25-class
                     crop_disease_risk label, i.e. crop + disease + severity
                     jointly, which is what app14104322 predicts.
  metric ........... CHOSEN (to match the CDD study): macro F1. Balanced
                     accuracy is reported alongside; no claim rests on it alone.
  test fraction .... CHOSEN: 1/7 = 14.3 %, not derived from the study, whose
                     stated 5:3:2 ratio gives a 20 % test share. The paper's
                     validation split is merged into train here because nothing
                     is tuned.
  regimes .......... CHOSEN (structural): random (series shared between the
                     sides) vs group-on-val_md5 (the minimal correction). Exactly
                     one factor differs; that is the whole design.
  COLLAPSE ......... CHOSEN: the 0.03 drop of this script's printed P1
                     outcome. The manuscript prints the two scores and their
                     intervals instead (sec:res-timestamp, tab:cdd), so no
                     sentence rests on the threshold.
  TOL .............. CHOSEN: the AI Hub comparable band, for the printed P2
                     outcome only; no CDD claim reads it.
  group key ........ CHOSEN (structural): val_md5, the timestamps-stripped env
                     value hash (780 groups). Chosen over csv_md5 (785 raw-file
                     groups) because two windows with identical values and
                     different timestamps are the same measurement.
  env features ..... CHOSEN (structural): the 5-stat/channel cache that
                     dacon_granularity.py writes (dacon_env_feats_5stat.parquet),
                     median-imputed in CV. Not uniformly weaker than the CDD
                     study's LSTM over the raw series: their LSTM sees 9 env
                     columns (temperature, humidity and dew point, each as mean/
                     max/min) over the raw series; ours sees 5 summary stats over
                     the 25 of the files' 51 value columns that carry a value
                     (dacon_granularity.py), 125 features. Weaker in TEMPORAL
                     modeling, RICHER in channel count. So a leak that this
                     representation exposes is not thereby a lower bound on
                     what the LSTM exposes, and env(random)=0.885 against their
                     env-only 0.8084 is not evidence that ours is weaker.
  timestamp features ... CHOSEN (structural): day_ord + hour, as in the AI Hub timestamp
                     comparison; not retuned here.
  estimator ........ CHOSEN: RandomForestClassifier(60, random_state=seed), as in
                     the AI Hub timestamp-baseline comparison. Continuity, not a new knob.
  nothing FITTED.

  BASELINE: RandomForestClassifier on 5-stat env summaries |
    obvious alternative: "an LSTM over the raw 48 h series, as the CDD study
    used, would extract more from the channel" |
    discharge: sec:res-cdd, by scope — the manuscript states each CDD comparison
    for the random forests that sec:reanalysis names, and no sentence says what an
    LSTM over the raw series would extract from it.

------------------------------------------------------------------------------
OUTCOME MAPPING (printed at the end of the run)
------------------------------------------------------------------------------
Readout: macro F1 over 7 seeds, 95 % CI, per classifier per regime.
The threshold COLLAPSE (0.03) is this script's own; TOL (0.02) is the AI Hub comparable
band. Neither reaches a manuscript sentence.

  P1  LOOKUP: env(random) - env(group) > COLLAPSE.
      => On a split that shares series, the environment classifier's score rests
         partly on recognizing series seen in training.

  P2  ENV TRANSFERS: env(group) >= env(random) - TOL.
      => Forbidding the shared window costs the series nothing, so its signal is
         not group identity, and the leakage argument does not carry at this task.

  P3  NO EFFECT TO EXPLAIN (inconclusive): env(random) - base(random) < COLLAPSE.
      => The series does not predict the 25-class label even under the leaking
         split, so this task is not where the reported increment lives.

  A drop between TOL and COLLAPSE is reported as inconclusive.

  Reported beside the mapping and outside it: the same contrast for a bare timestamp,
  which is not an environmental sensor. Our splits test other images than the study's
  split, so no score here is compared with a figure that the study reports
  (dacon_their_split.py makes that comparison on the study's own split).

Run:  TROPICAL_SCRATCH=/path/to/cdd python dacon_published_split.py
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])
HERE = Path(__file__).resolve().parent   # results land beside the script that
                                        # writes them
META = SCRATCH / "dacon_meta.parquet"      # per-image record, built by dacon_granularity.py
ENV_CACHE = SCRATCH / "dacon_env_feats_5stat.parquet"     # built by dacon_granularity.py
OUT = HERE / "dacon_published_split.csv"

SEEDS = [0, 1, 2, 3, 4, 5, 6]      # the AI Hub timestamp-baseline comparison's seed set
COLLAPSE = 0.03                    # the drop of the printed P1 outcome (header)
TOL = 0.02                         # == the AI Hub comparable band
N_TREES = 60
N_FOLDS = 7                        # test share 1/7 = 14.3%; CHOSEN (header)
GROUP_KEY = "val_md5"              # timestamps-stripped env value hash

# The CDD study's reported figures (10.3390/app14104322), for reference only.
# Not a threshold: our pipeline is not theirs, so only the SPLIT CONTRAST is ours
# to claim. Quoted so the printout sits next to the number it interrogates.
PAPER_IMAGE_ONLY_F1 = 0.8779   # 10.3390/app14104322 Table 2, ResNet-50, image
PAPER_MULTIMODAL_F1 = 0.9037   # ibid., image + environment (+2.58 %p)


def splits(regime, y, groups, seed):
    """Held-out test fold under a random split vs the minimal correction."""
    if regime == "random":
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        return next(iter(cv.split(np.zeros(len(y)), y)))
    cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    return next(iter(cv.split(np.zeros(len(y)), y, groups)))


def score(est, X, y, tr, te):
    est.fit(X[tr], y[tr])
    p = est.predict(X[te])
    return (f1_score(y[te], p, average="macro", zero_division=0),
            balanced_accuracy_score(y[te], p))


def main():
    assert META.exists(), f"missing {META}; run dacon_granularity.py"
    assert ENV_CACHE.exists(), f"missing {ENV_CACHE}; run dacon_granularity.py"
    meta = pd.read_parquet(META)
    meta["image"] = meta["image"].astype(str)
    meta["day_ord"] = pd.to_datetime(meta["date"], format="%Y%m%d").map(pd.Timestamp.toordinal)
    t = meta["time"].astype(str).str.zfill(4)
    meta["hour"] = t.str[:2].astype(int) + t.str[2:].astype(int) / 60.0
    env = pd.read_parquet(ENV_CACHE).reindex(meta["image"].values)

    y = meta["label"].to_numpy()
    groups = pd.factorize(meta[GROUP_KEY])[0]
    dates = meta["date"].to_numpy()
    Xc = meta[["day_ord", "hour"]].to_numpy(dtype=float)
    Xe = env.to_numpy(dtype=float)
    Xce = np.hstack([Xc, Xe])

    print("=" * 88)
    print(f"CDD — random vs grouped split.  n={len(meta)}  classes={len(set(y))}  "
          f"env groups={meta[GROUP_KEY].nunique()}")
    print(f"interrogating 10.3390/app14104322: image-only {PAPER_IMAGE_ONLY_F1} -> "
          f"multimodal {PAPER_MULTIMODAL_F1} (+2.58 %p) on its own split")
    print("=" * 88)

    def rf(seed, impute):
        # One thread per forest, the folds in parallel (below): a forest that predicts
        # on several threads sums its trees' votes in no fixed order, so an exact tie
        # resolves differently from run to run.
        e = RandomForestClassifier(N_TREES, n_jobs=1, random_state=seed)
        return make_pipeline(SimpleImputer(strategy="median"), e) if impute else e

    def fold(regime, seed):
        tr, te = splits(regime, y, groups, seed)
        leak = np.isin(groups[te], groups[tr]).mean()   # sanity check, always logged
        # test images whose photo date also occurs in training (sec:res-cdd: a held-out
        # series leaves its own date in training)
        date_in_train = np.isin(dates[te], dates[tr]).mean()
        r = dict(regime=regime, seed=seed, n_test=len(te), group_leak=leak,
                 date_in_train=date_in_train)
        for arm, X, imp in [("base", Xc, False), ("timestamp", Xc, False),
                            ("env", Xe, True), ("timestamp_env", Xce, True)]:
            est = DummyClassifier(strategy="most_frequent") if arm == "base" else rf(seed, imp)
            r[f"{arm}_f1"], r[f"{arm}_ba"] = score(est, X, y, tr, te)
        return r

    rows = Parallel(n_jobs=-1)(delayed(fold)(regime, seed)
                               for regime in ["random", "group"] for seed in SEEDS)
    for r in rows:
        print(f"  {r['regime']:7s} seed={r['seed']} leak={r['group_leak']:.3f} | macroF1  "
              f"base={r['base_f1']:.3f} timestamp={r['timestamp_f1']:.3f} "
              f"env={r['env_f1']:.3f} timestamp+env={r['timestamp_env_f1']:.3f}")

    long = pd.DataFrame(rows)
    long.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")

    tcrit = stats.t.ppf(0.975, len(SEEDS) - 1)

    def ci(x):
        m, sd = x.mean(), x.std(ddof=1)
        h = tcrit * sd / np.sqrt(len(x))
        return m, m - h, m + h

    print("\n" + "=" * 88)
    print("MACRO F1 by regime (mean [95% CI] over 7 seeds)")
    print("=" * 88)
    print(f"{'classifier':10s} {'random (shared series)':28s} {'group (minimal fix)':28s} {'drop':>8s}")
    summ = {}
    for arm in ["base", "timestamp", "env", "timestamp_env"]:
        a = long[long.regime == "random"][f"{arm}_f1"]
        b = long[long.regime == "group"][f"{arm}_f1"]
        ma, la, ha = ci(a)
        mb, lb, hb = ci(b)
        summ[arm] = (ma, mb)
        print(f"{arm:10s} {ma:.3f} [{la:.3f}, {ha:.3f}]      "
              f"{mb:.3f} [{lb:.3f}, {hb:.3f}]      {ma - mb:+.3f}")

    lk = long.groupby("regime")["group_leak"].mean()
    print(f"\ngroup leak (test rows whose env window is in train): "
          f"random={lk['random']:.3f}  group={lk['group']:.3f}")

    # ---- outcome mapping ----
    env_r, env_g = summ["env"]
    base_r = summ["base"][0]
    print("\n" + "=" * 88)
    print("OUTCOME")
    print("=" * 88)
    print(f"  env(random)={env_r:.3f}  env(group)={env_g:.3f}  base(random)={base_r:.3f}")
    print(f"  drop={env_r - env_g:+.3f} (COLLAPSE={COLLAPSE})   "
          f"advantage over the majority-class predictor={env_r - base_r:+.3f}")
    if env_r - base_r < COLLAPSE:
        v = ("P3 — NO EFFECT TO EXPLAIN. env does not predict the 25-class label even "
             "under the leaking split.")
    elif env_g >= env_r - TOL:
        v = ("P2 — ENV TRANSFERS. Forbidding the shared window costs the series nothing; "
             "the leakage argument does not carry at this task.")
    elif env_r - env_g > COLLAPSE:
        v = ("P1 — LOOKUP. The series' power at this task does not survive the "
             "minimal correction; on a split that shares series, the score rests partly "
             "on recognizing series seen in training.")
    else:
        v = (f"UNMAPPED: drop={env_r - env_g:+.3f} lies between TOL and COLLAPSE. "
             "Inconclusive.")
    print(f"\n  OUTCOME: {v}")
    print(f"\n  outside the mapping: timestamp(random)={summ['timestamp'][0]:.3f} — a bare "
          f"timestamp, no sensors, no images.")


if __name__ == "__main__":
    main()
