"""
CDD — the study's split, with the forest's max_features chosen on the study's validation partition.

dacon_their_split.py scores the timestamp, environment, and timestamp-and-environment forests
on the CDD study's own split (paper_work_seed777) at scikit-learn's default max_features
("sqrt"). The combined forest reads 127 columns with a value (2 timestamp + 125 environment),
so each split considers 11 of them and can use a timestamp column with probability 0.166. On
that split the timestamp alone scores above the timestamp and environment together
(sec:res-cdd). A referee can say the drop is the learner's: a forest that rarely sees the
timestamp at a split cannot use it. This script re-runs the three forests over a grid of
max_features, selects the value per forest and seed by macro-F1 on the study's validation
partition (the partition the study selects its model by), and scores the selected forest,
refitted on training and validation together as in dacon_their_split.py, on the study's test
partition. Every grid value's test score is written too. Writes dacon_their_split_tuned.csv
beside this script, the path the suite reads. Reads no image.

Degrees of freedom:
  split ......... MEASURED: the study's committed paper_work_seed777, as dacon_their_split.py.
  grid .......... CHOSEN: max_features {sqrt, 0.25, 0.5, 1.0}, from the default to every
                  column. Everything else is dacon_their_split.py's (60 trees, median
                  imputation fitted on the training rows, seeds 0-6).
  selection ..... CHOSEN: macro-F1 on the study's validation partition, for a forest trained
                  on the training partition; ties go to the earlier grid value.
  nothing FITTED beyond the selected value, which is recorded per row.

  BASELINE: RandomForestClassifier(60), max_features selected on the study's validation
    partition | obvious alternative: "an LSTM over the raw series, as the CDD study used" |
    tuning: max_features over the four-value grid above, for all three forests alike |
    discharge: sec:res-cdd, by scope (the paper states the scores for the random forest).

COMPARISON AND MEANINGS (committed before the run).
  Statistic: the paired difference timestamp - (timestamp and environment) over the seven
  seeds, 95% t interval, (a) with each forest at its selected max_features and (b) with both at
  max_features = 1.0, every column at every split.
  - Both intervals above zero: the paper keeps that adding the environment features to the
    timestamp lowers the forest's macro-F1 on the study's split, and adds that the drop
    persists when every split can use the timestamp.
  - Either interval containing zero or below it: the drop at the default max_features is a
    property of feature subsampling. The paper cuts the sentence that adding the environment
    lowers macro-F1 and prints the combined forest's score under (a) beside the default's.
  Reported whatever it shows, without a threshold: the selected combined forest's macro-F1 and
  its interval beside the 0.9037 that the study reports for its fusion model, and the selected
  timestamp forest's beside the same value; the range of each forest's test macro-F1 over the
  grid goes to the supplementary material.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])   # hard lookup: no fallback
HERE = Path(__file__).resolve().parent
META = SCRATCH / "dacon_meta.parquet"
ENV_CACHE = SCRATCH / "dacon_env_feats_5stat.parquet"
SPLIT_DIR = SCRATCH / "crop_disease" / "data"
OUT = HERE / "dacon_their_split_tuned.csv"

SEEDS = [0, 1, 2, 3, 4, 5, 6]          # dacon_their_split.py's seed set
N_TREES = 60                           # dacon_their_split.py's forest
GRID = ["sqrt", 0.25, 0.5, 1.0]        # CHOSEN (header)
THEIR_SPLIT = "paper_work_seed777"     # the set the study's training script reads
PAPER_MULTIMODAL_F1 = 0.9037           # 10.3390/app14104322 Table 2, ResNet-50, image + environment


def ids(name, part):
    return pd.read_csv(SPLIT_DIR / name / f"{part}.csv")["image"].astype(str).to_numpy()


def main():
    meta = pd.read_parquet(META)
    meta["image"] = meta["image"].astype(str)
    meta["day_ord"] = pd.to_datetime(meta["date"], format="%Y%m%d").map(pd.Timestamp.toordinal)
    t = meta["time"].astype(str).str.zfill(4)
    meta["hour"] = t.str[:2].astype(int) + t.str[2:].astype(int) / 60.0
    env = pd.read_parquet(ENV_CACHE)
    meta = meta.set_index("image")

    def features(i):
        m = meta.loc[i]
        Xc = m[["day_ord", "hour"]].to_numpy(dtype=float)
        Xe = env.reindex(i).to_numpy(dtype=float)
        return {"timestamp": Xc, "env": Xe, "timestamp_env": np.hstack([Xc, Xe])}, m["label"].to_numpy()

    tr, va, te = ids(THEIR_SPLIT, "train"), ids(THEIR_SPLIT, "val"), ids(THEIR_SPLIT, "test")
    Xtr, ytr = features(tr)
    Xva, yva = features(va)
    Xte, yte = features(te)
    Xtv = {k: np.vstack([Xtr[k], Xva[k]]) for k in Xtr}
    ytv = np.concatenate([ytr, yva])

    def forest(seed, mf, arm):
        e = RandomForestClassifier(N_TREES, max_features=mf, n_jobs=1, random_state=seed)
        return make_pipeline(SimpleImputer(strategy="median"), e) if arm != "timestamp" else e

    def run(seed, arm, mf):
        v = forest(seed, mf, arm).fit(Xtr[arm], ytr).predict(Xva[arm])
        p = forest(seed, mf, arm).fit(Xtv[arm], ytv).predict(Xte[arm])
        return dict(seed=seed, arm=arm, max_features=str(mf),
                    val_f1=f1_score(yva, v, average="macro", zero_division=0),
                    test_f1=f1_score(yte, p, average="macro", zero_division=0),
                    test_ba=balanced_accuracy_score(yte, p))

    long = pd.DataFrame(Parallel(n_jobs=-1)(delayed(run)(s, a, mf) for s in SEEDS
                                            for a in Xtr for mf in GRID))
    # selected value per seed and forest: best validation macro-F1, ties to the earlier value
    long["grid_pos"] = long["max_features"].map({str(m): i for i, m in enumerate(GRID)})
    best = (long.sort_values(["seed", "arm", "val_f1", "grid_pos"], ascending=[True, True, False, True])
            .groupby(["seed", "arm"]).head(1).index)
    long["selected"] = long.index.isin(best)
    long = long.drop(columns="grid_pos")
    long.to_csv(OUT, index=False)

    tcrit = stats.t.ppf(0.975, len(SEEDS) - 1)

    def ci(x):
        x = np.asarray(x)
        h = tcrit * x.std(ddof=1) / np.sqrt(len(x))
        return f"{x.mean():.4f} [{x.mean() - h:.4f}, {x.mean() + h:.4f}]"

    sel = long[long.selected].set_index(["seed", "arm"])
    print("selected max_features:\n", sel["max_features"].unstack("arm"))
    for arm in Xtr:
        print(f"  {arm:14s} selected  test macro-F1 {ci(sel.xs(arm, level='arm')['test_f1'])}")
        for mf in GRID:
            g = long[(long.arm == arm) & (long.max_features == str(mf))]
            print(f"  {arm:14s} mf={str(mf):5s} test macro-F1 {ci(g['test_f1'])}")
    d_sel = (sel.xs("timestamp", level="arm")["test_f1"]
             - sel.xs("timestamp_env", level="arm")["test_f1"])
    a1 = long[long.max_features == "1.0"].set_index(["seed", "arm"])["test_f1"]
    d_all = a1.xs("timestamp", level="arm") - a1.xs("timestamp_env", level="arm")
    print(f"\ntimestamp - timestamp_env, selected : {ci(d_sel)}")
    print(f"timestamp - timestamp_env, all cols : {ci(d_all)}")
    print(f"study's fusion macro-F1: {PAPER_MULTIMODAL_F1}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
