"""
The timestamp-baseline comparison with the random forest tuned, identically for every input.

timestamp_baseline_multiseed.py compares the timestamp classifier with the panel, full-panel
and farm-and-timestamp classifiers using one untuned 60-tree forest (sec:diagnostic). A
referee can say that an untuned forest under-uses five soil-chemistry columns more than it
under-uses two timestamp columns. This script gives every classifier the same search: the
same grid, the same inner cross-validation and the same selection metric, on the same drawn
rows, outer folds and seeds as timestamp_baseline_multiseed.py. Writes
timestamp_vs_panel_tuned.csv beside this script, the path the suite reads: one row per crop,
seed, classifier and outer fold, with the setting that the inner search selected.

Design (matched tuning):
  outer ......... timestamp_baseline_multiseed.py's: per crop and seed, 20,000 rows drawn at
                  random_state=seed from the rows carrying all five soil-chemistry variables,
                  StratifiedKFold(3, shuffle, seed).
  inner ......... GridSearchCV on each outer training fold, StratifiedKFold(3, shuffle, seed),
                  scoring accuracy, refit on the whole outer training fold.
  grid .......... RandomForestClassifier(200 trees) x max_features {sqrt, 0.5, 1.0} x
                  min_samples_leaf {1, 5, 20}: nine settings, the same for every input.
  classifiers ... panel (five soil-chemistry columns), timestamp (day + hour), full panel
                  (every panel column with more than one value in the crop), farm and
                  timestamp (day + hour + farm code).

Degrees of freedom:
  SEEDS ......... MEASURED-over: seeds 0-6, timestamp_baseline_multiseed.py's set.
  grid .......... CHOSEN: the two settings that control a forest's fit (columns per split and
                  leaf size) at three values each, from the scikit-learn default to all
                  columns and to a leaf of 20 rows; 200 trees, more than the untuned 60, so
                  that tree count does not limit the tuned forest. Not derived; the same for
                  every input, which is the property the comparison needs.
  selection ..... CHOSEN: inner accuracy, the comparison's first metric. Balanced accuracy is
                  read from the same refitted forests.
  nothing FITTED beyond the selected setting per outer fold, which is recorded per row.

  BASELINE: RandomForestClassifier, tuned over the grid above for every input |
    obvious alternative: "gradient boosting or a neural network would extract more from
    the soil chemistry" | tuning: the nine-setting grid above, inner 3-fold CV, the same for
    all four classifiers | discharge: sec:res-timestamp (this run) and sec:discussion,
    by scope: the paper states the comparison for the random forest, untuned and tuned.

CRITERIA AND MEANINGS (committed before the run).
  Statistic: per crop, the paired difference of two classifiers over the seven seeds, a
  paired t-test per crop, Holm's correction across the compared crops (claims-discipline,
  a suite is one comparison); accuracy first, balanced accuracy beside it.
  (1) Timestamp versus soil chemistry: count the crops with timestamp > panel after Holm, and
      the crops with panel > timestamp after Holm, under the tuned forest.
      - Where the tuned count of crops with timestamp > panel equals the untuned count, the
        paper states the count for both forests.
      - Where the two counts differ, every statement of the claim (abstract, Results heading,
        Conclusion) prints the count that holds under both forests, the crops where the two
        forests agree, and Results report each crop that changes direction.
      - If the tuned forest makes the panel more accurate than the timestamp in four or more
        of the seven crops, the claim that a timestamp predicts the class at least as well as
        the soil chemistry is withdrawn, and Results report that a tuned forest extracts more
        from the soil chemistry than from the timestamp, as a finding against the session
        reading.
  (2) Full panel versus timestamp under the tuned forest, reported beside the untuned count
      (untuned: the full panel ahead in all seven crops).
  (3) Full panel versus farm and timestamp under the tuned forest: read as the farm-identity
      check in timestamp_baseline_multiseed.py's header; the paper states what holds under
      both forests.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])  # extraction dir; see README.md
HERE = Path(__file__).resolve().parent
OUT = HERE / "timestamp_vs_panel_tuned.csv"
BIOCHEM = ["soil_pH", "soil_N", "soil_P", "soil_K", "ms_gec"]
AMBIENT = ["temp", "hum", "soil_temp", "soil_condition"]
TIMESTAMP = ["day_ord", "hour"]
SEEDS = [0, 1, 2, 3, 4, 5, 6]   # timestamp_baseline_multiseed.py's seed set
SAMPLE = 20000                  # timestamp_baseline_multiseed.py's draw per crop and seed
N_TREES = 200                   # CHOSEN (header)
GRID = {"max_features": ["sqrt", 0.5, 1.0], "min_samples_leaf": [1, 5, 20]}  # CHOSEN (header)

df = pd.read_parquet(SCRATCH / "panel.parquet")
df["measure_dt"] = pd.to_datetime(df["measure_dt"])
df["day_ord"] = df["measure_dt"].dt.normalize().map(pd.Timestamp.toordinal)
df["hour"] = df["measure_dt"].dt.hour + df["measure_dt"].dt.minute / 60.0


def tasks():
    for crop, g in df.groupby("crop_nm"):
        g = g.dropna(subset=BIOCHEM + TIMESTAMP)
        if len(g) < 1000 or g["cls"].nunique() < 2:
            continue
        usable = [c for c in BIOCHEM if g[c].nunique() > 1]
        full = [c for c in BIOCHEM + AMBIENT if g[c].nunique() > 1]
        for seed in SEEDS:
            sub = g.sample(n=min(SAMPLE, len(g)), random_state=seed)
            y = sub["cls"].to_numpy(dtype=object).astype(str)
            X = {"panel": sub[usable].to_numpy(dtype=float),
                 "timestamp": sub[TIMESTAMP].to_numpy(dtype=float),
                 "panel_full": sub[full].to_numpy(dtype=float),
                 "farm_ts": np.column_stack([sub[TIMESTAMP].to_numpy(dtype=float),
                                             pd.factorize(sub["farm"], sort=True)[0]])}
            folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X["panel"], y))
            for arm, Xa in X.items():
                for k, (tr, te) in enumerate(folds):
                    yield crop, seed, arm, k, Xa, y, tr, te


def fit(crop, seed, arm, k, X, y, tr, te):
    # One thread per forest and per search, the tasks in parallel: a forest that predicts on
    # several threads resolves an exact vote tie differently from run to run.
    gs = GridSearchCV(RandomForestClassifier(N_TREES, n_jobs=1, random_state=seed), GRID,
                      scoring="accuracy", n_jobs=1, refit=True,
                      cv=StratifiedKFold(3, shuffle=True, random_state=seed))
    gs.fit(X[tr], y[tr])
    p = gs.predict(X[te])
    bp = gs.best_params_
    return dict(crop=crop, seed=seed, arm=arm, fold=k, acc=accuracy_score(y[te], p),
                bal=balanced_accuracy_score(y[te], p),
                max_features=str(bp["max_features"]), min_samples_leaf=bp["min_samples_leaf"])


def holm(res, alpha=0.05):
    order = sorted(res, key=lambda c: res[c][1])
    rej, m = {}, len(order)
    for k, c in enumerate(order):
        rej[c] = bool(res[c][1] <= alpha / (m - k) and all(rej.values()))
    return rej


def main():
    # One row per crop, seed, classifier and outer fold, with the setting that the inner
    # search selected; the suite averages the folds within seed.
    folds = pd.DataFrame(Parallel(n_jobs=-1, verbose=5)(delayed(fit)(*t) for t in tasks()))
    folds.to_csv(OUT, index=False)
    long = (folds.groupby(["crop", "seed", "arm"])[["acc", "bal"]].mean()
            .unstack("arm"))
    long.columns = [f"{arm}" if m == "acc" else f"{arm}_bal" for m, arm in long.columns]
    long = long.reset_index()
    for a, b in (("timestamp", "panel"), ("panel_full", "timestamp"), ("panel_full", "farm_ts")):
        for sfx, name in (("", "accuracy"), ("_bal", "balanced accuracy")):
            res = {c: ((g[a + sfx] - g[b + sfx]).mean(),
                       stats.ttest_1samp(g[a + sfx] - g[b + sfx], 0).pvalue)
                   for c, g in long.groupby("crop")}
            rej = holm(res)
            print(f"\n{a} - {b}, {name}, tuned forest, paired over seeds, Holm across crops")
            for c, (md, p) in res.items():
                print(f"  {c:9s} {md:+.4f}  p={p:.2g}  {'Holm' if rej[c] else ''}")
            print(f">>> {a} above {b} in {sum(rej[c] and res[c][0] > 0 for c in res)}, "
                  f"below in {sum(rej[c] and res[c][0] < 0 for c in res)} of {len(res)} crops")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
