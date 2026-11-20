"""
Is the soil-chemistry panel a nutrient measurement or a proxy for the timestamp? — over seeds 0–6.

This is the timestamp-baseline comparison that the manuscript reports (sec:res-timestamp),
over a seed set, so that each crop's margin carries a confidence interval over seeds.
timestamp_baseline.py runs the same comparison over the same seeds in accuracy alone,
beside the within-session measurement.

The comparison (sec:diagnostic): a bare timestamp (day-ordinal + hour, no
chemistry) against the five soil-chemistry columns at predicting the disease class,
within crop. A timestamp that is no less accurate than the panel means that the
panel's accuracy gives no evidence that the panel does more than identify the
session.

PAIRED design (isolates the feature set): within each seed, ONE subsample and ONE
StratifiedKFold split feed BOTH models, and the RF seed is shared; only the feature
columns differ.  diff = timestamp - panel is therefore paired per seed, and the only thing
that varies within a (crop, seed) pair is chemistry-features vs timestamp-features.

Degrees of freedom:
  SEEDS ......... MEASURED-over: the seed set is the variance sample.
  timestamp features. CHOSEN (structural): day-ordinal + hour; no free parameter.
  panel features. CHOSEN (structural): 5 soil-chemistry columns of the 9 shipped.
  full-panel features. CHOSEN (structural): all 9 shipped columns, less any with at
                 most one value in the crop; a missing value is left to the forest.
  estimator ..... CHOSEN: RandomForestClassifier(60, random_state=seed), the same
                 estimator as timestamp_baseline.py and as the CDD analyses. Continuity, not a
                 new knob.
  crop floor .... CHOSEN: a crop enters at >= 1,000 complete rows and >= 2 classes;
                 every compared crop exceeds the 20,000-row draw, so any floor up to the
                 draw selects the same crops (sec:diagnostic).
  folds ......... CHOSEN: StratifiedKFold(3, shuffle, seed), shared by every classifier.
  nothing FITTED.

  BASELINE: RandomForestClassifier(60) on the 5 shipped biochem columns |
    obvious alternative: "a stronger learner, or engineered agronomic features,
    would beat a bare timestamp where a 60-tree forest on raw columns does not" |
    discharge: sec:diagnostic and sec:discussion, by scope. The design is PAIRED
    (sec:diagnostic: one estimator, one subsample, one fold split, shared RF seed;
    only the feature columns differ), so the objection is not a relative handicap of
    panel against timestamp. The paired design does not exclude the estimator in
    absolute terms, so the manuscript states the comparison for the random forest it
    names, and its limitations state that no other learner was tried.

Printed verdicts (this script's own thresholds; they reach no manuscript claim):
  Per crop, 95% CI of the paired mean diff (timestamp - panel) over seeds; tol = 0.02
  (the same value timestamp_baseline.py used for its own printed verdicts, timestamp_baseline.py:183, kept
   for continuity; that script's bands reach no manuscript claim, as its header says).
    supported  if in >=5 of the crops timestamp is comparable-or-better
               (the lower limit of the diff's 90% CI >= -tol, the timestamp
                criterion's definition).
    refuted    if in >=3 crops panel decisively beats timestamp (CI_high < -tol).
    otherwise  mixed.
  A comparable timestamp classifier already means that the panel's accuracy gives no
  evidence beyond session identity; the outcome against the claim is the panel
  decisively ahead.

BALANCED-ACCURACY CHECK.
  The timestamp criterion is stated in accuracy: the timestamp classifier is
  comparable to or better than the panel classifier in a majority of the compared
  crops, comparable meaning that the lower limit of the margin's 90% confidence
  interval over seeds is at least -0.02. The criterion was chosen after the data
  were seen, and the manuscript prints the margins to which it was applied rather
  than its verdict (sec:res-timestamp, sec:stats). The manuscript reports balanced
  accuracy as a check that no comparison depends on the sizes of the classes. This script computes both metrics
  from the same fits (one cross_validate pass). Only the metric differs — seeds,
  crops, folds, features, RF, and tol=0.02 are identical — so any divergence
  between the two is the metric. The verdict thresholds that this script prints
  (>= 5 crops in accuracy, >= 4 in balanced accuracy) are its own and reach no
  manuscript claim; the criterion's majority is 4 of the 7 compared crops.

FULL-PANEL CHECK (criterion committed before its run).
  The panel classifier reads the five soil-chemistry variables. A second panel
  classifier, the full-panel classifier, reads every panel variable that a crop's
  compared rows carry (BIOCHEM + AMBIENT: the five, air temperature, air humidity,
  soil temperature and soil moisture), less any that takes at most one value among
  those rows. A missing value is left to the forest, whose trees route it at each
  split (scikit-learn's native handling). Same drawn rows, folds, seed and forest as
  the five-variable classifier, so the pair differs only in the ambient columns.
  Reported whatever it shows, in accuracy and in balanced accuracy: the number of
  compared crops in which the timestamp classifier is comparable to or better than
  the full-panel classifier, by the timestamp criterion's band and 90% rule, and the
  number in which the full-panel classifier is superior. If the first count is a
  majority of the compared crops in accuracy, the timestamp criterion holds for the
  whole panel too. The manuscript prints the margins themselves (sec:res-timestamp).

FARM-IDENTITY CHECK (comparison and meanings committed before its run).
  The full-panel classifier is more accurate than the timestamp classifier in every
  compared crop. The timestamp names the day and time of a panel measurement but not
  its farm, so a fourth forest, the farm-and-timestamp classifier (farm_ts), reads
  the timestamp's two columns plus the farm, coded as an integer within the crop. It
  runs on the same drawn rows, folds, seed and forest, so it differs from the
  timestamp classifier only in the farm column. Farm and measurement time identify
  each panel measurement, so farm_ts holds every identity that the full panel can
  hold. Statistic: per crop, the paired difference panel_full - farm_ts over the seven
  seeds, a paired t-test per crop, Holm's correction across the compared crops; accuracy
  first, balanced accuracy beside it.
    No crop with panel_full > farm_ts after Holm: farm identity added to the
      timestamp reaches the full panel's accuracy, so the full panel's lead over the
      timestamp is no evidence of agronomic information.
    k >= 1 crops with panel_full > farm_ts after Holm: the paper does not credit the
      full panel's lead to farm identity in those crops. Results report that there the
      full panel predicts beyond farm and measurement time, as a finding against the
      session reading; no sentence calls the residual agronomic.
  Fig has a single farm, so there farm_ts carries the timestamp classifier's
  information and the outcome repeats the full-panel comparison; the check is
  informative in the crops with two or more farms.
    farm_ts features. CHOSEN (structural): day-ordinal + hour + farm code; no free parameter.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])  # extraction dir; see README.md
HERE = Path(__file__).resolve().parent   # results land beside the script that
                                         # writes them: the path the paper reads
OUT = HERE / "timestamp_vs_panel_multiseed.csv"
BIOCHEM = ["soil_pH", "soil_N", "soil_P", "soil_K", "ms_gec"]
AMBIENT = ["temp", "hum", "soil_temp", "soil_condition"]  # the panel's other four
TIMESTAMP = ["day_ord", "hour"]
SEEDS = [0, 1, 2, 3, 4, 5, 6]  # timestamp_baseline.py runs the same seeds.
TOL = 0.02                     # comparable band of the timestamp criterion. CHOSEN after
                               # the data were seen; it decides only the verdicts that
                               # this script prints. Same value as timestamp_baseline.py:183,
                               # kept for continuity -- not derived from it.
SAMPLE = 20000                 # rows drawn per crop and seed. CHOSEN: bounds the
                               # forests' computation on the largest crops (sec:diagnostic)

df = pd.read_parquet(SCRATCH / "panel.parquet")
df["measure_dt"] = pd.to_datetime(df["measure_dt"])
df["day_ord"] = df["measure_dt"].dt.normalize().map(pd.Timestamp.toordinal)
df["hour"] = df["measure_dt"].dt.hour + df["measure_dt"].dt.minute / 60.0


def cv_scores(X, y, est, cv):
    # ONE CV pass, both metrics from identical fits: accuracy and balanced accuracy
    # are then paired, not two separate runs.
    # One thread per forest (n_jobs=1 below), the folds in parallel: a forest that
    # predicts on several threads sums its trees' votes in no fixed order, so an
    # exact tie resolves differently from run to run.
    r = cross_validate(est, X, y, cv=cv, n_jobs=-1,
                       scoring=["accuracy", "balanced_accuracy"])
    return r["test_accuracy"].mean(), r["test_balanced_accuracy"].mean()


rows = []
for crop, g in df.groupby("crop_nm"):
    g = g.dropna(subset=BIOCHEM + TIMESTAMP)  # drops 한라봉 entirely (soil_K all-NaN), as timestamp_baseline.py does
    if len(g) < 1000 or g["cls"].nunique() < 2:
        continue
    usable = [c for c in BIOCHEM if g[c].nunique() > 1]
    full = [c for c in BIOCHEM + AMBIENT if g[c].nunique() > 1]
    for seed in SEEDS:
        sub = g.sample(n=min(SAMPLE, len(g)), random_state=seed)
        y = sub["cls"].to_numpy(dtype=object).astype(str)
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)  # same folds for both models
        Xp = sub[usable].to_numpy(dtype=float)
        Xc = sub[TIMESTAMP].to_numpy(dtype=float)
        base, base_b = cv_scores(Xp, y, DummyClassifier(strategy="most_frequent"), cv)
        panel, panel_b = cv_scores(Xp, y, RandomForestClassifier(60, n_jobs=1, random_state=seed), cv)
        timestamp, timestamp_b = cv_scores(Xc, y, RandomForestClassifier(60, n_jobs=1, random_state=seed), cv)
        full_p, full_pb = cv_scores(sub[full].to_numpy(dtype=float), y,
                                    RandomForestClassifier(60, n_jobs=1, random_state=seed), cv)
        Xf = np.column_stack([Xc, pd.factorize(sub["farm"], sort=True)[0]])
        farm_ts, farm_tsb = cv_scores(Xf, y, RandomForestClassifier(60, n_jobs=1, random_state=seed), cv)
        rows.append(dict(crop=crop, seed=seed, n=len(g), base=base, panel=panel, timestamp=timestamp,
                         base_bal=base_b, panel_bal=panel_b, timestamp_bal=timestamp_b,
                         panel_full=full_p, panel_full_bal=full_pb, full_vars="+".join(full),
                         farm_ts=farm_ts, farm_ts_bal=farm_tsb, n_farms=g["farm"].nunique()))
        print(f"  {crop:9s} seed={seed}  base={base:.3f}/{base_b:.3f}  "
              f"panel={panel:.3f}/{panel_b:.3f}  timestamp={timestamp:.3f}/{timestamp_b:.3f}  (acc/bal)",
              flush=True)

long = pd.DataFrame(rows)
long.to_csv(OUT, index=False)

def report(long, pcol, ccol, title, gate_count):
    tcrit = stats.t.ppf(0.975, len(SEEDS) - 1)
    tcrit90 = stats.t.ppf(0.95, len(SEEDS) - 1)   # the 90% CI's lower limit defines comparable
    print("\n" + "=" * 104)
    print(f"{title}, {len(SEEDS)} seeds, paired within seed")
    print("=" * 104)
    print(f"{'crop':9s} {'panel m±sd':>14s} {'timestamp m±sd':>14s} "
          f"{'diff c-p':>9s} {'95% CI':>18s} {'p_paired':>9s}   verdict")
    print("-" * 104)
    summ = []
    for crop, gg in long.groupby("crop"):
        p = gg[pcol].to_numpy()
        c = gg[ccol].to_numpy()
        d = c - p
        md, sd = d.mean(), d.std(ddof=1)
        hw = tcrit * sd / np.sqrt(len(d))
        lo, hi = md - hw, md + hw
        lo90 = md - tcrit90 * sd / np.sqrt(len(d))
        tt = stats.ttest_rel(c, p)
        if lo > TOL:
            v = "timestamp superior"
        elif hi < -TOL:
            v = "panel superior"
        elif lo90 >= -TOL:
            v = "comparable or better"
        else:
            v = "neither"
        summ.append(dict(crop=crop, panel=p.mean(), timestamp=c.mean(), md=md, lo=lo, hi=hi,
                         p=tt.pvalue, comp_or_better=lo90 >= -TOL,
                         panel_wins=hi < -TOL))
        print(f"{crop:9s} {p.mean():>7.3f}±{p.std(ddof=1):.3f} {c.mean():>7.3f}±{c.std(ddof=1):.3f} "
              f"{md:>+9.3f} [{lo:>+7.3f},{hi:>+7.3f}] {tt.pvalue:>9.3f}   {v}")
    s = pd.DataFrame(summ)
    n = len(s)
    nc, npw = int(s["comp_or_better"].sum()), int(s["panel_wins"].sum())
    print(f"crops timestamp comparable-or-better : {nc} / {n}   (threshold: >= {gate_count})")
    print(f"crops panel decisively ahead     : {npw} / {n}")
    print(f"crop mean:  panel={s['panel'].mean():.3f}  timestamp={s['timestamp'].mean():.3f}")
    return s, nc, npw


# Accuracy, the metric of the timestamp criterion; this script's printed threshold is >= 5 of 7.
_, ncp, npwp = report(long, "panel", "timestamp",
                      "ACCURACY — panel vs timestamp disease class", 5)
plain = ("supported — timestamp comparable-or-better than panel across seeds" if ncp >= 5
         else "refuted — panel carries signal beyond the timestamp" if npwp >= 3
         else "mixed")
print(f">>> accuracy: {plain}")

# Balanced accuracy, the manuscript's check; this script's printed threshold is >= 4 of 7.
_, ncb, npwb = report(long, "panel_bal", "timestamp_bal",
                      "BALANCED ACCURACY — panel vs timestamp disease class", 4)
balanced = ("timestamp comparable-or-better in >= 4 of 7 crops under balanced accuracy" if ncb >= 4
      else f"timestamp comparable-or-better in only {ncb}/7 crops under balanced accuracy")
print(f">>> balanced accuracy: {balanced}")

# The full-panel check, reported whatever it shows (header).
for pcol, ccol, name in (("panel_full", "timestamp", "ACCURACY"),
                         ("panel_full_bal", "timestamp_bal", "BALANCED ACCURACY")):
    _, nf, npf = report(long, pcol, ccol, f"{name} — full panel vs timestamp", 4)
    print(f">>> full panel, {name.lower()}: timestamp comparable-or-better in {nf}, "
          f"full panel superior in {npf}")


def holm_counts(long, a, b, alpha=0.05):
    """Per crop, paired t-test of a - b over seeds; Holm across crops.
    Returns {crop: (mean diff, p, rejected)}."""
    res = {c: ((g[a] - g[b]).mean(), stats.ttest_1samp(g[a] - g[b], 0).pvalue)
           for c, g in long.groupby("crop")}
    order = sorted(res, key=lambda c: res[c][1])
    rej, m = {}, len(order)
    for k, c in enumerate(order):
        if res[c][1] <= alpha / (m - k) and all(rej.values()):
            rej[c] = True
        else:
            rej[c] = False
    return {c: (res[c][0], res[c][1], rej[c]) for c in res}


# The farm-identity check, reported whatever it shows (header).
for a, b, name in (("panel_full", "farm_ts", "ACCURACY"),
                   ("panel_full_bal", "farm_ts_bal", "BALANCED ACCURACY")):
    r = holm_counts(long, a, b)
    print(f"\n{name} — full panel minus farm and timestamp, paired over seeds, Holm across crops")
    for c, (md, p, rj) in r.items():
        print(f"  {c:9s} {md:+.4f}  p={p:.2g}  {'Holm-significant' if rj else ''}")
    print(f">>> full panel above farm+timestamp in {sum(rj and md > 0 for md, p, rj in r.values())}, "
          f"below in {sum(rj and md < 0 for md, p, rj in r.values())} of {len(r)} crops")
print(f"\nwrote {OUT}")
