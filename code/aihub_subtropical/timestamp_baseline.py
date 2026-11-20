"""
Is the soil-chemistry panel a nutrient measurement or a proxy for the timestamp?

Over timestamp_baseline_multiseed.py's seed set. The timestamp-baseline comparison that the
manuscript reports, (a) against (b), is that script's, which also scores balanced accuracy;
(a) and (b) here repeat it in accuracy beside the within-session measurement. The manuscript
reads (c) and (d) from this file's timestamp_vs_panel.csv (sec:res-timestamp), with a
confidence interval over the seeds (sec:stats).

A within-crop panel->class signal is what "nutrient regime correlates with
disease incidence" would look like, and it is the reading the dataset's
sensor panel invites. It has a mundane rival. The panel comes from sensors
installed per farm and matched to the images afterwards (sec:corpora), and nearly
every panel tuple occurs in one farm-day session only (sec:res-diagnostic), so
the readings change from session to session. If disease classes were photographed on different days
or at different hours, "panel predicts class" is "timestamp predicts class", and
clustering the panel into agronomic regimes yields date strata, not nutrient
regimes.

Discriminating tests, per crop:
  (a) panel     -> class    [the reading under test]
  (b) timestamp -> class    [day ordinal + hour, no chemistry]
  (c) panel     -> class, within one (farm, day) session
                            [does the chemistry say anything once the timestamp is
                             held fixed?]
  (d) panel     -> class, over the rows that enter (c), pooled
                            [the same rows with the session not fixed, so the
                             advantage of (d) minus that of (c) is what fixing
                             the session removes]

  If (b) is comparable to or better than (a): the panel's accuracy gives no
  evidence that the panel does more than identify the session (sec:diagnostic).
  If (c) ~ the within-session base rate: the panel has no within-session signal
  at all, and its apparent confounding is between-session contrast wearing the
  panel's name — the within-session test that sec:diagnostic states.

Degrees of freedom:
  SEEDS ......... MEASURED-over: timestamp_baseline_multiseed.py's seed set, for the
                 subsample, the folds and the forest alike, so that the within-session
                 values carry the confidence interval that sec:stats requires before
                 one score is said to exceed another. The set replaced a single run at
                 seed 0.
  panel features. CHOSEN (structural): 5 soil-chemistry columns of the 9 shipped.
  timestamp features. CHOSEN (structural): day-ordinal + hour; no free parameter.
  crop floor, draw, folds, forest. CHOSEN: timestamp_baseline_multiseed.py's (1,000 rows,
                 20,000 rows, stratified 3-fold, 60 trees), so that (a) and (b) here equal
                 that script's seed by seed.
  session floors. CHOSEN (structural): a session enters (c) and (d) at >= 300 rows and
                 >= 2 classes of >= 3 rows. Three rows is the smallest class a
                 3-fold stratified split admits, so every fold holds each class;
                 300 rows is chosen, not derived. Not tuned. The session-floor
                 check below repeats (c) and (d) at other floors.
  verdict bands . CHOSEN: 0.02 and 0.05, for the printed per-crop verdicts only.
                 Neither reaches the manuscript — the timestamp criterion's band is
                 timestamp_baseline_multiseed.py's TOL, and the residual is published as a
                 difference, not as a verdict.
  floor sweep .. CHOSEN: 100 and 1,000 rows, a factor of about three either side of
                 300, with 300 itself rerun as a check on the main table.
  nothing FITTED.

SESSION-FLOOR CHECK (criterion committed before its run).
  (c) and (d) are repeated at session floors of 100, 300 and 1,000 rows (the class
  floor of 3 rows unchanged) on the same crops, rows and seeds, and written to
  within_session_floor_sweep.csv. Reported whatever it shows, at each floor: (i)
  whether the crop mean of the pooled minus the within-session advantage exceeds half
  the pooled advantage, with a paired interval over seeds that excludes zero; (ii) the
  range over crops of the within-session advantage, and whether each crop's interval
  over seeds lies above the comparable band's half-width (0.02); (iii) the compared
  crops that hold no session at that floor. The manuscript prints the values that
  (ii) reads, not its verdict: the range at 300 rows and, at the floor where an
  interval comes nearest zero, the lowest lower limit.

  BASELINE: RandomForestClassifier(60) on the 5 shipped biochem columns |
    obvious alternative: "a stronger learner would find the within-session
    chemistry a 60-tree forest on raw columns misses, so your residual is a
    property of the model, not of the panel" |
    discharge: sec:discussion, by scope: the manuscript names the panel
    classifier a random forest (sec:diagnostic), and its limitations state that
    no other learner was tried. The within-session residual that this file contributes runs
    against the claim rather than for it, and a stronger learner could only
    enlarge it; the manuscript reports it as a difference over seeds.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])  # extraction dir; see README.md
HERE = Path(__file__).resolve().parent   # results land beside the script that
                                         # writes them: the path the paper reads
OUT = HERE / "timestamp_vs_panel.csv"
BIOCHEM = ["soil_pH", "soil_N", "soil_P", "soil_K", "ms_gec"]
SEEDS = [0, 1, 2, 3, 4, 5, 6]  # timestamp_baseline_multiseed.py's seed set (sec:stats)
SWEEP = HERE / "within_session_floor_sweep.csv"
FLOOR_SWEEP = [100, 300, 1000]  # session floors of the session-floor check (header)

df = pd.read_parquet(SCRATCH / "panel.parquet")
df["measure_dt"] = pd.to_datetime(df["measure_dt"])
df["day_ord"] = df["measure_dt"].dt.normalize().map(pd.Timestamp.toordinal)
df["hour"] = df["measure_dt"].dt.hour + df["measure_dt"].dt.minute / 60.0
df["session"] = df["farm"] + "@" + df["measure_dt"].dt.normalize().astype(str)

TIMESTAMP = ["day_ord", "hour"]


def score(X, y, cv, seed):
    # One thread per forest, the folds in parallel: a forest that predicts on
    # several threads sums its trees' votes in no fixed order, so an exact tie
    # resolves differently from run to run.
    return cross_val_score(
        RandomForestClassifier(n_estimators=60, n_jobs=1, random_state=seed),
        X, y, cv=cv, n_jobs=-1).mean()


print("=" * 92)
print("PANEL vs TIMESTAMP — which predicts the disease class?  (within crop)")
print("=" * 92)
print(f"{'crop':9s} {'seed':>4s} {'n':>7s} {'base':>6s} {'(a)panel':>9s} {'(b)timestamp':>9s} "
      f"{'(c)panel|session':>17s}   verdict")
print("-" * 92)

results = []
for crop, g in df.groupby("crop_nm"):
    g = g.dropna(subset=BIOCHEM + TIMESTAMP)
    if len(g) < 1000 or g["cls"].nunique() < 2:
        print(f"{crop:9s} {len(g):>7,}  -- insufficient (panel incomplete: "
              f"{[c for c in BIOCHEM if df[df.crop_nm==crop][c].isna().all()]})")
        continue
    usable = [c for c in BIOCHEM if g[c].nunique() > 1]

    # The sessions that (c) scores, and their rows: which sessions qualify depends on
    # no seed, so every seed scores the same rows and varies only the folds and forests.
    sessions = []
    for sess, gs in g.groupby("session"):
        if len(gs) < 300 or gs["cls"].nunique() < 2:
            continue
        vc = gs["cls"].value_counts()
        keep = gs[gs["cls"].isin(vc[vc >= 3].index)]
        if keep["cls"].nunique() < 2:
            continue
        sessions.append(keep)

    for seed in SEEDS:
        sub = g.sample(n=min(20000, len(g)), random_state=seed)
        y = sub["cls"].to_numpy(dtype=object).astype(str)
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)

        base = cross_val_score(DummyClassifier(strategy="most_frequent"),
                               sub[usable].to_numpy(dtype=float), y, cv=cv).mean()
        a = score(sub[usable].to_numpy(dtype=float), y, cv, seed)
        b = score(sub[TIMESTAMP].to_numpy(dtype=float), y, cv, seed)

        # (c) within-session: pool per-session CV scores, weighted by session size.
        # The panel classifier and the majority-class predictor are scored on the
        # same sessions and the same folds, as (a) and base are over the whole crop.
        accs, accs_b, wts = [], [], []
        for keep in sessions:
            ys = keep["cls"].to_numpy(dtype=object).astype(str)
            Xs = keep[usable].to_numpy(dtype=float)
            cvs = StratifiedKFold(3, shuffle=True, random_state=seed)
            accs.append(score(Xs, ys, cvs, seed))
            accs_b.append(cross_val_score(DummyClassifier(strategy="most_frequent"),
                                          Xs, ys, cv=cvs).mean())
            wts.append(len(keep))
        c_in = np.average(accs, weights=wts) if accs else np.nan
        c_base = np.average(accs_b, weights=wts) if accs_b else np.nan

        # (d) the rows that (c) scores, pooled across sessions and cross-validated as
        # one table, so that (d) and (c) differ only in whether the session is fixed.
        pooled = pd.concat(sessions)
        yd = pooled["cls"].to_numpy(dtype=object).astype(str)
        Xd = pooled[usable].to_numpy(dtype=float)
        cvd = StratifiedKFold(3, shuffle=True, random_state=seed)
        d_in = score(Xd, yd, cvd, seed)
        d_base = cross_val_score(DummyClassifier(strategy="most_frequent"), Xd, yd, cv=cvd).mean()

        verdict = []
        if b >= a - 0.02:
            verdict.append("TIMESTAMP>=PANEL")
        if not np.isnan(c_in) and not np.isnan(c_base) and c_in <= c_base + 0.05:
            verdict.append("no within-session signal")
        v = "; ".join(verdict) if verdict else "panel adds beyond timestamp"
        print(f"{crop:9s} {seed:>4d} {len(g):>7,} {base:>6.3f} {a:>9.3f} {b:>9.3f} "
              f"{c_in:>9.3f}/{c_base:<7.3f}   {v}", flush=True)
        results.append(dict(crop=crop, seed=seed, n=len(g), base=base, panel=a, timestamp=b,
                            panel_in_sess=c_in, base_in_sess=c_base,
                            panel_kept=d_in, base_kept=d_base))

long = pd.DataFrame(results)
r = long.groupby("crop").mean(numeric_only=True)
print("\n" + "=" * 92)
print(f"SUMMARY, means over {len(SEEDS)} seeds")
print("=" * 92)
print(f"crops where TIMESTAMP alone matches/beats PANEL : "
      f"{(r['timestamp'] >= r['panel'] - 0.02).sum()} / {len(r)}")
print(f"crops where panel has NO within-session advantage over the majority-class predictor : "
      f"{(r['panel_in_sess'] <= r['base_in_sess'] + 0.05).sum()} / {len(r)}")
print(f"\nmean over crops:  panel={r['panel'].mean():.3f}  timestamp={r['timestamp'].mean():.3f}"
      f"  panel|session={r['panel_in_sess'].mean():.3f} "
      f"(within-session baseline={r['base_in_sess'].mean():.3f})")
long.to_csv(OUT, index=False)
print(f"\nwrote {OUT}")


def within_session(g, usable, seed, min_rows):
    """(c) and (d) of one crop and seed at a session floor of min_rows rows."""
    sessions = []
    for _, gs in g.groupby("session"):
        if len(gs) < min_rows or gs["cls"].nunique() < 2:
            continue
        vc = gs["cls"].value_counts()
        keep = gs[gs["cls"].isin(vc[vc >= 3].index)]
        if keep["cls"].nunique() >= 2:
            sessions.append(keep)
    if not sessions:
        return dict(n_sessions=0)
    accs, accs_b, wts = [], [], []
    for keep in sessions:
        ys = keep["cls"].to_numpy(dtype=object).astype(str)
        Xs = keep[usable].to_numpy(dtype=float)
        cvs = StratifiedKFold(3, shuffle=True, random_state=seed)
        accs.append(score(Xs, ys, cvs, seed))
        accs_b.append(cross_val_score(DummyClassifier(strategy="most_frequent"),
                                      Xs, ys, cv=cvs).mean())
        wts.append(len(keep))
    pooled = pd.concat(sessions)
    yd = pooled["cls"].to_numpy(dtype=object).astype(str)
    Xd = pooled[usable].to_numpy(dtype=float)
    cvd = StratifiedKFold(3, shuffle=True, random_state=seed)
    return dict(n_sessions=len(sessions),
                panel_in_sess=np.average(accs, weights=wts),
                base_in_sess=np.average(accs_b, weights=wts),
                panel_kept=score(Xd, yd, cvd, seed),
                base_kept=cross_val_score(DummyClassifier(strategy="most_frequent"),
                                          Xd, yd, cv=cvd).mean())


# The session-floor check (header), on the crops, rows and seeds of the main table.
sweep = []
for crop, g in df.groupby("crop_nm"):
    g = g.dropna(subset=BIOCHEM + TIMESTAMP)
    if len(g) < 1000 or g["cls"].nunique() < 2:
        continue
    usable = [c for c in BIOCHEM if g[c].nunique() > 1]
    for floor in FLOOR_SWEEP:
        for seed in SEEDS:
            sweep.append(dict(crop=crop, floor=floor, seed=seed,
                              **within_session(g, usable, seed, floor)))
        print(f"  floor sweep: {crop} at {floor} rows", flush=True)
pd.DataFrame(sweep).to_csv(SWEEP, index=False)
print(f"wrote {SWEEP}")
