"""
leakage_collapse_gate.py — the candidate splits for the image–sensor fusion experiment.

Reads panel.parquet (metadata + parsed labels), opens no image, trains no model. It runs
before any model is trained, because its outcome decides whether the fusion experiment has
a split to run on.

On CDD, no split by photo date keeps every class on both sides (sec:corpora), and its
region is constant, so it has no farm axis. The question for the AI Hub dataset is whether a
group-disjoint split exists that (a) has a leak rate near zero, where the leak rate is the
share of test images whose panel tuple recurs in training, because a fusion gain could rest
on that recurrence, while (b) keeping coverage, so that a drop in macro-F1 on that split
reflects the lower leak rate rather than classes absent from training.

Over all rows, farm identity explains more than half the variance of most panel variables
(sec:res-diagnostic; 24 farms, 22 growing one crop → panel ≈ farm ≈ crop), yet every class spans ≥13 photo
dates — so a farm-disjoint split loses coverage but a split by photo date might not. This
script measures which.

Degrees of freedom:
  MEASURED — every number below, from panel.parquet.
  CHOSEN (structural) — the candidate group axes {farm, session, date, month}, the panel
    vector = the 9 columns a fusion model would ingest, GroupKFold(5) as the disjoint
    partition (deterministic, no seed to tune), the 4:1 training-to-test ratio that
    5-fold implies.
  FITTED — none. No threshold is tuned to any target.

Thresholds:
  LEAK_FLOOR       = 0.50  # the random split's leak rate must reach this — else a group
                           #   split has no leak rate to lower.
  LEAK_REMOVED     = 0.10  # an axis passes iff its leak rate <= this.
  POSITIVITY_FLOOR = 0.80  # an axis keeps coverage iff >= this share of test images'
                           #   CELL (crop_part_class) is also present in train — below it the
                           #   group-disjoint drop in macro-F1 is dominated by classes absent
                           #   from training, confounding the lower leak rate with a harder task.
  Outcome:
    a split exists iff the random split's leak rate >= LEAK_FLOOR
        AND  some candidate axis has leak rate <= LEAK_REMOVED and cell coverage >=
             POSITIVITY_FLOOR. The printout names the passing axis with the lowest leak
             rate. Otherwise the AI Hub dataset, like CDD, admits no such split.
  LEAK_FLOOR, LEAK_REMOVED and POSITIVITY_FLOOR are this script's; the manuscript prints the
  leak rates and coverages rather than these thresholds (tab:candidates), and holds out photo
  dates because a farm-disjoint split leaves part of the test images with a combination of
  crop, plant part and class that no training image has (sec:fusion-design); the lowest leak
  rate among the passing axes selects the same axis. The month axis has fewer groups than folds and is skipped.
  session_in_train is the share of test images whose measurement session (farm, day on
  which the panel was measured) also occurs in training (sec:intro: a split that holds out
  whole photo dates keeps the session of most test images out of training).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])  # no fallback; see README.md
HERE = Path(__file__).resolve().parent   # results land beside the script that
                                         # writes them: the path the paper reads
PANEL_PARQUET = SCRATCH / "panel.parquet"

# The 9 columns a fusion model ingests as the sensor input — the leak rate is measured on
# THIS vector (the AI Hub analog of CDD's byte-identical environment series).
PANEL = ["temp", "hum", "soil_temp", "soil_condition", "soil_pH",
         "soil_N", "soil_P", "soil_K", "ms_gec"]
BIOCHEM = ["soil_pH", "soil_N", "soil_P", "soil_K", "ms_gec"]  # the soil-chemistry subset

# Thresholds (see module docstring).
LEAK_FLOOR = 0.50
LEAK_REMOVED = 0.10
POSITIVITY_FLOOR = 0.80
N_FOLDS = 5
RANDOM_SEED = 0  # only the random-baseline KFold shuffle; group axes are seed-free (GroupKFold)

OUT = HERE / "leakage_collapse_gate.csv"


def _keyify(frame, cols):
    # NaN-safe row key (pandas 3 astype(str) leaves NaN a float); mirrors
    # identifiability.py so the vector identity is defined the same way.
    arr = frame[cols].round(6).to_numpy(dtype=object)
    return pd.Series(["|".join(str(v) for v in row) for row in arr], index=frame.index)


def _fold_stats(df, key_col, label_col, cell_col, splitter, groups):
    """Average, over the splitter's folds, the leak rate and coverage."""
    leaks, pos_cls, pos_cell, farm_overlap, sess = [], [], [], [], []
    idx = np.arange(len(df))
    for tr, te in splitter.split(idx, groups=groups) if groups is not None \
            else splitter.split(idx):
        train, test = df.iloc[tr], df.iloc[te]
        train_keys = set(train[key_col])
        leaks.append(test[key_col].isin(train_keys).mean())              # byte-identical panel leak
        pos_cls.append(test[label_col].isin(set(train[label_col])).mean())
        pos_cell.append(test[cell_col].isin(set(train[cell_col])).mean())
        farm_overlap.append(test["farm"].isin(set(train["farm"])).mean())  # is the farm shared?
        sess.append(test["msession"].isin(set(train["msession"])).mean())
    return (float(np.mean(leaks)), float(np.mean(pos_cls)),
            float(np.mean(pos_cell)), float(np.mean(farm_overlap)), float(np.mean(sess)))


def main():
    assert PANEL_PARQUET.exists(), (
        f"panel.parquet missing at {PANEL_PARQUET}; regenerate via identifiability.py "
        f"(reads the metadata xlsx from $TROPICAL_SCRATCH/other/)"
    )
    df = pd.read_parquet(PANEL_PARQUET).reset_index(drop=True)
    df["pdate"] = pd.to_datetime(df["photo_dt"], errors="coerce").dt.normalize()
    df["month"] = df["pdate"].dt.to_period("M").astype(str)
    df["session"] = df["farm"].astype(str) + "|" + df["pdate"].astype(str)  # (farm, day)
    # the session that sec:corpora defines: farm and the day on which the panel was measured
    df["msession"] = (df["farm"].astype(str) + "@"
                      + pd.to_datetime(df["measure_dt"]).dt.normalize().astype(str))
    df["panel_key"] = _keyify(df, PANEL)      # the model's sensor input (the tuple whose recurrence the leak rate counts)
    df["bio_key"] = _keyify(df, BIOCHEM)      # the soil-chemistry subset (printed only)

    n = len(df)
    print("=" * 92)
    print("AI Hub dataset — candidate splits for the fusion experiment")
    print("=" * 92)
    print(f"images                 : {n:,}")
    print(f"farms / crops / classes / cells : "
          f"{df['farm'].nunique()} / {df['crop'].nunique()} / {df['cls'].nunique()} / {df['cell'].nunique()}")
    print(f"distinct panel vectors : {df['panel_key'].nunique():,} ({100*df['panel_key'].nunique()/n:.2f}%)   "
          f"biochem vectors: {df['bio_key'].nunique():,} ({100*df['bio_key'].nunique()/n:.2f}%)")
    print(f"photo-dates / sessions / months : "
          f"{df['pdate'].nunique()} / {df['session'].nunique()} / {df['month'].nunique()}")
    print(f"share of images whose panel vector is shared with >=1 other : "
          f"{100*df['panel_key'].duplicated(keep=False).mean():.2f}%")

    # ---- random split (leak rate high, coverage full) ----
    rows = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    leak, pcls, pcell, fo, si = _fold_stats(df, "panel_key", "cls", "cell", kf, groups=None)
    rows.append(dict(axis="random", n_groups=n, panel_leak=leak, cls_pos=pcls,
                     cell_pos=pcell, farm_overlap=fo, session_in_train=si))

    # ---- each candidate group-disjoint axis ----
    for axis in ["month", "date", "session", "farm"]:
        gcol = {"month": "month", "date": "pdate", "session": "session", "farm": "farm"}[axis]
        groups = df[gcol].astype(str).to_numpy()
        n_groups = df[gcol].nunique()
        if n_groups < N_FOLDS:
            print(f"  [skip] axis '{axis}' has {n_groups} groups < {N_FOLDS} folds")
            continue
        gkf = GroupKFold(n_splits=N_FOLDS)
        leak, pcls, pcell, fo, si = _fold_stats(df, "panel_key", "cls", "cell", gkf, groups=groups)
        rows.append(dict(axis=axis, n_groups=n_groups, panel_leak=leak, cls_pos=pcls,
                         cell_pos=pcell, farm_overlap=fo, session_in_train=si))

    res = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print("PER-AXIS (5-fold mean): panel_leak = test images whose 9-dim panel vector is in train;")
    print("  cell_pos = test images whose crop_part_class is in train; farm_overlap = farm shared")
    print("=" * 92)
    print(res.to_string(index=False,
          formatters={"panel_leak": "{:.4f}".format, "cls_pos": "{:.4f}".format,
                      "cell_pos": "{:.4f}".format, "farm_overlap": "{:.4f}".format,
                      "session_in_train": "{:.4f}".format}))

    # ---- outcome ----
    rnd = res.loc[res.axis == "random"].iloc[0]
    cand = res[(res.axis != "random") & (res.panel_leak <= LEAK_REMOVED)
               & (res.cell_pos >= POSITIVITY_FLOOR)]
    leak_ok = rnd.panel_leak >= LEAK_FLOOR
    go = leak_ok and len(cand) > 0
    print("\n" + "=" * 92)
    print("OUTCOME")
    print("=" * 92)
    print(f"  random-split panel leak {rnd.panel_leak:.4f} >= LEAK_FLOOR {LEAK_FLOOR}: "
          f"{'met — the random split has a leak rate to lower' if leak_ok else 'not met — no leak rate to lower'}")
    if len(cand):
        w = cand.sort_values("panel_leak").iloc[0]
        print(f"  axis that lowers the leak rate and keeps coverage: '{w.axis}' "
              f"(leak {w.panel_leak:.4f} <= {LEAK_REMOVED}, cell_pos {w.cell_pos:.4f} >= {POSITIVITY_FLOOR})")
    else:
        print(f"  axis that lowers the leak rate and keeps coverage: NONE "
              f"(no axis clears leak <= {LEAK_REMOVED} AND cell_pos >= {POSITIVITY_FLOOR})")
    print(f"\n  OUTCOME: {'a split exists — the fusion experiment runs on that axis' if go else 'no split lowers the leak rate and keeps coverage'}")

    res["verdict"] = "GO" if go else "NO-GO"
    res.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
