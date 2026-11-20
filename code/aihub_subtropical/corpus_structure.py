"""
The AI Hub dataset's structure and physical validity, as a committed
artifact.

The Datasets section and the granularity results print numbers about the
dataset itself — how many images, farms, crops and classes it holds, how many
of those classes occur on a single crop, how many
distinct tuples of the whole panel those images share, what percentage of rows share their panel tuple with
another row, how many panel tuples occur in one farm-day session only, when they were photographed, and how much of the panel is physically
impossible or exactly zero. This script emits them into `corpus_structure.json`,
which `test_assertions.py` reads to pin the macros that `paper/params.tex`
defines, so that a changed number shows as a diff rather than as a hand-typed value.
Beside them it emits the share of each panel variable's variance that farm
explains, for all nine variables, from which sec:res-diagnostic names the
variables whose share is at most half.

The dataset itself is access-gated and 700 GiB, so it cannot live in the
repository; the summary can, and it is imagery-free by construction — counts,
percentages and two dates, no image id, no label, no sensor value.

Degrees of freedom:
  SCRATCH       CHOSEN -- local extraction path, no effect on any measured value.
  PANEL         CHOSEN (structural) -- all nine panel columns, imported from
                leakage_collapse_gate.py, where the fusion experiment reads them too; the
                distinct panel tuples are counted over them.
  HEALTHY_CLS   MEASURED from the dataset's own code list: `999` is the
                pest-and-disease-free code, so the realized *disease* classes are
                the other codes. Excluding it is what makes the count a count of
                disease classes rather than of label values.
  SESSION       CHOSEN (structural) -- (farm, measurement day), the session the
                panel is constant over and the one sec:diagnostic's farm-day
                session means. 9% of rows are photographed on another day than
                the one on which their panel is measured (photo_day_differs_pct).
  nothing FITTED.

Run:
    TROPICAL_SCRATCH=/path/to/extraction python corpus_structure.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so the sibling import resolves
from leakage_collapse_gate import PANEL  # noqa: E402  the nine panel columns, stated once there

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus_structure.json"
SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])   # extraction dir; see README.md

NPK = ["soil_N", "soil_P", "soil_K"]
RSQ_CHANNELS = ["soil_pH", "soil_N", "soil_P", "soil_K"]   # the pH/N/P/K channels sec:res-diagnostic names
HEALTHY_CLS = "999"                                        # the dataset's pest-and-disease-free code


def r2_farm(df, column):
    """Fraction of `column`'s variance that farm identity alone explains.

    1 - Var(residual after removing each farm's mean) / Var(total), the
    between/within-farm decomposition; ddof=0 on both sides so the ratio is the
    population R^2 rather than a sample estimate.
    """
    s = df[[column, "farm"]].dropna()
    within = s.groupby("farm")[column].transform(lambda x: x - x.mean())
    return 1 - within.var(ddof=0) / s[column].var(ddof=0)


def main():
    panel = SCRATCH / "panel.parquet"
    if not panel.exists():
        raise SystemExit(
            f"missing {panel}; build it with identifiability.py against the "
            f"dataset's metadata spreadsheet (see README.md)"
        )
    df = pd.read_parquet(panel)

    measure_day = pd.to_datetime(df["measure_dt"]).dt.floor("D")
    photo_day = pd.to_datetime(df["photo_dt"]).dt.floor("D")
    # One key per row rather than a set, so the rows that share a tuple can be counted too.
    # The parquet round-trip yields arrow-backed dtypes, so the join goes through object
    # rather than a vectorized astype(str).
    tuple_key = pd.Series(["|".join(str(v) for v in row)
                           for row in df[PANEL].round(6).to_numpy(dtype=object)], index=df.index)
    rows_per_tuple = tuple_key.value_counts()
    # SESSION: (farm, measurement day), one key per row
    session = df["farm"].astype(str) + "@" + measure_day.astype(str)
    sessions_per_tuple = session.groupby(tuple_key).nunique()
    ph = df["soil_pH"]

    summary = {
        "images": len(df),
        "crops": int(df["crop_nm"].nunique()),
        "disease_classes": int(df.loc[df["cls"] != HEALTHY_CLS, "cls"].nunique()),
        # disease classes photographed on one crop only, the reason sec:diagnostic gives
        # for training the timestamp baseline's classifiers within each crop
        "single_crop_disease_classes": int(
            (df.loc[df["cls"] != HEALTHY_CLS].groupby("cls")["crop_nm"].nunique() == 1).sum()),
        "farms": int(df["farm"].nunique()),
        "panel_tuples": len(rows_per_tuple),
        # rows whose panel tuple recurs in at least one other row
        "panel_tuple_shared_pct": round(
            100 * int(rows_per_tuple[rows_per_tuple > 1].sum()) / len(df), 4),
        # panel tuples that occur in one farm-day session only, the measurement behind
        # the paper's word that a reading identifies its session
        "panel_tuples_one_session": int((sessions_per_tuple == 1).sum()),
        "photo_first": str(photo_day.min().date()),
        "photo_last": str(photo_day.max().date()),
        # rows photographed on another day than the one on which their panel was measured
        "photo_day_differs_pct": round(100 * float((photo_day != measure_day).mean()), 4),
        # pH is -log10[H+] and so bounded [0,14] by definition; anything outside is not pH.
        "ph_impossible_pct": round(100 * float(((ph < 0) | (ph > 14)).mean()), 4),
        # N, P and K written as exactly zero, over the three channels pooled and over
        # the values present: K is missing for whole crops, and a missing value is no zero.
        "npk_zero_pct": round(100 * float((df[NPK] == 0).to_numpy().sum()
                                          / df[NPK].notna().to_numpy().sum()), 4),
        "r2_farm_min": round(min(r2_farm(df, c) for c in RSQ_CHANNELS), 4),
        "r2_farm_max": round(max(r2_farm(df, c) for c in RSQ_CHANNELS), 4),
        "r2_farm_panel": {c: round(r2_farm(df, c), 4) for c in PANEL},
    }

    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for k, v in summary.items():
        print(f"{k:20s} {v}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
