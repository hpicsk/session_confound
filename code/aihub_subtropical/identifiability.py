"""
Where the sensor panel ships, and whether it is collinear with the disease class.

The panel ships in the metadata xlsx (NOT in the label JSON `environments`
block, which is an empty skeleton in all 246,600 shipped labels). It joins to
labels by image filename. Writes panel.parquet, which the later scripts read.

A panel that is collinear with a grouping of the images (crop, farm, class)
cannot be separated from that grouping: a slope reported for the panel is then
the grouping's contrast, and no amount of data or reweighting breaks a structural
collinearity. This is knowable from the assignment table alone, before any model
runs. The printout therefore counts the soil-chemistry tuples that occur under more
than one class, and prints each (crop, part, class) cell's soil-chemistry ranges:
cells whose ranges are disjoint do not overlap, so no comparison at equal soil
chemistry can separate them.

Reads metadata + label filenames only. No images, no model.

Degrees of freedom:
  PANEL, BIOCHEM  CHOSEN (structural) -- the nine panel columns that the xlsx ships, and
                  their five soil-chemistry columns.
  CROP_NM         CHOSEN (structural) -- the dataset's crop codes and crop names, from
                  its guide.
  nothing FITTED: panel.parquet holds the xlsx's values as shipped, joined by file name.
"""

import os
from pathlib import Path

import pandas as pd

SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])  # extraction dir; see README.md
_XLSX = sorted((SCRATCH / "other").rglob("*.xlsx"))
assert len(_XLSX) == 1, f"expected one metadata xlsx under {SCRATCH / 'other'}, found {len(_XLSX)}"
XLSX = _XLSX[0]

PANEL = ["temp", "hum", "soil_temp", "soil_condition", "soil_pH",
         "soil_N", "soil_P", "soil_K", "ms_gec"]
BIOCHEM = ["soil_pH", "soil_N", "soil_P", "soil_K", "ms_gec"]

CROP_NM = {"001": "애플망고", "002": "백향과", "003": "올리브", "004": "용과",
           "007": "바나나", "009": "레몬", "010": "무화과", "011": "한라봉"}


def load():
    df = pd.read_excel(XLSX, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"데이터 명": "image", "촬영일자": "photo_dt",
                            "농가 주소": "farm"})
    # HF001_01_003_00001.JPG -> crop=001 part=01 class=003
    parts = df["image"].str.replace(".JPG", "", regex=False).str.split("_")
    df["crop"] = parts.str[0].str.replace("HF", "", regex=False)
    df["part"] = parts.str[1]
    df["cls"] = parts.str[2]
    df["cell"] = df["crop"] + "_" + df["part"] + "_" + df["cls"]
    df["crop_nm"] = df["crop"].map(CROP_NM).fillna(df["crop"])
    return df


def main():
    df = load()
    print("=" * 80)
    print("PANEL METADATA — SHAPE AND JOIN")
    print("=" * 80)
    print(f"rows                     : {len(df):,}")
    print(f"unique images            : {df['image'].nunique():,}")
    print(f"distinct farms           : {df['farm'].nunique()}")
    print(f"distinct crops           : {df['crop'].nunique()}")
    print(f"distinct classes         : {df['cls'].nunique()}")
    print(f"distinct crop_part_class : {df['cell'].nunique()}")
    print(f"data_type values         : {df['data_type'].unique().tolist()}")
    print(f"panel null counts        : {df[PANEL].isna().sum().to_dict()}")

    print("\nfarms:")
    for f, c in df["farm"].value_counts().items():
        crops = sorted(df.loc[df.farm == f, "crop_nm"].unique())
        print(f"  {c:>7,}  {f}   crops={crops}")

    # ---------- VARIATION: does the panel vary image-to-image? ----------
    print("\n" + "=" * 80)
    print("VARIATION — distinct panel vectors vs. images")
    print("=" * 80)
    def keyify(frame, cols):
        # NaN-safe row key; pandas 3 astype(str) leaves NaN as a float, so
        # stringify element-wise rather than column-wise.
        arr = frame[cols].round(6).to_numpy(dtype=object)
        return pd.Series(["|".join(str(v) for v in row) for row in arr],
                         index=frame.index)

    full = keyify(df, PANEL)
    bio = keyify(df, BIOCHEM)
    print(f"distinct FULL panel vectors     : {full.nunique():,} / {len(df):,} images"
          f"  ({100*full.nunique()/len(df):.2f}%)")
    print(f"distinct BIOCHEM vectors (NPK/EC/pH): {bio.nunique():,} / {len(df):,} images"
          f"  ({100*bio.nunique()/len(df):.2f}%)")
    for c in PANEL:
        print(f"  {c:16s} distinct={df[c].nunique():>7,}  "
              f"min={df[c].min():>9.3f}  max={df[c].max():>9.3f}  std={df[c].std():>8.3f}")

    # ---------- IDENTIFIABILITY: is the panel collinear with the label? ----------
    print("\n" + "=" * 80)
    print("COLLINEARITY — does a biochem vector determine the class?")
    print("=" * 80)
    df["bio_key"] = bio
    g = df.groupby("bio_key")["cell"].nunique()
    gc = df.groupby("bio_key")["cls"].nunique()
    print(f"distinct biochem vectors                 : {len(g):,}")
    print(f"  ...spanning >1 (crop,part,class) cell  : {(g > 1).sum():,}  ({100*(g>1).mean():.2f}%)")
    print(f"  ...spanning >1 disease class code      : {(gc > 1).sum():,}  ({100*(gc>1).mean():.2f}%)")
    print(f"images whose biochem vector is unique to ONE cell: "
          f"{df['bio_key'].map(g).eq(1).sum():,} / {len(df):,} "
          f"({100*df['bio_key'].map(g).eq(1).mean():.2f}%)")

    # Per-cell biochem ranges: cells that occupy disjoint boxes do not overlap
    print("\n" + "=" * 80)
    print("PER-CELL BIOCHEM RANGES  (disjoint => the cells do not overlap)")
    print("=" * 80)
    agg = df.groupby(["crop_nm", "part", "cls"]).agg(
        n=("image", "size"),
        **{f"{c}_{s}": (c, s) for c in BIOCHEM for s in ("min", "max")})
    pd.set_option("display.width", 250, "display.max_columns", 50)
    for c in BIOCHEM:
        agg[c] = (agg[f"{c}_min"].round(3).astype(str) + "–" + agg[f"{c}_max"].round(3).astype(str))
    print(agg[["n"] + BIOCHEM].to_string())

    # ---------- Does class vary WITHIN a farm? ----------
    print("\n" + "=" * 80)
    print("FARM x CLASS — is the farm (hence its chemistry) the class?")
    print("=" * 80)
    ct = pd.crosstab(df["farm"], df["cls"])
    print(ct.to_string())

    # ---------- Predictability: can biochem alone recover the class? ----------
    print("\n" + "=" * 80)
    print("LEAKAGE PROBE — can the 5-dim biochem panel alone predict the class?")
    print("      (no pixels; if AUC/acc ~ 1.0 the panel IS the label)")
    print("=" * 80)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.dummy import DummyClassifier
    complete = df.dropna(subset=BIOCHEM)
    print(f"rows with complete biochem panel: {len(complete):,} / {len(df):,}")
    sub = complete.sample(n=min(40000, len(complete)), random_state=0)
    X, y = sub[BIOCHEM].values, sub["cell"].values
    base = cross_val_score(DummyClassifier(strategy="most_frequent"), X, y, cv=3, n_jobs=-1).mean()
    # One thread per forest, the folds in parallel: a forest that predicts on several
    # threads sums its trees' votes in no fixed order, so an exact tie resolves
    # differently from run to run.
    rf = cross_val_score(RandomForestClassifier(n_estimators=60, n_jobs=1, random_state=0),
                         X, y, cv=3, n_jobs=-1).mean()
    print(f"majority-class baseline accuracy      : {base:.4f}")
    print(f"RandomForest on 5 biochem vars only   : {rf:.4f}")
    print(f"  -> panel->cell mutual determination : {'TOTAL (panel IS the label)' if rf > 0.95 else 'partial'}")

    df.to_parquet(SCRATCH / "panel.parquet")
    print(f"\nwrote {SCRATCH/'panel.parquet'}")


if __name__ == "__main__":
    main()
