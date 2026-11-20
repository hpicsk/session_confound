"""CDD granularity probe: reads no image.

The dataset's documentation cannot settle one property: do same-region/same-window
samples carry informationally DISTINCT per-image environment series, or shared
readings, as on the AI Hub dataset? Answer it directly over all 5,767 train samples:

  A. distinctness  : distinct CSV-byte hashes / 5,767; distinct env-VALUE
                     hashes (timestamps stripped) / 5,767.
  B. grouping/meta : region / crop / disease / risk / date distributions; the window
                     length and span; how many samples share a (region,start,end) window.
Writes three tables. dacon_meta.parquet, the per-image record that the other CDD
scripts read (META_COLUMNS: each image's identifier and label, its crop, disease,
region, photo date and time, and the two hashes of its series), goes to
$TROPICAL_SCRATCH with the per-sample environmental features (ENV_CACHE), because
its labels and dates are the competition's data, which its rules forbid
redistributing. Beside this file go dacon_series_ids.parquet, each image's
identifier and the two hashes of its series (SERIES_ID_COLUMNS), and
dacon_granularity.json, the counts that the manuscript prints from the labels and
dates. Neither holds an image, a label, a date or a sensor value.

Degrees of freedom:
  series identity. CHOSEN (structural), and the choice the counts rest on: two
                 samples share an env series iff their CSVs agree cell-for-cell
                 after the 측정시각 (measurement-time) column is dropped. It is
                 what "informationally distinct" has to mean for a reading that
                 differs only in when it was written down, and it is what
                 sec:res-diagnostic's series count and shared-image share are
                 counts of. A looser rule (rounding, or a tolerance) would merge
                 more samples, lowering the series count and raising the
                 shared-image share; the strict one is the conservative direction
                 for a leakage claim.
  csv hash ..... CHOSEN (structural): the raw bytes, kept beside the value
                 hash so the two can be compared rather than assumed
                 equal. Neither is a threshold.
  env features .. CHOSEN (structural): five summary statistics (mean, min, max,
                 first valid, last valid) of each value column of a sample's
                 series, the timestamp column dropped. Missing values are
                 imputed inside cross-validation by the scripts that read them.
  scope ......... CHOSEN: all 5,767 labelled training samples, the whole
                 released split. No sampling, so no seed.
  nothing MEASURED-over (this is an enumeration of a fixed dataset, not an
  estimate with a spread), nothing FITTED, and no estimator — so no BASELINE
  tag: this file fits no model and publishes no comparison.
"""
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent          # the published tables land here
SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])  # dataset root; no fallback
META = SCRATCH / "dacon_meta.parquet"           # per-image labels and dates: never committed
SERIES_IDS = HERE / "dacon_series_ids.parquet"
GRANULARITY = HERE / "dacon_granularity.json"
ROOT = SCRATCH / "train" / "train"
ENV_CACHE = SCRATCH / "dacon_env_feats_5stat.parquet"  # filename encodes the stat set;
#                        a feature-definition change changes the path -> no stale reuse
TS_COL = "측정시각"                       # env-CSV timestamp col; dropped from features
# the columns of dacon_meta.parquet: what the CDD scripts and granularity() read,
# and nothing else of the dataset's annotations
META_COLUMNS = ["image", "label", "crop", "disease", "region", "date", "time",
                "csv_md5", "val_md5"]
SERIES_ID_COLUMNS = ["image", "csv_md5", "val_md5"]


def granularity(m):
    """The counts that the manuscript prints from the labels and dates."""
    photo_date = m["date"].str[:8]
    sizes = m.groupby("val_md5").size()
    shared = sizes > 1
    one_date = m.assign(photo_date=photo_date).groupby("val_md5")["photo_date"].nunique()
    one_disease = m.groupby("val_md5")["disease"].nunique()
    return dict(
        photo_dates=int(photo_date.nunique()),
        crops=int(m["crop"].nunique()),
        classes=int(m["label"].nunique()),
        single_date_classes=int((m.assign(photo_date=photo_date)
                                 .groupby("label")["photo_date"].nunique() == 1).sum()),
        regions=int(m["region"].nunique()),
        shared_series_single_date_pct=float(100 * (one_date[shared] == 1).mean()),
        shared_series_single_disease_pct=float(100 * (one_disease[shared] == 1).mean()),
    )


def build_env_features(meta):
    """Per-sample env summary: 5 stats (mean, min, max, first-valid, last-valid)
    per channel over the 51 env value-columns (255 features). Every file ships
    the same 51 columns, but 26 of them are empty in every file, so 125 features
    carry a value and the split scripts' median imputer drops the other 130.
    Cached: reading 5,767 CSVs is the slow step. Ragged/absent sensors -> NaN,
    imputed later inside CV. first-/last-valid use b/ffill so an absent endpoint
    doesn't discard a present reading. A series runs newest first, so first-valid
    is the reading nearest the photo time."""
    if ENV_CACHE.exists():
        return pd.read_parquet(ENV_CACHE)
    cols = None
    rows = {}
    for k, sid in enumerate(meta["image"].astype(str)):
        c = pd.read_csv(ROOT / sid / f"{sid}.csv")
        v = c.drop(columns=[TS_COL], errors="ignore").apply(pd.to_numeric, errors="coerce")
        if cols is None:
            cols = list(v.columns)
        v = v.reindex(columns=cols)
        s = pd.concat({"mean": v.mean(), "min": v.min(), "max": v.max(),
                       "first": v.bfill().iloc[0], "last": v.ffill().iloc[-1]})
        s.index = [f"{stat}::{ch}" for stat, ch in s.index]
        rows[sid] = s
        if k % 1000 == 0:
            print(f"  env feats ...{k}/{len(meta)}", flush=True)
    feats = pd.DataFrame(rows).T
    feats.index.name = "image"
    feats.to_parquet(ENV_CACHE)
    return feats


idx = pd.read_csv(SCRATCH / "train.csv")
idx["image"] = idx["image"].astype(str)

rows = []
for k, sid in enumerate(idx["image"]):
    d = ROOT / sid
    j = json.loads((d / f"{sid}.json").read_text())
    desc, ann = j["description"], j["annotations"]
    raw = (d / f"{sid}.csv").read_bytes()
    csv_md5 = hashlib.md5(raw).hexdigest()
    c = pd.read_csv(io.BytesIO(raw))
    tcol = c["측정시각"]
    # value hash: drop the timestamp col, hash the remaining cell content
    val_md5 = hashlib.md5(
        c.drop(columns=["측정시각"]).to_csv(index=False).encode()).hexdigest()
    rows.append(dict(
        image=sid, label=idx["label"].iloc[k],
        crop=ann.get("crop"), disease=ann.get("disease"), risk=ann.get("risk"),
        grow=ann.get("grow"), area=ann.get("area"), region=desc.get("region"),
        date=str(desc.get("date")), time=str(desc.get("time")),
        n_rows=len(c), t_start=str(tcol.iloc[0]), t_end=str(tcol.iloc[-1]),
        csv_md5=csv_md5, val_md5=val_md5))
    if k % 1500 == 0:
        print(f"  ...{k}/{len(idx)}", flush=True)

m = pd.DataFrame(rows)
m[META_COLUMNS].to_parquet(META)
m[SERIES_ID_COLUMNS].to_parquet(SERIES_IDS)
GRANULARITY.write_text(json.dumps(granularity(m[META_COLUMNS]), indent=2) + "\n",
                       encoding="utf-8")
build_env_features(m)
n = len(m)

print("\n" + "=" * 84)
print(f"A. INFORMATIONAL DISTINCTNESS of the per-sample env series  (N={n:,})")
print("=" * 84)
print(f"distinct CSV-byte hashes           : {m.csv_md5.nunique():,} / {n:,}"
      f"  ({100*m.csv_md5.nunique()/n:.2f}%)")
print(f"distinct env-VALUE hashes          : {m.val_md5.nunique():,} / {n:,}"
      f"  ({100*m.val_md5.nunique()/n:.2f}%)  (timestamps stripped)")
print(f"distinct (region,t_start,t_end)     : {m[['region','t_start','t_end']].drop_duplicates().shape[0]:,} / {n:,}")
dup = m.csv_md5.value_counts()
print(f"CSV hashes shared by >1 sample     : {(dup>1).sum():,}  "
      f"(max sharing {dup.max()} samples)")

print("\n" + "=" * 84)
print("B. GROUPING / METADATA")
print("=" * 84)
for col in ["region", "crop", "risk"]:
    print(f"\n{col} value counts:")
    print(m[col].value_counts().to_string())
print(f"\ndisease codes ({m.disease.nunique()}):")
print(m.disease.value_counts().to_string())
d = pd.to_datetime(m.date, format="%Y%m%d")
print(f"\ndate range: {d.min().date()} .. {d.max().date()}   distinct dates: {m.date.nunique()}")
print(f"window length rows: min={m.n_rows.min()} median={int(m.n_rows.median())} max={m.n_rows.max()}")
print(f"\nwrote {META}, {SERIES_IDS.name} and {GRANULARITY.name}")
