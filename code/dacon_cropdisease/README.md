# CDD dataset — analysis scripts

The diagnostic and the re-analysis of a published fusion study on the **CDD
dataset** (*Crop Disease Diagnosis based on Agricultural Environmental Changes*,
농업 환경 변화에 따른 작물 병해 진단, LG AI Research), the paper's second dataset.
Method and results: [`../../paper/`](../../paper/).

The scripts read no image: they read the label index, the per-sample environment
CSVs, and the annotation JSON only, and no detector is trained. The environment
series here is shared by many images rather than recorded per image.

## Claim checks

The root [`test_assertions.py`](../../test_assertions.py) reproduces every
quantitative claim from the committed, imagery-free tables in this directory
(`dacon_series_ids.parquet`, `dacon_granularity.json` and the per-seed result CSVs). Run `python test_assertions.py` from the repository root.

## Source & provenance

- Served by DACON as the "농업 환경 변화에 따른 작물 병해 진단 AI 경진대회" challenge
  (LG AI Research collected the data; DACON hosted the challenge), retrieved from
  <https://dacon.io/competitions/official/235870/overview/description>.
  **Redistribution prohibited** — the dataset stays in `$TROPICAL_SCRATCH`
  (git-ignored, required — no fallback), and so does `dacon_meta.parquet`, the
  per-image record that `dacon_granularity.py` builds and the other scripts read:
  the identifier, the label, the crop, disease, region, photo date and time from the
  annotation, and two hashes of the environment series. Only these scripts and
  imagery-free derived tables are versioned: `dacon_series_ids.parquet` (each
  image's identifier and the two hashes of its series), `dacon_granularity.json`
  (the counts that the manuscript prints from the labels and dates), and the result
  tables. None holds an image, a sensor value, or a per-image label or date. Those
  tables are written into this directory, the path the suite reads; there is no
  copy step.
- Per-sample layout, verified from the dataset (not the docs):
  `train/<id>/<id>.jpg` + `<id>.csv` (environment series) + `<id>.json` (annotation).
- **The CDD study's code** ([`hyunseoki/crop_disease`](https://github.com/hyunseoki/crop_disease),
  its Data Availability repository) is pinned at commit `c825d9d06f26cd2c6ab4ed619d07415991029a80`
  (2024-05-06) — the exact version every number derived from that repository was read
  from. **No license is declared** in that repository (default all-rights-reserved), so
  it is referenced for scholarly critique and reproduction, **not redistributed**; the
  clone stays in git-ignored scratch. Reproduce with
  `git clone https://github.com/hyunseoki/crop_disease && git -C crop_disease checkout c825d9d`.

## Run order

| # | script | answers |
|---|---|---|
| 1 | `dacon_granularity.py` | how many distinct environment series the images carry, and the per-sample metadata; writes `dacon_series_ids.parquet` and `dacon_granularity.json` here, and the per-image record `dacon_meta.parquet` and the environment feature table (`dacon_env_feats_5stat.parquet`) into `$TROPICAL_SCRATCH`, which steps 2 to 4 read |
| 2 | `dacon_audited_split_leakage.py` | the CDD study's **own** committed splits (`hyunseoki/crop_disease`). On `paper_work_seed777`, the set `src/main.py` reads for training, validation and test: **91.9%** of test images share their environment series with a training image (91.0% with their own label, 96.4% against training and validation together); on the other committed set, `paper_work_seed42`, **97.1%**. **31.5%** of the test identifiers appear verbatim in training, and 50.2% in training and validation together. Needs `--repo <clone>` and `$TROPICAL_SCRATCH`; reads no image |
| 3 | `dacon_published_split.py` | the environment, timestamp, and combined classifiers under a random split and a split grouped by environment series, over 7 split seeds: environment 0.885 → 0.653, timestamp 0.899 → 0.595, leak rate 95.9% → 0.0% |
| 4 | `dacon_their_split.py` | the same classifiers on the study's **own** committed identifiers, so one macro-F1 of ours and one of theirs are measured over the same test images. On `paper_work_seed777`: timestamp **0.910 [0.904, 0.916]**, environment 0.845, timestamp and environment 0.885, majority class 0.014. The timestamp and environment classifier reaches the study's image-only score (0.8779) but not its fusion score (0.9037); the timestamp classifier alone contains the fusion score in its interval. Reads no image; reads the study's clone at `$TROPICAL_SCRATCH/crop_disease` |
| 5 | `dacon_their_split_tuned.py` | the same three forests on the study's split with max_features chosen on the study's validation partition, and at every value of the grid (√p, 0.25p, 0.5p, p): the timestamp alone stays above the timestamp and the environment together, **+0.0226** at the chosen value and **+0.0173** with every variable at every split (paired over 7 forest seeds). Reads no image |

## Findings (N = 5,767 training samples)

1. **One environment series is shared by many images.** Only **780 distinct
   environment series for 5,767 images**; 96.4% of the images share their series
   with at least one other image.
2. **A series is a session reading.** 98.6% of the shared series span a single
   photo date, and 98.4% carry a single disease, so a classifier that predicts the
   disease from the series can do so by recognizing the session.
3. **No region axis; the grouping axis is time.** `region` is constant across all
   5,767 samples, and the 780 series spread over 64 photo dates, about 12 to a
   date, so a held-out series leaves its own date in training.
4. **No split by date keeps every class on both sides.** 4 of the 25 classes
   were photographed on a single date.

The CDD study (Applied Sciences 14(10):4322, 2024) reports a **+2.58 percentage-point**
macro-F1 gain from adding the environment series. Its text states a 5:3:2 ratio of
training, validation and test images but not how it drew its split. Its Data
Availability repository ships the split files: `paper_work_seed42` holds exactly that
ratio, whereas `paper_work_seed777`, the set its training script reads, holds 1,801,
1,082 and 1,154 images. Joining the committed identifiers to the series hashes shows
that 91.9% of the test images of `paper_work_seed777` carry their identical environment
series in training, so the split is not group-disjoint on the environment series. Its
reported ±0.59 intervals vary only the random seed of training (the split files are
fixed), so no split variance enters them.
