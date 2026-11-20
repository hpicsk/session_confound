# AI Hub dataset — analysis scripts

The diagnostic and the image–sensor fusion experiment for the **AI Hub dataset**
(국내 재배 아열대·열대 병해충 데이터). AI Hub keys it by a platform-side locator
rather than by title, so it is named rather than numbered here and in the paper;
[`SOURCES.md`](../../SOURCES.md) at the repository root carries the retrieval route and the key
mapping. Method and results: [`paper/`](../../paper/).

The Tier-1 scripts read no image: they read label JSON and the metadata xlsx only,
and no detector is trained. Every script resolves its working directory from
`TROPICAL_SCRATCH` and raises `KeyError` if it is unset — no fallback. Raw labels
and the image cache live in `$TROPICAL_SCRATCH` (git-ignored); the imagery-free
result tables are written **into this directory**, which is the path
`test_assertions.py` and the manuscript read. There is no copy step, because a copy
is what lets a re-run and the published paper drift apart.

## Setup

```bash
conda activate tropical
export DATA="/path/to/066.국내_재배_아열대·열대_병해충_데이터/3.개방데이터/1.데이터"
export TROPICAL_SCRATCH="/path/to/extraction"        # ~1.5 GB of JSON + a 20 MB xlsx
mkdir -p "$TROPICAL_SCRATCH"/{labels,sublabel,other}
```

Extract labels into **per-zip subdirectories** — the zip name is the only carrier of
crop/part/class provenance, so a flat extraction loses it:

```bash
# 96 label zips -> 246,600 JSON.  Redirect to /dev/null: piping unzip to `head`
# raises SIGPIPE and silently truncates the extraction after a few files.
for z in $(find "$DATA/Training" "$DATA/Validation" -iname '*.zip' | grep -i '라벨\|TL_\|VL_'); do
  unzip -qo "$z" -d "$TROPICAL_SCRATCH/labels/$(basename "$z" .zip)" > /dev/null
done
unzip -qo "$DATA/Sublabel/SbL.zip" '*.json' -d "$TROPICAL_SCRATCH/sublabel" > /dev/null
unzip -qo "$DATA/Other/Other.zip"           -d "$TROPICAL_SCRATCH/other"    > /dev/null

# verify the extraction is complete before trusting any result
find "$TROPICAL_SCRATCH/labels"   -name '*.json' | wc -l   # expect 246,600
find "$TROPICAL_SCRATCH/sublabel" -name '*.json' | wc -l   # expect  27,400
```

## Tier 1 — the diagnostic (run order)

| # | script | answers | needs |
|---|---|---|---|
| 1 | `scan_panel.py` | Does the sensor panel ship in the label files? (**0 / 246,600**) Writes `panel_scan.json`. | `labels/` |
| 2 | `identifiability.py` | Where does the panel ship, and is it collinear with the label? Writes `panel.parquet`. | `other/` |
| 3 | `timestamp_baseline.py` | The within-session measurement, over seeds 0–6: on the rows of the scored sessions, restricting prediction to single farm-day sessions cuts the panel classifier's advantage over the majority-class predictor **0.251 → 0.045**, averaged across crops and seeds. Writes `timestamp_vs_panel.csv`, and repeats the measurement at session minimums of 100, 300 and 1,000 rows into `within_session_floor_sweep.csv`. | `panel.parquet` |
| 4 | `timestamp_baseline_multiseed.py` | The timestamp-baseline comparison that the manuscript reports: seeds 0–6, accuracy and balanced accuracy (**the timestamp is more accurate than the panel in 5 of 7 crops, the panel in olive**, paired t over seeds with Holm's correction across crops), the full-panel classifier (**more accurate than the timestamp in all 7 crops**) and the farm-and-timestamp classifier (**the full panel stays more accurate in 4 of 7 crops**). Writes `timestamp_vs_panel_multiseed.csv`. | `panel.parquet` |
| 4b | `timestamp_baseline_tuned.py` | The same four classifiers on the same rows, folds and seeds, each forest tuned by one grid search (inner 3-fold cross-validation over max_features and min_samples_leaf, 200 trees). **The same five crops and olive; the full panel ahead of the timestamp in all 7 and of farm and timestamp in the same 4.** Writes `timestamp_vs_panel_tuned.csv`, one row per crop, seed, classifier and outer fold. | `panel.parquet` |
| 5 | `leakage_collapse_gate.py` | The candidate splits for the fusion experiment: the leak rate (the share of test images whose panel tuple recurs in training), and whether every class and cell lands in both training and test data. A **date-disjoint** split cuts the leak rate from **99.3% to 7.2%** while keeping all classes and cells on both sides. Writes `leakage_collapse_gate.csv`. | `panel.parquet` |
| 6 | `corpus_structure.py` | The dataset facts that the Datasets section prints — images, crops, disease classes, farms, distinct tuples of the whole panel, the percentage of rows sharing their panel tuple with another row, the panel tuples that occur in one farm-day session only, the share of rows photographed on another day than their panel was measured, the photo window, the physical-validity rates, and the share of each panel variable's variance that farm explains — as an artifact rather than as stdout, and the count of disease classes photographed on one crop only, which the timestamp baseline's Methods print. Writes `corpus_structure.json`, which `test_assertions.py` binds `params.tex` against. | `panel.parquet` |

```bash
python scan_panel.py && python identifiability.py && python timestamp_baseline.py \
  && python timestamp_baseline_multiseed.py && python timestamp_baseline_tuned.py \
  && python leakage_collapse_gate.py && python corpus_structure.py
```

`scan_panel.py` takes a few minutes (246,600 file reads); the rest are fast except the
random forests in `timestamp_baseline.py` and `timestamp_baseline_multiseed.py`.

## Tier 2 — image–sensor fusion experiment (GPU; subsampled imagery)

Late fusion (a ResNet-50 image branch and an LSTM branch over the nine panel
variables, the CDD study's model) on a 40,000-image subsample stratified by `cell`,
scored by disease-class macro-F1. The gain is
`δ = macroF1(image+sensor) − macroF1(image)`.

| script | role | result |
|---|---|---|
| `h4_materialize_splits.py` | the subsample and its in-distribution and date-disjoint splits; coverage check | prep |
| `h4_build_image_cache.py` | `(40000,256,256,3)` uint8 cache (git-ignored `*.npy`) | prep |
| `h4_fusion_collapse.py` | reduced budget (40 epochs) | the in-distribution gain has an interval that includes zero, `δ = +0.0013 [−0.0058, +0.0084]` (Supplementary Material) |
| `h4_fusion_matched_budget.py` | matched budget (300 epochs, batch 8) | the in-distribution gain is above zero, `δ = +0.0084 [+0.0028, +0.0140]`. Not because the sensor branch trains further: both models score lower than at 40 epochs, in every seed (`image −0.0141 [−0.0219, −0.0063]`, `fusion −0.0071 [−0.0102, −0.0039]`), so δ widens because the image model loses more (paired difference of the two changes `−0.0071 [−0.0140, −0.0001]`) |
| `h4_g4b_collapse.py` | date-disjoint split at the matched budget | at 5 seeds, both the gain `δ(date) = +0.0037 [−0.0041, +0.0115]` and its paired reduction relative to the in-distribution gain `+0.0046 [−0.0010, +0.0103]` have intervals that include zero |

Every committed result table here is imagery-free per-seed output; the corresponding
`.npy` image caches are git-ignored and regenerated from the dataset.

## Caveats

- **`timestamp_baseline.py` repeats the timestamp-baseline comparison in accuracy.** The comparison
  that the manuscript reports is `timestamp_baseline_multiseed.py`'s, which also scores balanced
  accuracy; the manuscript reads only `timestamp_baseline.py`'s within-session measurement. Both
  scripts run over seeds 0–6.
- **`identifiability.py` reads the xlsx by column *name*.** The 설명서's JSON schema
  orders `soil_temp` before `soil_condition`; the xlsx appears to swap them, so any
  positional read silently loads soil moisture as soil temperature.
- **한라봉 has no `soil_K`**, so its 39,000 images (14%) drop out of every
  analysis that requires the five soil-chemistry columns (`timestamp_baseline.py`,
  `timestamp_baseline_multiseed.py`) rather than failing in them.
- **Coverage and the regime check are different checks.** `h4_positivity_recheck.csv`
  and `leakage_collapse_gate.csv` measure coverage: whether every class and cell
  lands in both the training and the test data (`cls_pos`, `cell_pos`), so that the
  split can be scored. `identifiability.py`'s docstring asks a different question:
  whether an agronomic regime admits more than one disease class, so that a
  within-regime contrast exists at all. The first can read 1.0 while the second
  fails. Only the first is committed as an artifact — `identifiability.py` prints
  the second to stdout and writes only `panel.parquet` — so no manuscript sentence
  rests on the second for this dataset.
