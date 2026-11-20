# Sensor readings identify the session

Code and data for the paper *Environmental sensor readings in two crop disease
image datasets identify the session*.

The paper tests whether the environmental sensor readings of two paired
image–sensor crop-disease datasets (the AI Hub dataset and CDD) measure the
plant's environment or identify the session, one farm or greenhouse on one day. In
both datasets nearly every image shares its sensor reading with other images. On
the AI Hub dataset, a timestamp alone predicts the disease class more accurately than
the soil-chemistry readings in five of the seven crops compared, whether or not the
forests are tuned, so a fusion gain measured on a split that shares sessions between
training and test data cannot be separated from session identity. The full panel of
readings predicts beyond the farm and the timestamp together in four of the seven crops. None of the fusion studies
the paper cites compares its sensor readings with a timestamp, and none reports an
evaluation on a farm- or session-disjoint split.

## Layout

```
paper/               LaTeX manuscript (main.tex, params.tex, params_exempt.md, refs.bib)
                     and its Supplementary Material (supplementary.tex); the manuscript
                     builds with Elsevier's elsarticle class from TeX Live
paper/generated/     the manuscript's four figures, written by code/figures.py, and the
                     Supplementary Material's generated tables, written by code/tables.py
code/aihub_subtropical/   AI Hub dataset: analysis scripts + imagery-free tables
code/dacon_cropdisease/  CDD dataset: analysis scripts + imagery-free tables
code/survey/         coding of the cited fusion studies, which the manuscript's
                     sentences about those studies are checked against
code/figures.py      draws the figures from the committed tables (no dataset, no GPU)
code/tables.py       writes the supplement's generated tables from the committed tables
SOURCES.md           retrieval routes for third-party material, which is not redistributed here
LICENSE              MIT for the code, and what it does not cover
test_assertions.py   every manuscript claim, checked against the tables in code/
reproduce.sh         tier-1 verification entry point
requirements.txt     pinned dependencies (two tiers; see below)
```

## Reproduce

**Verify the claims (no dataset, no GPU).** The result tables (CSV/parquet) under
`code/` are committed. `test_assertions.py` reads them and checks every number
the manuscript reports.

```
pip install -r requirements.txt     # the tier-1 rows suffice
./reproduce.sh                       # runs test_assertions.py
```

**Regenerate the tables from raw imagery (dataset + GPU).** The two datasets are
access-restricted and not redistributed here (AI Hub terms; DACON
competition rules for CDD). With a dataset in hand, point `$TROPICAL_SCRATCH` at a working
directory and follow the run-order table in the dataset's own README:
[`code/aihub_subtropical/README.md`](code/aihub_subtropical/README.md) and
[`code/dacon_cropdisease/README.md`](code/dacon_cropdisease/README.md).

## What is and isn't committed

Scripts and small imagery-free derived tables (per-session metadata, split
membership, leakage counts, per-seed metrics) are versioned. Raw imagery, the
multi-gigabyte image caches, the raw environment series, and the CDD per-image
labels and dates are not — they live in `$TROPICAL_SCRATCH` and are reconstructed
from the datasets.

Every generator writes its result table into the directory the suite reads it from,
so re-running a script overwrites the committed table in place and a stale result
shows up as a diff. Every random forest runs on one thread, and the scripts run the
folds or seeds in parallel instead, because a forest that predicts on several threads
resolves an exact vote tie differently from run to run.

## Licensing

`LICENSE` covers the code (the scripts under `code/`, `test_assertions.py`,
`reproduce.sh` and `requirements.txt`) under MIT. It covers nothing else, and the
file says so in its own first section, because parts of this repository are not
ours to license. The result tables, `dacon_series_ids.parquet` and
`dacon_granularity.json` derive from the two datasets and stay under the AI Hub
usage policy and the CDD competition's rules; they are published so the paper's
results can be checked, and reuse of the AI Hub tables carries the attribution
to NIA that the policy requires. The rest of `paper/`, the survey coding and the
Markdown documents are all rights reserved.

## Paper

The figures need the Liberation Sans font, because the journal's guide allows only
Arial, Courier, Times and Symbol in illustrations and Liberation Sans has Arial's
metrics. Install the TrueType files of liberation-fonts 2.1.5
(<https://github.com/liberationfonts/liberation-fonts/releases/tag/2.1.5>) into
`~/.local/share/fonts`, and delete matplotlib's font cache (`fontlist-*.json` under
`python -c "import matplotlib; print(matplotlib.get_cachedir())"`). `code/figures.py`
stops if matplotlib cannot find the font, rather than falling back to another one.

Build the Supplementary Material and then the manuscript with
`latexmk -cd -pdf paper/supplementary.tex paper/main.tex`, after `python code/figures.py`
has drawn the figures from the committed tables; the manuscript reads the
Supplementary Material's table numbers from its build. Every number in either document
is a macro in `paper/params.tex`, and `test_assertions.py` binds each macro to the
committed table that measures it, asserting the macro is that value's correct
rounding. The enumeration is exhaustive and fails on any macro that has neither a
binding nor an entry in [`paper/params_exempt.md`](paper/params_exempt.md), which
records the values that no committed artifact can derive — the CDD study's own
reported figures — and why.
