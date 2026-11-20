"""
Whether the sensor panel ships in the label files.

The paper tests whether the paired soil-chemistry panel (soil_N/P/K/EC/pH)
identifies the farm and session rather than measuring each image. This script measures where
the panel ships and how it varies, before any model is fit. It reads only label
JSON, never an image, and never touches the statistics a model would later be fit
to.

Measures, over every shipped label:
  1. Panel POPULATION  — what fraction of images carry a non-empty panel at all.
  2. Panel VARIATION   — if populated, does it vary image-to-image, or repeat.

On the AI Hub dataset (1) answers 0, which settles (2) with it and leaves the panel
reachable only through the metadata table -- the population sec:corpora describes
when it separates the metadata rows carrying a panel from the labelled images.
An identifiability block below reports whether a panel vector spans more than
one disease class; it is gated on a populated panel and has therefore never run
here. It is kept because it is what this script would measure on a dataset that
ships one, and its output is not committed, claimed, or cited.

Writes panel_scan.json beside this script — the path the suite reads.

Degrees of freedom:
  panel fields .. MEASURED from the label schema: the nine channels the dataset
                 documents as the environment panel, and nothing else. Adding a
                 field could only raise the population count this reports, so
                 the fixed list is the conservative direction for a zero.
  populated ..... CHOSEN (structural): AI Hub encodes absent as an empty or
                 whitespace string rather than null, so a value counts as present
                 iff it strips to something. Reading the empty strings as present
                 is what would turn this zero into 246,600.
  scope ......... CHOSEN: every shipped label under $TROPICAL_SCRATCH/labels, no
                 sampling and so no seed.
  nothing MEASURED-over (this is an enumeration, not an estimate), nothing
  FITTED, and no estimator — so no BASELINE tag: this file fits no model.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])  # extraction dir; see README.md
HERE = Path(__file__).resolve().parent   # results land beside the script that
                                         # writes them: the path the paper reads
LABELS = SCRATCH / "labels"
OUT = HERE / "panel_scan.json"

PANEL_FIELDS = ["temp", "hum", "soil_condition", "soil_temp",
                "soil_pH", "soil_N", "soil_P", "soil_K", "soil_EC"]


def is_populated(v):
    """AI Hub encodes 'absent' as empty string or whitespace, not null."""
    return v is not None and str(v).strip() != ""


def main():
    files = sorted(LABELS.rglob("*.json"))
    print(f"scanning {len(files):,} label files\n", flush=True)

    n = 0
    field_pop = Counter()          # per-field populated count
    any_panel_pop = 0              # images with >=1 populated panel field
    full_panel_pop = 0             # images with ALL panel fields populated
    photo_dt_pop = 0
    measure_dt_pop = 0
    region_vals = Counter()
    # panel value -> set of disease classes, for the identifiability check
    panel_by_class = defaultdict(Counter)
    class_counts = Counter()
    populated_examples = []

    for i, f in enumerate(files):
        if i % 25000 == 0 and i:
            print(f"  ...{i:,}", flush=True)
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"PARSE FAIL {f}: {e}")
            continue
        n += 1

        env = d.get("environments", {}) or {}
        img = d.get("images", {}) or {}
        ann = d.get("annotations", {}) or {}

        cls = ann.get("object_class_sclas_code", "?")
        crop = ann.get("object_class_lclas_code", "?")
        part = ann.get("object_class_mlsfc_code", "?")
        class_counts[(crop, part, cls)] += 1

        pop_here = [fl for fl in PANEL_FIELDS if is_populated(env.get(fl))]
        for fl in pop_here:
            field_pop[fl] += 1
        if pop_here:
            any_panel_pop += 1
            if len(populated_examples) < 5:
                populated_examples.append((str(f.relative_to(LABELS)), env))
        if len(pop_here) == len(PANEL_FIELDS):
            full_panel_pop += 1
            key = tuple(str(env.get(fl)).strip() for fl in PANEL_FIELDS)
            panel_by_class[key][(crop, part, cls)] += 1

        if is_populated(img.get("image_potogrf_dt")):
            photo_dt_pop += 1
        if is_populated(env.get("measure_dt")):
            measure_dt_pop += 1
        region_vals[str(img.get("region_nm", "?")).strip()] += 1

    print("\n" + "=" * 78)
    print("RESULT — PANEL POPULATION")
    print("=" * 78)
    print(f"labels scanned                    : {n:,}")
    print(f"images w/ >=1 panel field         : {any_panel_pop:,}  ({100*any_panel_pop/n:.4f}%)")
    print(f"images w/ FULL panel (all 9)      : {full_panel_pop:,}  ({100*full_panel_pop/n:.4f}%)")
    print(f"images w/ image_potogrf_dt        : {photo_dt_pop:,}  ({100*photo_dt_pop/n:.4f}%)")
    print(f"images w/ measure_dt              : {measure_dt_pop:,}  ({100*measure_dt_pop/n:.4f}%)")
    print("\nper-field population:")
    for fl in PANEL_FIELDS:
        c = field_pop[fl]
        print(f"  {fl:16s} : {c:>8,}  ({100*c/n:7.4f}%)")

    print(f"\nregion_nm values ({len(region_vals)} distinct):")
    for v, c in region_vals.most_common(10):
        print(f"  {c:>8,}  {v!r}")

    print(f"\ndistinct (crop, part, class) cells: {len(class_counts)}")
    print(f"distinct disease class codes      : {len({c for _,_,c in class_counts})}")

    if populated_examples:
        print("\n" + "=" * 78)
        print("EXAMPLES OF POPULATED PANELS")
        print("=" * 78)
        for path, env in populated_examples:
            print(f"\n{path}\n  {json.dumps(env, ensure_ascii=False)}")
    else:
        print("\n" + "!" * 78)
        print("NO IMAGE IN THE ENTIRE SHIPPED DATASET CARRIES ANY PANEL VALUE.")
        print("!" * 78)

    # Identifiability: does any panel value span >1 disease class?
    if full_panel_pop:
        print("\n" + "=" * 78)
        print("IDENTIFIABILITY — distinct full-panel vectors vs. disease classes")
        print("=" * 78)
        print(f"distinct full-panel vectors: {len(panel_by_class):,} "
              f"over {full_panel_pop:,} images")
        spanning = sum(1 for k, v in panel_by_class.items() if len(v) > 1)
        print(f"panel vectors spanning >1 (crop,part,class) cell: {spanning:,} "
              f"/ {len(panel_by_class):,}")

    json.dump(
        {"n": n, "any_panel": any_panel_pop, "full_panel": full_panel_pop,
         "photo_dt": photo_dt_pop, "measure_dt": measure_dt_pop,
         "field_pop": dict(field_pop),
         "cells": {f"{a}_{b}_{c}": v for (a, b, c), v in class_counts.items()},
         "regions": dict(region_vals)},
        open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
