#!/usr/bin/env python3
"""Read survey_records.csv and tally its coded columns.

The records code each fusion study the manuscript cites, from the study's own
text, for the practices that the Introduction, the CDD re-analysis and the
Discussion state about those studies: whether a study compares its panel with a
timestamp, whether its evaluation split holds out a farm or a session, whether it
puts a majority-class accuracy beside a panel-alone one, and whether it releases
both its dataset and its split. The manuscript prints no count from this file.
test_assertions.py re-derives each of those sentences from the cells through
`tally`.

Two kinds of row, never mixed in one count
------------------------------------------
kind=study    a paper reporting a fusion result. Carries the practice columns
              (granularity disclosure, timestamp baseline, sensor-only baseline,
              comparator, split unit, temporal buffer, data availability).
kind=deposit  a dataset release. Carries only a measured granularity and a
              release status. A deposit reports no fusion result of its own, so
              every practice column on a deposit row is `n/a` and the row enters
              no practice count.

`panel_origin` records how a deposit's panel was obtained: `looked-up` where the
deposit's own metadata records the panel as looked up from an external table,
`not-stated` where the deposit ships no such column.

A deposit that also has a study (CDD) appears once as each kind, linked by
`corpus_ref`, and is counted once in each. `check_no_double_count` asserts it.

Per-column denominators are derived, never asserted
---------------------------------------------------
Three sentinels remove a row from a column's denominator, for three different
reasons, and the distinction is the point of the schema:

  inaccessible    we could not read the paper, so we cannot assert anything
                  about it. Removed from every column.
  n/a             the row is not the kind of object that could exhibit this
                  property (a deposit has no sensor-only baseline), or the
                  column is conditional and its precondition failed (no
                  comparator column without a sensor-only baseline to compare).
  not-evaluable   the dataset lacks the input the property needs. A dataset with
                  no time field cannot carry a timestamp baseline or a temporal
                  buffer, by anyone. Distinct from `no`, which says the study
                  declined an available test -- collapsing them would report as
                  researcher choice what is a property of the dataset.

Every other state stays in the denominator, including `not-established`: we
read the paper and could not establish the property, which is a codable
non-positive, not an absence of evidence.

Every cell is filled from the paper's own text, and none by inference from its
reported totals, however sound the arithmetic: the manuscript binds itself to
facts measured against a shipped dataset rather than read from documentation
(sec:corpora), and a coded cell is held to the same line.

Consequence: the columns do not share a denominator, and no denominator is
written down anywhere -- each is counted from the cells of its own column. The
comparator column is the clearest case: it is conditional on the sensor-only
baseline, so its denominator is that column's numerator and not the codable set.

Usage: python survey_records.py   (checks the records and prints each column's tally)
"""

import csv
import inspect
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RECORDS = HERE / "survey_records.csv"

# A row leaves a column's denominator for exactly these three reasons.
NOT_COUNTED = {"inaccessible", "n/a", "not-evaluable"}

# (tally name, csv column, states that count as the numerator, kind)
COLUMNS = [
    ("GranularityReported", "granularity_state", {"reported"}, "study"),
    ("GranularityDisclosed", "granularity_state", {"reported", "by-construction"}, "study"),
    ("TimestampBaseline", "timestamp_baseline", {"yes"}, "study"),
    ("SensorBaseline", "sensor_baseline", {"yes"}, "study"),
    ("Comparator", "comparator", {"yes"}, "study"),
    ("SplitAboveSample", "split_above_sample", {"yes"}, "study"),
    ("SessionDisjoint", "session_disjoint", {"yes"}, "study"),
    ("TemporalBuffer", "temporal_buffer", {"yes"}, "study"),
    ("DataPublic", "data_availability", {"public"}, "study"),
    ("ReproducibleRelease", "reproducible_release", {"yes"}, "study"),
]


def reproducible_release(data_availability, splits_released):
    """Derived rather than stored, and taking its inputs by name.

    A third party can re-run a study's evaluation only when the dataset and the
    partition indexing it are both released. It is the conjunction of two cells
    that are coded independently, so storing it would let the CSV disagree with
    itself. A row we could not read carries its sentinel through, because an
    unread paper is not a paper whose release we checked and found closed.

    The parameters are CSV column names because `source_cells` reads them off
    this signature: a count over this column is backed by the cells named here,
    and adding a third one carries it to that check by being written.
    """
    if data_availability in NOT_COUNTED:
        return data_availability
    return "yes" if data_availability == "public" and splits_released == "yes" else "no"


# Columns computed in `load` rather than read from the CSV.
DERIVED = {"reproducible_release": reproducible_release}


def source_cells(column):
    """The coded cells a column's value rests on: itself, or its derivation's inputs."""
    fn = DERIVED.get(column)
    return tuple(inspect.signature(fn).parameters) if fn else (column,)


def load():
    with RECORDS.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["reproducible_release"] = reproducible_release(
            r["data_availability"], r["splits_released"]
        )
    return rows


def tally(rows, column, positive, kind):
    """Numerator and denominator for one column, both derived from the cells."""
    pool = [r for r in rows if r["kind"] == kind]
    codable = [r for r in pool if r[column] not in NOT_COUNTED]
    return {
        "numerator": sum(1 for r in codable if r[column] in positive),
        "denominator": len(codable),
        "excluded": len(pool) - len(codable),
        "total": len(pool),
        "positive_ids": [r["record_id"] for r in codable if r[column] in positive],
    }


def check_no_double_count(rows, counts):
    ids = [r["record_id"] for r in rows]
    assert len(ids) == len(set(ids)), "record_id is not unique"

    deposits = {r["record_id"] for r in rows if r["kind"] == "deposit"}
    for r in rows:
        if r["corpus_ref"]:
            assert r["kind"] == "study", f"{r['record_id']}: corpus_ref on a deposit row"
            assert r["corpus_ref"] in deposits, f"{r['record_id']}: dangling corpus_ref"

    # Every row of the column's kind is in exactly one of three buckets.
    for stem, (column, positive, kind) in counts["_spec"].items():
        c = counts[stem]
        assert c["numerator"] <= c["denominator"], f"{stem}: numerator exceeds denominator"
        assert c["denominator"] + c["excluded"] == c["total"], f"{stem}: buckets do not sum"
        assert len(c["positive_ids"]) == c["numerator"], f"{stem}: numerator vs id list"

    # The conditional column's denominator is the parent column's numerator.
    assert counts["Comparator"]["denominator"] == counts["SensorBaseline"]["numerator"], (
        "comparator denominator must equal the number of studies that ran a "
        "sensor-only baseline"
    )


PANEL_ORIGIN_STATES = {"deposit": {"not-stated", "looked-up"}, "study": {"n/a"}}


def check_panel_origin_is_coded(rows):
    """Every row carries a panel_origin its kind admits, so a blank or a typo
    in the column fails here rather than passing as a third state."""
    for r in rows:
        assert r["kind"] in PANEL_ORIGIN_STATES, (
            f"{r['record_id']}: kind is {r['kind']!r}, and the coding table's kinds "
            f"are {sorted(PANEL_ORIGIN_STATES)}"
        )
        allowed = PANEL_ORIGIN_STATES[r["kind"]]
        assert r["panel_origin"] in allowed, (
            f"{r['record_id']}: panel_origin is {r['panel_origin']!r}, and a "
            f"{r['kind']} row admits {sorted(allowed)}"
        )


def main():
    rows = load()
    counts = {stem: tally(rows, col, pos, kind) for stem, col, pos, kind in COLUMNS}
    counts["_spec"] = {stem: (col, pos, kind) for stem, col, pos, kind in COLUMNS}
    check_no_double_count(rows, counts)
    check_panel_origin_is_coded(rows)
    del counts["_spec"]

    for stem, c in sorted(counts.items()):
        ids = ", ".join(c["positive_ids"]) or "-"
        print(f"{stem:24s} {c['numerator']:>2d} / {c['denominator']:<2d} "
              f"(excluded {c['excluded']}) {ids}")


if __name__ == "__main__":
    main()
