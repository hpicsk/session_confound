"""
CDD — leakage in the CDD study's own split. Reads no image.
Part of the analyses behind paper/main.tex.

------------------------------------------------------------------------------
WHAT THE STUDY STATES, AND WHAT ITS REPOSITORY SHIPS
------------------------------------------------------------------------------
The CDD study (10.3390/app14104322) contains exactly ONE sentence about its
split, in Section 4:

    "We divided the dataset into a ratio of 5:3:2 for training, validation, and
     testing sets, respectively."

and never states the mechanism. The paper's Data Availability Statement points
to https://github.com/hyunseoki/crop_disease (pinned at commit c825d9d,
2024-05-06 — the version these numbers were read from), and that repository
ships the split itself:

    data/paper_work_seed42/{train,val,test}.csv     2883 / 1730 / 1154  (= 5767, exactly 5:3:2)
    data/paper_work_seed777/{train,val,test}.csv    1801 / 1082 / 1154
    src/main.py:48-49,153  <- reads paper_work_seed777/{train,val,test}.csv

Each row is `image,label` — the CDD sample id and its 25-class code. Our
dacon_meta.parquet (in $TROPICAL_SCRATCH) carries the same `image` id plus `val_md5`, the
timestamps-stripped hash of that sample's environment window. Joining the two
answers the question as a FACT rather than an inference.

------------------------------------------------------------------------------
WHAT IS MEASURED
------------------------------------------------------------------------------
The decisive quantity is not a rate but a yes/no:

    Does any val_md5 appear on BOTH sides of their train/test boundary?

If yes, the split is NOT group-disjoint on the environment series — no
statistical argument required, and the possibility that the study grouped its
split is excluded. The rates below then say how much of the test set is affected.

    breach      : test images whose env window is also in train
    breach_lab  : ...and that window carries the test image's own label in train
    breach_trval: same, against train+val (their model selects on val, so val is
                  part of what the reported test number is conditioned on)

A cruder, premise-free companion sits INSIDE breach:

    direct      : test images whose OWN sample id (not merely its env window) is
                  also in train — a strict SUBSET of breach. No session=label
                  argument is needed: the model literally trains on the test row.
                  Unlike breach it is key-INDEPENDENT (an id-level property), so it
                  is identical across the two env keys and reported in every row for
                  self-containedness. It contaminates BOTH modalities equally, so it
                  inflates the ABSOLUTE benchmark numbers (image-only AND multimodal)
                  but largely CANCELS in the env INCREMENT — both branches memorize
                  the shared row. Complementary to breach, which is what makes the
                  env INCREMENT (not just the absolute numbers) a leakage artifact.
    direct_trval: same, against train+val.

Reported against the two reference points that dacon_published_split.py measures on
this dataset and the manuscript prints in sec:res-cdd, read from dacon_published_split.csv: a random
image-level split, and a StratifiedGroupKFold on val_md5 that shares nothing by
construction.

------------------------------------------------------------------------------
DEGREES OF FREEDOM
------------------------------------------------------------------------------
  splits ....... MEASURED: read verbatim from the CDD study's own repository.
                 Nothing is re-drawn, re-seeded, or re-stratified here.
  group key .... CHOSEN (structural): val_md5, as in every other CDD script
                 here. csv_md5 (785 raw-file groups vs val_md5's 780) is
                 reported alongside as the conservative variant.
  nothing FITTED, nothing trained, no image opened.

  BASELINE: none — this script fits no model and therefore fixes no
    architecture. The one judgement call is the group key, and both candidates
    are reported.

------------------------------------------------------------------------------
NO PASS/FAIL THRESHOLD
------------------------------------------------------------------------------
This is a property of files that already exist, not an experiment: the answer
is determined before the script runs and no researcher degree of freedom can
move it, so no threshold is set over it.

Run:  TROPICAL_SCRATCH=/path/to/cdd python dacon_audited_split_leakage.py --repo <path to cloned crop_disease>
Writes an imagery-free rate summary (per tag x key: split sizes, shared-group count,
breach / breach_lab / breach_trval) to dacon_audited_split_leakage.csv beside this
script, the path test_assertions.py reads.
"""
import argparse
import os
import pathlib
import sys

import pandas as pd

SCRATCH = pathlib.Path(os.environ["TROPICAL_SCRATCH"])   # hard lookup: no fallback
META = SCRATCH / "dacon_meta.parquet"   # per-image record, built by dacon_granularity.py

# The imagery-free rate summary is written beside this script, where test_assertions.py
# reads it. The summary holds only aggregate breach RATES and group counts — no image id,
# no label.
HERE = pathlib.Path(__file__).resolve().parent   # results land beside the script
                                                # that writes them
OUT = HERE / "dacon_audited_split_leakage.csv"

# Reference points for the printout, read from dacon_published_split.py's committed
# per-seed table rather than restated here. A bracket copied into a second file is what
# lets the paper and the repo end up quoting two different random-split leak rates.
DRAWN_SPLITS_CSV = HERE / "dacon_published_split.csv"


def reference_leaks():
    """{regime: mean group_leak} over dacon_published_split.py's seeds, on the same val_md5 key as below."""
    if not DRAWN_SPLITS_CSV.exists():
        sys.exit(f"missing {DRAWN_SPLITS_CSV} — the random and group-disjoint brackets are read from "
                 f"dacon_published_split.py's committed per-seed table; run it first")
    return pd.read_csv(DRAWN_SPLITS_CSV).groupby("regime")["group_leak"].mean()


def audit(meta, repo, tag, key):
    """Leakage across the CDD study's own train/val/test boundary."""
    d = repo / "data" / tag
    parts = {}
    for name in ("train", "val", "test"):
        f = d / f"{name}.csv"
        if not f.exists():
            sys.exit(f"missing {f} — is --repo pointing at a clone of hyunseoki/crop_disease?")
        # their CSV stores `image` as int, dacon_meta.parquet as str; join on str.
        ids = pd.read_csv(f)[["image"]].astype({"image": str})
        parts[name] = meta.merge(ids, on="image", how="inner")

    n_ids = {k: len(pd.read_csv(d / f"{k}.csv")) for k in parts}
    matched = {k: len(v) for k, v in parts.items()}
    if matched != n_ids:
        sys.exit(f"{tag}: id join is incomplete {matched} vs {n_ids} — refusing to report a rate "
                 f"over a partial join")

    tr, va, te = parts["train"], parts["val"], parts["test"]
    tr_keys, trval_keys = set(tr[key]), set(tr[key]) | set(va[key])
    tr_lab = tr.groupby(key)["label"].agg(set)

    breach = te[key].isin(tr_keys)
    breach_trval = te[key].isin(trval_keys)
    breach_lab = [row.label in tr_lab.get(getattr(row, key), set()) for row in te.itertuples()]

    # Direct same-sample overlap: test rows whose OWN id is in train(+val). Key-independent
    # (an id property, identical across env keys); a strict subset of breach. Premise-free —
    # the benchmark trains on the test row itself. See docstring for why it cancels in the delta.
    tr_ids, trval_ids = set(tr["image"]), set(tr["image"]) | set(va["image"])
    direct = te["image"].isin(tr_ids)
    direct_trval = te["image"].isin(trval_ids)

    shared = len(set(te[key]) & tr_keys)
    print(f"\n{tag}  [{key}]  train {len(tr)} / val {len(va)} / test {len(te)}")
    print(f"  env groups shared across the train|test boundary : {shared} of {te[key].nunique()} test groups")
    print(f"  test images whose env window is in TRAIN         : {breach.mean():.4f}")
    print(f"  ...and mapped there to their own label           : {pd.Series(breach_lab).mean():.4f}")
    print(f"  test images whose env window is in TRAIN or VAL   : {breach_trval.mean():.4f}")
    print(f"  test images whose OWN id is in TRAIN (direct)     : {direct.mean():.4f}")
    print(f"  ...direct against TRAIN or VAL                    : {direct_trval.mean():.4f}")
    return dict(
        tag=tag, key=key,
        n_train=len(tr), n_val=len(va), n_test=len(te),
        n_test_groups=int(te[key].nunique()), shared=shared,
        breach=breach.mean(), breach_lab=float(pd.Series(breach_lab).mean()),
        breach_trval=breach_trval.mean(),
        direct=float(direct.mean()), direct_trval=float(direct_trval.mean()),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=pathlib.Path,
                    help="clone of https://github.com/hyunseoki/crop_disease (checkout commit c825d9d)")
    args = ap.parse_args()

    meta = pd.read_parquet(META)
    print(f"dacon_meta.parquet: {len(meta)} images, "
          f"{meta.val_md5.nunique()} val_md5 groups, {meta.csv_md5.nunique()} csv_md5 groups")
    ref = reference_leaks()
    print(f"reference: random image-level split leaks {ref['random']:.4f}, "
          f"group-disjoint leaks {ref['group']:.4f} (dacon_published_split.csv)")

    rows = []
    for tag in ("paper_work_seed42", "paper_work_seed777"):
        for key in ("val_md5", "csv_md5"):
            rows.append(audit(meta, args.repo, tag, key))

    summary = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")

    shared_any = any(r["shared"] > 0 for r in rows)
    print("\n" + "=" * 78)
    if shared_any:
        print("The CDD study's split is NOT group-disjoint on the environment series:")
        print("environment windows appear on both sides of its own train|test boundary.")
        print("This is read off the authors' committed split files, not inferred from the text.")
    else:
        print("No environment window crosses the CDD study's train|test boundary.")
        print("The split IS group-disjoint. The leakage refutation does not apply to it and")
        print("must be withdrawn from the manuscript.")
    print("=" * 78)

    max_direct_trval = max(r["direct_trval"] for r in rows)
    if max_direct_trval > 0:
        print("\nAnd cruder than any window argument, needing no session=label premise:")
        print(f"up to {max_direct_trval:.1%} of test ids appear VERBATIM in train+val (same sample) —")
        print("the benchmark trains on its own test set. This inflates the absolute numbers")
        print("of both modalities equally, so it largely cancels in the env INCREMENT (which")
        print("the env-window breach above, not this, is what makes a leakage artefact).")
        print("=" * 78)
