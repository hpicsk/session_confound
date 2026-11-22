"""test_assertions.py — executable checks of the claims in paper/main.tex.

Every number that the manuscript prints is a macro in paper/params.tex. Each macro
is bound here to the committed, imagery-free table under code/ that measures it and
asserted to be that value's correct rounding, and a macro with neither a binding
nor an entry in paper/params_exempt.md fails. Directional claims are asserted on
per-seed data with paired 95% confidence intervals. Further checks read the
manuscript for sentences whose claim carries no number.

Run:  python test_assertions.py       (also pytest-discoverable)
"""

import ast
import csv
import fnmatch
import json
import math
import re
import subprocess
import sys
import tokenize
from datetime import date
from pathlib import Path

import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
AIHUB = HERE / "code" / "aihub_subtropical"   # AI Hub dataset: code + imagery-free CSVs
CDD = HERE / "code" / "dacon_cropdisease"   # CDD: code + imagery-free CSVs

PARAMS = HERE / "paper" / "params.tex"          # hand-authored macros the manuscript body prints
PARAMS_EXEMPT = HERE / "paper" / "params_exempt.md"   # macros with no data source, and why
MAIN = HERE / "paper" / "main.tex"
SUPP = HERE / "paper" / "supplementary.tex"   # the Supplementary Material, built from the same params.tex

# One level of nesting, which is all a macro body ever carries: the LaTeX thin-space
# thousands separator, 274{,}000.
_MACRO = r"\\newcommand\{\\%s\}\{((?:[^{}]|\{[^{}]*\})*)\}"


def _param(name):
    """The value \\name prints in the manuscript, read from paper/params.tex.

    A number this file guards against a CSV is not the number the paper publishes
    until the macro is read too; params.tex is hand-authored, so the two can drift
    with every assertion still green.
    """
    m = re.search(_MACRO % name, PARAMS.read_text(encoding="utf-8"))
    assert m is not None, f"params.tex defines no \\{name}"
    return m.group(1)


# The committed tables the scripts write beside themselves — never the git-ignored scratch paths.
TIMESTAMP_CSV = AIHUB / "timestamp_vs_panel_multiseed.csv"
CDD_SERIES_IDS = CDD / "dacon_series_ids.parquet"      # each image's identifier and two hashes of its series
CDD_GRANULARITY_JSON = CDD / "dacon_granularity.json"  # the counts printed from the CDD labels and dates
CDD_LEAK_CSV = CDD / "dacon_audited_split_leakage.csv"    # the leakage of the study's committed splits
CDD_DRAWN_SPLITS_CSV = CDD / "dacon_published_split.csv"         # the random and grouped splits we draw
CDD_STUDY_SPLIT_CSV = CDD / "dacon_their_split.csv"        # the classifiers on the study's own splits
THEIR_SPLIT_TAG = "paper_work_seed777"   # the set src/main.py:48-49,153 hardcodes

N_SEEDS = 7                   # seed set {0..6}

# The CDD study's reported macro-F1 (Applied Sciences 14(10):4322, 2024, Table 2):
# image-only 0.8779 -> multimodal 0.9037, a +0.0258 increment from an LSTM over 3 env
# channels. The study's numbers, not ours.
PAPER_IMAGE_F1 = 0.8779
PAPER_MULTIMODAL_F1 = 0.9037
PAPER_INCREMENT = 0.0258


# ---------------------------------------------------------------------------
# AI Hub dataset — the timestamp-baseline comparison over seeds
# ---------------------------------------------------------------------------
def _timestamp_summary(pcol="panel", ccol="timestamp"):
    # pcol/ccol select the metric: accuracy (panel/timestamp) or balanced accuracy
    # (panel_bal/timestamp_bal). Same paired-CI logic for both.
    assert TIMESTAMP_CSV.exists(), (
        f"timestamp-baseline per-seed data missing at {TIMESTAMP_CSV}; regenerate via "
        f"code/aihub_subtropical/timestamp_baseline_multiseed.py"
    )
    df = pd.read_csv(TIMESTAMP_CSV)
    tcrit = stats.t.ppf(0.975, N_SEEDS - 1)
    out = {}
    for crop, g in df.groupby("crop"):
        d = g[ccol].to_numpy() - g[pcol].to_numpy()  # paired within seed
        md, sd = d.mean(), d.std(ddof=1)
        hw = tcrit * sd / math.sqrt(len(d))
        lo, hi = md - hw, md + hw
        out[crop] = dict(
            n_seeds=len(d),
            panel=g[pcol].mean(), timestamp=g[ccol].mean(),
            md=md, lo=lo, hi=hi,
            # sec:stats: one score exceeds the other only where the interval of their
            # paired difference excludes zero
            timestamp_higher=lo > 0, panel_higher=hi < 0,
        )
    return df, out


WITHIN_SESSION_CSV = AIHUB / "timestamp_vs_panel.csv"   # timestamp_baseline.py's per-seed record
FLOOR_SWEEP_CSV = AIHUB / "within_session_floor_sweep.csv"  # its session-floor check


def _within_session_advantage():
    """Panel advantage over the majority-class predictor, within farm-day sessions and
    over the same sessions' scored rows pooled, one row per crop and seed.

    timestamp_baseline.py's committed table is the source for the manuscript's within-session
    sentences. It runs over the timestamp baseline's seed set, so each value carries the
    t interval over seeds that sec:stats requires before one score is said to exceed another.
    """
    df = pd.read_csv(WITHIN_SESSION_CSV)
    df["advantage_pooled"] = df["panel_kept"] - df["base_kept"]
    df["advantage_within"] = df["panel_in_sess"] - df["base_in_sess"]
    df["advantage_fall"] = df["advantage_pooled"] - df["advantage_within"]
    return df


def _floor_sweep():
    """{floor: {crop: (mean, lo, hi) of the within-session advantage over seeds}}."""
    df = pd.read_csv(FLOOR_SWEEP_CSV)
    df["advantage_within"] = df["panel_in_sess"] - df["base_in_sess"]
    return {floor: {crop: _seed_ci(g.sort_values("seed")["advantage_within"])
                    for crop, g in gf.groupby("crop")}
            for floor, gf in df.groupby("floor")}


def _seed_ci(x):
    """(mean, lo, hi): the 95% t interval over seeds of one value per seed."""
    x = pd.Series(x, dtype=float).to_numpy()
    hw = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return x.mean(), x.mean() - hw, x.mean() + hw


def _holm_crops(df, a, b, alpha=None):
    """Per crop, the paired t-test over seeds of column a minus column b, Holm's correction
    across the crops at code/tables.py's HOLM_ALPHA, the level sec:stats prints. {'md': {crop: mean}, 'higher': {crops a > b}, 'lower': {...}}."""
    alpha = _script_const("code/tables.py", "HOLM_ALPHA") if alpha is None else alpha
    res = {}
    for crop, g in df.groupby("crop"):
        g = g.sort_values("seed")
        d = g[a].to_numpy() - g[b].to_numpy()
        res[crop] = (d.mean(), stats.ttest_1samp(d, 0).pvalue)
    order, rej = sorted(res, key=lambda c: res[c][1]), {}
    for k, c in enumerate(order):
        rej[c] = bool(res[c][1] <= alpha / (len(order) - k) and all(rej.values()))
    return dict(md={c: r[0] for c, r in res.items()},
                higher={c for c in res if rej[c] and res[c][0] > 0},
                lower={c for c in res if rej[c] and res[c][0] < 0})


TUNED_CSV = AIHUB / "timestamp_vs_panel_tuned.csv"   # timestamp_baseline_tuned.py, one row per fold


def _tuned_long():
    """The tuned forests' scores, one row per crop and seed, the folds averaged within seed,
    in the column names of timestamp_vs_panel_multiseed.csv."""
    folds = pd.read_csv(TUNED_CSV)
    long = folds.groupby(["crop", "seed", "arm"])[["acc", "bal"]].mean().unstack("arm")
    long.columns = [arm if m == "acc" else f"{arm}_bal" for m, arm in long.columns]
    return long.reset_index()


def test_panel_reaches_no_label_branch():
    # sec:corpora separates the metadata rows that carry a panel from the labels over
    # images, and every panel number in this paper is read from the metadata table.
    # That separation is a measurement, not a reading of the dataset documentation:
    # scan_panel.py walks every shipped label and finds no panel in any of them.
    scan = json.loads((AIHUB / "panel_scan.json").read_text(encoding="utf-8"))
    assert scan["any_panel"] == 0 and scan["full_panel"] == 0, (
        f"Manuscript (sec:corpora) says that each metadata row 'also gives the farm, "
        f"the date and time at which the image was photographed, a soil and ambient "
        f"panel', "
        f"apart from the object-detection label files, two sources joined by the "
        f"metadata table, but "
        f"{scan['any_panel']:,} of {scan['n']:,} shipped labels carry a panel value "
        f"themselves; the panel would then have a second source and the row count "
        f"and the image count would no longer be measuring different things"
    )


def test_timestamp_within_session_advantage_collapses():
    # sec:res-within: the within-session restriction is what separates chemistry from the
    # session it was read in. Each half of the sentence is a conclusion that one score exceeds
    # another, so each is held to sec:stats's rule: the t interval over seeds excludes zero.
    df = _within_session_advantage()
    seeds = set(df["seed"])
    assert seeds == set(_script_const("code/aihub_subtropical/timestamp_baseline_multiseed.py", "SEEDS")), (
        f"sec:timestamp-methods averages the within-session advantage over the timestamp "
        f"comparison's seeds, 'and then over the crops and the seeds', but "
        f"timestamp_vs_panel.csv holds seeds {sorted(seeds)}"
    )
    # "the crops" of sec:res-within are the compared crops: "Each compared crop holds sessions
    # that meet the session minimum", so every compared crop holds a scored session.
    _, compared = _timestamp_summary()
    scored = set(df.dropna(subset=["advantage_within", "advantage_pooled"])["crop"])
    assert scored == set(compared), (
        f"sec:res-within says 'Each compared crop holds sessions that meet the session minimum', "
        f"so each of the {len(compared)} compared crops holds a scored "
        f"session, but the within-session measurement scores {sorted(scored)} of "
        f"{sorted(compared)}"
    )
    fall = df.groupby("seed")["advantage_fall"].mean()
    m, lo, hi = _seed_ci(fall)
    assert lo > 0, (
        f"Manuscript (sec:res-within) says the advantage 'falls from 0.251 with the "
        f"sessions pooled to 0.045 within single sessions', a fall whose interval paired "
        f"within seed must lie above zero, but the crop mean of the pooled minus the "
        f"within-session advantage is {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    )
    # sec:discussion ("most of their advantage on the AI Hub dataset comes from
    # differences between sessions") says "most": the fall exceeds half the pooled advantage.
    pooled = df.groupby("seed")["advantage_pooled"].mean().mean()
    within = df.groupby("seed")["advantage_within"].mean().mean()
    assert pooled - within > 0.5 * pooled, (
        f"sec:discussion says that 'most of their advantage on the AI Hub dataset comes "
        f"from differences between sessions', but the advantage falls "
        f"from {pooled:.3f} pooled to {within:.3f} within sessions"
    )
    resid = {crop: _seed_ci(g.sort_values("seed")["advantage_within"])
             for crop, g in df.groupby("crop")}
    under = {crop: r for crop, r in resid.items() if r[1] <= 0}
    assert not under, (
        f"Manuscript (sec:res-within) says that 'in each crop the within-session advantage "
        f"stays above zero', but the interval does not lie above zero in {under}"
    )
    # sec:stats counts a crop only where its paired t-test survives Holm's correction across
    # the crops, so "in each crop" is that count.
    holm = _holm_crops(df.assign(zero=0.0), "advantage_within", "zero")["higher"]
    assert holm == set(df["crop"]), (
        f"Manuscript (sec:res-within) says that 'in each crop the within-session advantage "
        f"stays above zero', a count that sec:stats takes after Holm's correction, but the "
        f"advantage lies above zero after the correction only in {sorted(holm)}"
    )


def test_timestamp_seed_set_complete():
    _, s = _timestamp_summary()
    assert len(s) == 7, f"the timestamp-baseline comparison covers 7 crops; found {len(s)}"
    # 설명서 documents 8 crops (애플망고·백향과·올리브·용과·바나나·레몬·무화과·한라봉);
    # the within-crop test runs on 7. Two conditions each exclude 한라봉 -- it ships
    # no soil_K (README caveat), and it carries a single labeled class across the
    # dataset -- and timestamp_baseline_multiseed.py's dropna applies the first before the
    # >= 2-class rule is reached. Pins sec:diagnostic's crop-inclusion sentence.
    expected_crops = {"레몬", "무화과", "바나나", "백향과", "애플망고", "올리브", "용과"}
    assert set(s) == expected_crops, (
        f"sec:timestamp-methods says the crop rule 'excludes only the Hallabong mandarin', "
        f"leaving 7 of the dataset's 8 crops, but the timestamp-baseline CSV covers {set(s)} "
        f"(missing: {expected_crops - set(s)}; unexpected: {set(s) - expected_crops})"
    )
    # The crop denominator prints from its own macro, so a change to the seed set cannot
    # silently rewrite the crop count.
    assert _param("timestampCrops") == str(len(s)), (
        f"\\timestampCrops prints {_param('timestampCrops')} but the comparison covers "
        f"{len(s)} crops"
    )
    for crop, r in s.items():
        assert r["n_seeds"] == N_SEEDS, (
            f"sec:timestamp-methods says 'We repeated the draw and the cross-validation for "
            f"each of 7 random seeds' but crop {crop} has {r['n_seeds']} seeds"
        )


TIMESTAMP_HIGHER = {"레몬", "무화과", "바나나", "애플망고", "용과"}  # sec:res-timestamp's five
OTHER_THREE = TIMESTAMP_HIGHER - {"무화과", "애플망고"}             # "the other three"


def test_the_timestamp_is_more_accurate_in_five_crops():
    # sec:res-timestamp: "the timestamp classifier is more accurate than the panel
    # classifier in 5 of 7 crops ... In passion fruit, the comparison detects no
    # difference between the two classifiers. In olive, the soil-chemistry classifier is more
    # accurate". The three sets are derived from every crop that the per-seed table holds.
    _, s = _timestamp_summary()
    higher = {c for c, r in s.items() if r["timestamp_higher"]}
    lower = {c for c, r in s.items() if r["panel_higher"]}
    assert higher == TIMESTAMP_HIGHER and lower == {"올리브"}, (
        f"sec:res-timestamp says the timestamp classifier is more accurate in lemon, fig, "
        f"banana, apple mango and dragon fruit and the soil-chemistry classifier in olive alone, but "
        f"the intervals give {sorted(higher)} and {sorted(lower)}"
    )
    assert set(s) - higher - lower == {"백향과"}, (
        f"sec:res-timestamp says the comparison detects no difference in passion fruit "
        f"alone, but it detects none in {sorted(set(s) - higher - lower)}"
    )
    assert _param("timestampHigherCrops") == str(len(higher)), (
        f"\\timestampHigherCrops prints {_param('timestampHigherCrops')} but the timestamp "
        f"classifier is more accurate in {len(higher)} crops"
    )
    small = [s[c]["md"] for c in OTHER_THREE]
    for macro, value in (("timestampMarginSmallLo", min(small)),
                         ("timestampMarginSmallHi", max(small))):
        assert math.isclose(float(_param(macro)), value, abs_tol=5e-4), (
            f"sec:res-timestamp prints \\{macro} as {_param(macro)} for the other three "
            f"crops, but the margins there give {value:+.4f}"
        )


def test_fig_and_apple_mango_carry_the_largest_timestamp_margins():
    _, s = _timestamp_summary()
    # The manuscript names these crops in English; the Korean keys are the CSV's own.
    for crop, name, claimed in [("무화과", "fig", 0.154), ("애플망고", "apple mango", 0.060)]:
        r = s[crop]
        assert math.isclose(r["md"], claimed, abs_tol=1e-3), (
            f"sec:res-timestamp prints the {name} timestamp margin as {claimed:+.3f} "
            f"but data shows {r['md']:+.3f}"
        )
        rest = max(v["md"] for c, v in s.items() if c not in ("무화과", "애플망고"))
        assert r["md"] > rest, (
            f"sec:res-timestamp prints fig's and apple mango's margins apart from the 'between' "
            f"range of the other three crops, as the two largest, but {name}'s "
            f"{r['md']:+.3f} is not above every other crop's"
        )


def test_olive_is_the_one_crop_with_a_negative_margin():
    _, s = _timestamp_summary()
    r = s["올리브"]
    assert math.isclose(r["md"], -0.020, abs_tol=1e-3), (
        f"sec:res-timestamp says 'in olive the soil-chemistry classifier is more accurate, by 0.020' "
        f"but data shows a timestamp margin of {r['md']:+.3f}"
    )
    df = pd.read_csv(TIMESTAMP_CSV)
    for pcol, ccol in (("panel", "timestamp"), ("panel_bal", "timestamp_bal")):
        md = (df[ccol] - df[pcol]).groupby(df["crop"]).mean()
        assert list(md[md < 0].index) == ["올리브"], (
            f"sec:res-timestamp names olive as the one crop in which the soil-chemistry classifier is "
            f"more accurate, but in {ccol} the crops with a negative mean margin are "
            f"{list(md[md < 0].index)}"
        )


def test_the_full_panel_is_more_accurate_than_the_timestamp_in_every_crop():
    # sec:res-timestamp: "The full-panel classifier, by contrast, is more accurate than the
    # timestamp classifier in every compared crop, in both metrics and with the forests
    # untuned or tuned." Over every crop that the per-seed tables hold, Holm across crops.
    for name, df in (("untuned", pd.read_csv(TIMESTAMP_CSV)), ("tuned", _tuned_long())):
        for sfx in ("", "_bal"):
            r = _holm_crops(df, "panel_full" + sfx, "timestamp" + sfx)
            short = sorted(set(df["crop"]) - r["higher"])
            assert not short, (
                f"sec:res-timestamp says the full-panel classifier 'is more accurate than the "
                f"timestamp classifier in every compared crop, in both metrics and with the "
                f"forests untuned or tuned', but with the {name} forests in "
                f"{'balanced accuracy' if sfx else 'accuracy'} it is not, after Holm, in {short}"
            )


FULL_BEYOND_FARM = {"레몬", "무화과", "애플망고", "올리브"}   # sec:res-timestamp's four crops


def test_farm_identity_does_not_account_for_the_full_panels_lead_in_four_crops():
    # sec:res-timestamp: "Adding the farm to the timestamp closes that lead in banana and
    # passion fruit, where the farm-and-timestamp classifier and the full-panel classifier do
    # not differ, and in dragon fruit, where the farm-and-timestamp classifier is the more
    # accurate. In the other 4 crops, lemon, fig, apple mango, and olive, the full-panel
    # classifier stays more accurate than the farm-and-timestamp classifier in both metrics and
    # with the forests untuned or tuned".
    for name, df in (("untuned", pd.read_csv(TIMESTAMP_CSV)), ("tuned", _tuned_long())):
        for sfx in ("", "_bal"):
            r = _holm_crops(df, "panel_full" + sfx, "farm_ts" + sfx)
            assert r["higher"] == FULL_BEYOND_FARM and r["lower"] == {"용과"}, (
                f"sec:res-timestamp says that in lemon, fig, apple mango, and olive 'the full-panel "
                f"classifier stays more accurate than the farm-and-timestamp classifier' and that in "
                f"dragon fruit 'the farm-and-timestamp classifier is the more accurate', but "
                f"with the {name} forests in {'balanced accuracy' if sfx else 'accuracy'} "
                f"Holm gives the full panel ahead in {sorted(r['higher'])} and behind in "
                f"{sorted(r['lower'])}"
            )
    # "One of these crops, fig, has all its images from a single farm, so in fig the farm adds
    # nothing to the timestamp."
    farms = pd.read_csv(TIMESTAMP_CSV).groupby("crop")["n_farms"].first()
    assert farms["무화과"] == 1 and (farms.drop("무화과") > 1).all(), (
        f"sec:res-timestamp says that fig 'has all its images from a single farm', but the compared "
        f"rows give farms per crop {farms.to_dict()}"
    )


def test_tuned_forests_give_the_same_five_crops_and_olive():
    # sec:res-timestamp: "With tuned forests, the timestamp classifier is more accurate than the
    # soil-chemistry classifier in the same five crops and less accurate in olive, and has the higher
    # balanced accuracy in the same five crops and the lower in olive", and "With untuned
    # forests, the timestamp classifier has the higher balanced accuracy in those five crops and
    # in passion fruit, and the lower in olive".
    expect = {("untuned", ""): TIMESTAMP_HIGHER, ("untuned", "_bal"): TIMESTAMP_HIGHER | {"백향과"},
              ("tuned", ""): TIMESTAMP_HIGHER, ("tuned", "_bal"): TIMESTAMP_HIGHER}
    for name, df in (("untuned", pd.read_csv(TIMESTAMP_CSV)), ("tuned", _tuned_long())):
        for sfx in ("", "_bal"):
            r = _holm_crops(df, "timestamp" + sfx, "panel" + sfx)
            assert r["higher"] == expect[(name, sfx)] and r["lower"] == {"올리브"}, (
                f"sec:res-timestamp says 'With tuned forests, the timestamp classifier is more "
                f"accurate than the soil-chemistry classifier in the same five crops and less accurate in "
                f"olive, and has the higher balanced accuracy in the same five crops and the lower "
                f"in olive', and adds passion fruit only for the untuned forests in balanced "
                f"accuracy, but the {name} forests in "
                f"{'balanced accuracy' if sfx else 'accuracy'} give the timestamp ahead in "
                f"{sorted(r['higher'])} and behind in {sorted(r['lower'])}"
            )
    # Every tuned run covers the untuned run's crops, seeds and three outer folds, and every
    # selected setting lies in the grid that tab:settings prints.
    folds = pd.read_csv(TUNED_CSV)
    grid = _script_const("code/aihub_subtropical/timestamp_baseline_tuned.py", "GRID")
    assert (folds.groupby(["crop", "seed", "arm"]).size() == 3).all() and \
        set(folds["crop"]) == set(pd.read_csv(TIMESTAMP_CSV)["crop"]), (
        "the tuned run does not cover every crop, seed and classifier on three outer folds")
    assert set(folds["max_features"]) <= {str(v) for v in grid["max_features"]} and \
        set(folds["min_samples_leaf"]) <= set(grid["min_samples_leaf"]), (
        "the tuned run selected a setting outside the grid that tab:settings prints")


def test_the_full_panel_reads_soil_moisture_in_apple_mango_alone():
    # sec:timestamp-methods: the full-panel classifier reads, beside the soil chemistry, "air
    # temperature, air humidity, and soil temperature" in every compared crop, "and in apple
    # mango, soil moisture as well", read off the variables
    # that timestamp_baseline_multiseed.py recorded for each crop.
    df = pd.read_csv(TIMESTAMP_CSV)
    read = df.groupby("crop")["full_vars"].first().str.split("+")
    moist = sorted(c for c, v in read.items() if "soil_condition" in v)
    assert moist == ["애플망고"], (
        f"sec:timestamp-methods says the full-panel classifier reads soil moisture 'in apple "
        f"mango' alone, but it reads soil moisture in {moist}"
    )
    lacking = {c: sorted({"temp", "hum", "soil_temp"} - set(v)) for c, v in read.items()}
    assert not any(lacking.values()), (
        f"sec:timestamp-methods says the full-panel classifier 'reads every panel variable "
        f"recorded in the rows', air temperature, air humidity and soil temperature in every "
        f"compared crop, but {lacking}"
    )


def test_the_within_session_sentences_across_session_minimums():
    # timestamp_baseline.py's session-floor check. Its 300-row rerun must reproduce the main
    # table, or the sweep measures something else.
    SESSION_MIN_ROWS = _script_literal("code/aihub_subtropical/timestamp_baseline.py",
                                       r"len\(gs\) < (\d+)")
    sweep = pd.read_csv(FLOOR_SWEEP_CSV)
    main = pd.read_csv(WITHIN_SESSION_CSV).set_index(["crop", "seed"])
    floors = _script_const("code/aihub_subtropical/timestamp_baseline.py", "FLOOR_SWEEP")
    same = sweep[sweep["floor"] == SESSION_MIN_ROWS].set_index(["crop", "seed"])
    for col in ("panel_in_sess", "base_in_sess", "panel_kept", "base_kept"):
        assert (same[col] == main.loc[same.index, col]).all(), (
            f"the session-floor check reruns the {SESSION_MIN_ROWS}-row minimum, but its {col} "
            f"differs from timestamp_vs_panel.csv"
        )
    sweep["pooled"] = sweep["panel_kept"] - sweep["base_kept"]
    sweep["within"] = sweep["panel_in_sess"] - sweep["base_in_sess"]
    for floor, g in sweep.groupby("floor"):
        per_seed = g.groupby("seed")[["pooled", "within"]].mean()
        _, lo, _ = _seed_ci(per_seed["pooled"] - per_seed["within"])
        pooled, within = per_seed["pooled"].mean(), per_seed["within"].mean()
        assert lo > 0 and pooled - within > 0.5 * pooled, (
            f"sec:res-within reports the within-session advantage 'At session minimums of "
            f"{min(floors)} and {max(floors):,} rows', where it must fall as at 300 rows, but at "
            f"{floor} rows it falls from {pooled:.3f} to {within:.3f} (fall's lower limit {lo:+.4f})"
        )
    by_floor = _floor_sweep()
    under = {(f, c) for f in (min(floors), max(floors))
             for c, r in by_floor[f].items() if r[1] <= 0}
    # sec:stats counts a crop only where its paired t-test survives Holm's correction.
    sweep["zero"] = 0.0
    for f in (min(floors), max(floors)):
        g = sweep[sweep["floor"] == f]
        holm = _holm_crops(g, "within", "zero")["higher"]
        under |= {(f, c) for c in set(g["crop"]) - holm}
    assert not under, (
        f"sec:res-within says that at session minimums of {min(floors)} and {max(floors):,} "
        f"rows 'the within-session advantage also stays above zero in every compared crop', "
        f"but it does not at {sorted(under)}"
    )
    assert sorted(floors)[1] == SESSION_MIN_ROWS, (
        f"tab:floors brackets the {SESSION_MIN_ROWS}-row minimum by the other two, but the "
        f"sweep's middle floor is {sorted(floors)[1]}"
    )


def test_every_compared_crop_exceeds_the_row_draw():
    # sec:timestamp-methods: the crop rule "excludes only the Hallabong mandarin", and the draw
    # takes 20,000 rows in each compared crop, so every compared crop has more than that. Derived from every crop
    # that the results CSV holds, not from a list.
    df = pd.read_csv(TIMESTAMP_CSV)
    sample = _script_const("code/aihub_subtropical/timestamp_baseline_multiseed.py", "SAMPLE")
    small = df.groupby("crop")["n"].first()
    small = small[small <= sample]
    assert small.empty, (
        f"sec:timestamp-methods says 'Each of the 7 compared crops has more than 20,000 rows "
        f"that carry all five soil-chemistry variables', the draw of {sample}, but "
        f"{dict(small)} keep no more"
    )


def test_in_balanced_accuracy_the_date_disjoint_gain_lies_above_zero():
    # sec:fusion: "In macro-F1, however, the date-disjoint gain itself is also not detectably
    # above zero, whereas in balanced accuracy it is." The matched-budget date-disjoint gain in
    # balanced accuracy, paired within seed.
    *_, s6 = _fusion_date_summary("test_ba")
    assert s6["d_lo"] > 0.0, (
        f"sec:fusion says of the date-disjoint gain 'whereas in balanced accuracy it is' above "
        f"zero, but its interval is [{s6['d_lo']:+.4f}, {s6['d_hi']:+.4f}]"
    )


def test_in_balanced_accuracy_the_environment_classifier_scores_higher_group_disjoint():
    # sec:res-within: "whereas in balanced accuracy the environment classifier scores higher
    # on the group-disjoint split" (tab:cdd prints the paired difference).
    _, s1 = _cdd_drawn_splits_summary("ba")
    hi = s1["group"]["timestamp_over_env_hi"]
    assert hi < 0, (
        f"sec:res-within claims that 'in balanced accuracy the environment classifier scores "
        f"higher on the group-disjoint split', but the paired timestamp-minus-"
        f"environment interval reaches {hi:+.4f}"
    )


def test_the_date_disjoint_test_detects_no_loss_in_either_metric():
    # sec:discussion: "On the AI Hub dataset, however, the date-disjoint test detected no loss of
    # the gain", in neither metric. A loss is a paired reduction (in-distribution gain minus
    # date-disjoint gain) whose interval lies above zero.
    for metric in ("test_f1", "test_ba"):
        *_, s6 = _fusion_date_summary(metric)
        assert s6["drop_lo"] <= 0.0, (
            f"sec:fusion says 'the date-disjoint split detects no loss of the gain on held-out "
            f"dates', but in "
            f"{metric} the paired reduction's "
            f"interval [{s6['drop_lo']:+.4f}, {s6['drop_hi']:+.4f}] lies above zero"
        )


# ---------------------------------------------------------------------------
# AI Hub dataset — the SAME comparison in BALANCED accuracy, the manuscript's check that
# no comparison depends on the sizes of the classes. timestamp_baseline_multiseed.py emits both
# metrics from identical fits (panel_bal/timestamp_bal); these check that the verdict is
# ROBUST to the metric swap.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CDD — granularity (the environment series is per session, not per image)
# ---------------------------------------------------------------------------
# The per-image record (labels, crops, diseases, photo dates) is the competition's
# data and stays in $TROPICAL_SCRATCH. dacon_granularity.py publishes two tables
# from it: each image's identifier with the two hashes of its series, and the
# counts that the manuscript prints from the labels and dates.
def _cdd_series_ids():
    assert CDD_SERIES_IDS.exists(), (
        f"CDD series identifiers missing at {CDD_SERIES_IDS}; regenerate via "
        f"dacon_granularity.py (reads the CDD files under $TROPICAL_SCRATCH, which are not redistributed)"
    )
    return pd.read_parquet(CDD_SERIES_IDS)


def _cdd_granularity():
    assert CDD_GRANULARITY_JSON.exists(), (
        f"CDD label and date counts missing at {CDD_GRANULARITY_JSON}; regenerate via "
        f"dacon_granularity.py (reads the CDD files under $TROPICAL_SCRATCH, which are not redistributed)"
    )
    return json.loads(CDD_GRANULARITY_JSON.read_text(encoding="utf-8"))


def test_cdd_sample_count():
    m = _cdd_series_ids()
    assert len(m) == 5767, (
        f"sec:res-diagnostic claims '5,767 images' (CDD granularity) "
        f"but dacon_series_ids.parquet has {len(m)}"
    )


def test_cdd_env_series_not_per_image():
    m = _cdd_series_ids()
    distinct = m["val_md5"].nunique()
    assert distinct == 780, (
        f"sec:res-diagnostic claims 'Only 780 distinct environment series exist for its 5,767 images' "
        f"but data shows {distinct}"
    )


def test_cdd_images_share_env_series():
    m = _cdd_series_ids()
    sizes = m.groupby("val_md5").size()
    pct = 100 * int(sizes[sizes > 1].sum()) / len(m)
    assert math.isclose(pct, 96.4, abs_tol=0.05), (
        f"sec:res-diagnostic claims 'Of the images, 96.4% share their series with at "
        f"least one other image' but data shows {pct:.1f}%"
    )


def test_cdd_session_is_one_greenhouse_day():
    pct = _cdd_granularity()["shared_series_single_date_pct"]
    assert math.isclose(pct, 98.6, abs_tol=0.05), (
        f"sec:res-diagnostic claims 'Of the shared series, 98.6% span a single photo date' but "
        f"{pct:.1f}% of the shared env-series clusters span exactly one photo date"
    )


def test_cdd_env_collinear_with_label():
    pct = _cdd_granularity()["shared_series_single_disease_pct"]
    assert math.isclose(pct, 98.4, abs_tol=0.05), (
        f"sec:res-diagnostic claims that of the shared series '98.4% carry a single disease' "
        f"(CDD bullet) but {pct:.1f}% of shared clusters map to one disease"
    )


def test_cdd_region_is_constant():
    regions = _cdd_granularity()["regions"]
    assert regions == 1, (
        f"sec:res-diagnostic reads 'nearly every shared series identifies one day', "
        f"which needs "
        f"region constant, or the series would be "
        f"a region signature rather than a session one; but region takes "
        f"{regions} values"
    )


# ---------------------------------------------------------------------------
# CDD — leakage in the CDD study's OWN committed splits.
#
# dacon_audited_split_leakage.py is a lookup that reads no image: it reads the split files shipped
# by the study's own repository (hyunseoki/crop_disease, named in its Data Availability Statement)
# and joins their train/val/test ids to the env-window hash (val_md5 / csv_md5) of the
# per-image record that dacon_granularity.py writes to $TROPICAL_SCRATCH.
# Nothing is re-drawn or re-seeded — the answer is a property of files that already exist. The
# committed summary holds only aggregate breach RATES and group counts (no image id, no label).
# 91.9% of the test images of seed777, the split their code reads, have their byte-identical
# env window in training (97.1% for seed42), against 95.9% for a random split that we draw and
# 0.0% for a group-disjoint one — so their split is not group-disjoint on the environment
# series. The same lookup also reports a cruder, premise-free companion — DIRECT same-sample
# overlap (test ids appearing verbatim in train / train+val): 31.5%/50.2% for seed777,
# 49.8%/79.8% for seed42, so the study scores its models partly on images they were trained
# on. Both are guarded here; regenerate via dacon_audited_split_leakage.py --repo <clone of
# hyunseoki/crop_disease>, which overwrites the committed table in place.
# ---------------------------------------------------------------------------
def _cdd_study_split_summary():
    assert CDD_LEAK_CSV.exists(), (
        f"leakage summary of the study's splits missing at {CDD_LEAK_CSV}; regenerate via "
        f"dacon_audited_split_leakage.py --repo <clone of hyunseoki/crop_disease>, "
        f"which reads the per-image record in $TROPICAL_SCRATCH and the clone's split files "
        f"and writes this path"
    )
    df = pd.read_csv(CDD_LEAK_CSV)
    val = {r.tag: r for r in df[df.key == "val_md5"].itertuples()}  # sec:res-cdd quotes val_md5
    return df, val


def test_cdd_study_split_join_is_complete():
    # sec:res-cdd provenance: "read off the authors' own files". Their repo ships
    # paper_work_seed42 = 2883/1730/1154 (the stated 5:3:2 over 5,767) and paper_work_seed777
    # = 1801/1082/1154; the join to the per-image record matched every committed id (the script
    # refuses to report a rate over a partial join), pinning the sizes the paper quotes.
    df, val = _cdd_study_split_summary()
    assert set(zip(df.tag, df.key)) == {
        ("paper_work_seed42", "val_md5"), ("paper_work_seed42", "csv_md5"),
        ("paper_work_seed777", "val_md5"), ("paper_work_seed777", "csv_md5"),
    }, f"the leakage summary should cover 2 committed splits x 2 env keys but has {set(zip(df.tag, df.key))}"
    s42, s777 = val["paper_work_seed42"], val["paper_work_seed777"]
    assert (s42.n_train, s42.n_val, s42.n_test) == (2883, 1730, 1154), (
        f"sec:reanalysis says that only the other set's partitions hold the ratio that the "
        f"study states, which needs paper_work_seed42 = 2883/1730/1154 (5:3:2 over 5,767), "
        f"but the join has {(s42.n_train, s42.n_val, s42.n_test)}"
    )
    assert (s777.n_train, s777.n_val, s777.n_test) == (1801, 1082, 1154), (
        f"sec:reanalysis says that only the other set holds the study's 5:3:2 ratio, and "
        f"sec:res-cdd reads the study's split, paper_work_seed777, as 1801/1082/1154 "
        f"but the join has {(s777.n_train, s777.n_val, s777.n_test)}"
    )


def test_cdd_study_split_leaks():
    # sec:res-cdd: "For 91.9% of its test images, the identical environment
    # series also occurs in training (91.0% with the test image's own class, 96.4% in
    # training and validation together)"; "Of the 485 distinct test series, 406 recur in
    # training" — paper_work_seed777,
    # val_md5. That is the set src/main.py hardcodes for train, val and test alike, so it
    # is the split behind the figures the study reports and the split every classifier of
    # _their_split_summary is scored on, so the leak and the direct overlap below read the
    # same split.
    _, val = _cdd_study_split_summary()
    r = val[THEIR_SPLIT_TAG]
    for got, claimed, name in [
        (r.breach, 0.9185, "'91.9%' env-window breach"),
        (r.breach_lab, 0.9099, "'91.0%' label-mapped breach"),
        (r.breach_trval, 0.9645, "'96.4%' train+val breach"),
    ]:
        assert math.isclose(got, claimed, abs_tol=1e-3), (
            f"sec:res-cdd claims {name} but data shows {got:.4f}"
        )
    assert (int(r.shared), int(r.n_test_groups)) == (406, 485), (
        f"sec:res-cdd claims 'Of the 485 distinct test series, 406 recur in training' but data "
        f"shows {int(r.shared)} of {int(r.n_test_groups)}"
    )
    # The same sentence goes on: "so the shared series are most of the test series". "Most"
    # is more than half of the distinct test series.
    assert 2 * int(r.shared) > int(r.n_test_groups), (
        f"sec:res-cdd says 'the shared series are most of the test series', but only "
        f"{int(r.shared)} of "
        f"{int(r.n_test_groups)} distinct test series recur in training"
    )


def test_the_leak_rate_is_the_same_under_the_raw_byte_hash():
    # sec:res-cdd: "the identical environment series also occurs in training, whether or not
    # the reading times are dropped before the series are matched". The Supplementary
    # Material's leak table (tab:leak) prints every leak rate twice, with the series matched
    # by their values (reading times dropped) and by their raw files, and the two rows agree. Held here, because a future build of the dataset that made the two
    # keys diverge would leave the table's two rows printing different rates that the text
    # of the paper calls by one name, the leak rate.
    df, _ = _cdd_study_split_summary()
    for tag, g in df.groupby("tag"):
        keys = set(g.key)
        assert keys == {"val_md5", "csv_md5"}, (
            f"tab:leak prints each set under both matching rules, but split {tag} carries "
            f"only {sorted(keys)}"
        )
        for col in ("breach", "breach_lab", "breach_trval", "direct", "direct_trval"):
            rates = {r.key: getattr(r, col) for r in g.itertuples()}
            assert math.isclose(rates["val_md5"], rates["csv_md5"], abs_tol=5e-4), (
                f"sec:res-cdd says the series occur in training 'whether or not the reading times "
                f"are dropped before the series are matched', but tab:leak's rows for {tag} "
                f"differ in {col}: {rates['val_md5']:.4f} under the "
                f"time-dropped hash and {rates['csv_md5']:.4f} under the raw-byte hash"
            )


def test_the_group_split_lowers_both_classifiers_without_reaching_base_rate():
    # sec:res-within: "On CDD, both the environment classifier and the timestamp classifier
    # score a lower macro-F1 on the group-disjoint split than on the random split" (each
    # interval entirely below the other, sec:stats), and "Neither classifier falls to the
    # 0.015 that the majority-class predictor scores on the group-disjoint split".
    _, s1 = _cdd_drawn_splits_summary()
    for arm in ("env", "timestamp"):
        assert s1["group"][arm + "_hi"] < s1["random"][arm + "_lo"], (
            f"sec:res-within says the {arm} classifier scores 'a lower macro-F1 on the "
            f"group-disjoint split than on the random split', but its group-disjoint interval "
            f"reaches {s1['group'][arm + '_hi']:.4f} and its random-split interval starts at "
            f"{s1['random'][arm + '_lo']:.4f}"
        )
        assert s1["group"][arm + "_lo"] > s1["group"]["base_hi"], (
            f"sec:res-within says 'Neither classifier falls to the 0.015 that the "
            f"majority-class predictor scores on the group-disjoint split', but the {arm} "
            f"classifier's interval reaches down to {s1['group'][arm + '_lo']:.4f}"
        )


def test_cdd_other_set_also_leaks():
    # sec:res-cdd: "The repository's other set of identifiers leaks 97.1% into its training
    # partition" (paper_work_seed42, val_md5).
    # Their repository ships two sets of identifiers and the training entry point reads
    # only one; this is the other.
    _, val = _cdd_study_split_summary()
    r = val["paper_work_seed42"]
    assert math.isclose(r.breach, 0.9714, abs_tol=1e-3), (
        f"sec:res-cdd claims the other committed set 'leaks 97.1%' but data shows {r.breach:.4f}"
    )


def test_cdd_study_split_is_not_group_disjoint():
    # sec:res-cdd: "neither shipped split is group-disjoint on the environment series",
    # read off the authors' own files. Every committed split (both seeds, both env
    # keys) shares env groups across its own train|test boundary, and every breach rate sits
    # nearer the random draw than the group-disjoint one — it leaks like a random split, the
    # opposite of disjoint. Both brackets are dacon_published_split.py's own two regimes,
    # measured over the same dataset on val_md5 as this check is, and they are the two rates
    # Table~\ref{tab:cdd} prints. Deriving them here rather than restating a reference measured
    # elsewhere keeps the inequality and the sentence reading the same two numbers.
    df, val = _cdd_study_split_summary()
    _, s1 = _cdd_drawn_splits_summary()
    random_leak, disjoint_leak = s1["random"]["leak"], s1["group"]["leak"]
    assert (df["shared"] > 0).all(), (
        f"sec:res-cdd's verdict needs env windows on both sides of the boundary (shared>0) for "
        f"every committed split, but some share none: "
        f"{df.loc[df['shared'] == 0, ['tag', 'key']].to_dict('records')}"
    )
    for r in df.itertuples():
        assert abs(r.breach - random_leak) < abs(r.breach - disjoint_leak), (
            f"sec:res-cdd reads the study's split as leaking at random level (ref "
            f"{random_leak:.4f}), the opposite of group-disjoint (ref {disjoint_leak:.4f}); "
            f"but {r.tag}/{r.key} breach {r.breach:.4f} is closer to disjoint than to random"
        )
    # No assertion that a committed split out-leaks the random draw. It need not:
    # paper_work_seed777 trains on half the dataset where the random draw trains on six
    # sevenths, and a smaller training pool covers fewer series. sec:res-cdd prints the
    # two reference rates as a bracket and claims only that the study's split sits at the
    # random end of it, which the loop above is what checks.


def test_cdd_study_split_direct_identifier_overlap():
    # A cruder, PREMISE-FREE companion to the env-window breach: the paper's test set overlaps
    # its own TRAIN set by sample id — same id + same label = literally training on the test
    # row. It needs no session=label argument (a strict subset of the window breach) and it
    # contaminates image-only AND multimodal equally, so it invalidates the ABSOLUTE benchmark
    # numbers but largely CANCELS in the env INCREMENT — complementary to the window breach
    # (which is what makes the +2.58%p increment, not just the absolute numbers, a leak). direct
    # is key-independent, so val_md5 and csv_md5 rows carry the same value.
    _, val = _cdd_study_split_summary()
    for tag, d, dtv, disp in [
        ("paper_work_seed777", 0.3146, 0.5017, "31.5% train∩test / 50.2% train+val∩test"),
        ("paper_work_seed42", 0.4983, 0.7981, "49.8% train∩test / 79.8% train+val∩test"),
    ]:
        r = val[tag]
        # sec:res-cdd prints the study's split; the other set's overlap is pinned unprinted
        who = "sec:res-cdd claims" if tag == THEIR_SPLIT_TAG else "this check pins"
        assert math.isclose(r.direct, d, abs_tol=1e-3), (
            f"{who} {tag} direct same-sample overlap ({disp}) but train∩test "
            f"data shows {r.direct:.4f}"
        )
        assert math.isclose(r.direct_trval, dtv, abs_tol=1e-3), (
            f"{who} {tag} direct same-sample overlap ({disp}) but "
            f"(train+val)∩test data shows {r.direct_trval:.4f}"
        )
        # structural: a same-id overlap is necessarily an env-window breach too (direct ⊂ breach).
        assert r.direct <= r.breach + 1e-9 and r.direct_trval <= r.breach_trval + 1e-9, (
            f"sec:res-cdd frames direct overlap as a SUBSET of the env-window breach ({tag}) "
            f"but direct {r.direct:.4f}/{r.direct_trval:.4f} exceeds breach "
            f"{r.breach:.4f}/{r.breach_trval:.4f}"
        )
    # seed777 is the split main.py/main_mixup.py hardcode, so its overlap is the one that
    # actually inflated the paper's reported numbers; it must be a real (nonzero) leak.
    assert val["paper_work_seed777"].direct > 0.3, (
        f"sec:res-cdd claims the split their code actually uses (seed777) trains on ~1/3 of "
        f"its own test set, but direct overlap is only {val['paper_work_seed777'].direct:.4f}"
    )


# ---------------------------------------------------------------------------
# CDD — the environment series' own power at the published 25-class task, on the splits we draw.
#
# dacon_published_split.py opens no image and trains no detector: it scores macro F1 of the
# environment series and a bare timestamp at the study's 25-class task under two splits
# differing in EXACTLY one respect — random StratifiedKFold (series shared between the sides) vs
# StratifiedGroupKFold on val_md5 (forbid the same env window on both sides). Its outcome
# mapping, which it prints and the manuscript does not use, reads P1 (LOOKUP) when the
# series clears the majority-class base by >= COLLAPSE,
# does NOT transfer (env_g < env_r - TOL), and loses more than COLLAPSE (env_r - env_g >
# COLLAPSE) — its power at this task is group identity, not agronomy. The run lands on P1:
# env 0.885 -> 0.653, timestamp 0.899 -> 0.595, val_md5 leak 0.959 -> 0.000. Guarded here from
# the committed per-seed CSV; regenerate via
#   TROPICAL_SCRATCH=<dataset root> python dacon_published_split.py
# which reads the redistribution-prohibited dataset from there and writes this table.
# ---------------------------------------------------------------------------
def _cdd_drawn_splits_summary(metric="f1"):
    assert CDD_DRAWN_SPLITS_CSV.exists(), (
        f"per-seed data of the splits we draw missing at {CDD_DRAWN_SPLITS_CSV}; regenerate via "
        f"dacon_published_split.py, which reads the redistribution-prohibited dataset "
        f"from $TROPICAL_SCRATCH and writes this path"
    )
    df = pd.read_csv(CDD_DRAWN_SPLITS_CSV)
    tcrit = stats.t.ppf(0.975, N_SEEDS - 1)

    def ci(x):
        m, sd = x.mean(), x.std(ddof=1)
        hw = tcrit * sd / math.sqrt(len(x))
        return m, m - hw, m + hw

    out = {}
    for regime, g in df.groupby("regime"):
        # the outcome mapping compares classifier MEANS per regime (random vs group are
        # unpaired — different test folds), mirroring dacon_published_split.py's summ[arm]=(mean_random, mean_group).
        # tab:cdd also prints each classifier's own interval, and sec:res-cdd reads the two
        # regimes' intervals against each other, so the spread is carried here too.
        r = dict(n_seeds=len(g), leak=g["group_leak"].mean())
        for arm in ["base", "timestamp", "env", "timestamp_env"]:
            r[arm], r[arm + "_lo"], r[arm + "_hi"] = ci(g[f"{arm}_{metric}"].to_numpy())
        # the classifiers share a fold within a seed, so a contrast between them is paired;
        # the classifier intervals above overlap and cannot settle one (sec:res-cdd).
        d = g[f"timestamp_{metric}"].to_numpy() - g[f"env_{metric}"].to_numpy()
        r["timestamp_over_env"], r["timestamp_over_env_lo"], r["timestamp_over_env_hi"] = ci(d)
        out[regime] = r
    return df, out


def test_cdd_drawn_splits_seed_set_complete():
    _, s = _cdd_drawn_splits_summary()
    assert set(s) == {"random", "group"}, (
        f"dacon_published_split.py runs 2 regimes (random split, group-disjoint "
        f"correction) but the CSV has regimes {sorted(s)}"
    )
    for regime, r in s.items():
        assert r["n_seeds"] == N_SEEDS, (
            f"dacon_published_split.py runs {N_SEEDS} seeds per regime but regime "
            f"'{regime}' has {r['n_seeds']}"
        )


def test_cdd_drawn_splits_env_power_falls_under_group_split():
    # sec:res-timestamp: "The environment classifier falls from 0.885 to 0.653" once the shared
    # env window is forbidden. dacon_published_split.py's own printed outcome mapping is held
    # in the test below.
    _, s = _cdd_drawn_splits_summary()
    er, eg = s["random"]["env"], s["group"]["env"]
    assert math.isclose(er, 0.885, abs_tol=1e-3), (
        f"sec:res-cdd claims env(random) '0.885' but data shows {er:.4f}"
    )
    assert math.isclose(eg, 0.653, abs_tol=1e-3), (
        f"sec:res-cdd claims env(group) '0.653' but data shows {eg:.4f}"
    )


def test_cdd_drawn_splits_timestamp_values_on_the_splits_we_draw():
    # tab:cdd's timestamp row, "0.899 -> 0.595", on the two splits this paper draws. The
    # comparison against the study's own reported figures is not read off these:
    # they are scored on folds we draw, so they and the study's numbers are measured
    # over different test images. test_their_split_timestamp_matches_their_reported_fusion
    # makes that comparison on the study's own identifiers instead.
    _, s = _cdd_drawn_splits_summary()
    cr, cg = s["random"]["timestamp"], s["group"]["timestamp"]
    assert math.isclose(cr, 0.899, abs_tol=1e-3), (
        f"sec:res-cdd claims timestamp(random) '0.899' but data shows {cr:.4f}"
    )
    assert math.isclose(cg, 0.595, abs_tol=1e-3), (
        f"sec:res-cdd claims timestamp(group) '0.595' but data shows {cg:.4f}"
    )


def test_cdd_detects_no_timestamp_direction():
    # sec:diagnostic's per-crop band is defined over the AI Hub dataset only. On CDD the
    # timestamp is above the environment series under one split and below it under the other;
    # neither paired margin separates, so sec:res-within says that the timestamp baseline
    # detects no difference, which is what the data supports; an interval that includes zero
    # shows no equivalence.
    _, s1 = _cdd_drawn_splits_summary()
    for regime in ("random", "group"):
        lo, hi = s1[regime]["timestamp_over_env_lo"], s1[regime]["timestamp_over_env_hi"]
        assert lo < 0 < hi, (
            f"sec:res-within claims that 'In macro-F1, the timestamp and the environment "
            f"classifiers do not differ on either split', but under the "
            f"{regime} split the paired timestamp-minus-environment "
            f"margin is {s1[regime]['timestamp_over_env']:+.4f} on [{lo:+.4f}, {hi:+.4f}], "
            f"which excludes zero"
        )


def test_cdd_drawn_splits_group_correction_removes_the_leak():
    # sec:res-cdd: "A random split that we draw leaks 95.9%, and a group-disjoint one none".
    # Under the random split 95.9% of test
    # rows carry a train env window (on val_md5); the group-disjoint correction drives it to
    # exactly zero — the 0.0 reference dacon_audited_split_leakage.py prints.
    _, s = _cdd_drawn_splits_summary()
    lr, lg = s["random"]["leak"], s["group"]["leak"]
    assert math.isclose(lr, 0.959, abs_tol=1e-3), (
        f"sec:res-cdd claims val_md5 leak(random) '0.959' but data shows {lr:.4f}"
    )
    assert lg == 0.0, (
        f"sec:res-cdd claims the group-disjoint correction removes the leak entirely "
        f"(leak(group) '0.000', the group-disjoint reference) but data shows {lg:.6f}"
    )


def test_cdd_drawn_splits_verdict_is_lookup_confirmed():
    # dacon_published_split.py's outcome mapping (P1/P2/P3) resolves to P1 iff env clears
    # base by >= COLLAPSE (else P3), does NOT transfer (env_g < env_r - TOL, else P2), and
    # loses more than COLLAPSE (env_r - env_g > COLLAPSE). Recomputed from the means,
    # guarding the branch INEQUALITIES rather than the 'P1' token.
    _, s = _cdd_drawn_splits_summary()
    er, eg, br = s["random"]["env"], s["group"]["env"], s["random"]["base"]
    assert er - br >= CDD_COLLAPSE, (
        f"dacon_published_split.py's outcome is P1 but env clears the majority-class base by only {er - br:+.4f} "
        f"(< {CDD_COLLAPSE}) — that is P3 (no effect to explain), not P1"
    )
    assert eg < er - CDD_TOL, (
        f"dacon_published_split.py's outcome is P1 but env(group)={eg:.4f} >= env(random) - TOL = "
        f"{er - CDD_TOL:.4f} — that is P2 (env transfers), under which the leakage "
        f"argument does not carry"
    )
    assert er - eg > CDD_COLLAPSE, (
        f"dacon_published_split.py's outcome is P1 (lookup), needing env_r - env_g > {CDD_COLLAPSE}, "
        f"but the drop is only {er - eg:+.4f}"
    )


# ---------------------------------------------------------------------------
# CDD, the study's own split — the same four classifiers scored on the identifiers the
# study committed, so one of our macro-F1s and one of theirs are measured over the
# same test images. dacon_their_split.py reads paper_work_seed777, the set
# src/main.py hardcodes for train, val and test alike, and trains on train+val
# because nothing is tuned. No image is opened, and no model of the study's is run.
# Their split is fixed, so these intervals carry estimator variance
# alone, which is why the comparison below asks whether their figure falls inside
# one of ours rather than which of the two is larger.
# ---------------------------------------------------------------------------
def _their_split_summary(metric="f1"):
    assert CDD_STUDY_SPLIT_CSV.exists(), (
        f"per-seed data on the study's splits missing at {CDD_STUDY_SPLIT_CSV}; regenerate via "
        f"dacon_their_split.py, which reads the redistribution-prohibited dataset and "
        f"the CDD study's committed split files from $TROPICAL_SCRATCH and "
        f"writes this path"
    )
    df = pd.read_csv(CDD_STUDY_SPLIT_CSV)
    tcrit = stats.t.ppf(0.975, N_SEEDS - 1)

    def ci(x):
        m, sd = x.mean(), x.std(ddof=1)
        hw = tcrit * sd / math.sqrt(len(x))
        return m, m - hw, m + hw

    out = {}
    for split, g in df.groupby("split"):
        r = dict(n_seeds=len(g), leak=g["group_leak"].mean(),
                 n_train=int(g["n_train"].iloc[0]), n_test=int(g["n_test"].iloc[0]))
        for arm in ["base", "timestamp", "env", "timestamp_env"]:
            r[arm], r[arm + "_lo"], r[arm + "_hi"] = ci(g[f"{arm}_{metric}"].to_numpy())
        # the classifiers share the one fixed fold within a seed, so a contrast between them
        # is paired; the classifier intervals above cannot settle one.
        for a, b_ in [("timestamp", "timestamp_env"), ("timestamp", "env"), ("timestamp_env", "env")]:
            d = g[f"{a}_{metric}"].to_numpy() - g[f"{b_}_{metric}"].to_numpy()
            k = f"{a}_over_{b_}"
            r[k], r[k + "_lo"], r[k + "_hi"] = ci(d)
        out[split] = r
    return df, out


def test_their_split_is_the_set_their_entry_point_reads():
    # sec:reanalysis: "The study's split is the set that the study's training script
    # reads." The classifiers
    # and the leak paragraph must therefore be scored on the same identifiers. Derive
    # that from the two artifacts rather than restating it: the run's own training size
    # is seed777's train+val, and the leak it recomputed is seed777's train+val breach.
    # The same sizes hold sec:reanalysis's "they train on the study's training and
    # validation partitions together".
    df, ts = _their_split_summary()
    assert THEIR_SPLIT_TAG in ts, (
        f"sec:res-cdd reports the classifiers on {THEIR_SPLIT_TAG} but the run covers {sorted(ts)}"
    )
    t = ts[THEIR_SPLIT_TAG]
    _, val = _cdd_study_split_summary()
    r = val[THEIR_SPLIT_TAG]
    assert (t["n_train"], t["n_test"]) == (r.n_train + r.n_val, r.n_test), (
        f"sec:reanalysis says the random forests 'train on the study's training and "
        f"validation partitions together', but the classifiers train on {t['n_train']} and "
        f"test on {t['n_test']} while {THEIR_SPLIT_TAG} commits {r.n_train}+{r.n_val} "
        f"train+val and {r.n_test} test; the classifiers and sec:res-cdd's leak paragraph are "
        f"reading different splits"
    )
    assert math.isclose(t["leak"], r.breach_trval, abs_tol=1e-3), (
        f"the classifiers' own series leak is {t['leak']:.4f} but the leakage table gives "
        f"{r.breach_trval:.4f} for the same split against train+val"
    )
    assert t["n_seeds"] == N_SEEDS, (
        f"sec:reanalysis says {N_SEEDS} seeds carry the random forest's variance on the "
        f"study's fixed split but the run has {t['n_seeds']}"
    )


def test_the_timestamp_lies_above_the_reported_fusion_score_on_the_other_set():
    # sec:res-cdd: "On the repository's other set of identifiers, the timestamp classifier also
    # reaches the study's fusion score": its whole 95% interval over the forest seeds.
    df = pd.read_csv(CDD_STUDY_SPLIT_CSV)
    x = df[df["split"] == "paper_work_seed42"]["timestamp_f1"]
    _, lo, _ = _seed_ci(x)
    reported = float(_params_defined()["paperMultiFone"])
    assert lo > reported, (
        f"sec:res-cdd claims that on the other set 'the timestamp classifier also reaches the "
        f"study's fusion score', but its interval reaches down to {lo:.4f}, below the study's "
        f"{reported}"
    )


def test_their_split_timestamp_matches_their_reported_fusion():
    # sec:res-cdd: "The timestamp classifier alone, however, matches that fusion score",
    # the study's 0.9037 for its fusion model, which sec:res-cdd's heading, the abstract, the contribution list
    # and the Conclusion also claim. sec:reanalysis defines "matches": the timestamp
    # classifier's 95% confidence interval contains the reported figure.
    _, ts = _their_split_summary()
    t = ts[THEIR_SPLIT_TAG]
    df = pd.read_csv(CDD_STUDY_SPLIT_CSV)
    _, lo, hi = _seed_ci(df[df["split"] == THEIR_SPLIT_TAG]["timestamp_f1"])
    assert lo <= PAPER_MULTIMODAL_F1 <= hi, (
        f"sec:res-cdd claims that the timestamp classifier matches the study's "
        f"{PAPER_MULTIMODAL_F1} for its fusion model on the study's own split, but its "
        f"interval is [{lo:.4f}, {hi:.4f}]"
    )
    assert t["timestamp_lo"] > PAPER_IMAGE_F1, (
        f"fig:cdd draws the timestamp classifier's interval on the study's split to the right of "
        f"the line for the study's image model ({PAPER_IMAGE_F1}), but the interval reaches "
        f"down to {t['timestamp_lo']:.4f}"
    )


def test_their_split_criterion_outcome():
    # The outcome mapping of dacon_their_split.py (C1/C2/C3) reads the TIMESTAMP-AND-ENVIRONMENT
    # classifier against the study's two reported figures. It falls on C2, and sec:res-cdd
    # reports the comparison that the mapping decided: "The timestamp and the environment
    # features together reach 0.8851, above the 0.8779 that the study reports for its image
    # model but below its fusion score".
    _, ts = _their_split_summary()
    ce = ts[THEIR_SPLIT_TAG]["timestamp_env"]
    assert PAPER_IMAGE_F1 <= ce < PAPER_MULTIMODAL_F1, (
        f"dacon_their_split.py's criterion falls on C2 — the environment features and the timestamp "
        f"together reach the study's image branch ({PAPER_IMAGE_F1}) and not its fusion "
        f"model ({PAPER_MULTIMODAL_F1}) — but timestamp+env is {ce:.4f}, which is outcome "
        f"{'C1' if ce >= PAPER_MULTIMODAL_F1 else 'C3'}; sec:res-cdd reports the wrong one"
    )


def test_their_split_panel_adds_nothing_to_the_timestamp():
    # sec:res-cdd: "so adding the environment features to the timestamp lowers the
    # forest's macro-F1 on this split", and the supplement's table of the other set
    # prints the same paired difference there. A direction on each set, so it is read
    # off the paired margin and not off the two classifiers' own intervals. Every set
    # that the run scored is checked, not the two named here.
    _, ts = _their_split_summary()
    assert len(ts) == 2 and THEIR_SPLIT_TAG in ts, (
        f"sec:res-cdd names the study's split and one other set, but the run scored {sorted(ts)}"
    )
    for tag, t in sorted(ts.items()):
        assert t["timestamp_over_timestamp_env_lo"] > 0, (
            f"sec:res-cdd claims the timestamp alone exceeds the timestamp and environment "
            f"features together on {tag}, but the paired margin is "
            f"{t['timestamp_over_timestamp_env']:+.4f} on [{t['timestamp_over_timestamp_env_lo']:+.4f}, "
            f"{t['timestamp_over_timestamp_env_hi']:+.4f}], which includes zero"
        )


CDD_MAXFEATURES_CSV = CDD / "dacon_their_split_tuned.csv"   # dacon_their_split_tuned.py


def test_the_timestamp_stays_higher_at_every_max_features():
    # sec:res-cdd: "The timestamp alone also scores higher than the timestamp and the
    # environment features together when the number of variables per node is chosen on the
    # validation partition, and when a forest considers every variable at every node". Paired
    # within seed; tab:maxfeatures prints both.
    df = pd.read_csv(CDD_MAXFEATURES_CSV)
    seeds = sorted(set(df["seed"]))
    assert seeds == _script_const("code/dacon_cropdisease/dacon_their_split.py", "SEEDS"), (
        f"tab:maxfeatures prints means over the study-split forests' seeds, but the grid run "
        f"holds seeds {seeds}"
    )
    assert (df.groupby(["seed", "arm"])["selected"].sum() == 1).all(), (
        "tab:maxfeatures prints one selected max_features per forest and seed, but the grid "
        "run marks another number"
    )

    def col(arm, rows):
        return df[(df["arm"] == arm) & rows].sort_values("seed")["test_f1"].to_numpy()

    for label, rows in (("chosen on the validation partition", df["selected"]),
                        ("every variable at every split", df["max_features"] == "1.0")):
        _, lo, hi = _seed_ci(col("timestamp", rows) - col("timestamp_env", rows))
        assert lo > 0, (
            f"sec:res-cdd says 'The timestamp alone also scores higher than the timestamp and the "
            f"environment features together when the number of variables per node is chosen on "
            f"the validation partition', and when a forest considers every variable, but with "
            f"{label} the paired difference "
            f"is [{lo:+.4f}, {hi:+.4f}]"
        )
    # The default row of the grid reproduces dacon_their_split.py's forests, seed by seed, so
    # the grid run scores the forests that tab:cdd-theirs prints and not other ones.
    theirs = pd.read_csv(CDD_STUDY_SPLIT_CSV)
    theirs = theirs[theirs["split"] == THEIR_SPLIT_TAG].sort_values("seed")
    for arm in ("timestamp", "env", "timestamp_env"):
        grid = col(arm, df["max_features"] == "sqrt")
        assert (abs(grid - theirs[f"{arm}_f1"].to_numpy()) < 1e-12).all(), (
            f"tab:maxfeatures' square-root column should reproduce tab:cdd-theirs' {arm} "
            f"forest seed by seed, but it does not"
        )


# ---------------------------------------------------------------------------
# The AI Hub fusion experiment at the reduced budget (40 epochs; paper/main.tex sec:fusion):
#   2 models (image-only `img`, late-logit fusion `all`) x 2 splits (iid/date) x 5
#   seeds {0..4}. At 40 epochs the in-distribution gain's CI includes 0 (tab:fusion-reduced;
#   sec:fusion-design reports that the CDD study's 300-epoch training was added after this
#   run). At 300 epochs the in-distribution gain's CI excludes 0, NOT because the longer
#   budget trains the sensor branch further: both models score LOWER at 300 epochs than here, in
#   every seed (see _budget_change below), and the gain widens only because the image-only model
#   loses more (sec:fusion).
#   These tests guard the MARGIN on the loaded per-seed CSV (paired 95% t-interval, mirroring
#   h4_fusion_collapse.py `ci()`), not a>b on one run. Imagery-free: the CSV carries
#   F1/BA/epoch counts per (split, seed, model), no image ids.
# ---------------------------------------------------------------------------
FUSION_REDUCED_CSV = AIHUB / "h4_fusion_collapse.csv"           # committed per-seed results
FUSION_COVERAGE_CSV = AIHUB / "h4_positivity_recheck.csv"    # committed coverage check
FUSION_REDUCED_SEEDS = {0, 1, 2, 3, 4}       # the fusion experiment's seed set
FUSION_REDUCED_ARMS = {"img", "all"}         # image-only model, late-logit fusion model
FUSION_REDUCED_SPLITS = {"iid", "date"}
# Coverage conditions (sec:fusion-design; fold-0 thresholds, h4_materialize_splits.py).
# These thresholds measure coverage: does every class and cell appear on BOTH sides of the
# partition, so the split can be scored. Whether an environment regime admits more than one
# disease class, so that a within-regime contrast exists at all, is a property of the
# dataset, not of a split, and no check here measures it for the AI Hub dataset. Reading
# cls_pos == 1.0 below as that property is the mistake the column name invites.

FUSION_IID_LEAK_MIN = 0.50           # the in-distribution split must RETAIN the panel leak rate
FUSION_DATE_LEAK_MAX = 0.10          # date-disjoint must REMOVE it
FUSION_MIN_CLASS_COVERAGE = 0.99            # class coverage on the two splits with a criterion (iid, date)
FUSION_MIN_CELL_COVERAGE = 0.80           # cell coverage on the same two splits


def _paired_ci(x):
    """mean, lo, hi of a paired 95% t-interval — the method is verbatim from
    h4_fusion_collapse.py `ci()` (the test must reproduce the script's number, so the
    interval is computed identically; pandas std defaults to ddof=1, matching np.std(ddof=1))."""
    x = pd.Series(x, dtype=float)
    m, sd = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, len(x) - 1) * sd / (len(x) ** 0.5)
    return m, m - h, m + h


def _fusion_reduced_summary(metric="test_f1"):
    assert FUSION_REDUCED_CSV.exists(), (
        f"reduced-budget fusion results missing at {FUSION_REDUCED_CSV}; produced by "
        f"code/aihub_subtropical/h4_fusion_collapse.py run, which writes this path (imagery-free)"
    )
    long = pd.read_csv(FUSION_REDUCED_CSV)
    w = long.pivot_table(index=["split", "seed"], columns="arm", values=metric).reset_index()
    w["delta"] = w["all"] - w["img"]           # paired within (split, seed): fusion - image-only
    summ = {}
    for s in FUSION_REDUCED_SPLITS:
        g = w[w.split == s]
        md, lo, hi = _paired_ci(g["delta"].values)
        summ[s] = dict(d=md, lo=lo, hi=hi, img=g["img"].mean(), all=g["all"].mean(), n=len(g))
    return long, summ


def _fusion_coverage():
    assert FUSION_COVERAGE_CSV.exists(), (
        f"coverage CSV missing at {FUSION_COVERAGE_CSV}; produced by "
        f"code/aihub_subtropical/h4_materialize_splits.py, which writes this path"
    )
    return pd.read_csv(FUSION_COVERAGE_CSV).set_index("axis")


def test_the_gate_and_the_trained_fold_stay_two_populations():
    # The paper prints the same quantity over two populations: the candidate split by photo
    # date leaks 7.2% on the whole spreadsheet (sec:fusion-design) and the realized
    # date-disjoint fold 7.3% on the subsample (sec:fusion). A reader who takes them for one
    # quantity reads a typo, so each sentence names its population, and this derives both
    # from their own artifacts so the distinction cannot be "fixed" away.
    gate = pd.read_csv(AIHUB / "leakage_collapse_gate.csv").set_index("axis")
    panel = json.loads((AIHUB / "corpus_structure.json").read_text(encoding="utf-8"))["images"]
    assert int(gate.loc["random", "n_groups"]) == panel, (
        f"tab:candidates compares candidate splits of the whole metadata spreadsheet, but "
        f"leakage_collapse_gate.csv's random axis covers "
        f"{int(gate.loc['random', 'n_groups'])} groups against the panel's {panel}"
    )
    sub = pd.read_csv(FUSION_COVERAGE_CSV)["subsample_n"].unique()
    assert len(sub) == 1 and int(sub[0]) < panel, (
        f"sec:fusion reports the realized folds' leak 'On the 40,000-image subsample', apart "
        f"from sec:fusion-design's whole-table "
        f"figure, but the coverage CSV declares subsample sizes {sub} against a panel of {panel}"
    )


def test_fusion_reduced_budget_run_is_complete():
    # The run is 2 models x 2 splits x 5 seeds = 20 trainings; a partial CSV would silently bias
    # the CIs, so pin the shape before any margin.
    long, _ = _fusion_reduced_summary()
    assert set(long["seed"]) == FUSION_REDUCED_SEEDS, (
        f"sec:fusion-design says 'The image-only and fusion models share the ResNet-50, the "
        f"training procedure, and the random seeds', 5 seeds by tab:settings, but the results CSV has "
        f"seeds {sorted(set(long['seed']))}"
    )
    assert set(long["arm"]) == FUSION_REDUCED_ARMS and set(long["split"]) == FUSION_REDUCED_SPLITS, (
        f"the reduced-budget run should cover models {FUSION_REDUCED_ARMS} x splits {FUSION_REDUCED_SPLITS} but has "
        f"models {set(long['arm'])} x splits {set(long['split'])}"
    )
    # Derived from the three sets above rather than written as 30, so the cell count
    # and the decomposition the message quotes cannot drift apart.
    cells = len(FUSION_REDUCED_ARMS) * len(FUSION_REDUCED_SPLITS) * len(FUSION_REDUCED_SEEDS)
    assert len(long) == cells, (
        f"the reduced-budget run is {cells} trainings (models x splits x seeds), "
        f"but the CSV has {len(long)} rows"
    )


def test_fusion_reduced_budget_coverage_conditions_hold():
    # The date-disjoint comparison is only valid if the in-distribution split RETAINS the panel
    # leak rate while date-disjoint REMOVES it, both keeping coverage (sec:fusion-design).
    p = _fusion_coverage()
    assert p.loc["iid", "panel_leak_fold0"] >= FUSION_IID_LEAK_MIN, (
        f"Manuscript (sec:fusion) rests on the in-distribution split retaining the panel leak rate (>= {FUSION_IID_LEAK_MIN}) "
        f"but data shows in-distribution leak rate {p.loc['iid', 'panel_leak_fold0']:.4f}"
    )
    assert p.loc["date", "panel_leak_fold0"] <= FUSION_DATE_LEAK_MAX, (
        f"Manuscript (sec:fusion-design) says that 'holding out photo dates also keeps the "
        f"readings of most test images out of training' (<= {FUSION_DATE_LEAK_MAX}) but data shows date "
        f"leak {p.loc['date', 'panel_leak_fold0']:.4f}"
    )
    # sec:intro states the class half as a universal, that the date-disjoint split
    # 'keeps every class in both the training and the test data', which the floor below
    # would pass at 0.995; the universal needs every class.
    assert p.loc["date", "cls_pos_fold0"] == 1.0, (
        f"sec:intro claims that the split by photo date 'keeps every class in both the "
        f"training and the test data' but the date split's class coverage is "
        f"{p.loc['date', 'cls_pos_fold0']:.4f}"
    )
    for ax in ("iid", "date"):
        ok = (p.loc[ax, "cls_pos_fold0"] >= FUSION_MIN_CLASS_COVERAGE
              and p.loc[ax, "cell_pos_fold0"] >= FUSION_MIN_CELL_COVERAGE)
        assert ok, (
            f"sec:intro says the split by photo date 'keeps every class in both the training and "
            f"the test data' — the property that makes the date-disjoint test runnable on this "
            f"dataset and not on CDD — and treats the {ax} split as keeping coverage "
            f"(cls>={FUSION_MIN_CLASS_COVERAGE}, cell>={FUSION_MIN_CELL_COVERAGE}); data shows "
            f"cls_pos {p.loc[ax, 'cls_pos_fold0']:.4f}, cell_pos {p.loc[ax, 'cell_pos_fold0']:.4f}"
        )


def test_fusion_reduced_budget_shows_no_in_distribution_gain():
    # sec:fusion-design: "Training first ran for a reduced budget of 40 epochs, which showed no
    # macro-F1 gain of the fusion model over the image-only model on the in-distribution split".
    # The paired 95% CI of the reduced-budget in-distribution gain in macro-F1 must include zero
    # (tab:fusion-reduced prints it). The sentence names macro-F1 because the balanced-accuracy
    # gain at that budget excludes zero.
    _, summ = _fusion_reduced_summary()
    lo, hi = summ["iid"]["lo"], summ["iid"]["hi"]
    assert lo < 0.0 < hi, (
        f"Manuscript (sec:fusion-design) says the reduced-budget training 'showed no macro-F1 "
        f"gain of the fusion model over the image-only model on the in-distribution split' but the CI is [{lo:+.4f}, {hi:+.4f}], which excludes 0"
    )


def test_fusion_reduced_budget_reported_deltas_match():
    # The two per-split reduced-budget deltas that tab:fusion prints, each with its paired 95%
    # CI, all including 0 at that budget (tab:fusion-reduced). At the CDD study's 300 epochs the
    # in-distribution gain lies above zero, and the date-disjoint run is repeated there too.
    _, summ = _fusion_reduced_summary()
    for split, d, lo, hi in [
        ("iid",  +0.0013, -0.0058, +0.0084),
        ("date", +0.0082, -0.0091, +0.0255),
    ]:
        s = summ[split]
        for got, claimed, name in [(s["d"], d, "mean"), (s["lo"], lo, "CI lo"), (s["hi"], hi, "CI hi")]:
            assert math.isclose(got, claimed, abs_tol=5e-4), (
                f"tab:fusion reports the reduced-budget {split} delta {name} {claimed:+.4f} but data shows {got:+.4f}"
            )
        assert s["lo"] < 0.0 < s["hi"], (
            f"tab:fusion shows all three reduced-budget intervals including 0 but "
            f"{split} CI [{s['lo']:+.4f}, {s['hi']:+.4f}] does not"
        )


def test_fusion_reduced_budget_image_baseline_is_strong():
    # tab:fusion prints the reduced-budget image-only macro-F1, 0.823: the reduced-budget gain whose interval includes zero
    # is not a broken pipeline; the image-only model reaches 0.823 in-distribution macro-F1, far
    # above 18-class chance, so `all` had a real bar to clear. The matched-budget level is a
    # separate macro (\imageOnlyFoneMatched, bound off the matched-budget CSV): the two budgets
    # must not be quoted for each other.
    _, summ = _fusion_reduced_summary()
    assert math.isclose(summ["iid"]["img"], 0.823, abs_tol=2e-3), (
        f"tab:fusion prints the reduced-budget image-only macro-F1 as 0.823 "
        f"but data shows in-distribution image-only {summ['iid']['img']:.4f}"
    )


def test_fusion_reduced_budget_iid_leak_is_near_total():
    # sec:fusion: a panel tuple that also occurs in training "is carried by 89.8% of the test
    # images of the in-distribution split", which is what a
    # session/timestamp shortcut would exploit. At the reduced budget the fusion model extracted
    # no gain from it; at the matched budget it does, and the date-disjoint run tests whether that
    # gain IS the leak rate. The 89.8% value itself is what this guards.
    p = _fusion_coverage()
    assert math.isclose(p.loc["iid", "panel_leak_fold0"], 0.898, abs_tol=1e-3), (
        f"Manuscript (sec:fusion) says a recurring panel tuple 'is carried by 89.8% of the test "
        f"images of the in-distribution split' but data "
        f"shows {p.loc['iid', 'panel_leak_fold0']:.4f}"
    )


# ---------------------------------------------------------------------------
# The AI Hub fusion experiment at the CDD study's matched training budget: 300 epochs,
#   batch 8, no early stopping (the CDD study's src/main.py:24 and
#   script/train_single_modal.sh:3), on the reduced-budget run's in-distribution split,
#   40,000-image subsample and seeds {0..4}. At the reduced budget the in-distribution
#   gain's interval included zero, and 40 epochs may undertrain the CDD study's
#   environmental branch. At the matched budget the gain's interval excludes zero
#   (sec:fusion), and the gain widens because both models score lower than at 40 epochs
#   in every seed and the image-only model loses more (_budget_change). Guard
#   the margin on the per-seed CSV (paired 95% t-interval, the reduced-budget run's
#   ci()), not a>b on one run. Imagery-free: F1/BA/epoch per (seed, model), no image ids.
# ---------------------------------------------------------------------------
FUSION_MATCHED_CSV = AIHUB / "h4_fusion_matched_budget.csv"   # committed per-seed results (300 epochs, batch 8)
FUSION_MATCHED_SEEDS = {0, 1, 2, 3, 4}       # same seeds as the reduced budget
FUSION_MATCHED_ARMS = {"img", "all"}         # image-only model, late-logit fusion model
FUSION_MATCHED_EPOCHS = 300                  # CDD src/main.py:24 (vs the reduced budget's 40) — the only budget change


def _fusion_matched_summary(metric="test_f1"):
    assert FUSION_MATCHED_CSV.exists(), (
        f"matched-budget fusion results missing at {FUSION_MATCHED_CSV}; produced by "
        f"code/aihub_subtropical/h4_fusion_matched_budget.py run, which writes this path (imagery-free)"
    )
    long = pd.read_csv(FUSION_MATCHED_CSV)
    w = long.pivot_table(index=["split", "seed"], columns="arm", values=metric).reset_index()
    w["delta"] = w["all"] - w["img"]                 # paired within seed: fusion - image-only
    g = w[w.split == "iid"]
    md, lo, hi = _paired_ci(g["delta"].values)       # identical paired-CI method as the reduced budget
    return long, dict(d=md, lo=lo, hi=hi, img=g["img"].mean(), all=g["all"].mean(), n=len(g))


def _budget_change():
    """{model: paired 40 -> 300 epoch change in in-distribution macro-F1}, within seed.

    The matched-budget run re-runs the reduced budget's own in-distribution split,
    subsample and seeds, so the two CSVs pair seed-for-seed. Each run's delta is a
    difference WITHIN one budget; this is the difference ACROSS them, which
    tab:fusion-reduced prints.
    """
    a = pd.read_csv(FUSION_REDUCED_CSV)
    b = pd.read_csv(FUSION_MATCHED_CSV)
    aw = a[a.split == "iid"].pivot_table(index="seed", columns="arm", values="test_f1")
    bw = b.pivot_table(index="seed", columns="arm", values="test_f1")
    out = {}
    for arm in sorted(FUSION_MATCHED_ARMS):
        paired = (bw[arm] - aw[arm]).values
        md, lo, hi = _paired_ci(paired)          # same paired CI as both runs report
        out[arm] = dict(md=md, lo=lo, hi=hi, worse=int((paired < 0).sum()), n=len(paired))
    # the two models' changes, differenced within seed: the gain's widening
    out["img_minus_all"] = _paired_ci(((bw["img"] - aw["img"]) - (bw["all"] - aw["all"])).values)
    return out


def test_fusion_matched_budget_arm_levels_at_matched_budget():
    # The matched-budget gain could be read as the environmental branch finally training.
    # The replication verdict (delta's CI excludes 0) does not say why; the model levels do.
    # Both models score LOWER at 300 epochs than at 40, in every seed, with CIs excluding
    # zero, and delta widens only because the image-only model falls further. The
    # criterion asks only about delta, so guard the model levels that the prose reports.
    ch = _budget_change()
    for arm in ("img", "all"):
        c = ch[arm]
        assert c["hi"] < 0.0 and c["worse"] == c["n"], (
            f"Manuscript (sec:fusion) says the image-only model's change from the reduced-budget "
            f"training 'is -0.0141, and the fusion model's is -0.0071', both lower in every seed, "
            f"but the {arm} model moves {c['md']:+.4f} "
            f"[{c['lo']:+.4f}, {c['hi']:+.4f}] with {c['worse']}/{c['n']} seeds lower"
        )
    md, lo, hi = ch["img_minus_all"]
    assert hi < 0.0, (
        f"Manuscript (sec:fusion) says 'the in-distribution gain comes from a fall of the "
        f"image-only model's score rather than from a rise of the fusion model's', a paired difference whose interval must exclude zero, but the "
        f"image-only change "
        f"minus the fusion change is {md:+.4f} "
        f"[{lo:+.4f}, {hi:+.4f}]"
    )


def test_fusion_matched_budget_run_is_iid_only_and_complete():
    # The matched-budget run is in-distribution ONLY (2 models x 5 seeds = 10 trainings); the
    # date-disjoint run is h4_g4b_collapse.py's, with its own CSV. A partial CSV would bias the
    # CI; a non-iid split would mix that run's rows into this table.
    long, _ = _fusion_matched_summary()
    assert set(long["seed"]) == FUSION_MATCHED_SEEDS, (
        f"the matched-budget run uses seeds {{0-4}} but the CSV has "
        f"{sorted(set(long['seed']))}"
    )
    assert set(long["arm"]) == FUSION_MATCHED_ARMS, (
        f"the matched-budget run is 2 models (img, all) but the CSV has {set(long['arm'])}"
    )
    assert set(long["split"]) == {"iid"}, (
        f"the matched-budget run is in-distribution only (date: h4_g4b_collapse.csv) but "
        f"the CSV has splits {set(long['split'])}"
    )
    cells = len(FUSION_MATCHED_ARMS) * len(FUSION_MATCHED_SEEDS)
    assert len(long) == cells, (
        f"the matched-budget run is 2 models x 5 seeds = {cells} trainings "
        f"but the CSV has {len(long)} rows, not {cells}"
    )


def test_fusion_matched_budget_matched_budget_was_actually_exercised():
    # The one thing that changed from the reduced budget is the budget: 300 epochs (CDD
    # src/main.py:24) vs 40. At least one model must select a best checkpoint past epoch 39, else
    # the 300-epoch budget was never
    # exercised and the run would not have tested the undertraining objection it exists to close.
    long, _ = _fusion_matched_summary()
    assert long["best_epoch"].max() > 39, (
        f"the matched budget is {FUSION_MATCHED_EPOCHS} epochs (vs the reduced budget's 40) to test the "
        f"undertraining objection, but no model's best checkpoint is past epoch 39 (max best_epoch "
        f"{int(long['best_epoch'].max())}) — the matched budget was not exercised"
    )


def test_fusion_matched_budget_iid_gain_lies_above_zero():
    # sec:fusion: "the fusion model raises macro-F1 relative to the image-only model by
    # δ = +0.0084" (tab:fusion: [+0.0028, +0.0140]). At 40 epochs the CI includes 0; at the CDD
    # study's 300 epochs it excludes 0. Guard the margin: CI excludes 0.
    _, s = _fusion_matched_summary()
    for got, claimed, name in [(s["d"], +0.0084, "mean"), (s["lo"], +0.0028, "CI lo"), (s["hi"], +0.0140, "CI hi")]:
        assert math.isclose(got, claimed, abs_tol=5e-4), (
            f"tab:fusion reports the matched-budget in-distribution Δ {name} {claimed:+.4f} but data shows {got:+.4f}"
        )
    assert s["lo"] > 0.0, (
        f"Manuscript (sec:fusion) says 'the fusion model raises macro-F1 relative to the "
        f"image-only model', a gain whose paired 95% CI excludes 0, but the CI is "
        f"[{s['lo']:+.4f}, {s['hi']:+.4f}], "
        f"which includes 0"
    )


# ---------------------------------------------------------------------------
# The AI Hub fusion experiment on the date-disjoint split at the matched budget (300 epochs).
#   The matched-budget in-distribution gain lies above zero (delta_iid +0.0084, CI excludes 0).
#   This run measures that gain on a date-disjoint split at the SAME budget, at n=5, paired to
#   the committed in-distribution gain:
#     delta(date) = macroF1(all;date) - macroF1(img;date), paired within seed
#     drop(seed)  = delta_iid(seed) - delta(date)(seed)  [delta_iid READ from the matched-budget CSV, NOT re-run]
#   Measured: delta(date) = +0.0037 [-0.0041, +0.0115] and drop = +0.0046 [-0.0010, +0.0103], both
#   CIs including 0 in macro-F1, so the paper reports that the gain did not detectably shrink
#   (sec:fusion). The CSV is date-only. Guard the MARGIN on the
#   loaded per-seed CSVs (paired
#   95% t-interval, same ci() as the other two runs), not a>b on one run.
#   Imagery-free: F1/BA/epoch per (split, seed, model), no image ids.
# ---------------------------------------------------------------------------
FUSION_DATE_CSV = AIHUB / "h4_g4b_collapse.csv"   # committed per-seed date-disjoint results (300 epochs, batch 8)
FUSION_DATE_SEEDS = {0, 1, 2, 3, 4}     # paired to the in-distribution gain
FUSION_DATE_ARMS = {"img", "all"}       # image-only model, late-logit fusion model


def _fusion_date_summary(metric="test_f1"):
    assert FUSION_DATE_CSV.exists(), (
        f"matched-budget date-disjoint results missing at {FUSION_DATE_CSV}; produced by "
        f"code/aihub_subtropical/h4_g4b_collapse.py date, which writes this path (imagery-free)"
    )
    raw = pd.read_csv(FUSION_DATE_CSV)
    date = raw[raw.split == "date"]
    dw = date.pivot_table(index="seed", columns="arm", values=metric)
    dw["delta"] = dw["all"] - dw["img"]            # delta(date), paired within seed
    # delta_iid READ from the committed matched-budget CSV (not re-run, not recomputed).
    iid = pd.read_csv(FUSION_MATCHED_CSV)
    iid = iid[iid.split == "iid"].pivot_table(index="seed", columns="arm", values=metric)
    iid["delta_iid"] = iid["all"] - iid["img"]
    j = dw.join(iid["delta_iid"], how="inner")     # pair delta(date) with delta_iid on the shared seed
    j["drop"] = j["delta_iid"] - j["delta"]        # within-seed reduction from the in-distribution gain
    d_m, d_lo, d_hi = _paired_ci(j["delta"].values)
    drop_m, drop_lo, drop_hi = _paired_ci(j["drop"].values)
    return raw, dw, j, dict(
        d=d_m, d_lo=d_lo, d_hi=d_hi,
        drop=drop_m, drop_lo=drop_lo, drop_hi=drop_hi, n=len(j),
        img=dw["img"].mean(),
    )


def test_fusion_date_disjoint_run_is_date_only_and_complete():
    # The matched-budget date run is date-disjoint, 2 models x 5 seeds = 10 trainings, paired to the
    # committed in-distribution gain. A partial CSV biases the CIs; a stray row of another
    # split would fold that split into the date outcome; a broken
    # pairing would break the drop.
    raw, _, j, _ = _fusion_date_summary()
    assert set(raw["split"]) == {"date"}, (
        f"the matched-budget date run is date-disjoint only "
        f"but the CSV has splits {set(raw['split'])}"
    )
    assert set(raw["seed"]) == FUSION_DATE_SEEDS, (
        f"the matched-budget date run uses seeds {sorted(FUSION_DATE_SEEDS)} but the CSV has "
        f"{sorted(set(raw['seed']))}"
    )
    assert set(raw["arm"]) == FUSION_DATE_ARMS, (
        f"the matched-budget date run is 2 models (img, all) but the CSV has {set(raw['arm'])}"
    )
    cells = len(FUSION_DATE_ARMS) * len(FUSION_DATE_SEEDS)
    assert len(raw) == cells, (
        f"the matched-budget date run is 2 models x 5 seeds = {cells} trainings "
        f"but the CSV has {len(raw)} rows, not {cells}"
    )
    assert len(j) == 5, (
        f"the paired reduction pairs delta(date) with the committed in-distribution delta on 5 shared seeds but the join has "
        f"{len(j)} — delta_iid and delta(date) are not paired over the seed set"
    )


def test_fusion_date_disjoint_date_gain_is_not_established():
    # tab:fusion: date-disjoint Δ(date) = +0.0037 [−0.0041, +0.0115], CI includes 0: the fusion
    # gain is not detectably above zero on the date-disjoint split at n=5.
    *_, s = _fusion_date_summary()
    for got, claimed, name in [(s["d"], +0.0037, "mean"), (s["d_lo"], -0.0041, "CI lo"), (s["d_hi"], +0.0115, "CI hi")]:
        assert math.isclose(got, claimed, abs_tol=5e-4), (
            f"tab:fusion reports the matched-budget date-disjoint Δ {name} {claimed:+.4f} but data shows {got:+.4f}"
        )
    assert s["d_lo"] < 0.0 < s["d_hi"], (
        f"sec:fusion says 'In macro-F1, however, the date-disjoint gain itself is also not "
        f"detectably above zero', but the CI is [{s['d_lo']:+.4f}, {s['d_hi']:+.4f}], which excludes 0"
    )


def test_fusion_date_disjoint_drop_does_not_clear_seed_noise():
    # tab:fusion: the paired reduction from the in-distribution gain is +0.0046 [−0.0010, +0.0103],
    # CI includes 0: the reduction does not clear seed noise at n=5. delta_iid is READ from the
    # committed matched-budget CSV, not recomputed to widen the drop.
    *_, s = _fusion_date_summary()
    for got, claimed, name in [(s["drop"], +0.0046, "mean"), (s["drop_lo"], -0.0010, "CI lo"), (s["drop_hi"], +0.0103, "CI hi")]:
        assert math.isclose(got, claimed, abs_tol=5e-4), (
            f"tab:fusion reports the paired reduction {name} {claimed:+.4f} but data shows {got:+.4f}"
        )
    assert s["drop_lo"] < 0.0 < s["drop_hi"], (
        f"sec:fusion says 'the paired reduction of +0.0046 is not detectably above zero' but the "
        f"CI is [{s['drop_lo']:+.4f}, {s['drop_hi']:+.4f}]"
    )


def test_the_gain_did_not_detectably_shrink():
    # sec:fusion's heading: "A small late-fusion gain did not detectably shrink on held-out
    # photo dates". The paired reduction's interval includes zero in both metrics, and the
    # in-distribution gain that it reduces lies above zero in macro-F1.
    for metric in ("test_f1", "test_ba"):
        *_, s = _fusion_date_summary(metric)
        assert s["drop_lo"] < 0.0 < s["drop_hi"], (
            f"sec:fusion says 'the date-disjoint split detects no loss of the gain on held-out dates', but in "
            f"{metric} the paired reduction's interval "
            f"[{s['drop_lo']:+.4f}, {s['drop_hi']:+.4f}] excludes zero"
        )
    _, s5 = _fusion_matched_summary()
    assert s5["lo"] > 0.0, (
        f"sec:fusion says 'the fusion model raises macro-F1 relative to the image-only model', "
        f"a gain whose interval lies above zero, but its interval is "
        f"[{s5['lo']:+.4f}, {s5['hi']:+.4f}]"
    )


# ===========================================================================
# The cited fusion studies (sec:intro, sec:reanalysis, sec:discussion)
#
# The manuscript prints no count over these studies. Its sentences about them
# are universals over the studies it cites ("None of the fusion studies we
# cite ..."), and each is re-derived here from code/survey/survey_records.csv
# through code/survey/survey_records.py, the reader that tallies the coded cells.
# ===========================================================================

SURVEY = HERE / "code" / "survey"


def _survey_records():
    sys.path.insert(0, str(SURVEY))
    try:
        import survey_records
    finally:
        sys.path.pop(0)
    return survey_records


def _survey_counts():
    records = _survey_records()
    rows = records.load()
    return rows, {
        stem: records.tally(rows, col, pos, kind)
        for stem, col, pos, kind in records.COLUMNS
    }


def test_survey_denominators_are_per_column():
    # The schema invariant the survey's wording rests on: a column's denominator
    # is derived from its own applicable set, so two columns may legitimately
    # disagree. Asserted here because one shared denominator would make a single
    # manuscript sentence count two populations.
    _, counts = _survey_counts()
    for stem, c in counts.items():
        assert c["denominator"] + c["excluded"] == c["total"], (
            f"{stem}: {c['denominator']} codable + {c['excluded']} excluded "
            f"!= {c['total']} rows"
        )
        assert c["numerator"] == len(c["positive_ids"]), f"{stem}: numerator vs id list"
    assert counts["Comparator"]["denominator"] == counts["SensorBaseline"]["numerator"], (
        "The comparator column is conditional on running a sensor-only baseline; "
        f"its denominator is {counts['Comparator']['denominator']} against "
        f"{counts['SensorBaseline']['numerator']} studies that ran one"
    )


def test_survey_practice_universals():
    rows, counts = _survey_counts()
    # sec:intro: "None of the fusion studies that we cite reports either of two checks":
    # "The first check compares how well the sensor readings predict the label with how
    # well a timestamp predicts it" and "The second evaluates a study's fusion model on a split that
    # keeps all images of each farm or each session in either the training or the test
    # data". The quantifier ranges over the studies that the paragraph's
    # first sentence cites, so the set is read off that \citep rather than off the CSV:
    # a study cited there and never coded would leave the universal unchecked, and
    # a coded study the sentence no longer cites would check a set it does not name.
    text = re.sub(r"(?<!\\)%.*", "", MAIN.read_text(encoding="utf-8"))
    m = re.search(r"whether its sensor readings identify the session\s*\\citep\{([^}]*)\}",
                  text)
    assert m, "sec:intro no longer cites the set its universal ranges over"
    cited = {k.strip() for k in m.group(1).split(",")}
    studies = [r for r in rows if r["kind"] == "study"]
    coded = {r["cite_key"] for r in studies}
    assert cited == coded, (
        f"sec:intro's universal ranges over {sorted(cited)}, but the coded studies "
        f"are {sorted(coded)}; code the missing ones or cite the coded ones"
    )
    codable = len([r for r in studies if r["access"] != "inaccessible"])
    c = counts["TimestampBaseline"]
    assert c["numerator"] == 0 and c["denominator"] == codable, (
        f"sec:intro claims that the cited studies apply neither of two checks, one of them "
        f"'compares how well the sensor readings predict the label with how well a "
        f"timestamp predicts it', but the coding has "
        f"{c['numerator']} of {c['denominator']} ({c['positive_ids']})"
    )
    c = counts["SessionDisjoint"]
    assert c["numerator"] == 0 and c["denominator"] == codable, (
        f"sec:intro claims that the cited studies apply neither of two checks, one of them "
        f"'evaluates a study's fusion model on a split that keeps all images of each farm or each "
        f"session in either the training or the test data' "
        f"but the coding has {c['numerator']} of {c['denominator']} "
        f"({c['positive_ids']})"
    )
    # sec:discussion ranges over the same studies. The comparator column is
    # conditional on a sensor-only baseline, so its denominator is the studies that
    # publish a panel-alone accuracy, which is the set the sentence speaks of.
    c = counts["Comparator"]
    assert c["numerator"] == 0, (
        f"sec:discussion claims 'The fusion studies that we cite and that publish a "
        f"sensor-only score ... set no majority-class score beside it' but the coding names "
        f"{c['positive_ids']}"
    )
    # The sentence names those studies by citation, so the cited set is the coded set.
    tex = " ".join((HERE / "paper" / "main.tex").read_text(encoding="utf-8").split())
    m = re.search(r"that publish a sensor-only score \\citep\{([^}]*)\}", tex)
    assert m, "sec:discussion no longer cites the studies that publish a sensor-only score"
    with open(HERE / "code" / "survey" / "survey_records.csv", encoding="utf-8") as f:
        coded = {r["cite_key"] for r in csv.DictReader(f) if r["sensor_baseline"] == "yes"}
    assert set(m.group(1).split(",")) == coded, (
        f"sec:discussion cites {sorted(m.group(1).split(','))} as the studies that publish "
        f"a sensor-only score, but the coding names {sorted(coded)}"
    )


def test_survey_the_counter_example_is_xiang():
    _, counts = _survey_counts()
    # sec:intro: "The split that comes nearest to the second check is that of Xiang et al.
    # (2026), which keeps all data of each plant or each infection event, one occurrence of
    # disease, in one of its training, validation, and test sets". The sentence names one
    # study, so the cell is pinned to that id rather than to a count -- a count of 1 would
    # still pass if the positive moved to another row.
    for stem, phrase in [
        ("SplitAboveSample",
         "keeps all data of each plant or each infection event"),
    ]:
        c = counts[stem]
        assert c["positive_ids"] == ["xiang2026"], (
            f"sec:intro names [partitioned2026] as the study that does "
            f"'{phrase}', but the coding names {c['positive_ids']}"
        )


def test_survey_cdd_is_the_only_reproducible_release():
    _, counts = _survey_counts()
    # Two manuscript sentences identify this single coded row, and each names it
    # by a different description: sec:intro calls it the study re-analyzed,
    # sec:discussion counts it as the one gain a re-analysis reaches. Assert the identity once and quote both, because
    # the row is the CDD study and not this paper's re-analysis of it, and a
    # sentence that swaps the two reads as true while naming the wrong object.
    assert counts["ReproducibleRelease"]["positive_ids"] == ["lee2024dacon"], (
        f"sec:intro claims 'the CDD study is the only fusion study that we cite whose "
        f"dataset and split were both released', and sec:discussion "
        f"claims 'A re-analysis can expose such a leak only where a study releases both "
        f"its dataset and its split, so our re-analysis covers one published gain', but the coding names "
        f"{counts['ReproducibleRelease']['positive_ids']}"
    )


def test_both_corpora_share_one_reading_across_many_images():
    # sec:res-diagnostic: "On both datasets, one sensor reading is shared across
    # many images". sec:diagnostic draws the line structurally -- one distinct
    # vector per image is a per-image measurement, anything below it is shared --
    # so a re-measurement putting either dataset at that line falsifies the
    # sentence. Read from the artifacts \numPanelTuples and \cddSeries are
    # bound to, not from the macros. The AI Hub reading is the tuple of all nine
    # panel variables, the panel tuple the Introduction defines.
    structure = json.loads((AIHUB / "corpus_structure.json").read_text(encoding="utf-8"))
    meta = _cdd_series_ids()
    shared = {
        "AI Hub": (structure["panel_tuples"], structure["images"]),
        "CDD": (meta["val_md5"].nunique(), len(meta)),
    }
    per_image = sorted(k for k, (readings, rows) in shared.items() if readings >= rows)
    assert not per_image, (
        f"sec:res-diagnostic claims 'On both datasets, one sensor reading is shared "
        f"across many images', but {per_image} carries one reading per image: {shared}"
    )


def test_both_classifiers_exceed_the_majority_class_predictor():
    # sec:res-timestamp: "Both classifiers exceed the majority-class predictor in every crop". A direction, so read off each crop's paired interval over seeds in
    # both metrics, over the crops that the per-seed table holds.
    df = pd.read_csv(TIMESTAMP_CSV)
    short = []
    for crop, g in df.groupby("crop"):
        g = g.sort_values("seed")
        for clf, base in (("panel", "base"), ("timestamp", "base"),
                          ("panel_bal", "base_bal"), ("timestamp_bal", "base_bal")):
            lo = _seed_ci(g[clf].to_numpy() - g[base].to_numpy())[1]
            if lo <= 0:
                short.append((crop, clf, round(lo, 4)))
    assert not short, (
        f"sec:res-timestamp says 'Both classifiers exceed the majority-class predictor in "
        f"every crop', but the paired interval reaches zero in {short}"
    )
    # sec:stats counts a crop only where its paired t-test survives Holm's correction across
    # the crops, so "every crop" is that count.
    for clf, base in (("panel", "base"), ("timestamp", "base"),
                      ("panel_bal", "base_bal"), ("timestamp_bal", "base_bal")):
        higher = _holm_crops(df, clf, base)["higher"]
        assert higher == set(df["crop"]), (
            f"sec:res-timestamp says 'Both classifiers exceed the majority-class predictor in "
            f"every crop', a count that sec:stats takes after Holm's correction, but {clf} "
            f"exceeds {base} after the correction only in {sorted(higher)}"
        )


def test_holding_out_photo_dates_keeps_most_sessions_out_of_training():
    # sec:intro: a split that holds out whole photo dates "keeps the session of most test
    # images out of training". leakage_collapse_gate.py measures it on the photo-date split of the whole
    # spreadsheet: the share of test images whose measurement session occurs in training.
    gate = pd.read_csv(AIHUB / "leakage_collapse_gate.csv").set_index("axis")
    shared = gate.loc["date", "session_in_train"]
    assert shared < 0.5, (
        f"sec:intro claims that a split that holds out whole photo dates 'keeps the session of "
        f"most test images out of training', but {shared:.1%} of the test images of the photo-date "
        f"split have their session in training"
    )


def test_a_held_out_series_leaves_its_date_in_training():
    # sec:res-within: "so a held-out series leaves other series of its photo date in training". The mean of
    # 12.2 series to a date does not imply it for every date, so it is read off the
    # group-disjoint test folds that dacon_published_split.py scored.
    df = pd.read_csv(CDD_DRAWN_SPLITS_CSV)
    g = df[df["regime"] == "group"]
    assert (g["date_in_train"] == 1.0).all(), (
        f"sec:res-within claims 'a held-out series leaves other series of its photo date in training', but on "
        f"the group-disjoint split the share of test images whose photo date occurs in "
        f"training is {dict(zip(g['seed'], g['date_in_train'].round(4)))}"
    )


def test_most_images_were_photographed_on_their_panel_day():
    # sec:intro says that a split holding out whole photo dates "keeps the session of most
    # test images out of training", which holds because most rows were photographed on the
    # day on which their panel was measured, the day of their session: sec:corpora prints
    # the share that was not. Read from the artifact that \pctPhotoDayDiffers is bound to.
    differs = json.loads((AIHUB / "corpus_structure.json").read_text(
        encoding="utf-8"))["photo_day_differs_pct"]
    assert differs < 50, (
        f"sec:intro claims that holding out whole photo dates 'keeps the session of most test "
        f"images out of training', but {differs:.1f}% of the rows were photographed on "
        f"another day than their panel was measured"
    )


FARM_SHARE_HALF = 0.5  # sec:res-diagnostic: farm explains "more than half the variance"
PANEL_NAMES = {             # the AI Hub dataset's panel columns, in the words sec:corpora lists
    "soil_N": "N", "soil_P": "P", "soil_K": "K", "soil_pH": "pH",
    "ms_gec": "electrical conductivity", "temp": "air temperature", "hum": "air humidity",
    "soil_temp": "soil temperature", "soil_condition": "soil moisture",
}


def test_farm_share_over_the_soil_chemistry_and_the_whole_panel():
    # sec:res-diagnostic says "farm identity explains more than half the variance of every
    # panel variable except air humidity and electrical conductivity" and that it explains
    # between \rsqFarmLo and \rsqFarmHi of the variance of "pH, N, P, and K, four of the five
    # soil-chemistry variables that the soil-chemistry classifier reads". The exceptions are derived over every
    # column the artifact scores rather than over the two the sentence names, and the
    # columns are checked against the panel sec:corpora lists, so a column added to
    # or dropped from the panel fails here instead of leaving "nine" behind.
    shares = json.loads((AIHUB / "corpus_structure.json").read_text(
        encoding="utf-8"))["r2_farm_panel"]
    assert set(shares) == set(PANEL_NAMES), (
        f"sec:corpora lists the panel as {sorted(PANEL_NAMES.values())}, but "
        f"corpus_structure.json scores the columns {sorted(shares)}"
    )
    listed = re.search(r"soil\s+and\s+ambient\s+panel\s*\(([^)]*)\)",
                       MAIN.read_text(encoding="utf-8"))
    assert listed, "sec:corpora no longer lists the panel after 'soil and ambient panel ('"
    assert len(listed.group(1).split(",")) == len(shares) == 9, (
        f"sec:res-diagnostic reads the farm share of every panel variable, the nine that "
        f"sec:corpora lists, but sec:corpora lists {len(listed.group(1).split(','))} variables and "
        f"corpus_structure.json scores {len(shares)}"
    )
    four = {c: shares[c] for c in ("soil_pH", "soil_N", "soil_P", "soil_K")}
    assert all(v > FARM_SHARE_HALF for v in four.values()), (
        f"sec:res-diagnostic says farm explains between \\rsqFarmLo and \\rsqFarmHi, more than "
        f"half, of the variance of each of 'pH, N, P, and K', "
        f"but the shares are {four}"
    )
    at_most_half = sorted(PANEL_NAMES[c] for c, v in shares.items() if v <= FARM_SHARE_HALF)
    assert at_most_half == ["air humidity", "electrical conductivity"], (
        f"sec:res-diagnostic says farm explains more than half the variance of each panel "
        f"variable 'except air humidity and electrical conductivity', but the variables at or "
        f"below half are {at_most_half}: {shares}"
    )


def test_methods_names_the_panel_variables_the_baseline_reads():
    # sec:timestamp-methods says that a crop enters the comparison if enough of its rows
    # "carry all five soil-chemistry variables (pH, N, P, K, and electrical conductivity)", and
    # that "The soil-chemistry classifier reads the five soil-chemistry variables". The timestamp baseline
    # (timestamp_baseline_multiseed.py) and the within-session measurement (timestamp_baseline.py) read the
    # columns their BIOCHEM names, not the nine sec:corpora lists, and the Methods
    # said "from the sensor panel" for as long as they did. So the sentence is
    # checked against both scripts' constant, not against the panel.
    read = {s: _script_const(f"code/aihub_subtropical/{s}", "BIOCHEM")
            for s in ("timestamp_baseline_multiseed.py", "timestamp_baseline.py")}
    assert len({tuple(v) for v in read.values()}) == 1, (
        f"sec:timestamp-methods names one set of panel variables for the timestamp baseline and "
        f"the within-session measurement, but the two scripts read {read}"
    )
    names = sorted(PANEL_NAMES[c] for c in read["timestamp_baseline_multiseed.py"])
    text = _rendered_manuscript()["sec:timestamp-methods"]
    m = re.search(r"carry all (\w+)\s+soil-chemistry variables \(([^)]*)\)", text)
    assert m and re.search(r"The soil-chemistry classifier reads the %s soil-chemistry\s+variables"
                           % m.group(1), text), (
        "sec:timestamp-methods no longer says which panel variables the timestamp "
        "baseline reads: 'carry all five soil-chemistry variables (...)' and 'The "
        "soil-chemistry classifier reads the five soil-chemistry variables'")
    listed = sorted(re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(2)))
    words = "zero one two three four five six seven eight nine ten".split()
    assert listed == names and m.group(1) == words[len(names)], (
        f"sec:timestamp-methods says that the soil-chemistry classifier reads the {m.group(1)} "
        f"soil-chemistry variables ({m.group(2)}), but timestamp_baseline_multiseed.py and timestamp_baseline.py read "
        f"{len(names)} columns: {names}"
    )


def test_methods_names_the_timestamp_the_baseline_reads():
    # The two datasets carry different timestamps. sec:timestamp-methods says the AI Hub
    # timestamp classifier reads "the day and the time of day at which a row's panel was
    # measured", and that the CDD one reads "the day and the time of day of the photograph". A sensor's measurement time and an image's capture time
    # differ on 9% of AI Hub rows, so each sentence is held to the column that its
    # scripts derive the day and the hour from, not to the word "timestamp".
    rendered = _rendered_manuscript()
    for section, phrase, scripts, day, hour in [
        ("sec:timestamp-methods", "the day and the time of day at which a row's panel was measured",
         ("code/aihub_subtropical/timestamp_baseline_multiseed.py", "code/aihub_subtropical/timestamp_baseline.py"),
         r'df\["day_ord"\] = df\["measure_dt"\]', r'df\["hour"\] = df\["measure_dt"\]'),
        ("sec:timestamp-methods", "the day and the time of day of the photograph",
         ("code/dacon_cropdisease/dacon_published_split.py",
          "code/dacon_cropdisease/dacon_their_split.py"),
         r'meta\["day_ord"\] = pd\.to_datetime\(meta\["date"\]', r't = meta\["time"\]'),
    ]:
        assert phrase in rendered[section], (
            f"{section} no longer says which timestamp the classifier reads: '{phrase}'"
        )
        for script in scripts:
            src = (HERE / script).read_text(encoding="utf-8")
            assert re.search(day, src) and re.search(hour, src), (
                f"{section} says the timestamp classifier reads '{phrase}', but {script} "
                f"no longer derives its day and hour from that column"
            )


def test_every_include_path_resolves_into_the_generated_tree():
    # Every path the document reads resolves inside a tree a
    # generator writes, and every artifact a generator writes is read by the document.
    # A build failure already catches a missing file; what it cannot catch is a second
    # copy of a generated artifact living where the generator does not write, because
    # the copy is a valid file and the run that produced it was correct -- the
    # generator then regenerates and the document keeps publishing the older run.
    # Enumerated rather than written around the four paths that exist today, which
    # would pass the moment a fifth is added.
    inc = re.findall(r"\\(?:input|include|includegraphics)(?:\[[^\]]*\])?\{([^}]*)\}",
                     MAIN.read_text(encoding="utf-8") + SUPP.read_text(encoding="utf-8"))
    paper = HERE / "paper"
    named = {p: (p if "." in p.rsplit("/", 1)[-1] else p + ".tex") for p in inc}
    missing = sorted(p for p, f in named.items() if not (paper / f).exists())
    assert not missing, f"main.tex or supplementary.tex includes {missing}, absent under paper/"
    # params.tex is the one hand-authored include; the binding tests below stand in
    # for the generator it does not have.
    outside = sorted(p for p in named if p != "params" and not p.startswith("generated/"))
    assert not outside, (
        f"main.tex or supplementary.tex reads {outside} from outside paper/generated/; if a generator writes "
        f"that artifact, the document is reading a copy that will not follow a re-run"
    )
    read = {f.rsplit("/", 1)[-1] for p, f in named.items() if p.startswith("generated/")}
    generated = paper / "generated"
    written = {f.name for f in generated.iterdir()} if generated.is_dir() else set()
    orphan = sorted(written - read)
    assert not orphan, (
        f"paper/generated/ holds {orphan}, which the document never reads; a generated "
        f"artifact no section points to is deleted, not kept"
    )


# ===========================================================================
# paper/params.tex — every macro the manuscript body prints, against its source.
#
# The other tests in this file guard a measured value against the sentence that
# reports it. None of them guards the macro in between, and params.tex is
# hand-authored, so a macro edited to a different number leaves every assertion
# green while the document publishes the new one. It has happened once already:
# \numSeeds printed the crop denominator at four sites including the abstract,
# undetected because the seed set and the crop set both happened to hold seven.
#
# The rule is a blacklist, not a whitelist. Every macro params.tex defines must
# either appear in _params_bindings() with a committed source, or be registered
# in paper/params_exempt.md with the reason it has none. Checking only the macros
# bound here would pass the moment a macro is added, which is the failure this
# exists to prevent.
# ===========================================================================

_CI_PRINTED = re.compile(r"^\[\s*([+-]?[\d.]+)\s*,\\,\s*([+-]?[\d.]+)\s*\]$")


def _prints(printed, value):
    """True iff `printed` is what `value` rounds to at the precision it prints.

    A macro is a rounded display of a measured value, so the test is that the
    display is that value's correct rounding — not that the two agree to some
    tolerance chosen here, which would be one more hand-authored number. Counts,
    percentages and confidence intervals all fall out of the same rule.
    """
    if isinstance(value, str):
        return printed == value
    if isinstance(value, tuple):                      # a printed CI, [lo,\,hi]
        m = _CI_PRINTED.match(printed)
        return bool(m) and all(_prints(m.group(i + 1), v) for i, v in enumerate(value))
    plain = printed.replace("{,}", "")                # 274{,}000 -> 274000
    decimals = len(plain.split(".")[1]) if "." in plain else 0
    sign = "+" if plain.startswith("+") else ""
    return format(value, f"{sign}.{decimals}f") == plain


def _params_defined():
    """{macro: the string it prints} for every macro paper/params.tex defines."""
    return dict(re.findall(_MACRO % r"(\w+)", PARAMS.read_text(encoding="utf-8")))


def _params_exempt():
    """{macro: (who measured it, reason)} from paper/params_exempt.md's table.

    The two kinds are not interchangeable. A figure the CDD study measured is
    cited to that study and is no exception to anything we claim about our own
    numbers; one we measured with no shipped table is, and the availability
    statement counts those.
    """
    rows = re.findall(r"^\|\s*`\\(\w+)`\s*\|([^|]+)\|([^|]+)\|",
                      PARAMS_EXEMPT.read_text(encoding="utf-8"), re.M)
    return {name: (who.strip(), reason.strip()) for name, who, reason in rows}


def _script_const(relpath, name):
    """A module-level constant's value, read from the script that ran with it.

    A protocol number the Methods section prints (an epoch budget, a batch size, a
    comparable band) has its source in the code that executed it, not in a results
    file. Handles the tuple form the epoch budget uses, EPOCHS, PATIENCE, T_0 = ...
    """
    text = (HERE / relpath).read_text(encoding="utf-8")
    for m in re.finditer(r"^([A-Z][\w \t,]*?)[ \t]*=(?!=)[ \t]*([^#\n]+)", text, re.M):
        names = [n.strip() for n in m.group(1).split(",")]
        if name not in names:
            continue
        if len(names) == 1:
            return ast.literal_eval(m.group(2).strip())
        values = [v.strip() for v in m.group(2).split(",")]
        if len(values) == len(names):
            return ast.literal_eval(values[names.index(name)])
    raise AssertionError(f"{relpath} defines no module constant {name}")


# Thresholds that the manuscript states (sec:reanalysis), read from the scripts that
# ran with them rather than copied, so a changed script changes the check.
CDD_COLLAPSE = float(_script_const("code/dacon_cropdisease/dacon_published_split.py", "COLLAPSE"))
CDD_TOL = float(_script_const("code/dacon_cropdisease/dacon_published_split.py", "TOL"))


def _script_tuple_const(relpath, name):
    """A constant bound in a module-level tuple assignment, read through the parse tree.

    `_script_const` splits the right-hand side on commas, so it cannot read a line
    whose values include a tuple of their own: PX, BS, LR, BETAS = 256, 64, 1e-3,
    (0.9, 0.99). The parse tree pairs each name with its value whatever the value
    holds.
    """
    tree = ast.parse((HERE / relpath).read_text(encoding="utf-8"))
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
            for t, v in zip(target.elts, value.elts):
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(v)
    raise AssertionError(f"{relpath} binds no {name} in a module-level tuple assignment")


def _script_literal(relpath, pattern):
    """The one value a script writes inline at every site that `pattern` matches.

    Some settings the Methods print were never named in the run scripts: a fold
    count passed straight to StratifiedKFold, a row floor in an `if`. The scripts
    are dated records and stay as they ran, so the setting is read where it is
    written, and every site must agree, or the Methods would describe one run of
    two.
    """
    found = set(re.findall(pattern, (HERE / relpath).read_text(encoding="utf-8")))
    assert len(found) == 1, f"{relpath} writes {sorted(found) or 'nothing'} for {pattern!r}"
    return ast.literal_eval(found.pop())


def _class_default(relpath, cls, arg):
    """A keyword default of `cls.__init__` in a script: the size the model was built at."""
    tree = ast.parse((HERE / relpath).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == "__init__":
                    names = [a.arg for a in f.args.args][-len(f.args.defaults):]
                    return ast.literal_eval(f.args.defaults[names.index(arg)])
    raise AssertionError(f"{relpath} defines no {cls}.__init__ default {arg}")


def _requirement(name):
    """The version requirements.txt pins for `name`, the version the runs used."""
    text = (HERE / "requirements.txt").read_text(encoding="utf-8")
    if name == "python":
        return re.search(r"Python (\d+\.\d+)", text).group(1)
    return re.search(rf"^{re.escape(name)}==([\w.]+)$", text, re.M).group(1)


def _params_bindings():
    """{macro: the value its committed source derives}, for every bound macro.

    One entry per number the manuscript publishes, naming where that number comes
    from. Macros absent here must be registered in paper/params_exempt.md.
    """
    b = {}

    # --- the corpus itself (corpus_structure.py -> corpus_structure.json) ---
    c = json.loads((AIHUB / "corpus_structure.json").read_text(encoding="utf-8"))
    first, last = date.fromisoformat(c["photo_first"]), date.fromisoformat(c["photo_last"])
    assert first.year == last.year, "the photo window spans two years; \\photoWindow prints one"
    b.update(
        numImages=c["images"], numCrops=c["crops"], numClasses=c["disease_classes"],
        numSingleCropClasses=c["single_crop_disease_classes"],
        numFarms=c["farms"],
        numPanelTuples=c["panel_tuples"],
        numTuplesOneSession=c["panel_tuples_one_session"],
        pctPanelTupleShared=c["panel_tuple_shared_pct"],
        photoWindow=f"{first:%B}--{last:%B} {last.year}",
        pctPhotoDayDiffers=c["photo_day_differs_pct"],
        rsqFarmLo=c["r2_farm_min"], rsqFarmHi=c["r2_farm_max"],
    )

    # --- timestamp vs panel (timestamp_vs_panel_multiseed.csv) ---
    df, s = _timestamp_summary()
    _, sb = _timestamp_summary("panel_bal", "timestamp_bal")
    beyond = _holm_crops(df, "panel_full", "farm_ts")
    b.update(
        numSeeds=df["seed"].nunique(), timestampCrops=len(s),
        timestampHigherCrops=len(_holm_crops(df, "timestamp", "panel")["higher"]),
        timestampMarginSmallLo=min(s[c]["md"] for c in OTHER_THREE),
        timestampMarginSmallHi=max(s[c]["md"] for c in OTHER_THREE),
        figWinPlain=s["무화과"]["md"], mangoWinPlain=s["애플망고"]["md"],
        oliveLeanAbs=-s["올리브"]["md"],
        fullBeyondFarmCrops=len(beyond["higher"]),
        fullBeyondFarmLo=min(beyond["md"][c] for c in beyond["higher"]),
        fullBeyondFarmHi=max(beyond["md"][c] for c in beyond["higher"]),
    )

    # --- within-session residual (timestamp_baseline.py -> timestamp_vs_panel.csv; per seed) ---
    ws = _within_session_advantage()
    resid = ws.groupby("crop")["advantage_within"].mean()   # each crop's mean over seeds
    per_seed = ws.groupby("seed")[["advantage_pooled", "advantage_within",
                                   "advantage_fall"]].mean()
    b.update(panelAdvantagePooled=ws["advantage_pooled"].mean(),
             panelAdvantageWithinSession=ws["advantage_within"].mean(),
             residualLo=resid.min(), residualHi=resid.max())
    floors = _script_const("code/aihub_subtropical/timestamp_baseline.py", "FLOOR_SWEEP")
    b.update(floorLow=min(floors), floorHigh=f"{max(floors):,}".replace(",", "{,}"))

    # --- the tuned forests (timestamp_baseline_tuned.py) and the CDD max_features grid
    # (dacon_their_split_tuned.py): the settings that the supplement's settings table prints ---
    tuned = "code/aihub_subtropical/timestamp_baseline_tuned.py"
    grid = _script_const(tuned, "GRID")
    leaves = [str(v) for v in grid["min_samples_leaf"]]
    assert grid["max_features"] == ["sqrt", 0.5, 1.0], (
        f"tab:settings describes the tuned forests' max_features grid as the square root, half "
        f"and all of the variables, but the script searches {grid['max_features']}"
    )
    cdd_grid = _script_const("code/dacon_cropdisease/dacon_their_split_tuned.py", "GRID")
    assert cdd_grid[0] == "sqrt", f"the CDD grid opens on {cdd_grid[0]}, not the square root"
    b.update(holmAlpha=_script_const("code/tables.py", "HOLM_ALPHA"),
             tunedTrees=_script_const(tuned, "N_TREES"),
             tunedLeafSizes=", ".join(leaves[:-1]) + ", or " + leaves[-1],
             tunedInnerFolds=_script_literal(tuned, r"cv=StratifiedKFold\((\d+), shuffle"),
             cddGridFractions=", ".join(str(v) for v in cdd_grid[1:-1]) + ", or "
             + str(cdd_grid[-1]))

    # --- protocol constants, from the scripts that ran with them ---
    b.update(
        subsampleSize=_script_const("code/aihub_subtropical/h4_materialize_splits.py",
                                    "SUBSAMPLE_TARGET"),
        lowBudgetEpochs=_script_const("code/aihub_subtropical/h4_fusion_collapse.py", "EPOCHS"),
        matchedEpochs=_script_const("code/aihub_subtropical/h4_fusion_matched_budget.py", "EPOCHS"),
        matchedBatch=_script_const("code/aihub_subtropical/h4_fusion_matched_budget.py", "BS"),
        # The CDD seed sets, each read off the file that ran rather than restated
        # here: the split seeds behind the two splits we draw, and the estimator
        # seeds behind the classifiers scored on the study's own fixed split.
        cddSeeds=len(_script_const(
            "code/dacon_cropdisease/dacon_published_split.py", "SEEDS")),
    )

    # --- image fusion: reduced budget, matched budget, matched-budget date run (h4_*.csv) ---
    _, s4 = _fusion_reduced_summary()
    _, s5 = _fusion_matched_summary()
    ch5 = _budget_change()
    *_, s6 = _fusion_date_summary()
    # Both leak-rate numbers come off the same fold of the same 40k subsample the models
    # consumed. leakage_collapse_gate.csv measures the same quantity on the full
    # 274k panel; that is the candidate-split comparison's frame, not the fusion split's.
    pos = _fusion_coverage()
    b.update(
        imageOnlyFone=s4["iid"]["img"], imageOnlyFoneMatched=s5["img"],
        imageOnlyFoneDate=s4["date"]["img"], imageOnlyFoneDateMatched=s6["img"],
        iidDeltaLowBudget=s4["iid"]["d"], iidDeltaLowBudgetCI=(s4["iid"]["lo"], s4["iid"]["hi"]),
        dateDeltaLowBudget=s4["date"]["d"],
        dateDeltaLowBudgetCI=(s4["date"]["lo"], s4["date"]["hi"]),
        iidDelta=s5["d"], iidDeltaCI=(s5["lo"], s5["hi"]), iidDeltaPoints=100 * s5["d"],
        imageOnlyBudgetChange=ch5["img"]["md"],
        fusionBudgetChange=ch5["all"]["md"],
        imageOnlyBudgetChangeCI=(ch5["img"]["lo"], ch5["img"]["hi"]),
        fusionBudgetChangeCI=(ch5["all"]["lo"], ch5["all"]["hi"]),
        budgetChangeDiff=ch5["img_minus_all"][0],
        budgetChangeDiffCI=ch5["img_minus_all"][1:],
        dateDelta=s6["d"], dateDeltaCI=(s6["d_lo"], s6["d_hi"]),
        dateReduction=s6["drop"], dateReductionCI=(s6["drop_lo"], s6["drop_hi"]),
        iidPanelLeak=100 * pos.loc["iid", "panel_leak_fold0"],
        datePanelLeak=100 * pos.loc["date", "panel_leak_fold0"],
        fusionSeeds=_fusion_seeds(s5, s6),
    )
    _, s4b = _fusion_reduced_summary("test_ba")
    _, s5b = _fusion_matched_summary("test_ba")
    *_, s6b = _fusion_date_summary("test_ba")
    for sp in ("iid", "date"):
        b[f"{sp}DeltaLowBudgetBA"] = s4b[sp]["d"]
        b[f"{sp}DeltaLowBudgetBACI"] = (s4b[sp]["lo"], s4b[sp]["hi"])
    b.update(iidDeltaBA=s5b["d"], iidDeltaBACI=(s5b["lo"], s5b["hi"]),
             dateDeltaBA=s6b["d"], dateDeltaBACI=(s6b["d_lo"], s6b["d_hi"]),
             dateReductionBA=s6b["drop"], dateReductionBACI=(s6b["drop_lo"], s6b["drop_hi"]))

    # --- the candidate-split comparison (leakage_collapse_gate.csv) ---
    # sec:fusion-design reports which splits met the comparison's two thresholds.
    # The comparison measures the full panel, not the fusion subsample, so its leak rates
    # and \iidPanelLeak, \datePanelLeak are two populations and not one number.
    gate = pd.read_csv(AIHUB / "leakage_collapse_gate.csv").set_index("axis")
    b.update(
        candidateSplits=len(gate),
        candidateFolds=_script_const("code/aihub_subtropical/leakage_collapse_gate.py", "N_FOLDS"),
        candidateRandomLeak=100 * gate.loc["random", "panel_leak"],
        candidateDateLeak=100 * gate.loc["date", "panel_leak"],
        candidateSessionLeak=100 * gate.loc["session", "panel_leak"],
        candidateFarmCoverage=100 * gate.loc["farm", "cell_pos"],
        farmUncoveredPct=100 * (1 - gate.loc["farm", "cell_pos"]),
        candidateRandomCoverage=100 * gate.loc["random", "cell_pos"],
        candidateDateCoverage=100 * gate.loc["date", "cell_pos"],
        candidateSessionCoverage=100 * gate.loc["session", "cell_pos"],
        candidateFarmLeak=100 * gate.loc["farm", "panel_leak"],
    )

    # --- CDD granularity (dacon_series_ids.parquet, dacon_granularity.json) ---
    m = _cdd_series_ids()
    cdd = _cdd_granularity()
    sizes = m.groupby("val_md5").size()
    b.update(
        cddImages=len(m), cddSeries=m["val_md5"].nunique(),
        cddSharedPct=100 * int(sizes[sizes > 1].sum()) / len(m),
        cddSingleDatePct=cdd["shared_series_single_date_pct"],
        cddSingleCodePct=cdd["shared_series_single_disease_pct"],
        cddDates=cdd["photo_dates"],
        cddSeriesPerDate=m["val_md5"].nunique() / cdd["photo_dates"],
    )

    # --- the audited split's leakage (dacon_audited_split_leakage.csv) ---
    _, val = _cdd_study_split_summary()
    r42, r777 = val["paper_work_seed42"], val[THEIR_SPLIT_TAG]
    b.update(
        cddLeak=100 * r777.breach,
        cddLeakTrainval=100 * r777.breach_trval,
        cddGroupsShared=int(r777.shared), cddGroupsTotal=int(r777.n_test_groups),
        cddDirectTest=100 * r777.direct, cddDirectTrainval=100 * r777.direct_trval,
    )

    # --- the environment series' power on the splits we draw (dacon_published_split.csv) ---
    _, s1 = _cdd_drawn_splits_summary()
    b.update(
        cddBaseRandom=s1["random"]["base"], cddBaseGroup=s1["group"]["base"],
        cddEnvRandom=s1["random"]["env"], cddEnvGroup=s1["group"]["env"],
        cddTimestampRandom=s1["random"]["timestamp"], cddTimestampGroup=s1["group"]["timestamp"],
        cddTimestampEnvRandom=s1["random"]["timestamp_env"],
        cddTimestampEnvGroup=s1["group"]["timestamp_env"],
        **{f"cdd{arm}{reg.capitalize()}CI": (s1[reg][k + "_lo"], s1[reg][k + "_hi"])
           for arm, k in [("Base", "base"), ("Env", "env"),
                          ("Timestamp", "timestamp"), ("TimestampEnv", "timestamp_env")]
           for reg in ("random", "group")},
        cddTimestampOverEnvRandom=s1["random"]["timestamp_over_env"],
        cddTimestampOverEnvRandomCI=(s1["random"]["timestamp_over_env_lo"], s1["random"]["timestamp_over_env_hi"]),
        cddTimestampOverEnvGroup=s1["group"]["timestamp_over_env"],
        cddTimestampOverEnvGroupCI=(s1["group"]["timestamp_over_env_lo"], s1["group"]["timestamp_over_env_hi"]),
        cddLeakRandom=100 * s1["random"]["leak"], cddLeakGroup=100 * s1["group"]["leak"],
    )
    _, s1b = _cdd_drawn_splits_summary("ba")
    for reg in ("random", "group"):
        for arm, k in [("Base", "base"), ("Env", "env"), ("Timestamp", "timestamp"),
                       ("TimestampEnv", "timestamp_env"), ("TimestampOverEnv", "timestamp_over_env")]:
            b[f"cdd{arm}{reg.capitalize()}BA"] = s1b[reg][k]
            b[f"cdd{arm}{reg.capitalize()}BACI"] = (s1b[reg][k + "_lo"], s1b[reg][k + "_hi"])

    # --- the same four classifiers on the study's own committed split (dacon_their_split.csv) ---
    _, ts = _their_split_summary()
    t = ts[THEIR_SPLIT_TAG]
    b.update(
        cddTheirSeeds=t["n_seeds"],
        cddTheirBase=t["base"], cddTheirBaseCI=(t["base_lo"], t["base_hi"]),
        cddTheirEnv=t["env"], cddTheirEnvCI=(t["env_lo"], t["env_hi"]),
        cddTheirTimestamp=t["timestamp"], cddTheirTimestampCI=(t["timestamp_lo"], t["timestamp_hi"]),
        cddTheirTimestampEnv=t["timestamp_env"],
        cddTheirTimestampEnvCI=(t["timestamp_env_lo"], t["timestamp_env_hi"]),
        cddTheirTimestampOverTimestampEnv=t["timestamp_over_timestamp_env"],
        cddTheirTimestampOverTimestampEnvCI=(t["timestamp_over_timestamp_env_lo"],
                                     t["timestamp_over_timestamp_env_hi"]),
    )
    o = ts["paper_work_seed42"]   # the repository's other committed set
    b.update(
        cddOtherBase=o["base"], cddOtherBaseCI=(o["base_lo"], o["base_hi"]),
        cddOtherEnv=o["env"], cddOtherEnvCI=(o["env_lo"], o["env_hi"]),
        cddOtherTimestamp=o["timestamp"], cddOtherTimestampCI=(o["timestamp_lo"], o["timestamp_hi"]),
        cddOtherTimestampEnv=o["timestamp_env"], cddOtherTimestampEnvCI=(o["timestamp_env_lo"], o["timestamp_env_hi"]),
        cddOtherTimestampOverTimestampEnv=o["timestamp_over_timestamp_env"],
        cddOtherTimestampOverTimestampEnvCI=(o["timestamp_over_timestamp_env_lo"], o["timestamp_over_timestamp_env_hi"]),
    )
    _, tsb = _their_split_summary("ba")
    for who, tag in (("Their", THEIR_SPLIT_TAG), ("Other", "paper_work_seed42")):
        for arm, k in [("Base", "base"), ("Env", "env"), ("Timestamp", "timestamp"),
                       ("TimestampEnv", "timestamp_env"),
                       ("TimestampOverTimestampEnv", "timestamp_over_timestamp_env")]:
            b[f"cdd{who}{arm}BA"] = tsb[tag][k]
            b[f"cdd{who}{arm}BACI"] = (tsb[tag][k + "_lo"], tsb[tag][k + "_hi"])

    # --- the settings tab:settings and the Methods print, from the scripts that ran ---
    # Each reader asserts that every script running the setting agrees, so the table
    # describes one protocol rather than the first script it was read from.
    timestamp_script = "code/aihub_subtropical/timestamp_baseline_multiseed.py"
    single = "code/aihub_subtropical/timestamp_baseline.py"
    drawn_splits_script = "code/dacon_cropdisease/dacon_published_split.py"
    trees = {_script_const(drawn_splits_script, "N_TREES"),
             _script_const("code/dacon_cropdisease/dacon_their_split.py", "N_TREES"),
             _script_literal(timestamp_script, r"RandomForestClassifier\((\d+), n_jobs"),
             _script_literal(single, r"RandomForestClassifier\(n_estimators=(\d+)")}
    assert len(trees) == 1, f"the random-forest analyses run forests of {sorted(trees)} trees"
    folds = {_script_literal(timestamp_script, r"StratifiedKFold\((\d+), shuffle"),
             _script_literal(single, r"StratifiedKFold\((\d+), shuffle")}
    assert len(folds) == 1, f"the timestamp baseline and timestamp_baseline.py cross-validate with {folds}"
    fusion = "code/aihub_subtropical/h4_fusion_collapse.py"
    assert (_script_const("code/aihub_subtropical/h4_materialize_splits.py", "N_FOLDS")
            == _script_const(fusion, "N_FOLDS")), "the fusion splits and training script disagree"
    b.update(
        forestTrees=trees.pop(),
        timestampMinRows=_script_literal(timestamp_script, r"len\(g\) < (\d+)"),
        timestampSampleRows=_script_const(timestamp_script, "SAMPLE"),
        timestampFolds=folds.pop(),
        sessionMinRows=_script_literal(single, r"len\(gs\) < (\d+)"),
        sessionMinClassRows=_script_literal(single, r"vc >= (\d+)"),
        cddFolds=_script_const(drawn_splits_script, "N_FOLDS"),
        fusionFolds=_script_const(fusion, "N_FOLDS"),
        imagePixels=_script_tuple_const(fusion, "PX"),
        lowBudgetBatch=_script_tuple_const(fusion, "BS"),
        learningRate=_script_tuple_const(fusion, "LR"),
        earlyStopPatience=_script_const(fusion, "PATIENCE"),
        restartEpochs=_script_const(fusion, "T_0"),
        lstmHidden=_class_default(fusion, "DaconLSTM", "hidden_dim"),
        cddCrops=cdd["crops"], cddClasses=cdd["classes"],
        cddSingleDateClasses=cdd["single_date_classes"],
        pythonVersion=_requirement("python"), pandasVersion=_requirement("pandas"),
        sklearnVersion=_requirement("scikit-learn"), torchVersion=_requirement("torch"),
        timmVersion=_requirement("timm"),
        albumentationsVersion=_requirement("albumentations"),
    )

    return b


def _fusion_seeds(s5, s6):
    """Seeds behind the fusion experiment, from the ones the two matched-budget runs share.

    sec:intro, sec:fusion-design and the caption of tab:fusion say "\\fusionSeeds{}
    seeds", and the date run pairs delta(date) against the matched-budget run's
    committed delta_iid within seed, so the count that backs those sentences is the size of that join rather than
    either CSV's own row count. A numeral, because \\numSeeds{} prints one in the same
    sentence.
    """
    assert s5["n"] == s6["n"], (
        f"the matched-budget run ran {s5['n']} seeds and the date run paired {s6['n']}; the "
        f"manuscript's '\\fusionSeeds{{}} seeds' speaks for both runs at once and can "
        f"no longer name one count"
    )
    return str(s6["n"])


def test_params_every_macro_is_bound_or_exempt():
    defined = _params_defined()
    bound, exempt = set(_params_bindings()), _params_exempt()
    unaccounted = sorted(set(defined) - bound - set(exempt))
    assert not unaccounted, (
        f"params.tex defines {len(unaccounted)} macro(s) with neither a data source in "
        f"_params_bindings() nor an entry in {PARAMS_EXEMPT.name}: {unaccounted}. Every "
        f"number the manuscript prints needs one or the other, or it is a hand-typed "
        f"value nothing can check"
    )
    both = sorted(bound & set(exempt))
    assert not both, (
        f"{both} are both bound to a source and registered as exempt; an exemption is for "
        f"a macro with no source, so one of the two entries is wrong"
    )
    stale = sorted(set(exempt) - set(defined))
    assert not stale, (
        f"{PARAMS_EXEMPT.name} exempts {stale}, which params.tex no longer defines"
    )
    assert not sorted(bound - set(defined)), (
        f"_params_bindings() binds {sorted(bound - set(defined))}, which params.tex does "
        f"not define"
    )


def test_params_macros_are_all_printed():
    # A macro the document never prints is a number carrying a test and a source with
    # no sentence spending them. It also hides: an unprinted macro cannot be
    # caught by reading the paper, only by looking for it.
    #
    # Every parameter file, not params.tex alone: an enumeration scoped to one file
    # stops being exhaustive the first time a macro is written somewhere else, which
    # is the reason _FRAMES above reads _param_files() too. The sibling checks stay
    # on params.tex on purpose: a binding and an exemption are properties of a
    # hand-authored value, and a generated macro answers to its generator.
    main = MAIN.read_text(encoding="utf-8") + SUPP.read_text(encoding="utf-8")
    unused = sorted(
        f"{path.name}:\\{n}" for path in _param_files()
        for n in dict(re.findall(_MACRO % r"(\w+)", path.read_text(encoding="utf-8")))
        if not re.search(r"\\" + n + r"(?![a-zA-Z])", main)
    )
    assert not unused, (
        f"{unused} are defined but neither main.tex nor supplementary.tex prints them; delete each one "
        f"(in the generator, for a generated file) or print it"
    )


def test_no_bare_seed_count_in_the_manuscript():
    # The enumeration above walks params.tex, so it can only see numbers that already
    # became macros. A count that never became one is invisible to it. Seed counts are
    # locked configuration, so each
    # is a macro or it is a number no artifact backs. On a
    # rhetorical hit ("one seed cannot separate an effect from noise"), reword to
    # "a single seed" rather than relax this.
    counts = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d[\d,{}.]*)"
    bare = re.findall(rf"\b{counts}\s+(?:pre-specified\s+)?seeds?\b",
                      MAIN.read_text(encoding="utf-8"))
    assert not bare, (
        f"main.tex prints {len(bare)} seed count(s) as literal text: {bare}. A seed count "
        f"is locked configuration, so it belongs in params.tex bound to the run that used "
        f"it, not typed into a sentence where a re-run cannot reach it"
    )


def test_methods_names_the_estimator_the_code_runs():
    # An estimator name is prose, so no macro binds it and no CSV contradicts it, and
    # every number the sentence introduces can be correct while the name is wrong.
    # Anchor it to the imports instead -- the tabular analyses use exactly one
    # estimator family, and the Methods must name
    # that one. A second family added to the code fails the first assertion, which is
    # the prompt to say so in the paper rather than a rule to relax.
    # The imports drive it: a family the code adopts with no entry below fails, which is
    # the prompt to describe it in the paper, not a line to add here.
    prose = {
        "RandomForestClassifier": "random forest",        # the panel and timestamp classifiers
        "DummyClassifier": "majority-class",              # the floor every classifier scores against
    }
    sources = "\n".join(p.read_text(encoding="utf-8")
                        for p in sorted((HERE / "code").rglob("*.py")))
    imported = {name.strip()
                for line in re.findall(r"^\s*from sklearn\.[\w.]+ import (.+)$", sources, re.M)
                for name in line.split(",")}
    estimators = {c for c in imported if c.endswith(("Classifier", "Regressor"))}
    document = " ".join(_rendered_manuscript().values())
    unnamed = sorted(c for c in estimators if prose.get(c, "\x00") not in document)
    assert not unnamed, (
        f"the tabular analyses import {unnamed}, which the manuscript never names; a reader "
        f"re-running the diagnostic would fit a different model than the one the "
        f"numbers came from"
    )
    assert "The timestamp baseline compares random forests" in _rendered_manuscript()["sec:timestamp-methods"], (
        "Manuscript (sec:timestamp-methods) must describe the timestamp baseline's model: 'The "
        "timestamp baseline compares random forests' -- "
        "timestamp_baseline_multiseed.py and timestamp_baseline.py both run "
        "RandomForestClassifier(60), and the model a Methods section names is the model "
        "a reader would re-run"
    )


def test_params_values_match_their_sources():
    defined = _params_defined()
    wrong = []
    for name, value in sorted(_params_bindings().items()):
        if not _prints(defined[name], value):
            measured = (value if isinstance(value, str) else
                        ", ".join(f"{v:.6g}" for v in value) if isinstance(value, tuple) else
                        f"{value:.6g}")
            wrong.append(f"\\{name} prints {defined[name]} but its source measures {measured}")
    assert not wrong, (
        "params.tex is what the manuscript publishes, and these macros no longer round "
        "to what their source measures:\n  " + "\n  ".join(wrong)
    )


def test_params_external_references_are_single_sourced():
    # The exempt macros are other people's numbers, transcribed once from a paper or a
    # log this repo cannot ship. Each is also transcribed into a constant above that this
    # file's own inequalities are written against, so the two copies are checked against
    # each other: a corrected reference value must move both, or the paper prints one
    # number while the check that reasons about it uses another.
    for macro, constant, value in [
        ("paperIncrement", "PAPER_INCREMENT", 100 * PAPER_INCREMENT),
        ("paperImageFone", "PAPER_IMAGE_F1", PAPER_IMAGE_F1),
        ("paperMultiFone", "PAPER_MULTIMODAL_F1", PAPER_MULTIMODAL_F1),
    ]:
        assert _prints(_param(macro), value), (
            f"\\{macro} prints {_param(macro)} but this file's {constant} is {value}; "
            f"the manuscript and this file would be citing different reference values"
        )


# ===========================================================================
# The population a number lives in.
#
# Every check above runs one direction: number -> source. It verifies that the
# value the paper prints is the value its artifact holds, and it is exhaustive
# along that axis. What it cannot see is a *join*: two numbers, each correctly
# bound to its own artifact, printed in one sentence as though measured over one
# population. A join survives a fully green run, because every binding involved
# is right.
#
# The AI Hub dataset carries four valid populations -- 274,000 panel rows, 249,434
# labeled images, 246,600 distributed images, the 40,000-image fusion subsample
# -- and CDD adds its own. A population recorded only in a params.tex comment is
# recorded in an artifact that nothing reads.
#
# So the population is declared, and read. Each block of params.tex carries a
# `% FRAME:` line; a macro may override it with `[frame:...]` on its own line;
# `none` marks a number that is not a count over a population (a threshold, an
# epoch budget, a seed count). Three rules follow.
# ===========================================================================

# frame id -> (unit noun the manuscript must use for members of this population,
#              a callable returning the population's size, or None if nothing
#              committed measures it).
#
# The unit noun is what makes the frames checkable from prose. Two populations of
# the AI Hub dataset are both countable in "images" in ordinary speech -- the panel
# table's rows and the labelled images -- so the manuscript fixes one word per
# population and the third rule below holds it to that.
#
# The sizes are what keep the scheme from being decorative: a frame that measures
# the same population as another is not a second frame, and declaring one would
# license a crossing that is real.
_FRAMES = {
    "aihub-panel": ("rows", lambda: json.loads(
        (AIHUB / "corpus_structure.json").read_text(encoding="utf-8"))["images"]),
    "aihub-timestamp-crops": (None, None),      # rows of the 7 compared crops carrying all
                                            # five soil-chemistry variables
    "aihub-subsample": ("images", lambda: int(
        _fusion_coverage()["subsample_n"].iloc[0])),
    "cdd": ("images", lambda: len(_cdd_series_ids())),
    "cdd-audited-split": (None, None),    # the CDD study's own committed split files
    "audited-study": (None, None),          # figures the CDD study reports, not ours
    "cited-study": (None, None),            # a figure another cited study reports
    "none": (None, None),
}


def _param_files():
    """Every file that defines a macro the manuscript prints.

    params.tex is hand-authored and the rest are generator output; the population
    rules apply to both, so the frame enumeration reads both.
    """
    return [PARAMS, *sorted((HERE / "paper" / "generated").glob("*_params.tex"))]


# Sentences that deliberately relate two populations. Each is registered by a
# distinctive phrase, with the reason, exactly as params_exempt.md registers a
# macro with no source: crossing frames is allowed when the sentence is *about*
# the crossing and names both sides, and is a finding otherwise.
_CROSS_FRAME_SENTENCES = {
    "the in-distribution gain was itself small,":
        "sec:discussion compares the fusion gain measured on the AI Hub subsample with the "
        "gain that the CDD study reports on its own dataset; the sentence names the CDD study "
        "as the source of the second number, and the comparison of two studies' gains is "
        "the point",
}


def _scan_frames(path):
    """({macro: frame id}, the frame still open at EOF) for one parameter file.

    `% FRAME: x` opens a block, a later `% FRAME:` replaces it, and `% ENDFRAME`
    closes it. The terminator is what bounds a block: scope is otherwise open to
    the end of the file, so a macro appended below the last block joins that
    block's population with nothing anywhere recording that anyone chose it.
    """
    frames, block = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"%\s*ENDFRAME\b", line.strip()):
            block = None
            continue
        header = re.match(r"%\s*FRAME:\s*([\w-]+)", line.strip())
        if header:
            block = header.group(1)
            continue
        m = re.search(_MACRO % r"(\w+)", line)
        if m:
            override = re.search(r"\[frame:([\w-]+)\]", line)
            frames[m.group(1)] = override.group(1) if override else block
    return frames, block


def _params_frames():
    """{macro: frame id} for every macro the manuscript's parameter files define."""
    frames = {}
    for path in _param_files():
        frames.update(_scan_frames(path)[0])   # a block never crosses a file boundary
    return frames


def _manuscript_sentences():
    """[(sentence, [macros it prints])] over raw main.tex, comments stripped.

    Raw rather than rendered: the check is about which macro stands where, and a
    rendered sentence has lost them. Table rows split as sentences do, since a row
    is a claim of its own, and so do semicolons: an item in the contribution list
    states one thing, and reading the four contributions as one sentence reported
    four populations as one. An
    abbreviation splits a sentence early, which can only hide a crossing, never
    invent one.
    """
    text = re.sub(r"(?<!\\)%.*", "", MAIN.read_text(encoding="utf-8"))
    text = text.split(r"\begin{document}", 1)[-1]
    out = []
    for sentence in re.split(r"(?<=\.)\s+|\\\\|;", text):
        macros = re.findall(r"\\(\w+)\{\}", sentence)
        if macros:
            out.append((" ".join(sentence.split()), macros))
    return out


def test_params_every_macro_declares_a_frame():
    # A blacklist, like the binding enumeration above: a macro added to a block
    # inherits its frame, and a macro added outside every block fails here rather
    # than silently joining whichever population its neighbour belongs to.
    #
    # The file is named because more than one defines macros now, and a report
    # that says params.tex for a macro in paper/generated/ sends the next reader
    # to the wrong file.
    frames = _params_frames()
    undeclared = sorted(f"{path.name}:\\{n}" for path in _param_files()
                        for n, f in _scan_frames(path)[0].items() if f is None)
    assert not undeclared, (
        f"{undeclared} are defined outside any `% FRAME:` block; every number "
        f"belongs to a population or is declared `[frame:none]`"
    )
    unknown = sorted({f for f in frames.values()} - set(_FRAMES))
    assert not unknown, (
        f"the parameter files declare frames {unknown}, absent from _FRAMES"
    )
    empty = sorted(set(_FRAMES) - {"none"} - set(frames.values()))
    assert not empty, f"_FRAMES declares {empty}, which no macro belongs to"
    sized = {f: s() for f, (_, s) in sorted(_FRAMES.items()) if s is not None}
    collision = sorted(a for a in sized for b in sized if a < b and sized[a] == sized[b])
    assert not collision, (
        f"{collision} measure the same population as another declared frame, so the "
        f"distinction between them is a label rather than a fact, and the crossing "
        f"rule below would be licensing a real one: {sized}"
    )


def test_no_param_file_ends_inside_an_open_frame_block():
    # The one way into this scheme that leaves no trace. Every other placement
    # puts a macro between two visible markers, so the population it joins is a
    # choice someone made and can be read back; a macro appended below the last
    # block inherits that block, and the checks downstream see it only when the
    # manuscript happens to print it next to a unit noun -- which a ratio, a
    # threshold, or an accuracy never does.
    #
    # Closing the last block makes the end of a file ground where a macro has no
    # frame, so the next one added there fails the declaration test above instead
    # of guessing. This is the tail; a block that ends mid-file needs no
    # terminator, because what follows it is the next marker, in plain sight.
    open_at_eof = sorted(p.name for p in _param_files() if _scan_frames(p)[1])
    assert not open_at_eof, (
        f"{open_at_eof} end inside a `% FRAME:` block, so a macro appended to the "
        f"end of one would join that population without anyone deciding it should. "
        f"Close the last block with `% ENDFRAME` (in the generator, for generated "
        f"files)."
    )


def test_one_sentence_names_one_population():
    # The join no number-to-source check can see. Each macro below is bound to the
    # artifact that measures it and each binding is correct; what is wrong is that
    # one sentence reports two of them as one quantity.
    frames = _params_frames()
    crossings, used = [], set()
    for sentence, macros in _manuscript_sentences():
        present = {frames.get(m) for m in macros if frames.get(m) not in (None, "none")}
        if len(present) < 2:
            continue
        registered = {phrase for phrase in _CROSS_FRAME_SENTENCES if phrase in sentence}
        if registered:
            used |= registered
            continue
        # audited-study is a provenance, not a population: it marks a figure the
        # CDD study reports rather than a set of items we measured. Setting one
        # of ours beside one of theirs is this paper's central move and is sound --
        # so long as the sentence says whose number is whose. Unmarked, it is a
        # splice of two numbers from two test-set definitions, so the attribution
        # is what is required here rather than the crossing forbidden.
        if "audited-study" in present and len(present) == 2 and \
                re.search(r"\b(their|reported|study's)\b", sentence):
            continue
        named = {m: frames[m] for m in macros if frames.get(m) not in (None, "none")}
        crossings.append(f"{sorted(present)}: {sentence[:150]}…\n      macros: {named}")
    assert not crossings, (
        f"{len(crossings)} sentence(s) print numbers measured over different "
        f"populations, with nothing in the sentence saying so. Register the sentence "
        f"in _CROSS_FRAME_SENTENCES if the crossing is the point and both sides are "
        f"named; otherwise one of the numbers is measured on the wrong "
        f"frame:\n  " + "\n  ".join(crossings)
    )
    stale = sorted(set(_CROSS_FRAME_SENTENCES) - used)
    assert not stale, (
        f"_CROSS_FRAME_SENTENCES registers {stale}, which no sentence printing numbers "
        f"from two populations contains. A registration that outlives its sentence "
        f"exempts nothing and exempts the next sentence that contains the phrase"
    )


def test_a_count_prints_its_own_unit_noun():
    # Two of the AI Hub dataset's populations are both "images" in ordinary speech, so the
    # sentence rule above cannot separate them when only one macro is present. The
    # manuscript fixes one noun per population; this holds it to that noun wherever
    # it prints a member count.
    frames, units = _params_frames(), {u for u, _ in _FRAMES.values() if u}
    text = re.sub(r"(?<!\\)%.*", "", MAIN.read_text(encoding="utf-8"))
    wrong = []
    for macro, frame in sorted(frames.items()):
        # An undeclared or unknown frame is reported by the declaration test,
        # which owns that failure; indexing it here would raise a KeyError from
        # the unit-noun check instead, which stops the run without saying why.
        unit = _FRAMES.get(frame, (None, None))[0]
        if unit is None:
            continue
        for m in re.finditer(r"\\" + macro + r"\{\}", text):
            # The unit a count is printed with is the noun beside it, so the window
            # stops at the sentence that prints it: a later sentence naming another
            # population is not this count's unit.
            window = re.split(r"(?<=\.)\s", text[m.end():m.end() + 60])[0]
            after = re.findall(r"[A-Za-z-]+", window)[:3]
            printed = next((w for w in after if w.lower() in units), None)
            if printed and printed.lower() != unit:
                wrong.append(
                    f"\\{macro} counts the {frame} population, whose unit is "
                    f"'{unit}', but the manuscript prints it as '{printed}': "
                    f"…{' '.join(after)}…"
                )
    assert not wrong, (
        f"{len(wrong)} count(s) are printed with another population's unit "
        f"noun:\n  " + "\n  ".join(wrong)
    )


def _scored_run_tables():
    """Every committed table whose rows are scored runs, by path.

    A run is what a seed indexes, so the seed column is the discriminator: the
    coding records, the candidate-split comparison and the split-leakage counts
    carry no seed and are outside a rule about configurations. Derived from the
    artifacts rather than listed, because a list of which tables owe the check
    passes the first time a script writes a new one.
    """
    out = {}
    for f in sorted((HERE / "code").rglob("*.csv")):
        df = pd.read_csv(f)
        if "seed" in df.columns:
            out[f] = df
    return out


def _score_columns(df):
    """The columns holding a score, read off the values rather than the names.

    Every metric, share and rate this pipeline writes falls in [0,1]; counts,
    epoch budgets, seeds and wall-clock do not. A validation column is a
    model-selection statistic and not a result, so it is outside the rule.
    An over-wide read is safe in this direction: an extra column can only supply
    one more value that might be printed, never hide one that is not.
    """
    cols = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or c.startswith("val_"):
            continue
        v = df[c].dropna()
        if len(v) and v.min() >= 0.0 and v.max() <= 1.0:
            cols.append(c)
    return cols


def _reaches_the_reader(values, bound):
    """True iff one of `values` is a number the manuscript binds.

    Matched against the values the bindings *derive*, never against the strings
    the macros print. A printed form is a rounding, so \\panelDecisiveCrops's "0"
    and \\cddTheirSeeds's "7" match any small number put beside them: written
    that way the check would pass with every configuration deleted. The bound
    value is the same float this function recomputes, so the comparison is equality.

    0 and 1 are the floor and ceiling of every share in these tables and identify
    no configuration, and an integral binding is a count rather than a score: a
    share scaled to a percentage can land on a small seed count and clear the check
    for a configuration bound nowhere.
    """
    scores = [b for b in bound if not float(b).is_integer()]
    return any(math.isclose(v, b, rel_tol=1e-9, abs_tol=1e-12)
               for v in values if pd.notna(v) and v not in (0.0, 1.0)
               for b in scores)


def _printed_in_generated_tables(values):
    """True iff one of `values` is printed, at its printed precision, in a table that
    code/tables.py writes into paper/generated/ for the Supplementary Material.

    Those tables print numbers that no macro binds, so a configuration whose scores
    reach the reader only there would fail the binding match. A printed value matches
    a value within half a unit of its last printed digit, and only printed values of
    three or more decimals count, because a value printed to fewer digits matches too
    much to identify a configuration.
    """
    printed = []
    for f in sorted((HERE / "paper" / "generated").glob("*.tex")):
        text = f.read_text(encoding="utf-8").replace("$-$", "-")
        printed += [(float(x), len(x.split(".")[1])) for x in re.findall(r"-?\d+\.\d{3,}", text)]
    return any(abs(v - p) <= 0.5 * 10 ** -d
               for v in values if pd.notna(v) and v not in (0.0, 1.0)
               for p, d in printed)


# A wide column prefix that names the evaluation fold rather than a configuration.
# One entry, keyed on the prefix and not on the tables carrying it, so a new script
# writing test_f1/test_ba is covered without an edit here. Registered rather
# than commented: an exemption a check enumerates is a decision.
_FOLD_PREFIXES = {"test": "the evaluation fold; these tables name their models in an `arm` column"}


def test_every_arm_the_runs_scored_reaches_the_reader():
    """Every configuration the results data contains reaches the reader, for a
    configuration that a table encodes as columns.

    `test_every_split_the_run_scored_leaves_a_number_in_the_paper` derives this
    for the fusion experiment's split column, where the configurations are rows.
    The other encoding is outside it: one column per configuration, one row per
    seed. A table can be correctly bound and hold no wrong number while a
    configuration it carries is reported nowhere, so no binding check fires.
    Configurations that a table encodes as rows -- the environment-only classifier
    on the study's own split, the group-disjoint regime of the splits we draw --
    are outside this check: a level of a condition column reaches the reader
    through a paired summary this recomputes only unpaired, so a check over levels
    fires on configurations that are reported.

    A configuration is a column-name prefix carrying more than one metric. It
    reaches the reader when the manuscript binds one of its means or a signed
    difference against another configuration, since an increment publishes the two
    configurations defining it. Matched against the values the bindings derive,
    never against the strings the macros print: a printed form is a rounding, so
    \\panelDecisiveCrops's "0" matches any small number put beside it.
    """
    bound = [v for v in _params_bindings().values() if isinstance(v, (int, float))]
    bound += [x for v in _params_bindings().values() if isinstance(v, tuple)
              for x in v if isinstance(x, (int, float))]
    silent = []
    for path, df in _scored_run_tables().items():
        score = _score_columns(df)
        cells = [g for c in df.columns
                 if not pd.api.types.is_numeric_dtype(df[c]) and 1 < df[c].nunique() <= 12
                 for _, g in df.groupby(c)] + [df]
        metrics = {}
        for c in score:
            prefix, _, metric = c.rpartition("_")
            metrics.setdefault(prefix or c, set()).add(metric if prefix else "")
        arms = {p for p, m in metrics.items() if len(m) > 1} - set(_FOLD_PREFIXES)
        for arm in sorted(arms):
            cols = [c for c in score if (c.rpartition("_")[0] or c) == arm]
            vals = [g[c].mean() for g in cells for c in cols]
            vals += [100 * v for v in list(vals)]
            for other in arms - {arm}:
                for c in cols:
                    # a metric-less column (accuracy) pairs with the other arm's own name
                    twin = other if c == arm else f"{other}_{c.rpartition('_')[2]}"
                    if twin in score:
                        vals += [d for g in cells
                                 for d in (g[c].mean() - g[twin].mean(),
                                           g[twin].mean() - g[c].mean())]
            if not (_reaches_the_reader(vals, bound) or _printed_in_generated_tables(vals)):
                silent.append(f"{path.name}: column {arm}")
    assert not silent, (
        "the runs scored these configurations and the committed tables carry every one, but "
        "each reaches the reader as no number at all -- not a level, not a "
        "difference against another configuration. A configuration the results data contains and no "
        "section mentions reads as suppression from the outside:\n  " + "\n  ".join(silent)
    )


def test_every_split_the_run_scored_leaves_a_number_in_the_paper():
    # A Methods promise the Results drop is invisible to every check above. The
    # rows are missing everywhere, so no binding can be wrong; the sentence that
    # promises them carries no number, so the frame check has nothing to read. What
    # is checkable is the other direction: a split the run actually scored must
    # reach the reader as some number -- a gain if it is reported, or the
    # measurement that declines it if it is not.
    #
    # Values are matched, not macro names: a hand-written list of which macro
    # belongs to which axis would pass the moment a fourth axis is added, which is
    # the failure this is written to prevent, one axis later.
    pos, (_, s4) = _fusion_coverage(), _fusion_reduced_summary()
    pool = [v for m, v in _params_defined().items()
            if _params_frames().get(m) == "aihub-subsample"]
    varying = [c for c in pos.columns
               if pd.api.types.is_numeric_dtype(pos[c]) and pos[c].nunique() > 1]
    silent = []
    for axis in sorted(set(pd.read_csv(FUSION_REDUCED_CSV)["split"])):
        # 0 and 1 are the floor and ceiling of every share in the table; they match
        # any macro that happens to print one, so they identify no axis.
        vals = [float(pos.loc[axis, c]) for c in varying]
        vals = [v for v in vals if v not in (0.0, 1.0)]
        vals += [100 * v for v in vals] + [s4[axis][k] for k in ("d", "lo", "hi", "img", "all")]
        if not any(_prints(shown, v) for shown in pool for v in vals):
            silent.append(axis)
    assert not silent, (
        f"sec:fusion trains on {sorted(set(pd.read_csv(FUSION_REDUCED_CSV)['split']))} and the "
        f"committed results carry every one, but {silent} reaches the reader as no number at "
        f"all: not a gain, not the coverage measurement that would decline it. A split the "
        f"training covered and the paper never mentions is a Methods promise the Results "
        f"drop, or a run withheld -- from the outside those read the same"
    )


# ===========================================================================
# The obligations a published number carries.
#
# A script whose output a published number rests on owes a degrees-of-freedom
# accounting in its header and, wherever it fixes an estimator, a BASELINE tag.
# Carried as prose, both go stale the moment a script's role changes, so the
# obligation is derived instead -- if a script's output is committed, the script
# is publishing, and it owes the tags.
# ===========================================================================

_DOF_TAGS = ("MEASURED", "CHOSEN", "FITTED")


def _committed_artifacts():
    """Every committed result artifact under code/, by path."""
    return sorted(f for f in (HERE / "code").rglob("*")
                  if f.suffix in {".csv", ".json", ".parquet", ".txt"})


# The artifacts that are records rather than pipeline inputs: nothing downstream
# consumes them, and that is what they are for. Registered here with the reason,
# the way params_exempt.md registers a macro with no source and
# _CROSS_FRAME_SENTENCES registers a deliberate crossing -- an escape hatch a check
# enumerates is a decision, a comment is not. A log that nothing reads and no
# sentence rests on is not a record, it is an orphan.
_RECORDS = {
    "leakage_collapse_gate.csv":
        "the candidate-split comparison for the fusion experiment, measured on the full "
        "274k panel rather than the fusion split's frame. sec:fusion-design reports "
        "it, and no later script consumes it: a comparison whose output a later run "
        "read could be re-scored after the fact",
}


_WRITES = re.compile(r"to_csv|to_parquet|write_text|json\.dump|open\([^)]*[\"']w|"
                     r"^\s*[A-Z_]*OUT[A-Z_]*\s*=", re.M)


def _names_artifact(script, artifact):
    """The lines of `script` that resolve `artifact`, split by where they resolve it.

    Only quoted occurrences: a header that mentions a filename in prose is
    describing the artifact, not resolving it, and counting those as a path inside
    the repository is how a second copy would pass this check.
    """
    quoted = re.compile(r"[\"'][^\"']*" + re.escape(artifact.name) + r"[\"']")
    inside, outside = [], []
    for line in script.read_text(encoding="utf-8").splitlines():
        if not quoted.search(line):
            continue
        (outside if re.search(r"\bSCRATCH\b|[\"']/home/", line) else
         inside).append(line.strip())
    return inside, outside


def _writers(artifact):
    """The scripts beside `artifact` that write it, directly or through a binding.

    The pipeline's usual idiom binds the path to a module-level name and writes
    through that name on a later line, so the filename and the write verb never
    share a line. Matching only lines that quote the filename saw the write when
    the name happened to contain OUT (which `_WRITES` keys on) and missed it
    otherwise: a script binding its output to a name of its own choosing published
    macros with no accounting, on a check that could not see it writing anything.
    """
    out = []
    for script in sorted(artifact.parent.glob("*.py")):
        src = script.read_text(encoding="utf-8")
        quoted = sum(_names_artifact(script, artifact), [])
        bound = re.findall(r"^\s*([A-Za-z_]\w*)\s*=\s*[^=\n]*[\"\'][^\"\']*"
                           + re.escape(artifact.name) + r"[\"\']", src, re.M)
        through = [line for line in src.splitlines()
                   if _WRITES.search(line)
                   and any(re.search(r"\b" + v + r"\b", line) for v in bound)]
        if any(_WRITES.search(line) for line in quoted) or through:
            out.append(script)
    return out


def _suite_consumers():
    """The script names whose reads put a number on the path into the paper.

    This file, plus the generators it imports: the survey and external-dataset
    macros are verified by re-running those emitters, so an artifact they open is
    on a number's path even though this file never names it. Read out of the
    imports rather than listed, so a generator the suite starts re-deriving from
    joins the set by being imported.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = {a.name + ".py" for node in ast.walk(tree)
             if isinstance(node, ast.Import) for a in node.names}
    return {Path(__file__).name} | {n for n in names if any((HERE / "code").rglob(n))}


def _resolves_as_path(artifact):
    """A quoted occurrence of `artifact` that is joined onto a directory.

    The name has to be reached *as a path* for the line to be consuming the file.
    A filename inside a longer string is the pipeline describing itself -- a
    failure message naming the check a run should consult, a provenance comment
    the generator emits into its own output -- and reading those as consumption
    is how an artifact nothing opens keeps its binding.
    """
    return re.compile(r"/\s*[\"\'][^\"\']*" + re.escape(artifact.name) + r"[\"\']")


def _readers_of(artifact):
    """The scripts that open `artifact`, as [(script name, the line)].

    The mirror of `_writers`, and the distinction the reachability check below did
    not draw: it asked whether the artifact's name occurred anywhere in the
    concatenated source of code/ and of this file, so a write statement, a
    docstring, and the text of a failure message each cleared an artifact of
    orphan status. An artifact's own writer is not evidence that anything consumes
    it, which is the one direction that matters -- the check exists to find the
    file the pipeline produces and nobody reads.
    """
    path = _resolves_as_path(artifact)
    sites = []
    for script in [*sorted((HERE / "code").rglob("*.py")), Path(__file__)]:
        for line in script.read_text(encoding="utf-8").splitlines():
            if path.search(line) and not _WRITES.search(line):
                sites.append((script.name, line.strip()))
    return sites


def test_a_script_that_publishes_a_number_carries_its_tags():
    # "Publishes" is derived, not declared: a script publishes when it writes an
    # artifact this suite reads, because that is the path a number takes from a run
    # into the paper. timestamp_baseline.py became one the day two of its numbers entered
    # sec:res-diagnostic, and nothing recomputed the obligation that came with it.
    consumers = _suite_consumers()
    missing = []
    for artifact in _committed_artifacts():
        if not any(s in consumers for s, _ in _readers_of(artifact)):
            continue                                   # not on any number's path
        for script in _writers(artifact):
            src = script.read_text(encoding="utf-8")
            header = ast.get_docstring(ast.parse(src)) or ""
            if not any(t in header for t in _DOF_TAGS):
                missing.append(
                    f"{script.name} writes {artifact.name}, which this suite reads, "
                    f"and its header carries no {'/'.join(_DOF_TAGS)} accounting")
            # The BASELINE tag covers a fixed architecture, encoder, optimizer or
            # reference implementation, not only a fixed sklearn estimator. Reading
            # the sklearn imports alone would leave every torch script outside the
            # rule. A script importing torch trains a model it fixed.
            fits = any(c.strip().endswith(("Classifier", "Regressor"))
                       for line in re.findall(r"^\s*from sklearn[\w.]* import (.+)$",
                                              src, re.M)
                       for c in line.split(","))
            fits = fits or bool(re.search(r"^\s*(?:import torch|from torch)", src, re.M))
            if fits and "BASELINE" not in header:
                missing.append(
                    f"{script.name} fixes an estimator and publishes {artifact.name}, "
                    f"and its header carries no BASELINE tag")
    assert not missing, (
        "a script whose output this suite reads is publishing, and owes the accounting "
        "a published number carries:\n  " + "\n  ".join(sorted(set(missing)))
    )


def test_committed_artifacts_are_reachable_from_the_pipeline():
    # The path property, one directory over from the include check above. The
    # generator writes to $TROPICAL_SCRATCH, a copy is committed under code/, and
    # this file reads the copy: a re-run cannot reach what the paper publishes, and
    # nothing compares the two. The access-restricted datasets make the freshness half
    # unenforceable here; this is the half that is.
    off_path, split, orphan = [], [], []
    for artifact in _committed_artifacts():
        rel = str(artifact.relative_to(HERE))
        if not _readers_of(artifact) and artifact.name not in _RECORDS:
            orphan.append(rel)
            continue
        # Every sibling script, not only the writer: the question is which file the
        # pipeline resolves under this name, and a reader reaching for the scratch
        # copy opens the same gap from the other end.
        sites = [_names_artifact(s, artifact)
                 for s in sorted(artifact.parent.glob("*.py"))]
        inside = [ln for i, _ in sites for ln in i]
        outside = [ln for _, o in sites for ln in o]
        if outside and not inside:
            off_path.append(rel)
        elif outside and inside:
            split.append(f"{rel} — {len(inside)} script(s) resolve the committed copy, "
                         f"{len(outside)} the scratch one")
    stale = sorted(set(_RECORDS) - {a.name for a in _committed_artifacts()})
    assert not stale, (
        f"_RECORDS registers {stale}, which no committed artifact is named by. A "
        f"registration that outlives its file exempts nothing and hides the next "
        f"artifact that takes the name"
    )
    assert not (orphan or off_path or split), "\n  ".join(
        ([f"{len(orphan)} committed artifact(s) are read by no script and no "
          f"assertion: {orphan}. A generated artifact nothing resolves is deleted, "
          f"not kept — its numbers otherwise reach the reader by "
          f"transcription."] if orphan else [])
        + ([f"{len(off_path)} committed artifact(s) sit off the pipeline's path: "
            f"every script resolves the name outside the repository, so no step puts "
            f"the file where this suite reads it and a re-run leaves the paper "
            f"publishing the older run: "] + off_path if off_path else [])
        + ([f"{len(split)} committed artifact(s) are resolved at two different paths, "
            f"so the pipeline holds two files under one name and nothing compares "
            f"them: "] + split if split else [])
    )


_SCRATCH_BINDING = re.compile(
    r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*SCRATCH\s*/\s*"
    r"[\"']([^\"']+\.(?:csv|json|parquet|npy|txt))[\"']")
_READS = re.compile(r"read_csv|read_parquet|np\.load|json\.load|open\(")


def test_no_result_is_published_only_to_the_working_directory():
    # Both ends need checking, and the reachability check above walks one: it starts
    # from the committed artifacts, so a result written only to $TROPICAL_SCRATCH
    # never enters its loop. The author of the run finds the file locally, this
    # suite stays green, and a clone has nothing to read -- the failure that check
    # exists to catch, entered from the side it cannot see.
    #
    # Result or cache is derived, not declared. The working directory is for
    # what nobody publishes, and what nobody publishes is read back: a cache earns
    # its place there by being loaded again, by its own writer or by a sibling. An
    # off-repo binding no script ever reads is not an intermediate, it is the run's
    # own output, and its destination is beside the script that writes it.
    scripts = {s: s.read_text(encoding="utf-8").splitlines()
               for s in sorted((HERE / "code").rglob("*.py"))}
    bound = {}          # artifact name -> [(script, variable, line no)]
    for script, lines in scripts.items():
        for i, line in enumerate(lines, 1):
            m = _SCRATCH_BINDING.match(line)
            if m:
                bound.setdefault(m.group(2), []).append((script, m.group(1), i))
    stranded = []
    for name, sites in sorted(bound.items()):
        # Resolve each binding inside its own script: a variable name is local, so
        # scanning every script for it reads a sibling's `out_path` as this `out`
        # being loaded again, and the check clears the artifact it was written to
        # catch. Sharing between scripts is already carried by `sites`, which groups
        # on the path rather than on the name bound to it.
        read_back = any(re.search(rf"\b{re.escape(var)}\b", ln) and _READS.search(ln)
                        for script, var, _ in sites
                        for ln in scripts[script])
        if not read_back:
            where = ", ".join(f"{s.name}:{n}" for s, _, n in sites)
            stranded.append(f"{name} ({where})")
    assert not stranded, (
        f"{len(stranded)} result artifact(s) are written only outside the "
        f"repository and read back by nothing, so they are outputs rather than "
        f"caches and no clone can reach them. Bind each to a path "
        f"beside its own script, as every other script in its directory does:\n  "
        + "\n  ".join(stranded)
    )


def _defined_section_labels():
    """Every label the manuscript and its Supplementary Material define, over their include closure."""
    text = MAIN.read_text(encoding="utf-8") + SUPP.read_text(encoding="utf-8")
    for name in re.findall(r"\\(?:input|include)\{([^}]*)\}", text):
        f = MAIN.parent / (name if name.endswith(".tex") else name + ".tex")
        if f.exists():
            text += f.read_text(encoding="utf-8")
    return set(re.findall(r"\\label\{([^}]*:[^}]*)\}", text))


def test_every_label_slug_a_repo_file_names_resolves():
    # Code cites the manuscript by its stable slug rather than by a rendered
    # number, and that stability is what hides the decay: the slug survives
    # renumbering, so it reads as the durable form, while folding or retitling a
    # section deletes it outright. Inside the document the typesetter reports the
    # dangling reference, which is why paper/*.tex is excluded here; outside it
    # nothing does, so a docstring or a failure message points at a section that no
    # longer exists with this suite green. Both ends derived: the labels from
    # the include closure, the references from version control. No dated-record
    # exemption applies here -- a path is a script's own name and stays findable, a
    # slug is only a pointer, so a record describing an older tree names its section
    # in words rather than keeping a slug the manuscript dropped.
    # Every namespace the manuscript labels in, derived from the labels rather than
    # written down: a list of prefixes is a membership list, and the next float kind
    # is not on it.
    defined = _defined_section_labels()
    prefixes = sorted({lab.split(":", 1)[0] for lab in defined})
    assert prefixes, "the manuscript defines no \\label, so this check reads nothing"
    pattern = r"\b(?:" + "|".join(map(re.escape, prefixes)) + r"):[A-Za-z0-9:_-]+"
    tracked = _tracked()
    dangling = {}
    for name in tracked:
        f = HERE / name
        if f.suffix in {".png", ".pdf", ".parquet", ".npy"} or not f.exists():
            continue
        if name.startswith("paper/") and f.suffix == ".tex":
            continue
        for m in re.finditer(pattern, f.read_text(encoding="utf-8")):
            slug = m.group(0).rstrip("-_:")
            if slug not in defined:
                dangling.setdefault(slug, set()).add(name)
    assert not dangling, (
        f"{len(dangling)} label slug(s) named in the repository resolve to no "
        f"\\label in the manuscript. Nothing dereferences these, so the pointer "
        f"survives the section it names:\n  "
        + "\n  ".join(f"{s} — named in {sorted(fs)}" for s, fs in sorted(dangling.items()))
    )


def test_no_bare_numeral_in_the_manuscript_body():
    # The macro enumeration walks params.tex, so it sees only numbers that already
    # became macros; a numeral typed straight into a sentence is invisible to it, the
    # way a bare seed count was. The whitelist is for numerals that are not
    # measurements: a confidence level, a definition, a model's name, a year.
    allowed = {
        "95",           # the confidence level, printed with every interval
        "90",           # the confidence level of the one-sided bound that defines "comparable"
        "0", "14",      # pH's definitional bounds, $[0,14]$
        "50",           # ResNet-50, a model's name
        "145", "02841", # the street number and postcode of the byline's postal address
    }
    # Both documents: a number typed into the Supplementary Material escapes the
    # binding as surely as one typed into the body.
    text = " ".join(re.sub(r"(?<!\\)%.*", "", doc.read_text(encoding="utf-8"))
                    .split(r"\begin{document}", 1)[-1] for doc in (MAIN, SUPP))
    for pattern in (r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
                    r"\\begin\{tabular\}.*?\\end\{tabular\}",
                    # a figure's width is layout, not a measurement
                    r"\\(?:label|ref|cite[a-z]*|input|ead|bibliography|bibliographystyle|"
                    r"includegraphics)(?:\[[^\]]*\])?\{[^}]*\}",
                    r"\\[a-zA-Z]+\{\}",
                    r"[Cc]riterion \d",    # a numbered criterion's cross-reference, not a value
                    r"\\eta\^2",          # the effect size's name, eta squared, not a value
                    r"Claude\s+Opus\s+\d[\d.]*",  # a model's version in the AI disclosure
                    r"\(\d\)"):           # an enumerator in a list, not a value
        text = re.sub(pattern, " ", text, flags=re.S)
    # A numeral that ends a sentence carries the sentence's period, which is not part
    # of the number.
    bare = sorted({n.rstrip(".") for n in re.findall(r"(?<![\w.\\-])\d[\d.]*", text)}
                  - allowed)
    assert not bare, (
        f"main.tex or supplementary.tex prints {bare} as literal text. A measurement in the body belongs in "
        f"params.tex bound to the artifact that measured it; a number that is not a "
        f"measurement belongs in the whitelist above with the reason it is not one"
    )


def test_the_readme_names_the_manuscript_by_its_title():
    # README.md names the paper by its title, and a retitled manuscript leaves the old
    # title there: nothing renders the README beside the PDF, so only this comparison
    # sees the two drift apart.
    tex = re.sub(r"(?<!\\)%.*", "", MAIN.read_text(encoding="utf-8"))
    m = re.search(r"\\title(?:\[[^\]]*\])?\{([^}]*)\}", tex)
    assert m, "main.tex has no \\title{...} for the README to name"
    title = " ".join(m.group(1).split())
    readme = " ".join((HERE / "README.md").read_text(encoding="utf-8").split())
    assert f"*{title}*" in readme, (
        f"README.md names the paper by a title other than main.tex's '{title}'"
    )


def test_signed_values_print_one_minus_sign():
    # A macro holding a signed number renders "-0.020" in a table cell and "-0.020"
    # in a paragraph: LaTeX draws a hyphen in text mode and a minus in math mode, so
    # the same value takes two glyphs an inch apart on one page. Both sites are
    # correct and no binding can see it, because nothing here compares a rendered
    # value against the same value rendered elsewhere. Derive the set from the values
    # rather than list the macros, or the next signed macro inherits the pass.
    signed = {m for m, v in _params_defined().items() if v.startswith(("+", "-", "["))}
    text = re.sub(r"(?<!\\)%.*", "", MAIN.read_text(encoding="utf-8"))
    outside = re.sub(r"\$[^$]*\$", " ", text.split(r"\begin{document}", 1)[-1])
    loose = sorted({m for m in re.findall(r"\\(\w+)\{\}", outside) if m in signed})
    assert not loose, (
        f"{loose} print a signed value outside math mode, so the manuscript draws a "
        f"hyphen where the same macro inside $...$ draws a minus. Wrap each use in "
        f"$...$; {len(signed)} macros carry a sign or an interval"
    )


# ===========================================================================
# The manuscript sentences this file quotes back.
#
# An assertion checks an identity; its message asserts a description ("the
# manuscript claims X"). Nothing compares the two, so the description can go
# stale while the suite stays green — the sentence is a string literal, and a
# passing test never prints it. Seven quotations in this file named sentences the
# paper does not contain, one of them ("is not a lucky seed") a sentence it never
# contained in any form; they were found by hand, and finding them by hand does
# not stop the next one.
#
# This renders main.tex with its macros expanded and checks that every sentence
# quoted inside a manuscript-attributed message is still in it. The convention it
# enforces: inside a message that names a manuscript section, quotation marks
# mean a quotation. Anything else the message wants to set off goes unquoted.
# ===========================================================================

_QUOTED = re.compile(r"""(?<![A-Za-z])'(.+?)'(?![A-Za-z])|"(.+?)\"""")
_INTERPOLATED = "\x00"           # stands in for an f-string's {} while scanning
_MIN_QUOTED_WORDS = 3            # a quoted phrase, not a column name or a code


def _rendered_manuscript():
    """{section label: that section's text}, macros expanded and markup stripped.

    What the document prints, which is what a quotation has to match: the
    manuscript body holds macro calls, not numbers, so a message quoting a
    sentence with a number in it can only be checked against the expanded text.
    Split by section because a message names one — a sentence that has moved to a
    different section is as stale a pointer as a sentence that has gone.
    """
    text = MAIN.read_text(encoding="utf-8")
    macros = dict(_params_defined())
    for generated in sorted((HERE / "paper" / "generated").glob("*.tex")):
        macros.update(re.findall(_MACRO % r"(\w+)", generated.read_text(encoding="utf-8")))
    text = re.sub(r"\\(\w+)\{\}", lambda m: macros.get(m.group(1), m.group(0)), text)
    for pattern, replacement in [
        (r"(?<!\\)%.*", ""),                            # comments, but not an escaped \%
        (r"\\label\{(sec:[^}]*)\}", "\x01\\1\x01"),     # section starts, kept as markers
        (r"\\(?:cite[a-z]*|label|ref|eqref)\{[^}]*\}", ""),
        (r"\\(?:emph|textbf|textit|mathrm|text)\{([^}]*)\}", r"\1"),
        (r"\{,\}", ","),                                # 274{,}000 -> 274,000
        (r"\\,", ""),                                   # thin space inside a printed CI
        (r"\\%", "%"), (r"\\&", "&"), (r"~", " "), (r"\$", ""),
        (r"``|''", '"'),
        (r"\s+", " "),
        (r"\s+([.,;:])", r"\1"),      # a sentence closing on a \citep loses its space
    ]:
        text = re.sub(pattern, replacement, text)
    parts = text.split("\x01")
    # parts[0] is the front matter before the first \label{sec:...}; after it the
    # split alternates label, body, label, body.
    sections = {"": parts[0]}
    sections.update(dict(zip(parts[1::2], parts[2::2])))
    return sections


def _assert_messages():
    """[(test name, message text)] for every assertion in this file that carries one.

    Reads its own source rather than running anything: a message on an assertion
    that passes is never evaluated, so the stale ones are exactly the ones no run
    reaches.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    messages = []
    for func in tree.body:
        if not (isinstance(func, ast.FunctionDef) and func.name.startswith("test_")):
            continue
        for node in ast.walk(func):
            if not (isinstance(node, ast.Assert) and node.msg is not None):
                continue
            parts = node.msg.values if isinstance(node.msg, ast.JoinedStr) else [node.msg]
            messages.append((func.name, "".join(
                p.value if isinstance(p, ast.Constant) and isinstance(p.value, str)
                else _INTERPOLATED
                for p in parts
            )))
    return messages


def _long_quotes(message):
    """The prose quotations in one message, minus the ones nothing could resolve."""
    for single, double in _QUOTED.findall(message):
        phrase = single or double
        if _INTERPOLATED in phrase or len(phrase.split()) < _MIN_QUOTED_WORDS:
            continue                                     # parameterized, or too short to be prose
        yield " ".join(phrase.split())


def _comment_blocks():
    """[(line number, text)] for every run of consecutive comment lines in this file.

    Tokenized rather than grepped, because this file quotes LaTeX and a `#` inside
    a string literal is not a comment.
    """
    with Path(__file__).open(encoding="utf-8") as fh:
        comments = {t.start[0]: t.string.lstrip("#").strip()
                    for t in tokenize.generate_tokens(fh.readline)
                    if t.type == tokenize.COMMENT}
    blocks, run, start = [], [], None
    for line in sorted(comments):
        if run and line == start + len(run):
            run.append(comments[line])
        else:
            if run:
                blocks.append((start, " ".join(run)))
            run, start = [comments[line]], line
    if run:
        blocks.append((start, " ".join(run)))
    return blocks


# A comment attributes a sentence to the paper in one form: the section label, a
# colon, then the quotation. That form alone does not separate a manuscript
# quotation from the others this file makes beside a section label -- a
# script header's outcome line, a run log, a phrase from the conventions file.
# What separates them is that a manuscript quotation is a manuscript sentence: it
# opens on a capital and closes on a full stop. A fragment quoted mid-comment is
# left to the message check, which reaches the quotations that carry a failure to
# a reader.
_ATTRIBUTED = re.compile(r"\b(sec:[\w-]+)\b[^:'\"]{0,40}:\s*(?=['\"])")
# A comment cannot interpolate, so it writes a macro's value as a bare N.
_COMMENT_PLACEHOLDER = re.compile(r"(?<![\w\\])N(?![\w])")


def _quoted_manuscript_comments():
    """[(site, quoted sentence, [label])] for every manuscript sentence a comment quotes.

    The message check below reads assertion messages, which is where a stale
    quotation misdirects someone chasing a failure. It is not where most of them
    are: this file explains itself in comments, they quote the paper as freely,
    and nothing evaluates a comment. A rewrite of the granularity results left four
    comment-quoted sentences pointing at prose that had gone, with the suite green.
    """
    claims = []
    for line, text in _comment_blocks():
        found = _ATTRIBUTED.search(text)
        if not found:
            continue
        for phrase in _long_quotes(text[found.end():]):
            if not (phrase[:1].isupper() and phrase.endswith((".", "!", "?"))):
                continue
            claims.append((f"{Path(__file__).name}:{line}",
                           _COMMENT_PLACEHOLDER.sub("...", phrase), [found.group(1)]))
    return claims


def _quoted_manuscript_claims():
    """[(test name, quoted phrase, labels)] for every sentence this file attributes to the paper."""
    claims = []
    for name, message in _assert_messages():
        labels = re.findall(r"\bsec:[\w-]+", message)
        if not labels and "anuscript" not in message:
            continue                                     # not attributed to the paper
        claims.extend((name, phrase, labels) for phrase in _long_quotes(message))
    return claims


def _quotes(rendered, phrase):
    """True iff `phrase` appears in `rendered`, allowing the two liberties a
    quotation may take: an ellipsis for elided words, and the case a phrase
    changes when a sentence-initial one is quoted mid-sentence."""
    at = 0
    for span in phrase.lower().split("..."):
        at = rendered.lower().find(span.strip(), at)
        if at < 0:
            return False
    return True


def test_manuscript_quotations_exist():
    sections = _rendered_manuscript()
    missing = []
    for test, phrase, labels in (_quoted_manuscript_claims()
                                 + _quoted_manuscript_comments()):
        unknown = [lab for lab in labels if lab not in sections]
        if unknown:
            missing.append(f"{test}: names {unknown}, which main.tex does not label")
            continue
        # A message naming sections must find its sentence in one of them; one that only
        # says "the manuscript" is checked against the whole document.
        searched = [sections[lab] for lab in labels] or list(sections.values())
        if not any(_quotes(text, phrase) for text in searched):
            where = " or ".join(labels) if labels else "main.tex"
            missing.append(f"{test}: {phrase!r} is not in {where}")
    assert not missing, (
        f"{len(missing)} message(s) or comment(s) attribute a sentence to a manuscript "
        f"section that does not contain it. A failing assertion would send its reader to a "
        f"sentence that is not there, and a comment explains the check against prose the "
        f"paper has dropped:\n  " + "\n  ".join(missing)
    )


# The manuscript is not the only thing this file quotes. A message that sends its
# reader to a section of a Markdown document such as README.md carries the identical
# defect and is invisible to the check above, which resolves \label{sec:...} and
# nothing else. Same shape, one artifact over.
_ARTIFACT = re.compile(r"\b([\w-]+\.md)\b(?:\s*(?:§|Sec\.?\s*)(\d+))?")


def _flatten_markdown(text):
    """Markdown as the prose a quotation can be matched against."""
    for pattern, replacement in [(r"[*`]", ""), ("×", "x"), ("−", "-"),
                                 ("≈", "~"), ("→", "->"), (r"\s+", " ")]:
        text = re.sub(pattern, replacement, text)
    return text.strip()


def _committed_document(name):
    """{section number: that section's text}, plus "" for the whole file.

    The markdown analogue of _rendered_manuscript. A message naming `file §N` is
    checked against section N alone, because a phrase that has moved to another
    section is as stale a pointer as one that has gone.
    """
    root = HERE / name
    hits = [root] if root.exists() else sorted(HERE.rglob(name))
    assert len(hits) == 1, (
        f"{name} resolves to {[str(h.relative_to(HERE)) for h in hits]}; a message "
        f"quoting it names no one file"
    )
    text = hits[0].read_text(encoding="utf-8")
    parts = re.split(r"^##\s+(\d+)\.\s", text, flags=re.M)
    sections = {"": _flatten_markdown(text)}
    sections.update({n: _flatten_markdown(body) for n, body in zip(parts[1::2], parts[2::2])})
    return sections


def _quoted_document_claims():
    """[(test, phrase, file, section)] for every phrase this file attributes to a document."""
    claims = []
    for name, message in _assert_messages():
        named = _ARTIFACT.findall(message)
        if len(named) != 1:
            continue          # nothing named, or more than one and no phrase belongs to a file
        doc, section = named[0]
        claims.extend((name, phrase, doc, section) for phrase in _long_quotes(message))
    return claims


def test_committed_document_quotations_exist():
    missing = []
    for test, phrase, doc, section in _quoted_document_claims():
        try:
            sections = _committed_document(doc)
        except AssertionError as exc:
            missing.append(f"{test}: {exc}")
            continue
        if section and section not in sections:
            missing.append(
                f"{test}: names {doc} §{section}, which has no numbered section {section}"
            )
            continue
        if not _quotes(sections[section], _flatten_markdown(phrase)):
            where = f"{doc} §{section}" if section else doc
            missing.append(f"{test}: {phrase!r} is not in {where}")
    assert not missing, (
        f"{len(missing)} assertion message(s) attribute a phrase to a committed document "
        f"that does not contain it. A failing assertion would send its reader to a "
        f"section of a document that does not hold the quoted phrase:\n  "
        + "\n  ".join(missing)
    )


def test_survey_rows_cite_a_reference_that_exists():
    # Every coded row backs a sentence about a study the manuscript cites, so a
    # cite_key with no entry is a claim about a source nobody can look up: a key is
    # only checkable once something resolves it. A scope filter inside a derived check
    # (`kind == "study"`) is a hand-written membership list in predicate form, so this
    # one runs over every row and the emptiness is the failure, not an exemption.
    keys = {re.match(r"@\w+\{([^,]+),", b).group(1)
            for b in re.findall(r"^@\w+\{[^,]+,", (HERE / "paper" / "refs.bib")
                                .read_text(encoding="utf-8"), re.M)}
    rows, _ = _survey_counts()
    blank = sorted(r["record_id"] for r in rows if not r["cite_key"])
    assert not blank, (
        f"survey_records.csv rows {blank} carry no cite_key; every row names "
        f"something the reader is told a number about, so each is a measurement "
        f"over a source that cannot be looked up"
    )
    missing = sorted({r["cite_key"] for r in rows} - keys)
    assert not missing, (
        f"survey_records.csv cites {missing}, which refs.bib does not define, so "
        f"each is a source a reader cannot look up"
    )





def test_availability_statement_matches_the_exemptions():
    # The availability statement is a claim about this repository, so it goes stale
    # the way any other claim does -- silently, and in the direction that flatters.
    # It said every number rested on a released table until the exemption list
    # existed to contradict it. Bind the count it prints to the list itself.
    ours = sorted(m for m, (who, _) in _params_exempt().items() if who == "this paper")
    word = {0: "no", 1: "one", 2: "two", 3: "three"}.get(len(ours), str(len(ours)))
    text = _rendered_manuscript()["sec:availability"]
    if not ours:
        assert "Every measurement that we report is read from these tables" in text \
                and "exception" not in text, (
            f"{PARAMS_EXEMPT.name} lists no macro that we measured without a shipped table, "
            f"so sec:availability should say 'Every measurement that we report is read from "
            f"these tables' and name no exception"
        )
        return
    assert f"{word} exception" in text, (
        f"Manuscript (sec:availability) says 'Every measurement that we report is read from "
        f"these tables' but {PARAMS_EXEMPT.name} now lists "
        f"{len(ours)} macro(s) we measured with no shipped table ({ours}); the sentence "
        f"must say '{word} exception' or the new entry must be given a source"
    )


def _git(*argv):
    """git output as text, or None where this copy carries no history."""
    if not (HERE / ".git").exists():
        return None
    return subprocess.run(("git", "-C", str(HERE)) + argv, check=True,
                          capture_output=True, text=True).stdout


def _tracked():
    """The files that this copy ships: the git index, or, in a release archive that
    carries no history, every file of the tree that .gitignore does not exclude."""
    listing = _git("ls-files")
    if listing is not None:
        return listing.split()
    rules = [r.strip() for r in (HERE / ".gitignore").read_text(encoding="utf-8").splitlines()
             if r.strip() and not r.startswith("#")]

    def ignored(rel):
        parts = rel.split("/")
        return any(r.rstrip("/") in parts[:-1] if r.endswith("/")
                   else fnmatch.fnmatch(rel, r) if "/" in r
                   else any(fnmatch.fnmatch(part, r) for part in parts)
                   for r in rules)

    files = (f.relative_to(HERE).as_posix() for f in HERE.rglob("*") if f.is_file())
    return sorted(f for f in files if not ignored(f))


def test_no_committed_source_claims_a_registry():
    # No plan was deposited in any registry, so no committed source may call the
    # criteria by the registry word: a term of art names an institution, not a
    # practice, and stale documentation is a bug of the same severity as a stale
    # formula. Invisible to every other check here, because the word prints no
    # number, so no macro binding reaches it.
    offenders = {}
    for path in _tracked():
        if not path.endswith((".py", ".md", ".tex", ".sh")):
            continue
        for n, line in enumerate((HERE / path).read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"pre-?regist", line, re.I):
                offenders.setdefault(path, []).append(n)
    assert not offenders, (
        "No plan for these criteria was deposited in any registry, so no committed "
        "file may use the registry word for them -- it asserts a third party that "
        "does not exist, and a reader of the released code meets it with no "
        "sentence anywhere to qualify it. State the criterion and its threshold, "
        "which bind by being stated and met; the timing words are not the fix, "
        "because each one re-asserts the same chronology no reader can audit. "
        "Found in:\n  "
        + "\n  ".join(f"{f}:{','.join(map(str, ns))}" for f, ns in offenders.items())
    )


# ---------------------------------------------------------------------------
# Checks that derive their expectation from an artifact --
# the script that ran, the per-seed CSV, version control -- rather than from the
# prose it guards, so reverting the prose fails the check rather than the reader.
# ---------------------------------------------------------------------------

def test_the_crop_exclusion_names_the_filter_that_fires():
    # sec:diagnostic explains why the timestamp-baseline comparison runs on fewer crops than the
    # dataset ships. That explanation is a Methods step, and the
    # script is the record of which step ran. timestamp_baseline_multiseed.py filters in a
    # fixed order -- drop rows with a missing panel channel, THEN require >=2
    # classes -- so a crop whose panel lacks a channel never reaches the class
    # rule, and attributing its absence to the class rule describes a filter the
    # code does not apply.
    script = (AIHUB / "timestamp_baseline_multiseed.py").read_text(encoding="utf-8")
    drop = script.index("dropna(subset=BIOCHEM")
    klass = script.index('nunique() < 2')
    assert drop < klass, (
        "timestamp_baseline_multiseed.py now applies the class rule before the panel dropna; "
        "sec:diagnostic's exclusion sentence describes the old order"
    )

    corpus_crops = json.loads(
        (AIHUB / "corpus_structure.json").read_text(encoding="utf-8"))["crops"]
    compared = pd.read_csv(TIMESTAMP_CSV)["crop"].nunique()
    assert corpus_crops - compared == 1, (
        f"sec:diagnostic accounts for exactly one excluded crop, but the dataset ships "
        f"{corpus_crops} and the comparison runs on {compared}"
    )

    # Both conditions hold of the excluded crop, so naming the class rule is not
    # false -- it is the wrong one to name alone, because the panel filter fires
    # first and is what the compared set is actually selected on. Methods describe
    # the steps the code runs, in the order it runs them.
    # Scoped to the section that makes the claim, not the document: a document-wide
    # search for the word can be satisfied by an unrelated sentence.
    text = re.sub(r"(?<!\\)%.*", "", MAIN.read_text(encoding="utf-8"))
    assert r"\label{sec:timestamp-methods}" in text, "main.tex no longer labels sec:timestamp-methods"
    body = re.split(r"\\(?:sub)?section\{", text.split(r"\label{sec:timestamp-methods}", 1)[-1], 1)[0]
    # In the manuscript's register, not the column name: code adopts the
    # manuscript's vocabulary, never the reverse.
    assert re.search(r"potassium|nutrient reading", body), (
        "sec:timestamp-methods must name the reason the script actually applies: the excluded "
        "crop ships no potassium reading, so its rows leave every analysis that "
        "requires all five soil-chemistry variables before the class rule is evaluated"
    )


def test_no_standing_raising_practice_claim_without_an_artifact():
    # A practice that raises a claim's standing -- a withholding,
    # a masked key, a held-out check -- is conferred by an artifact someone else
    # can inspect, not by the sentence asserting it. Nothing binds such a sentence
    # to a number, so the macro enumeration cannot see it: the check runs the other way,
    # from the phrase to the record that would have to exist.
    # Whitespace-normalized on both sides: the manuscript wraps, so a literal
    # substring match silently misses a phrase broken across a newline, and a check
    # that cannot see the claim it guards passes for the wrong reason. This file is
    # excluded from the corroborating set for the same reason -- it quotes the
    # phrases it searches for, so it would satisfy every one of them by itself.
    flat = lambda t: " ".join(re.sub(r"(?<!\\)%.*", "", t).split())
    text = flat(MAIN.read_text(encoding="utf-8"))
    phrases = {
        "withheld from all threshold setting":
            "a record naming the withheld crop and the thresholds it was kept out of",
        "out-of-sample check":
            "a record naming what was held out and what it checked",
    }
    tracked = _tracked()
    mine = Path(__file__).name
    elsewhere = [f for f in tracked
                 if f not in ("paper/main.tex", mine) and (HERE / f).is_file()]
    offenders = []
    for phrase, needs in phrases.items():
        if phrase not in text:
            continue
        found = [f for f in elsewhere
                 if phrase in flat((HERE / f).read_text(encoding="utf-8", errors="ignore"))]
        if not found:
            offenders.append(f"{phrase!r} -- needs {needs}; no tracked file records it")
    assert not offenders, (
        "The manuscript asserts a practice that raises the claim's standing while no "
        "artifact in the repository records it. Name the artifact or cut the "
        "sentence -- disclosure of the practice is not the practice:\n  "
        + "\n  ".join(offenders)
    )


# Names a repo file may point at without owning: the check cannot tell a path in
# this tree from one in somebody else's, so the ones that are not ours are decided
# here rather than left to a comment (an escape hatch a check enumerates).
_NAMED_PATH_EXEMPT = {
    "file.py":
        "the placeholder in the comment documenting the file:line check below",
}


def test_every_script_path_a_repo_file_names_resolves():
    # The weaker standard, the half the file:line check below skips as "not a path
    # this repo owns": every path a document *names* rather than opens -- a README
    # command, a docstring's cross-reference, an assertion's failure message. Nothing
    # dereferences these, so a deleted script leaves them pointing at nothing with
    # this suite green.
    # Scripts, documents and parameter files only. Data paths have their own check
    # (test_committed_artifacts_are_reachable_from_the_pipeline) and are where the
    # scratch and deposit names live, which no rule here separates from ours.
    # Derived from version control at both ends.
    tracked = _tracked()
    texts = {n: (HERE / n).read_text(encoding="utf-8", errors="replace")
             for n in tracked if n.endswith((".md", ".py", ".tex", ".sh", ".bib"))}
    top = {n.split("/")[0] for n in tracked}
    exts = (".py", ".md", ".sh", ".tex")
    tok = re.compile(r"(?<![\w*.-])([A-Za-z0-9][\w./-]*\.(?:py|md|sh|tex))\b")
    # A first segment this repo has no top-level entry for is another tree, and its
    # files are not ours to resolve. The upstream repository names them both ways --
    # src/model.py and a bare model.py -- so one prefixed mention licenses the bare
    # one, and a new upstream file needs no registration.
    foreign = {m[1] for text in texts.values()
               for m in re.findall(r"(?<![\w./-])((?:[\w.-]+/)+)([\w.-]+)", text)
               if m[0].split("/")[0] not in top}

    def resolves(cand, owner):
        base = cand.split("/")[-1]
        return (base in _NAMED_PATH_EXEMPT or base in foreign
                or (HERE / cand).exists() or (HERE / owner).parent.joinpath(cand).exists()
                or any(t == cand or t.endswith("/" + cand) for t in tracked))

    dangling = {}
    for name, text in texts.items():
        for i, line in enumerate(text.splitlines(), 1):
            for whole in tok.findall(line):
                # A slash joining two script names is prose, not one path.
                segs = [s for s in whole.split("/") if s.endswith(exts)]
                for cand in (segs if len(segs) > 1 else [whole]):
                    if not resolves(cand, name):
                        dangling.setdefault(cand, set()).add(f"{name}:{i}")
    assert not dangling, (
        f"{len(dangling)} path(s) a repository file names resolve to nothing. No "
        f"program dereferences them, so the pointer survives the file it names:\n  "
        + "\n  ".join(f"{c} — named at {sorted(s)}" for c, s in sorted(dangling.items()))
    )


def test_a_comment_citing_a_file_line_finds_its_value_there():
    # This extends "every path a repo document names must resolve" to the
    # sites a comment cites. A `file.py:NN` pointer is dereferenced by nobody, so
    # it rots silently: the line keeps existing while the thing it was cited for
    # moves. Where the comment says a value is inherited from that site, the site
    # has to carry the value.
    # Both phrasings a source file uses to cite a sibling line: "timestamp_baseline.py:115" and
    # "timestamp_baseline.py already used, line 115". The docstring form is the one a grep for
    # the punctuated form misses, and it carried the same dead pointer here.
    pat = re.compile(r"([\w./-]+\.py)(?::|,?\s+line\s+)(\d+)")
    offenders = []
    for path in sorted((HERE / "code").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for m in pat.finditer(text):
            target = next(iter(HERE.rglob(m.group(1))), None)
            if target is None:
                continue                       # not a path this repo owns
            tl = target.read_text(encoding="utf-8").splitlines()
            n, where = int(m.group(2)), text[:m.start()].count("\n") + 1
            if n > len(tl):
                offenders.append(
                    f"{path.name}:{where} cites {m.group(1)}:{n}, past that file's end")
                continue
            # The value the citation is offered as the source of, if it names one.
            near = text[max(0, m.start() - 220):m.end()]
            vals = re.findall(r"(?<![\w.])(\d+\.\d+)(?![\w.])", near)
            if vals and not any(v in tl[n - 1] for v in vals):
                offenders.append(
                    f"{path.name}:{where} cites {m.group(1)}:{n} as the source of "
                    f"{vals[-1]}, but that line is {tl[n - 1].strip()[:70]!r}")
    assert not offenders, (
        "a comment cites a file:line that does not carry what it is cited for:\n  "
        + "\n  ".join(offenders)
    )


def test_every_forest_predicts_on_one_thread():
    # A random forest that predicts on several threads sums its trees' votes in no
    # fixed order, so an exact tie resolves differently from run to run and a
    # committed table no longer regenerates byte for byte. Every forest in code/
    # therefore runs on one thread, and the scripts run the folds or seeds in
    # parallel. Derived from every forest that the scripts construct, read from the
    # syntax tree so that a docstring naming the estimator is not a call.
    calls, offenders = 0, []
    for path in sorted((HERE / "code").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", getattr(node.func, "attr", None))
                    == "RandomForestClassifier"):
                continue
            calls += 1
            n_jobs = {k.arg: k.value for k in node.keywords}.get("n_jobs")
            if not (isinstance(n_jobs, ast.Constant) and n_jobs.value == 1):
                offenders.append(f"{path.relative_to(HERE)}:{node.lineno}")
    assert calls, "no RandomForestClassifier call found under code/; the check reads nothing"
    assert not offenders, (
        "sec:availability says 'Every measurement that we report is read from these "
        "tables', but the committed result tables do not regenerate byte for byte from "
        "forests that are not fixed to one thread:\n  " + "\n  ".join(offenders)
    )


def _run_all():
    # Aggregating runner so a full audit shows EVERY violated claim, not just the
    # first. Only AssertionError (a claim violation) is collected; a missing file
    # or bad column propagates as a loud traceback (fail loud).
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures.append(t.__name__)
            print(f"FAIL  {t.__name__}\n      {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} claims verified")
    if failures:
        raise SystemExit(f"{len(failures)} manuscript claim(s) no longer hold: {failures}")


if __name__ == "__main__":
    _run_all()
