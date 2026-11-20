#!/usr/bin/env python3
"""Write the Supplementary Material's generated tables from the committed result tables.

Each table is written into paper/generated/, the directory that paper/supplementary.tex
inputs it from, so a re-run overwrites the published table in place. The script reads only
the imagery-free tables under code/; it needs no dataset, no GPU, and no environment
variable.

  supp_scores.tex       per-crop accuracy and balanced accuracy of the five AI Hub predictors
                        (timestamp_vs_panel_multiseed.csv)
  supp_margins.tex      per-crop margins of three pairs of AI Hub classifiers, untuned forests
                        (timestamp_vs_panel_multiseed.csv)
  supp_tuned.tex        the same margins with every forest tuned (timestamp_vs_panel_tuned.csv)
  supp_floors.tex       the within-session measurement at three session minimums
                        (within_session_floor_sweep.csv)
  supp_leak.tex         leak rates of the two identifier sets of the CDD study's repository
                        (dacon_audited_split_leakage.csv)
  supp_maxfeatures.tex  the CDD forests on the study's split over the max_features grid
                        (dacon_their_split_tuned.csv)

Degrees of freedom:
  intervals ....... MEASURED by the scripts that wrote each table; recomputed here with the
                    t rule over seeds that test_assertions.py applies, so a printed interval
                    is the interval the suite checks.
  Holm marks ...... the paired t-test over seeds per crop, Holm's correction across the
                    crops at 0.05, as in the manuscript's statistics paragraph (sec:stats).
  precision ....... CHOSEN, display only: three decimals for the AI Hub scores and margins,
                    the precision of the manuscript's AI Hub values; four for the CDD study's
                    split, the precision of the study's reported figures; one for percentages.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
AIHUB = ROOT / "code" / "aihub_subtropical"
CDD = ROOT / "code" / "dacon_cropdisease"
OUT = ROOT / "paper" / "generated"

HOLM_ALPHA = 0.05   # CHOSEN: the conventional level, the level of the 95% intervals (sec:stats)

CROPS = {"애플망고": "Apple mango", "바나나": "Banana", "용과": "Dragon fruit", "무화과": "Fig",
         "레몬": "Lemon", "올리브": "Olive", "백향과": "Passion fruit"}


def t_interval(x):
    x = np.asarray(x, dtype=float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def holm(pvalues, alpha=HOLM_ALPHA):
    """{key: rejected} under Holm's step-down correction."""
    order = sorted(pvalues, key=pvalues.get)
    rej, m = {}, len(order)
    for k, key in enumerate(order):
        rej[key] = bool(pvalues[key] <= alpha / (m - k) and all(rej.values()))
    return rej


def signed(v, d=3):
    s = f"{v:+.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def cell(x, d=3, mark=False, sign=True):
    """mean [95% interval]; a difference prints its sign, a score does not."""
    m, lo, hi = t_interval(x)
    f = (lambda v: signed(v, d)) if sign else (lambda v: f"{v:.{d}f}")
    return f"{f(m)}{'*' if mark else ''} [{f(lo)}, {f(hi)}]"


PAIRS = (("timestamp", "panel", "Timestamp minus soil chemistry"),
         ("timestamp", "panel_full", "Timestamp minus full panel"),
         ("farm_ts", "panel_full", "Farm and timestamp minus full panel"))


def margins_table(long, path):
    """long: one row per crop and seed, accuracy columns <arm>, balanced accuracy <arm>_bal."""
    col = r">{\centering\arraybackslash}p{0.25\textwidth}"
    lines = [r"\begin{tabular}{l" + col * len(PAIRS) + "}", r"\toprule",
             " & " + " & ".join(p[2] for p in PAIRS) + r" \\", r"\midrule"]
    for sfx, name in (("", "Accuracy"), ("_bal", "Balanced accuracy")):
        lines.append(rf"\multicolumn{{{len(PAIRS) + 1}}}{{l}}{{\emph{{{name}}}}} \\")
        diffs = {p: {c: (g.sort_values("seed")[p[0] + sfx].to_numpy()
                         - g.sort_values("seed")[p[1] + sfx].to_numpy())
                     for c, g in long.groupby("crop")} for p in PAIRS}
        marks = {p: holm({c: stats.ttest_1samp(d, 0).pvalue for c, d in diffs[p].items()})
                 for p in PAIRS}
        for crop in sorted(diffs[PAIRS[0]], key=CROPS.get):
            lines.append(CROPS[crop] + " & " + " & ".join(
                cell(diffs[p][crop], mark=marks[p][crop]) for p in PAIRS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scores_table(long, path):
    arms = (("base", "Majority class"), ("panel", "Soil chemistry"), ("timestamp", "Timestamp"),
            ("panel_full", "Full panel"), ("farm_ts", "Farm and timestamp"))
    col = r">{\centering\arraybackslash}p{0.12\textwidth}"
    lines = [r"\begin{tabular}{l" + col * len(arms) + "}", r"\toprule",
             " & " + " & ".join(a[1] for a in arms) + r" \\", r"\midrule"]
    for sfx, name in (("", "Accuracy"), ("_bal", "Balanced accuracy")):
        lines.append(rf"\multicolumn{{{len(arms) + 1}}}{{l}}{{\emph{{{name}}}}} \\")
        for crop, g in sorted(long.groupby("crop"), key=lambda kv: CROPS[kv[0]]):
            lines.append(CROPS[crop] + " & " + " & ".join(
                f"{g[a + sfx].mean():.3f}" for a, _ in arms) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def floors_table(path):
    df = pd.read_csv(AIHUB / "within_session_floor_sweep.csv")
    df["pooled"] = df["panel_kept"] - df["base_kept"]
    df["within"] = df["panel_in_sess"] - df["base_in_sess"]
    lines = [r"\begin{tabular}{rcccc}", r"\toprule",
             r"Session minimum (rows) & Pooled & Within sessions & Lowest crop & "
             r"Highest crop \\", r"\midrule"]
    for floor, g in df.groupby("floor"):
        per_seed = g.groupby("seed")[["pooled", "within"]].mean()
        crop = g.groupby("crop")["within"].mean()
        lines.append(f"{floor:,}".replace(",", "{,}") + f" & {per_seed['pooled'].mean():.3f} & "
                     f"{per_seed['within'].mean():.3f} & {crop.min():.3f} & "
                     f"{crop.max():.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def leak_table(path):
    df = pd.read_csv(CDD / "dacon_audited_split_leakage.csv")
    sets = {"paper_work_seed777": "The study's split", "paper_work_seed42": "The other set"}
    keys = {"val_md5": "values", "csv_md5": "raw files"}
    col = r">{\centering\arraybackslash}p{0.105\textwidth}"
    lines = [r"\begin{tabular}{ll" + col * 5 + "}", r"\toprule",
             r" & Series matched by & Into training & Same class & Training and validation "
             r"& Identifiers in training & Identifiers in training and validation \\",
             r"\midrule"]
    for tag, name in sets.items():
        for key, kname in keys.items():
            r = df[(df["tag"] == tag) & (df["key"] == key)].iloc[0]
            lines.append(f"{name} & {kname} & {100 * r.breach:.1f} & {100 * r.breach_lab:.1f} & "
                         f"{100 * r.breach_trval:.1f} & {100 * r.direct:.1f} & "
                         f"{100 * r.direct_trval:.1f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maxfeatures_table(path):
    df = pd.read_csv(CDD / "dacon_their_split_tuned.csv")
    grid = list(dict.fromkeys(df["max_features"]))
    arms = (("timestamp", "Timestamp"), ("env", "Environment"),
            ("timestamp_env", "Timestamp and environment"))
    head = {"sqrt": r"$\sqrt{p}$"}
    col = r">{\centering\arraybackslash}p{0.17\textwidth}"
    lines = [r"\begin{tabular}{l" + col * (len(arms) + 1) + "}", r"\toprule",
             r"Variables per node & " + " & ".join(a[1] for a in arms)
             + r" & Timestamp minus timestamp and environment \\", r"\midrule"]

    def col(arm, sel):
        g = df[(df["arm"] == arm) & sel].sort_values("seed")
        return g["test_f1"].to_numpy()

    for name, sel in [(head.get(m, m + "$p$"), df["max_features"] == m) for m in grid] + [
            ("Chosen on validation", df["selected"])]:
        cells = [cell(col(arm, sel), 4, sign=False) for arm, _ in arms]
        cells.append(cell(col("timestamp", sel) - col("timestamp_env", sel), 4))
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    untuned = pd.read_csv(AIHUB / "timestamp_vs_panel_multiseed.csv")
    folds = pd.read_csv(AIHUB / "timestamp_vs_panel_tuned.csv")
    tuned = folds.groupby(["crop", "seed", "arm"])[["acc", "bal"]].mean().unstack("arm")
    tuned.columns = [arm if m == "acc" else f"{arm}_bal" for m, arm in tuned.columns]
    tuned = tuned.reset_index()
    scores_table(untuned, OUT / "supp_scores.tex")
    margins_table(untuned, OUT / "supp_margins.tex")
    margins_table(tuned, OUT / "supp_tuned.tex")
    floors_table(OUT / "supp_floors.tex")
    leak_table(OUT / "supp_leak.tex")
    maxfeatures_table(OUT / "supp_maxfeatures.tex")


if __name__ == "__main__":
    main()
