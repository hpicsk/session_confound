#!/usr/bin/env python3
"""Draw the manuscript's four figures from the committed result tables.

Every figure that paper/main.tex includes is written here, into paper/generated/,
the directory the document reads them from, so a re-run overwrites the published
figure in place. The script reads only the imagery-free tables under code/ and
paper/params.tex; it needs no dataset, no GPU, and no environment variable, but it
needs the Liberation Sans font (FONT_FAMILY) and stops if matplotlib cannot find it.

  fig_concept.pdf      sec:intro, sec:methods -- how one sensor reading is attached
                       to many images of a session, and the two parts of the diagnostic
  fig_granularity.pdf  sec:res-diagnostic -- (a) the share of each AI Hub panel
                       variable's variance that farm explains; (b) CDD images by
                       how many images share their environment series
  fig_timestamp.pdf    sec:res-timestamp -- three accuracy margins per crop: the timestamp
                       classifier minus the soil-chemistry and the full-panel classifiers, and the
                       farm-and-timestamp classifier minus the full-panel classifier
  fig_cdd.pdf          sec:res-cdd -- macro-F1 of the three CDD predictors under
                       the two splits we draw and the study's own split, against
                       the two macro-F1 values the study reports

Degrees of freedom:
  intervals ....... MEASURED by the scripts that wrote each table; recomputed here
                    with the t rule over seeds that test_assertions.py applies, so a
                    plotted interval is the interval the suite checks.
  reference lines . the CDD study's reported values, read from paper/params.tex
                    (\\paperImageFone, \\paperMultiFone), which the text prints.
  SERIES_BINS ..... CHOSEN, display only: the bins of panel (b) of fig_granularity.
                    Every image falls in exactly one bin, so the bins change no
                    count, only how the counts are grouped.
  layout, colors .. CHOSEN, display only.
  nothing FITTED.

Run:  python code/figures.py
"""
import ast
import json
import math
import re
from pathlib import Path

import matplotlib
import matplotlib.font_manager
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AIHUB = HERE / "aihub_subtropical"
CDD = HERE / "dacon_cropdisease"
OUT = ROOT / "paper" / "generated"          # the directory main.tex includes from

# Figure widths. The journal's guide asks for images "near to the desired size of the
# printed version" and gives no width, so one column is 85 mm and two columns 180 mm.
MM = 1 / 25.4
ONE_COLUMN = 85 * MM
TWO_COLUMN = 180 * MM
FONT_PT = 8            # the smallest text at final size
# The journal's guide: "Only use the following fonts in your illustrations: Arial,
# Courier, Times, Symbol." Liberation Sans has Arial's metrics; matplotlib would
# otherwise fall back to DejaVu Sans with only a warning, so style() stops instead.
FONT_FAMILY = "Liberation Sans"
LABEL_PT = 9           # axis labels and panel headings

# The validated categorical slots (dataviz reference palette, light mode), in their
# fixed order, plus the ink and surface tones. Every series also carries a distinct
# marker, so no identity rests on color alone.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID = "#e1e0d9"

# The crops' English names, in the words the manuscript uses.
CROP_NAMES = {"레몬": "Lemon", "무화과": "Fig", "바나나": "Banana", "백향과": "Passion fruit",
              "애플망고": "Apple mango", "올리브": "Olive", "용과": "Dragon fruit"}

# The AI Hub panel columns, in the manuscript's words (sec:corpora).
PANEL_LABELS = {"soil_pH": "pH", "soil_N": "N", "soil_P": "P", "soil_K": "K",
                "ms_gec": "Electrical conductivity", "soil_condition": "Soil moisture",
                "soil_temp": "Soil temperature", "temp": "Air temperature",
                "hum": "Air humidity"}

# Panel (b) of fig_granularity: bins of "images that share one environment series".
SERIES_BINS = [(1, 1), (2, 5), (6, 10), (11, 20), (21, None)]

# The study's own split, the identifier set its training script reads.
THEIR_SPLIT = "paper_work_seed777"


def script_const(path, name):
    """A module-level constant read from the script that ran with it, unexecuted."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{path.name} defines no module constant {name}")


def param(name):
    """A number paper/params.tex defines, so a figure and the text print one value."""
    text = (ROOT / "paper" / "params.tex").read_text(encoding="utf-8")
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", text)
    if not m:
        raise KeyError(f"paper/params.tex defines no macro \\{name}")
    return float(m.group(1).replace("{,}", ""))


def t_interval(x):
    """Mean and 95% t interval over seeds: the rule test_assertions.py applies."""
    x = np.asarray(x, dtype=float)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return x.mean(), x.mean() - half, x.mean() + half


def style():
    for weight in ("normal", "bold"):
        props = matplotlib.font_manager.FontProperties(family=FONT_FAMILY, weight=weight)
        try:
            matplotlib.font_manager.findfont(props, fallback_to_default=False)
        except ValueError as err:
            raise SystemExit(f"matplotlib finds no {weight} {FONT_FAMILY}; install it "
                             f"(see README.md) and delete matplotlib's font cache") from err
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_PT,
        "axes.labelsize": LABEL_PT,
        "axes.titlesize": LABEL_PT,
        "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,          # embed TrueType, so the text stays editable
        "legend.frameon": False,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    # No creation date, so a re-run that changes nothing writes the same bytes.
    fig.savefig(OUT / name, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def panel_letter(ax, letter):
    """The panel label, outside the plotting area, as the guidelines ask."""
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=LABEL_PT + 1,
            fontweight="bold", ha="right", va="bottom", color=INK)


# ---------------------------------------------------------------------------
# fig:concept -- concept
# ---------------------------------------------------------------------------
def fig_concept():
    fig = plt.figure(figsize=(TWO_COLUMN, 78 * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 78)
    ax.axis("off")

    # (a) Two sessions, each with two sensor readings and the images taken after them.
    ax.text(3, 74, "a", fontsize=LABEL_PT + 1, fontweight="bold", va="top", color=INK)
    ax.text(9, 74, "One sensor reading is attached to many images", fontsize=LABEL_PT,
            va="top", color=INK)
    rows = [(54, "Farm 1, day 1", SLOT[0], "A"), (26, "Farm 2, day 2", SLOT[1], "B")]
    for y, name, color, label in rows:
        ax.text(3, y, name, fontsize=FONT_PT, va="center", color=INK)
        ax.annotate("", xy=(108, y), xytext=(24, y),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
        ax.text(108, y - 3.2, "time", fontsize=FONT_PT, ha="right", va="top", color=MUTED)
        for k, x0 in enumerate((27, 66)):
            fill = color if k == 0 else _lighter(color)
            ax.plot(x0, y, marker="D", ms=7, color=fill, mec=INK_2, mew=0.6, zorder=3)
            for j in range(4):
                xi = x0 + 7 + 8 * j
                ax.add_patch(Rectangle((xi - 2.6, y + 2.2), 5.2, 5.2, facecolor=fill,
                                       edgecolor=INK_2, lw=0.6))
        ax.text(24, y - 7.5, f"Every image of the session is labeled disease class {label}",
                fontsize=FONT_PT, va="center", color=INK_2)
    handles = [Line2D([], [], marker="D", ls="", ms=6, color=MUTED, mec=INK_2, mew=0.6,
                      label="Sensor reading"),
               Patch(facecolor=MUTED, edgecolor=INK_2, lw=0.6,
                     label="Image, filled with the reading it carries")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(22 / 180, 0.0),
              ncol=2, handlelength=1.4, columnspacing=1.6)

    # (b) The two parts of the diagnostic, both on metadata and labels.
    ax.text(118, 74, "b", fontsize=LABEL_PT + 1, fontweight="bold", va="top", color=INK)
    ax.text(124, 74, "Two parts of the diagnostic", fontsize=LABEL_PT, va="top", color=INK)
    boxes = [(41, "Granularity probe",
              "Counts the images and the\nsessions that share each\nreading, and the share of each\n"
              "variable's variance that farm\nidentity explains"),
             (10, "Timestamp baseline",
              "Predicts the label from the\ntimestamp alone and compares\n"
              "the prediction with one made\nfrom the sensor readings")]
    for y0, title, body in boxes:
        ax.add_patch(FancyBboxPatch((118, y0), 59, 27, boxstyle="round,pad=0,rounding_size=2",
                                    facecolor="white", edgecolor=INK_2, lw=0.8))
        ax.text(121, y0 + 24, title, fontsize=FONT_PT, fontweight="bold", va="top", color=INK)
        ax.text(121, y0 + 19.5, body, fontsize=FONT_PT, va="top", color=INK_2, linespacing=1.2)
    ax.add_patch(FancyArrowPatch((111, 40), (117, 40), arrowstyle="-|>", mutation_scale=9,
                                 color=INK_2, lw=0.8))
    save(fig, "fig_concept.pdf")


def _lighter(hex_color, amount=0.45):
    """The same hue mixed toward white, for the second reading of a session."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = [round(c + (255 - c) * amount) for c in (r, g, b)]
    return "#" + "".join(f"{c:02x}" for c in mix)


# ---------------------------------------------------------------------------
# fig:granularity -- granularity on both datasets
# ---------------------------------------------------------------------------
def fig_granularity():
    shares = json.loads((AIHUB / "corpus_structure.json").read_text(encoding="utf-8"))
    shares = shares["r2_farm_panel"]
    order = sorted(shares, key=shares.get)
    ids = pd.read_parquet(CDD / "dacon_series_ids.parquet")
    per_series = ids.groupby("val_md5").size()            # images sharing each series
    images_in = []
    for lo, hi in SERIES_BINS:
        inside = per_series[(per_series >= lo) & ((per_series <= hi) if hi else True)]
        images_in.append(int(inside.sum()))
    assert sum(images_in) == len(ids), "every CDD image belongs to exactly one bin"

    fig, (a, b) = plt.subplots(1, 2, figsize=(TWO_COLUMN, 62 * MM),
                               gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.55))
    y = np.arange(len(order))
    # The five soil-chemistry columns that the soil-chemistry classifier reads (sec:diagnostic),
    # read from the script that ran the comparison rather than restated here.
    read = set(script_const(ROOT / "code" / "aihub_subtropical" / "timestamp_baseline_multiseed.py", "BIOCHEM"))
    colors = [SLOT[0] if c in read else MUTED for c in order]
    a.barh(y, [shares[c] for c in order], height=0.62, color=colors)
    a.set_yticks(y, [PANEL_LABELS[c] for c in order])
    a.axvline(0.5, color=INK_2, lw=0.8, ls=(0, (3, 2)))
    a.set_xlim(0, 1)
    a.set_xlabel("Share of variance explained by farm")
    a.tick_params(axis="y", length=0)
    a.legend(handles=[Patch(color=SLOT[0], label="Read by the soil-chemistry classifier"),
                      Patch(color=MUTED, label="Other panel variables")],
             loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=1, borderaxespad=0)
    a.text(-0.5, 1.02, "a", transform=a.transAxes, fontsize=LABEL_PT + 1,
           fontweight="bold", ha="left", va="bottom", color=INK)

    labels = [f"{lo}" if lo == hi else (f"{lo}–{hi}" if hi else f"≥{lo}")
              for lo, hi in SERIES_BINS]
    x = np.arange(len(SERIES_BINS))
    b.bar(x, images_in, width=0.62, color=SLOT[0])
    b.set_xticks(x, labels)
    b.set_xlabel("Images sharing one series")
    b.set_ylabel("CDD images")
    b.yaxis.grid(True, color=GRID, lw=0.8)
    b.set_axisbelow(True)
    b.tick_params(axis="x", length=0)
    b.text(-0.3, 1.02, "b", transform=b.transAxes, fontsize=LABEL_PT + 1,
           fontweight="bold", ha="left", va="bottom", color=INK)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.84)
    save(fig, "fig_granularity.pdf")


# ---------------------------------------------------------------------------
# fig:timestamp -- timestamp minus soil chemistry, per crop
# ---------------------------------------------------------------------------
def fig_timestamp():
    df = pd.read_csv(AIHUB / "timestamp_vs_panel_multiseed.csv")
    pairs = (("timestamp", "panel", SLOT[0], "o", "Timestamp minus soil chemistry"),
             ("timestamp", "panel_full", SLOT[1], "D", "Timestamp minus full panel"),
             ("farm_ts", "panel_full", SLOT[2], "s", "Farm and timestamp minus full panel"))
    # Per pair, the paired t-test over seeds per crop and Holm's correction across the
    # crops (sec:stats); a margin that does not survive the correction is drawn hollow.
    stat = {}
    for a, b, *_ in pairs:
        diffs = {crop: g.sort_values("seed")[a].to_numpy() - g.sort_values("seed")[b].to_numpy()
                 for crop, g in df.groupby("crop")}
        pv = {c: stats.ttest_1samp(d, 0).pvalue for c, d in diffs.items()}
        order, rej = sorted(pv, key=pv.get), {}
        for k, c in enumerate(order):
            rej[c] = bool(pv[c] <= 0.05 / (len(order) - k) and all(rej.values()))
        stat[(a, b)] = {c: (t_interval(d), rej[c]) for c, d in diffs.items()}
    crops = sorted(stat[("timestamp", "panel")], key=lambda c: stat[("timestamp", "panel")][c][0][0])

    fig, ax = plt.subplots(figsize=(ONE_COLUMN, 100 * MM))
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    y = np.arange(len(crops))
    for offset, (a, b, color, marker, name) in zip((0.24, 0.0, -0.24), pairs):
        for i, c in enumerate(crops):
            (mean, lo, hi), sig = stat[(a, b)][c]
            ax.errorbar(mean, y[i] + offset, xerr=[[mean - lo], [hi - mean]], fmt=marker, ms=4.5,
                        color=color, mfc=color if sig else "white", mec=color if not sig else "white",
                        mew=0.8, elinewidth=1.0, capsize=0, zorder=3,
                        label=name if i == 0 else None)
    ax.set_yticks(y, [CROP_NAMES[c] for c in crops])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Accuracy margin")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.42, 1.02), ncol=1,
              handletextpad=0.4, columnspacing=1.0, frameon=False)
    fig.subplots_adjust(left=0.3, right=0.97, bottom=0.12, top=0.74)
    save(fig, "fig_timestamp.pdf")


# ---------------------------------------------------------------------------
# fig:cdd -- the CDD predictors on three splits
# ---------------------------------------------------------------------------
def fig_cdd():
    drawn = pd.read_csv(CDD / "dacon_published_split.csv")
    theirs = pd.read_csv(CDD / "dacon_their_split.csv")
    theirs = theirs[theirs["split"] == THEIR_SPLIT]
    splits = [("Random split (ours)", drawn[drawn["regime"] == "random"], SLOT[0], "o"),
              ("Group-disjoint split (ours)", drawn[drawn["regime"] == "group"], SLOT[1], "s"),
              ("The study's own split", theirs, SLOT[2], "^")]
    arms = [("env_f1", "Environment"), ("timestamp_f1", "Timestamp"),
            ("timestamp_env_f1", "Timestamp and environment")]
    reported = [(param("paperImageFone"), "Study's image model", (0, (3, 2))),
                (param("paperMultiFone"), "Study's fusion model", (0, (1, 1.5)))]

    fig, ax = plt.subplots(figsize=(TWO_COLUMN, 58 * MM))
    y = np.arange(len(arms))[::-1]
    for (name, frame, color, marker), offset in zip(splits, (0.22, 0.0, -0.22)):
        means, los, his = zip(*(t_interval(frame[col]) for col, _ in arms))
        means, los, his = map(np.array, (means, los, his))
        ax.errorbar(means, y + offset, xerr=[means - los, his - means], fmt=marker, ms=5,
                    color=color, mec="white", mew=0.6, elinewidth=1.0, capsize=0,
                    label=name, zorder=3)
    for value, name, dash in reported:
        ax.axvline(value, color=INK_2, lw=0.9, ls=dash, zorder=1, label=name)
    ax.set_yticks(y, [label for _, label in arms])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Macro-F1")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=2.2)
    fig.subplots_adjust(left=0.24, right=0.71, bottom=0.21, top=0.96)
    save(fig, "fig_cdd.pdf")


def main():
    style()
    fig_concept()
    fig_granularity()
    fig_timestamp()
    fig_cdd()
    print(f"wrote {sorted(p.name for p in OUT.glob('fig_*.pdf'))} to {OUT}")


if __name__ == "__main__":
    main()
