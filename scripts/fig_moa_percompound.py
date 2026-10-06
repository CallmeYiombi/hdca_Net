#!/usr/bin/env python3
"""Per-compound mechanism recovery figure (main text).

Panel A  16 probe drugs x 4 mask conditions, cell = runs in which the expected
         pathway reached the top 10.  Sequential single-hue ramp.
Panel B  median rank of the expected pathway, broad -> direct-target, log scale.

Data:  moa_summary_all.csv  (per-run `ranks`: 16 values, one per probe drug).

NOTE on drug order: the `ranks` column is NOT in the alphabetical order of
results/moa_probe_drugs.csv.  The order below was recovered from the published
broad-mask medians and independently confirmed against the direct-target
medians (16/16).  Do not reorder without re-running that check.
"""
import csv
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RANK_ORDER = ["trametinib", "dabrafenib", "erlotinib", "rucaparib", "lapatinib",
              "afatinib", "gefitinib", "vorinostat", "olaparib", "nilotinib",
              "crizotinib", "nutlin-3a", "selumetinib", "pictilisib", "imatinib",
              "panobinostat"]

# tag -> (column label, n runs).  Verified against the paper: brd 15.6%,
# prn 49.0%, dns 0.0% hit rate over the same six seeds.
CONDITIONS = [("brd", "broad\nmask"), ("prn", "direct\ntargets"),
              ("dns", "no mask\n(dense)"), ("sham", "randomized\ntargets")]
SHAM_TAGS = [f"random_mask_{i}" for i in range(1, 6)]

TOP_K = 10                      # a hit is a rank <= 10
N_PATHWAYS = 3143
CHANCE = 3.9                    # % , derived in Supplementary Methods

# --- palette (dataviz reference instance) -----------------------------------
SEQ = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE = "#ffffff"
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)


def load(path="moa_summary_all.csv"):
    rows = list(csv.DictReader(open(path)))
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append([float(v) for v in r["ranks"].split()])
    by_tag["sham"] = [x for t in SHAM_TAGS for x in by_tag[t]]
    return by_tag


def main():
    by_tag = load()
    hits, meds = {}, {}
    for tag, _ in CONDITIONS:
        runs = by_tag[tag]
        hits[tag] = [sum(1 for r in runs if r[i] <= TOP_K) for i in range(16)]
        meds[tag] = [st.median([r[i] for r in runs]) for i in range(16)]
        n = len(runs)
        rate = 100 * sum(hits[tag]) / (16 * n)
        print(f"{tag:5s} n_runs={n}  hit rate={rate:5.2f}%")

    # rows ordered by direct-target median rank, as in the probe table
    idx = sorted(range(16), key=lambda i: (meds["prn"][i], meds["brd"][i]))

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(7.1, 4.5), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.05})
    fig.patch.set_facecolor(SURFACE)

    # ---------------- Panel A : heatmap ----------------
    axA.set_facecolor(SURFACE)
    for c, (tag, _) in enumerate(CONDITIONS):
        n = len(by_tag[tag])
        for r, i in enumerate(idx):
            frac = hits[tag][i] / n
            axA.add_patch(plt.Rectangle((c + .02, r + .02), .96, .96,
                                        facecolor=CMAP(frac), edgecolor=SURFACE, lw=1.2))
            if hits[tag][i]:
                axA.text(c + .5, r + .5, f"{hits[tag][i]}/{n}", ha="center", va="center",
                         fontsize=6.6, color="#ffffff" if frac > .55 else INK)
    axA.set_xlim(0, 4); axA.set_ylim(16, 0)
    axA.set_xticks([c + .5 for c in range(4)])
    axA.set_xticklabels([lab for _, lab in CONDITIONS], fontsize=6.8, color=INK2)
    axA.set_yticks([r + .5 for r in range(16)])
    axA.set_yticklabels([RANK_ORDER[i] for i in idx], fontsize=7, color=INK)
    axA.tick_params(length=0)
    for s in axA.spines.values():
        s.set_visible(False)
    axA.set_title("A   runs recovering the expected pathway",
                  fontsize=7.8, color=INK, loc="left", pad=8)

    # ---------------- Panel B : dumbbell ----------------
    axB.set_facecolor(SURFACE)
    axB.set_xscale("log")
    for r, i in enumerate(idx):
        b, d = meds["brd"][i], meds["prn"][i]
        axB.plot([b, d], [r + .5, r + .5], color=MUTED, lw=1.0, zorder=1)
        axB.scatter([b], [r + .5], s=26, color=BLUE, zorder=2,
                    edgecolor=SURFACE, linewidth=.8)
        axB.scatter([d], [r + .5], s=26, color=ORANGE, zorder=3,
                    edgecolor=SURFACE, linewidth=.8)
    axB.axvline(TOP_K, color=INK2, lw=.9, ls=(0, (3, 2)), zorder=0)
    axB.text(TOP_K * 0.85, 0.25, "top 10", fontsize=6.4, color=INK2,
             va="center", ha="right")
    axB.set_xlim(.8, N_PATHWAYS * 1.4); axB.set_ylim(16, 0)
    axB.set_yticks([]); axB.tick_params(axis="x", labelsize=6.6, colors=INK2, length=2)
    axB.set_xlabel(f"median rank of expected pathway  (of {N_PATHWAYS:,})",
                   fontsize=7, color=INK2)
    axB.grid(axis="x", color="#e8e8e4", lw=.6, zorder=0)
    axB.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axB.spines[s].set_visible(False)
    axB.spines["bottom"].set_color("#d8d8d4")
    axB.set_title("B   where that pathway ranks", fontsize=7.8, color=INK, loc="left", pad=8)
    axB.scatter([], [], s=26, color=BLUE, label="broad mask")
    axB.scatter([], [], s=26, color=ORANGE, label="direct targets")
    leg = axB.legend(fontsize=6.6, frameon=False, loc="lower left",
                     handletextpad=.35, borderpad=.2, labelspacing=.35)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.subplots_adjust(left=.145, right=.985, top=.90, bottom=.115)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_moa_percompound.{ext}", dpi=400,
                    facecolor=SURFACE)
    print("wrote figures/fig_moa_percompound.{pdf,png}")


if __name__ == "__main__":
    main()
