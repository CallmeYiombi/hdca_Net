#!/usr/bin/env python3
"""Encoded versus post-hoc mechanism recovery (main text, Figure 3).

Hit rates cannot be compared directly across models whose candidate spaces differ by two
orders of magnitude (3,143 HCDT pathways against 34 KEGG cancer pathways), so the axis is
each model's recovery divided by the chance level of its own vocabulary, with the raw rates
carried alongside as text.

Data: moa_summary_all.csv, results/moa_path_attn.csv, results/drpreter_moa_6seed.json.
"""
import csv
import json
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANCE = {"HDCA-Net": 3.9, "DRPreter": 16.5}


def load():
    ga = [float(r["hit_rate"]) for r in csv.DictReader(open("moa_summary_all.csv"))
          if r["tag"] == "prn"]
    pa = [float(r["hit_rate"]) for r in csv.DictReader(open("results/moa_path_attn.csv"))
          if r["tag"] == "prn"]
    d = json.load(open("results/drpreter_moa_6seed.json"))
    # row 0 is drawn at the bottom, so list in reverse reading order
    return [("DRPreter", "pathway attention", d["path_seed_rates"]),
            ("DRPreter", "gene $\\rightarrow$ pathway", d["genepath_seed_rates"]),
            ("HDCA-Net", "pathway attention", pa),
            ("HDCA-Net", "gene $\\rightarrow$ pathway", ga)]


def main():
    rows = load()
    y = np.arange(len(rows))

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6.6, 2.8))

    for i, (model, _, vals) in enumerate(rows):
        c = CHANCE[model]
        e = [v / c for v in vals]
        mu, sd = st.mean(e), st.stdev(e)
        ax.plot([max(mu - sd, 1e-3), mu + sd], [i, i], color="black", lw=1.1, zorder=2)
        ax.plot(e, [i] * len(e), "o", ms=3, mfc="white", mec="#888888", mew=0.6, zorder=3)
        ax.plot([mu], [i], "o", ms=7.5, mec="black", mew=1.1, zorder=4,
                mfc="black" if model == "HDCA-Net" else "white")
        ax.text(mu, i + 0.26, f"{mu:.1f}$\\times$", ha="center", va="bottom", fontsize=9)
        ax.text(27, i, f"{st.mean(vals):.1f}%   (chance {c:.1f}%)",
                va="center", ha="left", fontsize=8.2)

    ax.axvline(1.0, color="black", lw=1.2, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(0.45, 25)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1\n(chance)", "2", "5", "10", "20"])
    ax.set_xlabel("MoA hit rate as a multiple of that model's own chance level")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}\n{ch}" for m, ch, _ in rows], fontsize=8.5)
    ax.set_ylim(-0.55, len(rows) - 0.35)
    ax.tick_params(axis="x", direction="out", length=3)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_moa_vs_drpreter.{ext}",
                    dpi=400 if ext == "png" else None, bbox_inches="tight")
    for model, channel, vals in rows:
        c = CHANCE[model]
        print(f"{model:9s} {channel[:24]:26s} {st.mean(vals):5.2f}%  "
              f"chance {c:4.1f}%  {st.mean(vals)/c:5.2f}x")
    print("wrote figures/fig_moa_vs_drpreter.{pdf,png}")


if __name__ == "__main__":
    main()
