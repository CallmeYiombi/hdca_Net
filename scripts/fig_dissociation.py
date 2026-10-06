#!/usr/bin/env python3
"""Predictive accuracy and mechanistic fidelity are separable axes (main text, Figure 4).

Two panels over the same four single-factor variants of the drug--gene mask, everything else
held fixed: cross-study accuracy on the left, mechanism recovery on the right, both axes
starting at zero.

Accuracy is recomputed from the per-pair predictions under --root rather than read from a
cached table. Use --check to print the numbers without plotting.
"""

import os
import argparse
import numpy as np
import pandas as pd

CAP = 13.82
COHORTS = {"CCLE": ("matrices_ccle_2015", "CCLE_2015", "CCLE"),
           "gCSI": ("matrices_gcsi_2019", "gCSI_2019", "gCSI")}

# variant -> (run directories, npz seed tag per run, display label, colour)
VARIANTS = [
    ("reference", [f"hdca_gdsc12_pruned_div03_s{s}" for s in range(1, 7)],
     [f"seed{s}" for s in range(1, 7)], "direct-target\n(reference)", "#2b7cd3"),
    ("broad", [f"hdca_gdsc12_broad_div03_s{s}" for s in range(1, 7)],
     [f"seed{s}" for s in range(1, 7)], "broad mask", "#e8622a"),
    ("randomized", [f"hdca_gdsc12_random_mask_{k}" for k in range(1, 6)],
     ["seed42"] * 5, "randomized mask", "#7a7a7a"),
    ("dense", [f"hdca_gdsc12_nomask_div03_s{s}" for s in range(1, 7)],
     [f"seed{s}" for s in range(1, 7)], "no mask", "#5b4ea8"),
]

# MoA hit rates as reported in the main text (Sections 4.2 and 4.4).
# probe = 16-drug curated probe, gene branch;  panel = 144-drug DGIdb panel.
# A None means that read-out is not plotted for this variant (matches the published figure).
MOA = {
    "reference":  {"probe": (49.0, 7.3), "panel": (44.1, 2.0)},
    "broad":      {"probe": (15.6, 3.4), "panel": (19.4, 1.8)},
    "randomized": {"probe": (3.8, 5.6),  "panel": (1.7, 1.3)},
    "dense":      {"probe": (0.0, 0.0),  "panel": None},
}


# label placement, kept out of the plotting loop so the four labels never collide:
# the two bottom variants sit within ~2 points of each other on the y axis.
LABEL_OFFSET = {"reference": (16, 12), "broad": (16, 8),
                "randomized": (10, 14), "dense": (-10, -20)}
LABEL_ALIGN = {"reference": "left", "broad": "left",
               "randomized": "left", "dense": "right"}


def align_mask(root, st_dir, drp_root, drp_tag):
    """Boolean mask over sample_table rows selecting the pairs DRPreter covers."""
    st = pd.read_csv(os.path.join(root, "data", st_dir, "sample_table.csv"))
    ic = pd.read_csv(os.path.join(drp_root, "Data_HDCA", f"IC_{drp_tag}.csv"))
    ref = set(zip(ic["DepMap_ID"].astype(str), ic["Drug name"].astype(str)))
    pairs = zip(st["model_id"].astype(str), st["drug_name"].astype(str))
    return np.array([p in ref for p in pairs], dtype=bool)


def variant_pcc(root, dirs, tags, cohort_tag, mask, cap=CAP):
    """Winsorized, DRPreter-aligned PCC for each run of one variant."""
    out = []
    for d, tag in zip(dirs, tags):
        f = os.path.join(root, "results", d, "cross_dataset", "gene_pathway",
                         f"crosspred_{cohort_tag}_{tag}.npz")
        z = np.load(f)
        yt, yp = z["y_true"][mask], z["y_pred"][mask]
        out.append(np.corrcoef(np.minimum(yt, cap), yp)[0, 1])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="server_snapshot")
    ap.add_argument("--drpreter_root", default="DRPreter-main")
    ap.add_argument("--out", default="figures/fig_dissociation")
    ap.add_argument("--check", action="store_true", help="print numbers, skip the plot")
    a = ap.parse_args()

    acc = {}
    for coh, (st_dir, coh_tag, drp_tag) in COHORTS.items():
        m = align_mask(a.root, st_dir, a.drpreter_root, drp_tag)
        for name, dirs, tags, _, _ in VARIANTS:
            v = variant_pcc(a.root, dirs, tags, coh_tag, m)
            acc[(name, coh)] = (v.mean(), v.std(ddof=1), len(v))

    print(f"{'variant':<12}" + "".join(f"{c:>22}" for c in COHORTS))
    for name, _, _, _, _ in VARIANTS:
        row = "".join(f"{acc[(name,c)][0]:>15.4f} +/-{acc[(name,c)][1]:.4f}" for c in COHORTS)
        print(f"{name:<12}{row}   n={acc[(name,'CCLE')][2]}")
    if a.check:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["direct-target", "broad", "randomized", "no mask"]
    order = [v[0] for v in VARIANTS]
    x = np.arange(len(order))
    width = 0.34

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # ---- left: cross-study accuracy, two cohorts -------------------------------
    for j, coh in enumerate(COHORTS):
        pos = x + (j - 0.5) * width
        mu = [acc[(n, coh)][0] for n in order]
        sd = [acc[(n, coh)][1] for n in order]
        axL.bar(pos, mu, width, yerr=sd, capsize=3, label=coh,
                color=("#3b3b3b" if j == 0 else "#bdbdbd"),
                edgecolor="black", linewidth=0.7,
                error_kw=dict(elinewidth=0.8, capthick=0.8))
    axL.set_ylabel("cross-study PCC")
    axL.set_ylim(0, 1.0)
    axL.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # ---- right: mechanism recovery, two read-outs ------------------------------
    for j, key in enumerate(("probe", "panel")):
        pos, mu, sd = [], [], []
        for i, n in enumerate(order):
            if MOA[n][key] is None:
                continue
            pos.append(x[i] + (j - 0.5) * width)
            mu.append(MOA[n][key][0]); sd.append(MOA[n][key][1])
        for p, m_ in zip(pos, mu):     # a measured zero is not a missing bar
            if m_ == 0:
                axR.text(p, 0.8, "0", ha="center", va="bottom", fontsize=8)
        axR.bar(pos, mu, width, yerr=sd, capsize=3,
                label=("16-drug probe" if j == 0 else "144-drug panel"),
                color=("#3b3b3b" if j == 0 else "#bdbdbd"),
                edgecolor="black", linewidth=0.7,
                error_kw=dict(elinewidth=0.8, capthick=0.8))
    axR.set_ylabel("MoA hit rate (%)")
    axR.set_ylim(0, 60)
    axR.set_yticks(range(0, 61, 10))

    for ax, tag in ((axL, "A"), (axR, "B")):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.tick_params(direction="out", length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.4)
        ax.text(-0.18, 1.04, tag, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom")

    fig.tight_layout()
    for ext in ("pdf", "png", "eps"):
        fig.savefig(f"{a.out}.{ext}", dpi=400 if ext == "png" else None,
                    bbox_inches="tight")
    print(f"\nwrote {a.out}.{{png,pdf,eps}}")


if __name__ == "__main__":
    main()
