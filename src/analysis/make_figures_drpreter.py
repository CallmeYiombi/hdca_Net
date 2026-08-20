"""Result figures for the manuscript.

Regenerates the performance panel, the MoA hit-rate panel and the HDCA-Net vs DRPreter
comparison from the aggregated result tables in results/.

Usage: python src/analysis/make_figures_drpreter.py
"""
import os, re, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

DRUG_MOA = {
    "erlotinib": ["EGFR", "ErbB"], "gefitinib": ["EGFR", "ErbB"],
    "lapatinib": ["EGFR", "ErbB", "HER2"], "afatinib": ["EGFR", "ErbB"],
    "vemurafenib": ["MAPK", "BRAF", "ERK"], "dabrafenib": ["MAPK", "BRAF", "ERK"],
    "trametinib": ["MAPK", "MEK", "ERK"], "selumetinib": ["MAPK", "MEK", "ERK"],
    "bkm120": ["PI3K", "AKT", "mTOR"], "gdc-0941": ["PI3K", "AKT"],
    "pictilisib": ["PI3K", "AKT"], "vorinostat": ["HDAC", "histone", "acetyl"],
    "panobinostat": ["HDAC", "histone", "acetyl"], "imatinib": ["BCR-ABL", "ABL", "KIT"],
    "nilotinib": ["BCR-ABL", "ABL"], "crizotinib": ["ALK", "MET"],
    "nutlin-3a": ["p53", "MDM2", "TP53"], "olaparib": ["PARP", "DNA repair", "BRCA"],
    "rucaparib": ["PARP", "DNA repair"],
}

def _wb(kw, text):
    return re.search(rf"\b{re.escape(kw)}\d*\b", text, re.IGNORECASE) is not None

def _moa_kws(drug_lower):
    for k, kws in DRUG_MOA.items():
        if k in drug_lower or drug_lower in k:
            return kws
    return []

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGDIR = os.path.join(ROOT, "manuscript", "figures")
INTERP = os.path.join(ROOT, "results", "interpret_hdca")
os.makedirs(FIGDIR, exist_ok=True)

ORDER = ["HDCA", "DRPreter", "PANCDR", "DeepCDR", "GraphDRP", "TGSA"]
HL = "#c0392b"
DRP = "#e67e22"
OTHER = "#95a5a6"

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved: manuscript/figures/{name}.pdf/.png")

def fig_accuracy():
    df = pd.read_csv(os.path.join(ROOT, "results", "cross_winsorized_table_6seed_subset.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, cset in zip(axes, ["CCLE", "gCSI"]):
        sub = df[df.cset == cset].set_index("model").reindex(ORDER)
        colors = [HL if m == "HDCA" else DRP if m == "DRPreter" else OTHER for m in ORDER]
        x = np.arange(len(ORDER))
        ax.bar(x, sub["PCC_win"], yerr=sub["PCC_win_std"], color=colors,
               capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=40, ha="right", fontsize=9)
        ax.set_title(f"GDSC -> {cset}", fontsize=11)
        ax.set_ylim(0.65, 0.80)
        ax.grid(axis="y", ls=":", alpha=0.5)
        for xi, (v, s) in enumerate(zip(sub["PCC_win"], sub["PCC_win_std"])):
            ax.text(xi, v + s + 0.002, f"{v:.3f}", ha="center", fontsize=7)
    axes[0].set_ylabel("Cross-dataset PCC (winsorized)", fontsize=10)
    fig.suptitle("Cross-dataset generalization (subset-aligned, 6 seeds)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig_cross_accuracy_6seed")

def fig_moa():
    df = pd.read_csv(os.path.join(ROOT, "results", "moa_compare_table.csv"))
    df = df[df.axis.isin(["gene->pathway", "pathway attention"])].reset_index(drop=True)
    labels = ["gene -> pathway\n(GradCAM / gene-attn)", "pathway attention\n(Transformer / path-attn)"]
    x = np.arange(len(df)); w = 0.36

    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - w/2, df["HDCA_mean"], w, yerr=df["HDCA_std"], capsize=4,
                color=HL, edgecolor="black", linewidth=0.5, label="HDCA (ours)")
    b2 = ax.bar(x + w/2, df["DRPreter_mean"], w, yerr=df["DRPreter_std"], capsize=4,
                color=DRP, edgecolor="black", linewidth=0.5, label="DRPreter")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MoA hit rate (%)  -- 16 drugs, 6 seeds", fontsize=10)
    ax.set_ylim(0, 68)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=9, frameon=False)

    for xi, r in df.iterrows():
        ax.text(xi - w/2, r["HDCA_mean"] + r["HDCA_std"] + 1, f"{r['HDCA_mean']:.1f}",
                ha="center", fontsize=8, color=HL)
        ax.text(xi + w/2, r["DRPreter_mean"] + r["DRPreter_std"] + 1, f"{r['DRPreter_mean']:.1f}",
                ha="center", fontsize=8, color=DRP)
        star = "**" if r["p_mannwhitney"] < 0.01 else ("*" if r["p_mannwhitney"] < 0.05 else "ns")
        top = max(r["HDCA_mean"] + r["HDCA_std"], r["DRPreter_mean"] + r["DRPreter_std"]) + 5
        ax.plot([xi - w/2, xi + w/2], [top, top], color="black", lw=0.8)
        ax.text(xi, top + 0.5, f"{star}  (+{r['delta_pp']:.0f}pp, p={r['p_mannwhitney']:.3f})",
                ha="center", fontsize=8)
    ax.set_title("Mechanism-of-action recovery: supervised vs post-hoc", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_moa_compare_drpreter")

def fig_performance_3panel():
    """Performance panel: random split, drug 5-fold and cross-study evaluation."""
    sub = pd.read_csv(os.path.join(ROOT, "results", "cross_winsorized_table_6seed_subset.csv"))

    rnd = {"GraphDRP": 0.867, "DeepCDR": 0.875, "TGSA": 0.863,
           "PANCDR": 0.933, "DRPreter": 0.9404, "HDCA": 0.865}
    d5m = {"GraphDRP": 0.372, "DeepCDR": 0.383, "TGSA": 0.352,
           "PANCDR": 0.362, "DRPreter": 0.4135, "HDCA": 0.407}
    d5s = {"GraphDRP": 0.070, "DeepCDR": 0.079, "TGSA": 0.086,
           "PANCDR": 0.125, "DRPreter": 0.0143, "HDCA": 0.056}
    colors = [HL if m == "HDCA" else DRP if m == "DRPreter" else OTHER for m in ORDER]
    x = np.arange(len(ORDER))

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))

    ax[0].bar(x, [rnd[m] for m in ORDER], color=colors, edgecolor="black", linewidth=0.5)
    ax[0].set_ylim(0.80, 0.96); ax[0].set_title("(a) Random split (in-distribution)", fontsize=10)
    ax[0].set_ylabel("PCC", fontsize=10)

    ax[1].bar(x, [d5m[m] for m in ORDER], yerr=[d5s[m] for m in ORDER], capsize=3,
              color=colors, edgecolor="black", linewidth=0.5)
    ax[1].set_ylim(0.25, 0.50); ax[1].set_title("(b) Drug 5-fold (unseen drug)", fontsize=10)
    ax[1].set_ylabel("PCC", fontsize=10)

    w = 0.38
    cc = sub[sub.cset == "CCLE"].set_index("model").reindex(ORDER)
    gc = sub[sub.cset == "gCSI"].set_index("model").reindex(ORDER)
    ax[2].bar(x - w/2, cc["PCC_win"], w, yerr=cc["PCC_win_std"], capsize=2,
              color=colors, edgecolor="black", linewidth=0.5, label="CCLE")
    ax[2].bar(x + w/2, gc["PCC_win"], w, yerr=gc["PCC_win_std"], capsize=2,
              color=colors, edgecolor="black", linewidth=0.5, alpha=0.55, label="gCSI")
    ax[2].set_ylim(0.65, 0.80); ax[2].set_title("(c) Cross-dataset (winsorized, subset, 6 seeds)", fontsize=10)
    ax[2].set_ylabel("PCC", fontsize=10)
    ax[2].text(0.02, 0.94, "solid=CCLE  faded=gCSI", transform=ax[2].transAxes, fontsize=7)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(ORDER, rotation=40, ha="right", fontsize=8)
        a.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    save(fig, "figA_performance")

def fig_moa_broad_direct():
    """MoA hit rate for the three interpretability channels, broad vs direct-target mask."""
    ch = ["pathway\nattention", "gene branch\n$p_{gene\\_align}$", "fused\n$p_{align}$"]
    broad = [54.2, 17.7, 46.9]; broad_s = [7.6, 2.6, 8.6]
    direct = [49.0, 49.0, 49.0]; direct_s = [7.3, 7.3, 4.7]
    x = np.arange(len(ch)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, broad, w, yerr=broad_s, capsize=4, color=OTHER,
           edgecolor="black", linewidth=0.5, label="Broad HCDT mask")
    ax.bar(x + w/2, direct, w, yerr=direct_s, capsize=4, color=HL,
           edgecolor="black", linewidth=0.5, label="Direct-target mask")
    ax.set_xticks(x); ax.set_xticklabels(ch, fontsize=9)
    ax.set_ylabel("MoA top-10 hit rate (%)  -- 16 drugs, 6 seeds", fontsize=10)
    ax.set_ylim(0, 68); ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=9, frameon=False)
    for xi in range(len(ch)):
        ax.text(xi - w/2, broad[xi] + broad_s[xi] + 1, f"{broad[xi]:.1f}", ha="center", fontsize=8)
        ax.text(xi + w/2, direct[xi] + direct_s[xi] + 1, f"{direct[xi]:.1f}", ha="center", fontsize=8, color=HL)
    ax.annotate("+31.3pp, $p<0.001$", xy=(1 + w/2, 63), ha="center", fontsize=8, color=HL)
    ax.set_title("MoA top-10 hit rate by channel and gene mask", fontsize=10)
    fig.tight_layout()
    save(fig, "figD_moa_hitrate")

def fig_moa_rank_dumbbell():
    """Per-drug rank of the expected MoA pathway, broad vs direct-target mask."""
    MAT = os.path.join(ROOT, "data", "matrices_gdsc12")
    st = pd.read_csv(os.path.join(MAT, "sample_table.csv"))
    low2idx = dict(zip(st["drug_name_lower"], st["drug_idx"]))
    pl = json.load(open(os.path.join(MAT, "id_maps.json")))["pathway_list"]
    Pb = np.load(os.path.join(INTERP, "cross_diag2", "p_gene_align.npy"))
    Pd = np.load(os.path.join(INTERP, "pruned_div03", "p_gene_align.npy"))
    drugs = ["afatinib", "crizotinib", "dabrafenib", "erlotinib", "gefitinib", "imatinib",
             "lapatinib", "nilotinib", "nutlin-3a (-)", "olaparib", "panobinostat",
             "pictilisib", "rucaparib", "selumetinib", "trametinib", "vorinostat"]
    label = {"nutlin-3a (-)": "nutlin-3a"}

    def best_rank(P, di, dl):
        kws = _moa_kws(dl)
        order = np.argsort(P[di])[::-1]
        for r, pi in enumerate(order):
            if any(_wb(k, pl[pi]) for k in kws):
                return r + 1
        return None

    MISS = 23
    rows = []
    for d in drugs:
        di = low2idx[d]
        rb = best_rank(Pb, di, d); rd = best_rank(Pd, di, d)
        rows.append((label.get(d, d),
                     rb if (rb and rb <= 20) else MISS,
                     rd if (rd and rd <= 20) else MISS))
    rows.sort(key=lambda t: (t[2], t[1]))
    names = [r[0] for r in rows]
    xb = np.array([r[1] for r in rows]); xd = np.array([r[2] for r in rows])
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.axvspan(0.5, 10.5, color="#e8f4f2", zorder=0, label="MoA hit zone (top-10)")
    for yi, (b, d_) in enumerate(zip(xb, xd)):
        ax.plot([b, d_], [yi, yi], color="#c9ccce", lw=2, zorder=1)
    ax.scatter(xb, y, s=55, color=OTHER, edgecolor="black", lw=0.5, zorder=3, label="Broad HCDT mask")
    ax.scatter(xd, y, s=60, color=HL, edgecolor="black", lw=0.5, zorder=3, label="Direct-target mask")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_ylim(-0.7, len(names) - 0.3); ax.invert_yaxis()
    ax.axvline(10.5, color="#7fb0aa", ls="--", lw=0.8, zorder=1)
    ax.set_xlim(0, 25)
    ax.set_xticks([1, 5, 10, 15, 20, MISS])
    ax.set_xticklabels(["1", "5", "10", "15", "20", "x\n>20"], fontsize=8)
    ax.set_xlabel("rank of expected MoA pathway in gene branch  (1 = top; x = not recovered)", fontsize=9)
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white")
    ax.set_title("Direct-target mask pulls each drug's MoA pathway into the top ranks",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "figE_moa_rank")

def fig_moa_heatmap():
    """Heatmap of gene-branch pathway scores with the MoA hits highlighted."""
    MAT = os.path.join(ROOT, "data", "matrices_gdsc12")
    BROAD = os.path.join(INTERP, "cross_diag2")
    DIRECT = os.path.join(INTERP, "pruned_div03")
    drugs = ["afatinib", "crizotinib", "dabrafenib", "erlotinib", "gefitinib", "imatinib",
             "lapatinib", "nilotinib", "nutlin-3a (-)", "olaparib", "panobinostat",
             "pictilisib", "rucaparib", "selumetinib", "trametinib", "vorinostat"]
    label = {"nutlin-3a (-)": "nutlin-3a"}
    st = pd.read_csv(os.path.join(MAT, "sample_table.csv"))
    low2idx = dict(zip(st["drug_name_lower"], st["drug_idx"]))
    pl = json.load(open(os.path.join(MAT, "id_maps.json")))["pathway_list"]
    Pb = np.load(os.path.join(BROAD, "p_gene_align.npy"))
    Pd = np.load(os.path.join(DIRECT, "p_gene_align.npy"))
    didx = [low2idx[d] for d in drugs]

    def rank_enc(P, di):
        order = np.argsort(P[di])[::-1]
        val = np.zeros(P.shape[1])
        for r, idx in enumerate(order[:10]):
            val[idx] = 10 - r
        return val

    def hits(P, di, dl):
        """Indices of the top-10 pathways that match the drug's expected keywords."""
        kws = _moa_kws(dl)
        top = np.argsort(P[di])[::-1][:10]
        return {int(pi) for pi in top if any(_wb(k, pl[pi]) for k in kws)}

    Rb = np.array([rank_enc(Pb, di) for di in didx])
    Rd = np.array([rank_enc(Pd, di) for di in didx])
    hb = {d: hits(Pb, di, d) for d, di in zip(drugs, didx)}
    hd = {d: hits(Pd, di, d) for d, di in zip(drugs, didx)}

    from collections import Counter

    hit_cols = []
    for hmap in (hb, hd):
        for s in hmap.values():
            for pi in s:
                if pi not in hit_cols:
                    hit_cols.append(pi)
    broad_cnt = Counter()
    for di in didx:
        broad_cnt.update(np.argsort(Pb[di])[::-1][:2].tolist())
    collapse_cols = [p for p, c in broad_cnt.most_common() if c >= 4][:3]

    cols = list(collapse_cols)
    for pi in hit_cols:
        if pi not in cols:
            cols.append(pi)

    def _short(n):
        return n if len(n) <= 34 else n[:32] + "..."
    colnames = [_short(pl[c]) for c in cols]
    colpos = {c: j for j, c in enumerate(cols)}
    Mb = Rb[:, cols]; Md = Rd[:, cols]

    cmap = LinearSegmentedColormap.from_list(
        "hdca_teal", ["#ffffff", "#d8ecec", "#7fc0c4", "#2b7a8f", "#123b47"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), gridspec_kw=dict(wspace=0.05))
    panels = [(axes[0], Mb, hb, "(a) Broad HCDT mask  --  shared generic columns (collapse)"),
              (axes[1], Md, hd, "(b) Direct-target mask  --  drug-specific MoA pathways")]
    for ax, M, hmap, title in panels:
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=10)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(colnames, rotation=50, ha="right", fontsize=8)
        ax.set_yticks(range(len(drugs)))
        ax.set_yticklabels([label.get(d, d) for d in drugs], fontsize=8)
        ax.set_title(title, fontsize=9.5, pad=8)
        if ax is axes[1]:
            ax.set_yticklabels([])
        ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(drugs), 1), minor=True)
        ax.grid(which="minor", color="#e6e6e6", lw=0.6)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_edgecolor("#cccccc")
        for di, d in enumerate(drugs):
            for pi in hmap.get(d, ()):
                if pi in colpos:
                    ax.add_patch(Rectangle((colpos[pi] - 0.5, di - 0.5), 1, 1, fill=False,
                                           edgecolor="#e8663b", lw=2.4, zorder=5))
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.01)
    cbar.set_label("within-drug pathway rank\n(10 = top, blank = outside top-10)", fontsize=8)
    axes[1].add_patch(Rectangle((0, 0), 0, 0, fill=False, edgecolor="#e8663b", lw=2.4,
                                label="known-MoA hit"))
    axes[1].legend(loc="lower right", fontsize=8, frameon=True, facecolor="white")
    fig.suptitle("Gene-branch ($p_{gene\\_align}$) pathway alignment for 16 known-MoA drugs "
                 "(box = expected MoA pathway reaches top-10)", fontsize=12, y=0.99)
    save(fig, "figE_moa_heatmap")

if __name__ == "__main__":
    fig_accuracy()
    fig_moa()
    fig_performance_3panel()
    fig_moa_broad_direct()
    fig_moa_rank_dumbbell()
    print("done.")
