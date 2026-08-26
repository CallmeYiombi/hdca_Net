"""
HDCA-Net interpretability analysis.

Per-drug averaged alignment scores -> MoA validation + heatmaps.

Extracts (averaged over the samples of each drug):
  gene_attn       (D, G)  -- gene-level attention over HCDT gene targets
  path_attn       (D, P)  -- pathway-level attention over HCDT pathway mask
  p_gene_align    (D, P)  - gene-aligned pathway vector (with cell context)
  p_path_align    (D, P)  - pathway-aligned vector (with cell context)
  align_weights   (D, 2)  -- drug-conditional [gene_branch, pathway_branch] softmax

Usage:
  python src/analysis/interpret_hdca.py \\
      --config configs/hdca_gdsc12_cross_diag.yaml \\
      --model_path results/hdca_gdsc12_diag2/cross_dataset/gene_pathway/best.pt \\
      --align both \\
      --out_dir results/interpret_hdca/cross \\
      --gpu 0
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.hdca_net import HDCANet
from src.data.dataset_hdca import HDCADataset

ALIGN_PRESETS = {
    "gene":    ("gene",),
    "pathway": ("pathway",),
    "both":    ("gene", "pathway"),
}

DRUG_MOA = {
    "erlotinib":    ["EGFR", "ErbB"],
    "gefitinib":    ["EGFR", "ErbB"],
    "lapatinib":    ["EGFR", "ErbB", "HER2"],
    "afatinib":     ["EGFR", "ErbB"],
    "vemurafenib":  ["MAPK", "BRAF", "ERK"],
    "dabrafenib":   ["MAPK", "BRAF", "ERK"],
    "trametinib":   ["MAPK", "MEK", "ERK"],
    "selumetinib":  ["MAPK", "MEK", "ERK"],
    "bkm120":       ["PI3K", "AKT", "mTOR"],
    "gdc-0941":     ["PI3K", "AKT"],
    "pictilisib":   ["PI3K", "AKT"],
    "vorinostat":   ["HDAC", "histone", "acetyl"],
    "panobinostat": ["HDAC", "histone", "acetyl"],
    "imatinib":     ["BCR-ABL", "ABL", "KIT"],
    "nilotinib":    ["BCR-ABL", "ABL"],
    "crizotinib":   ["ALK", "MET"],
    "nutlin-3a":    ["p53", "MDM2", "TP53"],
    "olaparib":     ["PARP", "DNA repair", "BRCA"],
    "rucaparib":    ["PARP", "DNA repair"],
}

def load_matrices(mat_dir, drug_gene_file="hcdt_drug_gene.npy",
                  drug_path_file="hcdt_drug_path_direct.npy"):
    load = lambda f: np.load(os.path.join(mat_dir, f))
    return {
        "drug_fp":               load("drug_fp.npy"),
        "cell_expr":             load("cell_expr.npy"),
        "hcdt_drug_gene":        load(drug_gene_file),
        "hcdt_drug_path_direct": load(drug_path_file),
        "hcdt_neg_drug_gene":    load("hcdt_neg_drug_gene.npy"),
        "gene_pathway":          load("gene_pathway.npy"),
        "sample_table":          pd.read_csv(os.path.join(mat_dir, "sample_table.csv")),
    }

def extract_scores(model, mat, cfg, device, align_branches):
    """Run inference over all samples, average scores per drug_idx."""
    sample_table = mat["sample_table"]
    num_drugs    = int(sample_table["drug_idx"].max()) + 1
    num_genes    = mat["cell_expr"].shape[1]
    num_pathways = mat["gene_pathway"].shape[1]

    sums = {
        "gene_attn":     np.zeros((num_drugs, num_genes),    dtype=np.float64),
        "path_attn":     np.zeros((num_drugs, num_pathways), dtype=np.float64),
        "p_gene_align":  np.zeros((num_drugs, num_pathways), dtype=np.float64),
        "p_path_align":  np.zeros((num_drugs, num_pathways), dtype=np.float64),
        "p_align":       np.zeros((num_drugs, num_pathways), dtype=np.float64),
        "align_weights": np.zeros((num_drugs, 2),            dtype=np.float64),
    }
    counts = np.zeros(num_drugs, dtype=np.int64)

    all_idx = np.arange(len(sample_table))
    ds = HDCADataset(
        sample_indices=all_idx, sample_table=sample_table,
        drug_fp=mat["drug_fp"], cell_expr=mat["cell_expr"],
        hcdt_drug_gene=mat["hcdt_drug_gene"],
        hcdt_drug_path=mat["hcdt_drug_path_direct"],
        hcdt_neg_drug_gene=mat["hcdt_neg_drug_gene"],
        y_mean=0.0, y_std=1.0,
    )
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg.get("num_workers", 4), pin_memory=True)

    model.eval()
    didx_col = sample_table["drug_idx"].values

    has_gene_branch    = "gene" in align_branches
    has_pathway_branch = "pathway" in align_branches
    two_branch         = has_gene_branch and has_pathway_branch

    pos = 0
    with torch.no_grad():
        for batch in loader:
            drug_fp, cell_expr, dg, dp, neg, y = batch
            out = model.get_alignment_scores(
                drug_fp.to(device), cell_expr.to(device),
                dg.to(device), dp.to(device),
            )
            B = drug_fp.size(0)
            d_indices = didx_col[pos: pos + B]
            pos += B

            ga  = out.get("gene_attn")
            pa  = out.get("path_attn")
            pga = out.get("p_gene_align")
            ppa = out.get("p_path_align")
            pal = out.get("p_align")
            aw  = out.get("align_weights")

            for i, di in enumerate(d_indices):
                if has_gene_branch and ga is not None:
                    sums["gene_attn"][di]    += ga[i].cpu().numpy()
                if has_pathway_branch and pa is not None:
                    sums["path_attn"][di]    += pa[i].cpu().numpy()
                if has_gene_branch and pga is not None:
                    sums["p_gene_align"][di] += pga[i].cpu().numpy()
                if has_pathway_branch and ppa is not None:
                    sums["p_path_align"][di] += ppa[i].cpu().numpy()
                if pal is not None:
                    sums["p_align"][di]      += pal[i].cpu().numpy()
                if two_branch and aw is not None:
                    sums["align_weights"][di] += aw[i].cpu().numpy()
                counts[di] += 1

    valid = counts > 0
    denom = counts[:, None].clip(1)
    avg = {k: np.where(counts[:, None] > 0, v / denom, 0)
           for k, v in sums.items() if v.ndim == 2 and v.shape[1] != 2}
    if two_branch:
        avg["align_weights"] = np.where(counts[:, None] > 0,
                                        sums["align_weights"] / denom, 0)
    avg["valid_drugs"] = np.where(valid)[0]
    avg["drug_count"]  = counts
    return avg

def get_drug_names(mat):
    tbl = mat["sample_table"]
    return dict(zip(tbl["drug_idx"], tbl["drug_name_lower"]))

def get_pathway_names(mat_dir):
    with open(os.path.join(mat_dir, "id_maps.json")) as f:
        return json.load(f).get("pathway_list")

def get_gene_names(mat_dir):
    with open(os.path.join(mat_dir, "id_maps.json")) as f:
        return json.load(f).get("gene_list")

def validate_moa(score_matrix, drug_names, pathway_names, valid_drugs, out_path,
                 top_k=10):
    rows = []
    for di in valid_drugs:
        dname = drug_names.get(di, "").lower()
        keywords = None
        for k, kws in DRUG_MOA.items():
            if k in dname or dname in k:
                keywords = kws
                break
        if keywords is None:
            continue

        scores = score_matrix[di]
        top_idx = np.argsort(scores)[::-1][:top_k]
        top_names  = [pathway_names[i] for i in top_idx]
        top_scores = scores[top_idx]

        hits = []
        for kw in keywords:
            for rank, pname in enumerate(top_names):
                if kw.lower() in pname.lower():
                    hits.append(f"{pname} (rank {rank+1})")
                    break

        rows.append({
            "drug":         dname,
            "expected_kws": ", ".join(keywords),
            "top5":         " | ".join(top_names[:5]),
            "top5_scores":  " | ".join(f"{s:.3f}" for s in top_scores[:5]),
            "moa_hits":     "; ".join(hits) if hits else "--",
            "hit":          len(hits) > 0,
        })

    if not rows:
        print("  [WARN] no known MoA drugs matched")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    n_hit = df["hit"].sum()
    print(f"  MoA top-{top_k} hit: {n_hit}/{len(df)} drugs")
    print(df[["drug", "expected_kws", "moa_hits"]].to_string(index=False))
    return df

def save_top_pathways(score_matrix, drug_names, pathway_names, valid_drugs,
                      out_path, top_k=10):
    rows = []
    for di in valid_drugs:
        top_idx = np.argsort(score_matrix[di])[::-1][:top_k]
        for rank, pi in enumerate(top_idx):
            rows.append({
                "drug_idx":     di,
                "drug_name":    drug_names.get(di, ""),
                "rank":         rank + 1,
                "pathway_idx":  pi,
                "pathway_name": pathway_names[pi] if pathway_names else pi,
                "score":        float(score_matrix[di, pi]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return df

def plot_heatmap(score_matrix, drug_labels, pathway_labels, title, out_path,
                 top_n_drugs=30, top_n_paths=40):
    if sns is None:
        print(f"  (seaborn not installed; skipping heatmap: {out_path})")
        return
    var = score_matrix.var(axis=0)
    top_p = np.argsort(var)[::-1][:top_n_paths]
    max_s = score_matrix[:, top_p].max(axis=1)
    top_d = np.argsort(max_s)[::-1][:top_n_drugs]
    sub   = score_matrix[np.ix_(top_d, top_p)]

    rows = [drug_labels[i] for i in top_d]
    cols = [pathway_labels[i] if pathway_labels else str(i) for i in top_p]
    fig, ax = plt.subplots(figsize=(max(14, top_n_paths * 0.35),
                                    max(10, top_n_drugs * 0.32)))
    sns.heatmap(sub, xticklabels=cols, yticklabels=rows, cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.3, linecolor="grey")
    ax.set_title(title, fontsize=13, pad=10)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

def save_align_weight_summary(align_weights, drug_names, valid_drugs, out_path):
    """Per-drug [gene_branch, pathway_branch] fusion softmax weights."""
    rows = []
    for di in valid_drugs:
        w = align_weights[di]
        rows.append({
            "drug_idx":      di,
            "drug_name":     drug_names.get(di, ""),
            "w_gene_branch": float(w[0]),
            "w_pathway_branch": float(w[1]),
        })
    df = pd.DataFrame(rows).sort_values("w_gene_branch", ascending=False)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  align_weights mean: gene={df['w_gene_branch'].mean():.3f}  "
          f"pathway={df['w_pathway_branch'].mean():.3f}")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--align",      choices=["gene", "pathway", "both"], default="both")
    parser.add_argument("--out_dir",    default="results/interpret_hdca")
    parser.add_argument("--gpu",        type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    elif "cuda_device" in cfg:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_device"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  align: {args.align}")

    os.makedirs(args.out_dir, exist_ok=True)
    mat_dir = cfg["mat_dir"]
    mat = load_matrices(mat_dir, cfg.get("hcdt_drug_gene_file", "hcdt_drug_gene.npy"),
                        cfg.get("hcdt_drug_path_file", "hcdt_drug_path_direct.npy"))
    cfg["fp_dim"]       = mat["drug_fp"].shape[1]
    cfg["num_genes"]    = mat["cell_expr"].shape[1]
    cfg["num_pathways"] = mat["gene_pathway"].shape[1]

    align_branches = ALIGN_PRESETS[args.align]
    gene_pathway_t = torch.tensor(mat["gene_pathway"], dtype=torch.float32)

    model = HDCANet(
        fp_dim=cfg["fp_dim"],
        num_genes=cfg["num_genes"],
        num_pathways=cfg["num_pathways"],
        gene_pathway_matrix=gene_pathway_t,
        drug_enc_hidden=cfg.get("drug_enc_hidden", 512),
        drug_enc_out=cfg.get("drug_enc_out", 256),
        gate_mode=cfg.get("gate_mode", "hard"),
        gate_gamma=cfg.get("gate_gamma", 2.0),
        path_gate_mode=cfg.get("path_gate_mode", "soft"),
        fp_input_dropout=0.0,
        use_mutation=cfg.get("use_mutation", False),
        dropout=cfg.get("dropout", 0.3),
        lambda_neg=cfg.get("lambda_neg", 0.1),
        mask_dropout=0.0,
        align_branches=align_branches,
        norm_type=cfg.get("norm_type", "layer"),
        gene_norm_type=cfg.get("gene_norm_type", None),
        path_norm_type=cfg.get("path_norm_type", None),
        cell_norm_type=cfg.get("cell_norm_type", None),
    ).to(device)

    print(f"Load: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print("\nExtracting alignment scores (per-drug averages)...")
    scores = extract_scores(model, mat, cfg, device, align_branches)

    pathway_names = get_pathway_names(mat_dir)
    gene_names    = get_gene_names(mat_dir)
    drug_names    = get_drug_names(mat)
    valid_drugs   = scores["valid_drugs"]
    print(f"  drugs: {len(valid_drugs)} / pathway: {len(pathway_names)} / gene: {len(gene_names)}")

    drug_labels = [drug_names.get(di, f"drug_{di}") for di in valid_drugs]

    if "pathway" in align_branches:
        print("\n[1] MoA validation -- path_attn (direct pathway attention)")
        validate_moa(scores["path_attn"], drug_names, pathway_names, valid_drugs,
                     os.path.join(args.out_dir, "moa_path_attn.csv"))

        print("\n[2] MoA validation -- p_path_align (with cell context)")
        validate_moa(scores["p_path_align"], drug_names, pathway_names, valid_drugs,
                     os.path.join(args.out_dir, "moa_p_path_align.csv"))

    if "gene" in align_branches:
        print("\n[3] MoA validation -- p_gene_align (gene branch -> pathway projection)")
        validate_moa(scores["p_gene_align"], drug_names, pathway_names, valid_drugs,
                     os.path.join(args.out_dir, "moa_p_gene_align.csv"))

    print("\n[4] MoA validation -- p_align (fused)")
    validate_moa(scores["p_align"], drug_names, pathway_names, valid_drugs,
                 os.path.join(args.out_dir, "moa_p_align.csv"))

    print("\nTop-10 pathways per drug...")
    if "pathway" in align_branches:
        save_top_pathways(scores["path_attn"], drug_names, pathway_names, valid_drugs,
                          os.path.join(args.out_dir, "top_path_attn.csv"))
    save_top_pathways(scores["p_align"], drug_names, pathway_names, valid_drugs,
                      os.path.join(args.out_dir, "top_p_align.csv"))

    if args.align == "both":
        print("\nAlign-branch weights per drug...")
        save_align_weight_summary(
            scores["align_weights"], drug_names, valid_drugs,
            os.path.join(args.out_dir, "align_weights.csv"),
        )

    print("\nHeatmaps...")
    if "pathway" in align_branches:
        plot_heatmap(
            scores["path_attn"][valid_drugs], drug_labels, pathway_names,
            title="HDCA-Net: Drug-Pathway Attention (path_attn)",
            out_path=os.path.join(args.out_dir, "heatmap_path_attn.png"),
        )
    plot_heatmap(
        scores["p_align"][valid_drugs], drug_labels, pathway_names,
        title="HDCA-Net: Fused Drug-Pathway Alignment (p_align)",
        out_path=os.path.join(args.out_dir, "heatmap_p_align.png"),
    )

    for k, v in scores.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            np.save(os.path.join(args.out_dir, f"{k}.npy"), v)
    print(f"\nDone. Output: {args.out_dir}/")

if __name__ == "__main__":
    main()
