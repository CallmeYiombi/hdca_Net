"""
HDCA-Net training & evaluation.

Three modes (choose with --mode):
  random   : 80/10/10 random split -- main test-set metrics
  drug5    : drug-stratified 5-fold -- generalization to unseen drugs
  cross    : train on GDSC1+2 -> evaluate on CCLE / gCSI

Ablation over alignment branches with --align (gene | pathway | both).

Examples
--------
  # Main random-split run (both branches)
  python src/train/train_hdca.py --config configs/hdca_gdsc12.yaml \\
      --mode random --align both --gpu 4

  # Branch ablation, drug-5fold
  python src/train/train_hdca.py --config configs/hdca_gdsc12.yaml \\
      --mode drug5 --align gene     --gpu 4
  python src/train/train_hdca.py --config configs/hdca_gdsc12.yaml \\
      --mode drug5 --align pathway  --gpu 4
  python src/train/train_hdca.py --config configs/hdca_gdsc12.yaml \\
      --mode drug5 --align both     --gpu 4

  # Cross-dataset evaluation
  python src/train/train_hdca.py --config configs/hdca_gdsc12.yaml \\
      --mode cross --align both \\
      --eval_dirs data/matrices_ccle_2015 data/matrices_gcsi_2019 --gpu 4
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.hdca_net import HDCANet
from src.data.dataset_mp import MPHCPNetDataset
from src.data.split import random_split, drug_kfold
from src.utils.seed import set_seed
from src.utils.metrics import compute_metrics

ALIGN_PRESETS = {
    "gene":    ("gene",),
    "pathway": ("pathway",),
    "both":    ("gene", "pathway"),
}

def load_matrices(mat_dir, drug_gene_file="hcdt_drug_gene.npy",
                  drug_path_file="hcdt_drug_path_direct.npy"):
    load = lambda f: np.load(os.path.join(mat_dir, f))
    return {
        "drug_fp":               load("drug_fp.npy"),
        "cell_expr":             load("cell_expr.npy"),
        "hcdt_drug_gene":        load(drug_gene_file),
        "hcdt_drug_disease":     load("hcdt_drug_disease.npy"),
        "hcdt_drug_path_direct": load(drug_path_file),
        "hcdt_neg_drug_gene":    load("hcdt_neg_drug_gene.npy"),
        "gene_pathway":          load("gene_pathway.npy"),
        "sample_table":          pd.read_csv(os.path.join(mat_dir, "sample_table.csv")),
    }

def build_model(cfg, gene_pathway_t, align_branches, device):
    return HDCANet(
        fp_dim=cfg["fp_dim"],
        num_genes=cfg["num_genes"],
        num_pathways=cfg["num_pathways"],
        gene_pathway_matrix=gene_pathway_t,
        drug_enc_hidden=cfg.get("drug_enc_hidden", 512),
        drug_enc_out=cfg.get("drug_enc_out", 256),
        gate_mode=cfg.get("gate_mode", "hard"),
        gate_gamma=cfg.get("gate_gamma", 2.0),
        path_gate_mode=cfg.get("path_gate_mode", "soft"),
        fp_input_dropout=cfg.get("fp_input_dropout", 0.0),
        use_mutation=cfg.get("use_mutation", False),
        dropout=cfg.get("dropout", 0.3),
        lambda_neg=cfg.get("lambda_neg", 0.1),
        mask_dropout=cfg.get("mask_dropout", 0.0),
        align_branches=align_branches,
        norm_type=cfg.get("norm_type", "layer"),
        gene_norm_type=cfg.get("gene_norm_type", None),
        path_norm_type=cfg.get("path_norm_type", None),
        cell_norm_type=cfg.get("cell_norm_type", None),
    ).to(device)

def make_loader(indices, sample_table, mat, cfg, shuffle, y_mean, y_std,
                cell_expr_override=None):
    cell_expr = cell_expr_override if cell_expr_override is not None else mat["cell_expr"]
    ds = MPHCPNetDataset(
        sample_indices=indices, sample_table=sample_table,
        drug_fp=mat["drug_fp"], cell_expr=cell_expr,
        hcdt_drug_gene=mat["hcdt_drug_gene"],
        hcdt_drug_disease=mat["hcdt_drug_disease"],
        hcdt_drug_path=mat["hcdt_drug_path_direct"],
        hcdt_neg_drug_gene=mat["hcdt_neg_drug_gene"],
        y_mean=y_mean, y_std=y_std,
    )
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg.get("num_workers", 4), pin_memory=True)

def run_epoch(model, loader, optimizer, device, cfg, train=True):
    model.train() if train else model.eval()
    lambda_neg = cfg.get("lambda_neg", 0.1)
    lambda_align_div = cfg.get("lambda_align_div", 0.0)
    lambda_gene_sparse = cfg.get("lambda_gene_sparse", 0.0)
    total_loss, all_y, all_pred = 0.0, [], []
    align_w_std_sum, align_w_mean_sum, n_batches_aw = 0.0, 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            drug_fp, cell_expr, dg, dr, dp, neg, y = batch
            drug_fp   = drug_fp.to(device);   cell_expr = cell_expr.to(device)
            dg        = dg.to(device);         dr        = dr.to(device)
            dp        = dp.to(device);         neg       = neg.to(device)
            y         = y.to(device)

            y_pred, neg_loss, align_weights, gene_entropy = model(
                drug_fp, cell_expr, dg, dr, dp,
                hcdt_neg_gene=neg if train else None,
            )
            loss = nn.functional.mse_loss(y_pred.squeeze(), y)
            if train and lambda_neg > 0:
                loss = loss + lambda_neg * neg_loss
            if train and lambda_gene_sparse > 0:

                loss = loss + lambda_gene_sparse * gene_entropy
            if (train and lambda_align_div > 0
                    and align_weights is not None
                    and align_weights.size(0) > 1):

                div = align_weights.std(dim=0).mean()
                loss = loss - lambda_align_div * div

            if align_weights is not None:
                align_w_std_sum  += align_weights.std(dim=0).mean().item()
                align_w_mean_sum += align_weights[:, 0].mean().item()
                n_batches_aw += 1

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            all_y.append(y.cpu().numpy())
            all_pred.append(y_pred.squeeze().detach().cpu().numpy())

    all_y    = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    m = compute_metrics(all_y, all_pred)
    m["loss"] = total_loss / len(all_y)
    if n_batches_aw > 0:
        m["align_w_std"]  = align_w_std_sum  / n_batches_aw
        m["align_w_mean0"] = align_w_mean_sum / n_batches_aw
    return m

def train_loop(model, tr_loader, va_loader, cfg, device, out_dir,
               epochs, early_stop, lr_patience,
               eval_hook=None, hook_every=5):
    os.makedirs(out_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg.get("weight_decay", 1e-5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=lr_patience, factor=0.5)

    best_val_loss, best_epoch = float("inf"), 0
    best_path    = os.path.join(out_dir, "best.pt")
    history_path = os.path.join(out_dir, "train_history.json")
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, tr_loader, optimizer, device, cfg, train=True)
        va = run_epoch(model, va_loader, optimizer, device, cfg, train=False)
        scheduler.step(va["loss"])

        rec = {
            "epoch": epoch,
            "tr_loss": tr["loss"], "tr_pcc": tr["pcc"], "tr_rmse": tr["rmse"],
            "va_loss": va["loss"], "va_pcc": va["pcc"], "va_rmse": va["rmse"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if "align_w_std" in tr:
            rec["align_w_std"]   = tr["align_w_std"]
            rec["align_w_mean0"] = tr["align_w_mean0"]
        if eval_hook is not None and (epoch == 1 or epoch % hook_every == 0):
            extra = eval_hook(epoch, model)
            if extra:
                rec.update(extra)

        history.append(rec)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            best_epoch    = epoch
            torch.save(model.state_dict(), best_path)

        if epoch % 5 == 0 or epoch == 1:
            extras = " ".join(
                f"{k.replace('cross_', '').replace('_pcc', '')}={v:.3f}"
                for k, v in rec.items()
                if k.startswith("cross_") and k.endswith("_pcc")
            )
            aw_msg = (f" | aw_std {rec['align_w_std']:.3f} mean0 {rec['align_w_mean0']:.3f}"
                      if "align_w_std" in rec else "")
            print(f"    ep {epoch:3d} | tr RMSE {tr['rmse']:.4f} PCC {tr['pcc']:.4f} "
                  f"| va RMSE {va['rmse']:.4f} PCC {va['pcc']:.4f} "
                  f"| {extras}{aw_msg} | {(time.time()-t0)/60:.1f}min")

        if epoch - best_epoch >= early_stop:
            print(f"    Early stop at epoch {epoch} (best {best_epoch})")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    return best_epoch

def test_metrics(model, te_loader, device, y_mean, y_std, save_path=None):
    model.eval()
    all_y, all_pred = [], []
    with torch.no_grad():
        for batch in te_loader:
            drug_fp, cell_expr, dg, dr, dp, neg, y = batch
            y_pred, *_ = model(drug_fp.to(device), cell_expr.to(device),
                                 dg.to(device), dr.to(device), dp.to(device))
            all_y.append(y.numpy())
            all_pred.append(y_pred.squeeze().cpu().numpy())
    y_true = np.concatenate(all_y)    * y_std + y_mean
    y_pred = np.concatenate(all_pred) * y_std + y_mean
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if save_path is not None:

        np.savez(save_path, y_true=y_true, y_pred=y_pred)
    return compute_metrics(y_true[mask], y_pred[mask])

def run_random(cfg, mat, gene_pathway_t, align_branches, device, out_root):
    set_seed(cfg["seed"])
    tr_idx, va_idx, te_idx = random_split(mat["sample_table"], seed=cfg["seed"])
    y_mean = float(mat["sample_table"].iloc[tr_idx]["ln_ic50"].mean())
    y_std  = float(mat["sample_table"].iloc[tr_idx]["ln_ic50"].std())

    def _loader(idx, shuffle):
        return make_loader(idx, mat["sample_table"], mat, cfg, shuffle, y_mean, y_std)

    model = build_model(cfg, gene_pathway_t, align_branches, device)
    out_dir = os.path.join(out_root, "random", "_".join(align_branches))
    best_ep = train_loop(
        model, _loader(tr_idx, True), _loader(va_idx, False), cfg, device, out_dir,
        epochs=cfg["epochs"],
        early_stop=cfg.get("early_stop", 30),
        lr_patience=cfg.get("lr_patience", 7),
    )
    m = test_metrics(model, _loader(te_idx, False), device, y_mean, y_std)
    print(f"  TEST | RMSE {m['rmse']:.4f}  PCC {m['pcc']:.4f}  "
          f"Spearman {m['spearman']:.4f}  (best ep {best_ep})")

    result = {"mode": "random", "align": "_".join(align_branches),
              "best_epoch": best_ep, **m}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result

def run_drug5(cfg, mat, gene_pathway_t, align_branches, device, out_root):
    epochs       = cfg.get("ablation_epochs", 60)
    early_stop   = cfg.get("ablation_early_stop", 15)
    lr_patience  = cfg.get("ablation_lr_patience", 5)

    fold_metrics = []
    t_start = time.time()
    for fold, tr_idx, va_idx, te_idx in drug_kfold(
            mat["sample_table"], n_splits=5, seed=cfg["seed"]):
        set_seed(cfg["seed"] + fold)
        y_mean = float(mat["sample_table"].iloc[tr_idx]["ln_ic50"].mean())
        y_std  = float(mat["sample_table"].iloc[tr_idx]["ln_ic50"].std())

        def _loader(idx, shuffle):
            return make_loader(idx, mat["sample_table"], mat, cfg, shuffle, y_mean, y_std)

        model = build_model(cfg, gene_pathway_t, align_branches, device)
        out_dir = os.path.join(out_root, "drug5", "_".join(align_branches),
                               f"fold{fold+1}")
        best_ep = train_loop(
            model, _loader(tr_idx, True), _loader(va_idx, False), cfg, device,
            out_dir, epochs=epochs, early_stop=early_stop, lr_patience=lr_patience,
        )
        m = test_metrics(model, _loader(te_idx, False), device, y_mean, y_std)
        m["fold"], m["best_epoch"] = fold + 1, best_ep
        fold_metrics.append(m)
        print(f"  Fold {fold+1} | RMSE {m['rmse']:.4f}  PCC {m['pcc']:.4f}  "
              f"(best ep {best_ep})")

    df = pd.DataFrame(fold_metrics)
    summary = {
        "mode":      "drug5",
        "align":     "_".join(align_branches),
        "rmse_mean": float(df["rmse"].mean()),
        "rmse_std":  float(df["rmse"].std()),
        "pcc_mean":  float(df["pcc"].mean()),
        "pcc_std":   float(df["pcc"].std()),
        "spearman_mean": float(df["spearman"].mean()),
        "spearman_std":  float(df["spearman"].std()),
        "time_min":  (time.time() - t_start) / 60,
    }
    out_summary = os.path.join(out_root, "drug5", "_".join(align_branches), "summary.json")
    os.makedirs(os.path.dirname(out_summary), exist_ok=True)
    with open(out_summary, "w") as f:
        json.dump({"folds": fold_metrics, "summary": summary}, f, indent=2)
    print(f"\n  Drug-5fold summary | RMSE {summary['rmse_mean']:.4f}+/-{summary['rmse_std']:.4f}  "
          f"PCC {summary['pcc_mean']:.4f}+/-{summary['pcc_std']:.4f}")
    return summary

def eval_on_external(model, eval_dir, mat, cfg, device, y_mean, y_std, save_dir=None):
    eval_path    = Path(eval_dir)
    sample_table = pd.read_csv(eval_path / "sample_table.csv")
    cell_expr    = np.load(eval_path / "cell_expr.npy")
    with open(eval_path / "eval_info.json") as f:
        info = json.load(f)

    all_idx = np.arange(len(sample_table))
    loader = make_loader(
        all_idx, sample_table, mat, cfg, shuffle=False,
        y_mean=y_mean, y_std=y_std, cell_expr_override=cell_expr,
    )
    save_path = None
    if save_dir is not None:
        ds_name = info.get("pset", eval_path.name)
        save_path = os.path.join(save_dir, f"crosspred_{ds_name}_seed{cfg['seed']}.npz")
    m = test_metrics(model, loader, device, y_mean, y_std, save_path=save_path)
    m.update({"dataset": info.get("pset", eval_path.name),
              "measure": info.get("measure", "?"),
              "n_samples": len(sample_table)})
    return m

def run_cross(cfg, mat, gene_pathway_t, align_branches, eval_dirs, device, out_root):
    out_dir = os.path.join(out_root, "cross_dataset", "_".join(align_branches))
    os.makedirs(out_dir, exist_ok=True)

    set_seed(cfg["seed"])

    sample_table = mat["sample_table"]
    val_strategy = cfg.get("val_strategy", "drug")
    val_frac     = cfg.get("val_frac", 0.1)
    rng = np.random.default_rng(cfg["seed"])

    if val_strategy == "drug":
        drugs = np.array(sorted(sample_table["drug_name_lower"].unique()))
        n_val = max(1, int(len(drugs) * val_frac))
        val_drugs   = set(drugs[rng.choice(len(drugs), n_val, replace=False)])
        train_drugs = set(drugs) - val_drugs
        tr_idx = sample_table.index[sample_table["drug_name_lower"].isin(train_drugs)].values
        va_idx = sample_table.index[sample_table["drug_name_lower"].isin(val_drugs)].values
    elif val_strategy == "random":
        all_idx = np.arange(len(sample_table))
        rng.shuffle(all_idx)
        n_val  = max(1, int(len(all_idx) * val_frac))
        va_idx = all_idx[:n_val]
        tr_idx = all_idx[n_val:]
    elif val_strategy == "cell":
        cells = np.array(sorted(sample_table["cell_name"].unique()))
        n_val = max(1, int(len(cells) * val_frac))
        val_cells = set(cells[rng.choice(len(cells), n_val, replace=False)])
        tr_idx = sample_table.index[~sample_table["cell_name"].isin(val_cells)].values
        va_idx = sample_table.index[ sample_table["cell_name"].isin(val_cells)].values
    else:
        raise ValueError(f"Unknown val_strategy: {val_strategy}")

    print(f"  val_strategy={val_strategy}  tr={len(tr_idx)}  va={len(va_idx)}")

    y_mean = float(sample_table.iloc[tr_idx]["ln_ic50"].mean())
    y_std  = float(sample_table.iloc[tr_idx]["ln_ic50"].std())

    def _loader(idx, shuffle):
        return make_loader(idx, sample_table, mat, cfg, shuffle, y_mean, y_std)

    model = build_model(cfg, gene_pathway_t, align_branches, device)

    cross_hook_every = int(cfg.get("cross_hook_every", 0))
    eval_hook = None
    if cross_hook_every > 0:
        def _hook(epoch, m):
            out = {}
            for ed in eval_dirs:
                me = eval_on_external(m, ed, mat, cfg, device, y_mean, y_std)
                ds = me["dataset"]
                out[f"cross_{ds}_pcc"]  = float(me["pcc"])
                out[f"cross_{ds}_scc"]  = float(me["spearman"])
                out[f"cross_{ds}_rmse"] = float(me["rmse"])
            return out
        eval_hook = _hook

    if cfg.get("eval_only"):
        best_path = os.path.join(out_dir, "best.pt")
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"  [eval_only] loaded {best_path} (training skipped; y_mean/y_std recomputed with the same seed)")
        best_ep = -1
    else:
        best_ep = train_loop(
            model, _loader(tr_idx, True), _loader(va_idx, False), cfg, device, out_dir,
            epochs=cfg["epochs"],
            early_stop=cfg.get("early_stop", 30),
            lr_patience=cfg.get("lr_patience", 7),
            eval_hook=eval_hook,
            hook_every=cross_hook_every if cross_hook_every > 0 else 5,
        )
        with open(os.path.join(out_dir, "train_stats.json"), "w") as f:
            json.dump({"y_mean": y_mean, "y_std": y_std, "best_epoch": best_ep}, f)

    results = []
    for eval_dir in eval_dirs:
        print(f"\n  Eval on {eval_dir}")
        m = eval_on_external(model, eval_dir, mat, cfg, device, y_mean, y_std, save_dir=out_dir)
        m["align"] = "_".join(align_branches)
        m["best_epoch"] = best_ep
        results.append(m)
        print(f"    {m['dataset']:12s} | RMSE {m['rmse']:.4f}  "
              f"PCC {m['pcc']:.4f}  Spearman {m['spearman']:.4f}")

    with open(os.path.join(out_dir, "cross_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode",   choices=["random", "drug5", "cross"], required=True)
    parser.add_argument("--align",  choices=["gene", "pathway", "both"], default="both")
    parser.add_argument("--eval_dirs", nargs="+",
                        help="cross mode only: external evaluation dataset directories")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="override cfg['seed'] (multi-seed significance runs)")
    parser.add_argument("--tag", type=str, default=None,
                        help="suffix appended to out_dir so seeds don't collide")
    parser.add_argument("--eval_only", action="store_true",
                        help="skip training, load best.pt, run cross evaluation and dump per-pair predictions (npz)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed

    cfg["eval_only"] = args.eval_only

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    elif "cuda_device" in cfg:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_device"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    align_branches = ALIGN_PRESETS[args.align]
    print(f"Device: {device}  |  mode: {args.mode}  |  align: {align_branches}")

    mat = load_matrices(cfg["mat_dir"],
                        cfg.get("hcdt_drug_gene_file", "hcdt_drug_gene.npy"),
                        cfg.get("hcdt_drug_path_file", "hcdt_drug_path_direct.npy"))
    cfg["fp_dim"]       = mat["drug_fp"].shape[1]
    cfg["num_genes"]    = mat["cell_expr"].shape[1]
    cfg["num_pathways"] = mat["gene_pathway"].shape[1]
    gene_pathway_t = torch.tensor(mat["gene_pathway"], dtype=torch.float32)

    out_root = cfg.get("out_dir", "results/hdca_gdsc12")
    if args.tag:
        out_root = f"{out_root}_{args.tag}"
    os.makedirs(out_root, exist_ok=True)

    if args.mode == "random":
        run_random(cfg, mat, gene_pathway_t, align_branches, device, out_root)
    elif args.mode == "drug5":
        run_drug5(cfg, mat, gene_pathway_t, align_branches, device, out_root)
    else:
        if not args.eval_dirs:
            raise SystemExit("--eval_dirs required for cross mode")
        run_cross(cfg, mat, gene_pathway_t, align_branches,
                  args.eval_dirs, device, out_root)

if __name__ == "__main__":
    main()
