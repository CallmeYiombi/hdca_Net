"""
Benchmark: GraphDRP, DeepCDR and TGSA against HDCA-Net.

All models use 2048-dim Morgan fingerprints as drug input for fair comparison.
Runs random-split (80/10/10), drug-5fold, and cross-dataset evaluations.

Usage:
  python src/train/train_baselines.py --mat_dir data/matrices_gdsc12 \\
         --hcdt_dir data/HCDT2.0 --split random --gpu 0
  python src/train/train_baselines.py --mat_dir data/matrices_gdsc12 \\
         --hcdt_dir data/HCDT2.0 --split drug  --gpu 0
  python src/train/train_baselines.py --mat_dir data/matrices_gdsc12 \\
         --hcdt_dir data/HCDT2.0 --split cross --gpu 0 \\
         --eval_dirs data/matrices_ccle_2015 data/matrices_gcsi_2019
"""
import os, sys, argparse, json, time, logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.baselines.graphdrp import GraphDRP
from src.models.baselines.deepcdr  import DeepCDR
from src.models.baselines.tgsa     import TGSA
from src.data.dataset_baseline     import BaselineDataset
from src.data.split                import random_split, drug_kfold
from src.utils.seed                import set_seed
from src.utils.metrics             import compute_metrics

EPOCHS      = 150
EARLY_STOP  = 30
LR_PATIENCE = 7
LR          = 3e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE  = 1024
DROPOUT     = 0.3
SEED        = 42
NUM_WORKERS = 4

def build_gene_pathway(hcdt_dir: str, gene_symbol_to_gidx: dict,
                       num_genes: int, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        return np.load(cache_path)

    pg = pd.read_excel(os.path.join(hcdt_dir, "Pathway_Gene.xlsx"))
    pg["path_clean"] = pg["PATH_NAME"].str.strip()
    pathway_list     = sorted(pg["path_clean"].dropna().unique())
    num_pathways     = len(pathway_list)
    p2idx            = {p: i for i, p in enumerate(pathway_list)}

    mat = np.zeros((num_genes, num_pathways), dtype=np.float32)
    for _, row in pg.iterrows():
        pidx = p2idx.get(row["path_clean"])
        if pidx is None or pd.isna(row.get("GENE_SYMBOL")):
            continue
        for sym in str(row["GENE_SYMBOL"]).split(","):
            gidx = gene_symbol_to_gidx.get(sym.strip().upper())
            if gidx is not None:
                mat[gidx, pidx] = 1.0

    np.save(cache_path, mat)
    logging.getLogger("baselines").info(f"gene_pathway built: {mat.shape}")
    return mat

def load_data(mat_dir: str, hcdt_dir: str):
    drug_fp    = np.load(os.path.join(mat_dir, "drug_fp.npy"))
    cell_expr  = np.load(os.path.join(mat_dir, "cell_expr.npy"))
    cell_mut   = np.load(os.path.join(mat_dir, "cell_mut.npy"))
    sample_tbl = pd.read_csv(os.path.join(mat_dir, "sample_table.csv"))

    with open(os.path.join(mat_dir, "id_maps_v2.json")) as f:
        id_maps = json.load(f)

    gene_sym_map = id_maps.get("gene_symbol_to_idx", {})
    num_genes    = drug_fp.shape[0]
    num_genes    = cell_expr.shape[1]

    cache = os.path.join(mat_dir, "gene_pathway.npy")
    gene_pathway = build_gene_pathway(hcdt_dir, gene_sym_map, num_genes, cache)

    return drug_fp, cell_expr, cell_mut, gene_pathway, sample_tbl

def make_model(name: str, fp_dim: int, num_genes: int,
               gene_pathway_t: torch.Tensor) -> nn.Module:
    if name == "GraphDRP":
        return GraphDRP(fp_dim=fp_dim, num_genes=num_genes, dropout=DROPOUT)
    if name == "DeepCDR":
        return DeepCDR(fp_dim=fp_dim, num_genes=num_genes, dropout=DROPOUT)
    if name == "TGSA":
        return TGSA(fp_dim=fp_dim, num_genes=num_genes,
                    gene_pathway_matrix=gene_pathway_t, dropout=DROPOUT)
    raise ValueError(name)

def make_loader(indices, sample_tbl, drug_fp, cell_expr, cell_mut,
                y_mean, y_std, shuffle):
    ds = BaselineDataset(indices, sample_tbl, drug_fp, cell_expr, cell_mut,
                         y_mean, y_std)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)

def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_y, all_pred = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for fp, expr, mut, y in loader:
            fp, expr, mut, y = fp.to(device), expr.to(device), mut.to(device), y.to(device)

            if isinstance(model, DeepCDR):
                y_pred = model(fp, expr, mut)
            else:
                y_pred = model(fp, expr)

            loss = nn.functional.mse_loss(y_pred.squeeze(), y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            all_y.append(y.cpu().numpy())
            all_pred.append(y_pred.squeeze().detach().cpu().numpy())

    y_cat = np.concatenate(all_y)
    m = compute_metrics(y_cat, np.concatenate(all_pred))
    m["loss"] = total_loss / len(y_cat)
    return m

def train_one(model, tr_loader, va_loader, te_loader,
              device, out_dir, y_mean, y_std):
    os.makedirs(out_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=0.5)

    best_val, best_ep = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        run_epoch(model, tr_loader, optimizer, device, train=True)
        va = run_epoch(model, va_loader, optimizer, device, train=False)
        scheduler.step(va["loss"])
        if va["loss"] < best_val:
            best_val, best_ep = va["loss"], epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        if epoch - best_ep >= EARLY_STOP:
            break

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                     map_location=device))
    model.eval()
    all_y, all_pred = [], []
    with torch.no_grad():
        for fp, expr, mut, y in te_loader:
            fp, expr, mut = fp.to(device), expr.to(device), mut.to(device)
            y_pred = model(fp, expr, mut) if isinstance(model, DeepCDR) else model(fp, expr)
            all_y.append(y.numpy())
            all_pred.append(y_pred.squeeze().cpu().numpy())

    y_true = np.concatenate(all_y) * y_std + y_mean
    y_pred = np.concatenate(all_pred) * y_std + y_mean
    return compute_metrics(y_true, y_pred), best_ep

def drug_based_train_val_split(sample_tbl, seed, val_frac=0.1):
    """Drug-stratified 90/10 validation split, identical to the HDCA-Net cross mode."""
    drugs = np.array(sorted(sample_tbl["drug_name_lower"].unique()))
    rng   = np.random.default_rng(seed)
    n_val = max(1, int(len(drugs) * val_frac))
    val_drugs   = set(drugs[rng.choice(len(drugs), n_val, replace=False)])
    train_drugs = set(drugs) - val_drugs
    tr_idx = sample_tbl.index[sample_tbl["drug_name_lower"].isin(train_drugs)].values
    va_idx = sample_tbl.index[sample_tbl["drug_name_lower"].isin(val_drugs)].values
    return tr_idx, va_idx

def eval_external(model, eval_dir, drug_fp, num_genes,
                  y_mean, y_std, device, log, save_path=None):
    """Run inference only on an external dataset (CCLE / gCSI) and return metrics."""
    eval_path    = Path(eval_dir)
    sample_table = pd.read_csv(eval_path / "sample_table.csv")
    cell_expr    = np.load(eval_path / "cell_expr.npy")

    mut_path = eval_path / "cell_mut.npy"
    if mut_path.exists():
        cell_mut = np.load(mut_path)
    else:
        log.info(f"  [{eval_path.name}] no cell_mut.npy; using a zero matrix")
        cell_mut = np.zeros((cell_expr.shape[0], num_genes), dtype=np.float32)

    info_path = eval_path / "eval_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    all_idx = np.arange(len(sample_table))
    ds = BaselineDataset(all_idx, sample_table, drug_fp, cell_expr, cell_mut,
                         y_mean, y_std)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    model.eval()
    all_y, all_pred = [], []
    with torch.no_grad():
        for fp, expr, mut, y in loader:
            fp, expr, mut = fp.to(device), expr.to(device), mut.to(device)
            y_pred = model(fp, expr, mut) if isinstance(model, DeepCDR) else model(fp, expr)
            all_y.append(y.numpy())
            all_pred.append(y_pred.squeeze().cpu().numpy())

    y_true = np.concatenate(all_y)    * y_std + y_mean
    y_pred = np.concatenate(all_pred) * y_std + y_mean
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if save_path is not None:

        np.savez(save_path, y_true=y_true, y_pred=y_pred)
    m = compute_metrics(y_true[mask], y_pred[mask])
    m["dataset"]   = info.get("pset", eval_path.name)
    m["measure"]   = info.get("measure", "?")
    m["n_samples"] = int(len(sample_table))
    return m

def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("baselines")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_dir",  default="data/matrices_gdsc12")
    parser.add_argument("--hcdt_dir", default="data/HCDT2.0")
    parser.add_argument("--out_dir",  default="results/baselines")
    parser.add_argument("--split",    default="random",
                        choices=["random", "drug", "cross"])
    parser.add_argument("--eval_dirs", nargs="+", default=None,
                        help="cross mode only: external evaluation dataset directories")
    parser.add_argument("--gpu",      type=int, default=0)
    parser.add_argument("--eval_only", action="store_true",
                        help="skip training, load cross_dataset/<model>/best.pt, evaluate and dump predictions (npz)")
    parser.add_argument("--seed", type=int, default=42,
                        help="multi-seed reproduction: cross best.pt is isolated under seed<N>/, npz files share the model directory")
    args = parser.parse_args()

    global SEED
    SEED = args.seed

    log_path = os.path.join(args.out_dir, f"train_{args.split}.log")
    log = setup_logger(log_path)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  log -> {log_path}")

    drug_fp, cell_expr, cell_mut, gene_pathway, sample_tbl = load_data(
        args.mat_dir, args.hcdt_dir)

    fp_dim         = drug_fp.shape[1]
    num_genes      = cell_expr.shape[1]
    gene_pathway_t = torch.from_numpy(gene_pathway).to(device)

    log.info(f"drug_fp={drug_fp.shape}  cell_expr={cell_expr.shape}  "
             f"gene_pathway={gene_pathway.shape}")

    MODELS = ["GraphDRP", "DeepCDR", "TGSA"]
    all_results = []

    if args.split == "random":
        log.info("=== Random split (80/10/10) ===")
        set_seed(SEED)
        tr_idx, va_idx, te_idx = random_split(sample_tbl, seed=SEED)
        y_mean = float(sample_tbl.iloc[tr_idx]["ln_ic50"].mean())
        y_std  = float(sample_tbl.iloc[tr_idx]["ln_ic50"].std())

        for mname in MODELS:
            set_seed(SEED)
            model   = make_model(mname, fp_dim, num_genes, gene_pathway_t).to(device)
            out_dir = os.path.join(args.out_dir, "random", mname)
            t0      = time.time()

            tr_loader = make_loader(tr_idx, sample_tbl, drug_fp, cell_expr, cell_mut,
                                    y_mean, y_std, shuffle=True)
            va_loader = make_loader(va_idx, sample_tbl, drug_fp, cell_expr, cell_mut,
                                    y_mean, y_std, shuffle=False)
            te_loader = make_loader(te_idx, sample_tbl, drug_fp, cell_expr, cell_mut,
                                    y_mean, y_std, shuffle=False)

            m, best_ep = train_one(model, tr_loader, va_loader, te_loader,
                                   device, out_dir, y_mean, y_std)
            elapsed = time.time() - t0
            log.info(f"[{mname:10s}] RMSE={m['rmse']:.4f}  PCC={m['pcc']:.4f}  "
                     f"SCC={m['spearman']:.4f}  best_ep={best_ep}  ({elapsed/60:.1f}min)")
            all_results.append({"split": "random", "model": mname,
                                 "fold": "-", "best_ep": best_ep, **m})

    elif args.split == "cross":
        if not args.eval_dirs:
            raise SystemExit("--eval_dirs required for cross mode")
        log.info(f"=== Cross-dataset (train: GDSC1+2, eval: {args.eval_dirs}) ===")
        set_seed(SEED)
        tr_idx, va_idx = drug_based_train_val_split(sample_tbl, SEED)
        y_mean = float(sample_tbl.iloc[tr_idx]["ln_ic50"].mean())
        y_std  = float(sample_tbl.iloc[tr_idx]["ln_ic50"].std())
        log.info(f"  train {len(tr_idx):,} / val {len(va_idx):,}  "
                 f"y_mean={y_mean:.3f}  y_std={y_std:.3f}")

        for mname in MODELS:
            set_seed(SEED)
            model     = make_model(mname, fp_dim, num_genes, gene_pathway_t).to(device)
            model_dir = os.path.join(args.out_dir, "cross_dataset", mname)
            out_dir   = os.path.join(model_dir, f"seed{SEED}")
            os.makedirs(out_dir, exist_ok=True)
            t0 = time.time()

            tr_loader = make_loader(tr_idx, sample_tbl, drug_fp, cell_expr, cell_mut,
                                    y_mean, y_std, shuffle=True)
            va_loader = make_loader(va_idx, sample_tbl, drug_fp, cell_expr, cell_mut,
                                    y_mean, y_std, shuffle=False)

            if args.eval_only:
                model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                                 map_location=device))
                log.info(f"  [eval_only] loaded {out_dir}/best.pt (training skipped)")
                best_ep = -1
            else:

                _, best_ep = train_one(model, tr_loader, va_loader, va_loader,
                                       device, out_dir, y_mean, y_std)
                with open(os.path.join(out_dir, "train_stats.json"), "w") as f:
                    json.dump({"y_mean": y_mean, "y_std": y_std,
                               "best_epoch": best_ep,
                               "n_train": int(len(tr_idx)),
                               "n_val":   int(len(va_idx))}, f, indent=2)

            ext_results = []
            for eval_dir in args.eval_dirs:
                ds_tag = Path(eval_dir).name
                m = eval_external(model, eval_dir, drug_fp, num_genes,
                                  y_mean, y_std, device, log,
                                  save_path=os.path.join(model_dir, f"crosspred_{ds_tag}_seed{SEED}.npz"))
                m.update({"model": mname, "best_epoch": best_ep})
                ext_results.append(m)
                log.info(f"  [{mname:10s} / {m['dataset']:12s}] "
                         f"RMSE={m['rmse']:.4f}  PCC={m['pcc']:.4f}  "
                         f"SCC={m['spearman']:.4f}  (n={m['n_samples']})")
                all_results.append({"split": "cross", **m,
                                    "fold": "-"})

            with open(os.path.join(out_dir, "cross_metrics.json"), "w") as f:
                json.dump(ext_results, f, indent=2)
            log.info(f"[{mname}] done ({(time.time()-t0)/60:.1f}min, best_ep={best_ep})")

    else:
        log.info("=== Drug 5-fold ===")
        for mname in MODELS:
            fold_metrics = []
            t0 = time.time()
            for fold, tr_idx, va_idx, te_idx in drug_kfold(sample_tbl, n_splits=5, seed=SEED):
                set_seed(SEED + fold)
                y_mean = float(sample_tbl.iloc[tr_idx]["ln_ic50"].mean())
                y_std  = float(sample_tbl.iloc[tr_idx]["ln_ic50"].std())

                model   = make_model(mname, fp_dim, num_genes, gene_pathway_t).to(device)
                out_dir = os.path.join(args.out_dir, "drug_fold", mname, f"fold{fold+1}")

                tr_loader = make_loader(tr_idx, sample_tbl, drug_fp, cell_expr, cell_mut,
                                        y_mean, y_std, shuffle=True)
                va_loader = make_loader(va_idx, sample_tbl, drug_fp, cell_expr, cell_mut,
                                        y_mean, y_std, shuffle=False)
                te_loader = make_loader(te_idx, sample_tbl, drug_fp, cell_expr, cell_mut,
                                        y_mean, y_std, shuffle=False)

                m, best_ep = train_one(model, tr_loader, va_loader, te_loader,
                                       device, out_dir, y_mean, y_std)
                fold_metrics.append(m)
                all_results.append({"split": "drug_fold", "model": mname,
                                    "fold": fold + 1, "best_ep": best_ep, **m})
                log.info(f"[{mname:10s} fold{fold+1}] RMSE={m['rmse']:.4f}  "
                         f"PCC={m['pcc']:.4f}  SCC={m['spearman']:.4f}  (ep {best_ep})")

            df      = pd.DataFrame(fold_metrics)
            elapsed = time.time() - t0
            log.info(f"[{mname} avg] RMSE={df['rmse'].mean():.4f}+/-{df['rmse'].std():.3f}  "
                     f"PCC={df['pcc'].mean():.4f}+/-{df['pcc'].std():.3f}  "
                     f"({elapsed/60:.1f}min)")

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"results_{args.split}.csv")
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    log.info(f"Saved -> {out_csv}")

if __name__ == "__main__":
    main()
