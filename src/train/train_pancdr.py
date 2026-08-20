"""
Train the PANCDR baseline (adversarial domain-adaptation CDR model) in our harness.

Usage
-----
# cross-dataset (GDSC1+2 -> CCLE / gCSI); target = unlabeled CCLE+gCSI expression
PYTHONPATH=. python src/train/train_pancdr.py \
    --mat_dir data/matrices_gdsc12 --split cross \
    --eval_dirs data/matrices_ccle_2015 data/matrices_gcsi_2019 \
    --gpu 6 --out_dir results/baselines

# random split (in-distribution, Table 1) and drug 5-fold (Table 2):
PYTHONPATH=. python src/train/train_pancdr.py --mat_dir data/matrices_gdsc12 --split random --gpu 6
PYTHONPATH=. python src/train/train_pancdr.py --mat_dir data/matrices_gdsc12 --split drug   --gpu 6

Notes
-----
* PANCDR needs an *unlabeled target-domain* expression set. For `cross` we feed the
  CCLE+gCSI expression (transductive). For `random`/`drug` there is no separate
  target domain, so the adversarial term is disabled (lam=0) and the model reduces
  to its graph-GCN + expression-encoder CDR predictor. This is reported honestly.
* Drug graphs (drug_atom_feats.npy / drug_adj_norm.npy) are shared across datasets
  (same drug_idx space), so they only need to be built once for GDSC12 via
  src/data/build_drug_graphs.py.
"""
import os, sys, argparse, json, time, logging, itertools
from pathlib import Path
from itertools import cycle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from src.models.baselines.pancdr import PANCDR
from src.data.dataset_pancdr import PancdrDataset, ExprOnlyDataset
from src.data.split import random_split, drug_kfold
from src.utils.seed import set_seed
from src.utils.metrics import compute_metrics

SEED = 42

def get_params(args):
    return dict(
        nz=args.nz, d_dim=args.d_dim, lr=args.lr, lr_adv=args.lr_adv,
        lam=args.lam, batch_size=args.batch_size, tgt_batch=args.tgt_batch,
        epochs=args.epochs, patience=args.patience, num_workers=args.num_workers,
    )

def zscore(expr, eps=1e-8):
    """Per-gene standardisation within a dataset (matches PANCDR z-scored expr)."""
    mu = expr.mean(axis=0, keepdims=True)
    sd = expr.std(axis=0, keepdims=True)
    return ((expr - mu) / (sd + eps)).astype(np.float32)

def valid_drug_mask(feats):
    """Drugs whose SMILES resolved have a non-zero atom-feature matrix; the rest
    are zero-graphs (unresolved names / biologics) and MUST be excluded -- feeding
    identical zero inputs for many drugs would corrupt training."""
    return np.abs(feats).reshape(feats.shape[0], -1).sum(axis=1) > 0

def load_matrices(mat_dir, log=None):
    def L(f):
        return np.load(os.path.join(mat_dir, f))
    feats = L("drug_atom_feats.npy")
    adj = L("drug_adj_norm.npy")
    expr = zscore(L("cell_expr.npy"))
    tbl = pd.read_csv(os.path.join(mat_dir, "sample_table.csv"))
    valid = valid_drug_mask(feats)
    n0 = len(tbl)
    tbl = tbl[valid[tbl["drug_idx"].values]].reset_index(drop=True)
    msg = (f"[graph coverage] valid drugs {int(valid.sum())}/{len(valid)}; "
           f"samples {len(tbl):,}/{n0:,} kept "
           f"(dropped {n0 - len(tbl):,} zero-graph rows)")
    (log.info if log else print)(msg)
    return feats, adj, expr, tbl, valid

def drug_based_train_val_split(sample_tbl, seed, val_frac=0.1):
    rng = np.random.default_rng(seed)
    drugs = np.array(sorted(sample_tbl["drug_name_lower"].unique()))
    n_val = max(1, int(len(drugs) * val_frac))
    val_drugs = set(drugs[rng.choice(len(drugs), n_val, replace=False)])
    train_drugs = set(drugs) - val_drugs
    tr = sample_tbl.index[sample_tbl["drug_name_lower"].isin(train_drugs)].values
    va = sample_tbl.index[sample_tbl["drug_name_lower"].isin(val_drugs)].values
    return tr, va

def make_loader(idx, tbl, feats, adj, expr, y_mean, y_std, bs, nw, shuffle):
    ds = PancdrDataset(idx, tbl, feats, adj, expr, y_mean, y_std)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      pin_memory=True, drop_last=shuffle)

def train_pancdr(model, tr_loader, va_loader, target_expr, params, device,
                 out_dir, y_mean, y_std, log):
    os.makedirs(out_dir, exist_ok=True)
    bce = torch.nn.BCELoss()
    mse = torch.nn.MSELoss()
    opt = torch.optim.Adam(itertools.chain(model.encoder.parameters(),
                                           model.cdr.parameters()), lr=params["lr"])
    opt_adv = torch.optim.Adam(model.disc.parameters(), lr=params["lr_adv"])

    use_adv = params["lam"] > 0 and target_expr is not None
    if use_adv:
        tgt_loader = DataLoader(ExprOnlyDataset(target_expr),
                                batch_size=params["tgt_batch"], shuffle=True,
                                drop_last=True, num_workers=params["num_workers"])

    best_p, best_ep, wait = -1.0, 0, 0
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        pairs = zip(tr_loader, cycle(tgt_loader)) if use_adv else tr_loader
        for batch in pairs:
            if use_adv:
                (drug_feat, drug_adj, gexpr, y), t_gexpr = batch
                t_gexpr = t_gexpr.to(device)
            else:
                drug_feat, drug_adj, gexpr, y = batch
            drug_feat = drug_feat.to(device)
            drug_adj = drug_adj.to(device)
            gexpr = gexpr.to(device)
            y = y.view(-1, 1).to(device)

            if use_adv:

                opt_adv.zero_grad()
                z_s, _, _ = model.encoder(gexpr)
                z_t, _, _ = model.encoder(t_gexpr)
                z_cat = torch.cat((z_s, z_t))
                d_true = torch.cat((torch.zeros(z_s.shape[0], 1, device=device),
                                    torch.ones(z_t.shape[0], 1, device=device)))
                d_loss = bce(model.disc(z_cat.detach()), d_true)
                d_loss.backward()
                opt_adv.step()

            opt.zero_grad()
            z_s, _, _ = model.encoder(gexpr)
            y_pred = model.cdr(drug_feat, drug_adj, z_s)
            cdr_loss = mse(y_pred, y)
            if use_adv:
                z_t, _, _ = model.encoder(t_gexpr)
                z_cat = torch.cat((z_s, z_t))
                d_flip = torch.cat((torch.ones(z_s.shape[0], 1, device=device),
                                    torch.zeros(z_t.shape[0], 1, device=device)))
                adv_loss = bce(model.disc(z_cat), d_flip)
                loss = cdr_loss + params["lam"] * adv_loss
            else:
                loss = cdr_loss
            loss.backward()
            opt.step()

        p_val = evaluate(model, va_loader, device, y_mean, y_std)["pcc"]
        if p_val >= best_p:
            best_p, best_ep, wait = p_val, epoch, 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        else:
            wait += 1
            if wait >= params["patience"]:
                break
        log.info(f"  ep {epoch:3d}  val_pcc={p_val:.4f}  (best {best_p:.4f}@{best_ep})")

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    return best_ep

@torch.no_grad()
def evaluate(model, loader, device, y_mean, y_std, save_path=None):
    model.eval()
    ys, preds = [], []
    for drug_feat, drug_adj, gexpr, y in loader:
        y_pred, _ = model(drug_feat.to(device), drug_adj.to(device),
                          gexpr.to(device), sample=False)
        ys.append(y.numpy())
        preds.append(y_pred.squeeze(-1).cpu().numpy())
    y_true = np.concatenate(ys) * y_std + y_mean
    y_pred = np.concatenate(preds) * y_std + y_mean
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if save_path is not None:

        np.savez(save_path, y_true=y_true, y_pred=y_pred)
    return compute_metrics(y_true[mask], y_pred[mask])

def eval_external(model, eval_dir, feats, adj, valid, y_mean, y_std, device, params, log, save_path=None):
    p = Path(eval_dir)
    tbl = pd.read_csv(p / "sample_table.csv")
    n0 = len(tbl)
    tbl = tbl[valid[tbl["drug_idx"].values]].reset_index(drop=True)
    expr = zscore(np.load(p / "cell_expr.npy"))
    log.info(f"  [{p.name}] eval drugs kept {tbl['drug_idx'].nunique()} | "
             f"samples {len(tbl):,}/{n0:,} (dropped {n0 - len(tbl):,} zero-graph)")
    info = json.loads((p / "eval_info.json").read_text()) if (p / "eval_info.json").exists() else {}
    loader = make_loader(np.arange(len(tbl)), tbl, feats, adj, expr,
                         y_mean, y_std, params["batch_size"], params["num_workers"], shuffle=False)
    m = evaluate(model, loader, device, y_mean, y_std, save_path=save_path)
    m["dataset"] = info.get("pset", p.name)
    m["n_samples"] = int(len(tbl))
    return m

def build_model(expr, feats, params, device):
    return PANCDR(n_genes=expr.shape[1], atom_feat_dim=feats.shape[-1],
                  nz=params["nz"], d_dim=params["d_dim"], is_regr=True).to(device)

def setup_logger(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lg = logging.getLogger("pancdr")
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for h in (logging.FileHandler(path), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--split", choices=["random", "drug", "cross"], required=True)
    ap.add_argument("--eval_dirs", nargs="+", default=None)
    ap.add_argument("--out_dir", default="results/baselines")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true",
                    help="skip training, load cross_dataset/PANCDR/best.pt, evaluate and dump predictions (npz)")
    ap.add_argument("--seed", type=int, default=42,
                    help="multi-seed reproduction: cross best.pt is isolated under seed<N>/, npz files share the model directory")
    ap.add_argument("--nz", type=int, default=256)
    ap.add_argument("--d_dim", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_adv", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--tgt_batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    global SEED
    SEED = args.seed

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = get_params(args)
    log = setup_logger(os.path.join(args.out_dir, f"pancdr_{args.split}.log"))
    log.info(f"device={device}  split={args.split}  params={params}")

    feats, adj, expr, tbl, valid = load_matrices(args.mat_dir, log)
    log.info(f"drug graphs {feats.shape}, expr {expr.shape}, samples {len(tbl):,}")
    results = []

    if args.split == "cross":
        if not args.eval_dirs:
            raise SystemExit("--eval_dirs required for cross")
        set_seed(SEED)
        tr, va = drug_based_train_val_split(tbl, SEED)
        y_mean = float(tbl.iloc[tr]["ln_ic50"].mean())
        y_std = float(tbl.iloc[tr]["ln_ic50"].std())
        target_expr = np.concatenate([zscore(np.load(Path(d) / "cell_expr.npy"))
                                      for d in args.eval_dirs], axis=0)
        model = build_model(expr, feats, params, device)
        model_dir = os.path.join(args.out_dir, "cross_dataset", "PANCDR")
        out_dir   = os.path.join(model_dir, f"seed{SEED}")
        os.makedirs(out_dir, exist_ok=True)
        tr_loader = make_loader(tr, tbl, feats, adj, expr, y_mean, y_std,
                                params["batch_size"], params["num_workers"], True)
        va_loader = make_loader(va, tbl, feats, adj, expr, y_mean, y_std,
                                params["batch_size"], params["num_workers"], False)
        if args.eval_only:
            model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                             map_location=device))
            log.info(f"  [eval_only] loaded {out_dir}/best.pt (training skipped)")
            best_ep = -1
        else:
            best_ep = train_pancdr(model, tr_loader, va_loader, target_expr, params,
                                   device, out_dir, y_mean, y_std, log)
        for d in args.eval_dirs:
            m = eval_external(model, d, feats, adj, valid, y_mean, y_std, device, params, log,
                              save_path=os.path.join(model_dir, f"crosspred_{Path(d).name}_seed{SEED}.npz"))
            m.update({"model": "PANCDR", "split": "cross", "best_epoch": best_ep})
            results.append(m)
            log.info(f"  [PANCDR/{m['dataset']}] RMSE={m['rmse']:.4f} "
                     f"PCC={m['pcc']:.4f} SCC={m['spearman']:.4f} n={m['n_samples']}")
        json.dump(results, open(os.path.join(out_dir, "cross_metrics.json"), "w"), indent=2)

    elif args.split == "random":
        set_seed(SEED)
        tr, va, te = random_split(tbl, seed=SEED)
        y_mean = float(tbl.iloc[tr]["ln_ic50"].mean())
        y_std = float(tbl.iloc[tr]["ln_ic50"].std())
        model = build_model(expr, feats, params, device)
        out_dir = os.path.join(args.out_dir, "random", "PANCDR")
        tr_loader = make_loader(tr, tbl, feats, adj, expr, y_mean, y_std,
                                params["batch_size"], params["num_workers"], True)
        va_loader = make_loader(va, tbl, feats, adj, expr, y_mean, y_std,
                                params["batch_size"], params["num_workers"], False)
        te_loader = make_loader(te, tbl, feats, adj, expr, y_mean, y_std,
                                params["batch_size"], params["num_workers"], False)
        best_ep = train_pancdr(model, tr_loader, va_loader, None, params,
                               device, out_dir, y_mean, y_std, log)
        m = evaluate(model, te_loader, device, y_mean, y_std)
        m.update({"model": "PANCDR", "split": "random", "best_epoch": best_ep})
        results.append(m)
        log.info(f"  [PANCDR/random] RMSE={m['rmse']:.4f} PCC={m['pcc']:.4f} SCC={m['spearman']:.4f}")
        json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)

    else:
        for fold, tr, va, te in drug_kfold(tbl, n_splits=5, seed=SEED):
            set_seed(SEED + fold)
            y_mean = float(tbl.iloc[tr]["ln_ic50"].mean())
            y_std = float(tbl.iloc[tr]["ln_ic50"].std())
            model = build_model(expr, feats, params, device)
            out_dir = os.path.join(args.out_dir, "drug_fold", "PANCDR", f"fold{fold+1}")
            tr_loader = make_loader(tr, tbl, feats, adj, expr, y_mean, y_std,
                                    params["batch_size"], params["num_workers"], True)
            va_loader = make_loader(va, tbl, feats, adj, expr, y_mean, y_std,
                                    params["batch_size"], params["num_workers"], False)
            te_loader = make_loader(te, tbl, feats, adj, expr, y_mean, y_std,
                                    params["batch_size"], params["num_workers"], False)
            best_ep = train_pancdr(model, tr_loader, va_loader, None, params,
                                   device, out_dir, y_mean, y_std, log)
            m = evaluate(model, te_loader, device, y_mean, y_std)
            m.update({"model": "PANCDR", "split": "drug", "fold": fold + 1, "best_epoch": best_ep})
            results.append(m)
            log.info(f"  [PANCDR/drug fold{fold+1}] PCC={m['pcc']:.4f} SCC={m['spearman']:.4f}")
        pcc = np.mean([r["pcc"] for r in results])
        scc = np.mean([r["spearman"] for r in results])
        log.info(f"  [PANCDR/drug5 mean] PCC={pcc:.4f}+/-{np.std([r['pcc'] for r in results]):.4f} SCC={scc:.4f}")
        json.dump(results, open(os.path.join(args.out_dir, "drug_fold", "PANCDR", "summary.json"), "w"), indent=2)

    pd.DataFrame(results).to_csv(
        os.path.join(args.out_dir, f"pancdr_results_{args.split}.csv"), index=False)
    log.info("done.")

if __name__ == "__main__":
    main()
