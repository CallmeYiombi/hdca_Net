"""Does the gene attention actually drive the prediction, or is it decorative?

The MoA results say the gene branch attends to the right genes. They do not say the
prediction depends on those genes: an attention map can agree with biology and still be
bypassed by the rest of the network. This intervenes on the input to find out.

For each test pair we read the gene attention, then re-predict with the expression of
selected genes set to zero -- the per-gene mean, since expression is z-scored -- and
measure how far the prediction moves:

  top-k        the k genes the model attends to most
  mask-random  k genes drawn from the same drug's mask row and disjoint from the top-k, so
               the contrast is not confounded by masked-versus-unmasked
  any-random   k genes drawn from outside the drug's mask, the loosest control

The two random conditions are averaged over --repeats independent draws per pair, so the
control is an expectation rather than a single draw.

If the attention is faithful, erasing top-k moves the prediction substantially more than
either control. If all three move it about equally, the attention is not what the
prediction rests on, whatever it agrees with.

Forward-only: no training, no gradients. Usage:
  python scripts/faithfulness_test.py [--seeds 1 2 3] [--n 512] [--device mps]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.hdca_net import HDCANet  # noqa: E402
from src.data.split import random_split  # noqa: E402

MD = "data/matrices_gdsc12"
# The six-seed reference runs keep only their per-pair dumps locally, so the default is
# the seed-42 reference checkpoint. On the server, pass the six with --ckpt.
DEFAULT_CKPT = ["results/hdca_gdsc12_pruned_div03/cross_dataset/gene_pathway/best.pt"]
KS = (1, 5, 10, 25)


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_run(ckpt, device, cfg):
    mat = {f: np.load(os.path.join(MD, f + ".npy")) for f in
           ("drug_fp", "cell_expr", "hcdt_drug_gene_pruned", "hcdt_drug_disease",
            "hcdt_drug_path_direct", "hcdt_neg_drug_gene", "gene_pathway")}
    gp = torch.tensor(mat["gene_pathway"], dtype=torch.float32, device=device)
    model = HDCANet(
        fp_dim=mat["drug_fp"].shape[1],
        num_genes=mat["cell_expr"].shape[1],
        num_pathways=mat["gene_pathway"].shape[1],
        gene_pathway_matrix=gp,
        drug_enc_hidden=cfg.get("drug_enc_hidden", 512),
        drug_enc_out=cfg.get("drug_enc_out", 256),
        gate_mode=cfg.get("gate_mode", "hard"),
        gate_gamma=cfg.get("gate_gamma", 2.0),
        fp_input_dropout=0.0,
        use_mutation=False,
        dropout=cfg.get("dropout", 0.3),
        lambda_neg=cfg.get("lambda_neg", 0.1),
        mask_dropout=0.0,
        align_branches=("gene", "pathway"),
    ).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    state = ck.get("model_state", ck.get("model_state_dict", ck))
    model.load_state_dict(state)
    model.eval()
    return model, mat


def scatter_zero(ex, gidx):
    e = ex.clone()
    e.scatter_(1, gidx, 0.0)
    return e


def predict(model, fp, ex, dg, dd, dp):
    y, _, _, _ = model(fp, ex, dg, dd, dp)
    return y.detach().float().cpu().numpy().ravel()


def run_seed(ckpt, idx, device, cfg, rng, repeats=5):
    model, mat = load_run(ckpt, device, cfg)
    fp = torch.tensor(mat["drug_fp"][idx["drug"]], dtype=torch.float32, device=device)
    ex = torch.tensor(mat["cell_expr"][idx["cell"]], dtype=torch.float32, device=device)
    dg = torch.tensor(mat["hcdt_drug_gene_pruned"][idx["drug"]], dtype=torch.float32, device=device)
    dd = torch.tensor(mat["hcdt_drug_disease"][idx["drug"]], dtype=torch.float32, device=device)
    dp = torch.tensor(mat["hcdt_drug_path_direct"][idx["drug"]], dtype=torch.float32, device=device)

    with torch.no_grad():
        base = predict(model, fp, ex, dg, dd, dp)
        attn = model.get_alignment_scores(fp, ex, dg, dp)["gene_attn"]  # (B, G)

        # --- genes that carry the explanation against the rest of the same mask ------
        # To answer the objection that the model is only handing back the annotation it was
        # given, erase, separately, the genes belonging to the pathway the gene branch ranks
        # first for that pair and the genes in the same mask that do not belong to it. Both
        # sets were supplied to the model, so a difference is predictive dependence rather
        # than possession of the annotation.
        gpm = torch.tensor(mat["gene_pathway"], dtype=torch.float32, device=device)
        A = gpm / torch.sqrt(gpm.sum(0).clamp(min=1.0))
        pg = (attn * ex) @ A                                           # (B, P)
        top_path = pg.argmax(dim=1)                                    # (B,)
        is_member = (gpm[:, top_path].T > 0)                           # (B, G)

        order = torch.argsort(attn, dim=1, descending=True)
        out, keep = {}, {}
        for k in KS:
            out[(k, "top-k")] = predict(model, fp, scatter_zero(ex, order[:, :k]),
                                        dg, dd, dp) - base
            acc = {"mask-random": [], "any-random": []}
            for _ in range(repeats):
                # k genes from the drug's own mask row, disjoint from the top-k. A pair
                # is usable only if the mask has 2k entries, otherwise the two sets
                # would necessarily overlap and the contrast would be vacuous.
                mr, usable = [], []
                for i in range(dg.shape[0]):
                    nz = torch.nonzero(dg[i], as_tuple=False).ravel()
                    top = set(order[i, :k].tolist())
                    rest = torch.tensor([g for g in nz.tolist() if g not in top],
                                        dtype=torch.long, device=device)
                    if len(rest) < k:
                        mr.append(order[i, :k])      # placeholder, masked out below
                        usable.append(False)
                    else:
                        pick = torch.tensor(rng.choice(len(rest), size=k, replace=False),
                                            device=device)
                        mr.append(rest[pick]); usable.append(True)
                keep[k] = np.array(usable)
                acc["mask-random"].append(
                    predict(model, fp, scatter_zero(ex, torch.stack(mr)), dg, dd, dp) - base)
                ar = []
                for i in range(dg.shape[0]):
                    nzi = torch.nonzero(dg[i], as_tuple=False).ravel().tolist()
                    cand = np.setdiff1d(np.arange(ex.shape[1]), np.array(nzi, dtype=int),
                                        assume_unique=False)
                    ar.append(torch.tensor(rng.choice(cand, size=k, replace=False),
                                           device=device))
                anyr = torch.stack(ar)
                acc["any-random"].append(
                    predict(model, fp, scatter_zero(ex, anyr), dg, dd, dp) - base)
            for cond, want in (("path-member", True), ("path-other", False)):
                picks, ok = [], []
                for i in range(dg.shape[0]):
                    nz = torch.nonzero(dg[i], as_tuple=False).ravel()
                    sel = [g for g in nz.tolist() if bool(is_member[i, g]) == want]
                    if len(sel) < k:
                        picks.append(order[i, :k]); ok.append(False)
                    else:
                        j = rng.choice(len(sel), size=k, replace=False)
                        picks.append(torch.tensor([sel[t] for t in j],
                                                  dtype=torch.long, device=device))
                        ok.append(True)
                d_ = predict(model, fp, scatter_zero(ex, torch.stack(picks)), dg, dd, dp) - base
                out[(k, cond)] = np.abs(d_)
                keep[(k, cond)] = np.array(ok)

            for name, deltas in acc.items():
                out[(k, name)] = np.mean(np.abs(np.stack(deltas)), axis=0)
    return base, out, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=DEFAULT_CKPT)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None,
                    help="write the results to this JSON file. The random controls are "
                         "re-drawn on every run, so the JSON is the record of a given run.")
    ap.add_argument("--repeats", type=int, default=5,
                    help="independent draws per pair for the random controls")
    args = ap.parse_args()

    device = pick_device(args.device)
    cfg = yaml.safe_load(open("configs/hdca_gdsc12_cross_pruned_div03.yaml")) \
        if os.path.exists("configs/hdca_gdsc12_cross_pruned_div03.yaml") else {}

    st = pd.read_csv(os.path.join(MD, "sample_table.csv"))
    dgm = np.load(os.path.join(MD, "hcdt_drug_gene_pruned.npy"))
    has_mask = dgm.sum(1) > 0
    _, _, te = random_split(st, seed=1)
    te = np.asarray(te)
    te = te[has_mask[st.drug_idx.values[te]]]          # gene branch has to be active
    rng = np.random.default_rng(0)
    pick = rng.choice(len(te), size=min(args.n, len(te)), replace=False)
    rows = st.iloc[te[pick]]
    idx = {"drug": rows.drug_idx.values, "cell": rows.cell_idx.values}
    y_std = float(st.ln_ic50.std())

    print(f"device {device} | {len(rows)} held-out pairs, "
          f"{rows.drug_idx.nunique()} compounds with a direct-target mask")
    print(f"predictions are z-scored; reported in log-IC50 units (x{y_std:.3f})\n")

    agg, nkeep = {}, {}
    for s in args.ckpt:
        _, out, keep = run_seed(s, idx, device, cfg, rng, args.repeats)
        for (k, name), d in out.items():
            # every condition is averaged over the same pairs -- the ones on which the
            # disjoint mask-random control is defined -- so the ratios compare like with like
            if name in ("path-member", "path-other"):
                m = keep[k] & keep[(k, "path-member")] & keep[(k, "path-other")]
            else:
                m = keep[k]
            if m.sum() == 0:
                continue
            agg.setdefault((k, name), []).append(float(np.abs(d[m]).mean()) * y_std)
            nkeep[k] = int(keep[k].sum())
            if name == "path-member":
                nkeep[(k, "path")] = int(m.sum())
        print(f"  {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(s))))} done")

    dump = dict(n_pairs=args.n, repeats=args.repeats, ckpt=list(args.ckpt),
                eligible={f"k={k}": int(nkeep[k]) for k in KS}, erasure={}, path_member={})

    print(f"\n{'k':>4}  {'top-k':>16}  {'mask-random':>16}  {'any-random':>16}   ratio")
    for k in KS:
        v = {n: np.array(agg[(k, n)]) for n in ("top-k", "mask-random", "any-random")}
        # note: mask-random is averaged over the usable pairs only
        r = v["top-k"].mean() / max(v["mask-random"].mean(), 1e-9)
        dump["erasure"][f"k={k}"] = dict(
            n_pairs=int(nkeep[k]), ratio=float(r),
            **{n.replace("-", "_"): dict(mean=float(v[n].mean()), sd=float(v[n].std()),
                                         per_seed=[float(x) for x in v[n]])
               for n in ("top-k", "mask-random", "any-random")})
        print(f"{k:>4}  " + "  ".join(f"{v[n].mean():8.4f}+/-{v[n].std():5.4f}"
                                      for n in ("top-k", "mask-random", "any-random"))
              + f"   {r:5.1f}x")
    print(f"\n{'k':>4}  {'path-member':>16}  {'path-other':>16}   ratio   n_pairs")
    for k in KS:
        if (k, "path-member") not in agg:
            continue
        a = np.array(agg[(k, "path-member")]); b = np.array(agg[(k, "path-other")])
        dump["path_member"][f"k={k}"] = dict(
            n_pairs=int(nkeep.get((k, "path"), 0)),
            ratio=float(a.mean() / max(b.mean(), 1e-9)),
            path_member=dict(mean=float(a.mean()), sd=float(a.std()),
                             per_seed=[float(x) for x in a]),
            path_other=dict(mean=float(b.mean()), sd=float(b.std()),
                            per_seed=[float(x) for x in b]))
        print(f"{k:>4}  {a.mean():8.4f}+/-{a.std():5.4f}  {b.mean():8.4f}+/-{b.std():5.4f}"
              f"   {a.mean()/max(b.mean(),1e-9):5.2f}x   {nkeep.get((k,'path'),0)}")
    print("  (both drawn from the drug's own mask; membership is in the pathway the gene")
    print("   branch ranks first for that pair, so the contrast is annotation-matched)")

    print(f"\nmean |change in predicted log-IC50| when the listed genes are erased;\nrandom conditions averaged over {args.repeats} draws per pair.")
    print("all three conditions are scored on the pairs whose mask has >=2k genes,")
    print("so the mask-random control is defined and disjoint from top-k:")
    print("   " + ", ".join(f"k={k}: {nkeep[k]}/{args.n}" for k in KS))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(dump, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
