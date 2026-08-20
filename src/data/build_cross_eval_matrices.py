"""Build evaluation matrices for the external datasets (CCLE, gCSI).

Maps the PharmacoGx-exported response tables and expression profiles onto the gene and drug
universe used for training, and writes the per-dataset sample table and matrices consumed by
the cross-study evaluation.
"""
import os, sys, re, json, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[2]
DATA    = ROOT / "data"
GDSC12  = DATA / "matrices_gdsc12"
RAW_DIR = DATA / "pharmacodb" / "raw"

PREFERRED_MEASURES = ["ic50_recomputed", "IC50", "aac_recomputed", "AAC", "auc_recomputed"]

def normalize_drug_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r'\s*\(.*?\)', '', name)
    name = re.sub(r'\s*(dihydrochloride|hydrochloride|hcl|2hcl|mesylate|'
                  r'sulfate|acetate|sodium|tartrate|maleate|fumarate)\b', '', name)
    name = re.sub(r'[-\s]+', '', name)
    return name.strip()

def load_sensitivity(pset_dir: Path):
    for measure in PREFERRED_MEASURES:
        f = pset_dir / f"sensitivity_{measure}.csv"
        if f.exists():
            df = pd.read_csv(f, low_memory=False, encoding="latin-1")
            return df, measure
    return None, None

def build_pset_matrices(pset_dir: Path, id_maps: dict,
                        expr_raw: pd.DataFrame, model_df: pd.DataFrame,
                        out_root: Path):
    pset_name = pset_dir.name
    safe_name = pset_name.lower().replace("-", "_").replace(".", "_")
    out_dir   = out_root / f"matrices_{safe_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PSet: {pset_name}  ->  {out_dir.name}")

    sens_df, measure = load_sensitivity(pset_dir)
    if sens_df is None:
        print(f"  [SKIP] no sensitivity table")
        return None
    print(f"  measure: {measure}")

    drug_col  = "drug_name" if "drug_name" in sens_df.columns else sens_df.columns[0]
    cell_cols = [c for c in sens_df.columns if c != drug_col]
    print(f"  raw: {len(sens_df)} drugs x {len(cell_cols)} cells")

    norm_to_didx = {k: int(v) for k, v in id_maps["drug_norm_to_idx"].items()}

    rows = []
    for _, row in sens_df.iterrows():
        pset_drug = str(row[drug_col])
        d_idx     = norm_to_didx.get(normalize_drug_name(pset_drug))
        if d_idx is None:
            continue
        for cell_name in cell_cols:
            val = row[cell_name]
            if pd.isna(val):
                continue
            rows.append({
                "drug_idx":   d_idx,
                "drug_name":  pset_drug,
                "cell_name":  str(cell_name),
                "response":   float(val),
            })

    if not rows:
        print(f"  [SKIP] no compounds shared with GDSC1+2")
        return None

    long = pd.DataFrame(rows)
    print(f"  shared compounds: {long['drug_idx'].nunique()} / {len(sens_df)}")

    stripped_to_model = dict(zip(
        model_df["StrippedCellLineName"].str.upper(),
        model_df["ModelID"]
    ))

    def to_model_id(name):
        stripped = re.sub(r'[-\s_]', '', str(name)).upper()
        return stripped_to_model.get(stripped)

    long["model_id"] = long["cell_name"].apply(to_model_id)
    long = long[long["model_id"].notna()].copy()
    long = long[long["model_id"].isin(expr_raw.index)].copy()

    if len(long) < 100:
        print(f"  [SKIP] too few samples ({len(long)})")
        return None

    n_cells = long["model_id"].nunique()
    n_drugs = long["drug_idx"].nunique()
    print(f"  final: {len(long):,} samples, {n_drugs} drugs, {n_cells} cells")

    cell_list    = sorted(long["model_id"].unique())
    cell_to_cidx = {c: i for i, c in enumerate(cell_list)}
    long["cell_idx"]        = long["model_id"].map(cell_to_cidx)

    pos_mask = long["response"] > 0
    if (~pos_mask).any():
        print(f"  [WARN] IC50 <= 0 dropping {(~pos_mask).sum()} samples")
        long = long[pos_mask]
    long["ln_ic50"]         = np.log(long["response"])
    long["drug_name_lower"] = long["drug_name"].str.lower().str.strip()

    sample_table = long[[
        "drug_idx", "cell_idx", "drug_name", "cell_name",
        "model_id", "ln_ic50", "drug_name_lower"
    ]].reset_index(drop=True)

    gene_list = id_maps["gene_list"]
    num_cells = len(cell_list)
    num_genes = len(gene_list)

    expr_sub  = expr_raw.reindex(index=cell_list, columns=gene_list).fillna(0).astype(np.float32)
    cell_expr = expr_sub.values
    gene_std  = cell_expr.std(axis=0)
    gene_std[gene_std == 0] = 1.0
    cell_expr = (cell_expr - cell_expr.mean(axis=0)) / gene_std

    np.save(out_dir / "cell_expr.npy", cell_expr)
    np.save(out_dir / "cell_mut.npy",  np.zeros((num_cells, num_genes), dtype=np.float32))
    sample_table.to_csv(out_dir / "sample_table.csv", index=False)

    eval_info = {
        "pset":             pset_name,
        "measure":          measure,
        "n_samples":        len(sample_table),
        "n_drugs":          n_drugs,
        "n_cells":          num_cells,
        "cell_expr_shape":  [num_cells, num_genes],
        "gdsc12_mat_dir":   str(GDSC12),
    }
    with open(out_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2)

    print(f"  saved: {out_dir}")
    return out_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psets", nargs="+", default=None,
                        help="PSet directory names under data/pharmacodb/raw (default: auto-detect CCLE/gCSI)")
    args = parser.parse_args()

    if args.psets:
        pset_dirs = [RAW_DIR / p for p in args.psets]
    else:
        pset_dirs = [d for d in sorted(RAW_DIR.iterdir())
                     if d.is_dir() and any(k in d.name.upper()
                                           for k in ["CCLE", "GCSI", "gCSI"])]
        if not pset_dirs:
            print(f"Could not find the CCLE/gCSI directories; pass --psets explicitly.")
            print(f"Available PSets:")
            for d in sorted(RAW_DIR.iterdir()):
                if d.is_dir():
                    print(f"  {d.name}")
            sys.exit(1)

    print(f"PSets: {[d.name for d in pset_dirs]}")

    print("\nLoading GDSC1+2 id_maps...")
    with open(GDSC12 / "id_maps.json") as f:
        id_maps = json.load(f)
    print(f"  Genes: {len(id_maps['gene_list'])}, Drugs: {id_maps['num_drugs']}")

    print("\nLoading DepMap expression matrix...")
    expr_raw = pd.read_csv(DATA / "OmicsExpressionTPMLogp1.csv", low_memory=False)
    if "IsDefaultEntryForModel" in expr_raw.columns:
        col      = expr_raw["IsDefaultEntryForModel"]
        mask     = col.astype(str).str.upper().isin({"YES", "TRUE", "1"})
        filtered = expr_raw[mask]
        expr_raw = filtered if len(filtered) > 0 else expr_raw
    expr_raw = expr_raw.drop_duplicates("ModelID", keep="first").set_index("ModelID")
    gene_list = id_maps["gene_list"]
    expr_raw  = expr_raw.reindex(columns=gene_list).fillna(0).astype(np.float32)
    print(f"  {expr_raw.shape[0]} cells x {expr_raw.shape[1]} genes (aligned to the GDSC1+2 gene universe)")

    model_df = pd.read_csv(DATA / "Model.csv")
    print(f"  Model.csv: {len(model_df)} rows")

    results = []
    for pset_dir in pset_dirs:
        if not pset_dir.exists():
            print(f"[WARN] {pset_dir} missing, skipped")
            continue
        out = build_pset_matrices(pset_dir, id_maps, expr_raw, model_df, DATA)
        if out:
            results.append(out)

    print(f"\n{'='*60}")
    print(f"Done. Created {len(results)} evaluation datasets:")
    for r in results:
        print(f"  {r}")

    if results:
        eval_dirs_str = " ".join(str(r) for r in results)
        print(f"\nNext:")
        print(f"  python src/train/eval_cross_dataset.py \\")
        print(f"    --config configs/mp_hcpnet_gdsc12.yaml \\")
        print(f"    --eval_dirs {eval_dirs_str} \\")
        print(f"    --gpu 4")

if __name__ == "__main__":
    main()
