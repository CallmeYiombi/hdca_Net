"""Build the GDSC1+GDSC2 training matrices.

Merges the two GDSC releases, matches drug and cell-line identifiers, and writes the sample
table together with the expression, fingerprint and HCDT prior matrices.
"""
import os, sys, json, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HCDT = DATA / "HCDT2.0"
OUT  = DATA / "matrices_gdsc12"
OUT.mkdir(parents=True, exist_ok=True)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT = True
except ImportError:
    print("[WARN] RDKit not found: pip install rdkit")
    RDKIT = False

def normalize_drug_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r'\s*\(.*?\)', '', name)
    name = re.sub(r'\s*(dihydrochloride|hydrochloride|hcl|2hcl|mesylate|'
                  r'sulfate|acetate|sodium|tartrate|maleate|fumarate)\b', '', name)
    name = re.sub(r'[-\s]+', '', name)
    return name.strip()

def smiles_to_fp(smiles: str, radius: int = 2, nbits: int = 2048):
    if not RDKIT or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

print("[1] Loading GDSC1 + GDSC2 dose-response data...")
g1 = pd.read_excel(DATA / "GDSC1_fitted_dose_response_27Oct23.xlsx")
g2 = pd.read_excel(DATA / "GDSC2_fitted_dose_response_27Oct23.xlsx")
print(f"  GDSC1: {len(g1):,} rows  ({g1['DRUG_NAME'].nunique()} drugs, {g1['CELL_LINE_NAME'].nunique()} cells)")
print(f"  GDSC2: {len(g2):,} rows  ({g2['DRUG_NAME'].nunique()} drugs, {g2['CELL_LINE_NAME'].nunique()} cells)")

gdsc = pd.concat([g1, g2], ignore_index=True)
gdsc["_key"] = gdsc["DRUG_NAME"].str.lower().str.strip() + "||" + gdsc["CELL_LINE_NAME"].str.strip()
gdsc = gdsc.sort_values("DATASET", ascending=False)
gdsc = gdsc.drop_duplicates("_key", keep="first").drop(columns="_key")
gdsc = gdsc[gdsc["LN_IC50"].notna()].reset_index(drop=True)
print(f"  Combined (dedup, GDSC2 priority): {len(gdsc):,} rows  "
      f"({gdsc['DRUG_NAME'].nunique()} drugs, {gdsc['CELL_LINE_NAME'].nunique()} cells)")

print("\n[2] Mapping SangerModelID -> DepMap ModelID...")
model_df = pd.read_csv(DATA / "Model.csv")
sanger_to_model = dict(zip(model_df["SangerModelID"], model_df["ModelID"]))

stripped_to_model = dict(zip(
    model_df["StrippedCellLineName"].str.upper(),
    model_df["ModelID"]
))

def get_model_id(row):
    mid = sanger_to_model.get(row.get("SANGER_MODEL_ID", ""))
    if mid:
        return mid
    stripped = re.sub(r'[-\s_]', '', str(row.get("CELL_LINE_NAME", ""))).upper()
    return stripped_to_model.get(stripped)

gdsc["ModelID"] = gdsc.apply(get_model_id, axis=1)
matched = gdsc["ModelID"].notna().sum()
print(f"  Matched: {matched:,}/{len(gdsc):,} rows ({100*matched/len(gdsc):.1f}%)")
gdsc = gdsc[gdsc["ModelID"].notna()].copy()

print("\n[3] Loading DepMap expression data...")
expr_raw = pd.read_csv(DATA / "OmicsExpressionTPMLogp1.csv", low_memory=False)

if "IsDefaultEntryForModel" in expr_raw.columns:
    col = expr_raw["IsDefaultEntryForModel"]
    mask = col.astype(str).str.upper().isin({"YES", "TRUE", "1"})
    filtered = expr_raw[mask]

    expr_raw = filtered.copy() if len(filtered) > 0 else expr_raw.copy()
    print(f"  IsDefaultEntryForModel filter: {len(filtered)} rows (fallback={'yes' if len(filtered)==0 else 'no'})")
expr_raw = expr_raw.drop_duplicates("ModelID", keep="first")
expr_raw = expr_raw.set_index("ModelID")

meta_cols = {"Unnamed: 0", "SequencingID", "ModelID", "IsDefaultEntryForModel",
             "ModelConditionID", "IsDefaultEntryForMC"}
gene_cols = [c for c in expr_raw.columns if c not in meta_cols and "(" in c]
expr_raw  = expr_raw[gene_cols].astype(np.float32)

print(f"  Expression: {expr_raw.shape[0]} cells x {len(gene_cols)} genes")

gdsc_model_ids = set(gdsc["ModelID"].unique())
common_cells   = sorted(gdsc_model_ids & set(expr_raw.index))
print(f"  GDSC12 unique ModelIDs: {len(gdsc_model_ids)}")
print(f"  Common with DepMap expr: {len(common_cells)}")

gdsc = gdsc[gdsc["ModelID"].isin(common_cells)].copy()
print(f"  Samples after cell filter: {len(gdsc):,}")

cell_to_cidx = {c: i for i, c in enumerate(common_cells)}
num_cells    = len(common_cells)
num_genes    = len(gene_cols)
gene_to_gidx = {g: i for i, g in enumerate(gene_cols)}

gene_symbol_to_gidx = {}
for i, g in enumerate(gene_cols):
    sym = g.split(" (")[0].upper()
    gene_symbol_to_gidx[sym] = i

print("  Building cell_expr matrix (z-score)...")
cell_expr_df = expr_raw.reindex(common_cells).fillna(0).astype(np.float32)
cell_expr = cell_expr_df.values
gene_std = cell_expr.std(axis=0)
gene_std[gene_std == 0] = 1.0
cell_expr = (cell_expr - cell_expr.mean(axis=0)) / gene_std
print(f"  cell_expr: {cell_expr.shape}")

print("\n[4] Building mutation matrix...")
mut_raw = pd.read_csv(DATA / "OmicsSomaticMutationsMatrixDamaging.csv", low_memory=False)
if "IsDefaultEntryForModel" in mut_raw.columns:
    col = mut_raw["IsDefaultEntryForModel"]
    mask = col.astype(str).str.upper().isin({"YES", "TRUE", "1"})
    filtered = mut_raw[mask]
    mut_raw = filtered.copy() if len(filtered) > 0 else mut_raw.copy()
mut_raw = mut_raw.drop_duplicates("ModelID", keep="first").set_index("ModelID")

mut_meta_cols = {"Unnamed: 0", "SequencingID", "ModelID", "ModelConditionID",
                 "IsDefaultEntryForModel", "IsDefaultEntryForMC"}
mut_gene_cols = [c for c in mut_raw.columns if c not in mut_meta_cols]

cell_mut = np.zeros((num_cells, num_genes), dtype=np.float32)
mut_covered = 0
for cell_id in common_cells:
    if cell_id not in mut_raw.index:
        continue
    cidx = cell_to_cidx[cell_id]
    row  = mut_raw.loc[cell_id]
    for gc in mut_gene_cols:
        val = row.get(gc, 0)
        if pd.notna(val) and val != 0:
            if gc in gene_to_gidx:
                cell_mut[cidx, gene_to_gidx[gc]] = 1.0
    mut_covered += 1
print(f"  cell_mut: {cell_mut.shape}  (covered {mut_covered}/{num_cells} cells)")

print("\n[5] Building drug universe & fingerprints...")

drug_smiles = {}
gdsc2_drug_file = DATA / "pharmacodb" / "raw" / "GDSC_2020_v2-8_2_" / "drug_info.csv"
if gdsc2_drug_file.exists():
    d_info = pd.read_csv(gdsc2_drug_file, encoding="latin-1")
    smiles_col = next((c for c in d_info.columns if "smiles" in c.lower()), None)
    name_col   = next((c for c in d_info.columns if "drug" in c.lower() and "name" in c.lower()), None)
    if smiles_col and name_col:
        for _, row in d_info.iterrows():
            drug_smiles[normalize_drug_name(row[name_col])] = str(row[smiles_col])
    print(f"  SMILES loaded from GDSC2 pharmacodb: {len(drug_smiles)} drugs")

drug_names = sorted(gdsc["DRUG_NAME"].str.strip().unique())
drug_to_didx  = {d: i for i, d in enumerate(drug_names)}
drug_norm_map = {normalize_drug_name(d): i for d, i in drug_to_didx.items()}
num_drugs = len(drug_names)
print(f"  Drug universe: {num_drugs} drugs")

drug_fp = np.zeros((num_drugs, 2048), dtype=np.float32)
fp_found = 0
for drug_name, d_idx in drug_to_didx.items():
    smiles = drug_smiles.get(normalize_drug_name(drug_name))
    fp = smiles_to_fp(smiles)
    if fp is not None:
        drug_fp[d_idx] = fp
        fp_found += 1
print(f"  FP computed: {fp_found}/{num_drugs} ({100*fp_found/num_drugs:.1f}%)")

print("\n[6] Building sample_table...")
gdsc["drug_idx"] = gdsc["DRUG_NAME"].str.strip().map(drug_to_didx)
gdsc["cell_idx"] = gdsc["ModelID"].map(cell_to_cidx)
gdsc = gdsc[gdsc["drug_idx"].notna() & gdsc["cell_idx"].notna()].copy()
gdsc["drug_idx"] = gdsc["drug_idx"].astype(int)
gdsc["cell_idx"] = gdsc["cell_idx"].astype(int)

sample_table = gdsc[[
    "drug_idx", "cell_idx", "DRUG_NAME", "CELL_LINE_NAME", "ModelID",
    "LN_IC50", "AUC", "DATASET", "CANCER_TYPE"
]].copy()
sample_table.columns = [
    "drug_idx", "cell_idx", "drug_name", "cell_name", "model_id",
    "ln_ic50", "auc", "dataset", "cancer_type"
]
sample_table["drug_name_lower"] = sample_table["drug_name"].str.lower().str.strip()
print(f"  sample_table: {len(sample_table):,} samples  "
      f"({sample_table['drug_idx'].nunique()} drugs, {sample_table['cell_idx'].nunique()} cells)")
print(f"  By dataset: {sample_table['dataset'].value_counts().to_dict()}")

print("\n[7] Building HCDT matrices...")

def hcdt_didx(hcdt_name):
    norm = normalize_drug_name(hcdt_name)
    return drug_norm_map.get(norm)

print("  gene_pathway...")
pg = pd.read_excel(HCDT / "Pathway_Gene.xlsx")
pg["path_clean"] = pg["PATH_NAME"].str.strip()
pathway_list    = sorted(pg["path_clean"].dropna().unique())
pathway_to_pidx = {p: i for i, p in enumerate(pathway_list)}
num_pathways    = len(pathway_list)

gene_pathway = np.zeros((num_genes, num_pathways), dtype=np.float32)
for _, row in pg.iterrows():
    p_idx = pathway_to_pidx.get(row["path_clean"])
    if p_idx is None or pd.isna(row.get("GENE_SYMBOL")):
        continue
    for sym in str(row["GENE_SYMBOL"]).split(","):
        g_idx = gene_symbol_to_gidx.get(sym.strip().upper())
        if g_idx is not None:
            gene_pathway[g_idx, p_idx] = 1.0
print(f"    gene_pathway: {gene_pathway.shape}")

print("  hcdt_drug_gene...")
dg = pd.read_csv(HCDT / "DRUG_GENE" / "DRUG-GENE.tsv", sep="\t")
hcdt_drug_gene = np.zeros((num_drugs, num_genes), dtype=np.float32)
for _, row in dg.iterrows():
    d_idx = hcdt_didx(row["DRUG_NAME"])
    g_idx = gene_symbol_to_gidx.get(str(row["GENE_SYMBOL"]).upper())
    if d_idx is not None and g_idx is not None:
        hcdt_drug_gene[d_idx, g_idx] = 1.0
cov = (hcdt_drug_gene.sum(axis=1) > 0).sum()
print(f"    HCDT name-match coverage: {cov}/{num_drugs} drugs ({100*cov/num_drugs:.1f}%)")

print("  hcdt_drug_gene augmentation (GDSC PUTATIVE_TARGET)...")

GENE_ALIASES = {
    "ABL": "ABL1", "AKT": "AKT1", "FGFR": "FGFR1",
    "ERK1": "MAPK3", "ERK2": "MAPK1", "ERK5": "MAPK7",
    "DNAPK": "PRKDC", "FAK1": "PTK2", "CRAF": "RAF1",
    "ALK5": "TGFBR1", "BMP": "BMP2", "CAMK2": "CAMK2A",
}

def parse_putative_targets(target_str):
    if pd.isna(target_str) or not isinstance(target_str, str):
        return []
    parts = re.split(r"[,/\s]+", target_str.strip())
    return [p.strip() for p in parts if re.match(r"^[A-Z][A-Z0-9]{1,9}$", p.strip())]

gdsc_drug_target = {}
for df_src in [g1, g2]:
    for _, row in df_src[["DRUG_NAME", "PUTATIVE_TARGET"]].drop_duplicates("DRUG_NAME").iterrows():
        norm = normalize_drug_name(row["DRUG_NAME"])
        gdsc_drug_target[norm] = parse_putative_targets(row.get("PUTATIVE_TARGET", ""))

aug_drugs, aug_entries = 0, 0
for norm, genes in gdsc_drug_target.items():
    d_idx = drug_norm_map.get(norm)
    if d_idx is None or hcdt_drug_gene[d_idx].sum() > 0:
        continue
    matched = []
    for sym in genes:
        canonical = GENE_ALIASES.get(sym, sym)
        g_idx = gene_symbol_to_gidx.get(canonical)
        if g_idx is not None:
            hcdt_drug_gene[d_idx, g_idx] = 1.0
            matched.append(canonical)
    if matched:
        aug_drugs += 1
        aug_entries += len(matched)

cov = (hcdt_drug_gene.sum(axis=1) > 0).sum()
print(f"    augmented {aug_drugs} drugs (+{aug_entries} drug-gene entries)")
print(f"    final coverage: {cov}/{num_drugs} drugs ({100*cov/num_drugs:.1f}%)")

CID_CACHE = DATA / "pubchem_cid_cache.json"
if CID_CACHE.exists():
    print("  hcdt_drug_gene PubChem CID augmentation...")
    with open(CID_CACHE) as f:
        cid_cache = json.load(f)
    dg_cid = dg.dropna(subset=["PUBCHEM_CID"]).copy()
    dg_cid["cid_int"] = dg_cid["PUBCHEM_CID"].astype(float).astype(int)
    cid_to_genes: dict = {}
    for _, row in dg_cid.iterrows():
        cid_to_genes.setdefault(int(row["cid_int"]), []).append(
            str(row["GENE_SYMBOL"]).upper()
        )
    cid_added = 0
    for drug_lower, cid in cid_cache.items():
        if cid is None:
            continue
        d_idx = drug_norm_map.get(drug_lower)
        if d_idx is None or hcdt_drug_gene[d_idx].sum() > 0:
            continue
        for sym in cid_to_genes.get(int(cid), []):
            canonical = GENE_ALIASES.get(sym, sym)
            g_idx = gene_symbol_to_gidx.get(canonical)
            if g_idx is not None:
                hcdt_drug_gene[d_idx, g_idx] = 1.0
                cid_added += 1
    cov = (hcdt_drug_gene.sum(axis=1) > 0).sum()
    print(f"    CID augmentation: +{cid_added} entries -> {cov}/{num_drugs} ({100*cov/num_drugs:.1f}%)")
else:
    print("  [skip] PubChem CID cache not found. Run: python src/data/fetch_pubchem_cids.py")

print("  lincs_drug_gene (placeholder)...")
lincs_path = OUT / "lincs_drug_gene.npy"
if lincs_path.exists():
    lincs_drug_gene = np.load(lincs_path)
    cov = (lincs_drug_gene.sum(axis=1) > 0).sum()
    print(f"    loaded existing lincs_drug_gene: {lincs_drug_gene.shape}, coverage {cov}/{num_drugs}")
else:
    lincs_drug_gene = np.zeros((num_drugs, num_genes), dtype=np.float32)
    print(f"    placeholder zeros {lincs_drug_gene.shape} -- run build_lincs_matrices.py for real data")

print("  hcdt_drug_path_direct...")
dp = pd.read_excel(HCDT / "DRUG_PATHWAY" / "Drug-Pathway.xlsx")
hcdt_drug_path_direct = np.zeros((num_drugs, num_pathways), dtype=np.float32)
for _, row in dp.iterrows():
    d_idx = hcdt_didx(row["drug_name"])
    p_idx = pathway_to_pidx.get(str(row["path_name"]).strip())
    if d_idx is not None and p_idx is not None:
        hcdt_drug_path_direct[d_idx, p_idx] = 1.0
cov = (hcdt_drug_path_direct.sum(axis=1) > 0).sum()
print(f"    Pathway coverage: {cov}/{num_drugs} drugs ({100*cov/num_drugs:.1f}%)")

print("  hcdt_neg_drug_gene...")
neg = pd.read_excel(HCDT / "negative DTIs.xlsx")
hcdt_neg_drug_gene = np.zeros((num_drugs, num_genes), dtype=np.float32)
for _, row in neg.iterrows():
    d_idx = hcdt_didx(row["DRUG_NAME"])
    g_idx = gene_symbol_to_gidx.get(str(row["GENE_SYMBOL"]).upper())
    if d_idx is not None and g_idx is not None:
        hcdt_neg_drug_gene[d_idx, g_idx] = 1.0
cov = (hcdt_neg_drug_gene.sum(axis=1) > 0).sum()
print(f"    Neg DTI coverage: {cov}/{num_drugs} drugs ({100*cov/num_drugs:.1f}%)")

print("\n[8] Saving to", OUT)
np.save(OUT / "drug_fp.npy",               drug_fp)
np.save(OUT / "cell_expr.npy",             cell_expr)
np.save(OUT / "cell_mut.npy",              cell_mut)
np.save(OUT / "hcdt_drug_gene.npy",        hcdt_drug_gene)
np.save(OUT / "lincs_drug_gene.npy",       lincs_drug_gene)
np.save(OUT / "hcdt_drug_path_direct.npy", hcdt_drug_path_direct)
np.save(OUT / "hcdt_neg_drug_gene.npy",    hcdt_neg_drug_gene)
np.save(OUT / "gene_pathway.npy",          gene_pathway)
sample_table.to_csv(OUT / "sample_table.csv", index=False)

id_maps = {
    "drug_name_to_idx":   {k: int(v) for k, v in drug_to_didx.items()},
    "drug_norm_to_idx":   {k: int(v) for k, v in drug_norm_map.items()},
    "cell_to_idx":        {k: int(v) for k, v in cell_to_cidx.items()},
    "gene_to_idx":        {k: int(v) for k, v in gene_to_gidx.items()},
    "gene_symbol_to_idx": {k: int(v) for k, v in gene_symbol_to_gidx.items()},
    "pathway_to_idx":     {k: int(v) for k, v in pathway_to_pidx.items()},
    "num_drugs":    num_drugs,
    "num_cells":    num_cells,
    "num_genes":    num_genes,
    "num_pathways": num_pathways,
    "gene_list":    gene_cols,
    "pathway_list": pathway_list,
    "cell_list":    common_cells,
    "drug_list":    drug_names,
}
with open(OUT / "id_maps.json", "w") as f:
    json.dump(id_maps, f, indent=2)

with open(OUT / "id_maps_v2.json", "w") as f:
    json.dump(id_maps, f, indent=2)

print("\n=== Done ===")
print(f"  drug_fp:               {drug_fp.shape}")
print(f"  cell_expr:             {cell_expr.shape}")
print(f"  cell_mut:              {cell_mut.shape}")
print(f"  hcdt_drug_gene:        {hcdt_drug_gene.shape}")
print(f"  lincs_drug_gene:       {lincs_drug_gene.shape}")
print(f"  hcdt_drug_path_direct: {hcdt_drug_path_direct.shape}")
print(f"  hcdt_neg_drug_gene:    {hcdt_neg_drug_gene.shape}")
print(f"  gene_pathway:          {gene_pathway.shape}")
print(f"  sample_table:          {sample_table.shape}")
print(f"\nOutput: {OUT}")
print("Next: set mat_dir: data/matrices_gdsc12 in configs/mp_hcpnet.yaml")
