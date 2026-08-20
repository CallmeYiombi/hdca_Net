"""MoA agreement of HDCA-Net against DRPreter, with significance tests.

Both models are scored on the same 16 drugs with the same word-boundary matcher, each through
its own native pathway prior. Compares two channels (gene-to-pathway projection and pathway
attention) with independent two-sample tests over six seeds per model.

Usage: python src/analysis/moa_compare.py
"""
import os, re, json, glob
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTERP = os.path.join(ROOT, "results", "interpret_hdca")
DRP_JSON = os.path.join(ROOT, "results", "drpreter_moa_6seed.json")
OUT = os.path.join(ROOT, "results", "moa_compare_table.csv")

AXIS_FILE = {"gene": "moa_p_gene_align.csv",
             "path": "moa_path_attn.csv",
             "fused": "moa_p_align.csv"}

def wb(kw, text):
    return re.search(rf"\b{re.escape(kw)}\d*\b", text, re.IGNORECASE) is not None

def hdca_seed_rate(csv):
    """Re-score the recorded pathway hits with the word-boundary matcher; returns hit rate (%)."""
    d = pd.read_csv(csv)
    hits = 0
    for _, r in d.iterrows():
        kws = [k.strip() for k in str(r["expected_kws"]).split(",")]
        mh = str(r["moa_hits"])
        if mh in ("--", "nan", ""):
            continue
        names = [re.sub(r"\s*\(rank \d+\)", "", x).strip() for x in mh.split(";")]
        if any(wb(k, nm) for nm in names for k in kws):
            hits += 1
    return 100.0 * hits / len(d)

def hdca_rates(cond, axis):
    rates = []
    for s in range(1, 7):
        f = os.path.join(INTERP, f"{cond}_s{s}", AXIS_FILE[axis])
        if os.path.exists(f):
            rates.append(hdca_seed_rate(f))
    return np.array(rates)

def ms(a):
    a = np.asarray(a, float)
    return f"{a.mean():.1f} +/- {a.std(ddof=1):.1f}"

def test(a, b):
    """Independent two-sample tests, a = HDCA-Net, b = DRPreter."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u, p_mw = mannwhitneyu(a, b, alternative="greater")
    t, p_t = ttest_ind(a, b, equal_var=False)
    return p_mw, p_t

def main():
    with open(DRP_JSON) as f:
        drp = json.load(f)
    drp_gene = np.array(drp["genepath_seed_rates"])
    drp_path = np.array(drp["path_seed_rates"])
    drp_raw  = np.array(drp["gene_raw_seed_rates"])

    hdca = {axis: hdca_rates("prn", axis) for axis in AXIS_FILE}
    base = {axis: hdca_rates("base", axis) for axis in AXIS_FILE}

    rows = []
    print("=" * 74)
    print(" HDCA(PRUNED lambda0.3) vs DRPreter -- MoA hit (6-seed, word-boundary, 16 drugs)")
    print("=" * 74)

    for axis, drp_arr, label in [("gene", drp_gene, "gene->pathway"),
                                 ("path", drp_path, "pathway attention")]:
        h = hdca[axis]
        p_mw, p_t = test(h, drp_arr)
        print(f"\n[{label}]")
        print(f"  HDCA     : {ms(h)} %   seeds={[round(float(x),1) for x in h]}")
        print(f"  DRPreter : {ms(drp_arr)} %   seeds={[round(float(x),1) for x in drp_arr]}")
        print(f"  delta = +{h.mean()-drp_arr.mean():.1f}pp   Mann-Whitney p={p_mw:.4f}  |  Welch t p={p_t:.4f}")
        rows.append(dict(axis=label, HDCA_mean=h.mean(), HDCA_std=h.std(ddof=1),
                         DRPreter_mean=drp_arr.mean(), DRPreter_std=drp_arr.std(ddof=1),
                         delta_pp=h.mean()-drp_arr.mean(), p_mannwhitney=p_mw, p_welch=p_t))

    print("\n[reference] HDCA-Net fused   :", ms(hdca["fused"]), "%")
    print("[reference] DRPreter raw gene:", ms(drp_raw), "%")
    print("[reference] HDCA-Net broad-mask gene:", ms(base["gene"]), "%  -> +{:.1f}pp with the direct-target mask".format(
        hdca["gene"].mean() - base["gene"].mean()))
    rows.append(dict(axis="fused (HDCA only)", HDCA_mean=hdca["fused"].mean(), HDCA_std=hdca["fused"].std(ddof=1),
                     DRPreter_mean=np.nan, DRPreter_std=np.nan, delta_pp=np.nan, p_mannwhitney=np.nan, p_welch=np.nan))
    rows.append(dict(axis="raw gene symbol", HDCA_mean=np.nan, HDCA_std=np.nan,
                     DRPreter_mean=drp_raw.mean(), DRPreter_std=drp_raw.std(ddof=1),
                     delta_pp=np.nan, p_mannwhitney=np.nan, p_welch=np.nan))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nsaved: {OUT}")
    print("\nNote: the two models are read out through different pathway priors "
          "(HCDT vs 34 KEGG pathways) under the same matcher and the same 16 drugs.")

if __name__ == "__main__":
    main()
