"""Assert that the local inputs still reproduce the manuscript's published values.

A stale local copy of the broad HCDT mask once produced a whole set of wrong Methods
numbers (240 annotated compounds instead of 435) and nothing caught it, because no run
recorded which mask it consumed. This script is the standing check: each entry
recomputes a published quantity from local files, so a silent data regression fails here
instead of in the manuscript.

Usage: python scripts/provenance_check.py
Exit status is non-zero if any check fails.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

FAIL = []


def check(name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:52s} got {got}   expected {want}")
    if not ok:
        FAIL.append(name)


def main():
    md = "data/matrices_gdsc12"

    print("panel and split (Methods, sec:train):")
    st = pd.read_csv(os.path.join(md, "sample_table.csv"))
    check("GDSC pairs", len(st), 342114)
    check("drugs", st["drug_name_lower"].nunique(), 542)
    check("cell lines", st["model_id"].nunique(), 718)
    check("duplicate (drug, cell) rows", int(st.duplicated(["drug_name_lower", "model_id"]).sum()), 0)
    pl = json.load(open(os.path.join(md, "id_maps.json")))["pathway_list"]
    check("pathways P", len(pl), 3143)

    print("\nmask generation, inferred from the runs (sec:pruned):")
    for tag, want in (("brd", 435), ("base", 435), ("prn", 421), ("dns", 542)):
        fs = sorted(glob.glob(f"results/interpret_hdca/{tag}_s*/p_gene_align.npy"))
        if not fs:
            print(f"  [skip] {tag}_s*: no local runs")
            continue
        act = [542 - int((np.abs(np.load(f)).max(1) == 0).sum()) for f in fs]
        check(f"{tag}_s* annotated compounds (all seeds equal)",
              act[0] if len(set(act)) == 1 else f"inconsistent {act}", want)

    print("\nmask FILES on disk (the historical failure point):")
    for f, want in (("hcdt_drug_gene_pruned.npy", 421), ("hcdt_drug_gene.npy", 435)):
        p = os.path.join(md, f)
        if not os.path.exists(p):
            print(f"  [skip] {f} absent")
            continue
        m = np.load(p)
        n = int((m.sum(1) > 0).sum())
        med = float(np.median(m.sum(1)[m.sum(1) > 0]))
        status = "ok " if n == want else "STALE"
        if n != want:
            FAIL.append(f"{f} is stale")
        print(f"  [{status}] {f:34s} {n}/542 annotated, median {med:.0f} genes   expected {want}")

    print("\npublished results reproduced from local files:")
    hits = pd.read_csv("moa_summary_all.csv")
    for tag, want in (("prn", 49.0), ("brd", 15.6), ("dns", 0.0), ("base", 17.7)):
        g = hits[hits.tag == tag]["hit_rate"]
        check(f"16-drug MoA, {tag}", round(float(g.mean()), 1), want, tol=0.05)

    sub = pd.read_csv("results/cross_winsorized_table_6seed_subset.csv")
    for model, cset, want in (("HDCA", "CCLE", 0.773), ("HDCA", "gCSI", 0.761),
                              ("DRPreter", "CCLE", 0.766), ("DRPreter", "gCSI", 0.758)):
        v = sub[(sub.model == model) & (sub.cset == cset)]["PCC_win"].iloc[0]
        check(f"cross-study PCC, {model} {cset}", round(float(v), 3), want, tol=0.0005)

    va = pd.read_csv("results/variants_aligned_summary.csv")
    for variant, cset, want in (("sham", "CCLE", 0.773), ("sham", "gCSI", 0.754)):
        v = va[(va.variant == variant) & (va.cset == cset)]["PCC"].iloc[0]
        check(f"aligned PCC, {variant} {cset}", round(float(v), 3), want, tol=0.0005)

    ng = json.load(open("results/negctrl_moa_dgidb.json"))
    check("DGIdb panel, evaluable compounds", ng["n_eval"], 144)
    check("DGIdb panel, DB-matched", ng["n_matched"], 208)

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) FAILED: {FAIL}")
        sys.exit(1)
    print("all checks passed -- local inputs reproduce the manuscript")


if __name__ == "__main__":
    main()
