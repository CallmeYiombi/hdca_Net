"""Reproduce the read-out degeneracy statistics quoted in Results (sec:cases, sec:sweep).

Definition recovered by audit (it reproduces 5% / 9% / 30% exactly):
  * average p_gene_align over the six seeds of a tag, then
  * take each compound's top-k pathways as an ORDERED tuple, then
  * among compounds whose gene branch is active (mask non-empty),
    report the size of the modal group -- compounds handed one and the same read-out.

The ordering matters: comparing top-k as unordered sets inflates every group,
and including the empty-mask rows adds a block of identical all-zero rows that is
an artefact of coverage, not of degeneracy.

Usage: python scripts/degeneracy_audit.py
"""
import collections
import glob
import os

import numpy as np

ROOT = "results/interpret_hdca"
TAGS = [("reference (direct-target)", "prn"), ("HDCA-Net-broad", "brd"), ("HDCA-Net-dense", "dns")]


def seed_mean(tag):
    paths = sorted(glob.glob(os.path.join(ROOT, f"{tag}_s*", "p_gene_align.npy")))
    if not paths:
        raise SystemExit(f"no p_gene_align.npy for tag {tag} under {ROOT}")
    return np.mean([np.load(p) for p in paths], axis=0), len(paths)


def report(matrix, k):
    active = np.abs(matrix).max(axis=1) > 0
    order = np.argsort(-matrix, axis=1, kind="stable")[:, :k]
    ordered = [tuple(row) for row in order]
    among_active = [t for t, a in zip(ordered, active) if a]
    counts = collections.Counter(among_active)
    modal = counts.most_common(1)[0][1]
    unique = sum(1 for t in ordered if counts[t] == 1 and active[ordered.index(t)])
    unique_all = sum(1 for t in among_active if counts[t] == 1)
    return {
        "n_all": matrix.shape[0],
        "n_active": int(active.sum()),
        "modal": modal,
        "modal_pct_active": 100.0 * modal / len(among_active),
        "modal_pct_all": 100.0 * modal / matrix.shape[0],
        "distinct": len(counts),
        "unique": unique_all,
    }


for label, tag in TAGS:
    matrix, n_seeds = seed_mean(tag)
    print(f"\n=== {label}  ({n_seeds} seeds, mean matrix) ===")
    for k in (5, 10):
        r = report(matrix, k)
        print(
            f"  top-{k:<2d} active {r['n_active']}/{r['n_all']}  "
            f"modal group {r['modal']} "
            f"({r['modal_pct_active']:.0f}% of active, {r['modal_pct_all']:.0f}% of all)  "
            f"distinct read-outs {r['distinct']}  "
            f"compounds with a read-out of their own {r['unique']}"
        )
