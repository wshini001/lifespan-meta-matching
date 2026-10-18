"""Test soft Yeo-7 network-pair enrichment using coverage-stratified permutations.

Permutes complete node membership profiles within cortical-coverage quartiles
and applies Benjamini-Hochberg correction across 28 network pairs.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

YEO = {
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal Attention",
    4: "Ventral Attention",
    5: "Limbic",
    6: "Frontoparietal Control",
    7: "Default Mode",
}


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def enrich(overlap, edges, pnf_column, n_permutations, seed, support_column=None):
    ov = overlap.sort_values("node").reset_index(drop=True)
    if len(ov) != 55 or ov.node.tolist() != list(range(1, 56)) or len(edges) != 1485:
        raise ValueError("Expected 55 ordered nodes and 1485 edges")
    W = ov[[f"abs_total_Yeo{k}" for k in range(1, 8)]].to_numpy(float)
    coverage = ov.cortical_abs_mass_fraction.to_numpy(float)
    i = edges.Node1.to_numpy(int) - 1
    j = edges.Node2.to_numpy(int) - 1
    abs_pnf = np.abs(edges[pnf_column].to_numpy(float))
    if (
        not np.isfinite(W).all()
        or not np.isfinite(coverage).all()
        or (not np.isfinite(abs_pnf).all())
    ):
        raise ValueError("Non-finite memberships, coverage or covariance")
    if min(i.min(), j.min()) < 0 or max(i.max(), j.max()) >= 55:
        raise ValueError("Edge nodes outside 1..55")
    observed_pairs = set(zip(np.minimum(i, j), np.maximum(i, j)))
    if observed_pairs != {(a, b) for a in range(55) for b in range(a + 1, 55)}:
        raise ValueError("Edges do not cover all unique node pairs exactly")
    boot = np.zeros(len(edges), bool)
    if support_column:
        series = edges[support_column]
        boot = (
            series.to_numpy(bool)
            if series.dtype == bool
            else series.astype(str)
            .str.lower()
            .isin(["true", "1", "t", "yes"])
            .to_numpy(bool)
        )
    B = int(n_permutations)
    if B < 2:
        raise ValueError("At least two permutations required")
    pairs = [(a, b) for a in range(7) for b in range(a, 7)]

    def calculate(X):
        A = X[i]
        C = X[j]
        E = A.T @ C
        M = A.T @ (C * abs_pnf[:, None])
        BS = A.T @ (C * boot.astype(float)[:, None])
        mean_abs = []
        support = []
        exposure = []
        for a, b in pairs:
            if a == b:
                e = E[a, a]
                m = M[a, a]
                q = BS[a, a]
            else:
                e = E[a, b] + E[b, a]
                m = M[a, b] + M[b, a]
                q = BS[a, b] + BS[b, a]
            exposure.append(e)
            mean_abs.append(m / e if e > 0 else np.nan)
            support.append(q / e if e > 0 else np.nan)
        return (
            np.asarray(mean_abs, dtype=float),
            np.asarray(support, dtype=float),
            np.asarray(exposure, dtype=float),
        )

    obs_mean, obs_support, obs_exposure = calculate(W)
    rank = pd.Series(coverage).rank(method="first")
    strata = pd.qcut(rank, q=4, labels=False).to_numpy(int)
    rng = np.random.default_rng(seed)
    null_mean = np.empty((B, len(pairs)), dtype=np.float32)
    null_support = np.empty((B, len(pairs)), dtype=np.float32)
    for repeat in range(B):
        permutation = np.arange(55)
        for stratum in range(4):
            indices = np.where(strata == stratum)[0]
            permutation[indices] = rng.permutation(indices)
        mean, support, _ = calculate(W[permutation])
        null_mean[repeat] = mean
        null_support[repeat] = support
    rows = []
    for x, (a, b) in enumerate(pairs):
        nm = null_mean[:, x].astype(float)
        ns = null_support[:, x].astype(float)
        p_mean = (1 + np.sum(nm >= obs_mean[x])) / (B + 1)
        p_support = (1 + np.sum(ns >= obs_support[x])) / (B + 1)
        sd_m = np.std(nm, ddof=1)
        sd_s = np.std(ns, ddof=1)
        rows.append(
            {
                "Yeo1": a + 1,
                "Yeo1_name": YEO[a + 1],
                "Yeo2": b + 1,
                "Yeo2_name": YEO[b + 1],
                "observed_mean_abs_pnf": obs_mean[x],
                "null_mean_abs_pnf": np.mean(nm),
                "null_sd_abs_pnf": sd_m,
                "enrichment_ratio_abs_pnf": obs_mean[x] / np.mean(nm),
                "z_abs_pnf": (obs_mean[x] - np.mean(nm)) / sd_m if sd_m > 0 else np.nan,
                "p_enrichment_abs_pnf": p_mean,
                "observed_bootstrap_support": obs_support[x],
                "null_bootstrap_support": np.mean(ns),
                "z_bootstrap_support": (
                    (obs_support[x] - np.mean(ns)) / sd_s if sd_s > 0 else np.nan
                ),
                "p_enrichment_bootstrap_support": p_support,
                "observed_soft_exposure": obs_exposure[x],
            }
        )
    res = pd.DataFrame(rows)
    res["q_enrichment_abs_pnf"] = bh_fdr(res["p_enrichment_abs_pnf"])
    res["q_enrichment_bootstrap_support"] = bh_fdr(
        res["p_enrichment_bootstrap_support"]
    )
    if support_column is None:
        res = res.drop(
            columns=[
                "observed_bootstrap_support",
                "null_bootstrap_support",
                "z_bootstrap_support",
                "p_enrichment_bootstrap_support",
                "q_enrichment_bootstrap_support",
            ]
        )
    return (res, null_mean, null_support, strata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-membership-csv", required=True)
    parser.add_argument("--edge-table-csv", required=True)
    parser.add_argument("--pnf-column", required=True)
    parser.add_argument("--support-column")
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result, null, support, strata = enrich(
        pd.read_csv(args.soft_membership_csv),
        pd.read_csv(args.edge_table_csv),
        args.pnf_column,
        args.n_permutations,
        args.seed,
        args.support_column,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "network_pair_enrichment.csv", index=False)
    np.savez_compressed(
        out / "permutation_null.npz",
        null_mean_abs_pnf=null,
        seed=args.seed,
        B=args.n_permutations,
    )
    if args.support_column:
        np.savez_compressed(
            out / "bootstrap_support_null.npz", null_bootstrap_support=support
        )
    (out / "settings.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "B": args.n_permutations,
                "covariance_column": args.pnf_column,
                "stratum_sizes": np.bincount(strata).tolist(),
                "null_dtype": "float32",
                "FDR_scope": "28 unique Yeo7 network pairs in this branch",
            },
            indent=2,
        )
        + "\n"
    )
    print("PASS: coverage-stratified network enrichment calculated")


if __name__ == "__main__":
    main()
