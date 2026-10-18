"""Combine five normalized evidence channels into descriptive domain scores.

Uses direct reconstructed-feature correlations for feature channels and ranks.
The channel weights are fixed; the integrated score is not a statistical test.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

DOMAINS = [
    "cognition",
    "sleep/circadian",
    "lifestyle/environment",
    "physiology",
    "early-life/development",
    "mental/affective",
]
WEIGHTS = np.array([0.25, 0.25, 0.15, 0.15, 0.2])
CHANNELS = ["pc_max", "raw_max", "raw_mean", "top20_count", "consensus"]


def calculate_matrix(pc, raw, consensus):
    threshold = float(raw.abs_incremental.nlargest(20).min())
    rows = []
    for domain in DOMAINS:
        p = pc.loc[pc.domain.eq(domain), "pc_abs_incremental"].astype(float)
        r = raw.loc[raw.domain.eq(domain), "abs_incremental"].astype(float)
        c = consensus.loc[consensus.domain.eq(domain), "consensus_score"].astype(float)
        if not len(p) or not len(r) or (not len(c)):
            raise ValueError("Missing evidence for domain: " + domain)
        rows.append(
            {
                "domain": domain,
                "pc_max": p.max(),
                "raw_max": r.max(),
                "raw_mean": r.mean(),
                "top20_count": int(r.ge(threshold).sum()),
                "consensus": c.max(),
            }
        )
    result = pd.DataFrame(rows)
    for column in CHANNELS:
        values = result[column].astype(float)
        maximum = values.max(skipna=True)
        result["norm_" + column] = (
            values / maximum if np.isfinite(maximum) and maximum > 0 else 0.0
        )
    normalized = result[["norm_" + name for name in CHANNELS]].to_numpy(float)
    result["integrated"] = [
        float(
            np.sum(row[np.isfinite(row)] * WEIGHTS[np.isfinite(row)])
            / WEIGHTS[np.isfinite(row)].sum()
        )
        for row in normalized
    ]
    return result


def calculate(raw, direct, pc, consensus):
    keys = ["block", "raw_col"]
    if len(raw) != 112 or len(direct) != 112 or len(pc) != 12:
        raise ValueError("Expected 112 reconstructed features and 12 PCs")
    if raw.duplicated(["block", "raw_feature"]).any() or direct.duplicated(keys).any():
        raise ValueError("Duplicate reconstruction keys")
    merged = raw.merge(
        direct[keys + ["recomputed_increment_r"]],
        left_on=["block", "raw_feature"],
        right_on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (
        not merged._merge.eq("both").all()
        or not np.isfinite(merged.recomputed_increment_r).all()
    ):
        raise ValueError("Missing or nonfinite direct feature correlations")
    corrected = raw.copy()
    corrected["incremental_corr"] = merged.recomputed_increment_r.to_numpy(float)
    corrected["abs_incremental"] = corrected.incremental_corr.abs()
    matrix = calculate_matrix(pc, corrected, consensus)
    ranking = merged[
        ["domain", "block", "raw_feature", "incremental_corr", "recomputed_increment_r"]
    ].rename(
        columns={
            "incremental_corr": "old_weighted_score",
            "recomputed_increment_r": "corrected_direct_r",
        }
    )
    ranking["abs_direct_r_for_rank"] = ranking.corrected_direct_r.abs().round(12)
    ranking["abs_old_score_for_tie"] = ranking.old_weighted_score.abs()
    ranked = []
    for domain in DOMAINS:
        subset = (
            ranking[ranking.domain.eq(domain)]
            .sort_values(
                ["abs_direct_r_for_rank", "abs_old_score_for_tie", "raw_feature"],
                ascending=[False, False, True],
                kind="stable",
            )
            .copy()
        )
        subset["direct_rank_within_domain"] = np.arange(1, len(subset) + 1)
        ranked.append(subset)
    ranking = pd.concat(ranked, ignore_index=True)
    bars = ranking[ranking.direct_rank_within_domain.le(8)].copy()
    if len(ranking) != 112 or len(bars) != 43:
        raise ValueError("Feature counts differ from the configured analysis")
    return (matrix, bars, ranking)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-evidence-csv", required=True)
    parser.add_argument("--direct-correlations-tsv", required=True)
    parser.add_argument("--pc-evidence-csv", required=True)
    parser.add_argument("--consensus-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    matrix, bars, ranking = calculate(
        pd.read_csv(args.raw_evidence_csv),
        pd.read_csv(args.direct_correlations_tsv, sep="\t"),
        pd.read_csv(args.pc_evidence_csv),
        pd.read_csv(args.consensus_csv),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out / "domain_evidence.csv", index=False)
    bars.to_csv(out / "displayed_features.csv", index=False)
    ranking.to_csv(out / "all_feature_ranks.csv", index=False)
    (out / "calculation_contract.json").write_text(
        json.dumps(
            {
                "weights": dict(zip(CHANNELS, WEIGHTS.tolist())),
                "pc_and_consensus_channels": "PC maximum association and cross-model consensus",
                "feature_channels": "Participant-level direct Pearson correlations",
                "top_feature_rule": "At or above the global twentieth-largest absolute correlation, including ties",
                "display_tie_rule": "Absolute direct r rounded to 12 decimals; old loading-weighted magnitude; feature identifier",
                "scope": "Descriptive integrated evidence; not a statistical test or causal importance",
            },
            indent=2,
        )
        + "\n"
    )
    print(matrix[["domain", "integrated"]].to_string(index=False))
    print("PASS: six domain scores and 43 displayed features calculated")


if __name__ == "__main__":
    main()
