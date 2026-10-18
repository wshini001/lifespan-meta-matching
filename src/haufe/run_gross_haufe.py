"""Calculate gross-motor total-increment Haufe patterns with equal child weights.

Each scan is weighted by the inverse number of scans from its child.
Node summaries use covariance divided by increment variance.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from haufe_engine import node_importance
from weighted_haufe_engine import child_weights, wcov, wcorr


def calculate(predictions, cohort, mapping):
    keys = ["scan_id", "imaging_numeric_id"]
    columns = [f"e{i}" for i in range(1, 1486)]
    for frame in [predictions, cohort]:
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate scan/child identities")
    data = predictions.merge(
        cohort[keys + columns], on=keys, how="left", validate="one_to_one"
    )
    if len(data) != 321 or data.imaging_numeric_id.nunique() != 166:
        raise ValueError("Expected 321 scans / 166 children")
    if not data.groupby("imaging_numeric_id").fold.nunique().eq(1).all():
        raise ValueError("A child occurs in multiple folds")
    x = data[columns].to_numpy(float)
    y = (data.y_pred_krr_plus_dnn_multibranch - data.y_pred_baseline).to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite aligned connectivity or increment")
    if len(mapping) != 1485 or mapping.edge_col.tolist() != columns:
        raise ValueError("Edge mapping order differs from connectivity columns")
    pairs = mapping[["Node1", "Node2"]].to_numpy(int)
    expected = {(i, j) for i in range(1, 56) for j in range(i + 1, 56)}
    if set(map(tuple, pairs)) != expected:
        raise ValueError("Invalid 55-node edge pairs")
    w = child_weights(data.imaging_numeric_id)
    cov = np.array([wcov(x[:, j], y, w) for j in range(1485)])
    cor = np.array([wcorr(x[:, j], y, w) for j in range(1485)])
    variance = wcov(y, y, w)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Non-positive increment variance")
    out = mapping[["Node1", "Node2", "ICA_Node1", "ICA_Node2"]].copy()
    out = out.rename(
        columns={
            "ICA_Node1": "FMRIB_ICA_component1",
            "ICA_Node2": "FMRIB_ICA_component2",
        }
    )
    out.insert(0, "edge", columns)
    out["pnf_cov_subject_total"] = cov
    out["haufe_weight"] = cov / variance
    out["haufe_r"] = cor
    nodes = node_importance(out, "haufe_weight")
    return (out, nodes, variance)


def main():
    parser = argparse.ArgumentParser()
    for argument in ["predictions-csv", "cohort-csv", "edge-map-csv", "out-dir"]:
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    edges, nodes, variance = calculate(
        pd.read_csv(args.predictions_csv),
        pd.read_csv(args.cohort_csv),
        pd.read_csv(args.edge_map_csv),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out / "haufe_edges.csv", index=False)
    nodes.to_csv(out / "haufe_nodes.csv", index=False)
    (out / "settings.json").write_text(
        json.dumps(
            {
                "n_scans": 321,
                "n_children": 166,
                "participant_balance": "normalized inverse number of scans per child",
                "increment": "final Phase4A prediction minus baseline prediction",
                "increment_variance": variance,
                "node_summary_value": "covariance divided by increment variance",
                "enrichment_value": "weighted covariance",
            },
            indent=2,
        )
    )
    print("PASS: Gross-motor weighted Haufe patterns calculated")


if __name__ == "__main__":
    main()
