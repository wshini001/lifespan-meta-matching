"""Calculate child-mean branch-specific Haufe covariances and correlations.

Branch definitions, active observations and covariance denominators are
specified in the supplied settings file.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from haufe_engine import compute_pnf, node_importance


def calculate(predictions, cohort, mapping, settings):
    keys = ["scan_id", "imaging_numeric_id"]
    edge_columns = mapping.edge_col.astype(str).tolist()
    if len(mapping) != 1485 or mapping.EdgeNumber.tolist() != list(range(1, 1486)):
        raise ValueError("Expected ordered 1485-edge mapping")
    pairs = mapping[["Node1", "Node2"]].to_numpy(int)
    expected = {(i, j) for i in range(1, 56) for j in range(i + 1, 56)}
    if set(map(tuple, pairs)) != expected or len(set(edge_columns)) != 1485:
        raise ValueError("Invalid edge identities")
    if [c for c in cohort.columns if c in set(edge_columns)] != edge_columns:
        raise ValueError("Raw cohort edge-column order differs")
    for frame in [predictions, cohort]:
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate scan/child keys")
    data = predictions.merge(
        cohort[keys + edge_columns], on=keys, how="left", validate="one_to_one"
    )
    if (
        len(data) != settings["n_scans"]
        or data.imaging_numeric_id.nunique() != settings["n_children"]
    ):
        raise ValueError("Unexpected aligned cohort size")
    if not np.isfinite(data[edge_columns].to_numpy(float)).all():
        raise ValueError("Missing or non-finite aligned connectivity")
    fold_column = settings["fold_column"]
    if fold_column == "auto":
        fold_column = "outer_fold" if "outer_fold" in data.columns else "fold"
    if not data.groupby("imaging_numeric_id")[fold_column].nunique().eq(1).all():
        raise ValueError("A child occurs in multiple outer folds")
    identity = [
        c
        for c in ["EdgeNumber", "Node1", "Node2", "edge_col", "ICA_Node1", "ICA_Node2"]
        if c in mapping
    ]
    edges = mapping[identity].copy()
    node_tables, samples = ([], {})
    for branch, spec in settings["branches"].items():
        use = data.copy()
        if "active_column" in spec:
            active = use[spec["active_column"]]
            if active.dtype != bool:
                active = (
                    active.astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"true": True, "false": False, "1": True, "0": False})
                )
            if active.isna().any():
                raise ValueError("Unrecognized active-fold flag")
            use = use.loc[active].copy()
        if "column" in spec:
            use["_increment"] = use[spec["column"]].to_numpy(float)
        else:
            use["_increment"] = use[spec["left"]] - use[spec["right"]]
        if not np.isfinite(use._increment).all():
            raise ValueError("Non-finite OOF increment")
        if (
            len(use) != spec["n_scans"]
            or use.imaging_numeric_id.nunique() != spec["n_children"]
        ):
            raise ValueError("Unexpected active branch sample: " + branch)
        if sorted(use[fold_column].unique().tolist()) != spec["folds"]:
            raise ValueError("Unexpected active folds: " + branch)
        child = (
            use[["imaging_numeric_id", fold_column, "_increment"] + edge_columns]
            .groupby("imaging_numeric_id", sort=True, as_index=False)
            .mean()
        )
        x = child[edge_columns].to_numpy(float)
        y = child._increment.to_numpy(float)
        result = compute_pnf(x, y, ddof=settings["covariance_ddof"])
        cov_column = branch + "__pnf_cov_subject"
        edges[cov_column] = result["cov"]
        edges[branch + "__pnf_corr_subject"] = result["corr"]
        within_x, within_y = ([], [])
        for _, part in child.groupby(fold_column, sort=True):
            xx, yy = (
                part[edge_columns].to_numpy(float),
                part._increment.to_numpy(float),
            )
            within_x.append(xx - xx.mean(axis=0, keepdims=True))
            within_y.append(yy - yy.mean())
        sensitivity = compute_pnf(
            np.vstack(within_x),
            np.concatenate(within_y),
            ddof=settings["covariance_ddof"],
        )
        edges[branch + "__pnf_cov_withinfold"] = sensitivity["cov"]
        edges[branch + "__pnf_corr_withinfold"] = sensitivity["corr"]
        nodes = node_importance(edges, cov_column)
        nodes.insert(0, "branch", branch)
        node_tables.append(nodes)
        samples[branch] = {"n_scans": len(use), "n_children": len(child)}
    return (edges, pd.concat(node_tables, ignore_index=True), samples)


def main():
    parser = argparse.ArgumentParser()
    for argument in [
        "predictions-csv",
        "cohort-csv",
        "edge-map-csv",
        "settings",
        "out-dir",
    ]:
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text())
    edges, nodes, samples = calculate(
        pd.read_csv(args.predictions_csv),
        pd.read_csv(args.cohort_csv),
        pd.read_csv(args.edge_map_csv),
        settings,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out / "haufe_edges.csv", index=False)
    nodes.to_csv(out / "haufe_nodes.csv", index=False)
    (out / "sample_contract.json").write_text(json.dumps(samples, indent=2))
    print("PASS: configured child-mean Haufe branches calculated")


if __name__ == "__main__":
    main()
