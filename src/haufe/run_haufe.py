"""Align OOF targets to RSFC and calculate child-mean Haufe patterns."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from haufe_engine import compute_pnf, node_importance


def align(master, cohort, settings):
    edges = [f"e{i}" for i in range(1, 1486)]
    if master.scan_id.duplicated().any() or cohort.scan_id.duplicated().any():
        raise ValueError("Duplicate scan identifiers")
    data = master.merge(
        cohort[["scan_id", "imaging_numeric_id", *edges]],
        on="scan_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_cohort"),
    )
    if (
        len(data) != settings["n_scans"]
        or data.imaging_numeric_id.nunique() != settings["n_children"]
    ):
        raise ValueError("Unexpected scan/child counts")
    if not np.array_equal(
        data.imaging_numeric_id.astype(str), data.imaging_numeric_id_cohort.astype(str)
    ):
        raise ValueError("Child identifiers differ between predictions and RSFC")
    target_columns = list(settings["targets"].values())
    if not np.isfinite(data[edges + target_columns].to_numpy(float)).all():
        raise ValueError("Non-finite edge or prediction values")
    if "fold" in data and data.groupby("imaging_numeric_id").fold.nunique().max() != 1:
        raise ValueError("A child occurs in multiple outer folds")
    child = (
        data[["imaging_numeric_id", *edges, *target_columns]]
        .groupby("imaging_numeric_id", as_index=False)
        .mean(numeric_only=True)
    )
    return (data, child, edges)


def calculate(master, cohort, mapping, settings, n_bootstrap=0):
    if settings["covariance_ddof"] != 1:
        raise ValueError("This entry point uses sample covariance")
    data, child, edges = align(master, cohort, settings)
    mapping = mapping.copy()
    if len(mapping) != 1485 or not np.array_equal(
        mapping.EdgeNumber.to_numpy(int), np.arange(1, 1486)
    ):
        raise ValueError("Expected edge mapping in exact EdgeNumber 1..1485 order")
    pairs = mapping[["Node1", "Node2"]].to_numpy(int)
    if set(map(tuple, np.sort(pairs, axis=1))) != {
        (i, j) for i in range(1, 56) for j in range(i + 1, 56)
    }:
        raise ValueError("Mapping does not contain all unique node pairs")
    output = mapping[
        [
            c
            for c in ["EdgeNumber", "Node1", "Node2", "ICA_Node1", "ICA_Node2"]
            if c in mapping
        ]
    ].copy()
    output["edge_col"] = edges
    x_scan, x_child = (data[edges].to_numpy(float), child[edges].to_numpy(float))
    for name, column in settings["targets"].items():
        for level, x, frame in [("scan", x_scan, data), ("subject", x_child, child)]:
            result = compute_pnf(x, frame[column].to_numpy(float))
            output[f"pnf_cov_{level}_{name}"] = result["cov"]
            output[f"pnf_corr_{level}_{name}"] = result["corr"]
    bootstrap = None
    if n_bootstrap:
        if n_bootstrap < 2:
            raise ValueError("At least two bootstrap draws required")
        rng = np.random.default_rng(settings["seed"])
        bootstrap = np.empty((n_bootstrap, 1485), dtype=np.float32)
        y = child[settings["targets"]["total"]].to_numpy(float)
        for draw in range(n_bootstrap):
            indices = rng.integers(0, len(child), size=len(child))
            bootstrap[draw] = compute_pnf(x_child[indices], y[indices])["cov"]
        lower = np.percentile(bootstrap, 2.5, axis=0)
        upper = np.percentile(bootstrap, 97.5, axis=0)
        output["bootstrap_mean_subject_total"] = np.mean(bootstrap, axis=0)
        output["bootstrap_ci2p5_subject_total"] = lower
        output["bootstrap_ci97p5_subject_total"] = upper
        output["bootstrap_sign_consistency"] = np.maximum(
            np.mean(bootstrap > 0, axis=0), np.mean(bootstrap < 0, axis=0)
        )
        output["bootstrap_sign_match_full"] = np.mean(
            np.sign(bootstrap)
            == np.sign(output.pnf_cov_subject_total.to_numpy())[None, :],
            axis=0,
        )
        output["bootstrap_ci_excludes_zero"] = (lower > 0) | (upper < 0)
    output["abs_pnf_subject_total"] = output.pnf_cov_subject_total.abs()
    output["abs_rank_subject_total"] = output.abs_pnf_subject_total.rank(
        method="min", ascending=False
    ).astype(int)
    nodes = node_importance(output, "pnf_cov_subject_total")
    if {"ICA_Node1", "ICA_Node2"} <= set(output):
        lookup = {}
        for row in output.itertuples():
            for node, ica in [(row.Node1, row.ICA_Node1), (row.Node2, row.ICA_Node2)]:
                if node in lookup and lookup[node] != ica:
                    raise ValueError("Inconsistent node-to-ICA mapping")
                lookup[node] = ica
        nodes["FMRIB_ICA_component"] = nodes.node.map(lookup)
    return (output, nodes, bootstrap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--edge-map-csv", required=True)
    parser.add_argument(
        "--settings", default=str(Path(__file__).parent / "config/fine_motor.json")
    )
    parser.add_argument("--n-bootstrap", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text())
    result, nodes, bootstrap = calculate(
        pd.read_csv(args.predictions_csv),
        pd.read_csv(args.cohort_csv),
        pd.read_csv(args.edge_map_csv),
        settings,
        args.n_bootstrap,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "haufe_edges.csv", index=False)
    nodes.to_csv(out / "haufe_nodes.csv", index=False)
    if bootstrap is not None:
        np.savez_compressed(out / "bootstrap_covariance.npz", covariance=bootstrap)
    (out / "settings.json").write_text(
        json.dumps({**settings, "n_bootstrap": args.n_bootstrap}, indent=2) + "\n"
    )
    print("PASS: child-mean Haufe patterns calculated")


if __name__ == "__main__":
    main()
