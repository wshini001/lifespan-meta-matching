"""Apply the five frozen UK Biobank KRR bundles for the Final12 PCs to HCP.

Each HCP RSFC row is centered and L2-normalized before correlation-kernel
prediction. Predictions are averaged across the requested UK Biobank seeds.
No HCP outcome is used in this transfer step.
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Sequence
import joblib
import numpy as np
import pandas as pd


def parse_seed_list(text: str) -> tuple[int, ...]:
    seeds = tuple((int(value.strip()) for value in text.split(",") if value.strip()))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def unitize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(matrix * matrix, axis=1, keepdims=True))
    norms[~np.isfinite(norms) | (norms == 0)] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def discover_bundles(directory: Path, seeds: Sequence[int]) -> list[Path]:
    paths = []
    for seed in seeds:
        expected = (
            directory
            / f"stage1_seed{seed}"
            / f"krr_stage1_pc_transfer_bundle_stage1_seed{seed}.joblib"
        )
        if expected.exists():
            paths.append(expected)
            continue
        candidates = sorted(directory.glob(f"**/*seed{seed}*.joblib"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Could not identify one bundle for seed {seed}: {candidates[:10]}"
            )
        paths.append(candidates[0])
    return paths


def load_and_validate_bundles(paths: Sequence[Path]):
    bundles = []
    reference_edges = None
    reference_targets = None
    for path in paths:
        bundle = joblib.load(path)
        required = ["edge_names", "target_names", "Z_train_full", "target_models"]
        missing = [key for key in required if key not in bundle]
        if missing:
            raise KeyError(f"Bundle {path} is missing {missing}")
        if bundle.get("bundle_type") != "ukb_stage1_krr_pc_transfer_bundle":
            raise RuntimeError(f"Unexpected bundle type in {path}")
        edges = [str(value) for value in bundle["edge_names"]]
        targets = [str(value) for value in bundle["target_names"]]
        if reference_edges is None:
            reference_edges, reference_targets = (edges, targets)
        elif edges != reference_edges or targets != reference_targets:
            raise RuntimeError("The UK Biobank bundles use inconsistent feature orders")
        if sorted(bundle["target_models"]) != sorted(targets):
            raise RuntimeError(f"Target-model mismatch in {path}")
        training = np.asarray(bundle["Z_train_full"])
        if training.ndim != 2 or not np.isfinite(training).all():
            raise RuntimeError(f"Invalid training matrix in {path}")
        bundles.append(bundle)
    if reference_edges != [f"e{i}" for i in range(1, 1486)]:
        raise RuntimeError("Expected the ordered edge contract e1..e1485")
    return (bundles, reference_edges, reference_targets)


def predict_target(
    hcp_edges: np.ndarray, training_edges: np.ndarray, model: dict, chunk_size: int
) -> np.ndarray:
    if model.get("status") != "ok":
        raise RuntimeError(
            f"Frozen model status is not ok: {model.get('failure_reason')}"
        )
    fit_indices = np.asarray(model["fit_train_idx"], dtype=np.int64)
    coefficients = np.asarray(model["dual_coef"], dtype=np.float64).ravel()
    if fit_indices.size == 0 or fit_indices.size != coefficients.size:
        raise RuntimeError("Invalid frozen KRR coefficients")
    if np.any(fit_indices < 0) or np.any(fit_indices >= training_edges.shape[0]):
        raise RuntimeError("Frozen KRR fit indices are out of bounds")
    training_fit = np.asarray(training_edges[fit_indices], dtype=np.float32)
    predictions = np.empty(hcp_edges.shape[0], dtype=np.float64)
    for start in range(0, hcp_edges.shape[0], max(1, chunk_size)):
        stop = min(hcp_edges.shape[0], start + max(1, chunk_size))
        kernel = np.asarray(hcp_edges[start:stop] @ training_fit.T, dtype=np.float64)
        predictions[start:stop] = kernel @ coefficients
    if not np.isfinite(predictions).all():
        raise RuntimeError("Transferred predictions contain non-finite values")
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-input", required=True)
    parser.add_argument("--ukb-bundle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Source seeds must be unique")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_paths = discover_bundles(Path(args.ukb_bundle_dir), seeds)
    bundles, edge_columns, targets = load_and_validate_bundles(bundle_paths)
    locked = pd.read_parquet(args.locked_input)
    required_metadata = ["Subject", "fold"]
    missing_metadata = [c for c in required_metadata if c not in locked.columns]
    if missing_metadata:
        raise KeyError(f"Locked HCP table is missing {missing_metadata}")
    if locked["Subject"].astype(str).duplicated().any():
        raise RuntimeError("Locked HCP table contains duplicate Subject IDs")
    missing_edges = [column for column in edge_columns if column not in locked.columns]
    if missing_edges:
        raise KeyError(f"Locked HCP table is missing edges: {missing_edges[:10]}")
    hcp_edges = locked[edge_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(hcp_edges).all():
        raise RuntimeError("Locked HCP edges contain non-finite values")
    hcp_edges = unitize_rows(hcp_edges)
    metadata_order = [
        "locked_row_idx",
        "edge_row_idx",
        "Subject",
        "ID",
        "fold",
        "family_group",
        "FluidCog",
        "Sex",
        "Age",
        "Age_in_Yrs",
        "Mother",
        "Father",
        "Race",
        "EarlyCog",
        "CrystalCog",
        "TotalCog",
    ]
    output = locked[[c for c in metadata_order if c in locked.columns]].copy()
    seed_predictions: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    for seed, bundle in zip(seeds, bundles):
        print(
            f"[HCP source] applying frozen UKB seed {seed}; {len(targets)} PCs",
            flush=True,
        )
        training_edges = np.asarray(bundle["Z_train_full"], dtype=np.float32)
        for target in targets:
            seed_predictions[target].append(
                predict_target(
                    hcp_edges,
                    training_edges,
                    bundle["target_models"][target],
                    args.chunk_size,
                ).astype(np.float32)
            )
    for target in targets:
        stack = np.vstack(seed_predictions[target]).astype(np.float64)
        output[f"{target}__zpred"] = np.mean(stack, axis=0).astype(np.float32)
    prediction_path = output_dir / "hcp_final12_pc_predictions.parquet"
    manifest_path = output_dir / "hcp_final12_pc_predictions_manifest.json"
    output.to_parquet(prediction_path, index=False)
    seed_from_path = []
    for path in bundle_paths:
        match = re.search("seed(\\d+)", str(path))
        seed_from_path.append(int(match.group(1)) if match else None)
    manifest = {
        "n_participants": int(len(output)),
        "n_edges": len(edge_columns),
        "n_pc_targets": len(targets),
        "targets": targets,
        "seeds_requested": list(seeds),
        "seeds_identified_from_paths": seed_from_path,
        "bundle_paths": [str(path) for path in bundle_paths],
        "locked_input": str(args.locked_input),
        "output": str(prediction_path),
        "hcp_outcome_used": False,
        "edge_order": "e1..e1485; semantic atlas equivalence must be established upstream",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
