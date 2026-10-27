"""Predict from standalone parameter exports without source participants/logs.

Output rows describe target participants and must remain in authorized storage.
This is a new bank-generation entry point, not a rerun of historical results.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .parameter_io import EDGES, predict_dnn, predict_krr, require, validate_krr
except ImportError:
    from parameter_io import EDGES, predict_dnn, predict_krr, require, validate_krr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        help="One export_source_parameters.py output directory",
    )
    parser.add_argument(
        "--cohort-csv", required=True, help="Authorized target table with e1..e1485"
    )
    parser.add_argument(
        "--id-column", required=True, help="Target alignment column to preserve"
    )
    parser.add_argument(
        "--prediction-scale",
        choices=["z", "scaled"],
        required=True,
        help="Select explicitly; raw-disease scaled output is still a residual score",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="New participant-level target output; keep private",
    )
    args = parser.parse_args()
    os.umask(0o077)
    destination = Path(args.out_csv)
    require(not destination.exists(), "Prediction output exists; refusing overwrite")
    metadata_path = Path(args.model_dir) / "model_metadata.json"
    meta = (
        json.loads(metadata_path.read_text())
        if metadata_path.is_file()
        else validate_krr(args.model_dir)
    )
    cohort = pd.read_csv(args.cohort_csv, usecols=[args.id_column] + EDGES)
    require(
        cohort[args.id_column].notna().all() and cohort[args.id_column].is_unique,
        "Use a nonmissing unique row identifier",
    )
    predictor = (
        predict_dnn
        if meta.get("model_kind") == "dnn"
        or meta.get("format") == "original_pytorch_state_dict_and_whitelisted_norm"
        else predict_krr
    )
    names, z, scaled = predictor(args.model_dir, cohort[EDGES].to_numpy())
    values = z if args.prediction_scale == "z" else scaled
    require(
        args.id_column not in names and np.isfinite(values).all(),
        "Invalid prediction columns/values",
    )
    result = pd.DataFrame(values, columns=names)
    result.insert(0, args.id_column, cohort[args.id_column].to_numpy())
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(
        json.dumps(
            {
                "rows": len(result),
                "targets": len(names),
                "prediction_scale": args.prediction_scale,
            }
        )
    )


if __name__ == "__main__":
    main()
