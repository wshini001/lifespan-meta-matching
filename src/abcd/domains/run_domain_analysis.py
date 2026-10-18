"""Run an ABCD domain-subset, leave-one-domain-out or multi-kernel analysis."""

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        choices=[
            "subset",
            "leave_domain",
            "linear_kernel",
            "three_kernel",
            "rbf_kernel",
        ],
        required=True,
    )
    parser.add_argument("--abcd-csv", required=True)
    parser.add_argument("--locked-predictions-csv", required=True)
    parser.add_argument("--pc-parquet", required=True)
    parser.add_argument("--freeze-tsv", required=True)
    parser.add_argument("--fold-manifest-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    cfg = json.loads(
        (
            Path(__file__).resolve().parent / "config" / f"{args.analysis}.json"
        ).read_text()
    )
    engine = importlib.import_module(args.analysis + "_engine")
    if engine.read_freeze_list(args.freeze_tsv) != cfg["pc_names"]:
        raise ValueError("Frozen PC identities/order differ from the configuration")
    manifest = pd.read_csv(args.fold_manifest_csv)
    if len(manifest) != cfg["expected_n"]:
        raise ValueError("Unexpected locked cohort size")
    params = cfg["parameters"].copy()
    params.update(
        abcd_csv=args.abcd_csv,
        locked_pred_csv=args.locked_predictions_csv,
        pc_pred_parquet=args.pc_parquet,
        freeze_tsv=args.freeze_tsv,
        fold_manifest_csv=args.fold_manifest_csv,
        out_dir=args.out_dir,
        expected_n=cfg["expected_n"],
        expected_covariates=cfg["covariate_columns"],
    )
    engine.run(SimpleNamespace(**params))


if __name__ == "__main__":
    main()
