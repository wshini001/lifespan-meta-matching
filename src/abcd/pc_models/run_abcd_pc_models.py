"""Run the ABCD KRR-PC, DNN-PC, expanded-penalty or combined-PC prediction model."""

import argparse
import json
from pathlib import Path
import pandas as pd
import nested_engine
import dual_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["krr12", "dnn12", "krr12_expanded", "dual12"], required=True
    )
    parser.add_argument("--abcd-csv", required=True)
    parser.add_argument("--pred-parquet")
    parser.add_argument("--dnn-pred-parquet")
    parser.add_argument("--krr-pred-parquet")
    parser.add_argument("--freeze-tsv", required=True)
    parser.add_argument("--fold-npz", required=True)
    parser.add_argument("--fold-manifest-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    config = json.loads(
        (Path(__file__).resolve().parent / "config" / f"{args.model}.json").read_text()
    )
    if nested_engine.read_freeze_list(args.freeze_tsv) != config["source_names"]:
        raise ValueError("Frozen PC identities or order differ from the configuration")
    if len(pd.read_csv(args.fold_manifest_csv)) != config["n_subjects"]:
        raise ValueError("Unexpected locked cohort size")
    shared = {
        k: config[k]
        for k in [
            "target_col",
            "id_col",
            "subjectkey_col",
            "outer_k",
            "inner_k",
            "seed",
            "abcd_alpha_grid",
            "pc_alpha_grid",
            "inner_metric",
            "y_tolerance",
        ]
    }
    shared.update(
        abcd_csv=args.abcd_csv,
        freeze_tsv=args.freeze_tsv,
        fold_npz=args.fold_npz,
        fold_manifest_csv=args.fold_manifest_csv,
        out_dir=args.out_dir,
        log_stem=config["output_stem"],
    )
    if args.model == "dual12":
        if not args.dnn_pred_parquet or not args.krr_pred_parquet:
            parser.error(
                "dual12 requires both --dnn-pred-parquet and --krr-pred-parquet"
            )
        dual_engine.run_dual_source_nested(
            **shared,
            dnn_pred_parquet=args.dnn_pred_parquet,
            krr_pred_parquet=args.krr_pred_parquet,
            fold_col=config["fold_col"],
            pc_feature_mode=config["pc_feature_mode"],
            site_col="",
            age_col="",
            sex_col="",
            edu_col="",
            motion_cols="",
            expected_covariates=config["covariate_columns"],
        )
    else:
        if not args.pred_parquet:
            parser.error("Two-stream models require --pred-parquet")
        nested_engine.run_nested_ensemble(
            **shared,
            pred_parquet=args.pred_parquet,
            expected_covariates=config["covariate_columns"],
            source_predictor_family=config["source_predictor_family"],
        )


if __name__ == "__main__":
    main()
