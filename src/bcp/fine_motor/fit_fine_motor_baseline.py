"""Fit an eight-covariate RSFC-only fine-motor baseline on child-grouped folds."""

import argparse
import json
from pathlib import Path
import baseline_engine as engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--inner-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--alphas", default="0.0001,0.001,0.004,0.01,0.04,0.1,0.4,1,4,10"
    )
    args = parser.parse_args()
    data = engine.load_bcp_data(
        args.cohort_csv,
        "fine_motor_percentile",
        "eligible_fine_motor_percentile_without_tr",
        "imaging_numeric_id",
    )
    folds, fold_id = engine.load_outer_folds_from_manifest(
        data["meta"], args.fold_manifest
    )
    result = engine.fit_eval_outer(
        data["X"],
        data["y"],
        data["groups"],
        data["meta"],
        data["n_edge"],
        data["n_cov"],
        folds,
        fold_id,
        [float(x) for x in args.alphas.split(",")],
        args.inner_k,
        args.seed,
        "cod",
        data["cov_cols"],
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, name in [
        ("pred_df", "baseline_predictions.csv"),
        ("fold_metrics_df", "fold_metrics.csv"),
        ("inner_scores_df", "inner_scores.csv"),
        ("impute_df", "imputation_means.csv"),
    ]:
        result[key].to_csv(out / name, index=False)
    (out / "baseline_metrics.json").write_text(json.dumps(result["pooled"], indent=2))
    print(json.dumps(result["pooled"], indent=2))


if __name__ == "__main__":
    main()
