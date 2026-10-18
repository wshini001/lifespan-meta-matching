"""Calculate ABCD participant-level PC and reconstructed-feature correlations."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from alignment_engine import corr_safe


def calculate(predictions, source, loadings, names):
    keys = ["subjectkey", "src_subject_id"]
    columns = [name + "__zpred" for name in names]
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError("Expected exactly twelve frozen PC names")
    predictions, source = (predictions.copy(), source.copy())
    for frame in [predictions, source]:
        for key in keys:
            frame[key] = frame[key].astype(str)
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate participant keys")
    merged = predictions.merge(
        source[keys + columns], on=keys, how="left", validate="one_to_one"
    )
    if len(merged) != 9954 or not np.isfinite(merged[columns].to_numpy(float)).all():
        raise ValueError("Expected 9954 aligned participants with complete frozen PCs")
    increment = merged.y_pred_ensemble_nested.to_numpy(
        float
    ) - merged.y_pred_abcd_nested.to_numpy(float)
    final = merged.y_pred_ensemble_nested.to_numpy(float)
    pcs = pd.DataFrame(
        [
            {
                "pc_name": name,
                "increment_r": corr_safe(merged[column], increment),
                "final_r": corr_safe(merged[column], final),
            }
            for name, column in zip(names, columns)
        ]
    )
    selected = (
        loadings[loadings.pc_name.isin(names)]
        .sort_values(["block", "raw_col", "pc_name"])
        .reset_index(drop=True)
    )
    if selected.duplicated(["block", "raw_col", "pc_name"]).any() or set(
        selected.pc_name
    ) != set(names):
        raise ValueError("Missing or duplicate frozen PC loading rows")
    rows = []
    for (block, feature), group in selected.groupby(["block", "raw_col"], sort=False):
        z = merged[[name + "__zpred" for name in group.pc_name]].to_numpy(float)
        weights = group.loading_aligned_to_saved_pc.to_numpy(float)
        reconstructed = z @ weights
        if not np.isfinite(reconstructed).all():
            raise ValueError("Non-finite reconstructed feature")
        rows.append(
            {
                "block": block,
                "raw_col": feature,
                "recomputed_increment_r": corr_safe(reconstructed, increment),
                "recomputed_final_r": corr_safe(reconstructed, final),
            }
        )
    raw = pd.DataFrame(rows)
    if len(raw) != 112:
        raise ValueError("Expected 112 reconstructed features")
    return (pcs, raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--pc-parquet", required=True)
    parser.add_argument("--loadings-csv", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    pred = pd.read_csv(args.predictions_csv)
    features = pd.read_csv(args.feature_manifest)
    names = features[features.family.eq("pc")].feature_id.astype(str).tolist()
    pc, raw = calculate(
        pred, pd.read_parquet(args.pc_parquet), pd.read_csv(args.loadings_csv), names
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pc.to_csv(out / "pc_correlations.csv", index=False)
    raw.to_csv(out / "reconstructed_correlations.tsv", sep="\t", index=False)
    print("PASS: ABCD participant-level direct correlations calculated")


if __name__ == "__main__":
    main()
