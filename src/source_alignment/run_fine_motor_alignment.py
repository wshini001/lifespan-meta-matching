"""Calculate fine-motor source and PC-reconstructed feature correlations.

Uses stage-specific prediction increments and the configured PC-active sample.
"""

import argparse
from pathlib import Path
import pandas as pd
from alignment_engine import align_source, feature_correlations, reconstruct_pc_features


def calculate(oof, banks, features, loadings):
    if len(oof) != 395 or oof.imaging_numeric_id.nunique() != 199:
        raise ValueError("Expected 395 scans from 199 children")
    if oof.groupby("imaging_numeric_id").fold.nunique().max() != 1:
        raise ValueError("Children span outer folds")
    parsed = oof.pc_effectively_active
    if parsed.dtype != bool:
        parsed = (
            parsed.astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    if parsed.isna().any():
        raise ValueError("Invalid PC activity values")
    oof = oof.copy()
    oof["pc_effectively_active"] = parsed.astype(bool)
    pc_active = oof[oof.pc_effectively_active]
    if (
        len(pc_active) != 315
        or pc_active.imaging_numeric_id.nunique() != 155
        or sorted(pc_active.fold.unique()) != [1, 2, 3, 4]
    ):
        raise ValueError("PC-active sample differs from the configured sample")
    rows = []
    reconstruction = None
    for family, branch, increment, expected in [
        ("disease", "disease_anchor_vs_baseline", "disease_oof_increment", 2),
        ("pc", "pc12_refinement_active", "pc12_oof_increment", 12),
        ("dnn", "dnnraw_M88_vs_pc12_anchor", "dnn_M88_oof_increment", 88),
    ]:
        metadata = features[features.family.eq(family)].copy()
        if len(metadata) != expected:
            raise ValueError("Unexpected source feature count: " + family)
        columns = metadata.source_column.tolist()
        aligned = align_source(oof, banks[family], columns)
        active = aligned.pc_effectively_active if family == "pc" else None
        result = feature_correlations(aligned, columns, increment, active)
        result = metadata.merge(result, on="source_column", validate="one_to_one")
        result["branch"] = branch
        rows.append(result)
        if family == "pc":
            raw, raw_metadata = reconstruct_pc_features(aligned, columns, loadings)
            reconstructed_columns = raw_metadata.source_column.tolist()
            result_raw = feature_correlations(
                raw, reconstructed_columns, increment, raw.pc_effectively_active
            )
            reconstruction = raw_metadata.merge(
                result_raw, on="source_column", validate="one_to_one"
            )
            reconstruction["feature_id"] = reconstruction.raw_col.astype(str)
            reconstruction["branch"] = "pc12_refinement_active_reconstructed_raw"
    return (pd.concat(rows, ignore_index=True), reconstruction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-csv", required=True)
    parser.add_argument("--disease-parquet", required=True)
    parser.add_argument("--pc-parquet", required=True)
    parser.add_argument("--dnn-parquet", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--loadings-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    banks = {
        name: pd.read_parquet(path)
        for name, path in [
            ("disease", args.disease_parquet),
            ("pc", args.pc_parquet),
            ("dnn", args.dnn_parquet),
        ]
    }
    source, raw = calculate(
        pd.read_csv(args.oof_csv),
        banks,
        pd.read_csv(args.feature_manifest, dtype={"feature_id": str}),
        pd.read_csv(args.loadings_csv),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source.to_csv(out / "source_correlations.csv", index=False)
    raw.to_csv(out / "reconstructed_correlations.csv", index=False)
    print("PASS: source-feature and direct PC-reconstructed correlations calculated")


if __name__ == "__main__":
    main()
