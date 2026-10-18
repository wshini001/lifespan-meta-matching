"""Align KRRRaw and DNNRaw predictions to visual-reception scans and folds."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def align(reference, bank, manifest):
    keys = ["scan_id", "imaging_numeric_id"]
    frames = []
    for original in [reference, bank]:
        frame = original.copy()
        frame["scan_id"] = frame["scan_id"].astype(str).str.strip()
        groups = pd.to_numeric(frame["imaging_numeric_id"], errors="raise").to_numpy(
            float
        )
        if (
            not np.isfinite(groups).all()
            or not np.equal(groups, np.floor(groups)).all()
        ):
            raise ValueError("Expected integer child identifiers")
        frame["imaging_numeric_id"] = groups.astype(np.int64).astype(str)
        if frame[keys].duplicated().any():
            raise ValueError("Duplicate scan/child identifiers")
        frames.append(frame)
    reference, bank = frames
    if len(reference) != 289 or reference["imaging_numeric_id"].nunique() != 159:
        raise ValueError("Expected 289 scans / 159 children")
    ordered = []
    for family in ["krrraw_top120", "dnnraw_top120"]:
        rows = manifest.loc[manifest["family"].eq(family)].sort_values(
            "family_feature_index_1based"
        )
        if rows["family_feature_index_1based"].tolist() != list(range(1, 121)):
            raise ValueError("Expected 120 consecutive features in each family")
        ordered.extend(rows["aligned_column"].astype(str).tolist())
    if len(set(ordered)) != 240 or not all((c.endswith("__zpred") for c in ordered)):
        raise ValueError("Expected 240 unique frozen z-predictions")
    folds = pd.to_numeric(reference["fold"], errors="raise").to_numpy(float)
    if (
        not np.isfinite(folds).all()
        or not np.equal(folds, np.floor(folds)).all()
        or set(folds) != set(range(5))
    ):
        raise ValueError("Expected integer visual folds 0–4")
    reference["fold"] = folds.astype(int)
    if reference.groupby("imaging_numeric_id")["fold"].nunique().max() != 1:
        raise ValueError("Child crosses visual folds")
    result = reference[keys + ["fold"]].merge(
        bank[keys + ordered], on=keys, how="left", validate="one_to_one"
    )
    if not np.isfinite(result[ordered].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite frozen source predictions")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-reference-csv", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--out-parquet", required=True)
    args = parser.parse_args()
    result = align(
        pd.read_csv(args.locked_reference_csv),
        pd.read_parquet(args.source_bank),
        pd.read_csv(args.feature_manifest, sep="\t"),
    )
    path = Path(args.out_parquet)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    print("PASS: 240 frozen raw-source predictions aligned to visual identifiers/folds")


if __name__ == "__main__":
    main()
