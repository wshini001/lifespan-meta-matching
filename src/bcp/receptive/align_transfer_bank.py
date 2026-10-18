"""Align frozen source predictions to the receptive-language scan and fold order."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def align(cohort, bank, manifest):
    keys = ["scan_id", "imaging_numeric_id"]
    cohort = cohort.copy()
    bank = bank.copy()
    for frame in (cohort, bank):
        for key in keys:
            frame[key] = (
                frame[key].astype(str).str.strip().str.replace("\\.0$", "", regex=True)
            )
    membership = cohort["in_n241_legacy"].astype(str).str.lower().isin(["true", "1"])
    locked = cohort.loc[membership].reset_index(drop=True)
    if len(locked) != 241 or locked["imaging_numeric_id"].nunique() != 141:
        raise ValueError("Expected 241 scans / 141 children")
    if locked[keys].duplicated().any() or bank[keys].duplicated().any():
        raise ValueError("Duplicate scan/child keys")
    locked["fold"] = pd.to_numeric(
        locked["fold_n319_historical"], errors="raise"
    ).astype(int)
    if set(locked["fold"]) != set(range(5)):
        raise ValueError("Expected outer folds 0–4")
    if locked.groupby("imaging_numeric_id")["fold"].nunique().max() != 1:
        raise ValueError("Child crosses outer folds")
    columns = (
        manifest.sort_values(["family", "family_feature_index_1based"])[
            "aligned_column"
        ]
        .astype(str)
        .tolist()
    )
    if len(columns) != 614 or len(set(columns)) != 614:
        raise ValueError("Expected 614 unique frozen source features")
    aligned = locked[keys + ["fold"]].merge(
        bank[keys + columns], on=keys, how="left", validate="one_to_one"
    )
    if not np.isfinite(aligned[columns].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite frozen source features")
    aligned.insert(0, "_locked_row", np.arange(len(aligned)))
    return aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--out-parquet", required=True)
    args = parser.parse_args()
    result = align(
        pd.read_csv(args.cohort_csv),
        pd.read_parquet(args.source_bank),
        pd.read_csv(args.feature_manifest, sep="\t"),
    )
    path = Path(args.out_parquet)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    print("PASS: frozen source predictions aligned to 241 scans / 141 children")


if __name__ == "__main__":
    main()
