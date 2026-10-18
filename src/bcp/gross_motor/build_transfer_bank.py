"""Align KRRRaw, DiseaseRaw and DNNRaw predictions to gross-motor scans."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["scan_id", "imaging_numeric_id"]


def normalize_keys(df):
    result = df.copy()
    result["scan_id"] = result["scan_id"].astype(str).str.strip()
    values = pd.to_numeric(result["imaging_numeric_id"], errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Child identifiers must be finite integers")
    result["imaging_numeric_id"] = values.astype(np.int64).astype(str)
    if result[KEYS].duplicated().any():
        raise ValueError("Duplicate scan/child keys")
    return result


def align_family(locked, source, manifest):
    source = normalize_keys(source)
    manifest = manifest.sort_values("family_feature_index_1based")
    columns = manifest.source_column.astype(str).tolist()
    if [str(c) for c in source if str(c).endswith("__zpred")] != columns:
        raise ValueError(
            "Source feature order differs from the source-feature specification"
        )
    aligned = locked[KEYS].merge(
        source[KEYS + columns], on=KEYS, how="left", validate="one_to_one"
    )
    values = aligned[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Missing source keys or non-finite source predictions")
    return pd.DataFrame(values, columns=manifest.aligned_column.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-manifest-csv", required=True)
    parser.add_argument("--krr-raw-parquet", required=True)
    parser.add_argument("--disease-raw-parquet", required=True)
    parser.add_argument("--dnn-raw-parquet", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    locked = normalize_keys(pd.read_csv(args.locked_manifest_csv))
    locked = locked.sort_values("_locked_row").reset_index(drop=True)
    if len(locked) != 321 or locked.imaging_numeric_id.nunique() != 166:
        raise ValueError("Expected 321 scans from 166 children")
    if not np.array_equal(locked._locked_row.to_numpy(int), np.arange(321)):
        raise ValueError("Unexpected locked row sequence")
    if locked.groupby("imaging_numeric_id").fold.nunique().max() != 1:
        raise ValueError("Children span outer folds")
    manifest = pd.read_csv(args.feature_manifest, sep="\t")
    bank = locked[["_locked_row", *KEYS, "fold"]].copy()
    for family, path in [
        ("krrraw_top120", args.krr_raw_parquet),
        ("diseaseraw330", args.disease_raw_parquet),
        ("dnnraw_mean5", args.dnn_raw_parquet),
    ]:
        part = align_family(
            locked, pd.read_parquet(path), manifest[manifest.family.eq(family)]
        )
        bank = pd.concat([bank, part], axis=1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank.to_parquet(out / "transfer_bank.parquet", index=False)
    print(
        json.dumps(
            {
                "n_scans": len(bank),
                "n_children": 166,
                "n_features": len(bank.columns) - 4,
            }
        )
    )


if __name__ == "__main__":
    main()
