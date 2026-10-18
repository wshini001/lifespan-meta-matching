"""Calculate direct source-feature correlations and PC reconstructions.

Child-balanced correlations use within-child means before Pearson correlation.
"""

import numpy as np
import pandas as pd


def corr_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def child_balanced_correlation(aligned, feature_column, increment_column):
    child_data = (
        aligned[["imaging_numeric_id", feature_column, increment_column]]
        .groupby("imaging_numeric_id", as_index=False)
        .mean(numeric_only=True)
    )
    return (
        corr_safe(child_data[feature_column], child_data[increment_column]),
        int(len(child_data)),
    )


def align_source(oof, source, columns):
    if source.scan_id.duplicated().any() or oof.scan_id.duplicated().any():
        raise ValueError("Duplicate scan identifiers")
    aligned = oof.merge(
        source[["scan_id", *columns]], on="scan_id", how="left", validate="one_to_one"
    )
    if not np.isfinite(aligned[columns].to_numpy(float)).all():
        raise ValueError("Missing or non-finite aligned source values")
    return aligned


def feature_correlations(aligned, columns, increment, active=None):
    selected = aligned if active is None else aligned.loc[active].copy()
    if not len(selected):
        raise ValueError("Empty active sample")
    records = []
    for column in columns:
        r, n = child_balanced_correlation(selected, column, increment)
        all_r, _ = child_balanced_correlation(aligned, column, increment)
        records.append(
            {
                "source_column": column,
                "n_scans": len(selected),
                "n_children": n,
                "scan_level_r": corr_safe(aligned[column], aligned[increment]),
                "child_balanced_r": r,
                "child_balanced_allfold_r": all_r,
            }
        )
    return pd.DataFrame(records)


def reconstruct_pc_features(aligned, columns, loadings):
    names = [column[:-7] for column in columns]
    selected = loadings[loadings.pc_name.astype(str).isin(names)].copy()
    if selected.duplicated(["block", "raw_col", "pc_name"]).any():
        raise ValueError("Duplicate loading rows")
    if set(selected.pc_name.astype(str)) != set(names):
        raise ValueError("Loading coverage differs from transferred PC set")
    selected["raw_key"] = (
        selected.block.astype(str) + "||" + selected.raw_col.astype(str)
    )
    frame = (
        selected.pivot(
            index="raw_key", columns="pc_name", values="loading_aligned_to_saved_pc"
        )
        .reindex(columns=names)
        .fillna(0.0)
    )
    matrix = frame.to_numpy(float)
    values = aligned[columns].to_numpy(float) @ matrix.T
    if not np.isfinite(values).all():
        raise ValueError("Non-finite reconstructed values")
    metadata = (
        selected[["raw_key", "block", "raw_col"]]
        .drop_duplicates("raw_key")
        .set_index("raw_key")
        .loc[frame.index]
        .reset_index()
    )
    result = aligned.copy()
    reconstructed_columns = [f"__raw_{index:03d}" for index in range(len(metadata))]
    result = pd.concat(
        [
            result,
            pd.DataFrame(values, columns=reconstructed_columns, index=result.index),
        ],
        axis=1,
    )
    metadata["source_column"] = reconstructed_columns
    return (result, metadata)
