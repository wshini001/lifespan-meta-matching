"""Prepare eight outcome covariates and select receptive-language baselines.

Fits education imputation and outcome residualization on training partitions.
"""

from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

TARGET = "receptive_language_percentile"
GROUP = "imaging_numeric_id"
KEYS = ["scan_id", GROUP]
EDU = "mother_edu_gt_college"
ALPHAS = [
    0.0001,
    0.001,
    0.004,
    0.01,
    0.04,
    0.1,
    0.4,
    1.0,
    4.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
]
GAMMAS = [0.0, 0.01, 0.04, 0.1, 0.4, 1.0, 4.0, 10.0, 30.0, 100.0]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_covariates(df: pd.DataFrame, include_education: bool) -> np.ndarray:
    age = pd.to_numeric(df["scan_age_months"], errors="coerce").to_numpy(float)
    sex = (
        df["sex_final"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"F": 0.0, "M": 1.0})
        .to_numpy(float)
    )
    fd = pd.to_numeric(df["FD"], errors="coerce").to_numpy(float)
    site_unc = (
        df["site_primary"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("UNC")
        .astype(float)
        .to_numpy()
    )
    columns = [age, age**2, sex, fd, site_unc, age * sex, age**2 * sex]
    if include_education:
        columns.append(pd.to_numeric(df[EDU], errors="coerce").to_numpy(float))
    matrix = np.column_stack(columns)
    if not include_education and (not np.isfinite(matrix).all()):
        raise RuntimeError("Nonfinite basic covariates")
    return matrix


def residualize_train_test(ytr, yte, ctr, cte):
    ctr = np.asarray(ctr, float).copy()
    cte = np.asarray(cte, float).copy()
    means = np.nanmean(ctr, axis=0)
    means[~np.isfinite(means)] = 0.0
    for matrix in (ctr, cte):
        bad = ~np.isfinite(matrix)
        if bad.any():
            matrix[bad] = np.take(means, np.where(bad)[1])
    model = LinearRegression().fit(ctr, ytr)
    return (ytr - model.predict(ctr), yte - model.predict(cte))


def choose_candidate(grid, family=None, guarded=False, adaptive=False):
    if adaptive:
        candidates = grid.copy()
    elif guarded:
        candidates = grid[grid["family"].isin(["null", family])].copy()
    else:
        candidates = grid[grid["family"] == family].copy()
    if guarded or adaptive:
        candidates["null_preference"] = candidates["is_null"].astype(int)
        candidates["alpha_sort"] = candidates["alpha"].fillna(np.inf)
        candidates = candidates.sort_values(
            [
                "pooled_inner_COD",
                "mean_inner_COD",
                "min_inner_COD",
                "null_preference",
                "gamma",
                "alpha_sort",
                "family",
            ],
            ascending=[False, False, False, False, True, False, True],
        )
    else:
        candidates = candidates.sort_values(
            ["mean_inner_COD", "alpha", "gamma"], ascending=[False, True, True]
        )
    return candidates.iloc[0].to_dict()


ENGINE_PATH = Path(__file__).resolve().parent / "connectivity_kernels.py"
