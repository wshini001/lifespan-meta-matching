"""Load visual-reception baseline features and child-grouped folds.

Uses full-table empirical-centile/logit signed features and eigenvalue-based
correlation-kernel ridge predictions.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

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
    40.0,
    100.0,
    400.0,
    1000.0,
    4000.0,
    10000.0,
    100000.0,
    1000000.0,
]
TARGET = "visual_reception_percentile"
GROUP = "imaging_numeric_id"
KEYS = ["scan_id", GROUP]


def load_engine():
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(ENGINE_PATH)
    specification = importlib.util.spec_from_file_location(
        "visual_phase3_engine", ENGINE_PATH
    )
    engine = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(engine)
    return engine


def kernel_predictions(kernel_train, kernel_test, y, alphas):
    symmetric = (kernel_train + kernel_train.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_y = eigenvectors.T @ y
    result = {}
    for alpha in alphas:
        dual = eigenvectors @ (projected_y / (eigenvalues + alpha))
        result[float(alpha)] = kernel_test @ dual
    return result


def load_data(engine):
    arguments = SimpleNamespace(
        cohort_csv=str(COHORT),
        fold_manifest_csv=str(FOLDS),
        fmri_res_csv=str(FMRI36),
        max_age_diff_days=7.0,
        expected_n=289,
        expected_children=159,
    )
    df, rsfc, fmri36, covariate_frame, fold_id, audit, edges, fmri36_columns = (
        engine.load_and_match(arguments)
    )
    signed = pd.read_csv(SIGNED72)
    signed_columns = sorted(
        (column for column in signed if column.startswith("fmrirescent_"))
    )
    if len(signed_columns) != 72:
        raise RuntimeError(f"Expected 72 signed features; found {len(signed_columns)}")
    left = df[KEYS].copy()
    right = signed[KEYS + signed_columns].copy()
    for key in KEYS:
        left[key] = engine.normalize_id(left[key])
        right[key] = engine.normalize_id(right[key])
    if right[KEYS].duplicated().any():
        raise RuntimeError("Duplicate signed72 keys")
    joined = (
        left.reset_index()
        .merge(right, on=KEYS, how="left", validate="one_to_one")
        .sort_values("index")
    )
    signed72 = joined[signed_columns].apply(pd.to_numeric, errors="coerce")
    if signed72.isna().any().any():
        raise RuntimeError("Missing signed72 values in locked N=289")
    if not np.isfinite(signed72.to_numpy(float)).all():
        raise RuntimeError("Non-finite signed72 values")
    if len(df) != 289 or df[GROUP].astype(str).nunique() != 159:
        raise RuntimeError("Expected locked N=289/159")
    return (
        df,
        rsfc,
        fmri36,
        signed72.to_numpy(float),
        covariate_frame,
        fold_id,
        audit,
        edges,
        fmri36_columns,
        signed_columns,
    )


ENGINE_PATH = Path(__file__).resolve().parent / "cohort_preprocessing.py"
COHORT = None
FOLDS = None
FMRI36 = None
SIGNED72 = None
