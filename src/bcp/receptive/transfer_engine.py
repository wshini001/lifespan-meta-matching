"""Build receptive-language baseline and PC inner-OOF prediction caches.

Provides training-partition source scaling, model selection and row alignment.
"""

from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

TAG = "receptive_language_n241_transfer_singlebranch_strict_nested_v1"
TARGET = "receptive_language_percentile"
GROUP = "imaging_numeric_id"
EDU = "mother_edu_gt_college"
KEYS = ["scan_id", GROUP]
EXPECTED_N = 241
EXPECTED_CHILDREN = 141
TRANSFER_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
W_BASE = [0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
FAMILY_SPECS = {
    "krrpc_final12": [12],
    "dnnpc_final12": [12],
    "krrraw_top120": [5, 10, 15, 20, 30, 40, 60, 80, 100, 120],
    "dnnraw_top120": [5, 10, 15, 20, 30, 40, 60, 80, 100, 120],
    "diseasepc_top20": [5, 8, 12, 16, 20],
    "diseaseraw330": [5, 10, 20, 30, 50, 80, 120, 180, 240, 330],
}


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace("\\.0$", "", regex=True)


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for key in KEYS:
        result[key] = normalize(result[key])
    return result


def metrics(y, prediction):
    y = np.asarray(y, float).ravel()
    prediction = np.asarray(prediction, float).ravel()
    if (
        y.shape != prediction.shape
        or not np.isfinite(y).all()
        or (not np.isfinite(prediction).all())
    ):
        raise RuntimeError("Invalid metric input")
    error = y - prediction
    mse = float(np.mean(error**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    cod = float(1.0 - np.sum(error**2) / sst)
    correlation = (
        float(np.corrcoef(y, prediction)[0, 1])
        if y.std() > 0 and prediction.std() > 0
        else np.nan
    )
    return {
        "n": int(len(y)),
        "COD": cod,
        "R2": cod,
        "Pearson_r": correlation,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
    }


def zscore_train_test(train, test):
    train = np.asarray(train, float)
    test = np.asarray(test, float)
    mean = train.mean(axis=0)
    sd = train.std(axis=0, ddof=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return ((train - mean) / sd, (test - mean) / sd)


def fit_transfer(train_x, y_train, test_x, alpha):
    train_z, test_z = zscore_train_test(train_x, test_x)
    return (
        Ridge(alpha=float(alpha), fit_intercept=True)
        .fit(train_z, y_train)
        .predict(test_z)
    )


def score_parts(y, prediction, fold_id):
    pooled = metrics(y, prediction)
    fold_cod = [
        metrics(y[fold_id == fold], prediction[fold_id == fold])["COD"]
        for fold in sorted(np.unique(fold_id))
    ]
    return {
        "pooled_inner_COD": pooled["COD"],
        "pooled_inner_RMSE": pooled["RMSE"],
        "mean_inner_COD": float(np.mean(fold_cod)),
        "min_inner_COD": float(np.min(fold_cod)),
        "sd_inner_COD": float(np.std(fold_cod, ddof=1)),
    }


def choose_branch(rows):
    return sorted(
        rows,
        key=lambda row: (
            -row["pooled_inner_COD"],
            -row["mean_inner_COD"],
            -row["min_inner_COD"],
            -int(row["selected_null"]),
            row["top_m"],
            -row["alpha"] if np.isfinite(row["alpha"]) else -np.inf,
        ),
    )[0]


def choose_stack(rows):
    return sorted(
        rows,
        key=lambda row: (
            -row["pooled_inner_COD"],
            -row["mean_inner_COD"],
            -row["min_inner_COD"],
            -row["w_base"],
            row["top_m"],
            -row["alpha"],
        ),
    )[0]


def baseline_cache_for_split(base, engine, area, rsfc, y_train, train, valid):
    area_train, area_valid = engine.kernel_train_test_for_block(
        area[train], area[valid], "area_rescent"
    )
    rsfc_train, rsfc_valid = engine.kernel_train_test_for_block(
        rsfc[train], rsfc[valid], "rsfc"
    )
    cache = {}
    for gamma in base.GAMMAS:
        kernel_train = rsfc_train + float(gamma) * area_train
        kernel_valid = rsfc_valid + float(gamma) * area_valid
        for alpha in base.ALPHAS:
            cache[float(alpha), float(gamma)] = (
                KernelRidge(alpha=float(alpha), kernel="precomputed")
                .fit(kernel_train, y_train)
                .predict(kernel_valid)
            )
    return cache


def select_baseline(base, y, inner_id, cache, outer_fold):
    rows = [
        {
            "family": "null",
            "alpha": np.nan,
            "gamma": 0.0,
            "is_null": True,
            **score_parts(y, np.zeros(len(y)), inner_id),
        }
    ]
    for (alpha, gamma), prediction in cache.items():
        rows.append(
            {
                "family": "rsfc_plus_area_rescent",
                "alpha": alpha,
                "gamma": gamma,
                "is_null": False,
                **score_parts(y, prediction, inner_id),
            }
        )
    grid = pd.DataFrame(rows)
    selected = base.choose_candidate(
        grid, family="rsfc_plus_area_rescent", guarded=True
    )
    selected["outer_fold"] = outer_fold
    selected["method"] = "baseline"
    selected["top_m"] = 0
    selected["w_base"] = 1.0 if selected["family"] != "null" else 0.0
    selected["selected_null"] = selected["family"] == "null"
    return (selected, grid)


def build_inner_oof(
    base, engine, area, rsfc, families, y_raw, cov, groups, outer_train, outer_fold
):
    n = len(outer_train)
    y_oof = np.full(n, np.nan)
    inner_id = np.full(n, -1, int)
    baseline_cache = {
        (float(alpha), float(gamma)): np.full(n, np.nan)
        for gamma in base.GAMMAS
        for alpha in base.ALPHAS
    }
    transfer_cache = {
        family: {
            (int(top_m), float(alpha)): np.full(n, np.nan)
            for top_m in FAMILY_SPECS[family]
            for alpha in TRANSFER_ALPHAS
        }
        for family in FAMILY_SPECS
    }
    inner_folds = engine.grouped_folds(groups[outer_train], 5, 100 + outer_fold)
    for inner_fold, (tr_relative, va_relative) in enumerate(inner_folds):
        train = outer_train[tr_relative]
        valid = outer_train[va_relative]
        y_train, y_valid = base.residualize_train_test(
            y_raw[train], y_raw[valid], cov[train], cov[valid]
        )
        y_oof[va_relative] = y_valid
        inner_id[va_relative] = inner_fold
        current_base = baseline_cache_for_split(
            base, engine, area, rsfc, y_train, train, valid
        )
        for key, prediction in current_base.items():
            baseline_cache[key][va_relative] = prediction
        for family, matrix in families.items():
            for top_m in FAMILY_SPECS[family]:
                for alpha in TRANSFER_ALPHAS:
                    transfer_cache[family][top_m, float(alpha)][va_relative] = (
                        fit_transfer(
                            matrix[train, :top_m], y_train, matrix[valid, :top_m], alpha
                        )
                    )
    arrays = [y_oof, *baseline_cache.values()]
    for family_cache in transfer_cache.values():
        arrays.extend(family_cache.values())
    if (inner_id < 0).any() or any((not np.isfinite(value).all() for value in arrays)):
        raise RuntimeError("Incomplete/nonfinite inner OOF predictions")
    return (y_oof, inner_id, baseline_cache, transfer_cache)


def select_inner(base, y, inner_id, baseline_cache, transfer_cache, outer_fold):
    selected_baseline, baseline_grid = select_baseline(
        base, y, inner_id, baseline_cache, outer_fold
    )
    if selected_baseline["selected_null"]:
        baseline_oof = np.zeros(len(y), float)
    else:
        baseline_oof = baseline_cache[
            float(selected_baseline["alpha"]), float(selected_baseline["gamma"])
        ]
    null_score = score_parts(y, np.zeros(len(y)), inner_id)
    grid_rows = []
    for row in baseline_grid.to_dict(orient="records"):
        grid_rows.append({"outer_fold": outer_fold, "method": "baseline", **row})
    selections = {"baseline": selected_baseline, "families": {}}
    for family, cache in transfer_cache.items():
        branch_rows = [
            {
                "outer_fold": outer_fold,
                "family": family,
                "method": "branch",
                "top_m": 0,
                "alpha": np.nan,
                "w_base": 0.0,
                "selected_null": True,
                **null_score,
            }
        ]
        stack_rows = []
        grid_rows.extend(branch_rows)
        for (top_m, alpha), branch_oof in cache.items():
            branch_row = {
                "outer_fold": outer_fold,
                "family": family,
                "method": "branch",
                "top_m": top_m,
                "alpha": alpha,
                "w_base": 0.0,
                "selected_null": False,
                **score_parts(y, branch_oof, inner_id),
            }
            branch_rows.append(branch_row)
            grid_rows.append(branch_row)
            for weight in W_BASE:
                prediction = weight * baseline_oof + (1.0 - weight) * branch_oof
                stack_row = {
                    "outer_fold": outer_fold,
                    "family": family,
                    "method": "stack",
                    "top_m": top_m,
                    "alpha": alpha,
                    "w_base": weight,
                    "selected_null": False,
                    **score_parts(y, prediction, inner_id),
                }
                stack_rows.append(stack_row)
                grid_rows.append(stack_row)
        selections["families"][family] = {
            "branch": choose_branch(branch_rows),
            "stack": choose_stack(stack_rows),
        }
    return (selections, grid_rows)


def fit_baseline_outer(base, engine, area, rsfc, y_train, train, test, selection):
    if selection["selected_null"]:
        return np.zeros(len(test), float)
    area_train, area_test = engine.kernel_train_test_for_block(
        area[train], area[test], "area_rescent"
    )
    rsfc_train, rsfc_test = engine.kernel_train_test_for_block(
        rsfc[train], rsfc[test], "rsfc"
    )
    gamma = float(selection["gamma"])
    return (
        KernelRidge(alpha=float(selection["alpha"]), kernel="precomputed")
        .fit(rsfc_train + gamma * area_train, y_train)
        .predict(rsfc_test + gamma * area_test)
    )


def load_data(base, engine):
    cohort = normalize_keys(pd.read_csv(COHORT))
    membership = cohort["in_n241_legacy"]
    if membership.dtype != bool:
        membership = membership.astype(str).str.lower().isin(["true", "1"])
    frame = cohort.loc[membership].copy().reset_index(drop=True)
    raw = pd.read_csv(RAW_AREA).rename(columns={"Unnamed: 0": "res_id_raw"})
    raw["res_id_raw"] = normalize(raw["res_id_raw"])
    area_columns = [column for column in raw if column.startswith("Area_")]
    if len(area_columns) != 70 or raw["res_id_raw"].duplicated().any():
        raise RuntimeError("Raw Area contract failed")
    frame = frame.merge(
        raw[["res_id_raw", *area_columns]],
        left_on="scan_id",
        right_on="res_id_raw",
        how="left",
        validate="one_to_one",
    )
    rsfc_columns = sorted(
        [column for column in frame if column.startswith("e") and column[1:].isdigit()],
        key=lambda value: int(value[1:]),
    )
    if len(frame) != EXPECTED_N or frame[GROUP].nunique() != EXPECTED_CHILDREN:
        raise RuntimeError("Expected locked N=241/141")
    if len(rsfc_columns) != 1485:
        raise RuntimeError("Expected 1485 RSFC features")
    area = frame[area_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    rsfc = frame[rsfc_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    y_raw = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(float)
    cov = base.make_covariates(frame, include_education=True)
    groups = frame[GROUP].to_numpy(str)
    fold_id = pd.to_numeric(frame["fold_n319_historical"], errors="raise").to_numpy(int)
    if not np.isfinite(area).all() or not np.isfinite(rsfc).all():
        raise RuntimeError("Nonfinite imaging features")
    return (frame, area, rsfc, y_raw, cov, groups, fold_id, area_columns, rsfc_columns)


def load_families(frame, fold_id):
    bank = normalize_keys(pd.read_parquet(BANK))
    manifest = pd.read_csv(FEATURE_MANIFEST, sep="\t")
    if bank[KEYS].duplicated().any():
        raise RuntimeError("Duplicate transfer-bank keys")
    aligned = frame[KEYS].merge(bank, on=KEYS, how="left", validate="one_to_one")
    if aligned["fold"].isna().any() or not np.array_equal(
        pd.to_numeric(aligned["fold"]).to_numpy(int), fold_id
    ):
        raise RuntimeError("Transfer-bank alignment/fold contract failed")
    families, columns_by_family = ({}, {})
    for family, top_m_values in FAMILY_SPECS.items():
        rows = manifest.loc[manifest["family"] == family].copy()
        rows["family_feature_index_1based"] = pd.to_numeric(
            rows["family_feature_index_1based"], errors="raise"
        ).astype(int)
        rows = rows.sort_values("family_feature_index_1based")
        columns = rows["aligned_column"].astype(str).tolist()
        expected = max(top_m_values)
        if len(columns) != expected or rows[
            "family_feature_index_1based"
        ].tolist() != list(range(1, expected + 1)):
            raise RuntimeError(f"{family} feature-order/count contract failed")
        if any((column not in aligned for column in columns)):
            raise RuntimeError(f"{family} missing aligned columns")
        matrix = aligned[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"{family} contains NaN/Inf")
        families[family] = matrix
        columns_by_family[family] = columns
    return (families, columns_by_family)


PHASE1_PY = Path(__file__).resolve().parent / "outcome_preprocessing.py"
COHORT = None
RAW_AREA = None
BANK = None
FEATURE_MANIFEST = None
