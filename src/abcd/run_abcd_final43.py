"""Fit the ABCD final43 nested two-stream ensemble.

Uses locked outer folds, inner-OOF stacking and training-fold outcome
residualization. Covariate imputation precedes fold-specific fitting;
missing counts before imputation are recorded in output metadata.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold


def read_freeze_list(freeze_tsv_path: str) -> List[str]:
    cols = []
    with open(freeze_tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            cols.append(s.split("\t")[0])
    if cols and cols[0].lower() == "col":
        cols = cols[1:]
    if not cols:
        raise ValueError(f"Freeze list is empty: {freeze_tsv_path}")
    return cols


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        raise RuntimeError("No finite values for metric computation.")
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = np.nan
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    cod = float(1.0 - sse / sst) if sst > 0 else np.nan
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Pearson_r": r, "COD": cod}


def _score_for_alpha(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return -np.inf
    if metric == "pearson":
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            return float(np.corrcoef(y_true, y_pred)[0, 1])
        return 0.0
    if metric == "cod":
        sse = float(np.sum((y_true - y_pred) ** 2))
        sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return float(1.0 - sse / sst) if sst > 0 else -np.inf
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"Unknown metric: {metric}")


def _unitize_rows(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    A = A - A.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(A, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return A / norm


def demean_norm(A: np.ndarray) -> np.ndarray:
    return _unitize_rows(A)


def residualize_train_test(
    y_train: np.ndarray, y_test: np.ndarray, cov_train: np.ndarray, cov_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    lr = LinearRegression()
    lr.fit(cov_train, y_train)
    y_train_res = y_train - lr.predict(cov_train)
    y_test_res = y_test - lr.predict(cov_test)
    return (y_train_res, y_test_res)


def _safe_sd(x: np.ndarray) -> float:
    sd = float(np.std(np.asarray(x, dtype=np.float64)))
    if not np.isfinite(sd) or sd == 0.0:
        return 1.0
    return sd


def _parse_float_list(s: str) -> Tuple[float, ...]:
    vals = tuple((float(x.strip()) for x in str(s).split(",") if x.strip()))
    if not vals:
        raise ValueError(f"Empty float list: {s}")
    return vals


def load_stage2a_pc_predictions(
    pred_parquet: str, freeze_tsv: str, id_col: str, subjectkey_col: str
) -> Dict[str, np.ndarray]:
    source_names = read_freeze_list(freeze_tsv)
    df = pd.read_parquet(pred_parquet)
    required = [id_col, subjectkey_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"PC Stage2A parquet missing columns: {missing}")
    z_names = [f"{name}__zpred" for name in source_names]
    missing_z = [c for c in z_names if c not in df.columns]
    if missing_z:
        raise KeyError(
            f"PC Stage2A parquet missing z-prediction columns: {missing_z[:10]}"
        )
    pc = df[z_names].to_numpy(dtype=np.float64)
    if not np.isfinite(pc).all():
        raise RuntimeError("PC prediction matrix contains NaN/Inf.")
    return {
        "id": df[id_col].astype(str).to_numpy(),
        "subjectkey": df[subjectkey_col].astype(str).to_numpy(),
        "pc": pc,
        "source_names": np.asarray(source_names, dtype=object),
    }


def load_fold_manifest(
    fold_manifest_csv: str,
    ids: np.ndarray,
    subjectkeys: np.ndarray,
    id_col: str,
    subjectkey_col: str,
) -> pd.DataFrame:
    manifest = pd.read_csv(fold_manifest_csv)
    required = [id_col, subjectkey_col, "_orig_row_idx", "fold"]
    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise KeyError(f"Fold manifest missing columns: {missing}")
    if len(manifest) != len(ids):
        raise RuntimeError(
            f"Manifest length mismatch: manifest={len(manifest)}, PC={len(ids)}"
        )
    if not np.array_equal(manifest[id_col].astype(str).to_numpy(), ids.astype(str)):
        raise RuntimeError("PC prediction order does not match fold manifest id order.")
    if not np.array_equal(
        manifest[subjectkey_col].astype(str).to_numpy(), subjectkeys.astype(str)
    ):
        raise RuntimeError(
            "PC prediction order does not match fold manifest subjectkey order."
        )
    if manifest[[id_col, subjectkey_col]].astype(str).duplicated().any():
        raise RuntimeError("Fold manifest contains duplicated subjects.")
    return manifest.copy()


def load_fullcv_folds(
    fold_npz: str, expected_n_subjects: int, expected_outer_k: int
) -> np.ndarray:
    z = np.load(fold_npz, allow_pickle=True)
    if "fold_id" not in z:
        raise KeyError(f"fold npz missing fold_id: {fold_npz}")
    fold_id = np.asarray(z["fold_id"], dtype=np.int64)
    if len(fold_id) != int(expected_n_subjects):
        raise RuntimeError(
            f"fold_id length mismatch: {len(fold_id)} vs {expected_n_subjects}"
        )
    outer_k_in_file = (
        int(np.asarray(z["outer_k"]).item())
        if "outer_k" in z
        else int(np.max(fold_id) + 1)
    )
    if outer_k_in_file != int(expected_outer_k):
        raise RuntimeError(
            f"outer_k mismatch: file={outer_k_in_file}, expected={expected_outer_k}"
        )
    unique = sorted(np.unique(fold_id).tolist())
    if unique != list(range(int(expected_outer_k))):
        raise RuntimeError(f"Unexpected fold labels: {unique}")
    return fold_id


def load_abcd_locked_data(
    abcd_csv: str,
    fold_manifest: pd.DataFrame,
    id_col: str,
    subjectkey_col: str,
    target_col: str,
) -> Dict[str, object]:
    df_all = pd.read_csv(abcd_csv)
    df_all["_orig_row_idx"] = np.arange(len(df_all), dtype=np.int64)
    keep_idx = fold_manifest["_orig_row_idx"].astype(np.int64).to_numpy()
    if np.any(keep_idx < 0) or np.any(keep_idx >= len(df_all)):
        raise RuntimeError(
            "Fold manifest contains _orig_row_idx outside ABCD csv row range."
        )
    df = (
        df_all.set_index("_orig_row_idx", drop=False)
        .loc[keep_idx]
        .reset_index(drop=True)
    )
    if id_col in df.columns:
        if not np.array_equal(
            df[id_col].astype(str).to_numpy(),
            fold_manifest[id_col].astype(str).to_numpy(),
        ):
            raise RuntimeError("ABCD rows do not match fold manifest id order.")
    if subjectkey_col in df.columns:
        if not np.array_equal(
            df[subjectkey_col].astype(str).to_numpy(),
            fold_manifest[subjectkey_col].astype(str).to_numpy(),
        ):
            raise RuntimeError("ABCD rows do not match fold manifest subjectkey order.")
    edge_cols = [f"e{i}" for i in range(1, 1486)]
    missing_edges = [c for c in edge_cols if c not in df.columns]
    if missing_edges:
        raise KeyError(f"ABCD csv missing RSFC edge columns: {missing_edges[:10]}")
    E = df[edge_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(E).all():
        raise RuntimeError("ABCD RSFC edge matrix contains NaN/Inf.")
    if target_col not in df.columns:
        raise KeyError(f"ABCD csv missing target column: {target_col}")
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise RuntimeError("ABCD target contains NaN/Inf.")
    cov = pd.DataFrame(index=df.index)
    need = [
        "Age_years",
        "Sex",
        "iqc_rsfmri_1_mean_motion",
        "iqc_rsfmri_2_mean_motion",
        "iqc_rsfmri_3_mean_motion",
        "iqc_rsfmri_4_mean_motion",
    ]
    missing_need = [c for c in need if c not in df.columns]
    if missing_need:
        raise KeyError(f"ABCD csv missing required covariate columns: {missing_need}")
    cov["Age_years"] = pd.to_numeric(df["Age_years"], errors="coerce").astype(
        np.float64
    )
    cov["Sex"] = pd.to_numeric(df["Sex"], errors="coerce").astype(np.float64)
    for i in range(1, 5):
        col = f"iqc_rsfmri_{i}_mean_motion"
        cov[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    if "Edu" in df.columns:
        cov["Edu"] = pd.to_numeric(df["Edu"], errors="coerce").astype(np.float64)
    if "site" in df.columns:
        site_dummies = pd.get_dummies(df["site"], prefix="site", drop_first=True)
        cov = pd.concat([cov, site_dummies], axis=1)
    cov["age_sq"] = cov["Age_years"] ** 2
    cov["age_sex"] = cov["Age_years"] * cov["Sex"]
    cov["age_sq_sex"] = cov["Age_years"] ** 2 * cov["Sex"]
    covariate_missing_counts = {str(c): int(cov[c].isna().sum()) for c in cov.columns}
    for col in cov.columns:
        if cov[col].isna().any():
            med = cov[col].median()
            cov[col] = cov[col].fillna(med)
    if cov.isna().any().any():
        bad = cov.columns[cov.isna().any()].tolist()
        raise RuntimeError(
            f"Covariates still contain NaN after median imputation: {bad}"
        )
    C = cov.to_numpy(dtype=np.float64)
    if not np.isfinite(C).all():
        raise RuntimeError("Covariate matrix contains NaN/Inf.")
    meta_cols = []
    for c in [
        "_orig_row_idx",
        subjectkey_col,
        id_col,
        "interview_date",
        "age",
        "Age_years",
        "Sex",
        "site",
        target_col,
    ]:
        if c in df.columns and c not in meta_cols:
            meta_cols.append(c)
    meta = df[meta_cols].copy()
    return {
        "E": E,
        "y": y,
        "cov": C,
        "cov_cols": list(cov.columns),
        "edge_cols": edge_cols,
        "meta": meta,
        "covariate_missing_counts": covariate_missing_counts,
    }


def fit_predict_abcd_krr(
    E_train: np.ndarray,
    E_test: np.ndarray,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    cov_train: np.ndarray,
    cov_test: np.ndarray,
    alpha_grid: Sequence[float],
    inner_metric: str,
    seed: int,
    inner_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    n_train = E_train.shape[0]
    n_splits = min(int(inner_k), n_train)
    if n_splits < 2:
        raise RuntimeError("Not enough training subjects for inner CV.")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    alpha_scores = []
    alpha_fold_scores = {}
    for alpha in alpha_grid:
        scores = []
        for tr_idx, va_idx in cv.split(np.arange(n_train)):
            y_tr_res, y_va_res = residualize_train_test(
                y_train_raw[tr_idx],
                y_train_raw[va_idx],
                cov_train[tr_idx, :],
                cov_train[va_idx, :],
            )
            Z_tr = _unitize_rows(E_train[tr_idx, :])
            Z_va = _unitize_rows(E_train[va_idx, :])
            K_tr = Z_tr @ Z_tr.T
            K_va = Z_va @ Z_tr.T
            model = KernelRidge(alpha=float(alpha), kernel="precomputed")
            model.fit(K_tr, y_tr_res)
            pred_va = np.asarray(model.predict(K_va), dtype=np.float64)
            scores.append(_score_for_alpha(y_va_res, pred_va, metric=inner_metric))
        alpha_scores.append(float(np.nanmean(scores)))
        alpha_fold_scores[str(float(alpha))] = [float(s) for s in scores]
    best_i = int(np.nanargmax(np.asarray(alpha_scores, dtype=np.float64)))
    best_alpha = float(list(alpha_grid)[best_i])
    y_train_res, y_test_res = residualize_train_test(
        y_train_raw, y_test_raw, cov_train, cov_test
    )
    Z_train = _unitize_rows(E_train)
    Z_test = _unitize_rows(E_test)
    K_train = Z_train @ Z_train.T
    K_test = Z_test @ Z_train.T
    final_model = KernelRidge(alpha=best_alpha, kernel="precomputed")
    final_model.fit(K_train, y_train_res)
    y_test_pred = np.asarray(final_model.predict(K_test), dtype=np.float64)
    info = {
        "best_alpha": best_alpha,
        "alpha_grid": [float(a) for a in alpha_grid],
        "alpha_mean_scores": [float(s) for s in alpha_scores],
        "alpha_fold_scores": alpha_fold_scores,
        "inner_metric": inner_metric,
    }
    return (y_test_res, y_test_pred, best_alpha, info)


def fit_predict_pc_krr(
    pc_train: np.ndarray,
    pc_test: np.ndarray,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    cov_train: np.ndarray,
    cov_test: np.ndarray,
    alpha_grid: Sequence[float],
    inner_metric: str,
    seed: int,
    inner_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    n_train = pc_train.shape[0]
    n_splits = min(int(inner_k), n_train)
    if n_splits < 2:
        raise RuntimeError("Not enough training subjects for inner CV.")
    X_all = demean_norm(pc_train)
    X_test = demean_norm(pc_test)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    alpha_scores = []
    alpha_fold_scores = {}
    for alpha in alpha_grid:
        scores = []
        for tr_idx, va_idx in cv.split(np.arange(n_train)):
            y_tr_res, y_va_res = residualize_train_test(
                y_train_raw[tr_idx],
                y_train_raw[va_idx],
                cov_train[tr_idx, :],
                cov_train[va_idx, :],
            )
            model = KernelRidge(alpha=float(alpha))
            model.fit(X_all[tr_idx, :], y_tr_res)
            pred_va = np.asarray(model.predict(X_all[va_idx, :]), dtype=np.float64)
            scores.append(_score_for_alpha(y_va_res, pred_va, metric=inner_metric))
        alpha_scores.append(float(np.nanmean(scores)))
        alpha_fold_scores[str(float(alpha))] = [float(s) for s in scores]
    best_i = int(np.nanargmax(np.asarray(alpha_scores, dtype=np.float64)))
    best_alpha = float(list(alpha_grid)[best_i])
    y_train_res, y_test_res = residualize_train_test(
        y_train_raw, y_test_raw, cov_train, cov_test
    )
    final_model = KernelRidge(alpha=best_alpha)
    final_model.fit(X_all, y_train_res)
    y_test_pred = np.asarray(final_model.predict(X_test), dtype=np.float64)
    info = {
        "best_alpha": best_alpha,
        "alpha_grid": [float(a) for a in alpha_grid],
        "alpha_mean_scores": [float(s) for s in alpha_scores],
        "alpha_fold_scores": alpha_fold_scores,
        "inner_metric": inner_metric,
    }
    return (y_test_res, y_test_pred, best_alpha, info)


def fit_ensemble_from_inner_oof(
    y_inner: np.ndarray, pred_abcd_inner: np.ndarray, pred_pc_inner: np.ndarray
) -> Tuple[LinearRegression, np.ndarray, np.ndarray, Dict[str, float]]:
    X_raw = np.column_stack([pred_abcd_inner, pred_pc_inner]).astype(np.float64)
    y_inner = np.asarray(y_inner, dtype=np.float64).ravel()
    if not np.isfinite(X_raw).all() or not np.isfinite(y_inner).all():
        raise RuntimeError("Inner OOF arrays contain NaN/Inf.")
    x_mu = X_raw.mean(axis=0, keepdims=True)
    x_sd = X_raw.std(axis=0, keepdims=True)
    x_sd[~np.isfinite(x_sd)] = 1.0
    x_sd[x_sd == 0.0] = 1.0
    X = (X_raw - x_mu) / x_sd
    lr = LinearRegression()
    lr.fit(X, y_inner)
    pred_inner = np.asarray(lr.predict(X), dtype=np.float64)
    train_metrics = compute_metrics(y_inner, pred_inner)
    train_metrics["coef_abcd_standardized"] = float(lr.coef_[0])
    train_metrics["coef_pc_standardized"] = float(lr.coef_[1])
    train_metrics["intercept"] = float(lr.intercept_)
    train_metrics["x_mu_abcd"] = float(x_mu[0, 0])
    train_metrics["x_mu_pc"] = float(x_mu[0, 1])
    train_metrics["x_sd_abcd"] = float(x_sd[0, 0])
    train_metrics["x_sd_pc"] = float(x_sd[0, 1])
    return (lr, x_mu, x_sd, train_metrics)


def generate_inner_oof_base_predictions(
    E: np.ndarray,
    pc: np.ndarray,
    y: np.ndarray,
    cov: np.ndarray,
    outer_train_idx: np.ndarray,
    outer_fold: int,
    inner_k: int,
    abcd_alpha_grid: Sequence[float],
    pc_alpha_grid: Sequence[float],
    inner_metric: str,
    seed: int,
    y_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict], List[dict]]:
    n_outer_train = len(outer_train_idx)
    pred_abcd = np.full(n_outer_train, np.nan, dtype=np.float64)
    pred_pc = np.full(n_outer_train, np.nan, dtype=np.float64)
    y_inner = np.full(n_outer_train, np.nan, dtype=np.float64)
    inner_rows = []
    alpha_rows = []
    cv = KFold(n_splits=int(inner_k), shuffle=True, random_state=int(seed))
    for inner_fold, (tr_rel, va_rel) in enumerate(cv.split(np.arange(n_outer_train))):
        tr_abs = outer_train_idx[tr_rel]
        va_abs = outer_train_idx[va_rel]
        y_va_abcd, p_va_abcd, alpha_abcd, info_abcd = fit_predict_abcd_krr(
            E_train=E[tr_abs, :],
            E_test=E[va_abs, :],
            y_train_raw=y[tr_abs],
            y_test_raw=y[va_abs],
            cov_train=cov[tr_abs, :],
            cov_test=cov[va_abs, :],
            alpha_grid=abcd_alpha_grid,
            inner_metric=inner_metric,
            seed=seed + 1000 * outer_fold + 10 * inner_fold + 1,
            inner_k=inner_k,
        )
        y_va_pc, p_va_pc, alpha_pc, info_pc = fit_predict_pc_krr(
            pc_train=pc[tr_abs, :],
            pc_test=pc[va_abs, :],
            y_train_raw=y[tr_abs],
            y_test_raw=y[va_abs],
            cov_train=cov[tr_abs, :],
            cov_test=cov[va_abs, :],
            alpha_grid=pc_alpha_grid,
            inner_metric=inner_metric,
            seed=seed + 1000 * outer_fold + 10 * inner_fold + 2,
            inner_k=inner_k,
        )
        max_y_diff = float(np.max(np.abs(y_va_abcd - y_va_pc)))
        if max_y_diff > float(y_tolerance):
            raise RuntimeError(
                f"Inner residualized y mismatch: outer={outer_fold}, inner={inner_fold}, max_diff={max_y_diff}"
            )
        pred_abcd[va_rel] = p_va_abcd
        pred_pc[va_rel] = p_va_pc
        y_inner[va_rel] = y_va_abcd
        for method, yy, pp, aa in [
            ("inner_abcd_base", y_va_abcd, p_va_abcd, alpha_abcd),
            ("inner_pc_base", y_va_pc, p_va_pc, alpha_pc),
        ]:
            m = compute_metrics(yy, pp)
            inner_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "method": method,
                    "n_train": int(len(tr_abs)),
                    "n_valid": int(len(va_abs)),
                    "alpha": float(aa),
                    **m,
                }
            )
        alpha_rows.append(
            {
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "stage": "inner_oof_base",
                "alpha_abcd": float(alpha_abcd),
                "alpha_pc": float(alpha_pc),
                "max_y_resid_diff_abcd_pc": max_y_diff,
                "abcd_alpha_scores": json.dumps(info_abcd["alpha_mean_scores"]),
                "pc_alpha_scores": json.dumps(info_pc["alpha_mean_scores"]),
            }
        )
        print(
            "[inner-oof] outer %d inner %d done: alpha_abcd=%s alpha_pc=%s"
            % (outer_fold, inner_fold, alpha_abcd, alpha_pc),
            flush=True,
        )
    if (
        not np.isfinite(pred_abcd).all()
        or not np.isfinite(pred_pc).all()
        or (not np.isfinite(y_inner).all())
    ):
        raise RuntimeError(f"Inner OOF generation failed for outer fold {outer_fold}.")
    return (y_inner, pred_abcd, pred_pc, inner_rows, alpha_rows)


def summarize_all_methods(
    pred_df: pd.DataFrame, prediction_cols: List[str]
) -> Tuple[pd.DataFrame, dict]:
    fold_rows = []
    pooled = {}
    y_all = pred_df["y_true_resid"].to_numpy(dtype=np.float64)
    for col in prediction_cols:
        pooled[col] = compute_metrics(y_all, pred_df[col].to_numpy(dtype=np.float64))
        for fold in sorted(pred_df["fold"].unique()):
            sub = pred_df[pred_df["fold"] == fold]
            m = compute_metrics(
                sub["y_true_resid"].to_numpy(dtype=np.float64),
                sub[col].to_numpy(dtype=np.float64),
            )
            fold_rows.append(
                {"method": col, "fold": int(fold), "n_test": int(len(sub)), **m}
            )
    fold_df = pd.DataFrame(fold_rows)
    across = {}
    for col in prediction_cols:
        sub = fold_df[fold_df["method"] == col]
        across[col] = {
            "R2_mean": float(sub["R2"].mean()),
            "R2_sd": float(sub["R2"].std(ddof=1)),
            "RMSE_mean": float(sub["RMSE"].mean()),
            "RMSE_sd": float(sub["RMSE"].std(ddof=1)),
            "MAE_mean": float(sub["MAE"].mean()),
            "MAE_sd": float(sub["MAE"].std(ddof=1)),
            "Pearson_r_mean": float(sub["Pearson_r"].mean()),
            "Pearson_r_sd": float(sub["Pearson_r"].std(ddof=1)),
            "COD_mean": float(sub["COD"].mean()),
            "COD_sd": float(sub["COD"].std(ddof=1)),
        }
    abcd_cod = pooled["y_pred_abcd_nested"]["COD"]
    abcd_r = pooled["y_pred_abcd_nested"]["Pearson_r"]
    for col in prediction_cols:
        pooled[col]["delta_COD_vs_abcd_nested"] = float(pooled[col]["COD"] - abcd_cod)
        pooled[col]["delta_Pearson_r_vs_abcd_nested"] = float(
            pooled[col]["Pearson_r"] - abcd_r
        )
        if abs(abcd_cod) > 0:
            pooled[col]["percent_COD_improvement_vs_abcd_nested"] = float(
                100.0 * (pooled[col]["COD"] - abcd_cod) / abs(abcd_cod)
            )
        else:
            pooled[col]["percent_COD_improvement_vs_abcd_nested"] = np.nan
    metrics = {"pooled": pooled, "across_fold_mean_sd": across}
    return (fold_df, metrics)


def run_nested_ensemble(
    abcd_csv: str,
    pred_parquet: str,
    freeze_tsv: str,
    fold_npz: str,
    fold_manifest_csv: str,
    out_dir: str,
    target_col: str,
    id_col: str,
    subjectkey_col: str,
    outer_k: int,
    inner_k: int,
    seed: int,
    abcd_alpha_grid: Sequence[float],
    pc_alpha_grid: Sequence[float],
    inner_metric: str,
    y_tolerance: float,
    log_stem: str,
    expected_covariates=None,
) -> None:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print("[config] Phase 4B KRR-source nested ensemble", flush=True)
    print(f"  abcd_csv={abcd_csv}", flush=True)
    print(f"  pred_parquet={pred_parquet}", flush=True)
    print(f"  freeze_tsv={freeze_tsv}", flush=True)
    print(f"  fold_npz={fold_npz}", flush=True)
    print(f"  fold_manifest_csv={fold_manifest_csv}", flush=True)
    print(f"  out_dir={out_dir}", flush=True)
    pc_data = load_stage2a_pc_predictions(
        pred_parquet=pred_parquet,
        freeze_tsv=freeze_tsv,
        id_col=id_col,
        subjectkey_col=subjectkey_col,
    )
    fold_manifest = load_fold_manifest(
        fold_manifest_csv=fold_manifest_csv,
        ids=pc_data["id"],
        subjectkeys=pc_data["subjectkey"],
        id_col=id_col,
        subjectkey_col=subjectkey_col,
    )
    fold_id = load_fullcv_folds(
        fold_npz=fold_npz,
        expected_n_subjects=len(fold_manifest),
        expected_outer_k=outer_k,
    )
    if not np.array_equal(fold_manifest["fold"].to_numpy(dtype=np.int64), fold_id):
        raise RuntimeError("fold npz fold_id does not match fold manifest.")
    abcd_data = load_abcd_locked_data(
        abcd_csv=abcd_csv,
        fold_manifest=fold_manifest,
        id_col=id_col,
        subjectkey_col=subjectkey_col,
        target_col=target_col,
    )
    if expected_covariates is not None and abcd_data["cov_cols"] != expected_covariates:
        raise RuntimeError(
            "Covariate columns/order differ from the reported configuration."
        )
    E = abcd_data["E"]
    y = abcd_data["y"]
    cov = abcd_data["cov"]
    pc = pc_data["pc"]
    meta = abcd_data["meta"].copy()
    if E.shape[0] != pc.shape[0] or E.shape[0] != len(fold_id):
        raise RuntimeError("N mismatch among ABCD edges, PC predictions, and fold ids.")
    n = E.shape[0]
    y_true_oof = np.full(n, np.nan, dtype=np.float64)
    pred_abcd_oof = np.full(n, np.nan, dtype=np.float64)
    pred_pc_oof = np.full(n, np.nan, dtype=np.float64)
    pred_ens_oof = np.full(n, np.nan, dtype=np.float64)
    pred_avg_oof = np.full(n, np.nan, dtype=np.float64)
    inner_metric_rows = []
    ensemble_weight_rows = []
    alpha_rows = []
    for outer_fold in range(int(outer_k)):
        print(
            "\n[outer] fold %d/%d starting at %.1fs"
            % (outer_fold + 1, outer_k, time.time() - t0),
            flush=True,
        )
        outer_train_idx = np.where(fold_id != outer_fold)[0]
        outer_test_idx = np.where(fold_id == outer_fold)[0]
        y_inner, pred_abcd_inner, pred_pc_inner, inner_rows, alpha_inner_rows = (
            generate_inner_oof_base_predictions(
                E=E,
                pc=pc,
                y=y,
                cov=cov,
                outer_train_idx=outer_train_idx,
                outer_fold=outer_fold,
                inner_k=inner_k,
                abcd_alpha_grid=abcd_alpha_grid,
                pc_alpha_grid=pc_alpha_grid,
                inner_metric=inner_metric,
                seed=seed + outer_fold,
                y_tolerance=y_tolerance,
            )
        )
        inner_metric_rows.extend(inner_rows)
        alpha_rows.extend(alpha_inner_rows)
        ens_model, ens_mu, ens_sd, ens_train_metrics = fit_ensemble_from_inner_oof(
            y_inner=y_inner,
            pred_abcd_inner=pred_abcd_inner,
            pred_pc_inner=pred_pc_inner,
        )
        y_test_abcd, p_test_abcd, alpha_abcd_final, info_abcd_final = (
            fit_predict_abcd_krr(
                E_train=E[outer_train_idx, :],
                E_test=E[outer_test_idx, :],
                y_train_raw=y[outer_train_idx],
                y_test_raw=y[outer_test_idx],
                cov_train=cov[outer_train_idx, :],
                cov_test=cov[outer_test_idx, :],
                alpha_grid=abcd_alpha_grid,
                inner_metric=inner_metric,
                seed=seed + 10000 + outer_fold,
                inner_k=inner_k,
            )
        )
        y_test_pc, p_test_pc, alpha_pc_final, info_pc_final = fit_predict_pc_krr(
            pc_train=pc[outer_train_idx, :],
            pc_test=pc[outer_test_idx, :],
            y_train_raw=y[outer_train_idx],
            y_test_raw=y[outer_test_idx],
            cov_train=cov[outer_train_idx, :],
            cov_test=cov[outer_test_idx, :],
            alpha_grid=pc_alpha_grid,
            inner_metric=inner_metric,
            seed=seed + 20000 + outer_fold,
            inner_k=inner_k,
        )
        max_y_diff = float(np.max(np.abs(y_test_abcd - y_test_pc)))
        if max_y_diff > float(y_tolerance):
            raise RuntimeError(
                f"Outer residualized y mismatch in fold {outer_fold}: {max_y_diff}"
            )
        X_test_ens_raw = np.column_stack([p_test_abcd, p_test_pc]).astype(np.float64)
        X_test_ens = (X_test_ens_raw - ens_mu) / ens_sd
        p_test_ens = np.asarray(ens_model.predict(X_test_ens), dtype=np.float64)
        p_test_avg = 0.5 * p_test_abcd + 0.5 * p_test_pc
        y_true_oof[outer_test_idx] = y_test_abcd
        pred_abcd_oof[outer_test_idx] = p_test_abcd
        pred_pc_oof[outer_test_idx] = p_test_pc
        pred_ens_oof[outer_test_idx] = p_test_ens
        pred_avg_oof[outer_test_idx] = p_test_avg
        fold_metrics_now = {
            "outer_fold": int(outer_fold),
            "n_train": int(len(outer_train_idx)),
            "n_test": int(len(outer_test_idx)),
            "alpha_abcd_final": float(alpha_abcd_final),
            "alpha_pc_final": float(alpha_pc_final),
            "max_y_resid_diff_final_abcd_pc": max_y_diff,
            "ensemble_intercept": float(ens_model.intercept_),
            "ensemble_coef_abcd_standardized": float(ens_model.coef_[0]),
            "ensemble_coef_pc_standardized": float(ens_model.coef_[1]),
            "ensemble_x_mu_abcd": float(ens_mu[0, 0]),
            "ensemble_x_mu_pc": float(ens_mu[0, 1]),
            "ensemble_x_sd_abcd": float(ens_sd[0, 0]),
            "ensemble_x_sd_pc": float(ens_sd[0, 1]),
        }
        for k, v in ens_train_metrics.items():
            fold_metrics_now[f"ensemble_inner_oof_{k}"] = float(v)
        ensemble_weight_rows.append(fold_metrics_now)
        alpha_rows.append(
            {
                "outer_fold": int(outer_fold),
                "inner_fold": -1,
                "stage": "outer_final_base",
                "alpha_abcd": float(alpha_abcd_final),
                "alpha_pc": float(alpha_pc_final),
                "max_y_resid_diff_abcd_pc": max_y_diff,
                "abcd_alpha_scores": json.dumps(info_abcd_final["alpha_mean_scores"]),
                "pc_alpha_scores": json.dumps(info_pc_final["alpha_mean_scores"]),
            }
        )
        tmp_npz = tmp_dir / f"{log_stem}_partial_after_outer{outer_fold}.npz"
        np.savez(
            tmp_npz,
            y_true_oof=y_true_oof,
            pred_abcd_oof=pred_abcd_oof,
            pred_pc_oof=pred_pc_oof,
            pred_ens_oof=pred_ens_oof,
            pred_avg_oof=pred_avg_oof,
            fold_id=fold_id,
            completed_outer_fold=np.int64(outer_fold),
        )
        m_abcd = compute_metrics(y_test_abcd, p_test_abcd)
        m_pc = compute_metrics(y_test_abcd, p_test_pc)
        m_ens = compute_metrics(y_test_abcd, p_test_ens)
        print(
            "[outer] fold %d done at %.1fs: ABCD COD=%.5f, PC COD=%.5f, ENS COD=%.5f"
            % (outer_fold, time.time() - t0, m_abcd["COD"], m_pc["COD"], m_ens["COD"]),
            flush=True,
        )
    for name, arr in [
        ("y_true_oof", y_true_oof),
        ("pred_abcd_oof", pred_abcd_oof),
        ("pred_pc_oof", pred_pc_oof),
        ("pred_ens_oof", pred_ens_oof),
        ("pred_avg_oof", pred_avg_oof),
    ]:
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{name} contains NaN/Inf after full run.")
    pred_df = meta.copy()
    pred_df["fold"] = fold_id
    pred_df["y_true_resid"] = y_true_oof
    pred_df["y_pred_abcd_nested"] = pred_abcd_oof
    pred_df["y_pred_pc_nested"] = pred_pc_oof
    pred_df["y_pred_avg_raw_nested"] = pred_avg_oof
    pred_df["y_pred_ensemble_nested"] = pred_ens_oof
    prediction_cols = [
        "y_pred_abcd_nested",
        "y_pred_pc_nested",
        "y_pred_avg_raw_nested",
        "y_pred_ensemble_nested",
    ]
    fold_df, metrics = summarize_all_methods(pred_df, prediction_cols)
    inner_df = pd.DataFrame(inner_metric_rows)
    weights_df = pd.DataFrame(ensemble_weight_rows)
    alpha_df = pd.DataFrame(alpha_rows)
    pred_csv = out_dir / f"predictions_concat_{log_stem}.csv"
    fold_csv = out_dir / f"fold_metrics_{log_stem}.csv"
    inner_csv = out_dir / f"inner_oof_metrics_{log_stem}.csv"
    weights_csv = out_dir / f"ensemble_weights_{log_stem}.csv"
    alpha_csv = out_dir / f"base_alpha_summary_{log_stem}.csv"
    result_npz = out_dir / f"{log_stem}_result.npz"
    metrics_json = out_dir / f"metrics_{log_stem}.json"
    metadata_json = out_dir / f"metadata_{log_stem}.json"
    pred_df.to_csv(pred_csv, index=False)
    fold_df.to_csv(fold_csv, index=False)
    inner_df.to_csv(inner_csv, index=False)
    weights_df.to_csv(weights_csv, index=False)
    alpha_df.to_csv(alpha_csv, index=False)
    np.savez(
        result_npz,
        y_true_resid=y_true_oof,
        y_pred_abcd_nested=pred_abcd_oof,
        y_pred_pc_nested=pred_pc_oof,
        y_pred_avg_raw_nested=pred_avg_oof,
        y_pred_ensemble_nested=pred_ens_oof,
        fold_id=fold_id,
        source_names=pc_data["source_names"],
    )
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    metadata = {
        "abcd_csv": str(abcd_csv),
        "pred_parquet": str(pred_parquet),
        "freeze_tsv": str(freeze_tsv),
        "fold_npz": str(fold_npz),
        "fold_manifest_csv": str(fold_manifest_csv),
        "out_dir": str(out_dir),
        "target_col": str(target_col),
        "id_col": str(id_col),
        "subjectkey_col": str(subjectkey_col),
        "outer_k": int(outer_k),
        "inner_k": int(inner_k),
        "seed": int(seed),
        "n_subjects": int(n),
        "n_edges": int(E.shape[1]),
        "n_transferred_pcs": int(pc.shape[1]),
        "source_names": [str(x) for x in pc_data["source_names"].tolist()],
        "covariate_columns": [str(x) for x in abcd_data["cov_cols"]],
        "abcd_alpha_grid": [float(a) for a in abcd_alpha_grid],
        "pc_alpha_grid": [float(a) for a in pc_alpha_grid],
        "inner_metric": str(inner_metric),
        "y_tolerance": float(y_tolerance),
        "method": "ABCD nested ensemble: RSFC KRR plus transferred-signature KRR",
        "source_predictor_family": "Mixed KRR phenotype-PC, DNN raw and KRR disease-PC source predictors",
        "pc_prediction_input_type": "Mean-five transferred predictions; source families specified by source_names",
        "design": "For each outer fold, ensemble weights are learned from inner-OOF base predictions inside the outer training fold only. Outer test fold is never used to learn ensemble weights.",
        "runtime_seconds": float(time.time() - t0),
        "covariate_missing_counts_before_imputation": abcd_data[
            "covariate_missing_counts"
        ],
    }
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("\n[done] ABCD nested ensemble completed.")
    print(f"[done] runtime seconds = {time.time() - t0:.1f}")
    print(f"[done] outputs = {out_dir}")
    print("[done] pooled metrics:")
    for method, m in metrics["pooled"].items():
        print(
            "  {method}: r={r:.6f}, COD={cod:.6f}, delta_COD={dcod:.6f}".format(
                method=method,
                r=m["Pearson_r"],
                cod=m["COD"],
                dcod=m["delta_COD_vs_abcd_nested"],
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--abcd-csv", required=True)
    parser.add_argument("--pred-parquet", required=True)
    parser.add_argument("--freeze-tsv", required=True)
    parser.add_argument("--fold-npz", required=True)
    parser.add_argument("--fold-manifest-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    feature_path = config_path.parent / config["source_features_file"]
    config["source_names"] = feature_path.read_text().splitlines()
    if len(config["source_names"]) != 43 or len(set(config["source_names"])) != 43:
        raise ValueError("Expected 43 unique source features in the feature list.")
    if read_freeze_list(args.freeze_tsv) != config["source_names"]:
        raise RuntimeError(
            "Transferred feature order differs from final43 configuration."
        )
    manifest = pd.read_csv(args.fold_manifest_csv)
    if len(manifest) != config["expected_n_subjects"]:
        raise RuntimeError("Locked cohort size differs from final43 configuration.")
    run_nested_ensemble(
        abcd_csv=args.abcd_csv,
        pred_parquet=args.pred_parquet,
        freeze_tsv=args.freeze_tsv,
        fold_npz=args.fold_npz,
        fold_manifest_csv=args.fold_manifest_csv,
        out_dir=args.out_dir,
        target_col=config["target_col"],
        id_col=config["id_col"],
        subjectkey_col=config["subjectkey_col"],
        outer_k=config["outer_k"],
        inner_k=config["inner_k"],
        seed=config["seed"],
        abcd_alpha_grid=config["abcd_alpha_grid"],
        pc_alpha_grid=config["pc_alpha_grid"],
        inner_metric=config["inner_metric"],
        y_tolerance=config["y_tolerance"],
        log_stem="abcd_final43_nested",
        expected_covariates=config["covariate_columns"],
    )


if __name__ == "__main__":
    main()
