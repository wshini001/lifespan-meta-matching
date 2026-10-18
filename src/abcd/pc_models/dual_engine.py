"""Fit a three-stream ABCD ensemble using RSFC, KRR-PC and DNN-PC predictions."""

from __future__ import annotations
from nested_engine import compute_metrics as compute_metrics

import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def read_freeze_list(path: str) -> List[str]:
    cols: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            cols.append(s.split("\t")[0])
    if cols and cols[0].lower() == "col":
        cols = cols[1:]
    if not cols:
        raise RuntimeError(f"Freeze list is empty: {path}")
    return cols


def parse_float_list(s: str, name: str = "float_list") -> Tuple[float, ...]:
    vals = tuple((float(x.strip()) for x in str(s).split(",") if x.strip()))
    if not vals:
        raise ValueError(f"{name} is empty: {s}")
    return vals


def score_for_alpha(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
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
    raise ValueError(f"Unknown inner_metric: {metric}")


def unitize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X - np.mean(X, axis=1, keepdims=True)
    norm = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
    norm[norm == 0.0] = 1.0
    return X / norm


def column_zscore_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    mu = np.mean(X_train, axis=0, keepdims=True)
    sd = np.std(X_train, axis=0, keepdims=True)
    sd[~np.isfinite(sd)] = 1.0
    sd[sd == 0.0] = 1.0
    return ((X_train - mu) / sd, (X_test - mu) / sd)


def transform_pc_train_test(
    X_train: np.ndarray, X_test: np.ndarray, mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    mode = str(mode)
    if mode == "row_unitize":
        if X_train.shape[1] < 2:
            return column_zscore_train_test(X_train, X_test)
        return (unitize_rows(X_train), unitize_rows(X_test))
    if mode == "column_zscore":
        return column_zscore_train_test(X_train, X_test)
    raise ValueError(f"Unknown pc_feature_mode: {mode}")


def make_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = s.astype(str).str.strip().str.lower()
    mapper = {"m": 1.0, "male": 1.0, "1": 1.0, "f": 0.0, "female": 0.0, "0": 0.0}
    if x.isin(list(mapper.keys())).mean() > 0.8:
        return x.map(mapper).astype(float)
    codes, _ = pd.factorize(s.astype(str))
    return pd.Series(codes, index=s.index).replace(-1, np.nan).astype(float)


def find_col(
    df: pd.DataFrame, candidates: Sequence[str], label: str, required: bool = True
) -> str | None:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(
            f"Cannot find required column for {label}. Candidates={candidates}. Available first 100 columns={cols[:100]}"
        )
    return None


def residualize_train_test(
    y_train: np.ndarray, y_test: np.ndarray, C_train: np.ndarray, C_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    lr = LinearRegression()
    lr.fit(C_train, y_train)
    return (y_train - lr.predict(C_train), y_test - lr.predict(C_test))


def load_pc_predictions(
    pred_parquet: str,
    freeze_tsv: str,
    fold_manifest: pd.DataFrame,
    subjectkey_col: str,
    id_col: str,
    label: str,
) -> Tuple[np.ndarray, List[str]]:
    pc_names = read_freeze_list(freeze_tsv)
    z_cols = [f"{p}__zpred" for p in pc_names]
    df = pd.read_parquet(pred_parquet)
    missing = [c for c in [subjectkey_col, id_col] if c not in df.columns]
    if missing:
        raise KeyError(f"{label} PC parquet missing ID columns: {missing}")
    missing_z = [c for c in z_cols if c not in df.columns]
    if missing_z:
        raise KeyError(f"{label} PC parquet missing zpred columns: {missing_z[:10]}")
    small = df[[subjectkey_col, id_col] + z_cols].copy()
    small[subjectkey_col] = small[subjectkey_col].astype(str)
    small[id_col] = small[id_col].astype(str)
    if small.duplicated([subjectkey_col, id_col]).any():
        dup = int(small.duplicated([subjectkey_col, id_col]).sum())
        raise RuntimeError(f"{label} PC parquet has duplicated subjects: {dup}")
    manifest = fold_manifest[[subjectkey_col, id_col]].copy()
    manifest[subjectkey_col] = manifest[subjectkey_col].astype(str)
    manifest[id_col] = manifest[id_col].astype(str)
    aligned = manifest.merge(
        small, on=[subjectkey_col, id_col], how="left", validate="one_to_one"
    )
    if len(aligned) != len(manifest):
        raise RuntimeError(f"{label} PC alignment length mismatch.")
    if aligned[z_cols].isna().any().any():
        missing_n = int(aligned[z_cols].isna().any(axis=1).sum())
        raise RuntimeError(
            f"{label} PC predictions missing after alignment for {missing_n} subjects."
        )
    X = aligned[z_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise RuntimeError(f"{label} PC matrix contains NaN/Inf after alignment.")
    return (X, pc_names)


def load_fold_manifest(
    fold_manifest_csv: str, subjectkey_col: str, id_col: str, fold_col: str
) -> pd.DataFrame:
    manifest = pd.read_csv(fold_manifest_csv)
    required = ["_orig_row_idx", subjectkey_col, id_col, fold_col]
    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise KeyError(f"Fold manifest missing required columns: {missing}")
    manifest = manifest.copy()
    manifest[subjectkey_col] = manifest[subjectkey_col].astype(str)
    manifest[id_col] = manifest[id_col].astype(str)
    manifest[fold_col] = manifest[fold_col].astype(int)
    manifest["_orig_row_idx"] = pd.to_numeric(
        manifest["_orig_row_idx"], errors="raise"
    ).astype(int)
    if manifest.duplicated([subjectkey_col, id_col]).any():
        dup = int(manifest.duplicated([subjectkey_col, id_col]).sum())
        raise RuntimeError(f"Fold manifest has duplicated subjects: {dup}")
    return manifest


def load_fold_npz(fold_npz: str, expected_n: int, outer_k: int) -> np.ndarray:
    z = np.load(fold_npz, allow_pickle=True)
    if "fold_id" not in z:
        raise KeyError(f"fold_npz missing fold_id: {fold_npz}")
    fold_id = np.asarray(z["fold_id"], dtype=np.int64)
    if len(fold_id) != int(expected_n):
        raise RuntimeError(
            f"fold_id length mismatch: {len(fold_id)} vs expected {expected_n}"
        )
    unique = sorted(np.unique(fold_id).tolist())
    if unique != list(range(int(outer_k))):
        raise RuntimeError(
            f"Unexpected fold labels in fold_npz: {unique}, expected {list(range(int(outer_k)))}"
        )
    return fold_id


def load_abcd_locked_data(
    abcd_csv: str,
    fold_manifest: pd.DataFrame,
    subjectkey_col: str,
    id_col: str,
    fold_col: str,
    target_col: str,
    site_col: str,
    age_col: str,
    sex_col: str,
    edu_col: str,
    motion_cols: str,
) -> Dict[str, object]:
    df_all = pd.read_csv(abcd_csv)
    df_all["_orig_row_idx"] = np.arange(len(df_all), dtype=np.int64)
    orig_idx = fold_manifest["_orig_row_idx"].to_numpy(dtype=np.int64)
    if np.any(orig_idx < 0) or np.any(orig_idx >= len(df_all)):
        raise RuntimeError(
            "_orig_row_idx in fold manifest is outside ABCD_final.csv row range."
        )
    if len(np.unique(orig_idx)) != len(orig_idx):
        raise RuntimeError("Fold manifest contains duplicated _orig_row_idx values.")
    df = df_all.iloc[orig_idx].copy().reset_index(drop=True)
    for c in [subjectkey_col, id_col]:
        if c not in df.columns:
            raise KeyError(f"ABCD csv missing ID column: {c}")
        df[c] = df[c].astype(str)
    if not np.array_equal(
        df[subjectkey_col].to_numpy(),
        fold_manifest[subjectkey_col].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "ABCD rows selected by _orig_row_idx do not match manifest subjectkey order."
        )
    if not np.array_equal(
        df[id_col].to_numpy(), fold_manifest[id_col].astype(str).to_numpy()
    ):
        raise RuntimeError(
            "ABCD rows selected by _orig_row_idx do not match manifest id order."
        )
    edge_cols = [f"e{i}" for i in range(1, 1486)]
    missing_edges = [c for c in edge_cols if c not in df.columns]
    if missing_edges:
        raise KeyError(f"ABCD csv missing RSFC edge columns: {missing_edges[:10]}")
    E = df[edge_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(E).all():
        raise RuntimeError("ABCD RSFC edge matrix contains NaN/Inf in locked rows.")
    if target_col not in df.columns:
        raise KeyError(f"ABCD csv missing target column: {target_col}")
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        bad = int((~np.isfinite(y)).sum())
        raise RuntimeError(f"Target contains non-finite values in locked rows: {bad}")
    resolved_age = (
        age_col
        if age_col
        else find_col(
            df,
            ["Age_years", "Age", "age", "interview_age", "interview_age_years"],
            "age",
        )
    )
    resolved_sex = (
        sex_col if sex_col else find_col(df, ["Sex", "sex", "gender", "sex_num"], "sex")
    )
    resolved_site = (
        site_col
        if site_col
        else find_col(
            df, ["site", "Site", "site_id", "site_id_l", "scanner_site"], "site"
        )
    )
    resolved_edu = (
        edu_col
        if edu_col
        else find_col(
            df,
            [
                "Edu",
                "edu",
                "education",
                "parent_education",
                "highest_parent_education",
                "demo_prnt_ed_v2",
            ],
            "education",
            required=False,
        )
    )
    baseline_motion_cols = [
        "iqc_rsfmri_1_mean_motion",
        "iqc_rsfmri_2_mean_motion",
        "iqc_rsfmri_3_mean_motion",
        "iqc_rsfmri_4_mean_motion",
    ]
    if motion_cols:
        resolved_motion = [c.strip() for c in motion_cols.split(",") if c.strip()]
        missing_motion = [c for c in resolved_motion if c not in df.columns]
        if missing_motion:
            raise KeyError(f"Motion columns not found: {missing_motion}")
    elif all((c in df.columns for c in baseline_motion_cols)):
        resolved_motion = baseline_motion_cols
    else:
        candidates = []
        for c in df.columns:
            low = c.lower()
            if any(
                (
                    p in low
                    for p in [
                        "motion",
                        "mean_fd",
                        "meanfd",
                        "numtrs",
                        "num_tr",
                        "rsfmri",
                    ]
                )
            ):
                if c not in edge_cols and c not in [subjectkey_col, id_col]:
                    if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8:
                        candidates.append(c)
        resolved_motion = candidates[:4]
        if len(resolved_motion) < 1:
            raise KeyError(
                "Could not autodetect motion covariates; pass --motion_cols explicitly."
            )
    cov = pd.DataFrame(index=df.index)
    cov["Age_years"] = pd.to_numeric(df[resolved_age], errors="coerce").astype(float)
    cov["Sex"] = make_numeric_series(df[resolved_sex])
    for c in resolved_motion:
        cov[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    if resolved_edu is not None:
        cov["Edu"] = make_numeric_series(df[resolved_edu])
    site = df[resolved_site].astype(str).fillna("missing")
    site_dummies = pd.get_dummies(site, prefix="site", drop_first=True, dtype=float)
    cov = pd.concat([cov, site_dummies], axis=1)
    cov["age_sq"] = cov["Age_years"] ** 2
    cov["age_sex"] = cov["Age_years"] * cov["Sex"]
    cov["age_sq_sex"] = cov["Age_years"] ** 2 * cov["Sex"]
    cov = cov.apply(pd.to_numeric, errors="coerce").astype(float)
    for c in cov.columns:
        if cov[c].isna().any():
            med = cov[c].median()
            if not np.isfinite(med):
                med = 0.0
            cov[c] = cov[c].fillna(med)
    C = cov.to_numpy(dtype=np.float64)
    if not np.isfinite(C).all():
        raise RuntimeError("Covariate matrix contains NaN/Inf.")
    meta_cols = ["_orig_row_idx", subjectkey_col, id_col]
    for c in ["interview_date", "age", "Age_years", "Sex", "site", target_col]:
        if c in df.columns and c not in meta_cols:
            meta_cols.append(c)
    meta = df[meta_cols].copy()
    meta[fold_col] = fold_manifest[fold_col].to_numpy(dtype=int)
    return {
        "meta": meta,
        "E": E,
        "y": y,
        "cov": C,
        "cov_cols": list(cov.columns),
        "edge_cols": edge_cols,
    }


def fit_predict_abcd_krr(
    E_train: np.ndarray,
    E_test: np.ndarray,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    C_train: np.ndarray,
    C_test: np.ndarray,
    alpha_grid: Sequence[float],
    inner_metric: str,
    seed: int,
    inner_k: int,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, object]]:
    n_train = len(y_train_raw)
    cv = KFold(
        n_splits=min(int(inner_k), n_train), shuffle=True, random_state=int(seed)
    )
    alpha_scores = []
    alpha_fold_scores: Dict[str, List[float]] = {}
    for alpha in alpha_grid:
        scores = []
        for tr_idx, va_idx in cv.split(np.arange(n_train)):
            y_tr_res, y_va_res = residualize_train_test(
                y_train_raw[tr_idx],
                y_train_raw[va_idx],
                C_train[tr_idx],
                C_train[va_idx],
            )
            Z_tr = unitize_rows(E_train[tr_idx])
            Z_va = unitize_rows(E_train[va_idx])
            K_tr = Z_tr @ Z_tr.T
            K_va = Z_va @ Z_tr.T
            model = KernelRidge(alpha=float(alpha), kernel="precomputed")
            model.fit(K_tr, y_tr_res)
            pred_va = np.asarray(model.predict(K_va), dtype=np.float64)
            scores.append(score_for_alpha(y_va_res, pred_va, inner_metric))
        alpha_scores.append(float(np.nanmean(scores)))
        alpha_fold_scores[str(float(alpha))] = [float(x) for x in scores]
    best_i = int(np.nanargmax(np.asarray(alpha_scores, dtype=np.float64)))
    best_alpha = float(list(alpha_grid)[best_i])
    y_train_res, y_test_res = residualize_train_test(
        y_train_raw, y_test_raw, C_train, C_test
    )
    Z_train = unitize_rows(E_train)
    Z_test = unitize_rows(E_test)
    K_train = Z_train @ Z_train.T
    K_test = Z_test @ Z_train.T
    model = KernelRidge(alpha=best_alpha, kernel="precomputed")
    model.fit(K_train, y_train_res)
    pred_test = np.asarray(model.predict(K_test), dtype=np.float64)
    info = {
        "best_alpha": best_alpha,
        "alpha_grid": [float(a) for a in alpha_grid],
        "alpha_mean_scores": [float(x) for x in alpha_scores],
        "alpha_fold_scores": alpha_fold_scores,
    }
    return (y_test_res, pred_test, best_alpha, info)


def fit_predict_pc_krr(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    C_train: np.ndarray,
    C_test: np.ndarray,
    alpha_grid: Sequence[float],
    inner_metric: str,
    seed: int,
    inner_k: int,
    pc_feature_mode: str,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, object]]:
    n_train = len(y_train_raw)
    X_all, X_te = transform_pc_train_test(X_train, X_test, mode=pc_feature_mode)
    cv = KFold(
        n_splits=min(int(inner_k), n_train), shuffle=True, random_state=int(seed)
    )
    alpha_scores = []
    alpha_fold_scores: Dict[str, List[float]] = {}
    for alpha in alpha_grid:
        scores = []
        for tr_idx, va_idx in cv.split(np.arange(n_train)):
            y_tr_res, y_va_res = residualize_train_test(
                y_train_raw[tr_idx],
                y_train_raw[va_idx],
                C_train[tr_idx],
                C_train[va_idx],
            )
            model = KernelRidge(alpha=float(alpha))
            model.fit(X_all[tr_idx], y_tr_res)
            pred_va = np.asarray(model.predict(X_all[va_idx]), dtype=np.float64)
            scores.append(score_for_alpha(y_va_res, pred_va, inner_metric))
        alpha_scores.append(float(np.nanmean(scores)))
        alpha_fold_scores[str(float(alpha))] = [float(x) for x in scores]
    best_i = int(np.nanargmax(np.asarray(alpha_scores, dtype=np.float64)))
    best_alpha = float(list(alpha_grid)[best_i])
    y_train_res, y_test_res = residualize_train_test(
        y_train_raw, y_test_raw, C_train, C_test
    )
    model = KernelRidge(alpha=best_alpha)
    model.fit(X_all, y_train_res)
    pred_test = np.asarray(model.predict(X_te), dtype=np.float64)
    info = {
        "best_alpha": best_alpha,
        "alpha_grid": [float(a) for a in alpha_grid],
        "alpha_mean_scores": [float(x) for x in alpha_scores],
        "alpha_fold_scores": alpha_fold_scores,
        "pc_feature_mode": pc_feature_mode,
    }
    return (y_test_res, pred_test, best_alpha, info)


def generate_inner_oof_predictions(
    E: np.ndarray,
    X_dnn: np.ndarray,
    X_krr: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    outer_train_idx: np.ndarray,
    outer_fold: int,
    inner_k: int,
    abcd_alpha_grid: Sequence[float],
    pc_alpha_grid: Sequence[float],
    inner_metric: str,
    seed: int,
    pc_feature_mode: str,
    y_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], List[dict]]:
    n = len(outer_train_idx)
    y_inner = np.full(n, np.nan, dtype=np.float64)
    p_abcd = np.full(n, np.nan, dtype=np.float64)
    p_dnn = np.full(n, np.nan, dtype=np.float64)
    p_krr = np.full(n, np.nan, dtype=np.float64)
    inner_rows: List[dict] = []
    alpha_rows: List[dict] = []
    cv = KFold(n_splits=int(inner_k), shuffle=True, random_state=int(seed))
    for inner_fold, (tr_rel, va_rel) in enumerate(cv.split(np.arange(n))):
        tr_abs = outer_train_idx[tr_rel]
        va_abs = outer_train_idx[va_rel]
        y_abcd, pred_abcd, alpha_abcd, info_abcd = fit_predict_abcd_krr(
            E_train=E[tr_abs],
            E_test=E[va_abs],
            y_train_raw=y[tr_abs],
            y_test_raw=y[va_abs],
            C_train=C[tr_abs],
            C_test=C[va_abs],
            alpha_grid=abcd_alpha_grid,
            inner_metric=inner_metric,
            seed=seed + 1000 * outer_fold + 10 * inner_fold + 1,
            inner_k=inner_k,
        )
        y_dnn, pred_dnn, alpha_dnn, info_dnn = fit_predict_pc_krr(
            X_train=X_dnn[tr_abs],
            X_test=X_dnn[va_abs],
            y_train_raw=y[tr_abs],
            y_test_raw=y[va_abs],
            C_train=C[tr_abs],
            C_test=C[va_abs],
            alpha_grid=pc_alpha_grid,
            inner_metric=inner_metric,
            seed=seed + 1000 * outer_fold + 10 * inner_fold + 2,
            inner_k=inner_k,
            pc_feature_mode=pc_feature_mode,
        )
        y_krr, pred_krr, alpha_krr, info_krr = fit_predict_pc_krr(
            X_train=X_krr[tr_abs],
            X_test=X_krr[va_abs],
            y_train_raw=y[tr_abs],
            y_test_raw=y[va_abs],
            C_train=C[tr_abs],
            C_test=C[va_abs],
            alpha_grid=pc_alpha_grid,
            inner_metric=inner_metric,
            seed=seed + 1000 * outer_fold + 10 * inner_fold + 3,
            inner_k=inner_k,
            pc_feature_mode=pc_feature_mode,
        )
        max_y_diff = max(
            float(np.max(np.abs(y_abcd - y_dnn))), float(np.max(np.abs(y_abcd - y_krr)))
        )
        if max_y_diff > float(y_tolerance):
            raise RuntimeError(
                f"Inner residualized target mismatch outer={outer_fold}, inner={inner_fold}: max_y_diff={max_y_diff}"
            )
        y_inner[va_rel] = y_abcd
        p_abcd[va_rel] = pred_abcd
        p_dnn[va_rel] = pred_dnn
        p_krr[va_rel] = pred_krr
        for method, yy, pp, aa in [
            ("inner_abcd_base", y_abcd, pred_abcd, alpha_abcd),
            ("inner_dnn_pc_base", y_dnn, pred_dnn, alpha_dnn),
            ("inner_krr_pc_base", y_krr, pred_krr, alpha_krr),
        ]:
            m = compute_metrics(yy, pp)
            inner_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "method": method,
                    "best_alpha": float(aa),
                    "n": int(len(yy)),
                    **m,
                }
            )
        for method, info in [
            ("abcd_base", info_abcd),
            ("dnn_pc_base", info_dnn),
            ("krr_pc_base", info_krr),
        ]:
            for a, s in zip(info["alpha_grid"], info["alpha_mean_scores"]):
                alpha_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(inner_fold),
                        "method": method,
                        "alpha": float(a),
                        "mean_inner_score_nested_search": float(s),
                        "selected_alpha": float(info["best_alpha"]),
                        "inner_metric": str(inner_metric),
                    }
                )
        print(
            f"[inner-oof] outer {outer_fold} inner {inner_fold} done: alpha_abcd={alpha_abcd} alpha_dnn={alpha_dnn} alpha_krr={alpha_krr}",
            flush=True,
        )
    if not (
        np.isfinite(y_inner).all()
        and np.isfinite(p_abcd).all()
        and np.isfinite(p_dnn).all()
        and np.isfinite(p_krr).all()
    ):
        raise RuntimeError("Inner OOF arrays contain missing values after inner CV.")
    return (y_inner, p_abcd, p_dnn, p_krr, inner_rows, alpha_rows)


def fit_linear_ensemble(
    y: np.ndarray, pred_dict: Dict[str, np.ndarray], feature_names: Sequence[str]
) -> Tuple[LinearRegression, np.ndarray, np.ndarray, Dict[str, float]]:
    X_raw = np.column_stack(
        [np.asarray(pred_dict[name], dtype=np.float64) for name in feature_names]
    )
    y = np.asarray(y, dtype=np.float64).ravel()
    if not np.isfinite(X_raw).all() or not np.isfinite(y).all():
        raise RuntimeError("Ensemble input contains NaN/Inf.")
    x_mu = X_raw.mean(axis=0, keepdims=True)
    x_sd = X_raw.std(axis=0, keepdims=True)
    x_sd[~np.isfinite(x_sd)] = 1.0
    x_sd[x_sd == 0.0] = 1.0
    X = (X_raw - x_mu) / x_sd
    lr = LinearRegression()
    lr.fit(X, y)
    pred = lr.predict(X)
    m = compute_metrics(y, pred)
    m["intercept"] = float(lr.intercept_)
    for i, name in enumerate(feature_names):
        m[f"coef_{name}_standardized"] = float(lr.coef_[i])
        m[f"x_mu_{name}"] = float(x_mu[0, i])
        m[f"x_sd_{name}"] = float(x_sd[0, i])
    return (lr, x_mu, x_sd, m)


def apply_linear_ensemble(
    lr: LinearRegression,
    x_mu: np.ndarray,
    x_sd: np.ndarray,
    pred_dict: Dict[str, np.ndarray],
    feature_names: Sequence[str],
) -> np.ndarray:
    X_raw = np.column_stack(
        [np.asarray(pred_dict[name], dtype=np.float64) for name in feature_names]
    )
    X = (X_raw - x_mu) / x_sd
    return np.asarray(lr.predict(X), dtype=np.float64)


def add_ensemble_weight_row(
    rows: List[dict],
    outer_fold: int,
    ensemble_name: str,
    feature_names: Sequence[str],
    train_metrics: Dict[str, float],
) -> None:
    row = {
        "outer_fold": int(outer_fold),
        "ensemble_name": str(ensemble_name),
        "n_features": int(len(feature_names)),
        "feature_names": ",".join(feature_names),
        "inner_oof_R2": float(train_metrics["R2"]),
        "inner_oof_RMSE": float(train_metrics["RMSE"]),
        "inner_oof_MAE": float(train_metrics["MAE"]),
        "inner_oof_Pearson_r": float(train_metrics["Pearson_r"]),
        "inner_oof_COD": float(train_metrics["COD"]),
        "intercept": float(train_metrics["intercept"]),
    }
    for name in feature_names:
        for suffix in ["standardized", ""]:
            key = f"coef_{name}_{suffix}".rstrip("_")
            if key in train_metrics:
                row[key] = float(train_metrics[key])
        if f"coef_{name}_standardized" in train_metrics:
            row[f"coef_{name}_standardized"] = float(
                train_metrics[f"coef_{name}_standardized"]
            )
        if f"x_mu_{name}" in train_metrics:
            row[f"x_mu_{name}"] = float(train_metrics[f"x_mu_{name}"])
        if f"x_sd_{name}" in train_metrics:
            row[f"x_sd_{name}"] = float(train_metrics[f"x_sd_{name}"])
    rows.append(row)


def run_dual_source_nested(
    abcd_csv: str,
    dnn_pred_parquet: str,
    krr_pred_parquet: str,
    freeze_tsv: str,
    fold_npz: str,
    fold_manifest_csv: str,
    out_dir: str,
    target_col: str,
    subjectkey_col: str,
    id_col: str,
    fold_col: str,
    outer_k: int,
    inner_k: int,
    seed: int,
    abcd_alpha_grid: Sequence[float],
    pc_alpha_grid: Sequence[float],
    inner_metric: str,
    y_tolerance: float,
    pc_feature_mode: str,
    site_col: str,
    age_col: str,
    sex_col: str,
    edu_col: str,
    motion_cols: str,
    log_stem: str,
    expected_covariates=None,
) -> None:
    t0 = time.time()
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    print("[config] Dual-source nested ensemble")
    print(f"  abcd_csv={abcd_csv}")
    print(f"  dnn_pred_parquet={dnn_pred_parquet}")
    print(f"  krr_pred_parquet={krr_pred_parquet}")
    print(f"  freeze_tsv={freeze_tsv}")
    print(f"  fold_npz={fold_npz}")
    print(f"  fold_manifest_csv={fold_manifest_csv}")
    print(f"  out_dir={out_dir_p}")
    print(f"  pc_feature_mode={pc_feature_mode}")
    fold_manifest = load_fold_manifest(
        fold_manifest_csv, subjectkey_col, id_col, fold_col
    )
    folds_from_npz = load_fold_npz(
        fold_npz, expected_n=len(fold_manifest), outer_k=outer_k
    )
    if not np.array_equal(
        folds_from_npz, fold_manifest[fold_col].to_numpy(dtype=np.int64)
    ):
        raise RuntimeError(
            "fold_id in fold_npz does not match fold column in fold_manifest_csv."
        )
    X_dnn, dnn_names = load_pc_predictions(
        dnn_pred_parquet,
        freeze_tsv,
        fold_manifest,
        subjectkey_col,
        id_col,
        label="DNN-source",
    )
    X_krr, krr_names = load_pc_predictions(
        krr_pred_parquet,
        freeze_tsv,
        fold_manifest,
        subjectkey_col,
        id_col,
        label="KRR-source",
    )
    if list(dnn_names) != list(krr_names):
        raise RuntimeError(
            "DNN-source and KRR-source PC names differ after freeze_tsv loading."
        )
    abcd = load_abcd_locked_data(
        abcd_csv=abcd_csv,
        fold_manifest=fold_manifest,
        subjectkey_col=subjectkey_col,
        id_col=id_col,
        fold_col=fold_col,
        target_col=target_col,
        site_col=site_col,
        age_col=age_col,
        sex_col=sex_col,
        edu_col=edu_col,
        motion_cols=motion_cols,
    )
    if expected_covariates is not None and abcd["cov_cols"] != expected_covariates:
        raise ValueError("Covariate identities/order differ from the configuration")
    E = abcd["E"]
    y = abcd["y"]
    C = abcd["cov"]
    meta = abcd["meta"].copy()
    folds = fold_manifest[fold_col].to_numpy(dtype=np.int64)
    n = len(y)
    if (
        E.shape[0] != n
        or X_dnn.shape[0] != n
        or X_krr.shape[0] != n
        or (C.shape[0] != n)
    ):
        raise RuntimeError("Input matrix row count mismatch after loading/alignment.")
    unique_folds = sorted(np.unique(folds).tolist())
    if unique_folds != list(range(int(outer_k))):
        raise RuntimeError(f"Unexpected folds: {unique_folds}")
    pred_rows: List[pd.DataFrame] = []
    fold_metric_rows: List[dict] = []
    inner_rows_all: List[dict] = []
    weight_rows: List[dict] = []
    alpha_rows_all: List[dict] = []
    for outer_fold in unique_folds:
        print(
            f"\n[outer] fold {outer_fold + 1}/{outer_k} starting at {time.time() - t0:.1f}s",
            flush=True,
        )
        test_idx = np.where(folds == outer_fold)[0]
        train_idx = np.where(folds != outer_fold)[0]
        y_inner, p_in_abcd, p_in_dnn, p_in_krr, inner_rows, alpha_rows = (
            generate_inner_oof_predictions(
                E=E,
                X_dnn=X_dnn,
                X_krr=X_krr,
                y=y,
                C=C,
                outer_train_idx=train_idx,
                outer_fold=int(outer_fold),
                inner_k=inner_k,
                abcd_alpha_grid=abcd_alpha_grid,
                pc_alpha_grid=pc_alpha_grid,
                inner_metric=inner_metric,
                seed=seed,
                pc_feature_mode=pc_feature_mode,
                y_tolerance=y_tolerance,
            )
        )
        inner_rows_all.extend(inner_rows)
        alpha_rows_all.extend(alpha_rows)
        inner_pred_dict = {"abcd": p_in_abcd, "dnn_pc": p_in_dnn, "krr_pc": p_in_krr}
        ensemble_specs = {
            "ensemble_abcd_dnn": ["abcd", "dnn_pc"],
            "ensemble_abcd_krr": ["abcd", "krr_pc"],
            "ensemble_dnn_krr_pc": ["dnn_pc", "krr_pc"],
            "ensemble_three_stream": ["abcd", "dnn_pc", "krr_pc"],
        }
        ensemble_models = {}
        for ens_name, feat_names in ensemble_specs.items():
            lr, x_mu, x_sd, train_metrics = fit_linear_ensemble(
                y_inner, inner_pred_dict, feat_names
            )
            ensemble_models[ens_name] = (lr, x_mu, x_sd, feat_names)
            add_ensemble_weight_row(
                weight_rows, int(outer_fold), ens_name, feat_names, train_metrics
            )
            inner_rows_all.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": -1,
                    "method": f"inner_oof_{ens_name}",
                    "best_alpha": np.nan,
                    "n": int(len(y_inner)),
                    **compute_metrics(
                        y_inner,
                        apply_linear_ensemble(
                            lr, x_mu, x_sd, inner_pred_dict, feat_names
                        ),
                    ),
                }
            )
        y_test_abcd, pred_abcd, alpha_abcd, info_abcd = fit_predict_abcd_krr(
            E_train=E[train_idx],
            E_test=E[test_idx],
            y_train_raw=y[train_idx],
            y_test_raw=y[test_idx],
            C_train=C[train_idx],
            C_test=C[test_idx],
            alpha_grid=abcd_alpha_grid,
            inner_metric=inner_metric,
            seed=seed + 100000 + int(outer_fold) * 10 + 1,
            inner_k=inner_k,
        )
        y_test_dnn, pred_dnn, alpha_dnn, info_dnn = fit_predict_pc_krr(
            X_train=X_dnn[train_idx],
            X_test=X_dnn[test_idx],
            y_train_raw=y[train_idx],
            y_test_raw=y[test_idx],
            C_train=C[train_idx],
            C_test=C[test_idx],
            alpha_grid=pc_alpha_grid,
            inner_metric=inner_metric,
            seed=seed + 100000 + int(outer_fold) * 10 + 2,
            inner_k=inner_k,
            pc_feature_mode=pc_feature_mode,
        )
        y_test_krr, pred_krr, alpha_krr, info_krr = fit_predict_pc_krr(
            X_train=X_krr[train_idx],
            X_test=X_krr[test_idx],
            y_train_raw=y[train_idx],
            y_test_raw=y[test_idx],
            C_train=C[train_idx],
            C_test=C[test_idx],
            alpha_grid=pc_alpha_grid,
            inner_metric=inner_metric,
            seed=seed + 100000 + int(outer_fold) * 10 + 3,
            inner_k=inner_k,
            pc_feature_mode=pc_feature_mode,
        )
        max_y_diff = max(
            float(np.max(np.abs(y_test_abcd - y_test_dnn))),
            float(np.max(np.abs(y_test_abcd - y_test_krr))),
        )
        if max_y_diff > float(y_tolerance):
            raise RuntimeError(
                f"Outer residualized target mismatch outer={outer_fold}: max_y_diff={max_y_diff}"
            )
        test_pred_dict = {"abcd": pred_abcd, "dnn_pc": pred_dnn, "krr_pc": pred_krr}
        pred_ens_abcd_dnn = apply_linear_ensemble(
            *ensemble_models["ensemble_abcd_dnn"][:3],
            test_pred_dict,
            ensemble_models["ensemble_abcd_dnn"][3],
        )
        pred_ens_abcd_krr = apply_linear_ensemble(
            *ensemble_models["ensemble_abcd_krr"][:3],
            test_pred_dict,
            ensemble_models["ensemble_abcd_krr"][3],
        )
        pred_ens_dnn_krr_pc = apply_linear_ensemble(
            *ensemble_models["ensemble_dnn_krr_pc"][:3],
            test_pred_dict,
            ensemble_models["ensemble_dnn_krr_pc"][3],
        )
        pred_ens_three = apply_linear_ensemble(
            *ensemble_models["ensemble_three_stream"][:3],
            test_pred_dict,
            ensemble_models["ensemble_three_stream"][3],
        )
        pred_avg_all3 = (pred_abcd + pred_dnn + pred_krr) / 3.0
        pred_avg_dnn_krr_pc = (pred_dnn + pred_krr) / 2.0
        fold_meta = meta.iloc[test_idx].copy().reset_index(drop=True)
        fold_pred = fold_meta.copy()
        fold_pred["outer_fold"] = int(outer_fold)
        fold_pred["y_true_resid"] = y_test_abcd
        fold_pred["y_pred_abcd_nested"] = pred_abcd
        fold_pred["y_pred_dnn_pc_nested"] = pred_dnn
        fold_pred["y_pred_krr_pc_nested"] = pred_krr
        fold_pred["y_pred_avg_dnn_krr_pc_nested"] = pred_avg_dnn_krr_pc
        fold_pred["y_pred_avg_all3_nested"] = pred_avg_all3
        fold_pred["y_pred_ensemble_abcd_dnn_nested"] = pred_ens_abcd_dnn
        fold_pred["y_pred_ensemble_abcd_krr_nested"] = pred_ens_abcd_krr
        fold_pred["y_pred_ensemble_dnn_krr_pc_nested"] = pred_ens_dnn_krr_pc
        fold_pred["y_pred_ensemble_three_stream_nested"] = pred_ens_three
        pred_rows.append(fold_pred)
        method_preds = {
            "y_pred_abcd_nested": pred_abcd,
            "y_pred_dnn_pc_nested": pred_dnn,
            "y_pred_krr_pc_nested": pred_krr,
            "y_pred_avg_dnn_krr_pc_nested": pred_avg_dnn_krr_pc,
            "y_pred_avg_all3_nested": pred_avg_all3,
            "y_pred_ensemble_abcd_dnn_nested": pred_ens_abcd_dnn,
            "y_pred_ensemble_abcd_krr_nested": pred_ens_abcd_krr,
            "y_pred_ensemble_dnn_krr_pc_nested": pred_ens_dnn_krr_pc,
            "y_pred_ensemble_three_stream_nested": pred_ens_three,
        }
        for method, pred in method_preds.items():
            m = compute_metrics(y_test_abcd, pred)
            fold_metric_rows.append(
                {
                    "fold": int(outer_fold),
                    "method": method,
                    "n": int(len(test_idx)),
                    **m,
                }
            )
        for method, alpha, info in [
            ("abcd_base", alpha_abcd, info_abcd),
            ("dnn_pc_base", alpha_dnn, info_dnn),
            ("krr_pc_base", alpha_krr, info_krr),
        ]:
            for a, s in zip(info["alpha_grid"], info["alpha_mean_scores"]):
                alpha_rows_all.append(
                    {
                        "outer_fold": int(outer_fold),
                        "inner_fold": -1,
                        "method": method,
                        "alpha": float(a),
                        "mean_inner_score_final_outer_base": float(s),
                        "selected_alpha": float(alpha),
                        "inner_metric": str(inner_metric),
                    }
                )
        print(
            "[outer] fold {f} done at {t:.1f}s: ABCD COD={a:.5f}, DNN-PC COD={d:.5f}, KRR-PC COD={k:.5f}, three-stream COD={e:.5f}".format(
                f=int(outer_fold),
                t=time.time() - t0,
                a=compute_metrics(y_test_abcd, pred_abcd)["COD"],
                d=compute_metrics(y_test_abcd, pred_dnn)["COD"],
                k=compute_metrics(y_test_abcd, pred_krr)["COD"],
                e=compute_metrics(y_test_abcd, pred_ens_three)["COD"],
            ),
            flush=True,
        )
    pred_all = pd.concat(pred_rows, axis=0, ignore_index=True)
    pred_all["_tmp_order"] = pred_all["_orig_row_idx"].map(
        {
            int(v): i
            for i, v in enumerate(fold_manifest["_orig_row_idx"].to_numpy(dtype=int))
        }
    )
    pred_all = (
        pred_all.sort_values("_tmp_order")
        .drop(columns=["_tmp_order"])
        .reset_index(drop=True)
    )
    method_cols = [
        "y_pred_abcd_nested",
        "y_pred_dnn_pc_nested",
        "y_pred_krr_pc_nested",
        "y_pred_avg_dnn_krr_pc_nested",
        "y_pred_avg_all3_nested",
        "y_pred_ensemble_abcd_dnn_nested",
        "y_pred_ensemble_abcd_krr_nested",
        "y_pred_ensemble_dnn_krr_pc_nested",
        "y_pred_ensemble_three_stream_nested",
    ]
    pooled_metrics: Dict[str, dict] = {}
    y_all = pred_all["y_true_resid"].to_numpy(dtype=np.float64)
    base_metrics = compute_metrics(
        y_all, pred_all["y_pred_abcd_nested"].to_numpy(dtype=np.float64)
    )
    best_prev_cod = np.nanmax(
        [
            base_metrics["COD"],
            compute_metrics(
                y_all,
                pred_all["y_pred_ensemble_abcd_dnn_nested"].to_numpy(dtype=np.float64),
            )["COD"],
            compute_metrics(
                y_all,
                pred_all["y_pred_ensemble_abcd_krr_nested"].to_numpy(dtype=np.float64),
            )["COD"],
        ]
    )
    for method in method_cols:
        m = compute_metrics(y_all, pred_all[method].to_numpy(dtype=np.float64))
        m["delta_COD_vs_abcd_nested"] = float(m["COD"] - base_metrics["COD"])
        m["delta_Pearson_r_vs_abcd_nested"] = float(
            m["Pearson_r"] - base_metrics["Pearson_r"]
        )
        if abs(base_metrics["COD"]) > 0:
            m["percent_COD_improvement_vs_abcd_nested"] = float(
                (m["COD"] - base_metrics["COD"]) / abs(base_metrics["COD"]) * 100.0
            )
        else:
            m["percent_COD_improvement_vs_abcd_nested"] = np.nan
        m["delta_COD_vs_best_two_stream_reference"] = float(m["COD"] - best_prev_cod)
        pooled_metrics[method] = m
    fold_metrics = pd.DataFrame(fold_metric_rows)
    baseline_by_fold = (
        fold_metrics[fold_metrics["method"] == "y_pred_abcd_nested"]
        .set_index("fold")["COD"]
        .to_dict()
    )
    for i, row in fold_metrics.iterrows():
        fold_metrics.loc[i, "delta_COD_vs_abcd_nested"] = float(
            row["COD"] - baseline_by_fold[int(row["fold"])]
        )
    pred_csv = out_dir_p / f"predictions_concat_{log_stem}.csv"
    fold_csv = out_dir_p / f"fold_metrics_{log_stem}.csv"
    inner_csv = out_dir_p / f"inner_oof_metrics_{log_stem}.csv"
    weights_csv = out_dir_p / f"ensemble_weights_{log_stem}.csv"
    alpha_csv = out_dir_p / f"base_alpha_summary_{log_stem}.csv"
    metrics_json = out_dir_p / f"metrics_{log_stem}.json"
    metadata_json = out_dir_p / f"metadata_{log_stem}.json"
    pred_all.to_csv(pred_csv, index=False)
    fold_metrics.to_csv(fold_csv, index=False)
    pd.DataFrame(inner_rows_all).to_csv(inner_csv, index=False)
    pd.DataFrame(weight_rows).to_csv(weights_csv, index=False)
    pd.DataFrame(alpha_rows_all).to_csv(alpha_csv, index=False)
    metrics = {
        "pooled": pooled_metrics,
        "fold_count": {str(f): int(np.sum(folds == f)) for f in unique_folds},
        "best_previous_two_stream_reference_COD": float(best_prev_cod),
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    metadata = {
        "abcd_csv": str(abcd_csv),
        "dnn_pred_parquet": str(dnn_pred_parquet),
        "krr_pred_parquet": str(krr_pred_parquet),
        "freeze_tsv": str(freeze_tsv),
        "fold_npz": str(fold_npz),
        "fold_manifest_csv": str(fold_manifest_csv),
        "out_dir": str(out_dir_p),
        "target_col": str(target_col),
        "subjectkey_col": str(subjectkey_col),
        "id_col": str(id_col),
        "fold_col": str(fold_col),
        "outer_k": int(outer_k),
        "inner_k": int(inner_k),
        "seed": int(seed),
        "n_subjects": int(n),
        "n_edges": int(E.shape[1]),
        "n_transferred_pcs": int(X_dnn.shape[1]),
        "pc_names": [str(x) for x in dnn_names],
        "covariate_columns": [str(x) for x in abcd["cov_cols"]],
        "abcd_alpha_grid": [float(a) for a in abcd_alpha_grid],
        "pc_alpha_grid": [float(a) for a in pc_alpha_grid],
        "inner_metric": str(inner_metric),
        "pc_feature_mode": str(pc_feature_mode),
        "y_tolerance": float(y_tolerance),
        "method": "Dual-source nested ensemble: ABCD RSFC-only KRR + DNN-source PC KRR stacking + KRR-source PC KRR stacking",
        "design": "For each outer fold, all ensemble weights are learned from inner-OOF base predictions inside the outer training fold only. Outer test fold is never used to learn ensemble weights.",
        "runtime_seconds": float(time.time() - t0),
    }
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("\n[done] Dual-source nested ensemble completed.")
    print(f"[done] runtime seconds = {time.time() - t0:.1f}")
    print(f"[done] outputs = {out_dir_p}")
    print("[done] pooled metrics:")
    for method in method_cols:
        m = pooled_metrics[method]
        print(
            "  {method}: r={r:.6f}, COD={cod:.6f}, delta_COD={dcod:.6f}, delta_vs_best_prev={dbest:.6f}".format(
                method=method,
                r=m["Pearson_r"],
                cod=m["COD"],
                dcod=m["delta_COD_vs_abcd_nested"],
                dbest=m["delta_COD_vs_best_two_stream_reference"],
            )
        )
