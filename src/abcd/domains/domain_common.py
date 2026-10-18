"""Shared numerical helpers for ABCD domain-subset and multi-kernel analyses."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def apply_ensemble(ens, pred_abcd, pred_pc):
    X = np.column_stack([pred_abcd, pred_pc]).astype(np.float64)
    Xz = (X - ens["mu"]) / ens["sd"]
    return ens["model"].predict(Xz)


def bootstrap_delta(y, pred_base, pred_model, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        vals.append(
            cod_score(y[idx], pred_model[idx]) - cod_score(y[idx], pred_base[idx])
        )
    vals = np.asarray(vals, dtype=float)
    return {
        "boot_mean_delta_COD": float(np.nanmean(vals)),
        "boot_ci2p5_delta_COD": float(np.nanpercentile(vals, 2.5)),
        "boot_ci97p5_delta_COD": float(np.nanpercentile(vals, 97.5)),
    }


def cod_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst <= 0:
        return np.nan
    return 1.0 - sse / sst


def column_zscore_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    mu = np.mean(X_train, axis=0, keepdims=True)
    sd = np.std(X_train, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return ((X_train - mu) / sd, (X_test - mu) / sd)


def combine_kernels(K_e: np.ndarray, K_p: np.ndarray, weight_pc: float) -> np.ndarray:
    w = float(weight_pc)
    return (1.0 - w) * K_e + w * K_p


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return {
        "R2_sklearn": float(r2_score(y_true, y_pred)),
        "COD": float(cod_score(y_true, y_pred)),
        "Pearson_r": float(safe_corr(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def find_col(df: pd.DataFrame, candidates: List[str], name: str, required: bool = True):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(
            f"Cannot find required column for {name}. Candidates={candidates}. Available first 80 columns={cols[:80]}"
        )
    return None


def fit_ensemble_from_inner_oof(pred_abcd, pred_pc, y):
    X = np.column_stack([pred_abcd, pred_pc]).astype(np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    model = LinearRegression()
    model.fit(Xz, y)
    return {
        "model": model,
        "mu": mu,
        "sd": sd,
        "coef_abcd_standardized": float(model.coef_[0]),
        "coef_pc_standardized": float(model.coef_[1]),
        "intercept": float(model.intercept_),
    }


def fit_predict_abcd(E_train, y_train, E_test, alpha_grid, inner_k, seed, metric):
    best_alpha, best_score = select_alpha_abcd(
        E_train, y_train, alpha_grid, inner_k, seed, metric
    )
    Z_train = unitize_rows(E_train)
    Z_test = unitize_rows(E_test)
    K_train = Z_train @ Z_train.T
    K_test = Z_test @ Z_train.T
    model = KernelRidge(alpha=best_alpha, kernel="precomputed")
    model.fit(K_train, y_train)
    pred = model.predict(K_test)
    return (pred, best_alpha, best_score)


def fit_predict_pc(
    X_train, y_train, X_test, alpha_grid, inner_k, seed, metric, pc_feature_mode
):
    best_alpha, best_score = select_alpha_pc(
        X_train, y_train, alpha_grid, inner_k, seed, metric, pc_feature_mode
    )
    X_tr, X_te = transform_pc_train_test(X_train, X_test, pc_feature_mode)
    model = KernelRidge(alpha=best_alpha)
    model.fit(X_tr, y_train)
    pred = model.predict(X_te)
    return (pred, best_alpha, best_score)


def fit_predict_precomputed(
    K_train: np.ndarray, y_train: np.ndarray, K_test: np.ndarray, alpha: float
) -> np.ndarray:
    model = KernelRidge(alpha=float(alpha), kernel="precomputed")
    model.fit(K_train, y_train)
    return model.predict(K_test)


def generate_inner_oof_abcd(
    E, y_raw, C, outer_train_idx, inner_k, alpha_grid, seed, metric
):
    pred_oof = np.full(len(outer_train_idx), np.nan, dtype=np.float64)
    y_oof = np.full(len(outer_train_idx), np.nan, dtype=np.float64)
    alpha_rows = []
    kf = KFold(n_splits=inner_k, shuffle=True, random_state=seed)
    for inner_fold, (tr_rel, va_rel) in enumerate(kf.split(outer_train_idx)):
        tr_abs = outer_train_idx[tr_rel]
        va_abs = outer_train_idx[va_rel]
        y_tr_resid, y_va_resid = residualize_train_test(
            y_raw[tr_abs], y_raw[va_abs], C[tr_abs], C[va_abs]
        )
        pred, alpha, score = fit_predict_abcd(
            E[tr_abs],
            y_tr_resid,
            E[va_abs],
            alpha_grid,
            inner_k,
            seed + inner_fold + 11,
            metric,
        )
        pred_oof[va_rel] = pred
        y_oof[va_rel] = y_va_resid
        alpha_rows.append(
            {
                "inner_fold": inner_fold,
                "stream": "abcd",
                "alpha": alpha,
                "inner_score": score,
            }
        )
    if np.isnan(pred_oof).any() or np.isnan(y_oof).any():
        raise RuntimeError("ABCD inner OOF prediction has NaN")
    return (pred_oof, y_oof, alpha_rows)


def generate_inner_oof_pc(
    Xpc,
    y_raw,
    C,
    outer_train_idx,
    inner_k,
    alpha_grid,
    seed,
    metric,
    pc_feature_mode,
    subset_name,
):
    pred_oof = np.full(len(outer_train_idx), np.nan, dtype=np.float64)
    y_oof = np.full(len(outer_train_idx), np.nan, dtype=np.float64)
    alpha_rows = []
    kf = KFold(n_splits=inner_k, shuffle=True, random_state=seed)
    for inner_fold, (tr_rel, va_rel) in enumerate(kf.split(outer_train_idx)):
        tr_abs = outer_train_idx[tr_rel]
        va_abs = outer_train_idx[va_rel]
        y_tr_resid, y_va_resid = residualize_train_test(
            y_raw[tr_abs], y_raw[va_abs], C[tr_abs], C[va_abs]
        )
        pred, alpha, score = fit_predict_pc(
            Xpc[tr_abs],
            y_tr_resid,
            Xpc[va_abs],
            alpha_grid,
            inner_k,
            seed + inner_fold + 101,
            metric,
            pc_feature_mode,
        )
        pred_oof[va_rel] = pred
        y_oof[va_rel] = y_va_resid
        alpha_rows.append(
            {
                "inner_fold": inner_fold,
                "stream": "pc",
                "subset": subset_name,
                "alpha": alpha,
                "inner_score": score,
            }
        )
    if np.isnan(pred_oof).any() or np.isnan(y_oof).any():
        raise RuntimeError(f"PC inner OOF prediction has NaN for subset {subset_name}")
    return (pred_oof, y_oof, alpha_rows)


def load_locked_reference(
    locked_pred_csv: str, subjectkey_col: str, id_col: str, fold_col: str
) -> pd.DataFrame:
    df = pd.read_csv(locked_pred_csv)
    missing = [c for c in [subjectkey_col, id_col, fold_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Locked prediction CSV missing columns: {missing}. Columns={list(df.columns)}"
        )
    out = df[[subjectkey_col, id_col, fold_col]].copy()
    out[subjectkey_col] = out[subjectkey_col].astype(str)
    out[id_col] = out[id_col].astype(str)
    out[fold_col] = out[fold_col].astype(int)
    if out.duplicated([subjectkey_col, id_col]).any():
        raise RuntimeError("Locked prediction CSV has duplicated subjects")
    return out


def load_pc_predictions(
    pc_pred_parquet: str, freeze_tsv: str, subjectkey_col: str, id_col: str
) -> Tuple[pd.DataFrame, List[str]]:
    pc_names = read_freeze_list(freeze_tsv)
    df = pd.read_parquet(pc_pred_parquet)
    missing_id = [c for c in [subjectkey_col, id_col] if c not in df.columns]
    if missing_id:
        raise KeyError(f"PC prediction parquet missing ID columns: {missing_id}")
    z_cols = [f"{p}__zpred" for p in pc_names]
    missing_z = [c for c in z_cols if c not in df.columns]
    if missing_z:
        raise KeyError(f"PC prediction parquet missing zpred columns: {missing_z[:10]}")
    out = df[[subjectkey_col, id_col] + z_cols].copy()
    out[subjectkey_col] = out[subjectkey_col].astype(str)
    out[id_col] = out[id_col].astype(str)
    for c in z_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float64)
    if out.duplicated([subjectkey_col, id_col]).any():
        raise RuntimeError("PC prediction parquet has duplicated subjects")
    if out[z_cols].isna().any().any():
        raise RuntimeError("PC prediction parquet contains NaN in zpred columns")
    return (out, pc_names)


def make_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = s.astype(str).str.strip().str.lower()
    mapper = {"m": 1.0, "male": 1.0, "1": 1.0, "f": 0.0, "female": 0.0, "0": 0.0}
    if x.isin(list(mapper.keys())).mean() > 0.8:
        return x.map(mapper).astype(float)
    codes, _ = pd.factorize(s.astype(str))
    codes = pd.Series(codes, index=s.index).replace(-1, np.nan).astype(float)
    return codes


def parse_pc_block(pc_name: str) -> str:
    if "__pc" not in pc_name:
        return "unknown"
    return pc_name.split("__pc")[0]


def parse_weight_grid(s: str) -> List[float]:
    vals = parse_float_grid(s, name="weight grid")
    bad = [v for v in vals if v < 0.0 or v > 1.0]
    if bad:
        raise ValueError(f"Kernel weights must be in [0, 1]. Bad values: {bad}")
    vals = sorted(set((float(v) for v in vals)))
    if 0.0 not in vals:
        raise ValueError("weight_grid must include 0.0 for ABCD-only sanity check.")
    if 1.0 not in vals:
        raise ValueError("weight_grid must include 1.0 for PC-only sanity check.")
    return vals


def read_freeze_list(path: str) -> List[str]:
    cols = []
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


def residualize_train_test(
    y_train, y_test, C_train, C_test
) -> Tuple[np.ndarray, np.ndarray]:
    model = LinearRegression()
    model.fit(C_train, y_train)
    return (y_train - model.predict(C_train), y_test - model.predict(C_test))


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return np.nan
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def score_by_metric(y_true, y_pred, metric: str) -> float:
    if metric == "pearson":
        return safe_corr(y_true, y_pred)
    if metric == "cod":
        return cod_score(y_true, y_pred)
    raise ValueError(f"Unknown metric: {metric}")


def select_alpha_abcd(E_train, y_train, alpha_grid, inner_k, seed, metric):
    kf = KFold(n_splits=inner_k, shuffle=True, random_state=seed)
    scores = []
    for alpha in alpha_grid:
        fold_scores = []
        for tr, va in kf.split(E_train):
            Z_tr = unitize_rows(E_train[tr])
            Z_va = unitize_rows(E_train[va])
            K_tr = Z_tr @ Z_tr.T
            K_va = Z_va @ Z_tr.T
            model = KernelRidge(alpha=alpha, kernel="precomputed")
            model.fit(K_tr, y_train[tr])
            pred = model.predict(K_va)
            fold_scores.append(
                safe_corr(y_train[va], pred)
                if metric == "pearson"
                else cod_score(y_train[va], pred)
            )
        scores.append(float(np.nanmean(fold_scores)))
    best_idx = int(np.nanargmax(scores))
    return (float(alpha_grid[best_idx]), float(scores[best_idx]))


def select_alpha_pc(
    X_train, y_train, alpha_grid, inner_k, seed, metric, pc_feature_mode
):
    kf = KFold(n_splits=inner_k, shuffle=True, random_state=seed)
    scores = []
    for alpha in alpha_grid:
        fold_scores = []
        for tr, va in kf.split(X_train):
            X_tr, X_va = transform_pc_train_test(
                X_train[tr], X_train[va], pc_feature_mode
            )
            model = KernelRidge(alpha=alpha)
            model.fit(X_tr, y_train[tr])
            pred = model.predict(X_va)
            fold_scores.append(
                safe_corr(y_train[va], pred)
                if metric == "pearson"
                else cod_score(y_train[va], pred)
            )
        scores.append(float(np.nanmean(fold_scores)))
    best_idx = int(np.nanargmax(scores))
    return (float(alpha_grid[best_idx]), float(scores[best_idx]))


def transform_pc_train_test(
    X_train: np.ndarray, X_test: np.ndarray, mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    mode = str(mode)
    if mode == "column_zscore":
        return column_zscore_train_test(X_train, X_test)
    if mode == "row_unitize":
        if X_train.shape[1] < 2:
            return column_zscore_train_test(X_train, X_test)
        return (unitize_rows(X_train), unitize_rows(X_test))
    raise ValueError(f"Unknown pc_feature_mode: {mode}")


def unitize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X - np.mean(X, axis=1, keepdims=True)
    norm = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
    norm[norm == 0] = 1.0
    return X / norm


def parse_float_grid(s: str, name: str = "grid") -> List[float]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if not vals:
        raise ValueError(f"Empty {name}")
    return vals
