"""Participant bootstrap, paired-error tests and fixed-weight PC-stream permutation for ABCD."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        raise RuntimeError("No finite samples available for metric computation.")
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = np.nan
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    cod = float(1.0 - sse / sst) if sst > 0 else np.nan
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Pearson_r": pearson_r, "COD": cod}


def _json_float(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, float):
        return x
    return x


def _safe_percentile(x: np.ndarray, qs: List[float]) -> List[float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [np.nan for _ in qs]
    return [float(v) for v in np.percentile(x, qs)]


def load_predictions(
    pred_csv: str,
    subjectkey_col: str,
    id_col: str,
    fold_col: str,
    y_col: str,
    method_cols: List[str],
) -> pd.DataFrame:
    pred_csv = str(pred_csv)
    df = pd.read_csv(pred_csv)
    required = [subjectkey_col, id_col, fold_col, y_col] + method_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Prediction CSV missing required columns: {missing}; file={pred_csv}"
        )
    if df[[subjectkey_col, id_col]].astype(str).duplicated().any():
        n_dup = int(df[[subjectkey_col, id_col]].astype(str).duplicated().sum())
        raise RuntimeError(f"Duplicated subjects in prediction file: {n_dup}")
    for col in [fold_col, y_col] + method_cols:
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all():
            bad = int((~np.isfinite(arr)).sum())
            raise RuntimeError(
                f"Column {col} contains NaN/Inf after numeric conversion: {bad}"
            )
        df[col] = arr
    df[fold_col] = df[fold_col].astype(np.int64)
    folds = sorted(df[fold_col].unique().tolist())
    if folds != list(range(len(folds))):
        raise RuntimeError(f"Unexpected fold labels: {folds}")
    if len(folds) < 2:
        raise RuntimeError(f"Expected at least two folds, got {folds}")
    return df


def load_weights(weights_csv: str) -> pd.DataFrame:
    weights_csv = str(weights_csv)
    w = pd.read_csv(weights_csv)
    required = [
        "outer_fold",
        "ensemble_intercept",
        "ensemble_coef_abcd_standardized",
        "ensemble_coef_pc_standardized",
        "ensemble_x_mu_abcd",
        "ensemble_x_mu_pc",
        "ensemble_x_sd_abcd",
        "ensemble_x_sd_pc",
    ]
    missing = [c for c in required if c not in w.columns]
    if missing:
        raise KeyError(
            f"Ensemble weights CSV missing required columns: {missing}; file={weights_csv}"
        )
    if w["outer_fold"].duplicated().any():
        dup = w.loc[w["outer_fold"].duplicated(), "outer_fold"].tolist()
        raise RuntimeError(
            f"Ensemble weights file contains duplicated outer_fold rows: {dup}"
        )
    for c in required:
        arr = pd.to_numeric(w[c], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all():
            raise RuntimeError(f"Weight column {c} contains NaN/Inf.")
        w[c] = arr
    if np.any(w["ensemble_x_sd_abcd"].to_numpy(dtype=np.float64) <= 0):
        raise RuntimeError("ensemble_x_sd_abcd must be positive for all folds.")
    if np.any(w["ensemble_x_sd_pc"].to_numpy(dtype=np.float64) <= 0):
        raise RuntimeError("ensemble_x_sd_pc must be positive for all folds.")
    w["outer_fold"] = w["outer_fold"].astype(np.int64)
    return w


def summarize_methods(
    df: pd.DataFrame,
    fold_col: str,
    y_col: str,
    method_cols: List[str],
    baseline_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, dict]]:
    y = df[y_col].to_numpy(dtype=np.float64)
    pooled_rows = []
    pooled_dict = {}
    baseline_metrics = compute_metrics(y, df[baseline_col].to_numpy(dtype=np.float64))
    baseline_cod = baseline_metrics["COD"]
    baseline_r = baseline_metrics["Pearson_r"]
    for method in method_cols:
        pred = df[method].to_numpy(dtype=np.float64)
        m = compute_metrics(y, pred)
        m["method"] = method
        m["delta_COD_vs_baseline"] = float(m["COD"] - baseline_cod)
        m["delta_Pearson_r_vs_baseline"] = float(m["Pearson_r"] - baseline_r)
        m["percent_COD_improvement_vs_baseline"] = (
            float(100.0 * (m["COD"] - baseline_cod) / abs(baseline_cod))
            if abs(baseline_cod) > 0
            else np.nan
        )
        pooled_rows.append(m)
        pooled_dict[method] = {k: _json_float(v) for k, v in m.items() if k != "method"}
    fold_rows = []
    for fold in sorted(df[fold_col].unique()):
        sub = df[df[fold_col] == fold]
        y_fold = sub[y_col].to_numpy(dtype=np.float64)
        base_fold = compute_metrics(
            y_fold, sub[baseline_col].to_numpy(dtype=np.float64)
        )
        base_cod_fold = base_fold["COD"]
        base_r_fold = base_fold["Pearson_r"]
        for method in method_cols:
            m = compute_metrics(y_fold, sub[method].to_numpy(dtype=np.float64))
            fold_rows.append(
                {
                    "fold": int(fold),
                    "method": method,
                    "n": int(len(sub)),
                    **m,
                    "delta_COD_vs_baseline": float(m["COD"] - base_cod_fold),
                    "delta_Pearson_r_vs_baseline": float(m["Pearson_r"] - base_r_fold),
                }
            )
    return (pd.DataFrame(pooled_rows), pd.DataFrame(fold_rows), pooled_dict)


def run_bootstrap(
    df: pd.DataFrame,
    y_col: str,
    method_cols: List[str],
    baseline_col: str,
    ensemble_col: str,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    n = len(df)
    y = df[y_col].to_numpy(dtype=np.float64)
    preds = {c: df[c].to_numpy(dtype=np.float64) for c in method_cols}
    draws = {
        "COD_baseline": np.empty(n_boot, dtype=np.float64),
        "COD_ensemble": np.empty(n_boot, dtype=np.float64),
        "delta_COD_ensemble_vs_baseline": np.empty(n_boot, dtype=np.float64),
        "Pearson_r_baseline": np.empty(n_boot, dtype=np.float64),
        "Pearson_r_ensemble": np.empty(n_boot, dtype=np.float64),
        "delta_Pearson_r_ensemble_vs_baseline": np.empty(n_boot, dtype=np.float64),
        "mean_squared_error_improvement": np.empty(n_boot, dtype=np.float64),
    }
    method_cod = {c: np.empty(n_boot, dtype=np.float64) for c in method_cols}
    method_r = {c: np.empty(n_boot, dtype=np.float64) for c in method_cols}
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        for c in method_cols:
            m = compute_metrics(y_b, preds[c][idx])
            method_cod[c][b] = m["COD"]
            method_r[c][b] = m["Pearson_r"]
        draws["COD_baseline"][b] = method_cod[baseline_col][b]
        draws["COD_ensemble"][b] = method_cod[ensemble_col][b]
        draws["delta_COD_ensemble_vs_baseline"][b] = (
            method_cod[ensemble_col][b] - method_cod[baseline_col][b]
        )
        draws["Pearson_r_baseline"][b] = method_r[baseline_col][b]
        draws["Pearson_r_ensemble"][b] = method_r[ensemble_col][b]
        draws["delta_Pearson_r_ensemble_vs_baseline"][b] = (
            method_r[ensemble_col][b] - method_r[baseline_col][b]
        )
        err_base = (y_b - preds[baseline_col][idx]) ** 2
        err_ens = (y_b - preds[ensemble_col][idx]) ** 2
        draws["mean_squared_error_improvement"][b] = float(np.mean(err_base - err_ens))
        if (b + 1) % 1000 == 0:
            print(f"[bootstrap] {b + 1}/{n_boot}", flush=True)
    summary_rows = []
    for name, arr in draws.items():
        q025, q50, q975 = _safe_percentile(arr, [2.5, 50, 97.5])
        summary_rows.append(
            {
                "quantity": name,
                "mean": float(np.nanmean(arr)),
                "median": q50,
                "ci_2p5": q025,
                "ci_97p5": q975,
                "n_boot": int(n_boot),
            }
        )
    for c in method_cols:
        for metric_name, source in [("COD", method_cod), ("Pearson_r", method_r)]:
            arr = source[c]
            q025, q50, q975 = _safe_percentile(arr, [2.5, 50, 97.5])
            summary_rows.append(
                {
                    "quantity": f"{metric_name}_{c}",
                    "mean": float(np.nanmean(arr)),
                    "median": q50,
                    "ci_2p5": q025,
                    "ci_97p5": q975,
                    "n_boot": int(n_boot),
                }
            )
    all_draws = {}
    all_draws.update(draws)
    for c in method_cols:
        all_draws[f"COD_{c}"] = method_cod[c]
        all_draws[f"Pearson_r_{c}"] = method_r[c]
    return (pd.DataFrame(summary_rows), all_draws)


def run_paired_error_tests(
    df: pd.DataFrame, y_col: str, baseline_col: str, ensemble_col: str
) -> Dict[str, float]:
    y = df[y_col].to_numpy(dtype=np.float64)
    p_base = df[baseline_col].to_numpy(dtype=np.float64)
    p_ens = df[ensemble_col].to_numpy(dtype=np.float64)
    se_base = (y - p_base) ** 2
    se_ens = (y - p_ens) ** 2
    improvement = se_base - se_ens
    t_two = stats.ttest_1samp(improvement, 0.0, alternative="two-sided")
    t_greater = stats.ttest_1samp(improvement, 0.0, alternative="greater")
    try:
        w_two = stats.wilcoxon(
            improvement, alternative="two-sided", zero_method="wilcox"
        )
        w_greater = stats.wilcoxon(
            improvement, alternative="greater", zero_method="wilcox"
        )
        wilcoxon_stat_two = float(w_two.statistic)
        wilcoxon_p_two = float(w_two.pvalue)
        wilcoxon_stat_greater = float(w_greater.statistic)
        wilcoxon_p_greater = float(w_greater.pvalue)
    except Exception:
        wilcoxon_stat_two = np.nan
        wilcoxon_p_two = np.nan
        wilcoxon_stat_greater = np.nan
        wilcoxon_p_greater = np.nan
    return {
        "n": int(len(improvement)),
        "mean_squared_error_baseline": float(np.mean(se_base)),
        "mean_squared_error_ensemble": float(np.mean(se_ens)),
        "mean_squared_error_improvement": float(np.mean(improvement)),
        "median_squared_error_improvement": float(np.median(improvement)),
        "fraction_subjects_improved": float(np.mean(improvement > 0.0)),
        "paired_t_stat_two_sided": float(t_two.statistic),
        "paired_t_p_two_sided": float(t_two.pvalue),
        "paired_t_stat_greater": float(t_greater.statistic),
        "paired_t_p_greater": float(t_greater.pvalue),
        "wilcoxon_stat_two_sided": wilcoxon_stat_two,
        "wilcoxon_p_two_sided": wilcoxon_p_two,
        "wilcoxon_stat_greater": wilcoxon_stat_greater,
        "wilcoxon_p_greater": wilcoxon_p_greater,
    }


def reconstruct_ensemble_from_weights(
    df: pd.DataFrame, weights: pd.DataFrame, fold_col: str, abcd_col: str, pc_col: str
) -> np.ndarray:
    pred = np.full(len(df), np.nan, dtype=np.float64)
    for _, row in weights.iterrows():
        fold = int(row["outer_fold"])
        idx = np.where(df[fold_col].to_numpy(dtype=np.int64) == fold)[0]
        if idx.size == 0:
            continue
        abcd = df.iloc[idx][abcd_col].to_numpy(dtype=np.float64)
        pc = df.iloc[idx][pc_col].to_numpy(dtype=np.float64)
        x_abcd = (abcd - float(row["ensemble_x_mu_abcd"])) / float(
            row["ensemble_x_sd_abcd"]
        )
        x_pc = (pc - float(row["ensemble_x_mu_pc"])) / float(row["ensemble_x_sd_pc"])
        pred[idx] = (
            float(row["ensemble_intercept"])
            + float(row["ensemble_coef_abcd_standardized"]) * x_abcd
            + float(row["ensemble_coef_pc_standardized"]) * x_pc
        )
    if not np.isfinite(pred).all():
        raise RuntimeError("Reconstructed ensemble prediction contains NaN/Inf.")
    return pred


def run_pc_stream_permutation(
    df: pd.DataFrame,
    weights: pd.DataFrame,
    y_col: str,
    fold_col: str,
    baseline_col: str,
    abcd_col: str,
    pc_col: str,
    observed_ensemble_col: str,
    n_perm: int,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    y = df[y_col].to_numpy(dtype=np.float64)
    p_base = df[baseline_col].to_numpy(dtype=np.float64)
    p_obs = df[observed_ensemble_col].to_numpy(dtype=np.float64)
    observed_delta_cod = (
        compute_metrics(y, p_obs)["COD"] - compute_metrics(y, p_base)["COD"]
    )
    observed_delta_r = (
        compute_metrics(y, p_obs)["Pearson_r"] - compute_metrics(y, p_base)["Pearson_r"]
    )
    perm_delta_cod = np.empty(n_perm, dtype=np.float64)
    perm_delta_r = np.empty(n_perm, dtype=np.float64)
    df_work = df.copy()
    fold_arr = df_work[fold_col].to_numpy(dtype=np.int64)
    pc_original = df_work[pc_col].to_numpy(dtype=np.float64)
    for p in range(int(n_perm)):
        pc_perm = pc_original.copy()
        for fold in sorted(np.unique(fold_arr)):
            idx = np.where(fold_arr == fold)[0]
            pc_perm[idx] = rng.permutation(pc_perm[idx])
        df_work[pc_col] = pc_perm
        pred_perm = reconstruct_ensemble_from_weights(
            df=df_work,
            weights=weights,
            fold_col=fold_col,
            abcd_col=abcd_col,
            pc_col=pc_col,
        )
        m_perm = compute_metrics(y, pred_perm)
        m_base = compute_metrics(y, p_base)
        perm_delta_cod[p] = m_perm["COD"] - m_base["COD"]
        perm_delta_r[p] = m_perm["Pearson_r"] - m_base["Pearson_r"]
        if (p + 1) % 500 == 0:
            print(f"[permutation] {p + 1}/{n_perm}", flush=True)
    p_ge_cod = float((1 + np.sum(perm_delta_cod >= observed_delta_cod)) / (n_perm + 1))
    p_ge_r = float((1 + np.sum(perm_delta_r >= observed_delta_r)) / (n_perm + 1))
    cod_q = _safe_percentile(perm_delta_cod, [2.5, 50, 97.5])
    r_q = _safe_percentile(perm_delta_r, [2.5, 50, 97.5])
    summary = pd.DataFrame(
        [
            {
                "quantity": "delta_COD_ensemble_vs_baseline",
                "observed": float(observed_delta_cod),
                "perm_mean": float(np.mean(perm_delta_cod)),
                "perm_ci_2p5": cod_q[0],
                "perm_median": cod_q[1],
                "perm_ci_97p5": cod_q[2],
                "one_sided_p_perm_ge_observed": p_ge_cod,
                "n_perm": int(n_perm),
            },
            {
                "quantity": "delta_Pearson_r_ensemble_vs_baseline",
                "observed": float(observed_delta_r),
                "perm_mean": float(np.mean(perm_delta_r)),
                "perm_ci_2p5": r_q[0],
                "perm_median": r_q[1],
                "perm_ci_97p5": r_q[2],
                "one_sided_p_perm_ge_observed": p_ge_r,
                "n_perm": int(n_perm),
            },
        ]
    )
    draws = np.column_stack([perm_delta_cod, perm_delta_r])
    return (summary, draws)
