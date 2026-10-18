"""Train UKB KRR source models for phenotype PCs or disease PCs.

RSFC is row-centered and L2-normalized. Source targets are normalized on
observed fitting entries, without covariate residualization in this script.
Validation zero-baseline COD selects alpha; refitting on the combined
training and validation sets is controlled by the configuration.
"""

from __future__ import annotations
import argparse
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def safe_json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (not np.isfinite(obj)):
        return None
    return obj


def read_freeze_list(path: str) -> List[str]:
    cols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                cols.append(s.split("\t")[0])
    if cols and cols[0].lower() == "col":
        cols = cols[1:]
    if not cols:
        raise ValueError(f"freeze list empty: {path}")
    if pd.Index(cols).duplicated().any():
        dup = pd.Index(cols)[pd.Index(cols).duplicated()].tolist()
        raise ValueError(f"freeze list contains duplicated names: {dup[:10]}")
    return cols


def get_edge_cols_from_header(csv_path: str) -> List[str]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    pairs = []
    for c in header:
        m = re.fullmatch("e(\\d+)", str(c))
        if m:
            pairs.append((int(m.group(1)), c))
    pairs.sort(key=lambda x: x[0])
    cols = [c for _, c in pairs]
    expected = [f"e{i}" for i in range(1, 1486)]
    if cols != expected:
        missing = [c for c in expected if c not in cols]
        extra = [c for c in cols if c not in expected]
        raise ValueError(
            f"edge columns must be exactly e1..e1485. n={len(cols)}, missing_first10={missing[:10]}, extra_first10={extra[:10]}"
        )
    return cols


def get_parquet_columns(path: str) -> List[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        return pd.read_parquet(path).columns.tolist()


def can_stratify(labels: Optional[np.ndarray], test_size: float) -> bool:
    if labels is None:
        return False
    labels = np.asarray(labels)
    if labels.size == 0:
        return False
    uniq, cnt = np.unique(labels, return_counts=True)
    if uniq.size < 2:
        return False
    return bool(
        not (cnt * test_size < 1).any() and (not (cnt * (1.0 - test_size) < 1).any())
    )


def unitize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X - np.mean(X, axis=1, keepdims=True)
    norm = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    norm[norm == 0.0] = 1.0
    return X / norm


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = (x[m], y[m])
    if x.size < 3 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cod_centered(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = (y_true[m], y_pred[m])
    if y_true.size == 0:
        return np.nan
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return np.nan if sst <= 0.0 else float(1.0 - sse / sst)


def predictive_cod_zero_baseline(y_true_z: np.ndarray, y_pred_z: np.ndarray) -> float:
    y_true_z = np.asarray(y_true_z, dtype=np.float64).ravel()
    y_pred_z = np.asarray(y_pred_z, dtype=np.float64).ravel()
    m = np.isfinite(y_true_z) & np.isfinite(y_pred_z)
    y_true_z, y_pred_z = (y_true_z[m], y_pred_z[m])
    if y_true_z.size == 0:
        return np.nan
    sse = float(np.sum((y_true_z - y_pred_z) ** 2))
    denom = float(np.sum(y_true_z**2))
    return np.nan if denom <= 0.0 else float(1.0 - sse / denom)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = (y_true[m], y_pred[m])
    if y_true.size == 0:
        return dict(
            n=0,
            R2_sklearn=np.nan,
            COD_centered=np.nan,
            Pearson_r=np.nan,
            RMSE=np.nan,
            MAE=np.nan,
            predictive_COD_zero_baseline=np.nan,
        )
    return {
        "n": int(y_true.size),
        "R2_sklearn": float(r2_score(y_true, y_pred)) if y_true.size >= 2 else np.nan,
        "COD_centered": cod_centered(y_true, y_pred),
        "Pearson_r": safe_corr(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "predictive_COD_zero_baseline": predictive_cod_zero_baseline(y_true, y_pred),
    }


def load_stage1_pc_data(
    ukb_final_csv: str,
    cov_parquet: str,
    freeze_tsv: str,
    site_col: str = "QC_Imaging_Site",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Optional[np.ndarray],
    List[str],
]:
    for p, label in [
        (ukb_final_csv, "UKB_final_csv"),
        (cov_parquet, "cov_parquet"),
        (freeze_tsv, "freeze_tsv"),
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")
    y_names = read_freeze_list(freeze_tsv)
    pq_cols = get_parquet_columns(cov_parquet)
    if "eid" not in pq_cols:
        raise KeyError(f"cov_parquet lacks eid: {cov_parquet}")
    missing_targets = [c for c in y_names if c not in pq_cols]
    if missing_targets:
        raise KeyError(
            f"cov_parquet missing freeze targets, first10={missing_targets[:10]}"
        )
    ind_cols_exist = [
        f"{y}__is_missing" for y in y_names if f"{y}__is_missing" in pq_cols
    ]
    cov_df = pd.read_parquet(cov_parquet, columns=["eid"] + y_names + ind_cols_exist)
    edge_cols = get_edge_cols_from_header(ukb_final_csv)
    header = pd.read_csv(ukb_final_csv, nrows=0).columns.tolist()
    x_cols = ["eid"] + edge_cols
    need_site = site_col in header
    if need_site:
        x_cols.append(site_col)
    ukb_df = pd.read_csv(ukb_final_csv, usecols=x_cols)
    cov_df["eid"] = pd.to_numeric(cov_df["eid"], errors="raise").astype(np.int64)
    ukb_df["eid"] = pd.to_numeric(ukb_df["eid"], errors="raise").astype(np.int64)
    if cov_df["eid"].duplicated().any():
        raise ValueError(
            f"cov_parquet contains duplicated eid examples={cov_df.loc[cov_df.eid.duplicated(), 'eid'].head(10).tolist()}"
        )
    if ukb_df["eid"].duplicated().any():
        raise ValueError(
            f"UKB_final.csv contains duplicated eid examples={ukb_df.loc[ukb_df.eid.duplicated(), 'eid'].head(10).tolist()}"
        )
    n_cov, n_ukb = (cov_df["eid"].nunique(), ukb_df["eid"].nunique())
    merged = ukb_df.merge(cov_df, on="eid", how="inner", sort=False)
    if merged.empty:
        raise RuntimeError("eid join produced 0 rows")
    n_merged = merged["eid"].nunique()
    if n_merged != n_cov:
        raise RuntimeError(
            f"eid join lost target-bank samples: ukb={n_ukb}, cov={n_cov}, merged={n_merged}"
        )
    if n_merged != n_ukb:
        print(
            f"[load] using target-bank subjects only: ukb={n_ukb}, cov={n_cov}, merged={n_merged}"
        )
    X = (
        merged[edge_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    Y = merged[y_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(Y)
    for j, y in enumerate(y_names):
        ind = f"{y}__is_missing"
        if ind in merged.columns:
            mask[:, j] = np.isfinite(Y[:, j]) & (
                pd.to_numeric(merged[ind], errors="coerce").to_numpy() == 0
            )
    Y_safe = Y.copy()
    Y_safe[~np.isfinite(Y_safe)] = 0.0
    if not np.isfinite(X).all():
        raise RuntimeError(f"X contains NaN/Inf, n_bad={int((~np.isfinite(X)).sum())}")
    site_vec = None
    if need_site:
        site_vec = merged[site_col].astype(str).fillna("missing").to_numpy()
        if np.unique(site_vec).size < 2:
            site_vec = None
    eid = merged["eid"].to_numpy(dtype=np.int64)
    print("[load] done")
    print(
        f"  X={X.shape}, Y={Y_safe.shape}, supervised={int(mask.sum())}/{mask.size}, targets={len(y_names)}"
    )
    print(
        f"  indicators_found={len(ind_cols_exist)}, site_available={site_vec is not None}"
    )
    return (X, Y_safe, mask.astype(bool), eid, y_names, site_vec, edge_cols)


def make_splits(
    n: int,
    eid: np.ndarray,
    site_vec: Optional[np.ndarray],
    test_size: float,
    valid_size: float,
    seed: int,
) -> Dict[str, object]:
    idx_all = np.arange(n, dtype=np.int64)
    if can_stratify(site_vec, test_size):
        train_val_idx, test_idx = train_test_split(
            idx_all,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=site_vec,
        )
        site_trv = site_vec[train_val_idx]
        use_outer = True
    else:
        train_val_idx, test_idx = train_test_split(
            idx_all, test_size=test_size, random_state=seed, shuffle=True
        )
        site_trv = None
        use_outer = False
    rel = np.arange(train_val_idx.size, dtype=np.int64)
    if can_stratify(site_trv, valid_size):
        train_rel, valid_rel = train_test_split(
            rel,
            test_size=valid_size,
            random_state=seed,
            shuffle=True,
            stratify=site_trv,
        )
        use_inner = True
    else:
        train_rel, valid_rel = train_test_split(
            rel, test_size=valid_size, random_state=seed, shuffle=True
        )
        use_inner = False
    train_idx = train_val_idx[train_rel]
    valid_idx = train_val_idx[valid_rel]
    concat = np.concatenate([train_idx, valid_idx, test_idx])
    if (
        concat.size != n
        or np.unique(concat).size != n
        or (not np.array_equal(np.sort(concat), idx_all))
    ):
        raise RuntimeError("train/valid/test split is not a clean partition")
    return {
        "train_idx": train_idx.astype(np.int64),
        "valid_idx": valid_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
        "train_val_idx": train_val_idx.astype(np.int64),
        "eid_train": eid[train_idx].astype(np.int64),
        "eid_valid": eid[valid_idx].astype(np.int64),
        "eid_test": eid[test_idx].astype(np.int64),
        "eid_train_val": eid[train_val_idx].astype(np.int64),
        "use_stratify_outer": bool(use_outer),
        "use_stratify_inner": bool(use_inner),
    }


def fit_one_target(
    target_name: str,
    y: np.ndarray,
    m: np.ndarray,
    Z_train: np.ndarray,
    Z_valid: np.ndarray,
    Z_test: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    alphas: Sequence[float],
    min_train_obs: int,
    min_valid_obs: int,
    min_test_obs: int,
    refit_on_train_valid: bool,
    Z_train_val: Optional[np.ndarray],
    train_val_idx: np.ndarray,
) -> Tuple[Dict[str, object], Dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.asarray(m, dtype=bool).ravel()
    obs_tr, obs_va, obs_te = (m[train_idx], m[valid_idx], m[test_idx])
    n_tr, n_va, n_te = (int(obs_tr.sum()), int(obs_va.sum()), int(obs_te.sum()))
    summary: Dict[str, object] = {
        "target": target_name,
        "n_train_obs_inner": n_tr,
        "n_valid_obs": n_va,
        "n_test_obs": n_te,
        "selected_alpha": None,
        "predictive_COD_valid_z": None,
        "valid_Pearson_r_z": None,
        "valid_COD_centered_z": None,
        "test_predictive_COD_z": None,
        "test_COD_centered_z": None,
        "test_Pearson_r_z": None,
        "test_RMSE_z": None,
        "test_MAE_z": None,
        "fit_n_obs": None,
        "fit_source": "inner_train",
        "fallback_used": False,
        "failure_reason": None,
    }
    yt_out = np.full(test_idx.size, np.nan, dtype=np.float32)
    yp_out = np.full(test_idx.size, np.nan, dtype=np.float32)
    mt_out = obs_te.astype(bool)

    def failed(reason: str, y_mu: float = 0.0, y_sigma: float = 1.0):
        summary["fallback_used"] = True
        summary["failure_reason"] = reason
        rec = dict(
            status="failed",
            failure_reason=reason,
            fit_train_idx=np.array([], dtype=np.int64),
            dual_coef=np.array([], dtype=np.float64),
            alpha=np.nan,
            y_mu=float(y_mu),
            y_sigma=float(y_sigma),
            fit_source="none",
        )
        return (rec, summary, yt_out, yp_out, mt_out)

    if n_tr < min_train_obs:
        return failed(f"not enough observed training samples: {n_tr} < {min_train_obs}")
    if n_va < min_valid_obs:
        return failed(
            f"not enough observed validation samples: {n_va} < {min_valid_obs}"
        )
    y_tr_raw = y[train_idx][obs_tr]
    y_mu = float(np.mean(y_tr_raw))
    y_sigma = float(np.std(y_tr_raw))
    if not np.isfinite(y_sigma) or y_sigma < 1e-12:
        y_sigma = 1.0
    y_tr_z = ((y[train_idx][obs_tr] - y_mu) / y_sigma).astype(np.float64)
    y_va_z = ((y[valid_idx][obs_va] - y_mu) / y_sigma).astype(np.float64)
    Z_fit = Z_train[obs_tr]
    K_fit = Z_fit @ Z_fit.T
    K_valid = Z_valid[obs_va] @ Z_fit.T
    best_alpha = None
    best_score = -np.inf
    best_valid_pred = None
    alpha_search = []
    for alpha in alphas:
        try:
            model = KernelRidge(alpha=float(alpha), kernel="precomputed")
            model.fit(K_fit, y_tr_z)
            pred_va = np.asarray(model.predict(K_valid), dtype=np.float64).ravel()
            score = predictive_cod_zero_baseline(y_va_z, pred_va)
            alpha_search.append(
                {
                    "alpha": float(alpha),
                    "predictive_COD_valid_z": score,
                    "COD_centered_valid_z": cod_centered(y_va_z, pred_va),
                    "Pearson_r_valid_z": safe_corr(y_va_z, pred_va),
                }
            )
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_alpha = float(alpha)
                best_valid_pred = pred_va
        except Exception as e:
            alpha_search.append(
                {
                    "alpha": float(alpha),
                    "predictive_COD_valid_z": np.nan,
                    "error": repr(e),
                }
            )
    if best_alpha is None:
        return failed(
            "all alpha candidates failed or produced non-finite validation scores",
            y_mu,
            y_sigma,
        )
    if refit_on_train_valid:
        if Z_train_val is None:
            raise ValueError("Z_train_val required for refit")
        obs_refit = m[train_val_idx]
        if int(obs_refit.sum()) >= min_train_obs:
            y_refit_raw = y[train_val_idx][obs_refit]
            y_mu_final = float(np.mean(y_refit_raw))
            y_sigma_final = float(np.std(y_refit_raw))
            if not np.isfinite(y_sigma_final) or y_sigma_final < 1e-12:
                y_sigma_final = 1.0
            y_final = ((y_refit_raw - y_mu_final) / y_sigma_final).astype(np.float64)
            idx_final_base = np.where(obs_refit)[0].astype(np.int64)
            Z_final = Z_train_val[obs_refit]
            fit_source = "outer_train_val_refit"
        else:
            summary["fallback_used"] = True
            summary["failure_reason"] = (
                "refit requested but not enough train_val observations; used inner_train fit"
            )
            y_mu_final, y_sigma_final = (y_mu, y_sigma)
            y_final = y_tr_z
            idx_final_base = np.where(obs_tr)[0].astype(np.int64)
            Z_final = Z_fit
            fit_source = "inner_train_fallback_after_refit_failure"
    else:
        y_mu_final, y_sigma_final = (y_mu, y_sigma)
        y_final = y_tr_z
        idx_final_base = np.where(obs_tr)[0].astype(np.int64)
        Z_final = Z_fit
        fit_source = "inner_train"
    final_model = KernelRidge(alpha=float(best_alpha), kernel="precomputed")
    final_model.fit(Z_final @ Z_final.T, y_final)
    dual = np.asarray(final_model.dual_coef_, dtype=np.float64).ravel()
    if n_te >= min_test_obs:
        y_te_z = ((y[test_idx][obs_te] - y_mu_final) / y_sigma_final).astype(np.float64)
        pred_te = np.asarray(
            final_model.predict(Z_test[obs_te] @ Z_final.T), dtype=np.float64
        ).ravel()
        mt = compute_metrics(y_te_z, pred_te)
        summary.update(
            {
                "test_predictive_COD_z": mt["predictive_COD_zero_baseline"],
                "test_COD_centered_z": mt["COD_centered"],
                "test_Pearson_r_z": mt["Pearson_r"],
                "test_RMSE_z": mt["RMSE"],
                "test_MAE_z": mt["MAE"],
            }
        )
        yt_out[obs_te] = y_te_z.astype(np.float32)
        yp_out[obs_te] = pred_te.astype(np.float32)
    else:
        summary["failure_reason"] = (
            f"not enough observed test samples for metrics: {n_te} < {min_test_obs}"
        )
    mv = compute_metrics(y_va_z, best_valid_pred)
    summary.update(
        {
            "selected_alpha": float(best_alpha),
            "predictive_COD_valid_z": float(best_score),
            "valid_Pearson_r_z": mv["Pearson_r"],
            "valid_COD_centered_z": mv["COD_centered"],
            "fit_n_obs": int(idx_final_base.size),
            "fit_source": fit_source,
        }
    )
    rec = {
        "status": "ok",
        "target": target_name,
        "fit_train_idx": idx_final_base.astype(np.int64),
        "dual_coef": dual.astype(np.float64),
        "alpha": float(best_alpha),
        "y_mu": float(y_mu_final),
        "y_sigma": float(y_sigma_final),
        "fit_source": fit_source,
        "n_fit_obs": int(idx_final_base.size),
        "selection_metric_name": "predictive_COD_valid_z",
        "predictive_COD_valid_z": float(best_score),
        "alpha_search": alpha_search,
    }
    if dual.size != idx_final_base.size:
        raise RuntimeError(f"{target_name}: dual_coef length != fit_train_idx length")
    return (rec, summary, yt_out, yp_out, mt_out)


def run_one_seed(
    X,
    Y,
    mask,
    eid,
    y_names,
    site_vec,
    edge_names,
    seed,
    alphas,
    out_dir,
    test_size,
    valid_size,
    min_train_obs,
    min_valid_obs,
    min_test_obs,
    refit_on_train_valid,
    compress,
    save_test_predictions,
):
    run_tag = f"stage1_seed{int(seed)}"
    seed_dir = out_dir / run_tag
    seed_dir.mkdir(parents=True, exist_ok=True)
    split = make_splits(X.shape[0], eid, site_vec, test_size, valid_size, int(seed))
    tr, va, te, trv = (
        split["train_idx"],
        split["valid_idx"],
        split["test_idx"],
        split["train_val_idx"],
    )
    print(
        f"\n[{run_tag}] n={X.shape[0]} train={len(tr)} valid={len(va)} test={len(te)} refit={refit_on_train_valid}"
    )
    print(
        f"  stratify_outer={split['use_stratify_outer']} stratify_inner={split['use_stratify_inner']}"
    )
    np.savez(
        seed_dir / f"krr_stage1_pc_outer_inner_split_{run_tag}.npz",
        train_idx=tr,
        valid_idx=va,
        test_idx=te,
        train_val_idx=trv,
        eid_train=split["eid_train"],
        eid_valid=split["eid_valid"],
        eid_test=split["eid_test"],
        eid_train_val=split["eid_train_val"],
        seed=np.array(int(seed), dtype=np.int64),
        test_size=np.array(float(test_size)),
        valid_size=np.array(float(valid_size)),
        use_stratify_outer=np.array(split["use_stratify_outer"]),
        use_stratify_inner=np.array(split["use_stratify_inner"]),
    )
    Z_tr = unitize_rows(X[tr])
    Z_va = unitize_rows(X[va])
    Z_te = unitize_rows(X[te])
    Z_trv = unitize_rows(X[trv]) if refit_on_train_valid else None
    Z_bundle = Z_trv if refit_on_train_valid else Z_tr
    eid_bundle = split["eid_train_val"] if refit_on_train_valid else split["eid_train"]
    fit_index_space = "outer_train_val" if refit_on_train_valid else "inner_train"
    target_models = {}
    rows = []
    y_true_test_z = np.full((len(te), len(y_names)), np.nan, dtype=np.float32)
    y_pred_test_z = np.full((len(te), len(y_names)), np.nan, dtype=np.float32)
    test_valid_mask = np.zeros((len(te), len(y_names)), dtype=bool)
    for j, target in enumerate(y_names):
        print(f"  target {j + 1:02d}/{len(y_names):02d}: {target}")
        rec, summ, yt, yp, mt = fit_one_target(
            target,
            Y[:, j],
            mask[:, j],
            Z_tr,
            Z_va,
            Z_te,
            tr,
            va,
            te,
            alphas,
            min_train_obs,
            min_valid_obs,
            min_test_obs,
            refit_on_train_valid,
            Z_trv,
            trv,
        )
        target_models[target] = rec
        rows.append(summ)
        y_true_test_z[:, j] = yt
        y_pred_test_z[:, j] = yp
        test_valid_mask[:, j] = mt
        print(
            f"    alpha={summ.get('selected_alpha')} validCOD0={summ.get('predictive_COD_valid_z')} testCOD0={summ.get('test_predictive_COD_z')} r={summ.get('test_Pearson_r_z')}"
        )
    metrics = pd.DataFrame(rows)
    metrics_csv = seed_dir / f"krr_stage1_pc_target_metrics_{run_tag}.csv"
    metrics.to_csv(metrics_csv, index=False)
    seed_summary = {
        "run_tag": run_tag,
        "seed": int(seed),
        "n_targets": int(len(y_names)),
        "n_total": int(X.shape[0]),
        "n_outer_train_val": int(len(trv)),
        "n_inner_train": int(len(tr)),
        "n_valid": int(len(va)),
        "n_test": int(len(te)),
        "use_stratify_outer": bool(split["use_stratify_outer"]),
        "use_stratify_inner": bool(split["use_stratify_inner"]),
        "refit_on_train_valid": bool(refit_on_train_valid),
        "fit_index_space": fit_index_space,
        "mean_valid_predictive_COD_z": float(
            np.nanmean(metrics["predictive_COD_valid_z"].to_numpy(float))
        ),
        "mean_valid_Pearson_r_z": float(
            np.nanmean(metrics["valid_Pearson_r_z"].to_numpy(float))
        ),
        "mean_test_predictive_COD_z": float(
            np.nanmean(metrics["test_predictive_COD_z"].to_numpy(float))
        ),
        "mean_test_COD_centered_z": float(
            np.nanmean(metrics["test_COD_centered_z"].to_numpy(float))
        ),
        "mean_test_Pearson_r_z": float(
            np.nanmean(metrics["test_Pearson_r_z"].to_numpy(float))
        ),
        "n_failed_targets": int(metrics["fallback_used"].astype(bool).sum()),
    }
    if save_test_predictions:
        np.savez_compressed(
            seed_dir / f"krr_stage1_pc_test_predictions_{run_tag}.npz",
            eid_test=split["eid_test"],
            target_names=np.asarray(y_names, dtype=object),
            y_true_z=y_true_test_z,
            y_pred_z=y_pred_test_z,
            valid_mask=test_valid_mask,
        )
    bundle = {
        "bundle_type": "ukb_stage1_krr_pc_transfer_bundle",
        "run_tag": run_tag,
        "metadata": {
            "created_at": _dt.datetime.now().isoformat(),
            "seed": int(seed),
            "n_total_subjects_loaded": int(X.shape[0]),
            "n_outer_train_val_subjects": int(len(trv)),
            "n_inner_train_subjects": int(len(tr)),
            "n_valid_subjects": int(len(va)),
            "n_test_subjects": int(len(te)),
            "n_edges": int(len(edge_names)),
            "n_targets": int(len(y_names)),
            "target_names": list(y_names),
            "alphas": [float(a) for a in alphas],
            "test_size": float(test_size),
            "valid_size": float(valid_size),
            "use_stratify_outer": bool(split["use_stratify_outer"]),
            "use_stratify_inner": bool(split["use_stratify_inner"]),
            "refit_on_train_valid": bool(refit_on_train_valid),
            "fit_index_space": fit_index_space,
            "feature_transform": "unitize_rows",
            "kernel_type": "corr_precomputed",
            "selection_metric_name": "predictive_COD_valid_z",
            "selection_metric_definition": "1 - SSE / sum(y_valid_z^2), y_valid_z normalized by inner training observed entries",
            "feature_order_contract": "target dataset must use edge_names in saved order",
            "row_identity_contract": "Z_train_full rows correspond to eid_train_full in saved order",
            "target_model_index_contract": "target_models[target]['fit_train_idx'] indexes rows of Z_train_full",
            "y_norm_contract": "y_mu/y_sigma computed from final fit set observed entries only",
            "inference_contract": {
                "test_feature_transform": "unitize_rows",
                "kernel_formula": "K_test_fit = Z_test @ Z_train_full[fit_train_idx].T",
                "prediction_formula_z": "y_pred_z = K_test_fit @ dual_coef",
                "prediction_formula_raw": "y_pred_raw = y_pred_z * y_sigma + y_mu",
                "recommended_downstream_column": "use y_pred_z as <target>__zpred for Stage2 stacking",
            },
        },
        "edge_names": list(edge_names),
        "target_names": list(y_names),
        "Z_train_full": Z_bundle.astype(np.float32),
        "eid_train_full": eid_bundle.astype(np.int64),
        "eid_valid": split["eid_valid"].astype(np.int64),
        "eid_test": split["eid_test"].astype(np.int64),
        "target_models": target_models,
        "target_summaries": rows,
        "seed_summary": seed_summary,
    }
    if bundle["Z_train_full"].shape[0] != len(bundle["eid_train_full"]):
        raise RuntimeError("bundle row mismatch")
    if pd.Index(bundle["eid_train_full"]).duplicated().any():
        raise RuntimeError("bundle eid_train_full duplicates")
    for target, rec in target_models.items():
        if rec.get("status") == "ok":
            idx = np.asarray(rec["fit_train_idx"], dtype=np.int64)
            dual = np.asarray(rec["dual_coef"], dtype=np.float64)
            if (
                idx.size == 0
                or idx.min() < 0
                or idx.max() >= bundle["Z_train_full"].shape[0]
            ):
                raise RuntimeError(f"{target}: bad fit_train_idx")
            if dual.ndim != 1 or dual.size != idx.size:
                raise RuntimeError(f"{target}: bad dual_coef length")
    bundle_path = seed_dir / f"krr_stage1_pc_transfer_bundle_{run_tag}.joblib"
    joblib.dump(bundle, bundle_path, compress=int(compress))
    summary_json = seed_dir / f"krr_stage1_pc_summary_{run_tag}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            sanitize_for_json(
                {
                    "run_tag": run_tag,
                    "seed_summary": seed_summary,
                    "target_summaries": rows,
                    "output_files": {
                        "metrics_csv": str(metrics_csv),
                        "bundle_joblib": str(bundle_path),
                    },
                }
            ),
            f,
            indent=2,
            ensure_ascii=False,
            default=safe_json_default,
        )
    seed_summary.update(
        metrics_csv=str(metrics_csv),
        summary_json=str(summary_json),
        bundle_joblib=str(bundle_path),
    )
    print(f"[saved] {metrics_csv}")
    print(f"[saved] {summary_json}")
    print(f"[saved] {bundle_path}")
    return seed_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ukb-csv", required=True)
    parser.add_argument("--target-parquet", required=True)
    parser.add_argument("--freeze-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    X, Y, mask, eid, targets, sites, edges = load_stage1_pc_data(
        args.ukb_csv, args.target_parquet, args.freeze_tsv
    )
    if targets != config["target_names"]:
        raise RuntimeError(
            "Freeze targets/order do not match the selected source-family configuration."
        )
    if len(eid) != config["expected_n_subjects"]:
        raise RuntimeError(
            "Source cohort size differs from the reported source-family configuration."
        )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in config["seeds"]:
        summaries.append(
            run_one_seed(
                X,
                Y,
                mask,
                eid,
                targets,
                sites,
                edges,
                int(seed),
                config["alphas"],
                out,
                config["test_size"],
                config["valid_size"],
                config["min_train_obs"],
                config["min_valid_obs"],
                config["min_test_obs"],
                config["refit_on_train_valid"],
                3,
                True,
            )
        )
    pd.DataFrame(summaries).to_csv(out / "krr_stage1_pc_seed_summary.csv", index=False)
    manifest = dict(
        script=Path(__file__).name,
        inputs=dict(
            ukb_csv=args.ukb_csv,
            target_parquet=args.target_parquet,
            freeze_tsv=args.freeze_tsv,
        ),
        config=config,
        data_shape=dict(
            n_subjects=len(eid), n_edges=len(edges), n_targets=len(targets)
        ),
        target_names=targets,
        seed_summaries=summaries,
    )
    (out / "krr_stage1_pc_manifest.json").write_text(
        json.dumps(sanitize_for_json(manifest), indent=2)
    )


if __name__ == "__main__":
    main()
