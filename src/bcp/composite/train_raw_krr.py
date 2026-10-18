"""Train UKB raw-phenotype KRR source models.

Uses site-stratified 80/20 splitting, observed-target masks, training-set
target normalization and validation predictive-COD penalty selection.
"""

import argparse
import os
import re
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def load_data(csv_path, target_name=None, target_index=None):
    y_parquet_path = os.environ["STAGE1_Y_PARQUET"]
    freeze_base_path = os.environ["STAGE1_FREEZE_BASE"]

    def _normalize_eid(s, name):
        x = pd.to_numeric(s, errors="raise")
        if not np.isfinite(x).all():
            raise ValueError(f"{name}Input validation or source fitting error: ")
        if not np.equal(x, np.floor(x)).all():
            raise ValueError(f"{name}Input validation or source fitting error: ")
        return x.astype(np.int64).astype(str)

    df_head = pd.read_csv(csv_path, nrows=0)
    cols = df_head.columns.tolist()
    if "eid" not in cols:
        raise KeyError("Input validation or source fitting error: ")
    e_cols = []
    for c in cols:
        m = re.fullmatch("e(\\d+)", str(c))
        if m:
            e_cols.append((int(m.group(1)), c))
    if not e_cols:
        raise ValueError("Input validation or source fitting error: ")
    e_cols.sort(key=lambda x: x[0])
    e_names = [c for _, c in e_cols]
    site_col = (
        "QC_Imaging_Site"
        if "QC_Imaging_Site" in cols
        else "QC Imaging Site" if "QC Imaging Site" in cols else None
    )
    x_usecols = ["eid"] + e_names + ([site_col] if site_col is not None else [])
    df_x = pd.read_csv(csv_path, usecols=x_usecols)
    freeze_df = pd.read_csv(freeze_base_path, sep="\t")
    candidate_cols = ["col", "feature", "column", "name", "colname"]
    preferred = [c for c in candidate_cols if c in freeze_df.columns]
    if len(preferred) == 0:
        raise ValueError(
            f"Input validation or source fitting error: {freeze_base_path}Input validation or source fitting error: {list(freeze_df.columns)}"
        )
    if len(preferred) > 1:
        raise ValueError(
            f"{freeze_base_path}Input validation or source fitting error: {preferred}Input validation or source fitting error: "
        )
    feature_col = preferred[0]
    y_raw = freeze_df[feature_col]
    n_na = int(y_raw.isna().sum())
    if n_na > 0:
        raise ValueError(
            f"{freeze_base_path}Input validation or source fitting error: {feature_col}Input validation or source fitting error: {n_na}Input validation or source fitting error: "
        )
    all_y_cols = y_raw.astype(str).str.strip().tolist()
    blank_names = [x for x in all_y_cols if x == ""]
    if blank_names:
        raise ValueError(
            f"{freeze_base_path}Input validation or source fitting error: {feature_col}Input validation or source fitting error: "
        )
    bad_literal_names = [x for x in all_y_cols if x.lower() == "nan"]
    if bad_literal_names:
        raise ValueError(
            f"{freeze_base_path}Input validation or source fitting error: {feature_col}Input validation or source fitting error: "
        )
    if len(all_y_cols) == 0:
        raise ValueError("Input validation or source fitting error: ")
    if "eid" in all_y_cols:
        raise ValueError("Input validation or source fitting error: ")
    dup_cols = (
        pd.Series(all_y_cols)[pd.Series(all_y_cols).duplicated()].unique().tolist()
    )
    if dup_cols:
        raise ValueError(f"Input validation or source fitting error: {dup_cols[:10]}")
    if target_name is not None:
        target_name = str(target_name).strip()
        if target_name == "":
            target_name = None
    if target_index is not None:
        if isinstance(target_index, bool):
            raise ValueError("Input validation or source fitting error: ")
        if isinstance(target_index, str):
            target_index_str = target_index.strip()
            if target_index_str == "":
                target_index = None
            else:
                if re.fullmatch("[+-]?\\d+", target_index_str) is None:
                    raise ValueError(
                        f"Input validation or source fitting error: {target_index}"
                    )
                target_index = int(target_index_str)
        elif isinstance(target_index, (int, np.integer)):
            target_index = int(target_index)
        else:
            raise ValueError(
                f"Input validation or source fitting error: {type(target_index).__name__}Input validation or source fitting error: "
            )
    if target_name is not None and target_index is not None:
        raise ValueError("Input validation or source fitting error: ")
    try:
        import pyarrow.parquet as pq

        parquet_cols = pq.read_schema(y_parquet_path).names
    except Exception:
        parquet_cols = pd.read_parquet(y_parquet_path).columns.tolist()
    if "eid" not in parquet_cols:
        raise KeyError("Input validation or source fitting error: ")
    missing_y = [c for c in all_y_cols if c not in parquet_cols]
    if missing_y:
        raise KeyError(f"Input validation or source fitting error: {missing_y[:10]}")
    all_indicator_map = {
        c: f"{c}__is_missing" for c in all_y_cols if f"{c}__is_missing" in parquet_cols
    }
    all_y_with_indicator = list(all_indicator_map.keys())
    all_y_without_indicator = [c for c in all_y_cols if c not in all_indicator_map]
    if len(all_y_cols) != 120:
        raise ValueError(f"Input validation or source fitting error: {len(all_y_cols)}")
    if len(all_y_with_indicator) != 101 or len(all_y_without_indicator) != 19:
        raise ValueError(
            "120-base setup sanity check failed: expected 101 with indicator and 19 without indicator"
        )
    if target_name is not None:
        if target_name not in all_y_cols:
            raise ValueError(f"Input validation or source fitting error: {target_name}")
        y_cols = [target_name]
    elif target_index is not None:
        if target_index < 0 or target_index >= len(all_y_cols):
            raise IndexError(
                f"Input validation or source fitting error: {target_index}Input validation or source fitting error: {len(all_y_cols) - 1})"
            )
        y_cols = [all_y_cols[target_index]]
    else:
        y_cols = all_y_cols
    indicator_map = {c: all_indicator_map[c] for c in y_cols if c in all_indicator_map}
    _y_with_indicator = list(indicator_map.keys())
    _y_without_indicator = [c for c in y_cols if c not in indicator_map]
    indicator_cols = list(indicator_map.values())
    pq_usecols = ["eid"] + y_cols + indicator_cols
    if len(pq_usecols) != len(set(pq_usecols)):
        dup_usecols = (
            pd.Series(pq_usecols)[pd.Series(pq_usecols).duplicated()].unique().tolist()
        )
        raise ValueError(
            f"Input validation or source fitting error: {dup_usecols[:10]}"
        )
    df_y = pd.read_parquet(y_parquet_path, columns=pq_usecols)
    df_x["eid"] = _normalize_eid(df_x["eid"], "UKB_final.csv")
    df_y["eid"] = _normalize_eid(df_y["eid"], "imputed parquet")
    if df_x["eid"].duplicated().any():
        raise ValueError("Input validation or source fitting error: ")
    if df_y["eid"].duplicated().any():
        raise ValueError("Input validation or source fitting error: ")
    df_x = df_x.copy()
    df_x["_order"] = np.arange(len(df_x))
    df_xy = df_x.merge(df_y, on="eid", how="inner", sort=False)
    df_xy = df_xy.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    if len(df_xy) == 0:
        raise ValueError("Input validation or source fitting error: ")
    X_edge = (
        df_xy[e_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    )
    if np.isnan(X_edge).any():
        raise ValueError("Input validation or source fitting error: ")
    Y = df_xy[y_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(Y).any():
        raise ValueError("Input validation or source fitting error: ")
    Y_mask = np.ones_like(Y, dtype=bool)
    for j, c in enumerate(y_cols):
        ind_col = indicator_map.get(c, None)
        if ind_col is not None:
            ind_num = pd.to_numeric(df_xy[ind_col], errors="coerce")
            if ind_num.isna().any():
                raise ValueError(f"{ind_col}Input validation or source fitting error: ")
            if not ind_num.isin([0, 1]).all():
                bad_vals = ind_num[~ind_num.isin([0, 1])].unique().tolist()
                raise ValueError(
                    f"{ind_col}Input validation or source fitting error: {bad_vals[:10]}"
                )
            Y_mask[:, j] = ind_num.to_numpy(dtype=np.int8) == 0
        else:
            Y_mask[:, j] = True
    site_vec = None
    if site_col is not None:
        if df_xy[site_col].isna().any():
            raise ValueError(f"{site_col}Input validation or source fitting error: ")
        site_vec = df_xy[site_col].astype(str).to_numpy()
        if np.unique(site_vec).size < 2:
            site_vec = None
    if not X_edge.shape[0] == Y.shape[0] == Y_mask.shape[0]:
        raise RuntimeError("aligned arrays have inconsistent row counts")
    if len(y_cols) == 1:
        pass
    eid_aligned = df_xy["eid"].astype(str).to_numpy()
    edge_names = list(e_names)
    return (X_edge, Y, Y_mask, y_cols, site_vec, eid_aligned, edge_names)


def prepare_stage1_split_and_y(
    X, Y, Y_mask, eid, site=None, valid_size=0.2, random_state=0
):
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Input validation or source fitting error: {X.shape[0]} vs {Y.shape[0]}"
        )
    if Y.shape != Y_mask.shape:
        raise ValueError(
            f"Input validation or source fitting error: {Y.shape} vs {Y_mask.shape}"
        )
    if len(eid) != X.shape[0]:
        raise ValueError(
            f"Input validation or source fitting error: {len(eid)} vs {X.shape[0]}"
        )
    if site is not None and len(site) != X.shape[0]:
        raise ValueError(
            f"Input validation or source fitting error: {len(site)} vs {X.shape[0]}"
        )
    if not 0 < valid_size < 1:
        raise ValueError(f"Input validation or source fitting error: {valid_size}")
    if random_state is None:
        raise ValueError("Input validation or source fitting error: ")
    Y_mask = np.asarray(Y_mask).astype(bool, copy=False)
    if not np.isfinite(X).all():
        raise ValueError("Input validation or source fitting error: ")
    if not np.isfinite(Y).all():
        raise ValueError("Input validation or source fitting error: ")
    use_stratify_valid = False
    split_cache_path = os.getenv("STAGE1_SPLIT_CACHE_NPZ", "").strip() or None
    row_idx = np.arange(X.shape[0], dtype=np.int64)
    if eid is None:
        raise ValueError("Input validation or source fitting error: ")
    eid_arr = np.asarray(eid)
    if eid_arr.ndim != 1:
        raise ValueError(f"Input validation or source fitting error: {eid_arr.ndim}")
    if eid_arr.shape[0] != X.shape[0]:
        raise ValueError(
            f"Input validation or source fitting error: {eid_arr.shape[0]} vs X.shape[0]={X.shape[0]}"
        )
    if np.unique(eid_arr).size != eid_arr.shape[0]:
        raise ValueError("Input validation or source fitting error: ")
    current_eid_sha1 = hashlib.sha1(
        "\n".join(eid_arr.astype(str).tolist()).encode("utf-8")
    ).hexdigest()
    if split_cache_path is not None and os.path.isfile(split_cache_path):
        with np.load(split_cache_path, allow_pickle=False) as split_npz:
            required = {
                "train_idx",
                "valid_idx",
                "n_subjects",
                "random_state",
                "valid_size",
                "eid_sha1",
            }
            missing = required.difference(set(split_npz.files))
            if missing:
                raise KeyError(
                    f"Input validation or source fitting error: {sorted(missing)}"
                )
            cache_n_subjects = int(np.asarray(split_npz["n_subjects"]).ravel()[0])
            cache_random_state = int(np.asarray(split_npz["random_state"]).ravel()[0])
            cache_valid_size = float(np.asarray(split_npz["valid_size"]).ravel()[0])
            cache_eid_sha1_raw = np.asarray(split_npz["eid_sha1"]).ravel()[0]
            if isinstance(cache_eid_sha1_raw, bytes):
                cache_eid_sha1 = cache_eid_sha1_raw.decode("utf-8")
            else:
                cache_eid_sha1 = str(cache_eid_sha1_raw)
            if cache_n_subjects != X.shape[0]:
                raise ValueError(
                    f"Input validation or source fitting error: {cache_n_subjects} vs {X.shape[0]}"
                )
            if cache_random_state != int(random_state):
                raise ValueError(
                    f"Input validation or source fitting error: {cache_random_state} vs {random_state}"
                )
            if not np.isclose(cache_valid_size, float(valid_size)):
                raise ValueError(
                    f"Input validation or source fitting error: {cache_valid_size} vs {valid_size}"
                )
            if cache_eid_sha1 != current_eid_sha1:
                raise ValueError("Input validation or source fitting error: ")
            train_idx = np.asarray(split_npz["train_idx"], dtype=np.int64).ravel()
            valid_idx = np.asarray(split_npz["valid_idx"], dtype=np.int64).ravel()
            if train_idx.size == 0 or valid_idx.size == 0:
                raise ValueError("Input validation or source fitting error: ")
            if train_idx.min() < 0 or valid_idx.min() < 0:
                raise ValueError("Input validation or source fitting error: ")
            if train_idx.max() >= X.shape[0] or valid_idx.max() >= X.shape[0]:
                raise ValueError("Input validation or source fitting error: ")
            if np.unique(train_idx).size != train_idx.size:
                raise ValueError("Input validation or source fitting error: ")
            if np.unique(valid_idx).size != valid_idx.size:
                raise ValueError("Input validation or source fitting error: ")
            if np.intersect1d(train_idx, valid_idx).size > 0:
                raise ValueError("Input validation or source fitting error: ")
            all_idx = np.concatenate([train_idx, valid_idx])
            if np.unique(all_idx).size != X.shape[0]:
                raise ValueError("Input validation or source fitting error: ")
            if all_idx.size != X.shape[0]:
                raise ValueError(
                    f"Input validation or source fitting error: {all_idx.size} vs {X.shape[0]}"
                )
            if "use_stratify_valid" in split_npz.files:
                raw_flag = np.asarray(split_npz["use_stratify_valid"]).ravel()[0]
                if isinstance(raw_flag, (np.bool_, bool)):
                    use_stratify_valid = bool(raw_flag)
                elif isinstance(raw_flag, (np.integer, int)):
                    use_stratify_valid = bool(int(raw_flag))
                else:
                    raise ValueError("Input validation or source fitting error: ")
    elif site is not None:
        site_arr = np.asarray(site)
        if site_arr.ndim != 1:
            raise ValueError(
                f"Input validation or source fitting error: {site_arr.ndim}"
            )
        if site_arr.shape[0] != X.shape[0]:
            raise ValueError(
                f"Input validation or source fitting error: {site_arr.shape[0]} vs X.shape[0]={X.shape[0]}"
            )
        try:
            train_idx, valid_idx, _, _, _, _ = train_test_split(
                row_idx,
                eid_arr,
                site_arr,
                test_size=valid_size,
                random_state=random_state,
                stratify=site_arr,
            )
            use_stratify_valid = True
        except ValueError as _e:
            train_idx, valid_idx, _, _, _, _ = train_test_split(
                row_idx,
                eid_arr,
                site_arr,
                test_size=valid_size,
                random_state=random_state,
                shuffle=True,
            )
            use_stratify_valid = False
    else:
        train_idx, valid_idx, _, _ = train_test_split(
            row_idx,
            eid_arr,
            test_size=valid_size,
            random_state=random_state,
            shuffle=True,
        )
    X_train = X[train_idx]
    X_valid = X[valid_idx]
    Y_train = Y[train_idx]
    Y_valid = Y[valid_idx]
    M_train = Y_mask[train_idx]
    M_valid = Y_mask[valid_idx]
    eid_train = eid_arr[train_idx]
    eid_valid = eid_arr[valid_idx]
    if site is not None:
        site_arr = np.asarray(site)
        site_train = site_arr[train_idx]
        site_valid = site_arr[valid_idx]
    else:
        site_train, site_valid = (None, None)
    n_targets = Y.shape[1]
    Y_train_z = np.full(Y_train.shape, np.nan, dtype=np.float32)
    Y_valid_z = np.full(Y_valid.shape, np.nan, dtype=np.float32)
    y_mu = np.full((n_targets,), np.nan, dtype=np.float32)
    y_sigma = np.full((n_targets,), np.nan, dtype=np.float32)
    n_train_obs_per_target = np.zeros((n_targets,), dtype=np.int32)
    for j in range(n_targets):
        tr_obs = M_train[:, j]
        va_obs = M_valid[:, j]
        n_tr_obs = int(tr_obs.sum())
        n_train_obs_per_target[j] = n_tr_obs
        if n_tr_obs < 2:
            raise ValueError(
                f"Input validation or source fitting error: {j}Input validation or source fitting error: {n_tr_obs}Input validation or source fitting error: "
            )
        elif n_tr_obs < 10:
            pass
        train_j = Y_train[tr_obs, j].astype(np.float64)
        mu_j = float(np.mean(train_j))
        centered_train = train_j - mu_j
        sigma_j = float(np.std(centered_train, ddof=0))
        Y_train_z[tr_obs, j] = centered_train.astype(np.float32)
        if np.any(va_obs):
            Y_valid_z[va_obs, j] = (
                Y_valid[va_obs, j].astype(np.float64) - mu_j
            ).astype(np.float32)
        if sigma_j > 0:
            Y_train_z[tr_obs, j] /= np.float32(sigma_j)
            if np.any(va_obs):
                Y_valid_z[va_obs, j] /= np.float32(sigma_j)
        else:
            Y_train_z[tr_obs, j] = 0.0
            if np.any(va_obs):
                Y_valid_z[va_obs, j] = 0.0
        y_mu[j] = np.float32(mu_j)
        y_sigma[j] = np.float32(sigma_j)
    if site is not None:
        pass
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "Y_train": Y_train,
        "Y_valid": Y_valid,
        "M_train": M_train,
        "M_valid": M_valid,
        "Y_train_z": Y_train_z,
        "Y_valid_z": Y_valid_z,
        "y_mu": y_mu,
        "y_sigma": y_sigma,
        "n_train_obs_per_target": n_train_obs_per_target,
        "eid_train": np.asarray(eid_train).astype(str),
        "eid_valid": np.asarray(eid_valid).astype(str),
        "site_train": site_train,
        "site_valid": site_valid,
        "use_stratify_valid": use_stratify_valid,
    }


def _unitize_rows(A):
    A = np.asarray(A, dtype=np.float64)
    A = A - A.mean(axis=1, keepdims=True)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return A / n


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = np.nan
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    cod = float(1.0 - sse / sst) if sst > 0 else np.nan
    return {"R²": r2, "RMSE": rmse, "Pearson_r": r, "COD": cod}


def _predictive_cod_znormed(y_true_z, y_pred_z):
    y_true_z = np.asarray(y_true_z, dtype=np.float64).ravel()
    y_pred_z = np.asarray(y_pred_z, dtype=np.float64).ravel()
    if y_true_z.size == 0 or y_pred_z.size == 0:
        return np.nan
    if y_true_z.shape != y_pred_z.shape:
        raise ValueError(
            f"Input validation or source fitting error: {y_true_z.shape} vs {y_pred_z.shape}"
        )
    if not np.isfinite(y_true_z).all() or not np.isfinite(y_pred_z).all():
        return np.nan
    denom = float(np.sum(y_true_z**2))
    if denom <= 0:
        return np.nan
    sse = float(np.sum((y_true_z - y_pred_z) ** 2))
    return float(1.0 - sse / denom)


def _fit_stage1_single_target(
    K_train_full,
    K_valid_full,
    y_train_z_col,
    y_valid_z_col,
    m_train_col,
    m_valid_col,
    alphas,
    target_name,
):
    tr_obs = np.asarray(m_train_col).astype(bool, copy=False)
    va_obs = np.asarray(m_valid_col).astype(bool, copy=False)
    idx_tr = np.where(tr_obs)[0]
    idx_va = np.where(va_obs)[0]
    n_tr = int(idx_tr.size)
    n_va = int(idx_va.size)
    if n_tr < 2:
        raise ValueError(
            f"{target_name}Input validation or source fitting error: {n_tr}Input validation or source fitting error: "
        )
    if n_va < 1:
        raise ValueError(f"{target_name}Input validation or source fitting error: ")
    if alphas is None or len(alphas) == 0:
        raise ValueError(f"{target_name}Input validation or source fitting error: ")
    ytr = np.asarray(y_train_z_col[tr_obs], dtype=np.float64).ravel()
    yva = np.asarray(y_valid_z_col[va_obs], dtype=np.float64).ravel()
    if not np.isfinite(ytr).all():
        raise ValueError(f"{target_name}Input validation or source fitting error: ")
    if not np.isfinite(yva).all():
        raise ValueError(f"{target_name}Input validation or source fitting error: ")
    if K_train_full.shape[0] != K_train_full.shape[1]:
        raise ValueError(
            f"{target_name}Input validation or source fitting error: {K_train_full.shape}"
        )
    if K_valid_full.shape[1] != K_train_full.shape[0]:
        raise ValueError(
            f"{target_name}Input validation or source fitting error: {K_valid_full.shape} vs {K_train_full.shape}"
        )
    if idx_tr.max() >= K_train_full.shape[0] or idx_va.max() >= K_valid_full.shape[0]:
        raise ValueError(f"{target_name}Input validation or source fitting error: ")
    Ktr = K_train_full[np.ix_(idx_tr, idx_tr)]
    Kva = K_valid_full[np.ix_(idx_va, idx_tr)]
    if not np.isfinite(Ktr).all():
        raise ValueError(f"{target_name}Input validation or source fitting error: ")
    if not np.isfinite(Kva).all():
        raise ValueError(f"{target_name}Input validation or source fitting error: ")
    best_alpha = None
    best_score = None
    best_model = None
    best_pred = None
    for a in alphas:
        model = KernelRidge(kernel="precomputed", alpha=float(a))
        model.fit(Ktr, ytr)
        pred = model.predict(Kva)
        if not np.isfinite(pred).all():
            continue
        score = _predictive_cod_znormed(yva, pred)
        if not np.isfinite(score):
            continue
        if best_score is None or score > best_score:
            best_score = float(score)
            best_alpha = float(a)
            best_model = model
            best_pred = np.asarray(pred, dtype=np.float64)
    if (
        best_alpha is None
        or best_model is None
        or best_pred is None
        or (best_score is None)
    ):
        fallback_alpha = float(alphas[0])
        fallback_model = KernelRidge(kernel="precomputed", alpha=fallback_alpha)
        fallback_model.fit(Ktr, ytr)
        fallback_pred = fallback_model.predict(Kva)
        if not np.isfinite(fallback_pred).all():
            raise ValueError(
                f"{target_name}: fallback alpha={fallback_alpha}Input validation or source fitting error: "
            )
        fallback_dual_coef = np.asarray(
            fallback_model.dual_coef_, dtype=np.float64
        ).ravel()
        if fallback_dual_coef.ndim != 1 or fallback_dual_coef.size != n_tr:
            raise ValueError(
                f"{target_name}Input validation or source fitting error: {fallback_dual_coef.size}, n_tr={n_tr}"
            )
        valid_metrics = compute_metrics(yva, fallback_pred)
        valid_metrics["selection_metric_name"] = "predictive_COD_valid_z"
        valid_metrics["selection_metric_value"] = None
        valid_metrics["n_train_obs"] = int(n_tr)
        valid_metrics["n_valid_obs"] = int(n_va)
        valid_metrics["fallback_used"] = True
        valid_metrics["failure_reason"] = "all_validation_scores_invalid"
        return {
            "dual_coef": fallback_dual_coef,
            "train_obs_idx": idx_tr.astype(np.int32),
            "valid_obs_idx": idx_va.astype(np.int32),
            "y_valid_true_z": yva.astype(np.float32),
            "y_valid_pred_z": np.asarray(fallback_pred, dtype=np.float32),
            "alpha": fallback_alpha,
            "metrics": valid_metrics,
        }
    valid_metrics = compute_metrics(yva, best_pred)
    valid_metrics["selection_metric_name"] = "predictive_COD_valid_z"
    valid_metrics["selection_metric_value"] = float(best_score)
    valid_metrics["n_train_obs"] = int(n_tr)
    valid_metrics["n_valid_obs"] = int(n_va)
    valid_metrics["fallback_used"] = False
    valid_metrics["failure_reason"] = None
    best_dual_coef = np.asarray(best_model.dual_coef_, dtype=np.float64).ravel()
    if best_dual_coef.ndim != 1 or best_dual_coef.size != n_tr:
        raise ValueError(
            f"{target_name}Input validation or source fitting error: {best_dual_coef.size}, n_tr={n_tr}"
        )
    return {
        "dual_coef": best_dual_coef,
        "train_obs_idx": idx_tr.astype(np.int32),
        "valid_obs_idx": idx_va.astype(np.int32),
        "y_valid_true_z": yva.astype(np.float32),
        "y_valid_pred_z": best_pred.astype(np.float32),
        "alpha": best_alpha,
        "metrics": valid_metrics,
    }


def train_stage1_krr(
    X, Y, Y_mask, y_cols, edge_names, eid, site=None, valid_size=0.2, random_state=0
):
    if Y.ndim != 2:
        raise ValueError(f"Input validation or source fitting error: {Y.shape}")
    if Y_mask.shape != Y.shape:
        raise ValueError(
            f"Input validation or source fitting error: {Y_mask.shape} vs {Y.shape}"
        )
    if len(y_cols) != Y.shape[1]:
        raise ValueError(
            f"len(y_cols)={len(y_cols)}Input validation or source fitting error: {Y.shape[1]}Input validation or source fitting error: "
        )
    if len(set(y_cols)) != len(y_cols):
        raise ValueError("Input validation or source fitting error: ")
    if len(edge_names) != X.shape[1]:
        raise ValueError(
            f"len(edge_names)={len(edge_names)}Input validation or source fitting error: {X.shape[1]}Input validation or source fitting error: "
        )
    if len(eid) != X.shape[0]:
        raise ValueError(
            f"len(eid)={len(eid)}Input validation or source fitting error: {X.shape[0]}Input validation or source fitting error: "
        )
    alphas_env = os.getenv("ALPHAS", "").strip()
    if alphas_env:
        try:
            alphas = [float(a.strip()) for a in alphas_env.split(",") if a.strip()]
        except Exception as e:
            raise ValueError(
                f"Input validation or source fitting error: {alphas_env}"
            ) from e
        if len(alphas) == 0:
            raise ValueError("Input validation or source fitting error: ")
    else:
        alphas = [
            1e-05,
            0.0001,
            0.001,
            0.004,
            0.007,
            0.01,
            0.04,
            0.07,
            0.1,
            0.4,
            0.7,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            10.0,
            15.0,
            20.0,
        ]
    split_pack = prepare_stage1_split_and_y(
        X, Y, Y_mask, eid, site=site, valid_size=valid_size, random_state=random_state
    )
    Z_train_full = _unitize_rows(split_pack["X_train"])
    Z_valid_full = _unitize_rows(split_pack["X_valid"])
    if not np.isfinite(Z_train_full).all():
        raise ValueError("Input validation or source fitting error: ")
    if not np.isfinite(Z_valid_full).all():
        raise ValueError("Input validation or source fitting error: ")
    K_train_full = Z_train_full @ Z_train_full.T
    K_valid_full = Z_valid_full @ Z_train_full.T
    if not np.isfinite(K_train_full).all():
        raise ValueError("Input validation or source fitting error: ")
    if not np.isfinite(K_valid_full).all():
        raise ValueError("Input validation or source fitting error: ")
    target_models = {}
    target_summaries = []
    for j, target_name in enumerate(y_cols):
        try:
            ret_j = _fit_stage1_single_target(
                K_train_full=K_train_full,
                K_valid_full=K_valid_full,
                y_train_z_col=split_pack["Y_train_z"][:, j],
                y_valid_z_col=split_pack["Y_valid_z"][:, j],
                m_train_col=split_pack["M_train"][:, j],
                m_valid_col=split_pack["M_valid"][:, j],
                alphas=alphas,
                target_name=target_name,
            )
        except Exception as e:
            raise RuntimeError(
                f"Input validation or source fitting error: {target_name}"
            ) from e
        target_models[target_name] = {
            "dual_coef": ret_j["dual_coef"],
            "train_obs_idx": ret_j["train_obs_idx"],
            "fit_train_idx": ret_j["train_obs_idx"],
            "valid_obs_idx": ret_j["valid_obs_idx"],
            "best_alpha": ret_j["alpha"],
            "y_mu": float(split_pack["y_mu"][j]),
            "y_sigma": float(split_pack["y_sigma"][j]),
            "n_train_obs": int(ret_j["metrics"]["n_train_obs"]),
        }
        selection_metric_name = ret_j["metrics"].get(
            "selection_metric_name", "predictive_COD_valid_z"
        )
        selection_metric_value = ret_j["metrics"].get("selection_metric_value", None)
        target_summaries.append(
            {
                "target": target_name,
                "best_alpha": ret_j["alpha"],
                "n_train_obs": int(ret_j["metrics"]["n_train_obs"]),
                "n_valid_obs": int(ret_j["metrics"]["n_valid_obs"]),
                "R2_valid_z": float(ret_j["metrics"]["R²"]),
                "RMSE_valid_z": float(ret_j["metrics"]["RMSE"]),
                "Pearson_r_valid_z": float(ret_j["metrics"]["Pearson_r"]),
                "COD_valid_z": float(ret_j["metrics"]["COD"]),
                "selection_metric_name": selection_metric_name,
                "predictive_COD_valid_z": (
                    None
                    if selection_metric_value is None
                    else float(selection_metric_value)
                ),
                "fallback_used": bool(ret_j["metrics"].get("fallback_used", False)),
                "failure_reason": ret_j["metrics"].get("failure_reason", None),
            }
        )
    del K_train_full, K_valid_full
    return {
        "Z_train_full": Z_train_full,
        "edge_names": list(edge_names),
        "eid_train_full": split_pack["eid_train"],
        "eid_valid": split_pack["eid_valid"],
        "target_models": target_models,
        "target_summaries": target_summaries,
        "y_cols": y_cols,
        "use_stratify_valid": split_pack["use_stratify_valid"],
        "valid_size": float(valid_size),
        "random_state": int(random_state),
        "alphas": alphas,
        "n_edges": int(X.shape[1]),
        "n_train_subjects": int(Z_train_full.shape[0]),
        "n_valid_subjects": int(split_pack["X_valid"].shape[0]),
        "feature_transform": "unitize_rows",
        "kernel_type": "corr_precomputed",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ukb-csv", required=True)
    p.add_argument("--targets-parquet", required=True)
    p.add_argument("--freeze-tsv", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--config", default=str(Path(__file__).parent / "config/raw_krr_training.json")
    )
    p.add_argument("--split-cache-npz")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    os.environ["STAGE1_Y_PARQUET"] = args.targets_parquet
    os.environ["STAGE1_FREEZE_BASE"] = args.freeze_tsv
    os.environ["ALPHAS"] = ",".join((str(a) for a in config["alphas"]))
    if args.split_cache_npz:
        os.environ["STAGE1_SPLIT_CACHE_NPZ"] = args.split_cache_npz
    else:
        os.environ.pop("STAGE1_SPLIT_CACHE_NPZ", None)
    X, Y, mask, names, site, ids, edges = load_data(
        args.ukb_csv, target_name=args.target
    )
    if len(X) != config["n_subjects"] or len(edges) != 1485:
        raise ValueError("Source cohort dimensions differ from the recorded analysis.")
    result = train_stage1_krr(
        X,
        Y,
        mask,
        names,
        edges,
        ids,
        site=site,
        valid_size=config["valid_size"],
        random_state=config["random_state"],
    )
    if result["use_stratify_valid"] != config["use_stratify_valid"]:
        raise ValueError(
            "Source split stratification differs from the recorded analysis."
        )
    metadata = {
        k: result[k]
        for k in [
            "n_edges",
            "n_train_subjects",
            "n_valid_subjects",
            "valid_size",
            "random_state",
            "use_stratify_valid",
            "alphas",
            "feature_transform",
            "kernel_type",
        ]
    }
    metadata["selection_metric_name"] = "predictive_COD_valid_z"
    token = re.sub("[^A-Za-z0-9._-]+", "_", args.target).strip("_")[:120]
    bundle = {
        "bundle_type": "ukb_stage1_krr_transfer_bundle",
        "run_id": f"seed0__target_{token}",
        "metadata": metadata,
        "y_cols": result["y_cols"],
        "edge_names": result["edge_names"],
        "eid_train_full": result["eid_train_full"],
        "eid_valid": result["eid_valid"],
        "Z_train_full": result["Z_train_full"],
        "target_models": result["target_models"],
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / f"source_model_{token}.joblib")
    (out / f"source_model_{token}.json").write_text(
        json.dumps(
            {"metadata": metadata, "target_summaries": result["target_summaries"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
