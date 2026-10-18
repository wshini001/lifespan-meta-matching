"""Fit fine-motor RSFC KRR models on locked child-grouped folds.

Uses eight outcome covariates, training-mean education imputation and
training-partition outcome residualization. Cohort arrays use float32.
"""

import json
import re
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def str2bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {
            "n": 0,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Pearson_r": np.nan,
            "COD": np.nan,
            "MSE": np.nan,
            "neg_MSE": np.nan,
        }
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = np.nan
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    cod = float(1.0 - sse / sst) if sst > 0 else np.nan
    return {
        "n": int(len(y_true)),
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson_r": r,
        "COD": cod,
        "MSE": mse,
        "neg_MSE": -mse,
    }


def score_for_alpha(y_true, y_pred, metric):
    m = compute_metrics(y_true, y_pred)
    if metric == "pearson":
        val = m["Pearson_r"]
    elif metric == "cod":
        val = m["COD"]
    elif metric == "neg_mse":
        val = m["neg_MSE"]
    else:
        raise ValueError(metric)
    if not np.isfinite(val):
        return -np.inf
    return float(val)


def unitize_rows(A):
    A = np.asarray(A, dtype=np.float64)
    A = A - A.mean(axis=1, keepdims=True)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return A / n


def make_balanced_group_folds(groups, n_splits=5, seed=0):
    groups = np.asarray(groups).astype(str)
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Not enough unique groups for {n_splits}-fold CV: n_unique_groups={len(unique_groups)}"
        )
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(unique_groups))
    rng.shuffle(order)
    shuffled_groups = unique_groups[order]
    shuffled_counts = group_counts[order]
    fold_group_sets = [set() for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    for g, cnt in zip(shuffled_groups, shuffled_counts):
        k = int(np.argmin(fold_sizes))
        fold_group_sets[k].add(g)
        fold_sizes[k] += int(cnt)
    splits = []
    all_idx = np.arange(len(groups))
    for k in range(n_splits):
        test_mask = np.isin(groups, list(fold_group_sets[k]))
        te = all_idx[test_mask]
        tr = all_idx[~test_mask]
        if len(te) == 0 or len(tr) == 0:
            raise RuntimeError(f"Empty fold generated at fold {k}")
        splits.append((tr, te))
    return splits


def build_covariates_full_plus_motheredu(df):
    cov = pd.DataFrame(index=df.index)
    age = pd.to_numeric(df["scan_age_months"], errors="coerce").astype(float)
    age_sq = age**2
    sex_str = df["sex_final"].astype(str).str.upper().str.strip()
    sex_male = sex_str.map({"F": 0.0, "M": 1.0}).astype(float)
    fd = pd.to_numeric(df["FD"], errors="coerce").astype(float)
    site = df["site_primary"].astype(str).str.strip()
    site_dum = pd.get_dummies(site, prefix="site", drop_first=True).astype(float)
    mother_edu = pd.to_numeric(df["mother_edu_gt_college"], errors="coerce").astype(
        float
    )
    cov["scan_age_months"] = age
    cov["age_sq"] = age_sq
    cov["sex_male"] = sex_male
    cov["FD"] = fd
    cov = pd.concat([cov, site_dum], axis=1)
    cov["age_sex"] = cov["scan_age_months"] * cov["sex_male"]
    cov["age_sq_sex"] = cov["age_sq"] * cov["sex_male"]
    cov["mother_edu_gt_college"] = mother_edu
    return cov


def impute_covariates_train_test(C_train, C_test):
    C_train = np.asarray(C_train, dtype=float)
    C_test = np.asarray(C_test, dtype=float)
    means = np.nanmean(C_train, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    Ctr = C_train.copy()
    Cte = C_test.copy()
    tr_bad = ~np.isfinite(Ctr)
    te_bad = ~np.isfinite(Cte)
    if tr_bad.any():
        Ctr[tr_bad] = np.take(means, np.where(tr_bad)[1])
    if te_bad.any():
        Cte[te_bad] = np.take(means, np.where(te_bad)[1])
    if not np.isfinite(Ctr).all() or not np.isfinite(Cte).all():
        raise RuntimeError(
            "Non-finite covariates remain after train-fold mean imputation."
        )
    return (Ctr, Cte, means)


def residualize_train_test(y_train, y_test, C_train, C_test):
    Ctr_imp, Cte_imp, means = impute_covariates_train_test(C_train, C_test)
    lr = LinearRegression()
    lr.fit(Ctr_imp, y_train)
    y_train_res = y_train - lr.predict(Ctr_imp)
    y_test_res = y_test - lr.predict(Cte_imp)
    return (y_train_res, y_test_res, means, lr)


def load_bcp_data(cohort_csv, target_col, completecase_col, group_col):
    df = pd.read_csv(cohort_csv)
    df["_row_idx_locked"] = np.arange(len(df), dtype=np.int64)
    required = [
        target_col,
        completecase_col,
        group_col,
        "scan_id",
        "scan_age_months",
        "sex_final",
        "FD",
        "site_primary",
        "mother_edu_gt_college",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    edge_cols = []
    for c in df.columns:
        m = re.fullmatch("e(\\d+)", str(c))
        if m:
            edge_cols.append((int(m.group(1)), c))
    edge_cols.sort(key=lambda x: x[0])
    edge_names = [c for _, c in edge_cols]
    if len(edge_names) != 1485:
        raise ValueError(f"Expected 1485 edge columns, found {len(edge_names)}")
    mask = str2bool_series(df[completecase_col])
    df_use = df.loc[mask].copy().reset_index(drop=True)
    if len(df_use) == 0:
        raise RuntimeError("No rows left after complete-case filtering.")
    E = df_use[edge_names].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    y = pd.to_numeric(df_use[target_col], errors="coerce").astype(np.float32)
    cov = build_covariates_full_plus_motheredu(df_use)
    non_edu_cov_cols = [c for c in cov.columns if c != "mother_edu_gt_college"]
    bad_edge = E.isna().any(axis=1)
    bad_y = y.isna()
    bad_non_edu_cov = cov[non_edu_cov_cols].isna().any(axis=1)
    if bad_edge.any() or bad_y.any() or bad_non_edu_cov.any():
        msg = {
            "bad_edge_rows": int(bad_edge.sum()),
            "bad_target_rows": int(bad_y.sum()),
            "bad_non_edu_covariate_rows": int(bad_non_edu_cov.sum()),
            "non_edu_covariates": non_edu_cov_cols,
        }
        raise RuntimeError(
            "Complete-case filtering still left missing values outside mother_edu: "
            + json.dumps(msg, indent=2)
        )
    X = np.hstack([E.to_numpy(np.float32), cov.to_numpy(np.float32)]).astype(np.float32)
    y_np = y.to_numpy(np.float32)
    groups = df_use[group_col].astype(str).to_numpy()
    meta_cols = [
        "_row_idx_locked",
        "scan_id",
        "imaging_subject_stem",
        "imaging_numeric_id",
        "scan_age_days",
        "scan_age_months",
        "FD",
        "TR",
        "site_primary",
        "sex_final",
        "PSCID",
        "CandID",
        "Candidate_Age",
        "Candidate_Age_num",
        "age_diff_months",
        target_col,
        "mother_education_raw",
        "mother_edu_gt_college",
        "flag_duplicate_numeric_age",
        "n_same_numeric_age",
        "flag_numeric_id_prefix_conflict",
        "n_unique_imaging_stems",
    ]
    meta_cols = [c for c in meta_cols if c in df_use.columns]
    meta = df_use[meta_cols].copy()
    info = {
        "n": int(len(df_use)),
        "n_unique_groups": int(len(np.unique(groups))),
        "n_edge": int(len(edge_names)),
        "n_cov": int(cov.shape[1]),
        "edge_names_n": int(len(edge_names)),
        "covariates": list(cov.columns),
        "mother_edu_nonmissing_scans": int(
            df_use["mother_edu_gt_college"].notna().sum()
        ),
        "mother_edu_missing_scans": int(df_use["mother_edu_gt_college"].isna().sum()),
        "mother_edu_value_counts": {
            str(k): int(v)
            for k, v in df_use["mother_edu_gt_college"]
            .value_counts(dropna=False)
            .items()
        },
    }
    return {
        "X": X,
        "y": y_np,
        "groups": groups,
        "meta": meta,
        "edge_names": edge_names,
        "cov_cols": list(cov.columns),
        "n_edge": len(edge_names),
        "n_cov": cov.shape[1],
        "df_use": df_use,
        "info": info,
    }


def load_outer_folds_from_manifest(meta, fold_manifest_csv):
    fm = pd.read_csv(fold_manifest_csv)
    key = ["scan_id", "imaging_numeric_id"]
    for c in key + ["fold"]:
        if c not in fm.columns:
            raise KeyError(f"fold manifest missing column: {c}")
    for c in key:
        if c not in meta.columns:
            raise KeyError(f"meta missing column: {c}")
    if fm[key].astype(str).duplicated().any():
        raise RuntimeError(
            "Fold manifest has duplicate scan_id/imaging_numeric_id keys."
        )
    leak = fm.groupby("imaging_numeric_id")["fold"].nunique()
    if int((leak > 1).sum()) > 0:
        raise RuntimeError("Subject leakage detected in supplied fold manifest.")
    tmp = meta.reset_index().copy()
    for c in key:
        tmp[c] = tmp[c].astype(str)
        fm[c] = fm[c].astype(str)
    mm = tmp.merge(fm[key + ["fold"]], on=key, how="left", validate="one_to_one")
    if mm["fold"].isna().any():
        bad = mm.loc[mm["fold"].isna(), key].head(10)
        raise RuntimeError(f"Some rows are missing fold assignment:\n{bad}")
    mm = mm.sort_values("index")
    fold_id = mm["fold"].astype(int).to_numpy()
    all_idx = np.arange(len(meta))
    folds = []
    for f in sorted(np.unique(fold_id)):
        te = all_idx[fold_id == f]
        tr = all_idx[fold_id != f]
        if len(te) == 0 or len(tr) == 0:
            raise RuntimeError(f"Empty fold from manifest: {f}")
        folds.append((tr, te))
    return (folds, fold_id)


def inner_select_alpha(
    X_train_val,
    y_train_val,
    groups_train_val,
    n_edge,
    n_cov,
    alphas,
    inner_k,
    seed,
    alpha_select_metric,
):
    n_unique_groups = len(np.unique(groups_train_val))
    n_inner = min(inner_k, n_unique_groups)
    if n_inner < 2:
        raise RuntimeError("Need at least 2 unique groups for inner CV.")
    inner_splits = make_balanced_group_folds(
        groups_train_val, n_splits=n_inner, seed=seed
    )
    alpha_scores = {float(a): [] for a in alphas}
    impute_rows = []
    for inner_i, (tr, va) in enumerate(inner_splits):
        Xtr_edge = X_train_val[tr, :n_edge]
        Xva_edge = X_train_val[va, :n_edge]
        if n_cov > 0:
            Ctr = X_train_val[tr, n_edge:]
            Cva = X_train_val[va, n_edge:]
            ytr_res, yva_res, cov_means, _ = residualize_train_test(
                y_train_val[tr], y_train_val[va], Ctr, Cva
            )
            impute_rows.append(
                {
                    "inner_fold": int(inner_i),
                    **{
                        f"mean_impute_cov_{j}": float(v)
                        for j, v in enumerate(cov_means)
                    },
                }
            )
        else:
            ytr_res = y_train_val[tr]
            yva_res = y_train_val[va]
        Ztr = unitize_rows(Xtr_edge)
        Zva = unitize_rows(Xva_edge)
        Ktr = Ztr @ Ztr.T
        Kva = Zva @ Ztr.T
        for a in alphas:
            a = float(a)
            model = KernelRidge(kernel="precomputed", alpha=a)
            model.fit(Ktr, ytr_res)
            pred = model.predict(Kva)
            score = score_for_alpha(yva_res, pred, alpha_select_metric)
            alpha_scores[a].append(score)
    rows = []
    for a, scores in alpha_scores.items():
        rows.append(
            {
                "alpha": float(a),
                f"mean_inner_{alpha_select_metric}": float(np.mean(scores)),
                f"sd_inner_{alpha_select_metric}": (
                    float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                ),
                "n_inner_folds": int(len(scores)),
            }
        )
    score_col = f"mean_inner_{alpha_select_metric}"
    score_df = (
        pd.DataFrame(rows)
        .sort_values([score_col, "alpha"], ascending=[False, True])
        .reset_index(drop=True)
    )
    best_alpha = float(score_df.iloc[0]["alpha"])
    best_score = float(score_df.iloc[0][score_col])
    impute_df = pd.DataFrame(impute_rows) if len(impute_rows) else pd.DataFrame()
    return (best_alpha, best_score, score_df, impute_df)


def fit_eval_outer(
    X,
    y,
    groups,
    meta,
    n_edge,
    n_cov,
    outer_splits,
    fold_id,
    alphas,
    inner_k,
    seed,
    alpha_select_metric,
    cov_cols,
):
    all_pred_rows = []
    fold_metric_rows = []
    inner_rows_all = []
    impute_rows_all = []
    for fold, (tr, te) in enumerate(outer_splits):
        print(
            f"\n[OUTER {fold}] train n={len(tr)}, test n={len(te)}, train groups={len(np.unique(groups[tr]))}, test groups={len(np.unique(groups[te]))}"
        )
        X_train_val = X[tr]
        y_train_val = y[tr]
        groups_train_val = groups[tr]
        X_test = X[te]
        y_test = y[te]
        best_alpha, best_inner_score, inner_score_df, inner_impute_df = (
            inner_select_alpha(
                X_train_val=X_train_val,
                y_train_val=y_train_val,
                groups_train_val=groups_train_val,
                n_edge=n_edge,
                n_cov=n_cov,
                alphas=alphas,
                inner_k=inner_k,
                seed=seed + 1000 + fold,
                alpha_select_metric=alpha_select_metric,
            )
        )
        inner_score_df["outer_fold"] = fold
        inner_score_df["alpha_select_metric"] = alpha_select_metric
        inner_rows_all.append(inner_score_df)
        if len(inner_impute_df):
            inner_impute_df["outer_fold"] = fold
            impute_rows_all.append(inner_impute_df)
        print(
            f"[OUTER {fold}] best alpha={best_alpha}, best inner {alpha_select_metric}={best_inner_score:.6f}"
        )
        if n_cov > 0:
            y_train_res, y_test_res, outer_cov_means, _ = residualize_train_test(
                y_train_val, y_test, X_train_val[:, n_edge:], X_test[:, n_edge:]
            )
        else:
            y_train_res = y_train_val
            y_test_res = y_test
            outer_cov_means = np.array([], dtype=float)
        Z_train = unitize_rows(X_train_val[:, :n_edge])
        Z_test = unitize_rows(X_test[:, :n_edge])
        K_train = Z_train @ Z_train.T
        K_test = Z_test @ Z_train.T
        model = KernelRidge(kernel="precomputed", alpha=best_alpha)
        model.fit(K_train, y_train_res)
        y_pred = model.predict(K_test)
        metrics = compute_metrics(y_test_res, y_pred)
        metrics["fold"] = int(fold)
        metrics["alpha"] = float(best_alpha)
        metrics[f"best_inner_{alpha_select_metric}"] = float(best_inner_score)
        metrics["alpha_select_metric"] = alpha_select_metric
        metrics["n_train"] = int(len(tr))
        metrics["n_test"] = int(len(te))
        metrics["n_train_groups"] = int(len(np.unique(groups[tr])))
        metrics["n_test_groups"] = int(len(np.unique(groups[te])))
        for name, val in zip(cov_cols, outer_cov_means):
            metrics[f"outer_train_mean_impute_{name}"] = float(val)
        fold_metric_rows.append(metrics)
        pred_meta = meta.iloc[te].reset_index(drop=True).copy()
        pred_meta.insert(0, "subj_idx", te)
        pred_meta["fold"] = fold
        pred_meta["y_true_raw"] = y_test
        pred_meta["y_true_resid"] = y_test_res
        pred_meta["y_pred"] = y_pred
        pred_meta["alpha"] = best_alpha
        pred_meta[f"best_inner_{alpha_select_metric}"] = best_inner_score
        pred_meta["alpha_select_metric"] = alpha_select_metric
        for name, val in zip(cov_cols, outer_cov_means):
            pred_meta[f"outer_train_mean_impute_{name}"] = float(val)
        all_pred_rows.append(pred_meta)
        print(f"[OUTER {fold}] metrics:", metrics)
    pred_df = pd.concat(all_pred_rows, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    inner_scores_df = pd.concat(inner_rows_all, ignore_index=True)
    impute_df = (
        pd.concat(impute_rows_all, ignore_index=True)
        if len(impute_rows_all)
        else pd.DataFrame()
    )
    pooled = compute_metrics(
        pred_df["y_true_resid"].to_numpy(), pred_df["y_pred"].to_numpy()
    )
    fold_summary = {}
    for metric in ["R2", "RMSE", "MAE", "Pearson_r", "COD", "MSE"]:
        vals = fold_metrics_df[metric].to_numpy(float)
        fold_summary[f"{metric}_mean"] = float(np.nanmean(vals))
        fold_summary[f"{metric}_sd"] = (
            float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        )
    return {
        "pred_df": pred_df,
        "fold_metrics_df": fold_metrics_df,
        "inner_scores_df": inner_scores_df,
        "impute_df": impute_df,
        "pooled": pooled,
        "fold_summary": fold_summary,
        "fold_id": fold_id,
    }
