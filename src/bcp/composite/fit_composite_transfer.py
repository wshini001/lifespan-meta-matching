"""Fit composite prediction branches using connectivity and transferred features.

Uses supplied outer folds and an ordered raw-feature list. PC kernel weights
are fixed at zero for this raw-source branch. Outcome residualization and feature scaling are fitted on each outer
training set before inner kernel-parameter selection.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    sse = np.sum((y - p) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    cod = 1.0 - sse / sst if sst > 0 else np.nan
    r = (
        np.corrcoef(y, p)[0, 1]
        if len(y) > 2 and np.std(y) > 0 and (np.std(p) > 0)
        else np.nan
    )
    mse = mean_squared_error(y, p)
    return {
        "n": int(len(y)),
        "COD": float(cod),
        "R2": float(cod),
        "Pearson_r": float(r),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y, p)),
        "MSE": float(mse),
        "neg_MSE": float(-mse),
    }


def unitize_rows(X):
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n <= 0] = 1.0
    return X / n


def zscore_train_test(X_train, X_test):
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd[sd <= 0] = 1.0
    return ((X_train - mu) / sd, (X_test - mu) / sd)


def make_kernel_train_test(X_train, X_test, mode):
    if mode == "unitize_corr":
        Ztr = unitize_rows(X_train)
        Zte = unitize_rows(X_test)
        return (Ztr @ Ztr.T, Zte @ Ztr.T)
    if mode == "zscore_linear":
        Ztr, Zte = zscore_train_test(X_train, X_test)
        scale = max(Ztr.shape[1], 1)
        return (Ztr @ Ztr.T / scale, Zte @ Ztr.T / scale)
    raise ValueError(f"Unknown kernel mode: {mode}")


def grouped_folds(groups, k, seed):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise RuntimeError(
            f"Need at least 2 unique groups for grouped CV, got {len(uniq)}."
        )
    k = min(int(k), len(uniq))
    rng = np.random.default_rng(seed)
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    fold_groups = np.array_split(shuffled, k)
    folds = []
    for fg in fold_groups:
        te = np.isin(groups, fg)
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        folds.append((np.where(tr)[0], np.where(te)[0]))
    if len(folds) < 2:
        raise RuntimeError("Could not create at least 2 non-empty grouped folds.")
    return folds


def residualize_train_test(y, cov, tr, te):
    y = np.asarray(y, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    Xtr = np.column_stack([np.ones(len(tr)), cov[tr]])
    beta = np.linalg.lstsq(Xtr, y[tr], rcond=None)[0]
    Xall = np.column_stack([np.ones(len(y)), cov])
    resid = y - Xall @ beta
    return resid


def solve_krr(K_train, y_train, K_test, alpha):
    n = K_train.shape[0]
    coef = np.linalg.solve(K_train + float(alpha) * np.eye(n), y_train)
    return K_test @ coef


def select_and_predict(
    Ks_train,
    Ks_test,
    y_train,
    groups_train,
    alphas,
    gamma_area_grid,
    gamma_pc_grid,
    gamma_raw_grid,
    seed,
    inner_k,
):
    inner = grouped_folds(groups_train, inner_k, seed)
    rows = []
    best = None
    for alpha in alphas:
        for ga in gamma_area_grid:
            for gp in gamma_pc_grid:
                for gr in gamma_raw_grid:
                    preds = np.zeros(len(y_train), dtype=float)
                    used = np.zeros(len(y_train), dtype=bool)
                    for itr, iva in inner:
                        Ktr = (
                            Ks_train["rsfc"][np.ix_(itr, itr)]
                            + ga * Ks_train["area"][np.ix_(itr, itr)]
                            + gp * Ks_train["pc"][np.ix_(itr, itr)]
                            + gr * Ks_train["raw"][np.ix_(itr, itr)]
                        )
                        Kva = (
                            Ks_train["rsfc"][np.ix_(iva, itr)]
                            + ga * Ks_train["area"][np.ix_(iva, itr)]
                            + gp * Ks_train["pc"][np.ix_(iva, itr)]
                            + gr * Ks_train["raw"][np.ix_(iva, itr)]
                        )
                        preds[iva] = solve_krr(Ktr, y_train[itr], Kva, alpha)
                        used[iva] = True
                    score = metrics(y_train[used], preds[used])["COD"]
                    row = {
                        "alpha": float(alpha),
                        "gamma_area": float(ga),
                        "gamma_pc": float(gp),
                        "gamma_raw": float(gr),
                        "mean_inner_cod": float(score),
                    }
                    rows.append(row)
                    if best is None or score > best["mean_inner_cod"]:
                        best = row
    alpha = best["alpha"]
    ga = best["gamma_area"]
    gp = best["gamma_pc"]
    gr = best["gamma_raw"]
    Ktr = (
        Ks_train["rsfc"]
        + ga * Ks_train["area"]
        + gp * Ks_train["pc"]
        + gr * Ks_train["raw"]
    )
    Kte = (
        Ks_test["rsfc"]
        + ga * Ks_test["area"]
        + gp * Ks_test["pc"]
        + gr * Ks_test["raw"]
    )
    pred = solve_krr(Ktr, y_train, Kte, alpha)
    return (pred, best, pd.DataFrame(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-csv", required=True)
    ap.add_argument("--official-pred-csv", required=True)
    ap.add_argument("--pc-parquet", required=True)
    ap.add_argument("--raw-parquet", required=True)
    ap.add_argument("--raw-feature-list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument(
        "--raw-kernel",
        choices=["zscore_linear", "unitize_corr"],
        default="zscore_linear",
    )
    ap.add_argument(
        "--pc-kernel", choices=["zscore_linear", "unitize_corr"], default="unitize_corr"
    )
    ap.add_argument("--inner-k", type=int, default=5)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.cohort_csv)
    df["_bcp_row"] = np.arange(len(df), dtype=np.int64)
    official = pd.read_csv(args.official_pred_csv).copy()
    official["_pred_row"] = np.arange(len(official), dtype=np.int64)
    pc = pd.read_parquet(args.pc_parquet)
    raw = pd.read_parquet(args.raw_parquet)
    df = df.merge(
        official[
            [
                "scan_id",
                "imaging_numeric_id",
                "fold",
                "y_true_resid",
                "y_pred",
                "_pred_row",
            ]
        ],
        on=["scan_id", "imaging_numeric_id"],
        how="left",
        validate="one_to_one",
    )
    if df["fold"].isna().any():
        raise RuntimeError("Missing official baseline fold after merge.")
    for name, trans in [("pc", pc), ("raw", raw)]:
        keys = ["_bcp_row", "scan_id", "imaging_numeric_id"]
        if not all((k in trans.columns for k in keys)):
            raise RuntimeError(f"{name} transferred file missing identity keys.")
        check = (
            df[keys].reset_index(drop=True).equals(trans[keys].reset_index(drop=True))
        )
        if not check:
            raise RuntimeError(f"{name} transferred file row identity/order mismatch.")
    edge_cols = sorted(
        [c for c in df.columns if c.startswith("e") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    area_cols = [c for c in df.columns if c.startswith("rescentarea_")]
    if len(edge_cols) != 1485:
        raise RuntimeError(f"Expected 1485 RSFC edges, got {len(edge_cols)}.")
    if len(area_cols) == 0:
        raise RuntimeError("No rescentarea columns found.")
    pc_cols = sorted(
        [
            c
            for c in pc.columns
            if c.endswith("__zpred") and (not c.endswith("__zpred_seed_sd"))
        ]
    )
    with open(args.raw_feature_list, "r", encoding="utf-8") as f:
        raw_targets = [x.strip() for x in f if x.strip()]
    raw_cols = [f"{t}__zpred" if not t.endswith("__zpred") else t for t in raw_targets]
    missing_raw = [c for c in raw_cols if c not in raw.columns]
    if missing_raw:
        raise RuntimeError(f"Missing raw zpred columns: {missing_raw}")
    y = df["composite_percentile"].to_numpy(float)
    cov = pd.DataFrame(
        {
            "scan_age_months": df["scan_age_months"].astype(float),
            "age_sq": df["scan_age_months"].astype(float) ** 2,
            "sex_male": (df["sex_final"].astype(str).str.upper() == "M").astype(float),
            "FD": df["FD"].astype(float),
            "site_UNC": (df["site_primary"].astype(str).str.upper() == "UNC").astype(
                float
            ),
        }
    )
    cov["age_sex"] = cov["scan_age_months"] * cov["sex_male"]
    cov["age_sq_sex"] = cov["age_sq"] * cov["sex_male"]
    groups = df["imaging_numeric_id"].to_numpy()
    folds = sorted(df["fold"].unique())
    alphas = [0.1, 0.4, 1.0, 4.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    gamma_area_grid = [0.0, 0.04, 0.1, 0.4, 1.0, 4.0, 10.0]
    gamma_pc_grid = [0.0]
    gamma_raw_grid = [0.0, 0.01, 0.04, 0.1, 0.4, 1.0, 4.0, 10.0]
    pred_base = np.zeros(len(df), dtype=float)
    pred_multi = np.zeros(len(df), dtype=float)
    y_true_resid_oof = np.full(len(df), np.nan, dtype=float)
    fold_rows = []
    inner_rows = []
    X_rsfc = df[edge_cols].to_numpy(float)
    X_area = df[area_cols].to_numpy(float)
    X_pc = pc[pc_cols].to_numpy(float)
    X_raw = raw[raw_cols].to_numpy(float)
    if (
        not np.isfinite(X_rsfc).all()
        or not np.isfinite(X_area).all()
        or (not np.isfinite(X_pc).all())
        or (not np.isfinite(X_raw).all())
    ):
        raise RuntimeError("Non-finite feature values detected.")
    for fold in folds:
        te = np.where(df["fold"].to_numpy() == fold)[0]
        tr = np.where(df["fold"].to_numpy() != fold)[0]
        y_resid = residualize_train_test(y, cov.to_numpy(float), tr, te)
        ytr = y_resid[tr]
        K_rsfc_tr, K_rsfc_te = make_kernel_train_test(
            X_rsfc[tr], X_rsfc[te], "unitize_corr"
        )
        K_area_tr, K_area_te = make_kernel_train_test(
            X_area[tr], X_area[te], "zscore_linear"
        )
        K_pc_tr, K_pc_te = make_kernel_train_test(X_pc[tr], X_pc[te], args.pc_kernel)
        K_raw_tr, K_raw_te = make_kernel_train_test(
            X_raw[tr], X_raw[te], args.raw_kernel
        )
        Ks_tr = {"rsfc": K_rsfc_tr, "area": K_area_tr, "pc": K_pc_tr, "raw": K_raw_tr}
        Ks_te = {"rsfc": K_rsfc_te, "area": K_area_te, "pc": K_pc_te, "raw": K_raw_te}
        base_pred, base_best, base_inner = select_and_predict(
            Ks_tr,
            Ks_te,
            ytr,
            groups[tr],
            alphas=alphas,
            gamma_area_grid=gamma_area_grid,
            gamma_pc_grid=[0.0],
            gamma_raw_grid=[0.0],
            seed=1000 + int(fold),
            inner_k=args.inner_k,
        )
        multi_pred, multi_best, multi_inner = select_and_predict(
            Ks_tr,
            Ks_te,
            ytr,
            groups[tr],
            alphas=alphas,
            gamma_area_grid=gamma_area_grid,
            gamma_pc_grid=gamma_pc_grid,
            gamma_raw_grid=gamma_raw_grid,
            seed=2000 + int(fold),
            inner_k=args.inner_k,
        )
        pred_base[te] = base_pred
        pred_multi[te] = multi_pred
        y_true_resid_oof[te] = y_resid[te]
        m_base = metrics(y_true_resid_oof[te], base_pred)
        m_multi = metrics(y_true_resid_oof[te], multi_pred)
        fold_rows.append(
            {
                "fold": int(fold),
                "n_test": int(len(te)),
                "n_test_groups": int(len(np.unique(groups[te]))),
                "baseline_COD": m_base["COD"],
                "baseline_Pearson_r": m_base["Pearson_r"],
                "multi_COD": m_multi["COD"],
                "multi_Pearson_r": m_multi["Pearson_r"],
                "delta_COD": m_multi["COD"] - m_base["COD"],
                "delta_Pearson_r": m_multi["Pearson_r"] - m_base["Pearson_r"],
                "baseline_alpha": base_best["alpha"],
                "baseline_gamma_area": base_best["gamma_area"],
                "multi_alpha": multi_best["alpha"],
                "multi_gamma_area": multi_best["gamma_area"],
                "multi_gamma_pc": multi_best["gamma_pc"],
                "multi_gamma_raw": multi_best["gamma_raw"],
                "baseline_best_inner_cod": base_best["mean_inner_cod"],
                "multi_best_inner_cod": multi_best["mean_inner_cod"],
            }
        )
        base_inner["outer_fold"] = int(fold)
        base_inner["stream"] = "baseline"
        multi_inner["outer_fold"] = int(fold)
        multi_inner["stream"] = "multikernel"
        inner_rows.append(base_inner)
        inner_rows.append(multi_inner)
    if not np.isfinite(y_true_resid_oof).all():
        raise RuntimeError("y_true_resid_oof contains NaN/Inf after outer CV.")
    official_resid = df["y_true_resid"].to_numpy(float)
    resid_max_abs_diff = float(np.max(np.abs(y_true_resid_oof - official_resid)))
    resid_mean_abs_diff = float(np.mean(np.abs(y_true_resid_oof - official_resid)))
    pooled_base = metrics(y_true_resid_oof, pred_base)
    pooled_multi = metrics(y_true_resid_oof, pred_multi)
    pred_df = df[
        [
            "_bcp_row",
            "scan_id",
            "imaging_numeric_id",
            "fold",
            "composite_percentile",
            "y_true_resid",
            "y_pred",
            "_pred_row",
        ]
    ].copy()
    pred_df["y_true_resid_official"] = pred_df["y_true_resid"].to_numpy(float)
    pred_df["y_true_resid"] = y_true_resid_oof
    pred_df["y_pred_baseline_only"] = pred_base
    pred_df["y_pred_multikernel"] = pred_multi
    fold_df = pd.DataFrame(fold_rows)
    inner_df = pd.concat(inner_rows, ignore_index=True)
    metrics_obj = {
        "run_tag": args.run_tag,
        "created_at": datetime.now().isoformat(),
        "pooled": {
            "baseline_only": pooled_base,
            "multikernel": pooled_multi,
            "delta_COD": pooled_multi["COD"] - pooled_base["COD"],
            "delta_Pearson_r": pooled_multi["Pearson_r"] - pooled_base["Pearson_r"],
            "relative_COD_improvement_percent": (
                (pooled_multi["COD"] - pooled_base["COD"])
                / abs(pooled_base["COD"])
                * 100
                if pooled_base["COD"] != 0
                else np.nan
            ),
            "n_folds_gamma_pc_gt0": int((fold_df["multi_gamma_pc"] > 0).sum()),
            "n_folds_gamma_raw_gt0": int((fold_df["multi_gamma_raw"] > 0).sum()),
            "gamma_pc_by_fold": {
                str(r.fold): float(r.multi_gamma_pc) for r in fold_df.itertuples()
            },
            "gamma_raw_by_fold": {
                str(r.fold): float(r.multi_gamma_raw) for r in fold_df.itertuples()
            },
            "residualization_check_vs_official": {
                "max_abs_diff": resid_max_abs_diff,
                "mean_abs_diff": resid_mean_abs_diff,
            },
        },
        "config": {
            "pc_kernel": args.pc_kernel,
            "raw_kernel": args.raw_kernel,
            "n_pc_features": len(pc_cols),
            "n_raw_features": len(raw_cols),
            "pc_features": pc_cols,
            "raw_features": raw_cols,
            "alphas": alphas,
            "gamma_area_grid": gamma_area_grid,
            "gamma_pc_grid": gamma_pc_grid,
            "gamma_raw_grid": gamma_raw_grid,
        },
    }
    run_config = {
        "cohort_csv": args.cohort_csv,
        "official_pred_csv": args.official_pred_csv,
        "pc_parquet": args.pc_parquet,
        "raw_parquet": args.raw_parquet,
        "raw_feature_list": args.raw_feature_list,
        "run_tag": args.run_tag,
        "target": "composite_percentile residualized within outer fold using official covariates",
    }
    pred_df.to_csv(out_dir / f"predictions_concat_{args.run_tag}.csv", index=False)
    fold_df.to_csv(out_dir / f"fold_metrics_{args.run_tag}.csv", index=False)
    inner_df.to_csv(out_dir / f"inner_scores_{args.run_tag}.csv", index=False)
    with open(out_dir / f"metrics_{args.run_tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, indent=2)
    with open(out_dir / f"run_config_{args.run_tag}.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)


if __name__ == "__main__":
    main()
