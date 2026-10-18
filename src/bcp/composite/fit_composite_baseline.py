"""Fit a child-grouped composite baseline using RSFC and surface-area features.

Uses seven outcome covariates. Outcome residualization and feature scaling
are fitted within each inner training partition. Use --alpha-select-metric cod
for the manuscript configuration; the command-line default is Pearson correlation.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mse = mean_squared_error(y, p)
    sse = np.sum((y - p) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r = np.corrcoef(y, p)[0, 1] if y.std() > 0 and p.std() > 0 else np.nan
    return {
        "n": int(len(y)),
        "COD": float(1 - sse / sst),
        "R2": float(r2_score(y, p)),
        "Pearson_r": float(r),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y, p)),
        "MSE": float(mse),
        "neg_MSE": float(-mse),
    }


def score_value(y, p, metric):
    m = metrics(y, p)
    if metric == "pearson":
        return m["Pearson_r"]
    if metric == "cod":
        return m["COD"]
    if metric == "neg_mse":
        return m["neg_MSE"]
    raise ValueError(metric)


def grouped_folds(groups, k, seed):
    groups = np.asarray(groups).astype(str)
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = np.arange(len(uniq))
    rng.shuffle(order)
    uniq = uniq[order]
    counts = counts[order]
    fold_sets = [set() for _ in range(k)]
    fold_sizes = np.zeros(k, dtype=int)
    for g, c in zip(uniq, counts):
        j = int(np.argmin(fold_sizes))
        fold_sets[j].add(g)
        fold_sizes[j] += int(c)
    idx = np.arange(len(groups))
    return [
        (idx[~np.isin(groups, list(s))], idx[np.isin(groups, list(s))])
        for s in fold_sets
    ]


def make_covariates(df):
    age = pd.to_numeric(df["scan_age_months"], errors="coerce").astype(float)
    sex = (df["sex_final"].astype(str) == "M").astype(float)
    fd = pd.to_numeric(df["FD"], errors="coerce").astype(float)
    site = (df["site_primary"].astype(str) == "UNC").astype(float)
    age2 = age**2
    return pd.DataFrame(
        {
            "scan_age_months": age,
            "age_sq": age2,
            "sex_male": sex,
            "FD": fd,
            "site_UNC": site,
            "age_sex": age * sex,
            "age_sq_sex": age2 * sex,
        }
    )


def zscore_train_test(Xtr, Xte):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd == 0] = 1
    return ((Xtr - mu) / sd, (Xte - mu) / sd)


def kernel(Xtr, Xte=None):
    if Xte is None:
        return Xtr @ Xtr.T / Xtr.shape[1]
    return Xte @ Xtr.T / Xtr.shape[1]


def residualize(ytr, yte, Ctr, Cte):
    lr = LinearRegression()
    lr.fit(Ctr, ytr)
    return (ytr - lr.predict(Ctr), yte - lr.predict(Cte))


def inner_select(X_rsfc, X_area, y, C, groups, tr_idx, alphas, gammas, metric, seed):
    inner = grouped_folds(groups[tr_idx], 5, seed)
    Xr = X_rsfc[tr_idx]
    Xa = X_area[tr_idx]
    yo = y[tr_idx]
    Co = C[tr_idx]
    rows = []
    for alpha in alphas:
        for gamma in gammas:
            vals = []
            for itr, iva in inner:
                yr, yv = residualize(yo[itr], yo[iva], Co[itr], Co[iva])
                Xr_tr, Xr_va = zscore_train_test(Xr[itr], Xr[iva])
                Xa_tr, Xa_va = zscore_train_test(Xa[itr], Xa[iva])
                Ktr = kernel(Xr_tr) + gamma * kernel(Xa_tr)
                Kva = kernel(Xr_tr, Xr_va) + gamma * kernel(Xa_tr, Xa_va)
                model = KernelRidge(alpha=alpha, kernel="precomputed")
                model.fit(Ktr, yr)
                pred = model.predict(Kva)
                vals.append(score_value(yv, pred, metric))
            rows.append(
                {
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    f"mean_inner_{metric}": float(np.mean(vals)),
                }
            )
    inner_df = (
        pd.DataFrame(rows)
        .sort_values(
            [f"mean_inner_{metric}", "alpha", "gamma"], ascending=[False, True, True]
        )
        .reset_index(drop=True)
    )
    best = inner_df.iloc[0]
    return (
        float(best["alpha"]),
        float(best["gamma"]),
        float(best[f"mean_inner_{metric}"]),
        inner_df,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--target-col", default="composite_percentile")
    ap.add_argument(
        "--alpha-select-metric",
        choices=["pearson", "cod", "neg_mse"],
        default="pearson",
    )
    ap.add_argument("--outer-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.cohort_csv)
    required = [
        args.target_col,
        "scan_age_months",
        "sex_final",
        "FD",
        "site_primary",
        "imaging_numeric_id",
    ]
    df = df.dropna(subset=required).reset_index(drop=True)
    rsfc_cols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    area_cols = [c for c in df.columns if c.startswith("rescentarea_")]
    if len(rsfc_cols) != 1485:
        pass
    if len(area_cols) == 0:
        raise RuntimeError("No rescentarea_ columns found")
    X_rsfc_df = df[rsfc_cols].apply(pd.to_numeric, errors="coerce")
    X_area_df = df[area_cols].apply(pd.to_numeric, errors="coerce")
    good = ~(X_rsfc_df.isna().any(axis=1) | X_area_df.isna().any(axis=1))
    df = df.loc[good].reset_index(drop=True)
    X_rsfc = X_rsfc_df.loc[good].to_numpy(float)
    X_area = X_area_df.loc[good].to_numpy(float)
    y = pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(float)
    C = make_covariates(df).to_numpy(float)
    groups = df["imaging_numeric_id"].astype(str).to_numpy()
    alphas = [0.0001, 0.001, 0.004, 0.01, 0.04, 0.1, 0.4, 1, 4, 10, 30, 100, 300, 1000]
    gammas = [0, 0.01, 0.04, 0.1, 0.4, 1, 4, 10]
    outer = grouped_folds(groups, args.outer_k, args.seed)
    pred_parts = []
    fold_rows = []
    inner_parts = []
    for fold, (tr, te) in enumerate(outer):
        yr, yt = residualize(y[tr], y[te], C[tr], C[te])
        alpha, gamma, best_inner, inner_df = inner_select(
            X_rsfc,
            X_area,
            y,
            C,
            groups,
            tr,
            alphas,
            gammas,
            args.alpha_select_metric,
            args.seed + 100 + fold,
        )
        Xr_tr, Xr_te = zscore_train_test(X_rsfc[tr], X_rsfc[te])
        Xa_tr, Xa_te = zscore_train_test(X_area[tr], X_area[te])
        Ktr = kernel(Xr_tr) + gamma * kernel(Xa_tr)
        Kte = kernel(Xr_tr, Xr_te) + gamma * kernel(Xa_tr, Xa_te)
        model = KernelRidge(alpha=alpha, kernel="precomputed")
        model.fit(Ktr, yr)
        pred = model.predict(Kte)
        met = metrics(yt, pred)
        met.update(
            {
                "fold": fold,
                "alpha": alpha,
                "gamma": gamma,
                f"best_inner_{args.alpha_select_metric}": best_inner,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_train_groups": int(len(np.unique(groups[tr]))),
                "n_test_groups": int(len(np.unique(groups[te]))),
            }
        )
        fold_rows.append(met)
        inner_df["outer_fold"] = fold
        inner_parts.append(inner_df)
        cols_to_keep = [
            "scan_id",
            "imaging_numeric_id",
            args.target_col,
            "scan_age_months",
            "sex_final",
            "FD",
            "site_primary",
            "res_id",
            "res_age_diff_days",
        ]
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        pr = df.iloc[te][cols_to_keep].copy()
        pr["fold"] = fold
        pr["y_true_resid"] = yt
        pr["y_pred"] = pred
        pred_parts.append(pr)
    pred_df = pd.concat(pred_parts, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)
    inner_df = pd.concat(inner_parts, ignore_index=True)
    pooled = metrics(pred_df["y_true_resid"], pred_df["y_pred"])
    pred_df.to_csv(out / f"predictions_concat_{args.run_tag}.csv", index=False)
    fold_df.to_csv(out / f"fold_metrics_{args.run_tag}.csv", index=False)
    inner_df.to_csv(out / f"inner_scores_{args.run_tag}.csv", index=False)
    with open(out / f"metrics_{args.run_tag}.json", "w") as f:
        json.dump(
            {"pooled": pooled, "per_fold": fold_df.to_dict(orient="records")},
            f,
            indent=2,
        )
    config = {
        "run_tag": args.run_tag,
        "cohort_csv": args.cohort_csv,
        "target_col": args.target_col,
        "model": "K_RSFC + gamma * K_rescentArea",
        "alpha_select_metric": args.alpha_select_metric,
        "n": int(len(df)),
        "n_unique_groups": int(len(np.unique(groups))),
        "n_rsfc_features": int(len(rsfc_cols)),
        "n_rescentarea_features": int(len(area_cols)),
        "covariates": [
            "scan_age_months",
            "age_sq",
            "sex_male",
            "FD",
            "site_UNC",
            "age_sex",
            "age_sq_sex",
        ],
    }
    with open(out / f"run_config_{args.run_tag}.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
