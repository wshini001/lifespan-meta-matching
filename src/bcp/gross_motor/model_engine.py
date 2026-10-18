"""Prepare training partitions and fit child-balanced gross-motor models."""

import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error

TARGET = "gross_motor_percentile"
GROUP = "imaging_numeric_id"
KEYS = ["scan_id", GROUP]
BASELINE_ALPHA = 1.0
EXPECTED_COVARIATES = [
    "scan_age_months",
    "age_sq",
    "sex_male",
    "FD",
    "site_UNC",
    "age_sex",
    "age_sq_sex",
    "mother_edu_gt_college",
]


def metrics(y, p):
    y, p = (np.asarray(y, float), np.asarray(p, float))
    mse = float(mean_squared_error(y, p))
    sst = float(np.sum((y - y.mean()) ** 2))
    cod = float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else np.nan
    r = float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else np.nan
    return {
        "n": int(len(y)),
        "COD": cod,
        "R2": cod,
        "Pearson_r": r,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y, p)),
        "MSE": mse,
    }


def grouped_folds(groups, k, seed):
    groups = np.asarray(groups).astype(str)
    uniq, counts = np.unique(groups, return_counts=True)
    if len(uniq) < k:
        raise RuntimeError("Too few groups for grouped CV")
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(uniq))
    rng.shuffle(order)
    bins, sizes = ([set() for _ in range(k)], np.zeros(k, int))
    for g, count in zip(uniq[order], counts[order]):
        j = int(np.argmin(sizes))
        bins[j].add(g)
        sizes[j] += int(count)
    idx = np.arange(len(groups))
    return [
        (idx[~np.isin(groups, list(s))], idx[np.isin(groups, list(s))]) for s in bins
    ]


def covariates(df):
    age = pd.to_numeric(df["scan_age_months"], errors="coerce").astype(float)
    sex = df["sex_final"].astype(str).str.upper().str.strip().map({"F": 0.0, "M": 1.0})
    site = pd.get_dummies(
        df["site_primary"].astype(str).str.strip(), prefix="site", drop_first=True
    ).astype(float)
    c = pd.DataFrame(
        {
            "scan_age_months": age,
            "age_sq": age**2,
            "sex_male": sex,
            "FD": pd.to_numeric(df["FD"], errors="coerce"),
        },
        index=df.index,
    )
    c = pd.concat([c, site], axis=1)
    c["age_sex"] = c["scan_age_months"] * c["sex_male"]
    c["age_sq_sex"] = c["age_sq"] * c["sex_male"]
    c["mother_edu_gt_college"] = pd.to_numeric(
        df["mother_edu_gt_college"], errors="coerce"
    )
    return c


def impute_train_test(c_tr, c_te):
    c_tr, c_te = (np.asarray(c_tr, float).copy(), np.asarray(c_te, float).copy())
    means = np.nanmean(c_tr, axis=0)
    means[~np.isfinite(means)] = 0
    for a in (c_tr, c_te):
        bad = ~np.isfinite(a)
        if bad.any():
            a[bad] = np.take(means, np.where(bad)[1])
    return (c_tr, c_te, means)


def residualize(y_tr, y_te, c_tr, c_te):
    c_tr, c_te, means = impute_train_test(c_tr, c_te)
    model = LinearRegression().fit(c_tr, y_tr)
    nuisance_tr, nuisance_te = (model.predict(c_tr), model.predict(c_te))
    return (
        y_tr - nuisance_tr,
        y_te - nuisance_te,
        means,
        model,
        nuisance_tr,
        nuisance_te,
    )


def locked_folds(df, manifest_path):
    fm = pd.read_csv(manifest_path)
    keys = ["scan_id", GROUP]
    if any((c not in fm for c in keys + ["fold"])):
        raise RuntimeError("Bad fold manifest")
    a, b = (df[keys].copy(), fm[keys + ["fold"]].copy())
    for c in keys:
        a[c] = a[c].astype(str).str.replace("\\.0$", "", regex=True)
        b[c] = b[c].astype(str).str.replace("\\.0$", "", regex=True)
    if b[keys].duplicated().any():
        raise RuntimeError("Duplicate fold-manifest key")
    m = (
        a.reset_index()
        .merge(b, on=keys, how="left", validate="one_to_one")
        .sort_values("index")
    )
    if m["fold"].isna().any():
        raise RuntimeError("Some eligible rows lack supplied folds")
    fold_id = m["fold"].astype(int).to_numpy()
    if (
        pd.DataFrame({GROUP: df[GROUP].astype(str), "fold": fold_id})
        .groupby(GROUP)
        .fold.nunique()
        .max()
        > 1
    ):
        raise RuntimeError("Subject leakage in supplied folds")
    idx = np.arange(len(df))
    folds = [(idx[fold_id != f], idx[fold_id == f]) for f in sorted(np.unique(fold_id))]
    return (folds, fold_id)


def normalize_keys(df):
    out = df.copy()
    for col in KEYS:
        out[col] = out[col].astype(str).str.strip().str.replace("\\.0$", "", regex=True)
    return out


def load_data(engine, cohort_csv, fold_manifest_csv):
    df = normalize_keys(pd.read_csv(cohort_csv).reset_index(drop=True))
    required = KEYS + [
        TARGET,
        "scan_age_months",
        "sex_final",
        "FD",
        "site_primary",
        "mother_edu_gt_college",
    ]
    missing = [col for col in required if col not in df]
    if missing:
        raise RuntimeError(f"Missing cohort columns: {missing}")
    if df[KEYS].duplicated().any():
        raise RuntimeError("Duplicated scan/child key")
    edges = sorted(
        [col for col in df if re.fullmatch("e\\d+", str(col))],
        key=lambda col: int(col[1:]),
    )
    if edges != [f"e{i}" for i in range(1, 1486)]:
        raise RuntimeError("Expected ordered e1..e1485")
    x = df[edges].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(float)
    cdf = engine.covariates(df)
    if list(cdf.columns) != EXPECTED_COVARIATES:
        raise RuntimeError(
            f"Covariate contract changed: expected {EXPECTED_COVARIATES}, got {list(cdf.columns)}"
        )
    nonedu = [col for col in cdf if col != "mother_edu_gt_college"]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Non-finite RSFC or target values")
    if cdf[nonedu].isna().any().any():
        raise RuntimeError("Unexpected missing non-education covariate")
    folds, fold_id = engine.locked_folds(df, fold_manifest_csv)
    return (df, x, y, cdf, folds, fold_id, edges)


def fit_zscore(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    sd = np.std(x_train, axis=0, ddof=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return ((x_train - mean) / sd, (x_test - mean) / sd)


def linear_kernel(x_test, x_train):
    return x_test @ x_train.T / float(x_train.shape[1])


def standardize(train, test):
    mu = np.mean(train, axis=0)
    sd = np.std(train, axis=0, ddof=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return ((train - mu) / sd, (test - mu) / sd)


def child_weights(groups):
    groups = np.asarray(groups).astype(str)
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv].astype(float)
    return w / np.mean(w)


def fit_baseline(refine, xtr, ytr, gtr, xte):
    ztr, zte = refine.fit_zscore(xtr, xte)
    return (
        KernelRidge(alpha=BASELINE_ALPHA, kernel="precomputed")
        .fit(
            refine.linear_kernel(ztr, ztr), ytr, sample_weight=refine.child_weights(gtr)
        )
        .predict(refine.linear_kernel(zte, ztr))
    )


def fit_transfer(xtr, ytr, gtr, xte, alpha):
    ztr, zte = standardize(xtr, xte)
    return (
        Ridge(alpha=float(alpha), fit_intercept=True)
        .fit(ztr, ytr, sample_weight=child_weights(gtr))
        .predict(zte)
    )
