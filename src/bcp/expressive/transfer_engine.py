"""Build inner-OOF baseline predictions and select a disease anchor.

Source branches fit the covariate-residualized outcome with feature scaling
estimated from each training partition.
"""

from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

TARGET = "expressive_language_percentile"
GROUP = "imaging_numeric_id"
KEYS = ["scan_id", GROUP]
BASE_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
DISEASE_RAW_M = [5, 10, 20, 30, 50, 80, 120, 180, 240, 330]
DISEASE_W_BASE = [0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]


def load_module(path: str, name: str):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scan_id"] = out["scan_id"].astype(str).str.strip()
    x = pd.to_numeric(out[GROUP], errors="raise").to_numpy(float)
    if not np.isfinite(x).all() or not np.equal(x, np.floor(x)).all():
        raise RuntimeError("imaging_numeric_id must contain finite integers")
    out[GROUP] = x.astype(np.int64).astype(str)
    return out


def metrics(y, pred):
    y = np.asarray(y, float).ravel()
    pred = np.asarray(pred, float).ravel()
    if (
        y.shape != pred.shape
        or not np.isfinite(y).all()
        or (not np.isfinite(pred).all())
    ):
        raise RuntimeError("Invalid metric inputs")
    err = y - pred
    mse = float(np.mean(err**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    cod = float(1.0 - np.sum(err**2) / sst)
    r = float(np.corrcoef(y, pred)[0, 1]) if y.std() > 0 and pred.std() > 0 else np.nan
    return {
        "n": int(len(y)),
        "COD": cod,
        "R2": cod,
        "Pearson_r": r,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(err))),
        "MSE": mse,
    }


def zscore_train_test(xtr, xte):
    xtr = np.asarray(xtr, float)
    xte = np.asarray(xte, float)
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0, ddof=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return ((xtr - mu) / sd, (xte - mu) / sd)


def fit_baseline(fmri_tr, ytr, fmri_te, alpha):
    ztr, zte = zscore_train_test(fmri_tr, fmri_te)
    ktr = ztr @ ztr.T / float(ztr.shape[1])
    kte = zte @ ztr.T / float(ztr.shape[1])
    return (
        KernelRidge(alpha=float(alpha), kernel="precomputed").fit(ktr, ytr).predict(kte)
    )


def fit_transfer(xtr, ytr, xte, alpha):
    ztr, zte = zscore_train_test(xtr, xte)
    return Ridge(alpha=float(alpha), fit_intercept=True).fit(ztr, ytr).predict(zte)


def fold_cod_stats(y, pred, fold_local):
    vals = []
    for f in sorted(np.unique(fold_local)):
        ix = fold_local == f
        vals.append(metrics(y[ix], pred[ix])["COD"])
    return (float(np.mean(vals)), float(np.min(vals)), float(np.std(vals, ddof=1)))


def scoring_fields(y, pred, fold_local):
    pooled = metrics(y, pred)
    mean_cod, min_cod, sd_cod = fold_cod_stats(y, pred, fold_local)
    return {
        "pooled_inner_COD": pooled["COD"],
        "pooled_inner_RMSE": pooled["RMSE"],
        "mean_inner_COD": mean_cod,
        "min_inner_COD": min_cod,
        "sd_inner_COD": sd_cod,
    }


def source_family_matrix(aligned, feature_manifest, family, expected_n):
    rows = feature_manifest.loc[feature_manifest["family"] == family].copy()
    rows["family_feature_index_1based"] = pd.to_numeric(
        rows["family_feature_index_1based"], errors="raise"
    ).astype(int)
    rows = rows.sort_values("family_feature_index_1based")
    cols = rows["aligned_column"].astype(str).tolist()
    if len(cols) != expected_n:
        raise RuntimeError(f"{family}: expected {expected_n} features, got {len(cols)}")
    if rows["family_feature_index_1based"].tolist() != list(range(1, expected_n + 1)):
        raise RuntimeError(f"{family}: feature order is not exactly 1..{expected_n}")
    missing = [c for c in cols if c not in aligned.columns]
    if missing:
        raise RuntimeError(f"{family}: missing aligned columns: {missing[:10]}")
    x = aligned[cols].to_numpy(float)
    if not np.isfinite(x).all():
        raise RuntimeError(f"{family}: contains NaN/Inf")
    return (x, cols)


def build_inner_oof(
    engine, fmri, branch_specs, y_raw, cov, groups, outer_train, inner_k, seed
):
    n = len(outer_train)
    y_oof = np.full(n, np.nan)
    inner_id = np.full(n, -1, int)
    base_cache = {a: np.full(n, np.nan) for a in BASE_ALPHAS}
    branch_cache = {
        name: {
            (m, a): np.full(n, np.nan) for m in spec["m_values"] for a in RIDGE_ALPHAS
        }
        for name, spec in branch_specs.items()
    }
    splits = engine.grouped_folds(groups[outer_train], inner_k, seed)
    for inner_fold, (tr_local, va_local) in enumerate(splits):
        tr = outer_train[tr_local]
        va = outer_train[va_local]
        ytr, yva = engine.residualize(y_raw[tr], y_raw[va], cov[tr], cov[va])
        y_oof[va_local] = yva
        inner_id[va_local] = inner_fold
        for alpha in BASE_ALPHAS:
            base_cache[alpha][va_local] = fit_baseline(fmri[tr], ytr, fmri[va], alpha)
        for name, spec in branch_specs.items():
            x = spec["X"]
            for m in spec["m_values"]:
                width = int(spec["width_fn"](m))
                for alpha in RIDGE_ALPHAS:
                    branch_cache[name][m, alpha][va_local] = fit_transfer(
                        x[tr, :width], ytr, x[va, :width], alpha
                    )
    arrays = [y_oof, *base_cache.values()]
    for cache in branch_cache.values():
        arrays.extend(cache.values())
    if (inner_id < 0).any() or any((not np.isfinite(a).all() for a in arrays)):
        raise RuntimeError("Incomplete/nonfinite inner OOF predictions")
    return (y_oof, inner_id, base_cache, branch_cache)


def select_disease_anchor(y, inner_id, base_cache, disease_cache, outer_fold):
    base_rows = []
    for alpha, pb in base_cache.items():
        mean_cod, min_cod, sd_cod = fold_cod_stats(y, pb, inner_id)
        base_rows.append(
            {
                "outer_fold": outer_fold,
                "stage": "baseline",
                "branch": "fmri_zlinear",
                "mode": "baseline",
                "top_m": 0,
                "alpha_base": float(alpha),
                "alpha_branch": np.nan,
                "anchor_weight": 1.0,
                "branch_weight": 0.0,
                "gamma": 0.0,
                "mean_inner_COD": mean_cod,
                "min_inner_COD": min_cod,
                "sd_inner_COD": sd_cod,
                "pooled_inner_COD": metrics(y, pb)["COD"],
                "pooled_inner_RMSE": metrics(y, pb)["RMSE"],
            }
        )
    best_base = sorted(
        base_rows, key=lambda r: (-r["mean_inner_COD"], r["alpha_base"])
    )[0]
    pb = base_cache[best_base["alpha_base"]]
    rows = list(base_rows)
    candidates = []
    for (m, alpha), pdisease in disease_cache.items():
        for w in DISEASE_W_BASE:
            pred = float(w) * pb + (1.0 - float(w)) * pdisease
            s = scoring_fields(y, pred, inner_id)
            row = {
                "outer_fold": outer_fold,
                "stage": "disease_anchor",
                "branch": "diseaseraw330",
                "mode": "convex",
                "top_m": int(m),
                "alpha_base": float(best_base["alpha_base"]),
                "alpha_branch": float(alpha),
                "anchor_weight": float(w),
                "branch_weight": 1.0 - float(w),
                "gamma": 0.0,
                **s,
            }
            rows.append(row)
            candidates.append(row)
    best = sorted(
        candidates,
        key=lambda r: (
            -r["mean_inner_COD"],
            -r["anchor_weight"],
            r["top_m"],
            -r["alpha_branch"],
        ),
    )[0]
    anchor_inner = (
        best["anchor_weight"] * pb
        + best["branch_weight"] * disease_cache[best["top_m"], best["alpha_branch"]]
    )
    return (best_base, best, anchor_inner, rows)
