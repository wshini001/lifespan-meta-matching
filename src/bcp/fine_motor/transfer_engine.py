"""Build inner prediction caches and select convex disease/baseline blends.

Ridge feature scaling is estimated within each training partition.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

LOCKED_BASELINE = {
    "COD": 0.03140914639561576,
    "Pearson_r": 0.1823312430801972,
    "RMSE": 28.859959267048005,
    "MAE": 24.454667362621464,
    "MSE": 832.89724889567,
    "SSE": 328994.41331378964,
    "N": 395,
}
FROZEN_M72 = {
    "COD": 0.046583018799620235,
    "Pearson_r": 0.21657892283812702,
    "RMSE": 28.633007917944745,
    "MAE": 24.35312963643682,
    "MSE": 819.8491424290866,
    "SSE": 323840.4112594892,
    "N": 395,
}


def parse_float_list(s):
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise RuntimeError(f"Empty float list: {s}")
    return vals


def metrics(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise RuntimeError(f"metrics shape mismatch: y={y.shape}, p={p.shape}")
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) < 3:
        raise RuntimeError("Need at least 3 finite samples")
    sse = float(np.sum((y - p) ** 2))
    denom_centered = float(np.sum((y - y.mean()) ** 2))
    denom_zero = float(np.sum(y**2))
    mse = float(np.mean((y - p) ** 2))
    if np.std(y) > 0 and np.std(p) > 0:
        r = float(np.corrcoef(y, p)[0, 1])
    else:
        r = float("nan")
    return {
        "n": int(len(y)),
        "COD": float(1.0 - sse / denom_centered),
        "COD_centered": float(1.0 - sse / denom_centered),
        "COD_zero": float(1.0 - sse / denom_zero) if denom_zero > 0 else float("nan"),
        "Pearson_r": r,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(y - p))),
        "MSE": mse,
        "SSE": sse,
        "denom_centered": denom_centered,
        "denom_zero": denom_zero,
    }


def align_to_meta(df, meta, label):
    df = df.copy()
    meta = meta.copy()
    key = ["scan_id", "imaging_numeric_id"]
    for c in key:
        if c not in df.columns:
            raise RuntimeError(f"{label}: missing key column {c}")
        if c not in meta.columns:
            raise RuntimeError(f"meta: missing key column {c}")
        df[c] = df[c].astype(str)
        meta[c] = meta[c].astype(str)
    if df[key].duplicated().any():
        raise RuntimeError(f"{label}: duplicated scan/subject keys")
    if meta[key].duplicated().any():
        raise RuntimeError("meta: duplicated scan/subject keys")
    if len(df) != len(meta):
        raise RuntimeError(f"{label}: N={len(df)}, expected {len(meta)}")
    df["_align_present_tmp"] = 1
    idx = pd.MultiIndex.from_frame(meta[key])
    aligned = df.set_index(key).reindex(idx)
    if aligned["_align_present_tmp"].isna().any():
        raise RuntimeError(f"{label}: locked BCP keys do not match")
    aligned = aligned.drop(columns=["_align_present_tmp"]).reset_index()
    if len(aligned) != len(meta):
        raise RuntimeError(f"{label}: aligned N mismatch")
    for c in key:
        if not np.array_equal(
            aligned[c].astype(str).to_numpy(), meta[c].astype(str).to_numpy()
        ):
            raise RuntimeError(f"{label}: row order mismatch after alignment: {c}")
    return aligned


def _safe_std_fit(X):
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return (mu, sd)


def _safe_std_apply(X, mu, sd):
    return (np.asarray(X, dtype=float) - mu) / sd


def ridge_predict(X_train, y_train, X_test, alpha):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if not np.isfinite(X_train).all():
        raise RuntimeError("ridge X_train contains NaN/Inf")
    if not np.isfinite(X_test).all():
        raise RuntimeError("ridge X_test contains NaN/Inf")
    mu, sd = _safe_std_fit(X_train)
    Xtr = _safe_std_apply(X_train, mu, sd)
    Xte = _safe_std_apply(X_test, mu, sd)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(Xtr, y_train)
    return np.asarray(model.predict(Xte), dtype=float)


def krr_predict_from_z(Z_train, y_train, Z_test, alpha):
    Z_train = np.asarray(Z_train, dtype=float)
    Z_test = np.asarray(Z_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    K_train = Z_train @ Z_train.T
    K_test = Z_test @ Z_train.T
    model = KernelRidge(kernel="precomputed", alpha=float(alpha))
    model.fit(K_train, y_train)
    return np.asarray(model.predict(K_test), dtype=float)


def build_inner_context(
    eng, Z_outer, C_outer, y_raw_outer, groups_outer, inner_k, seed
):
    Z_outer = np.asarray(Z_outer, dtype=float)
    C_outer = np.asarray(C_outer, dtype=float)
    y_raw_outer = np.asarray(y_raw_outer, dtype=float)
    groups_outer = np.asarray(groups_outer).astype(str)
    n = len(y_raw_outer)
    if len(Z_outer) != n or len(C_outer) != n or len(groups_outer) != n:
        raise RuntimeError("build_inner_context length mismatch")
    n_groups = len(np.unique(groups_outer))
    n_inner = min(int(inner_k), n_groups)
    if n_inner < 2:
        raise RuntimeError("Need at least 2 inner folds")
    splits = eng.make_balanced_group_folds(
        groups_outer, n_splits=n_inner, seed=int(seed)
    )
    y_inner = np.full(n, np.nan, dtype=float)
    visit = np.zeros(n, dtype=int)
    contexts = []
    for inner_fold, (tr, va) in enumerate(splits):
        tr = np.asarray(tr, dtype=int)
        va = np.asarray(va, dtype=int)
        overlap = set(groups_outer[tr]).intersection(set(groups_outer[va]))
        if overlap:
            raise RuntimeError(f"Inner subject leakage fold={inner_fold}")
        ytr_res, yva_res, cov_means, _ = eng.residualize_train_test(
            y_raw_outer[tr], y_raw_outer[va], C_outer[tr], C_outer[va]
        )
        ytr_res = np.asarray(ytr_res, dtype=float)
        yva_res = np.asarray(yva_res, dtype=float)
        y_inner[va] = yva_res
        visit[va] += 1
        contexts.append(
            {
                "inner_fold": int(inner_fold),
                "tr": tr,
                "va": va,
                "ytr_res": ytr_res,
                "yva_res": yva_res,
                "cov_means": np.asarray(cov_means, dtype=float),
            }
        )
    if not np.all(visit == 1):
        raise RuntimeError(
            "Each outer-train row must appear exactly once as inner validation"
        )
    if not np.isfinite(y_inner).all():
        raise RuntimeError("Nonfinite strict inner OOF target")
    return (y_inner, contexts)


def build_baseline_oof_cache(Z_outer, contexts, alphas):
    Z_outer = np.asarray(Z_outer, dtype=float)
    cache = {}
    for alpha in alphas:
        alpha = float(alpha)
        oof = np.full(len(Z_outer), np.nan, dtype=float)
        for ctx in contexts:
            tr = ctx["tr"]
            va = ctx["va"]
            oof[va] = krr_predict_from_z(
                Z_outer[tr], ctx["ytr_res"], Z_outer[va], alpha
            )
        if not np.isfinite(oof).all():
            raise RuntimeError(f"Nonfinite baseline OOF alpha={alpha}")
        cache[alpha] = oof
    return cache


def build_branch_oof_cache(X_outer, contexts, alphas):
    X_outer = np.asarray(X_outer, dtype=float)
    if not np.isfinite(X_outer).all():
        raise RuntimeError("Branch matrix contains NaN/Inf")
    cache = {}
    for alpha in alphas:
        alpha = float(alpha)
        oof = np.full(len(X_outer), np.nan, dtype=float)
        for ctx in contexts:
            tr = ctx["tr"]
            va = ctx["va"]
            oof[va] = ridge_predict(X_outer[tr], ctx["ytr_res"], X_outer[va], alpha)
        if not np.isfinite(oof).all():
            raise RuntimeError(f"Nonfinite branch OOF alpha={alpha}")
        cache[alpha] = oof
    return cache


def _is_better(row, best):
    if best is None:
        return True
    if row["inner_oof_COD"] > best["inner_oof_COD"] + 1e-15:
        return True
    if abs(row["inner_oof_COD"] - best["inner_oof_COD"]) <= 1e-15:
        if row["inner_oof_RMSE"] < best["inner_oof_RMSE"] - 1e-15:
            return True
        if abs(row["inner_oof_RMSE"] - best["inner_oof_RMSE"]) <= 1e-15:
            if row["w_base"] > best["w_base"]:
                return True
    return False


def select_stack(y_inner, base_cache, branch_cache, w_grid, method, outer_fold):
    y_inner = np.asarray(y_inner, dtype=float)
    rows = []
    best = None
    for alpha_base in base_cache:
        alpha_base = float(alpha_base)
        p = np.asarray(base_cache[alpha_base], dtype=float)
        m = metrics(y_inner, p)
        row = {
            "method": str(method),
            "outer_fold": int(outer_fold),
            "alpha_base": alpha_base,
            "alpha_branch": np.nan,
            "w_base": 1.0,
            "w_branch": 0.0,
            "inner_oof_COD": m["COD"],
            "inner_oof_r": m["Pearson_r"],
            "inner_oof_RMSE": m["RMSE"],
            "inner_oof_MAE": m["MAE"],
        }
        rows.append(row)
        if _is_better(row, best):
            best = row
    for alpha_base in base_cache:
        alpha_base = float(alpha_base)
        p_base = np.asarray(base_cache[alpha_base], dtype=float)
        for alpha_branch in branch_cache:
            alpha_branch = float(alpha_branch)
            p_branch = np.asarray(branch_cache[alpha_branch], dtype=float)
            for w_base in w_grid:
                w_base = float(w_base)
                if w_base >= 1.0:
                    continue
                w_branch = 1.0 - w_base
                p = w_base * p_base + w_branch * p_branch
                m = metrics(y_inner, p)
                row = {
                    "method": str(method),
                    "outer_fold": int(outer_fold),
                    "alpha_base": alpha_base,
                    "alpha_branch": alpha_branch,
                    "w_base": w_base,
                    "w_branch": w_branch,
                    "inner_oof_COD": m["COD"],
                    "inner_oof_r": m["Pearson_r"],
                    "inner_oof_RMSE": m["RMSE"],
                    "inner_oof_MAE": m["MAE"],
                }
                rows.append(row)
                if _is_better(row, best):
                    best = row
    if best is None:
        raise RuntimeError("No legal stack candidate")
    return (best, rows)
