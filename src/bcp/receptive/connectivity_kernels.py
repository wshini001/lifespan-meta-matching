"""Construct child-grouped folds and training-partition connectivity/area kernels."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    mse = float(mean_squared_error(y, p))
    sse = float(np.sum((y - p) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r = float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else np.nan
    return {
        "n": int(len(y)),
        "COD": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "R2": float(r2_score(y, p)),
        "Pearson_r": r,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y, p)),
        "MSE": mse,
        "neg_MSE": -mse,
    }


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


def zscore_train_test(Xtr, Xte):
    Xtr = np.asarray(Xtr, dtype=float)
    Xte = np.asarray(Xte, dtype=float)
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[~np.isfinite(sd)] = 1.0
    sd[sd == 0.0] = 1.0
    return ((Xtr - mu) / sd, (Xte - mu) / sd)


def linear_kernel_train_test(Xtr, Xte):
    Xtr_z, Xte_z = zscore_train_test(Xtr, Xte)
    p = max(1, Xtr_z.shape[1])
    return (Xtr_z @ Xtr_z.T / p, Xte_z @ Xtr_z.T / p)


def centile_logit_split_train_test(Xtr, Xte):
    Xtr = np.asarray(Xtr, dtype=float)
    Xte = np.asarray(Xte, dtype=float)
    if not np.isfinite(Xtr).all() or not np.isfinite(Xte).all():
        raise RuntimeError("Non-finite values in area_rescent train/test arrays.")
    ntr, p = Xtr.shape
    if ntr <= 0 or p <= 0:
        raise RuntimeError(f"Invalid area_rescent train shape: {Xtr.shape}")
    if Xte.shape[1] != p:
        raise RuntimeError(
            f"Train/test feature mismatch: Xtr={Xtr.shape}, Xte={Xte.shape}"
        )
    Xtr_out = np.empty((Xtr.shape[0], 2 * p), dtype=float)
    Xte_out = np.empty((Xte.shape[0], 2 * p), dtype=float)
    sorted_tr = np.sort(Xtr, axis=0)
    for j in range(p):
        cent_tr = np.searchsorted(sorted_tr[:, j], Xtr[:, j], side="left").astype(
            float
        ) / float(ntr)
        cent_te = np.searchsorted(sorted_tr[:, j], Xte[:, j], side="left").astype(
            float
        ) / float(ntr)
        logit_tr = np.log((cent_tr + 0.0001) / (1.0001 - cent_tr))
        logit_te = np.log((cent_te + 0.0001) / (1.0001 - cent_te))
        Xtr_out[:, 2 * j] = np.maximum(logit_tr, 0.0)
        Xtr_out[:, 2 * j + 1] = np.minimum(logit_tr, 0.0)
        Xte_out[:, 2 * j] = np.maximum(logit_te, 0.0)
        Xte_out[:, 2 * j + 1] = np.minimum(logit_te, 0.0)
    return (Xtr_out, Xte_out)


def kernel_train_test_for_block(Xtr, Xte, block_name):
    if block_name in {"rsfc", "fmrires"}:
        return linear_kernel_train_test(Xtr, Xte)
    if block_name == "area_rescent":
        Xtr_t, Xte_t = centile_logit_split_train_test(Xtr, Xte)
        return linear_kernel_train_test(Xtr_t, Xte_t)
    raise ValueError(f"Unknown block_name: {block_name}")
