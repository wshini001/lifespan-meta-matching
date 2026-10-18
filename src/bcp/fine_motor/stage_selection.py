"""Select disease, PC12 and fixed-size DNN prediction stages.

The fine-motor entry script uses the best_any candidate from the fixed-size
DNN search.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def better(row, best):
    if best is None:
        return True
    if row["inner_COD"] > best["inner_COD"] + 1e-15:
        return True
    if abs(row["inner_COD"] - best["inner_COD"]) <= 1e-15:
        if row["inner_RMSE"] < best["inner_RMSE"] - 1e-15:
            return True
        if abs(row["inner_RMSE"] - best["inner_RMSE"]) <= 1e-15:
            if row["anchor_weight"] > best["anchor_weight"] + 1e-15:
                return True
            if abs(row["anchor_weight"] - best["anchor_weight"]) <= 1e-15:
                m1 = row.get("M", np.inf)
                m2 = best.get("M", np.inf)
                if pd.isna(m1):
                    m1 = np.inf
                if pd.isna(m2):
                    m2 = np.inf
                if m1 < m2:
                    return True
    return False


def primary_inner_from_cfg(cfg, base_cache, dis_cache):
    ab = float(cfg["alpha_base"])
    if pd.isna(cfg["alpha_branch"]) or float(cfg["w_branch"]) <= 0:
        return np.asarray(base_cache[ab], float)
    ad = float(cfg["alpha_branch"])
    return float(cfg["w_base"]) * np.asarray(base_cache[ab], float) + float(
        cfg["w_branch"]
    ) * np.asarray(dis_cache[ad], float)


def predict_primary(H, Ztr, ytr, Zte, Xdis_tr, Xdis_te, cfg):
    pbase = H.krr_predict_from_z(Ztr, ytr, Zte, float(cfg["alpha_base"]))
    if pd.isna(cfg["alpha_branch"]) or float(cfg["w_branch"]) <= 0:
        return np.asarray(pbase, float)
    pdis = H.ridge_predict(Xdis_tr, ytr, Xdis_te, float(cfg["alpha_branch"]))
    return float(cfg["w_base"]) * np.asarray(pbase, float) + float(
        cfg["w_branch"]
    ) * np.asarray(pdis, float)


def select_pc_anchor(H, y, primary, pc_cache, weights):
    m0 = H.metrics(y, primary)
    best = {
        "alpha_pc": np.nan,
        "anchor_weight": 1.0,
        "pc_weight": 0.0,
        "inner_COD": m0["COD"],
        "inner_RMSE": m0["RMSE"],
        "inner_r": m0["Pearson_r"],
    }
    for alpha, ppc in pc_cache.items():
        for wa in weights:
            wa = float(wa)
            pred = wa * primary + (1.0 - wa) * np.asarray(ppc, float)
            m = H.metrics(y, pred)
            row = {
                "alpha_pc": float(alpha),
                "anchor_weight": wa,
                "pc_weight": 1.0 - wa,
                "inner_COD": m["COD"],
                "inner_RMSE": m["RMSE"],
                "inner_r": m["Pearson_r"],
            }
            if better(row, best):
                best = row
    return best


def make_inner_fold_id(groups, inner_k):
    groups = np.asarray(groups).astype(str)
    n_splits = min(int(inner_k), len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    fold_id = np.full(len(groups), -1, dtype=int)
    dummy_x = np.zeros((len(groups), 1))
    dummy_y = np.zeros(len(groups))
    for f, (_, va) in enumerate(gkf.split(dummy_x, dummy_y, groups)):
        fold_id[va] = f
    if np.any(fold_id < 0):
        raise RuntimeError("Inner fold assignment incomplete")
    chk = pd.DataFrame({"group": groups, "fold": fold_id})
    span = chk.groupby("group")["fold"].nunique().max()
    if int(span) != 1:
        raise RuntimeError("Inner subject leakage detected")
    return fold_id


def stability_stats(y, anchor_pred, candidate_pred, inner_fold_id):
    y = np.asarray(y, float)
    anchor_pred = np.asarray(anchor_pred, float)
    candidate_pred = np.asarray(candidate_pred, float)
    deltas = []
    for f in sorted(np.unique(inner_fold_id).tolist()):
        idx = inner_fold_id == f
        mse_anchor = float(np.mean((y[idx] - anchor_pred[idx]) ** 2))
        mse_candidate = float(np.mean((y[idx] - candidate_pred[idx]) ** 2))
        deltas.append(mse_anchor - mse_candidate)
    deltas = np.asarray(deltas, float)
    return {
        "n_positive_folds": int(np.sum(deltas > 0)),
        "median_delta_MSE": float(np.median(deltas)),
        "mean_delta_MSE": float(np.mean(deltas)),
        "fold_delta_MSE": deltas.tolist(),
    }


def select_dnn_for_M(H, y, anchor, dnn_cache, weights, inner_fold_id, M):
    anchor_m = H.metrics(y, anchor)
    best_any = {
        "M": int(M),
        "alpha_dnn": np.nan,
        "anchor_weight": 1.0,
        "dnn_weight": 0.0,
        "inner_COD": anchor_m["COD"],
        "inner_RMSE": anchor_m["RMSE"],
        "inner_r": anchor_m["Pearson_r"],
        "n_positive_folds": 0,
        "median_delta_MSE": 0.0,
        "mean_delta_MSE": 0.0,
        "fold_delta_MSE": [0.0] * len(np.unique(inner_fold_id)),
    }
    best_gate3 = None
    best_gate4 = None
    for alpha, pdnn in dnn_cache.items():
        pdnn = np.asarray(pdnn, float)
        for wa in weights:
            wa = float(wa)
            if wa >= 1.0:
                continue
            pred = wa * anchor + (1.0 - wa) * pdnn
            mm = H.metrics(y, pred)
            ss = stability_stats(y, anchor, pred, inner_fold_id)
            row = {
                "M": int(M),
                "alpha_dnn": float(alpha),
                "anchor_weight": wa,
                "dnn_weight": 1.0 - wa,
                "inner_COD": mm["COD"],
                "inner_RMSE": mm["RMSE"],
                "inner_r": mm["Pearson_r"],
                **ss,
            }
            if better(row, best_any):
                best_any = row
            pooled_improves = row["inner_COD"] > anchor_m["COD"] + 1e-15
            gate3 = (
                pooled_improves
                and row["n_positive_folds"] >= 3
                and (row["median_delta_MSE"] > 0)
            )
            gate4 = (
                pooled_improves
                and row["n_positive_folds"] >= 4
                and (row["median_delta_MSE"] > 0)
            )
            if gate3 and better(row, best_gate3):
                best_gate3 = row
            if gate4 and better(row, best_gate4):
                best_gate4 = row
    return (best_any, best_gate3, best_gate4)


def outer_predict_from_cfg(H, anchor_outer, Xtr, ytr, Xte, cfg):
    if cfg is None:
        return np.asarray(anchor_outer, float).copy()
    if pd.isna(cfg["alpha_dnn"]) or float(cfg["dnn_weight"]) <= 0:
        return np.asarray(anchor_outer, float).copy()
    pdnn = H.ridge_predict(Xtr, ytr, Xte, float(cfg["alpha_dnn"]))
    return float(cfg["anchor_weight"]) * np.asarray(anchor_outer, float) + float(
        cfg["dnn_weight"]
    ) * np.asarray(pdnn, float)
