"""Fit visual-reception KRRRaw-anchor and DNNRaw prediction models.

Runs sequential and adaptive candidate combinations. The manuscript uses
krr_anchor_plus_dnnraw_sequential with the supplied baseline features.
"""

from __future__ import annotations
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

TARGET = "visual_reception_percentile"
GROUP = "imaging_numeric_id"
KEYS = ["scan_id", GROUP]
EXPECTED_N = 289
EXPECTED_CHILDREN = 159
EXPECTED_BASELINE = {"COD": 0.012575393003604107, "Pearson_r": 0.12358626296353256}
EXPECTED_PHASE6 = {
    "baseline_plus_krrraw_stack": 0.020634419170838192,
    "baseline_plus_dnnraw_stack": 0.019770043516681435,
}
RAW_M = [5, 10, 15, 20, 30, 40, 60, 80, 100, 120]
PAIRED_M = [5, 10, 20, 40, 60, 80, 120]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
W_BASE = [0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
W_ANCHOR = [0.7, 0.8, 0.9, 0.95, 1.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase4-py",
        default=str(Path(__file__).resolve().parent / "baseline_inputs.py"),
    )
    parser.add_argument("--transfer-bank-parquet", required=True)
    parser.add_argument("--transfer-feature-manifest-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--run-tag",
        default="visual_reception_n289_rawdual_anchorfirst_strict_nested_v1",
    )
    parser.add_argument("--inner-k", type=int, default=5)
    parser.add_argument("--inner-seed-base", type=int, default=8400)
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--fold-manifest-csv", required=True)
    parser.add_argument("--fmri-res-csv", required=True)
    parser.add_argument("--signed72-csv", required=True)
    return parser.parse_args()


def load_module(path, name):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def normalize_keys(frame):
    result = frame.copy()
    result["scan_id"] = result["scan_id"].astype(str).str.strip()
    values = pd.to_numeric(result[GROUP], errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise RuntimeError("imaging_numeric_id must contain finite integers")
    result[GROUP] = values.astype(np.int64).astype(str)
    return result


def metrics(y, prediction):
    y = np.asarray(y, float).ravel()
    prediction = np.asarray(prediction, float).ravel()
    if y.shape != prediction.shape:
        raise RuntimeError("Metric shape mismatch")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise RuntimeError("Non-finite metric input")
    error = y - prediction
    mse = float(np.mean(error**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    cod = float(1.0 - np.sum(error**2) / sst)
    correlation = (
        float(np.corrcoef(y, prediction)[0, 1])
        if y.std() > 0 and prediction.std() > 0
        else np.nan
    )
    return {
        "n": int(len(y)),
        "COD": cod,
        "R2": cod,
        "Pearson_r": correlation,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
    }


def zscore_train_test(train, test):
    train = np.asarray(train, float)
    test = np.asarray(test, float)
    mean = train.mean(axis=0)
    sd = train.std(axis=0, ddof=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return ((train - mean) / sd, (test - mean) / sd)


def ridge_prediction(train_x, y_train, test_x, alpha):
    train_z, test_z = zscore_train_test(train_x, test_x)
    return (
        Ridge(alpha=float(alpha), fit_intercept=True)
        .fit(train_z, y_train)
        .predict(test_z)
    )


def baseline_predictions(phase4, engine, train_x, y_train, test_x, alphas):
    train_unit = engine.unitize_rows(train_x)
    test_unit = engine.unitize_rows(test_x)
    kernel_train = train_unit @ train_unit.T
    kernel_test = test_unit @ train_unit.T
    return phase4.kernel_predictions(kernel_train, kernel_test, y_train, alphas)


def score(y, prediction, inner_id):
    pooled = metrics(y, prediction)
    fold_cod = [
        metrics(y[inner_id == fold], prediction[inner_id == fold])["COD"]
        for fold in sorted(np.unique(inner_id))
    ]
    return {
        "pooled_inner_COD": pooled["COD"],
        "pooled_inner_RMSE": pooled["RMSE"],
        "mean_inner_COD": float(np.mean(fold_cod)),
        "min_inner_COD": float(np.min(fold_cod)),
        "sd_inner_COD": float(np.std(fold_cod, ddof=1)),
    }


def rank_rows(rows, tie_fields):
    return sorted(
        rows,
        key=lambda row: (
            -row["pooled_inner_COD"],
            -row["mean_inner_COD"],
            -row["min_inner_COD"],
            *[row[field] for field in tie_fields],
        ),
    )[0]


def load_family(feature_manifest, aligned, family, expected):
    rows = feature_manifest.loc[feature_manifest["family"] == family].copy()
    rows["family_feature_index_1based"] = pd.to_numeric(
        rows["family_feature_index_1based"], errors="raise"
    ).astype(int)
    rows = rows.sort_values("family_feature_index_1based")
    columns = rows["aligned_column"].astype(str).tolist()
    if len(columns) != expected:
        raise RuntimeError(
            f"{family}: expected {expected} features, got {len(columns)}"
        )
    if rows["family_feature_index_1based"].tolist() != list(range(1, expected + 1)):
        raise RuntimeError(f"{family}: feature-order contract failed")
    missing = [column for column in columns if column not in aligned]
    if missing:
        raise RuntimeError(f"{family}: missing columns {missing[:10]}")
    matrix = aligned[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{family}: nonfinite values")
    return (matrix, columns)


def build_inner_oof(
    phase4,
    engine,
    signed72,
    krrraw,
    dnnraw,
    y_raw,
    covariates,
    groups,
    outer_train,
    inner_k,
    seed,
):
    n = len(outer_train)
    y_oof = np.full(n, np.nan)
    inner_id = np.full(n, -1, int)
    baseline_cache = {float(alpha): np.full(n, np.nan) for alpha in phase4.ALPHAS}
    krr_cache = {
        (m, float(alpha)): np.full(n, np.nan) for m in RAW_M for alpha in RIDGE_ALPHAS
    }
    dnn_cache = {
        (m, float(alpha)): np.full(n, np.nan) for m in RAW_M for alpha in RIDGE_ALPHAS
    }
    paired_cache = {
        (m, float(alpha)): np.full(n, np.nan)
        for m in PAIRED_M
        for alpha in RIDGE_ALPHAS
    }
    splits = engine.grouped_folds(groups[outer_train], inner_k, seed)
    for inner_fold, (train_relative, valid_relative) in enumerate(splits):
        train = outer_train[train_relative]
        valid = outer_train[valid_relative]
        y_train, y_valid = engine.residualize(
            y_raw[train], y_raw[valid], covariates[train], covariates[valid]
        )
        y_oof[valid_relative] = y_valid
        inner_id[valid_relative] = inner_fold
        current_baseline = baseline_predictions(
            phase4, engine, signed72[train], y_train, signed72[valid], phase4.ALPHAS
        )
        for alpha, prediction in current_baseline.items():
            baseline_cache[float(alpha)][valid_relative] = prediction
        for m in RAW_M:
            for alpha in RIDGE_ALPHAS:
                krr_cache[m, float(alpha)][valid_relative] = ridge_prediction(
                    krrraw[train, :m], y_train, krrraw[valid, :m], alpha
                )
                dnn_cache[m, float(alpha)][valid_relative] = ridge_prediction(
                    dnnraw[train, :m], y_train, dnnraw[valid, :m], alpha
                )
        for m in PAIRED_M:
            paired_train = np.column_stack((krrraw[train, :m], dnnraw[train, :m]))
            paired_valid = np.column_stack((krrraw[valid, :m], dnnraw[valid, :m]))
            for alpha in RIDGE_ALPHAS:
                paired_cache[m, float(alpha)][valid_relative] = ridge_prediction(
                    paired_train, y_train, paired_valid, alpha
                )
    arrays = [
        y_oof,
        *baseline_cache.values(),
        *krr_cache.values(),
        *dnn_cache.values(),
        *paired_cache.values(),
    ]
    if (inner_id < 0).any() or any((not np.isfinite(x).all() for x in arrays)):
        raise RuntimeError("Incomplete/nonfinite inner OOF predictions")
    return (y_oof, inner_id, baseline_cache, krr_cache, dnn_cache, paired_cache)


def select_baseline(y, inner_id, cache, outer_fold, grid_rows):
    null = np.zeros(len(y), float)
    rows = [
        {
            "outer_fold": outer_fold,
            "method": "baseline",
            "family": "null",
            "top_m": 0,
            "alpha": np.nan,
            "w_base": 0.0,
            "w_anchor": np.nan,
            "selected_null": True,
            **score(y, null, inner_id),
        }
    ]
    for alpha, prediction in cache.items():
        rows.append(
            {
                "outer_fold": outer_fold,
                "method": "baseline",
                "family": "signed72_corr",
                "top_m": 0,
                "alpha": alpha,
                "w_base": 1.0,
                "w_anchor": np.nan,
                "selected_null": False,
                **score(y, prediction, inner_id),
            }
        )
    grid_rows.extend(rows)
    best = sorted(
        rows,
        key=lambda row: (
            -row["pooled_inner_COD"],
            -row["mean_inner_COD"],
            -row["min_inner_COD"],
            -int(row["selected_null"]),
            -(row["alpha"] if np.isfinite(row["alpha"]) else np.inf),
        ),
    )[0]
    prediction = null if best["selected_null"] else cache[float(best["alpha"])]
    return (best, prediction)


def select_base_stack(
    y, inner_id, base_prediction, branch_cache, family, outer_fold, grid_rows
):
    rows = []
    for (m, alpha), branch_prediction in branch_cache.items():
        for weight in W_BASE:
            prediction = weight * base_prediction + (1.0 - weight) * branch_prediction
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": "base_stack",
                    "family": family,
                    "top_m": m,
                    "alpha": alpha,
                    "w_base": weight,
                    "w_anchor": np.nan,
                    "selected_null": False,
                    **score(y, prediction, inner_id),
                }
            )
    grid_rows.extend(rows)
    best = sorted(
        rows,
        key=lambda row: (
            -row["pooled_inner_COD"],
            -row["mean_inner_COD"],
            -row["min_inner_COD"],
            -row["w_base"],
            row["top_m"],
            -row["alpha"],
        ),
    )[0]
    branch = branch_cache[int(best["top_m"]), float(best["alpha"])]
    prediction = best["w_base"] * base_prediction + (1.0 - best["w_base"]) * branch
    return (best, prediction)


def select_sequential(
    y, inner_id, anchor_prediction, augment_cache, family, outer_fold, grid_rows
):
    rows = []
    for (m, alpha), augment_prediction in augment_cache.items():
        for weight in W_ANCHOR:
            prediction = (
                weight * anchor_prediction + (1.0 - weight) * augment_prediction
            )
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": "sequential",
                    "family": family,
                    "top_m": m,
                    "alpha": alpha,
                    "w_base": np.nan,
                    "w_anchor": weight,
                    "selected_null": False,
                    **score(y, prediction, inner_id),
                }
            )
    grid_rows.extend(rows)
    best = sorted(
        rows,
        key=lambda row: (
            -row["pooled_inner_COD"],
            -row["mean_inner_COD"],
            -row["min_inner_COD"],
            -row["w_anchor"],
            row["top_m"],
            -row["alpha"],
        ),
    )[0]
    augment = augment_cache[int(best["top_m"]), float(best["alpha"])]
    prediction = (
        best["w_anchor"] * anchor_prediction + (1.0 - best["w_anchor"]) * augment
    )
    return (best, prediction)


def candidate_prediction(
    selection,
    matrix_train,
    y_train,
    matrix_test,
    base_prediction=None,
    anchor_prediction=None,
):
    m = int(selection["top_m"])
    alpha = float(selection["alpha"])
    branch = ridge_prediction(matrix_train[:, :m], y_train, matrix_test[:, :m], alpha)
    if selection["method"] == "base_stack":
        weight = float(selection["w_base"])
        return weight * base_prediction + (1.0 - weight) * branch
    if selection["method"] == "sequential":
        weight = float(selection["w_anchor"])
        return weight * anchor_prediction + (1.0 - weight) * branch
    raise RuntimeError("Unknown prediction method")


def main():
    args = parse_args()
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase4 = load_module(args.phase4_py, "visual_phase4")
    phase4.COHORT = Path(args.cohort_csv)
    phase4.FOLDS = Path(args.fold_manifest_csv)
    phase4.FMRI36 = Path(args.fmri_res_csv)
    phase4.SIGNED72 = Path(args.signed72_csv)
    engine = phase4.load_engine()
    df, _, _, signed72, covariate_frame, fold_id, audit, _, _, signed72_columns = (
        phase4.load_data(engine)
    )
    df = normalize_keys(df)
    y_raw = pd.to_numeric(df[TARGET], errors="raise").to_numpy(float)
    covariates = covariate_frame.to_numpy(float)
    groups = df[GROUP].to_numpy(str)
    if len(df) != EXPECTED_N or len(np.unique(groups)) != EXPECTED_CHILDREN:
        raise RuntimeError("Visual N289 cohort contract failed")
    bank = normalize_keys(pd.read_parquet(args.transfer_bank_parquet))
    manifest = pd.read_csv(args.transfer_feature_manifest_tsv, sep="\t")
    if bank[KEYS].duplicated().any():
        raise RuntimeError("Transfer bank contains duplicate keys")
    aligned = df[KEYS].merge(bank, on=KEYS, how="left", validate="one_to_one")
    if aligned["fold"].isna().any():
        raise RuntimeError("Transfer bank alignment failed")
    if not np.array_equal(pd.to_numeric(aligned["fold"]).to_numpy(int), fold_id):
        raise RuntimeError("Transfer bank fold contract failed")
    krrraw, krr_columns = load_family(manifest, aligned, "krrraw_top120", 120)
    dnnraw, dnn_columns = load_family(manifest, aligned, "dnnraw_top120", 120)
    indices = np.arange(len(df))
    outer_folds = [
        (indices[fold_id != fold], indices[fold_id == fold]) for fold in range(5)
    ]
    methods = [
        "null",
        "baseline_signed72_corr",
        "baseline_plus_krrraw_stack",
        "baseline_plus_dnnraw_stack",
        "baseline_plus_pairedraw_stack",
        "krr_anchor_plus_dnnraw_sequential",
        "dnn_anchor_plus_krrraw_sequential",
        "adaptive_dualraw_refinement",
    ]
    predictions = {method: np.full(len(df), np.nan) for method in methods}
    y_oof = np.full(len(df), np.nan)
    inner_rows, selected_rows, fold_rows = ([], [], [])
    for outer_fold, (train, test) in enumerate(outer_folds):
        print(f"[START] outer fold {outer_fold}", flush=True)
        inner_y, inner_id, base_cache, krr_cache, dnn_cache, paired_cache = (
            build_inner_oof(
                phase4,
                engine,
                signed72,
                krrraw,
                dnnraw,
                y_raw,
                covariates,
                groups,
                train,
                args.inner_k,
                args.inner_seed_base + outer_fold,
            )
        )
        current_grid = []
        base_sel, base_inner = select_baseline(
            inner_y, inner_id, base_cache, outer_fold, current_grid
        )
        krr_sel, krr_inner = select_base_stack(
            inner_y, inner_id, base_inner, krr_cache, "krrraw", outer_fold, current_grid
        )
        dnn_sel, dnn_inner = select_base_stack(
            inner_y, inner_id, base_inner, dnn_cache, "dnnraw", outer_fold, current_grid
        )
        paired_sel, paired_inner = select_base_stack(
            inner_y,
            inner_id,
            base_inner,
            paired_cache,
            "pairedraw",
            outer_fold,
            current_grid,
        )
        kd_sel, kd_inner = select_sequential(
            inner_y,
            inner_id,
            krr_inner,
            dnn_cache,
            "krr_anchor_plus_dnnraw",
            outer_fold,
            current_grid,
        )
        dk_sel, dk_inner = select_sequential(
            inner_y,
            inner_id,
            dnn_inner,
            krr_cache,
            "dnn_anchor_plus_krrraw",
            outer_fold,
            current_grid,
        )
        refinement_candidates = [
            ("baseline_plus_pairedraw_stack", paired_sel, paired_inner),
            ("krr_anchor_plus_dnnraw_sequential", kd_sel, kd_inner),
            ("dnn_anchor_plus_krrraw_sequential", dk_sel, dk_inner),
        ]
        adaptive_name, adaptive_sel, adaptive_inner = sorted(
            refinement_candidates,
            key=lambda item: (
                -item[1]["pooled_inner_COD"],
                -item[1]["mean_inner_COD"],
                -item[1]["min_inner_COD"],
                item[0],
            ),
        )[0]
        inner_rows.extend(current_grid)
        y_train, y_test = engine.residualize(
            y_raw[train], y_raw[test], covariates[train], covariates[test]
        )
        y_oof[test] = y_test
        predictions["null"][test] = 0.0
        if base_sel["selected_null"]:
            base_test = np.zeros(len(test), float)
        else:
            base_alpha = float(base_sel["alpha"])
            base_test = baseline_predictions(
                phase4, engine, signed72[train], y_train, signed72[test], [base_alpha]
            )[base_alpha]
        predictions["baseline_signed72_corr"][test] = base_test
        krr_test = candidate_prediction(
            krr_sel, krrraw[train], y_train, krrraw[test], base_prediction=base_test
        )
        dnn_test = candidate_prediction(
            dnn_sel, dnnraw[train], y_train, dnnraw[test], base_prediction=base_test
        )
        paired_train = np.column_stack(
            (
                krrraw[train, : int(paired_sel["top_m"])],
                dnnraw[train, : int(paired_sel["top_m"])],
            )
        )
        paired_test_matrix = np.column_stack(
            (
                krrraw[test, : int(paired_sel["top_m"])],
                dnnraw[test, : int(paired_sel["top_m"])],
            )
        )
        paired_branch_test = ridge_prediction(
            paired_train, y_train, paired_test_matrix, paired_sel["alpha"]
        )
        paired_weight = float(paired_sel["w_base"])
        paired_test = (
            paired_weight * base_test + (1.0 - paired_weight) * paired_branch_test
        )
        kd_test = candidate_prediction(
            kd_sel, dnnraw[train], y_train, dnnraw[test], anchor_prediction=krr_test
        )
        dk_test = candidate_prediction(
            dk_sel, krrraw[train], y_train, krrraw[test], anchor_prediction=dnn_test
        )
        current = {
            "null": np.zeros(len(test), float),
            "baseline_signed72_corr": base_test,
            "baseline_plus_krrraw_stack": krr_test,
            "baseline_plus_dnnraw_stack": dnn_test,
            "baseline_plus_pairedraw_stack": paired_test,
            "krr_anchor_plus_dnnraw_sequential": kd_test,
            "dnn_anchor_plus_krrraw_sequential": dk_test,
        }
        current["adaptive_dualraw_refinement"] = current[adaptive_name]
        for method, values in current.items():
            predictions[method][test] = values
        selections = [
            ("baseline_signed72_corr", base_sel),
            ("baseline_plus_krrraw_stack", krr_sel),
            ("baseline_plus_dnnraw_stack", dnn_sel),
            ("baseline_plus_pairedraw_stack", paired_sel),
            ("krr_anchor_plus_dnnraw_sequential", kd_sel),
            ("dnn_anchor_plus_krrraw_sequential", dk_sel),
        ]
        for method, selection in selections:
            selected_rows.append({"reported_method": method, **selection})
        selected_rows.append(
            {
                "reported_method": "adaptive_dualraw_refinement",
                "adaptive_selected_method": adaptive_name,
                **adaptive_sel,
            }
        )
        base_fold = metrics(y_test, base_test)
        for method, values in current.items():
            result = metrics(y_test, values)
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": method,
                    "n_train": len(train),
                    "n_test": len(test),
                    **result,
                    "delta_COD_vs_baseline_fold": result["COD"] - base_fold["COD"],
                }
            )
        print(
            f"[DONE] outer fold {outer_fold}: train={len(train)} test={len(test)} adaptive={adaptive_name}",
            flush=True,
        )
    if not np.isfinite(y_oof).all() or any(
        (not np.isfinite(value).all() for value in predictions.values())
    ):
        raise RuntimeError("Incomplete/nonfinite outer OOF predictions")
    baseline_metrics = metrics(y_oof, predictions["baseline_signed72_corr"])
    for key, expected in EXPECTED_BASELINE.items():
        if not np.isclose(baseline_metrics[key], expected, rtol=0, atol=1e-10):
            raise RuntimeError(
                f"Frozen baseline reproduction failed for {key}: {baseline_metrics[key]} vs {expected}"
            )
    for method, expected in EXPECTED_PHASE6.items():
        observed = metrics(y_oof, predictions[method])["COD"]
        if not np.isclose(observed, expected, rtol=0, atol=1e-10):
            raise RuntimeError(
                f"Phase6 reproduction failed for {method}: {observed} vs {expected}"
            )
    summary_rows = []
    for method in methods:
        result = metrics(y_oof, predictions[method])
        delta = result["COD"] - baseline_metrics["COD"]
        summary_rows.append(
            {
                "method": method,
                **result,
                "baseline_COD": baseline_metrics["COD"],
                "delta_COD_vs_baseline": delta,
                "relative_COD_improvement_percent_vs_baseline": delta
                / abs(baseline_metrics["COD"])
                * 100.0,
                "delta_r_vs_baseline": (
                    result["Pearson_r"] - baseline_metrics["Pearson_r"]
                    if np.isfinite(result["Pearson_r"])
                    else np.nan
                ),
                "delta_RMSE_vs_baseline": result["RMSE"] - baseline_metrics["RMSE"],
                "folds_improved_COD": (
                    int(
                        sum(
                            (
                                row["delta_COD_vs_baseline_fold"] > 0
                                for row in fold_rows
                                if row["method"] == method
                            )
                        )
                    )
                    if method != "baseline_signed72_corr"
                    else 0
                ),
                "meets_COD_0p025": bool(result["COD"] >= 0.025),
                "meets_COD_0p030": bool(result["COD"] >= 0.03),
            }
        )
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["COD", "Pearson_r"], ascending=False)
        .reset_index(drop=True)
    )
    wide = df[KEYS + [TARGET, "mother_edu_gt_college"]].copy()
    wide["fold"] = fold_id
    wide["y_true_resid"] = y_oof
    for method in methods:
        wide[f"y_pred_{method}"] = predictions[method]
    tag = args.run_tag
    paths = {
        "summary": output / f"summary_{tag}.csv",
        "fold_metrics": output / f"fold_metrics_{tag}.csv",
        "selected": output / f"selected_hyperparameters_{tag}.csv",
        "inner_grid": output / f"inner_grid_{tag}.csv.gz",
        "predictions": output / f"predictions_wide_{tag}.csv",
        "manifest": output / f"manifest_{tag}.json",
    }
    summary.to_csv(paths["summary"], index=False)
    pd.DataFrame(fold_rows).to_csv(paths["fold_metrics"], index=False)
    pd.DataFrame(selected_rows).to_csv(paths["selected"], index=False)
    pd.DataFrame(inner_rows).to_csv(
        paths["inner_grid"], index=False, compression="gzip"
    )
    wide.to_csv(paths["predictions"], index=False)
    manifest_out = {
        "status": "PASS",
        "run_tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "visual-reception dual-raw prediction",
        "n": len(df),
        "n_children": len(np.unique(groups)),
        "baseline": {
            "method": "signed72_corr_guarded",
            "metrics_reproduced": baseline_metrics,
        },
        "phase6_reproduction": EXPECTED_PHASE6,
        "success_targets": {"minimum_residual_COD": 0.025, "strong_residual_COD": 0.03},
        "families": {
            "krrraw_top120": len(krr_columns),
            "dnnraw_top120": len(dnn_columns),
            "pairedraw": "first M from each frozen source order",
        },
        "grids": {
            "baseline_alphas": phase4.ALPHAS,
            "raw_top_m": RAW_M,
            "paired_top_m": PAIRED_M,
            "ridge_alphas": RIDGE_ALPHAS,
            "w_base": W_BASE,
            "w_anchor": W_ANCHOR,
        },
        "contracts": {
            "same_locked_n289_outer_folds": True,
            "outer_and_inner_grouped_by_child": True,
            "maternal_education_in_y_residualization": True,
            "y_residualization_refit_inside_every_training_split": True,
            "x_not_covariate_residualized": True,
            "feature_scaling_training_only": True,
            "all_hyperparameters_selected_inside_outer_train": True,
            "outer_test_used_for_selection": False,
            "baseline_fallback_allowed": True,
            "selection_metric": "pooled inner COD",
            "development_note": "Candidate families were narrowed after Phase6 outer results; Phase7 is a development estimate, not independent confirmation.",
        },
        "inputs_read_only": {
            "phase4_py": str(Path(args.phase4_py).resolve()),
            "transfer_bank_parquet": str(Path(args.transfer_bank_parquet).resolve()),
            "transfer_feature_manifest_tsv": str(
                Path(args.transfer_feature_manifest_tsv).resolve()
            ),
        },
        "audit": audit,
        "signed72_features": signed72_columns,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nPASS visual-reception dual-raw prediction")


if __name__ == "__main__":
    main()
