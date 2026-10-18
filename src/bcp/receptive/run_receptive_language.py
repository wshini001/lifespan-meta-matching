"""Fit receptive-language PC anchors and conditional disease/raw corrections.

Uses child-grouped outer folds and cached inner-OOF baseline and PC predictions
for target-adaptive feature ranking and correction selection.
"""

from __future__ import annotations
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

PHASE3_PY = Path(__file__).resolve().parent / "transfer_engine.py"
TAG = "receptive_language_n241_targetadaptive_conditionalraw_strict_nested_v1"
FAMILY_SPECS = {
    "krrraw_targetadaptive": {
        "source_family": "krrraw_top120",
        "m_values": [3, 5, 10, 20],
        "priority": "primary",
    },
    "dnnraw_targetadaptive": {
        "source_family": "dnnraw_top120",
        "m_values": [3, 5, 10, 20],
        "priority": "primary",
    },
    "diseaseraw_targetadaptive": {
        "source_family": "diseaseraw330",
        "m_values": [3, 5, 10],
        "priority": "sensitivity",
    },
}
RAW_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
CORRECTION_WEIGHTS = [0.1, 0.25, 0.5, 0.75, 1.0]
GATE_THRESHOLD = 0.0
MIN_POOLED_GAIN = 0.00025
MIN_INNER_FOLDS_IMPROVED = 3
MAX_MIN_INNER_COD_DROP = 0.005
EXPECTED = {
    "baseline_COD": 0.03196035,
    "phase3_anchor_COD": 0.039171885,
    "phase9_COD": 0.042715397992571846,
}


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_subset(p3, y, prediction, fold_id, mask):
    return p3.score_parts(y[mask], prediction[mask], fold_id[mask])


def baseline_choice(p3, y, fold_id, cache, mask):
    rows = [
        {
            "selected_null": True,
            "alpha": np.nan,
            "gamma": 0.0,
            **score_subset(p3, y, np.zeros(len(y)), fold_id, mask),
        }
    ]
    for (alpha, gamma), prediction in cache.items():
        rows.append(
            {
                "selected_null": False,
                "alpha": alpha,
                "gamma": gamma,
                **score_subset(p3, y, prediction, fold_id, mask),
            }
        )
    selected = sorted(
        rows,
        key=lambda row: (
            -row["pooled_inner_COD"],
            -row["mean_inner_COD"],
            -row["min_inner_COD"],
            -int(row["selected_null"]),
            row["gamma"],
            -row["alpha"] if np.isfinite(row["alpha"]) else -np.inf,
        ),
    )[0]
    prediction = (
        np.zeros(len(y), dtype=float)
        if selected["selected_null"]
        else cache[float(selected["alpha"]), float(selected["gamma"])]
    )
    return (selected, prediction)


def anchor_choice(p3, y, fold_id, baseline_cache, transfer_cache, mask):
    baseline, baseline_prediction = baseline_choice(
        p3, y, fold_id, baseline_cache, mask
    )
    rows = []
    for (top_m, alpha), branch in transfer_cache["krrpc_final12"].items():
        for weight in p3.W_BASE:
            prediction = (
                float(weight) * baseline_prediction + (1.0 - float(weight)) * branch
            )
            rows.append(
                {
                    "top_m": int(top_m),
                    "alpha": float(alpha),
                    "w_base": float(weight),
                    **score_subset(p3, y, prediction, fold_id, mask),
                }
            )
    anchor = sorted(
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
    branch = transfer_cache["krrpc_final12"][
        int(anchor["top_m"]), float(anchor["alpha"])
    ]
    prediction = (
        float(anchor["w_base"]) * baseline_prediction
        + (1.0 - float(anchor["w_base"])) * branch
    )
    return (baseline, anchor, prediction)


def phase9_crossfit_oof(p3, y, inner_id, baseline_cache, transfer_cache):
    prediction = np.full(len(y), np.nan)
    audit = []
    for held_fold in sorted(np.unique(inner_id)):
        train_mask = inner_id != held_fold
        valid_mask = inner_id == held_fold
        baseline, baseline_all = baseline_choice(
            p3, y, inner_id, baseline_cache, train_mask
        )
        _, anchor, anchor_all = anchor_choice(
            p3, y, inner_id, baseline_cache, transfer_cache, train_mask
        )
        use_anchor = bool(float(anchor["mean_inner_COD"]) > GATE_THRESHOLD)
        prediction[valid_mask] = (
            anchor_all[valid_mask] if use_anchor else baseline_all[valid_mask]
        )
        audit.append(
            {
                "held_inner_fold": int(held_fold),
                "use_anchor": use_anchor,
                "anchor_alpha": float(anchor["alpha"]),
                "anchor_w_base": float(anchor["w_base"]),
                "anchor_mean_inner_COD_training_folds": float(anchor["mean_inner_COD"]),
                "baseline_alpha": float(baseline["alpha"]),
                "baseline_gamma": float(baseline["gamma"]),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete cross-fitted Phase9 prediction")
    return (prediction, audit)


def selected_anchor_outer(p3, families, train, test, y_train, baseline_test, selection):
    top_m = int(selection["top_m"])
    branch = p3.fit_transfer(
        families["krrpc_final12"][train, :top_m],
        y_train,
        families["krrpc_final12"][test, :top_m],
        float(selection["alpha"]),
    )
    return (
        float(selection["w_base"]) * baseline_test
        + (1.0 - float(selection["w_base"])) * branch
    )


def fold_cod(p3, y, prediction, fold_id):
    return np.asarray(
        [
            p3.metrics(y[fold_id == fold], prediction[fold_id == fold])["COD"]
            for fold in sorted(np.unique(fold_id))
        ]
    )


def orthogonalize_train_test(X_train, X_test, C_train, C_test):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    C_train = np.asarray(C_train, dtype=float)
    C_test = np.asarray(C_test, dtype=float)
    center = C_train.mean(axis=0)
    scale = C_train.std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    C_train_z = (C_train - center) / scale
    C_test_z = (C_test - center) / scale
    D_train = np.column_stack([np.ones(len(C_train_z)), C_train_z])
    D_test = np.column_stack([np.ones(len(C_test_z)), C_test_z])
    coefficients, *_ = np.linalg.lstsq(D_train, X_train, rcond=None)
    return (X_train - D_train @ coefficients, X_test - D_test @ coefficients)


def child_balanced_abs_correlation(X, y, groups):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups).astype(str)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(float)
    weights /= weights.sum()
    y_centered = y - np.sum(weights * y)
    X_centered = X - np.sum(weights[:, None] * X, axis=0)
    covariance = np.sum(weights[:, None] * X_centered * y_centered[:, None], axis=0)
    x_variance = np.sum(weights[:, None] * X_centered**2, axis=0)
    y_variance = float(np.sum(weights * y_centered**2))
    denominator = np.sqrt(np.maximum(x_variance * y_variance, 0.0))
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 1e-12,
    )
    return np.abs(correlation)


def ranked_conditional_matrices(
    raw_train,
    raw_valid,
    pc_train,
    pc_valid,
    phase9_train,
    phase9_valid,
    residual_train,
    groups_train,
):
    C_train = np.column_stack([pc_train, phase9_train])
    C_valid = np.column_stack([pc_valid, phase9_valid])
    X_train, X_valid = orthogonalize_train_test(raw_train, raw_valid, C_train, C_valid)
    association = child_balanced_abs_correlation(X_train, residual_train, groups_train)
    ranking = np.argsort(-association, kind="stable")
    return (X_train, X_valid, ranking, association)


def build_targetadaptive_oof(
    p3, families, groups, outer_train, y, inner_id, phase9_oof
):
    cache = {}
    ranking_audit = []
    for label, spec in FAMILY_SPECS.items():
        for top_m in spec["m_values"]:
            for alpha in RAW_ALPHAS:
                cache[label, int(top_m), float(alpha)] = np.full(len(y), np.nan)
    pc = families["krrpc_final12"]
    residual = y - phase9_oof
    for held_fold in sorted(np.unique(inner_id)):
        train_mask = inner_id != held_fold
        valid_mask = inner_id == held_fold
        train = outer_train[train_mask]
        valid = outer_train[valid_mask]
        for label, spec in FAMILY_SPECS.items():
            family = spec["source_family"]
            X_train, X_valid, ranking, association = ranked_conditional_matrices(
                families[family][train],
                families[family][valid],
                pc[train],
                pc[valid],
                phase9_oof[train_mask],
                phase9_oof[valid_mask],
                residual[train_mask],
                groups[train],
            )
            ranking_audit.append(
                {
                    "held_inner_fold": int(held_fold),
                    "family": label,
                    "top20_indices": json.dumps(ranking[:20].astype(int).tolist()),
                    "top20_abs_childbalanced_r": json.dumps(
                        association[ranking[:20]].astype(float).tolist()
                    ),
                }
            )
            for top_m in spec["m_values"]:
                selected = ranking[: int(top_m)]
                for alpha in RAW_ALPHAS:
                    cache[label, int(top_m), float(alpha)][valid_mask] = (
                        p3.fit_transfer(
                            X_train[:, selected],
                            residual[train_mask],
                            X_valid[:, selected],
                            float(alpha),
                        )
                    )
    for key, prediction in cache.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Incomplete target-adaptive OOF prediction: {key}")
    return (cache, ranking_audit)


def select_candidates(p3, y, inner_id, phase9_oof, cache, outer_fold):
    phase9_score = p3.score_parts(y, phase9_oof, inner_id)
    phase9_folds = fold_cod(p3, y, phase9_oof, inner_id)
    selected = {}
    grid = []
    eligible_all = []

    def fallback(label):
        return {
            "outer_fold": outer_fold,
            "family": label,
            "top_m": 0,
            "alpha": np.nan,
            "correction_weight": 0.0,
            "selected_phase9_fallback": True,
            "delta_pooled_inner_COD_vs_phase9": 0.0,
            "inner_folds_improved_vs_phase9": 0,
            "passes_stability_guard": True,
            **phase9_score,
        }

    for label, spec in FAMILY_SPECS.items():
        eligible = []
        for top_m in spec["m_values"]:
            for alpha in RAW_ALPHAS:
                correction = cache[label, int(top_m), float(alpha)]
                for weight in CORRECTION_WEIGHTS:
                    prediction = phase9_oof + float(weight) * correction
                    score = p3.score_parts(y, prediction, inner_id)
                    improved = int(
                        np.sum(fold_cod(p3, y, prediction, inner_id) > phase9_folds)
                    )
                    delta = float(
                        score["pooled_inner_COD"] - phase9_score["pooled_inner_COD"]
                    )
                    stable = bool(
                        delta >= MIN_POOLED_GAIN
                        and improved >= MIN_INNER_FOLDS_IMPROVED
                        and (score["mean_inner_COD"] >= phase9_score["mean_inner_COD"])
                        and (
                            score["min_inner_COD"]
                            >= phase9_score["min_inner_COD"] - MAX_MIN_INNER_COD_DROP
                        )
                    )
                    row = {
                        "outer_fold": outer_fold,
                        "family": label,
                        "priority": spec["priority"],
                        "top_m": int(top_m),
                        "alpha": float(alpha),
                        "correction_weight": float(weight),
                        "selected_phase9_fallback": False,
                        "delta_pooled_inner_COD_vs_phase9": delta,
                        "inner_folds_improved_vs_phase9": improved,
                        "passes_stability_guard": stable,
                        **score,
                    }
                    grid.append(row)
                    if stable:
                        eligible.append(row)
                        eligible_all.append(row)

        def key(row):
            return (
                -row["pooled_inner_COD"],
                -row["mean_inner_COD"],
                -row["min_inner_COD"],
                row["top_m"],
                row["correction_weight"],
                -row["alpha"],
            )

        selected[label] = sorted(eligible, key=key)[0] if eligible else fallback(label)
    primary_eligible = [
        row
        for row in eligible_all
        if FAMILY_SPECS[row["family"]]["priority"] == "primary"
    ]
    selected["adaptive_primary"] = (
        sorted(
            primary_eligible,
            key=lambda row: (
                -row["pooled_inner_COD"],
                -row["mean_inner_COD"],
                -row["min_inner_COD"],
                row["family"],
                row["top_m"],
                row["correction_weight"],
                -row["alpha"],
            ),
        )[0]
        if primary_eligible
        else fallback("phase9_fallback")
    )
    return (selected, grid)


def fit_outer_correction(
    p3,
    families,
    family_columns,
    groups,
    train,
    test,
    residual_target,
    phase9_train,
    phase9_test,
    selection,
):
    if selection["selected_phase9_fallback"]:
        return (np.zeros(len(test), dtype=float), [], [])
    label = selection["family"]
    source_family = FAMILY_SPECS[label]["source_family"]
    pc = families["krrpc_final12"]
    X_train, X_test, ranking, association = ranked_conditional_matrices(
        families[source_family][train],
        families[source_family][test],
        pc[train],
        pc[test],
        phase9_train,
        phase9_test,
        residual_target,
        groups[train],
    )
    chosen = ranking[: int(selection["top_m"])]
    correction = p3.fit_transfer(
        X_train[:, chosen],
        residual_target,
        X_test[:, chosen],
        float(selection["alpha"]),
    )
    names = [family_columns[source_family][int(i)] for i in chosen]
    strengths = association[chosen].astype(float).tolist()
    return (correction, names, strengths)


def method_summary(p3, y, predictions, method, baseline, phase9, fold_rows):
    current = p3.metrics(y, predictions[method])
    delta = current["COD"] - baseline["COD"]
    return {
        "method": method,
        **current,
        "baseline_COD": baseline["COD"],
        "phase9_COD": phase9["COD"],
        "delta_COD_vs_baseline": delta,
        "delta_COD_vs_phase9": current["COD"] - phase9["COD"],
        "relative_COD_improvement_percent_vs_baseline": 100.0
        * delta
        / abs(baseline["COD"]),
        "folds_improved_vs_phase9": (
            int(
                sum(
                    (
                        row["delta_COD_vs_phase9_fold"] > 0
                        for row in fold_rows
                        if row["method"] == method
                    )
                )
            )
            if method != "phase9_inner_stability_gated_krrpc"
            else 0
        ),
        "meets_COD_0p040": bool(current["COD"] >= 0.04),
        "meets_COD_0p050": bool(current["COD"] >= 0.05),
    }


def main():
    global OUT
    parser = argparse.ArgumentParser(
        description="Fit receptive-language conditional raw transfer."
    )
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--area-csv", required=True)
    parser.add_argument("--transfer-bank", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    p3 = load_module(PHASE3_PY, "receptive_phase3")
    p3.COHORT = Path(args.cohort_csv)
    p3.RAW_AREA = Path(args.area_csv)
    p3.BANK = Path(args.transfer_bank)
    p3.FEATURE_MANIFEST = Path(args.feature_manifest)
    base = p3.load_module(p3.PHASE1_PY, "receptive_phase1")
    engine = base.load_module(base.ENGINE_PATH, "receptive_phase0d_engine")
    frame, area, rsfc, y_raw, cov, groups, fold_id, _, _ = p3.load_data(base, engine)
    families, family_columns = p3.load_families(frame, fold_id)
    family_methods = {label: f"phase9_plus_{label}_guarded" for label in FAMILY_SPECS}
    methods = [
        "null",
        "formal_baseline",
        "phase3_krrpc_anchor",
        "phase9_inner_stability_gated_krrpc",
        *family_methods.values(),
        "adaptive_primary_targetadaptive_raw_exploratory",
    ]
    predictions = {method: np.full(len(frame), np.nan) for method in methods}
    y_oof = np.full(len(frame), np.nan)
    selected_rows = []
    grid_rows = []
    fold_rows = []
    ranking_rows = []
    crossfit_rows = []
    index = np.arange(len(frame))
    for outer_fold in range(5):
        train = index[fold_id != outer_fold]
        test = index[fold_id == outer_fold]
        print(f"[START] outer fold {outer_fold}", flush=True)
        inner_y, inner_id, baseline_cache, transfer_cache = p3.build_inner_oof(
            base, engine, area, rsfc, families, y_raw, cov, groups, train, outer_fold
        )
        stage1, _ = p3.select_inner(
            base, inner_y, inner_id, baseline_cache, transfer_cache, outer_fold
        )
        baseline_selection = stage1["baseline"]
        anchor_selection = stage1["families"]["krrpc_final12"]["stack"]
        use_anchor = bool(float(anchor_selection["mean_inner_COD"]) > GATE_THRESHOLD)
        phase9_cv, crossfit_audit = phase9_crossfit_oof(
            p3, inner_y, inner_id, baseline_cache, transfer_cache
        )
        crossfit_rows.extend(
            [{"outer_fold": outer_fold, **row} for row in crossfit_audit]
        )
        raw_cache, rank_audit = build_targetadaptive_oof(
            p3, families, groups, train, inner_y, inner_id, phase9_cv
        )
        ranking_rows.extend([{"outer_fold": outer_fold, **row} for row in rank_audit])
        selected, grid = select_candidates(
            p3, inner_y, inner_id, phase9_cv, raw_cache, outer_fold
        )
        grid_rows.extend(grid)
        y_train, y_test = base.residualize_train_test(
            y_raw[train], y_raw[test], cov[train], cov[test]
        )
        y_oof[test] = y_test
        baseline_test = p3.fit_baseline_outer(
            base, engine, area, rsfc, y_train, train, test, baseline_selection
        )
        anchor_test = selected_anchor_outer(
            p3, families, train, test, y_train, baseline_test, anchor_selection
        )
        phase9_test = anchor_test if use_anchor else baseline_test
        residual_target = inner_y - phase9_cv
        current = {
            "null": np.zeros(len(test), dtype=float),
            "formal_baseline": baseline_test,
            "phase3_krrpc_anchor": anchor_test,
            "phase9_inner_stability_gated_krrpc": phase9_test,
        }
        for label, method in family_methods.items():
            selection = selected[label]
            correction, names, strengths = fit_outer_correction(
                p3,
                families,
                family_columns,
                groups,
                train,
                test,
                residual_target,
                phase9_cv,
                phase9_test,
                selection,
            )
            current[method] = (
                phase9_test.copy()
                if selection["selected_phase9_fallback"]
                else phase9_test + float(selection["correction_weight"]) * correction
            )
            selected_rows.append(
                {
                    "reported_method": method,
                    **selection,
                    "selected_feature_names": json.dumps(names),
                    "selected_abs_childbalanced_r": json.dumps(strengths),
                }
            )
        adaptive = selected["adaptive_primary"]
        correction, names, strengths = fit_outer_correction(
            p3,
            families,
            family_columns,
            groups,
            train,
            test,
            residual_target,
            phase9_cv,
            phase9_test,
            adaptive,
        )
        adaptive_method = "adaptive_primary_targetadaptive_raw_exploratory"
        current[adaptive_method] = (
            phase9_test.copy()
            if adaptive["selected_phase9_fallback"]
            else phase9_test + float(adaptive["correction_weight"]) * correction
        )
        selected_rows.append(
            {
                "reported_method": adaptive_method,
                **adaptive,
                "selected_feature_names": json.dumps(names),
                "selected_abs_childbalanced_r": json.dumps(strengths),
            }
        )
        for method, prediction in current.items():
            predictions[method][test] = prediction
        base_fold = p3.metrics(y_test, baseline_test)
        phase9_fold = p3.metrics(y_test, phase9_test)
        for method, prediction in current.items():
            met = p3.metrics(y_test, prediction)
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": method,
                    "n_test": int(len(test)),
                    **met,
                    "delta_COD_vs_baseline_fold": met["COD"] - base_fold["COD"],
                    "delta_COD_vs_phase9_fold": met["COD"] - phase9_fold["COD"],
                }
            )
        print(
            f"[DONE] outer fold {outer_fold}: phase9={('anchor' if use_anchor else 'baseline')} adaptive={adaptive['family']} M={adaptive['top_m']} weight={adaptive['correction_weight']} fallback={adaptive['selected_phase9_fallback']}",
            flush=True,
        )
    if not np.isfinite(y_oof).all():
        raise RuntimeError("Incomplete outer-OOF outcome")
    for method, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Incomplete predictions for {method}")
    baseline_metrics = p3.metrics(y_oof, predictions["formal_baseline"])
    anchor_metrics = p3.metrics(y_oof, predictions["phase3_krrpc_anchor"])
    phase9_metrics = p3.metrics(
        y_oof, predictions["phase9_inner_stability_gated_krrpc"]
    )
    for label, observed, expected in [
        ("baseline", baseline_metrics["COD"], EXPECTED["baseline_COD"]),
        ("phase3", anchor_metrics["COD"], EXPECTED["phase3_anchor_COD"]),
        ("phase9", phase9_metrics["COD"], EXPECTED["phase9_COD"]),
    ]:
        if not np.isclose(observed, expected, atol=5e-07, rtol=0.0):
            raise RuntimeError(
                f"{label} reproduction failed: observed={observed}, expected={expected}"
            )
    summary = pd.DataFrame(
        [
            method_summary(
                p3,
                y_oof,
                predictions,
                method,
                baseline_metrics,
                phase9_metrics,
                fold_rows,
            )
            for method in methods
        ]
    ).sort_values(["COD", "Pearson_r"], ascending=False)
    wide = frame[["scan_id", "imaging_numeric_id"]].copy()
    wide["outer_fold"] = fold_id
    wide["y_true_resid"] = y_oof
    for method, prediction in predictions.items():
        wide[method] = prediction
    paths = {
        "summary": OUT / f"summary_{TAG}.csv",
        "fold_metrics": OUT / f"fold_metrics_{TAG}.csv",
        "selected": OUT / f"selected_hyperparameters_{TAG}.csv",
        "inner_grid": OUT / f"inner_grid_{TAG}.csv.gz",
        "ranking_audit": OUT / f"ranking_audit_{TAG}.csv.gz",
        "crossfit_audit": OUT / f"phase9_crossfit_audit_{TAG}.csv",
        "predictions": OUT / f"predictions_wide_{TAG}.csv",
        "manifest": OUT / f"manifest_{TAG}.json",
    }
    summary.to_csv(paths["summary"], index=False)
    pd.DataFrame(fold_rows).to_csv(paths["fold_metrics"], index=False)
    pd.DataFrame(selected_rows).to_csv(paths["selected"], index=False)
    pd.DataFrame(grid_rows).to_csv(paths["inner_grid"], index=False, compression="gzip")
    pd.DataFrame(ranking_rows).to_csv(
        paths["ranking_audit"], index=False, compression="gzip"
    )
    pd.DataFrame(crossfit_rows).to_csv(paths["crossfit_audit"], index=False)
    wide.to_csv(paths["predictions"], index=False)
    manifest = {
        "status": "PASS",
        "tag": TAG,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": int(len(frame)),
        "n_children": int(frame["imaging_numeric_id"].nunique()),
        "formal_baseline": "rsfc_plus_area_rescent_guarded_with_maternal_education",
        "family_specs": FAMILY_SPECS,
        "raw_alphas": RAW_ALPHAS,
        "correction_weights": CORRECTION_WEIGHTS,
        "selection_guard": {
            "minimum_pooled_inner_COD_gain": MIN_POOLED_GAIN,
            "minimum_inner_folds_improved": MIN_INNER_FOLDS_IMPROVED,
            "mean_inner_COD_must_not_decrease": True,
            "maximum_min_inner_COD_drop": MAX_MIN_INNER_COD_DROP,
        },
        "contracts": {
            "same_locked_n241_outer_folds": True,
            "outer_and_inner_grouped_by_child": True,
            "maternal_education_in_y_residualization": True,
            "phase9_prediction_crossfitted_inside_outer_train": True,
            "raw_orthogonalization_training_only": True,
            "raw_ranking_training_only": True,
            "raw_ranking_child_balanced": True,
            "M_alpha_weight_selected_inside_outer_train": True,
            "outer_test_used_for_selection": False,
            "disease_family_sensitivity_only": True,
            "adaptive_primary_family_exploratory_only": True,
            "post_hoc_after_phase11": True,
        },
        "inputs_read_only": {
            "phase3_engine": str(PHASE3_PY),
            "transfer_bank": str(p3.BANK),
            "feature_manifest": str(p3.FEATURE_MANIFEST),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))
    print("\nReceptive-language conditional raw transfer completed")


if __name__ == "__main__":
    main()
