"""Run the manuscript HCP FluidCog RSFC plus Final12-PC transfer model.

The preassigned outer folds and all inner folds are family-grouped. Outcome
residualization, PC standardization, KRR penalties, and the PC correction
weight are estimated only from training data. The outer test fold is used
only once for evaluation. RSFC features are row-normalized but are not
covariate-residualized.
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression


def parse_float_list(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one numeric value is required")
    return values


def correlation(y_true: np.ndarray, prediction: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(prediction)
    y_true, prediction = (y_true[valid], prediction[valid])
    if len(y_true) < 3 or np.std(y_true) <= 0 or np.std(prediction) <= 0:
        return np.nan
    return float(np.corrcoef(y_true, prediction)[0, 1])


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(prediction)
    y_true, prediction = (y_true[valid], prediction[valid])
    sse = float(np.sum((y_true - prediction) ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    cod = float(1.0 - sse / sst) if sst > 0 else np.nan
    return {
        "n": int(len(y_true)),
        "Pearson_r": correlation(y_true, prediction),
        "COD": cod,
        "RMSE": float(np.sqrt(np.mean((y_true - prediction) ** 2))),
        "MAE": float(np.mean(np.abs(y_true - prediction))),
        "SSE": sse,
    }


def score(y_true: np.ndarray, prediction: np.ndarray, name: str) -> float:
    values = metrics(y_true, prediction)
    return values["Pearson_r"] if name == "pearson" else values["COD"]


def unitize_rows(matrix: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(matrix * matrix, axis=1, keepdims=True))
    return matrix / np.maximum(norms, epsilon)


def standardize_train_apply(
    train: np.ndarray, test: np.ndarray, epsilon: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale = np.where(scale < epsilon, 1.0, scale)
    return ((train - mean) / scale, (test - mean) / scale)


def residualize_train_apply(
    y_train: np.ndarray,
    y_test: np.ndarray,
    covariates_train: np.ndarray,
    covariates_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = LinearRegression()
    model.fit(covariates_train, y_train)
    return (
        y_train - model.predict(covariates_train),
        y_test - model.predict(covariates_test),
    )


def make_group_folds(groups: np.ndarray, n_folds: int, seed: int):
    groups = np.asarray(groups).astype(str)
    rng = np.random.default_rng(seed)
    unique_groups, counts = np.unique(groups, return_counts=True)
    order = np.arange(len(unique_groups))
    rng.shuffle(order)
    unique_groups, counts = (unique_groups[order], counts[order])
    fold_load = np.zeros(n_folds, dtype=int)
    group_to_fold = {}
    for group, count in sorted(zip(unique_groups, counts), key=lambda item: -item[1]):
        fold = int(np.argmin(fold_load))
        group_to_fold[group] = fold
        fold_load[fold] += int(count)
    fold_id = np.array([group_to_fold[group] for group in groups], dtype=int)
    indices = np.arange(len(groups))
    splits = []
    for fold in range(n_folds):
        validation = indices[fold_id == fold]
        training = indices[fold_id != fold]
        if len(validation) == 0 or len(training) == 0:
            raise RuntimeError("An inner fold is empty")
        if set(groups[training]) & set(groups[validation]):
            raise RuntimeError("Family leakage detected in an inner split")
        splits.append((training, validation))
    return splits


def load_covariates(path: str, name: str) -> list[str]:
    definitions = json.loads(Path(path).read_text(encoding="utf-8"))
    if name not in definitions:
        raise KeyError(f"Unknown covariate set {name}; available: {list(definitions)}")
    return list(definitions[name])


def read_input_table(path):
    return (
        pd.read_csv(path)
        if Path(path).suffix.lower() == ".csv"
        else pd.read_parquet(path)
    )


def load_data(args: argparse.Namespace) -> dict:
    locked = read_input_table(args.locked_input)
    pc = read_input_table(args.pc_predictions)
    locked["Subject"] = locked["Subject"].astype(str)
    pc["Subject"] = pc["Subject"].astype(str)
    if not locked["Subject"].equals(pc["Subject"]):
        raise RuntimeError(
            "Subject order differs between locked input and PC predictions"
        )
    if "fold" in pc.columns and (not locked["fold"].equals(pc["fold"])):
        raise RuntimeError("Fold labels differ between locked input and PC predictions")
    edge_columns = sorted(
        [c for c in locked.columns if c.startswith("e") and c[1:].isdigit()],
        key=lambda value: int(value[1:]),
    )
    if edge_columns != [f"e{i}" for i in range(1, 1486)]:
        raise RuntimeError("Expected the ordered RSFC edge contract e1..e1485")
    covariate_columns = load_covariates(args.covariates_json, args.covariate_set)
    for column in [args.target, args.group, "fold", *covariate_columns]:
        if column not in locked.columns:
            raise KeyError(f"Locked HCP table is missing {column}")
    pc_columns = [column for column in pc.columns if column.endswith("__zpred")]
    if len(pc_columns) != args.expected_pc_count:
        raise RuntimeError(
            f"Expected {args.expected_pc_count} transferred PC columns; found {len(pc_columns)}"
        )
    arrays = {
        "y": locked[args.target].to_numpy(dtype=np.float64),
        "covariates": locked[covariate_columns].to_numpy(dtype=np.float64),
        "edges": locked[edge_columns].to_numpy(dtype=np.float64),
        "pcs": pc[pc_columns].to_numpy(dtype=np.float64),
    }
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values in {name}")
    metadata_columns = [
        "locked_row_idx",
        "edge_row_idx",
        "Subject",
        "fold",
        args.group,
        args.target,
        "Sex",
        "Age_in_Yrs",
    ]
    return {
        **arrays,
        "fold_id": locked["fold"].to_numpy(dtype=int),
        "groups": locked[args.group].astype(str).to_numpy(),
        "metadata": locked[[c for c in metadata_columns if c in locked.columns]].copy(),
        "edge_columns": edge_columns,
        "pc_columns": pc_columns,
        "covariate_columns": covariate_columns,
    }


def fit_rsfc_stream(
    edges,
    y,
    covariates,
    outer_train,
    outer_test,
    inner_splits,
    alpha_grid,
    selection_metric,
):
    normalized_edges = unitize_rows(edges)
    best = None
    alpha_rows = []
    for alpha in alpha_grid:
        oof_prediction = np.full(len(outer_train), np.nan)
        oof_y = np.full(len(outer_train), np.nan)
        fold_scores = []
        for train_relative, validation_relative in inner_splits:
            train = outer_train[train_relative]
            validation = outer_train[validation_relative]
            y_train, y_validation = residualize_train_apply(
                y[train], y[validation], covariates[train], covariates[validation]
            )
            model = KernelRidge(alpha=float(alpha), kernel="precomputed")
            model.fit(normalized_edges[train] @ normalized_edges[train].T, y_train)
            prediction = model.predict(
                normalized_edges[validation] @ normalized_edges[train].T
            )
            oof_prediction[validation_relative] = prediction
            oof_y[validation_relative] = y_validation
            fold_scores.append(score(y_validation, prediction, selection_metric))
        mean_score = float(np.nanmean(fold_scores))
        alpha_rows.append(
            {"stream": "rsfc", "alpha": float(alpha), "inner_score": mean_score}
        )
        if best is None or mean_score > best["inner_score"]:
            best = {
                "alpha": float(alpha),
                "inner_score": mean_score,
                "oof_prediction": oof_prediction.copy(),
                "oof_y": oof_y.copy(),
            }
    y_train, y_test = residualize_train_apply(
        y[outer_train], y[outer_test], covariates[outer_train], covariates[outer_test]
    )
    model = KernelRidge(alpha=best["alpha"], kernel="precomputed")
    model.fit(normalized_edges[outer_train] @ normalized_edges[outer_train].T, y_train)
    best["test_prediction"] = model.predict(
        normalized_edges[outer_test] @ normalized_edges[outer_train].T
    )
    best["test_y"] = y_test
    best["alpha_rows"] = alpha_rows
    return best


def fit_pc_stream(
    pcs,
    y,
    covariates,
    outer_train,
    outer_test,
    inner_splits,
    alpha_grid,
    selection_metric,
):
    best = None
    alpha_rows = []
    for alpha in alpha_grid:
        oof_prediction = np.full(len(outer_train), np.nan)
        oof_y = np.full(len(outer_train), np.nan)
        fold_scores = []
        for train_relative, validation_relative in inner_splits:
            train = outer_train[train_relative]
            validation = outer_train[validation_relative]
            y_train, y_validation = residualize_train_apply(
                y[train], y[validation], covariates[train], covariates[validation]
            )
            x_train, x_validation = standardize_train_apply(pcs[train], pcs[validation])
            model = KernelRidge(alpha=float(alpha), kernel="linear")
            model.fit(x_train, y_train)
            prediction = model.predict(x_validation)
            oof_prediction[validation_relative] = prediction
            oof_y[validation_relative] = y_validation
            fold_scores.append(score(y_validation, prediction, selection_metric))
        mean_score = float(np.nanmean(fold_scores))
        alpha_rows.append(
            {"stream": "pc", "alpha": float(alpha), "inner_score": mean_score}
        )
        if best is None or mean_score > best["inner_score"]:
            best = {
                "alpha": float(alpha),
                "inner_score": mean_score,
                "oof_prediction": oof_prediction.copy(),
                "oof_y": oof_y.copy(),
            }
    y_train, y_test = residualize_train_apply(
        y[outer_train], y[outer_test], covariates[outer_train], covariates[outer_test]
    )
    x_train, x_test = standardize_train_apply(pcs[outer_train], pcs[outer_test])
    model = KernelRidge(alpha=best["alpha"], kernel="linear")
    model.fit(x_train, y_train)
    best["test_prediction"] = model.predict(x_test)
    best["test_y"] = y_test
    best["alpha_rows"] = alpha_rows
    return best


def select_pc_weight(y, rsfc_prediction, pc_prediction, grid, metric_name):
    rows = []
    best = None
    for weight in grid:
        prediction = rsfc_prediction + weight * pc_prediction
        values = metrics(y, prediction)
        row = {
            "pc_weight": float(weight),
            "inner_Pearson_r": values["Pearson_r"],
            "inner_COD": values["COD"],
            "inner_RMSE": values["RMSE"],
            "inner_MAE": values["MAE"],
        }
        rows.append(row)
        value = values["Pearson_r"] if metric_name == "pearson" else values["COD"]
        if best is None or value > best["selection_value"]:
            best = {**row, "selection_value": value}
    return (best, rows)


def summarize(predictions: pd.DataFrame, prediction_columns: list[str]):
    fold_rows = []
    pooled = {}
    for column in prediction_columns:
        pooled[column] = metrics(predictions["y_true_resid"], predictions[column])
        for fold, subset in predictions.groupby("fold"):
            values = metrics(subset["y_true_resid"], subset[column])
            fold_rows.append({"method": column, "fold": int(fold), **values})
    baseline_cod = pooled["y_pred_rsfc_nested"]["COD"]
    baseline_r = pooled["y_pred_rsfc_nested"]["Pearson_r"]
    for column in prediction_columns:
        pooled[column]["delta_COD_vs_rsfc"] = pooled[column]["COD"] - baseline_cod
        pooled[column]["delta_Pearson_r_vs_rsfc"] = (
            pooled[column]["Pearson_r"] - baseline_r
        )
        pooled[column]["relative_COD_gain_percent"] = (
            100.0 * (pooled[column]["COD"] - baseline_cod) / baseline_cod
            if baseline_cod != 0
            else np.nan
        )
    return (pd.DataFrame(fold_rows), {"pooled": pooled})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-input", required=True)
    parser.add_argument("--pc-predictions", required=True)
    parser.add_argument("--covariates-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="FluidCog")
    parser.add_argument("--group", default="family_group")
    parser.add_argument("--covariate-set", default="basic_age_sex")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-pc-count", type=int, default=12)
    parser.add_argument("--expected-participants", type=int, default=1080)
    parser.add_argument("--expected-family-groups", type=int, default=465)
    parser.add_argument("--rsfc-alpha-grid", default="0.4,1.0,4.0,10.0")
    parser.add_argument("--pc-alpha-grid", default="5.0,10.0,15.0,20.0")
    parser.add_argument(
        "--pc-weight-grid",
        default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7",
    )
    parser.add_argument(
        "--alpha-selection-metric", choices=["pearson", "cod"], default="pearson"
    )
    parser.add_argument(
        "--weight-selection-metric", choices=["pearson", "cod"], default="cod"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rsfc_alpha_grid = parse_float_list(args.rsfc_alpha_grid)
    pc_alpha_grid = parse_float_list(args.pc_alpha_grid)
    pc_weight_grid = parse_float_list(args.pc_weight_grid)
    data = load_data(args)
    if args.expected_participants > 0 and len(data["y"]) != args.expected_participants:
        raise RuntimeError(
            f"Expected {args.expected_participants} participants; found {len(data['y'])}"
        )
    observed_groups = int(np.unique(data["groups"]).size)
    if (
        args.expected_family_groups > 0
        and observed_groups != args.expected_family_groups
    ):
        raise RuntimeError(
            f"Expected {args.expected_family_groups} family groups; found {observed_groups}"
        )
    folds = sorted(np.unique(data["fold_id"]).tolist())
    if len(folds) != args.outer_folds:
        raise RuntimeError(f"Expected {args.outer_folds} outer folds; found {folds}")
    n_participants = len(data["y"])
    rsfc_predictions = np.full(n_participants, np.nan)
    pc_predictions = np.full(n_participants, np.nan)
    combined_predictions = np.full(n_participants, np.nan)
    residualized_outcome = np.full(n_participants, np.nan)
    selected_rows, alpha_rows, weight_rows = ([], [], [])
    for outer_fold in folds:
        print(f"[HCP] fitting outer fold {outer_fold}", flush=True)
        outer_train = np.where(data["fold_id"] != outer_fold)[0]
        outer_test = np.where(data["fold_id"] == outer_fold)[0]
        training_groups = data["groups"][outer_train]
        test_groups = data["groups"][outer_test]
        if set(training_groups) & set(test_groups):
            raise RuntimeError(f"Family leakage in outer fold {outer_fold}")
        inner_splits = make_group_folds(
            training_groups, args.inner_folds, args.seed + 1000 * int(outer_fold)
        )
        rsfc = fit_rsfc_stream(
            data["edges"],
            data["y"],
            data["covariates"],
            outer_train,
            outer_test,
            inner_splits,
            rsfc_alpha_grid,
            args.alpha_selection_metric,
        )
        pc = fit_pc_stream(
            data["pcs"],
            data["y"],
            data["covariates"],
            outer_train,
            outer_test,
            inner_splits,
            pc_alpha_grid,
            args.alpha_selection_metric,
        )
        if np.max(np.abs(rsfc["oof_y"] - pc["oof_y"])) > 1e-08:
            raise RuntimeError(f"Inner outcome residuals differ in fold {outer_fold}")
        if np.max(np.abs(rsfc["test_y"] - pc["test_y"])) > 1e-08:
            raise RuntimeError(f"Outer outcome residuals differ in fold {outer_fold}")
        best_weight, tested_weights = select_pc_weight(
            rsfc["oof_y"],
            rsfc["oof_prediction"],
            pc["oof_prediction"],
            pc_weight_grid,
            args.weight_selection_metric,
        )
        weight = float(best_weight["pc_weight"])
        combined = rsfc["test_prediction"] + weight * pc["test_prediction"]
        rsfc_predictions[outer_test] = rsfc["test_prediction"]
        pc_predictions[outer_test] = pc["test_prediction"]
        combined_predictions[outer_test] = combined
        residualized_outcome[outer_test] = rsfc["test_y"]
        rsfc_metrics = metrics(rsfc["test_y"], rsfc["test_prediction"])
        combined_metrics = metrics(rsfc["test_y"], combined)
        selected_rows.append(
            {
                "outer_fold": int(outer_fold),
                "n_train": int(len(outer_train)),
                "n_test": int(len(outer_test)),
                "n_train_groups": int(np.unique(training_groups).size),
                "n_test_groups": int(np.unique(test_groups).size),
                "rsfc_alpha": float(rsfc["alpha"]),
                "pc_alpha": float(pc["alpha"]),
                "pc_weight": weight,
                "inner_weight_COD": float(best_weight["inner_COD"]),
                "inner_weight_Pearson_r": float(best_weight["inner_Pearson_r"]),
                "outer_rsfc_COD": rsfc_metrics["COD"],
                "outer_combined_COD": combined_metrics["COD"],
                "outer_delta_COD": combined_metrics["COD"] - rsfc_metrics["COD"],
            }
        )
        for row in [*rsfc["alpha_rows"], *pc["alpha_rows"]]:
            alpha_rows.append({**row, "outer_fold": int(outer_fold)})
        for row in tested_weights:
            weight_rows.append({**row, "outer_fold": int(outer_fold)})
    predictions = data["metadata"].copy()
    predictions["y_true_resid"] = residualized_outcome
    predictions["y_pred_rsfc_nested"] = rsfc_predictions
    predictions["y_pred_pc_nested"] = pc_predictions
    predictions["y_pred_rsfc_plus_pc_nested"] = combined_predictions
    output_columns = [
        "y_pred_rsfc_nested",
        "y_pred_pc_nested",
        "y_pred_rsfc_plus_pc_nested",
    ]
    for column in ["y_true_resid", *output_columns]:
        if not np.isfinite(predictions[column].to_numpy()).all():
            raise RuntimeError(f"Non-finite output values in {column}")
    fold_metrics, metric_summary = summarize(predictions, output_columns)
    files = {
        "predictions": output_dir / "hcp_rsfc_pc_predictions.csv",
        "fold_metrics": output_dir / "hcp_rsfc_pc_fold_metrics.csv",
        "selected_parameters": output_dir / "hcp_rsfc_pc_selected_parameters.csv",
        "inner_alpha_scores": output_dir / "hcp_rsfc_pc_inner_alpha_scores.csv",
        "inner_weight_scores": output_dir / "hcp_rsfc_pc_inner_weight_scores.csv",
        "metrics": output_dir / "hcp_rsfc_pc_metrics.json",
        "metadata": output_dir / "hcp_rsfc_pc_metadata.json",
    }
    predictions.to_csv(files["predictions"], index=False)
    fold_metrics.to_csv(files["fold_metrics"], index=False)
    pd.DataFrame(selected_rows).to_csv(files["selected_parameters"], index=False)
    pd.DataFrame(alpha_rows).to_csv(files["inner_alpha_scores"], index=False)
    pd.DataFrame(weight_rows).to_csv(files["inner_weight_scores"], index=False)
    files["metrics"].write_text(json.dumps(metric_summary, indent=2), encoding="utf-8")
    metadata = {
        "created_at": datetime.now().isoformat(),
        "model": "family-grouped nested RSFC plus transferred Final12-PC additive prediction",
        "inputs": {
            "locked_input": args.locked_input,
            "pc_predictions": args.pc_predictions,
            "covariates_json": args.covariates_json,
        },
        "configuration": {
            "target": args.target,
            "covariates": data["covariate_columns"],
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "seed": args.seed,
            "rsfc_alpha_grid": rsfc_alpha_grid,
            "pc_alpha_grid": pc_alpha_grid,
            "pc_weight_grid": pc_weight_grid,
            "alpha_selection_metric": args.alpha_selection_metric,
            "weight_selection_metric": args.weight_selection_metric,
        },
        "data_shape": {
            "n_participants": n_participants,
            "n_edges": len(data["edge_columns"]),
            "n_transferred_pcs": len(data["pc_columns"]),
        },
        "leakage_controls": {
            "outer_test_used_for_selection": False,
            "inner_cv_family_grouped": True,
            "outcome_residualization_train_only": True,
            "pc_standardization_train_only": True,
            "rsfc_covariate_residualized": False,
        },
        "outputs": {key: str(value) for key, value in files.items()},
    }
    files["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metric_summary["pooled"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
