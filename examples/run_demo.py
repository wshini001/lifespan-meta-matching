"""Run a common RSFC/PC demonstration for five simulated BCP outcomes."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge

from generate_simulated_data import OUTCOMES

ALPHAS = (0.1, 1.0)
BASELINE_WEIGHTS = (0.0, 0.5, 1.0)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict_stream(kernel_helpers, x, y_train, train, test, alpha):
    train_kernel, test_kernel = kernel_helpers.linear_kernel_train_test(
        x[train], x[test]
    )
    model = KernelRidge(alpha=alpha, kernel="precomputed").fit(train_kernel, y_train)
    return model.predict(test_kernel)


def fit_outcome(cohort, source, outcome, preprocessing, kernels):
    groups = cohort.imaging_numeric_id.to_numpy()
    x = cohort[[f"e{i}" for i in range(1, 1486)]].to_numpy(float)
    z = source[[f"simulated_pc{i:02d}__zpred" for i in range(1, 13)]].to_numpy(float)
    covariates = preprocessing.make_covariates(cohort, include_education=True)
    y = cohort[outcome].to_numpy(float)
    assert covariates.shape[1] == 8
    result = cohort[["scan_id", "imaging_numeric_id"]].copy()
    for column in ["y_true_resid", "y_pred_baseline", "y_pred_transfer"]:
        result[column] = np.nan
    result["fold"] = -1
    selections = []
    for fold, (train, test) in enumerate(kernels.grouped_folds(groups, 5, 42)):
        assert not set(groups[train]) & set(groups[test])
        inner_y = np.full(len(train), np.nan)
        streams = [np.full((len(ALPHAS), len(train)), np.nan) for _ in range(2)]
        for inner_train, inner_test in kernels.grouped_folds(
            groups[train], 3, 100 + fold
        ):
            fit, valid = train[inner_train], train[inner_test]
            assert not set(groups[fit]) & set(groups[valid])
            y_fit, y_valid = preprocessing.residualize_train_test(
                y[fit], y[valid], covariates[fit], covariates[valid]
            )
            inner_y[inner_test] = y_valid
            for stream, features in zip(streams, [x, z]):
                for index, alpha in enumerate(ALPHAS):
                    stream[index, inner_test] = predict_stream(
                        kernels, features, y_fit, fit, valid, alpha
                    )
        best = [
            int(np.argmin(np.mean((stream - inner_y) ** 2, axis=1)))
            for stream in streams
        ]
        inner_base, inner_pc = [stream[index] for stream, index in zip(streams, best)]
        weight = min(
            BASELINE_WEIGHTS,
            key=lambda w: np.mean((inner_y - w * inner_base - (1 - w) * inner_pc) ** 2),
        )
        y_fit, y_test = preprocessing.residualize_train_test(
            y[train], y[test], covariates[train], covariates[test]
        )
        baseline, pc = [
            predict_stream(kernels, features, y_fit, train, test, ALPHAS[index])
            for features, index in zip([x, z], best)
        ]
        result.loc[test, "y_true_resid"] = y_test
        result.loc[test, "y_pred_baseline"] = baseline
        result.loc[test, "y_pred_transfer"] = weight * baseline + (1 - weight) * pc
        result.loc[test, "fold"] = fold
        selections.append(
            dict(
                fold=fold,
                baseline_alpha=ALPHAS[best[0]],
                pc_alpha=ALPHAS[best[1]],
                baseline_weight=weight,
            )
        )
    if result.groupby("imaging_numeric_id").fold.nunique().ne(1).any():
        raise ValueError("A child appears in multiple outer folds")
    if (
        not np.isfinite(result[["y_true_resid", "y_pred_baseline", "y_pred_transfer"]])
        .all()
        .all()
    ):
        raise ValueError("Missing out-of-fold predictions")
    return result, pd.DataFrame(selections)


def run(work, input_dir):
    root = Path(__file__).resolve().parents[1]
    work.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(input_dir / "simulated_cohort.csv")
    source = pd.read_csv(input_dir / "simulated_pc_predictions.csv")
    keys = ["scan_id", "imaging_numeric_id"]
    if not cohort[keys].equals(source[keys]):
        raise ValueError("Simulated cohort and source rows differ")
    module_dir = root / "src/bcp/receptive"
    preprocessing = load_module(
        module_dir / "outcome_preprocessing.py", "demo_preprocessing"
    )
    kernels = load_module(module_dir / "connectivity_kernels.py", "demo_kernels")
    for index, outcome in enumerate(OUTCOMES):
        print(f"[BCP demo] {outcome}", flush=True)
        predictions, selections = fit_outcome(
            cohort, source, outcome, preprocessing, kernels
        )
        out = work / outcome
        out.mkdir(exist_ok=True)
        predictions.to_csv(out / "predictions.csv", index=False)
        selections.to_csv(out / "selected_parameters.csv", index=False)
        config = dict(
            seed=42,
            seed_offset=index,
            bootstrap_batch_size=50,
            signflip_batch_size=50,
            n_cluster_bootstrap=200,
            n_cluster_signflip_permutations=200,
            outcome="SIMULATED " + outcome,
            y_column="y_true_resid",
            baseline_column="y_pred_baseline",
            final_column="y_pred_transfer",
            id_column="imaging_numeric_id",
            fold_column="fold",
            cluster_unit="child",
            n_rows=len(cohort),
            n_clusters=cohort.imaging_numeric_id.nunique(),
        )
        config_path = out / "inference_settings.json"
        config_path.write_text(json.dumps(config, indent=2))
        subprocess.run(
            [
                sys.executable,
                str(root / "src/statistics/infer_prediction_gain.py"),
                "--config",
                str(config_path),
                "--predictions-csv",
                str(out / "predictions.csv"),
                "--cluster-column",
                "imaging_numeric_id",
                "--out-dir",
                str(out / "inference"),
            ],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir", help="Optional directory to retain generated demo files"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "simulated_data",
        help="Directory containing simulated cohort and PC-feature CSV inputs",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    if args.work_dir:
        run(Path(args.work_dir).resolve(), input_dir)
    else:
        with tempfile.TemporaryDirectory(
            prefix="meta_matching_bcp_simulated_"
        ) as directory:
            run(Path(directory), input_dir)


if __name__ == "__main__":
    main()
