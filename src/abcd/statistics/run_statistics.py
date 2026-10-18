"""Calculate ABCD 12-PC route statistics from OOF predictions and ensemble weights."""

import argparse
import json
from pathlib import Path
import numpy as np
import statistics_engine as engine


def run_route(route, prediction_path, weights_path, out):
    config = Path(__file__).resolve().parent / "config"
    cfg = json.loads((config / f"{route}.json").read_text())
    methods = [cfg[k] for k in ["baseline_col", "pc_col", "avg_col", "ensemble_col"]]
    data = engine.load_predictions(
        str(prediction_path),
        cfg["subjectkey_col"],
        cfg["id_col"],
        cfg["fold_col"],
        cfg["y_col"],
        methods,
    )
    keys = [cfg["subjectkey_col"], cfg["id_col"]]
    if (
        len(data) != cfg["expected_n"]
        or data[keys].isna().any().any()
        or set(data[cfg["fold_col"]]) != set(range(5))
    ):
        raise ValueError(
            "Expected locked cohort size, nonmissing IDs and five outer folds"
        )
    weights = engine.load_weights(str(weights_path))
    if set(weights.outer_fold) != set(data[cfg["fold_col"]]):
        raise ValueError("Prediction/weight folds differ")
    reconstructed = engine.reconstruct_ensemble_from_weights(
        data, weights, cfg["fold_col"], cfg["baseline_col"], cfg["pc_col"]
    )
    error = float(
        np.max(np.abs(reconstructed - data[cfg["ensemble_col"]].to_numpy(float)))
    )
    if error > 1e-08:
        raise ValueError(
            f"Saved weights do not reconstruct the ensemble: maximum error={error:.12e}"
        )
    print(
        f"PASS: {route}; weights reconstruct saved ensemble; maximum error={error:.3e}",
        flush=True,
    )
    pooled, folds, _ = engine.summarize_methods(
        data, cfg["fold_col"], cfg["y_col"], methods, cfg["baseline_col"]
    )
    paired = engine.run_paired_error_tests(
        data, cfg["y_col"], cfg["baseline_col"], cfg["ensemble_col"]
    )
    print(
        f"[{route}] Running {cfg['n_boot']} participant bootstraps (seed={cfg['seed']})",
        flush=True,
    )
    bootstrap, _ = engine.run_bootstrap(
        data,
        cfg["y_col"],
        methods,
        cfg["baseline_col"],
        cfg["ensemble_col"],
        cfg["n_boot"],
        cfg["seed"],
    )
    print(
        f"[{route}] Running {cfg['n_perm']} within-fold PC-stream permutations (seed={cfg['seed'] + 991}); weights fixed",
        flush=True,
    )
    permutation, _ = engine.run_pc_stream_permutation(
        data,
        weights,
        cfg["y_col"],
        cfg["fold_col"],
        cfg["baseline_col"],
        cfg["baseline_col"],
        cfg["pc_col"],
        cfg["ensemble_col"],
        cfg["n_perm"],
        cfg["seed"] + 991,
    )
    out.mkdir(parents=True, exist_ok=True)
    for name, table in [
        ("pooled_metrics", pooled),
        ("fold_metrics", folds),
        ("bootstrap", bootstrap),
        ("permutation", permutation),
    ]:
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "paired_tests.json").write_text(json.dumps(paired, indent=2) + "\n")
    print(f"Calculated statistics for {route}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", choices=["dnn12", "krr12"], required=True)
    p.add_argument("--predictions-csv", required=True)
    p.add_argument("--weights-csv", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    run_route(
        args.route,
        Path(args.predictions_csv),
        Path(args.weights_csv),
        Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
