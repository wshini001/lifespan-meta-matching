"""Calculate paired cluster inference on fixed outer-fold OOF predictions.

Uses percentile delta-COD bootstrap intervals and paired squared-error sign
flips with separate random streams. Models are not refitted during resampling.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def aggregate_pairs(y, baseline, final, clusters):
    y = np.asarray(y, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    final = np.asarray(final, dtype=np.float64)
    clusters = np.asarray(clusters)
    if (
        y.ndim != 1
        or baseline.shape != y.shape
        or final.shape != y.shape
        or clusters.shape != y.shape
    ):
        raise ValueError(
            "Outcomes, predictions and cluster labels must be aligned 1D arrays."
        )
    if len(y) < 2 or not np.isfinite(np.column_stack([y, baseline, final])).all():
        raise ValueError("Nonfinite outcome/prediction or fewer than two observations.")
    if pd.isna(clusters).any():
        raise ValueError("Missing cluster labels.")
    codes, labels = pd.factorize(clusters, sort=False)
    n_clusters = len(labels)
    if n_clusters < 2:
        raise ValueError("At least two independent resampling clusters are required.")
    centered = y - y.mean()
    sse_baseline = (y - baseline) ** 2
    sse_final = (y - final) ** 2
    sums = np.column_stack(
        [
            np.bincount(codes, minlength=n_clusters),
            np.bincount(codes, weights=centered, minlength=n_clusters),
            np.bincount(codes, weights=centered**2, minlength=n_clusters),
            np.bincount(codes, weights=sse_baseline, minlength=n_clusters),
            np.bincount(codes, weights=sse_final, minlength=n_clusters),
        ]
    )
    return sums, dict(n_rows=len(y), n_clusters=n_clusters)


def cod_from_sums(sums):
    n, sy, sy2, sseb, ssef = np.asarray(sums, dtype=np.float64).T
    sst = sy2 - sy**2 / n
    if np.any(sst <= 0) or not np.isfinite(sst).all():
        raise ValueError("Nonpositive or nonfinite resampled outcome variance.")
    baseline = 1 - sseb / sst
    final = 1 - ssef / sst
    delta = (sseb - ssef) / sst
    return baseline, final, delta


def paired_cluster_bootstrap(sums, rng, n_resamples, batch_size=250):
    if n_resamples < 1 or batch_size < 1:
        raise ValueError("Positive bootstrap count and batch size are required.")
    n_clusters = len(sums)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        sampled = sums[indices].sum(axis=1)
        _, _, deltas[start:stop] = cod_from_sums(sampled)
    return deltas


def paired_cluster_signflip(sums, rng, n_permutations, batch_size=500):
    if n_permutations < 1 or batch_size < 1:
        raise ValueError("Positive permutation count and batch size are required.")
    reductions = sums[:, 3] - sums[:, 4]
    observed = float(reductions.sum())
    statistics = np.empty(n_permutations, dtype=np.float64)
    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        signs = (
            rng.integers(
                0, 2, size=(stop - start, len(reductions)), dtype=np.int8
            ).astype(float)
            * 2
            - 1
        )
        statistics[start:stop] = signs @ reductions
    one = (1 + int(np.count_nonzero(statistics >= observed))) / (n_permutations + 1)
    two = (1 + int(np.count_nonzero(np.abs(statistics) >= abs(observed)))) / (
        n_permutations + 1
    )
    return observed, statistics, float(one), float(two)


def run_inference(
    y,
    baseline,
    final,
    clusters,
    seed,
    n_bootstrap,
    n_signflip,
    seed_offset=0,
    bootstrap_batch_size=250,
    signflip_batch_size=500,
):
    sums, counts = aggregate_pairs(y, baseline, final, clusters)
    b, f, delta = cod_from_sums(sums.sum(axis=0))
    rng_boot = np.random.default_rng(seed + seed_offset)
    rng_perm = np.random.default_rng(seed + 1000 + seed_offset)
    boot = paired_cluster_bootstrap(sums, rng_boot, n_bootstrap, bootstrap_batch_size)
    observed, null, p_one, p_two = paired_cluster_signflip(
        sums, rng_perm, n_signflip, signflip_batch_size
    )
    lo, hi = np.quantile(boot, [0.025, 0.975], method="linear")
    relative = float(100 * delta / b) if b != 0 else None
    result = dict(
        **counts,
        baseline_COD=float(b),
        final_COD=float(f),
        delta_COD=float(delta),
        delta_COD_CI2p5=float(lo),
        delta_COD_CI97p5=float(hi),
        relative_improvement_pct=relative,
        delta_CI_as_pct_baseline_low=float(100 * lo / b) if b > 0 else None,
        delta_CI_as_pct_baseline_high=float(100 * hi / b) if b > 0 else None,
        bootstrap_p_delta_le0=float(
            (1 + np.count_nonzero(boot <= 0)) / (n_bootstrap + 1)
        ),
        permutation_p_one_sided=p_one,
        permutation_p_two_sided=p_two,
        observed_SSE_reduction=observed,
    )
    return result, boot, null


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions-csv", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cluster-column")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    frame = pd.read_csv(args.predictions_csv)
    fields = [config["y_column"], config["baseline_column"], config["final_column"]]
    arrays = [pd.to_numeric(frame[c], errors="raise").to_numpy(float) for c in fields]
    if args.cluster_column:
        clusters = frame[args.cluster_column].to_numpy()
    else:
        if config.get("cluster_unit") != "participant":
            p.error("A cluster column is required for this analysis.")
        ids = frame[config["id_column"]]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(
                "Participant-level analysis requires a unique, nonmissing ID per row."
            )
        clusters = ids.to_numpy()
    result, boot, null = run_inference(
        *arrays,
        clusters,
        seed=config["seed"],
        n_bootstrap=config["n_cluster_bootstrap"],
        n_signflip=config["n_cluster_signflip_permutations"],
        seed_offset=config["seed_offset"],
        bootstrap_batch_size=config["bootstrap_batch_size"],
        signflip_batch_size=config["signflip_batch_size"],
    )
    for key in ["n_rows", "n_clusters"]:
        if key in config and result[key] != config[key]:
            raise RuntimeError(f"{key} differs from the recorded analysis.")
    result["outcome"] = config["outcome"]
    if config.get("fold_column"):
        fold_gains = []
        for fold in sorted(frame[config["fold_column"]].unique()):
            mask = frame[config["fold_column"]].eq(fold).to_numpy()
            y, b, f = [a[mask] for a in arrays]
            sst = float(np.sum((y - y.mean()) ** 2))
            if sst <= 0:
                raise ValueError("Fold outcome variance is zero.")
            fold_gains.append(
                float((np.sum((y - b) ** 2) - np.sum((y - f) ** 2)) / sst)
            )
        result["folds_improved"] = sum(x > 0 for x in fold_gains)
        result["folds_total"] = len(fold_gains)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out / "prediction_gain_summary.csv", index=False)
    np.savez_compressed(
        out / "prediction_gain_replicates.npz",
        bootstrap_delta_COD=boot,
        signflip_SSE_reduction=null,
    )
    metadata = dict(
        config=config,
        predictions_csv=args.predictions_csv,
        cluster_column=args.cluster_column,
        cluster_order="first appearance in input rows",
        rng="Independent NumPy default_rng streams for bootstrap and signflip",
        bootstrap_seed=config["seed"] + config["seed_offset"],
        signflip_seed=config["seed"] + 1000 + config["seed_offset"],
        signflip_draw_dtype="int8",
        quantile_method="linear",
        bootstrap_stratification="none",
        bootstrap_batch_size=config["bootstrap_batch_size"],
        signflip_batch_size=config["signflip_batch_size"],
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
    )
    (out / "prediction_gain_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
