"""Apply frozen UKB phenotype-PC KRR bundles to BCP RSFC.

Checks scan order, child fold membership, target order and edge names.
Averages predictions from seed 0 through seed 4.
"""

import argparse
import datetime as dt
import json
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, np.generic):
        return sanitize_for_json(obj.item())
    if isinstance(obj, float) and (not np.isfinite(obj)):
        return None
    return obj


def unitize_rows(X, dtype=np.float32):
    X = np.asarray(X, dtype=np.float64)
    X = X - np.mean(X, axis=1, keepdims=True)
    norm = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    norm[~np.isfinite(norm)] = 1.0
    norm[norm == 0.0] = 1.0
    Z = X / norm
    return Z.astype(dtype, copy=False)


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def load_fold_manifest(path, scan_col, group_col):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    required = ["_bcp_row", scan_col, group_col, "fold"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Fold manifest missing required columns: {missing}")
    out = df.copy()
    out["_bcp_row"] = pd.to_numeric(out["_bcp_row"], errors="raise").astype(np.int64)
    out[scan_col] = out[scan_col].astype(str)
    out[group_col] = out[group_col].astype(str)
    out["fold"] = pd.to_numeric(out["fold"], errors="raise").astype(np.int64)
    if out.duplicated("_bcp_row").any():
        raise RuntimeError("Fold manifest contains duplicated _bcp_row.")
    if out[[scan_col, group_col]].duplicated().any():
        raise RuntimeError("Fold manifest contains duplicated scan/group keys.")
    gfold = out.groupby(group_col)["fold"].nunique()
    if int((gfold > 1).sum()) != 0:
        raise RuntimeError("Subject leakage detected in fold manifest.")
    return out


def load_bcp_locked_rows(
    cohort_csv, fold_manifest, edge_names, scan_col, group_col, target_col
):
    p = Path(cohort_csv)
    if not p.exists():
        raise FileNotFoundError(p)
    df_all = (
        pd.read_csv(p).reset_index(drop=False).rename(columns={"index": "_bcp_row_all"})
    )
    idx = fold_manifest["_bcp_row"].to_numpy(dtype=np.int64)
    if np.any(idx < 0) or np.any(idx >= len(df_all)):
        raise RuntimeError("Fold manifest contains _bcp_row outside BCP cohort range.")
    df = df_all.iloc[idx].copy().reset_index(drop=True)
    df["_bcp_row"] = idx
    if not np.array_equal(
        df[scan_col].astype(str).to_numpy(),
        fold_manifest[scan_col].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "BCP rows selected by _bcp_row do not match fold manifest scan_id order."
        )
    if not np.array_equal(
        df[group_col].astype(str).to_numpy(),
        fold_manifest[group_col].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "BCP rows selected by _bcp_row do not match fold manifest imaging_numeric_id order."
        )
    missing_edges = [c for c in edge_names if c not in df.columns]
    if missing_edges:
        raise KeyError(
            f"BCP cohort missing edge columns from bundle edge_names, first 10: {missing_edges[:10]}"
        )
    E = (
        df[list(edge_names)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    if not np.isfinite(E).all():
        bad = int((~np.isfinite(E)).sum())
        raise RuntimeError(f"BCP edge matrix contains NaN/Inf: {bad}")
    meta_cols = ["_bcp_row", scan_col, group_col]
    for c in [
        "fold",
        target_col,
        "scan_age_days",
        "scan_age_months",
        "sex_final",
        "FD",
        "site_primary",
        "res_id",
        "res_age_diff_days",
        "y_true_resid",
        "y_pred",
    ]:
        if c in fold_manifest.columns and c not in meta_cols:
            meta_cols.append(c)
        elif c in df.columns and c not in meta_cols:
            meta_cols.append(c)
    meta = df[[c for c in meta_cols if c in df.columns]].copy()
    meta["fold"] = fold_manifest["fold"].to_numpy(dtype=np.int64)
    for c in ["y_true_resid", "y_pred", "_pred_row"]:
        if c in fold_manifest.columns:
            meta[c] = fold_manifest[c].to_numpy()
    audit = {
        "cohort_csv": str(p),
        "n_rows_full": int(len(df_all)),
        "n_locked": int(len(df)),
        "n_unique_scans": int(df[scan_col].astype(str).nunique()),
        "n_unique_subjects": int(df[group_col].astype(str).nunique()),
        "n_edges": int(E.shape[1]),
        "fold_counts": {
            str(k): int(v)
            for k, v in pd.Series(meta["fold"]).value_counts().sort_index().items()
        },
        "meta_columns": list(meta.columns),
    }
    return (meta, E, audit)


def load_bundle(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    b = joblib.load(p)
    if not isinstance(b, dict):
        raise RuntimeError(f"Bundle is not dict: {p}")
    required = [
        "edge_names",
        "target_names",
        "Z_train_full",
        "target_models",
        "metadata",
    ]
    missing = [k for k in required if k not in b]
    if missing:
        raise KeyError(f"Bundle missing required keys: {missing}")
    if b.get("bundle_type") != "ukb_stage1_krr_pc_transfer_bundle":
        raise RuntimeError(f"Unexpected bundle_type in {p}: {b.get('bundle_type')}")
    md = b["metadata"]
    if md.get("feature_transform") != "unitize_rows":
        raise RuntimeError(
            f"Unexpected feature_transform in {p}: {md.get('feature_transform')}"
        )
    if md.get("kernel_type") != "corr_precomputed":
        raise RuntimeError(f"Unexpected kernel_type in {p}: {md.get('kernel_type')}")
    Z = np.asarray(b["Z_train_full"], dtype=np.float32)
    if Z.ndim != 2:
        raise RuntimeError(f"Z_train_full is not 2D in {p}: {Z.shape}")
    if not np.isfinite(Z).all():
        raise RuntimeError(f"Z_train_full contains NaN/Inf in {p}")
    return b


def validate_bundles(bundle_paths):
    bundles = []
    edge_ref = None
    target_ref = None
    rows = []
    for p in bundle_paths:
        b = load_bundle(p)
        edge_names = [str(x) for x in b["edge_names"]]
        target_names = [str(x) for x in b["target_names"]]
        if edge_ref is None:
            edge_ref = edge_names
            target_ref = target_names
        else:
            if edge_names != edge_ref:
                raise RuntimeError(f"edge_names mismatch: {p}")
            if target_names != target_ref:
                raise RuntimeError(f"target_names mismatch: {p}")
        tm = b["target_models"]
        if set(tm.keys()) != set(target_names):
            raise RuntimeError(f"target_models keys do not match target_names in {p}")
        Z = np.asarray(b["Z_train_full"], dtype=np.float32)
        row_norm = np.linalg.norm(Z, axis=1)
        row_mean = Z.mean(axis=1)
        rows.append(
            {
                "path": str(p),
                "run_tag": b.get("run_tag"),
                "seed": b.get("metadata", {}).get("seed"),
                "n_train_full_rows": int(Z.shape[0]),
                "n_edges": int(Z.shape[1]),
                "n_targets": int(len(target_names)),
                "feature_transform": b.get("metadata", {}).get("feature_transform"),
                "kernel_type": b.get("metadata", {}).get("kernel_type"),
                "fit_index_space": b.get("metadata", {}).get("fit_index_space"),
                "refit_on_train_valid": b.get("metadata", {}).get(
                    "refit_on_train_valid"
                ),
                "Z_row_mean_min": float(row_mean.min()),
                "Z_row_mean_median": float(np.median(row_mean)),
                "Z_row_mean_max": float(row_mean.max()),
                "Z_row_norm_min": float(row_norm.min()),
                "Z_row_norm_median": float(np.median(row_norm)),
                "Z_row_norm_max": float(row_norm.max()),
            }
        )
        bundles.append(b)
    audit = {"n_bundles": int(len(bundles)), "bundle_rows": rows}
    return (bundles, edge_ref, target_ref, audit)


def predict_one_target_chunked(Z_bcp, Z_train_full, model, chunk_size):
    status = str(model.get("status", ""))
    if status != "ok":
        raise RuntimeError(
            f"Target model status is not ok: {status}; reason={model.get('failure_reason')}"
        )
    fit_idx = np.asarray(model["fit_train_idx"], dtype=np.int64)
    dual = np.asarray(model["dual_coef"], dtype=np.float64).ravel()
    y_mu = float(model["y_mu"])
    y_sigma = float(model["y_sigma"])
    if fit_idx.ndim != 1:
        raise RuntimeError("fit_train_idx is not 1D.")
    if dual.ndim != 1:
        raise RuntimeError("dual_coef is not 1D.")
    if fit_idx.size != dual.size:
        raise RuntimeError(
            f"dual_coef length {dual.size} != fit_train_idx length {fit_idx.size}."
        )
    if fit_idx.size == 0:
        raise RuntimeError("fit_train_idx is empty.")
    if np.any(fit_idx < 0) or np.any(fit_idx >= Z_train_full.shape[0]):
        raise RuntimeError("fit_train_idx out of bounds for Z_train_full.")
    if not np.isfinite(dual).all():
        raise RuntimeError("dual_coef contains NaN/Inf.")
    if not np.isfinite(y_mu) or not np.isfinite(y_sigma):
        raise RuntimeError("y_mu/y_sigma contains NaN/Inf.")
    if abs(y_sigma) < 1e-12:
        y_sigma = 1.0
    Z_fit = np.asarray(Z_train_full[fit_idx], dtype=np.float32)
    n = Z_bcp.shape[0]
    pred_z = np.empty(n, dtype=np.float64)
    cs = int(max(1, chunk_size))
    for start in range(0, n, cs):
        end = min(n, start + cs)
        K = np.asarray(Z_bcp[start:end] @ Z_fit.T, dtype=np.float64)
        pred_z[start:end] = K @ dual
    pred_raw = pred_z * y_sigma + y_mu
    return (pred_z, pred_raw)


def predict_one_bundle(bundle, Z_bcp, meta, target_names, chunk_size):
    Z_train = np.asarray(bundle["Z_train_full"], dtype=np.float32)
    target_models = bundle["target_models"]
    seed = bundle.get("metadata", {}).get("seed", None)
    run_tag = str(bundle.get("run_tag", f"stage1_seed{seed}"))
    out = meta.copy()
    rows = []
    for j, target in enumerate(target_names):
        model = target_models[target]
        pred_z, pred_raw = predict_one_target_chunked(
            Z_bcp=Z_bcp, Z_train_full=Z_train, model=model, chunk_size=chunk_size
        )
        if not np.isfinite(pred_z).all() or not np.isfinite(pred_raw).all():
            raise RuntimeError(f"Prediction contains NaN/Inf for target {target}")
        out[str(target)] = pred_raw.astype(np.float32)
        out[f"{target}__zpred"] = pred_z.astype(np.float32)
        rows.append(
            {
                "run_tag": run_tag,
                "seed": int(seed) if seed is not None else np.nan,
                "target": str(target),
                "alpha": float(model.get("alpha", np.nan)),
                "fit_source": str(model.get("fit_source", "")),
                "n_fit_obs": int(
                    model.get("n_fit_obs", len(model.get("fit_train_idx", [])))
                ),
                "y_mu": float(model.get("y_mu", np.nan)),
                "y_sigma": float(model.get("y_sigma", np.nan)),
                "pred_z_mean": float(np.mean(pred_z)),
                "pred_z_sd": float(np.std(pred_z)),
                "pred_z_min": float(np.min(pred_z)),
                "pred_z_max": float(np.max(pred_z)),
                "pred_raw_mean": float(np.mean(pred_raw)),
                "pred_raw_sd": float(np.std(pred_raw)),
            }
        )
    pred_cols = []
    for t in target_names:
        pred_cols.extend([str(t), f"{t}__zpred"])
    first_cols = [
        "_bcp_row",
        "scan_id",
        "imaging_numeric_id",
        "fold",
        "composite_percentile",
        "scan_age_days",
        "scan_age_months",
        "sex_final",
        "FD",
        "site_primary",
        "res_id",
        "res_age_diff_days",
        "y_true_resid",
        "y_pred",
        "_pred_row",
    ]
    first_cols = [c for c in first_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in first_cols and c not in pred_cols]
    out = out[first_cols + other_cols + pred_cols].copy()
    return (out, pd.DataFrame(rows))


def make_mean_predictions(seed_dfs, target_names):
    if not seed_dfs:
        raise RuntimeError("No seed prediction dataframes provided.")
    base = seed_dfs[0].copy()
    id_cols = [
        "_bcp_row",
        "scan_id",
        "imaging_numeric_id",
        "fold",
        "composite_percentile",
        "scan_age_days",
        "scan_age_months",
        "sex_final",
        "FD",
        "site_primary",
        "res_id",
        "res_age_diff_days",
        "y_true_resid",
        "y_pred",
        "_pred_row",
    ]
    id_cols = [c for c in id_cols if c in base.columns]
    mean_df = base[id_cols].copy()
    rows = []
    for target in target_names:
        raw_stack = np.vstack(
            [df[str(target)].to_numpy(dtype=np.float64) for df in seed_dfs]
        )
        z_stack = np.vstack(
            [df[f"{target}__zpred"].to_numpy(dtype=np.float64) for df in seed_dfs]
        )
        mean_raw = np.mean(raw_stack, axis=0)
        mean_z = np.mean(z_stack, axis=0)
        sd_raw = (
            np.std(raw_stack, axis=0, ddof=1)
            if len(seed_dfs) > 1
            else np.zeros_like(mean_raw)
        )
        sd_z = (
            np.std(z_stack, axis=0, ddof=1)
            if len(seed_dfs) > 1
            else np.zeros_like(mean_z)
        )
        mean_df[str(target)] = mean_raw.astype(np.float32)
        mean_df[f"{target}__zpred"] = mean_z.astype(np.float32)
        mean_df[f"{target}__raw_seed_sd"] = sd_raw.astype(np.float32)
        mean_df[f"{target}__zpred_seed_sd"] = sd_z.astype(np.float32)
        rows.append(
            {
                "target": str(target),
                "mean5_z_mean": float(np.mean(mean_z)),
                "mean5_z_sd": float(np.std(mean_z)),
                "mean5_z_min": float(np.min(mean_z)),
                "mean5_z_max": float(np.max(mean_z)),
                "mean5_raw_mean": float(np.mean(mean_raw)),
                "mean5_raw_sd": float(np.std(mean_raw)),
                "mean_seed_sd_z": float(np.mean(sd_z)),
                "mean_seed_sd_raw": float(np.mean(sd_raw)),
            }
        )
    pred_cols = []
    for t in target_names:
        pred_cols.extend(
            [str(t), f"{t}__zpred", f"{t}__raw_seed_sd", f"{t}__zpred_seed_sd"]
        )
    mean_df = mean_df[id_cols + pred_cols].copy()
    return (mean_df, pd.DataFrame(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bcp-cohort-csv", required=True)
    ap.add_argument("--fold-manifest-csv", required=True)
    ap.add_argument("--krr-bundle-paths", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scan-col", default="scan_id")
    ap.add_argument("--group-col", default="imaging_numeric_id")
    ap.add_argument("--target-col", default="composite_percentile")
    ap.add_argument("--chunk-size", type=int, default=256)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    per_seed_dir = out_dir / "per_seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    bundle_paths = [Path(p) for p in args.krr_bundle_paths]
    fold_manifest = load_fold_manifest(
        args.fold_manifest_csv, args.scan_col, args.group_col
    )
    bundles, edge_names, target_names, bundle_audit = validate_bundles(bundle_paths)
    if [b["metadata"].get("seed") for b in bundles] != [0, 1, 2, 3, 4]:
        raise ValueError("Supply exactly five bundles in seed 0 to 4 order.")
    if len(edge_names) != 1485 or edge_names[0] != "e1" or edge_names[-1] != "e1485":
        raise RuntimeError(
            f"Unexpected edge_names contract: n={len(edge_names)}, first={edge_names[:3]}, last={edge_names[-3:]}"
        )
    meta, E, bcp_audit = load_bcp_locked_rows(
        cohort_csv=args.bcp_cohort_csv,
        fold_manifest=fold_manifest,
        edge_names=edge_names,
        scan_col=args.scan_col,
        group_col=args.group_col,
        target_col=args.target_col,
    )
    Z_bcp = unitize_rows(E, dtype=np.float32)
    if not np.isfinite(Z_bcp).all():
        raise RuntimeError("Z_bcp contains NaN/Inf after unitize_rows.")
    del E
    row_mean = Z_bcp.mean(axis=1)
    row_norm = np.linalg.norm(Z_bcp, axis=1)
    bcp_audit["Z_bcp_row_mean_min_median_max"] = [
        float(row_mean.min()),
        float(np.median(row_mean)),
        float(row_mean.max()),
    ]
    bcp_audit["Z_bcp_row_norm_min_median_max"] = [
        float(row_norm.min()),
        float(np.median(row_norm)),
        float(row_norm.max()),
    ]
    seed_dfs = []
    target_summary_all = []
    for i, (bundle, path) in enumerate(zip(bundles, bundle_paths)):
        seed = bundle.get("metadata", {}).get("seed", i)
        df_seed, target_summary = predict_one_bundle(
            bundle=bundle,
            Z_bcp=Z_bcp,
            meta=meta,
            target_names=target_names,
            chunk_size=args.chunk_size,
        )
        target_summary["bundle_path"] = str(path)
        target_summary_all.append(target_summary)
        out_seed = (
            per_seed_dir / f"bcp_krr_transferred_pc_stage1_seed{int(seed)}.parquet"
        )
        df_seed.to_parquet(out_seed, index=False)
        seed_dfs.append(df_seed)
    mean_df, mean_summary = make_mean_predictions(seed_dfs, target_names)
    mean_path = out_dir / "bcp_krr_transferred_pc_mean5.parquet"
    compat_path = out_dir / "bcp_predicted_source_mean5_krr.parquet"
    mean_df.to_parquet(mean_path, index=False)
    mean_df.to_parquet(compat_path, index=False)
    seed_summary_path = out_dir / "bcp_krr_transferred_pc_per_seed_target_summary.csv"
    mean_summary_path = out_dir / "bcp_krr_transferred_pc_mean5_target_summary.csv"
    pd.concat(target_summary_all, axis=0, ignore_index=True).to_csv(
        seed_summary_path, index=False
    )
    mean_summary.to_csv(mean_summary_path, index=False)
    diagnostic_rows = []
    if args.target_col in mean_df.columns:
        y = pd.to_numeric(mean_df[args.target_col], errors="coerce").to_numpy(
            dtype=np.float64
        )
        y_resid = (
            pd.to_numeric(mean_df["y_true_resid"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            if "y_true_resid" in mean_df.columns
            else None
        )
        for t in target_names:
            z = mean_df[f"{t}__zpred"].to_numpy(dtype=np.float64)
            diagnostic_rows.append(
                {
                    "target": t,
                    "corr_zpred_raw_bcp_target": safe_corr(z, y),
                    "corr_zpred_baseline_resid_target": (
                        safe_corr(z, y_resid) if y_resid is not None else np.nan
                    ),
                    "zpred_mean": float(np.mean(z)),
                    "zpred_sd": float(np.std(z)),
                }
            )
    diagnostic_path = out_dir / "bcp_krr_transferred_pc_diagnostic_correlations.csv"
    pd.DataFrame(diagnostic_rows).to_csv(diagnostic_path, index=False)
    run_manifest = {
        "script": "infer_pc_krr.py",
        "created_at": dt.datetime.now().isoformat(),
        "inputs": {
            "bcp_cohort_csv": args.bcp_cohort_csv,
            "fold_manifest_csv": args.fold_manifest_csv,
            "krr_bundle_paths": [str(p) for p in bundle_paths],
        },
        "config": {
            "scan_col": args.scan_col,
            "group_col": args.group_col,
            "target_col": args.target_col,
            "chunk_size": int(args.chunk_size),
        },
        "bcp_audit": bcp_audit,
        "bundle_audit": bundle_audit,
        "target_names": target_names,
        "edge_names_first_last": [edge_names[0], edge_names[-1]],
        "inference_contract": {
            "feature_transform_for_bcp_edges": "unitize_rows",
            "kernel_formula": "K_test_fit = Z_bcp @ Z_train_full[fit_train_idx].T",
            "z_prediction_formula": "y_pred_z = K_test_fit @ dual_coef",
            "raw_prediction_formula": "y_pred_raw = y_pred_z * y_sigma + y_mu",
            "mean5_rule": "average predictions across seed0..seed4",
        },
        "outputs": {
            "out_dir": str(out_dir),
            "per_seed_dir": str(per_seed_dir),
            "mean5_parquet": str(mean_path),
            "compat_mean5_parquet": str(compat_path),
            "per_seed_target_summary_csv": str(seed_summary_path),
            "mean5_target_summary_csv": str(mean_summary_path),
            "diagnostic_correlations_csv": str(diagnostic_path),
        },
    }
    manifest_path = out_dir / "bcp_krr_transferred_pc_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(sanitize_for_json(run_manifest), f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
