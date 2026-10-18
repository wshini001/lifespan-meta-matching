"""Apply frozen UKB raw-phenotype KRR models to BCP RSFC.

Uses float32 row-centered, L2-normalized edges for kernel prediction.
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def unitize_rows(X: np.ndarray, dtype=np.float32) -> np.ndarray:
    X = np.asarray(X, dtype=dtype)
    X = X - X.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return X / denom


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x = x[ok]
    y = y[ok]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def get_edge_cols(df: pd.DataFrame) -> list[str]:
    edge_cols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    edge_cols = sorted(edge_cols, key=lambda x: int(x[1:]))
    expected = [f"e{i}" for i in range(1, len(edge_cols) + 1)]
    if edge_cols != expected:
        raise RuntimeError("BCP edge columns are not contiguous e1...eN.")
    return edge_cols


def load_bcp_locked(cohort_csv: str, baseline_pred_csv: str):
    cohort = pd.read_csv(cohort_csv)
    cohort["_bcp_row"] = np.arange(len(cohort), dtype=np.int64)
    edge_cols = get_edge_cols(cohort)
    if len(edge_cols) != 1485:
        raise RuntimeError(f"Expected 1485 BCP edge columns, got {len(edge_cols)}.")
    required = [
        "scan_id",
        "imaging_numeric_id",
        "composite_percentile",
        "scan_age_days",
        "scan_age_months",
        "sex_final",
        "FD",
        "site_primary",
    ]
    missing = [c for c in required if c not in cohort.columns]
    if missing:
        raise RuntimeError(f"BCP cohort missing required columns: {missing}")
    pred = pd.read_csv(baseline_pred_csv)
    required_pred = ["scan_id", "imaging_numeric_id", "fold", "y_true_resid", "y_pred"]
    missing_pred = [c for c in required_pred if c not in pred.columns]
    if missing_pred:
        raise RuntimeError(
            f"Baseline prediction CSV missing required columns: {missing_pred}"
        )
    if cohort.duplicated(["scan_id", "imaging_numeric_id"]).any():
        raise RuntimeError("BCP cohort has duplicated scan_id/imaging_numeric_id keys.")
    if pred.duplicated(["scan_id", "imaging_numeric_id"]).any():
        raise RuntimeError(
            "Baseline prediction CSV has duplicated scan_id/imaging_numeric_id keys."
        )
    pred = pred.copy()
    pred["_pred_row"] = np.arange(len(pred), dtype=np.int64)
    merged = cohort.merge(
        pred[
            [
                "scan_id",
                "imaging_numeric_id",
                "fold",
                "y_true_resid",
                "y_pred",
                "_pred_row",
            ]
        ],
        on=["scan_id", "imaging_numeric_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["fold"].isna().any():
        bad = merged.loc[
            merged["fold"].isna(), ["scan_id", "imaging_numeric_id"]
        ].head()
        raise RuntimeError(f"Missing baseline fold after merge. Preview:\n{bad}")
    if not np.isfinite(merged[edge_cols].to_numpy(float)).all():
        raise RuntimeError("BCP edge matrix contains NaN/Inf.")
    meta_cols = [
        "_bcp_row",
        "scan_id",
        "imaging_numeric_id",
        "composite_percentile",
        "scan_age_days",
        "scan_age_months",
        "sex_final",
        "FD",
        "site_primary",
    ]
    for c in ["res_id", "res_age_diff_days"]:
        if c in merged.columns:
            meta_cols.append(c)
    meta_cols += ["fold", "y_true_resid", "y_pred", "_pred_row"]
    meta = merged[meta_cols].copy()
    meta["fold"] = pd.to_numeric(meta["fold"], errors="raise").astype(int)
    E = merged[edge_cols].to_numpy(dtype=np.float32)
    Z = unitize_rows(E, dtype=np.float32)
    if not np.isfinite(Z).all():
        raise RuntimeError("BCP unitized edge matrix contains NaN/Inf.")
    audit = {
        "n_rows": int(len(meta)),
        "n_unique_scans": int(meta["scan_id"].nunique()),
        "n_unique_subjects": int(meta["imaging_numeric_id"].nunique()),
        "n_edges": int(len(edge_cols)),
        "fold_counts": {
            str(k): int(v) for k, v in meta["fold"].value_counts().sort_index().items()
        },
        "Z_bcp_row_mean_min_median_max": [
            float(Z.mean(axis=1).min()),
            float(np.median(Z.mean(axis=1))),
            float(Z.mean(axis=1).max()),
        ],
        "Z_bcp_row_norm_min_median_max": [
            float(np.linalg.norm(Z, axis=1).min()),
            float(np.median(np.linalg.norm(Z, axis=1))),
            float(np.linalg.norm(Z, axis=1).max()),
        ],
    }
    return (meta, Z, edge_cols, audit)


def predict_one_bundle(
    bundle_path: str, Z_bcp: np.ndarray, expected_edge_cols: list[str], chunk_size: int
):
    b = joblib.load(bundle_path)
    if b.get("bundle_type") != "ukb_stage1_krr_transfer_bundle":
        raise RuntimeError(
            f"Unexpected bundle_type in {bundle_path}: {b.get('bundle_type')}"
        )
    edge_names = list(b["edge_names"])
    if edge_names != expected_edge_cols:
        raise RuntimeError(f"Edge names mismatch in bundle: {bundle_path}")
    y_cols = list(b["y_cols"])
    if len(y_cols) != 1:
        raise RuntimeError(f"Expected single-target bundle, got y_cols={y_cols}")
    target = y_cols[0]
    target_models = b["target_models"]
    if target not in target_models:
        raise RuntimeError(
            f"Target {target} missing in target_models for {bundle_path}"
        )
    model = target_models[target]
    for key in ["fit_train_idx", "dual_coef", "y_mu", "y_sigma"]:
        if key not in model:
            raise RuntimeError(f"Model for {target} missing key {key}: {bundle_path}")
    Z_train = np.asarray(b["Z_train_full"], dtype=np.float32)
    fit_idx = np.asarray(model["fit_train_idx"], dtype=np.int64)
    dual = np.asarray(model["dual_coef"], dtype=np.float64).ravel()
    y_mu = float(model["y_mu"])
    y_sigma = float(model["y_sigma"])
    if fit_idx.ndim != 1 or dual.ndim != 1:
        raise RuntimeError(f"Bad fit_idx/dual shape for {target}")
    if fit_idx.size != dual.size:
        raise RuntimeError(f"dual length != fit_idx length for {target}")
    if fit_idx.size == 0:
        raise RuntimeError(f"Empty fit_idx for {target}")
    if fit_idx.min() < 0 or fit_idx.max() >= Z_train.shape[0]:
        raise RuntimeError(f"fit_idx out of bounds for {target}")
    if not np.isfinite(dual).all():
        raise RuntimeError(f"dual_coef has NaN/Inf for {target}")
    if not np.isfinite(Z_train).all():
        raise RuntimeError(f"Z_train_full has NaN/Inf for {target}")
    if not np.isfinite(y_mu) or not np.isfinite(y_sigma) or y_sigma <= 0:
        raise RuntimeError(f"Bad y_mu/y_sigma for {target}: {y_mu}, {y_sigma}")
    Z_fit = Z_train[fit_idx]
    pred_z = np.zeros(Z_bcp.shape[0], dtype=np.float64)
    for start in range(0, Z_bcp.shape[0], chunk_size):
        end = min(start + chunk_size, Z_bcp.shape[0])
        K = np.asarray(Z_bcp[start:end], dtype=np.float32) @ Z_fit.T
        pred_z[start:end] = K @ dual
    pred_raw = pred_z * y_sigma + y_mu
    summary = {
        "target": target,
        "bundle_path": str(bundle_path),
        "run_id": b.get("run_id"),
        "best_alpha": float(model.get("best_alpha", np.nan)),
        "n_fit_obs": int(fit_idx.size),
        "y_mu": y_mu,
        "y_sigma": y_sigma,
        "pred_z_mean": float(np.mean(pred_z)),
        "pred_z_sd": float(np.std(pred_z)),
        "pred_z_min": float(np.min(pred_z)),
        "pred_z_max": float(np.max(pred_z)),
        "pred_raw_mean": float(np.mean(pred_raw)),
        "pred_raw_sd": float(np.std(pred_raw)),
        "pred_raw_min": float(np.min(pred_raw)),
        "pred_raw_max": float(np.max(pred_raw)),
    }
    return (target, pred_raw.astype(np.float32), pred_z.astype(np.float32), summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bcp-cohort-csv", required=True)
    ap.add_argument("--baseline-pred-csv", required=True)
    ap.add_argument("--topM-manifest-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-tag", default="bcp_stage1_infer_krrraw_topM_v1")
    ap.add_argument("--chunk-size", type=int, default=128)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, Z_bcp, edge_cols, bcp_audit = load_bcp_locked(
        args.bcp_cohort_csv, args.baseline_pred_csv
    )
    top = pd.read_csv(args.topM_manifest_csv)
    required_cols = ["target", "rank", "bundle_path"]
    missing = [c for c in required_cols if c not in top.columns]
    if missing:
        raise RuntimeError(f"Top30 manifest missing columns: {missing}")
    top = top.sort_values("rank").reset_index(drop=True)
    if len(top) < 1:
        raise RuntimeError(f"Expected at least 1 KRR raw target, got {len(top)}.")
    out = meta.copy()
    summaries = []
    for i, row in top.iterrows():
        target_expected = row["target"]
        bundle_path = Path(row["bundle_path"])
        if not bundle_path.exists():
            raise FileNotFoundError(bundle_path)
        target, pred_raw, pred_z, summary = predict_one_bundle(
            bundle_path=str(bundle_path),
            Z_bcp=Z_bcp,
            expected_edge_cols=edge_cols,
            chunk_size=args.chunk_size,
        )
        if target != target_expected:
            raise RuntimeError(
                f"Target mismatch: manifest={target_expected}, bundle={target}"
            )
        out[target] = pred_raw
        out[f"{target}__zpred"] = pred_z
        summary["rank"] = int(row["rank"])
        summaries.append(summary)
    target_names = [r["target"] for r in summaries]
    zcols = [f"{t}__zpred" for t in target_names]
    if not np.isfinite(out[zcols].to_numpy(float)).all():
        raise RuntimeError("Output zpred columns contain NaN/Inf.")
    target_summary = pd.DataFrame(summaries)
    diag_rows = []
    for t in target_names:
        z = out[f"{t}__zpred"].to_numpy(float)
        diag_rows.append(
            {
                "target": t,
                "corr_zpred_raw_bcp_target": safe_corr(z, out["composite_percentile"]),
                "corr_zpred_baseline_resid_target": safe_corr(z, out["y_true_resid"]),
                "zpred_mean": float(np.mean(z)),
                "zpred_sd": float(np.std(z)),
            }
        )
    diag = pd.DataFrame(diag_rows).sort_values(
        "corr_zpred_baseline_resid_target", ascending=False
    )
    parquet_path = out_dir / f"bcp_krrraw_transferred_top{len(target_names)}.parquet"
    compat_path = (
        out_dir / f"bcp_predicted_source_krrraw_top{len(target_names)}.parquet"
    )
    summary_path = (
        out_dir / f"bcp_krrraw_transferred_top{len(target_names)}_target_summary.csv"
    )
    diag_path = (
        out_dir
        / f"bcp_krrraw_transferred_top{len(target_names)}_diagnostic_correlations.csv"
    )
    manifest_path = (
        out_dir / f"bcp_krrraw_transferred_top{len(target_names)}_manifest.json"
    )
    out.to_parquet(parquet_path, index=False)
    out.to_parquet(compat_path, index=False)
    target_summary.to_csv(summary_path, index=False)
    diag.to_csv(diag_path, index=False)
    manifest = {
        "script": "infer_raw_krr.py",
        "created_at": datetime.now().isoformat(),
        "run_tag": args.run_tag,
        "inputs": {
            "bcp_cohort_csv": str(args.bcp_cohort_csv),
            "baseline_pred_csv": str(args.baseline_pred_csv),
            "topM_manifest_csv": str(args.topM_manifest_csv),
        },
        "outputs": {
            "parquet": str(parquet_path),
            "compat_parquet": str(compat_path),
            "target_summary_csv": str(summary_path),
            "diagnostic_correlations_csv": str(diag_path),
        },
        "bcp_audit": bcp_audit,
        "n_targets": int(len(target_names)),
        "target_names": target_names,
        "edge_names_first_last": [edge_cols[0], edge_cols[-1]],
        "inference_contract": {
            "feature_transform": "unitize_rows",
            "kernel_type": "corr_precomputed",
            "formula": "K_test_fit = Z_bcp @ Z_train_full[fit_train_idx].T; y_pred_z = K_test_fit @ dual_coef; y_pred_raw = y_pred_z * y_sigma + y_mu",
        },
    }
    json.dump(manifest, open(manifest_path, "w"), indent=2)


if __name__ == "__main__":
    main()
