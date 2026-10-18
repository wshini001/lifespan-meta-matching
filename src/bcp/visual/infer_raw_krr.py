"""Apply frozen UKB raw-KRR models to BCP RSFC.

Uses float32 row-centered, L2-normalized edges and float64 dual prediction.
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import joblib
import numpy as np
import pandas as pd


def parse_int_list(s: str) -> List[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise RuntimeError(f"Empty M list: {s}")
    return vals


def unitize_rows(X: np.ndarray, dtype=np.float32) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    norm[~np.isfinite(norm)] = 1.0
    norm[norm == 0.0] = 1.0
    return (X / norm).astype(dtype, copy=False)


def get_edge_cols(edge_manifest_tsv: str) -> List[str]:
    em = pd.read_csv(edge_manifest_tsv, sep="\t")
    need = ["edge_index_1based", "edge_name"]
    missing = [c for c in need if c not in em.columns]
    if missing:
        raise RuntimeError(f"edge manifest missing columns: {missing}")
    em = em.sort_values("edge_index_1based")
    edge_cols = em["edge_name"].astype(str).tolist()
    expected = [f"e{i}" for i in range(1, len(edge_cols) + 1)]
    if edge_cols != expected:
        raise RuntimeError("edge manifest is not contiguous e1..eN")
    if len(edge_cols) != 1485:
        raise RuntimeError(f"expected 1485 edges, got {len(edge_cols)}")
    return edge_cols


def load_bcp_locked(
    cohort_csv: str,
    locked_row_manifest_csv: str,
    edge_cols: Sequence[str],
    scan_col: str,
    group_col: str,
    target_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    cohort = pd.read_csv(cohort_csv)
    manifest = pd.read_csv(locked_row_manifest_csv)
    key_cols = [scan_col, group_col]
    for c in key_cols:
        cohort[c] = cohort[c].astype(str)
        manifest[c] = manifest[c].astype(str)
    if manifest[key_cols].duplicated().any():
        raise RuntimeError("locked manifest duplicated keys")
    if cohort[key_cols].duplicated().any():
        raise RuntimeError("cohort duplicated keys")
    required_manifest = [
        "_locked_row",
        "_cohort_all_row",
        "_pred_row",
        scan_col,
        group_col,
        "fold",
    ]
    missing = [c for c in required_manifest if c not in manifest.columns]
    if missing:
        raise RuntimeError(f"locked manifest missing columns: {missing}")
    locked = manifest.merge(
        cohort[key_cols + list(edge_cols)],
        on=key_cols,
        how="left",
        validate="one_to_one",
    )
    locked = locked.sort_values("_locked_row").reset_index(drop=True)
    if len(locked) != 395:
        raise RuntimeError(f"expected N=395, got {len(locked)}")
    if not np.array_equal(locked["_locked_row"].to_numpy(int), np.arange(len(locked))):
        raise RuntimeError("_locked_row is not 0..N-1")
    if locked[group_col].astype(str).nunique() != 199:
        raise RuntimeError("expected 199 unique subjects")
    if locked[list(edge_cols)].isna().any().any():
        raise RuntimeError("edge columns have missing values after merge")
    X_edge = (
        locked[list(edge_cols)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    if not np.isfinite(X_edge).all():
        raise RuntimeError("BCP edge matrix has NaN/Inf")
    meta_cols = [
        "_locked_row",
        "_cohort_all_row",
        "_pred_row",
        scan_col,
        group_col,
        "fold",
    ]
    for c in [
        "scan_age_days",
        "scan_age_months",
        "sex_final",
        "FD",
        "site_primary",
        "mother_education_raw",
        "mother_edu_gt_college",
    ]:
        if c in locked.columns:
            meta_cols.append(c)
    meta = locked[meta_cols].copy()
    meta["fold"] = pd.to_numeric(meta["fold"], errors="raise").astype(int)
    audit = {
        "n": int(len(meta)),
        "n_subjects": int(meta[group_col].astype(str).nunique()),
        "fold_counts": {
            str(k): int(v) for k, v in meta["fold"].value_counts().sort_index().items()
        },
        "edge_shape": list(X_edge.shape),
        "edge_finite": bool(np.isfinite(X_edge).all()),
    }
    return (meta, X_edge, audit)


def load_raw_master_manifest(path: str, max_m: int) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    required = [
        "target",
        "rank",
        "predictive_COD_valid_z",
        "bundle_path",
        "bundle_path_exists",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"raw manifest missing columns: {missing}")
    df = df.sort_values("rank").reset_index(drop=True)
    if len(df) < max_m:
        raise RuntimeError(f"manifest has {len(df)} rows but max M={max_m}")
    sub = df.iloc[:max_m].copy()
    if not sub["bundle_path_exists"].astype(bool).all():
        bad = sub.loc[
            ~sub["bundle_path_exists"].astype(bool), ["rank", "target", "bundle_path"]
        ]
        raise RuntimeError(f"some bundle_path_exists false:\n{bad}")
    missing_files = [
        p for p in sub["bundle_path"].astype(str).tolist() if not Path(p).exists()
    ]
    if missing_files:
        raise RuntimeError(f"some bundle paths do not exist, first={missing_files[:5]}")
    if sub["target"].astype(str).duplicated().any():
        raise RuntimeError("duplicated target in raw manifest")
    return sub


def get_model_from_bundle(bundle: dict, expected_target: str):
    y_cols = bundle.get("y_cols", bundle.get("target_names", []))
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    y_cols = [str(x) for x in y_cols]
    if len(y_cols) != 1:
        raise RuntimeError(f"raw bundle expected one y_col, got {y_cols}")
    if y_cols[0] != str(expected_target):
        raise RuntimeError(
            f"bundle target mismatch: expected={expected_target}, got={y_cols[0]}"
        )
    tm = bundle.get("target_models", {})
    if not isinstance(tm, dict) or expected_target not in tm:
        raise RuntimeError(f"target_models missing expected target {expected_target}")
    return tm[expected_target]


def validate_raw_bundle(
    bundle: dict, edge_cols: Sequence[str], target: str, bundle_path: str
) -> dict:
    if bundle.get("bundle_type") != "ukb_stage1_krr_transfer_bundle":
        raise RuntimeError(
            f"bad bundle_type for {bundle_path}: {bundle.get('bundle_type')}"
        )
    metadata = bundle.get("metadata", {})
    if metadata.get("feature_transform") != "unitize_rows":
        raise RuntimeError(
            f"bad feature_transform in {bundle_path}: {metadata.get('feature_transform')}"
        )
    if metadata.get("kernel_type") != "corr_precomputed":
        raise RuntimeError(
            f"bad kernel_type in {bundle_path}: {metadata.get('kernel_type')}"
        )
    edge_names = [str(x) for x in bundle.get("edge_names", [])]
    if edge_names != list(edge_cols):
        raise RuntimeError(f"edge_names mismatch in {bundle_path}")
    Z = np.asarray(bundle["Z_train_full"])
    if Z.ndim != 2 or Z.shape[1] != 1485:
        raise RuntimeError(f"bad Z_train_full shape in {bundle_path}: {Z.shape}")
    if not np.isfinite(Z).all():
        raise RuntimeError(f"Z_train_full non-finite in {bundle_path}")
    model = get_model_from_bundle(bundle, str(target))
    need = ["fit_train_idx", "dual_coef", "y_mu", "y_sigma"]
    missing = [k for k in need if k not in model]
    if missing:
        raise RuntimeError(f"model missing {missing} in {bundle_path}")
    fit_idx = np.asarray(model["fit_train_idx"], dtype=np.int64)
    dual = np.asarray(model["dual_coef"], dtype=np.float64).ravel()
    if fit_idx.ndim != 1 or dual.ndim != 1 or len(fit_idx) != len(dual):
        raise RuntimeError(f"bad fit_idx/dual shape in {bundle_path}")
    if len(fit_idx) == 0:
        raise RuntimeError(f"empty fit_idx in {bundle_path}")
    if fit_idx.min() < 0 or fit_idx.max() >= Z.shape[0]:
        raise RuntimeError(f"fit_idx out of bounds in {bundle_path}")
    if not np.isfinite(dual).all():
        raise RuntimeError(f"dual_coef non-finite in {bundle_path}")
    return model


def predict_raw_one_bundle(
    Z_bcp: np.ndarray, bundle: dict, model: dict, chunk_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    Z_train = np.asarray(bundle["Z_train_full"], dtype=np.float32)
    fit_idx = np.asarray(model["fit_train_idx"], dtype=np.int64)
    dual = np.asarray(model["dual_coef"], dtype=np.float64).ravel()
    y_mu = float(model["y_mu"])
    y_sigma = float(model["y_sigma"])
    if not np.isfinite(y_mu) or not np.isfinite(y_sigma):
        raise RuntimeError("non-finite y_mu/y_sigma")
    if abs(y_sigma) < 1e-12:
        y_sigma = 1.0
    Z_fit = Z_train[fit_idx]
    pred_z = np.empty(Z_bcp.shape[0], dtype=np.float64)
    cs = max(1, int(chunk_size))
    for s in range(0, Z_bcp.shape[0], cs):
        e = min(Z_bcp.shape[0], s + cs)
        K = np.asarray(Z_bcp[s:e] @ Z_fit.T, dtype=np.float64)
        pred_z[s:e] = K @ dual
    pred_raw = pred_z * y_sigma + y_mu
    if not np.isfinite(pred_z).all() or not np.isfinite(pred_raw).all():
        raise RuntimeError("prediction has NaN/Inf")
    return (pred_z, pred_raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-csv", required=True)
    ap.add_argument("--locked-row-manifest-csv", required=True)
    ap.add_argument("--edge-manifest-tsv", required=True)
    ap.add_argument("--raw-master-manifest-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--m-list", default="5,10,15,20,23,30,40,60,80,100,120")
    ap.add_argument("--target-col", default="fine_motor_percentile")
    ap.add_argument("--scan-col", default="scan_id")
    ap.add_argument("--group-col", default="imaging_numeric_id")
    ap.add_argument("--chunk-size", type=int, default=128)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_list = parse_int_list(args.m_list)
    max_m = max(m_list)
    print("[load] edge manifest", flush=True)
    edge_cols = get_edge_cols(args.edge_manifest_tsv)
    print("[load] BCP locked N395", flush=True)
    meta, X_edge, bcp_audit = load_bcp_locked(
        cohort_csv=args.cohort_csv,
        locked_row_manifest_csv=args.locked_row_manifest_csv,
        edge_cols=edge_cols,
        scan_col=args.scan_col,
        group_col=args.group_col,
        target_col=args.target_col,
    )
    print("[transform] unitize BCP edges", flush=True)
    Z_bcp = unitize_rows(X_edge, dtype=np.float32)
    del X_edge
    bcp_audit["Z_bcp_shape"] = list(Z_bcp.shape)
    bcp_audit["Z_bcp_row_norm_min_median_max"] = [
        float(np.linalg.norm(Z_bcp, axis=1).min()),
        float(np.median(np.linalg.norm(Z_bcp, axis=1))),
        float(np.linalg.norm(Z_bcp, axis=1).max()),
    ]
    print("[load] raw master manifest", flush=True)
    raw_manifest = load_raw_master_manifest(args.raw_master_manifest_csv, max_m=max_m)
    all_pred = {}
    target_summary_rows = []
    print(f"[predict] KRR raw source targets 1..{max_m}", flush=True)
    for _, row in raw_manifest.iterrows():
        rank = int(row["rank"])
        target = str(row["target"])
        bundle_path = str(row["bundle_path"])
        print(f"  rank {rank:03d}/{max_m:03d}: {target}", flush=True)
        bundle = joblib.load(bundle_path)
        model = validate_raw_bundle(
            bundle, edge_cols=edge_cols, target=target, bundle_path=bundle_path
        )
        pred_z, pred_raw = predict_raw_one_bundle(
            Z_bcp=Z_bcp, bundle=bundle, model=model, chunk_size=args.chunk_size
        )
        zcol = f"{target}__zpred"
        rawcol = f"{target}"
        all_pred[zcol] = pred_z.astype(np.float32)
        all_pred[rawcol] = pred_raw.astype(np.float32)
        target_summary_rows.append(
            {
                "rank": rank,
                "target": target,
                "bundle_path": bundle_path,
                "source_predictive_COD_valid_z": float(row["predictive_COD_valid_z"]),
                "source_Pearson_r_valid_z": float(row.get("Pearson_r_valid_z", np.nan)),
                "source_selected_alpha": float(
                    row.get("selected_alpha", row.get("best_alpha", np.nan))
                ),
                "model_best_alpha": float(model.get("best_alpha", np.nan)),
                "n_fit_obs": int(len(model["fit_train_idx"])),
                "y_mu": float(model["y_mu"]),
                "y_sigma": float(model["y_sigma"]),
                "bcp_pred_z_mean": float(np.mean(pred_z)),
                "bcp_pred_z_sd": float(np.std(pred_z)),
                "bcp_pred_z_min": float(np.min(pred_z)),
                "bcp_pred_z_max": float(np.max(pred_z)),
            }
        )
    target_summary = pd.DataFrame(target_summary_rows)
    target_summary_path = (
        out_dir / "bcp_krrraw_source_topM_target_summary_all120_v1.csv"
    )
    target_summary.to_csv(target_summary_path, index=False)
    output_paths = {}
    for M in m_list:
        sub_manifest = raw_manifest.iloc[:M].copy()
        out = meta.copy()
        pred_cols = []
        for target in sub_manifest["target"].astype(str).tolist():
            rawcol = f"{target}"
            zcol = f"{target}__zpred"
            out[rawcol] = all_pred[rawcol]
            out[zcol] = all_pred[zcol]
            pred_cols.extend([rawcol, zcol])
        pq = out_dir / f"bcp_krrraw_source_top{M}_mean1.parquet"
        csv_manifest = out_dir / f"bcp_krrraw_source_top{M}_feature_manifest.csv"
        diag = out_dir / f"bcp_krrraw_source_top{M}_diagnostic_correlations.csv"
        out.to_parquet(pq, index=False)
        sub_manifest.to_csv(csv_manifest, index=False)
        diag_df = target_summary[target_summary["rank"] <= M].copy()
        diag_df.to_csv(diag, index=False)
        output_paths[str(M)] = {
            "parquet": str(pq),
            "feature_manifest": str(csv_manifest),
            "diagnostic_correlations": str(diag),
            "n_features_raw_targets": int(M),
            "n_zpred_columns": int(M),
        }
        print(f"[saved] M={M}: {pq}", flush=True)
    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "script": "infer_raw_krr.py",
        "analysis_stage": "Phase2B_KRRRaw_SourceTopM_Inference_v1",
        "inputs": {
            "cohort_csv": str(args.cohort_csv),
            "locked_row_manifest_csv": str(args.locked_row_manifest_csv),
            "edge_manifest_tsv": str(args.edge_manifest_tsv),
            "raw_master_manifest_csv": str(args.raw_master_manifest_csv),
        },
        "config": {
            "m_list": m_list,
            "max_m": int(max_m),
            "target_col": args.target_col,
            "scan_col": args.scan_col,
            "group_col": args.group_col,
            "chunk_size": int(args.chunk_size),
        },
        "bcp_audit": bcp_audit,
        "inference_contract": {
            "feature_transform": "unitize_rows",
            "kernel_formula": "K_test_fit = Z_bcp @ Z_train_full[fit_train_idx].T",
            "prediction_formula_z": "y_pred_z = K_test_fit @ dual_coef",
            "prediction_formula_raw": "y_pred_raw = y_pred_z * y_sigma + y_mu",
            "recommended_downstream_columns": "<target>__zpred",
        },
        "outputs": {
            "out_dir": str(out_dir),
            "target_summary_all120": str(target_summary_path),
            "by_M": output_paths,
        },
    }
    manifest_path = out_dir / "bcp_krrraw_source_topM_inference_manifest_v1.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n[done]")
    print(
        json.dumps(
            {
                "status": "PASS",
                "n": int(len(meta)),
                "n_subjects": int(meta[args.group_col].astype(str).nunique()),
                "m_list": m_list,
                "target_summary": str(target_summary_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
