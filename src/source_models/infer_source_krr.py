"""Apply five matching frozen UKB KRR bundles to the locked ABCD cohort.

Use explicit bundle paths and a family configuration to match phenotype
or disease-PC models. RSFC inputs are row-centered and L2-normalized.
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
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
        obj = obj.item()
    if isinstance(obj, float) and (not np.isfinite(obj)):
        return None
    return obj


def unitize_rows(X: np.ndarray, dtype=np.float32) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X - np.mean(X, axis=1, keepdims=True)
    norm = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    norm[~np.isfinite(norm)] = 1.0
    norm[norm == 0.0] = 1.0
    Z = X / norm
    return Z.astype(dtype, copy=False)


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def infer_seed_from_path(path: Path) -> Optional[int]:
    m = re.search("seed(\\d+)", str(path))
    if not m:
        return None
    return int(m.group(1))


def load_fold_manifest(
    path: str, subjectkey_col: str, id_col: str, fold_col: str
) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fold manifest not found: {p}")
    df = pd.read_csv(p)
    required = ["_orig_row_idx", subjectkey_col, id_col, fold_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Fold manifest missing required columns: {missing}")
    out = df.copy()
    out["_orig_row_idx"] = pd.to_numeric(out["_orig_row_idx"], errors="raise").astype(
        np.int64
    )
    out[subjectkey_col] = out[subjectkey_col].astype(str)
    out[id_col] = out[id_col].astype(str)
    out[fold_col] = pd.to_numeric(out[fold_col], errors="raise").astype(np.int64)
    if out.duplicated("_orig_row_idx").any():
        raise RuntimeError("Fold manifest contains duplicated _orig_row_idx values.")
    if out[[subjectkey_col, id_col]].duplicated().any():
        raise RuntimeError("Fold manifest contains duplicated subject IDs.")
    folds = sorted(out[fold_col].unique().tolist())
    if folds != list(range(len(folds))):
        raise RuntimeError(f"Unexpected fold labels in fold manifest: {folds}")
    return out


def validate_locked_reference(
    path: str,
    manifest: pd.DataFrame,
    subjectkey_col: str,
    id_col: str,
    fold_col: str,
    strict: bool = True,
) -> Dict[str, object]:
    if path is None or str(path).strip() == "":
        return {"used": False, "reason": "empty path"}
    p = Path(path)
    if not p.exists():
        msg = f"Locked reference CSV not found: {p}"
        if strict:
            raise FileNotFoundError(msg)
        return {"used": False, "reason": msg}
    ref = pd.read_csv(p)
    required = [subjectkey_col, id_col, fold_col]
    missing = [c for c in required if c not in ref.columns]
    if missing:
        raise KeyError(f"Locked reference CSV missing required columns: {missing}")
    if len(ref) != len(manifest):
        raise RuntimeError(
            f"Locked reference length mismatch: ref={len(ref)}, manifest={len(manifest)}"
        )
    ref_subject = ref[subjectkey_col].astype(str).to_numpy()
    ref_id = ref[id_col].astype(str).to_numpy()
    ref_fold = pd.to_numeric(ref[fold_col], errors="raise").to_numpy(dtype=np.int64)
    man_subject = manifest[subjectkey_col].astype(str).to_numpy()
    man_id = manifest[id_col].astype(str).to_numpy()
    man_fold = manifest[fold_col].to_numpy(dtype=np.int64)
    if not np.array_equal(ref_subject, man_subject):
        raise RuntimeError(
            "Locked reference subjectkey order does not match fold manifest."
        )
    if not np.array_equal(ref_id, man_id):
        raise RuntimeError(
            "Locked reference src_subject_id order does not match fold manifest."
        )
    if not np.array_equal(ref_fold, man_fold):
        raise RuntimeError("Locked reference fold order does not match fold manifest.")
    audit = {
        "used": True,
        "path": str(p),
        "n": int(len(ref)),
        "columns_found": [
            c
            for c in ["_orig_row_idx", subjectkey_col, id_col, fold_col, "y_true_resid"]
            if c in ref.columns
        ],
    }
    if "_orig_row_idx" in ref.columns:
        ref_idx = pd.to_numeric(ref["_orig_row_idx"], errors="raise").to_numpy(
            dtype=np.int64
        )
        if not np.array_equal(
            ref_idx, manifest["_orig_row_idx"].to_numpy(dtype=np.int64)
        ):
            raise RuntimeError(
                "Locked reference _orig_row_idx order does not match fold manifest."
            )
        audit["orig_row_idx_checked"] = True
    else:
        audit["orig_row_idx_checked"] = False
    return audit


def load_abcd_locked_rows(
    abcd_csv: str,
    manifest: pd.DataFrame,
    edge_names: Sequence[str],
    subjectkey_col: str,
    id_col: str,
    fold_col: str,
    target_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    p = Path(abcd_csv)
    if not p.exists():
        raise FileNotFoundError(f"ABCD csv not found: {p}")
    df_all = pd.read_csv(p)
    df_all["_orig_row_idx"] = np.arange(len(df_all), dtype=np.int64)
    idx = manifest["_orig_row_idx"].to_numpy(dtype=np.int64)
    if np.any(idx < 0) or np.any(idx >= len(df_all)):
        raise RuntimeError(
            "Fold manifest contains _orig_row_idx outside ABCD_final.csv row range."
        )
    df = df_all.iloc[idx].copy().reset_index(drop=True)
    if subjectkey_col not in df.columns:
        raise KeyError(f"ABCD csv missing subjectkey column: {subjectkey_col}")
    if id_col not in df.columns:
        raise KeyError(f"ABCD csv missing id column: {id_col}")
    if not np.array_equal(
        df[subjectkey_col].astype(str).to_numpy(),
        manifest[subjectkey_col].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "ABCD rows selected by _orig_row_idx do not match manifest subjectkey order."
        )
    if not np.array_equal(
        df[id_col].astype(str).to_numpy(), manifest[id_col].astype(str).to_numpy()
    ):
        raise RuntimeError(
            "ABCD rows selected by _orig_row_idx do not match manifest id order."
        )
    missing_edges = [c for c in edge_names if c not in df.columns]
    if missing_edges:
        raise KeyError(
            f"ABCD csv missing edge columns from bundle edge_names, first 10: {missing_edges[:10]}"
        )
    E = (
        df[list(edge_names)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    if not np.isfinite(E).all():
        bad = int((~np.isfinite(E)).sum())
        raise RuntimeError(
            f"ABCD edge matrix contains NaN/Inf after locked row selection: {bad}"
        )
    meta_cols = ["_orig_row_idx", subjectkey_col, id_col]
    for c in ["interview_date", "age", "Age_years", "Sex", "site", target_col]:
        if c in df.columns and c not in meta_cols:
            meta_cols.append(c)
    meta = df[meta_cols].copy()
    meta[fold_col] = manifest[fold_col].to_numpy(dtype=np.int64)
    audit = {
        "abcd_csv": str(p),
        "n_rows_abcd_full": int(len(df_all)),
        "n_locked": int(len(df)),
        "n_edges": int(E.shape[1]),
        "meta_columns": list(meta.columns),
        "fold_counts": {
            str(k): int(v)
            for k, v in pd.Series(meta[fold_col]).value_counts().sort_index().items()
        },
    }
    return (meta, E, audit)


def load_bundle(path: Path) -> Dict[str, object]:
    bundle = joblib.load(path)
    required = ["edge_names", "target_names", "Z_train_full", "target_models"]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise KeyError(f"Bundle {path} missing required keys: {missing}")
    if str(bundle.get("bundle_type", "")) != "ukb_stage1_krr_pc_transfer_bundle":
        raise RuntimeError(
            f"Unexpected bundle_type in {path}: {bundle.get('bundle_type')}. Expected ukb_stage1_krr_pc_transfer_bundle."
        )
    Z = np.asarray(bundle["Z_train_full"])
    if Z.ndim != 2:
        raise RuntimeError(f"Z_train_full is not 2D in {path}: shape={Z.shape}")
    if not np.isfinite(Z).all():
        raise RuntimeError(f"Z_train_full contains NaN/Inf in {path}")
    return bundle


def validate_bundles(
    bundle_paths: Sequence[Path],
) -> Tuple[List[Dict[str, object]], List[str], List[str], Dict[str, object]]:
    bundles = []
    first_edges = None
    first_targets = None
    audit_rows = []
    for p in bundle_paths:
        b = load_bundle(p)
        edges = [str(x) for x in b["edge_names"]]
        targets = [str(x) for x in b["target_names"]]
        if first_edges is None:
            first_edges = edges
            first_targets = targets
        else:
            if edges != first_edges:
                raise RuntimeError(f"edge_names mismatch for bundle {p}")
            if targets != first_targets:
                raise RuntimeError(f"target_names mismatch for bundle {p}")
        target_models = b["target_models"]
        if sorted(target_models.keys()) != sorted(targets):
            missing = sorted(set(targets) - set(target_models.keys()))
            extra = sorted(set(target_models.keys()) - set(targets))
            raise RuntimeError(
                f"target_models/target_names mismatch in {p}. missing={missing}, extra={extra}"
            )
        Z = np.asarray(b["Z_train_full"])
        seed = b.get("metadata", {}).get("seed", infer_seed_from_path(p))
        audit_rows.append(
            {
                "path": str(p),
                "run_tag": str(b.get("run_tag", "")),
                "seed": int(seed) if seed is not None else None,
                "n_train_full_rows": int(Z.shape[0]),
                "n_edges": int(Z.shape[1]),
                "n_targets": int(len(targets)),
                "fit_index_space": str(
                    b.get("metadata", {}).get("fit_index_space", "")
                ),
                "refit_on_train_valid": bool(
                    b.get("metadata", {}).get("refit_on_train_valid", False)
                ),
            }
        )
        bundles.append(b)
    audit = {"n_bundles": int(len(bundles)), "bundle_rows": audit_rows}
    return (bundles, first_edges or [], first_targets or [], audit)


def predict_one_target_chunked(
    Z_abcd: np.ndarray,
    Z_train_full: np.ndarray,
    model: Dict[str, object],
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    status = str(model.get("status", ""))
    if status != "ok":
        raise RuntimeError(
            f"Target model status is not ok: {status}, reason={model.get('failure_reason')}"
        )
    fit_idx = np.asarray(model["fit_train_idx"], dtype=np.int64)
    dual = np.asarray(model["dual_coef"], dtype=np.float64).ravel()
    y_mu = float(model["y_mu"])
    y_sigma = float(model["y_sigma"])
    if fit_idx.ndim != 1:
        raise RuntimeError("fit_train_idx is not 1D")
    if dual.ndim != 1:
        raise RuntimeError("dual_coef is not 1D")
    if fit_idx.size != dual.size:
        raise RuntimeError(
            f"dual_coef length {dual.size} != fit_train_idx length {fit_idx.size}"
        )
    if fit_idx.size == 0:
        raise RuntimeError("fit_train_idx is empty")
    if np.any(fit_idx < 0) or np.any(fit_idx >= Z_train_full.shape[0]):
        raise RuntimeError("fit_train_idx out of bounds for Z_train_full")
    if not np.isfinite(dual).all():
        raise RuntimeError("dual_coef contains NaN/Inf")
    if not np.isfinite(y_mu) or not np.isfinite(y_sigma):
        raise RuntimeError("y_mu/y_sigma is NaN/Inf")
    if abs(y_sigma) < 1e-12:
        y_sigma = 1.0
    Z_fit = np.asarray(Z_train_full[fit_idx], dtype=np.float32)
    n = Z_abcd.shape[0]
    pred_z = np.empty(n, dtype=np.float64)
    cs = int(max(1, chunk_size))
    for start in range(0, n, cs):
        end = min(n, start + cs)
        K = np.asarray(Z_abcd[start:end] @ Z_fit.T, dtype=np.float64)
        pred_z[start:end] = K @ dual
    pred_raw = pred_z * y_sigma + y_mu
    return (pred_z, pred_raw)


def predict_one_bundle(
    bundle: Dict[str, object],
    Z_abcd: np.ndarray,
    meta: pd.DataFrame,
    target_names: Sequence[str],
    chunk_size: int,
    subjectkey_col: str,
    id_col: str,
    fold_col: str,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Z_train = np.asarray(bundle["Z_train_full"], dtype=np.float32)
    target_models = bundle["target_models"]
    seed = bundle.get("metadata", {}).get("seed", None)
    run_tag = str(bundle.get("run_tag", f"stage1_seed{seed}"))
    out = meta.copy()
    rows = []
    for j, target in enumerate(target_names):
        print(f"    target {j + 1:02d}/{len(target_names):02d}: {target}", flush=True)
        model = target_models[target]
        pred_z, pred_raw = predict_one_target_chunked(
            Z_abcd=Z_abcd, Z_train_full=Z_train, model=model, chunk_size=chunk_size
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
    preferred = [
        "_orig_row_idx",
        subjectkey_col,
        id_col,
        fold_col,
        target_col,
        "interview_date",
        "age",
        "Age_years",
        "Sex",
        "site",
    ]
    first_cols = [c for c in preferred if c in out.columns]
    pred_cols = []
    for t in target_names:
        pred_cols.extend([str(t), f"{t}__zpred"])
    other_cols = [c for c in out.columns if c not in first_cols and c not in pred_cols]
    out = out[first_cols + other_cols + pred_cols].copy()
    return (out, pd.DataFrame(rows))


def make_mean_predictions(
    seed_dfs: Sequence[pd.DataFrame],
    target_names: Sequence[str],
    subjectkey_col: str,
    id_col: str,
    fold_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not seed_dfs:
        raise RuntimeError("No seed prediction dataframes were provided.")
    base = seed_dfs[0].copy()
    id_cols = [
        c
        for c in [
            "_orig_row_idx",
            subjectkey_col,
            id_col,
            fold_col,
            "intelligence_score",
            "interview_date",
            "age",
            "Age_years",
            "Sex",
            "site",
        ]
        if c in base.columns
    ]
    mean_df = base[id_cols].copy()
    rows = []
    for target in target_names:
        raw_cols = [df[str(target)].to_numpy(dtype=np.float64) for df in seed_dfs]
        z_cols = [df[f"{target}__zpred"].to_numpy(dtype=np.float64) for df in seed_dfs]
        raw_stack = np.vstack(raw_cols)
        z_stack = np.vstack(z_cols)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--abcd-csv", required=True)
    parser.add_argument("--fold-manifest-csv", required=True)
    parser.add_argument("--locked-reference-csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    paths = [Path(p) for p in args.bundle]
    bundles, edges, targets, audit = validate_bundles(paths)
    seeds = [int(b["metadata"]["seed"]) for b in bundles]
    if sorted(seeds) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Exactly one bundle for each seed 0-4 is required.")
    if targets != config["target_names"]:
        raise RuntimeError(
            "Bundle targets/order do not match the source-family configuration."
        )
    if edges != [f"e{i}" for i in range(1, 1486)]:
        raise RuntimeError("Bundle edges must be e1 through e1485 in numeric order.")
    for b in bundles:
        metadata = b["metadata"]
        for key in [
            "n_total_subjects_loaded",
            "alphas",
            "test_size",
            "valid_size",
            "refit_on_train_valid",
        ]:
            expected = (
                config["expected_n_subjects"]
                if key == "n_total_subjects_loaded"
                else config[key]
            )
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"Bundle {key} differs from the source-family configuration."
                )
    order = np.argsort(seeds)
    bundles = [bundles[int(k)] for k in order]
    manifest = load_fold_manifest(
        args.fold_manifest_csv, "subjectkey", "src_subject_id", "fold"
    )
    reference_audit = validate_locked_reference(
        args.locked_reference_csv,
        manifest,
        "subjectkey",
        "src_subject_id",
        "fold",
        strict=True,
    )
    meta, E, cohort_audit = load_abcd_locked_rows(
        args.abcd_csv,
        manifest,
        edges,
        "subjectkey",
        "src_subject_id",
        "fold",
        "intelligence_score",
    )
    Z = unitize_rows(E, dtype=np.float32)
    out = Path(args.out_dir)
    per_seed = out / "per_seed"
    per_seed.mkdir(parents=True, exist_ok=True)
    frames, summaries = ([], [])
    for b in bundles:
        seed = int(b["metadata"]["seed"])
        frame, summary = predict_one_bundle(
            b,
            Z,
            meta,
            targets,
            args.chunk_size,
            "subjectkey",
            "src_subject_id",
            "fold",
            "intelligence_score",
        )
        frame.to_parquet(
            per_seed / f"abcd_krr_transferred_pc_stage1_seed{seed}.parquet", index=False
        )
        frames.append(frame)
        summaries.append(summary)
    mean, mean_summary = make_mean_predictions(
        frames, targets, "subjectkey", "src_subject_id", "fold"
    )
    generic = out / "abcd_predicted_source_mean5_krr.parquet"
    mean.to_parquet(generic, index=False)
    if config["source_family"] == "disease_pc":
        mean.to_parquet(
            out / "abcd_predicted_disease_pc_top20_mean5_krr.parquet", index=False
        )
        selected = [
            t
            for t in targets
            if t
            not in {
                "disease_phecode__pc06",
                "disease_phecode__pc09",
                "disease_phecode__pc11",
                "disease_phecode__pc14",
            }
        ]
        columns = [c for c in meta.columns if c in mean.columns]
        columns += [c for t in selected for c in (t, f"{t}__zpred")]
        mean[columns].to_parquet(
            out / "abcd_predicted_disease_pc_top20_exploratory16_mean5_krr.parquet",
            index=False,
        )
        (out / "feature_freeze_disease_exploratory16.tsv").write_text(
            "\n".join(selected) + "\n"
        )
    pd.concat(summaries, ignore_index=True).to_csv(
        out / "per_seed_target_summary.csv", index=False
    )
    mean_summary.to_csv(out / "mean5_target_summary.csv", index=False)
    output_manifest = dict(
        script=Path(__file__).name,
        config=config,
        input_bundles=args.bundle,
        fold_manifest=args.fold_manifest_csv,
        locked_reference_audit=reference_audit,
        cohort_audit=cohort_audit,
        bundle_audit=audit,
        target_names=targets,
        output_parquet=str(generic),
    )
    (out / "abcd_krr_transferred_pc_manifest.json").write_text(
        json.dumps(sanitize_for_json(output_manifest), indent=2)
    )


if __name__ == "__main__":
    main()
