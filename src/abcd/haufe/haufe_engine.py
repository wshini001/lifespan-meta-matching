"""Calculate ABCD KRR-PC12 Haufe patterns and ROI-index summaries.

Edges are assigned ROI pairs by row-wise traversal of the upper triangle.
"""

from __future__ import annotations
import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {path}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def fuzzy_col(
    df: pd.DataFrame,
    include: Sequence[str],
    exclude: Sequence[str] = (),
    numeric: bool = False,
) -> Optional[str]:
    hits: List[str] = []
    for c in df.columns:
        s = str(c).lower()
        if all((k.lower() in s for k in include)) and (
            not any((k.lower() in s for k in exclude))
        ):
            if numeric:
                vals = pd.to_numeric(df[c], errors="coerce")
                if vals.notna().sum() == 0:
                    continue
            hits.append(c)
    if not hits:
        return None
    return sorted(hits, key=lambda x: (len(str(x)), str(x).lower()))[0]


def detect_subject_col(df: pd.DataFrame, label: str) -> str:
    col = first_existing_col(
        df,
        [
            "subjectkey",
            "src_subject_id",
            "subject_id",
            "participant_id",
            "eid",
            "id",
            "ID",
            "SubjID",
        ],
    )
    if col is None:
        raise KeyError(
            f"Could not detect subject id column in {label}. Columns={list(df.columns)[:30]}"
        )
    return col


def detect_prediction_col(
    df: pd.DataFrame, role: str, required: bool = True
) -> Optional[str]:
    role = role.lower()
    if role == "ensemble":
        candidates = [
            "y_pred_ensemble_nested",
            "y_pred_ensemble",
            "pred_ensemble",
            "ensemble_pred",
            "prediction_ensemble",
            "yhat_ensemble",
            "y_pred_final",
            "final_prediction",
        ]
        fuzzy_include = ["ensemble"]
        fuzzy_exclude = ["avg", "raw", "weight"]
    elif role == "abcd":
        candidates = [
            "y_pred_abcd_nested",
            "y_pred_abcd",
            "pred_abcd",
            "abcd_pred",
            "yhat_abcd",
            "y_pred_abcd_only",
            "baseline_pred",
            "y_pred_baseline",
        ]
        fuzzy_include = ["abcd"]
        fuzzy_exclude = ["weight"]
    elif role == "pc":
        candidates = [
            "y_pred_pc_nested",
            "y_pred_pc",
            "pred_pc",
            "pc_pred",
            "yhat_pc",
            "y_pred_krr_pc_nested",
        ]
        fuzzy_include = ["pc"]
        fuzzy_exclude = ["weight"]
    elif role == "target":
        candidates = [
            "y_true",
            "target",
            "y",
            "intelligence_score",
            "y_resid",
            "resid_y",
        ]
        fuzzy_include = ["true"]
        fuzzy_exclude = []
    else:
        raise ValueError(f"Unknown role: {role}")
    col = first_existing_col(df, candidates)
    if col is None:
        col = fuzzy_col(df, fuzzy_include, fuzzy_exclude, numeric=True)
    if col is None and required:
        raise KeyError(
            f"Could not detect {role} prediction column. Columns={list(df.columns)}"
        )
    return col


def detect_fold_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing_col(
        df,
        [
            "outer_fold",
            "fold",
            "fold_id",
            "test_fold",
            "cv_fold",
            "outer_fold_id",
            "split",
        ],
    )


def resolve_post_merge_col(
    df: pd.DataFrame, base_col: Optional[str], prefer: str = "x"
) -> Optional[str]:
    if base_col is None:
        return None
    if base_col in df.columns:
        return base_col
    suffixes = ["_x", "_y"] if prefer == "x" else ["_y", "_x"]
    for suf in suffixes:
        cand = f"{base_col}{suf}"
        if cand in df.columns:
            return cand
    lower = {str(c).lower(): c for c in df.columns}
    for suf in suffixes:
        cand = f"{base_col}{suf}".lower()
        if cand in lower:
            return lower[cand]
    if str(base_col).lower() in lower:
        return lower[str(base_col).lower()]
    return None


def detect_edge_cols(
    df: pd.DataFrame, expected_n_edges: Optional[int] = None
) -> List[str]:
    edge_cols = []
    for c in df.columns:
        m = re.fullmatch("e(\\d+)", str(c))
        if m:
            edge_cols.append((int(m.group(1)), c))
    edge_cols = [c for _, c in sorted(edge_cols, key=lambda x: x[0])]
    if not edge_cols:
        raise ValueError("Could not detect RSFC edge columns matching e1, e2, ..., eN.")
    if expected_n_edges is not None and len(edge_cols) != expected_n_edges:
        raise ValueError(
            f"Expected {expected_n_edges} edge columns but found {len(edge_cols)}."
        )
    return edge_cols


def load_abcd_edges_with_optional_locked_cohort(
    abcd_csv: Path,
    edge_cols: Sequence[str],
    abcd_subject_col: str,
    selected_cohort_csv: Optional[Path],
    id_col: str = "src_subject_id",
    subjectkey_col: str = "subjectkey",
    target_col: str = "intelligence_score",
) -> pd.DataFrame:
    header = pd.read_csv(abcd_csv, nrows=0)
    available_cols = set(header.columns)
    meta_candidates = [abcd_subject_col, subjectkey_col, id_col, target_col]
    read_cols = []
    for c in meta_candidates:
        if c in available_cols and c not in read_cols:
            read_cols.append(c)
    for c in edge_cols:
        if c not in available_cols:
            raise KeyError(
                f"Edge column {c} was detected in preview but is missing from full ABCD header."
            )
        read_cols.append(c)
    abcd = pd.read_csv(abcd_csv, usecols=read_cols)
    abcd["_orig_row_idx"] = np.arange(len(abcd), dtype=np.int64)
    if selected_cohort_csv is None or str(selected_cohort_csv).strip() == "":
        if abcd[abcd_subject_col].astype(str).duplicated().any():
            dup = int(abcd[abcd_subject_col].astype(str).duplicated().sum())
            raise ValueError(
                f"ABCD CSV has duplicated subject rows and no selected cohort was provided. duplicated rows={dup}. Provide --selected_cohort_csv to lock to Stage2A cohort."
            )
        print(
            f"[load_abcd] no selected cohort provided; using full ABCD rows N={len(abcd)}"
        )
        return abcd
    selected_cohort_csv = Path(selected_cohort_csv)
    if not selected_cohort_csv.exists():
        raise FileNotFoundError(f"selected cohort CSV not found: {selected_cohort_csv}")
    selected = pd.read_csv(selected_cohort_csv)
    required = ["_orig_row_idx"]
    for c in [subjectkey_col, id_col]:
        if c in selected.columns:
            required.append(c)
    missing = [c for c in required if c not in selected.columns]
    if missing:
        raise KeyError(f"selected cohort csv missing required columns: {missing}")
    selected["_orig_row_idx"] = pd.to_numeric(
        selected["_orig_row_idx"], errors="raise"
    ).astype(np.int64)
    keep_idx = selected["_orig_row_idx"].to_numpy(dtype=np.int64)
    if np.any(keep_idx < 0) or np.any(keep_idx >= len(abcd)):
        raise RuntimeError(
            "selected cohort contains _orig_row_idx outside ABCD_final.csv row range."
        )
    if (
        abcd_subject_col in selected.columns
        and selected[abcd_subject_col].astype(str).duplicated().any()
    ):
        dup = int(selected[abcd_subject_col].astype(str).duplicated().sum())
        raise RuntimeError(
            f"selected cohort csv contains duplicated {abcd_subject_col}: {dup}"
        )
    if id_col in selected.columns and selected[id_col].astype(str).duplicated().any():
        dup = int(selected[id_col].astype(str).duplicated().sum())
        raise RuntimeError(f"selected cohort csv contains duplicated {id_col}: {dup}")
    locked = (
        abcd.set_index("_orig_row_idx", drop=False).loc[keep_idx].reset_index(drop=True)
    )
    for c in [subjectkey_col, id_col]:
        if c in locked.columns and c in selected.columns:
            if not np.array_equal(
                locked[c].astype(str).to_numpy(), selected[c].astype(str).to_numpy()
            ):
                raise RuntimeError(
                    f"Locked cohort {c} order does not match selected cohort {c} order."
                )
    if abcd_subject_col not in locked.columns:
        raise KeyError(
            f"Locked ABCD table does not contain merge subject column: {abcd_subject_col}"
        )
    if locked[abcd_subject_col].astype(str).duplicated().any():
        dup = int(locked[abcd_subject_col].astype(str).duplicated().sum())
        raise RuntimeError(
            f"Locked ABCD cohort still has duplicated {abcd_subject_col}: {dup}"
        )
    bad_edges = (
        locked[list(edge_cols)].apply(pd.to_numeric, errors="coerce").isna().any(axis=1)
    )
    if bad_edges.any():
        raise RuntimeError(
            f"Locked cohort contains {int(bad_edges.sum())} rows with NaN/non-numeric RSFC edge values; do not silently drop rows because this would break alignment."
        )
    print(
        f"[load_abcd] using Stage2A locked cohort: N={len(locked)}, selected_cohort_csv={selected_cohort_csv}"
    )
    return locked


def infer_n_roi(n_edges: int) -> int:
    n = (1 + math.sqrt(1 + 8 * n_edges)) / 2
    n_int = int(round(n))
    if n_int * (n_int - 1) // 2 != n_edges:
        raise ValueError(
            f"Number of edges {n_edges} is not triangular; cannot infer n_roi."
        )
    return n_int


def make_edge_index(n_roi: int, edge_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    k = 0
    for i in range(n_roi):
        for j in range(i + 1, n_roi):
            rows.append(
                {
                    "edge_col": edge_cols[k],
                    "edge_number": k + 1,
                    "roi_i": i + 1,
                    "roi_j": j + 1,
                    "edge_label": f"ROI{i + 1:02d}-ROI{j + 1:02d}",
                }
            )
            k += 1
    return pd.DataFrame(rows)


def sanitize_numeric_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return (
        df.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    )


def compute_pnf(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X2 = X[ok, :]
    y2 = y[ok]
    n = int(X2.shape[0])
    if n < 5:
        raise ValueError("Too few complete subjects to compute PNF.")
    Xc = X2 - np.nanmean(X2, axis=0, keepdims=True)
    yc = y2 - np.nanmean(y2)
    cov = Xc.T @ yc / max(n - 1, 1)
    x_sd = np.sqrt(np.sum(Xc * Xc, axis=0) / max(n - 1, 1))
    y_sd = float(np.sqrt(np.sum(yc * yc) / max(n - 1, 1)))
    denom = x_sd * y_sd
    corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)
    return {
        "cov": cov,
        "corr": corr,
        "n_complete": np.repeat(n, X.shape[1]).astype(int),
    }


def vector_to_matrix(values: np.ndarray, n_roi: int) -> np.ndarray:
    mat = np.full((n_roi, n_roi), np.nan, dtype=float)
    k = 0
    for i in range(n_roi):
        mat[i, i] = 0.0
        for j in range(i + 1, n_roi):
            v = float(values[k])
            mat[i, j] = v
            mat[j, i] = v
            k += 1
    return mat


def compute_node_importance(
    edge_table: pd.DataFrame, n_roi: int, prefix: str = "pnf_cov_incremental"
) -> pd.DataFrame:
    signed = np.zeros(n_roi, dtype=float)
    abs_sum = np.zeros(n_roi, dtype=float)
    pos_sum = np.zeros(n_roi, dtype=float)
    neg_sum = np.zeros(n_roi, dtype=float)
    edge_count = np.zeros(n_roi, dtype=int)
    for _, r in edge_table.iterrows():
        i = int(r["roi_i"]) - 1
        j = int(r["roi_j"]) - 1
        v = float(r[prefix])
        if not np.isfinite(v):
            continue
        signed[i] += v
        signed[j] += v
        abs_sum[i] += abs(v)
        abs_sum[j] += abs(v)
        if v >= 0:
            pos_sum[i] += v
            pos_sum[j] += v
        else:
            neg_sum[i] += v
            neg_sum[j] += v
        edge_count[i] += 1
        edge_count[j] += 1
    out = pd.DataFrame(
        {
            "roi_index": np.arange(1, n_roi + 1),
            "roi_label": [f"ROI{i:02d}" for i in range(1, n_roi + 1)],
            "node_signed_pnf_sum": signed,
            "node_abs_pnf_sum": abs_sum,
            "node_positive_pnf_sum": pos_sum,
            "node_negative_pnf_sum": neg_sum,
            "n_edges": edge_count,
        }
    )
    out["node_abs_pnf_mean"] = out["node_abs_pnf_sum"] / np.maximum(out["n_edges"], 1)
    out["node_importance_rank"] = (
        out["node_abs_pnf_sum"].rank(method="min", ascending=False).astype(int)
    )
    out = out.sort_values("node_importance_rank").reset_index(drop=True)
    return out


def sign_consistency(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    signs = np.sign(vals)
    signs = signs[signs != 0]
    if signs.size == 0:
        return 0.0
    return float(max(np.mean(signs > 0), np.mean(signs < 0)))


def compute_fold_stability(
    X: np.ndarray, y: np.ndarray, folds: pd.Series
) -> pd.DataFrame:
    unique_folds = [f for f in pd.unique(folds) if pd.notna(f)]
    unique_folds = sorted(unique_folds)
    fold_covs: List[np.ndarray] = []
    for f in unique_folds:
        mask = folds.to_numpy() == f
        if int(mask.sum()) < 5:
            continue
        res = compute_pnf(X[mask, :], y[mask])
        fold_covs.append(res["cov"])
    if not fold_covs:
        return pd.DataFrame()
    arr = np.vstack(fold_covs)
    out = pd.DataFrame(
        {
            "fold_pnf_incremental_cov_mean": np.nanmean(arr, axis=0),
            "fold_pnf_incremental_cov_std": (
                np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.nan
            ),
            "fold_pnf_incremental_sign_consistency": [
                sign_consistency(arr[:, k]) for k in range(arr.shape[1])
            ],
            "n_folds_for_stability": arr.shape[0],
        }
    )
    return out


def run(args: argparse.Namespace) -> None:
    abcd_csv = Path(args.abcd_csv)
    pred_csv = Path(args.pred_csv)
    out_dir = ensure_dir(Path(args.out_dir))
    print("[config] make_section3_7_haufe_style_rsfc_pnf_v3")
    print(f"  abcd_csv={abcd_csv}")
    print(f"  pred_csv={pred_csv}")
    print(f"  selected_cohort_csv={args.selected_cohort_csv}")
    print(f"  out_dir={out_dir}")
    print(f"  expected_edges={args.expected_edges}")
    if not abcd_csv.exists():
        raise FileNotFoundError(f"ABCD CSV not found: {abcd_csv}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")
    pred = pd.read_csv(pred_csv)
    pred_subject_col = detect_subject_col(pred, "prediction CSV")
    ensemble_col = detect_prediction_col(pred, "ensemble", required=True)
    abcd_col = detect_prediction_col(pred, "abcd", required=True)
    pc_col = detect_prediction_col(pred, "pc", required=False)
    target_col = detect_prediction_col(pred, "target", required=False)
    fold_col = detect_fold_col(pred)
    print("[columns] prediction columns used:")
    print(f"  subject={pred_subject_col}")
    print(f"  ensemble={ensemble_col}")
    print(f"  abcd_only={abcd_col}")
    print(f"  pc_stream={pc_col}")
    print(f"  target={target_col}")
    print(f"  fold={fold_col}")
    if pred[pred_subject_col].duplicated().any():
        dup = int(pred[pred_subject_col].duplicated().sum())
        raise ValueError(f"Prediction CSV has duplicated subject rows: {dup}")
    abcd_preview = pd.read_csv(abcd_csv, nrows=5)
    abcd_subject_col = detect_subject_col(abcd_preview, "ABCD CSV")
    edge_cols = detect_edge_cols(abcd_preview, expected_n_edges=args.expected_edges)
    selected_cohort_csv = (
        Path(args.selected_cohort_csv)
        if str(args.selected_cohort_csv).strip()
        else None
    )
    abcd = load_abcd_edges_with_optional_locked_cohort(
        abcd_csv=abcd_csv,
        edge_cols=edge_cols,
        abcd_subject_col=abcd_subject_col,
        selected_cohort_csv=selected_cohort_csv,
        id_col=args.id_col,
        subjectkey_col=args.subjectkey_col,
        target_col=args.target_col,
    )
    n_edges = len(edge_cols)
    n_roi = infer_n_roi(n_edges)
    edge_index = make_edge_index(n_roi, edge_cols)
    pred_keep_cols = [pred_subject_col, ensemble_col, abcd_col]
    if pc_col:
        pred_keep_cols.append(pc_col)
    if target_col:
        pred_keep_cols.append(target_col)
    if fold_col:
        pred_keep_cols.append(fold_col)
    pred2 = pred[pred_keep_cols].copy()
    pred2["_merge_subject_id"] = pred2[pred_subject_col].astype(str)
    abcd["_merge_subject_id"] = abcd[abcd_subject_col].astype(str)
    merged = pred2.merge(
        abcd, on="_merge_subject_id", how="inner", validate="one_to_one"
    )
    print(f"[merge] pred={len(pred2)}, abcd={len(abcd)}, merged={len(merged)}")
    if len(merged) < args.min_subjects:
        raise ValueError(
            f"Too few aligned subjects: {len(merged)} < {args.min_subjects}"
        )
    ensemble_merge_col = resolve_post_merge_col(merged, ensemble_col, prefer="x")
    abcd_merge_col = resolve_post_merge_col(merged, abcd_col, prefer="x")
    pc_merge_col = (
        resolve_post_merge_col(merged, pc_col, prefer="x") if pc_col else None
    )
    target_merge_col = (
        resolve_post_merge_col(merged, target_col, prefer="x") if target_col else None
    )
    fold_merge_col = (
        resolve_post_merge_col(merged, fold_col, prefer="x") if fold_col else None
    )
    print("[columns] post-merge columns used:")
    print(f"  ensemble={ensemble_merge_col}")
    print(f"  abcd_only={abcd_merge_col}")
    print(f"  pc_stream={pc_merge_col}")
    print(f"  target={target_merge_col}")
    print(f"  fold={fold_merge_col}")
    if ensemble_merge_col is None or abcd_merge_col is None:
        raise KeyError(
            "Could not resolve ensemble or ABCD-only prediction columns after merge."
        )
    y_ensemble = pd.to_numeric(merged[ensemble_merge_col], errors="coerce").to_numpy(
        dtype=np.float64
    )
    y_abcd = pd.to_numeric(merged[abcd_merge_col], errors="coerce").to_numpy(
        dtype=np.float64
    )
    y_incremental = y_ensemble - y_abcd
    y_pc = (
        pd.to_numeric(merged[pc_merge_col], errors="coerce").to_numpy(dtype=np.float64)
        if pc_merge_col
        else None
    )
    y_target = (
        pd.to_numeric(merged[target_merge_col], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if target_merge_col
        else None
    )
    X = sanitize_numeric_matrix(merged, edge_cols)
    row_ok = (
        np.isfinite(y_ensemble)
        & np.isfinite(y_abcd)
        & np.isfinite(y_incremental)
        & np.all(np.isfinite(X), axis=1)
    )
    if not row_ok.all():
        raise ValueError(
            "Non-finite primary inputs; the locked cohort requires finite values"
        )
    X = X[row_ok, :]
    y_ensemble = y_ensemble[row_ok]
    y_abcd = y_abcd[row_ok]
    y_incremental = y_incremental[row_ok]
    if y_pc is not None:
        y_pc = y_pc[row_ok]
    if y_target is not None:
        y_target = y_target[row_ok]
    fold_series = merged.loc[row_ok, fold_merge_col] if fold_merge_col else None
    pnf_ensemble = compute_pnf(X, y_ensemble)
    pnf_abcd = compute_pnf(X, y_abcd)
    pnf_incremental = compute_pnf(X, y_incremental)
    pnf_pc = compute_pnf(X, y_pc) if y_pc is not None else None
    pnf_target = (
        compute_pnf(X, y_target)
        if y_target is not None and np.isfinite(y_target).sum() >= args.min_subjects
        else None
    )
    edge_table = edge_index.copy()
    edge_table["pnf_cov_final"] = pnf_ensemble["cov"]
    edge_table["pnf_corr_final"] = pnf_ensemble["corr"]
    edge_table["pnf_cov_abcd_only"] = pnf_abcd["cov"]
    edge_table["pnf_corr_abcd_only"] = pnf_abcd["corr"]
    edge_table["pnf_cov_incremental"] = pnf_incremental["cov"]
    edge_table["pnf_corr_incremental"] = pnf_incremental["corr"]
    if pnf_pc is not None:
        edge_table["pnf_cov_pc_stream"] = pnf_pc["cov"]
        edge_table["pnf_corr_pc_stream"] = pnf_pc["corr"]
    if pnf_target is not None:
        edge_table["pnf_cov_target"] = pnf_target["cov"]
        edge_table["pnf_corr_target"] = pnf_target["corr"]
    edge_table["abs_pnf_cov_incremental"] = edge_table["pnf_cov_incremental"].abs()
    edge_table["abs_pnf_corr_incremental"] = edge_table["pnf_corr_incremental"].abs()
    edge_table["incremental_pnf_sign"] = np.where(
        edge_table["pnf_cov_incremental"] >= 0, "positive", "negative"
    )
    edge_table["incremental_abs_rank"] = (
        edge_table["abs_pnf_cov_incremental"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    if fold_series is not None:
        stab = compute_fold_stability(
            X, y_incremental, fold_series.reset_index(drop=True)
        )
        if not stab.empty and len(stab) == len(edge_table):
            edge_table = pd.concat([edge_table, stab], axis=1)
    edge_table = edge_table.sort_values("incremental_abs_rank").reset_index(drop=True)
    top_pos = (
        edge_table.sort_values("pnf_cov_incremental", ascending=False)
        .head(args.top_edge_n)
        .copy()
    )
    top_neg = (
        edge_table.sort_values("pnf_cov_incremental", ascending=True)
        .head(args.top_edge_n)
        .copy()
    )
    top_edges = (
        pd.concat(
            [
                top_pos.assign(direction="positive"),
                top_neg.assign(direction="negative"),
            ],
            axis=0,
        )
        .sort_values("abs_pnf_cov_incremental", ascending=False)
        .reset_index(drop=True)
    )
    node_df = compute_node_importance(edge_table, n_roi, prefix="pnf_cov_incremental")
    mat_final = vector_to_matrix(
        edge_index.merge(
            edge_table[["edge_col", "pnf_cov_final"]], on="edge_col", how="left"
        )["pnf_cov_final"].to_numpy(),
        n_roi,
    )
    mat_abcd = vector_to_matrix(
        edge_index.merge(
            edge_table[["edge_col", "pnf_cov_abcd_only"]], on="edge_col", how="left"
        )["pnf_cov_abcd_only"].to_numpy(),
        n_roi,
    )
    mat_incremental = vector_to_matrix(
        edge_index.merge(
            edge_table[["edge_col", "pnf_cov_incremental"]], on="edge_col", how="left"
        )["pnf_cov_incremental"].to_numpy(),
        n_roi,
    )
    np.save(out_dir / "section3_7_pnf_matrix_final_cov.npy", mat_final)
    np.save(out_dir / "section3_7_pnf_matrix_abcd_only_cov.npy", mat_abcd)
    np.save(out_dir / "section3_7_pnf_matrix_incremental_cov.npy", mat_incremental)
    labels = [f"ROI{i:02d}" for i in range(1, n_roi + 1)]
    pd.DataFrame(mat_final, index=labels, columns=labels).to_csv(
        out_dir / "section3_7_pnf_matrix_final_cov.csv"
    )
    pd.DataFrame(mat_abcd, index=labels, columns=labels).to_csv(
        out_dir / "section3_7_pnf_matrix_abcd_only_cov.csv"
    )
    pd.DataFrame(mat_incremental, index=labels, columns=labels).to_csv(
        out_dir / "section3_7_pnf_matrix_incremental_cov.csv"
    )
    inc_final_edge_corr = float(
        np.corrcoef(edge_table["pnf_cov_final"], edge_table["pnf_cov_incremental"])[
            0, 1
        ]
    )
    inc_abcd_edge_corr = float(
        np.corrcoef(edge_table["pnf_cov_abcd_only"], edge_table["pnf_cov_incremental"])[
            0, 1
        ]
    )
    summary = {
        "script": "haufe_engine.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_subjects": int(X.shape[0]),
        "n_edges": int(n_edges),
        "n_roi": int(n_roi),
        "edge_column_rule": "Columns e1..e1485, sorted numerically; ROI pairs assigned by row-wise upper-triangle traversal, starting with ROI01-ROI02.",
        "selected_cohort_csv": (
            str(selected_cohort_csv) if selected_cohort_csv is not None else ""
        ),
        "prediction_columns": {
            "subject_col_pred": pred_subject_col,
            "subject_col_abcd": abcd_subject_col,
            "ensemble": ensemble_col,
            "abcd_only": abcd_col,
            "pc_stream": pc_col,
            "target": target_col,
            "fold": fold_col,
            "ensemble_after_merge": ensemble_merge_col,
            "abcd_only_after_merge": abcd_merge_col,
            "pc_stream_after_merge": pc_merge_col,
            "target_after_merge": target_merge_col,
            "fold_after_merge": fold_merge_col,
        },
        "primary_pnf_definition": "cov(RSFC edge, y_pred_ensemble - y_pred_abcd_only)",
        "edge_level_corr_final_vs_incremental_pnf": inc_final_edge_corr,
        "edge_level_corr_abcd_only_vs_incremental_pnf": inc_abcd_edge_corr,
        "top_positive_incremental_edges": top_pos[
            ["edge_col", "edge_label", "pnf_cov_incremental", "pnf_corr_incremental"]
        ]
        .head(10)
        .to_dict(orient="records"),
        "top_negative_incremental_edges": top_neg[
            ["edge_col", "edge_label", "pnf_cov_incremental", "pnf_corr_incremental"]
        ]
        .head(10)
        .to_dict(orient="records"),
        "top_node_abs_incremental_importance": node_df[
            ["roi_label", "node_abs_pnf_sum", "node_importance_rank"]
        ]
        .head(10)
        .to_dict(orient="records"),
        "interpretation_limitations": [
            "The analysis is edge-level and ROI-index-level rather than named-network-level.",
            "PNF values are predictive associations, not causal effects.",
        ],
    }
    save_csv(edge_table, out_dir / "section3_7_haufe_pnf_edge_table.csv")
    save_csv(top_edges, out_dir / "section3_7_haufe_pnf_top_edges_incremental.csv")
    save_csv(
        top_pos, out_dir / "section3_7_haufe_pnf_top_positive_edges_incremental.csv"
    )
    save_csv(
        top_neg, out_dir / "section3_7_haufe_pnf_top_negative_edges_incremental.csv"
    )
    save_csv(node_df, out_dir / "section3_7_haufe_pnf_node_importance_incremental.csv")
    input_manifest = pd.DataFrame(
        [
            {
                "file_role": "abcd_csv",
                "path": str(abcd_csv),
                "exists": abcd_csv.exists(),
                "mode": "read_only",
            },
            {
                "file_role": "selected_cohort_csv",
                "path": (
                    str(selected_cohort_csv) if selected_cohort_csv is not None else ""
                ),
                "exists": bool(
                    selected_cohort_csv is not None and selected_cohort_csv.exists()
                ),
                "mode": "read_only",
            },
            {
                "file_role": "prediction_csv",
                "path": str(pred_csv),
                "exists": pred_csv.exists(),
                "mode": "read_only",
            },
        ]
    )
    save_csv(input_manifest, out_dir / "section3_7_input_file_manifest.csv")
    write_json(summary, out_dir / "section3_7_haufe_pnf_summary_metrics.json")
    print("[done] Section 3.7 Haufe-style RSFC PNF v3 completed.")
    print(f"[done] output dir = {out_dir}")
    print(f"[done] aligned subjects = {X.shape[0]}")
    print(f"[done] edges = {n_edges}, inferred ROI count = {n_roi}")
    print("[done] top positive incremental PNF edges:")
    for _, r in top_pos.head(5).iterrows():
        print(
            f"  {r['edge_label']} {r['edge_col']}: cov={r['pnf_cov_incremental']:+.6g}, corr={r['pnf_corr_incremental']:+.6g}"
        )
    print("[done] top negative incremental PNF edges:")
    for _, r in top_neg.head(5).iterrows():
        print(
            f"  {r['edge_label']} {r['edge_col']}: cov={r['pnf_cov_incremental']:+.6g}, corr={r['pnf_corr_incremental']:+.6g}"
        )
    print("[done] top ROI-index node importance:")
    for _, r in node_df.head(8).iterrows():
        print(
            f"  {r['roi_label']}: abs_sum={r['node_abs_pnf_sum']:.6g}, signed_sum={r['node_signed_pnf_sum']:+.6g}"
        )
