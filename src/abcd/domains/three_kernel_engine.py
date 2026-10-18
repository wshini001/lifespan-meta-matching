"""Fit ABCD KRR models combining RSFC, cognition-PC and noncognition-PC kernels."""

from domain_common import (
    bootstrap_delta,
    compute_metrics,
    find_col,
    fit_predict_precomputed,
    load_locked_reference,
    load_pc_predictions,
    make_numeric_series,
    parse_pc_block,
    residualize_train_test,
    score_by_metric,
    transform_pc_train_test,
    unitize_rows,
)
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def parse_float_grid(s: str, name: str = "grid") -> List[float]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if not vals:
        raise ValueError(f"Empty {name}")
    return vals


def load_abcd_locked_data(
    abcd_csv: str,
    locked: pd.DataFrame,
    fold_manifest_csv: str,
    subjectkey_col: str,
    id_col: str,
    fold_col: str,
    target_col: str,
    site_col: str,
    age_col: str,
    sex_col: str,
    edu_col: str,
    motion_cols: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    abcd_path = Path(abcd_csv)
    if not abcd_path.exists():
        raise FileNotFoundError(
            f"ABCD csv not found: {abcd_path}. Please rerun with --abcd_csv pointing to the actual ABCD_final.csv."
        )
    manifest_path = Path(fold_manifest_csv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fold manifest CSV not found: {manifest_path}")
    df_all = pd.read_csv(abcd_path)
    df_all["_orig_row_idx"] = np.arange(len(df_all), dtype=np.int64)
    manifest = pd.read_csv(manifest_path)
    required_manifest_cols = ["_orig_row_idx", subjectkey_col, id_col, fold_col]
    missing_manifest = [c for c in required_manifest_cols if c not in manifest.columns]
    if missing_manifest:
        raise KeyError(f"Fold manifest missing required columns: {missing_manifest}")
    if len(manifest) != len(locked):
        raise RuntimeError(
            f"Manifest and locked prediction CSV length mismatch: manifest={len(manifest)}, locked={len(locked)}"
        )
    for c in [subjectkey_col, id_col]:
        manifest[c] = manifest[c].astype(str)
        locked[c] = locked[c].astype(str)
    manifest[fold_col] = manifest[fold_col].astype(int)
    locked[fold_col] = locked[fold_col].astype(int)
    if not np.array_equal(
        manifest[subjectkey_col].to_numpy(), locked[subjectkey_col].to_numpy()
    ):
        raise RuntimeError(
            "Fold manifest subjectkey order does not match locked prediction CSV."
        )
    if not np.array_equal(manifest[id_col].to_numpy(), locked[id_col].to_numpy()):
        raise RuntimeError(
            "Fold manifest src_subject_id order does not match locked prediction CSV."
        )
    if not np.array_equal(manifest[fold_col].to_numpy(), locked[fold_col].to_numpy()):
        raise RuntimeError(
            "Fold manifest fold order does not match locked prediction CSV."
        )
    orig_idx = pd.to_numeric(manifest["_orig_row_idx"], errors="raise").to_numpy(
        dtype=np.int64
    )
    if np.any(orig_idx < 0) or np.any(orig_idx >= len(df_all)):
        raise RuntimeError(
            "Fold manifest contains _orig_row_idx outside ABCD_final.csv row range."
        )
    if len(np.unique(orig_idx)) != len(orig_idx):
        raise RuntimeError("Fold manifest contains duplicated _orig_row_idx values.")
    df = df_all.iloc[orig_idx].copy().reset_index(drop=True)
    for c in [subjectkey_col, id_col]:
        if c not in df.columns:
            raise KeyError(f"ABCD csv missing required ID column: {c}")
        df[c] = df[c].astype(str)
    if not np.array_equal(
        df[subjectkey_col].to_numpy(), manifest[subjectkey_col].to_numpy()
    ):
        raise RuntimeError(
            "ABCD rows selected by _orig_row_idx do not match manifest subjectkey."
        )
    if not np.array_equal(df[id_col].to_numpy(), manifest[id_col].to_numpy()):
        raise RuntimeError(
            "ABCD rows selected by _orig_row_idx do not match manifest src_subject_id."
        )
    df[fold_col] = manifest[fold_col].to_numpy(dtype=int)
    edge_cols = [f"e{i}" for i in range(1, 1486)]
    missing_edges = [c for c in edge_cols if c not in df.columns]
    if missing_edges:
        raise KeyError(
            f"Missing edge columns in ABCD csv, first 10: {missing_edges[:10]}"
        )
    resolved_age = (
        age_col
        if age_col
        else find_col(
            df,
            ["Age_years", "Age", "age", "interview_age", "interview_age_years"],
            "age",
        )
    )
    resolved_sex = (
        sex_col if sex_col else find_col(df, ["Sex", "sex", "gender", "sex_num"], "sex")
    )
    resolved_site = (
        site_col
        if site_col
        else find_col(
            df, ["site", "Site", "site_id", "site_id_l", "scanner_site"], "site"
        )
    )
    resolved_edu = (
        edu_col
        if edu_col
        else find_col(
            df,
            [
                "Edu",
                "edu",
                "education",
                "parent_education",
                "highest_parent_education",
                "demo_prnt_ed_v2",
            ],
            "education",
            required=False,
        )
    )
    baseline_motion_cols = [
        "iqc_rsfmri_1_mean_motion",
        "iqc_rsfmri_2_mean_motion",
        "iqc_rsfmri_3_mean_motion",
        "iqc_rsfmri_4_mean_motion",
    ]
    if motion_cols:
        resolved_motion = [c.strip() for c in motion_cols.split(",") if c.strip()]
        missing_motion = [c for c in resolved_motion if c not in df.columns]
        if missing_motion:
            raise KeyError(f"Motion columns not found: {missing_motion}")
    elif all((c in df.columns for c in baseline_motion_cols)):
        resolved_motion = list(baseline_motion_cols)
    else:
        patterns = ["motion", "mean_fd", "meanfd", "numtrs", "num_tr", "rsfmri"]
        candidates = []
        for c in df.columns:
            low = c.lower()
            if any((pat in low for pat in patterns)):
                if c not in edge_cols and c not in [subjectkey_col, id_col]:
                    if (
                        pd.api.types.is_numeric_dtype(df[c])
                        or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8
                    ):
                        candidates.append(c)
        resolved_motion = candidates[:4]
        if len(resolved_motion) < 1:
            raise KeyError(
                f"Could not autodetect motion covariates. Please pass --motion_cols col1,col2,... Available first 120 columns={list(df.columns[:120])}"
            )
    if target_col not in df.columns:
        raise KeyError(f"ABCD csv missing target column: {target_col}")
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        bad = int((~np.isfinite(y)).sum())
        raise RuntimeError(
            f"Target contains NaN/Inf in locked manifest-selected rows: {bad}"
        )
    E = df[edge_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(E).all():
        bad = int((~np.isfinite(E)).sum())
        raise RuntimeError(
            f"ABCD edge matrix contains NaN/Inf after manifest row selection: {bad}"
        )
    cov = pd.DataFrame(index=df.index)
    cov["Age_years"] = pd.to_numeric(df[resolved_age], errors="coerce").astype(float)
    cov["Sex"] = make_numeric_series(df[resolved_sex])
    for c in resolved_motion:
        cov[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    if resolved_edu is not None:
        cov["Edu"] = make_numeric_series(df[resolved_edu])
    site = df[resolved_site].astype(str).fillna("missing")
    site_dum = pd.get_dummies(site, prefix="site", drop_first=True, dtype=float)
    cov = pd.concat([cov, site_dum], axis=1)
    cov["age_sq"] = cov["Age_years"] ** 2
    cov["age_sex"] = cov["Age_years"] * cov["Sex"]
    cov["age_sq_sex"] = cov["Age_years"] ** 2 * cov["Sex"]
    cov = cov.apply(pd.to_numeric, errors="coerce").astype(float)
    for c in cov.columns:
        if cov[c].isna().any():
            med = cov[c].median()
            if not np.isfinite(med):
                med = 0.0
            cov[c] = cov[c].fillna(med)
    if cov.isna().any().any():
        bad = cov.columns[cov.isna().any()].tolist()
        raise RuntimeError(
            f"Covariates still contain NaN after median imputation: {bad}"
        )
    C = cov.to_numpy(dtype=np.float64)
    if not np.isfinite(C).all():
        raise RuntimeError("Covariate matrix contains NaN/Inf.")
    cov_names = list(cov.columns)
    meta_cols = ["_orig_row_idx", subjectkey_col, id_col, fold_col]
    optional_cols = ["interview_date", "age", "Age_years", "Sex", "site", target_col]
    for c in optional_cols:
        if c in df.columns and c not in meta_cols:
            meta_cols.append(c)
    meta = df[meta_cols].copy()
    meta[fold_col] = manifest[fold_col].to_numpy(dtype=int)
    print(
        "[load_abcd_locked_data] selected exact ABCD rows using fold manifest _orig_row_idx"
    )
    print(f"[load_abcd_locked_data] manifest rows = {len(manifest)}")
    print(f"[load_abcd_locked_data] selected rows = {len(df)}")
    print(f"[load_abcd_locked_data] covariates = {cov_names}")
    return (meta, E, y, C, cov_names)


def build_pc_groups(pc_names: List[str]) -> Dict[str, List[str]]:
    if not pc_names:
        raise RuntimeError("Empty PC list.")
    bad = [p for p in pc_names if parse_pc_block(p) == "unknown"]
    if bad:
        raise RuntimeError(f"Some PC names do not follow '<domain>__pcXX': {bad}")
    cognition = [p for p in pc_names if parse_pc_block(p) == "cognition_core"]
    non_cognition = [p for p in pc_names if parse_pc_block(p) != "cognition_core"]
    if len(cognition) == 0:
        raise RuntimeError("No cognition_core PCs found.")
    if len(non_cognition) == 0:
        raise RuntimeError("No non-cognition PCs found.")
    return {
        "cognition_core": cognition,
        "non_cognition": non_cognition,
        "all12": list(pc_names),
    }


def parse_three_kernel_weight_grid(
    cog_grid_s: str, noncog_grid_s: str, max_total_pc_weight: float
) -> pd.DataFrame:
    cog_grid = sorted(set(parse_float_grid(cog_grid_s, name="cognition weight grid")))
    noncog_grid = sorted(
        set(parse_float_grid(noncog_grid_s, name="non-cognition weight grid"))
    )
    bad_cog = [v for v in cog_grid if v < 0.0 or v > 1.0]
    bad_non = [v for v in noncog_grid if v < 0.0 or v > 1.0]
    if bad_cog or bad_non:
        raise ValueError(
            f"Weights must be in [0, 1]. bad_cog={bad_cog}, bad_noncog={bad_non}"
        )
    if 0.0 not in cog_grid:
        raise ValueError("weight_cog_grid must include 0.0 for RSFC-only sanity check.")
    if 0.0 not in noncog_grid:
        raise ValueError(
            "weight_noncog_grid must include 0.0 for RSFC-only sanity check."
        )
    max_total = float(max_total_pc_weight)
    if max_total <= 0.0 or max_total > 1.0:
        raise ValueError("--max_total_pc_weight must be in (0, 1].")
    rows = []
    for w_cog in cog_grid:
        for w_non in noncog_grid:
            total = float(w_cog + w_non)
            if total <= max_total + 1e-12:
                rows.append(
                    {
                        "weight_cog": float(w_cog),
                        "weight_noncog": float(w_non),
                        "weight_rsfc": float(1.0 - total),
                        "total_pc_weight": float(total),
                    }
                )
    if not rows:
        raise RuntimeError(
            "Three-kernel weight grid is empty after max_total_pc_weight filtering."
        )
    grid = (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(
            ["total_pc_weight", "weight_cog", "weight_noncog"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )
    if not ((grid["weight_cog"] == 0.0) & (grid["weight_noncog"] == 0.0)).any():
        raise RuntimeError(
            "Three-kernel grid does not include RSFC-only point w_cog=0, w_noncog=0."
        )
    return grid


def make_three_base_kernels_train_test(
    E_train: np.ndarray,
    E_test: np.ndarray,
    Xcog_train: np.ndarray,
    Xcog_test: np.ndarray,
    Xnon_train: np.ndarray,
    Xnon_test: np.ndarray,
    pc_feature_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Ze_tr = unitize_rows(E_train)
    Ze_te = unitize_rows(E_test)
    K_e_train = Ze_tr @ Ze_tr.T
    K_e_test = Ze_te @ Ze_tr.T
    Zc_tr, Zc_te = transform_pc_train_test(Xcog_train, Xcog_test, pc_feature_mode)
    K_c_train = Zc_tr @ Zc_tr.T
    K_c_test = Zc_te @ Zc_tr.T
    Zn_tr, Zn_te = transform_pc_train_test(Xnon_train, Xnon_test, pc_feature_mode)
    K_n_train = Zn_tr @ Zn_tr.T
    K_n_test = Zn_te @ Zn_tr.T
    return (K_e_train, K_e_test, K_c_train, K_c_test, K_n_train, K_n_test)


def combine_three_kernels(
    K_e: np.ndarray,
    K_c: np.ndarray,
    K_n: np.ndarray,
    weight_cog: float,
    weight_noncog: float,
) -> np.ndarray:
    w_cog = float(weight_cog)
    w_non = float(weight_noncog)
    w_rsfc = 1.0 - w_cog - w_non
    if w_rsfc < -1e-10:
        raise ValueError(
            f"Invalid weights: w_rsfc={w_rsfc}, w_cog={w_cog}, w_noncog={w_non}"
        )
    return w_rsfc * K_e + w_cog * K_c + w_non * K_n


def run_inner_search_three_kernel(
    E: np.ndarray,
    Xcog: np.ndarray,
    Xnon: np.ndarray,
    y_raw: np.ndarray,
    C: np.ndarray,
    outer_train_idx: np.ndarray,
    alpha_grid: List[float],
    weight_grid_df: pd.DataFrame,
    inner_k: int,
    seed: int,
    metric: str,
    pc_feature_mode: str,
    outer_fold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    kf = KFold(n_splits=inner_k, shuffle=True, random_state=seed)
    for inner_fold, (tr_rel, va_rel) in enumerate(kf.split(outer_train_idx)):
        tr_abs = outer_train_idx[tr_rel]
        va_abs = outer_train_idx[va_rel]
        y_tr_resid, y_va_resid = residualize_train_test(
            y_raw[tr_abs], y_raw[va_abs], C[tr_abs], C[va_abs]
        )
        K_e_tr, K_e_va, K_c_tr, K_c_va, K_n_tr, K_n_va = (
            make_three_base_kernels_train_test(
                E[tr_abs],
                E[va_abs],
                Xcog[tr_abs],
                Xcog[va_abs],
                Xnon[tr_abs],
                Xnon[va_abs],
                pc_feature_mode=pc_feature_mode,
            )
        )
        for _, wr in weight_grid_df.iterrows():
            w_cog = float(wr["weight_cog"])
            w_non = float(wr["weight_noncog"])
            w_rsfc = float(wr["weight_rsfc"])
            total_pc = float(wr["total_pc_weight"])
            K_tr_w = combine_three_kernels(K_e_tr, K_c_tr, K_n_tr, w_cog, w_non)
            K_va_w = combine_three_kernels(K_e_va, K_c_va, K_n_va, w_cog, w_non)
            for alpha in alpha_grid:
                pred = fit_predict_precomputed(K_tr_w, y_tr_resid, K_va_w, alpha)
                score = score_by_metric(y_va_resid, pred, metric)
                rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(inner_fold),
                        "alpha": float(alpha),
                        "weight_rsfc": w_rsfc,
                        "weight_cog": w_cog,
                        "weight_noncog": w_non,
                        "total_pc_weight": total_pc,
                        "metric": metric,
                        "score": float(score),
                        "n_train": int(len(tr_abs)),
                        "n_valid": int(len(va_abs)),
                    }
                )
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("Inner search produced no rows.")
    summary = (
        detail.groupby(
            [
                "alpha",
                "weight_rsfc",
                "weight_cog",
                "weight_noncog",
                "total_pc_weight",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            n_inner=("score", "size"),
        )
        .sort_values(
            ["mean_score", "total_pc_weight", "weight_cog", "weight_noncog", "alpha"],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return (detail, summary)


def choose_best_three_kernel(summary: pd.DataFrame, mode: str) -> Dict[str, float]:
    df = summary.copy()
    if mode == "rsfc_only":
        df = df[
            np.isclose(df["weight_cog"], 0.0) & np.isclose(df["weight_noncog"], 0.0)
        ].copy()
    elif mode == "cognition_auxiliary":
        df = df[np.isclose(df["weight_noncog"], 0.0)].copy()
    elif mode == "noncognition_auxiliary":
        df = df[np.isclose(df["weight_cog"], 0.0)].copy()
    elif mode == "three_kernel":
        pass
    else:
        raise ValueError(f"Unknown selection mode: {mode}")
    if df.empty:
        raise RuntimeError(f"No inner-search rows available for mode={mode}")
    df = df.sort_values(
        ["mean_score", "total_pc_weight", "weight_cog", "weight_noncog", "alpha"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    r = df.iloc[0]
    return {
        "alpha": float(r["alpha"]),
        "weight_rsfc": float(r["weight_rsfc"]),
        "weight_cog": float(r["weight_cog"]),
        "weight_noncog": float(r["weight_noncog"]),
        "total_pc_weight": float(r["total_pc_weight"]),
        "inner_mean_score": float(r["mean_score"]),
        "inner_std_score": (
            float(r["std_score"]) if pd.notna(r["std_score"]) else np.nan
        ),
        "n_inner": int(r["n_inner"]),
    }


def fit_predict_three_kernel_final(
    E_train: np.ndarray,
    Xcog_train: np.ndarray,
    Xnon_train: np.ndarray,
    y_train_resid: np.ndarray,
    E_test: np.ndarray,
    Xcog_test: np.ndarray,
    Xnon_test: np.ndarray,
    alpha: float,
    weight_cog: float,
    weight_noncog: float,
    pc_feature_mode: str,
) -> np.ndarray:
    K_e_train, K_e_test, K_c_train, K_c_test, K_n_train, K_n_test = (
        make_three_base_kernels_train_test(
            E_train,
            E_test,
            Xcog_train,
            Xcog_test,
            Xnon_train,
            Xnon_test,
            pc_feature_mode=pc_feature_mode,
        )
    )
    K_train = combine_three_kernels(
        K_e_train, K_c_train, K_n_train, weight_cog, weight_noncog
    )
    K_test = combine_three_kernels(
        K_e_test, K_c_test, K_n_test, weight_cog, weight_noncog
    )
    return fit_predict_precomputed(K_train, y_train_resid, K_test, alpha)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_grid = parse_float_grid(args.alpha_grid, name="alpha grid")
    weight_grid_df = parse_three_kernel_weight_grid(
        args.weight_cog_grid, args.weight_noncog_grid, args.max_total_pc_weight
    )
    locked = load_locked_reference(
        args.locked_pred_csv, args.subjectkey_col, args.id_col, args.fold_col
    )
    pc_df, pc_names = load_pc_predictions(
        args.pc_pred_parquet, args.freeze_tsv, args.subjectkey_col, args.id_col
    )
    pc_groups = build_pc_groups(pc_names)
    meta, E, y_raw, C, cov_names = load_abcd_locked_data(
        abcd_csv=args.abcd_csv,
        locked=locked,
        fold_manifest_csv=args.fold_manifest_csv,
        subjectkey_col=args.subjectkey_col,
        id_col=args.id_col,
        fold_col=args.fold_col,
        target_col=args.target_col,
        site_col=args.site_col,
        age_col=args.age_col,
        sex_col=args.sex_col,
        edu_col=args.edu_col,
        motion_cols=args.motion_cols,
    )
    if getattr(args, "expected_n", None) is not None and len(meta) != args.expected_n:
        raise ValueError("Unexpected aligned cohort size")
    if (
        getattr(args, "expected_covariates", None) is not None
        and cov_names != args.expected_covariates
    ):
        raise ValueError("Covariate identities/order differ from the configuration")
    merged_pc = meta[[args.subjectkey_col, args.id_col]].merge(
        pc_df, on=[args.subjectkey_col, args.id_col], how="left", validate="one_to_one"
    )
    if len(merged_pc) != len(meta):
        raise RuntimeError("PC merge changed N")
    z_cols_all = [f"{p}__zpred" for p in pc_names]
    if merged_pc[z_cols_all].isna().any().any():
        raise RuntimeError("PC zpred has NaN after alignment")
    cog_cols = [f"{p}__zpred" for p in pc_groups["cognition_core"]]
    non_cols = [f"{p}__zpred" for p in pc_groups["non_cognition"]]
    Xcog = merged_pc[cog_cols].to_numpy(dtype=np.float64)
    Xnon = merged_pc[non_cols].to_numpy(dtype=np.float64)
    folds = meta[args.fold_col].to_numpy(dtype=int)
    unique_folds = sorted(np.unique(folds).tolist())
    if args.only_outer_folds:
        requested = [
            int(x.strip()) for x in args.only_outer_folds.split(",") if x.strip()
        ]
        unique_folds = [f for f in unique_folds if f in requested]
        if not unique_folds:
            raise ValueError(f"No requested folds found. requested={requested}")
    print("[PASS] inputs loaded and aligned")
    print("[INFO] N =", len(meta))
    print("[INFO] edges shape =", E.shape)
    print("[INFO] cognition PC shape =", Xcog.shape)
    print("[INFO] non-cognition PC shape =", Xnon.shape)
    print("[INFO] cov shape =", C.shape)
    print("[INFO] n_cov =", len(cov_names))
    print("[INFO] cognition PCs =", pc_groups["cognition_core"])
    print("[INFO] non-cognition PCs =", pc_groups["non_cognition"])
    print(
        "[INFO] folds =",
        {int(f): int(np.sum(folds == f)) for f in sorted(np.unique(folds).tolist())},
    )
    print("[INFO] active outer folds =", unique_folds)
    print("[INFO] alpha_grid =", alpha_grid)
    print("[INFO] weight_grid_rows =", len(weight_grid_df))
    print("[INFO] weight_grid preview =")
    print(weight_grid_df.to_string(index=False))
    print("[INFO] pc_feature_mode =", args.pc_feature_mode)
    if args.dry_run:
        print("[DRY RUN] stopping before model fitting")
        return
    pred_records = meta.copy()
    pred_records["y_raw"] = y_raw
    pred_records["y_true_resid"] = np.nan
    pred_records["y_pred_rsfc_only_w0"] = np.nan
    pred_records["y_pred_rsfc_plus_cognition_selected"] = np.nan
    pred_records["y_pred_rsfc_plus_noncognition_selected"] = np.nan
    pred_records["y_pred_three_kernel_selected"] = np.nan
    fold_metric_rows = []
    inner_detail_rows = []
    inner_summary_rows = []
    selected_rows = []
    method_to_col = {
        "rsfc_only_w0": "y_pred_rsfc_only_w0",
        "rsfc_plus_cognition_selected": "y_pred_rsfc_plus_cognition_selected",
        "rsfc_plus_noncognition_selected": "y_pred_rsfc_plus_noncognition_selected",
        "three_kernel_selected": "y_pred_three_kernel_selected",
    }
    for fold in unique_folds:
        print(f"[INFO] outer fold {fold}: inner search", flush=True)
        test_idx = np.where(folds == fold)[0]
        train_idx = np.where(folds != fold)[0]
        y_train_resid, y_test_resid = residualize_train_test(
            y_raw[train_idx], y_raw[test_idx], C[train_idx], C[test_idx]
        )
        pred_records.loc[test_idx, "y_true_resid"] = y_test_resid
        detail, summary = run_inner_search_three_kernel(
            E=E,
            Xcog=Xcog,
            Xnon=Xnon,
            y_raw=y_raw,
            C=C,
            outer_train_idx=train_idx,
            alpha_grid=alpha_grid,
            weight_grid_df=weight_grid_df,
            inner_k=args.inner_k,
            seed=args.seed + fold * 1000,
            metric=args.inner_metric,
            pc_feature_mode=args.pc_feature_mode,
            outer_fold=fold,
        )
        inner_detail_rows.append(detail)
        summary["outer_fold"] = int(fold)
        inner_summary_rows.append(summary)
        best_rsfc = choose_best_three_kernel(summary, mode="rsfc_only")
        best_cog = choose_best_three_kernel(summary, mode="cognition_auxiliary")
        best_non = choose_best_three_kernel(summary, mode="noncognition_auxiliary")
        best_three = choose_best_three_kernel(summary, mode="three_kernel")
        configs = [
            ("rsfc_only_w0", best_rsfc),
            ("rsfc_plus_cognition_selected", best_cog),
            ("rsfc_plus_noncognition_selected", best_non),
            ("three_kernel_selected", best_three),
        ]
        for method, cfg in configs:
            print(
                f"[INFO] outer fold {fold}: final {method} alpha={cfg['alpha']} w_rsfc={cfg['weight_rsfc']} w_cog={cfg['weight_cog']} w_noncog={cfg['weight_noncog']}",
                flush=True,
            )
            pred = fit_predict_three_kernel_final(
                E_train=E[train_idx],
                Xcog_train=Xcog[train_idx],
                Xnon_train=Xnon[train_idx],
                y_train_resid=y_train_resid,
                E_test=E[test_idx],
                Xcog_test=Xcog[test_idx],
                Xnon_test=Xnon[test_idx],
                alpha=cfg["alpha"],
                weight_cog=cfg["weight_cog"],
                weight_noncog=cfg["weight_noncog"],
                pc_feature_mode=args.pc_feature_mode,
            )
            pred_records.loc[test_idx, method_to_col[method]] = pred
            metrics = compute_metrics(y_test_resid, pred)
            metrics.update(
                {
                    "outer_fold": int(fold),
                    "method": method,
                    "n": int(len(test_idx)),
                    "selected_alpha": cfg["alpha"],
                    "selected_weight_rsfc": cfg["weight_rsfc"],
                    "selected_weight_cog": cfg["weight_cog"],
                    "selected_weight_noncog": cfg["weight_noncog"],
                    "selected_total_pc_weight": cfg["total_pc_weight"],
                    "inner_mean_score": cfg["inner_mean_score"],
                    "inner_std_score": cfg["inner_std_score"],
                }
            )
            fold_metric_rows.append(metrics)
            selected_rows.append(
                {
                    "outer_fold": int(fold),
                    "method": method,
                    "selected_alpha": cfg["alpha"],
                    "selected_weight_rsfc": cfg["weight_rsfc"],
                    "selected_weight_cog": cfg["weight_cog"],
                    "selected_weight_noncog": cfg["weight_noncog"],
                    "selected_total_pc_weight": cfg["total_pc_weight"],
                    "inner_mean_score": cfg["inner_mean_score"],
                    "inner_std_score": cfg["inner_std_score"],
                    "n_inner": cfg["n_inner"],
                }
            )
    active_eval_mask = np.isin(folds, np.asarray(unique_folds, dtype=int))
    pred_eval_records = pred_records.loc[active_eval_mask].copy().reset_index(drop=True)
    print("[INFO] evaluation rows =", len(pred_eval_records), flush=True)
    print(
        "[INFO] evaluation fold counts =",
        {
            int(f): int(
                np.sum(pred_eval_records[args.fold_col].to_numpy(dtype=int) == f)
            )
            for f in sorted(
                np.unique(pred_eval_records[args.fold_col].to_numpy(dtype=int)).tolist()
            )
        },
        flush=True,
    )
    required_pred_cols = ["y_true_resid"] + list(method_to_col.values())
    for c in required_pred_cols:
        if pred_eval_records[c].isna().any():
            bad = int(pred_eval_records[c].isna().sum())
            raise RuntimeError(
                f"Prediction column has missing values in active folds only: {c}, n_missing={bad}"
            )
    y = pred_eval_records["y_true_resid"].to_numpy(dtype=np.float64)
    pred_rsfc = pred_eval_records["y_pred_rsfc_only_w0"].to_numpy(dtype=np.float64)
    pooled_rows = []
    rsfc_metrics = compute_metrics(y, pred_rsfc)
    rsfc_cod = rsfc_metrics["COD"]
    rsfc_r = rsfc_metrics["Pearson_r"]
    methods = [
        ("rsfc_only_w0", "y_pred_rsfc_only_w0"),
        ("rsfc_plus_cognition_selected", "y_pred_rsfc_plus_cognition_selected"),
        ("rsfc_plus_noncognition_selected", "y_pred_rsfc_plus_noncognition_selected"),
        ("three_kernel_selected", "y_pred_three_kernel_selected"),
    ]
    for method, col in methods:
        pred = pred_eval_records[col].to_numpy(dtype=np.float64)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"Prediction column has NaN/Inf: {col}")
        m = compute_metrics(y, pred)
        m.update(
            {
                "method": method,
                "n": int(len(pred_eval_records)),
                "delta_COD_vs_rsfc": float(m["COD"] - rsfc_cod),
                "delta_Pearson_r_vs_rsfc": float(m["Pearson_r"] - rsfc_r),
                "percent_COD_improvement_vs_rsfc": (
                    float(100.0 * (m["COD"] - rsfc_cod) / rsfc_cod)
                    if rsfc_cod != 0
                    else np.nan
                ),
            }
        )
        if method != "rsfc_only_w0" and args.n_boot > 0:
            m.update(bootstrap_delta(y, pred_rsfc, pred, args.n_boot, args.seed + 9876))
        pooled_rows.append(m)
    pred_csv = out_dir / "predictions_concat_phase6b_three_kernel_krr_v1.csv"
    pooled_csv = out_dir / "pooled_metrics_phase6b_three_kernel_krr_v1.csv"
    fold_csv = out_dir / "fold_metrics_phase6b_three_kernel_krr_v1.csv"
    inner_detail_csv = out_dir / "inner_search_detail_phase6b_three_kernel_krr_v1.csv"
    inner_summary_csv = out_dir / "inner_search_summary_phase6b_three_kernel_krr_v1.csv"
    selected_csv = out_dir / "selected_hyperparams_phase6b_three_kernel_krr_v1.csv"
    weight_grid_csv = out_dir / "weight_grid_phase6b_three_kernel_krr_v1.csv"
    pred_eval_records.to_csv(pred_csv, index=False)
    pooled_df = pd.DataFrame(pooled_rows)
    fold_df = pd.DataFrame(fold_metric_rows)
    inner_detail_df = pd.concat(inner_detail_rows, axis=0, ignore_index=True)
    inner_summary_df = pd.concat(inner_summary_rows, axis=0, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    pooled_df.to_csv(pooled_csv, index=False)
    fold_df.to_csv(fold_csv, index=False)
    inner_detail_df.to_csv(inner_detail_csv, index=False)
    inner_summary_df.to_csv(inner_summary_csv, index=False)
    selected_df.to_csv(selected_csv, index=False)
    weight_grid_df.to_csv(weight_grid_csv, index=False)
    manifest = {
        "script": "three_kernel_engine.py",
        "abcd_csv": args.abcd_csv,
        "locked_pred_csv": args.locked_pred_csv,
        "fold_manifest_csv": args.fold_manifest_csv,
        "pc_pred_parquet": args.pc_pred_parquet,
        "freeze_tsv": args.freeze_tsv,
        "out_dir": str(out_dir),
        "n_total": int(len(meta)),
        "n_eval": int(len(pred_eval_records)),
        "fold_counts": {
            str(f): int(np.sum(folds == f)) for f in sorted(np.unique(folds).tolist())
        },
        "eval_fold_counts": {
            str(int(f)): int(
                np.sum(pred_eval_records[args.fold_col].to_numpy(dtype=int) == f)
            )
            for f in sorted(
                np.unique(pred_eval_records[args.fold_col].to_numpy(dtype=int)).tolist()
            )
        },
        "active_outer_folds": [int(f) for f in unique_folds],
        "cov_names": cov_names,
        "pc_groups": pc_groups,
        "pc_feature_mode": args.pc_feature_mode,
        "alpha_grid": alpha_grid,
        "weight_cog_grid": sorted(
            set(parse_float_grid(args.weight_cog_grid, name="cognition weight grid"))
        ),
        "weight_noncog_grid": sorted(
            set(
                parse_float_grid(
                    args.weight_noncog_grid, name="non-cognition weight grid"
                )
            )
        ),
        "max_total_pc_weight": float(args.max_total_pc_weight),
        "n_weight_grid_rows": int(len(weight_grid_df)),
        "inner_k": int(args.inner_k),
        "inner_metric": args.inner_metric,
        "n_boot": int(args.n_boot),
        "outputs": [
            pred_csv.name,
            pooled_csv.name,
            fold_csv.name,
            inner_detail_csv.name,
            inner_summary_csv.name,
            selected_csv.name,
            weight_grid_csv.name,
            "phase6b_three_kernel_krr_manifest.json",
        ],
    }
    with open(
        out_dir / "phase6b_three_kernel_krr_manifest.json", "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("[done] Phase 6B three-kernel KRR completed")
    print("[done] output dir =", out_dir)
    print("[done] pooled results:")
    print(
        pooled_df[
            [
                "method",
                "COD",
                "Pearson_r",
                "delta_COD_vs_rsfc",
                "percent_COD_improvement_vs_rsfc",
            ]
        ].to_string(index=False)
    )
    print("[done] selected weights for three_kernel_selected:")
    print(
        selected_df[selected_df["method"] == "three_kernel_selected"][
            [
                "outer_fold",
                "selected_alpha",
                "selected_weight_rsfc",
                "selected_weight_cog",
                "selected_weight_noncog",
                "selected_total_pc_weight",
                "inner_mean_score",
            ]
        ].to_string(index=False)
    )
