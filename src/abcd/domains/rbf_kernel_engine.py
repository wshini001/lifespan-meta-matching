"""Fit ABCD KRR models combining an RSFC kernel and a phenotype-PC RBF kernel."""

from domain_common import (
    bootstrap_delta,
    combine_kernels,
    compute_metrics,
    find_col,
    fit_predict_precomputed,
    load_locked_reference,
    load_pc_predictions,
    make_numeric_series,
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


def parse_phase6c_search_grid(
    weight_grid_s: str, gamma_scale_grid_s: str, include_pc_only: bool = True
) -> pd.DataFrame:
    weights = sorted(set(parse_float_grid(weight_grid_s, name="weight grid")))
    gamma_scales = sorted(
        set(parse_float_grid(gamma_scale_grid_s, name="gamma scale grid"))
    )
    bad_w = [w for w in weights if w < 0.0 or w >= 1.0]
    if bad_w:
        raise ValueError(
            f"Phase 6C --weight_grid is for selected multi-kernel search and must contain values in [0, 1). Bad values: {bad_w}. PC-RBF-only is added separately."
        )
    if 0.0 not in weights:
        raise ValueError(
            "Phase 6C --weight_grid must include 0.0 for the internal RSFC-only sanity check."
        )
    bad_g = [g for g in gamma_scales if g <= 0.0]
    if bad_g:
        raise ValueError(f"Gamma scales must be positive. Bad values: {bad_g}")
    rows = []
    for w in weights:
        if np.isclose(w, 0.0):
            rows.append(
                {
                    "candidate_role": "rsfc_only",
                    "weight_pc": 0.0,
                    "weight_rsfc": 1.0,
                    "gamma_scale": np.nan,
                    "gamma_scale_label": "NA",
                }
            )
        else:
            for gs in gamma_scales:
                rows.append(
                    {
                        "candidate_role": "rbf_multikernel",
                        "weight_pc": float(w),
                        "weight_rsfc": float(1.0 - w),
                        "gamma_scale": float(gs),
                        "gamma_scale_label": f"{float(gs):.8g}",
                    }
                )
    if include_pc_only:
        for gs in gamma_scales:
            rows.append(
                {
                    "candidate_role": "pc_rbf_only",
                    "weight_pc": 1.0,
                    "weight_rsfc": 0.0,
                    "gamma_scale": float(gs),
                    "gamma_scale_label": f"{float(gs):.8g}",
                }
            )
    grid = pd.DataFrame(rows)
    grid = grid.drop_duplicates(
        subset=["candidate_role", "weight_pc", "gamma_scale_label"]
    ).reset_index(drop=True)
    return grid


def squared_euclidean_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    D = A2 + B2 - 2.0 * (A @ B.T)
    np.maximum(D, 0.0, out=D)
    return D


def estimate_rbf_gamma_base(
    X_train: np.ndarray, max_samples: int, seed: int
) -> Tuple[float, float, int]:
    X_train = np.asarray(X_train, dtype=np.float64)
    n = X_train.shape[0]
    if n < 2:
        raise RuntimeError(
            "Cannot estimate RBF gamma from fewer than 2 training samples."
        )
    m = int(min(max(2, max_samples), n))
    if m < n:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=m, replace=False)
        X = X_train[idx]
    else:
        X = X_train
    D = squared_euclidean_distances(X, X)
    tri = D[np.triu_indices(D.shape[0], k=1)]
    tri = tri[np.isfinite(tri) & (tri > 0.0)]
    if tri.size == 0:
        median_sqdist = 1.0
    else:
        median_sqdist = float(np.median(tri))
        if not np.isfinite(median_sqdist) or median_sqdist <= 0.0:
            median_sqdist = 1.0
    gamma_base = 1.0 / median_sqdist
    return (float(gamma_base), float(median_sqdist), int(m))


def make_rsfc_kernels_train_test(
    E_train: np.ndarray, E_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    Ze_tr = unitize_rows(E_train)
    Ze_te = unitize_rows(E_test)
    K_e_train = Ze_tr @ Ze_tr.T
    K_e_test = Ze_te @ Ze_tr.T
    return (K_e_train, K_e_test)


def make_pc_rbf_kernels_from_scaled(
    X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, gamma: float
) -> Tuple[np.ndarray, np.ndarray]:
    D_train = squared_euclidean_distances(X_train_scaled, X_train_scaled)
    D_test = squared_euclidean_distances(X_test_scaled, X_train_scaled)
    K_train = np.exp(-float(gamma) * D_train)
    K_test = np.exp(-float(gamma) * D_test)
    return (K_train, K_test)


def run_inner_search_pcrbf(
    E: np.ndarray,
    Xpc: np.ndarray,
    y_raw: np.ndarray,
    C: np.ndarray,
    outer_train_idx: np.ndarray,
    alpha_grid: List[float],
    search_grid: pd.DataFrame,
    inner_k: int,
    seed: int,
    metric: str,
    pc_rbf_feature_mode: str,
    gamma_median_max_samples: int,
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
        K_e_tr, K_e_va = make_rsfc_kernels_train_test(E[tr_abs], E[va_abs])
        Xp_tr, Xp_va = transform_pc_train_test(
            Xpc[tr_abs], Xpc[va_abs], pc_rbf_feature_mode
        )
        gamma_base, median_sqdist, n_gamma_sample = estimate_rbf_gamma_base(
            Xp_tr,
            max_samples=gamma_median_max_samples,
            seed=int(seed + 10000 * (outer_fold + 1) + inner_fold),
        )
        rsfc_rows = search_grid[search_grid["candidate_role"] == "rsfc_only"]
        if len(rsfc_rows) != 1:
            raise RuntimeError(
                f"Expected exactly one RSFC-only search-grid row, found {len(rsfc_rows)}"
            )
        for alpha in alpha_grid:
            pred = fit_predict_precomputed(K_e_tr, y_tr_resid, K_e_va, alpha)
            score = score_by_metric(y_va_resid, pred, metric)
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "alpha": float(alpha),
                    "candidate_role": "rsfc_only",
                    "weight_pc": 0.0,
                    "weight_rsfc": 1.0,
                    "gamma_scale": np.nan,
                    "gamma_scale_label": "NA",
                    "gamma": np.nan,
                    "gamma_base": float(gamma_base),
                    "median_sqdist": float(median_sqdist),
                    "n_gamma_sample": int(n_gamma_sample),
                    "metric": metric,
                    "score": float(score),
                    "n_train": int(len(tr_abs)),
                    "n_valid": int(len(va_abs)),
                }
            )
        gamma_rows = search_grid[search_grid["candidate_role"] != "rsfc_only"].copy()
        for gamma_label, sub_grid in gamma_rows.groupby(
            "gamma_scale_label", sort=False
        ):
            gamma_scale = float(sub_grid["gamma_scale"].iloc[0])
            gamma = float(gamma_scale * gamma_base)
            K_p_tr, K_p_va = make_pc_rbf_kernels_from_scaled(Xp_tr, Xp_va, gamma)
            for _, gr in sub_grid.iterrows():
                weight_pc = float(gr["weight_pc"])
                role = str(gr["candidate_role"])
                K_tr_w = combine_kernels(K_e_tr, K_p_tr, weight_pc)
                K_va_w = combine_kernels(K_e_va, K_p_va, weight_pc)
                for alpha in alpha_grid:
                    pred = fit_predict_precomputed(K_tr_w, y_tr_resid, K_va_w, alpha)
                    score = score_by_metric(y_va_resid, pred, metric)
                    rows.append(
                        {
                            "outer_fold": int(outer_fold),
                            "inner_fold": int(inner_fold),
                            "alpha": float(alpha),
                            "candidate_role": role,
                            "weight_pc": weight_pc,
                            "weight_rsfc": float(1.0 - weight_pc),
                            "gamma_scale": gamma_scale,
                            "gamma_scale_label": str(gamma_label),
                            "gamma": gamma,
                            "gamma_base": float(gamma_base),
                            "median_sqdist": float(median_sqdist),
                            "n_gamma_sample": int(n_gamma_sample),
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
                "candidate_role",
                "weight_pc",
                "weight_rsfc",
                "gamma_scale_label",
                "metric",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            gamma_scale=("gamma_scale", "median"),
            gamma=("gamma", "median"),
            gamma_base=("gamma_base", "mean"),
            median_sqdist=("median_sqdist", "mean"),
            n_gamma_sample=("n_gamma_sample", "median"),
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            n_inner=("score", "size"),
        )
        .sort_values(
            ["mean_score", "candidate_role", "weight_pc", "gamma_scale", "alpha"],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return (detail, summary)


def choose_best_from_summary(summary: pd.DataFrame, mode: str) -> Dict[str, float]:
    df = summary.copy()
    if mode == "rsfc_only":
        df = df[df["candidate_role"] == "rsfc_only"].copy()
    elif mode == "pc_rbf_only":
        df = df[df["candidate_role"] == "pc_rbf_only"].copy()
    elif mode == "rbf_multikernel_selected":
        df = df[df["candidate_role"].isin(["rsfc_only", "rbf_multikernel"])].copy()
    else:
        raise ValueError(f"Unknown choose mode: {mode}")
    if df.empty:
        raise RuntimeError(f"No inner-search rows found for mode={mode}")
    df = df.sort_values(
        ["mean_score", "weight_pc", "gamma_scale", "alpha"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    r = df.iloc[0]
    return {
        "alpha": float(r["alpha"]),
        "candidate_role": str(r["candidate_role"]),
        "weight_pc": float(r["weight_pc"]),
        "weight_rsfc": float(r["weight_rsfc"]),
        "gamma_scale": (
            float(r["gamma_scale"]) if pd.notna(r["gamma_scale"]) else np.nan
        ),
        "gamma_scale_label": str(r["gamma_scale_label"]),
        "gamma": float(r["gamma"]) if pd.notna(r["gamma"]) else np.nan,
        "gamma_base": float(r["gamma_base"]) if pd.notna(r["gamma_base"]) else np.nan,
        "median_sqdist": (
            float(r["median_sqdist"]) if pd.notna(r["median_sqdist"]) else np.nan
        ),
        "n_gamma_sample": (
            int(r["n_gamma_sample"]) if pd.notna(r["n_gamma_sample"]) else 0
        ),
        "inner_mean_score": float(r["mean_score"]),
        "inner_std_score": (
            float(r["std_score"]) if pd.notna(r["std_score"]) else np.nan
        ),
        "n_inner": int(r["n_inner"]),
    }


def fit_predict_pcrbf_final(
    E_train: np.ndarray,
    Xpc_train: np.ndarray,
    y_train_resid: np.ndarray,
    E_test: np.ndarray,
    Xpc_test: np.ndarray,
    alpha: float,
    weight_pc: float,
    gamma_scale: float,
    pc_rbf_feature_mode: str,
    gamma_median_max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    K_e_train, K_e_test = make_rsfc_kernels_train_test(E_train, E_test)
    w = float(weight_pc)
    if np.isclose(w, 0.0):
        pred = fit_predict_precomputed(K_e_train, y_train_resid, K_e_test, alpha)
        info = {
            "gamma_scale": np.nan,
            "gamma": np.nan,
            "gamma_base": np.nan,
            "median_sqdist": np.nan,
            "n_gamma_sample": 0,
        }
        return (pred, info)
    Xp_tr, Xp_te = transform_pc_train_test(Xpc_train, Xpc_test, pc_rbf_feature_mode)
    gamma_base, median_sqdist, n_gamma_sample = estimate_rbf_gamma_base(
        Xp_tr, max_samples=gamma_median_max_samples, seed=seed
    )
    gamma = float(gamma_scale * gamma_base)
    K_p_train, K_p_test = make_pc_rbf_kernels_from_scaled(Xp_tr, Xp_te, gamma)
    K_train = combine_kernels(K_e_train, K_p_train, w)
    K_test = combine_kernels(K_e_test, K_p_test, w)
    pred = fit_predict_precomputed(K_train, y_train_resid, K_test, alpha)
    info = {
        "gamma_scale": float(gamma_scale),
        "gamma": float(gamma),
        "gamma_base": float(gamma_base),
        "median_sqdist": float(median_sqdist),
        "n_gamma_sample": int(n_gamma_sample),
    }
    return (pred, info)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_grid = parse_float_grid(args.alpha_grid, name="alpha grid")
    search_grid = parse_phase6c_search_grid(
        args.weight_grid, args.gamma_scale_grid, include_pc_only=True
    )
    locked = load_locked_reference(
        args.locked_pred_csv, args.subjectkey_col, args.id_col, args.fold_col
    )
    pc_df, pc_names = load_pc_predictions(
        args.pc_pred_parquet, args.freeze_tsv, args.subjectkey_col, args.id_col
    )
    if args.pc_subset != "all12":
        raise ValueError("Phase 6C currently supports --pc_subset all12 only.")
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
    Xpc_all = merged_pc[z_cols_all].to_numpy(dtype=np.float64)
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
    print("[INFO] PC shape =", Xpc_all.shape)
    print("[INFO] cov shape =", C.shape)
    print("[INFO] n_cov =", len(cov_names))
    print("[INFO] n_pc_all =", len(pc_names))
    print("[INFO] pc_names =", pc_names)
    print(
        "[INFO] folds =",
        {int(f): int(np.sum(folds == f)) for f in sorted(np.unique(folds).tolist())},
    )
    print("[INFO] active outer folds =", unique_folds)
    print("[INFO] alpha_grid =", alpha_grid)
    print("[INFO] search_grid rows =", len(search_grid))
    print(search_grid.to_string(index=False))
    print("[INFO] pc_rbf_feature_mode =", args.pc_rbf_feature_mode)
    print("[INFO] gamma_median_max_samples =", args.gamma_median_max_samples)
    if args.dry_run:
        print("[DRY RUN] stopping before model fitting")
        return
    pred_records = meta.copy()
    pred_records["y_raw"] = y_raw
    pred_records["y_true_resid"] = np.nan
    pred_records["y_pred_rsfc_only_w0"] = np.nan
    pred_records["y_pred_pc_rbf_only_w1"] = np.nan
    pred_records["y_pred_rbf_multikernel_selected"] = np.nan
    fold_metric_rows = []
    inner_detail_rows = []
    inner_summary_rows = []
    selected_rows = []
    method_to_col = {
        "rsfc_only_w0": "y_pred_rsfc_only_w0",
        "pc_rbf_only_w1": "y_pred_pc_rbf_only_w1",
        "rbf_multikernel_selected": "y_pred_rbf_multikernel_selected",
    }
    for fold in unique_folds:
        print(f"[INFO] outer fold {fold}: inner search", flush=True)
        test_idx = np.where(folds == fold)[0]
        train_idx = np.where(folds != fold)[0]
        y_train_resid, y_test_resid = residualize_train_test(
            y_raw[train_idx], y_raw[test_idx], C[train_idx], C[test_idx]
        )
        pred_records.loc[test_idx, "y_true_resid"] = y_test_resid
        detail, summary = run_inner_search_pcrbf(
            E=E,
            Xpc=Xpc_all,
            y_raw=y_raw,
            C=C,
            outer_train_idx=train_idx,
            alpha_grid=alpha_grid,
            search_grid=search_grid,
            inner_k=args.inner_k,
            seed=args.seed + fold * 1000,
            metric=args.inner_metric,
            pc_rbf_feature_mode=args.pc_rbf_feature_mode,
            gamma_median_max_samples=args.gamma_median_max_samples,
            outer_fold=fold,
        )
        inner_detail_rows.append(detail)
        summary["outer_fold"] = int(fold)
        inner_summary_rows.append(summary)
        best_rsfc = choose_best_from_summary(summary, mode="rsfc_only")
        best_pc_rbf = choose_best_from_summary(summary, mode="pc_rbf_only")
        best_multi = choose_best_from_summary(summary, mode="rbf_multikernel_selected")
        configs = [
            ("rsfc_only_w0", best_rsfc),
            ("pc_rbf_only_w1", best_pc_rbf),
            ("rbf_multikernel_selected", best_multi),
        ]
        for method, cfg in configs:
            print(
                f"[INFO] outer fold {fold}: final {method} alpha={cfg['alpha']} w_pc={cfg['weight_pc']} gamma_scale={cfg['gamma_scale_label']}",
                flush=True,
            )
            pred, final_info = fit_predict_pcrbf_final(
                E_train=E[train_idx],
                Xpc_train=Xpc_all[train_idx],
                y_train_resid=y_train_resid,
                E_test=E[test_idx],
                Xpc_test=Xpc_all[test_idx],
                alpha=cfg["alpha"],
                weight_pc=cfg["weight_pc"],
                gamma_scale=cfg["gamma_scale"] if pd.notna(cfg["gamma_scale"]) else 1.0,
                pc_rbf_feature_mode=args.pc_rbf_feature_mode,
                gamma_median_max_samples=args.gamma_median_max_samples,
                seed=args.seed + 20000 + fold,
            )
            pred_records.loc[test_idx, method_to_col[method]] = pred
            metrics = compute_metrics(y_test_resid, pred)
            metrics.update(
                {
                    "outer_fold": int(fold),
                    "method": method,
                    "n": int(len(test_idx)),
                    "selected_alpha": cfg["alpha"],
                    "selected_candidate_role": cfg["candidate_role"],
                    "selected_weight_pc": cfg["weight_pc"],
                    "selected_weight_rsfc": cfg["weight_rsfc"],
                    "selected_gamma_scale": final_info["gamma_scale"],
                    "selected_gamma": final_info["gamma"],
                    "selected_gamma_base": final_info["gamma_base"],
                    "selected_median_sqdist": final_info["median_sqdist"],
                    "selected_n_gamma_sample": final_info["n_gamma_sample"],
                    "inner_gamma_scale": cfg["gamma_scale"],
                    "inner_gamma": cfg["gamma"],
                    "inner_gamma_base": cfg["gamma_base"],
                    "inner_median_sqdist": cfg["median_sqdist"],
                    "inner_n_gamma_sample": cfg["n_gamma_sample"],
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
                    "selected_candidate_role": cfg["candidate_role"],
                    "selected_weight_pc": cfg["weight_pc"],
                    "selected_weight_rsfc": cfg["weight_rsfc"],
                    "selected_gamma_scale": final_info["gamma_scale"],
                    "selected_gamma": final_info["gamma"],
                    "selected_gamma_base": final_info["gamma_base"],
                    "selected_median_sqdist": final_info["median_sqdist"],
                    "selected_n_gamma_sample": final_info["n_gamma_sample"],
                    "inner_gamma_scale": cfg["gamma_scale"],
                    "inner_gamma": cfg["gamma"],
                    "inner_gamma_base": cfg["gamma_base"],
                    "inner_median_sqdist": cfg["median_sqdist"],
                    "inner_n_gamma_sample": cfg["n_gamma_sample"],
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
        ("pc_rbf_only_w1", "y_pred_pc_rbf_only_w1"),
        ("rbf_multikernel_selected", "y_pred_rbf_multikernel_selected"),
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
    pred_csv = out_dir / "predictions_concat_phase6c_pcrbf_multikernel_krr_v1.csv"
    pooled_csv = out_dir / "pooled_metrics_phase6c_pcrbf_multikernel_krr_v1.csv"
    fold_csv = out_dir / "fold_metrics_phase6c_pcrbf_multikernel_krr_v1.csv"
    inner_detail_csv = (
        out_dir / "inner_search_detail_phase6c_pcrbf_multikernel_krr_v1.csv"
    )
    inner_summary_csv = (
        out_dir / "inner_search_summary_phase6c_pcrbf_multikernel_krr_v1.csv"
    )
    selected_csv = out_dir / "selected_hyperparams_phase6c_pcrbf_multikernel_krr_v1.csv"
    search_grid_csv = out_dir / "search_grid_phase6c_pcrbf_multikernel_krr_v1.csv"
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
    search_grid.to_csv(search_grid_csv, index=False)
    manifest = {
        "script": "rbf_kernel_engine.py",
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
        "pc_names": pc_names,
        "pc_subset": args.pc_subset,
        "pc_rbf_feature_mode": args.pc_rbf_feature_mode,
        "alpha_grid": alpha_grid,
        "weight_grid": parse_float_grid(args.weight_grid, name="weight grid"),
        "gamma_scale_grid": parse_float_grid(
            args.gamma_scale_grid, name="gamma scale grid"
        ),
        "gamma_median_max_samples": int(args.gamma_median_max_samples),
        "n_search_grid_rows": int(len(search_grid)),
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
            search_grid_csv.name,
            "phase6c_pcrbf_multikernel_krr_manifest.json",
        ],
    }
    with open(
        out_dir / "phase6c_pcrbf_multikernel_krr_manifest.json", "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("[done] Phase 6C PC-RBF multi-kernel KRR completed")
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
    print("[done] selected RBF multi-kernel settings:")
    print(
        selected_df[selected_df["method"] == "rbf_multikernel_selected"][
            [
                "outer_fold",
                "selected_alpha",
                "selected_weight_pc",
                "selected_gamma_scale",
                "selected_gamma",
                "inner_mean_score",
            ]
        ].to_string(index=False)
    )
