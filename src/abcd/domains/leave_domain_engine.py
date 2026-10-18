"""Fit ABCD prediction models after removing one phenotype-PC domain at a time."""

from domain_common import (
    apply_ensemble,
    bootstrap_delta,
    compute_metrics,
    find_col,
    fit_ensemble_from_inner_oof,
    fit_predict_abcd,
    fit_predict_pc,
    generate_inner_oof_abcd,
    generate_inner_oof_pc,
    load_locked_reference,
    load_pc_predictions,
    make_numeric_series,
    parse_pc_block,
    residualize_train_test,
)
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def parse_float_grid(s: str) -> List[float]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if not vals:
        raise ValueError("Empty alpha grid")
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
            f"ABCD csv not found: {abcd_path}. Please rerun with --abcd_csv pointing to your actual ABCD_final.csv."
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


def build_subsets(pc_names: List[str]) -> Dict[str, List[str]]:
    if not pc_names:
        raise RuntimeError("Empty PC list.")
    domains = sorted({parse_pc_block(p) for p in pc_names})
    if "unknown" in domains:
        bad = [p for p in pc_names if parse_pc_block(p) == "unknown"]
        raise RuntimeError(f"Some PC names do not follow '<domain>__pcXX': {bad}")
    subsets: Dict[str, List[str]] = {"all12": list(pc_names)}
    for block in domains:
        removed = [p for p in pc_names if parse_pc_block(p) == block]
        keep = [p for p in pc_names if parse_pc_block(p) != block]
        if not removed:
            raise RuntimeError(
                f"No PCs removed for leave-one-domain-out block: {block}"
            )
        if len(keep) == len(pc_names):
            raise RuntimeError(
                f"Leave-one-domain-out failed to remove PCs for block: {block}"
            )
        if len(keep) == 0:
            raise RuntimeError(
                f"Leave-one-domain-out removed all PCs for block: {block}"
            )
        subsets[f"without_{block}"] = keep
    return subsets


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_abcd = parse_float_grid(args.abcd_alpha_grid)
    alpha_pc = parse_float_grid(args.pc_alpha_grid)
    locked = load_locked_reference(
        args.locked_pred_csv, args.subjectkey_col, args.id_col, args.fold_col
    )
    pc_df, pc_names = load_pc_predictions(
        args.pc_pred_parquet, args.freeze_tsv, args.subjectkey_col, args.id_col
    )
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
    Xpc_all_df = merged_pc[z_cols_all].copy()
    folds = meta[args.fold_col].to_numpy(dtype=int)
    unique_folds = sorted(np.unique(folds).tolist())
    subsets = build_subsets(pc_names)
    if args.only_subsets:
        requested = [s.strip() for s in args.only_subsets.split(",") if s.strip()]
        subsets = {k: subsets[k] for k in requested if k in subsets}
        missing = sorted(set(requested) - set(subsets))
        if missing:
            raise ValueError(
                f"Requested subset names not found: {missing}. Available={sorted(build_subsets(pc_names).keys())}"
            )
    print("[PASS] inputs loaded and aligned")
    print("[INFO] N =", len(meta))
    print("[INFO] edges shape =", E.shape)
    print("[INFO] cov shape =", C.shape)
    print("[INFO] n_cov =", len(cov_names))
    print("[INFO] n_pc_all =", len(pc_names))
    print("[INFO] folds =", {int(f): int(np.sum(folds == f)) for f in unique_folds})
    print("[INFO] subsets =", {k: v for k, v in subsets.items()})
    if args.dry_run:
        print("[DRY RUN] stopping before model fitting")
        return
    n = len(meta)
    pred_records = meta.copy()
    pred_records["y_raw"] = y_raw
    pred_records["y_true_resid"] = np.nan
    pred_records["y_pred_abcd_nested"] = np.nan
    fold_metric_rows = []
    pooled_rows = []
    weight_rows = []
    alpha_rows = []
    for subset_name in subsets:
        pred_records[f"y_pred_pc_{subset_name}"] = np.nan
        pred_records[f"y_pred_avg_{subset_name}"] = np.nan
        pred_records[f"y_pred_ensemble_{subset_name}"] = np.nan
    for fold in unique_folds:
        print(f"[INFO] outer fold {fold}", flush=True)
        test_idx = np.where(folds == fold)[0]
        train_idx = np.where(folds != fold)[0]
        y_train_resid, y_test_resid = residualize_train_test(
            y_raw[train_idx], y_raw[test_idx], C[train_idx], C[test_idx]
        )
        pred_records.loc[test_idx, "y_true_resid"] = y_test_resid
        print(f"[INFO] outer fold {fold}: ABCD inner OOF", flush=True)
        abcd_oof, y_oof, ar = generate_inner_oof_abcd(
            E=E,
            y_raw=y_raw,
            C=C,
            outer_train_idx=train_idx,
            inner_k=args.inner_k,
            alpha_grid=alpha_abcd,
            seed=args.seed + fold * 1000,
            metric=args.inner_metric,
        )
        for r in ar:
            r.update({"outer_fold": fold, "subset": "shared"})
            alpha_rows.append(r)
        print(f"[INFO] outer fold {fold}: ABCD final train/test", flush=True)
        abcd_test, abcd_alpha, abcd_score = fit_predict_abcd(
            E[train_idx],
            y_train_resid,
            E[test_idx],
            alpha_abcd,
            args.inner_k,
            args.seed + 10000 + fold,
            args.inner_metric,
        )
        pred_records.loc[test_idx, "y_pred_abcd_nested"] = abcd_test
        alpha_rows.append(
            {
                "outer_fold": fold,
                "inner_fold": -1,
                "stream": "abcd_final",
                "subset": "shared",
                "alpha": abcd_alpha,
                "inner_score": abcd_score,
            }
        )
        for subset_name, subset_pcs in subsets.items():
            print(
                f"[INFO] outer fold {fold}: subset {subset_name} n_pc={len(subset_pcs)}",
                flush=True,
            )
            subset_cols = [f"{p}__zpred" for p in subset_pcs]
            Xpc = Xpc_all_df[subset_cols].to_numpy(dtype=np.float64)
            pc_oof, y_pc_oof, pr = generate_inner_oof_pc(
                Xpc=Xpc,
                y_raw=y_raw,
                C=C,
                outer_train_idx=train_idx,
                inner_k=args.inner_k,
                alpha_grid=alpha_pc,
                seed=args.seed + fold * 1000,
                metric=args.inner_metric,
                pc_feature_mode=args.pc_feature_mode,
                subset_name=subset_name,
            )
            for r in pr:
                r.update({"outer_fold": fold})
                alpha_rows.append(r)
            if np.max(np.abs(y_pc_oof - y_oof)) > 1e-06:
                print(
                    f"[WARN] y_oof mismatch between ABCD and PC inner residuals for subset {subset_name}, fold {fold}"
                )
            ens = fit_ensemble_from_inner_oof(abcd_oof, pc_oof, y_oof)
            pc_test, pc_alpha, pc_score = fit_predict_pc(
                Xpc[train_idx],
                y_train_resid,
                Xpc[test_idx],
                alpha_pc,
                args.inner_k,
                args.seed + 20000 + fold,
                args.inner_metric,
                args.pc_feature_mode,
            )
            avg_test = 0.5 * (abcd_test + pc_test)
            ens_test = apply_ensemble(ens, abcd_test, pc_test)
            pred_records.loc[test_idx, f"y_pred_pc_{subset_name}"] = pc_test
            pred_records.loc[test_idx, f"y_pred_avg_{subset_name}"] = avg_test
            pred_records.loc[test_idx, f"y_pred_ensemble_{subset_name}"] = ens_test
            weight_rows.append(
                {
                    "outer_fold": fold,
                    "subset": subset_name,
                    "n_pcs": len(subset_pcs),
                    "pc_names": "|".join(subset_pcs),
                    "coef_abcd_standardized": ens["coef_abcd_standardized"],
                    "coef_pc_standardized": ens["coef_pc_standardized"],
                    "intercept": ens["intercept"],
                    "mu_abcd": float(ens["mu"][0]),
                    "mu_pc": float(ens["mu"][1]),
                    "sd_abcd": float(ens["sd"][0]),
                    "sd_pc": float(ens["sd"][1]),
                    "pc_final_alpha": pc_alpha,
                    "pc_final_inner_score": pc_score,
                }
            )
            alpha_rows.append(
                {
                    "outer_fold": fold,
                    "inner_fold": -1,
                    "stream": "pc_final",
                    "subset": subset_name,
                    "alpha": pc_alpha,
                    "inner_score": pc_score,
                }
            )
            for method, pred in [
                ("abcd", abcd_test),
                ("pc", pc_test),
                ("avg", avg_test),
                ("ensemble", ens_test),
            ]:
                m = compute_metrics(y_test_resid, pred)
                m.update(
                    {
                        "fold": fold,
                        "subset": subset_name,
                        "method": method,
                        "n": int(len(test_idx)),
                    }
                )
                fold_metric_rows.append(m)
    if (
        pred_records["y_true_resid"].isna().any()
        or pred_records["y_pred_abcd_nested"].isna().any()
    ):
        raise RuntimeError("Missing pooled y or ABCD predictions")
    y = pred_records["y_true_resid"].to_numpy(dtype=np.float64)
    abcd_pred = pred_records["y_pred_abcd_nested"].to_numpy(dtype=np.float64)
    abcd_m = compute_metrics(y, abcd_pred)
    for subset_name in subsets:
        for method, col in [
            ("abcd", "y_pred_abcd_nested"),
            ("pc", f"y_pred_pc_{subset_name}"),
            ("avg", f"y_pred_avg_{subset_name}"),
            ("ensemble", f"y_pred_ensemble_{subset_name}"),
        ]:
            pred = pred_records[col].to_numpy(dtype=np.float64)
            if not np.isfinite(pred).all():
                raise RuntimeError(f"Prediction column has NaN/Inf: {col}")
            m = compute_metrics(y, pred)
            m.update(
                {
                    "subset": subset_name,
                    "method": method,
                    "n": int(n),
                    "delta_COD_vs_abcd": float(m["COD"] - abcd_m["COD"]),
                    "delta_Pearson_r_vs_abcd": float(
                        m["Pearson_r"] - abcd_m["Pearson_r"]
                    ),
                    "percent_COD_improvement_vs_abcd": (
                        float(100.0 * (m["COD"] - abcd_m["COD"]) / abcd_m["COD"])
                        if abcd_m["COD"] != 0
                        else np.nan
                    ),
                    "n_pcs": len(subsets[subset_name]),
                    "pc_names": "|".join(subsets[subset_name]),
                }
            )
            if method == "ensemble" and args.n_boot > 0:
                m.update(
                    bootstrap_delta(y, abcd_pred, pred, args.n_boot, args.seed + 9876)
                )
            pooled_rows.append(m)
    pred_csv = out_dir / "predictions_concat_phase5b_leave_one_domain_out_nested_v1.csv"
    pooled_csv = out_dir / "pooled_metrics_phase5b_leave_one_domain_out_nested_v1.csv"
    fold_csv = out_dir / "fold_metrics_phase5b_leave_one_domain_out_nested_v1.csv"
    weights_csv = (
        out_dir / "ensemble_weights_phase5b_leave_one_domain_out_nested_v1.csv"
    )
    alphas_csv = out_dir / "alpha_summary_phase5b_leave_one_domain_out_nested_v1.csv"
    pred_records.to_csv(pred_csv, index=False)
    pd.DataFrame(pooled_rows).to_csv(pooled_csv, index=False)
    pd.DataFrame(fold_metric_rows).to_csv(fold_csv, index=False)
    pd.DataFrame(weight_rows).to_csv(weights_csv, index=False)
    pd.DataFrame(alpha_rows).to_csv(alphas_csv, index=False)
    pooled_df = pd.DataFrame(pooled_rows)
    _fold_df = pd.DataFrame(fold_metric_rows)
    manifest = {
        "abcd_csv": args.abcd_csv,
        "locked_pred_csv": args.locked_pred_csv,
        "pc_pred_parquet": args.pc_pred_parquet,
        "freeze_tsv": args.freeze_tsv,
        "out_dir": str(out_dir),
        "n": int(n),
        "fold_counts": {str(f): int(np.sum(folds == f)) for f in unique_folds},
        "cov_names": cov_names,
        "subsets": subsets,
        "pc_feature_mode": args.pc_feature_mode,
        "inner_k": int(args.inner_k),
        "inner_metric": args.inner_metric,
        "n_boot": int(args.n_boot),
        "outputs": [
            pred_csv.name,
            pooled_csv.name,
            fold_csv.name,
            weights_csv.name,
            alphas_csv.name,
            "phase5b_leave_one_domain_out_manifest.json",
        ],
    }
    with open(
        out_dir / "phase5b_leave_one_domain_out_manifest.json", "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("[done] Phase 5B leave-one-domain-out ablation completed")
    print("[done] output dir =", out_dir)
    print("[done] pooled ensemble results:")
    ens_rows = pooled_df[pooled_df["method"] == "ensemble"].sort_values(
        "COD", ascending=False
    )
    print(
        ens_rows[
            [
                "subset",
                "n_pcs",
                "COD",
                "Pearson_r",
                "delta_COD_vs_abcd",
                "percent_COD_improvement_vs_abcd",
            ]
        ].to_string(index=False)
    )
