"""Train UKB DiseaseRaw KRR models with training-set nuisance regression and scaling.

The default penalty grid extends to 1000; --alphas overrides the grid.
"""

import argparse
import gc
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

DEFAULT_ALPHAS = [
    1e-05,
    0.0001,
    0.001,
    0.004,
    0.007,
    0.01,
    0.04,
    0.07,
    0.1,
    0.4,
    0.7,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    50.0,
    75.0,
    100.0,
    150.0,
    200.0,
    300.0,
    500.0,
    750.0,
    1000.0,
]
EXPECTED_SITES = {11025, 11026, 11027}
COVARIATE_NAMES = [
    "Age",
    "Sex",
    "head_motion1",
    "Edu",
    "site_11026",
    "site_11027",
    "age_sq",
    "age_sex",
    "age_sq_sex",
]


def normalize_eid_array(x):
    s = pd.to_numeric(pd.Series(np.asarray(x).ravel()), errors="raise")
    if not np.isfinite(s.to_numpy(dtype=float)).all():
        raise ValueError("eid contains non-finite values")
    vals = s.to_numpy(dtype=float)
    if not np.equal(vals, np.floor(vals)).all():
        raise ValueError("eid contains non-integer-style values")
    return s.astype(np.int64).astype(str).to_numpy()


def normalize_phecode(x):
    return f"{float(x):.2f}"


def safe_name(x):
    return re.sub("[^A-Za-z0-9_.-]+", "_", str(x))


def unitize_rows(A):
    A = np.asarray(A, dtype=np.float64)
    A = A - A.mean(axis=1, keepdims=True)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return A / n


def predictive_cod(y_true_z, y_pred_z):
    y = np.asarray(y_true_z, dtype=np.float64).ravel()
    p = np.asarray(y_pred_z, dtype=np.float64).ravel()
    denom = float(np.sum(y**2))
    if denom <= 0:
        return np.nan
    sse = float(np.sum((y - p) ** 2))
    return float(1.0 - sse / denom)


def centered_cod(y_true, y_pred):
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((y - p) ** 2) / denom)


def pearson_r(y_true, y_pred):
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    if y.size < 2:
        return np.nan
    if np.std(y) <= 0 or np.std(p) <= 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def rmse(y_true, y_pred):
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((y - p) ** 2)))


def load_freeze330(path):
    df = pd.read_csv(path, sep="\t")
    candidates = ["phecode_key", "phecode_norm", "phecode", "target"]
    col = None
    for c in candidates:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError(
            f"Cannot identify phecode column in freeze file: {list(df.columns)}"
        )
    keys = [normalize_phecode(x) for x in df[col]]
    if len(keys) != 330:
        raise ValueError(f"Expected 330 frozen targets; got {len(keys)}")
    if len(set(keys)) != 330:
        raise ValueError("Frozen phecode list contains duplicates after normalization")
    return (df, keys)


def find_edges(columns):
    tmp = []
    for c in columns:
        m = re.fullmatch("e(\\d+)", str(c))
        if m:
            tmp.append((int(m.group(1)), c))
    tmp.sort(key=lambda z: z[0])
    nums = [x[0] for x in tmp]
    names = [x[1] for x in tmp]
    if nums != list(range(1, 1486)):
        raise ValueError(f"Expected exact e1..e1485; found {len(names)} edges")
    return names


def resolve_phecode_columns(path, target_keys):
    head = pd.read_csv(path, sep="\t", nrows=0)
    cols = list(head.columns)
    if "IID" in cols:
        id_col = "IID"
    elif "eid" in cols:
        id_col = "eid"
    else:
        raise KeyError(f"Could not find IID/eid in phecode matrix: {cols[:10]}")
    target_set = set(target_keys)
    mapping = {}
    for c in cols:
        if c == id_col:
            continue
        try:
            key = normalize_phecode(c)
        except Exception:
            continue
        if key in target_set:
            if key in mapping:
                raise ValueError(
                    f"Duplicate phecode mapping for normalized key {key}: {mapping[key]} and {c}"
                )
            mapping[key] = c
    missing = [x for x in target_keys if x not in mapping]
    if missing:
        raise KeyError(
            f"{len(missing)} frozen phecodes missing from raw matrix; first 20: {missing[:20]}"
        )
    return (id_col, mapping)


def build_covariates(df, edu_median):
    required = ["Age", "Sex", "head_motion1", "Edu", "QC_Imaging_Site"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"UKB missing covariate columns: {missing}")
    age = pd.to_numeric(df["Age"], errors="raise").to_numpy(dtype=np.float64)
    sex = pd.to_numeric(df["Sex"], errors="raise").to_numpy(dtype=np.float64)
    motion = pd.to_numeric(df["head_motion1"], errors="raise").to_numpy(
        dtype=np.float64
    )
    edu = pd.to_numeric(df["Edu"], errors="coerce").to_numpy(dtype=np.float64)
    edu = np.where(np.isfinite(edu), edu, float(edu_median))
    site_float = pd.to_numeric(df["QC_Imaging_Site"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.equal(site_float, np.floor(site_float)).all():
        raise ValueError("QC_Imaging_Site contains non-integer-style values")
    site = site_float.astype(np.int64)
    bad_sites = sorted(set(np.unique(site).tolist()) - EXPECTED_SITES)
    if bad_sites:
        raise ValueError(f"Unexpected UKB imaging sites: {bad_sites}")
    if not np.isfinite(age).all():
        raise ValueError("Age contains NaN/Inf")
    if not np.isfinite(sex).all():
        raise ValueError("Sex contains NaN/Inf")
    if not np.isfinite(motion).all():
        raise ValueError("head_motion1 contains NaN/Inf")
    if not np.isfinite(edu).all():
        raise ValueError("Edu still contains NaN/Inf after imputation")
    C = np.column_stack(
        [
            age,
            sex,
            motion,
            edu,
            (site == 11026).astype(np.float64),
            (site == 11027).astype(np.float64),
            age**2,
            age * sex,
            age**2 * sex,
        ]
    )
    if C.shape[1] != 9:
        raise RuntimeError(f"Expected nuisance matrix with 9 columns; got {C.shape}")
    if not np.isfinite(C).all():
        raise ValueError("Nuisance design contains NaN/Inf")
    rank_with_intercept = np.linalg.matrix_rank(
        np.column_stack([np.ones(C.shape[0], dtype=np.float64), C])
    )
    if rank_with_intercept < 10:
        raise ValueError(
            f"Nuisance design rank deficient: rank={rank_with_intercept}, expected 10"
        )
    return C


def load_aligned_data(ukb_csv, phecode_matrix, freeze330, split_npz):
    freeze_df, target_keys = load_freeze330(freeze330)
    split = np.load(split_npz, allow_pickle=True)
    required_split = [
        "eid_inner_train",
        "eid_validation",
        "eid_outer_test",
        "edu_median_inner_train",
    ]
    missing_split = [k for k in required_split if k not in split.files]
    if missing_split:
        raise KeyError(f"Frozen split NPZ missing: {missing_split}")
    eid_inner = normalize_eid_array(split["eid_inner_train"])
    eid_val = normalize_eid_array(split["eid_validation"])
    eid_test = normalize_eid_array(split["eid_outer_test"])
    edu_median = float(np.asarray(split["edu_median_inner_train"]).ravel()[0])
    if len(eid_inner) != 15787:
        raise ValueError(f"Unexpected inner-train N={len(eid_inner)}")
    if len(eid_val) != 3947:
        raise ValueError(f"Unexpected validation N={len(eid_val)}")
    if len(eid_test) != 4934:
        raise ValueError(f"Unexpected outer-test N={len(eid_test)}")
    if np.intersect1d(eid_inner, eid_val).size:
        raise ValueError("inner/validation overlap")
    if np.intersect1d(eid_inner, eid_test).size:
        raise ValueError("inner/test overlap")
    if np.intersect1d(eid_val, eid_test).size:
        raise ValueError("validation/test overlap")
    ukb_head = pd.read_csv(ukb_csv, nrows=0)
    edge_names = find_edges(ukb_head.columns)
    ukb_cols = [
        "eid",
        "Age",
        "Sex",
        "head_motion1",
        "Edu",
        "QC_Imaging_Site",
    ] + edge_names
    ukb = pd.read_csv(ukb_csv, usecols=ukb_cols)
    ukb["eid_key"] = normalize_eid_array(ukb["eid"])
    if ukb["eid_key"].duplicated().any():
        raise ValueError("UKB contains duplicated eid")
    id_col, col_map = resolve_phecode_columns(phecode_matrix, target_keys)
    phe_usecols = [id_col] + [col_map[k] for k in target_keys]
    phe = pd.read_csv(phecode_matrix, sep="\t", usecols=phe_usecols)
    phe["eid_key"] = normalize_eid_array(phe[id_col])
    if phe["eid_key"].duplicated().any():
        raise ValueError("Phecode matrix contains duplicated IID/eid")
    rename_map = {col_map[k]: k for k in target_keys}
    phe = phe.rename(columns=rename_map)
    phe = phe[["eid_key"] + target_keys]
    merged = ukb.merge(
        phe, on="eid_key", how="left", validate="one_to_one", indicator=True
    )
    if not (merged["_merge"] == "both").all():
        nbad = int((merged["_merge"] != "both").sum())
        raise ValueError(f"{nbad} UKB imaging subjects not found in phecode matrix")
    merged = merged.drop(columns=["_merge"])
    if len(merged) != 24668:
        raise ValueError(f"Expected N=24668 after alignment; got {len(merged)}")
    row_map = pd.Series(np.arange(len(merged), dtype=np.int64), index=merged["eid_key"])

    def get_idx(eids, name):
        miss = [x for x in eids if x not in row_map.index]
        if miss:
            raise KeyError(
                f"{name}: {len(miss)} frozen eids missing; first 10={miss[:10]}"
            )
        return row_map.loc[eids].to_numpy(dtype=np.int64)

    idx_inner = get_idx(eid_inner, "inner_train")
    idx_val = get_idx(eid_val, "validation")
    idx_test = get_idx(eid_test, "outer_test")
    X = merged[edge_names].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("RSFC edges contain NaN/Inf")
    C = build_covariates(merged, edu_median=edu_median)
    return {
        "merged": merged,
        "target_keys": target_keys,
        "edge_names": edge_names,
        "X": X,
        "C": C,
        "idx_inner": idx_inner,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "eid_inner": eid_inner,
        "eid_val": eid_val,
        "eid_test": eid_test,
        "edu_median": edu_median,
        "freeze_df": freeze_df,
    }


def count_binary(y):
    y = np.asarray(y, dtype=np.float64)
    obs = np.isfinite(y)
    yy = y[obs]
    if yy.size == 0:
        return (0, 0, 0)
    bad = ~np.isin(yy, [0.0, 1.0])
    if bad.any():
        raise ValueError(
            f"Observed raw disease values are not binary: {np.unique(yy[bad])[:10]}"
        )
    return (int(yy.size), int(np.sum(yy == 1.0)), int(np.sum(yy == 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ukb-csv", required=True)
    ap.add_argument("--phecode-matrix", required=True)
    ap.add_argument("--freeze330", required=True)
    ap.add_argument("--split-npz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--target-index",
        type=int,
        required=True,
        help="0-based index in frozen all330 list",
    )
    ap.add_argument("--alphas", default=",".join((str(x) for x in DEFAULT_ALPHAS)))
    args = ap.parse_args()
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if not alphas:
        raise ValueError("alpha grid is empty")
    pack = load_aligned_data(
        ukb_csv=args.ukb_csv,
        phecode_matrix=args.phecode_matrix,
        freeze330=args.freeze330,
        split_npz=args.split_npz,
    )
    targets = pack["target_keys"]
    j = int(args.target_index)
    if j < 0 or j >= len(targets):
        raise IndexError(f"target-index must be 0..{len(targets) - 1}; got {j}")
    target = targets[j]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target_dir = out_root / f"seed{args.seed}" / f"target_{j:03d}_{safe_name(target)}"
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "summary.json"
    if summary_path.exists():
        raise RuntimeError(f"Output already exists: {summary_path}")
    print("===== PHASE5C KRR CONTRACT =====")
    print("seed:", args.seed)
    print("target index:", j)
    print("target:", target)
    print("N inner train:", len(pack["idx_inner"]))
    print("N validation:", len(pack["idx_val"]))
    print("N outer test:", len(pack["idx_test"]))
    print("Edu median:", pack["edu_median"])
    print("residualize_y: True")
    print("refit_on_train_validation: False")
    print("kernel: row-centered L2-unitized Pearson")
    print("alphas:", alphas)
    X = pack["X"]
    C = pack["C"]
    merged = pack["merged"]
    ii = pack["idx_inner"]
    iv = pack["idx_val"]
    it = pack["idx_test"]
    Z_inner = unitize_rows(X[ii])
    Z_val = unitize_rows(X[iv])
    Z_test = unitize_rows(X[it])
    y_all = pd.to_numeric(merged[target], errors="coerce").to_numpy(dtype=np.float64)
    y_inner = y_all[ii]
    y_val = y_all[iv]
    y_test = y_all[it]
    n_inner_obs, n_inner_case, n_inner_control = count_binary(y_inner)
    n_val_obs, n_val_case, n_val_control = count_binary(y_val)
    n_test_obs, n_test_case, n_test_control = count_binary(y_test)
    print("\n===== RAW DISEASE COUNTS =====")
    print("inner:", n_inner_obs, n_inner_case, n_inner_control)
    print("validation:", n_val_obs, n_val_case, n_val_control)
    print("test:", n_test_obs, n_test_case, n_test_control)
    m_inner = np.isfinite(y_inner)
    m_val = np.isfinite(y_val)
    m_test = np.isfinite(y_test)
    if m_inner.sum() < 2:
        raise RuntimeError("Too few observed inner-training subjects")
    if m_val.sum() < 1:
        raise RuntimeError("No observed validation subjects")
    if m_test.sum() < 1:
        raise RuntimeError("No observed outer-test subjects")
    C_inner = C[ii]
    C_val = C[iv]
    C_test = C[it]
    nuisance = LinearRegression(fit_intercept=True)
    nuisance.fit(C_inner[m_inner], y_inner[m_inner])
    resid_inner = y_inner[m_inner] - nuisance.predict(C_inner[m_inner])
    resid_val = y_val[m_val] - nuisance.predict(C_val[m_val])
    resid_test = y_test[m_test] - nuisance.predict(C_test[m_test])
    resid_mu = float(np.mean(resid_inner))
    resid_sd = float(np.std(resid_inner, ddof=0))
    if not np.isfinite(resid_sd) or resid_sd <= 0:
        raise RuntimeError(f"Invalid residual training SD: {resid_sd}")
    ytr_z = (resid_inner - resid_mu) / resid_sd
    yva_z = (resid_val - resid_mu) / resid_sd
    yte_z = (resid_test - resid_mu) / resid_sd
    Ztr = Z_inner[m_inner]
    Zva = Z_val[m_val]
    Zte = Z_test[m_test]
    print("\nBuilding training kernel ...", flush=True)
    Ktr = Ztr @ Ztr.T
    print("Ktr shape:", Ktr.shape, "dtype:", Ktr.dtype, flush=True)
    print("Building validation kernel ...", flush=True)
    Kva = Zva @ Ztr.T
    best_alpha = None
    best_score = None
    best_model = None
    _best_val_pred = None
    alpha_rows = []
    for alpha in alphas:
        print(f"alpha={alpha:g}", flush=True)
        model = KernelRidge(kernel="precomputed", alpha=float(alpha))
        model.fit(Ktr, ytr_z)
        pred = model.predict(Kva)
        score = predictive_cod(yva_z, pred)
        row = {
            "alpha": float(alpha),
            "predictive_COD_valid_z": score,
            "Pearson_r_valid_z": pearson_r(yva_z, pred),
            "COD_centered_valid_z": centered_cod(yva_z, pred),
            "RMSE_valid_z": rmse(yva_z, pred),
        }
        alpha_rows.append(row)
        print(
            "  valid COD=",
            row["predictive_COD_valid_z"],
            "r=",
            row["Pearson_r_valid_z"],
            flush=True,
        )
        if np.isfinite(score) and (best_score is None or score > best_score):
            best_score = float(score)
            best_alpha = float(alpha)
            best_model = model
            _best_val_pred = np.asarray(pred, dtype=np.float64)
    if best_model is None:
        raise RuntimeError("No finite alpha candidate")
    pd.DataFrame(alpha_rows).to_csv(
        target_dir / "alpha_validation_metrics.csv", index=False
    )
    del Kva
    gc.collect()
    print("\nBuilding outer-test kernel ...", flush=True)
    Kte = Zte @ Ztr.T
    pred_test = best_model.predict(Kte)
    test_predictive_cod = predictive_cod(yte_z, pred_test)
    test_r = pearson_r(yte_z, pred_test)
    test_centered_cod = centered_cod(yte_z, pred_test)
    test_rmse = rmse(yte_z, pred_test)
    if n_test_case > 0 and n_test_control > 0:
        test_auc = float(roc_auc_score(y_test[m_test], pred_test))
    else:
        test_auc = np.nan
    val_metrics = [x for x in alpha_rows if np.isclose(x["alpha"], best_alpha)][0]
    dual_coef = np.asarray(best_model.dual_coef_, dtype=np.float64).ravel()
    fit_eid = pack["eid_inner"][m_inner]
    bundle = {
        "bundle_type": "ukb_raw_disease_residualized_krr_stage1_target",
        "version": "Phase5C_v1",
        "seed": int(args.seed),
        "target_index": int(j),
        "target": target,
        "best_alpha": float(best_alpha),
        "dual_coef": dual_coef,
        "fit_eid": np.asarray(fit_eid).astype(str),
        "edge_names": list(pack["edge_names"]),
        "feature_transform": "row_center_then_l2_unitize",
        "kernel_type": "Pearson_correlation_precomputed",
        "target_space": "covariate_residual_z",
        "covariate_names": list(COVARIATE_NAMES),
        "site_reference": 11025,
        "edu_median_inner_train": float(pack["edu_median"]),
        "nuisance_intercept": float(nuisance.intercept_),
        "nuisance_coef": np.asarray(nuisance.coef_, dtype=np.float64),
        "y_mean": float(resid_mu),
        "y_std": float(resid_sd),
        "residual_mean": float(resid_mu),
        "residual_std": float(resid_sd),
        "refit_on_train_validation": False,
        "selection_metric": "predictive_COD_valid_z",
        "prediction_scale": "z",
    }
    bundle_path = target_dir / "krr_transfer_bundle.joblib"
    dump(bundle, bundle_path, compress=3)
    np.savez_compressed(
        target_dir / "outer_test_predictions.npz",
        eid=pack["eid_test"][m_test],
        y_raw=y_test[m_test].astype(np.float32),
        y_true_z=yte_z.astype(np.float32),
        y_pred_z=np.asarray(pred_test, dtype=np.float32),
    )
    summary = {
        "seed": int(args.seed),
        "target_index": int(j),
        "target": target,
        "residualize_y": True,
        "refit_on_train_validation": False,
        "n_inner_train": int(len(ii)),
        "n_validation": int(len(iv)),
        "n_outer_test": int(len(it)),
        "n_inner_obs": n_inner_obs,
        "n_inner_case": n_inner_case,
        "n_inner_control": n_inner_control,
        "n_validation_obs": n_val_obs,
        "n_validation_case": n_val_case,
        "n_validation_control": n_val_control,
        "n_test_obs": n_test_obs,
        "n_test_case": n_test_case,
        "n_test_control": n_test_control,
        "edu_median_inner_train": float(pack["edu_median"]),
        "residual_train_mean": resid_mu,
        "residual_train_sd": resid_sd,
        "best_alpha": best_alpha,
        "predictive_COD_valid_z": float(val_metrics["predictive_COD_valid_z"]),
        "Pearson_r_valid_z": float(val_metrics["Pearson_r_valid_z"]),
        "COD_centered_valid_z": float(val_metrics["COD_centered_valid_z"]),
        "RMSE_valid_z": float(val_metrics["RMSE_valid_z"]),
        "predictive_COD_test_z": test_predictive_cod,
        "Pearson_r_test_z": test_r,
        "COD_centered_test_z": test_centered_cod,
        "RMSE_test_z": test_rmse,
        "AUC_raw_disease_test": test_auc,
        "bundle_path": str(bundle_path),
        "kernel_type": "Pearson_correlation_precomputed",
        "feature_transform": "row_center_then_l2_unitize",
        "covariate_names": COVARIATE_NAMES,
        "site_reference": 11025,
        "selection_metric": "predictive_COD_valid_z",
        "target_space": "covariate_residual_z",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n===== FINAL RESULT =====")
    print("target:", target)
    print("best alpha:", best_alpha)
    print("valid predictive COD:", summary["predictive_COD_valid_z"])
    print("valid r:", summary["Pearson_r_valid_z"])
    print("test predictive COD:", summary["predictive_COD_test_z"])
    print("test r:", summary["Pearson_r_test_z"])
    print("test AUC:", summary["AUC_raw_disease_test"])
    print("\n===== OUTPUTS =====")
    print(summary_path)
    print(bundle_path)
    print("\nPASS Phase5C residualized disease KRR target", j)


if __name__ == "__main__":
    main()
