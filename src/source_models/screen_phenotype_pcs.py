"""Select UKB phenotype PCs using a source-side KRR screen.

Holds out a screening subset, evaluates repeated random splits and retains
PCs above the configured mean test-correlation threshold. The remaining
cohort is used for source-model training.
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


def get_edge_cols(csv_path: Path):
    hdr = pd.read_csv(csv_path, nrows=0)
    cols = list(hdr.columns)
    edge_cols = [c for c in cols if re.fullmatch("e\\d+", str(c))]
    edge_cols = sorted(edge_cols, key=lambda x: int(x[1:]))
    if len(edge_cols) == 0:
        raise RuntimeError("No edge columns found in UKB_final.csv")
    return edge_cols


def prepare_screen():
    cand = pd.read_parquet(CAND_PARQUET)
    if "eid" not in cand.columns:
        raise RuntimeError("Candidate parquet missing eid")
    freeze = pd.read_csv(FREEZE_TSV, sep="\t")
    if "col" not in freeze.columns:
        raise RuntimeError("Freeze TSV missing col")
    pc_cols = freeze["col"].astype(str).tolist()
    missing_pc_cols = [c for c in pc_cols if c not in cand.columns]
    if missing_pc_cols:
        raise RuntimeError(f"Candidate parquet missing PC columns: {missing_pc_cols}")
    cand = cand[["eid"] + pc_cols].copy()
    cand["eid"] = cand["eid"].astype(str)
    edge_cols = get_edge_cols(UKB_FINAL)
    ukb = pd.read_csv(UKB_FINAL, usecols=["eid"] + edge_cols)
    ukb["eid"] = ukb["eid"].astype(str)
    merged = cand.merge(ukb, on="eid", how="inner", validate="one_to_one")
    if len(merged) != len(cand):
        raise RuntimeError(
            f"eid overlap mismatch: candidate={len(cand)} merged={len(merged)}"
        )
    n_total = len(merged)
    if SCREEN_N >= n_total:
        raise RuntimeError(
            f"SCREEN_N={SCREEN_N} must be smaller than n_total={n_total}"
        )
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_total)
    screen_idx = np.sort(perm[:SCREEN_N])
    main_idx = np.sort(perm[SCREEN_N:])
    screen_df = merged.iloc[screen_idx].copy().reset_index(drop=True)
    main_df = merged.iloc[main_idx].copy().reset_index(drop=True)
    screen_parquet = OUT_DIR / "pc_source_bank_v1_screen_1000.parquet"
    main_parquet = OUT_DIR / "pc_source_bank_v1_main_after_screen.parquet"
    screen_df.to_parquet(screen_parquet, index=False)
    main_df.to_parquet(main_parquet, index=False)
    screen_pc_csv = OUT_DIR / "pc_source_bank_v1_screen_1000_pc_only.csv.gz"
    main_pc_csv = OUT_DIR / "pc_source_bank_v1_main_after_screen_pc_only.csv.gz"
    screen_df[["eid"] + pc_cols].to_csv(screen_pc_csv, index=False, compression="gzip")
    main_df[["eid"] + pc_cols].to_csv(main_pc_csv, index=False, compression="gzip")
    screen_subj_txt = OUT_DIR / "pc_source_bank_v1_screen_1000_subjects.txt"
    main_subj_txt = OUT_DIR / "pc_source_bank_v1_main_subjects.txt"
    screen_subj_txt.write_text(
        "\n".join(screen_df["eid"].tolist()) + "\n", encoding="utf-8"
    )
    main_subj_txt.write_text(
        "\n".join(main_df["eid"].tolist()) + "\n", encoding="utf-8"
    )
    screen_edges_npy = OUT_DIR / "pc_source_bank_v1_screen_1000_edges.npy"
    np.save(screen_edges_npy, screen_df[edge_cols].to_numpy(dtype=np.float32))
    summary = {
        "seed": SEED,
        "screen_n": int(len(screen_df)),
        "main_n": int(len(main_df)),
        "n_total": int(n_total),
        "n_pc": int(len(pc_cols)),
        "n_edges": int(len(edge_cols)),
        "candidate_parquet": str(CAND_PARQUET),
        "ukb_final_csv": str(UKB_FINAL),
        "screen_parquet": str(screen_parquet),
        "main_parquet": str(main_parquet),
        "screen_pc_csv": str(screen_pc_csv),
        "main_pc_csv": str(main_pc_csv),
        "screen_subj_txt": str(screen_subj_txt),
        "main_subj_txt": str(main_subj_txt),
        "screen_edges_npy": str(screen_edges_npy),
        "he2022_alignment": "Post-PCA screening uses a fixed 1000-subject holdout cohort, separated from the main experiments, aligned to He2022 stage-2 re-screening logic.",
    }
    summary_json = OUT_DIR / "pc_source_bank_v1_screening_split_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("n_total =", n_total)
    print("screen_n =", len(screen_df))
    print("main_n =", len(main_df))
    print("n_pc =", len(pc_cols))
    print("n_edges =", len(edge_cols))
    print("screen_parquet =", screen_parquet)
    print("main_parquet =", main_parquet)
    print("screen_pc_csv =", screen_pc_csv)
    print("main_pc_csv =", main_pc_csv)
    print("screen_subj_txt =", screen_subj_txt)
    print("main_subj_txt =", main_subj_txt)
    print("screen_edges_npy =", screen_edges_npy)
    print("summary_json =", summary_json)


def safe_corr(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2:
        return np.nan
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cod_znormed(real, pred):
    real = np.asarray(real, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    ok = np.isfinite(real) & np.isfinite(pred)
    real = real[ok]
    pred = pred[ok]
    if len(real) == 0:
        return np.nan
    denom = float(np.sum(real**2))
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((real - pred) ** 2) / denom)


def mics_z_norm(train_y, *others):
    train_y = np.asarray(train_y, dtype=float).reshape(-1)
    mu = float(np.mean(train_y))
    sigma = float(np.std(train_y))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    out = [(train_y - mu) / sigma]
    for arr in others:
        arr = np.asarray(arr, dtype=float).reshape(-1)
        out.append((arr - mu) / sigma)
    return (out, mu, sigma)


def unitize_rows(X):
    X = np.asarray(X, dtype=float)
    row_mean = np.mean(X, axis=1, keepdims=True)
    X = X - row_mean
    row_norm = np.linalg.norm(X, axis=1, keepdims=True)
    row_norm[row_norm == 0] = 1.0
    return X / row_norm


def fit_predict_krr_precomputed(K_train, K_pred, y_train, alpha, with_bias=True):
    K_train = np.asarray(K_train, dtype=float)
    K_pred = np.asarray(K_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    n = K_train.shape[0]
    kr = K_train + float(alpha) * np.eye(n, dtype=float)
    try:
        if with_bias:
            ones = np.ones((n, 1), dtype=float)
            inv_kr = np.linalg.solve(kr, np.eye(n, dtype=float))
            denom = float((ones.T @ inv_kr @ ones).item())
            if abs(denom) < 1e-12:
                beta = 0.0
            else:
                beta = float((ones.T @ inv_kr @ y_train.reshape(-1, 1)).item() / denom)
            coef = np.linalg.solve(kr, y_train - beta)
            pred = K_pred @ coef + beta
        else:
            coef = np.linalg.solve(kr, y_train)
            pred = K_pred @ coef
    except np.linalg.LinAlgError:
        pred = np.full(K_pred.shape[0], np.nan, dtype=float)
    return np.asarray(pred, dtype=float).reshape(-1)


def summarize_alpha(values):
    vals = pd.Series(values).dropna()
    if len(vals) == 0:
        return (np.nan, np.nan, 0)
    vc = vals.value_counts()
    top_alpha = float(vc.index[0])
    top_count = int(vc.iloc[0])
    mean_alpha = float(vals.mean())
    return (top_alpha, mean_alpha, top_count)


def run_screen():
    screen_df = pd.read_parquet(SCREEN_PARQUET)
    main_df = pd.read_parquet(MAIN_PARQUET)
    cand_df = pd.read_parquet(CANDIDATE_PARQUET)
    manifest_df = pd.read_csv(CANDIDATE_MANIFEST, sep="\t")
    freeze_df = pd.read_csv(FREEZE_TSV, sep="\t")
    if "eid" not in screen_df.columns:
        raise RuntimeError("screen parquet missing eid")
    if "eid" not in main_df.columns:
        raise RuntimeError("main parquet missing eid")
    if "eid" not in cand_df.columns:
        raise RuntimeError("candidate parquet missing eid")
    if "col" not in freeze_df.columns:
        raise RuntimeError("freeze tsv missing col")
    pc_cols = freeze_df["col"].astype(str).tolist()
    missing_pc = [c for c in pc_cols if c not in screen_df.columns]
    if missing_pc:
        raise RuntimeError(f"screen parquet missing PC cols: {missing_pc}")
    edge_cols = [c for c in screen_df.columns if re.fullmatch("e\\d+", str(c))]
    if len(edge_cols) == 0:
        raise RuntimeError("No edge columns found in screen parquet")
    X_edges = np.load(SCREEN_EDGES_NPY).astype(np.float64)
    if X_edges.shape != (len(screen_df), len(edge_cols)):
        raise RuntimeError(
            f"screen_edges_npy shape mismatch: {X_edges.shape} vs expected ({len(screen_df)}, {len(edge_cols)})"
        )
    X = unitize_rows(X_edges)
    _n_subjects = len(screen_df)
    n_pc = len(pc_cols)
    test_corr_mat = np.full((n_pc, N_RNG), np.nan, dtype=float)
    test_cod_mat = np.full((n_pc, N_RNG), np.nan, dtype=float)
    val_corr_mat = np.full((n_pc, N_RNG), np.nan, dtype=float)
    best_alpha_mat = np.full((n_pc, N_RNG), np.nan, dtype=float)
    n_valid_subjects_vec = np.full(n_pc, np.nan, dtype=float)
    n_successful_splits_vec = np.zeros(n_pc, dtype=int)
    long_rows = []
    summary_rows = []
    _rng_master = np.random.RandomState(SEED)
    for j, pc_name in enumerate(pc_cols):
        y_all = screen_df[pc_name].to_numpy(dtype=float)
        valid_mask = np.isfinite(y_all)
        valid_idx = np.where(valid_mask)[0]
        n_valid = int(len(valid_idx))
        n_valid_subjects_vec[j] = n_valid
        print(f"[PC {j + 1}/{n_pc}] {pc_name}: n_valid={n_valid}", flush=True)
        if n_valid < MIN_VALID_N:
            summary_rows.append(
                {
                    "pc_name": pc_name,
                    "n_valid_subjects": n_valid,
                    "n_successful_splits": 0,
                    "mean_test_corr": np.nan,
                    "std_test_corr": np.nan,
                    "mean_test_cod": np.nan,
                    "std_test_cod": np.nan,
                    "top_best_alpha": np.nan,
                    "mean_best_alpha": np.nan,
                    "top_best_alpha_count": 0,
                    "pass_screen_corr_gt_0p1": False,
                    "screen_threshold_corr": CORR_THRESHOLD,
                    "status": "too_few_valid_subjects",
                }
            )
            continue
        Xv = X[valid_idx, :]
        yv = y_all[valid_idx]
        for r in range(N_RNG):
            rng = np.random.RandomState(SEED + r)
            perm = rng.permutation(n_valid)
            n_train = int(np.floor(0.6 * n_valid))
            n_valid_split = int(np.floor(0.2 * n_valid))
            n_test = int(n_valid - n_train - n_valid_split)
            if min(n_train, n_valid_split, n_test) < 2:
                continue
            tr = perm[:n_train]
            va = perm[n_train : n_train + n_valid_split]
            te = perm[n_train + n_valid_split :]
            Xtr, Xva, Xte = (Xv[tr], Xv[va], Xv[te])
            ytr, yva, yte = (yv[tr], yv[va], yv[te])
            (ytr_z, yva_z, yte_z), _, _ = mics_z_norm(ytr, yva, yte)
            Ktr = Xtr @ Xtr.T
            Kva = Xva @ Xtr.T
            Kte = Xte @ Xtr.T
            best_alpha = np.nan
            best_val_corr = -np.inf
            _best_te_pred = None
            for alpha in ALPHA_GRID:
                pred_va = fit_predict_krr_precomputed(
                    K_train=Ktr,
                    K_pred=Kva,
                    y_train=ytr_z,
                    alpha=alpha,
                    with_bias=WITH_BIAS,
                )
                cur_val_corr = safe_corr(yva_z, pred_va)
                if np.isfinite(cur_val_corr) and cur_val_corr > best_val_corr:
                    best_val_corr = cur_val_corr
                    best_alpha = float(alpha)
            if not np.isfinite(best_alpha):
                continue
            pred_te = fit_predict_krr_precomputed(
                K_train=Ktr,
                K_pred=Kte,
                y_train=ytr_z,
                alpha=best_alpha,
                with_bias=WITH_BIAS,
            )
            cur_test_corr = safe_corr(yte_z, pred_te)
            cur_test_cod = cod_znormed(yte_z, pred_te)
            test_corr_mat[j, r] = cur_test_corr
            test_cod_mat[j, r] = cur_test_cod
            val_corr_mat[j, r] = best_val_corr
            best_alpha_mat[j, r] = best_alpha
            n_successful_splits_vec[j] += 1
            long_rows.append(
                {
                    "pc_name": pc_name,
                    "rng_id": r,
                    "n_valid_subjects": n_valid,
                    "n_train": n_train,
                    "n_valid": n_valid_split,
                    "n_test": n_test,
                    "best_alpha": best_alpha,
                    "val_corr": best_val_corr,
                    "test_corr": cur_test_corr,
                    "test_cod": cur_test_cod,
                }
            )
        mean_test_corr = (
            float(np.nanmean(test_corr_mat[j, :]))
            if np.isfinite(test_corr_mat[j, :]).any()
            else np.nan
        )
        std_test_corr = (
            float(np.nanstd(test_corr_mat[j, :]))
            if np.isfinite(test_corr_mat[j, :]).any()
            else np.nan
        )
        mean_test_cod = (
            float(np.nanmean(test_cod_mat[j, :]))
            if np.isfinite(test_cod_mat[j, :]).any()
            else np.nan
        )
        std_test_cod = (
            float(np.nanstd(test_cod_mat[j, :]))
            if np.isfinite(test_cod_mat[j, :]).any()
            else np.nan
        )
        top_best_alpha, mean_best_alpha, top_best_alpha_count = summarize_alpha(
            best_alpha_mat[j, :]
        )
        pass_screen = bool(
            np.isfinite(mean_test_corr) and mean_test_corr > CORR_THRESHOLD
        )
        summary_rows.append(
            {
                "pc_name": pc_name,
                "n_valid_subjects": n_valid,
                "n_successful_splits": int(n_successful_splits_vec[j]),
                "mean_test_corr": mean_test_corr,
                "std_test_corr": std_test_corr,
                "mean_test_cod": mean_test_cod,
                "std_test_cod": std_test_cod,
                "top_best_alpha": top_best_alpha,
                "mean_best_alpha": mean_best_alpha,
                "top_best_alpha_count": top_best_alpha_count,
                "pass_screen_corr_gt_0p1": pass_screen,
                "screen_threshold_corr": CORR_THRESHOLD,
                "status": (
                    "ok" if int(n_successful_splits_vec[j]) > 0 else "all_splits_failed"
                ),
            }
        )
    long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(summary_rows)
    final_manifest = manifest_df.merge(
        summary_df, how="left", left_on="pc_name", right_on="pc_name"
    )
    final_manifest = final_manifest.sort_values("pc_name").reset_index(drop=True)
    pass_df = summary_df.loc[summary_df["pass_screen_corr_gt_0p1"]].copy()
    pass_df = pass_df.sort_values("pc_name").reset_index(drop=True)
    pass_pc_cols = pass_df["pc_name"].tolist()
    if len(pass_pc_cols) == 0:
        raise RuntimeError(
            "No PCs passed post-PCA KRR screening; cannot write final freeze"
        )
    final_freeze = pd.DataFrame({"col": pass_pc_cols})
    final_all = cand_df[["eid"] + pass_pc_cols].copy()
    edge_cols_main = [c for c in main_df.columns if re.fullmatch("e\\d+", str(c))]
    final_main = main_df[["eid"] + pass_pc_cols + edge_cols_main].copy()
    long_csv = OUT_DIR / "post_pca_krr_screening_v1_long.csv"
    summary_csv = OUT_DIR / "post_pca_krr_screening_v1_summary.csv"
    result_npz = OUT_DIR / "post_pca_krr_screening_v1_result.npz"
    final_manifest_tsv = OUT_DIR / "pc_source_bank_v1_final_manifest.tsv"
    final_freeze_tsv = OUT_DIR / "feature_freeze_pc_v1_final.tsv"
    final_all_parquet = OUT_DIR / "pc_source_bank_v1_final.parquet"
    final_all_csvgz = OUT_DIR / "pc_source_bank_v1_final.csv.gz"
    final_main_parquet = OUT_DIR / "pc_source_bank_v1_main_after_screen_final.parquet"
    final_main_csvgz = OUT_DIR / "pc_source_bank_v1_main_after_screen_final.csv.gz"
    pass_tsv = OUT_DIR / "pc_source_bank_v1_pass_list.tsv"
    metadata_json = OUT_DIR / "post_pca_krr_screening_v1_metadata.json"
    long_df.to_csv(long_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    np.savez_compressed(
        result_npz,
        pc_names=np.array(pc_cols, dtype=object),
        test_corr=test_corr_mat,
        test_cod=test_cod_mat,
        val_corr=val_corr_mat,
        best_alpha=best_alpha_mat,
        n_valid_subjects=n_valid_subjects_vec,
        n_successful_splits=n_successful_splits_vec,
        seed=np.array(SEED, dtype=int),
        n_rng=np.array(N_RNG, dtype=int),
        corr_threshold=np.array(CORR_THRESHOLD, dtype=float),
    )
    final_manifest.to_csv(final_manifest_tsv, sep="\t", index=False)
    final_freeze.to_csv(final_freeze_tsv, sep="\t", index=False)
    final_all.to_parquet(final_all_parquet, index=False)
    final_all.to_csv(final_all_csvgz, index=False, compression="gzip")
    final_main.to_parquet(final_main_parquet, index=False)
    final_main.to_csv(final_main_csvgz, index=False, compression="gzip")
    pass_df.to_csv(pass_tsv, sep="\t", index=False)
    metadata = {
        "seed": SEED,
        "n_rng": N_RNG,
        "corr_threshold": CORR_THRESHOLD,
        "alpha_grid": ALPHA_GRID,
        "with_bias": WITH_BIAS,
        "min_valid_n": MIN_VALID_N,
        "screen_parquet": str(SCREEN_PARQUET),
        "main_parquet": str(MAIN_PARQUET),
        "screen_edges_npy": str(SCREEN_EDGES_NPY),
        "freeze_tsv_candidate": str(FREEZE_TSV),
        "candidate_parquet": str(CANDIDATE_PARQUET),
        "he2022_alignment": "Post-PCA source-side KRR screening on a fixed 1000-subject cohort; 100 random 60/20/20 splits; retain PCs with mean test Pearson correlation > 0.1.",
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("n_pc_candidate =", len(pc_cols))
    print("n_pc_pass =", len(pass_pc_cols))
    print("long_csv =", long_csv)
    print("summary_csv =", summary_csv)
    print("result_npz =", result_npz)
    print("final_manifest_tsv =", final_manifest_tsv)
    print("final_freeze_tsv =", final_freeze_tsv)
    print("final_all_parquet =", final_all_parquet)
    print("final_main_parquet =", final_main_parquet)
    print("pass_tsv =", pass_tsv)
    print("metadata_json =", metadata_json)
    print("\n[TOP_BY_MEAN_TEST_CORR]")
    tmp = summary_df.sort_values("mean_test_corr", ascending=False)
    print(
        tmp[
            [
                "pc_name",
                "n_valid_subjects",
                "n_successful_splits",
                "mean_test_corr",
                "std_test_corr",
                "mean_test_cod",
                "pass_screen_corr_gt_0p1",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


def main():
    global CAND_PARQUET, CANDIDATE_PARQUET, FREEZE_TSV, CANDIDATE_MANIFEST
    global UKB_FINAL, OUT_DIR, SCREEN_PARQUET, MAIN_PARQUET, SCREEN_EDGES_NPY
    global SEED, SCREEN_N, N_RNG, CORR_THRESHOLD, ALPHA_GRID, WITH_BIAS, MIN_VALID_N
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ukb-csv", required=True)
    p.add_argument("--pca-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config/phenotype_pc_screening.json"),
    )
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    SEED = config["seed"]
    SCREEN_N = config["screen_n"]
    N_RNG = config["n_rng"]
    CORR_THRESHOLD = config["correlation_threshold"]
    ALPHA_GRID = config["alpha_grid"]
    WITH_BIAS = config["with_bias"]
    MIN_VALID_N = config["min_valid_n"]
    PCA_OUT = Path(args.pca_dir)
    CAND_PARQUET = CANDIDATE_PARQUET = PCA_OUT / "pc_source_bank_v1_candidate.parquet"
    FREEZE_TSV = PCA_OUT / "feature_freeze_pc_v1_candidate.tsv"
    CANDIDATE_MANIFEST = PCA_OUT / "pc_source_bank_v1_candidate_manifest.tsv"
    UKB_FINAL = Path(args.ukb_csv)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_screen()
    SCREEN_PARQUET = OUT_DIR / "pc_source_bank_v1_screen_1000.parquet"
    MAIN_PARQUET = OUT_DIR / "pc_source_bank_v1_main_after_screen.parquet"
    SCREEN_EDGES_NPY = OUT_DIR / "pc_source_bank_v1_screen_1000_edges.npy"
    OUT_DIR = OUT_DIR / "krr_screening_output_v1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_screen()


if __name__ == "__main__":
    main()
