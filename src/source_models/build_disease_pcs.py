"""Build and align UKB disease-PC source targets.

Uses the ordered disease list, column-mean imputation for PCA,
StandardScaler and full-SVD PCA.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def normalize_phecode(x):
    if pd.isna(x):
        return np.nan
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x).strip()


def build_disease_pcs():
    print("Loading retained phecodes...")
    retained = pd.read_csv(RETAINED_TSV, sep="\t")
    retained["phecode"] = retained["phecode"].astype(str)
    retained["phecode_norm"] = retained["phecode"].map(normalize_phecode)
    phecodes = retained["phecode"].tolist()
    print("Retained phecodes:", len(phecodes))
    print("Loading phecode annotation...")
    annot = pd.read_csv(ANNOTATION_CSV)
    annot["phecode_norm"] = annot["phecode"].map(normalize_phecode)
    annot_cols = [
        "phecode_norm",
        "phenotype",
        "phecode_exclude_range",
        "sex",
        "rollup",
        "leaf",
        "category_number",
        "category",
        "cases",
    ]
    annot = annot[[c for c in annot_cols if c in annot.columns]].drop_duplicates(
        "phecode_norm"
    )
    retained_annot = retained.merge(annot, on="phecode_norm", how="left")
    retained_annot.to_csv(
        f"{OUT_DIR}/retained_strict_phecodes_with_annotation.tsv", sep="\t", index=False
    )
    n_missing_annot = (
        int(retained_annot["phenotype"].isna().sum())
        if "phenotype" in retained_annot.columns
        else len(retained_annot)
    )
    print("Missing annotation rows:", n_missing_annot)
    usecols = ["FID", "IID"] + phecodes
    print("Loading imaging-overlap phecode matrix...")
    df = pd.read_csv(IN_MATRIX, sep="\t", usecols=usecols, na_values=["NA"])
    print("Input matrix:", df.shape)
    ids = df[["FID", "IID"]].copy()
    X_raw = df[phecodes].apply(pd.to_numeric, errors="coerce")
    qc_rows = []
    for c in phecodes:
        x = X_raw[c]
        qc_rows.append(
            {
                "phecode": c,
                "phecode_norm": normalize_phecode(c),
                "n_total": len(x),
                "n_nonmissing": int(x.notna().sum()),
                "n_missing_or_excluded": int(x.isna().sum()),
                "missing_or_exclusion_rate": float(x.isna().mean()),
                "n_case": int((x == 1).sum(skipna=True)),
                "n_control": int((x == 0).sum(skipna=True)),
                "mean_nonmissing": float(x.mean(skipna=True)),
                "std_nonmissing": float(x.std(skipna=True)),
            }
        )
    qc = pd.DataFrame(qc_rows)
    qc = qc.merge(annot, on="phecode_norm", how="left")
    qc.to_csv(
        f"{OUT_DIR}/disease_pc_input_qc_with_annotation.tsv", sep="\t", index=False
    )
    col_means = X_raw.mean(axis=0, skipna=True)
    X_imp = X_raw.fillna(col_means)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_z = scaler.fit_transform(X_imp.values.astype(np.float64))
    std_params = pd.DataFrame(
        {
            "phecode": phecodes,
            "phecode_norm": [normalize_phecode(c) for c in phecodes],
            "impute_mean": col_means.values,
            "scaler_mean_after_impute": scaler.mean_,
            "scaler_scale_after_impute": scaler.scale_,
        }
    ).merge(annot, on="phecode_norm", how="left")
    std_params.to_csv(
        f"{OUT_DIR}/disease_pc_imputation_and_scaling_params.tsv", sep="\t", index=False
    )
    print("Running PCA...")
    pca = PCA(n_components=N_PCS, svd_solver="full", random_state=0)
    scores = pca.fit_transform(X_z)
    pc_names = [f"disease_phecode__pc{i:02d}" for i in range(1, N_PCS + 1)]
    scores_df = ids.copy()
    for j, pc in enumerate(pc_names):
        scores_df[pc] = scores[:, j]
    scores_path = f"{OUT_DIR}/disease_pc_scores_candidate_top{N_PCS}.tsv.gz"
    scores_df.to_csv(scores_path, sep="\t", index=False, compression="gzip")
    evr = pd.DataFrame(
        {
            "pc": pc_names,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    evr.to_csv(f"{OUT_DIR}/disease_pc_explained_variance.tsv", sep="\t", index=False)
    loadings = (
        pd.DataFrame(pca.components_.T, index=phecodes, columns=pc_names)
        .reset_index()
        .rename(columns={"index": "phecode"})
    )
    loadings["phecode_norm"] = loadings["phecode"].map(normalize_phecode)
    loadings = loadings.merge(qc, on=["phecode", "phecode_norm"], how="left")
    loadings.to_csv(
        f"{OUT_DIR}/disease_pc_loadings_with_annotation.tsv", sep="\t", index=False
    )
    top_rows = []
    for pc in pc_names:
        keep_cols = [
            "phecode",
            "phecode_norm",
            pc,
            "n_case",
            "n_control",
            "missing_or_exclusion_rate",
            "mean_nonmissing",
            "phenotype",
            "category",
            "category_number",
            "sex",
            "leaf",
            "rollup",
            "cases",
        ]
        keep_cols = [c for c in keep_cols if c in loadings.columns]
        tmp = loadings[keep_cols].copy()
        tmp["pc"] = pc
        tmp["loading"] = tmp[pc]
        tmp["abs_loading"] = tmp["loading"].abs()
        tmp = tmp.sort_values("abs_loading", ascending=False).head(25)
        out_cols = [
            "pc",
            "phecode",
            "phecode_norm",
            "phenotype",
            "category",
            "loading",
            "abs_loading",
            "n_case",
            "n_control",
            "missing_or_exclusion_rate",
            "mean_nonmissing",
            "category_number",
            "sex",
            "leaf",
            "rollup",
            "cases",
        ]
        out_cols = [c for c in out_cols if c in tmp.columns]
        top_rows.append(tmp[out_cols])
    top = pd.concat(top_rows, axis=0, ignore_index=True)
    top.to_csv(
        f"{OUT_DIR}/disease_pc_top25_loadings_each_pc_with_annotation.tsv",
        sep="\t",
        index=False,
    )
    freeze = pd.DataFrame(
        {
            "feature": pc_names,
            "source_block": "disease_phecode",
            "pc_index": list(range(1, N_PCS + 1)),
        }
    )
    freeze_path = f"{OUT_DIR}/feature_freeze_disease_pc_candidate_top{N_PCS}.tsv"
    freeze.to_csv(freeze_path, sep="\t", index=False)
    manifest = {
        "input_matrix": IN_MATRIX,
        "retained_tsv": RETAINED_TSV,
        "annotation_csv": ANNOTATION_CSV,
        "out_dir": OUT_DIR,
        "n_subjects": int(df.shape[0]),
        "n_retained_phecodes": int(len(phecodes)),
        "n_missing_annotation": n_missing_annot,
        "n_pcs": int(N_PCS),
        "scores_path": scores_path,
        "freeze_path": freeze_path,
        "na_handling": "NA treated as missing/excluded; column mean imputation for PCA only",
        "standardization": "StandardScaler after PCA-only mean imputation",
        "sex_specific": "excluded in retained strict list",
        "note": "Original Owen and Zehui files are read-only; all derived outputs are written under the user's MM_test11_disease_icd_block directory.",
    }
    with open(f"{OUT_DIR}/make_disease_pc_source_bank_v1_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Done.")
    print(json.dumps(manifest, indent=2))
    print(evr.head(20))


def prepare_targets():
    print("Loading disease PC scores...")
    scores = pd.read_csv(SCORES_TSV, sep="\t")
    pc_cols = [c for c in scores.columns if c.startswith("disease_phecode__pc")]
    if len(pc_cols) == 0:
        raise RuntimeError("No disease_phecode__pc columns found.")
    print("Loading UKB_final eid...")
    ukb = pd.read_csv(UKB_FINAL, usecols=["eid"])
    ukb["eid"] = pd.to_numeric(ukb["eid"], errors="raise").astype(np.int64)
    scores["IID"] = pd.to_numeric(scores["IID"], errors="raise").astype(np.int64)
    if scores["IID"].duplicated().any():
        dup = scores.loc[scores["IID"].duplicated(), "IID"].head(10).tolist()
        raise RuntimeError(f"Duplicated IID in disease PC scores: {dup}")
    target = scores.rename(columns={"IID": "eid"})[["eid"] + pc_cols].copy()
    target["eid"] = target["eid"].astype(np.int64)
    n_overlap = target["eid"].isin(set(ukb["eid"])).sum()
    print("UKB_final N:", len(ukb))
    print("Disease PC target N:", len(target))
    print("Overlap:", n_overlap)
    if n_overlap != len(target):
        raise RuntimeError("Not all disease PC subjects overlap UKB_final.")
    X = target[pc_cols].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise RuntimeError("Disease PC target matrix contains NaN/Inf.")
    parquet_path = f"{OUT_DIR}/disease_pc_source_bank_candidate_top20_for_krr.parquet"
    freeze_path = f"{OUT_DIR}/feature_freeze_disease_pc_candidate_top20_noheader.tsv"
    manifest_path = f"{OUT_DIR}/prepare_disease_pc_for_krr_stage1_v1_manifest.json"
    target.to_parquet(parquet_path, index=False)
    with open(freeze_path, "w") as f:
        for c in pc_cols:
            f.write(c + "\n")
    desc = target[pc_cols].describe().T
    desc.to_csv(f"{OUT_DIR}/disease_pc_target_distribution_for_krr.tsv", sep="\t")
    manifest = {
        "input_scores_tsv": SCORES_TSV,
        "ukb_final": UKB_FINAL,
        "out_dir": OUT_DIR,
        "parquet_path": parquet_path,
        "freeze_path": freeze_path,
        "n_subjects": int(len(target)),
        "n_ukb_final": int(len(ukb)),
        "n_overlap": int(n_overlap),
        "n_targets": int(len(pc_cols)),
        "target_names": pc_cols,
        "note": "Adapter file for reusing ukb_krr_stage1_pc_v1.py without modifying original KRR code.",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Done.")
    print(json.dumps(manifest, indent=2))


def main():
    global IN_MATRIX, RETAINED_TSV, ANNOTATION_CSV, OUT_DIR, N_PCS, UKB_FINAL, SCORES_TSV
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--disease-matrix", required=True)
    p.add_argument("--ukb-csv", required=True)
    p.add_argument("--annotation-csv", required=True)
    p.add_argument("--retained-tsv", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    IN_MATRIX = args.disease_matrix
    RETAINED_TSV = args.retained_tsv
    ANNOTATION_CSV = args.annotation_csv
    UKB_FINAL = args.ukb_csv
    N_PCS = 20
    retained = pd.read_csv(RETAINED_TSV, sep="\t")
    if len(retained) != 330 or not retained["phecode"].is_unique:
        raise ValueError("Expected 330 unique ordered PheCodes.")
    base = Path(args.out_dir)
    OUT_DIR = str(base / "source_bank_disease_pc_v1")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    build_disease_pcs()
    SCORES_TSV = str(Path(OUT_DIR) / "disease_pc_scores_candidate_top20.tsv.gz")
    OUT_DIR = str(base / "stage1_krr_disease_pc_inputs_v1")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    prepare_targets()


if __name__ == "__main__":
    main()
