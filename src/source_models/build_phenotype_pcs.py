"""Build the six-block UKB phenotype PCA bank.

Converts specified missing-value codes to missing entries and applies
standardized EM PCA with the 95%-or-six-component rule.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.multivariate.pca import PCA as SMPCA

SPECIAL_NUMERIC = {-1.0, -3.0, -7.0, -10.0, -11.0, -13.0, -17.0, -21.0, 999.0, 9999.0}
SPECIAL_STRING = {str(int(x)) for x in SPECIAL_NUMERIC}


def read_table_auto(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep="\\s+", engine="python")
    return df


def normalize_id_col(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    if "eid" in cols:
        out = df.copy()
        out["eid"] = out["eid"].astype(str)
        return out
    if "ID" in cols:
        out = df.copy().rename(columns={"ID": "eid"})
        out["eid"] = out["eid"].astype(str)
        return out
    first = cols[0]
    out = df.copy().rename(columns={first: "eid"})
    out["eid"] = out["eid"].astype(str)
    return out


def clean_numeric_series(s: pd.Series) -> pd.Series:
    s_str = s.astype(str).str.strip()
    s_num = pd.to_numeric(s, errors="coerce")
    special_mask = s_str.isin(SPECIAL_STRING) | s_num.isin(SPECIAL_NUMERIC)
    s_num = s_num.mask(special_mask, np.nan)
    return s_num.astype(float)


def build_inputs():
    manifest = pd.read_csv(MANIFEST, sep="\t")
    required = {"raw_col", "block", "source_table", "block_input_order"}
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")
    cohort = pd.read_csv(UKB_FINAL, usecols=["eid"])
    cohort["eid"] = cohort["eid"].astype(str)
    cohort = cohort.drop_duplicates(subset=["eid"]).reset_index(drop=True)
    cohort_n = len(cohort)
    wide = cohort.copy()
    summary_rows = []
    for source_table, sub in manifest.groupby("source_table", sort=True):
        if source_table not in TABLE_PATHS:
            raise RuntimeError(f"Unknown source_table: {source_table}")
        path = TABLE_PATHS[source_table]
        df = read_table_auto(path)
        df = normalize_id_col(df)
        dup_n = int(df["eid"].duplicated().sum())
        if dup_n > 0:
            raise RuntimeError(f"{source_table} has duplicated eid rows: {dup_n}")
        needed_cols = ["eid"] + sub["raw_col"].tolist()
        missing_cols = [c for c in needed_cols if c not in df.columns]
        if missing_cols:
            raise RuntimeError(f"{source_table} missing columns: {missing_cols}")
        df = df[needed_cols].copy()
        for col in sub["raw_col"]:
            df[col] = clean_numeric_series(df[col])
        merged = cohort.merge(df, on="eid", how="left", validate="one_to_one")
        for col in sub["raw_col"]:
            wide[col] = merged[col]
    wide_parquet = OUT_DIR / "source_bank_v2_pca_input_wide.parquet"
    wide_csvgz = OUT_DIR / "source_bank_v2_pca_input_wide.csv.gz"
    wide.to_parquet(wide_parquet, index=False)
    wide.to_csv(wide_csvgz, index=False, compression="gzip")
    for block, sub in manifest.groupby("block", sort=True):
        cols = sub.sort_values("block_input_order")["raw_col"].tolist()
        block_df = wide[["eid"] + cols].copy()
        block_path = BLOCK_DIR / f"{block}.parquet"
        block_csvgz = BLOCK_DIR / f"{block}.csv.gz"
        block_df.to_parquet(block_path, index=False)
        block_df.to_csv(block_csvgz, index=False, compression="gzip")
        x = block_df[cols]
        n_rows = len(block_df)
        n_cols = len(cols)
        total_missing = int(x.isna().sum().sum())
        mean_missing_rate = float(x.isna().mean().mean()) if n_cols > 0 else np.nan
        min_nonmissing = int(x.notna().sum().min()) if n_cols > 0 else 0
        max_nonmissing = int(x.notna().sum().max()) if n_cols > 0 else 0
        summary_rows.append(
            {
                "block": block,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "total_missing": total_missing,
                "mean_missing_rate": round(mean_missing_rate, 6),
                "min_nonmissing": min_nonmissing,
                "max_nonmissing": max_nonmissing,
                "parquet": str(block_path),
                "csv_gz": str(block_csvgz),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("block").reset_index(drop=True)
    summary_path = OUT_DIR / "source_bank_v2_pca_block_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print("cohort_n =", cohort_n)
    print("wide_shape =", wide.shape)
    print("wide_parquet =", wide_parquet)
    print("wide_csvgz =", wide_csvgz)
    print("block_summary =", summary_path)
    print(summary.to_string(index=False))


def choose_keep_k(rsquare, threshold=0.95):
    rsq = np.asarray(rsquare, dtype=float).reshape(-1)
    idx = np.where(rsq >= threshold)[0]
    if len(idx) > 0:
        return int(idx[0] + 1)
    return int(len(rsq))


def run_block_pca():
    manifest = pd.read_csv(INPUT_MANIFEST, sep="\t")
    plan = pd.read_csv(PCA_PLAN, sep="\t")
    wide = pd.read_parquet(WIDE_INPUT)
    if "eid" not in wide.columns:
        raise RuntimeError("wide input missing eid column")
    eid = wide["eid"].astype(str).copy()
    out_wide = pd.DataFrame({"eid": eid})
    block_summary_rows = []
    pc_manifest_rows = []
    freeze_rows = []
    for _, plan_row in plan.sort_values("block").iterrows():
        block = plan_row["block"]
        sub = manifest.loc[manifest["block"] == block].copy()
        sub = sub.sort_values("block_input_order")
        cols = sub["raw_col"].tolist()
        if len(cols) == 0:
            raise RuntimeError(f"{block}: no input columns found in manifest")
        X = wide[cols].to_numpy(dtype=float)
        n_subjects_total = X.shape[0]
        n_raw_cols = X.shape[1]
        ncomp_max = int(min(6, n_raw_cols))
        if ncomp_max < 1:
            raise RuntimeError(f"{block}: invalid ncomp_max={ncomp_max}")
        fit_mask = np.sum(~np.isnan(X), axis=1) >= ncomp_max
        n_fit = int(fit_mask.sum())
        if n_fit <= ncomp_max:
            raise RuntimeError(
                f"{block}: too few subjects for PCA fit (n_fit={n_fit}, ncomp_max={ncomp_max})"
            )
        X_fit = X[fit_mask, :]
        pca = SMPCA(X_fit, ncomp=ncomp_max, standardize=True, missing="fill-em")
        factors = np.asarray(pca.factors, dtype=float)
        if factors.ndim == 1:
            factors = factors[:, None]
        available_k = int(factors.shape[1])
        rsquare_raw = np.asarray(pca.rsquare, dtype=float).reshape(-1)
        if rsquare_raw.size == available_k + 1:
            rsquare = rsquare_raw[1:]
        elif rsquare_raw.size >= available_k:
            rsquare = rsquare_raw[:available_k]
        else:
            raise RuntimeError(
                f"{block}: rsquare length too short: rsquare={rsquare_raw.size}, available_k={available_k}"
            )
        keep_k = choose_keep_k(rsquare, threshold=0.95)
        keep_k = int(min(keep_k, available_k, ncomp_max))
        print(
            f"[DEBUG] {block}: ncomp_max={ncomp_max}, factors_shape={factors.shape}, rsquare_raw_len={rsquare_raw.size}, rsquare_used_len={rsquare.size}, keep_k={keep_k}"
        )
        factors = factors[:, :keep_k]
        if factors.shape != (n_fit, keep_k):
            raise RuntimeError(
                f"{block}: factor shape mismatch: factors={factors.shape}, expected=({n_fit}, {keep_k})"
            )
        pc_names = [f"{block}__pc{i + 1:02d}" for i in range(keep_k)]
        block_pc = pd.DataFrame({"eid": eid})
        for name in pc_names:
            block_pc[name] = np.nan
        row_idx = np.where(fit_mask)[0]
        col_idx = [block_pc.columns.get_loc(c) for c in pc_names]
        block_pc.iloc[row_idx, col_idx] = factors
        block_parquet = BLOCK_DIR / f"{block}__pcs.parquet"
        block_csvgz = BLOCK_DIR / f"{block}__pcs.csv.gz"
        block_pc.to_parquet(block_parquet, index=False)
        block_pc.to_csv(block_csvgz, index=False, compression="gzip")
        for name in pc_names:
            out_wide[name] = block_pc[name].to_numpy()
        block_summary_rows.append(
            {
                "block": block,
                "n_subjects_total": n_subjects_total,
                "n_raw_cols": n_raw_cols,
                "ncomp_max": ncomp_max,
                "n_fit_subjects": n_fit,
                "fit_rate": round(n_fit / n_subjects_total, 6),
                "keep_k": keep_k,
                "cum_r2_last_kept": round(float(rsquare[keep_k - 1]), 6),
                "cum_r2_all_max": round(float(rsquare[-1]), 6),
                "pcs_parquet": str(block_parquet),
                "pcs_csv_gz": str(block_csvgz),
            }
        )
        for i, name in enumerate(pc_names, start=1):
            freeze_rows.append({"col": name})
            pc_manifest_rows.append(
                {
                    "pc_name": name,
                    "block": block,
                    "component_index": i,
                    "n_raw_cols": n_raw_cols,
                    "ncomp_max": ncomp_max,
                    "keep_k": keep_k,
                    "cum_r2_at_component": round(float(rsquare[i - 1]), 6),
                    "n_fit_subjects": n_fit,
                    "fit_rate": round(n_fit / n_subjects_total, 6),
                    "source_table_set": "|".join(
                        sorted(sub["source_table"].astype(str).unique().tolist())
                    ),
                    "input_raw_cols": "|".join(cols),
                }
            )
    candidate_parquet = OUT_DIR / "pc_source_bank_v1_candidate.parquet"
    candidate_csvgz = OUT_DIR / "pc_source_bank_v1_candidate.csv.gz"
    out_wide.to_parquet(candidate_parquet, index=False)
    out_wide.to_csv(candidate_csvgz, index=False, compression="gzip")
    freeze_tsv = OUT_DIR / "feature_freeze_pc_v1_candidate.tsv"
    pd.DataFrame(freeze_rows).to_csv(freeze_tsv, sep="\t", index=False)
    manifest_tsv = OUT_DIR / "pc_source_bank_v1_candidate_manifest.tsv"
    summary_tsv = OUT_DIR / "pc_source_bank_v1_block_pca_summary.tsv"
    pd.DataFrame(pc_manifest_rows).to_csv(manifest_tsv, sep="\t", index=False)
    pd.DataFrame(block_summary_rows).sort_values("block").to_csv(
        summary_tsv, sep="\t", index=False
    )
    print("candidate_shape =", out_wide.shape)
    print("candidate_parquet =", candidate_parquet)
    print("candidate_csvgz =", candidate_csvgz)
    print("freeze_tsv =", freeze_tsv)
    print("manifest_tsv =", manifest_tsv)
    print("summary_tsv =", summary_tsv)
    print("\n[BLOCK_PCA_SUMMARY]")
    print(pd.DataFrame(block_summary_rows).sort_values("block").to_string(index=False))


def main():
    global MANIFEST, INPUT_MANIFEST, PCA_PLAN, UKB_FINAL, TABLE_PATHS
    global OUT_DIR, BLOCK_DIR, WIDE_INPUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ukb-csv", required=True)
    p.add_argument("--phenotype-dir", required=True)
    p.add_argument("--input-manifest", required=True)
    p.add_argument("--pca-plan", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    MANIFEST = INPUT_MANIFEST = Path(args.input_manifest)
    PCA_PLAN = Path(args.pca_plan)
    UKB_FINAL = Path(args.ukb_csv)
    manifest = pd.read_csv(MANIFEST, sep="\t")
    TABLE_PATHS = {
        name: Path(args.phenotype_dir) / name
        for name in manifest["source_table"].unique()
    }
    for name in TABLE_PATHS:
        if Path(name).name != name:
            raise ValueError("Source table names must be filenames.")
    base = Path(args.out_dir)
    OUT_DIR = base / "pca_inputs"
    BLOCK_DIR = OUT_DIR / "blocks"
    BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    build_inputs()
    WIDE_INPUT = OUT_DIR / "source_bank_v2_pca_input_wide.parquet"
    OUT_DIR = base / "pca_outputs"
    BLOCK_DIR = OUT_DIR / "blocks"
    BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    run_block_pca()


if __name__ == "__main__":
    main()
