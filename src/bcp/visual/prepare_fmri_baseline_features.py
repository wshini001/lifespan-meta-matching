"""Prepare visual-reception fMRI baseline features and align scans by child.

Computes empirical centiles across the supplied feature table, applies a
logit transform and splits positive and negative values into separate columns.
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import argparse


def parse_ids(s):
    out = []
    for x in s.astype(str):
        parts = x.split("_")
        stem = parts[0]
        age = pd.to_numeric(parts[1], errors="coerce") if len(parts) > 1 else np.nan
        m = re.search("(\\d+)$", stem)
        num = str(int(m.group(1))) if m else np.nan
        out.append((x, stem, num, age))
    return pd.DataFrame(
        out, columns=["fmri_id", "fmri_stem", "fmri_numeric_id", "fmri_age_days"]
    )


def centile_logit_split(X):
    out = {}
    _n = X.shape[0]
    for col in X.columns:
        vals = X[col].to_numpy(float)
        cent = np.array([(vals < v).mean() for v in vals], dtype=float)
        logit = np.log((cent + 0.0001) / (1.0001 - cent))
        out[f"fmrirescent_{col}+"] = np.maximum(logit, 0)
        out[f"fmrirescent_{col}-"] = np.minimum(logit, 0)
    return pd.DataFrame(out)


def nearest_match(locked, raw):
    cand = locked[
        ["locked_row", "scan_id", "imaging_numeric_id_str", "scan_age_days_num"]
    ].merge(
        raw[["fmri_id", "fmri_stem", "fmri_numeric_id", "fmri_age_days"]],
        left_on="imaging_numeric_id_str",
        right_on="fmri_numeric_id",
        how="left",
    )
    cand["fmri_age_diff_days"] = (
        cand["scan_age_days_num"] - cand["fmri_age_days"]
    ).abs()
    cand["_sort"] = cand["fmri_age_diff_days"].fillna(1000000000.0)
    cand = cand.sort_values(["locked_row", "_sort", "fmri_age_days", "fmri_id"])
    return cand.groupby("locked_row", as_index=False).first().drop(columns=["_sort"])


def cc_mask(df):
    mask = df["composite_percentile"].notna()
    for c in ["scan_age_months", "sex_final", "FD", "site_primary"]:
        mask &= df[c].notna()
    return mask


def main():
    global LOCKED, FMRI, OUT, THR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-cohort-csv", required=True)
    parser.add_argument("--fmri-res-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-age-diff-days", type=float, default=30.0)
    args = parser.parse_args()
    LOCKED = args.locked_cohort_csv
    FMRI = args.fmri_res_csv
    OUT = Path(args.out_dir)
    THR = args.max_age_diff_days
    OUT.mkdir(exist_ok=True, parents=True)
    locked = pd.read_csv(LOCKED).reset_index(drop=True).reset_index(names="locked_row")
    locked["imaging_numeric_id_str"] = (
        pd.to_numeric(locked["imaging_numeric_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    locked["scan_age_days_num"] = pd.to_numeric(
        locked["scan_age_days"], errors="coerce"
    )
    fmri0 = pd.read_csv(FMRI)
    id_col = fmri0.columns[0]
    parsed = parse_ids(fmri0[id_col])
    feat_cols = [c for c in fmri0.columns if c != id_col]
    X = fmri0[feat_cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("fmri_res.csv has missing or nonnumeric feature cells")
    Xtr = centile_logit_split(X)
    fmri = pd.concat([parsed, Xtr], axis=1)
    fmri_feats = list(Xtr.columns)
    near = nearest_match(locked, fmri)
    full = locked.merge(
        near.drop(columns=["scan_id", "imaging_numeric_id_str", "scan_age_days_num"]),
        on="locked_row",
        how="left",
    ).merge(fmri[["fmri_id"] + fmri_feats], on="fmri_id", how="left")
    tables = {
        "nearest": full,
        "fmrirescent_exact": full[
            full["fmri_id"].notna() & (full["fmri_age_diff_days"] == 0)
        ].copy(),
        "fmrirescent_le30d": full[
            full["fmri_id"].notna() & (full["fmri_age_diff_days"] <= THR)
        ].copy(),
    }
    outputs = {}
    for name, df in tables.items():
        df = df.reset_index(drop=True)
        p = OUT / f"BCP_age1mo_{name}_v1.csv"
        df.to_csv(p, index=False)
        outputs[name] = str(p)
        cc = df.loc[cc_mask(df)].copy().reset_index(drop=True)
        pcc = OUT / f"BCP_age1mo_{name}_composite_fullcov_completecase_v1.csv"
        cc.to_csv(pcc, index=False)
        outputs[name + "_cc"] = str(pcc)
    pd.DataFrame({"feature": fmri_feats}).to_csv(
        OUT / "fmrirescent_feature_manifest.tsv", sep="\t", index=False
    )
    summary = {
        "locked_rows": len(locked),
        "raw_fmri_rows": len(fmri0),
        "raw_fmri_features": len(feat_cols),
        "fmrirescent_features": len(fmri_feats),
        "exact_rows": len(tables["fmrirescent_exact"]),
        "exact_cc": int(cc_mask(tables["fmrirescent_exact"]).sum()),
        "le30d_rows": len(tables["fmrirescent_le30d"]),
        "le30d_cc": int(cc_mask(tables["fmrirescent_le30d"]).sum()),
        "outputs": outputs,
    }
    with open(OUT / "summary_v1.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
