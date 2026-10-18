"""Align BCP scans with surface-area measurements.

Applies full-table empirical-centile, logit and positive-negative transforms,
then matches measurements by child and age. Produces exact-match and
nearest-age cohort tables.
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


def parse_ids(s):
    rows = []
    for x in s.astype(str):
        parts = x.split("_")
        stem = parts[0]
        age = pd.to_numeric(parts[1], errors="coerce") if len(parts) > 1 else np.nan
        m = re.search("(\\d+)$", stem)
        numeric_id = str(int(m.group(1))) if m else np.nan
        rows.append((x, stem, numeric_id, age))
    return pd.DataFrame(
        rows, columns=["res_id", "res_stem", "res_numeric_id", "res_age_days"]
    )


def centile_logit_split(X, prefix="rescentarea"):
    out = {}
    for col in X.columns:
        vals = X[col].to_numpy(float)
        cent = np.array([(vals < v).mean() for v in vals], dtype=float)
        logit = np.log((cent + 0.0001) / (1.0001 - cent))
        safe_col = str(col).replace(" ", "_")
        out[f"{prefix}_{safe_col}+"] = np.maximum(logit, 0)
        out[f"{prefix}_{safe_col}-"] = np.minimum(logit, 0)
    return pd.DataFrame(out)


def nearest_match(locked, res):
    cand = locked[
        ["locked_row", "scan_id", "imaging_numeric_id_str", "scan_age_days_num"]
    ].merge(
        res[["res_id", "res_stem", "res_numeric_id", "res_age_days"]],
        left_on="imaging_numeric_id_str",
        right_on="res_numeric_id",
        how="left",
    )
    cand["res_age_diff_days"] = (cand["scan_age_days_num"] - cand["res_age_days"]).abs()
    cand["_sort"] = cand["res_age_diff_days"].fillna(1000000000.0)
    cand = cand.sort_values(["locked_row", "_sort", "res_age_days", "res_id"])
    cand = cand.groupby("locked_row", as_index=False).first().drop(columns=["_sort"])
    return cand


def completecase_mask(df):
    mask = df["composite_percentile"].notna()
    for c in ["scan_age_months", "sex_final", "FD", "site_primary"]:
        mask &= df[c].notna()
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locked-cohort-csv", required=True)
    p.add_argument("--area-csv", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    LOCKED = args.locked_cohort_csv
    RES = args.area_csv
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    THR_DAYS = 30
    locked = pd.read_csv(LOCKED).reset_index(drop=True).reset_index(names="locked_row")
    locked["imaging_numeric_id_str"] = (
        pd.to_numeric(locked["imaging_numeric_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    locked["scan_age_days_num"] = pd.to_numeric(
        locked["scan_age_days"], errors="coerce"
    )
    res0 = pd.read_csv(RES)
    id_col = res0.columns[0]
    parsed = parse_ids(res0[id_col])
    area_cols = [c for c in res0.columns if c.startswith("Area_")]
    if len(area_cols) == 0:
        raise RuntimeError("No Area_ columns found in res.csv")
    area_raw = res0[area_cols].apply(pd.to_numeric, errors="coerce")
    if area_raw.isna().any().any():
        bad = area_raw.columns[area_raw.isna().any(axis=0)].tolist()[:20]
        raise RuntimeError(f"Area columns contain missing/non-numeric values: {bad}")
    area_trans = centile_logit_split(area_raw, prefix="rescentarea")
    res_area = pd.concat([parsed, area_trans], axis=1)
    area_feats = list(area_trans.columns)
    near = nearest_match(locked, res_area)
    full = locked.merge(
        near.drop(columns=["scan_id", "imaging_numeric_id_str", "scan_age_days_num"]),
        on="locked_row",
        how="left",
    ).merge(res_area[["res_id"] + area_feats], on="res_id", how="left")
    tables = {
        "nearest": full,
        "area_exact": full[
            full["res_id"].notna() & (full["res_age_diff_days"] == 0)
        ].copy(),
        "area_le30d": full[
            full["res_id"].notna() & (full["res_age_diff_days"] <= THR_DAYS)
        ].copy(),
    }
    outputs = {}
    for name, df in tables.items():
        df = df.reset_index(drop=True)
        p = OUT / f"BCP_age1mo_rescentarea_{name}_v1.csv"
        df.to_csv(p, index=False)
        outputs[name] = str(p)
        cc = df.loc[completecase_mask(df)].copy().reset_index(drop=True)
        pcc = (
            OUT / f"BCP_age1mo_rescentarea_{name}_composite_fullcov_completecase_v1.csv"
        )
        cc.to_csv(pcc, index=False)
        outputs[name + "_cc"] = str(pcc)
    pd.DataFrame({"feature": area_feats}).to_csv(
        OUT / "rescentarea_feature_manifest.tsv", sep="\t", index=False
    )
    summary_rows = []
    for thr in [0, 1, 7, 14, 30, 60, 90]:
        sub = full[full["res_id"].notna() & (full["res_age_diff_days"] <= thr)].copy()
        summary_rows.append(
            {
                "res_age_diff_threshold_days": thr,
                "n_scans": len(sub),
                "n_unique_subjects": sub["imaging_numeric_id"].astype(str).nunique(),
                "composite_nonmissing": int(sub["composite_percentile"].notna().sum()),
                "composite_fullcov_completecase": (
                    int(completecase_mask(sub).sum()) if len(sub) else 0
                ),
                "median_age_diff_days": (
                    float(sub["res_age_diff_days"].median()) if len(sub) else np.nan
                ),
                "max_age_diff_days": (
                    float(sub["res_age_diff_days"].max()) if len(sub) else np.nan
                ),
            }
        )
    threshold_summary = pd.DataFrame(summary_rows)
    threshold_summary.to_csv(OUT / "rescentarea_threshold_summary_v1.csv", index=False)
    summary = {
        "inputs": {"locked_csv": LOCKED, "res_csv": RES, "threshold_days": THR_DAYS},
        "locked": {
            "rows": int(len(locked)),
            "unique_scans": int(locked["scan_id"].nunique()),
            "unique_subjects": int(locked["imaging_numeric_id"].astype(str).nunique()),
            "n_edge_cols": int(
                len(
                    [c for c in locked.columns if c.startswith("e") and c[1:].isdigit()]
                )
            ),
        },
        "res": {
            "rows": int(len(res0)),
            "area_raw_features": int(len(area_cols)),
            "rescentarea_features": int(len(area_feats)),
        },
        "matching": {
            "nearest_matched_rows": int(full["res_id"].notna().sum()),
            "exact_rows": int(len(tables["area_exact"])),
            "exact_completecase": int(completecase_mask(tables["area_exact"]).sum()),
            "le30d_rows": int(len(tables["area_le30d"])),
            "le30d_completecase": int(completecase_mask(tables["area_le30d"]).sum()),
        },
        "outputs": outputs,
    }
    with open(OUT / "summary_v1.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
