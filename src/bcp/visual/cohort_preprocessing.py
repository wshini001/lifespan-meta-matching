"""Match visual-reception inputs and construct child-grouped folds.

Fits education imputation and outcome nuisance regression on each training
partition. Connectivity inputs are not covariate-residualized.
"""

import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

TARGET = "visual_reception_percentile"
ELIGIBLE = "eligible_visual_reception_percentile_without_tr"
GROUP = "imaging_numeric_id"
KEYS = ["scan_id", GROUP]


def normalize_id(series):
    return series.astype(str).str.replace("\\.0$", "", regex=True)


def grouped_folds(groups, k, seed):
    groups = np.asarray(groups).astype(str)
    uniq, counts = np.unique(groups, return_counts=True)
    if len(uniq) < k:
        raise RuntimeError("Too few children for grouped CV")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    bins, sizes = ([set() for _ in range(k)], np.zeros(k, int))
    for j in order:
        b = int(np.argmin(sizes))
        bins[b].add(uniq[j])
        sizes[b] += int(counts[j])
    idx = np.arange(len(groups))
    return [
        (idx[~np.isin(groups, list(s))], idx[np.isin(groups, list(s))]) for s in bins
    ]


def make_covariates(df):
    age = pd.to_numeric(df["scan_age_months"], errors="coerce").astype(float)
    sex = df["sex_final"].astype(str).str.upper().str.strip().map({"F": 0.0, "M": 1.0})
    site = pd.get_dummies(
        df["site_primary"].astype(str).str.strip(), prefix="site", drop_first=True
    ).astype(float)
    c = pd.DataFrame(
        {
            "scan_age_months": age,
            "age_sq": age**2,
            "sex_male": sex,
            "FD": pd.to_numeric(df["FD"], errors="coerce"),
        },
        index=df.index,
    )
    c = pd.concat([c, site], axis=1)
    c["age_sex"] = c["scan_age_months"] * c["sex_male"]
    c["age_sq_sex"] = c["age_sq"] * c["sex_male"]
    c["mother_edu_gt_college"] = pd.to_numeric(
        df["mother_edu_gt_college"], errors="coerce"
    )
    return c


def impute_train_test(c_tr, c_te):
    c_tr, c_te = (np.asarray(c_tr, float).copy(), np.asarray(c_te, float).copy())
    means = np.nanmean(c_tr, axis=0)
    means[~np.isfinite(means)] = 0.0
    for arr in (c_tr, c_te):
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.take(means, np.where(bad)[1])
    return (c_tr, c_te)


def residualize(y_tr, y_te, c_tr, c_te):
    c_tr, c_te = impute_train_test(c_tr, c_te)
    model = LinearRegression().fit(c_tr, y_tr)
    return (y_tr - model.predict(c_tr), y_te - model.predict(c_te))


def parse_fmri_ids(series):
    rows = []
    for value in series.astype(str):
        parts = value.split("_")
        stem = parts[0]
        age = pd.to_numeric(parts[1], errors="coerce") if len(parts) > 1 else np.nan
        match = re.search("(\\d+)$", stem)
        numeric_id = str(int(match.group(1))) if match else np.nan
        rows.append((value, numeric_id, age))
    return pd.DataFrame(rows, columns=["fmri_id", "fmri_numeric_id", "fmri_age_days"])


def load_and_match(args):
    cohort = pd.read_csv(args.cohort_csv)
    required = KEYS + [
        TARGET,
        ELIGIBLE,
        "scan_age_months",
        "scan_age_days",
        "sex_final",
        "FD",
        "site_primary",
        "mother_edu_gt_college",
    ]
    missing = [c for c in required if c not in cohort]
    if missing:
        raise RuntimeError(f"Missing cohort columns: {missing}")
    keep = cohort[ELIGIBLE].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    cohort = cohort.loc[keep].copy().reset_index(drop=True)
    if (len(cohort), cohort[GROUP].astype(str).nunique()) != (395, 199):
        raise RuntimeError("Eligible cohort is not the locked N=395/199 cohort")
    cohort["_locked_row"] = np.arange(len(cohort))
    cohort["_numeric_id"] = normalize_id(cohort[GROUP])
    cohort["_scan_age_days"] = pd.to_numeric(cohort["scan_age_days"], errors="raise")
    edges = sorted(
        [c for c in cohort if re.fullmatch("e\\d+", str(c))], key=lambda x: int(x[1:])
    )
    if edges != [f"e{i}" for i in range(1, 1486)]:
        raise RuntimeError("Expected ordered e1..e1485")
    fmri = pd.read_csv(args.fmri_res_csv)
    id_col = fmri.columns[0]
    fmri_cols = [c for c in fmri if c != id_col]
    if len(fmri_cols) != 36:
        raise RuntimeError(f"Expected 36 fmri_res features, found {len(fmri_cols)}")
    values = fmri[fmri_cols].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(float)).all():
        raise RuntimeError("Non-finite fmri_res values")
    parsed = parse_fmri_ids(fmri[id_col])
    fmri_table = pd.concat([parsed, values.reset_index(drop=True)], axis=1)
    if fmri_table[["fmri_numeric_id", "fmri_age_days"]].duplicated().any():
        raise RuntimeError("Duplicate fmri subject-age keys")
    cand = cohort[
        ["_locked_row", "scan_id", GROUP, "_numeric_id", "_scan_age_days"]
    ].merge(
        fmri_table[["fmri_id", "fmri_numeric_id", "fmri_age_days"]],
        left_on="_numeric_id",
        right_on="fmri_numeric_id",
        how="left",
    )
    cand["fmri_age_diff_days"] = (cand["_scan_age_days"] - cand["fmri_age_days"]).abs()
    nearest = (
        cand.dropna(subset=["fmri_id"])
        .sort_values(["_locked_row", "fmri_age_diff_days", "fmri_age_days", "fmri_id"])
        .groupby("_locked_row", as_index=False)
        .first()
    )
    nearest = nearest.loc[nearest.fmri_age_diff_days <= args.max_age_diff_days].copy()
    matched = (
        cohort.merge(
            nearest[["_locked_row", "fmri_id", "fmri_age_days", "fmri_age_diff_days"]],
            on="_locked_row",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            fmri_table,
            on=["fmri_id", "fmri_age_days"],
            how="left",
            validate="many_to_one",
        )
        .reset_index(drop=True)
    )
    if args.expected_n > 0 and len(matched) != args.expected_n:
        raise RuntimeError(f"Expected N={args.expected_n}, got {len(matched)}")
    n_children = matched[GROUP].astype(str).nunique()
    if args.expected_children > 0 and n_children != args.expected_children:
        raise RuntimeError(
            f"Expected {args.expected_children} children, got {n_children}"
        )
    if matched[KEYS].astype(str).duplicated().any():
        raise RuntimeError("Duplicate scan/child keys after fmri matching")
    cdf = make_covariates(matched)
    nonedu = [c for c in cdf if c != "mother_edu_gt_college"]
    if cdf[nonedu].isna().any().any():
        raise RuntimeError("Non-education covariate missingness")
    if pd.to_numeric(matched[TARGET], errors="coerce").isna().any():
        raise RuntimeError("Missing target")
    if (
        matched[edges + fmri_cols]
        .apply(pd.to_numeric, errors="coerce")
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError("Missing X values")
    fold_manifest = pd.read_csv(args.fold_manifest_csv)
    left, right = (matched[KEYS].copy(), fold_manifest[KEYS + ["fold"]].copy())
    for key in KEYS:
        left[key], right[key] = (normalize_id(left[key]), normalize_id(right[key]))
    if right[KEYS].duplicated().any():
        raise RuntimeError("Duplicate fold-manifest keys")
    joined = (
        left.reset_index()
        .merge(right, on=KEYS, how="left", validate="one_to_one")
        .sort_values("index")
    )
    if joined.fold.isna().any():
        raise RuntimeError("Missing supplied fold")
    fold_id = joined.fold.astype(int).to_numpy()
    if (
        pd.DataFrame({GROUP: matched[GROUP].astype(str), "fold": fold_id})
        .groupby(GROUP)
        .fold.nunique()
        .max()
        > 1
    ):
        raise RuntimeError("Child leakage in outer folds")
    if sorted(np.unique(fold_id)) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Expected outer folds 0..4")
    audit = {
        "eligible_rows": 395,
        "eligible_children": 199,
        "matched_rows": int(len(matched)),
        "matched_children": int(n_children),
        "max_allowed_age_diff_days": float(args.max_age_diff_days),
        "actual_age_diff_min": float(matched.fmri_age_diff_days.min()),
        "actual_age_diff_median": float(matched.fmri_age_diff_days.median()),
        "actual_age_diff_max": float(matched.fmri_age_diff_days.max()),
        "age_diff_counts": {
            str(k): int(v)
            for k, v in matched.fmri_age_diff_days.value_counts().sort_index().items()
        },
        "mother_edu_observed": int(matched.mother_edu_gt_college.notna().sum()),
        "mother_edu_missing": int(matched.mother_edu_gt_college.isna().sum()),
        "fold_counts": {
            str(k): int(v)
            for k, v in pd.Series(fold_id).value_counts().sort_index().items()
        },
    }
    return (
        matched,
        matched[edges].to_numpy(float),
        matched[fmri_cols].to_numpy(float),
        cdf,
        fold_id,
        audit,
        edges,
        fmri_cols,
    )


def unitize_rows(x):
    x = np.asarray(x, float)
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[~np.isfinite(norm) | (norm == 0)] = 1.0
    return x / norm
