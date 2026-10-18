"""Calculate direct and PC-reconstructed source-feature correlations with ABCD prediction increments."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def corr_np(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    x = x[m]
    y = y[m]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def clean_label(x):
    s = str(x)
    s = s.replace("_", " ")
    s = s.replace(".", " ")
    s = s.replace("/", " / ")
    s = " ".join(s.split())
    return s


def direct_domain(feature):
    f = feature.lower()
    if "blood_pressure" in f:
        return "physiology_direct"
    if any((k in f for k in ["sleep", "snoring", "getting_up", "nap"])):
        return "sleep_circadian_direct"
    if any((k in f for k in ["meat", "bread", "water", "fish", "cheese"])):
        return "diet_lifestyle_direct"
    if any(
        (
            k in f
            for k in ["computer", "risk", "outdoors", "sun", "mobile", "phone", "tv"]
        )
    ):
        return "lifestyle_behavior_direct"
    return "dnn_raw_direct"


def fold_corr_summary(score, target, fold, global_corr):
    vals = []
    for f in sorted(pd.unique(fold)):
        m = fold == f
        vals.append(corr_np(score[m], target[m]))
    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    if np.isfinite(global_corr) and global_corr != 0:
        sign_consistency = float(np.mean(np.sign(finite) == np.sign(global_corr)))
    else:
        sign_consistency = np.nan
    return (
        float(np.nanmean(finite)),
        float(np.nanmin(finite)),
        float(np.nanmax(finite)),
        sign_consistency,
    )


def make_record(
    source_type, domain, feature_id, display_feature, score, pred_df, extra=None
):
    y_inc = pred_df["incremental_vs_abcd"].to_numpy()
    y_final = pred_df["y_pred_ensemble_nested"].to_numpy()
    y_abcd = pred_df["y_pred_abcd_nested"].to_numpy()
    y_true = pred_df["y_true_resid"].to_numpy()
    fold = pred_df["fold"].to_numpy()
    r_inc = corr_np(score, y_inc)
    r_final = corr_np(score, y_final)
    r_abcd = corr_np(score, y_abcd)
    r_true = corr_np(score, y_true)
    mean_fc, min_fc, max_fc, sign_cons = fold_corr_summary(score, y_inc, fold, r_inc)
    row = {
        "source_type": source_type,
        "domain": domain,
        "feature_id": feature_id,
        "display_feature": display_feature,
        "corr_with_incremental_vs_abcd": r_inc,
        "abs_corr_with_incremental_vs_abcd": (
            abs(r_inc) if np.isfinite(r_inc) else np.nan
        ),
        "corr_with_final_ensemble_pred": r_final,
        "abs_corr_with_final_ensemble_pred": (
            abs(r_final) if np.isfinite(r_final) else np.nan
        ),
        "corr_with_abcd_only_pred": r_abcd,
        "corr_with_observed_residual_iq": r_true,
        "abs_corr_with_observed_residual_iq": (
            abs(r_true) if np.isfinite(r_true) else np.nan
        ),
        "mean_fold_corr_with_incremental_vs_abcd": mean_fc,
        "min_fold_corr_with_incremental_vs_abcd": min_fc,
        "max_fold_corr_with_incremental_vs_abcd": max_fc,
        "fold_sign_consistency": sign_cons,
        "reconstructed_score_mean": float(np.nanmean(score)),
        "reconstructed_score_sd": float(np.nanstd(score)),
    }
    if extra:
        row.update(extra)
    return row


def calculate_alignments(df, freeze, pc_loadings, disease_loadings):
    _final12_pcs = [
        x for x in freeze if "__pc" in x and (not x.startswith("disease_phecode__pc"))
    ]
    disease_pcs = [x for x in freeze if x.startswith("disease_phecode__pc")]
    dnn_raw = [x for x in freeze if "__pc" not in x]
    rows = []
    load = pc_loadings
    required_load_cols = ["block", "raw_col", "pc_name", "loading_aligned_to_saved_pc"]
    for c in required_load_cols:
        if c not in load.columns:
            raise KeyError(f"Missing final12 loading column: {c}")
    load = pc_loadings
    for raw_col, sub in load.groupby("raw_col"):
        score = np.zeros(df.shape[0], dtype=float)
        pcs_used = []
        loadings_used = []
        block = str(sub["block"].iloc[0])
        for _, r in sub.iterrows():
            pc = str(r["pc_name"])
            zc = f"{pc}__zpred"
            if zc in df.columns:
                w = float(r["loading_aligned_to_saved_pc"])
                score += w * df[zc].to_numpy(dtype=float)
                pcs_used.append(pc)
                loadings_used.append(w)
        if len(pcs_used) == 0:
            continue
        display = clean_label(raw_col)
        rows.append(
            make_record(
                source_type="final12_pc_reconstructed_raw",
                domain=block,
                feature_id=raw_col,
                display_feature=display,
                score=score,
                pred_df=df,
                extra={
                    "n_model_features_used_for_reconstruction": len(pcs_used),
                    "model_features_used_for_reconstruction": "|".join(pcs_used),
                    "max_abs_loading_used": float(np.max(np.abs(loadings_used))),
                },
            )
        )
    for f in dnn_raw:
        zc = f"{f}__zpred"
        score = df[zc].to_numpy(dtype=float)
        rows.append(
            make_record(
                source_type="dnn_raw_direct",
                domain=direct_domain(f),
                feature_id=f,
                display_feature=clean_label(f),
                score=score,
                pred_df=df,
                extra={
                    "n_model_features_used_for_reconstruction": 1,
                    "model_features_used_for_reconstruction": f,
                    "max_abs_loading_used": np.nan,
                },
            )
        )
    dl = disease_loadings
    pc_cols = [c for c in disease_pcs if c in dl.columns]
    if len(pc_cols) == 0:
        raise KeyError(
            "No selected disease PC loading columns found in disease loading file."
        )
    phecode_col = "phecode" if "phecode" in dl.columns else dl.columns[0]
    phenotype_col = "phenotype" if "phenotype" in dl.columns else None
    category_col = "category" if "category" in dl.columns else None
    disease_z = np.column_stack(
        [df[f"{pc}__zpred"].to_numpy(dtype=float) for pc in pc_cols]
    )
    for _, r in dl.iterrows():
        weights = (
            pd.to_numeric(r[pc_cols], errors="coerce").fillna(0).to_numpy(dtype=float)
        )
        if np.all(weights == 0):
            continue
        score = disease_z @ weights
        phe = str(r[phecode_col])
        phenotype = str(r[phenotype_col]) if phenotype_col else ""
        category = str(r[category_col]) if category_col else "disease_phecode"
        if phenotype and phenotype.lower() != "nan":
            display = f"{phe}: {phenotype}"
        else:
            display = phe
        rows.append(
            make_record(
                source_type="disease_phecode_reconstructed",
                domain=f"disease::{category}",
                feature_id=phe,
                display_feature=display,
                score=score,
                pred_df=df,
                extra={
                    "n_model_features_used_for_reconstruction": len(pc_cols),
                    "model_features_used_for_reconstruction": "|".join(pc_cols),
                    "max_abs_loading_used": float(np.max(np.abs(weights))),
                },
            )
        )
    all_df = pd.DataFrame(rows)
    all_df = all_df.sort_values(
        "abs_corr_with_incremental_vs_abcd", ascending=False
    ).reset_index(drop=True)
    all_df.insert(
        0, "rank_abs_corr_with_incremental_vs_abcd", np.arange(1, all_df.shape[0] + 1)
    )
    return all_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--feature-parquet", required=True)
    parser.add_argument("--freeze-tsv", required=True)
    parser.add_argument("--pc-loadings-csv", required=True)
    parser.add_argument("--disease-loadings-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    pred = pd.read_csv(args.predictions_csv)
    feat = pd.read_parquet(args.feature_parquet)
    freeze = [
        line.strip()
        for line in Path(args.freeze_tsv).read_text().splitlines()
        if line.strip()
    ]
    if len(freeze) != 43 or len(set(freeze)) != 43:
        raise ValueError("Expected 43 unique frozen feature names")
    counts = (
        sum(
            ("__pc" in x and (not x.startswith("disease_phecode__pc")) for x in freeze)
        ),
        sum(("__pc" not in x for x in freeze)),
        sum((x.startswith("disease_phecode__pc") for x in freeze)),
    )
    if counts != (12, 15, 16):
        raise ValueError(f"Unexpected source composition: {counts}")
    needed = [
        "_orig_row_idx",
        "y_true_resid",
        "y_pred_abcd_nested",
        "y_pred_ensemble_nested",
        "fold",
    ]
    zcols = [f"{x}__zpred" for x in freeze]
    for frame, name in [(pred, "predictions"), (feat, "feature bank")]:
        if (
            frame["_orig_row_idx"].isna().any()
            or frame["_orig_row_idx"].duplicated().any()
        ):
            raise ValueError(f"{name}: missing or duplicate row IDs")
    df = pred[needed].merge(
        feat[["_orig_row_idx"] + zcols],
        on="_orig_row_idx",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(df) != 9954 or not df["_merge"].eq("both").all():
        raise ValueError("Expected 9,954 fully matched prediction rows")
    if not np.isfinite(df[zcols + needed[1:]].to_numpy(float)).all():
        raise ValueError("Non-finite prediction, outcome, fold or source values")
    df["incremental_vs_abcd"] = df["y_pred_ensemble_nested"] - df["y_pred_abcd_nested"]
    pc_loadings = pd.read_csv(args.pc_loadings_csv)
    disease_loadings = pd.read_csv(args.disease_loadings_tsv, sep="\t")
    result = calculate_alignments(df, freeze, pc_loadings, disease_loadings)
    counts = result["source_type"].value_counts().to_dict()
    expected = {
        "final12_pc_reconstructed_raw": 112,
        "dnn_raw_direct": 15,
        "disease_phecode_reconstructed": 330,
    }
    if counts != expected:
        raise ValueError(f"Unexpected reconstruction counts: {counts}")
    keys = ["source_type", "feature_id"]
    if result.duplicated(keys).any():
        raise ValueError("Duplicate source/feature output keys")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "final43_source_alignments.tsv", sep="\t", index=False)
    print("PASS: final43 direct and reconstructed source alignments calculated")


if __name__ == "__main__":
    main()
