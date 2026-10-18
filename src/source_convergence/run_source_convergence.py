"""Compare fine-motor PC and DNN stage-specific source-feature alignments.

Matches common source features and estimates uncertainty using child bootstrap
resampling within outer folds.
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


def parse_bool(series):
    if series.dtype == bool:
        return series
    parsed = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if parsed.isna().any():
        bad = sorted(series.loc[parsed.isna()].astype(str).unique().tolist())
        raise RuntimeError(f"Could not parse Boolean values: {bad}")
    return parsed.astype(bool)


def clean_name(value):
    raw = str(value)
    explicit = {
        "molested_aschild": "Molested as a child",
        "num_fsisters": "Number of sisters",
        "num_fbrothers": "Number of brothers",
        "birthweightknown": "Birth weight",
        "breastfed_asbaby": "Breastfed as a baby",
        "placeofbirth_nc": "Place of birth: North coordinate",
        "placeofbirth_ec": "Place of birth: East coordinate",
        "maternalsmoke": "Maternal smoking",
    }
    if raw in explicit:
        return explicit[raw]
    value = re.sub("^\\d+_", "", raw)
    value = value.replace("_", " ").replace("/", " / ")
    return re.sub("\\s+", " ", value).strip()


def canonical_text(value):
    value = str(value).strip().lower()
    value = re.sub("__zpred$", "", value)
    return re.sub("[^a-z0-9]+", "", value)


def field_id(value):
    tokens = re.findall("(?<!\\d)(\\d{3,6})(?!\\d)", str(value))
    unique = sorted(set(tokens))
    return unique[0] if len(unique) == 1 else None


def corr_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = x[ok]
    y = y[ok]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rank_average(values):
    return pd.Series(np.asarray(values, float)).rank(method="average").to_numpy()


def spearman_safe(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return corr_safe(rank_average(x[ok]), rank_average(y[ok]))


def column_correlations(matrix, vector):
    matrix = np.asarray(matrix, float)
    vector = np.asarray(vector, float)
    if matrix.ndim != 2 or len(vector) != matrix.shape[0]:
        raise ValueError("column_correlations shape mismatch")
    xm = matrix - np.mean(matrix, axis=0, keepdims=True)
    ym = vector - np.mean(vector)
    denominator = np.sqrt(np.sum(xm * xm, axis=0) * np.sum(ym * ym))
    result = np.full(matrix.shape[1], np.nan, dtype=float)
    good = denominator > 0
    result[good] = np.sum(xm[:, good] * ym[:, None], axis=0) / denominator[good]
    return result


def convergence_metrics(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3:
        raise RuntimeError("Fewer than three finite common-feature estimates")
    nonzero = (x != 0) & (y != 0)
    same_direction = (
        float(np.mean(np.sign(x[nonzero]) == np.sign(y[nonzero])))
        if nonzero.any()
        else np.nan
    )
    return {
        "n_features": int(len(x)),
        "pearson_r": corr_safe(x, y),
        "spearman_rho": spearman_safe(x, y),
        "same_direction_fraction": same_direction,
    }


def center_within_fold(matrix, folds):
    matrix = np.asarray(matrix, float).copy()
    folds = np.asarray(folds)
    for fold in np.unique(folds):
        mask = folds == fold
        matrix[mask] -= np.mean(matrix[mask], axis=0, keepdims=True)
    return matrix


def scalar_ci(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [np.nan, np.nan]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def classify_feature(feature_id):
    value = str(feature_id).lower()
    groups = [
        (
            "Mental health",
            [
                "worry",
                "worrier",
                "guilty",
                "sensitivity",
                "neurotic",
                "tense",
                "miserable",
                "mood",
                "irritab",
                "lonely",
                "confide",
                "risk_taking",
                "risk taking",
                "fed_up",
                "anxious",
                "nervous",
                "embarrassment",
            ],
        ),
        (
            "Early life / social",
            [
                "molested",
                "abuse",
                "hated",
                "siblings",
                "sister",
                "brother",
                "birth",
                "maternal",
                "family",
                "friend",
                "breastfed",
                "child",
            ],
        ),
        (
            "Physical / sleep",
            [
                "blood_pressure",
                "blood pressure",
                "snoring",
                "sleep",
                "nap_",
                "nap ",
                "getting_up",
                "getting up",
                "walked",
                "walking",
                "physical_activity",
                "physical activity",
            ],
        ),
        (
            "Lifestyle / diet",
            [
                "intake",
                "smoked",
                "smoking",
                "alcohol",
                "beer",
                "wine",
                "bread",
                "fish",
                "fruit",
                "meat",
                "driving",
                "computer",
                "watching_tv",
                "outdoors",
                "sun/uv",
                "sun_uv",
            ],
        ),
        (
            "Cognition",
            [
                "fluid_intelligence",
                "reaction",
                "matching",
                "memory",
                "numeric",
                "trail",
                "prospective",
                "cognitive",
                "digits_remembered",
            ],
        ),
    ]
    for domain, keywords in groups:
        if any((keyword in value for keyword in keywords)):
            return domain
    return "Other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-csv", required=True)
    parser.add_argument("--pc-parquet", required=True)
    parser.add_argument("--dnn-parquet", required=True)
    parser.add_argument("--ranking-csv", required=True)
    parser.add_argument("--loadings-csv", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    oof_path = args.oof_csv
    pc_path = args.pc_parquet
    dnn_path = args.dnn_parquet
    dnn_rank_path = args.ranking_csv
    loading_path = args.loadings_csv
    SEED = args.seed
    N_BOOTSTRAP = args.n_bootstrap
    MIN_COMMON_FEATURES = 10
    if N_BOOTSTRAP < 1:
        raise ValueError("n-bootstrap must be positive")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(oof_path)
    pc = pd.read_parquet(pc_path)
    dnn = pd.read_parquet(dnn_path)
    dnn_rank = pd.read_csv(dnn_rank_path)
    loadings = pd.read_csv(loading_path)
    required_oof = {
        "scan_id",
        "imaging_numeric_id",
        "fold",
        "pc_effectively_active",
        "pc12_oof_increment",
        "dnn_M88_oof_increment",
        "total_transfer_oof_increment",
    }
    missing_oof = required_oof.difference(oof.columns)
    if missing_oof:
        raise RuntimeError(f"OOF table missing columns: {sorted(missing_oof)}")
    if len(oof) != 395 or oof["scan_id"].duplicated().any():
        raise RuntimeError("Expected 395 unique locked OOF scans")
    if oof["imaging_numeric_id"].nunique() != 199:
        raise RuntimeError("Expected 199 children in locked OOF table")
    oof["pc_effectively_active"] = parse_bool(oof["pc_effectively_active"])
    pc_columns = [column for column in pc.columns if column.endswith("__zpred")]
    if len(pc_columns) != 12:
        raise RuntimeError(
            f"Expected 12 PC prediction columns; observed {len(pc_columns)}"
        )
    pc_names = [column[:-7] for column in pc_columns]
    dnn_rank["rank"] = pd.to_numeric(dnn_rank["rank"], errors="raise").astype(int)
    rank88 = (
        dnn_rank.loc[dnn_rank["rank"] <= 88].sort_values("rank").reset_index(drop=True)
    )
    if len(rank88) != 88:
        raise RuntimeError(f"Expected fixed DNNRaw Top88; observed {len(rank88)}")
    dnn_targets = rank88["target"].astype(str).tolist()
    dnn_columns = [f"{target}__zpred" for target in dnn_targets]
    for label, frame, columns in [("PC", pc, pc_columns), ("DNNRaw", dnn, dnn_columns)]:
        if frame["scan_id"].duplicated().any():
            raise RuntimeError(f"{label}: duplicate scan_id")
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise RuntimeError(f"{label}: missing columns {missing[:10]}")

    def align_to_oof(frame, columns, label):
        aligned = oof.merge(
            frame[["scan_id"] + columns],
            on="scan_id",
            how="left",
            validate="one_to_one",
        )
        if aligned[columns].isna().any().any():
            raise RuntimeError(f"{label}: missing values after scan alignment")
        return aligned

    pc_aligned = align_to_oof(pc, pc_columns, "PC")
    dnn_aligned = align_to_oof(dnn, dnn_columns, "DNNRaw")
    required_loading = {"pc_name", "block", "raw_col", "loading_aligned_to_saved_pc"}
    missing_loading = required_loading.difference(loadings.columns)
    if missing_loading:
        raise RuntimeError(f"Loading table missing columns: {sorted(missing_loading)}")
    lf = loadings.loc[loadings["pc_name"].astype(str).isin(pc_names)].copy()
    if lf.duplicated(["block", "raw_col", "pc_name"]).any():
        raise RuntimeError("Duplicate block/raw_col/pc loading rows")
    if set(pc_names).difference(lf["pc_name"].astype(str)):
        raise RuntimeError("Aligned loading table does not cover all 12 PC axes")
    lf["raw_key"] = lf["block"].astype(str) + "||" + lf["raw_col"].astype(str)
    loading_frame = (
        lf.pivot(
            index="raw_key", columns="pc_name", values="loading_aligned_to_saved_pc"
        )
        .reindex(columns=pc_names)
        .fillna(0.0)
    )
    raw_keys = loading_frame.index.tolist()
    loading_matrix = loading_frame.to_numpy(float)
    raw_metadata = (
        lf[["raw_key", "block", "raw_col"]]
        .drop_duplicates("raw_key")
        .set_index("raw_key")
        .loc[raw_keys]
        .reset_index()
    )
    if raw_metadata["raw_col"].astype(str).duplicated().any():
        duplicates = raw_metadata.loc[
            raw_metadata["raw_col"].astype(str).duplicated(keep=False),
            ["raw_col", "block"],
        ]
        raise RuntimeError(
            "A raw phenotype appears in more than one PC block; strict one-to-one matching is ambiguous:\n"
            + duplicates.to_string(index=False)
        )
    pc_reconstructed = pc_aligned[pc_columns].to_numpy(float) @ loading_matrix.T
    if not np.isfinite(pc_reconstructed).all():
        raise RuntimeError("PC reconstructed matrix contains NaN/Inf")
    pc_features = raw_metadata.copy()
    pc_features["pc_raw_col"] = pc_features["raw_col"].astype(str)
    pc_features["canonical"] = pc_features["pc_raw_col"].map(canonical_text)
    pc_features["field_id"] = pc_features["pc_raw_col"].map(field_id)
    pc_features["pc_index"] = np.arange(len(pc_features), dtype=int)
    dnn_features = rank88[["rank", "target", "cod_mean_z"]].copy()
    dnn_features["dnn_target"] = dnn_features["target"].astype(str)
    dnn_features["canonical"] = dnn_features["dnn_target"].map(canonical_text)
    dnn_features["field_id"] = dnn_features["dnn_target"].map(field_id)
    dnn_features["dnn_index"] = np.arange(len(dnn_features), dtype=int)
    matches = []
    used_pc = set()
    used_dnn = set()

    def add_unique_matches(key_column, method, require_nonnull=False):
        pc_grouped = {}
        dnn_grouped = {}
        for index, value in pc_features[key_column].items():
            if require_nonnull and (value is None or pd.isna(value)):
                continue
            pc_grouped.setdefault(str(value), []).append(index)
        for index, value in dnn_features[key_column].items():
            if require_nonnull and (value is None or pd.isna(value)):
                continue
            dnn_grouped.setdefault(str(value), []).append(index)
        for value in sorted(set(pc_grouped).intersection(dnn_grouped)):
            p = [index for index in pc_grouped[value] if index not in used_pc]
            d = [index for index in dnn_grouped[value] if index not in used_dnn]
            if len(p) == 1 and len(d) == 1:
                pi, di = (p[0], d[0])
                matches.append((pi, di, method))
                used_pc.add(pi)
                used_dnn.add(di)

    pc_features["exact"] = pc_features["pc_raw_col"]
    dnn_features["exact"] = dnn_features["dnn_target"]
    add_unique_matches("exact", "exact_string")
    add_unique_matches("canonical", "canonical_full_label")
    add_unique_matches("field_id", "unique_UKB_field_id", require_nonnull=True)
    match_rows = []
    for pc_row_index, dnn_row_index, method in matches:
        p = pc_features.loc[pc_row_index]
        d = dnn_features.loc[dnn_row_index]
        loading_vector = loading_matrix[int(p["pc_index"])]
        dominant_pc = pc_names[int(np.argmax(np.abs(loading_vector)))]
        match_rows.append(
            {
                "pc_index": int(p["pc_index"]),
                "dnn_index": int(d["dnn_index"]),
                "pc_raw_key": str(p["raw_key"]),
                "pc_block": str(p["block"]),
                "pc_raw_col": str(p["pc_raw_col"]),
                "dnn_target": str(d["dnn_target"]),
                "dnn_source_rank": int(d["rank"]),
                "dnn_source_UKB_COD": float(d["cod_mean_z"]),
                "match_method": method,
                "feature": clean_name(d["dnn_target"]),
                "domain": classify_feature(d["dnn_target"]),
                "dominant_pc": dominant_pc,
                "dominant_pc_abs_loading": float(np.max(np.abs(loading_vector))),
            }
        )
    match_table = pd.DataFrame(match_rows)
    if len(match_table) < MIN_COMMON_FEATURES:
        audit_path = out / "fine_motor_PC_DNN_common_feature_match_audit_v2.csv"
        match_table.to_csv(audit_path, index=False)
        raise RuntimeError(
            f"Only {len(match_table)} strict common features were found; expected at least {MIN_COMMON_FEATURES}. Audit written to {audit_path}. No convergence statistics were written."
        )
    if (
        match_table["pc_index"].duplicated().any()
        or match_table["dnn_index"].duplicated().any()
    ):
        raise RuntimeError("Strict common-feature mapping is not one-to-one")
    match_table = match_table.sort_values("dnn_source_rank").reset_index(drop=True)
    match_path = out / "fine_motor_PC_DNN_common_feature_match_audit_v2.csv"
    match_table.to_csv(match_path, index=False)
    active_mask = pc_aligned["pc_effectively_active"].to_numpy(bool)
    if int(active_mask.sum()) != 315:
        raise RuntimeError(
            f"Expected 315 PC-active scans; observed {active_mask.sum()}"
        )
    active_meta = oof.loc[
        active_mask,
        [
            "imaging_numeric_id",
            "fold",
            "pc12_oof_increment",
            "dnn_M88_oof_increment",
            "total_transfer_oof_increment",
        ],
    ].copy()
    if sorted(active_meta["fold"].unique().tolist()) != [1, 2, 3, 4]:
        raise RuntimeError("Primary common sample must contain active folds 1--4")
    if (active_meta.groupby("imaging_numeric_id")["fold"].nunique() > 1).any():
        raise RuntimeError("A child occurs in more than one outer fold")
    pc_indices = match_table["pc_index"].to_numpy(int)
    dnn_indices = match_table["dnn_index"].to_numpy(int)
    pc_scan_matrix = pc_reconstructed[active_mask][:, pc_indices]
    dnn_scan_matrix = dnn_aligned.loc[active_mask, dnn_columns].to_numpy(float)[
        :, dnn_indices
    ]
    child_ids = sorted(active_meta["imaging_numeric_id"].unique().tolist())
    child_index = {child: index for index, child in enumerate(child_ids)}
    groups = active_meta["imaging_numeric_id"].map(child_index).to_numpy(int)
    n_children = len(child_ids)
    if n_children != 155:
        raise RuntimeError(f"Expected 155 PC-active children; observed {n_children}")

    def aggregate_matrix_by_child(matrix):
        frame = pd.DataFrame(np.asarray(matrix, float))
        frame.insert(0, "child_index", groups)
        return (
            frame.groupby("child_index", sort=True)
            .mean()
            .reindex(range(n_children))
            .to_numpy(float)
        )

    def aggregate_vector_by_child(values):
        frame = pd.DataFrame(
            {"child_index": groups, "value": np.asarray(values, float)}
        )
        return (
            frame.groupby("child_index", sort=True)["value"]
            .mean()
            .reindex(range(n_children))
            .to_numpy(float)
        )

    X_pc = aggregate_matrix_by_child(pc_scan_matrix)
    X_dnn = aggregate_matrix_by_child(dnn_scan_matrix)
    y_pc = aggregate_vector_by_child(active_meta["pc12_oof_increment"])
    y_dnn = aggregate_vector_by_child(active_meta["dnn_M88_oof_increment"])
    y_total = aggregate_vector_by_child(active_meta["total_transfer_oof_increment"])
    child_fold = (
        active_meta.assign(child_index=groups)
        .groupby("child_index", sort=True)["fold"]
        .first()
        .reindex(range(n_children))
        .to_numpy(int)
    )
    for label, array in [
        ("PC child matrix", X_pc),
        ("DNN child matrix", X_dnn),
        ("PC increment", y_pc),
        ("DNN increment", y_dnn),
        ("total increment", y_total),
    ]:
        if not np.isfinite(array).all():
            raise RuntimeError(f"{label} contains NaN/Inf")
    pc_stage_r = column_correlations(X_pc, y_pc)
    dnn_stage_r = column_correlations(X_dnn, y_dnn)
    pc_total_r = column_correlations(X_pc, y_total)
    dnn_total_r = column_correlations(X_dnn, y_total)
    representation_r = np.array(
        [corr_safe(X_pc[:, index], X_dnn[:, index]) for index in range(X_pc.shape[1])]
    )
    feature_results = match_table.copy()
    feature_results["pc_stage_child_balanced_r"] = pc_stage_r
    feature_results["dnn_stage_child_balanced_r"] = dnn_stage_r
    feature_results["pc_total_child_balanced_r"] = pc_total_r
    feature_results["dnn_total_child_balanced_r"] = dnn_total_r
    feature_results["pc_dnn_representation_r"] = representation_r
    feature_results["same_stage_direction"] = np.sign(pc_stage_r) == np.sign(
        dnn_stage_r
    )
    feature_results["concordant_strength"] = np.minimum(
        np.abs(pc_stage_r), np.abs(dnn_stage_r)
    )
    feature_results["quadrant"] = np.select(
        [
            (pc_stage_r > 0) & (dnn_stage_r > 0),
            (pc_stage_r < 0) & (dnn_stage_r < 0),
            (pc_stage_r >= 0) & (dnn_stage_r < 0),
            (pc_stage_r < 0) & (dnn_stage_r >= 0),
        ],
        [
            "concordant_positive",
            "concordant_negative",
            "discordant_+/-",
            "discordant_-/+",
        ],
        default="zero",
    )
    primary = convergence_metrics(pc_stage_r, dnn_stage_r)
    total_sensitivity = convergence_metrics(pc_total_r, dnn_total_r)

    def dominant_pc_balanced_metrics(x_values, y_values):
        grouped = (
            pd.DataFrame(
                {
                    "dominant_pc": feature_results["dominant_pc"].to_numpy(),
                    "x": np.asarray(x_values, float),
                    "y": np.asarray(y_values, float),
                }
            )
            .groupby("dominant_pc", as_index=False)
            .agg(x=("x", "mean"), y=("y", "mean"))
        )
        metrics = convergence_metrics(grouped["x"], grouped["y"])
        metrics["n_dominant_pc_groups"] = int(len(grouped))
        return (metrics, grouped)

    dominant_pc_balanced, dominant_pc_grouped = dominant_pc_balanced_metrics(
        pc_stage_r, dnn_stage_r
    )
    X_pc_wc = center_within_fold(X_pc, child_fold)
    X_dnn_wc = center_within_fold(X_dnn, child_fold)
    y_pc_wc = center_within_fold(y_pc[:, None], child_fold)[:, 0]
    y_dnn_wc = center_within_fold(y_dnn[:, None], child_fold)[:, 0]
    pc_wc_r = column_correlations(X_pc_wc, y_pc_wc)
    dnn_wc_r = column_correlations(X_dnn_wc, y_dnn_wc)
    within_fold_centered = convergence_metrics(pc_wc_r, dnn_wc_r)
    lofo_rows = []
    for held_out_fold in [1, 2, 3, 4]:
        keep = child_fold != held_out_fold
        x = column_correlations(X_pc[keep], y_pc[keep])
        y = column_correlations(X_dnn[keep], y_dnn[keep])
        metrics = convergence_metrics(x, y)
        metrics.update(
            {
                "analysis": "leave_one_active_fold_out",
                "held_out_fold": held_out_fold,
                "n_children": int(keep.sum()),
            }
        )
        lofo_rows.append(metrics)
    lofo = pd.DataFrame(lofo_rows)
    rng = np.random.default_rng(SEED)
    fold_indices = {fold: np.flatnonzero(child_fold == fold) for fold in [1, 2, 3, 4]}
    bootstrap = np.empty((N_BOOTSTRAP, 5), dtype=float)
    print(
        f"Running {N_BOOTSTRAP:,} fold-stratified child bootstraps across {len(feature_results)} fixed common features"
    )
    for iteration in range(N_BOOTSTRAP):
        sampled = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in fold_indices.values()
            ]
        )
        x = column_correlations(X_pc[sampled], y_pc[sampled])
        y = column_correlations(X_dnn[sampled], y_dnn[sampled])
        stage = convergence_metrics(x, y)
        xt = column_correlations(X_pc[sampled], y_total[sampled])
        yt = column_correlations(X_dnn[sampled], y_total[sampled])
        total = convergence_metrics(xt, yt)
        pc_balanced, _ = dominant_pc_balanced_metrics(x, y)
        bootstrap[iteration] = [
            stage["pearson_r"],
            stage["spearman_rho"],
            stage["same_direction_fraction"],
            total["pearson_r"],
            pc_balanced["pearson_r"],
        ]
    bootstrap_frame = pd.DataFrame(
        bootstrap,
        columns=[
            "stage_pearson_r",
            "stage_spearman_rho",
            "stage_same_direction_fraction",
            "total_pearson_r",
            "dominant_PC_balanced_pearson_r",
        ],
    )
    summary = {
        "primary_stage_specific_convergence": primary,
        "primary_stage_specific_bootstrap_95_CI": {
            "pearson_r": scalar_ci(bootstrap_frame.stage_pearson_r),
            "spearman_rho": scalar_ci(bootstrap_frame.stage_spearman_rho),
            "same_direction_fraction": scalar_ci(
                bootstrap_frame.stage_same_direction_fraction
            ),
        },
        "same_total_increment_sensitivity": total_sensitivity,
        "within_fold_centered_sensitivity": within_fold_centered,
        "sample": {
            "n_scans": int(active_mask.sum()),
            "n_children": n_children,
            "active_folds": [1, 2, 3, 4],
        },
        "settings": {"seed": SEED, "n_bootstrap": N_BOOTSTRAP},
    }
    feature_results.to_csv(out / "feature_convergence.csv", index=False)
    bootstrap_frame.to_csv(out / "bootstrap.csv.gz", index=False, compression="gzip")
    lofo.to_csv(out / "leave_one_fold_out.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print("PASS: stage-specific source convergence calculated")


if __name__ == "__main__":
    main()
