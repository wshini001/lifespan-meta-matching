"""Rank disease targets by average held-out UKB predictive COD across five seeds."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def normalize_target(x):
    return f"{float(x):.2f}"


def to_bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    z = s.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not z.isin(mapping).all():
        bad = sorted(z.loc[~z.isin(mapping)].unique().tolist())
        raise ValueError(f"Cannot parse boolean values: {bad}")
    return z.map(mapping).astype(bool)


def load_universe(path, expected_n):
    df = pd.read_csv(path, sep="\t")
    candidates = ["phecode_key", "phecode_norm", "phecode", "target"]
    col = None
    for c in candidates:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise RuntimeError(
            f"Cannot identify phecode column in {path}; columns={list(df.columns)}"
        )
    values = {normalize_target(x) for x in df[col]}
    if len(values) != expected_n:
        raise RuntimeError(
            f"{path}: expected {expected_n} unique targets, found {len(values)}"
        )
    return values


def read_seed(path, seed):
    df = pd.read_csv(path)
    required = [
        "seed",
        "target_index",
        "target",
        "best_alpha",
        "residualize_y",
        "refit_on_train_validation",
        "n_inner_train",
        "n_validation",
        "n_outer_test",
        "predictive_COD_valid_z",
        "predictive_COD_test_z",
        "Pearson_r_test_z",
        "AUC_raw_disease_test",
        "bundle_path",
        "kernel_type",
        "feature_transform",
        "selection_metric",
        "target_space",
        "alpha_resolution_stage",
        "extreme_shrinkage_boundary",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"seed{seed}: missing columns {missing}")
    if len(df) != 330:
        raise RuntimeError(f"seed{seed}: expected N=330; found {len(df)}")
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["target_index"] = pd.to_numeric(df["target_index"], errors="raise").astype(int)
    df["best_alpha"] = pd.to_numeric(df["best_alpha"], errors="raise").astype(float)
    df["target_key"] = df["target"].map(normalize_target)
    df["residualize_y"] = to_bool_series(df["residualize_y"])
    df["refit_on_train_validation"] = to_bool_series(df["refit_on_train_validation"])
    df["extreme_shrinkage_boundary"] = to_bool_series(df["extreme_shrinkage_boundary"])
    if not (df["seed"] == seed).all():
        raise RuntimeError(f"seed{seed}: seed field mismatch")
    if df["target_index"].nunique() != 330:
        raise RuntimeError(f"seed{seed}: duplicate target_index")
    if set(df["target_index"]) != set(range(330)):
        raise RuntimeError(f"seed{seed}: target index contract failed")
    if df["target_key"].nunique() != 330:
        raise RuntimeError(f"seed{seed}: duplicate normalized target labels")
    if not df["residualize_y"].all():
        raise RuntimeError(f"seed{seed}: non-residualized row found")
    if df["refit_on_train_validation"].any():
        raise RuntimeError(f"seed{seed}: unexpected refit found")
    if not (df["n_inner_train"] == 15787).all():
        raise RuntimeError(f"seed{seed}: inner-train N contract failed")
    if not (df["n_validation"] == 3947).all():
        raise RuntimeError(f"seed{seed}: validation N contract failed")
    if not (df["n_outer_test"] == 4934).all():
        raise RuntimeError(f"seed{seed}: outer-test N contract failed")
    if not (df["selection_metric"] == "predictive_COD_valid_z").all():
        raise RuntimeError(f"seed{seed}: selection metric mismatch")
    if not (df["target_space"] == "covariate_residual_z").all():
        raise RuntimeError(f"seed{seed}: target-space mismatch")
    if not (df["kernel_type"] == "Pearson_correlation_precomputed").all():
        raise RuntimeError(f"seed{seed}: kernel mismatch")
    if not (df["feature_transform"] == "row_center_then_l2_unitize").all():
        raise RuntimeError(f"seed{seed}: feature transform mismatch")
    bad_extreme = df.loc[
        df["extreme_shrinkage_boundary"] & ~np.isclose(df["best_alpha"], 100000.0)
    ]
    if len(bad_extreme):
        raise RuntimeError(f"seed{seed}: extreme-shrinkage flag without alpha=100000")
    bad_100k = df.loc[
        np.isclose(df["best_alpha"], 100000.0) & ~df["extreme_shrinkage_boundary"]
    ]
    if len(bad_100k):
        raise RuntimeError(f"seed{seed}: alpha=100000 without extreme-shrinkage flag")
    missing_bundles = []
    for _, r in df.iterrows():
        p = Path(str(r["bundle_path"]))
        if not p.is_file():
            missing_bundles.append((int(r["target_index"]), str(p)))
    if missing_bundles:
        print("Missing bundle examples:")
        for x in missing_bundles[:10]:
            print(x)
        raise RuntimeError(f"seed{seed}: {len(missing_bundles)} bundles missing")
    return df.sort_values("target_index").reset_index(drop=True)


def fisher_mean(r):
    r = np.asarray(r, dtype=float)
    r = np.clip(r, -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(r))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0-csv", required=True)
    ap.add_argument("--resolved-root", required=True)
    ap.add_argument("--brain50", required=True)
    ap.add_argument("--brain19", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {0: Path(args.seed0_csv)}
    root = Path(args.resolved_root)
    for s in [1, 2, 3, 4]:
        paths[s] = root / f"seed{s}" / f"phase5c_seed{s}_resolved_all330_v5.csv"
    for s, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"seed{s}: {p}")
    seed_dfs = {}
    print("===== FIVE-SEED INPUT CONTRACT =====")
    for s in range(5):
        df = read_seed(paths[s], s)
        seed_dfs[s] = df
        print(
            f"seed{s}: N={len(df)} stage2={int((df['alpha_resolution_stage'] == 'stage2_highalpha').sum())} extreme={int(df['extreme_shrinkage_boundary'].sum())}"
        )
    ref = seed_dfs[0][["target_index", "target_key"]].set_index("target_index")[
        "target_key"
    ]
    for s in [1, 2, 3, 4]:
        cur = seed_dfs[s][["target_index", "target_key"]].set_index("target_index")[
            "target_key"
        ]
        if not ref.equals(cur):
            bad = ref.index[ref != cur].tolist()
            raise RuntimeError(
                f"seed{s}: target identity mismatch; first indices={bad[:10]}"
            )
    print("PASS exact target identity across seeds0-4")
    long = pd.concat([seed_dfs[s] for s in range(5)], axis=0, ignore_index=True)
    if len(long) != 1650:
        raise RuntimeError(f"Expected 1650 seed-target rows; got {len(long)}")
    long_out = outdir / "phase5c_krr_residualized_5seed_long_v5.csv"
    long.to_csv(long_out, index=False)
    seed_rows = []
    for s in range(5):
        d = seed_dfs[s]
        seed_rows.append(
            {
                "seed": s,
                "n_targets": len(d),
                "n_stage2": int(
                    (d["alpha_resolution_stage"] == "stage2_highalpha").sum()
                ),
                "n_extreme_shrinkage": int(d["extreme_shrinkage_boundary"].sum()),
                "mean_test_COD": float(d["predictive_COD_test_z"].mean()),
                "mean_test_r": float(d["Pearson_r_test_z"].mean()),
                "mean_test_AUC": float(d["AUC_raw_disease_test"].mean()),
                "n_test_COD_gt0": int((d["predictive_COD_test_z"] > 0).sum()),
                "n_test_COD_gt001": int((d["predictive_COD_test_z"] > 0.001).sum()),
                "n_test_r_gt03": int((d["Pearson_r_test_z"] > 0.03).sum()),
                "n_test_r_gt05": int((d["Pearson_r_test_z"] > 0.05).sum()),
            }
        )
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary_out = outdir / "phase5c_krr_5seed_seed_summary_v5.csv"
    seed_summary.to_csv(seed_summary_out, index=False)
    cod_pivot = long.pivot(
        index="target_index", columns="seed", values="predictive_COD_test_z"
    ).sort_index()
    if cod_pivot.shape != (330, 5):
        raise RuntimeError(f"COD pivot shape={cod_pivot.shape}")
    rank_pivot = cod_pivot.rank(axis=0, method="average")
    spearman = rank_pivot.corr(method="pearson")
    spearman_out = outdir / "phase5c_krr_5seed_COD_spearman_v5.csv"
    spearman.to_csv(spearman_out)
    top30 = {}
    top50 = {}
    for s in range(5):
        x = seed_dfs[s].sort_values("predictive_COD_test_z", ascending=False)
        top30[s] = set(x.head(30)["target_index"].astype(int))
        top50[s] = set(x.head(50)["target_index"].astype(int))

    def jaccard_matrix(sets):
        mat = np.zeros((5, 5), dtype=float)
        for i in range(5):
            for j in range(5):
                u = sets[i] | sets[j]
                inter = sets[i] & sets[j]
                mat[i, j] = len(inter) / len(u) if len(u) else np.nan
        return pd.DataFrame(
            mat,
            index=[f"seed{x}" for x in range(5)],
            columns=[f"seed{x}" for x in range(5)],
        )

    top30_j = jaccard_matrix(top30)
    top50_j = jaccard_matrix(top50)
    top30_j.to_csv(outdir / "phase5c_krr_5seed_top30_jaccard_v5.csv")
    top50_j.to_csv(outdir / "phase5c_krr_5seed_top50_jaccard_v5.csv")
    brain50 = load_universe(args.brain50, expected_n=50)
    brain19 = load_universe(args.brain19, expected_n=19)
    if not brain19.issubset(brain50):
        raise RuntimeError("brain_core19 is not a subset of brain_firstbatch50")
    target_rows = []
    for target_index, g in long.groupby("target_index", sort=True):
        if len(g) != 5:
            raise RuntimeError(f"target {target_index}: expected 5 seeds; got {len(g)}")
        g = g.sort_values("seed")
        target_key = g["target_key"].iloc[0]
        cod = g["predictive_COD_test_z"].to_numpy(dtype=float)
        r = g["Pearson_r_test_z"].to_numpy(dtype=float)
        auc = g["AUC_raw_disease_test"].to_numpy(dtype=float)
        alpha = g["best_alpha"].to_numpy(dtype=float)
        extreme = g["extreme_shrinkage_boundary"].astype(bool).to_numpy()
        stage2 = (g["alpha_resolution_stage"] == "stage2_highalpha").to_numpy()
        target_rows.append(
            {
                "target_index": int(target_index),
                "target": target_key,
                "mean_test_COD": float(np.mean(cod)),
                "sd_test_COD": float(np.std(cod, ddof=1)),
                "median_test_COD": float(np.median(cod)),
                "min_test_COD": float(np.min(cod)),
                "max_test_COD": float(np.max(cod)),
                "n_seed_COD_gt0": int(np.sum(cod > 0)),
                "n_seed_COD_gt001": int(np.sum(cod > 0.001)),
                "mean_test_r_fisher": fisher_mean(r),
                "mean_test_r_raw": float(np.mean(r)),
                "sd_test_r": float(np.std(r, ddof=1)),
                "min_test_r": float(np.min(r)),
                "max_test_r": float(np.max(r)),
                "n_seed_r_gt03": int(np.sum(r > 0.03)),
                "n_seed_r_gt05": int(np.sum(r > 0.05)),
                "mean_test_AUC": float(np.mean(auc)),
                "sd_test_AUC": float(np.std(auc, ddof=1)),
                "n_seed_extreme_shrinkage": int(np.sum(extreme)),
                "n_seed_stage2": int(np.sum(stage2)),
                "alpha_min": float(np.min(alpha)),
                "alpha_median": float(np.median(alpha)),
                "alpha_max": float(np.max(alpha)),
                "n_seed_top30_COD": int(
                    sum((int(target_index) in top30[s] for s in range(5)))
                ),
                "n_seed_top50_COD": int(
                    sum((int(target_index) in top50[s] for s in range(5)))
                ),
                "is_brain_firstbatch50": target_key in brain50,
                "is_brain_core19": target_key in brain19,
            }
        )
    agg = pd.DataFrame(target_rows)
    if len(agg) != 330:
        raise RuntimeError(f"Expected aggregated N=330; got {len(agg)}")
    if int(agg["is_brain_firstbatch50"].sum()) != 50:
        raise RuntimeError("Aggregated brain50 membership != 50")
    if int(agg["is_brain_core19"].sum()) != 19:
        raise RuntimeError("Aggregated brain19 membership != 19")
    ranking = agg.sort_values(
        ["mean_test_COD", "n_seed_COD_gt0", "mean_test_r_fisher", "target_index"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "source_rank", np.arange(1, len(ranking) + 1))
    ranking["stable_COD_positive_4of5"] = ranking["n_seed_COD_gt0"] >= 4
    ranking["stable_COD_positive_5of5"] = ranking["n_seed_COD_gt0"] == 5
    ranking_out = outdir / "phase5c_krr_residualized_5seed_source_ranking_v5.csv"
    ranking.to_csv(ranking_out, index=False)
    universe_rows = []
    universes = {
        "all_disease330": np.ones(len(ranking), dtype=bool),
        "brain_firstbatch50": ranking["is_brain_firstbatch50"].to_numpy(),
        "brain_core19": ranking["is_brain_core19"].to_numpy(),
    }
    for name, mask in universes.items():
        d = ranking.loc[mask].copy()
        universe_rows.append(
            {
                "universe": name,
                "n_targets": len(d),
                "mean_of_5seed_mean_COD": float(d["mean_test_COD"].mean()),
                "median_of_5seed_mean_COD": float(d["mean_test_COD"].median()),
                "n_mean_COD_gt0": int((d["mean_test_COD"] > 0).sum()),
                "n_stable_positive_4of5": int(d["stable_COD_positive_4of5"].sum()),
                "n_stable_positive_5of5": int(d["stable_COD_positive_5of5"].sum()),
                "n_in_overall_top30": int((d["source_rank"] <= 30).sum()),
                "n_in_overall_top50": int((d["source_rank"] <= 50).sum()),
            }
        )
    universe_summary = pd.DataFrame(universe_rows)
    universe_out = outdir / "phase5c_krr_5seed_universe_summary_v5.csv"
    universe_summary.to_csv(universe_out, index=False)
    manifest = {
        "version": "Phase5C_5seed_source_ranking_v5",
        "n_targets": 330,
        "seeds": [0, 1, 2, 3, 4],
        "primary_ranking_metric": "mean predictive_COD_test_z across seeds0-4",
        "tie_breakers": [
            "n_seed_COD_gt0 descending",
            "mean_test_r_fisher descending",
            "target_index ascending",
        ],
        "alpha_selection_data": "within-seed validation only",
        "source_ranking_data": "within-seed held-out outer test only",
        "bcp_used_for_source_ranking": False,
        "stage2_trigger": "Stage1 validation-selected best_alpha == 1000",
        "stage2_max_alpha": 100000,
        "alpha_above_100000_allowed": False,
        "extreme_shrinkage_definition": "resolved best_alpha == 100000",
        "pearson_aggregation": "Fisher-z mean across five outer-test splits",
        "auc_role": "diagnostic only; not used for source ranking",
        "important_interpretation": "The five source splits are repeated random held-out splits and are not treated as five statistically independent experiments.",
        "seed_input_files": {str(s): str(paths[s]) for s in range(5)},
        "long_table": str(long_out),
        "source_ranking": str(ranking_out),
        "seed_summary": str(seed_summary_out),
        "cod_spearman": str(spearman_out),
        "universe_summary": str(universe_out),
    }
    manifest_out = (
        outdir / "phase5c_krr_residualized_5seed_source_ranking_manifest_v5.json"
    )
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("\n===== FIVE-SEED SOURCE SUMMARY =====")
    print(seed_summary.to_string(index=False))
    print("\n===== COD RANK SPEARMAN =====")
    print(spearman.round(3).to_string())
    print("\n===== TOP30 JACCARD =====")
    print(top30_j.round(3).to_string())
    print("\n===== TARGET-LEVEL 5-SEED COD DISTRIBUTION =====")
    print(
        ranking["mean_test_COD"]
        .quantile([0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 1])
        .to_string()
    )
    print("\nmean COD > 0:", int((ranking["mean_test_COD"] > 0).sum()))
    print("stable COD positive >=4/5:", int(ranking["stable_COD_positive_4of5"].sum()))
    print("stable COD positive 5/5:", int(ranking["stable_COD_positive_5of5"].sum()))
    print("mean COD > .001:", int((ranking["mean_test_COD"] > 0.001).sum()))
    print("\n===== EXTREME SHRINKAGE ACROSS SEEDS =====")
    print(ranking["n_seed_extreme_shrinkage"].value_counts().sort_index().to_string())
    print("\n===== FROZEN UNIVERSES =====")
    print(universe_summary.to_string(index=False))
    print("\n===== TOP 40 SOURCE TARGETS =====")
    show = [
        "source_rank",
        "target",
        "mean_test_COD",
        "sd_test_COD",
        "min_test_COD",
        "n_seed_COD_gt0",
        "n_seed_COD_gt001",
        "mean_test_r_fisher",
        "sd_test_r",
        "mean_test_AUC",
        "n_seed_top30_COD",
        "n_seed_extreme_shrinkage",
        "is_brain_firstbatch50",
        "is_brain_core19",
    ]
    print(ranking[show].head(40).to_string(index=False))
    print("\n===== SAVED =====")
    print(ranking_out)
    print(manifest_out)
    print("\nPASS Phase5C residualized KRR 5-seed source aggregation and ranking")


if __name__ == "__main__":
    main()
