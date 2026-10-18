"""Fit expressive-language disease-anchor and DNN additive prediction models.

Evaluates Top20 and Top18 source-ranked DNN branches. The Top18 additive
model corresponds to the manuscript prediction.
"""

from __future__ import annotations
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

EXPECTED_N = 289
EXPECTED_CHILDREN = 159
EXPECTED_BASELINE = {"COD": 0.025346825546223917, "Pearson_r": 0.17230204241486013}
EXPECTED_DISEASE_ANCHOR = {"COD": 0.0314329409546229, "Pearson_r": 0.17927591717488914}
PRIMARY_M = 20
SENSITIVITY_M = [15, 18, 22, 25]
ALL_M = [15, 18, 20, 22, 25]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
GAMMA_OLD = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
GAMMA_EXPANDED = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
CONVEX_ANCHOR_W = [0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase9-script",
        default=str(Path(__file__).resolve().parent / "transfer_engine.py"),
    )
    p.add_argument(
        "--engine-py",
        default=str(Path(__file__).resolve().parent / "cohort_preprocessing.py"),
    )
    p.add_argument("--transfer-bank-parquet", required=True)
    p.add_argument("--transfer-feature-manifest-tsv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--run-tag",
        default="expressive_language_n289_disease_dnnraw20_focused_validation_v1",
    )
    p.add_argument("--inner-k", type=int, default=5)
    p.add_argument("--inner-seed-base", type=int, default=7200)
    p.add_argument("--bootstrap-reps", type=int, default=10000)
    p.add_argument("--permutation-reps", type=int, default=100000)
    p.add_argument("--inference-seed", type=int, default=10202026)
    p.add_argument("--cohort-csv", required=True)
    p.add_argument("--fold-manifest-csv", required=True)
    p.add_argument("--fmri-res-csv", required=True)
    return p.parse_args()


def check_close(label, observed, expected, atol=1e-10):
    for key, value in expected.items():
        if not np.isclose(observed[key], value, rtol=0, atol=atol):
            raise RuntimeError(
                f"{label} reproduction failed for {key}: {observed[key]} vs {value}"
            )


def select_additive(
    core, y, inner_id, anchor, cache, m, gammas, outer_fold, selection_label
):
    rows = []
    fallback_score = core.scoring_fields(y, anchor, inner_id)
    best = {
        "outer_fold": outer_fold,
        "selection_label": selection_label,
        "mode": "additive",
        "top_m": int(m),
        "alpha_branch": np.nan,
        "gamma": 0.0,
        "anchor_weight": 1.0,
        "branch_weight": 0.0,
        **fallback_score,
    }
    rows.append(best.copy())
    for alpha in RIDGE_ALPHAS:
        branch = cache[int(m), float(alpha)]
        for gamma in gammas:
            if gamma <= 0:
                continue
            candidate_pred = anchor + float(gamma) * branch
            row = {
                "outer_fold": outer_fold,
                "selection_label": selection_label,
                "mode": "additive",
                "top_m": int(m),
                "alpha_branch": float(alpha),
                "gamma": float(gamma),
                "anchor_weight": 1.0,
                "branch_weight": float(gamma),
                **core.scoring_fields(y, candidate_pred, inner_id),
            }
            rows.append(row)
            key = (
                -row["pooled_inner_COD"],
                row["pooled_inner_RMSE"],
                row["gamma"],
                -row["alpha_branch"],
            )
            best_key = (
                -best["pooled_inner_COD"],
                best["pooled_inner_RMSE"],
                best["gamma"],
                -best["alpha_branch"] if np.isfinite(best["alpha_branch"]) else -np.inf,
            )
            if key < best_key:
                best = row
    return (best, rows)


def select_convex(core, y, inner_id, anchor, cache, m, outer_fold):
    rows = []
    score = core.scoring_fields(y, anchor, inner_id)
    best = {
        "outer_fold": outer_fold,
        "selection_label": "primary_M20_convex_comparator",
        "mode": "convex",
        "top_m": int(m),
        "alpha_branch": np.nan,
        "gamma": 0.0,
        "anchor_weight": 1.0,
        "branch_weight": 0.0,
        **score,
    }
    rows.append(best.copy())
    for alpha in RIDGE_ALPHAS:
        branch = cache[int(m), float(alpha)]
        for w in CONVEX_ANCHOR_W:
            if w >= 1.0:
                continue
            candidate_pred = float(w) * anchor + (1.0 - float(w)) * branch
            row = {
                "outer_fold": outer_fold,
                "selection_label": "primary_M20_convex_comparator",
                "mode": "convex",
                "top_m": int(m),
                "alpha_branch": float(alpha),
                "gamma": 0.0,
                "anchor_weight": float(w),
                "branch_weight": 1.0 - float(w),
                **core.scoring_fields(y, candidate_pred, inner_id),
            }
            rows.append(row)
            key = (
                -row["pooled_inner_COD"],
                row["pooled_inner_RMSE"],
                -row["anchor_weight"],
                -row["alpha_branch"],
            )
            best_key = (
                -best["pooled_inner_COD"],
                best["pooled_inner_RMSE"],
                -best["anchor_weight"],
                -best["alpha_branch"] if np.isfinite(best["alpha_branch"]) else -np.inf,
            )
            if key < best_key:
                best = row
    return (best, rows)


def apply_outer(core, anchor, x, ytr, tr, te, cfg):
    if not np.isfinite(cfg["alpha_branch"]) or cfg["branch_weight"] <= 0:
        return np.asarray(anchor, float).copy()
    m = int(cfg["top_m"])
    branch = core.fit_transfer(x[tr, :m], ytr, x[te, :m], float(cfg["alpha_branch"]))
    if cfg["mode"] == "additive":
        return anchor + float(cfg["gamma"]) * branch
    return float(cfg["anchor_weight"]) * anchor + float(cfg["branch_weight"]) * branch


def cluster_bootstrap(core, y, predictions, groups, reps, seed):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_indices = {g: np.flatnonzero(groups == g) for g in unique_groups}
    methods = list(predictions)
    out = {
        m: {"COD": [], "delta_vs_baseline": [], "delta_vs_anchor": []} for m in methods
    }
    for _ in range(int(reps)):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        ix = np.concatenate([group_indices[g] for g in sampled])
        base_cod = core.metrics(y[ix], predictions["baseline_fmri_zlinear"][ix])["COD"]
        anchor_cod = core.metrics(y[ix], predictions["diseaseraw_anchor"][ix])["COD"]
        for method in methods:
            cod = core.metrics(y[ix], predictions[method][ix])["COD"]
            out[method]["COD"].append(cod)
            out[method]["delta_vs_baseline"].append(cod - base_cod)
            out[method]["delta_vs_anchor"].append(cod - anchor_cod)
    rows = []
    for method in methods:
        for estimand, values in out[method].items():
            arr = np.asarray(values, float)
            rows.append(
                {
                    "method": method,
                    "estimand": estimand,
                    "bootstrap_reps": int(reps),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "ci2.5": float(np.quantile(arr, 0.025)),
                    "ci97.5": float(np.quantile(arr, 0.975)),
                    "probability_gt_0": float(np.mean(arr > 0)),
                }
            )
    return pd.DataFrame(rows)


def paired_cluster_signflip(y, reference, candidate, groups, reps, seed):
    unique_groups = np.unique(groups)
    improvements = np.asarray(
        [
            np.sum((y[groups == g] - reference[groups == g]) ** 2)
            - np.sum((y[groups == g] - candidate[groups == g]) ** 2)
            for g in unique_groups
        ],
        float,
    )
    observed = float(improvements.sum())
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = int(reps)
    while remaining > 0:
        batch = min(5000, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, len(improvements)))
        null = signs @ improvements
        exceed += int(np.sum(null >= observed))
        remaining -= batch
    return {
        "n_children": int(len(unique_groups)),
        "permutation_reps": int(reps),
        "observed_SSE_improvement": observed,
        "one_sided_p": float((exceed + 1) / (int(reps) + 1)),
        "children_with_positive_SSE_improvement": int(np.sum(improvements > 0)),
        "median_child_SSE_improvement": float(np.median(improvements)),
    }


def main():
    args = parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "phase9_core", Path(args.phase9_script).resolve()
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import Phase9 script")
    phase9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase9)
    engine = phase9.load_module(args.engine_py, "expressive_phase3_engine_phase10")
    eng_args = SimpleNamespace(
        cohort_csv=args.cohort_csv,
        fold_manifest_csv=args.fold_manifest_csv,
        fmri_res_csv=args.fmri_res_csv,
        max_age_diff_days=7.0,
        expected_n=EXPECTED_N,
        expected_children=EXPECTED_CHILDREN,
    )
    df, _, fmri, cdf, fold_id, audit, _, fmri_cols = engine.load_and_match(eng_args)
    df = phase9.normalize_keys(df)
    y_raw = pd.to_numeric(df[phase9.TARGET], errors="raise").to_numpy(float)
    cov = cdf.to_numpy(float)
    groups = df[phase9.GROUP].to_numpy(str)
    bank = phase9.normalize_keys(pd.read_parquet(args.transfer_bank_parquet))
    fman = pd.read_csv(args.transfer_feature_manifest_tsv, sep="\t")
    if bank[phase9.KEYS].duplicated().any():
        raise RuntimeError("Transfer bank contains duplicate keys")
    aligned = df[phase9.KEYS].merge(
        bank, on=phase9.KEYS, how="left", validate="one_to_one"
    )
    if aligned["fold"].isna().any() or not np.array_equal(
        pd.to_numeric(aligned["fold"]).to_numpy(int), fold_id
    ):
        raise RuntimeError("Transfer bank alignment/fold contract failed")
    disease_x, disease_cols = phase9.source_family_matrix(
        aligned, fman, "diseaseraw330", 330
    )
    dnn_x, dnn_cols = phase9.source_family_matrix(aligned, fman, "dnnraw_top120", 120)
    branch_specs = {
        "diseaseraw330": {
            "X": disease_x,
            "m_values": phase9.DISEASE_RAW_M,
            "width_fn": lambda m: m,
        },
        "dnnraw120": {"X": dnn_x, "m_values": ALL_M, "width_fn": lambda m: m},
    }
    methods = [
        "baseline_fmri_zlinear",
        "diseaseraw_anchor",
        "dnnraw_M20_additive_gamma_le020",
        "dnnraw_M20_additive_gamma_extended",
        "dnnraw_M20_convex_comparator",
        *[f"dnnraw_M{m}_additive_gamma_extended_sensitivity" for m in SENSITIVITY_M],
    ]
    pred = {m: np.full(len(df), np.nan) for m in methods}
    y_oof = np.full(len(df), np.nan)
    selected_rows = []
    inner_rows = []
    fold_rows = []
    idx = np.arange(len(df))
    outer_folds = [(idx[fold_id != f], idx[fold_id == f]) for f in range(5)]
    for outer_fold, (tr, te) in enumerate(outer_folds):
        y_inner, inner_id, base_cache, caches = phase9.build_inner_oof(
            engine,
            fmri,
            branch_specs,
            y_raw,
            cov,
            groups,
            tr,
            args.inner_k,
            args.inner_seed_base + outer_fold,
        )
        best_base, disease_cfg, disease_inner, disease_grid = (
            phase9.select_disease_anchor(
                y_inner, inner_id, base_cache, caches["diseaseraw330"], outer_fold
            )
        )
        inner_rows.extend(disease_grid)
        selected_rows.extend(
            [
                {**best_base, "selected_method": "baseline_fmri_zlinear"},
                {**disease_cfg, "selected_method": "diseaseraw_anchor"},
            ]
        )
        ytr, yte = engine.residualize(y_raw[tr], y_raw[te], cov[tr], cov[te])
        y_oof[te] = yte
        pb = phase9.fit_baseline(fmri[tr], ytr, fmri[te], best_base["alpha_base"])
        pred["baseline_fmri_zlinear"][te] = pb
        dm = int(disease_cfg["top_m"])
        pdisease = phase9.fit_transfer(
            disease_x[tr, :dm], ytr, disease_x[te, :dm], disease_cfg["alpha_branch"]
        )
        anchor_outer = (
            disease_cfg["anchor_weight"] * pb + disease_cfg["branch_weight"] * pdisease
        )
        pred["diseaseraw_anchor"][te] = anchor_outer
        cfg_old, rows = select_additive(
            phase9,
            y_inner,
            inner_id,
            disease_inner,
            caches["dnnraw120"],
            PRIMARY_M,
            GAMMA_OLD,
            outer_fold,
            "primary_M20_old_gamma_reproduction",
        )
        inner_rows.extend(rows)
        pred["dnnraw_M20_additive_gamma_le020"][te] = apply_outer(
            phase9, anchor_outer, dnn_x, ytr, tr, te, cfg_old
        )
        selected_rows.append(
            {**cfg_old, "selected_method": "dnnraw_M20_additive_gamma_le020"}
        )
        cfg_primary, rows = select_additive(
            phase9,
            y_inner,
            inner_id,
            disease_inner,
            caches["dnnraw120"],
            PRIMARY_M,
            GAMMA_EXPANDED,
            outer_fold,
            "primary_M20_expanded_gamma",
        )
        inner_rows.extend(rows)
        pred["dnnraw_M20_additive_gamma_extended"][te] = apply_outer(
            phase9, anchor_outer, dnn_x, ytr, tr, te, cfg_primary
        )
        selected_rows.append(
            {**cfg_primary, "selected_method": "dnnraw_M20_additive_gamma_extended"}
        )
        cfg_convex, rows = select_convex(
            phase9,
            y_inner,
            inner_id,
            disease_inner,
            caches["dnnraw120"],
            PRIMARY_M,
            outer_fold,
        )
        inner_rows.extend(rows)
        pred["dnnraw_M20_convex_comparator"][te] = apply_outer(
            phase9, anchor_outer, dnn_x, ytr, tr, te, cfg_convex
        )
        selected_rows.append(
            {**cfg_convex, "selected_method": "dnnraw_M20_convex_comparator"}
        )
        for m in SENSITIVITY_M:
            cfg, rows = select_additive(
                phase9,
                y_inner,
                inner_id,
                disease_inner,
                caches["dnnraw120"],
                m,
                GAMMA_EXPANDED,
                outer_fold,
                f"sensitivity_M{m}_expanded_gamma",
            )
            inner_rows.extend(rows)
            name = f"dnnraw_M{m}_additive_gamma_extended_sensitivity"
            pred[name][te] = apply_outer(phase9, anchor_outer, dnn_x, ytr, tr, te, cfg)
            selected_rows.append({**cfg, "selected_method": name})
        base_fold = phase9.metrics(yte, pb)
        anchor_fold = phase9.metrics(yte, anchor_outer)
        for method in methods:
            values = pred[method][te]
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"Incomplete prediction: {method}, fold {outer_fold}"
                )
            met = phase9.metrics(yte, values)
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": method,
                    "n_train": len(tr),
                    "n_test": len(te),
                    **met,
                    "delta_COD_vs_baseline_fold": met["COD"] - base_fold["COD"],
                    "delta_COD_vs_disease_anchor_fold": met["COD"] - anchor_fold["COD"],
                }
            )
        print(
            f"[DONE] fold {outer_fold}: primary gamma={cfg_primary['gamma']:.3f}, alpha={cfg_primary['alpha_branch']}",
            flush=True,
        )
    if not np.isfinite(y_oof).all() or any(
        (not np.isfinite(v).all() for v in pred.values())
    ):
        raise RuntimeError("Incomplete/nonfinite OOF predictions")
    baseline_metrics = phase9.metrics(y_oof, pred["baseline_fmri_zlinear"])
    anchor_metrics = phase9.metrics(y_oof, pred["diseaseraw_anchor"])
    check_close("Frozen baseline", baseline_metrics, EXPECTED_BASELINE)
    check_close("Disease anchor", anchor_metrics, EXPECTED_DISEASE_ANCHOR)
    summary_rows = []
    for method in methods:
        met = phase9.metrics(y_oof, pred[method])
        db = met["COD"] - baseline_metrics["COD"]
        da = met["COD"] - anchor_metrics["COD"]
        summary_rows.append(
            {
                "method": method,
                **met,
                "frozen_baseline_COD": baseline_metrics["COD"],
                "delta_COD_vs_frozen_baseline": db,
                "relative_COD_improvement_percent_vs_frozen_baseline": 100.0
                * db
                / abs(baseline_metrics["COD"]),
                "diseaseraw_anchor_COD": anchor_metrics["COD"],
                "delta_COD_vs_diseaseraw_anchor": da,
                "relative_COD_improvement_percent_vs_diseaseraw_anchor": 100.0
                * da
                / abs(anchor_metrics["COD"]),
                "delta_r_vs_frozen_baseline": met["Pearson_r"]
                - baseline_metrics["Pearson_r"],
                "folds_improved_vs_diseaseraw_anchor": (
                    int(
                        sum(
                            (
                                r["delta_COD_vs_disease_anchor_fold"] > 0
                                for r in fold_rows
                                if r["method"] == method
                            )
                        )
                    )
                    if method != "diseaseraw_anchor"
                    else 0
                ),
                "result_role": (
                    "locked_reference"
                    if method == "baseline_fmri_zlinear"
                    else (
                        "strict_nested_anchor"
                        if method == "diseaseraw_anchor"
                        else (
                            "focused_primary"
                            if method == "dnnraw_M20_additive_gamma_extended"
                            else (
                                "phase9_reproduction"
                                if method.endswith("gamma_le020")
                                else (
                                    "focused_comparator"
                                    if method.endswith("convex_comparator")
                                    else "fixed_M_sensitivity"
                                )
                            )
                        )
                    )
                ),
            }
        )
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["COD", "Pearson_r"], ascending=False)
        .reset_index(drop=True)
    )
    bootstrap = cluster_bootstrap(
        phase9, y_oof, pred, groups, args.bootstrap_reps, args.inference_seed
    )
    inference_rows = []
    for method in methods:
        if method in {"baseline_fmri_zlinear", "diseaseraw_anchor"}:
            continue
        test = paired_cluster_signflip(
            y_oof,
            pred["diseaseraw_anchor"],
            pred[method],
            groups,
            args.permutation_reps,
            args.inference_seed + methods.index(method),
        )
        inference_rows.append(
            {"candidate": method, "reference": "diseaseraw_anchor", **test}
        )
    inference = pd.DataFrame(inference_rows)
    tag = args.run_tag
    paths = {
        "summary": out / f"summary_{tag}.csv",
        "fold_metrics": out / f"fold_metrics_{tag}.csv",
        "selected": out / f"selected_hyperparameters_{tag}.csv",
        "inner_grid": out / f"inner_grid_{tag}.csv.gz",
        "predictions": out / f"predictions_wide_{tag}.csv",
        "cluster_bootstrap": out / f"child_cluster_bootstrap_{tag}.csv",
        "paired_permutation": out / f"paired_cluster_permutation_{tag}.csv",
        "manifest": out / f"manifest_{tag}.json",
    }
    summary.to_csv(paths["summary"], index=False)
    pd.DataFrame(fold_rows).to_csv(paths["fold_metrics"], index=False)
    pd.DataFrame(selected_rows).to_csv(paths["selected"], index=False)
    pd.DataFrame(inner_rows).to_csv(
        paths["inner_grid"], index=False, compression="gzip"
    )
    bootstrap.to_csv(paths["cluster_bootstrap"], index=False)
    inference.to_csv(paths["paired_permutation"], index=False)
    wide = df[phase9.KEYS + [phase9.TARGET]].copy()
    wide["fold"] = fold_id
    wide["y_true_resid"] = y_oof
    for method in methods:
        wide[f"y_pred_{method}"] = pred[method]
    wide.to_csv(paths["predictions"], index=False)
    manifest = {
        "status": "PASS",
        "run_tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "focused validation of frozen disease-anchor plus source-ranked DNN-raw Top20",
        "n": len(df),
        "n_children": len(np.unique(groups)),
        "primary_model": "dnnraw_M20_additive_gamma_extended",
        "baseline": baseline_metrics,
        "disease_anchor": anchor_metrics,
        "grids": {
            "primary_M": PRIMARY_M,
            "sensitivity_M": SENSITIVITY_M,
            "ridge_alphas": RIDGE_ALPHAS,
            "gamma_old": GAMMA_OLD,
            "gamma_expanded": GAMMA_EXPANDED,
            "convex_anchor_weights": CONVEX_ANCHOR_W,
        },
        "contracts": {
            "same_locked_n289_outer_folds": True,
            "outer_and_inner_grouped_by_child": True,
            "disease_anchor_exact_phase6_reproduction_required": True,
            "dnnraw_top20_source_ranked_before_bcp_outcome_modeling": True,
            "primary_M20_frozen_before_phase10": True,
            "alpha_and_gamma_selected_inside_outer_training_only": True,
            "exact_anchor_fallback": True,
            "neighbor_M_results_are_sensitivity_only": True,
            "outer_test_used_for_selection": False,
            "bootstrap_clustered_by_child": True,
            "paired_permutation_clustered_by_child": True,
        },
        "inputs_read_only": {
            "phase9_script": str(Path(args.phase9_script).resolve()),
            "engine_py": str(Path(args.engine_py).resolve()),
            "transfer_bank_parquet": str(Path(args.transfer_bank_parquet).resolve()),
            "transfer_feature_manifest_tsv": str(
                Path(args.transfer_feature_manifest_tsv).resolve()
            ),
        },
        "dnnraw_top20_columns": dnn_cols[:20],
        "diseaseraw_columns": disease_cols,
        "audit": audit,
        "fmri_features": fmri_cols,
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))
    print("\n===== PAIRED CHILD-CLUSTER PERMUTATION =====")
    print(inference.to_string(index=False))
    print("\nPASS focused disease + DNN-raw Top20 validation")


if __name__ == "__main__":
    main()
