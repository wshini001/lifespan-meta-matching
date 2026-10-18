"""Fit the fine-motor DiseaseRaw, PC12 and fixed-Top88 DNN prediction sequence.

Uses supplied source predictions, child-grouped folds and convex blends.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import baseline_engine as engine
import transfer_engine as transfer
import stage_selection as selection


def fit_sequence(data, outer_splits, disease_x, pc_x, dnn_x, settings):
    x = np.asarray(data["X"], float)
    y_raw = np.asarray(data["y"], float)
    groups = np.asarray(data["groups"]).astype(str)
    z = engine.unitize_rows(x[:, : data["n_edge"]])
    c = x[:, data["n_edge"] :]
    m = int(settings["M"])
    y_oof = np.full(len(y_raw), np.nan)
    outputs = {
        name: np.full(len(y_raw), np.nan)
        for name in ["pred_disease_top2", "pred_disease_top2_pc12", "pred_final_M88"]
    }
    records = []
    for fold, (tr, te) in enumerate(outer_splits):
        ytr, yte, _, _ = engine.residualize_train_test(
            y_raw[tr], y_raw[te], c[tr], c[te]
        )
        y_oof[te] = yte
        y_inner, contexts = transfer.build_inner_context(
            engine,
            z[tr],
            c[tr],
            y_raw[tr],
            groups[tr],
            settings["inner_k"],
            settings["inner_seed_base"] + fold,
        )
        base_cache = transfer.build_baseline_oof_cache(
            z[tr], contexts, settings["baseline_alphas"]
        )
        dis_cache = transfer.build_branch_oof_cache(
            disease_x[tr], contexts, settings["ridge_alphas"]
        )
        pc_cache = transfer.build_branch_oof_cache(
            pc_x[tr], contexts, settings["ridge_alphas"]
        )
        disease_cfg, _ = transfer.select_stack(
            y_inner,
            base_cache,
            dis_cache,
            settings["disease_baseline_weights"],
            "disease_top2_primary",
            fold,
        )
        primary_inner = selection.primary_inner_from_cfg(
            disease_cfg, base_cache, dis_cache
        )
        primary_outer = selection.predict_primary(
            transfer, z[tr], ytr, z[te], disease_x[tr], disease_x[te], disease_cfg
        )
        pc_cfg = selection.select_pc_anchor(
            transfer, y_inner, primary_inner, pc_cache, settings["pc_anchor_weights"]
        )
        if pd.isna(pc_cfg["alpha_pc"]):
            anchor_inner, anchor_outer = (primary_inner.copy(), primary_outer.copy())
        else:
            a = float(pc_cfg["alpha_pc"])
            wa, wp = (float(pc_cfg["anchor_weight"]), float(pc_cfg["pc_weight"]))
            anchor_inner = wa * primary_inner + wp * pc_cache[a]
            ppc_outer = transfer.ridge_predict(pc_x[tr], ytr, pc_x[te], a)
            anchor_outer = wa * primary_outer + wp * ppc_outer
        dnn_cache = transfer.build_branch_oof_cache(
            dnn_x[tr, :m], contexts, settings["ridge_alphas"]
        )
        bins = selection.make_inner_fold_id(groups[tr], settings["inner_k"])
        dnn_cfg, _, _ = selection.select_dnn_for_M(
            transfer,
            y_inner,
            anchor_inner,
            dnn_cache,
            settings["dnn_anchor_weights"],
            bins,
            m,
        )
        final_outer = selection.outer_predict_from_cfg(
            transfer, anchor_outer, dnn_x[tr, :m], ytr, dnn_x[te, :m], dnn_cfg
        )
        outputs["pred_disease_top2"][te] = primary_outer
        outputs["pred_disease_top2_pc12"][te] = anchor_outer
        outputs["pred_final_M88"][te] = final_outer
        records.extend(
            [
                {"outer_fold": fold, "stage": "disease", **disease_cfg},
                {"outer_fold": fold, "stage": "pc12", **pc_cfg},
                {
                    "outer_fold": fold,
                    "stage": "dnnraw",
                    **{
                        k: dnn_cfg[k]
                        for k in [
                            "M",
                            "alpha_dnn",
                            "anchor_weight",
                            "dnn_weight",
                            "inner_COD",
                            "inner_RMSE",
                            "inner_r",
                        ]
                    },
                },
            ]
        )
        print(f"Completed outer fold {fold}", flush=True)
    if not np.isfinite(y_oof).all() or any(
        (not np.isfinite(p).all() for p in outputs.values())
    ):
        raise ValueError("Incomplete out-of-fold predictions")
    return (y_oof, outputs, pd.DataFrame(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config_dir = Path(__file__).resolve().parent / "config"
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--official-baseline-pred", required=True)
    parser.add_argument("--disease-bank", required=True)
    parser.add_argument("--pc-parquet", required=True)
    parser.add_argument("--dnn-all-parquet", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--settings", default=str(config_dir / "fine_motor_settings.json")
    )
    parser.add_argument(
        "--disease-top2-manifest", default=str(config_dir / "disease_top2.tsv")
    )
    parser.add_argument(
        "--dnn-rank-manifest", default=str(config_dir / "raw_dnn_ranking.csv")
    )
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text())
    if int(settings["M"]) != 88:
        raise ValueError("This release replays the fixed-M88 manuscript route")
    data = engine.load_bcp_data(
        args.cohort_csv,
        "fine_motor_percentile",
        "eligible_fine_motor_percentile_without_tr",
        "imaging_numeric_id",
    )
    if (
        len(data["y"]) != 395
        or len(np.unique(data["groups"])) != 199
        or data["n_edge"] != 1485
        or (data["n_cov"] != 8)
    ):
        raise ValueError(
            "Expected 395 scans, 199 children, 1485 edges and 8 covariates"
        )
    splits, fold_id = engine.load_outer_folds_from_manifest(
        data["meta"], args.fold_manifest
    )
    if set(fold_id) != set(range(5)):
        raise ValueError("Expected supplied outer folds 0–4")
    frames = {}
    for label, path in [
        ("official", args.official_baseline_pred),
        ("disease", args.disease_bank),
        ("pc", args.pc_parquet),
        ("dnn", args.dnn_all_parquet),
    ]:
        frame = pd.read_csv(path) if label == "official" else pd.read_parquet(path)
        frames[label] = transfer.align_to_meta(frame, data["meta"], label)
        if "fold" in frame and (
            not np.array_equal(frames[label]["fold"].to_numpy(int), fold_id)
        ):
            raise ValueError(f"{label}: supplied outer fold mismatch")
    top = pd.read_csv(args.disease_top2_manifest, sep="\t")
    dis_cols = top["prediction_column"].astype(str).tolist()
    if dis_cols != ["phecode_250.20__zpred", "phecode_278.10__zpred"]:
        raise ValueError("Disease Top2 order differs from the supplied source manifest")
    pc_cols = [c for c in frames["pc"] if str(c).endswith("__zpred")]
    if len(pc_cols) != 12:
        raise ValueError("Expected 12 frozen PC predictions")
    rank = pd.read_csv(args.dnn_rank_manifest).sort_values("rank")
    if rank["rank"].tolist() != list(range(1, 121)):
        raise ValueError("Expected frozen DNN ranks 1–120")
    dnn_cols = (rank["target"].astype(str) + "__zpred").tolist()
    matrices = [
        frames["disease"][dis_cols].to_numpy(float),
        frames["pc"][pc_cols].to_numpy(float),
        frames["dnn"][dnn_cols].to_numpy(float),
    ]
    if any((not np.isfinite(x).all() for x in matrices)):
        raise ValueError("Nonfinite frozen source predictions")
    y, predictions, params = fit_sequence(data, splits, *matrices, settings)
    official = frames["official"]["y_pred"].to_numpy(float)
    if "y_true_resid" in frames["official"] and (
        not np.allclose(
            y, frames["official"]["y_true_resid"].to_numpy(float), rtol=0, atol=1e-10
        )
    ):
        raise ValueError("Residualized outcome differs from the supplied baseline")
    predictions = {"pred_official_baseline": official, **predictions}
    summary = pd.DataFrame(
        [{"model": name, **transfer.metrics(y, p)} for name, p in predictions.items()]
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wide = data["meta"][["scan_id", "imaging_numeric_id"]].copy()
    wide["fold"] = fold_id
    wide["y_true_resid"] = y
    for name, p in predictions.items():
        wide[name] = p
    wide.to_csv(out / "fine_motor_predictions.csv", index=False)
    summary.to_csv(out / "performance_ladder.csv", index=False)
    params.to_csv(out / "selected_parameters.csv", index=False)
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "inputs": vars(args),
                "settings": settings,
                "covariates": data["cov_cols"],
                "pc_columns": pc_cols,
                "disease_columns": dis_cols,
                "dnn_columns": dnn_cols[:88],
            },
            indent=2,
        )
    )
    print(summary[["model", "COD", "Pearson_r"]].to_string(index=False))
    print("Fine-motor fixed-Top88 sequence completed")


if __name__ == "__main__":
    main()
