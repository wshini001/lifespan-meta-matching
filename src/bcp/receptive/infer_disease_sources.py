"""Apply frozen DiseaseRaw KRR bundles to BCP RSFC.

Averages seed-specific standardized residual predictions. Individual domain
cohorts are subsequently aligned by scan and child identifiers.
"""

import argparse
import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

EXPECTED_FOLDS = {0: 80, 1: 80, 2: 79, 3: 78, 4: 78}


def normalize_target(x):
    return f"{float(x):.2f}"


def normalize_eids(x):
    s = pd.to_numeric(pd.Series(np.asarray(x).ravel()), errors="raise")
    v = s.to_numpy(dtype=float)
    if not np.isfinite(v).all():
        raise ValueError("eid contains NaN/Inf")
    if not np.equal(v, np.floor(v)).all():
        raise ValueError("eid contains non-integer values")
    return s.astype(np.int64).astype(str).to_numpy()


def get_edges(columns):
    tmp = []
    for c in columns:
        m = re.fullmatch("e(\\d+)", str(c))
        if m:
            tmp.append((int(m.group(1)), c))
    tmp.sort()
    nums = [x[0] for x in tmp]
    names = [x[1] for x in tmp]
    if nums != list(range(1, 1486)):
        raise RuntimeError(f"Expected exact e1..e1485; found {len(names)} edges")
    return names


def unitize_rows(A):
    A = np.asarray(A, dtype=np.float64)
    A = A - A.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(A, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise RuntimeError("Zero-norm RSFC row after centering")
    A = A / norm
    if not np.isfinite(A).all():
        raise RuntimeError("Unitized RSFC contains NaN/Inf")
    return A


def get_edges_from_manifest(edge_manifest_tsv):
    m = pd.read_csv(edge_manifest_tsv, sep="\t")
    required = ["edge_index_1based", "edge_name"]
    missing = [c for c in required if c not in m.columns]
    if missing:
        raise RuntimeError(f"Edge manifest missing {missing}")
    m = m.sort_values("edge_index_1based").reset_index(drop=True)
    edges = m["edge_name"].astype(str).tolist()
    expected = [f"e{i}" for i in range(1, 1486)]
    if edges != expected:
        raise RuntimeError("Edge manifest is not exact e1..e1485")
    return edges


def load_bcp(cohort_csv, locked_row_manifest_csv, edge_manifest_tsv):
    cohort = pd.read_csv(cohort_csv)
    edges = get_edges_from_manifest(edge_manifest_tsv)
    cohort_edges = get_edges(cohort.columns)
    if cohort_edges != edges:
        raise RuntimeError("BCP cohort edge order != frozen edge manifest")
    key_cols = ["scan_id", "imaging_numeric_id"]
    missing = [c for c in key_cols if c not in cohort.columns]
    if missing:
        raise RuntimeError(f"BCP cohort missing keys: {missing}")
    manifest = pd.read_csv(locked_row_manifest_csv)
    required_manifest = [
        "_locked_row",
        "_cohort_all_row",
        "_pred_row",
        "scan_id",
        "imaging_numeric_id",
        "fold",
    ]
    missing = [c for c in required_manifest if c not in manifest.columns]
    if missing:
        raise RuntimeError(f"Locked row manifest missing {missing}")
    if len(manifest) != 395:
        raise RuntimeError(f"Expected locked manifest N=395; got {len(manifest)}")
    if cohort[key_cols].astype(str).duplicated().any():
        raise RuntimeError(
            "Full BCP cohort has duplicated scan_id/imaging_numeric_id keys"
        )
    if manifest[key_cols].astype(str).duplicated().any():
        raise RuntimeError(
            "Locked manifest has duplicated scan_id/imaging_numeric_id keys"
        )
    for c in key_cols:
        cohort[c] = cohort[c].astype(str)
        manifest[c] = manifest[c].astype(str)
    locked = manifest.merge(
        cohort[key_cols + edges], on=key_cols, how="left", validate="one_to_one"
    )
    if len(locked) != 395:
        raise RuntimeError(f"Expected 395 locked rows after merge; got {len(locked)}")
    locked = locked.sort_values("_locked_row").reset_index(drop=True)
    if not np.array_equal(
        locked["_locked_row"].to_numpy(dtype=int), np.arange(395, dtype=int)
    ):
        raise RuntimeError("_locked_row is not exact 0..394")
    if locked["imaging_numeric_id"].astype(str).nunique() != 199:
        raise RuntimeError("Expected 199 unique locked subjects")
    locked["fold"] = pd.to_numeric(locked["fold"], errors="raise").astype(int)
    counts = locked["fold"].value_counts().sort_index().to_dict()
    if counts != EXPECTED_FOLDS:
        raise RuntimeError(f"Unexpected locked fold counts: {counts}")
    E = locked[edges].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if E.shape != (395, 1485):
        raise RuntimeError(f"Unexpected locked edge matrix shape: {E.shape}")
    if not np.isfinite(E).all():
        raise RuntimeError("Locked BCP RSFC contains NaN/Inf")
    Z = unitize_rows(E)
    meta = locked[
        [
            "_locked_row",
            "_cohort_all_row",
            "_pred_row",
            "scan_id",
            "imaging_numeric_id",
            "fold",
        ]
    ].copy()
    return (meta, Z, edges)


def load_ukb(ukb_csv, expected_edges):
    head = pd.read_csv(ukb_csv, nrows=0)
    edges = get_edges(head.columns)
    if edges != expected_edges:
        raise RuntimeError("UKB/BCP edge order mismatch")
    if "eid" not in head.columns:
        raise KeyError("UKB missing eid")
    df = pd.read_csv(ukb_csv, usecols=["eid"] + edges)
    if len(df) != 24668:
        raise RuntimeError(f"Expected UKB N=24668; got {len(df)}")
    eid = normalize_eids(df["eid"])
    if len(set(eid)) != len(eid):
        raise RuntimeError("UKB eid is not unique")
    E = df[edges].to_numpy(dtype=np.float64)
    if not np.isfinite(E).all():
        raise RuntimeError("UKB RSFC contains NaN/Inf")
    Z = unitize_rows(E)
    eid_to_col = {e: i for i, e in enumerate(eid)}
    return (eid, eid_to_col, Z)


def read_ranking(path, targets_arg):
    df = pd.read_csv(path)
    required = [
        "source_rank",
        "target",
        "mean_test_COD",
        "n_seed_COD_gt0",
        "mean_test_r_fisher",
        "is_brain_firstbatch50",
        "is_brain_core19",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Ranking missing {missing}")
    if len(df) != 330:
        raise RuntimeError(f"Ranking N={len(df)}, expected 330")
    df["target_key"] = df["target"].map(normalize_target)
    if df["target_key"].nunique() != 330:
        raise RuntimeError("Ranking target labels not unique")
    df["source_rank"] = pd.to_numeric(df["source_rank"], errors="raise").astype(int)
    if set(df["source_rank"]) != set(range(1, 331)):
        raise RuntimeError("source_rank contract failed")
    df = df.sort_values("source_rank").reset_index(drop=True)
    if targets_arg:
        wanted = [normalize_target(x) for x in targets_arg.split(",") if x.strip()]
        if len(wanted) != len(set(wanted)):
            raise RuntimeError("Duplicate requested targets")
        available = set(df["target_key"])
        missing = [x for x in wanted if x not in available]
        if missing:
            raise RuntimeError(f"Requested targets missing: {missing}")
        lookup = df.set_index("target_key")
        df = lookup.loc[wanted].reset_index()
        if "target_key" not in df.columns:
            raise RuntimeError("Internal error: target_key lost after target subset")
        if df["target_key"].tolist() != wanted:
            raise RuntimeError("Requested target order was not preserved")
    return df


def read_master(path, seed, ranking_targets):
    df = pd.read_csv(path)
    required = [
        "seed",
        "target_index",
        "target",
        "best_alpha",
        "n_inner_obs",
        "bundle_path",
        "alpha_resolution_stage",
        "extreme_shrinkage_boundary",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"seed{seed} master missing {missing}")
    if len(df) != 330:
        raise RuntimeError(f"seed{seed} master N={len(df)}")
    df["target_key"] = df["target"].map(normalize_target)
    if df["target_key"].nunique() != 330:
        raise RuntimeError(f"seed{seed} targets not unique")
    if set(df["target_key"]) != set(ranking_targets):
        raise RuntimeError(f"seed{seed} target set != ranking")
    seeds = pd.to_numeric(df["seed"], errors="raise").astype(int)
    if not (seeds == seed).all():
        raise RuntimeError(f"seed{seed} seed field mismatch")
    return df.set_index("target_key")


def validate_bundle(b, row, seed, target, edges):
    required = [
        "seed",
        "target_index",
        "target",
        "best_alpha",
        "dual_coef",
        "fit_eid",
        "edge_names",
        "feature_transform",
        "kernel_type",
        "target_space",
        "prediction_scale",
        "refit_on_train_validation",
    ]
    missing = [k for k in required if k not in b]
    if missing:
        raise RuntimeError(f"seed{seed} target={target}: bundle missing {missing}")
    if int(b["seed"]) != seed:
        raise RuntimeError(f"seed mismatch target={target}")
    if normalize_target(b["target"]) != target:
        raise RuntimeError(f"bundle target mismatch: {target}")
    if int(b["target_index"]) != int(row["target_index"]):
        raise RuntimeError(f"target_index mismatch: {target}")
    if not np.isclose(
        float(b["best_alpha"]), float(row["best_alpha"]), rtol=0, atol=1e-10
    ):
        raise RuntimeError(f"best_alpha mismatch: {target}")
    if list(b["edge_names"]) != edges:
        raise RuntimeError(f"edge_names mismatch: {target}")
    if b["feature_transform"] != "row_center_then_l2_unitize":
        raise RuntimeError(f"feature transform mismatch: {target}")
    if b["kernel_type"] != "Pearson_correlation_precomputed":
        raise RuntimeError(f"kernel mismatch: {target}")
    if b["target_space"] != "covariate_residual_z":
        raise RuntimeError(f"target space mismatch: {target}")
    if b["prediction_scale"] != "z":
        raise RuntimeError(f"prediction scale mismatch: {target}")
    if bool(b["refit_on_train_validation"]):
        raise RuntimeError(f"unexpected refit: {target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bcp-cohort-csv", required=True)
    ap.add_argument("--locked-row-manifest-csv", required=True)
    ap.add_argument("--edge-manifest-tsv", required=True)
    ap.add_argument("--ukb-csv", required=True)
    ap.add_argument("--seed0-csv", required=True)
    ap.add_argument("--resolved-root", required=True)
    ap.add_argument("--ranking-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument(
        "--targets", default="", help="Optional comma-separated phecodes for smoke test"
    )
    args = ap.parse_args()
    out = Path(args.out_dir)
    per_seed = out / "per_seed"
    out.mkdir(parents=True, exist_ok=True)
    per_seed.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise RuntimeError("No seeds requested")
    ranking_all = read_ranking(args.ranking_csv, "")
    ranking = read_ranking(args.ranking_csv, args.targets)
    target_keys = ranking["target_key"].astype(str).tolist()
    print("===== PHASE5D CONTRACT =====")
    print("seeds:", seeds)
    print("targets:", len(target_keys))
    print("BCP target used for selection: False")
    print("prediction space: covariate_residual_z")
    print("mean aggregation: arithmetic mean across seed zpred")
    print("\n===== LOAD BCP =====")
    meta, Z_bcp, edges = load_bcp(
        args.bcp_cohort_csv, args.locked_row_manifest_csv, args.edge_manifest_tsv
    )
    print("BCP N:", len(meta))
    print("BCP Z:", Z_bcp.shape)
    print("fold counts:", meta["fold"].value_counts().sort_index().to_dict())
    print("\n===== LOAD UKB =====")
    ukb_eid, eid_to_col, Z_ukb = load_ukb(args.ukb_csv, edges)
    print("UKB N:", len(ukb_eid))
    print("UKB Z:", Z_ukb.shape)
    print("\n===== CROSS KERNEL =====")
    K_all = Z_bcp @ Z_ukb.T
    expected_k_shape = (len(meta), len(ukb_eid))
    if K_all.shape != expected_k_shape:
        raise RuntimeError(
            f"Unexpected cross-kernel shape {K_all.shape}; expected {expected_k_shape}"
        )
    if not np.isfinite(K_all).all():
        raise RuntimeError("Cross-kernel contains NaN/Inf")
    print("K_BCP_UKB:", K_all.shape, K_all.dtype)
    masters = {}
    ranking_target_set = set(ranking_all["target_key"])
    root = Path(args.resolved_root)
    for seed in seeds:
        if seed == 0:
            p = Path(args.seed0_csv)
        else:
            p = root / f"seed{seed}" / f"phase5c_seed{seed}_resolved_all330_v5.csv"
        if not p.is_file():
            raise FileNotFoundError(p)
        masters[seed] = read_master(p, seed, ranking_target_set)
    predictions = {t: [] for t in target_keys}
    artifact_rows = []
    for seed in seeds:
        print(f"\n===== SEED {seed} =====", flush=True)
        master = masters[seed]
        seed_df = meta.copy()
        for k, target in enumerate(target_keys):
            row = master.loc[target]
            bundle_path = Path(str(row["bundle_path"]))
            if not bundle_path.is_file():
                raise FileNotFoundError(bundle_path)
            b = joblib.load(bundle_path)
            validate_bundle(b, row, seed, target, edges)
            fit_eid = normalize_eids(b["fit_eid"])
            dual = np.asarray(b["dual_coef"], dtype=np.float64).ravel()
            if len(fit_eid) != len(dual):
                raise RuntimeError(f"seed{seed} target={target}: fit_eid N != dual N")
            if len(set(fit_eid)) != len(fit_eid):
                raise RuntimeError(f"seed{seed} target={target}: duplicate fit_eid")
            if len(fit_eid) != int(row["n_inner_obs"]):
                raise RuntimeError(
                    f"seed{seed} target={target}: fit N={len(fit_eid)} != n_inner_obs={row['n_inner_obs']}"
                )
            missing_eid = [x for x in fit_eid if x not in eid_to_col]
            if missing_eid:
                raise RuntimeError(
                    f"seed{seed} target={target}: {len(missing_eid)} fit_eids missing UKB"
                )
            idx = np.fromiter(
                (eid_to_col[x] for x in fit_eid), dtype=np.int64, count=len(fit_eid)
            )
            pred_z = K_all[:, idx] @ dual
            if pred_z.shape != (len(meta),):
                raise RuntimeError(
                    f"bad prediction shape seed{seed} target={target}: {pred_z.shape}"
                )
            if not np.isfinite(pred_z).all():
                raise RuntimeError(f"nonfinite prediction seed{seed} target={target}")
            col = f"phecode_{target}__zpred"
            seed_df[col] = pred_z
            predictions[target].append(pred_z.copy())
            artifact_rows.append(
                {
                    "seed": seed,
                    "target": target,
                    "source_rank": int(
                        ranking.loc[
                            ranking["target_key"] == target, "source_rank"
                        ].iloc[0]
                    ),
                    "best_alpha": float(row["best_alpha"]),
                    "n_fit": len(fit_eid),
                    "alpha_resolution_stage": str(row["alpha_resolution_stage"]),
                    "extreme_shrinkage_boundary": bool(
                        row["extreme_shrinkage_boundary"]
                    ),
                    "bundle_path": str(bundle_path),
                    "bcp_zpred_mean": float(np.mean(pred_z)),
                    "bcp_zpred_sd": float(np.std(pred_z)),
                    "bcp_zpred_min": float(np.min(pred_z)),
                    "bcp_zpred_max": float(np.max(pred_z)),
                }
            )
            if len(target_keys) <= 10 or (k + 1) % 50 == 0:
                print(
                    f"seed{seed}: {k + 1}/{len(target_keys)} target={target}",
                    flush=True,
                )
        seed_path = per_seed / f"bcp_rawdisease_krr_resid_seed{seed}.parquet"
        seed_df.to_parquet(seed_path, index=False)
        print("saved:", seed_path, flush=True)
    print("\n===== MEAN ACROSS SEEDS =====")
    mean_df = meta.copy()
    summary_rows = []
    rank_lookup = ranking.set_index("target_key")
    for target in target_keys:
        stack = np.stack(predictions[target], axis=1)
        if stack.shape != (len(meta), len(seeds)):
            raise RuntimeError(f"bad seed stack: {target} {stack.shape}")
        zmean = np.mean(stack, axis=1)
        zsd = np.std(stack, axis=1, ddof=0)
        mean_df[f"phecode_{target}__zpred"] = zmean
        mean_df[f"phecode_{target}__zpred_seed_sd"] = zsd
        rr = rank_lookup.loc[target]
        summary_rows.append(
            {
                "source_rank": int(rr["source_rank"]),
                "target": target,
                "source_mean_test_COD": float(rr["mean_test_COD"]),
                "source_n_seed_COD_gt0": int(rr["n_seed_COD_gt0"]),
                "source_mean_test_r_fisher": float(rr["mean_test_r_fisher"]),
                "is_brain_firstbatch50": bool(rr["is_brain_firstbatch50"]),
                "is_brain_core19": bool(rr["is_brain_core19"]),
                "bcp_mean5_zpred_mean": float(np.mean(zmean)),
                "bcp_mean5_zpred_sd": float(np.std(zmean)),
                "bcp_mean_seed_sd": float(np.mean(zsd)),
            }
        )
    n_targets = len(target_keys)
    if seeds == [0, 1, 2, 3, 4] and n_targets == 330:
        mean_name = "bcp_rawdisease_krr_resid_mean5_all330_v1.parquet"
    else:
        mean_name = f"bcp_rawdisease_krr_resid_mean{len(seeds)}_n{n_targets}_v1.parquet"
    mean_path = out / mean_name
    mean_df.to_parquet(mean_path, index=False)
    artifact = pd.DataFrame(artifact_rows)
    artifact_path = out / "bcp_rawdisease_krr_resid_artifact_summary_v1.csv"
    artifact.to_csv(artifact_path, index=False)
    target_summary = pd.DataFrame(summary_rows).sort_values("source_rank")
    target_summary_path = out / "bcp_rawdisease_krr_resid_target_summary_v1.csv"
    target_summary.to_csv(target_summary_path, index=False)
    feature_manifest = target_summary[
        [
            "source_rank",
            "target",
            "source_mean_test_COD",
            "source_n_seed_COD_gt0",
            "source_mean_test_r_fisher",
            "is_brain_firstbatch50",
            "is_brain_core19",
        ]
    ].copy()
    feature_manifest["prediction_column"] = feature_manifest["target"].map(
        lambda x: f"phecode_{x}__zpred"
    )
    feature_manifest["seed_sd_column"] = feature_manifest["target"].map(
        lambda x: f"phecode_{x}__zpred_seed_sd"
    )
    feature_manifest_path = out / "bcp_rawdisease_krr_resid_feature_manifest_v1.tsv"
    feature_manifest.to_csv(feature_manifest_path, sep="\t", index=False)
    manifest = {
        "version": "Phase5D_RawDisease_KRR_BCP_Inference_v1",
        "n_bcp": int(len(meta)),
        "n_ukb": int(len(ukb_eid)),
        "n_edges": len(edges),
        "seeds": seeds,
        "n_targets": n_targets,
        "bcp_target_used_for_model_selection": False,
        "bcp_target_used_for_source_ranking": False,
        "locked_row_manifest_csv": str(Path(args.locked_row_manifest_csv).resolve()),
        "edge_manifest_tsv": str(Path(args.edge_manifest_tsv).resolve()),
        "source_ranking_csv": str(Path(args.ranking_csv).resolve()),
        "prediction_space": "covariate_residual_z",
        "feature_transform": "row_center_then_l2_unitize",
        "kernel": "Pearson_correlation_cross_kernel",
        "formula": "K_BCP_UKB = Z_BCP @ Z_UKB.T; y_pred_z = K_BCP_UKB[:, fit_eid_columns] @ dual_coef",
        "mean5_rule": "arithmetic mean of seed-specific covariate_residual_z predictions",
        "fold_counts": {str(k): int(v) for k, v in EXPECTED_FOLDS.items()},
        "outputs": {
            "mean_predictions": str(mean_path),
            "artifact_summary": str(artifact_path),
            "target_summary": str(target_summary_path),
            "feature_manifest": str(feature_manifest_path),
        },
    }
    manifest_path = out / "bcp_rawdisease_krr_resid_inference_manifest_v1.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\n===== FINAL =====")
    print("BCP N:", len(meta))
    print("UKB N:", len(ukb_eid))
    print("targets:", n_targets)
    print("seeds:", seeds)
    print("mean output:", mean_path)
    print("feature manifest:", feature_manifest_path)
    print("\nPASS Phase5D raw-disease residualized-KRR BCP inference")


if __name__ == "__main__":
    main()
