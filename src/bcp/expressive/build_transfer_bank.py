"""Align DiseaseRaw and DNNRaw predictions to the expressive-language cohort.

Matches scan and child identifiers and retains the specified feature order.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

GROUP = "imaging_numeric_id"
KEYS = ["scan_id", GROUP]
LOCKED_COLUMNS = ["_locked_row", "scan_id", GROUP, "fold"]
EXPECTED_N = 289
EXPECTED_CHILDREN = 159
FAMILIES = {
    "diseaseraw330": {"arg": "diseaseraw330", "prefix": "diseaseraw__", "n": 330},
    "dnnraw_top120": {"arg": "dnnraw_top120", "prefix": "dnnraw__", "n": 120},
}
FORBIDDEN_TOKENS = (
    "expressive_language",
    "fine_motor",
    "gross_motor",
    "composite_percentile",
    "y_true",
    "y_pred",
    "baseline",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--locked-reference-csv", required=True)
    p.add_argument("--diseaseraw330", required=True)
    p.add_argument("--dnnraw-top120", required=True)
    p.add_argument("--diseaseraw-feature-manifest-tsv", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-tag", default="expressive_language_n289_transfer_bank_v2")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_scan(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def normalize_group(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="raise").to_numpy(float)
    if not np.isfinite(x).all() or not np.equal(x, np.floor(x)).all():
        raise RuntimeError("imaging_numeric_id must contain finite integers")
    return pd.Series(x.astype(np.int64).astype(str), index=s.index)


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scan_id"] = normalize_scan(out["scan_id"])
    out[GROUP] = normalize_group(out[GROUP])
    return out


def load_disease_lookup(path: str | None) -> tuple[pd.DataFrame | None, str | None]:
    if path is None:
        return (None, None)
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    meta = pd.read_csv(p, sep="\t")
    candidates = ["prediction_column", "source_column", "aligned_column"]
    key = next((c for c in candidates if c in meta.columns), None)
    if key is None:
        raise RuntimeError(
            "Disease-raw manifest needs prediction_column, source_column, or aligned_column"
        )
    if meta[key].astype(str).duplicated().any():
        raise RuntimeError(f"Disease-raw manifest has duplicate {key}")
    return (meta.set_index(meta[key].astype(str), drop=False), str(p))


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    locked_path = Path(args.locked_reference_csv).resolve()
    if not locked_path.is_file():
        raise FileNotFoundError(locked_path)
    source_paths = {
        family: Path(getattr(args, spec["arg"])).resolve()
        for family, spec in FAMILIES.items()
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    locked = normalize_keys(pd.read_csv(locked_path))
    needed = KEYS + ["fold"]
    missing = [c for c in needed if c not in locked.columns]
    if missing:
        raise RuntimeError(f"Locked reference missing columns: {missing}")
    if len(locked) != EXPECTED_N or locked[GROUP].nunique() != EXPECTED_CHILDREN:
        raise RuntimeError(
            f"Expected {EXPECTED_N} scans/{EXPECTED_CHILDREN} children; got {len(locked)}/{locked[GROUP].nunique()}"
        )
    if locked[KEYS].duplicated().any():
        raise RuntimeError("Locked reference contains duplicate scan/child keys")
    fold = pd.to_numeric(locked["fold"], errors="raise").to_numpy(float)
    if not np.equal(fold, np.floor(fold)).all() or sorted(np.unique(fold).tolist()) != [
        0,
        1,
        2,
        3,
        4,
    ]:
        raise RuntimeError("Locked reference fold must be integer 0..4")
    locked["fold"] = fold.astype(int)
    locked.insert(0, "_locked_row", np.arange(EXPECTED_N, dtype=int))
    if locked.groupby(GROUP)["fold"].nunique().max() != 1:
        raise RuntimeError("Child leakage detected in locked outer folds")
    disease_lookup, disease_manifest_path = load_disease_lookup(
        args.diseaseraw_feature_manifest_tsv
    )
    bank = locked[LOCKED_COLUMNS].copy()
    family_audit: list[dict] = []
    feature_rows: list[dict] = []
    hashes: dict[str, str] = {}
    for family, spec in FAMILIES.items():
        path = source_paths[family]
        observed_sha = sha256(path)
        hashes[family] = observed_sha
        src = normalize_keys(pd.read_parquet(path))
        if src[KEYS].duplicated().any():
            raise RuntimeError(f"{family} has duplicate scan/child keys")
        zcols = [str(c) for c in src.columns if str(c).endswith("__zpred")]
        if len(zcols) != int(spec["n"]):
            raise RuntimeError(
                f"{family}: expected {spec['n']} exact __zpred columns; got {len(zcols)}"
            )
        src_x = src[KEYS + zcols].copy()
        src_x[zcols] = src_x[zcols].apply(pd.to_numeric, errors="coerce")
        aligned = locked[KEYS].merge(
            src_x, on=KEYS, how="left", validate="one_to_one", indicator=True
        )
        missing_keys = int((aligned["_merge"] != "both").sum())
        aligned = aligned.drop(columns="_merge")
        x = aligned[zcols].to_numpy(float)
        if missing_keys or not np.isfinite(x).all():
            raise RuntimeError(
                f"{family}: missing keys={missing_keys}, nonfinite cells={int((~np.isfinite(x)).sum())}"
            )
        constant = [c for c in zcols if aligned[c].nunique(dropna=True) <= 1]
        if constant:
            raise RuntimeError(f"{family}: constant N289 features: {constant[:10]}")
        renamed = []
        for j, source_col in enumerate(zcols, start=1):
            base = source_col[: -len("__zpred")]
            output_col = f"{spec['prefix']}{base}__zpred"
            if output_col in bank.columns or output_col in renamed:
                raise RuntimeError(f"Output feature collision: {output_col}")
            renamed.append(output_col)
            row = {
                "family": family,
                "family_feature_index_1based": j,
                "source_file": str(path),
                "source_column": source_col,
                "aligned_column": output_col,
                "prediction_space": "source-target z prediction",
                "source_order_frozen": True,
                "eligible_for_target_nested_selection": True,
            }
            if family == "diseaseraw330" and disease_lookup is not None:
                lookup_keys = [
                    source_col,
                    output_col,
                    source_col.replace("phecode_", ""),
                ]
                hit = next((k for k in lookup_keys if k in disease_lookup.index), None)
                if hit is not None:
                    rr = disease_lookup.loc[hit]
                    for field in [
                        "source_rank",
                        "target",
                        "source_mean_test_COD",
                        "source_n_seed_COD_gt0",
                        "source_mean_test_r_fisher",
                        "is_brain_firstbatch50",
                        "is_brain_core19",
                    ]:
                        if field in rr.index:
                            value = rr[field]
                            row[field] = (
                                value.item() if hasattr(value, "item") else value
                            )
            feature_rows.append(row)
        bank[renamed] = aligned[zcols].set_axis(renamed, axis=1)
        family_audit.append(
            {
                "family": family,
                "status": "PASS",
                "source_rows": int(len(src)),
                "source_unique_scans": int(src["scan_id"].nunique()),
                "source_extra_scans": int(src["scan_id"].nunique() - EXPECTED_N),
                "expressive_missing_keys": missing_keys,
                "n_features": int(len(zcols)),
                "missing_feature_cells": int(np.isnan(x).sum()),
                "nonfinite_feature_cells": int((~np.isfinite(x)).sum()),
                "constant_features_n289": int(len(constant)),
                "sha256": observed_sha,
            }
        )
    feature_cols = [c for c in bank.columns if c not in LOCKED_COLUMNS]
    expected_total = sum((int(x["n"]) for x in FAMILIES.values()))
    if len(feature_cols) != expected_total or len(set(feature_cols)) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} unique transfer features; got {len(feature_cols)}"
        )
    bad_names = [
        c
        for c in feature_cols
        if any((token in c.lower() for token in FORBIDDEN_TOKENS))
    ]
    if bad_names:
        raise RuntimeError(f"Outcome/baseline-like feature names: {bad_names[:10]}")
    if not np.isfinite(bank[feature_cols].to_numpy(float)).all():
        raise RuntimeError("Final transfer bank contains NaN/Inf")
    tag = args.run_tag
    paths = {
        "feature_bank": out / f"{tag}.parquet",
        "feature_manifest": out / f"{tag}_feature_manifest.tsv",
        "family_audit": out / f"{tag}_family_audit.csv",
        "manifest": out / f"{tag}_manifest.json",
    }
    bank.to_parquet(paths["feature_bank"], index=False)
    pd.DataFrame(feature_rows).to_csv(paths["feature_manifest"], sep="\t", index=False)
    pd.DataFrame(family_audit).to_csv(paths["family_audit"], index=False)
    manifest = {
        "status": "PASS",
        "run_tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_scans": EXPECTED_N,
        "n_children": EXPECTED_CHILDREN,
        "n_transfer_features": expected_total,
        "fold_counts": {
            str(k): int(v) for k, v in bank["fold"].value_counts().sort_index().items()
        },
        "families": {
            family: {
                "n_features": int(spec["n"]),
                "source_file": str(source_paths[family]),
                "sha256": hashes[family],
            }
            for family, spec in FAMILIES.items()
        },
        "contracts": {
            "outcome_free_bank": True,
            "only_exact___zpred_columns_used": True,
            "target_columns_used": False,
            "baseline_prediction_columns_used": False,
            "raw_prediction_columns_used": False,
            "seed_sd_columns_used": False,
            "source_fold_used": False,
            "expressive_fold_reattached_from_locked_reference": True,
            "source_order_frozen_before_expressive_modeling": True,
            "all_target_side_model_and_topm_selection_must_be_nested": True,
        },
        "inputs_read_only": {
            "locked_reference_csv": str(locked_path),
            "diseaseraw_feature_manifest_tsv": disease_manifest_path,
        },
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "shape": list(bank.shape),
                "n_transfer_features": expected_total,
                "outputs": {k: str(v) for k, v in paths.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
