"""Assemble the fixed final43 transferred predictions in locked cohort order.

No features are selected or fitted using the ABCD outcome. Supply the
mean-five prediction banks and the ordered source-feature list.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def index_bank(frame, keys):
    frame = frame.copy()
    for key in keys:
        frame[key] = frame[key].astype(str)
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate participant keys in prediction bank.")
    return frame.set_index(keys, drop=False)


def assemble(config, manifest, pc, raw, disease):
    keys = [config["subjectkey_col"], config["id_col"]]
    locked = index_bank(manifest, keys)
    names = config["source_names"]
    if len(names) != 43 or len(set(names)) != 43:
        raise RuntimeError("Expected 43 unique transferred feature names.")
    if len(locked) != config["expected_n_subjects"]:
        raise RuntimeError("Locked cohort size differs from the configuration.")
    banks = [index_bank(frame, keys) for frame in [pc, raw, disease]]
    for bank in banks:
        if not locked.index.isin(bank.index).all():
            raise RuntimeError("Prediction bank is missing locked participants.")
    pc, raw, disease = [bank.reindex(locked.index) for bank in banks]
    meta_cols = [
        c
        for c in [
            "_orig_row_idx",
            *keys,
            "fold",
            "intelligence_score",
            "interview_date",
            "age",
            "Age_years",
            "Sex",
            "site",
        ]
        if c in pc.columns
    ]
    out = pc[meta_cols].copy()
    out["fold"] = locked["fold"].to_numpy(dtype=int)
    out["_orig_row_idx"] = locked["_orig_row_idx"].to_numpy(dtype=int)
    for bank, selected in [
        (pc, names[:12]),
        (raw, names[12:27]),
        (disease, names[27:]),
    ]:
        for name in selected:
            z = f"{name}__zpred"
            if z not in bank.columns:
                raise KeyError(f"Missing transferred prediction: {z}")
            for col in [name, z, f"{name}__raw_seed_sd", f"{name}__zpred_seed_sd"]:
                if col in bank.columns:
                    out[col] = bank[col].to_numpy()
    values = out[[f"{name}__zpred" for name in names]].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("Transferred prediction matrix contains NaN/Inf.")
    return out.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in [
        "config",
        "fold-manifest-csv",
        "pc-parquet",
        "raw-parquet",
        "disease-parquet",
        "out-dir",
    ]:
        p.add_argument(f"--{flag}", required=True)
    args = p.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    feature_path = config_path.parent / config["source_features_file"]
    config["source_names"] = feature_path.read_text().splitlines()
    if len(config["source_names"]) != 43 or len(set(config["source_names"])) != 43:
        raise ValueError("Expected 43 unique source features in the feature list.")
    frame = assemble(
        config,
        pd.read_csv(args.fold_manifest_csv),
        pd.read_parquet(args.pc_parquet),
        pd.read_parquet(args.raw_parquet),
        pd.read_parquet(args.disease_parquet),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "abcd_final43_predictions.parquet", index=False)
    (out / "final43_feature_freeze.tsv").write_text(
        "\n".join(config["source_names"]) + "\n"
    )
    metadata = dict(
        config=config, inputs=vars(args), n_subjects=len(frame), n_features=43
    )
    (out / "final43_feature_manifest.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
