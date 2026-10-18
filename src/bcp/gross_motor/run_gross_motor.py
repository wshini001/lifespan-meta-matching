"""Fit gross-motor KRR and DNN multibranch models using child-grouped nested CV."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import model_engine as E
import transfer_engine as T
from build_transfer_bank import normalize_keys


def load_bank(df, fold_id, bank_path, manifest_path):
    bank = normalize_keys(pd.read_parquet(bank_path))
    manifest = pd.read_csv(manifest_path, sep="\t")
    arrays = []
    for family, count in [
        ("krrraw_top120", 120),
        ("diseaseraw330", 330),
        ("dnnraw_mean5", 120),
    ]:
        columns = (
            manifest[manifest.family.eq(family)]
            .sort_values("family_feature_index_1based")
            .aligned_column.tolist()
        )
        if len(columns) != count or len(set(columns)) != count:
            raise ValueError("Unexpected feature count or duplicate columns: " + family)
        aligned = df[E.KEYS].merge(
            bank[E.KEYS + ["fold"] + columns],
            on=E.KEYS,
            how="left",
            validate="one_to_one",
        )
        if not np.array_equal(pd.to_numeric(aligned.fold).to_numpy(int), fold_id):
            raise ValueError("Transfer bank fold mismatch")
        matrix = aligned[columns].to_numpy(float)
        if not np.isfinite(matrix).all():
            raise ValueError("Non-finite transfer features")
        arrays.append(matrix)
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--fold-manifest-csv", required=True)
    parser.add_argument("--locked-manifest-csv", required=True)
    parser.add_argument("--transfer-bank-parquet", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument(
        "--settings", default=str(Path(__file__).parent / "config/gross_settings.json")
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text())
    for key, name in {
        "raw_plateau": "RAW_PLATEAU",
        "disease_plateau": "DISEASE_PLATEAU",
        "dnn_m": "DNN_M",
        "alphas": "ALPHAS",
        "w_raw": "W_RAW",
        "w_disease": "W_DIS",
        "w_dnn": "W_DNN",
    }.items():
        if settings["grids"][key] != getattr(T, name):
            raise ValueError("Settings differ from the configured grid: " + key)
    if settings["baseline_alpha"] != E.BASELINE_ALPHA:
        raise ValueError("Settings differ from the configured baseline penalty")
    df, rsfc, y, cdf, folds, fid, _ = E.load_data(
        E, args.cohort_csv, args.fold_manifest_csv
    )
    df = normalize_keys(df)
    groups = df[E.GROUP].to_numpy(str)
    cov = cdf.to_numpy(float)
    if len(df) != 321 or len(np.unique(groups)) != 166 or len(folds) != 5:
        raise ValueError("Expected 321 scans, 166 children and five outer folds")
    locked = normalize_keys(pd.read_csv(args.locked_manifest_csv))
    check = df[E.KEYS].merge(
        locked[E.KEYS + ["fold"]], on=E.KEYS, how="left", validate="one_to_one"
    )
    if not np.array_equal(pd.to_numeric(check.fold).to_numpy(int), fid):
        raise ValueError("Locked fold mismatch")
    raw, disease, dnn = load_bank(
        df, fid, args.transfer_bank_parquet, args.feature_manifest
    )
    methods = ["baseline", "dnn_branch", "krr_plateau_core", "krr_plus_dnn_multibranch"]
    predictions = {name: np.full(len(df), np.nan) for name in methods}
    residuals = np.full(len(df), np.nan)
    selected = []
    for fold, (train, test) in enumerate(folds):
        reps = T.make_reps(
            E,
            E,
            E,
            rsfc,
            raw,
            disease,
            dnn,
            y,
            cov,
            groups,
            train,
            settings["inner_k"],
            settings["inner_repeats"],
            settings["inner_seed_base"] + 100000 * fold,
        )
        ra, da, bd, core, joint = T.select(E, reps)
        y_train, y_test, *_ = E.residualize(y[train], y[test], cov[train], cov[test])
        residuals[test] = y_test
        pc, pb, pn = T.fit(
            E,
            E,
            E,
            rsfc,
            raw,
            disease,
            dnn,
            y_train,
            groups[train],
            train,
            test,
            ra,
            da,
            bd,
            core,
        )
        pj, _, _ = T.fit(
            E,
            E,
            E,
            rsfc,
            raw,
            disease,
            dnn,
            y_train,
            groups[train],
            train,
            test,
            ra,
            da,
            bd,
            joint,
        )
        for name, values in zip(methods, [pb, pn, pc, pj]):
            predictions[name][test] = values
        for name, choice in [
            ("krr_plateau_core", core),
            ("krr_plus_dnn_multibranch", joint),
        ]:
            selected.append(
                {
                    "outer_fold": fold,
                    "method": name,
                    **choice,
                    "dnn_repr": bd["repr"],
                    "dnn_m": bd["size"],
                    "dnn_alpha": bd["alpha"],
                    "raw_alphas": json.dumps(ra),
                    "disease_alphas": json.dumps(da),
                }
            )
        print(f"Completed outer fold {fold}", flush=True)
    wide = df[E.KEYS + [E.TARGET]].copy()
    wide["fold"] = fid
    wide["y_true_resid"] = residuals
    for name in methods:
        wide["y_pred_" + name] = predictions[name]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(selected).to_csv(out / "selected_parameters.csv", index=False)
    summary = pd.DataFrame(
        [
            {"method": name, **E.metrics(residuals, predictions[name])}
            for name in methods
        ]
    )
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
