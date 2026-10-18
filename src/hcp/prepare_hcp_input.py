"""Prepare the locked HCP FluidCog table used in the transfer analysis.

The script joins the averaged 55-node full-correlation RSFC matrix
(1,485 upper-triangle edges) to HCP demographics, creates the prespecified
age/sex covariates, and assigns family-grouped five-fold splits. RSFC features
are not covariate-residualized here or in the downstream model.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

EDGE_COLUMNS = [f"e{i}" for i in range(1, 1486)]
COVARIATES = ["Age_in_Yrs", "Sex", "age_sq", "age_sex", "age_sq_sex"]


def clean_identifier(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace("\\.0$", "", regex=True)


def family_groups(frame: pd.DataFrame) -> pd.Series:
    if "Mother" not in frame.columns or "Father" not in frame.columns:
        return "singleton_" + frame["Subject"].astype(str)
    result = []
    for subject, mother, father in zip(
        frame["Subject"].astype(str), frame["Mother"], frame["Father"]
    ):
        if pd.notna(mother) and pd.notna(father):
            mother_id = str(mother).replace(".0", "")
            father_id = str(father).replace(".0", "")
            result.append(f"M{mother_id}_F{father_id}")
        else:
            result.append(f"singleton_{subject}")
    return pd.Series(result, index=frame.index, dtype="object")


def balanced_group_folds(groups: Sequence[str], n_folds: int, seed: int) -> np.ndarray:
    groups = np.asarray(groups, dtype=object)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    rng = np.random.default_rng(seed)
    unique_groups, counts = np.unique(groups, return_counts=True)
    order = np.arange(unique_groups.size)
    rng.shuffle(order)
    order = sorted(order.tolist(), key=lambda i: (-counts[i], rng.random()))
    fold_sizes = np.zeros(n_folds, dtype=int)
    group_to_fold: dict[object, int] = {}
    for group_index in order:
        fold = int(np.argmin(fold_sizes))
        group_to_fold[unique_groups[group_index]] = fold
        fold_sizes[fold] += int(counts[group_index])
    fold_id = np.array([group_to_fold[group] for group in groups], dtype=np.int64)
    for group in unique_groups:
        if np.unique(fold_id[groups == group]).size != 1:
            raise RuntimeError(f"Family group was split across folds: {group}")
    if sorted(np.unique(fold_id).tolist()) != list(range(n_folds)):
        raise RuntimeError("Not all requested fold labels were created")
    return fold_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-file", required=True, help="EdgeFull100_Avg.txt")
    parser.add_argument(
        "--demographics-csv", required=True, help="HCP demographics CSV"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="FluidCog")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-participants", type=int, default=1080)
    parser.add_argument("--expected-family-groups", type=int, default=465)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    edge_path = Path(args.edge_file)
    demographics_path = Path(args.demographics_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(edge_path, sep="\\s+", header=None)
    if edges.shape[1] != 1486:
        raise RuntimeError(
            f"Expected Subject plus 1,485 edges; observed shape {edges.shape}"
        )
    subjects = clean_identifier(edges.iloc[:, 0])
    if subjects.duplicated().any():
        raise RuntimeError("The RSFC file contains duplicate Subject IDs")
    edge_frame = pd.DataFrame(
        edges.iloc[:, 1:].to_numpy(dtype=np.float32), columns=EDGE_COLUMNS
    )
    edge_frame.insert(0, "Subject", subjects.to_numpy())
    edge_frame.insert(1, "edge_row_idx", np.arange(len(edge_frame), dtype=np.int64))
    if not np.isfinite(edge_frame[EDGE_COLUMNS].to_numpy(dtype=float)).all():
        raise RuntimeError("The RSFC matrix contains non-finite values")
    demographics = pd.read_csv(demographics_path)
    required = ["ID", "Sex", "Age", args.target, "Mother", "Father"]
    missing = [column for column in required if column not in demographics.columns]
    if missing:
        raise KeyError(f"Missing demographic columns: {missing}")
    demographics = demographics.copy()
    demographics["ID"] = clean_identifier(demographics["ID"])
    if demographics["ID"].duplicated().any():
        raise RuntimeError("The demographics file contains duplicate IDs")
    optional = ["Race", "EarlyCog", "CrystalCog", "TotalCog"]
    demographic_columns = required + [c for c in optional if c in demographics.columns]
    demographic_columns = list(dict.fromkeys(demographic_columns))
    merged = edge_frame.merge(
        demographics[demographic_columns],
        left_on="Subject",
        right_on="ID",
        how="left",
        validate="one_to_one",
    )
    merged[args.target] = pd.to_numeric(merged[args.target], errors="coerce")
    merged["Age"] = pd.to_numeric(merged["Age"], errors="coerce")
    merged["Age_in_Yrs"] = merged["Age"]
    merged["Sex"] = pd.to_numeric(merged["Sex"], errors="coerce")
    invalid_sex = merged.loc[merged["Sex"].notna() & ~merged["Sex"].isin([0, 1]), "Sex"]
    if not invalid_sex.empty:
        raise RuntimeError(
            f"Sex must be coded 0/1; observed {sorted(invalid_sex.unique())}"
        )
    merged["age_sq"] = merged["Age_in_Yrs"] ** 2
    merged["age_sex"] = merged["Age_in_Yrs"] * merged["Sex"]
    merged["age_sq_sex"] = merged["age_sq"] * merged["Sex"]
    merged["family_group"] = family_groups(merged)
    lock_columns = [args.target, *COVARIATES, *EDGE_COLUMNS]
    lock_values = merged[lock_columns].apply(pd.to_numeric, errors="coerce")
    lock_mask = np.isfinite(lock_values.to_numpy(dtype=np.float64)).all(axis=1)
    locked = merged.loc[lock_mask].copy().reset_index(drop=True)
    if locked.empty or locked["Subject"].duplicated().any():
        raise RuntimeError("Invalid locked cohort")
    locked.insert(0, "locked_row_idx", np.arange(len(locked), dtype=np.int64))
    locked["fold"] = balanced_group_folds(
        locked["family_group"].astype(str).to_numpy(), args.n_folds, args.seed
    )
    if locked.groupby("family_group")["fold"].nunique().max() != 1:
        raise RuntimeError("Family leakage detected after fold assignment")
    if args.expected_participants > 0 and len(locked) != args.expected_participants:
        raise RuntimeError(
            f"Expected {args.expected_participants} participants; found {len(locked)}"
        )
    observed_groups = int(locked["family_group"].nunique())
    if (
        args.expected_family_groups > 0
        and observed_groups != args.expected_family_groups
    ):
        raise RuntimeError(
            f"Expected {args.expected_family_groups} family groups; found {observed_groups}"
        )
    metadata = [
        "locked_row_idx",
        "edge_row_idx",
        "Subject",
        "ID",
        "fold",
        args.target,
        "Sex",
        "Age",
        "Age_in_Yrs",
        "Mother",
        "Father",
        "family_group",
        "Race",
        "EarlyCog",
        "CrystalCog",
        "TotalCog",
        "age_sq",
        "age_sex",
        "age_sq_sex",
    ]
    metadata = [column for column in metadata if column in locked.columns]
    locked = locked[metadata + EDGE_COLUMNS]
    locked_path = output_dir / "hcp_fluidcog_locked_input.parquet"
    audit_path = output_dir / "hcp_fluidcog_locked_input_audit.json"
    locked.to_parquet(locked_path, index=False)
    family_sizes = locked["family_group"].value_counts()
    audit = {
        "target": args.target,
        "n_participants": int(len(locked)),
        "n_family_groups": int(locked["family_group"].nunique()),
        "n_edges": 1485,
        "fold_counts": {
            str(k): int(v)
            for k, v in locked["fold"].value_counts().sort_index().items()
        },
        "maximum_family_group_size": int(family_sizes.max()),
        "covariates": COVARIATES,
        "rsfc_covariate_residualized": False,
        "edge_order": "e1..e1485 preserve the input EdgeFull100_Avg.txt order",
        "inputs": {
            "edge_file": str(edge_path),
            "demographics_csv": str(demographics_path),
        },
        "output": str(locked_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
