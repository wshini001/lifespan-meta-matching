"""Combine composite OOF predictions with fixed 0.70/0.30 weights.

Blends the internal prediction and the Top81 raw-source branch.
The supplied baseline prediction is used for performance comparison.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def metrics(y, p):
    sst = np.sum((y - y.mean()) ** 2)
    return {
        "COD": float(1 - np.sum((y - p) ** 2) / sst),
        "Pearson_r": float(np.corrcoef(y, p)[0, 1]),
    }


def build_blend(official, branches):
    keys = ["scan_id", "imaging_numeric_id"]
    off = official[keys + ["fold", "y_true_resid", "y_pred"]].rename(
        columns={
            "fold": "fold_official",
            "y_true_resid": "y_true",
            "y_pred": "pred_official",
        }
    )
    branch = branches[
        keys + ["fold", "y_true_resid", "y_pred_baseline_only", "y_pred_multikernel"]
    ].rename(
        columns={
            "fold": "fold_branch",
            "y_true_resid": "y_branch",
            "y_pred_baseline_only": "pred_internal",
            "y_pred_multikernel": "pred_top81",
        }
    )
    frame = off.merge(branch, on=keys, how="inner", validate="one_to_one")
    if len(frame) != len(off) or len(frame) != len(branch):
        raise ValueError("Branch sample coverage differs.")
    if not np.array_equal(frame.fold_official, frame.fold_branch):
        raise ValueError("Branch folds differ.")
    if np.max(abs(frame.y_true - frame.y_branch)) > 1e-08:
        raise ValueError("Branch residual outcomes differ.")
    if not np.isfinite(
        frame[["y_true", "pred_official", "pred_internal", "pred_top81"]].to_numpy(
            float
        )
    ).all():
        raise ValueError("Nonfinite branch predictions.")
    frame["fold"] = frame.fold_official.astype(int)
    if frame.groupby("imaging_numeric_id")["fold"].nunique().gt(1).any():
        raise ValueError("A child occurs in multiple outer folds.")
    frame["pred_fixed_w_internal_0p70"] = (
        0.7 * frame.pred_internal + (1 - 0.7) * frame.pred_top81
    )
    return frame[
        keys
        + [
            "fold",
            "y_true",
            "pred_official",
            "pred_internal",
            "pred_top81",
            "pred_fixed_w_internal_0p70",
        ]
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-pred-csv", required=True)
    p.add_argument("--branch-pred-csv", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    frame = build_blend(
        pd.read_csv(args.official_pred_csv), pd.read_csv(args.branch_pred_csv)
    )
    if len(frame) != 319 or frame.imaging_numeric_id.nunique() != 166:
        raise ValueError("Expected 319 scans in 166 children.")
    y = frame.y_true.to_numpy(float)
    summary = {
        name: metrics(y, frame[name].to_numpy(float))
        for name in [
            "pred_official",
            "pred_internal",
            "pred_top81",
            "pred_fixed_w_internal_0p70",
        ]
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "composite_predictions.csv", index=False)
    (out / "composite_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("PASS: fixed composite blend created; prediction columns aligned.")


if __name__ == "__main__":
    main()
