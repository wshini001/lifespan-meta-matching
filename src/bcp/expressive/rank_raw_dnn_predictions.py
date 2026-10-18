"""Select raw-DNN prediction features in UKB source-performance ranking order."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def rank_predictions(bank, ranking):
    ranking = ranking.sort_values("rank")
    if ranking["rank"].tolist() != list(range(1, 121)):
        raise ValueError("Expected frozen ranks 1–120")
    columns = (ranking["target"].astype(str) + "__zpred").tolist()
    keys = ["scan_id", "imaging_numeric_id"]
    if len(set(columns)) != 120 or bank[keys].duplicated().any():
        raise ValueError("Duplicate target columns or scan/child keys")
    result = bank[keys + columns].copy()
    if not np.isfinite(result[columns].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite source predictions")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", required=True)
    parser.add_argument("--ranking-csv", required=True)
    parser.add_argument("--out-parquet", required=True)
    args = parser.parse_args()
    result = rank_predictions(
        pd.read_parquet(args.source_parquet), pd.read_csv(args.ranking_csv)
    )
    path = Path(args.out_parquet)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    print("PASS: 120 frozen source predictions selected in recorded rank order")


if __name__ == "__main__":
    main()
