"""Generate fully simulated longitudinal inputs for five BCP-style outcomes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOMES = (
    "receptive_language_percentile",
    "expressive_language_percentile",
    "visual_reception_percentile",
    "fine_motor_percentile",
    "gross_motor_percentile",
)


def generate(output_dir, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_children = 36
    visits = rng.integers(2, 4, size=n_children)
    child = np.repeat(np.arange(n_children), visits)
    visit = np.concatenate([np.arange(count) for count in visits])
    n = len(child)
    latent = rng.normal(size=(n_children, 3))[child]
    latent = latent + rng.normal(scale=0.25, size=(n, 3))
    edges = latent @ rng.normal(scale=0.2, size=(3, 1485)) + rng.normal(size=(n, 1485))
    pcs = latent @ rng.normal(size=(3, 12)) + rng.normal(scale=0.5, size=(n, 12))
    age = rng.uniform(3, 22, n_children)[child] + 3 * visit
    sex = rng.integers(0, 2, n_children)[child]
    site = rng.integers(0, 2, n_children)[child]
    education = rng.integers(0, 2, n_children)[child]
    motion = rng.uniform(0.05, 0.4, n)
    cohort = pd.DataFrame(edges, columns=[f"e{i}" for i in range(1, 1486)])
    cohort["scan_id"] = [f"BCP_{i + 1}_visit{v + 1}" for i, v in zip(child, visit)]
    cohort["imaging_numeric_id"] = [f"BCP_{i + 1}" for i in child]
    cohort["scan_age_months"] = age
    cohort["sex_final"] = np.where(sex == 1, "M", "F")
    cohort["FD"] = motion
    cohort["site_primary"] = np.where(site == 1, "UNC", "UMN")
    cohort["mother_edu_gt_college"] = education
    effects = np.array(
        [[1, 0.2, 0.1], [0.7, 0.6, 0.1], [0.1, 1, 0.3], [0.3, 0.2, 1], [0.2, 0.6, 0.8]]
    )
    covariate_effect = (
        0.5 * age
        - 0.01 * age**2
        + sex
        + 0.03 * age * sex
        - 0.001 * age**2 * sex
        - 5 * motion
        + 2 * site
        + 3 * education
    )
    for outcome, effect in zip(OUTCOMES, effects):
        score = (
            40 + covariate_effect + 8 * latent @ effect + rng.normal(scale=6, size=n)
        )
        cohort[outcome] = np.clip(score, 0, 100)
    source = pd.DataFrame(
        pcs, columns=[f"simulated_pc{i+1:02d}__zpred" for i in range(12)]
    )
    for key in ["scan_id", "imaging_numeric_id"]:
        source[key] = cohort[key]
    cohort.to_csv(output_dir / "simulated_cohort.csv", index=False)
    source.to_csv(output_dir / "simulated_pc_predictions.csv", index=False)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.out_dir, args.seed)


if __name__ == "__main__":
    main()
