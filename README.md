# Cross-age meta-matching

This repository implements the training and transfer of UK Biobank brain–phenotype prediction models to HCP, ABCD and BCP cohorts. It includes cohort-specific prediction pipelines, statistical inference and interpretation of transferred signatures and connectivity patterns.

## Repository structure

| Directory | Contents |
| --- | --- |
| `src/source_models/` | Source phenotype preparation, PCA, KRR/DNN training and frozen-model inference |
| `src/abcd/` | ABCD prediction, PC-based transfer and domain analyses |
| `src/hcp/` | HCP input preparation, source-model inference and nested prediction |
| `src/bcp/` | BCP composite and five developmental-domain prediction pipelines |
| Other modules in `src/` | Statistical inference, signature alignment, Haufe patterns and network enrichment |
| `examples/` | Simulated-data generation and demonstration scripts |
| `docs/` | Module descriptions, input formats and development instructions |

See the [module guide](docs/MODULES.md) for individual scripts.

## Installation

Use Python 3.10 or later. From the repository root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Simulated demonstration

Run the demonstration from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 python examples/run_demo.py
```

The demonstration generates longitudinal scans for 36 simulated children with five developmental outcomes: receptive language, expressive language, visual reception, fine motor and gross motor. It uses 1,485 connectivity features, 12 PC features and eight age, sex, motion, site and maternal-education covariates.

Each outcome uses the same demonstration model with child-grouped nested cross-validation, training-fold outcome residualization, and an RSFC/PC prediction blend. Bootstrap and sign-flip inference resample children as clusters. The example reuses BCP preprocessing helpers; domain-specific prediction pipelines are located in their respective `src/bcp/` modules.

By default, generated files are stored in a temporary directory and deleted on exit. To retain them for inspection:

```bash
OPENBLAS_NUM_THREADS=1 python examples/run_demo.py --work-dir work/simulated_demo
```

## Usage

Run the entry script for the desired cohort or analysis and supply its input paths and configuration. For example, display the HCP prediction arguments with:

```bash
python src/hcp/run_hcp_rsfc_pc_nested_cv.py --help
```

For prediction-gain inference, use `src/statistics/infer_prediction_gain.py` with the corresponding cohort configuration. ABCD PC-route paired tests and PC-stream permutations are implemented separately in `src/abcd/statistics/`.

See the [input guide](docs/INPUTS.md) for required file formats and the [module guide](docs/MODULES.md) for analysis entry points. Formatting tools and commands are described in the [development guide](docs/DEVELOPMENT.md).

## Reference

He, T. et al. Meta-matching as a simple framework to translate phenotypic predictive models from big to small data. *Nature Neuroscience* **25**, 795–804 (2022). https://doi.org/10.1038/s41593-022-01059-9
