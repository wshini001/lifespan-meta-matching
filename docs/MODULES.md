# Module guide

The code follows four steps: source-model training and inference, cohort prediction, statistical inference, and interpretation.

## Source models and prediction

| Directory | Purpose |
| --- | --- |
| `src/source_models/` | Phenotype PCA, KRR/DNN source-model training and inference |
| `src/abcd/` | ABCD main prediction, PC models, domain analyses, statistics and interpretation |
| `src/hcp/` | HCP input preparation, source inference, nested prediction and alignment |
| `src/bcp/` | Composite and five developmental-domain prediction pipelines |

ABCD subdirectories are `pc_models/`, `domains/`, `statistics/`, `alignment/` and `haufe/`. BCP subdirectories are `composite/`, `receptive/`, `expressive/`, `visual/`, `fine_motor/` and `gross_motor/`.

The ABCD main-model configuration references `config/source_features.txt`, which lists the transferred features in input order.

## Statistics and interpretation

| Directory | Purpose |
| --- | --- |
| `src/statistics/` | Prediction-gain bootstrap and sign-flip inference |
| `src/abcd/statistics/` | ABCD 12-PC paired tests and PC-stream permutations |
| `src/abcd/alignment/`, `src/hcp/alignment/`, `src/source_alignment/` | Source-signature and reconstructed-phenotype alignments |
| `src/source_convergence/` | Compare phenotype alignments across source families |
| `src/domain_evidence/` | Integrate domain-level evidence |
| `src/abcd/haufe/`, `src/haufe/` | Connectivity patterns associated with prediction increments |
| `src/network_membership/` | ICA-to-Yeo network memberships |
| `src/network_enrichment/` | Network-pair enrichment tests |

## Source-model configurations

`src/source_models/config/phenotype_pc_screening.json` controls source-side PC screening. `phenotype_pc_krr.json` and `disease_pc_krr.json` control KRR training for phenotype PCs and disease PCs, respectively. These files specify candidate penalties, random seeds, split proportions and input checks.

Run entry scripts by file path with `--help` to view their arguments. Module `config/` directories contain analysis settings; `*_engine.py` files provide calculation helpers.

See [INPUTS.md](INPUTS.md) for input formats and the [README](../README.md) for installation and the simulated example.
