# Inputs

| Workflow | Required inputs |
| --- | --- |
| Source models | Connectivity and phenotype tables, phenotype definitions, selected-feature lists and source-model settings |
| Cohort prediction | Cohort and covariate tables, transferred source predictions, fold assignments and feature definitions |
| Signature interpretation | Out-of-fold predictions, source-prediction tables, PCA loadings and phenotype labels |
| Haufe analysis | Connectivity features, prediction increments, edge-to-node mapping and branch settings |
| Network membership | ICA spatial maps, retained-component list and spatially aligned Yeo atlas |
| Network enrichment | Network memberships and edge-pattern coefficients |
| Statistical inference | Outcomes, out-of-fold predictions, fold assignments and grouping identifiers where applicable |

Provide input paths through the relevant entry script. Use `--help` to view its arguments and configuration options. Participant or scan alignment, feature order and fold assignments must match the selected pipeline. Grouped analyses must keep observations from the same child or family within one fold.

## Simulated example

`examples/generate_simulated_data.py` generates a longitudinal cohort table with five developmental outcomes and a matching PC-feature table for the BCP demonstration. Column definitions are provided in the generator. The example uses CSV inputs; supported formats for other workflows are specified by their entry scripts.

See the [README](../README.md) to run the example and [MODULES.md](MODULES.md) to locate the analysis scripts.
