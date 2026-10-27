# Source parameters and restricted training artifacts

The repository distributes code and two generated demonstration tables. It does
not distribute participant records or trained source models. The historical
training and cohort pipelines remain available for use in authorized research
storage; their serialized artifacts are not publication packages.

## Source banks and their use

| Source bank | Typical saved files in the review collection | Target uses |
| --- | --- | --- |
| Raw-phenotype DNN, 120 outputs | One network per seed; five seeds | ABCD raw-feature augmentation and BCP analyses |
| Phenotype-PC DNN, 12 outputs | One network per seed; five seeds | Comparison/PC-transfer analyses |
| Phenotype-PC KRR, Final12 | One multi-output bundle per seed; five seeds | ABCD, HCP and BCP |
| Disease-PC KRR, 20 outputs | One multi-output bundle per seed; five seeds | ABCD and BCP; ABCD final43 selects 16 outputs |
| Raw-phenotype KRR | 120 single-target models | BCP composite/domain analyses |
| Raw-disease KRR | 330 single-target models per seed; five seeds | BCP domain analyses |

The full six-bank review collection has 1,790 model files, not 1,790 unique
phenotypes. The three source families in ABCD final43 are phenotype-PC KRR,
raw-phenotype DNN and disease-PC KRR. HCP's manuscript model uses Final12
phenotype-PC KRR. The additional raw KRR banks are relevant to BCP; their
inclusion differs across outcomes and model variants.

## Why the original artifact and a parameter export differ

Historical PC/raw KRR bundles contain per-participant RSFC matrices, training
indices, IDs and dual coefficients. The raw-disease bundles contain fit IDs
and require an external UKB RSFC table for the original inference method.
Deleting IDs alone does not remove participant-level feature rows.

For these **dot-product/correlation kernels only**, write
`beta = Z_fit.T @ dual_coef`. Prediction then uses `Z_new @ beta`. The standalone
export contains beta, output names, edge order, and target means/standard
deviations, with non-participant validation metadata. It contains no fit IDs,
training matrices, fit indices, dual coefficient vectors or participant outcomes.
This is a conversion of trained parameters, not a new fit or the original
joblib serialization. Floating-point order changes: synthetic comparisons use
`atol=1e-5, rtol=1e-5`. They do not establish equivalence on every historical
cohort prediction or replace real-input/downstream validation before changing
scientific results.

DNN state dictionaries can be copied byte-identically after checking their
keys, tensors and module metadata. Normalization parameters and output names
are exported separately. Full training logs, split files, participant PC scores
and predicted phenotype tables are excluded. A state-dictionary field check is
not a model-inversion or membership-inference assessment.

## Conversion in authorized local storage

Run from the repository root. Paths below are placeholders; keep exports in
private research storage. Never upload a legacy joblib/PTH from an untrusted
source for deserialization.

```bash
python src/source_models/export_source_parameters.py \
  --kind krr --source /work/project/private/pc_bundle.joblib \
  --out-dir /work/project/private/parameter_exports/pc_seed0

python src/source_models/export_source_parameters.py \
  --kind krr --source /work/project/private/raw_disease_bundle.joblib \
  --ukb-csv /authorized/current/UKB_final.csv --expected-ukb-rows 24668 \
  --out-dir /work/project/private/parameter_exports/disease_seed0_target000

python src/source_models/export_source_parameters.py \
  --kind dnn --source /work/project/private/stage1_seed0/dnn_stage1_model_stage1_seed0.pth \
  --out-dir /work/project/private/parameter_exports/dnn_seed0
```

Each destination must be new. The exporter validates a staged directory before
moving it into place. Raw-disease conversion uses the FC input locally, aligns
fit IDs, and does not copy that input into the export. This entry point exports
one model at a time; for multiseed/single-target banks, use the final resolved
model manifest to enumerate models rather than globbing intermediate fits.
The raw-disease prediction-only export omits nuisance-regression coefficients;
its rescaled prediction remains a covariate-residual score, not a diagnosis
probability or recovered raw phenotype.

## Standalone inference

```bash
python src/source_models/infer_source_parameters.py \
  --model-dir /work/project/private/parameter_exports/pc_seed0 \
  --cohort-csv /work/project/private/target_cohort.csv \
  --id-column scan_id --prediction-scale scaled \
  --out-csv /work/project/private/predictions/pc_seed0.csv
```

This entry point also reads the October 2026 private review exports with
`conversion_and_validation.json` for KRR, or original-weight DNN directories
with `model_metadata.json`. Those review files remain outside this repository.
An optional original `source_nuisance_parameters.npz` in the review bundle is
not needed for prediction. The new generic exporter has an explicit versioned
metadata schema; the validator distinguishes the two formats.

Select the prediction scale explicitly and preserve the original freeze-list
order and cohort alignment. Do not average across banks or automatically
replace historical Stage2A outputs. Five-seed averaging must use the same target
order and the scale expected by the downstream pipeline. Target predictions are
participant-level data too and must remain in authorized storage.

| Bank | Input transform retained |
| --- | --- |
| PC KRR | float64 row centering/L2, then float32 |
| Raw-phenotype KRR | float32 row centering/L2 |
| Raw-disease KRR | float64 row centering/L2; zero-norm rows rejected |
| DNN | Original RSFC cast to float32, without row unitization |

The existing cohort-specific inference scripts retain their historical bundle
interfaces. Use the new standalone entry point for parameter exports; a bare
parameters NPZ is not a drop-in replacement for a legacy joblib bundle.

## Preserve the historical workflow

All pre-existing source training and cohort-specific inference files remain
unchanged. The existing KRR readers require historical participant-bearing
bundles; the existing DNN reader can require a training log for output names.
Use the standalone entry point above when applying the reviewed parameter-only
models, rather than substituting its files into a legacy reader.

During DNN conversion, the exporter obtains verified target names from a
standalone manifest when available, or reads only `target_names` from the trusted
historical log. The exported `model_metadata.json` then carries names and edge
order; inference from that export needs neither a source training log nor a
participant split file. Original logs and split files are not copied.

For multi-seed inference, apply each required seed in its original target order,
choose the historically required prediction scale and average the corresponding
columns. The single-model CSV uses target names as columns. Historical bank
builders may expect `<target>__zpred`, additional cohort metadata and paired
identifiers; align those by the locked participant/scan keys before passing a
bank to the unchanged target pipeline. Do not join participants by row position.

The documented historical training/target workflow remains the route for
reproducing the original analysis with authorized inputs. Standalone inference
provides an alternative for the reviewed parameter exports; its schema checks
and generated-input tests are not a complete re-analysis of every historical
prediction, split or scientific result. No real UKB or target-cohort inputs or
trained models are distributed in this code repository.

## Publication boundary

`.gitignore` reduces accidental staging. `scripts/check_public_tree.py` rejects
tracked model/data/archive files outside the two exact simulated CSV paths,
checks obvious participant-array fields in JSON, and regenerates both examples
to compare their values. It is run in CI. This check does not inspect all storage,
GitHub releases or hidden data under arbitrary field names.

Removing a file from the current tree does not remove Git history. Inspect all
relevant refs, releases and external assets separately if a real-data exposure
is found. Do not rewrite history without identifying the affected objects and
coordinating that change.

Parameter conversion does not delete original files, prove that a model cannot
reveal participant information, or grant permission to retain, share or publish
it. Such decisions require the actual project/UKB requirements. This repository
makes no statement that all original data or backups have been deleted.
