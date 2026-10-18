# Running the code

Run commands from the repository root. The order is: prepare UKB source models, apply the frozen models to a target cohort, fit the target-cohort model, then calculate statistics and interpretation.

UKB models are transferred independently to ABCD, HCP and BCP. Choose the target-cohort workflow you need; the three target workflows can be run independently.

## Start with the simulated example

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python examples/run_demo.py
```

The example reads the two bundled CSV files in `examples/simulated_data/`. They contain generated longitudinal scans for 36 artificial children (`BCP_1` to `BCP_36`), five developmental outcomes and matching PC features. `run_demo.py` fits child-grouped nested prediction models and calculates bootstrap and sign-flip statistics. Analysis outputs are temporary by default.

To keep the analysis outputs:

```bash
OPENBLAS_NUM_THREADS=1 python examples/run_demo.py --work-dir work/simulated_demo
```

To regenerate the simulated inputs in a separate folder and run them:

```bash
python examples/generate_simulated_data.py --out-dir work/simulated_inputs
OPENBLAS_NUM_THREADS=1 python examples/run_demo.py --input-dir work/simulated_inputs
```

The example demonstrates a common RSFC/PC model. The domain-specific models are listed under Step 4.

## Step 1: Prepare and train UKB source models

Use `src/source_models/` for the shared source models:

| Order | Script | Purpose |
| --- | --- | --- |
| 1 | `build_phenotype_pcs.py` | Derive phenotype PCs within UKB phenotype blocks. |
| 2 | `screen_phenotype_pcs.py` | Select phenotype PCs using a source-side prediction screen. |
| 3 | `build_disease_pcs.py` | Derive PCs from UKB disease phenotypes, for disease-PC routes. |
| 4 | `train_source_krr.py` or `train_source_dnn.py` | Train the required UKB connectivity-to-phenotype models. |
| 5 | `infer_source_krr.py` or `infer_source_dnn.py` | Apply the frozen source models to target connectivity features. |

Raw-phenotype models use their phenotype targets directly. Choose the source-model family required by the target workflow. BCP composite raw-KRR training uses `src/bcp/composite/train_raw_krr.py`; individual disease-source training uses `src/bcp/receptive/train_disease_source.py`.

The output of source inference is a table of transferred phenotype predictions. These predictions become input features for target-cohort modeling.

## Step 2: UKB to ABCD

Run these scripts in `src/abcd/` in order:

| Order | Script | Purpose |
| --- | --- | --- |
| 1 | `lock_abcd_folds.py` | Assign the participant-level outer folds for the selected cohort. |
| 2 | `assemble_transferred_features.py` | Assemble phenotype-PC, raw-DNN and disease-PC predictions in the configured feature order. |
| 3 | `run_abcd_final43.py` | Fit the RSFC baseline and transferred-feature stream, then combine their predictions using inner cross-validation. |

Supply the cohort, fold files and assembled source-feature inputs to the prediction script. For the two phenotype-PC routes, use `pc_models/run_abcd_pc_models.py` instead. `domains/run_domain_analysis.py` runs domain-subset, leave-one-domain-out and multi-kernel analyses after their inputs have been prepared.

## Step 3: UKB to HCP

Run these scripts in `src/hcp/` in order:

| Order | Script | Purpose |
| --- | --- | --- |
| 1 | `prepare_hcp_input.py` | Align connectivity, participant information, covariates and family-grouped folds. |
| 2 | `apply_ukb_pc_models.py` | Apply the frozen UKB PC models to HCP connectivity. |
| 3 | `run_hcp_rsfc_pc_nested_cv.py` | Fit the RSFC and PC models and select their additive combination within the training data. |

## Step 4: UKB to BCP

Prepare the outcome-specific cohort, child-grouped folds and required source-prediction banks. Follow the sequence for the selected outcome in `src/bcp/`:

| Outcome | Run order | Purpose |
| --- | --- | --- |
| Composite | `composite/prepare_composite_cohort.py` → `fit_composite_baseline.py` → `fit_composite_transfer.py` → `blend_composite_predictions.py` | Prepare the cohort, fit the baseline and raw-KRR transfer branch, then form the final fixed-weight blend. |
| Receptive language | `receptive/align_transfer_bank.py` → `run_receptive_language.py` | Align source predictions, fit the PC anchor and select conditional source corrections. |
| Expressive language | `expressive/build_transfer_bank.py` → `run_expressive_language.py` | Align disease and raw-DNN predictions, fit the disease anchor and add the DNN contribution. |
| Visual reception | `visual/prepare_fmri_baseline_features.py` → `align_transfer_bank.py` → `run_visual_reception.py` | Prepare signed fMRI baseline features, align source predictions and fit KRR/DNN transfer stages. |
| Fine motor | `fine_motor/fit_fine_motor_baseline.py` → `run_fine_motor.py` | Fit the RSFC baseline and the Disease Top2, PC12 and fixed-Top88 DNN sequence. |
| Gross motor | `gross_motor/build_transfer_bank.py` → `run_gross_motor.py` | Align the source banks and fit the child-balanced KRR/disease/DNN combination. |

Script names without a repeated directory prefix remain in the outcome directory shown in the same row. Source-inference and ranking helpers in these directories prepare the required banks and feature lists before model fitting; see [docs/MODULES.md](docs/MODULES.md).

## Step 5: Statistics and interpretation

Use the out-of-fold predictions from the selected target model:

| Analysis | Entry point or directory | Purpose |
| --- | --- | --- |
| Prediction gain | `src/statistics/infer_prediction_gain.py` | Calculate COD gain, cluster-bootstrap intervals and paired sign-flip tests with the matching cohort configuration. |
| ABCD PC-route statistics | `src/abcd/statistics/run_statistics.py` | Run paired tests, bootstrap intervals and PC-stream permutations. |
| Source-signature alignment | `src/abcd/alignment/`, `src/hcp/alignment/`, `src/source_alignment/` | Relate source predictions or reconstructed phenotypes to prediction increments. |
| Connectivity patterns | `src/abcd/haufe/`, `src/haufe/` | Calculate edge patterns and node involvement from prediction increments. |
| Network analysis | `src/network_membership/run_network_membership.py` → `src/network_enrichment/run_network_enrichment.py` | Calculate ICA-to-Yeo memberships, then test network-pair enrichment using edge patterns. |

## Input paths and settings

Each entry script accepts its own input paths and options. For example:

```bash
python src/abcd/run_abcd_final43.py --help
python src/bcp/receptive/run_receptive_language.py --help
```

Use the configuration for the selected analysis. Preserve participant or scan alignment, feature order and fold assignments between steps. See [docs/INPUTS.md](docs/INPUTS.md) for input conventions and [README.md](README.md) for installation and the repository overview.
