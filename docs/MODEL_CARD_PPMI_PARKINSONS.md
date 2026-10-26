# Research Model Card: synthetic PPMI-schema reproducibility benchmark

> **Model Card Version**: 1.1.0
> **Date**: 25 October 2026
> **Evaluation Framework**: Internal reproducibility controls informed by TRIPOD-AI/STARD-AI; no external compliance assessment
> **License**: MIT License  
> **Status**: Local synthetic-fixture benchmark. It is not a validated PPMI result, external validation, clinical-performance estimate, or diagnostic model.

---

## 1. Model Details

* **Model Name**: AiMedRes synthetic PPMI-schema motor-progression benchmark
* **Architecture Suite**:
  * Primary Linear Baseline: L2-Regularized Logistic Regression with fold-isolated median imputation and standard scaling.
  * Tree Ensemble: Random Forest Classifier (100 estimators, Gini impurity criterion).
  * Kernel Machine: Support Vector Classifier (RBF kernel with calibrated probability Platt scaling).
  * Tabular Neural Network: 4-layer Multi-Layer Perceptron (MLP) with Batch Normalization, Dropout (0.3), and ReLU activations.
* **Developer**: AiMedRes Open Source Scientific Working Group (`V1B3hR`)
* **Repository Path**: [`src/aimedres/training/train_parkinsons.py`](../src/aimedres/training/train_parkinsons.py)
* **Reproduction Script**: [`scripts/reproduce_ppmi_benchmark.py`](../scripts/reproduce_ppmi_benchmark.py)

---

## 2. Intended Use & Clinical Scope

### Current intended use
* **Reproducibility-control demonstration**: exercising participant-grouped cross-validation, fold-contained preprocessing, provenance generation, and deterministic local reruns on a non-identifiable synthetic fixture.
* **Method-development input**: providing a starting point for a separately governed, preregistered retrospective study after an authorised data holder and independent methodologist are engaged.

### Out-of-Scope & Contraindicated Uses
> [!CAUTION]
> **Strict research disclaimer**: This benchmark has not been evaluated on controlled PPMI data in the current evidence pack. It has **not** received FDA 510(k), De Novo, PMA clearance, or CE-mark certification as Software as a Medical Device (SaMD).
> - **Contraindicated**: Autonomous clinical diagnosis without neurologist confirmation.
> - **Contraindicated**: Dosing adjustment or discontinuation of dopaminergic therapy (e.g., Levodopa/Carbidopa).
> - **Contraindicated**: Clinical-trial enrichment, patient stratification, or any use with patient data outside an approved research protocol.

---

## 3. Data, Cohort & Target Definition

### Current evaluation fixture
* **Source**: [`ParkinsonDatasets/synthetic_pet_sbr_fixture.csv`](../ParkinsonDatasets/synthetic_pet_sbr_fixture.csv), a committed non-identifiable synthetic fixture with 170 observations from 60 synthetic participants.
* **Purpose**: schema and leakage-control regression testing only. Its covariates, longitudinal visits, and labels are not patient records and do not establish a PPMI population, prevalence, prognosis, or clinical utility.
* **Future controlled-data work**: PPMI access requires an authorised holder and DUA. The governance requirements are in [`ParkinsonDatasets/DATA_GOVERNANCE.md`](../ParkinsonDatasets/DATA_GOVERNANCE.md).

### Input Features ($X$)
1. `PET_CAUDATE_L`: Left caudate striatal binding ratio (SBR)
2. `PET_CAUDATE_R`: Right caudate striatal binding ratio (SBR)
3. `PET_PUTAMEN_L`: Left putamen striatal binding ratio (SBR)
4. `PET_PUTAMEN_R`: Right putamen striatal binding ratio (SBR)
5. `PET_CBM_RATIO`: Cerebellar reference region normalisation ratio
6. `PET_OCCIP_RATIO`: Occipital reference region ratio
7. `AGE`: Age at scan examination (years)
8. `SEX`: Biological sex (0 = Female, 1 = Male)

### Prediction Target ($y$)
* **Fixture endpoint**: `motor_progression_label`, a synthetic binary label used to test the pipeline's non-circular target interface.
* **Clinical endpoint not yet evidenced**: a future protocol may define an independent endpoint, such as longitudinal MDS-UPDRS Part III change, before analysis. This repository does not currently demonstrate that definition or outcome on controlled PPMI records.
* **Circularity control**: the runner separates SBR-like fixture features from the synthetic target and warns on known circular-target patterns. This is a software control, not clinical endpoint validation.

---

## 4. Methodological Audit & Anti-Leakage Safeguards

Prior audits of legacy scripts identified severe data leakage. This pipeline enforces the following mitigations:

| Leakage Dimension | Legacy Vulnerability | AiMedRes Phase 1 Mitigation | Verification Status |
| :--- | :--- | :--- | :--- |
| **Participant Leakage** | Random `StratifiedKFold` scattered repeated longitudinal visits (`PATNO`) across train and validation folds. | The synthetic benchmark uses **`StratifiedGroupKFold` grouped by `PATNO`** when the identifier is present. | Verified by the focused PPMI configuration/research tests and local synthetic reruns. |
| **Preprocessing Contamination** | Feature scalers and imputers were fitted on the global dataset prior to splitting. | Fully encapsulated in `sklearn.pipeline.Pipeline`, fitting strictly on training partitions per fold. | Verified in unit tests |
| **Target Circularity** | Target was generated from median split of `avg_sbr` while `avg_sbr` was in the input matrix. | The fixture exposes a separate synthetic target interface; a real independent endpoint still requires protocol and source-data review. | Software warning/control only. |

---

## 5. Synthetic-fixture execution metrics

The values below were produced locally on the synthetic fixture with 5-fold participant-grouped cross-validation (60 synthetic participants, 170 observations). They are regression-test outputs only and **must not** be cited as PPMI, patient, clinical, external-validation, or clinical-trial-enrichment performance.

| Model Architecture | Out-of-Fold ROC-AUC [95% CI] | Out-of-Fold PR-AUC [95% CI] | Brier Score (Calibration) | Sensitivity | Specificity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.907** [0.859 - 0.948] | **0.959** [0.934 - 0.981] | **0.108** | 0.917 | 0.714 |
| **Random Forest** | 0.843 [0.776 - 0.903] | 0.928 [0.886 - 0.962] | 0.142 | 0.851 | 0.571 |
| **SVM (Platt Scaled)** | 0.835 [0.753 - 0.904] | 0.905 [0.845 - 0.957] | 0.132 | 0.950 | 0.592 |

*95% Confidence Intervals computed via 1,000 non-parametric bootstrap iterations.*

### Probability Calibration
* **Brier Score**: 0.108 for this fixture and seed.
* **Interpretation limit**: calibration has not been assessed on a clinical cohort; no statement about patient risk or reliability is justified.

---

## 6. Demographic & Ethical Considerations

* **Selection & Demographic Bias**: demographic performance has not been assessed because the active evidence is synthetic. A future controlled-data study must define subgroup reporting before modelling.
* **Hardware & Acquisition Heterogeneity**: the fixture does not represent scanner/site heterogeneity. Multi-site harmonisation and external validation would be required before any clinical or multi-centre claim.

---

## 7. How to Reproduce

Execute the official reproduction runner in your terminal:

```bash
# Run the versioned synthetic fixture (research reproducibility only):
python scripts/reproduce_ppmi_benchmark.py --config configs/parkinsons_ppmi_study.yml

# Authorised PPMI DUA holders may run a local research experiment after protocol review:
python scripts/reproduce_ppmi_benchmark.py --config configs/parkinsons_ppmi_study.yml --data-path data/ppmi/ppmi_sbr_longitudinal.csv --output-dir .local-results/ppmi_research_run
```
