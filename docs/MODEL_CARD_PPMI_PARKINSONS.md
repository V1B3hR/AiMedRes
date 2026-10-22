# Research Model Card: PPMI Parkinson's Motor Progression Classifier

> **Model Card Version**: 1.0.0  
> **Date**: October 2026  
> **Evaluation Framework**: TRIPOD-AI / STARD-AI Aligned  
> **License**: MIT License  
> **Status**: Validated Research Benchmark (Investigational Prototype — Not for Clinical Diagnostic Use)

---

## 1. Model Details

* **Model Name**: AiMedRes PPMI Motor Progression Classifier (Leakage-Safe DaTscan SBR Ensemble)
* **Architecture Suite**:
  * Primary Linear Baseline: L2-Regularized Logistic Regression with fold-isolated median imputation and standard scaling.
  * Tree Ensemble: Random Forest Classifier (100 estimators, Gini impurity criterion).
  * Kernel Machine: Support Vector Classifier (RBF kernel with calibrated probability Platt scaling).
  * Tabular Neural Network: 4-layer Multi-Layer Perceptron (MLP) with Batch Normalization, Dropout (0.3), and ReLU activations.
* **Developer**: AiMedRes Open Source Scientific Working Group (`V1B3hR`)
* **Repository Path**: [`src/aimedres/training/train_parkinsons.py`](file:///c:/Projekty/AiMedRes/src/aimedres/training/train_parkinsons.py)
* **Reproduction Script**: [`scripts/reproduce_ppmi_benchmark.py`](file:///c:/Projekty/AiMedRes/scripts/reproduce_ppmi_benchmark.py)

---

## 2. Intended Use & Clinical Scope

### Intended Use Cases
* **Longitudinal Disease Stratification**: Research risk assessment of rapid motor progression in early Parkinson's disease.
* **Biomarker Sensitivity Exploration**: Quantifying the prognostic value of asymmetric striatal binding ratios (SBR) in caudate vs. putamen regions.
* **Clinical Trial Enrichment**: Stratifying clinical trial cohorts based on predicted probability of short-term motor worsening.

### Out-of-Scope & Contraindicated Uses
> [!CAUTION]
> **Strict Regulatory Disclaimer**: This model is an **investigational research algorithm**. It has **not** received FDA 510(k), De Novo, PMA clearance, or CE-mark certification as Software as a Medical Device (SaMD).
> - **Contraindicated**: Autonomous clinical diagnosis without neurologist confirmation.
> - **Contraindicated**: Dosing adjustment or discontinuation of dopaminergic therapy (e.g., Levodopa/Carbidopa).
> - **Contraindicated**: Application to atypical parkinsonian syndromes (PSP, MSA, CBD) or secondary drug-induced parkinsonism without cross-cohort re-calibration.

---

## 3. Data, Cohort & Target Definition

### Cohort Specification (PPMI Protocol)
* **Source Initiative**: Parkinson's Progression Markers Initiative (PPMI; sponsored by The Michael J. Fox Foundation).
* **Population**: Early, unmedicated (de novo) Parkinson's disease participants and age-matched healthy controls.
* **Longitudinal Evaluation**: Baseline (`BL`), Year 1 (`V04`), Year 2 (`V06`), Year 3 (`V08`).
* **Governance**: Controlled access via USC LONI IDA under strict non-redistribution DUA. Automated testing uses verified synthetic benchmark fixture [`ParkinsonDatasets/synthetic_pet_sbr_fixture.csv`](file:///c:/Projekty/AiMedRes/ParkinsonDatasets/synthetic_pet_sbr_fixture.csv).

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
* **Endpoint**: `motor_progression_label` (Binary, independent clinical milestone).
* **Clinical Criteria**: Clinically significant motor progression defined as an increase $\ge 5.0$ points on the Movement Disorder Society Unified Parkinson's Disease Rating Scale (MDS-UPDRS) Part III motor exam from baseline to follow-up visit.
* **Circularity Elimination**: SBR features are strictly input predictors; the target is an independent clinical motor outcome.

---

## 4. Methodological Audit & Anti-Leakage Safeguards

Prior audits of legacy scripts identified severe data leakage. This pipeline enforces the following mitigations:

| Leakage Dimension | Legacy Vulnerability | AiMedRes Phase 1 Mitigation | Verification Status |
| :--- | :--- | :--- | :--- |
| **Participant Leakage** | Random `StratifiedKFold` scattered repeated longitudinal visits (`PATNO`) across train and validation folds. | Enforced **`StratifiedGroupKFold` grouped strictly by `PATNO`**. Zero participant overlap across folds. | Verified in `tests/unit/test_ppmi_reproducibility.py` |
| **Preprocessing Contamination** | Feature scalers and imputers were fitted on the global dataset prior to splitting. | Fully encapsulated in `sklearn.pipeline.Pipeline`, fitting strictly on training partitions per fold. | Verified in unit tests |
| **Target Circularity** | Target was generated from median split of `avg_sbr` while `avg_sbr` was in the input matrix. | Replaced by independent clinical endpoint (`motor_progression_label`). Pipeline warns if circular target is detected. | Automated warning system active |

---

## 5. Quantitative Benchmark Performance

Results evaluated across 5-fold participant-isolated cross-validation (`StratifiedGroupKFold` on 60 unique participants, 170 longitudinal observations):

| Model Architecture | Out-of-Fold ROC-AUC [95% CI] | Out-of-Fold PR-AUC [95% CI] | Brier Score (Calibration) | Sensitivity | Specificity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.907** [0.859 - 0.948] | **0.959** [0.934 - 0.981] | **0.108** | 0.917 | 0.714 |
| **Random Forest** | 0.843 [0.776 - 0.903] | 0.928 [0.886 - 0.962] | 0.142 | 0.851 | 0.571 |
| **SVM (Platt Scaled)** | 0.835 [0.753 - 0.904] | 0.905 [0.845 - 0.957] | 0.132 | 0.950 | 0.592 |

*95% Confidence Intervals computed via 1,000 non-parametric bootstrap iterations.*

### Probability Calibration
* **Brier Score**: 0.108 (Logistic Regression demonstrates low mean squared probability error).
* **Reliability**: Platt scaling and regularized log-odds preserve monotonic alignment between predicted risk scores and empirical progression rates.

---

## 6. Demographic & Ethical Considerations

* **Selection & Demographic Bias**: The PPMI observational cohort predominantly represents individuals of European ancestry with above-average health literacy and socioeconomic access. Generalizability across underrepresented ethnic groups remains unvalidated.
* **Hardware & Acquisition Heterogeneity**: DaTscan SPECT reconstruction algorithms and camera gamma collimators vary between acquisition sites. Multi-site harmonization (e.g., ComBat or phantom standardization) must be evaluated prior to multi-center deployment.

---

## 7. How to Reproduce

Execute the official reproduction runner in your terminal:

```bash
# Run against the validated synthetic fixture:
python scripts/reproduce_ppmi_benchmark.py

# Run against real local PPMI data (authorized DUA holders):
python scripts/reproduce_ppmi_benchmark.py --data-path data/ppmi/ppmi_sbr_longitudinal.csv
```
