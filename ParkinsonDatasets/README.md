# Parkinson's Datasets & Benchmarking Resources

> **Governance Notice**: Real participant-level clinical and imaging data from the Parkinson's Progression Markers Initiative (PPMI) is subject to the PPMI Data Use Agreement (DUA) and must not be distributed publicly. See [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md) for full compliance protocols and data access registration procedures.

---

## 1. Directory Overview

This directory provides data loading hooks, schema definitions, and synthetic fixtures for the **AiMedRes Parkinson's Research Pipeline**:

- **`DATA_GOVERNANCE.md`**: Official compliance policy and procedures for authorized PPMI access.
- **`synthetic_pet_sbr_fixture.csv`**: A 100% synthetic, non-PHI benchmark dataset for automated CI/CD and open-source reproducibility testing (170 longitudinal records across 60 simulated participants).
- **`PPMI_*.pdf`**: Reference documentation detailing imaging acquisition protocols and SBR calculation methods.

---

## 2. Scientific Methodology & Audit Notes

A scientific audit of historical Parkinson’s scripts in this repository noted that earlier iterations utilized:
1. **Circular Target Formulation**: Binary labels split on the median of SBR measurements while using those exact same SBR measurements as inputs, producing artificial 99-100% accuracy.
2. **Patient-Level Data Leakage**: Random row splitting instead of grouping repeated visits by participant ID (`PATNO`), resulting in cross-fold data leakage.
3. **Pre-Split Data Transformation**: Fitting imputers and scalers prior to cross-validation splits.

### Current Architecture (Phase 1 Compliant):
- **Participant Grouping**: Cross-validation uses `GroupKFold` grouped strictly on `PATNO`.
- **Fold-Internal Preprocessing**: All scaling and imputation operations are encapsulated within an `sklearn.pipeline.Pipeline` fitted solely on training folds.
- **Non-Circular Clinical Outcomes**: Models evaluate independent clinical motor endpoints (e.g., `motor_progression_label`) rather than predicting functions of input features.

---

## 3. Usage & Execution

### Running on the Synthetic Fixture (Open Source & CI/CD)
```bash
python src/aimedres/training/train_parkinsons.py \
  --data-path ParkinsonDatasets/synthetic_pet_sbr_fixture.csv \
  --output-dir results/parkinsons_synthetic \
  --folds 5
```

### Running in an Authorized Institutional Environment
```bash
# Point to your authorized, restricted local PPMI data directory:
python src/aimedres/training/train_parkinsons.py \
  --data-path /secure/restricted/PPMI_Data/ \
  --output-dir results/parkinsons_official \
  --folds 5
```
