# 📜 PPMI Data Governance & Compliance Policy

> **Crucial Rule**: Under the Parkinson’s Progression Markers Initiative (PPMI) Data Use Agreement (DUA), **individual patient-level research data must NOT be redistributed publicly** (e.g., via public GitHub repositories, open cloud buckets, or unauthenticated downloads).

---

## 1. Regulatory Context & PPMI Data Use Agreement

The Parkinson’s Progression Markers Initiative (PPMI) is a landmark observational study sponsored by The Michael J. Fox Foundation for Parkinson’s Research. Access to PPMI data requires:
1. Registration with the PPMI study portal at [ppmi-info.org](https://www.ppmi-info.org/).
2. Execution of the PPMI Data Use Agreement.
3. Explicit commitment **never to redistribute** primary PPMI datasets or attempt participant re-identification.
4. Appropriate citation of PPMI in all research publications and preprints.

---

## 2. Repository Data Handling Policy

To maintain compliance with institutional guidelines and open-source standards:

1. **No Restricted PPMI Records in Public Version Control**:
   - Any raw PPMI clinical or imaging CSVs downloaded from the PPMI portal must be maintained in **restricted, encrypted, institution-approved storage** (`.gitignore` protects against accidental tracking).
   - Only non-PHI metadata manifests, schema definitions, and synthetic fixtures may be committed to Git.
2. **Synthetic CI/CD Fixture**:
   - For automated testing, continuous integration, and pipeline validation, use `ParkinsonDatasets/synthetic_pet_sbr_fixture.csv`.
   - This fixture contains 100% synthetic, non-identifiable participant profiles with realistic covariate distributions, longitudinal visit structures, and non-circular progression outcomes.
3. **Reproducible Local Execution**:
   - Authorized researchers who have executed the PPMI DUA can point the training pipeline to their secure local download directory using `--data-path /path/to/restricted/ppmi/`.

---

## 3. Scientific Methodology & Audit Findings

An internal audit of legacy Parkinson’s pipeline scripts identified two methodological flaws that must be strictly avoided:

### ⚠️ Flaw 1: Circular Prediction Target
- **Legacy Approach**: Derived the binary label `status` by taking the median split of the average striatal binding ratio (`avg_sbr`), while simultaneously passing those same SBR features into the input feature matrix.
- **Correction**: Binary or continuous outcomes must represent **independent clinical endpoints** (e.g., longitudinal MDS-UPDRS motor progression, Hoehn & Yahr staging, or verified diagnostic category), never a direct mathematical partition of the model's own input features.

### ⚠️ Flaw 2: Longitudinal Visit Leakage Across Cross-Validation Folds
- **Legacy Approach**: Split rows randomly via standard `StratifiedKFold`. Because PPMI contains repeated longitudinal visits for individual participants (`PATNO`), records from the same patient leaked into both training and validation folds.
- **Correction**: Model evaluation must enforce **participant-level grouping (`GroupKFold` grouped on `PATNO`)** so that all visits for any given individual remain strictly within a single fold. Preprocessing (imputation and scaling) must be fit *strictly* within training folds.

---

## 4. How Authorized Researchers Obtain PPMI Data

1. Visit [https://www.ppmi-info.org/access-data-specimens/data](https://www.ppmi-info.org/access-data-specimens/data).
2. Create an account and submit a Data Access Application describing your research intent.
3. Upon approval, download the relevant standardized tabular packages (e.g., `PET_SBR_Analysis`, `MDS_UPDRS_Part_III`).
4. Place files in a designated local directory outside git or under an ignored folder:
   ```bash
   python src/aimedres/training/train_parkinsons.py --data-path /path/to/local/ppmi_files/
   ```

For a comprehensive guide covering LONI IDA (PPMI, ADNI, MarkVCID) and open-access research repositories (OASIS, PRO-ACT, PhysioNet), see [docs/DATASET_CATALOG.md](../docs/DATASET_CATALOG.md).
