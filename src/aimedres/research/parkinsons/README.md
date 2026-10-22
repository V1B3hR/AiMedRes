# Parkinson's Progression Research Package (Phase 2.1)

Decoupled, typed, scientifically rigorous research pipeline for Parkinson's disease progression modeling.

## 🔬 Core Methodological Principles

1. **Independent Clinical Target**:
   Predicts clinically verified motor progression milestone ($\Delta\text{MDS-UPDRS Part III} \ge 5.0$ points), avoiding circular feature-to-label leakage.
2. **Participant-Level Grouping (`StratifiedGroupKFold`)**:
   Longitudinal patient visits are grouped strictly by `PATNO`, ensuring 0 visit leakage across training and validation folds.
3. **Fold-Encapsulated Preprocessing**:
   Imputation and feature scaling are encapsulated within scikit-learn `Pipeline` objects and fitted exclusively on training splits.
4. **Statistical Rigor**:
   Reports 95% non-parametric bootstrap confidence intervals (1,000 iterations), PR-AUC, and Brier score probability calibration.
5. **Cryptographic Provenance**:
   Automatically records SHA-256 hashes of input dataset fixtures, saved model checkpoints, and runtime environment specifications in `provenance_manifest.json`.

---

## 🚀 Quick Usage

```python
from aimedres.research.parkinsons import (
    ParkinsonsPipelineConfig,
    ParkinsonsResearchPipeline,
)

# 1. Initialize with typed Pydantic configuration
config = ParkinsonsPipelineConfig(
    data_path="ParkinsonDatasets/synthetic_pet_sbr_fixture.csv",
    output_dir="results/ppmi_reproducibility_benchmark",
    n_folds=5,
    random_state=42,
    epochs=25,
)

# 2. Instantiate and execute the pipeline
pipeline = ParkinsonsResearchPipeline(config)
results = pipeline.run()

# 3. Inspect results and provenance manifest
print("Best Classical Model:", results["classical_results"])
print("Cryptographic Manifest:", results["manifest"]["model_artifacts"])
```

---

## 🧪 Testing

```bash
pytest tests/unit/test_research_parkinsons.py
```
