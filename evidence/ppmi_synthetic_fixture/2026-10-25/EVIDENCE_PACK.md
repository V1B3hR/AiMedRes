# AiMedRes Evidence Pack — synthetic PPMI-schema fixture

> **Research-only.** This pack demonstrates one local, reproducible execution on a non-identifiable synthetic fixture. It is not PPMI evidence, patient-level performance, clinical validation, regulatory evidence, or a deployable model.

## Frozen inputs

- Study configuration: `configs/parkinsons_ppmi_study.yml` (`SHA-256`: `9e303ec951d88632793aa12c8deb81d9eb627775ed7aba60de12b3062a40fe01`)
- Synthetic fixture: `ParkinsonDatasets/synthetic_pet_sbr_fixture.csv` (`SHA-256`: `e803e33abbaea34197e5506e5b73111d0766edc54bc898883926ccb2e44687ae`)
- Git HEAD: `404a65b86fc1a5a75e48385133b5c31a48efc9a3`
- Worktree clean: `False`. The source-file hashes in `evidence_manifest.json` are the authoritative snapshot for this run.

## Control-run result

Execution time: 11.21 seconds.  Five-fold `StratifiedGroupKFold` was used with `PATNO` as group identifier.

| Model | ROC-AUC | 95% CI | Brier score |
|---|---:|---:|---:|
| Logistic Regression | 0.907 | 0.859–0.948 | 0.108 |
| Random Forest | 0.843 | 0.776–0.903 | 0.142 |
| SVM | 0.835 | 0.753–0.904 | 0.132 |

## Reproduction

```powershell
python scripts/reproduce_ppmi_benchmark.py --config configs/parkinsons_ppmi_study.yml --evidence-dir evidence/ppmi_synthetic_fixture/<new-empty-directory>
```

The directory must be new or empty; the runner refuses to overwrite an existing evidence pack. Full parameters, runtime context, Git state and source hashes are in `evidence_manifest.json`.

## Explicit limits

- The source fixture is synthetic; the values above must never be reported as PPMI or clinical-model metrics.
- A dirty worktree cannot substitute for a release commit or signed tag.
- An independent rerun, methodological review, authorized controlled-data work and external validation are still required.
