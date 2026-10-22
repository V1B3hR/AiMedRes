# AiMedRes: SaMD Requirements & Verification Traceability Matrix

> **Document ID**: AIMEDRES-QMS-TRM-001  
> **Version**: 1.0.0  
> **Effective Date**: October 2026  
> **Standards Alignment**: IEC 62304:2006+A1:2015 (Class B) | ISO 14971:2019 | FDA Software Guidance (2023)  
> **Verification Coverage**: 100% of SaMD requirements mapped to automated test assertions in `tests/unit/`  

---

## 1. Scope, Purpose & Regulatory Standards

This Software as a Medical Device (SaMD) Traceability Matrix establishes end-to-end design control linkage between:
1. **Clinical User Needs (UN)**
2. **Software Requirement Specifications (SRS)**
3. **ISO 14971 Clinical Hazard Analyses & Risk Controls (HAZ / RC)**
4. **IEC 62304 Software Verification Tests (VER)**

In accordance with **IEC 62304 Section 5.1.1** (Software development planning) and **Section 5.7.4** (Software verification traceability), every clinical hazard and functional requirement must be programmatically verifiable through deterministic regression tests in the repository's continuous integration gateway.

---

## 2. Software Safety Classification (IEC 62304 § 4.3)

AiMedRes is classified as **IEC 62304 Software Safety Class B**:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                 IEC 62304 SOFTWARE SAFETY CLASSIFICATION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Class A: No injury or damage to health is possible.                         │
│ Class B: Non-serious injury is possible. (ASSIGNED CLASSIFICATION)          │
│ Class C: Death or serious injury is possible.                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Rationale: The software provides adjunctive clinical decision support to    │
│ licensed neurologists. Erroneous output could misdirect non-urgent staging  │
│ or delay routine follow-up, but independent clinical examination and HCP    │
│ reviewability (FDA § 3060 Criterion 4) mitigate hazards below Class C.     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. ISO 14971 Hazard Analysis & Risk Control Catalogue

| Hazard ID | Hazard Description | Pre-Mitigation Risk | Risk Control Measure (RC) | Post-Mitigation Risk | Verification Test ID |
| :--- | :--- | :---: | :--- | :---: | :--- |
| `HAZ-001` | **Participant-Level Longitudinal Leakage**: Patient repeat visits appear in both train and validation splits, causing overfitting and inflated false confidence. | **HIGH** | `RC-001`: Enforce `StratifiedGroupKFold` on participant IDs (`PATNO` / `Subject ID` / `patient_id`) with zero cross-fold subject intersection. | **LOW** | `VER-001`<br>`VER-002` |
| `HAZ-002` | **Circular Target Leakage**: Clinical rating derivatives (e.g. `CDR` when predicting dementia, or baseline UPDRS) are ingested into features, producing tautological predictions. | **HIGH** | `RC-002`: Automated audit filter in dataset loaders dropping clinical target derivatives from the input matrix prior to feature assembly. | **LOW** | `VER-003`<br>`VER-004` |
| `HAZ-003` | **Preprocessing Contamination**: Scaling or imputation parameters computed on pooled dataset prior to cross-validation, leaking test distributions. | **MED** | `RC-003`: Encapsulate imputers and scalers within `sklearn.pipeline.Pipeline`, fitting strictly on training fold indices. | **LOW** | `VER-005` |
| `HAZ-004` | **Insecure Identity / Production Credential Compromise**: Weak keys or in-memory credentials permitting unauthorized data access or session tampering. | **HIGH** | `RC-004`: `AuthDatabaseStore` with SQLite persistence, PBKDF2 password hashing + salt, instant API key/session revocation, and fail-closed secret validation. | **LOW** | `VER-006`<br>`VER-007` |
| `HAZ-005` | **Unvalidated Cognitive Agent Execution in Clinical Mode**: Heuristic simulation agents executing unvalidated clinical reasoning in production clinical mode. | **HIGH** | `RC-005`: Quarantine cognitive agent modules in `src/aimedres/simulation/` with runtime fail-closed guard `assert_simulation_allowed()`. | **LOW** | `VER-008` |
| `HAZ-006` | **Silent Point-Estimate Failure**: Models outputting point estimates without confidence intervals, masking predictive uncertainty from clinicians. | **MED** | `RC-006`: Compute 95% bootstrap confidence intervals ($B = 1,000$) and Brier score calibration loss on out-of-fold predictions. | **LOW** | `VER-009` |
| `HAZ-007` | **Software Supply-Chain Vulnerabilities**: Insecure third-party packages or untracked dependencies deployed to production. | **MED** | `RC-007`: Automated CycloneDX 1.5 Software Bill of Materials (SBOM) generator in `scripts/generate_sbom.py` capturing all direct and transitive dependencies. | **LOW** | `VER-010` |
| `HAZ-008` | **Unprivileged Deployment Breach**: Container executing with root privileges or lacking health probes, causing container escape or silent crash loops. | **MED** | `RC-008`: Enforce `runAsNonRoot: true`, non-root user `uid: 10001`, `readOnlyRootFilesystem: true`, and verified liveness/readiness probes in Helm/K8s. | **LOW** | `VER-011` |

---

## 4. Complete SaMD Requirements Traceability Matrix

The table below maps each Clinical User Need through functional requirements, risk controls, verification tests, and the exact automated unit test executing in CI:

| User Need (UN) | Software Requirement (SRS) | Hazard ID | Risk Control | Verification Test ID | Automated Test Suite Location | CI Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **UN-01** (Accurate Progression Trajectory) | **SRS-01**: The system shall isolate longitudinal patient evaluations across cross-validation splits. | `HAZ-001` | `RC-001` | `VER-001` | `tests/unit/test_ppmi_reproducibility.py::test_zero_patno_leakage_across_cross_validation_folds` | **PASS** |
| **UN-01** (Accurate Progression Trajectory) | **SRS-02**: The system shall verify zero patient overlap across 5 folds for all open clinical datasets (OASIS, PRO-ACT, PTB-XL, Telemonitoring). | `HAZ-001` | `RC-001` | `VER-002` | `tests/unit/test_open_access_datasets.py::test_oasis_zero_subject_leakage`<br>`tests/unit/test_open_access_datasets.py::test_parkinsons_telemonitoring_zero_subject_leakage` | **PASS** |
| **UN-02** (Scientific Target Validity) | **SRS-03**: The system shall exclude circular clinical rating scales (`CDR`, baseline UPDRS) from feature matrices. | `HAZ-002` | `RC-002` | `VER-003` | `tests/unit/test_open_access_datasets.py::test_oasis_loader_and_schema`<br>`tests/unit/test_open_access_datasets.py::test_parkinsons_telemonitoring_loader_and_schema` | **PASS** |
| **UN-02** (Scientific Target Validity) | **SRS-04**: The system shall evaluate independent motor progression targets distinct from baseline severity. | `HAZ-002` | `RC-002` | `VER-004` | `tests/unit/test_ppmi_reproducibility.py::test_feature_label_independence` | **PASS** |
| **UN-03** (Methodological Reproducibility) | **SRS-05**: The system shall encapsulate preprocessing scalers and imputers within fold-isolated pipelines. | `HAZ-003` | `RC-003` | `VER-005` | `tests/unit/test_open_access_datasets.py::test_end_to_end_oasis_leakage_safe_training` | **PASS** |
| **UN-04** (Clinical Data Security) | **SRS-06**: The system shall fail-closed in production if encryption secrets are missing, default, or under 32 characters. | `HAZ-004` | `RC-004` | `VER-006` | `tests/unit/test_security_and_architecture.py::test_fail_closed_validation_in_production_mode` | **PASS** |
| **UN-04** (Clinical Data Security) | **SRS-07**: The system shall persist credentials in SQLite, verify PBKDF2 password hashes, and instantly revoke API keys. | `HAZ-004` | `RC-004` | `VER-007` | `tests/unit/test_security_and_architecture.py::test_auth_database_store_lifecycle` | **PASS** |
| **UN-05** (Clinical Governance) | **SRS-08**: The system shall quarantine experimental cognitive simulation agents with fail-closed runtime checks. | `HAZ-005` | `RC-005` | `VER-008` | `tests/unit/test_security_and_architecture.py::test_simulation_quarantine_enforcement_in_production` | **PASS** |
| **UN-06** (Clinical Decision Transparency) | **SRS-09**: The system shall report 95% bootstrap confidence intervals and Brier reliability calibration. | `HAZ-006` | `RC-006` | `VER-009` | `tests/unit/test_ppmi_reproducibility.py::test_end_to_end_pipeline_execution` | **PASS** |
| **UN-07** (Supply Chain Integrity) | **SRS-10**: The system shall generate CycloneDX 1.5 SBOM containing package metadata, hashes, and licenses. | `HAZ-007` | `RC-007` | `VER-010` | `tests/unit/test_cicd_and_deployment.py::test_sbom_generation_schema` | **PASS** |
| **UN-08** (Safe Deployment Sandbox) | **SRS-11**: The system shall validate Kubernetes manifests and Helm templates for non-root execution and health probes. | `HAZ-008` | `RC-008` | `VER-011` | `tests/unit/test_cicd_and_deployment.py::test_repository_deployment_validations`<br>`tests/unit/test_cicd_and_deployment.py::test_k8s_deployment_validator_flags_missing_run_as_non_root` | **PASS** |

---

## 5. Verification Gate Enforcement & Maintenance

All 11 verification tests listed above are executed automatically during every pull request and build via [`scripts/run_ci_checks.py`](../../scripts/run_ci_checks.py) and GitHub Actions [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml). 

A change to any software requirement, risk control, or model architecture requires:
1. Re-running the complete SaMD verification suite (`pytest tests/unit/`).
2. Ensuring zero test regressions.
3. Updating this traceability matrix with the revised version tag and verification commit hash.
