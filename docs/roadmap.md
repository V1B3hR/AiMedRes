# 🧭 AiMedRes: Consolidated Production-Ready AI/ML Roadmap (2026–2027)

> **Document Status**: Living Master Roadmap (Single Source of Truth)  
> **Supersedes**: All prior fragmentary roadmaps (`docs/roadmapfuture.md`, `docs/roadmapsecurity.md`, and unverified legacy milestone claims).  
> **Orientation**: Scientific Rigor, Honest Capability Classification, Leakage-Proof Validation, and Responsible Clinical AI Translation.

---

## 1. Executive Summary & Strategic Reset

AiMedRes has developed extensive scaffolding across conventional disease-model training pipelines, an adaptive "DuetMind" neural agent framework, clinical decision support (CDS) interfaces, a Flask API, and React/React Native frontends. While this scaffolding represents substantial software engineering and interface development, an objective scientific audit reveals that **the repository does not yet substantiate claims of validated, production-ready, or clinically usable AI**.

To transition from an exploratory multi-component prototype to an institutionally credible medical research platform, this **Consolidated Roadmap** executes a fundamental strategic reset:

1. **Containment & Truthfulness First**: Immediately reconcile documentation, licensing, and capability claims with actual verified code.
2. **Singular Research Wedge**: Halt parallel attempts to simultaneously mature seven disease models, EHR write-back integrations, multi-agent clinical diagnostic loops, and mobile apps. Instead, focus all rigorous AI/ML efforts on a single, high-value clinical target: **Retrospective Parkinson’s Disease progression and severity analysis using the Parkinson’s Progression Markers Initiative (PPMI)**.
3. **Leakage-Proof Scientific Rigor**: Redesign the Parkinson’s pipeline from the ground up to eliminate circular prediction tasks, participant-level leakage across longitudinal visits, and pre-cross-validation scaling/imputation.
4. **Software & Security Hardening**: Decouple research analysis code from production inference services, implement real identity and secrets management, enforce strict CI gates, and prepare for external institutional review.

---

## 2. Current Baseline & Honest Capability Audit

The table below establishes an unvarnished audit of the repository, categorizing every major area into three reality tiers:
- **Tier 1 — Implemented Prototype Code**: Genuinely present executable code, tests, and functional scripts.
- **Tier 2 — Partially Implemented or Heuristic Simulation**: Modules that function via static rules, heuristic approximations, or in-memory mock logic rather than verified clinical models.
- **Tier 3 — Documentation or Planning Only**: Documents, templates, or roadmaps describing future milestones without corresponding operational reality or third-party validation.

### 2.1 Capability Audit Matrix

| Area | Tier | What is Genuinely Present | What is Missing, Inconsistent, or Flawed |
|---|:---:|---|---|
| **Training Infrastructure** | Tier 1 | 7 default jobs in `src/aimedres/cli/train.py`; orchestrator records Git SHA, GPU details, logs, JSON summaries. Classical ML and PyTorch. | Reproducibility incomplete: outputs ignored/deleted, DVC has no configured remote, package definitions disagree, no end-to-end verified PPMI artifact. |
| **Parkinson's / PPMI Work** | Tier 2 | `train_parkinsons.py` loads `ParkinsonDatasets/` CSVs, extracts SBR features, trains classical & neural models, exports reports. | **Critical scientific flaws**: Circular prediction (label created from the same SBR features used as inputs); preprocessing fitted before CV; longitudinal visits not grouped by `PATNO` (data leakage). Restricted PPMI data stored directly in Git. |
| **Adaptive NN & Agents** | Tier 2 | `AdaptiveNeuralNetwork`, `DuetMindAgent`, specialist agent roles, state handling, memory store, and safety heuristics in `src/aimedres/core/` and `src/aimedres/agents/`. | Experimental simulation code only. `specialized_medical_agents.py` generates heuristic diagnoses, simulated peer review, and synthetic outcome accuracy. Not validated clinical reasoning. |
| **Clinical Decision Support (CDS)** | Tier 2 | Risk scoring logic, intervention recommendation structures, FHIR/HL7 conversion helpers, human-in-the-loop audit stubs in `src/aimedres/clinical/`. | Risk engine is hand-authored static heuristic rules. EHR "sync" methods are explicitly simulated mocks. Not connected to validated disease models or real partner EHRs. |
| **API & Applications** | Tier 1 / 2 | Flask API in `src/aimedres/api/server.py`, rate limiting, security headers, DICOM routes, React CDS dashboard, Expo mobile companion UI. | Main prediction endpoint instantiates a fresh generic network rather than serving a registered, validated model. In-memory auth/token store with fallback development secrets. UI contract broader than backend reality. |
| **Privacy & Security** | Tier 1 / 3 | Regex-based PHI scrubbing, DICOM tag clearing, encryption helpers (AES-256 / Kyber-768 hybrid prototype), security documentation. | Pattern scrubbing is not a formal HIPAA de-identification certification or data-use authorization system. DICOM stripper clears limited fields. No third-party penetration testing or external SOC audit completed. |
| **Deployment & MLOps** | Tier 1 / 3 | Dockerfiles, Docker Compose, Kubernetes manifests (`k8s/`), Helm chart (`helm/aimedres/`), Prometheus/Grafana configs (`monitoring/`). | Deployment assets and declarative configurations only, not proof of deployed live operations. Image publishing, signed provenance, cloud overlays, and automated tracing remain unfinished. |
| **Documentation & Compliance** | Tier 3 | Extensive documentation: training guides, security policies, model cards, FDA Q-Sub generator (`src/aimedres/compliance/`). | Claims overstate maturity. Generates draft FDA/QMS skeletons, but no formal FDA submission has been made, no clinical pilot has run with live patients, and no certified QMS exists. |
| **Packaging & Governance** | Tier 1 / 2 | `pyproject.toml`, `setup.py`, requirements files, MIT `LICENSE`. | **Metadata conflicts**: README and legal notes previously discussed GPL-3.0 while `LICENSE` and `pyproject.toml` specified MIT; placeholder author emails (`your-email@example.com`) in `setup.py`; dependency specifications drifted across requirements files. |
| **Testing & CI** | Tier 1 / 2 | Meaningful unit and integration test suites in `tests/`, Vitest frontend tests, performance benchmarks (k6, Lighthouse CI). | CI activity historically focused on dependency updates. Incomplete integration test passes due to missing optional medical packages (`pydicom`). No automated participant leakage regression tests in CI. |

---

## 3. The 5-Phase Production-Ready AI/ML Roadmap

```mermaid
flowchart TD
    subgraph P0["Phase 0: Immediate Containment (Weeks 0-2)"]
        P0A[PPMI Data Governance & Restricted Data Removal] --> P0B[Freeze Unsupported Claims in Docs & README]
        P0B --> P0C[Consolidate Single Research Focus: Parkinson's]
        P0C --> P0D[Reconcile Packaging, Licenses MIT & Dependencies]
    end

    subgraph P1["Phase 1: Research-Grade PPMI Pipeline (Months 1-2)"]
        P1A[Pre-Registered Clinical Study Protocol] --> P1B[Defensible Clinical Target Definition]
        P1B --> P1C[Leakage-Proof PATNO GroupKFold Pipeline]
        P1C --> P1D[Data Manifest & Synthetic CI Benchmark Fixture]
    end

    subgraph P2["Phase 2: Reproducibility & Scientific Quality (Months 2-4)"]
        P2A[Typed Pydantic Config & Pinned Lockfiles] --> P2B[Leakage & Scientific Regression Test Suite]
        P2B --> P2C[MLflow Tracking & Model Registry]
        P2C --> P2D[Honest Model Card & Peer Review Signoff]
    end

    subgraph P3["Phase 3: Software Hardening & Decoupling (Months 4-8)"]
        P3A[Decouple research/ service/ ui/] --> P3B[Serve Registered Model via Flask API]
        P3B --> P3C[OAuth2/OIDC Auth & DB Credential Store]
        P3C --> P3D[Hardened Non-Clinical Research Sandbox K8s]
    end

    subgraph P4["Phase 4: Institutional Pilot Readiness (Months 8-15)"]
        P4A[Institutional IRB & Intended-Use Specification] --> P4B[Temporal & External Multi-Site Validation]
        P4B --> P4C[Clinician-in-the-Loop Usability Testing]
        P4C --> P4D[Quality Management & SaMD Traceability]
    end

    P0 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
```

---

### Phase 0: Immediate Containment, Data Governance & Truthfulness (Weeks 0–2)

**Goal**: Establish an honest baseline, eliminate legal and data-governance vulnerabilities, reconcile package metadata, and freeze speculative distractions.

- [x] **0.1 PPMI & Data-Use Governance Enforcement**
  - **Action**: Audit `ParkinsonDatasets/`. Under PPMI Data Use Agreements (DUA), raw patient records cannot be redistributed in public open-source repositories.
  - **Deliverable**: Isolate raw PPMI files from public distribution, add `ParkinsonDatasets/DATA_GOVERNANCE.md` detailing credential requirements, and introduce a verified, non-PHI synthetic fixture (`synthetic_pet_sbr_fixture.csv`) for open-source CI/CD execution.
- [x] **0.2 Claim Containment & Repository Truthfulness**
  - **Action**: Update `README.md`, `docs/KNOWN_LIMITATIONS.md`, and model cards to remove premature claims of "FDA submission readiness", "certified HIPAA compliance", and "multi-hospital network operational".
  - **Deliverable**: Explicit classification markers across all docs designating components as *Prototype*, *Simulated*, *Planned*, or *Validated*.
- [x] **0.3 Single Research Focus & Scope Freeze**
  - **Action**: Concentrate core AI/ML development exclusively on retrospective Parkinson’s disease progression and severity modeling using PPMI data.
  - **Deliverable**: Freeze feature expansion on the other six disease models, the Expo mobile app, and simulated multi-agent consensus until the primary research pipeline achieves scientific peer-review readiness.
- [x] **0.4 Packaging, Metadata & License Reconciliation**
  - **Action**: Resolve license divergence. The repository license is **MIT** (per `LICENSE` and `pyproject.toml`). Remove all stale references to GPL-3.0 in `README.md`, `setup.py`, templates, and documentation.
  - **Deliverable**: Clean `setup.py` without placeholder values (`your-email@example.com`), aligned with `pyproject.toml` and standard requirements files.

---

### Phase 1: Research-Grade PPMI Pipeline (Months 1–2)

**Goal**: Transform the Parkinson’s ML script into a scientifically defensible, statistically sound, leakage-free predictive pipeline.

- [ ] **1.1 Pre-Registered Clinical Study Protocol**
  - **Objective**: Author a formal study protocol prior to model optimization.
  - **Specification**:
    - **Target Outcome**: Transition away from the circular median-split striatal binding ratio (SBR). Adopt a clinically defensible endpoint: e.g., longitudinal motor progression (MDS-UPDRS Part III annual rate of change $\Delta \ge 5$ points) or baseline dopaminergic deficit classification verified against healthy controls.
    - **Index Time & Horizon**: Baseline visit (BL) features predicting 12-month (V04) and 24-month (V06) progression.
    - **Inclusion/Exclusion Rules**: Defined PPMI cohort filter (e.g., Early PD with DaTscan visual confirmation, baseline age $\ge 30$, completed motor assessments).
    - **Statistical Analysis Plan (SAP)**: Pre-specified primary metric (Balanced Accuracy, AUC-ROC, Brier Score), calibration assessment, and subgroup stratification (age, sex, genetic status).
- [ ] **1.2 Leakage-Proof Participant-Level Modeling**
  - **Objective**: Redesign model validation to prevent all forms of data leakage.
  - **Specification**:
    - **Participant Splitting**: Enforce strict participant-level splitting using `GroupKFold` or `StratifiedGroupKFold` grouped on `PATNO`. Ensure no individual patient's longitudinal visits appear in both training and validation/test folds.
    - **Pipeline Encapsulation**: Encapsulate all imputation (`SimpleImputer`) and feature scaling (`StandardScaler`, `RobustScaler`) inside an `sklearn.pipeline.Pipeline`. Preprocessing must be fitted *strictly on training folds* and only applied to validation/test folds.
    - **Holdout Test Set**: Reserve an untouched 20% holdout cohort (split by `PATNO`) evaluated exactly once after cross-validation tuning.
    - **Baseline Comparators**: Compare model performance against simple, interpretable clinical baselines (e.g., Logistic Regression with age + disease duration alone).
    - **Statistical Uncertainty**: Report 95% bootstrap confidence intervals for all primary metrics.
- [ ] **1.3 Data Manifest & Provenance Tracking**
  - **Objective**: Create a complete, non-PHI experiment manifest.
  - **Specification**:
    - Record PPMI data release versions, exact file names, SHA-256 checksums, transformation logs, and cohort inclusion counts.
    - Store all raw data in secure, institution-approved storage; commit only configuration files, metadata manifests, and synthetic fixtures to Git.
- [ ] **1.4 Synthetic PPMI Benchmark Fixture**
  - **Objective**: Maintain a 100% synthetic dataset that mimics the schema, covariance structure, and longitudinal visit hierarchy of PPMI.
  - **Specification**: Permits automated regression testing in GitHub Actions without violating data privacy or third-party DUAs.

---

### Phase 2: Reproducibility & Scientific Quality (Months 2–4)

**Goal**: Deliver clean packaging, deterministic rerunability, automated scientific regression testing, and transparent model reporting.

- [ ] **2.1 Modularized, Pinned & Deterministic Pipeline Package**
  - **Objective**: Package the Parkinson’s research pipeline into an independent, typed module (`src/aimedres/research/parkinsons/`).
  - **Specification**:
    - Strictly typed Pydantic configuration (`ParkinsonsConfig`) managing hyperparameters, paths, and random seeds.
    - Deterministic execution guarantees (`PYTHONHASHSEED`, NumPy seed, PyTorch deterministic mode).
    - Pinned dependency lockfile (`requirements-lock.txt` or Poetry/uv lock) and container image digest.
    - Automated export of machine-readable run reports (JSON + markdown metrics).
- [ ] **2.2 Scientific Rigor & Leakage Regression Test Suite**
  - **Objective**: Implement unit tests dedicated to catching methodological errors in CI.
  - **Specification**:
    - `test_zero_patno_leakage`: Asserts that `set(train_patno) & set(val_patno) == empty`.
    - `test_feature_label_independence`: Asserts that target labels are not mathematical derivations of the input feature matrix.
    - `test_pipeline_idempotency`: Asserts that two runs with identical seed produce bit-identical evaluation metrics.
- [ ] **2.3 Live Experiment Tracking & Model Registry**
  - **Objective**: Integrate MLflow (or Weights & Biases) for verifiable experiment provenance.
  - **Specification**:
    - Log all training runs, parameters, metrics curves, feature importance plots, and serialized model artifacts.
    - Register candidate models with explicit versioning and lineage to the exact data manifest and Git commit.
- [ ] **2.4 Evidence-Based Model Card & External Methodological Signoff**
  - **Objective**: Publish a comprehensive, honest model card adhering to FDA Good Machine Learning Practice (GMLP) principles.
  - **Specification**:
    - Intended use, exclusions, limitations, calibration curves, subgroup disparities (fairness audit across sex and age deciles), and failure mode analyses.
    - Complete independent review checklist signed off by a consulting biostatistician and a movement-disorder neurologist prior to public dissemination.

---

### Phase 3: Software Hardening & Architecture Decoupling (Months 4–8)

**Goal**: Decouple research analytics from production services, harden security boundaries, and eliminate architectural anti-patterns.

- [ ] **3.1 Domain-Driven Architectural Decoupling**
  - **Objective**: Restructure the repository into clear architectural boundaries:
    ```text
    src/aimedres/
      research/       # Reproducible scientific pipelines (PPMI, validation, manifests)
      service/        # Production inference API, model serving, security middleware
      ui/             # Clinical decision-support dashboard and client interfaces
      simulation/     # Quarantined experimental agent logic and cognitive simulations
    ```
  - **Specification**: Quarantine heuristic agent rules (`specialized_medical_agents.py`, DuetMind cognitive simulations) away from the production inference path. Clearly demarcate them as simulation research prototypes.
- [ ] **3.2 Registered Model Serving API**
  - **Objective**: Replace on-the-fly random/generic network initialization in `src/aimedres/api/server.py` with validated model artifact serving.
  - **Specification**:
    - API endpoints load versioned, registered models from the local model store or MLflow registry.
    - Enforce input schema validation with Pydantic; reject out-of-distribution or malformed feature payloads.
    - Include model version, metadata hash, and uncertainty estimates in every API response.
- [ ] **3.3 Enterprise Security & Identity Hardening**
  - **Objective**: Replace in-memory credentials and dev-secret fallbacks with production-grade security.
  - **Specification**:
    - Remove hardcoded default secrets and fail closed if required environment secrets are absent.
    - Integrate OAuth2 / OpenID Connect (OIDC) authentication with role-based access control (RBAC).
    - Database-backed, audited session and API token storage (PostgreSQL).
- [ ] **3.4 Production CI/CD Gateways**
  - **Objective**: Establish strict gating workflows before merge.
  - **Specification**:
    - Automated linting (Ruff/Black), strict type checking (Mypy), security vulnerability scanning (Bandit, Trivy), and container image signing (Cosign).
    - Software Bill of Materials (SBOM) generation (SPDX/CycloneDX).
- [ ] **3.5 Non-Clinical Research Sandbox Deployment**
  - **Objective**: Deploy the decoupled platform to an isolated Kubernetes research sandbox.
  - **Specification**:
    - Validate Helm chart deployments with least-privilege RBAC and network policies.
    - Connect Prometheus, Grafana, and centralized log aggregation (Loki/Elasticsearch) with real metrics and alerts.

---

### Phase 4: Institutional Pilot Readiness & Clinical Evaluation (Months 8–15)

**Goal**: Prepare for ethical, institutional evaluation under formal oversight and rigorous clinical validation.

- [ ] **4.1 Institutional IRB & Intended-Use Definition**
  - **Objective**: Partner with an academic medical center or clinical research institution.
  - **Specification**:
    - Formulate formal Institutional Review Board (IRB) / Ethics Committee protocol.
    - Author clear Intended Use and Indications for Use (IFU) statements: Research Use Only (RUO) or Non-Device Clinical Decision Support (CDS) under FDA 21st Century Cures Act criteria.
- [ ] **4.2 Multi-Site & Temporal External Validation**
  - **Objective**: Validate the model on temporally and institutionally distinct cohorts.
  - **Specification**:
    - Evaluate on independent cohorts (e.g., BioFIND, DATATOP, or distinct geographic PPMI clinical sites).
    - Measure discrimination (AUC-ROC, C-index), calibration (Brier score, calibration slope/intercept), and clinical net benefit via Decision Curve Analysis (DCA).
- [ ] **4.3 Clinician-in-the-Loop Usability & Human Factors**
  - **Objective**: Conduct formal usability studies with neurologists and clinical fellows.
  - **Specification**:
    - Assess explainability artifacts (SHAP, feature attributions) to ensure they assist rather than mislead clinical reasoning.
    - Measure decision turnaround time, cognitive workload, and user error rates.
- [ ] **4.4 Quality Management & SaMD Traceability**
  - **Objective**: Establish design controls proportional to prospective medical software.
  - **Specification**:
    - Requirements traceability matrix linking clinical needs to software verification tests.
    - Risk management documentation conforming to ISO 14971 principles.
    - Software lifecycle management processes conforming to IEC 62304 standards.

---

## 4. First Major Milestone: The PPMI Parkinson's Research Reproducibility Pack

The single most credible deliverable to present to an institutional partner, IRB, or scientific review committee is the **PPMI Parkinson's Research Reproducibility Pack**. Rather than adding speculative features, all immediate engineering converges on completing this pack:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│               PPMI PARKINSON'S RESEARCH REPRODUCIBILITY PACK                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Zero Restricted Records in Git + Synthetic CI Fixture                   │
│    (ParkinsonDatasets/DATA_GOVERNANCE.md & synthetic_pet_sbr_fixture.csv)   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Pre-Registered Study Protocol & Data-Access Statement                    │
│    (Defensible non-circular endpoint, inclusion/exclusion, statistical plan)│
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Leakage-Proof Participant-Level Pipeline                                 │
│    (GroupKFold on PATNO, fold-internal scaling, holdout test isolation)     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. Single Deterministic Execution Command                                   │
│    (aimedres train parkinsons --config configs/parkinsons_ppmi_study.yml)   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. Machine-Readable Experiment Manifest & Evaluation Report                 │
│    (Data checksums, cohort counts, bootstrap CIs, calibration metrics)      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6. Automated CI Pipeline with Fixture Verification                          │
│    (Zero-leakage regression tests passing on every pull request)            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 7. Honest, Evidence-Based Model Card                                        │
│    (Intended use, GMLP compliance, subgroup fairness, known failure modes)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 8. Independent Review Checklist                                             │
│    (Signed off by biostatistician and movement-disorder neurologist)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Development & Execution Instructions

### 5.1 Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/V1B3hR/AiMedRes.git
cd AiMedRes

# 2. Configure Python environment (Python 3.10 - 3.12)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install core dependencies and development tools
pip install -r requirements.txt
pip install -r requirements-ml.txt
pip install -r requirements-dev.txt

# 4. Run test suite (verifying leakage-safe pipelines and unit coverage)
pytest tests/
```

### 5.2 Executing the Parkinson's Research Pipeline

```bash
# Preview discovered training jobs without execution
python run_all_training.py --only parkinsons --dry-run

# Run leakage-safe training on the synthetic PPMI benchmark fixture
python src/aimedres/training/train_parkinsons.py \
  --data-path ParkinsonDatasets/synthetic_pet_sbr_fixture.csv \
  --output-dir results/parkinsons_synthetic_run \
  --folds 5

# When operating in an institution-approved restricted environment with registered PPMI access:
# python src/aimedres/training/train_parkinsons.py \
#   --data-path /secure/restricted/PPMI_Data/ \
#   --output-dir results/parkinsons_ppmi_official \
#   --folds 5
```

### 5.3 User Interfaces (Research Preview)

```bash
# Clinical Decision Support Web Dashboard
cd frontend
npm install
npm run dev

# Mobile Clinical Companion (Development Preview)
cd ../mobile
npm install
npm start
```

---

## 6. Governance & Review Cadence

- **Bi-Weekly Technical Review**: Verification of zero-leakage test runs, data governance audits, and model metric deltas.
- **Monthly Protocol Review**: Alignment with external clinical advisory members (neurology/biostatistics) regarding endpoints, covariates, and subgroup balance.
- **Milestone Gate Review**: Formal transition gate before moving between roadmap phases. Phase advancement requires 100% completion of acceptance criteria and automated test passes.
