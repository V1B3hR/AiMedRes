# 🧠 AiMedRes

> **Advanced AI Medical Research Assistant**  
> **Adaptive neural architectures + multi-agent research simulations**
> **Safety‑aware, explainable AI for neurological and mental health research**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/V1B3hR/AiMedRes)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)](https://github.com/V1B3hR/AiMedRes)

---

> **⚠️ RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS ⚠️**  
> This software has **NOT** been approved by the FDA or any other regulatory authority.  
> It is intended exclusively for research and development purposes.  
> Clinical decisions must only be made by licensed healthcare professionals.  
> See the full [Disclaimer](#️-disclaimer) below.

---

## 📊 Implementation Status & Capability Classification

In accordance with our [Consolidated Roadmap](docs/roadmap.md) and [Known Limitations](docs/KNOWN_LIMITATIONS.md), AiMedRes classifies all components into three transparent capability tiers:

### 🛠️ Tier 1: Implemented Prototype Code
Working code, training pipelines, and interfaces developed as research scaffolding:
| Capability | Location | Reality & Notes |
|---|---|---|
| Parkinson's & Open Research Package | `src/aimedres/research/` | Modularized, typed Pydantic config, `LocalExperimentTracker` (SQLite/JSON), leakage-safe cross-validation, open dataset connectors (OASIS-2, PRO-ACT, PTB-XL, UCI Telemonitoring), and cryptographic provenance manifest (`provenance_manifest.json`) |
| Dataset evaluation tooling | `scripts/`, `results/` | Download and participant-isolated evaluation utilities; any recorded metrics require independent replication and are not clinical validation |
| Maintained research API | `src/aimedres/service/` | Application factory (`create_app`) exposing research-model metadata only. Online inference is disabled; FHIR, case-review, explainability, DICOM, and P3 route modules are retained as unexposed research scaffolding until separately validated. |
| Usability-supporting code | `src/aimedres/clinical/usability.py` | Feature-attribution, guideline-concordance, turnaround-time, and survey helpers; no completed clinician usability study is claimed |
| Clinical governance templates | `docs/clinical/` | Intended-use and protocol documents for institutional adaptation; they are not an IRB approval or a regulatory determination |
| SaMD traceability templates | `docs/compliance/` | Traceability and risk-management scaffolding for future quality work; not a certified QMS or device classification |
| Quarantined Simulation Agents | `src/aimedres/simulation/` | Heuristic multi-agent prototypes are unavailable through the maintained API and must be explicitly labelled when run locally for research. |
| Database-Backed Auth & Security | `src/aimedres/security/auth.py` | `AuthDatabaseStore` (SQLite), PBKDF2 hashing with salt, instant API key & JWT session revocation, audit logging, and fail-closed production secret validation |
| Clinical CDS & Imaging Web UI | `frontend/` | React + Vite + Three.js viewer; prototype CDS dashboards |
| Mobile Companion UI | `mobile/` | Expo/React Native prototype interface |
| Pattern-Based PHI Scrubbing | `src/aimedres/security/` | Regex-based detection; technical safeguard (not a formal HIPAA certification) |
| DICOM Header Anonymization | `src/aimedres/integration/` | Strips specified tags; requires institutional review before PHI exposure |
| Container & deployment assets | `k8s/`, `helm/`, `monitoring/` | Configuration assets, Prometheus material, a CycloneDX 1.5 SBOM generator, and a local CI runner; not an operated production environment |
| Maintained Unit-Test Gate | `tests/unit/` | Runs the supported research, security, deployment, and selected training checks. See [`debug/README.md`](debug/README.md) for the current command, scope, and clean-environment procedure; it is not evidence that every historical test or prototype module remains supported. |

### 🔬 Tier 2: Partially Implemented or Heuristic Simulations
Components operating via heuristics, mock data, or simulation loops (not evidence-based clinical models):
| Component | Location | Reality & Notes |
|---|---|---|
| Multi-Agent Consultation | `src/aimedres/agents/` | Heuristic diagnostic scoring & simulated peer review |
| Clinical Risk Engine | `src/aimedres/clinical/` | Hand-authored static scoring rules; not linked to verified clinical models |
| EHR Ingress / Sync | `src/aimedres/integration/` | Synthetic FHIR/HL7 converters; sync methods are simulated mocks |
| Multi-Hospital Network | `src/aimedres/clinical/` | In-memory simulation framework, not an active multi-site clinical network |
| Post-Quantum Hybrid Crypto | `src/aimedres/security/` | Experimental Kyber768 prototype; research evaluation only |

### 📄 Tier 3: Documentation & Planning Artifacts Only
Declarative documentation, templates, and roadmap specifications (not completed deliverables):
| Item | Location | Reality & Notes |
|---|---|---|
| FDA Pre-Submission (Q-Sub) | `src/aimedres/compliance/` | Generates draft question packages; **no FDA submission has been made** |
| Quality Management System (QMS) | `docs/` | Template SOP skeletons; **not a certified QMS** |
| Clinical Pilot Programs | `src/aimedres/clinical/` | Study design tracking skeletons; **no live patient trials conducted** |
| Third-Party HIPAA/GDPR Audits | `docs/` | Internal technical controls; **formal third-party audit pending** |
| External Penetration Testing | `docs/` | Documented security posture; **external testing pending** |

### 🧭 Research Roadmap Status

See [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) and **[`docs/roadmap.md`](docs/roadmap.md)** for implemented research artefacts and outstanding validation work. Completion of an in-repository roadmap item is not clinical or production readiness:
- **P0 (Presentation readiness and audit)**: In progress. A synthetic-fixture [Evidence Pack](evidence/ppmi_synthetic_fixture/2026-10-25/EVIDENCE_PACK.md) now records one local control run, but the claim review, independent rerun, external security assurance and methodological review remain incomplete.
- **P1 (Independent replication and partnership)**: Future gate. The repository contains participant-grouped evaluation controls and synthetic-fixture tooling, but no independently reproduced controlled-data result.
- **P2 (Research funding application)**: Future gate. A concept note may request funding for validation only after P0/P1 evidence and partner roles are in place.
- **P3 (Institutional validation and regulatory decision)**: Future gate. Protocol, intended-use, usability, and traceability artefacts are planning inputs, not institutional approval, clinical validation, or a certified QMS.

**Primary Milestone**: The **PPMI Parkinson's Research Reproducibility Pack** (see [docs/roadmap.md](docs/roadmap.md)).  
**Deployment Plan**: [healthcaredeploymentplan.md](healthcaredeploymentplan.md).

## 🎯 Mission

AiMedRes supports reproducible research experiments in neurological health. The
active focus is retrospective Parkinson's/PPMI analysis; other components are
exploratory scaffolding. It combines:
- Adaptive neural network topologies
- Multi-agent medical reasoning
- Biological state + memory simulation
- Explainability & auditability prototypes

---

## 🏥 Current Research Focus

- **Parkinson's disease / PPMI** — maintained reproducibility and
  leakage-control work.
- **Open research datasets** — connector and evaluation utilities, subject to
  their own access terms and independent validation.
- **Other disease models, simulations, and UI concepts** — retained as legacy
  or experimental code, not active clinical-model work.

---

## 🚀 Exploratory feature inventory

> The following inventory identifies code and documentation in the repository.
> It does not assert clinical validation, regulatory compliance, production
> operation, or live health-system integration.

### 🧩 Intelligence & Architecture

  **Research prototypes for security, auditing, and explainability**
  
- Adaptive neural evolution engine (dynamic layer & pathway adjustment)
- Multi-agent consultation & consensus system
- Biological state simulators (energy, mood, circadian influences)
- Dual-store + prioritized replay memory consolidation
- Multi-Hospital Network and Specialty Modules
- **3D Brain Visualization (P3-1)** - Interactive anatomical mapping with AI explainability overlays
- Multi-Modal AI Integration (EHR, imaging, notes)

### 📊 Clinical AI

  **Clinical-integration and security scaffolding:**

- FHIR/HL7 conversion helpers and integration scaffolding; no live EHR connector is validated
- Pattern-based PHI detection and audit helpers; not HIPAA/GDPR certification
- Experimental cryptography and access-control components
- Documentation and partial prototypes for monitoring, backup, and recovery
- Risk scoring & uncertainty estimates
- Explainable prediction frames (feature attributions + causal hints)
- Configurable safety thresholds & override gating
- Quantitative performance dashboards (CLI + API + dashboard module)
- Predictive Healthcare Analytics (NEW)
- Population-level Insights
- **Advanced DICOM/3D Viewer (P3-1)** - Smooth streaming viewer with treatment simulation

### 🏥 Integration Layer

- EHR and event-streaming integration scaffolding; no live connector is validated
- Audit and provenance-tagging helpers; operational guarantees require deployment review
- Compliance scaffolding (HIPAA/FDA alignment docs)
- Regulatory and deployment planning material

### 🔒 Security & Cryptography (P3-2)

- Experimental cryptographic and key-management helpers
- Security architecture and audit-log prototypes
- Privacy and compliance documentation for institutional adaptation

### 🚢 MLOps & Deployment (P3-3)

- CI, SBOM, and deployment configuration scaffolding
- Experiment tracking, model-comparison, and monitoring prototypes
- Planning material for rollback, canary, and operations workflows

> ⚠️ **Note:** K8s/Helm deployment assets, Prometheus/Grafana/Alertmanager
> configurations, and baseline k6/Lighthouse artefacts are included in-repo.
> They require environment-specific validation and formal external review. See
> [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).

  **Evidence boundary:**
- Model metrics must be tied to a versioned dataset, protocol, and independent
  validation report before being used as a performance claim.
- CLI, API, and UI code are research and prototype surfaces, not validated
  real-world clinical workflows.

**Deployment planning stack (see KNOWN_LIMITATIONS for remaining gaps):**
- Deployment assets: Docker (root `Dockerfile`), K8s manifests (`k8s/`), Helm chart (`helm/aimedres/`), blue/green/canary strategies documented
- Monitoring assets: Prometheus/Grafana/Alertmanager (`monitoring/`), ELK and SIEM integration guidance in deployment docs

---

## 🔁 Legacy training labels (not production status)

| Model / Variant          | Dataset(s)             | Target Task                    | Status          | Notes                      |
|------------------------- |-----------------------|-------------------------------|-----------------|----------------------------|
| AD_EARLY_V2              | ADNI + INTERNAL_SET_V1| MCI→AD conversion (12–24m)     | Legacy experiment | Requires independent validation |
| AD_SCREEN_V1             | ADNI subset           | Screening classifier           | Legacy experiment | Requires independent validation |
| MULTI_AGENT_CONSENSUS_V3 | Simulated + Expert    | Agreement score                | Simulation | Not a clinical model |
| MEMORY_CONSOLIDATION_V4  | Synthetic episodic    | Retention @24h                 | Simulation | Not a clinical model |
| LATENCY_OPT_BATCH_OPT    | Prototype benchmark   | Latency experiment             | Exploratory | Hardware- and workload-specific |

---

## 📈 Performance evidence

The repository does not publish a single current performance snapshot for
clinical use. Reproduce the maintained synthetic PPMI check locally and review
the generated artefacts, model card, dataset provenance, and validation
protocol before making any research-performance statement.

---

## ⚕️ For Healthcare Deployment

### Deploying AiMedRes in a Hospital or Clinical Setting

- **Technical, compliance, and operational planning documentation is available in [healthcaredeploymentplan.md](healthcaredeploymentplan.md)**
- **Reference the plan for scripts, templates, rollback, support, backup/DR, escalation, audit sample code, checklists, and SOPs**
- **Any actual clinical deployment requires institutional review, local regulatory validation, and sign-off by qualified personnel**
- **Incident management, governance, and continuous improvement must be established by your institution**

> ⚠️ The deployment plan is a **planning and reference document**, not a guarantee of production readiness. Infrastructure assets for K8s/Helm and observability are included, while formal security audits, external certifications, and institution-specific rollout validation remain future work — see [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).

### What You Get

- A research-grade ML platform with security scaffolding and compliance documentation stubs
- Built-in scripts and guides for local adaptation
- Compliance documentation generators for HIPAA, GDPR, NIST, and FDA pre-submission planning

**Start here:**  
[healthcaredeploymentplan.md](healthcaredeploymentplan.md) — deployment planning guide.

---

## 🛠️ Install / Quickstart

```bash
git clone https://github.com/V1B3hR/AiMedRes.git
cd AiMedRes
pip install -r requirements.txt
# For dev/docs: pip install -e ".[dev,docs]"
```

See “Deployment” section below for healthcare deployment workflow.

---

## 🤖 Supported research usage

The maintained command path is the leakage-safe Parkinson's research runner:

```bash
python scripts/reproduce_ppmi_benchmark.py --config configs/parkinsons_ppmi_study.yml --output-dir .test-tmp/ppmi_benchmark
```

To create a safe-to-commit public Evidence Pack from the committed synthetic fixture, add `--evidence-dir evidence/ppmi_synthetic_fixture/<new-empty-directory>`. The runner rejects this option for controlled or patient-level data.

See [debug/README.md](debug/README.md) for its scope, synthetic fixture, data
governance requirements, and clean-environment validation procedure. The
legacy multi-model and agent workflows below are exploratory only.

---

## 🚂 Legacy multi-model training workflows

AiMedRes retains seven exploratory training jobs behind a unified orchestrator.
They are not comprehensive medical models and do not represent validated
clinical capabilities:

1. **ALS** (Amyotrophic Lateral Sclerosis)
2. **Alzheimer's Disease**
3. **Parkinson's Disease**
4. **Brain MRI Classification**
5. **Cardiovascular Disease**
6. **Diabetes Prediction**
7. **Specialized Medical Agents**

### Quick Start - Train All Models

```bash
# Simple wrapper script (easiest)
./train_all_models.sh

# With parallel execution (faster)
./train_all_models.sh --parallel --max-workers 4

# With custom training parameters
./train_all_models.sh --parallel --max-workers 6 --epochs 50 --folds 5

# Unified CLI (recommended)
aimedres train

# With custom parameters
aimedres train --epochs 20 --folds 5

# Parallel execution for faster training
aimedres train --parallel --max-workers 4

# Larger experimental run; this is not a production configuration
aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128
```

### Run Specific Models

```bash
# Only selected models
aimedres train --only als alzheimers parkinsons

# Exclude certain models
aimedres train --exclude brain_mri

# Dry run (preview commands without execution)
aimedres train --dry-run --epochs 10

# With batch size for compatible models
aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128
```

### Run Single Model

```bash
# Legacy exploratory Alzheimer's training script
python src/aimedres/training/train_alzheimers.py

# Custom parameters
python src/aimedres/training/train_als.py --epochs 50 --folds 3 --output-dir my_results
```

### List Available Training Jobs

```bash
aimedres train --list

# Or use the wrapper
./train_all_models.sh --dry-run
```

### Complete Documentation

For comprehensive documentation on running all models, see:
- **[RUN_ALL_MODELS_GUIDE.md](RUN_ALL_MODELS_GUIDE.md)** - Complete usage guide
- **[run_all_models_demo.sh](run_all_models_demo.sh)** - Interactive demonstration
- **[train_all_models.sh](train_all_models.sh)** - Simple wrapper script

### Output & Results

Training writes experiment artefacts to `results/`:
- Trained models (`.pkl`, `.pth`)
- Metrics (CSV, JSON)
- Logs, plots

### Run exploratory examples

```bash
python examples/basic/run_all_demo.py
python examples/clinical/alzheimer_demo.py
python examples/advanced/parallel_mode.py
python examples/enterprise/production_demo.py  # demonstration code only
```
See [examples/README.md](examples/README.md) for more.

### Advanced Options

```bash
aimedres train --config my_config.yaml
aimedres train --retries 2
aimedres train --allow-partial-success
```

### Documentation

- **Examples Guide**: [examples/README.md](examples/README.md)
- **Training Usage Guide**: [docs/TRAINING_USAGE.md](docs/TRAINING_USAGE.md)
- **Training Scripts Reference**: [src/aimedres/training/README.md](src/aimedres/training/README.md)
- **Documentation Index**: [docs/](docs/)

### Reproducibility

- Set `AIMEDRES_SEED=42`
- Use `--seed` in scripts
- Configs stored under `results/<job_id>/config.yaml`

---

## 🧪 Model Zoo (legacy and exploratory)

| Name                 | Task                  | Status           | Checkpoint | Notes                               |
|----------------------|-----------------------|------------------|------------|-------------------------------------|
| ad_early_v2          | 12–24m conversion     | Exploratory | (planned)  | Requires independent validation |
| ad_screen_v1         | Screening classifier  | Exploratory | (planned)  | Requires independent validation |
| consensus_v3         | Multi-agent aggregator| Experimental     | (planned)  | Weighted disagreement resolution    |
| memory_v4            | Consolidation kernel  | Experimental     | (planned)  | Synaptic tagging + decay curves     |

---

## 📁 Project Structure

```
src/aimedres/
  core/
  training/
  clinical/
  compliance/
  integration/
  dashboards/
  cli/
  agents/
  agent_memory/
  security/
  api/
  utils/
examples/
  basic/
  clinical/
  advanced/
  enterprise/
tests/
  unit/
    test_security/
    test_training/
  integration/
  performance/
  regression/
scripts/
mlops/
docs/
configs/
data/
```
- Examples are now organized by category
- Tests are split into unit, integration, performance
- Added clinical, compliance, integration, dashboards packages
- Unified CLI under src/aimedres/cli/

---

## 🛡️ Safety and compliance scaffolding

The following components are technical controls and templates. They do not
establish legal compliance, clinical safety, or formal certification.

### Human-in-the-loop components

- Approval, rationale, audit, and override mechanisms in prototype code
- Tamper-evident logging helpers that require deployment-specific review

### PHI-protection components

- Pattern-based PHI detection and de-identification helpers
- Repository checks for known data-handling risks
- No Safe Harbor certification or guarantee of complete de-identification

### Additional Security Features
- Configurable risk thresholds
- Bias & drift monitoring (dashboards in progress)
- Regulatory pathway planning material
- Vulnerability disclosure process (see [SECURITY.md](SECURITY.md))

---

## 📚 Documentation

| Topic                 | Link                                                      |
|-----------------------|-----------------------------------------------------------|
| Technical Architecture| docs/architecture.md                                      |
| Medical Applications  | docs/medical-applications.md                              |
| API Reference         | [docs/API_REFERENCE.md](docs/API_REFERENCE.md)            |
| Compliance Tracking   | docs/compliance.md                                        |
| Publications / Notes  | docs/publications.md                                      |
| Known Limitations     | [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md)    |
| Agent Memory System   | [agent_memory/README.md](agent_memory/README.md)          |
| Development Roadmap   | [docs/roadmap.md](docs/roadmap.md)                        |

(Missing docs: generate via `scripts/scaffold_docs.py`)

---

## 🛣️ Roadmap

The [consolidated roadmap](docs/roadmap.md) is the source of truth. Prior
roadmap snapshots and historical PR titles are not current capability claims.
The next research gates are independent replication, documented external
validation, and institutional review—not feature-count expansion.

---

## 🧾 Data Sources & Ethics

| Dataset                | Usage                             | Access                    |
|------------------------|-----------------------------------|---------------------------|
| PPMI                   | Parkinson's research workflow     | Controlled; authorized users only |
| ADNI                   | Alzheimer's progression           | Licensed / user-supplied  |
| INTERNAL_SIM_V1        | Synthetic multimodal cases        | Generated                 |
| CLINICAL_NOTES_PROTOTYPE| Context enrichment (planned)     | Pending de-ID pipeline    |

Ethics:
- Do not commit raw PHI or controlled data to the public repository
- Use synthetic fixtures for public tests and examples
- Treat fairness analysis and de-identification as protocol-specific work, not
  automatic properties of the code

---

## 🤝 Contributing

Focus areas:
- reproducible Parkinson's research and validation tooling;
- safety, security, and dependency maintenance;
- synthetic-data and provenance safeguards;
- documentation that distinguishes prototypes from verified evidence.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the current contribution process.

---

## 📢 Citing

If you use AiMedRes in your research, please cite:

```bibtex
@software{aimedres2025,
  title   = {AiMedRes: Adaptive Multi-Agent Clinical Reasoning with Biological Memory Enhancement},
  author  = {AiMedRes Development Team},
  year    = {2025},
  version = {1.0.0},
  url     = {https://github.com/V1B3hR/AiMedRes}
}
```

*Academic publication in preparation.*

---

## ⚖️ Disclaimer

**⚠️ IMPORTANT: RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS ⚠️**

This software is intended for **research and development purposes only**. It is **NOT** a medical device and has **NOT** been approved by the FDA or any other regulatory authority for clinical diagnosis or treatment. 

**Key Limitations:**
- This software does **NOT** provide medical diagnosis
- This software is **NOT** intended to replace professional medical judgment
- Clinical decisions must **ONLY** be made by licensed healthcare professionals
- Results generated by this software should be considered experimental and require validation
- No warranties are provided regarding accuracy, reliability, or fitness for any particular purpose

By using this software, you acknowledge and agree to these limitations and disclaimers.

---

## 📜 License

AiMedRes is licensed under the **MIT License**.

This means:
- ✅ You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
- ✅ Commercial and non-commercial research use are permitted
- ✅ Simple attribution requirement (include the copyright notice and permission notice in copies)

See the [LICENSE](LICENSE) file for full legal terms.

For research collaboration or data-sharing inquiries, please open an issue or contact the maintainers.

---

## 📞 Contact

- Lead: [V1B3hR](https://github.com/V1B3hR)
- Issues: https://github.com/V1B3hR/AiMedRes/issues
- Discussions: https://github.com/V1B3hR/AiMedRes/discussions
- Collaboration: Open to research & clinical partners

---

*Advancing responsible AI for neurological health.* 🧠

> **Note:** This summary is based on the 30 most recent PRs/issues. There are 162 total; for a complete changelog and more details, visit [all closed PRs/issues](https://github.com/V1B3hR/AiMedRes/issues?q=is%3Apr+is%3Aclosed).
