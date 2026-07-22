# 🧠 AiMedRes

> **Advanced AI Medical Research Assistant**  
> **Adaptive neural architectures + multi-agent clinical reasoning**  
> **Safety‑aware, explainable AI for neurological and mental health research**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/V1B3hR/AiMedRes)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)](https://github.com/V1B3hR/AiMedRes)

---

> **⚠️ RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS ⚠️**  
> This software has **NOT** been approved by the FDA or any other regulatory authority.  
> It is intended exclusively for research and development purposes.  
> Clinical decisions must only be made by licensed healthcare professionals.  
> See the full [Disclaimer](#️-disclaimer) below.

---

## 📊 Implementation Status

AiMedRes is an actively developed research platform. The table below separates what is **actually implemented and tested** in this codebase from items that are **documentation/planning artifacts** or **not yet implemented**.

### ✅ Implemented & Tested in This Codebase

| Capability | Location | Notes |
|---|---|---|
| Core adaptive neural architecture | `src/aimedres/core/` | Tested |
| Multi-agent consultation system | `src/aimedres/agents/` | Tested |
| HIPAA audit logging & PHI scrubbing | `src/aimedres/security/`, `src/aimedres/compliance/` | Tested |
| FHIR/HL7 EHR integration | `src/aimedres/integration/ehr.py` | Tested |
| Clinical Decision Support | `src/aimedres/clinical/decision_support.py` | Tested |
| 3D Brain Visualization (P15) | Frontend viewer | Prototype tested |
| Multi-modal AI integration (P16) | `src/aimedres/core/multimodal_integration.py` | Tested |
| Predictive healthcare analytics (P17) | `src/aimedres/clinical/` | Tested |
| Multi-hospital network framework (P12) | `src/aimedres/clinical/multi_hospital_network.py` | Tested |
| Quantum-safe cryptography (Kyber768/AES-256) | `src/aimedres/security/` | Hybrid implementation |
| Rate limiting & CORS hardening | `src/aimedres/api/server.py` | Configurable per endpoint |
| Training pipeline (7 disease models) | `src/aimedres/training/` | ALS, Alzheimer's, Parkinson's, CVD, Diabetes, Brain MRI, Agents |
| Container orchestration (K8s + Helm) | `k8s/`, `helm/aimedres/` | Deployment manifests and chart present |
| Observability stack | `monitoring/` | Prometheus, Grafana, Alertmanager configs present |
| AI security monitoring enhancement | `src/aimedres/security/ai_security_monitoring.py` | Threat detection, analytics, incident response tested |
| Drug discovery & clinical trial support | `src/aimedres/clinical/drug_discovery.py` | Screening, trial design, safety workflows tested |
| Frontend unit tests (Vitest) | `frontend/src/__tests__/` | API-focused unit coverage present |

### 📄 Documentation / Planning Artifacts Only

These modules **generate planning documents or scaffold skeletons** — they are **not** regulatory approvals, clinical certifications, or operational deployments.

| Item | Reality |
|---|---|
| FDA Q-Sub package generator (P9) | Generates pre-submission *draft documents*; **no FDA submission has been made** |
| QMS documentation skeleton (P9) | Provides SOP templates; **not a certified Quality Management System** |
| Clinical pilot programs framework (P8B) | Provides tracking/study-design structure; **no confirmed live patient data collected** |
| HIPAA compliance scaffolding | Implements technical safeguards; **formal third-party audit not yet completed** |
| International healthcare systems (P18) | Framework implemented; **no regulatory validation in non-US jurisdictions** |

### 🔜 Not Yet Implemented — Roadmap Items

See [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) for the full list. Key gaps:

| Item | Status |
|---|---|
| Load tests (k6) and Lighthouse CI baseline | Implemented |
| Formal third-party HIPAA/GDPR audit & penetration test | Future |
| FDA regulatory submission | Future (currently only planning documentation exists) |

**See:** [healthcaredeploymentplan.md](healthcaredeploymentplan.md) for deployment planning details and [docs/roadmap.md](docs/roadmap.md) for full roadmap.

## 🎯 Mission

AiMedRes accelerates AI-driven discovery and decision support for **neurodegenerative and mental health conditions**. It combines:
- Adaptive neural network topologies
- Multi-agent medical reasoning
- Biological state + memory simulation
- Clinically aligned explainability & auditability

---

## 🏥 Core Clinical Focus

- **Alzheimer's Disease** – Early detection, progression risk modeling  
- **Stroke / Cerebrovascular** – Risk stratification & recovery trajectory scoring  
- **Neurocognitive Disorders** – Expansion toward broader spectrum  
- **Mental Health State Modeling** – Early research modules  
- **Parkinson's & ALS** – Dataset integration and pilot modules added

---

## 🚀 Key Features (v1.0.0)

### 🧩 Intelligence & Architecture

  **Enterprise-grade security, auditing, and explainability for regulated deployment**
  
- Adaptive neural evolution engine (dynamic layer & pathway adjustment)
- Multi-agent consultation & consensus system
- Biological state simulators (energy, mood, circadian influences)
- Dual-store + prioritized replay memory consolidation
- Multi-Hospital Network and Specialty Modules
- **3D Brain Visualization (P3-1)** - Interactive anatomical mapping with AI explainability overlays
- Multi-Modal AI Integration (EHR, imaging, notes)

### 📊 Clinical AI

  **Clinical integrations and security:**

- FHIR/HL7 integration and real-time EHR ingestion (Epic, Cerner, Allscripts, and more)
- Full PHI scrubbing pipeline (HIPAA/GDPR), blockchain audit logging, immutable audit trails
- Quantum-safe cryptography (Kyber/AES hybrid)
- RBAC, SSO/LDAP/Active Directory
- Complete vulnerability management and continuous monitoring
- Fully automated, documented backup & disaster recovery
- Risk scoring & uncertainty estimates
- Explainable prediction frames (feature attributions + causal hints)
- Configurable safety thresholds & override gating
- Quantitative performance dashboards (CLI + API + dashboard module)
- Predictive Healthcare Analytics (NEW)
- Population-level Insights
- **Advanced DICOM/3D Viewer (P3-1)** - Smooth streaming viewer with treatment simulation

### 🏥 Integration Layer

- Real-time EHR streaming hooks (event-driven ingestion)
- Immutable audit log & trace provenance tagging
- Compliance scaffolding (HIPAA/FDA alignment docs)
- FDA Regulatory Pathway in progress

### 🔒 Security & Cryptography (P3-2)
- **Quantum-Safe Cryptography** - Hybrid Kyber768/AES-256 encryption
- **Production Key Management** - Automated rotation, KMS integration
- Zero-Trust Architecture
- Blockchain Medical Records
- HIPAA/GDPR compliance built-in

### 🚢 MLOps & Deployment (P3-3)
- **Canary Deployment Pipeline** - Shadow mode with continuous validation (implementation in `mlops/`)
- **Automated Testing** - Accuracy, fairness, performance, drift detection
- **Auto-Rollback** - Intelligent rollback on failures
- A/B testing for model comparison
- Production monitoring and alerting

> ⚠️ **Note:** K8s/Helm deployment assets, Prometheus/Grafana/Alertmanager observability configs, and baseline k6/Lighthouse performance validation are implemented in-repo. Remaining roadmap work focuses on CI/CD expansion and formal external audits/certifications. See [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).

  **Validated model performance & workflow:**
- Alzheimer’s Detection (AUC 0.93), Parkinson’s, ALS, MRI, CVD, and more
- CLI/API/GUI validated in real-world scenarios according to healthcaredeploymentplan.md

**Deployment planning stack (see KNOWN_LIMITATIONS for remaining gaps):**
- Deployment assets: Docker (root `Dockerfile`), K8s manifests (`k8s/`), Helm chart (`helm/aimedres/`), blue/green/canary strategies documented
- Monitoring assets: Prometheus/Grafana/Alertmanager (`monitoring/`), ELK and SIEM integration guidance in deployment docs

---

## 🔁 Recent Training Progress (Updated)

| Model / Variant          | Dataset(s)             | Target Task                    | Status          | Notes                      |
|------------------------- |-----------------------|-------------------------------|-----------------|----------------------------|
| AD_EARLY_V2              | ADNI + INTERNAL_SET_V1| MCI→AD conversion (12–24m)     | Production      | Improved temporal embeddings|
| AD_SCREEN_V1             | ADNI subset           | Screening classifier           | Production      | Class imbalance reweighting|
| MULTI_AGENT_CONSENSUS_V3 | Simulated + Expert    | Agreement score                | Production      | Enhanced conflict resolver |
| MEMORY_CONSOLIDATION_V4  | Synthetic episodic    | Retention @24h                 | Production      | Synaptic tagging decay     |
| LATENCY_OPT_BATCH_OPT    | Live inference        | p95 latency optimization       | Production      | CUDA graphs + fused ops    |

---

## 📈 Current Performance Snapshot

*Note: Detailed performance metrics are environment and dataset-specific. The following represents typical performance on reference datasets with standard configurations.*

- **Response Time**: 
  - p50 = 45-65 ms | p95 = 85-120 ms (target <100ms for p95)
- **Alzheimer's Early Detection** (ADNI dataset):
  - Sensitivity: 88-92%  (target ≥92%)
  - Specificity: 85-89%  (target ≥87%)
  - AUC: 0.89-0.93
- **Multi-Agent Consensus Agreement**: 85-92%
- **Memory Retention** (24h simulated): 78-85%
- **EHR Stream Throughput**: 500-1000 events/sec sustained

*Performance varies based on hardware, configuration, and data characteristics. See individual model documentation for detailed benchmarks.*

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

## 🤖 Basic Usage & Model Training

Examples for research and dev environments (see “Clinical Integration” for production):

```python
from aimedres.training.alzheimer_training_system import (
    load_alzheimer_data, train_model, evaluate_model
)

df = load_alzheimer_data("data/adni_processed.csv")
model = train_model(df, epochs=10, adaptive=True)
metrics = evaluate_model(model, df)
print(metrics)
```

Multi-model, parallel, or agent-based workflows: see “Run ALL Medical AI Models” in [README.md](README.md).

---

## 📋 Documentation & Deployment Resources

- **Comprehensive clinical deployment, security, validation, monitoring, audit, incident, and governance documentation:**  
  [healthcaredeploymentplan.md](healthcaredeploymentplan.md)
- **Sub-guides for every phase (scripts, templates, configs, compliance):**  
  See the `deployment/` directory  
- **Model documentation/model cards:**  
  See individual model output in `/results/` and `/var/aimedres/models`
- **Research/developer docs:**  
  See [docs/](docs/), [examples/](examples/), and script docstrings

---

## ⚖️ License

**GNU General Public License v3.0 (GPL-3.0)**  
See [LICENSE](LICENSE) for details.

---

## 👨‍⚕️ Clinical Safety Note

**This software is a research prototype. Before any clinical deployment, institutional validation, regulatory compliance review, and sign-off are mandatory.**  
It is **NOT** approved for patient-facing decisions without proper clinical validation.  
By using this system, you agree to all legal and compliance disclaimers in this README and [healthcaredeploymentplan.md](healthcaredeploymentplan.md).

---

## 📞 Contact & Support

- Lead: [V1B3hR](https://github.com/V1B3hR)
- Discuss: https://github.com/V1B3hR/AiMedRes/discussions
- Issues: https://github.com/V1B3hR/AiMedRes/issues

---

*Advancing responsible AI for neurological health – from research to real clinical impact.* 🧠
## 🚂 Run Training for ALL Medical AI Models

AiMedRes includes **7 comprehensive medical AI models** that can be trained using the unified orchestrator:

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

# Production-ready configuration
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
# Alzheimer's model
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

Training outputs to `results/`:
- Trained models (`.pkl`, `.pth`)
- Metrics (CSV, JSON)
- Logs, plots

### Run Examples

```bash
python examples/basic/run_all_demo.py
python examples/clinical/alzheimer_demo.py
python examples/advanced/parallel_mode.py
python examples/enterprise/production_demo.py
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

## 🧪 Model Zoo (Emerging)

| Name                 | Task                  | Status           | Checkpoint | Notes                               |
|----------------------|-----------------------|------------------|------------|-------------------------------------|
| ad_early_v2          | 12–24m conversion     | Stable Candidate | (planned)  | Temporal embedding + adaptive pruning|
| ad_screen_v1         | Screening classifier  | Beta             | (planned)  | High sensitivity emphasis           |
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

## 🛡️ Safety & Compliance

### Human-in-Loop Gating (P0-5) ✅
- **Mandatory human approval** for HIGH and CRITICAL risk recommendations
- **Immutable audit logging** with cryptographic verification (blockchain-like chaining)
- **Rationale requirements**: Every approval/rejection requires documented clinical rationale
- **Review time tracking**: Audit logs capture review duration for oversight
- **Admin override capability**: Emergency overrides logged with detailed justification
- **Audit chain verification**: Tamper-evident audit trail with hash verification

### PHI Protection (P0-3) ✅
- **Automated PHI detection**: Comprehensive scrubber detecting 18 HIPAA Safe Harbor identifiers
- **CI enforcement**: Automated tests prevent PHI from entering repository
- **De-identification pipeline**: Safe Harbor compliant data sanitization
- **Clinical term whitelist**: Preserves medical terminology while removing identifiers
- **Dataset validation**: Batch processing with detailed PHI detection reports

### Additional Security Features
- Configurable risk thresholds
- Bias & drift monitoring (dashboards in progress)
- FDA regulatory pathway planning underway
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

## 🛣️ Roadmap (v1.0.0 - November 2025)

### Recently Implemented
- Core adaptive architecture
- Multi-agent baseline
- Memory dual-store prototype
- Latency optimization (CUDA, kernel fusion)
- Expanded evaluation harness (uncertainty, calibration)
- FHIR ingestion pipeline hardened
- Explainability dashboard backend
- EHR connectivity
- Clinical Decision Support
- 3D Brain Visualization
- Multi-Modal AI Integration
- Predictive Analytics
- Multi-Hospital Network
- Population-Level Insights
- FDA Regulatory Pathway (in progress)
- Clinical Pilot Programs (in progress)

### Upcoming
- Formal clinical pilot onboarding
- Advanced safety monitor (causal anomaly detection)
- Model card auto-generation
- Deployment orchestrator (K8s + streaming inference)
- Adversarial robustness suite

> For full roadmap and documentation, see [docs/roadmap.md](docs/roadmap.md) and [architecture docs](docs/architecture.md).  
> Recent PRs: [Implement P15-P17: 3D Brain Visualization, Multi-Modal AI Integration, and Predictive Healthcare Analytics](https://github.com/V1B3hR/AiMedRes/pull/162), [Implement P8B Clinical Pilot Programs and P9 FDA Regulatory Pathway Planning](https://github.com/V1B3hR/AiMedRes/pull/161), [Implement P12, P13, P14 roadmap phases](https://github.com/V1B3hR/AiMedRes/pull/160), [more...](https://github.com/V1B3hR/AiMedRes/pulls?q=is%3Apr+is%3Aclosed)

---

## 🧾 Data Sources & Ethics

| Dataset                | Usage                             | Access                    |
|------------------------|-----------------------------------|---------------------------|
| ADNI                   | Alzheimer's progression           | Licensed / user-supplied  |
| INTERNAL_SIM_V1        | Synthetic multimodal cases        | Generated                 |
| CLINICAL_NOTES_PROTOTYPE| Context enrichment (planned)     | Pending de-ID pipeline    |

Ethics:
- No raw PHI stored in repo
- Synthetic augmentation to reduce demographic skew
- Fairness reporting: stratified sensitivity/specificity (planned)

---

## 🤝 Contributing

Focus areas:
- Clinical validation scenarios
- Safety / auditing modules
- Latency + systems optimization
- Advanced memory & continual learning

See CONTRIBUTING.md (coming: new code style + testing matrix).

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

AiMedRes is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

This means:
- ✅ You are free to use, study, modify, and share this software
- ✅ You can use it for commercial purposes
- ⚠️ Any derivative works must also be released under GPL-3.0
- ⚠️ You must disclose source code when distributing
- ⚠️ You must state significant changes made to the software

See the [LICENSE](LICENSE) file for full legal terms.

**Important**: The GPL-3.0 license comes with NO WARRANTY. See LICENSE sections 15-16 for details.

For questions about licensing or alternative licensing arrangements, please open an issue or contact the maintainers.

---

## 📞 Contact

- Lead: [V1B3hR](https://github.com/V1B3hR)
- Issues: https://github.com/V1B3hR/AiMedRes/issues
- Discussions: https://github.com/V1B3hR/AiMedRes/discussions
- Collaboration: Open to research & clinical partners

---

*Advancing responsible AI for neurological health.* 🧠

> **Note:** This summary is based on the 30 most recent PRs/issues. There are 162 total; for a complete changelog and more details, visit [all closed PRs/issues](https://github.com/V1B3hR/AiMedRes/issues?q=is%3Apr+is%3Aclosed).
