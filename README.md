# 🧠 AiMedRes

> **Advanced AI Medical Research Assistant**  
> **Adaptive neural architectures + multi-agent clinical reasoning**  
> **Safety‑aware, explainable AI for neurological and mental health research**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active%20development-green.svg)](https://github.com/V1B3hR/AiMedRes)

---

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

## 🚀 Key Features (2025)

### 🧩 Intelligence & Architecture
- Adaptive neural evolution engine (dynamic layer & pathway adjustment)
- Multi-agent consultation & consensus system
- Biological state simulators (energy, mood, circadian influences)
- Dual-store + prioritized replay memory consolidation
- Multi-Hospital Network and Specialty Modules
- 3D Brain Visualization (NEW)
- Multi-Modal AI Integration (EHR, imaging, notes)

### 📊 Clinical AI
- Risk scoring & uncertainty estimates
- Explainable prediction frames (feature attributions + causal hints)
- Configurable safety thresholds & override gating
- Quantitative performance dashboards (CLI + API + dashboard module)
- Predictive Healthcare Analytics (NEW)
- Population-level Insights

### 🏥 Integration Layer
- FHIR / HL7 interface modules
- Real-time EHR streaming hooks (event-driven ingestion)
- Immutable audit log & trace provenance tagging
- Compliance scaffolding (HIPAA/FDA alignment docs)
- FDA Regulatory Pathway in progress

---

## 🔁 Recent Training Progress (Updated)

| Model / Variant          | Dataset(s)             | Target Task                    | Best Metric      | Prev Metric     | Δ           | Notes                      |
|------------------------- |-----------------------|-------------------------------|------------------|-----------------|-------------|----------------------------|
| AD_EARLY_V2              | ADNI + INTERNAL_SET_V1| MCI→AD conversion (12–24m)     | AUC = PLACEHOLDER| PLACEHOLDER     | +PLACEHOLDER| Improved temporal embeddings|
| AD_SCREEN_V1             | ADNI subset           | Screening classifier           | Sens = PLACEHOLDER / Spec = PLACEHOLDER | Sens = PLACEHOLDER / Spec = PLACEHOLDER | +PLACEHOLDER | Class imbalance reweighting|
| MULTI_AGENT_CONSENSUS_V3 | Simulated + Expert Annotation | Agreement score         | PLACEHOLDER%     | PLACEHOLDER%    | +PLACEHOLDER| New conflict resolver      |
| MEMORY_CONSOLIDATION_V4  | Synthetic episodic tasks | Retention @24h              | PLACEHOLDER%     | PLACEHOLDER%    | +PLACEHOLDER| Added synaptic tagging decay|
| LATENCY_OPT_BATCH_OPT    | Live inference harness | p95 latency                   | PLACEHOLDER ms   | PLACEHOLDER ms  | -PLACEHOLDER ms | CUDA graphs + fused ops  |

---

## 📈 Current Performance Snapshot

- Response Time: p50 = PLACEHOLDER ms | p95 = PLACEHOLDER ms (target <100ms)
- Alzheimer's Early Detection:
  - Sensitivity: PLACEHOLDER%  (target ≥92%)
  - Specificity: PLACEHOLDER%  (target ≥87%)
  - AUC: PLACEHOLDER
- Multi-Agent Consensus Agreement: PLACEHOLDER%
- Memory Retention (24h simulated): PLACEHOLDER%
- EHR Stream Throughput: PLACEHOLDER events/sec sustained

---

## 🛠️ Installation

```bash
git clone https://github.com/V1B3hR/AiMedRes.git
cd AiMedRes
pip install -e .
# optional extras: pip install -e ".[dev]"
```

Quick import check:
```bash
python -c "from aimedres.training import AlzheimerTrainingPipeline; print('✓ Installation successful')"
```

### Requirements
- Python 3.10+
- Core: NumPy, Pandas, Scikit-learn
- Optional: PyTorch, XGBoost, MLflow (for advanced features)

See [docs/requirements.md](docs/requirements.md) for full details.

---

## 📁 Repository Structure

```
AiMedRes/
├── src/aimedres/           # Main package (all code here)
│   ├── training/           # Training pipelines (Alzheimer's, ALS, etc.)
│   ├── agents/             # Specialized medical agents
│   ├── agent_memory/       # Memory consolidation systems
│   ├── clinical/           # Clinical decision support
│   ├── compliance/         # HIPAA, FDA, regulatory modules
│   ├── core/               # Core utilities and agents
│   ├── dashboards/         # Visualization dashboards
│   └── ...
├── docs/                   # Documentation (guides, API, architecture)
├── tests/                  # Test suite
├── examples/               # Example scripts and demos
├── mlops/                  # Production MLOps infrastructure
└── configs/                # Configuration files
```

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for v0.2.0 cleanup details.

---

## 💡 Basic Usage

### Training (Alzheimer's pipeline)
```python
from aimedres.training.alzheimer_training_system import (
    load_alzheimer_data, train_model, evaluate_model
)

df = load_alzheimer_data("data/adni_processed.csv")
model = train_model(df, epochs=10, adaptive=True)
metrics = evaluate_model(model, df)
print(metrics)
```

### Multi-Agent Consensus
```python
from aimedres.agents.dialogue_manager import MultiAgentConsultation
from aimedres.agents.medical_reasoning import ClinicalDecisionAgent

consultation = MultiAgentConsultation([
    ClinicalDecisionAgent(role="neurologist"),
    ClinicalDecisionAgent(role="radiologist"),
    ClinicalDecisionAgent(role="cognitive_assessor"),
])

case = consultation.load_case("examples/cases/case_001.json")
recommendation = consultation.analyze_case(case)
print(recommendation.summary())
```

### Memory Consolidation
```python
from aimedres.agent_memory.memory_consolidation import MemoryConsolidator
from aimedres.agent_memory.memory_store import MemoryStore

store = MemoryStore()
consolidator = MemoryConsolidator(store=store, strategy="dual_store")
consolidator.ingest({"type": "clinical_event", "content": "...", "priority": 0.87})
consolidator.run_cycle()
```

---

## 🚂 Run Training

### Quick Start - Train All Models

```bash
# Unified CLI (recommended)
aimedres train

# Legacy script (still supported)
python run_all_training.py

# With custom parameters
aimedres train --epochs 20 --folds 5

# Parallel execution
aimedres train --parallel --max-workers 4

# Production-ready config
aimedres train --parallel --max-workers 6 --epochs 50 --folds 5
```

### Run Specific Models

```bash
# Only selected models
aimedres train --only als alzheimers parkinsons

# Exclude certain models
aimedres train --exclude brain_mri

# Dry run
aimedres train --dry-run --epochs 10
```

### Run Single Model

```bash
# Alzheimer's model default
python src/aimedres/training/train_alzheimers.py

# Custom parameters
python src/aimedres/training/train_als.py --epochs 50 --folds 3 --output-dir my_results
```

### List Available Training Jobs

```bash
aimedres train --list
```

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
- **Training Usage Guide**: [TRAINING_USAGE.md](TRAINING_USAGE.md)
- **Training Scripts Reference**: [src/aimedres/training/README.md](src/aimedres/training/README.md)
- **Implementation Details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Refactoring Summary**: [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)

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

- Configurable risk thresholds & human-in-loop gating
- Immutable audit trails (hash chain/ledger planned)
- Bias & drift monitoring (dashboards in progress)
- Privacy: de-identification utilities (PHI scrubber planned)
- FDA regulatory pathway planning underway

---

## 📚 Documentation

| Topic                 | Link                               |
|-----------------------|------------------------------------|
| Technical Architecture| docs/architecture.md               |
| Medical Applications  | docs/medical-applications.md       |
| API Reference         | docs/api-reference.md              |
| Compliance Tracking   | docs/compliance.md                 |
| Publications / Notes  | docs/publications.md               |

(Missing docs: generate via `scripts/scaffold_docs.py`)

---

## 🛣️ Roadmap (2025-10-10, Recent Progress)

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

> For full roadmap and documentation, see [ROADMAP.md](ROADMAP.md) and [architecture docs](docs/architecture.md).  
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

## 📚 Documentation

For comprehensive documentation, see the [docs/](docs/) directory:

- **Getting Started**: [Quick Reference](docs/guides/QUICK_REFERENCE.md), [Training Guide](docs/guides/TRAINING_USAGE.md)
- **Architecture**: [System Architecture](docs/ARCHITECTURE_REFACTOR_PLAN.md), [MLOps](docs/MLOPS_ARCHITECTURE.md)
- **API Reference**: [API Docs](docs/API_REFERENCE.md), [Clinical Decision Support](docs/CLINICAL_DECISION_SUPPORT_README.md)
- **Compliance**: [Data Handling](docs/DATA_HANDLING_PROCEDURES.md), [Security](docs/SECURITY_GUIDELINES.md)

Full documentation index: [docs/README.md](docs/README.md)

---

## 🤝 Contributing

We welcome contributions! Please see:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [security.md](security.md) - Security policies
- [CHANGELOG.md](CHANGELOG.md) - Version history

---

## ⚖️ Disclaimer

This software is for **research & development**. It does **not** provide medical diagnosis. Clinical decisions must be made by licensed professionals.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🌟 Project Status

- ✅ v0.2.0 - Repository cleanup and standardization complete
- ✅ Core training pipelines operational
- ✅ MLOps infrastructure implemented
- 🔄 Production deployment guides (in progress)
- 🔄 Enhanced testing coverage (in progress)

See [docs/roadmap.md](docs/roadmap.md) and [RELEASE_NOTES.md](RELEASE_NOTES.md) for details.

---

## 📞 Contact

- Lead: [V1B3hR](https://github.com/V1B3hR)
- Issues: https://github.com/V1B3hR/AiMedRes/issues
- Collaboration: Open to research & clinical partners

---

*Advancing responsible AI for neurological health.* 🧠
