# 🧠 AiMedRes

> **Advanced AI Medical Research Assistant**  
> **Adaptive neural architectures + multi-agent clinical reasoning**  
> **Safety‑aware, explainable AI for neurological and mental health research**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)
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
pip install -r requirements.txt
# optional extras: pip install -e ".[dev,docs]"
```

Quick import check:
```bash
python -c "from aimedres.training.alzheimer_training_system import load_alzheimer_data; print('OK')"
```

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

## 📢 Citing

(Add once first preprint is available)
```
@article{aimedres2026,
  title   = {AiMedRes: Adaptive Multi-Agent Clinical Reasoning with Biological Memory Enhancement},
  author  = {...},
  year    = {2026},
  journal = {Preprint}
}
```

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
