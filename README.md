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
- **Neurocognitive Disorders** – Expansion toward broader spectrum (in progress)  
- **Mental Health State Modeling** – (roadmap / prototype status)  

---

## 🚀 Key Features

### 🧩 Intelligence & Architecture
- Adaptive neural evolution engine (dynamic layer & pathway adjustment)
- Multi-agent consultation & consensus system
- Biological state simulators (energy, mood, circadian influences)
- Dual-store + prioritized replay memory consolidation

### 📊 Clinical AI
- Risk scoring & uncertainty estimates
- Explainable prediction frames (feature attributions + causal hints)
- Configurable safety thresholds & override gating
- Quantitative performance dashboards (CLI + API + dashboard module)

### 🏥 Integration Layer
- FHIR / HL7 interface modules
- Real-time EHR streaming hooks (event-driven ingestion)
- Immutable audit log & trace provenance tagging
- Compliance scaffolding (HIPAA/FDA alignment docs)

---

## 🔁 Recent Training Progress (Updated)

| Model / Variant | Dataset(s) | Target Task | Best Metric | Prev Metric | Δ | Notes |
|-----------------|-----------|-------------|-------------|-------------|----|-------|
| AD_EARLY_V2 | ADNI + INTERNAL_SET_V1 | MCI→AD conversion (12–24m) | AUC = PLACEHOLDER | PLACEHOLDER | +PLACEHOLDER | Improved temporal embeddings |
| AD_SCREEN_V1 | ADNI subset | Screening classifier | Sens = PLACEHOLDER / Spec = PLACEHOLDER | Sens = PLACEHOLDER / Spec = PLACEHOLDER | +PLACEHOLDER | Class imbalance reweighting |
| MULTI_AGENT_CONSENSUS_V3 | Simulated + Expert Annotation | Agreement score |  PLACEHOLDER% | PLACEHOLDER% | +PLACEHOLDER | New conflict resolver |
| MEMORY_CONSOLIDATION_V4 | Synthetic episodic tasks | Retention @24h |  PLACEHOLDER% | PLACEHOLDER% | +PLACEHOLDER | Added synaptic tagging decay |
| LATENCY_OPT_BATCH_OPT | Live inference harness | p95 latency |  PLACEHOLDER ms | PLACEHOLDER ms | -PLACEHOLDER ms | CUDA graphs + fused ops |

(Replace PLACEHOLDER values with actual results; I can regenerate this table.)

---

## 📈 Current Performance Snapshot

Update these with your latest numbers:

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

## 🔬 Reproducing Training Runs

```bash
# 1. Environment
pip install -r requirements.txt

# 2. Data prep (ADNI example — requires credentials / license)
python scripts/prepare_adni.py --input ~/raw_adni --output data/adni_processed.csv

# 3. Train
python scripts/train_alzheimer.py \
  --data data/adni_processed.csv \
  --model out/models/ad_early_v2.pt \
  --adaptive \
  --epochs 25 \
  --batch-size 64

# 4. Evaluate
python scripts/eval_alzheimer.py --model out/models/ad_early_v2.pt --data data/adni_processed.csv
```

Determinism options:
- Set `AIMEDRES_SEED=42`
- Use `--deterministic` flag in training script
- Logged configs stored under `out/runs/<timestamp>/config.yaml`

---

## 🧪 Model Zoo (Emerging)

| Name | Task | Status | Checkpoint | Notes |
|------|------|--------|------------|-------|
| ad_early_v2 | 12–24m conversion | Stable Candidate | (planned) | Temporal embedding + adaptive pruning |
| ad_screen_v1 | Screening classifier | Beta | (planned) | High sensitivity emphasis |
| consensus_v3 | Multi-agent aggregator | Experimental | (planned) | Weighted disagreement resolution |
| memory_v4 | Consolidation kernel | Experimental | (planned) | Synaptic tagging + decay curves |

(If you want, I can script automatic table generation from a registry file.)

---

## 📁 Project Structure (Consolidated)

```
src/aimedres/
  training/                  # Disease-specific training pipelines & core training infrastructure
    train_alzheimers.py      # Alzheimer's disease classification
    train_als.py             # ALS classification
    train_parkinsons.py      # Parkinson's disease classification
    train_brain_mri.py       # Brain MRI image classification
    train_cardiovascular.py  # Cardiovascular risk prediction
    train_diabetes.py        # Diabetes classification
    automation_system.py     # Training automation
    custom_pipeline.py       # Dynamic pipeline builder
    orchestration.py         # Workflow orchestration
  agents/                    # Medical reasoning & specialized agents
    specialized_medical_agents.py  # Multi-agent medical simulation
  agent_memory/              # Memory consolidation & storage
    memory_consolidation.py  # Dual-store consolidation system
    embed_memory.py          # Vector memory store
    agent_extensions.py      # Plugin system
  core/                      # Core components
    neural_network.py        # Adaptive neural networks
    agent.py                 # DuetMind agent framework
    config.py                # Configuration management
  security/                  # Security & validation
  api/                       # REST API
  utils/                     # Utilities
scripts/                     # CLI utilities
tests/                       # Unit / integration tests
examples/                    # Usage examples
```

---

## 🛡️ Safety & Compliance

- Configurable risk thresholds & human-in-loop gating
- Immutable audit trails (planned: hash chain / ledger)
- Bias & drift monitoring hooks (framework present; dashboards WIP)
- Privacy: de-identification utilities (PHI scrubber module planned)

---

## 📚 Documentation

| Topic | Link |
|-------|------|
| Technical Architecture | docs/architecture.md |
| Medical Applications | docs/medical-applications.md |
| API Reference | docs/api-reference.md |
| Compliance Tracking | docs/compliance.md |
| Publications / Notes | docs/publications.md |

(Generate missing docs via `scripts/scaffold_docs.py` – planned.)

---

## 🛣️ Roadmap (High-Level Update)

### Active (Now)
- [x] Core adaptive architecture
- [x] Multi-agent baseline
- [x] Memory dual-store prototype
- [🟧] Latency optimization pass (CUDA graph + kernel fusion)
- [🟧] Expanded evaluation harness (uncertainty + calibration)
- [🟧] FHIR ingestion pipeline hardening
- [🟧] Explainability dashboard backend
- [ ] Formal clinical pilot onboarding
- [ ] Regulatory pre-assessment packet

### Upcoming
- [ ] Parkinson's & ALS dataset integration
- [ ] Adversarial robustness testing suite
- [ ] Advanced safety monitor (causal anomaly detection)
- [ ] Model card auto-generation
- [ ] Deployment orchestrator (K8s + streaming inference)

---

## 🧾 Data Sources & Ethics

| Dataset | Usage | Access |
|---------|-------|--------|
| ADNI | Alzheimer's progression training/eval | Licensed / user-supplied |
| INTERNAL_SIM_V1 | Synthetic multimodal cases | Generated |
| CLINICAL_NOTES_PROTOTYPE (planned) | Context enrichment | Pending de-ID pipeline |

Ethics:
- No raw PHI stored in repo
- Synthetic augmentation to reduce demographic skew
- Planned fairness reporting: stratified sensitivity/specificity

---

## 🤝 Contributing

Focus areas:
- Clinical validation scenarios
- Safety / auditing modules
- Latency + systems optimization
- Advanced memory & continual learning
See CONTRIBUTING.md (coming update: new code style + testing matrix).

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

This software is for **research & development**. It does **not** provide medical diagnosis. Clinical decisions must be made by licensed professionals.

---

## 📞 Contact

- Lead: [V1B3hR](https://github.com/V1B3hR)
- Issues: https://github.com/V1B3hR/AiMedRes/issues
- Discussions: https://github.com/V1B3hR/AiMedRes/discussions
- Collaboration: Open to research & clinical partners

---

*Advancing responsible AI for neurological health.* 🧠
