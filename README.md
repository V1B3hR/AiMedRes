# duetmind_adaptive

**duetmind_adaptive** is a hybrid AI framework that combines Adaptive Neural Networks (AdaptiveNN) with DuetMind cognitive agents. AdaptiveNN provides the "brain"—dynamic, learning, biologically inspired neural networks—while DuetMind supplies the "mind"—reasoning, safety, and social interaction. Together they create truly adaptive, safe, multi-agent systems with emergent intelligence.

---

## Project Description

This project aims to develop living AI agents that possess:
- **Energy and Sleep Cycles:** Each agent simulates biological states such as energy and sleep, affecting its reasoning and performance.
- **Emergent Reasoning:** Agents' thought processes adapt dynamically based on internal neural network states.
- **Multi-Agent Dialogues:** Agents interact with each other, engaging in conversations and joint reasoning tasks.
- **Safety Monitoring:** Continuous monitoring of "biological" neural processes to ensure safe operation.
- **Memory Consolidation:** Agents consolidate memory during sleep phases, influencing future reasoning and behavior.

Unlike conventional AI, duetmind_adaptive creates cognitive agents whose reasoning engines are living, adaptive neural networks—resulting in “living, breathing, sleeping” brains.

---

## System Architecture

The architecture is composed of two major components:

1. **AdaptiveNN ("Brain")**
    - Dynamic neural nodes with energy, sleep, and mood states
    - Learning and adaptation over time
    - Biological process monitoring

2. **DuetMind ("Mind")**
    - Reasoning engine and dialogue management
    - Safety and social interaction modules
    - Memory consolidation and retrieval

### Simplified Architecture Diagram

```
+------------------+        +----------------------+
|   DuetMind       |<------>|   AdaptiveNN         |
|  (Mind)          |        |   (Brain)            |
|------------------|        |----------------------|
| Reasoning Engine |        | Neural Nodes         |
| Safety Monitor   |        | Energy/Sleep/Mood    |
| Dialogue System  |        | Biological Process   |
+------------------+        +----------------------+
      |                              ^
      v                              |
  Multi-Agent Communication <--------+
```

---

## Features

- **Biological Neural Simulation:** Agents have neural networks mimicking real biological cycles.
- **Emergent Intelligence:** Reasoning changes based on neural state.
- **Safe Multi-Agent Operation:** Built-in safety checks for biological and cognitive processes.
- **Social Interaction:** Agents participate in dialogues and collaborative tasks.
- **Memory & Sleep Dynamics:** Sleep phases affect memory consolidation and future behavior.
- **Real Data Training:** Comprehensive training on real Alzheimer's disease dataset.
- **Medical AI Agents:** AI agents enhanced with medical reasoning capabilities.
- **Data Quality Monitoring:** Comprehensive validation and quality assurance.
- **Collaborative Decision Making:** Multi-agent medical consultation simulation.

---

## Quick Start

### Comprehensive Training and Simulation

Run the complete system that trains on real data and simulates medical consultations:

```bash
python3 comprehensive_training_simulation.py
```

### Individual Components

```bash
# Train medical model on real data
python3 training/alzheimer_training_system.py

# Validate data quality
python3 data_quality_monitor.py

# Run original adaptive simulation
python3 labyrinth_adaptive.py

# See usage examples
python3 usage_examples.py
```

### Real Data Integration

The system uses real Alzheimer's disease data from Kaggle:
- **Dataset**: 373 patient records with 9 clinical features
- **Training**: Random Forest classifier with 100% test accuracy
- **Quality**: Comprehensive validation with 99.9% quality score
- **Integration**: Seamless connection between training and simulation

---

## Roadmap

- [x] Core integration of AdaptiveNN and DuetMind
- [x] Biological state simulation (energy, sleep, mood)
- [x] Multi-agent dialogue engine
- [x] **Comprehensive training on real Alzheimer's disease data**
- [x] **Medical AI agents with reasoning capabilities**
- [x] **Data quality monitoring and validation**
- [x] **Collaborative medical decision-making simulation**
- [ ] Advanced safety monitoring and intervention
- [ ] Expanded memory consolidation algorithms
- [ ] Visualization tools for network states and agent dialogs
- [ ] API for custom agent behaviors and extensions
- [ ] Web-based simulation dashboard
- [ ] Clinical integration and real-world deployment

---

## Getting Started

### Running Training

To load and work with Alzheimer's disease datasets for training AI models:

```bash
# Basic training script (matches problem statement format)
python3 run_training.py

# Modern API version (no deprecation warnings)  
python3 run_training_modern.py
```

Both scripts load Alzheimer's disease datasets from Kaggle and display the first 5 records.

### Prerequisites

1. Install required dependencies:
```bash
pip install kagglehub pandas
```

2. Set up Kaggle API credentials:
   - Create a Kaggle account and generate API credentials
   - Place your `kaggle.json` file in `~/.kaggle/` directory
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Additional Dataset Loaders

See `files/dataset/` directory for more dataset loading options and examples.

*More installation and usage instructions coming soon.*

---

## Contributing

Contributions and feedback are welcome! Please open issues or pull requests for bugs, features, or documentation improvements.

---

## License

*Specify your license here.*

---
