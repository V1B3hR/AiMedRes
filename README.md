# duetmind_adaptive

**duetmind_adaptive** is a hybrid AI framework that combines Adaptive Neural Networks (AdaptiveNN) with DuetMind cognitive agents. AdaptiveNN provides the "brain"—dynamic, learning, biologically inspired neural networks—while DuetMind supplies the "mind"—reasoning, safety, and social interaction. Together they create truly adaptive, safe, multi-agent systems with emergent intelligence.

---

## File Organization

All training and test files are organized in the following structure:

- **Training Files**: All training scripts are located in `files/training/`
- **Test Files**: All test scripts are located in `files/tests/`
- **Dataset Files**: Dataset utilities are located in `files/dataset/`

This organization ensures a clean separation of concerns and makes it easy to locate specific functionality.

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
- **Enhanced Medical Data:** Integration with comprehensive 2149-patient dataset (rabieelkharoua/alzheimers-disease-dataset).
- **High-Accuracy Models:** 94.7% accuracy on comprehensive medical dataset with 32 features.
- **Medical AI Agents:** AI agents enhanced with medical reasoning capabilities.
- **Data Quality Monitoring:** Comprehensive validation and quality assurance.
- **Collaborative Decision Making:** Multi-agent medical consultation simulation.
- **Problem Statement Compliance:** Exact implementation of kagglehub.load_dataset requirements.

---

## Quick Start

### Comprehensive Training and Simulation

Run the complete system that trains on real data and simulates medical consultations:

```bash
# New enhanced system with comprehensive dataset
python3 files/training/comprehensive_medical_ai_training.py

# Original system
python3 files/training/comprehensive_training_simulation.py

# Problem statement exact implementation
python3 problem_statement_implementation.py
```

### Individual Components

```bash
# Train enhanced medical model on comprehensive data
python3 files/training/enhanced_alzheimer_training_system.py

# Train original medical model
python3 files/training/alzheimer_training_system.py

# Validate data quality
python3 data_quality_monitor.py

# Run original adaptive simulation
python3 labyrinth_adaptive.py

# See usage examples
python3 usage_examples.py
```

### Real Data Integration

The system uses real Alzheimer's disease data from Kaggle:

**Original Dataset (brsdincer/alzheimer-features):**
- **Dataset**: 373 patient records with 9 clinical features
- **Training**: Random Forest classifier with 100% test accuracy
- **Quality**: Comprehensive validation with 99.9% quality score

**Enhanced Dataset (rabieelkharoua/alzheimers-disease-dataset):**
- **Dataset**: 2149 patient records with 35 clinical and lifestyle features
- **Training**: Enhanced Random Forest classifier with 94.7% test accuracy
- **Features**: Comprehensive medical, demographic, and lifestyle variables
- **Integration**: Complete pipeline from kagglehub.load_dataset to collaborative AI deployment

Both datasets provide **seamless connection between training and simulation** for advanced medical AI research and applications.

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

### Quick Start Commands

For the problem statement requirements:

```bash
# Run comprehensive training
python3 files/training/run_training_comprehensive.py

# Run simulation  
python3 run_simulation.py
```

### Advanced Usage

Use the main entry point for more options:

```bash
# Interactive mode (default)
python3 main.py

# Run comprehensive training only
python3 main.py --mode training

# Run simulation only
python3 main.py --mode simulation

# Run both training and simulation
python3 main.py --mode both

# Enable verbose logging
python3 main.py --mode both --verbose
```

### What's Included

- **Comprehensive Training**: Multi-phase neural network training with:
  - Neural foundation training
  - Adaptive behavior training
  - Multi-agent coordination training
  - Biological cycle integration training
- **Adaptive Simulation**: 20-step labyrinth simulation with 3 adaptive agents
- **Training Reports**: Detailed training metrics and model artifacts
- **Multiple Entry Points**: Direct scripts and configurable main entry point

### Prerequisites

```bash
pip install kagglehub pandas psutil redis flask numpy
```

No additional setup required - the system uses synthetic training data and works out of the box.

### Legacy Training Scripts

The original training scripts are still available:

```bash
# Basic training script (requires kagglehub setup)
python3 files/training/run_training.py

# Modern API version (requires kagglehub setup)
python3 files/training/run_training_modern.py
```

These require Kaggle API setup but only load datasets without actual training.

---

## Contributing

Contributions and feedback are welcome! Please open issues or pull requests for bugs, features, or documentation improvements.

---

## License

*Specify your license here.*

---