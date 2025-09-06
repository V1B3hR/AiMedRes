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

---

## Roadmap

- [x] Core integration of AdaptiveNN and DuetMind
- [x] Biological state simulation (energy, sleep, mood)
- [x] Multi-agent dialogue engine
- [ ] Advanced safety monitoring and intervention
- [ ] Expanded memory consolidation algorithms
- [ ] Visualization tools for network states and agent dialogs
- [ ] API for custom agent behaviors and extensions
- [ ] Comprehensive documentation and tutorials

---

## Getting Started

### Quick Start Commands

For the problem statement requirements:

```bash
# Run comprehensive training
python3 run_training_comprehensive.py

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
# Install required dependency
pip install numpy
```

No additional setup required - the system uses synthetic training data and works out of the box.

### Legacy Training Scripts

The original training scripts are still available:

```bash
# Basic training script (requires kagglehub setup)
python3 run_training.py

# Modern API version (requires kagglehub setup)
python3 run_training_modern.py
```

These require Kaggle API setup but only load datasets without actual training.

---

## Contributing

Contributions and feedback are welcome! Please open issues or pull requests for bugs, features, or documentation improvements.

---

## License

*Specify your license here.*

---
