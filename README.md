# duetmind_adaptive

**duetmind_adaptive** is a hybrid AI framework that combines Adaptive Neural Networks (AdaptiveNN) with DuetMind cognitive agents. AdaptiveNN provides the "brain"—dynamic, learning, biologically inspired neural networks—while DuetMind agents drive reasoning, memory, and social interaction.

---

## File Organization

All training and test files are organized in the following structure:

- **Training Files**: All training scripts are located in `training/`
- **Test Files**: All test scripts are located in `files/tests/`
- **Dataset Files**: Dataset utilities are located in `files/dataset/`
- **Documentation Files**: All documentation is located in `docs/`
- **Neural Network Core**: The foundational neural network logic is implemented in `neuralnet.py`

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
    - **Core implementation in [`neuralnet.py`](./neuralnet.py)**

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

- **Biological Neural Simulation:** Agents have neural networks mimicking real biological cycles. See [`neuralnet.py`](./neuralnet.py) for core implementation.
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
- **Comprehensive Training Test:** Full system validation with detailed reporting.

---

## Quick Start

### Comprehensive Training and Simulation

Run the complete system that trains on real data and simulates medical consultations:

```bash
# New enhanced system with comprehensive dataset
python3 training/comprehensive_medical_ai_training.py

# Original system
python3 training/comprehensive_training_simulation.py

# Problem statement exact implementation
python3 problem_statement_implementation.py
```

### Individual Components

```bash
# Train enhanced medical model on comprehensive data
python3 training/enhanced_alzheimer_training_system.py

# Train original medical model
python3 training/alzheimer_training_system.py

# Validate data quality
python3 data_quality_monitor.py

# Run original adaptive simulation
python3 labyrinth_adaptive.py

# See usage examples
python3 usage_examples.py

# Run standalone neural network simulation
python3 neuralnet.py
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
# Run comprehensive training test and report
python3 run_comprehensive_training_test.py

# Run comprehensive training only
python3 full_training.py --mode comprehensive

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
- **Comprehensive Training Test**: Full system validation and reporting with:
  - Module import validation
  - Model persistence testing
  - Basic training validation
  - Kaggle dataset training validation
  - Agent simulation validation
  - End-to-end comprehensive training test
  - Detailed text and JSON reports
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
python3 training/run_training.py

# Modern API version (requires kagglehub setup)
python3 training/run_training_modern.py
```

These require Kaggle API setup but only load datasets without actual training.

---

## neuralnet.py: Neural Network Core

[`neuralnet.py`](./neuralnet.py) implements the foundational neural network logic powering agent cognition and adaptation.  
It includes:
- Biological state simulation (energy, sleep, mood)
- Memory and social signal processing
- Resource management and neural node dynamics

**Usage:**  
Most system scripts import and use `neuralnet.py` as the core adaptive neural network engine, but you can also run or extend it directly for custom neural network experiments.

---

## MLOps Architecture

**duetmind_adaptive** now includes a comprehensive MLOps (Machine Learning Operations) architecture that provides reproducible pipelines, automated training, and advanced agent memory capabilities.

### Quick Start with MLOps

```bash
# Setup MLOps environment
make setup-env

# Run the complete ML pipeline
make reproduce

# Start MLOps infrastructure (Postgres + MLflow + MinIO)
make infra-up

# View experiment tracking
make mlflow-ui
```

### MLOps Components

#### 🔄 **Data Pipeline**
- **Automated ingestion** from multiple data sources
- **Feature engineering** with schema validation (Pandera)
- **Data versioning** with DVC integration
- **Quality monitoring** and drift detection

#### 🧪 **Experiment Tracking**
- **MLflow integration** for model versioning and metrics
- **Feature hashing** for schema consistency
- **Performance monitoring** with automated alerts
- **Model registry** with promotion workflows

#### 🚀 **Production Infrastructure**
- **Docker Compose** stack (PostgreSQL + MinIO + MLflow)
- **Database migrations** with Alembic
- **Object storage** for model artifacts
- **CI/CD pipeline** with GitHub Actions

#### 🧠 **Agent Memory System**
- **Vector embeddings** with pgvector for semantic memory
- **Agent session management** with persistent memory
- **Semantic similarity search** for reasoning retrieval
- **Memory associations** for knowledge graphs

### MLOps Commands

```bash
# Data operations
make data              # Ingest raw data
make features          # Build and validate features  
make train             # Train model with MLflow logging
make validate          # Run schema validation
make hash-features     # Compute feature consistency hash

# Infrastructure
make infra-up          # Start MLOps infrastructure
make db-migrate        # Run database migrations
make backfill-metadata # Register existing data assets

# Development
make test              # Run test suite
make lint              # Code quality checks
make format            # Auto-format code
make reproduce         # Full DVC pipeline reproduction
```

### Architecture Phases

The MLOps system is implemented in three incremental phases:

1. **Phase 1 - Reproducibility** ✅
   - Data pipelines, schema validation, MLflow tracking, DVC integration

2. **Phase 2 - Infrastructure** ✅  
   - PostgreSQL metadata, Docker containers, model registry, migrations

3. **Phase 3 - Advanced Features** ✅
   - Vector memory, CI/CD automation, drift detection, agent embeddings

### Key Features

- **📊 Experiment Tracking**: MLflow integration with SQLite/PostgreSQL backend
- **🔍 Schema Validation**: Pandera contracts for data quality assurance
- **📦 Data Versioning**: DVC pipeline for reproducible data workflows
- **🏗️ Infrastructure**: Docker Compose for local development and testing
- **🤖 Agent Memory**: pgvector-based semantic memory for AI agents
- **⚡ Automation**: GitHub Actions CI/CD for automated retraining
- **📈 Monitoring**: Data drift detection and model performance tracking

### Documentation

- **[MLOps Architecture Guide](./docs/MLOPS_ARCHITECTURE.md)**: Comprehensive architecture documentation
- **[Model Promotion Guidelines](./mlops/registry/model_promotion.md)**: Model lifecycle management
- **[Configuration Reference](./params.yaml)**: Central parameter configuration

### Agent Memory Demo

The vector memory system enables agents to store and retrieve semantic memories:

```bash
# Run agent memory demonstration
python agent_memory/embed_memory.py
```

This demonstrates:
- Storing agent reasoning as vector embeddings
- Semantic similarity search for memory retrieval  
- Memory associations and importance scoring
- Session-based memory isolation

---

## Contributing

Contributions and feedback are welcome! Please open issues or pull requests for bugs, features, or documentation improvements.

---

## License

*Specify your license here.*

---
