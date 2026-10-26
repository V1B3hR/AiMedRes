# AiMedRes Documentation Index

**Version**: 1.0.0 | **Last Updated**: October 2026

Welcome to the AiMedRes documentation! This index helps you find the right documentation for your needs.

> **Current status:** AiMedRes is a research prototype, not clinical software. This index distinguishes the active evidence path from historical implementation and planning material. The [research-readiness roadmap](roadmap.md) and [Known Limitations](KNOWN_LIMITATIONS.md) take precedence whenever a document conflicts with them.

## 📖 Getting Started

- [README](../README.md) - Project overview and quick start
- [CONTRIBUTING](../CONTRIBUTING.md) - How to contribute to the project
- [CHANGELOG](../CHANGELOG.md) - Version history and changes

## 🚂 Training & Usage

### Scientific Research & Reproducibility
- [Parkinson's Research Package](../src/aimedres/research/parkinsons/README.md) - Decoupled, typed research pipeline & cryptographic provenance manifest
- [Local Experiment Tracker](../src/aimedres/research/tracking.py) - SQLite & JSON experiment lineage, parameter logging, and metric tracking
- [Domain Decoupled Architecture](../src/aimedres/service/) - Maintained research-metadata API (`service/`), research pipelines (`research/`), and quarantined simulations (`simulation/`)
- [PPMI Benchmark Reproduction Script](../scripts/reproduce_ppmi_benchmark.py) - One-command PPMI reproducibility runner
- [Synthetic PPMI Evidence Pack](../evidence/ppmi_synthetic_fixture/2026-10-25/EVIDENCE_PACK.md) - Versioned local control run, source hashes, and explicit non-clinical limits
- [Model Card - synthetic PPMI-schema benchmark](MODEL_CARD_PPMI_PARKINSONS.md) - Research-only scope, local fixture metrics, and validation limits
- [PPMI Data Governance](../ParkinsonDatasets/DATA_GOVERNANCE.md) - DUA compliance and restricted data protocols

### Training Guides
- [Training Usage Guide](TRAINING_USAGE.md) - Historical training reference; commands require re-verification
- [Training Scripts Reference](../src/aimedres/training/README.md) - Training script documentation
- [ALS Training Guide](ALS_TRAINING_GUIDE.md) - ALS-specific training
- [Brain MRI Training](BRAIN_MRI_TRAINING_IMPLEMENTATION.md) - Historical prototype reference; not a validated imaging model
- [Run All Guide](RUN_ALL_GUIDE.md) - Historical orchestration reference; requires re-verification

### Quick Starts
- [Quickstart Parallel Training](QUICKSTART_PARALLEL_TRAINING.md) - Parallel training quickstart
- [Parallel Mode README](PARALLEL_MODE_README.md) - Parallel training modes
- [Parallel 6 Workers Configuration](PARALLEL_6WORKERS_50EPOCHS_5FOLDS.md) - Advanced parallel setup

## 🏢 Architecture & Deployment Planning

> These are design and prototype references, not evidence of a deployed production system.

- [MLOps Architecture](MLOPS_ARCHITECTURE.md) - MLOps design and deployment planning
- [Production MLOps Guide](PRODUCTION_MLOPS_GUIDE.md) - Proposed MLOps capabilities; not deployed
- [Monitoring Enhancements](MONITORING_ENHANCEMENTS.md) - System monitoring and observability
- [Performance Testing Guide](PERFORMANCE_TESTING.md) - k6 and Lighthouse CI baseline
- [Phase 2B README](PHASE2B_README.md) - Phase 2B features

## 🤖 Agents & Features

- [Specialized Agents Implementation](SPECIALIZED_AGENTS_IMPLEMENTATION.md) - Medical AI agents
- [Specialized Agents Training](SPECIALIZED_AGENTS_TRAINING.md) - Training specialized agents

## 📊 Clinical & Compliance Planning

> These documents are draft frameworks and templates; they are not institutional approvals, regulatory submissions, or evidence of clinical validation.

- [Clinical Decision Support](CLINICAL_DECISION_SUPPORT_README.md) - Prototype CDS design reference
- [Parkinson & ALS Clinical Support](PARKINSON_ALS_CLINICAL_DECISION_SUPPORT_README.md) - Prototype support design reference
- [Clinical Validation Framework](CLINICAL_VALIDATION_FRAMEWORK.md) - Draft validation framework
- [Data Handling Procedures](DATA_HANDLING_PROCEDURES.md) - Data management and handling
- [Dataset Catalog & Acquisition Guide](DATASET_CATALOG.md) - USC LONI IDA (PPMI, ADNI, MarkVCID) and open-access public dataset guide

## 🔒 Security & Privacy

- [Security Guidelines](SECURITY_GUIDELINES.md) - Security best practices
- [Security Configuration](SECURITY_CONFIGURATION.md) - Security setup
- [Security Assessment](SECURITY_ASSESSMENT.md) - Internal controls reference, not an external assessment
- [Phase 2B Security Guide](PHASE2B_SECURITY_GUIDE.md) - Prototype security-feature reference

## 🖼️ Visualization & Dashboards

- [Web Simulation Dashboard](WEB_SIMULATION_DASHBOARD_README.md) - Interactive web dashboards
- [Enhanced Features Documentation](ENHANCED_FEATURES_DOCUMENTATION.md) - Enhanced UI features

## 📚 Reference Documentation

- [API Reference](API_REFERENCE.md) - API documentation; verify against maintained research API
- [Automation & Scalability](AUTOMATION_SCALABILITY.md) - Scaling and automation
- [Data Imbalance Resolution](DATA_IMBALANCE_RESOLUTION.md) - Handling imbalanced datasets
- [Imaging De-identification Guide](IMAGING_DEIDENTIFICATION_GUIDE.md) - Medical imaging privacy
- [Model Card - Imaging](MODEL_CARD_IMAGING.md) - Imaging model documentation
- [Model Card - synthetic PPMI-schema benchmark](MODEL_CARD_PPMI_PARKINSONS.md) - Research-only scope and PPMI validation limits

## 🔄 Training Systems

- [Cardiovascular Training](CARDIOVASCULAR_TRAINING_SUMMARY.md) - Cardiovascular disease models
- [Structured Alzheimer Training](STRUCTURED_ALZ_TRAINING.md) - Alzheimer's training pipeline
- [Full Training Guide](FULL_TRAINING_GUIDE.md) - Complete training reference
- [Training README](TRAINING_README.md) - Training system overview
- [Remote Training Documentation](REMOTE_TRAINING_DOCUMENTATION.md) - Remote/distributed training

## 🗺️ Roadmap & Planning

- [Known Limitations](KNOWN_LIMITATIONS.md) - Technical constraints, audit gaps, and future improvements
- [PPMI Data Governance](../ParkinsonDatasets/DATA_GOVERNANCE.md) - Compliance policy and access requirements
- [Requirements](requirements.md) - System requirements

### Active Roadmap & Funding
- [Research-readiness Roadmap](roadmap.md) - Single source of truth for presentation readiness, audit gates, and next actions
- [Funding Strategy: UK and Poland](FUNDING_STRATEGY_UK_AND_POLAND.md) - UK-first funding routes and conditional Poland/Kujawsko-Pomorskie directions
- [Institutional Research Partnership Brief](INSTITUTIONAL_RESEARCH_PARTNERSHIP_BRIEF.md) - Shareable, research-only concept note, pre-proposal starter text and first-contact email

## 📝 Additional Resources

- [Enhanced Memory Consolidation](ENHANCED_MEMORY_CONSOLIDATION_README.md) - Memory system documentation
- [User Experience Guide](USER_EXPERIENCE_GUIDE.md) - User experience guidelines
- [Comprehensive Training Documentation](COMPREHENSIVE_TRAINING_DOCUMENTATION.md) - Detailed training docs

## 📦 Archive

Historical implementation summaries and phase completion documents can be found in [archive/](archive/).

---

**Need help?** Check the [README](../README.md) for quick start instructions or [CONTRIBUTING](../CONTRIBUTING.md) for how to get involved.
