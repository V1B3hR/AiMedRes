# Specialized Medical Agents Training

## Overview

The `train_specialized_agents.py` script trains machine learning models for specialized medical agents (Radiologist, Neurologist, Pathologist) that can work together for enhanced medical diagnosis and consensus-based decision making.

## Features

- **Multi-Agent Training**: Trains separate models for different specialized agent roles
- **Cross-Validation**: Configurable k-fold cross-validation (default: 5 folds)
- **Neural Networks**: Optional neural network training with configurable epochs (default: 50)
- **Consensus Metrics**: Calculates multi-agent consensus and diversity metrics
- **Comprehensive Output**: Saves models, metrics, and detailed training summaries

## Usage

### Standalone Execution

```bash
# Basic usage (uses default parameters)
python src/aimedres/training/train_specialized_agents.py

# With custom parameters
python src/aimedres/training/train_specialized_agents.py \
    --output-dir specialized_agents_results \
    --folds 5 \
    --epochs 50

# With custom dataset
python src/aimedres/training/train_specialized_agents.py \
    --data-path /path/to/dataset.csv \
    --output-dir my_results \
    --folds 10 \
    --epochs 100
```

### Via Training Orchestrator

The script is automatically discovered by `run_all_training.py` and can be run as part of the unified training pipeline:

```bash
# Run all training jobs including specialized agents
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5

# Run only specialized agents training
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --only train_specialized_agents

# Dry-run to see the command
python run_all_training.py --epochs 50 --folds 5 --only train_specialized_agents --dry-run
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | str | `specialized_agents_results` | Directory to save training outputs |
| `--data-path` | str | `None` | Path to dataset CSV file (if None, downloads from Kaggle) |
| `--folds` | int | `5` | Number of cross-validation folds |
| `--epochs` | int | `50` | Number of training epochs for neural networks |

## Trained Agents

The script trains models for three specialized medical agents:

1. **Radiologist Agent** (Random Forest)
   - Specializes in image-based diagnostic patterns
   - High analytical reasoning capability

2. **Neurologist Agent** (Gradient Boosting)
   - Specializes in cognitive assessment
   - Strong logical reasoning

3. **Pathologist Agent** (Logistic Regression)
   - Specializes in systematic analysis
   - Baseline diagnostic capability

## Output Structure

After training, the following directory structure is created:

```
<output-dir>/
├── agent_models/
│   ├── radiologist_rf_model.pkl
│   ├── neurologist_gb_model.pkl
│   ├── pathologist_lr_model.pkl
│   └── *_neural_model.pth (if PyTorch available)
├── metrics/
│   └── agent_training_results.json
├── preprocessors/
│   ├── agent_preprocessor.pkl
│   └── label_encoder.pkl
└── agent_training_summary.txt
```

## Metrics

The training pipeline calculates and reports:

- **Per-Agent Metrics**:
  - Accuracy (with standard deviation)
  - Balanced Accuracy
  - F1 Score (weighted)
  - ROC AUC (for binary classification)

- **Consensus Metrics**:
  - Mean Accuracy across all agents
  - Mean F1 Score
  - Diversity Score (measures opinion variance)
  - Number of participating agents

## Example Output

```
RADIOLOGIST Cross-Validation Results:
  Accuracy:          0.8750 (+/- 0.0245)
  Balanced Accuracy: 0.8623 (+/- 0.0289)
  F1 Weighted:       0.8698 (+/- 0.0267)
  ROC AUC:           0.9124 (+/- 0.0198)

Consensus Metrics:
  Mean Accuracy:     0.8533 (+/- 0.0312)
  Mean F1:           0.8421
  Diversity Score:   0.0312
  Number of Agents:  3
```

## Integration with Specialized Medical Agents

The trained models can be loaded and used with the `specialized_medical_agents.py` module for:

- Multi-agent diagnostic consensus
- Ensemble predictions
- Explainable AI analysis
- Clinical decision support

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- torch (optional, for neural networks)
- kagglehub (optional, for automatic dataset download)

## Verification

To verify the training pipeline is working correctly:

```bash
python test_specialized_agents_integration.py
```

This will run comprehensive tests to ensure:
1. CLI flags are properly configured
2. Auto-discovery by `run_all_training.py` works
3. Command construction is correct
4. Training executes successfully and produces expected outputs

## Notes

- If `kagglehub` is not available, you must provide `--data-path` pointing to a dataset CSV
- Neural network training is skipped if PyTorch is not installed
- The script uses the Alzheimer's Disease dataset by default for training medical reasoning models
- Cross-validation ensures robust model evaluation
- All models are saved in pickle format for easy deployment
