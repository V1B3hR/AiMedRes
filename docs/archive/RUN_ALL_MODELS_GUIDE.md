# Running All Medical AI Models - Complete Guide

**Version**: 1.0.0 | **Last Updated**: November 2025

## Overview

The AiMedRes platform provides a unified orchestrator that can train **ALL 7 medical AI models** simultaneously or individually. This guide shows you how to use it.

## Available Medical AI Models

The orchestrator includes the following medical AI models:

1. **ALS (Amyotrophic Lateral Sclerosis)** - Progressive neurodegenerative disease prediction
2. **Alzheimer's Disease** - Cognitive decline and dementia prediction
3. **Parkinson's Disease** - Movement disorder prediction
4. **Brain MRI Classification** - Brain tumor classification from MRI scans
5. **Cardiovascular Disease** - Heart disease prediction
6. **Diabetes** - Diabetes risk prediction
7. **Specialized Medical Agents** - Multi-agent medical analysis system

## Quick Start - Run All Models

### Option 1: Using the CLI wrapper (Recommended)

```bash
# Run all models sequentially
./aimedres train

# Run all models in parallel (faster)
./aimedres train --parallel --max-workers 4

# Run all models with custom training parameters
./aimedres train --parallel --max-workers 6 --epochs 50 --folds 5
```

### Option 2: Using Python directly

```bash
# Run all models sequentially
python3 src/aimedres/cli/train.py

# Run all models in parallel
python3 src/aimedres/cli/train.py --parallel --max-workers 4
```

## Running Specific Models

### Run a Single Model

```bash
# Train only ALS model
./aimedres train --only als

# Train only Alzheimer's model
./aimedres train --only alzheimers

# Train only Parkinson's model
./aimedres train --only parkinsons
```

### Run Multiple Specific Models

```bash
# Train ALS, Alzheimer's, and Parkinson's models
./aimedres train --only als alzheimers parkinsons

# Train with parallel execution
./aimedres train --parallel --max-workers 3 --only als alzheimers parkinsons
```

## Advanced Usage

### Custom Training Parameters

```bash
# Set epochs for all compatible models
./aimedres train --epochs 100

# Set cross-validation folds
./aimedres train --folds 10

# Set batch size (for models that support it)
./aimedres train --batch 256

# Combine multiple parameters
./aimedres train --epochs 100 --folds 10 --batch 256
```

### Parallel Training with Custom Parameters

```bash
# Recommended: Train all models in parallel with optimized settings
./aimedres train \
  --parallel \
  --max-workers 6 \
  --epochs 50 \
  --folds 5 \
  --batch 128
```

### Dry Run Mode

Preview commands without actually running training:

```bash
# See what commands would be executed
./aimedres train --dry-run

# Preview parallel execution
./aimedres train --dry-run --parallel --max-workers 4 --epochs 50
```

### List Available Models

```bash
# List all available training jobs
./aimedres train --list

# List with filters applied
./aimedres train --list --only als alzheimers
```

## Output Management

### Custom Output Directory

```bash
# Specify custom output directory
./aimedres train --base-output-dir my_results

# Each model will create a subdirectory:
# - my_results/als_comprehensive_results/
# - my_results/alzheimer_comprehensive_results/
# - etc.
```

### Logs and Summaries

```bash
# Custom log directory
./aimedres train --logs-dir my_logs

# Custom summary directory
./aimedres train --summary-dir my_summaries
```

## Model-Specific Arguments

### ALS Model

```bash
# ALS with specific dataset choice
./aimedres train --only als --extra-arg --dataset-choice=als-progression
```

### Parkinson's Model

```bash
# Parkinson's with custom data path
./aimedres train --only parkinsons --extra-arg --data-path=ParkinsonDatasets
```

## Filtering and Exclusion

### Exclude Specific Models

```bash
# Train all models except Brain MRI
./aimedres train --exclude brain_mri

# Train all except multiple models
./aimedres train --exclude brain_mri specialized_agents
```

### Disable Auto-Discovery

```bash
# Use only the 7 default models, no auto-discovery
./aimedres train --no-auto-discover
```

## Error Handling and Retries

### Automatic Retries

```bash
# Retry failed jobs up to 2 times
./aimedres train --retries 2
```

### Allow Partial Success

```bash
# Exit with code 0 even if some non-critical jobs fail
./aimedres train --allow-partial-success
```

## Complete Example: Production Training Run

```bash
#!/bin/bash
# production_training.sh
# Complete training run with all optimizations

./aimedres train \
  --parallel \
  --max-workers 6 \
  --epochs 50 \
  --folds 5 \
  --batch 128 \
  --base-output-dir production_results \
  --logs-dir production_logs \
  --summary-dir production_summaries \
  --retries 2 \
  --allow-partial-success \
  --verbose
```

## Monitoring and Results

### Check Training Status

During training, you'll see:
- Real-time progress for each model
- Success/failure indicators
- Execution times
- Resource utilization

### Review Results

After training completes:

1. **Output directories**: `results/<model>_comprehensive_results/`
   - Trained models
   - Metrics and evaluations
   - Visualizations

2. **Log files**: `logs/<model_id>/run_*.log`
   - Detailed training logs per model
   - Error messages if any

3. **Summary JSON**: `summaries/training_summary_<timestamp>.json`
   - Complete training metadata
   - All model results
   - Execution statistics

## Troubleshooting

### View Verbose Output

```bash
./aimedres train --verbose
```

### Test Configuration First

```bash
# Dry run to check command generation
./aimedres train --dry-run --verbose
```

### Run Single Model for Testing

```bash
# Test with just one model
./aimedres train --only als --epochs 5 --folds 2
```

## Performance Tips

1. **Use parallel execution** for faster training: `--parallel --max-workers 4`
2. **Adjust epochs and folds** based on your needs
3. **Monitor resource usage** - reduce `--max-workers` if system is overloaded
4. **Use `--dry-run`** to verify commands before actual execution

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Train All Medical AI Models

on:
  workflow_dispatch:
    inputs:
      epochs:
        description: 'Number of training epochs'
        default: '50'
      folds:
        description: 'Cross-validation folds'
        default: '5'

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Train all models
        run: |
          ./aimedres train \
            --parallel \
            --max-workers 4 \
            --epochs ${{ github.event.inputs.epochs }} \
            --folds ${{ github.event.inputs.folds }} \
            --allow-partial-success
```

## Summary

The orchestrator makes it easy to train all medical AI models:

- **7 medical AI models** available out of the box
- **Parallel execution** for faster training
- **Flexible parameters** for customization
- **Automatic discovery** of new training scripts
- **Comprehensive logging** and results tracking
- **Error handling** with retries and partial success

For more details, see:
- `./aimedres train --help` - Full command reference
- `src/aimedres/cli/train.py` - Orchestrator implementation
- `tests/integration/test_run_all_training.py` - Usage examples
