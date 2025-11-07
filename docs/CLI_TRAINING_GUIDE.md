# Quick Start: Training Medical AI Models with aimedres CLI

**Version**: 1.0.0 | **Last Updated**: November 2025

## Command Overview

The `aimedres train` command provides a modern, unified interface for training medical AI models with support for parallel execution and customizable parameters.

## Basic Usage

```bash
aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128
```

## Command-Line Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--parallel` | Enable parallel execution of training jobs | Off | `--parallel` |
| `--max-workers N` | Maximum number of concurrent workers | 4 | `--max-workers 6` |
| `--epochs N` | Number of training epochs | Script default | `--epochs 50` |
| `--folds N` | Number of cross-validation folds | Script default | `--folds 5` |
| `--batch N` | Batch size for training | Script default | `--batch 128` |
| `--only MODEL [MODEL ...]` | Train only specified models | All models | `--only als alzheimers` |
| `--exclude MODEL [MODEL ...]` | Exclude specified models | None | `--exclude diabetes` |
| `--list` | List available training jobs and exit | - | `--list` |
| `--dry-run` | Show commands without executing | - | `--dry-run` |
| `--config FILE` | Load job configuration from YAML | None | `--config jobs.yaml` |

## Examples

### 1. Train All Models with Parallel Execution

```bash
aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128
```

This will:
- Train all discovered medical AI models in parallel
- Use up to 6 concurrent workers
- Apply 50 epochs to compatible neural network models
- Use 5-fold cross-validation where supported
- Set batch size to 128 for compatible models

### 2. Train Specific Models

```bash
aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128 \
               --only als alzheimers parkinsons
```

Trains only ALS, Alzheimer's, and Parkinson's disease prediction models.

### 3. Sequential Training (No Parallelism)

```bash
aimedres train --epochs 50 --folds 5 --batch 128
```

Trains models one at a time without parallel execution.

### 4. List Available Training Jobs

```bash
aimedres train --list
```

Displays all available training jobs with their supported parameters.

### 5. Preview Commands (Dry Run)

```bash
aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128 --dry-run
```

Shows what commands would be executed without actually running them.

### 6. Exclude Specific Models

```bash
aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 \
               --exclude brain_mri cardiovascular
```

Trains all models except Brain MRI and Cardiovascular.

## Available Medical AI Models

The following models are automatically discovered and available for training:

### Core Disease Prediction Models
- **als** - ALS (Amyotrophic Lateral Sclerosis) progression prediction
- **alzheimers** - Alzheimer's Disease detection and staging
- **parkinsons** - Parkinson's Disease prediction
- **brain_mri** - Brain MRI classification
- **cardiovascular** - Cardiovascular disease prediction
- **diabetes** - Diabetes prediction

### Parameter Support by Model

| Model | Epochs | Folds | Batch Size | Output Dir |
|-------|--------|-------|------------|------------|
| ALS | ✅ | ✅ | ✅ | ✅ |
| Alzheimer's | ✅ | ✅ | ❌ | ✅ |
| Parkinson's | ✅ | ✅ | ❌ | ✅ |
| Brain MRI | ✅ | ❌ | ✅ | ✅ |
| Cardiovascular | ✅ | ✅ | ❌ | ✅ |
| Diabetes | ✅ | ✅ | ❌ | ✅ |

**Note**: Parameters are intelligently applied only to compatible scripts. Unsupported parameters are automatically skipped for each model.

## Output and Results

### Directory Structure

```
results/
├── als_comprehensive_results/
├── alzheimer_comprehensive_results/
├── parkinsons_comprehensive_results/
├── brain_mri_comprehensive_results/
├── cardiovascular_comprehensive_results/
└── diabetes_comprehensive_results/

logs/
├── orchestrator.log
├── als/
├── alzheimers/
└── ...

summaries/
└── training_summary_YYYYMMDD_HHMMSS.json
```

### Summary Report

After training completes, a JSON summary is generated in `summaries/` containing:
- Training pipeline metadata
- Job status (success/failed/skipped)
- Execution times and durations
- Commands executed
- GPU information
- Git commit hash

## Advanced Usage

### Using a Configuration File

Create a YAML configuration file `jobs.yaml`:

```yaml
jobs:
  - name: "ALS Custom"
    script: "src/aimedres/training/train_als.py"
    output: "als_custom_results"
    id: "als_custom"
    args:
      dataset-choice: "als-progression"
      learning-rate: 0.001
    supports_epochs: true
    supports_folds: true
    supports_batch: true
```

Then run:

```bash
aimedres train --config jobs.yaml --parallel --max-workers 6
```

### Environment Variables

Set these environment variables to customize behavior:

```bash
export AIMEDRES_LOG_LEVEL=DEBUG
export AIMEDRES_GPU_DEVICE=0
```

## Troubleshooting

### Issue: Parallel mode not starting

**Solution**: Ensure you have enough system resources and reduce `--max-workers`:

```bash
aimedres train --parallel --max-workers 2 --epochs 50 --folds 5
```

### Issue: Out of memory errors

**Solution**: Reduce batch size:

```bash
aimedres train --parallel --max-workers 2 --epochs 50 --folds 5 --batch 64
```

### Issue: Model-specific parameter not working

**Solution**: Check parameter support with `--list`:

```bash
aimedres train --list
```

Look for parameter support indicators (e.g., `batch=True` or `batch=False`).

## Backward Compatibility

The legacy command format is still supported:

```bash
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128
```

However, the modern `aimedres train` command is recommended for new workflows.

## Getting Help

```bash
# General help
aimedres --help

# Training command help
aimedres train --help

# List available jobs
aimedres train --list
```

## Performance Tips

1. **Use parallel execution** for multiple models: `--parallel --max-workers 6`
2. **Start with dry-run** to preview commands: `--dry-run`
3. **Train specific models** during development: `--only als`
4. **Use appropriate batch sizes** based on your GPU memory
5. **Monitor resources** when using parallel execution

## Next Steps

- Review the [examples/cli/](../examples/cli/) directory for more examples
- Check the [tests/integration/](../tests/integration/) directory for validation tests
- Explore individual training scripts in [src/aimedres/training/](../src/aimedres/training/)
