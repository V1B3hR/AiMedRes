# Core Training Orchestrator

This directory contains a convenient wrapper for running all medical AI training from the core module.

## Quick Start

### From the Core Directory

```bash
cd /path/to/AiMedRes/src/aimedres/core
python run_all_training.py
```

This will run all 6 disease prediction models with default settings.

## Usage Examples

### List Available Models

```bash
python run_all_training.py --list
```

This shows all 6 core disease prediction models:
- ALS (Amyotrophic Lateral Sclerosis)
- Alzheimer's Disease
- Parkinson's Disease
- Brain MRI Classification
- Cardiovascular Disease Prediction
- Diabetes Prediction

### Run with Custom Parameters

```bash
# Run all models with 50 epochs and 5 folds
python run_all_training.py --epochs 50 --folds 5

# Run specific models only
python run_all_training.py --only als alzheimers parkinsons

# Run in parallel with 6 workers
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5
```

### Dry Run (Preview Commands)

```bash
python run_all_training.py --dry-run
```

### Exclude Specific Models

```bash
python run_all_training.py --exclude brain_mri
```

## Available Options

- `--list` - List all available training jobs
- `--epochs N` - Set number of epochs for all models
- `--folds N` - Set number of cross-validation folds
- `--only [MODEL...]` - Run only specified models
- `--exclude [MODEL...]` - Exclude specified models
- `--dry-run` - Show commands without executing
- `--parallel` - Run models in parallel
- `--max-workers N` - Number of parallel workers (default: 4)
- `--no-auto-discover` - Disable auto-discovery of training scripts
- `--verbose` - Enable verbose logging

## Model IDs

Use these IDs with `--only` and `--exclude`:
- `als` - ALS (Amyotrophic Lateral Sclerosis)
- `alzheimers` - Alzheimer's Disease
- `parkinsons` - Parkinson's Disease
- `brain_mri` - Brain MRI Classification
- `cardiovascular` - Cardiovascular Disease Prediction
- `diabetes` - Diabetes Prediction

## Output

Results are saved to `results/` directory at the repository root:
- `results/als_comprehensive_results/`
- `results/alzheimer_comprehensive_results/`
- `results/parkinsons_comprehensive_results/`
- `results/brain_mri_comprehensive_results/`
- `results/cardiovascular_comprehensive_results/`
- `results/diabetes_comprehensive_results/`

Logs are saved to `logs/` directory.

## How It Works

This script is a lightweight wrapper that:
1. Determines the repository root (3 levels up from core)
2. Changes to the repository root directory
3. Delegates to the main `run_all_training.py` at the repository root
4. Ensures all paths and imports work correctly

This allows you to conveniently run training from anywhere within the core module while maintaining compatibility with the main training orchestrator.
