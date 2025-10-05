# Run All Training - Quick Start Guide

## Overview

The AiMedRes training orchestrator provides a powerful "**run all**" capability that automatically discovers and executes all medical AI training scripts in the repository.

## Quick Start

### 1. Run All Training (Simplest)

```bash
python run_all_training.py
```

This command will:
- ✅ Auto-discover all training scripts (12 jobs found)
- ✅ Run them sequentially with default parameters
- ✅ Save results to `results/` directory
- ✅ Generate logs in `logs/` directory
- ✅ Create summary reports in `summaries/` directory

### 2. Run All with Custom Parameters

```bash
python run_all_training.py --epochs 50 --folds 5
```

Customize training parameters:
- `--epochs N` - Number of training epochs
- `--folds N` - Number of cross-validation folds
- `--verbose` - Show detailed progress

### 3. Run in Parallel (Faster)

```bash
python run_all_training.py --parallel --max-workers 4
```

Speed up training by running multiple models simultaneously:
- `--parallel` - Enable parallel execution
- `--max-workers N` - Number of concurrent jobs (default: 4)

### 4. Run Specific Models

```bash
# Run only ALS and Alzheimer's
python run_all_training.py --only als alzheimers

# Exclude specific models
python run_all_training.py --exclude diabetes brain_mri
```

### 5. Preview Commands (Dry Run)

```bash
python run_all_training.py --dry-run --epochs 10
```

See what commands would be executed without actually running them.

### 6. List Available Jobs

```bash
python run_all_training.py --list
```

Display all discovered training jobs and their configurations.

## Convenience Script

For a simplified experience, use the convenience shell script:

```bash
./run_medical_training.sh
```

This automatically runs all models with optimized parameters (50 epochs, 5 folds).

## What Gets Trained?

The orchestrator discovers and runs the following medical AI models:

1. **ALS** (Amyotrophic Lateral Sclerosis) - Neurodegenerative disease prediction
2. **Alzheimer's Disease** - Dementia progression prediction
3. **Parkinson's Disease** - Movement disorder classification
4. **Brain MRI Classification** - Brain imaging analysis
5. **Cardiovascular Disease** - Heart disease risk assessment
6. **Diabetes Prediction** - Type 2 diabetes risk prediction
7. **Additional pipeline models** - MLOps baseline models

**Total: 12 training jobs discovered**

## Advanced Options

### Custom Output Directory

```bash
python run_all_training.py --base-output-dir /path/to/results
```

### Retry Failed Jobs

```bash
python run_all_training.py --retries 2
```

### Custom Configuration

```bash
python run_all_training.py --config my_config.yaml
```

### Add Extra Arguments

```bash
python run_all_training.py --extra-arg --batch-size=32 --extra-arg --lr=0.001
```

## Verification

Verify the system is working correctly:

```bash
# Run comprehensive verification
python verify_run_all.py

# Expected output: 5/5 checks passed ✅
```

## Output Structure

After running, you'll find:

```
results/
├── als_comprehensive_results/
├── alzheimer_comprehensive_results/
├── parkinsons_comprehensive_results/
├── cardiovascular_results/
├── diabetes_results/
├── brain_mri_results/
└── ... (other model results)

logs/
├── orchestrator.log
├── als.log
├── alzheimers.log
└── ... (individual job logs)

summaries/
└── training_summary_YYYYMMDD_HHMMSS.json
```

## GitHub Actions Integration

The orchestrator can be triggered via GitHub Actions for automated training:

1. Go to **Actions** tab in GitHub
2. Select **"Training Orchestrator"** workflow
3. Click **"Run workflow"**
4. Configure parameters (epochs, folds, parallel mode)
5. Click **"Run workflow"** button

Results are automatically uploaded as artifacts.

## Troubleshooting

### Missing Dependencies

If you see dependency errors:

```bash
pip install -r requirements-ml.txt
```

### No Training Scripts Found

Ensure you're in the repository root:

```bash
cd /path/to/AiMedRes
python run_all_training.py --list
```

### Job Failures

View detailed logs:

```bash
# Orchestrator log
cat logs/orchestrator.log

# Specific job log
cat logs/als.log
```

Enable retries for transient failures:

```bash
python run_all_training.py --retries 2
```

## Performance Tips

1. **Use Parallel Mode**: Reduce total time by running models concurrently
   ```bash
   python run_all_training.py --parallel --max-workers 4
   ```

2. **Filter Jobs**: Run only the models you need
   ```bash
   python run_all_training.py --only als alzheimers parkinsons
   ```

3. **Reduce Epochs**: For testing, use fewer epochs
   ```bash
   python run_all_training.py --epochs 5 --dry-run  # Preview first
   ```

4. **GPU Acceleration**: The orchestrator automatically detects and uses GPUs if available

## Example Workflows

### Quick Test Run
```bash
# Test with minimal parameters
python run_all_training.py --dry-run --epochs 1
```

### Production Training
```bash
# Full training with optimal parameters
python run_all_training.py --epochs 100 --folds 10 --parallel --max-workers 6
```

### Selective Training
```bash
# Train only neurodegenerative disease models
python run_all_training.py --only als alzheimers parkinsons --epochs 50
```

### Debug Mode
```bash
# Verbose output with retries
python run_all_training.py --verbose --retries 3 --only als
```

## Documentation

For more details, see:
- `IMPLEMENTATION_SUMMARY.md` - Complete implementation documentation
- `TRAINING_ORCHESTRATOR_SUMMARY.md` - Technical architecture
- `TRAINING_USAGE.md` - Training script usage guide
- `run_all_training.py --help` - Full command-line reference

## Testing

Verify functionality with the automated test suite:

```bash
python test_run_all_training.py
# Expected: 4/4 tests passed ✅
```

## Support

For issues or questions:
1. Check the documentation files listed above
2. Review the logs in `logs/` directory
3. Run verification: `python verify_run_all.py`
4. Open an issue on GitHub

---

**🎉 The "Run All" training orchestrator is production-ready and fully operational!**
