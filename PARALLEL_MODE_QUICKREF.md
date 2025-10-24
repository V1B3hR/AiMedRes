# Parallel Mode Quick Reference

## Basic Command

```bash
python run_all_training.py --parallel --max-workers 6
```

## Quick Examples

| Command | Description |
|---------|-------------|
| `python run_all_training.py --parallel --max-workers 6` | Run all jobs with 6 workers |
| `python run_all_training.py --parallel --max-workers 6 --dry-run` | Preview parallel execution |
| `python run_all_training.py --parallel --max-workers 6 --only als` | Run only ALS in parallel |
| `python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5` | Parallel with custom params |
| `python run_all_training.py --parallel --max-workers 6 --list` | List all jobs |
| `python run_all_training.py --parallel --max-workers 6 --exclude mlops` | Exclude specific jobs |

## Worker Selection Guide

| Your System | Recommended Command |
|-------------|-------------------|
| 4 CPU cores, 16GB RAM | `--parallel --max-workers 3` |
| 6 CPU cores, 32GB RAM | `--parallel --max-workers 6` |
| 8 CPU cores, 64GB RAM | `--parallel --max-workers 8` |
| Single GPU | `--parallel --max-workers 1` |
| Multiple GPUs | `--parallel --max-workers <num_gpus>` |

## Common Options

| Option | Values | Description |
|--------|--------|-------------|
| `--parallel` | flag | Enable parallel execution |
| `--max-workers` | integer | Number of concurrent jobs |
| `--epochs` | integer | Training epochs |
| `--folds` | integer | Cross-validation folds |
| `--batch` | integer | Batch size |
| `--only` | job ids | Run specific jobs only |
| `--exclude` | job ids | Exclude specific jobs |
| `--dry-run` | flag | Preview without executing |
| `--list` | flag | List available jobs |
| `--retries` | integer | Retry attempts per job |

## Verification

Test that parallel mode works:
```bash
python verify_parallel_mode.py
```

## Full Documentation

See [PARALLEL_MODE_GUIDE.md](PARALLEL_MODE_GUIDE.md) for complete documentation.
