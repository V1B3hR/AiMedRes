# Parallel Training Mode

## Overview

The AiMedRes training orchestrator supports running multiple training jobs in parallel to significantly reduce total training time. This feature is ideal for multi-core systems and when training multiple models simultaneously.

## Command Syntax

```bash
python run_all_training.py --parallel --max-workers <N>
```

Where `<N>` is the number of concurrent workers (training jobs that can run simultaneously).

## Examples

### Basic Parallel Execution

Run all training jobs in parallel with 6 workers:
```bash
python run_all_training.py --parallel --max-workers 6
```

### Parallel with Specific Jobs

Run only ALS and Alzheimer's models in parallel:
```bash
python run_all_training.py --parallel --max-workers 6 --only als alzheimers
```

### Parallel with Custom Parameters

Run with 50 epochs and 5-fold cross-validation:
```bash
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5
```

### Preview Parallel Execution (Dry-Run)

See what commands will be executed without actually running them:
```bash
python run_all_training.py --parallel --max-workers 6 --dry-run
```

### Parallel with Batch Size

Run with custom batch size:
```bash
python run_all_training.py --parallel --max-workers 6 --batch 128
```

## Performance Considerations

### Choosing the Number of Workers

The optimal number of workers depends on:

1. **CPU Cores**: Generally, set workers to the number of CPU cores available
   ```bash
   # For 6-core CPU
   python run_all_training.py --parallel --max-workers 6
   ```

2. **Memory**: Each worker runs a complete training job, ensure sufficient RAM
   - Recommended: 8GB RAM minimum per worker for medical AI models
   - For systems with 48GB RAM: `--max-workers 6` is safe

3. **GPU Resources**: 
   - Single GPU: `--max-workers 1` (avoid GPU conflicts)
   - Multiple GPUs: Set workers equal to number of GPUs
   - CPU-only training: Set workers equal to CPU cores

### Resource Usage Guidelines

| System Resources | Recommended Workers | Example Command |
|-----------------|-------------------|-----------------|
| 4 cores, 16GB RAM | 2-3 workers | `--parallel --max-workers 3` |
| 6 cores, 32GB RAM | 4-6 workers | `--parallel --max-workers 6` |
| 8 cores, 64GB RAM | 6-8 workers | `--parallel --max-workers 8` |
| 16 cores, 128GB RAM | 8-12 workers | `--parallel --max-workers 12` |

## How It Works

### Parallel Execution Flow

1. **Job Discovery**: Orchestrator discovers all available training scripts
2. **Job Filtering**: Apply `--only` or `--exclude` filters if specified
3. **Worker Pool**: Create ThreadPoolExecutor with specified max workers
4. **Concurrent Execution**: Submit all jobs to the pool
5. **Progress Tracking**: Monitor job completion and collect results
6. **Summary Generation**: Generate comprehensive summary when all jobs complete

### Implementation Details

- **Technology**: Python's `concurrent.futures.ThreadPoolExecutor`
- **Actual Workers**: `min(max_workers, num_jobs)` - never creates more workers than jobs
- **Thread Safety**: Each job runs in isolated subprocess with dedicated log file
- **Interrupt Handling**: Graceful shutdown on Ctrl+C, completing current jobs

## Monitoring Parallel Execution

### Real-Time Monitoring

During parallel execution, you'll see:
```
⚠️  Parallel mode enabled. Ensure sufficient resources.
[als] 🚀 Starting attempt 1/1
[alzheimers] 🚀 Starting attempt 1/1
[parkinsons] 🚀 Starting attempt 1/1
```

### Log Files

Each job writes to its own log file:
```
logs/
├── orchestrator.log          # Main orchestrator log
├── als/
│   └── run_20251024_100329.log
├── alzheimers/
│   └── run_20251024_100329.log
└── parkinsons/
    └── run_20251024_100329.log
```

### Summary Report

After completion, find the summary at:
```
summaries/training_summary_YYYYMMDD_HHMMSS.json
```

## Error Handling

### Job Failures in Parallel Mode

- Failed jobs don't stop other running jobs
- All jobs continue to completion
- Summary shows which jobs succeeded/failed
- Exit code reflects overall status

### Retry Logic

Add retries for transient failures:
```bash
python run_all_training.py --parallel --max-workers 6 --retries 2
```

### Interrupt Handling

- **First Ctrl+C**: Graceful shutdown, completes current jobs
- **Second Ctrl+C**: Immediate exit

## Advanced Usage

### Excluding Jobs from Parallel Execution

```bash
# Run all except MLOps pipelines
python run_all_training.py --parallel --max-workers 6 --exclude train_baseline_models train_model
```

### Custom Discovery Patterns

```bash
# Discover only specific patterns
python run_all_training.py --parallel --max-workers 6 --include-pattern "train_*cardiovascular*.py"
```

### Combining with Extra Arguments

```bash
# Pass additional arguments to all jobs
python run_all_training.py --parallel --max-workers 6 \
  --extra-arg --verbose \
  --extra-arg --early-stopping
```

## Comparison: Sequential vs Parallel

### Sequential Execution (Default)

```bash
python run_all_training.py
```

- Jobs run one at a time
- Total time = sum of all job times
- Example: 7 jobs × 30 min each = 210 minutes (3.5 hours)

### Parallel Execution

```bash
python run_all_training.py --parallel --max-workers 6
```

- Up to 6 jobs run simultaneously
- Total time ≈ longest job time × ceil(num_jobs / workers)
- Example: 7 jobs ÷ 6 workers × 30 min ≈ 60 minutes (1 hour)

**Speedup**: ~3.5x faster with 6 workers!

## Troubleshooting

### Issue: "Out of Memory" Errors

**Solution**: Reduce the number of workers
```bash
python run_all_training.py --parallel --max-workers 2
```

### Issue: GPU Conflicts

**Solution**: Use sequential execution for GPU-intensive models
```bash
python run_all_training.py  # Sequential, no --parallel
```

Or use one worker per GPU:
```bash
# For 2 GPUs
python run_all_training.py --parallel --max-workers 2
```

### Issue: Jobs Starting but Not Completing

**Solution**: Check individual job logs in `logs/<job_id>/` directory

### Issue: Slower Than Expected

**Causes**:
- I/O bottleneck (disk too slow)
- CPU oversubscription (too many workers)
- Memory swapping

**Solution**: Reduce workers or check system resources
```bash
# Monitor system resources
htop  # or top on Linux
```

## Best Practices

1. **Start with Dry-Run**: Always test with `--dry-run` first
   ```bash
   python run_all_training.py --parallel --max-workers 6 --dry-run
   ```

2. **Monitor Resources**: Watch CPU, RAM, and disk I/O during execution

3. **Use Appropriate Workers**: Don't exceed your CPU core count

4. **Consider Job Duration**: Longer jobs benefit more from parallelization

5. **Check Logs**: Review individual job logs for detailed debugging

6. **Use --only for Testing**: Test parallel mode with a subset first
   ```bash
   python run_all_training.py --parallel --max-workers 3 --only als alzheimers --epochs 5
   ```

## See Also

- [RUN_ALL_MODELS_GUIDE.md](RUN_ALL_MODELS_GUIDE.md) - Complete training guide
- [QUICKSTART_TRAINING.md](QUICKSTART_TRAINING.md) - Quick start guide
- [examples/advanced/parallel_6workers_50epochs_5folds.py](examples/advanced/parallel_6workers_50epochs_5folds.py) - Example script

## Verification

To verify parallel mode is working correctly:

```bash
python verify_parallel_mode.py
```

This runs a comprehensive test suite to validate all parallel mode functionality.
