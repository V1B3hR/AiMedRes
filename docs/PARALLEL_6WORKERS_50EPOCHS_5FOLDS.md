# Parallel Training with 6 Workers, 50 Epochs, and 5 Folds

## Command

```bash
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5
```

## Status

✅ **VERIFIED AND WORKING**

## Description

This command orchestrates parallel training of all discovered medical AI models with the following configuration:

- **Parallel Execution**: Enabled with up to 6 concurrent workers
- **Epochs**: 50 epochs for neural network training (applied to compatible scripts)
- **Cross-Validation**: 5-fold cross-validation (applied to compatible scripts)

## What It Does

1. **Discovers Training Jobs**: Automatically finds all training scripts in the repository
2. **Parallel Execution**: Runs up to 6 training jobs simultaneously
3. **Parameter Application**: Applies `--epochs 50` and `--folds 5` to all compatible scripts
4. **Result Organization**: Saves results to organized directories

## Discovered Jobs

The command will discover and train approximately 12 medical AI models, including:

- ALS (Amyotrophic Lateral Sclerosis) prediction
- Alzheimer's disease detection
- Parkinson's disease prediction
- Cardiovascular disease risk assessment
- Diabetes progression modeling
- Brain MRI analysis
- And more...

## Output Structure

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
└── orchestrator.log
└── [job_id]/
    └── run_[timestamp].log

summaries/
└── training_summary_[timestamp].json
```

## Verification

Run the verification script to confirm the command works:

```bash
python verify_parallel_6workers_50epochs_5folds.py
```

## Examples

### Run All Jobs (Full Training)

```bash
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5
```

### Dry Run (Preview Commands)

```bash
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --dry-run
```

### Run Specific Jobs Only

```bash
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --only als alzheimers parkinsons
```

### List Available Jobs

```bash
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --list
```

## Performance Characteristics

### Resource Usage

- **CPU**: Up to 6 cores/threads in use simultaneously
- **Memory**: Depends on models being trained (recommend 16GB+ RAM)
- **Disk**: Results and logs stored in `results/`, `logs/`, and `summaries/` directories

### Execution Time

With 12 jobs and 6 workers:
- **Sequential**: ~12 × T (where T = time per job)
- **Parallel (6 workers)**: ~2 × T (approximately 6× speedup)

Actual speedup depends on:
- Individual job duration
- System resources
- I/O performance

## Advanced Options

### Control Worker Count

```bash
# Use 4 workers instead of 6
python run_all_training.py --parallel --max-workers 4 --epochs 50 --folds 5
```

### Add Retry Logic

```bash
# Retry failed jobs up to 2 times
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --retries 2
```

### Verbose Logging

```bash
# Enable verbose output
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --verbose
```

### Custom Output Directory

```bash
# Change base output directory
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --base-output-dir my_results
```

## Implementation Details

### Code Location

File: `run_all_training.py`

### Parallel Execution Logic

```python
if args.parallel and len(jobs) > 1:
    logger.info("⚠️  Parallel mode enabled. Ensure sufficient resources.")
    max_workers = min(args.max_workers, len(jobs))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job") as pool:
        future_map = {
            pool.submit(run_job, job, ...): job
            for job in jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            result_job = future.result()
            executed_jobs.append(result_job)
```

### Parameter Propagation

The `--epochs` and `--folds` parameters are automatically applied to compatible training scripts:

```python
def build_command(self, python_exec, global_epochs, global_folds, extra_args, base_output_dir):
    cmd = [python_exec, self.script]
    
    if self.supports_epochs:
        epochs = self.args.get("epochs", global_epochs)
        if epochs is not None:
            cmd += ["--epochs", str(epochs)]
    
    if self.supports_folds:
        folds = self.args.get("folds", global_folds)
        if folds is not None:
            cmd += ["--folds", str(folds)]
```

## Testing

### Automated Tests

The functionality is tested in `test_run_all_training.py`:

```bash
python test_run_all_training.py
```

Expected output:
```
Test 1: List all training jobs... ✓
Test 2: Dry run with parameters... ✓
Test 3: Parallel execution mode... ✓
Test 4: Job filtering... ✓
Test 5: Parallel execution with custom parameters... ✓

Test Results: 5/5 tests passed
```

### Manual Verification

```bash
# Quick verification (dry-run mode)
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --dry-run --only als

# Full verification script
python verify_parallel_6workers_50epochs_5folds.py
```

## Troubleshooting

### Issue: "Parallel mode enabled. Ensure sufficient resources."

**Solution**: This is informational. The orchestrator will use at most `min(max_workers, num_jobs)` workers.

### Issue: Some scripts don't accept --epochs or --folds

**Solution**: This is expected. The orchestrator detects which scripts support these parameters and only applies them to compatible scripts.

### Issue: Out of memory errors

**Solution**: Reduce the number of workers:
```bash
python run_all_training.py --parallel --max-workers 2 --epochs 50 --folds 5
```

### Issue: Want to see what will run before executing

**Solution**: Use `--dry-run` mode:
```bash
python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --dry-run
```

## Related Commands

### Sequential Execution (No Parallel)

```bash
python run_all_training.py --epochs 50 --folds 5
```

### Different Worker Counts

```bash
# 2 workers (more conservative)
python run_all_training.py --parallel --max-workers 2 --epochs 50 --folds 5

# 4 workers (balanced)
python run_all_training.py --parallel --max-workers 4 --epochs 50 --folds 5

# 8 workers (aggressive, requires sufficient resources)
python run_all_training.py --parallel --max-workers 8 --epochs 50 --folds 5
```

### Different Training Parameters

```bash
# Quick training (10 epochs, 3 folds)
python run_all_training.py --parallel --max-workers 6 --epochs 10 --folds 3

# Moderate training (30 epochs, 5 folds)
python run_all_training.py --parallel --max-workers 6 --epochs 30 --folds 5

# Extensive training (100 epochs, 10 folds)
python run_all_training.py --parallel --max-workers 6 --epochs 100 --folds 10
```

## GitHub Actions Integration

This command can be executed via GitHub Actions workflow:

### Manual Trigger

1. Go to Actions tab in GitHub
2. Select "Training Orchestrator" workflow
3. Click "Run workflow"
4. Set parameters:
   - `epochs`: 50
   - `folds`: 5
   - `parallel`: true
   - `max_workers`: 6
5. Click "Run workflow"

### Workflow Configuration

File: `.github/workflows/training-orchestrator.yml`

The workflow accepts the same parameters and executes the training in GitHub's CI environment.

## Conclusion

The command `python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5` is:

- ✅ **Implemented**: Code is complete and functional
- ✅ **Tested**: Automated tests verify all functionality
- ✅ **Verified**: Dedicated verification script confirms correct operation
- ✅ **Documented**: This document and inline code documentation
- ✅ **Production-Ready**: Ready for immediate use

**No additional changes required** - the feature is complete and working as designed.
