# Training Orchestrator Workflow Architecture

## Overview

The Training Orchestrator workflow provides a GitHub Actions-based automation layer for running the unified training orchestrator (`run_all_training.py`) with configurable parameters.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions UI                         │
│  (Manual Workflow Dispatch with 15 Input Parameters)        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Training Orchestrator Workflow Job                 │
│                  (ubuntu-latest runner)                      │
├─────────────────────────────────────────────────────────────┤
│  1. Checkout Repository                                      │
│  2. Setup Python 3.10                                        │
│  3. Install Dependencies (requirements-*.txt)                │
│  4. Configure Kaggle Credentials (if secrets present)        │
│  5. Build Command Arguments (dynamic based on inputs)        │
│  6. Run Training Orchestrator (run_all_training.py)          │
│  7. Upload Artifacts (results + logs + summaries)            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              run_all_training.py Orchestrator                │
├─────────────────────────────────────────────────────────────┤
│  • Auto-discover training scripts (train_*.py)               │
│  • Load built-in job definitions                             │
│  • Merge with config file (if provided)                      │
│  • Filter jobs (--only, --exclude)                           │
│  • Build commands with parameters                            │
│  • Execute jobs (sequential or parallel)                     │
│  • Generate summary reports                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Training Scripts                           │
├─────────────────────────────────────────────────────────────┤
│  • training/train_als.py                                     │
│  • training/train_alzheimers.py                              │
│  • training/train_parkinsons.py                              │
│  • training/train_diabetes.py                                │
│  • training/train_brain_mri.py                               │
│  • files/training/*.py                                       │
│  • scripts/train_*.py                                        │
│  • mlops/pipelines/train_*.py                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Output Artifacts                         │
├─────────────────────────────────────────────────────────────┤
│  results/        - Model outputs, predictions                │
│  logs/           - Execution logs per job                    │
│  summaries/      - JSON training summaries                   │
└─────────────────────────────────────────────────────────────┘
```

## Workflow Input Parameters

### Training Configuration
- **epochs**: Number of training epochs (passed to scripts that support it)
- **folds**: Number of cross-validation folds (passed to scripts that support it)

### Job Selection
- **only**: Whitelist of job IDs to run
- **exclude**: Blacklist of job IDs to skip

### Execution Mode
- **parallel**: Enable parallel job execution
- **max_workers**: Number of parallel workers (default: 4)
- **retries**: Number of retry attempts on failure (default: 0)

### Advanced Options
- **config**: Path to YAML config file with custom job definitions
- **extra_args**: Additional arguments to pass to all training scripts
- **dry_run**: Preview commands without executing
- **list_only**: List discovered jobs without running
- **no_auto_discover**: Disable automatic script discovery
- **discover_roots**: Limit discovery to specific directories
- **allow_partial_success**: Don't fail if some non-critical jobs fail
- **verbose**: Enable detailed logging

## Data Flow

```
User Input (GitHub UI)
    ↓
Workflow Parameters Processing
    ↓
Command Construction (build_args step)
    ↓
Orchestrator Execution
    ↓
┌─────────────────────────────────┐
│  Job Discovery & Filtering       │
│  • Auto-discovery of train_*.py  │
│  • Built-in defaults             │
│  • YAML config (optional)        │
│  • --only/--exclude filters      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Command Building per Job        │
│  • Add --output-dir              │
│  • Add --epochs (if supported)   │
│  • Add --folds (if supported)    │
│  • Add extra args                │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Job Execution                   │
│  • Sequential or Parallel        │
│  • Per-job logging               │
│  • Retry on failure (optional)   │
│  • Status tracking               │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Results & Artifacts             │
│  • Training models               │
│  • Metrics & reports             │
│  • Execution logs                │
│  • JSON summary                  │
└─────────────────────────────────┘
    ↓
Upload to GitHub Artifacts
    • training-results (30 days retention)
    • training-summary (90 days retention)
```

## Key Components

### 1. Workflow File
**Location**: `.github/workflows/training-orchestrator.yml`

**Purpose**: Defines the GitHub Actions workflow with input parameters, job steps, and artifact uploads.

**Trigger**: `workflow_dispatch` (manual trigger from GitHub UI)

### 2. Orchestrator Script
**Location**: `run_all_training.py`

**Purpose**: Unified training orchestrator that discovers, manages, and executes training jobs.

**Features**:
- Auto-discovery of training scripts
- Job filtering and selection
- Parallel execution support
- Command building with parameter injection
- Comprehensive logging and reporting

### 3. Training Scripts
**Location**: Multiple directories
- `training/*.py`
- `files/training/*.py`
- `scripts/train_*.py`
- `mlops/pipelines/train_*.py`

**Naming Convention**: Scripts following `train_*.py` pattern are auto-discovered

**Interface**: Each script should accept:
- `--output-dir`: Directory for outputs
- `--epochs`: Number of training epochs (optional)
- `--folds`: Number of CV folds (optional)
- Custom arguments specific to the model/dataset

### 4. Output Structure
```
results/
├── {job_name}_results/
│   ├── models/
│   ├── metrics/
│   └── predictions/
logs/
├── {job_id}.log
summaries/
└── training_summary_{timestamp}.json
```

## Execution Modes

### Sequential Mode (Default)
```
Job 1 → Job 2 → Job 3 → ... → Job N
```
- Jobs run one after another
- Lower resource usage
- Easier to debug
- Longer total time

### Parallel Mode
```
Job 1 ┐
Job 2 ├─→ max_workers concurrent jobs
Job 3 ┘
...
```
- Multiple jobs run simultaneously
- Higher resource usage
- Faster total time
- Requires sufficient resources

## Error Handling

### Job Failure Modes
1. **Critical Failure** (non-optional job fails)
   - Workflow exits with error
   - Can be overridden with `allow_partial_success`

2. **Optional Failure** (optional job fails)
   - Logged but doesn't fail workflow
   - Continues to next job

3. **Retry Logic** (if retries > 0)
   - Failed jobs are retried up to N times
   - 2-second delay between retries

### Workflow Failure Handling
```
Job Execution
    ↓
  Failed?
    ↓
  Retry?  ────Yes───→ Retry (up to N times)
    ↓ No
Optional Job?
    ↓ No
allow_partial_success?
    ↓ No
Exit with Error
```

## Artifact Management

### training-results Artifact
- **Retention**: 30 days
- **Contents**: 
  - `results/` - All model outputs and predictions
  - `logs/` - Per-job execution logs
  - `summaries/` - Training summaries

### training-summary Artifact
- **Retention**: 90 days (longer for historical tracking)
- **Contents**:
  - `summaries/` - JSON summaries only
  - Lighter weight for long-term storage

## Security Considerations

### Secrets Management
- Kaggle credentials stored as GitHub repository secrets
- Secrets never logged or exposed in outputs
- Conditional execution only if secrets are present

### Repository Access
- Workflow runs with repository-scoped token
- No external network access beyond package installation
- All code executed from checked-out repository

## Extension Points

### Adding New Training Scripts
1. Create script following `train_*.py` pattern
2. Accept standard parameters (`--output-dir`, `--epochs`, `--folds`)
3. Auto-discovered on next workflow run

### Custom Job Configuration
1. Create YAML config file with job definitions
2. Pass via `config` workflow parameter
3. Overrides auto-discovered jobs

### Custom Parameters
1. Use `extra_args` workflow parameter
2. Arguments passed to all jobs
3. Format: `--batch-size=32 --learning-rate=1e-3`

## Monitoring & Debugging

### Viewing Progress
- Check workflow run page in GitHub Actions
- Real-time log streaming during execution
- Per-job logs in uploaded artifacts

### Debugging Failures
1. Enable `verbose: true` for detailed logs
2. Use `dry_run: true` to preview commands
3. Use `list_only: true` to see discovered jobs
4. Check individual job logs in artifacts

### Performance Monitoring
- Summary JSON includes timing information
- Per-job duration tracked
- Total pipeline duration calculated

## Best Practices

### For Users
- Start with `list_only: true` to verify job selection
- Use `dry_run: true` before actual execution
- Enable `verbose: true` for troubleshooting
- Use `parallel: true` only with sufficient resources

### For Developers
- Follow `train_*.py` naming convention
- Accept standard parameters for compatibility
- Log progress and errors clearly
- Create focused, single-purpose training scripts

## Future Enhancements

Potential areas for expansion:
- Scheduled workflow triggers (cron)
- Matrix builds for parameter sweeps
- Email/Slack notifications on completion
- Integration with MLflow tracking
- Automatic model deployment on success
- Resource usage tracking and reporting
