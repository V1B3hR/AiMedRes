# GitHub Actions Workflows

## Training Orchestrator Workflow

The `training-orchestrator.yml` workflow allows you to run the unified training orchestrator (`run_all_training.py`) through GitHub Actions with configurable parameters.

### How to Use

1. Go to the **Actions** tab in the GitHub repository
2. Select the **Training Orchestrator** workflow from the left sidebar
3. Click the **Run workflow** button
4. Configure the parameters as needed (all parameters are optional)
5. Click **Run workflow** to start the training

### Available Parameters

#### Training Configuration

- **epochs**: Global default number of epochs for training (if supported by training scripts)
  - Type: Number
  - Example: `20`

- **folds**: Global default number of folds for cross-validation (if supported)
  - Type: Number
  - Example: `5`

#### Job Selection

- **only**: Run only specific job IDs (space-separated)
  - Type: String
  - Example: `als alzheimers diabetes`
  - Runs only the specified training jobs

- **exclude**: Exclude specific job IDs from execution (space-separated)
  - Type: String
  - Example: `parkinsons cardiovascular`
  - Excludes these jobs from the training run

#### Execution Mode

- **parallel**: Enable parallel execution of training jobs
  - Type: Boolean
  - Default: `false`
  - When enabled, multiple training jobs run simultaneously

- **max_workers**: Number of workers for parallel mode
  - Type: Number
  - Default: `4`
  - Only applies when `parallel` is enabled

- **retries**: Number of retry attempts per job if it fails
  - Type: Number
  - Default: `0`

#### Advanced Options

- **extra_args**: Additional arguments to append to all training jobs
  - Type: String
  - Example: `--batch-size=32 --learning-rate=3e-4`
  - These arguments are passed to each training script

- **config**: Path to YAML config file with job definitions
  - Type: String
  - Example: `configs/training_jobs.yaml`
  - Overrides default job configurations

- **dry_run**: Show commands without executing (dry run mode)
  - Type: Boolean
  - Default: `false`
  - Useful for testing and debugging

- **list_only**: List selected jobs and exit without running training
  - Type: Boolean
  - Default: `false`
  - Shows what jobs would be executed

- **no_auto_discover**: Disable auto-discovery of training scripts
  - Type: Boolean
  - Default: `false`
  - Uses only explicitly configured jobs

- **discover_roots**: Root directories to search for training scripts
  - Type: String (space-separated)
  - Example: `training files/training`
  - Restricts auto-discovery to specified directories

- **allow_partial_success**: Exit successfully even if some non-optional jobs fail
  - Type: Boolean
  - Default: `false`
  - Useful for running multiple independent training jobs

- **verbose**: Enable verbose console logging
  - Type: Boolean
  - Default: `false`
  - Provides more detailed output

### Examples

#### Example 1: Run All Training Jobs with 20 Epochs
```yaml
epochs: 20
folds: 5
```

#### Example 2: Run Specific Training Jobs in Parallel
```yaml
only: als alzheimers diabetes
parallel: true
max_workers: 3
epochs: 15
```

#### Example 3: Dry Run to Preview Commands
```yaml
dry_run: true
list_only: false
verbose: true
```

#### Example 4: Run with Custom Arguments
```yaml
epochs: 25
extra_args: --batch-size=64 --learning-rate=1e-3
parallel: true
```

#### Example 5: Exclude Specific Jobs
```yaml
exclude: parkinsons cardiovascular
epochs: 20
allow_partial_success: true
```

### Output Artifacts

The workflow automatically uploads the following artifacts after execution:

1. **training-results** (retained for 30 days)
   - Contains: `results/`, `logs/`, and `summaries/` directories
   - All training outputs, logs, and performance metrics

2. **training-summary** (retained for 90 days)
   - Contains: `summaries/` directory only
   - High-level summary of training execution and results

You can download these artifacts from the workflow run page in the Actions tab.

### Requirements

The workflow automatically:
- Checks out the repository
- Sets up Python 3.10
- Installs dependencies from `requirements-ml.txt`, `requirements.txt`, and `requirements-dev.txt`
- Configures Kaggle credentials (if `KAGGLE_USERNAME` and `KAGGLE_KEY` secrets are set)

### Kaggle Integration

If your training scripts require Kaggle datasets, add the following secrets to your repository:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add new repository secrets:
   - `KAGGLE_USERNAME`: Your Kaggle username
   - `KAGGLE_KEY`: Your Kaggle API key

The workflow will automatically configure Kaggle authentication when these secrets are present.

### Troubleshooting

- **Jobs not discovered**: Ensure training scripts follow the `train_*.py` naming convention, or use the `discover_roots` parameter
- **Missing dependencies**: Check that all required dependencies are in the requirements files
- **Kaggle authentication failed**: Verify that `KAGGLE_USERNAME` and `KAGGLE_KEY` secrets are correctly set
- **Job failures**: Use `verbose: true` and check the workflow logs for detailed error messages

### Related Documentation

- Main training orchestrator documentation: [`run_all_training.py`](../../run_all_training.py)
- Training usage guide: [`TRAINING_USAGE.md`](../../TRAINING_USAGE.md)
- Repository README: [`README.md`](../../README.md)
