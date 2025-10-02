# Training Orchestrator Workflow - Quick Start Guide

## 🚀 Running Training Jobs via GitHub Actions

### Basic Steps

1. Navigate to **Actions** tab in GitHub
2. Select **Training Orchestrator** workflow
3. Click **Run workflow** button
4. Configure parameters (all optional)
5. Click green **Run workflow** button to start

### Common Use Cases

#### 📊 Run All Training Jobs (Default)
```
Leave all fields empty or at defaults
```
This will auto-discover and run all training scripts with default settings.

#### 🎯 Run Specific Models with Custom Epochs
```
only: als alzheimers diabetes
epochs: 20
folds: 5
```

#### ⚡ Parallel Execution for Speed
```
parallel: true
max_workers: 4
epochs: 15
```

#### 🔍 Preview Commands (Dry Run)
```
dry_run: true
list_only: false
verbose: true
```

#### 🚫 Exclude Certain Jobs
```
exclude: parkinsons cardiovascular
epochs: 20
```

#### ⚙️ Advanced Custom Arguments
```
epochs: 25
extra_args: --batch-size=64 --learning-rate=1e-3
parallel: true
```

## 📥 Accessing Results

After workflow completes:

1. Go to the workflow run page
2. Scroll to **Artifacts** section at the bottom
3. Download:
   - `training-results` - Full training outputs, logs, and results
   - `training-summary` - High-level summary (retained longer)

## 🔑 Kaggle Setup (Optional)

If training requires Kaggle datasets:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add secrets:
   - `KAGGLE_USERNAME` = your Kaggle username
   - `KAGGLE_KEY` = your Kaggle API key

## 📋 Available Training Jobs

Common job IDs you can use in `only` or `exclude`:

- `als` - ALS (Amyotrophic Lateral Sclerosis)
- `alzheimers` - Alzheimer's Disease
- `parkinsons` - Parkinson's Disease
- `diabetes` - Diabetes Risk Classification
- `brain_mri` - Brain MRI Classification
- `cardiovascular` - Cardiovascular Risk

Use `list_only: true` to see all discovered jobs without running them.

## 🛟 Troubleshooting

**Workflow not visible?**
- Ensure `.github/workflows/training-orchestrator.yml` exists in main branch

**Job fails immediately?**
- Check if required dependencies are in requirements files
- Verify Kaggle secrets are set (if needed)

**Want to see detailed logs?**
- Set `verbose: true` in workflow inputs

## 📚 More Information

- Detailed documentation: [workflows/README.md](workflows/README.md)
- Training usage: [TRAINING_USAGE.md](../TRAINING_USAGE.md)
- Orchestrator source: [run_all_training.py](../run_all_training.py)
