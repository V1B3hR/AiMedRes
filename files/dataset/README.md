# Kaggle Dataset Loader for duetmind_adaptive

This directory contains scripts to load the Alzheimer features dataset from Kaggle using kagglehub.

## Files

- `exact_problem_statement.py` - Exact implementation from the problem statement with proper file_path
- `simple_kaggle_loader.py` - Modern API version with updated kagglehub calls  
- `load_kaggle_dataset.py` - Robust script with automatic file detection and error handling
- `create_test_data.py` - Creates test data for validation purposes
- `problem_statement_corrected.py` - Corrected version of the exact problem statement with syntax fixes
- `problem_statement_modern_api.py` - Modern API version addressing the problem statement

## Prerequisites

1. Install required dependencies:
```bash
pip install kagglehub pandas
```

2. Set up Kaggle API credentials:
   - Create a Kaggle account and generate API credentials
   - Place your `kaggle.json` file in `~/.kaggle/` directory
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

### Exact Problem Statement Implementation

```python
python3 exact_problem_statement.py
```

This script implements exactly what was requested in the problem statement, with the only change being the file_path set to "alzheimer.csv".

### Simple Loader (modern API)

```python
python3 simple_kaggle_loader.py
```

This script loads the "brsdincer/alzheimer-features" dataset using the specified file path.

### Advanced Loader (with error handling)

```python
python3 load_kaggle_dataset.py
```

This script attempts to load the dataset with automatic file detection, trying common file names if the primary file isn't found.

## Dataset Information

- **Dataset**: brsdincer/alzheimer-features
- **Source**: Kaggle
- **Format**: CSV files with Alzheimer's disease features
- **Usage**: Research and analysis of Alzheimer's disease patterns

## File Path Configuration

The `file_path` variable in the scripts specifies which file to load from the dataset. Common values:

- `"alzheimer.csv"` - Main dataset file
- `"data.csv"` - Alternative common name
- `"alzheimer_features.csv"` - Descriptive name
- `""` - Empty string (not supported by kagglehub for pandas datasets)

## Troubleshooting

1. **Network Issues**: Ensure you have internet access to download from Kaggle
2. **Authentication**: Verify your Kaggle API credentials are correctly configured
3. **File Path**: Check the actual file names in the dataset if you get file not found errors
4. **Dependencies**: Make sure kagglehub and pandas are installed

## Integration with duetmind_adaptive

These scripts provide data loading capabilities for the duetmind_adaptive AI framework, specifically for Alzheimer's disease research and analysis applications.