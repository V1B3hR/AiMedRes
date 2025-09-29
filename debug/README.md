# Phase 1 Debugging Usage Guide

## Overview

The Phase 1 debugging script implements the first phase of the AiMedRes debugging process as outlined in `debug/debuglist.md`. It performs comprehensive environment and reproducibility checks.

## Running Phase 1 Debugging

### Basic Usage
```bash
python debug/phase1_environment_debug.py
```

### With Verbose Output
```bash
python debug/phase1_environment_debug.py --verbose
```

### View Help
```bash
python debug/phase1_environment_debug.py --help
```

## What Phase 1 Checks

### Subphase 1.1: Environment Setup Verification
- ✅ Python version (requires >=3.10)
- ✅ Platform information
- ⚠️ Core dependencies (SQLAlchemy, psycopg2, pgvector, sentence_transformers, yaml)
- ⚠️ ML dependencies (numpy, pandas, scikit-learn, matplotlib, seaborn, torch, xgboost, kagglehub, scipy, joblib)
- ✅ GPU/CUDA availability check
- ✅ Environment variables from .env file

### Subphase 1.2: Reproducibility Checks ✅ PASSING
- ✅ Random seed implementation in training scripts
- ✅ Reproducibility test with NumPy and Python random
- ✅ Environment documentation generation

### Subphase 1.3: Version Control Verification ✅ PASSING
- ✅ Git repository status
- ✅ .gitignore patterns for ML projects
- ✅ DVC (Data Version Control) setup
- ✅ Output directory structure

## Current Status: 2/3 Subphases Passing

### ✅ What's Working
- Python environment meets requirements
- Random seeding implemented correctly
- Version control properly configured
- Environment variables configured
- Core ML packages (numpy, pandas, scikit-learn) installed

### ⚠️ Missing Dependencies
To achieve full Phase 1 success, install these packages:
```bash
# Core dependencies
pip install SQLAlchemy psycopg2-binary pgvector sentence-transformers

# ML dependencies  
pip install torch matplotlib seaborn xgboost kagglehub

# Or install all at once from requirements
pip install -r requirements-ml.txt
```

## Generated Files

The script generates several documentation files:

1. **`debug/phase1_results.json`** - Detailed results summary
2. **`debug/environment_snapshot.json`** - Complete environment documentation
3. **`.env`** - Environment variables (created from .env.example)

## Next Steps

Once Phase 1 fully passes (3/3 subphases), you can proceed to:
- Phase 2: Data Integrity & Preprocessing Debugging
- Phase 3: Code Sanity & Logical Error Checks
- etc.

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install using pip as shown above
2. **CUDA not available**: This is expected in CPU-only environments
3. **Uncommitted changes**: This is just a warning, not a failure
4. **Environment variables not loading**: Make sure .env file exists and has correct format

### Clean Environment Test
To test in a clean environment:
```bash
# Reset and test
git status  # Check current state
python debug/phase1_environment_debug.py --verbose
```

## Implementation Details

The Phase 1 script is a comprehensive tool that:
- Automatically loads .env files
- Checks package versions against requirements
- Tests reproducibility with actual computations
- Documents the complete environment state
- Provides actionable recommendations

This establishes a solid foundation for the subsequent debugging phases in the AiMedRes debugging methodology.