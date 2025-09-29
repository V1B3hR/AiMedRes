# AiMedRes Debugging Usage Guide

## Overview

The AiMedRes debugging process implements a comprehensive, phase-based approach to debugging AI/ML systems. This guide covers both Phase 1 (Environment & Reproducibility) and Phase 2 (Data Integrity & Preprocessing) debugging.

## Phase 1: Environment & Reproducibility Checks ✅ COMPLETE

### Running Phase 1 Debugging

#### Basic Usage
```bash
python debug/phase1_environment_debug.py
```

#### With Verbose Output
```bash
python debug/phase1_environment_debug.py --verbose
```

#### View Help
```bash
python debug/phase1_environment_debug.py --help
```

### What Phase 1 Checks

#### Subphase 1.1: Environment Setup Verification ✅ COMPLETE
- ✅ Python version (requires >=3.10)
- ✅ Platform information
- ✅ Core dependencies (SQLAlchemy, psycopg2, pgvector, sentence_transformers, yaml)
- ✅ ML dependencies (numpy, pandas, scikit-learn, matplotlib, seaborn, torch, xgboost, kagglehub, scipy, joblib)
- ✅ GPU/CUDA availability check
- ✅ Environment variables from .env file

#### Subphase 1.2: Reproducibility Checks ✅ COMPLETE
- ✅ Random seed implementation in training scripts
- ✅ Reproducibility test with NumPy and Python random
- ✅ Environment documentation generation

#### Subphase 1.3: Version Control Verification ✅ COMPLETE
- ✅ Git repository status
- ✅ .gitignore patterns for ML projects
- ✅ DVC (Data Version Control) setup
- ✅ Output directory structure

## Phase 2: Data Integrity & Preprocessing Debugging ✅ COMPLETE

### Running Phase 2 Debugging

#### Basic Usage
```bash
python debug/phase2_data_integrity_debug.py
```

#### With Verbose Output
```bash
python debug/phase2_data_integrity_debug.py --verbose
```

#### Custom Output Directory
```bash
python debug/phase2_data_integrity_debug.py --output-dir custom_debug_output
```

### What Phase 2 Checks

#### Subphase 2.1: Data Integrity Validation ✅ COMPLETE
- ✅ Missing values analysis across all datasets
- ✅ Duplicate detection and reporting
- ✅ Outlier identification using IQR method
- ✅ Data type validation and consistency checks
- ✅ Column-level analysis for data quality metrics

#### Subphase 2.2: Preprocessing Routines Check ✅ COMPLETE
- ✅ Training script analysis for preprocessing patterns
- ✅ Feature scaling implementation verification
- ✅ Data encoding pattern detection
- ✅ Cross-validation and splitting methodology review
- ✅ Preprocessing pipeline testing

#### Subphase 2.3: Data Visualization & Class Balance ✅ COMPLETE
- ✅ Missing data heatmaps generation
- ✅ Feature distribution analysis and visualization
- ✅ Class balance assessment and visualization
- ✅ Correlation matrix generation for numeric features
- ✅ Automated visualization export to debug/visualizations/

### Phase 2 Generated Files

The Phase 2 script generates several output files:

1. **`debug/phase2_results.json`** - Comprehensive Phase 2 results and findings
2. **`debug/visualizations/`** - Directory containing:
   - `correlation_*.png` - Feature correlation matrices
   - `distributions_*.png` - Feature distribution plots
   - `missing_data_*.png` - Missing data pattern heatmaps
   - `class_balance_*.png` - Class balance visualizations

### Current Status: All Phases Complete

#### ✅ Phase 1 Complete
- All environment dependencies installed and verified
- Reproducibility tests passing
- Version control properly configured

#### ✅ Phase 2 Complete  
- Data integrity validation passed
- Preprocessing routines verified
- Visualizations generated successfully

## Next Steps

With Phases 1 and 2 complete, you can proceed to:
- Phase 3: Code Sanity & Logical Error Checks
- Phase 4: Model Architecture Verification
- etc. (as outlined in debug/debuglist.md)

## Troubleshooting

### Phase 1 Common Issues

1. **Missing dependencies**: Install using pip as shown above
2. **CUDA not available**: This is expected in CPU-only environments
3. **Uncommitted changes**: This is just a warning, not a failure
4. **Environment variables not loading**: Make sure .env file exists and has correct format

### Phase 2 Common Issues

1. **No data files found**: Ensure CSV data files exist in the repository
2. **Visualization errors**: Check matplotlib backend configuration for headless environments
3. **Preprocessing analysis incomplete**: Review training scripts for standard preprocessing patterns

### Clean Environment Test
To test in a clean environment:
```bash
# Reset and test both phases
git status  # Check current state
python debug/phase1_environment_debug.py --verbose
python debug/phase2_data_integrity_debug.py --verbose
```

## Implementation Details

The Phase 1 script is a comprehensive tool that:
- Automatically loads .env files
- Checks package versions against requirements
- Tests reproducibility with actual computations
- Documents the complete environment state
- Provides actionable recommendations

This establishes a solid foundation for the subsequent debugging phases in the AiMedRes debugging methodology.