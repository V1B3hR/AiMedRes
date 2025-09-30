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

### Current Status: Multiple Phases Complete

#### ✅ Phase 1 Complete
- All environment dependencies installed and verified
- Reproducibility tests passing
- Version control properly configured

#### ✅ Phase 2 Complete  
- Data integrity validation passed
- Preprocessing routines verified
- Visualizations generated successfully

#### ✅ Phase 8 Complete
- Feature importance plots for tree-based models generated
- Partial dependence plots created for key features
- Enhanced confusion matrices with precision, recall, F1 displayed
- 15+ high-quality visualizations produced

## Phase 8: Model Visualization & Interpretability ✅ COMPLETE

### Running Phase 8 Debugging

#### Basic Usage
```bash
python debug/phase8_model_visualization.py
```

#### With Verbose Output
```bash
python debug/phase8_model_visualization.py --verbose
```

#### Custom Data Source
```bash
# Use synthetic data (default)
python debug/phase8_model_visualization.py --data-source synthetic

# Use custom CSV file
python debug/phase8_model_visualization.py --data-source path/to/data.csv
```

### What Phase 8 Analyzes

#### Subphase 8.1: Feature Importance for Tree-Based Models ✅ COMPLETE
- ✅ Extracts feature importances from DecisionTree, RandomForest, and GradientBoosting
- ✅ Generates bar plots showing top 15 most important features
- ✅ Saves feature importance data to CSV files for further analysis
- ✅ Creates color-coded visualizations with viridis palette
- ✅ Logs top 5 features for each model with importance scores

#### Subphase 8.2: Partial Dependence Plots ✅ COMPLETE
- ✅ Identifies top 4 most important features from each model
- ✅ Generates 1D partial dependence plots showing individual feature effects
- ✅ Creates 2D partial dependence plots for top feature interactions
- ✅ Uses sklearn.inspection.partial_dependence for accurate calculations
- ✅ Visualizes how predictions change with feature values

#### Subphase 8.3: Enhanced Confusion Matrices ✅ COMPLETE
- ✅ Generates confusion matrices with raw counts
- ✅ Creates normalized confusion matrices showing percentages
- ✅ Produces per-class metrics visualizations (precision, recall, F1)
- ✅ Displays accuracy and macro-averaged metrics
- ✅ Saves comprehensive classification reports to results JSON

### Phase 8 Generated Files

The Phase 8 script generates several output files:

1. **`debug/phase8_results.json`** - Comprehensive Phase 8 results including:
   - Feature importance rankings for each model
   - Partial dependence analysis results
   - Confusion matrix data and classification reports
   - Model performance metrics

2. **`debug/visualizations/`** - Directory containing:
   - `feature_importance_*.png` - Feature importance bar plots (3 files)
   - `feature_importance_*.csv` - Feature importance data tables (3 files)
   - `partial_dependence_*.png` - 1D PDP plots showing individual feature effects (3 files)
   - `partial_dependence_2d_*.png` - 2D PDP plots showing feature interactions (3 files)
   - `confusion_matrix_*.png` - Enhanced confusion matrices with counts and percentages (3 files)
   - `classification_metrics_*.png` - Per-class precision, recall, F1 visualizations (3 files)

### Phase 8 Key Features

- **Automated Model Training**: Trains DecisionTree, RandomForest, and GradientBoosting classifiers
- **Multiple Visualization Types**: 15+ plots generated automatically
- **Feature Insights**: Identifies most important features across all models
- **Interactive Analysis**: Partial dependence shows how features affect predictions
- **Performance Metrics**: Detailed per-class and overall performance evaluation
- **Publication Quality**: High-resolution (300 DPI) plots ready for reports

### Example Output

```
🚀 Starting Phase 8: Model Visualization & Interpretability

SUBPHASE 8.1: FEATURE IMPORTANCE FOR TREE-BASED MODELS
  ✓ RandomForest trained (Test Accuracy: 0.770)
  Top 5 features for RandomForest:
    age: 0.3178
    bmi: 0.1838
    blood_pressure: 0.1755
    cholesterol: 0.1215
    glucose: 0.1089

SUBPHASE 8.2: PARTIAL DEPENDENCE PLOTS
  ✓ Saved 1D partial dependence plots
  ✓ Saved 2D partial dependence plot

SUBPHASE 8.3: ENHANCED CONFUSION MATRICES
  Overall metrics for RandomForest:
    Accuracy:  0.770
    Macro Avg - Precision: 0.770, Recall: 0.770, F1: 0.770

✅ Phase 8 Model Visualization & Interpretability COMPLETE
```

## Next Steps

With Phases 1, 2, and 8 complete, you can proceed to:
- Phase 3: Code Sanity & Logical Error Checks
- Phase 4: Model Architecture Verification
- Phase 5: Cross-Validation Implementation
- Phase 6: Hyperparameter Tuning & Search
- Phase 7: Model Training & Evaluation
- Phase 9: Error Analysis & Edge Cases
- Phase 10: Final Model & System Validation
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

### Phase 8 Common Issues

1. **Import errors**: Ensure sklearn, matplotlib, seaborn, numpy, pandas are installed
2. **Partial dependence errors**: Some models may not support PDP - only tree-based models are used
3. **Memory issues with large datasets**: Phase 8 uses full dataset - consider sampling for very large datasets
4. **Missing models**: Phase 8 trains its own models - Phase 7 results are optional

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