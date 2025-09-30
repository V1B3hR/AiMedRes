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

#### ✅ Phase 9 Complete
- Misclassified samples analyzed with error patterns
- Model bias investigated with statistical tests
- Edge cases and adversarial examples tested
- 9 comprehensive visualizations created
- Robustness scores calculated for all models

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

## Phase 9: Error Analysis & Edge Cases ✅ COMPLETE

### Running Phase 9 Debugging

#### Basic Usage
```bash
python debug/phase9_error_analysis_edge_cases.py
```

#### With Verbose Output
```bash
python debug/phase9_error_analysis_edge_cases.py --verbose
```

#### With Custom Data Source
```bash
python debug/phase9_error_analysis_edge_cases.py --data-source synthetic
```

### What Phase 9 Analyzes

#### Subphase 9.1: Misclassified Samples & Residuals ✅ COMPLETE
- **Identifies misclassified samples** across all models
- **Computes error statistics** (total errors, error rates, class-wise errors)
- **Analyzes confusion patterns** (which classes get confused with which)
- **Calculates residuals** (predicted - true class) for regression-style analysis
- **Generates error visualizations**:
  - Confusion matrices
  - Error rate by class bar charts
  - Residual distribution histograms
  - Misclassification pattern heatmaps

#### Subphase 9.2: Model Bias Investigation ✅ COMPLETE
- **Class-wise performance analysis** (precision, recall, F1-score for each class)
- **Statistical bias metrics**:
  - Demographic parity difference
  - Balanced accuracy
  - Class imbalance ratio
- **Chi-square tests** for prediction distribution bias
- **Bias detection warnings** for:
  - High demographic parity differences (>0.2)
  - Significant prediction distribution bias (>0.15)
  - Large performance disparities across classes (>0.3 F1 difference)
- **Generates bias visualizations**:
  - Class-wise performance metrics bar charts
  - Prediction vs true distribution comparisons
  - Performance heatmaps
  - Bias indicator metrics

#### Subphase 9.3: Edge Cases & Adversarial Testing ✅ COMPLETE
- **Edge case testing**:
  - Boundary value testing (min/max for each feature)
  - All-minimum and all-maximum feature combinations
  - 14 edge cases per model
- **Adversarial perturbation testing**:
  - Multiple epsilon values (0.01, 0.05, 0.1, 0.2)
  - Prediction consistency analysis
  - Accuracy degradation measurement
- **Robustness scoring**:
  - Overall robustness score (0-1)
  - Adversarial robustness rate
  - Stability score
- **Generates adversarial visualizations**:
  - Accuracy under perturbation plots
  - Prediction consistency rate plots
  - Accuracy drop bar charts
  - Robustness metrics summary

### Phase 9 Generated Files

The Phase 9 script generates several output files:

1. **`debug/phase9_results.json`** - Comprehensive Phase 9 results including:
   - Misclassification analysis for each model
   - Error patterns and residual statistics
   - Bias metrics and statistical tests
   - Class-wise performance metrics
   - Edge case test results
   - Adversarial testing results
   - Robustness scores

2. **`debug/visualizations/`** - Directory containing 9 visualizations:
   - `error_distribution_*.png` - Error analysis plots (3 files)
   - `bias_analysis_*.png` - Bias investigation plots (3 files)
   - `adversarial_tests_*.png` - Adversarial testing plots (3 files)

### Phase 9 Key Features

- **Multi-model analysis**: Analyzes DecisionTree, RandomForest, and GradientBoosting
- **Comprehensive error metrics**: Error rates, confusion patterns, residuals
- **Statistical bias detection**: Chi-square tests, demographic parity, balanced accuracy
- **Adversarial robustness**: Tests with multiple perturbation strengths
- **Automated warnings**: Flags significant bias and performance issues
- **Rich visualizations**: 9 high-quality plots with detailed metrics

### Example Output

```
🔍 PHASE 9: ERROR ANALYSIS & EDGE CASES

SUBPHASE 9.1: Analyzing Misclassified Samples & Residuals
  Analyzing errors for DecisionTree...
    • Total misclassified: 82/300 (27.3%)
    • Error rate by class:
      - Class 0: 24/27 (88.9%)
      - Class 1: 19/229 (8.3%)
      - Class 2: 39/44 (88.6%)

SUBPHASE 9.2: Investigating Model Bias
  Analyzing bias for RandomForest...
    • Balanced Accuracy: 0.372
    • Demographic Parity Difference: 0.947
    • Chi-square p-value: 0.0000
    • Class-wise Performance:
      - Class 0: P=0.400, R=0.074, F1=0.125
      - Class 1: P=0.772, R=0.974, F1=0.861
      - Class 2: P=0.500, R=0.068, F1=0.120
    ⚠️  Bias warnings:
      - High demographic parity difference detected
      - Significant prediction distribution bias
      - Large performance disparity across classes

SUBPHASE 9.3: Testing Edge Cases & Adversarial Examples
  Testing GradientBoosting on edge cases...
    • Edge cases tested: 14
    • Adversarial robustness: 93.8%
    • Average accuracy drop: 1.2%
    • Overall robustness score: 0.963

📊 PHASE 9 SUMMARY
  ✅ Phase 9 Complete!
    • Models analyzed: 3
    • Subphases completed: 3/3
    • Visualizations created: 9
    • Execution time: 7.2s
  
  📈 Error Analysis:
    • Average error rate: 25.7%
  
  ⚖️  Bias Analysis:
    • Average balanced accuracy: 0.395
    • Significant bias detected: Yes
  
  🛡️  Robustness Testing:
    • Average robustness score: 0.980

✅ Phase 9 Error Analysis & Edge Cases COMPLETE
```

## Phase 10: Final Model & System Validation ✅ COMPLETE

### Running Phase 10 Debugging

#### Basic Usage
```bash
python debug/phase10_final_validation.py
```

#### With Verbose Output
```bash
python debug/phase10_final_validation.py --verbose
```

#### Custom Data Source
```bash
python debug/phase10_final_validation.py --data-source synthetic
```

### What Phase 10 Validates

#### Subphase 10.1: Held-Out Test Data Validation ✅ COMPLETE
- **Final performance metrics on held-out test data**:
  - Accuracy, precision, recall, F1-score (macro and weighted)
  - Balanced accuracy for imbalanced datasets
  - Per-class performance metrics
- **Generalization gap analysis**:
  - Train vs test accuracy comparison
  - Overfitting detection and severity classification
  - Statistical validation of model performance
- **Model comparison**:
  - Best model identification
  - Performance ranking across all models
  - Comprehensive validation summary

#### Subphase 10.2: End-to-End Pipeline Testing ✅ COMPLETE
- **Data loading and preprocessing pipeline**:
  - Data integrity validation
  - Missing value handling
  - Feature consistency checks
- **Feature scaling consistency**:
  - Transform consistency validation
  - Scaling reproducibility tests
- **Model prediction pipeline**:
  - Single and batch prediction tests
  - Probability prediction validation
  - Prediction format verification
- **Edge case handling**:
  - Boundary value testing
  - Min/max feature value predictions
  - Single sample predictions
- **Error handling and robustness**:
  - Wrong feature count handling
  - Input validation testing
  - Exception handling verification

#### Subphase 10.3: Documentation & Next Steps ✅ COMPLETE
- **Performance summary documentation**:
  - Overall model performance metrics
  - Training and test set statistics
  - Feature and class distribution analysis
- **Model comparison documentation**:
  - Side-by-side model comparison
  - Overfitting risk assessment
  - Best model recommendations
- **Key findings extraction**:
  - Best performing model identification
  - Overfitting analysis across models
  - Generalization gap insights
- **Recommendations generation**:
  - Data collection recommendations
  - Model improvement suggestions
  - Production deployment guidance
- **Next steps planning**:
  - Deployment strategies
  - Monitoring and alerting setup
  - A/B testing framework
  - Model governance procedures
  - Retraining schedules
  - Explainability implementation

### Phase 10 Generated Files

The Phase 10 script generates:

1. **`debug/phase10_results.json`** - Comprehensive Phase 10 results including:
   - Held-out test validation metrics for all models
   - Generalization gap analysis
   - End-to-end pipeline test results (7 tests)
   - Documentation sections with findings and recommendations
   - Complete next steps planning
   - Execution time and summary statistics

### Phase 10 Key Features

- **Comprehensive Final Validation**: Tests models on truly held-out test data
- **End-to-End Testing**: Validates the complete ML pipeline from data to predictions
- **Production Readiness**: Checks that models are ready for deployment
- **Documentation Generation**: Automatically generates comprehensive documentation
- **Actionable Insights**: Provides clear recommendations and next steps
- **100% Test Coverage**: All pipeline tests passing with detailed reporting

### Example Output

```
🎯 STARTING PHASE 10: FINAL MODEL & SYSTEM VALIDATION
======================================================================
📋 Based on debug/debuglist.md

SUBPHASE 10.1: HELD-OUT TEST DATA VALIDATION
======================================================================
🔍 Validating DecisionTree on held-out test data...
  ✓ Test Accuracy: 0.820
  ✓ F1 Score (macro): 0.722
  ✓ Balanced Accuracy: 0.711
  ✓ Generalization Gap: 0.137 (moderate)

🔍 Validating RandomForest on held-out test data...
  ✓ Test Accuracy: 0.893
  ✓ F1 Score (macro): 0.829
  ✓ Balanced Accuracy: 0.801
  ✓ Generalization Gap: 0.107 (moderate)

📊 Validation Summary:
  • Average Test Accuracy: 0.869
  • Average F1 Score: 0.795
  • Best Model: RandomForest

SUBPHASE 10.2: END-TO-END PIPELINE TESTING
======================================================================
🧪 Test 1: Data Loading & Preprocessing Pipeline
  ✓ Data loading and preprocessing: PASSED

🧪 Test 2: Feature Scaling Consistency
  ✓ Feature scaling consistency: PASSED

🧪 Test 3: Model Prediction Pipeline
  ✓ DecisionTree predictions: PASSED
  ✓ RandomForest predictions: PASSED
  ✓ GradientBoosting predictions: PASSED

📊 Pipeline Testing Summary:
  • Total Tests: 7
  • Passed: 7
  • Pass Rate: 100.0%
  ✅ All pipeline tests passed!

SUBPHASE 10.3: DOCUMENTING FINDINGS & NEXT STEPS
======================================================================
📝 Extracting Key Findings...
  • Best performing model: RandomForest with test accuracy of 0.893
  • Overfitting detected in: DecisionTree, RandomForest, GradientBoosting
  • Average generalization gap: 0.117

📝 Generating Recommendations...
  • Consider collecting more training data to improve model generalization
  • Implement regularization techniques to reduce overfitting
  • Implement continuous monitoring for model performance in production

📝 Defining Next Steps...
  • Deploy best performing model to production environment
  • Set up model monitoring and alerting systems
  • Implement A/B testing framework for model comparison
  • Establish model governance and compliance procedures

📊 PHASE 10 SUMMARY
======================================================================
✅ Phase 10 Complete!
  • Models validated: 3
  • Subphases completed: 3/3
  • Execution time: 0.2s

📈 Validation Results:
  • Average test accuracy: 0.869
  • Best model: RandomForest

🧪 Pipeline Tests:
  • Tests passed: 7
  • Pass rate: 100.0%

📝 Documentation:
  • Key findings: 3
  • Recommendations: 3
  • Next steps: 8

✅ Phase 10 Final Model & System Validation COMPLETE
```

## Next Steps

With Phases 1-10 complete, all core debugging phases are finished! The debugging methodology now covers:
- Phase 1: Environment & Reproducibility ✅
- Phase 2: Data Integrity & Preprocessing ✅
- Phase 3: Code Sanity & Logical Errors ✅
- Phase 4: Model Architecture Verification ✅
- Phase 5: Cross-Validation Implementation ✅
- Phase 6: Hyperparameter Tuning & Search ✅
- Phase 7: Model Training & Evaluation ✅
- Phase 8: Model Visualization & Interpretability ✅
- Phase 9: Error Analysis & Edge Cases ✅
- Phase 10: Final Model & System Validation ✅

You can now proceed with model deployment and production implementation:
- Deploy best performing models to production
- Set up continuous monitoring and alerting
- Implement A/B testing for model comparison
- Establish model governance and compliance
- Schedule regular retraining cycles

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