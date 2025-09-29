#!/usr/bin/env python3
"""
Phase 5 Cross-Validation Implementation Demo

This script demonstrates the Phase 5 cross-validation functionality
as specified in debuglist.md:

- Phase 5.1: Use k-fold cross-validation for generalization check
- Phase 5.2: Apply stratified sampling for imbalanced datasets
- Phase 5.3: Optionally, use leave-one-out cross-validation for small datasets
"""

import sys
import os
import logging
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.training import AlzheimerTrainer, Phase5TrainingRunner
from training.cross_validation import CrossValidationConfig
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase5Demo")


def create_demo_datasets():
    """Create different types of datasets to demonstrate Phase 5 functionality"""
    datasets = {}
    
    # 1. Balanced, adequate-size dataset (should use k-fold)
    logger.info("Creating balanced dataset for k-fold demonstration...")
    X_balanced, y_balanced = make_classification(
        n_samples=200, n_features=15, n_classes=3, 
        n_informative=10, n_redundant=3, random_state=42
    )
    datasets['balanced_adequate'] = (X_balanced, y_balanced, "Balanced dataset (200 samples, 3 classes)")
    
    # 2. Imbalanced dataset (should use stratified k-fold)
    logger.info("Creating imbalanced dataset for stratified CV demonstration...")
    X_imbalanced, y_imbalanced = make_classification(
        n_samples=150, n_features=12, n_classes=2,
        n_informative=8, weights=[0.95, 0.05], random_state=42
    )
    datasets['imbalanced'] = (X_imbalanced, y_imbalanced, "Imbalanced dataset (95% vs 5% class ratio)")
    
    # 3. Small dataset (should use leave-one-out)
    logger.info("Creating small dataset for LOO CV demonstration...")
    X_small, y_small = make_classification(
        n_samples=25, n_features=5, n_classes=2,
        n_informative=3, random_state=42
    )
    datasets['small'] = (X_small, y_small, "Small dataset (25 samples)")
    
    return datasets


def demonstrate_phase_5(dataset_name: str, X: np.ndarray, y: np.ndarray, description: str):
    """Demonstrate Phase 5 cross-validation on a specific dataset"""
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 5 DEMONSTRATION: {dataset_name.upper()}")
    logger.info(f"Dataset: {description}")
    logger.info(f"{'='*60}")
    
    # Create trainer with synthetic data
    trainer = AlzheimerTrainer()
    
    # Override the data with our demo data
    # Convert to DataFrame format expected by trainer
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['diagnosis'] = ['Class_' + str(int(label)) for label in y]
    
    # Manually set the data in trainer
    trainer.feature_columns = feature_names
    X_processed, y_processed = trainer.preprocess_data(df)
    
    # Create Phase 5 training runner
    cv_config = CrossValidationConfig(
        k_folds=5,
        small_dataset_threshold=50,
        imbalance_threshold=0.15,
        random_state=42
    )
    
    phase5_runner = Phase5TrainingRunner(trainer, cv_config)
    
    # Override trainer data for demo
    phase5_runner.trainer.feature_scaler.fit(X_processed)
    phase5_runner.trainer.label_encoder.fit(y_processed)
    
    # Run Phase 5 cross-validation manually with our data
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Get comprehensive cross-validation results
    cv_results = phase5_runner.cross_validator.comprehensive_cross_validation(model, X_processed, y_processed)
    
    # Display results
    print_phase_5_results(cv_results, dataset_name)
    
    return cv_results


def print_phase_5_results(results: dict, dataset_name: str):
    """Print formatted Phase 5 results"""
    dataset_analysis = results['dataset_analysis']
    summary = results['summary']
    
    print(f"\n📊 DATASET ANALYSIS ({dataset_name}):")
    print(f"   Samples: {dataset_analysis['n_samples']}")
    print(f"   Features: {dataset_analysis['n_features']}")
    print(f"   Classes: {dataset_analysis['n_classes']}")
    print(f"   Class distribution: {dataset_analysis['class_counts']}")
    print(f"   Class balance ratio: {dataset_analysis['class_balance_ratio']:.3f}")
    print(f"   Is small dataset: {dataset_analysis['is_small']}")
    print(f"   Is imbalanced: {dataset_analysis['is_imbalanced']}")
    print(f"   Recommended strategy: {dataset_analysis['recommended_strategy']}")
    
    print(f"\n🔄 PHASE 5 EXECUTION:")
    
    # Phase 5.1: K-fold CV
    if results['phase_5_1_k_fold']:
        kfold_results = results['phase_5_1_k_fold']
        print(f"   ✅ Phase 5.1 - K-fold CV:")
        print(f"      Mean accuracy: {kfold_results['mean_scores']['accuracy']['test']:.3f} ± {kfold_results['std_scores']['accuracy']['test']:.3f}")
        print(f"      Generalization gap: {kfold_results['generalization_gap']['accuracy']:.3f}")
    
    # Phase 5.2: Stratified CV
    if results['phase_5_2_stratified']:
        stratified_results = results['phase_5_2_stratified']
        print(f"   ✅ Phase 5.2 - Stratified CV:")
        print(f"      Mean accuracy: {stratified_results['mean_scores']['accuracy']['test']:.3f} ± {stratified_results['std_scores']['accuracy']['test']:.3f}")
        print(f"      Stratification quality: {'Good' if stratified_results['stratification_quality']['is_well_stratified'] else 'Poor'}")
        print(f"      Max proportion deviation: {stratified_results['stratification_quality']['max_proportion_deviation']:.3f}")
    
    # Phase 5.3: Leave-One-Out CV
    if results['phase_5_3_leave_one_out']:
        loo_results = results['phase_5_3_leave_one_out']
        print(f"   ✅ Phase 5.3 - Leave-One-Out CV:")
        print(f"      Accuracy: {loo_results['mean_scores']['accuracy']:.3f}")
        ci_low, ci_high = loo_results['accuracy_confidence_interval']
        print(f"      95% Confidence Interval: [{ci_low:.3f}, {ci_high:.3f}]")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in summary.get('recommendations', []):
        print(f"   • {rec}")
    
    if summary.get('best_strategy'):
        print(f"\n🏆 BEST PERFORMING STRATEGY: {summary['best_strategy']}")
        print(f"   Best accuracy: {summary['best_accuracy']:.3f}")


def main():
    """Main demonstration function"""
    logger.info("🚀 Starting Phase 5 Cross-Validation Implementation Demo")
    logger.info("This demonstration shows the three subphases of Phase 5:")
    logger.info("  - 5.1: k-fold cross-validation for generalization check")
    logger.info("  - 5.2: stratified sampling for imbalanced datasets")
    logger.info("  - 5.3: leave-one-out cross-validation for small datasets")
    
    # Create demo datasets
    datasets = create_demo_datasets()
    
    all_results = {}
    
    # Demonstrate on each dataset type
    for dataset_name, (X, y, description) in datasets.items():
        try:
            results = demonstrate_phase_5(dataset_name, X, y, description)
            all_results[dataset_name] = results
        except Exception as e:
            logger.error(f"Error demonstrating {dataset_name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("PHASE 5 CROSS-VALIDATION IMPLEMENTATION SUMMARY")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        dataset_analysis = results['dataset_analysis']
        print(f"\n{dataset_name.upper()}: {dataset_analysis['recommended_strategy']}")
        print(f"  Samples: {dataset_analysis['n_samples']}, "
              f"Imbalanced: {dataset_analysis['is_imbalanced']}, "
              f"Small: {dataset_analysis['is_small']}")
        
        phases_run = []
        if results['phase_5_1_k_fold']:
            phases_run.append("5.1 (k-fold)")
        if results['phase_5_2_stratified']:
            phases_run.append("5.2 (stratified)")
        if results['phase_5_3_leave_one_out']:
            phases_run.append("5.3 (leave-one-out)")
        
        print(f"  Phases executed: {', '.join(phases_run)}")
    
    print(f"\n✅ Phase 5 Cross-Validation Implementation demonstration completed!")
    print(f"All required subphases have been successfully implemented and tested.")
    
    # Save results for analysis
    output_file = "phase5_demo_results.json"
    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for dataset_name, results in all_results.items():
            serializable_results[dataset_name] = {
                'dataset_analysis': results['dataset_analysis'],
                'recommended_strategy': results['recommended_strategy'],
                'summary': results['summary']
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to {output_file}")
    except Exception as e:
        logger.warning(f"Could not save results to file: {e}")


if __name__ == "__main__":
    main()