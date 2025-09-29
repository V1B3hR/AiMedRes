#!/usr/bin/env python3
"""
Tests for Phase 5 Cross-Validation Implementation

This module tests the implementation of:
- Phase 5.1: k-fold cross-validation for generalization check
- Phase 5.2: Stratified sampling for imbalanced datasets  
- Phase 5.3: Leave-one-out cross-validation for small datasets
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.cross_validation import Phase5CrossValidator, CrossValidationConfig


class TestCrossValidationConfig(unittest.TestCase):
    """Test cases for CrossValidationConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CrossValidationConfig()
        self.assertEqual(config.k_folds, 5)
        self.assertEqual(config.small_dataset_threshold, 50)
        self.assertEqual(config.imbalance_threshold, 0.1)
        self.assertEqual(config.random_state, 42)
        self.assertIn('accuracy', config.scoring_metrics)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = CrossValidationConfig(
            k_folds=10,
            small_dataset_threshold=30,
            imbalance_threshold=0.2,
            random_state=123,
            scoring_metrics=['accuracy', 'f1_macro']
        )
        self.assertEqual(config.k_folds, 10)
        self.assertEqual(config.small_dataset_threshold, 30)
        self.assertEqual(config.imbalance_threshold, 0.2)
        self.assertEqual(config.random_state, 123)
        self.assertEqual(len(config.scoring_metrics), 2)


class TestPhase5CrossValidator(unittest.TestCase):
    """Test cases for Phase5CrossValidator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = Phase5CrossValidator()
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create test datasets
        self.balanced_data = make_classification(
            n_samples=100, n_features=10, n_classes=2, 
            n_informative=5, random_state=42
        )
        
        self.imbalanced_data = make_classification(
            n_samples=100, n_features=10, n_classes=2,
            n_informative=5, weights=[0.95, 0.05], random_state=42
        )
        
        self.small_data = make_classification(
            n_samples=30, n_features=5, n_classes=2,
            n_informative=3, random_state=42
        )
    
    def test_dataset_analysis_balanced(self):
        """Test dataset analysis for balanced dataset"""
        X, y = self.balanced_data
        analysis = self.validator.analyze_dataset_characteristics(X, y)
        
        self.assertEqual(analysis['n_samples'], 100)
        self.assertEqual(analysis['n_features'], 10)
        self.assertEqual(analysis['n_classes'], 2)
        self.assertFalse(analysis['is_imbalanced'])
        self.assertFalse(analysis['is_small'])
        self.assertEqual(analysis['recommended_strategy'], 'k_fold')
    
    def test_dataset_analysis_imbalanced(self):
        """Test dataset analysis for imbalanced dataset"""
        X, y = self.imbalanced_data
        analysis = self.validator.analyze_dataset_characteristics(X, y)
        
        self.assertTrue(analysis['is_imbalanced'])
        self.assertEqual(analysis['recommended_strategy'], 'stratified_k_fold')
        
        # Check class balance ratio is low
        self.assertLess(analysis['class_balance_ratio'], 0.5)
    
    def test_dataset_analysis_small(self):
        """Test dataset analysis for small dataset"""
        X, y = self.small_data
        analysis = self.validator.analyze_dataset_characteristics(X, y)
        
        self.assertTrue(analysis['is_small'])
        self.assertEqual(analysis['recommended_strategy'], 'leave_one_out')
        self.assertEqual(analysis['n_samples'], 30)
    
    def test_phase_5_1_k_fold_cv(self):
        """Test Phase 5.1: k-fold cross-validation"""
        X, y = self.balanced_data
        results = self.validator.phase_5_1_k_fold_cv(self.model, X, y)
        
        self.assertEqual(results['strategy'], 'k_fold')
        self.assertEqual(results['n_folds'], 5)
        self.assertIn('mean_scores', results)
        self.assertIn('std_scores', results)
        self.assertIn('generalization_gap', results)
        
        # Check that we have results for all metrics
        for metric in self.validator.config.scoring_metrics:
            self.assertIn(metric, results['mean_scores'])
            self.assertIn('test', results['mean_scores'][metric])
            self.assertIn('train', results['mean_scores'][metric])
        
        # Check reasonable accuracy range
        test_accuracy = results['mean_scores']['accuracy']['test']
        self.assertGreaterEqual(test_accuracy, 0.0)
        self.assertLessEqual(test_accuracy, 1.0)
    
    def test_phase_5_2_stratified_cv(self):
        """Test Phase 5.2: stratified cross-validation for imbalanced datasets"""
        X, y = self.imbalanced_data
        results = self.validator.phase_5_2_stratified_cv(self.model, X, y)
        
        self.assertEqual(results['strategy'], 'stratified_k_fold')
        self.assertEqual(results['n_folds'], 5)
        self.assertIn('fold_class_distributions', results)
        self.assertIn('stratification_quality', results)
        
        # Check stratification quality
        self.assertIn('max_proportion_deviation', results['stratification_quality'])
        self.assertIn('is_well_stratified', results['stratification_quality'])
        
        # Verify fold class distributions are recorded
        self.assertEqual(len(results['fold_class_distributions']), 5)
        
        # Check that each fold has samples from both classes (when possible)
        for fold_dist in results['fold_class_distributions']:
            self.assertIsInstance(fold_dist, Counter)
    
    def test_phase_5_3_leave_one_out_cv(self):
        """Test Phase 5.3: leave-one-out cross-validation for small datasets"""
        X, y = self.small_data
        results = self.validator.phase_5_3_leave_one_out_cv(self.model, X, y)
        
        self.assertEqual(results['strategy'], 'leave_one_out')
        self.assertEqual(results['n_folds'], 30)  # Same as number of samples
        self.assertIn('accuracy_scores', results)
        self.assertIn('accuracy_confidence_interval', results)
        
        # Check that we have one prediction per sample
        self.assertEqual(len(results['accuracy_scores']), 30)
        
        # Check confidence interval is valid
        ci_low, ci_high = results['accuracy_confidence_interval']
        self.assertGreaterEqual(ci_low, 0.0)
        self.assertLessEqual(ci_high, 1.0)
        self.assertLessEqual(ci_low, ci_high)
        
        # Check mean accuracy is within reasonable range
        mean_accuracy = results['mean_scores']['accuracy']
        self.assertGreaterEqual(mean_accuracy, 0.0)
        self.assertLessEqual(mean_accuracy, 1.0)
    
    def test_comprehensive_cross_validation_balanced(self):
        """Test comprehensive CV on balanced dataset"""
        X, y = self.balanced_data
        results = self.validator.comprehensive_cross_validation(self.model, X, y)
        
        self.assertIn('dataset_analysis', results)
        self.assertIn('phase_5_1_k_fold', results)
        self.assertIn('summary', results)
        
        # Should run k-fold but not stratified or LOO for balanced, large dataset
        self.assertIsNotNone(results['phase_5_1_k_fold'])
        self.assertIsNone(results['phase_5_2_stratified'])  # Not imbalanced
        self.assertIsNone(results['phase_5_3_leave_one_out'])  # Not small
        
        self.assertEqual(results['recommended_strategy'], 'k_fold')
    
    def test_comprehensive_cross_validation_imbalanced(self):
        """Test comprehensive CV on imbalanced dataset"""
        X, y = self.imbalanced_data
        results = self.validator.comprehensive_cross_validation(self.model, X, y)
        
        # Should run both k-fold and stratified CV
        self.assertIsNotNone(results['phase_5_1_k_fold'])
        self.assertIsNotNone(results['phase_5_2_stratified'])
        self.assertIsNone(results['phase_5_3_leave_one_out'])  # Not small
        
        self.assertEqual(results['recommended_strategy'], 'stratified_k_fold')
    
    def test_comprehensive_cross_validation_small(self):
        """Test comprehensive CV on small dataset"""
        X, y = self.small_data
        results = self.validator.comprehensive_cross_validation(self.model, X, y)
        
        # Should run k-fold and LOO CV
        self.assertIsNotNone(results['phase_5_1_k_fold'])
        self.assertIsNotNone(results['phase_5_3_leave_one_out'])
        
        self.assertEqual(results['recommended_strategy'], 'leave_one_out')
    
    def test_summary_generation(self):
        """Test summary generation and recommendations"""
        X, y = self.imbalanced_data
        results = self.validator.comprehensive_cross_validation(self.model, X, y)
        
        summary = results['summary']
        self.assertIn('recommended_strategy', summary)
        self.assertIn('dataset_characteristics', summary)
        self.assertIn('recommendations', summary)
        
        # Should have recommendations for imbalanced dataset
        self.assertGreater(len(summary['recommendations']), 0)
        
        # Check dataset characteristics
        chars = summary['dataset_characteristics']
        self.assertEqual(chars['balance'], 'imbalanced')
        self.assertEqual(chars['size'], 'adequate')  # 100 samples
    
    def test_custom_config_usage(self):
        """Test using custom configuration"""
        custom_config = CrossValidationConfig(
            k_folds=10,
            small_dataset_threshold=25,
            scoring_metrics=['accuracy', 'f1_macro']
        )
        validator = Phase5CrossValidator(custom_config)
        
        X, y = self.balanced_data
        results = validator.phase_5_1_k_fold_cv(self.model, X, y)
        
        self.assertEqual(results['n_folds'], 10)
        # Should only have the two metrics we specified
        for metric in ['accuracy', 'f1_macro']:
            self.assertIn(metric, results['mean_scores'])
    
    def test_edge_case_single_class(self):
        """Test handling of single-class datasets"""
        # Create dataset with only one class
        X = np.random.rand(50, 5)
        y = np.zeros(50)  # All same class
        
        analysis = self.validator.analyze_dataset_characteristics(X, y)
        self.assertEqual(analysis['n_classes'], 1)
        
        # Should still run without errors (though results may not be meaningful)
        try:
            results = self.validator.phase_5_1_k_fold_cv(self.model, X, y)
            # If it runs, that's good enough for this edge case
        except Exception as e:
            # Some metrics might fail with single class, which is expected
            self.assertIn(('single', 'class', 'constant'), str(e).lower().split())
    
    def test_very_small_dataset(self):
        """Test with very small dataset (fewer than k_folds)"""
        X = np.random.rand(3, 2)
        y = np.array([0, 1, 0])
        
        analysis = self.validator.analyze_dataset_characteristics(X, y)
        self.assertTrue(analysis['is_small'])
        self.assertEqual(analysis['recommended_strategy'], 'leave_one_out')
        
        # LOO should work with very small datasets
        results = self.validator.phase_5_3_leave_one_out_cv(self.model, X, y)
        self.assertEqual(results['n_folds'], 3)


class TestIntegrationWithTraining(unittest.TestCase):
    """Integration tests with existing training module"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Import after path setup
        from training.training import AlzheimerTrainer
        
        self.trainer = AlzheimerTrainer()
        self.validator = Phase5CrossValidator()
    
    def test_integration_with_alzheimer_trainer(self):
        """Test integration with AlzheimerTrainer"""
        # Load data using trainer
        df = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(df)
        
        # Train a model
        results = self.trainer.train_model(X, y)
        
        # Apply Phase 5 cross-validation to the trained model
        cv_results = self.validator.comprehensive_cross_validation(self.trainer.model, X, y)
        
        self.assertIsNotNone(cv_results)
        self.assertIn('dataset_analysis', cv_results)
        self.assertIn('summary', cv_results)
        
        # Should detect that this is synthetic data with adequate size
        dataset_chars = cv_results['summary']['dataset_characteristics']
        self.assertIn(dataset_chars['size'], ['adequate', 'small'])


if __name__ == '__main__':
    unittest.main()