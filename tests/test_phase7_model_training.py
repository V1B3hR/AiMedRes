#!/usr/bin/env python3
"""
Test suite for Phase 7: Model Training & Evaluation Debug Script

This test validates the Phase 7 debugging implementation.
"""

import sys
import os
import tempfile
import unittest
from pathlib import Path
import json
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from debug.phase7_model_training_evaluation import Phase7ModelTrainingEvaluator


class TestPhase7ModelTrainingEvaluator(unittest.TestCase):
    """Test cases for Phase 7 debugging functionality"""
    
    def test_evaluator_initialization(self):
        """Test evaluator initializes correctly"""
        evaluator = Phase7ModelTrainingEvaluator(verbose=False, data_source="synthetic")
        self.assertEqual(evaluator.data_source, "synthetic")
        self.assertEqual(evaluator.verbose, False)
        self.assertEqual(evaluator.results, {})
        self.assertIsInstance(evaluator.trained_models, dict)
        self.assertIsInstance(evaluator.baseline_models, dict)
        self.assertIsInstance(evaluator.evaluation_metrics, list)

    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        evaluator = Phase7ModelTrainingEvaluator()
        data, target_col = evaluator.generate_synthetic_data(n_samples=100)
        
        self.assertEqual(data.shape[0], 100)
        self.assertEqual(target_col, 'target')
        self.assertIn(target_col, data.columns)
        self.assertEqual(data.shape[1], 7)  # 6 features + 1 target
        
        # Check data types
        self.assertIn(str(data['age'].dtype), ['int64', 'float64'])
        self.assertIn(str(data['target'].dtype), ['int64', 'int32'])
        
        # Check value ranges
        self.assertGreaterEqual(data['age'].min(), 18)
        self.assertLessEqual(data['age'].max(), 95)
        self.assertGreaterEqual(data['bmi'].min(), 15)
        self.assertLessEqual(data['bmi'].max(), 50)

    def test_load_synthetic_data(self):
        """Test loading synthetic data"""
        evaluator = Phase7ModelTrainingEvaluator(data_source="synthetic")
        data, target_col = evaluator.load_data()
        
        self.assertEqual(data.shape[0], 1000)  # default n_samples
        self.assertEqual(target_col, 'target')
        self.assertEqual(data.shape[1], 7)  # 6 features + 1 target

    def test_subphase_7_1_cross_validation(self):
        """Test Subphase 7.1: Train with cross-validation"""
        evaluator = Phase7ModelTrainingEvaluator()
        data, target_col = evaluator.generate_synthetic_data(n_samples=200)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        results = evaluator.subphase_7_1_train_with_cross_validation(X, y, n_folds=3)
        
        # Check result structure
        self.assertIn('cv_results', results)
        self.assertIn('n_folds', results)
        self.assertEqual(results['n_folds'], 3)
        
        # Check that models were trained
        cv_results = results['cv_results']
        self.assertGreater(len(cv_results), 0)
        
        # Check that at least one model has valid results
        valid_models = [name for name, res in cv_results.items() if 'error' not in res]
        self.assertGreater(len(valid_models), 0)
        
        # Check result structure for valid models
        for name in valid_models:
            model_results = cv_results[name]
            self.assertIn('train_accuracy', model_results)
            self.assertIn('test_accuracy', model_results)
            self.assertIn('test_precision', model_results)
            self.assertIn('test_recall', model_results)
            self.assertIn('test_f1', model_results)
            self.assertIn('overfitting_gap', model_results)
            self.assertIn('training_time', model_results)
            
            # Check metric structure
            self.assertIn('mean', model_results['test_accuracy'])
            self.assertIn('std', model_results['test_accuracy'])
            self.assertIn('scores', model_results['test_accuracy'])
            
            # Validate metric ranges
            self.assertGreaterEqual(model_results['test_accuracy']['mean'], 0)
            self.assertLessEqual(model_results['test_accuracy']['mean'], 1)
            self.assertGreaterEqual(model_results['test_f1']['mean'], 0)
            self.assertLessEqual(model_results['test_f1']['mean'], 1)
            self.assertGreater(model_results['training_time'], 0)
        
        # Check trained models were stored
        self.assertGreater(len(evaluator.trained_models), 0)

    def test_subphase_7_2_record_metrics(self):
        """Test Subphase 7.2: Record metrics"""
        evaluator = Phase7ModelTrainingEvaluator()
        data, target_col = evaluator.generate_synthetic_data(n_samples=200)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # First train models
        _ = evaluator.subphase_7_1_train_with_cross_validation(X, y, n_folds=3)
        
        # Then record metrics
        results = evaluator.subphase_7_2_record_metrics(X, y, test_size=0.2)
        
        # Check result structure
        self.assertIn('metrics_record', results)
        self.assertIn('train_size', results)
        self.assertIn('test_size', results)
        
        # Check sizes
        self.assertEqual(results['train_size'] + results['test_size'], 200)
        self.assertEqual(results['test_size'], int(200 * 0.2))
        
        # Check metrics for each model
        metrics_record = results['metrics_record']
        self.assertGreater(len(metrics_record), 0)
        
        for name, metrics in metrics_record.items():
            if 'error' not in metrics:
                self.assertIn('training_metrics', metrics)
                self.assertIn('test_metrics', metrics)
                self.assertIn('confusion_matrix', metrics)
                self.assertIn('classification_report', metrics)
                
                # Check training metrics
                train_metrics = metrics['training_metrics']
                self.assertIn('accuracy', train_metrics)
                self.assertIn('precision_macro', train_metrics)
                self.assertIn('recall_macro', train_metrics)
                self.assertIn('f1_macro', train_metrics)
                
                # Check test metrics
                test_metrics = metrics['test_metrics']
                self.assertIn('accuracy', test_metrics)
                self.assertIn('precision_macro', test_metrics)
                self.assertIn('recall_macro', test_metrics)
                self.assertIn('f1_macro', test_metrics)
                
                # Validate ranges
                self.assertGreaterEqual(test_metrics['accuracy'], 0)
                self.assertLessEqual(test_metrics['accuracy'], 1)
                self.assertGreaterEqual(test_metrics['f1_macro'], 0)
                self.assertLessEqual(test_metrics['f1_macro'], 1)
        
        # Check evaluation metrics were stored
        self.assertGreater(len(evaluator.evaluation_metrics), 0)

    def test_subphase_7_3_baseline_comparison(self):
        """Test Subphase 7.3: Baseline comparison"""
        evaluator = Phase7ModelTrainingEvaluator()
        data, target_col = evaluator.generate_synthetic_data(n_samples=200)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Train models first
        _ = evaluator.subphase_7_1_train_with_cross_validation(X, y, n_folds=3)
        _ = evaluator.subphase_7_2_record_metrics(X, y)
        
        # Compare with baseline
        results = evaluator.subphase_7_3_compare_with_baseline(X, y)
        
        # Check result structure
        self.assertIn('baseline_results', results)
        self.assertIn('comparison_summary', results)
        
        # Check baseline results
        baseline_results = results['baseline_results']
        self.assertGreater(len(baseline_results), 0)
        
        for name, metrics in baseline_results.items():
            if 'error' not in metrics:
                self.assertIn('train_accuracy', metrics)
                self.assertIn('test_accuracy', metrics)
                self.assertIn('precision', metrics)
                self.assertIn('recall', metrics)
                self.assertIn('f1', metrics)
                self.assertIn('overfitting_gap', metrics)
                
                # Validate ranges
                self.assertGreaterEqual(metrics['test_accuracy'], 0)
                self.assertLessEqual(metrics['test_accuracy'], 1)
                self.assertGreaterEqual(metrics['f1'], 0)
                self.assertLessEqual(metrics['f1'], 1)
        
        # Check comparison summary
        comparison_summary = results['comparison_summary']
        if comparison_summary:  # Only if evaluation_metrics were populated
            self.assertIn('best_by_accuracy', comparison_summary)
            self.assertIn('best_by_f1', comparison_summary)
            self.assertIn('least_overfit', comparison_summary)
            
            # Check structure of best models
            for key in ['best_by_accuracy', 'best_by_f1', 'least_overfit']:
                best_model = comparison_summary[key]
                self.assertIn('model', best_model)
                self.assertIn('test_accuracy', best_model)
                self.assertIn('test_f1', best_model)
        
        # Check baseline models were stored
        self.assertGreater(len(evaluator.baseline_models), 0)

    def test_full_phase_7_run(self):
        """Test complete Phase 7 run"""
        evaluator = Phase7ModelTrainingEvaluator(verbose=False, data_source="synthetic")
        
        # Run Phase 7
        results = evaluator.run_phase_7()
        
        # Check overall structure
        self.assertIn('timestamp', results)
        self.assertIn('data_source', results)
        self.assertIn('data_shape', results)
        self.assertIn('cv_results', results)
        self.assertIn('metrics_results', results)
        self.assertIn('comparison_results', results)
        self.assertIn('evaluation_metrics', results)
        
        # Check data shape
        self.assertIn('n_samples', results['data_shape'])
        self.assertIn('n_features', results['data_shape'])
        self.assertEqual(results['data_shape']['n_samples'], 1000)
        
        # Check that all subphases completed
        self.assertIsNotNone(results['cv_results'])
        self.assertIsNotNone(results['metrics_results'])
        self.assertIsNotNone(results['comparison_results'])
        
        # Check that results file exists
        results_path = evaluator.output_dir / "phase7_results.json"
        assert results_path.exists()
        
        # Verify JSON is valid
        with open(results_path) as f:
            loaded_results = json.load(f)
            self.assertIsNotNone(loaded_results)
            self.assertIn('timestamp', loaded_results)

    def test_results_saving(self):
        """Test that results are saved correctly"""
        evaluator = Phase7ModelTrainingEvaluator()
        
        # Create test results with various data types
        test_results = {
            'timestamp': '2024-01-01T00:00:00',
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_array': np.array([1, 2, 3]),
            'dict_with_numpy': {
                'value': np.float64(2.71),
                'array': np.array([4, 5, 6])
            },
            'list_with_numpy': [np.int64(1), np.float64(2.0)]
        }
        
        # Save results
        evaluator.save_results(test_results)
        
        # Check file exists
        results_path = evaluator.output_dir / "phase7_results.json"
        assert results_path.exists()
        
        # Load and verify
        with open(results_path) as f:
            loaded = json.load(f)
            self.assertEqual(loaded['timestamp'], '2024-01-01T00:00:00')
            self.assertEqual(loaded['numpy_int'], 42)
            self.assertEqual(loaded['numpy_float'], 3.14)
            self.assertEqual(loaded['numpy_array'], [1, 2, 3])
            self.assertEqual(loaded['dict_with_numpy']['value'], 2.71)
            self.assertEqual(loaded['list_with_numpy'], [1, 2.0])

    def test_categorical_feature_handling(self):
        """Test handling of categorical features"""
        evaluator = Phase7ModelTrainingEvaluator()
        
        # Create data with categorical features
        data = evaluator.generate_synthetic_data(n_samples=100)[0]
        data['category'] = np.random.choice(['A', 'B', 'C'], size=100)
        
        X = data.drop(columns=['target'])
        y = data['target']
        
        # Encode categorical features before passing to subphase
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X['category'] = le.fit_transform(X['category'])
        
        # Train with categorical features (now encoded)
        results = evaluator.subphase_7_1_train_with_cross_validation(X, y, n_folds=3)
        
        # Should handle categorical features without error
        self.assertIn('cv_results', results)
        self.assertGreater(len(results['cv_results']), 0)

    def test_overfitting_detection(self):
        """Test overfitting detection in evaluator"""
        evaluator = Phase7ModelTrainingEvaluator()
        data, target_col = evaluator.generate_synthetic_data(n_samples=100)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Train models and record metrics
        _ = evaluator.subphase_7_1_train_with_cross_validation(X, y, n_folds=3)
        _ = evaluator.subphase_7_2_record_metrics(X, y)
        
        # Check evaluation metrics have overfitting gap
        self.assertGreater(len(evaluator.evaluation_metrics), 0)
        for metric in evaluator.evaluation_metrics:
            self.assertIn('overfitting_gap', metric)
            # Gap should be a reasonable number
            self.assertGreaterEqual(metric['overfitting_gap'], -1)
            self.assertLessEqual(metric['overfitting_gap'], 1)

    def test_model_storage(self):
        """Test that trained models are properly stored"""
        evaluator = Phase7ModelTrainingEvaluator()
        data, target_col = evaluator.generate_synthetic_data(n_samples=200)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Train models
        _ = evaluator.subphase_7_1_train_with_cross_validation(X, y, n_folds=3)
        
        # Check trained models
        self.assertGreater(len(evaluator.trained_models), 0)
        for name, model in evaluator.trained_models.items():
            # Model should have fit method
            self.assertTrue(hasattr(model, 'fit'))
            self.assertTrue(hasattr(model, 'predict'))
            
            # Model should be able to make predictions
            predictions = model.predict(X.values[:10] if hasattr(X, 'values') else X[:10])
            self.assertEqual(len(predictions), 10)

    def test_error_handling_invalid_data_source(self):
        """Test error handling for invalid data source"""
        evaluator = Phase7ModelTrainingEvaluator(data_source="nonexistent_file.csv")
        
        # Should fall back to synthetic data without crashing
        data, target_col = evaluator.load_data()
        self.assertIsNotNone(data)
        self.assertIsNotNone(target_col)
        self.assertEqual(data.shape[0], 1000)  # synthetic data default

    def test_metrics_completeness(self):
        """Test that all required metrics are computed"""
        evaluator = Phase7ModelTrainingEvaluator()
        data, target_col = evaluator.generate_synthetic_data(n_samples=200)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Train and evaluate
        _ = evaluator.subphase_7_1_train_with_cross_validation(X, y, n_folds=3)
        results = evaluator.subphase_7_2_record_metrics(X, y)
        
        # Check all required metrics are present
        for name, metrics in results['metrics_record'].items():
            if 'error' not in metrics:
                # Training metrics
                self.assertIn('accuracy', metrics['training_metrics'])
                self.assertIn('precision_macro', metrics['training_metrics'])
                self.assertIn('precision_weighted', metrics['training_metrics'])
                self.assertIn('recall_macro', metrics['training_metrics'])
                self.assertIn('recall_weighted', metrics['training_metrics'])
                self.assertIn('f1_macro', metrics['training_metrics'])
                self.assertIn('f1_weighted', metrics['training_metrics'])
                
                # Test metrics
                self.assertIn('accuracy', metrics['test_metrics'])
                self.assertIn('precision_macro', metrics['test_metrics'])
                self.assertIn('precision_weighted', metrics['test_metrics'])
                self.assertIn('recall_macro', metrics['test_metrics'])
                self.assertIn('recall_weighted', metrics['test_metrics'])
                self.assertIn('f1_macro', metrics['test_metrics'])
                self.assertIn('f1_weighted', metrics['test_metrics'])
                
                # Additional outputs
                self.assertIn('confusion_matrix', metrics)
                self.assertIn('classification_report', metrics)


if __name__ == '__main__':
    unittest.main()
