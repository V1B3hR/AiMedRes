#!/usr/bin/env python3
"""
Test for Phase5TrainingRunner integration with the main training module
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.training import AlzheimerTrainer, Phase5TrainingRunner
from training.cross_validation import CrossValidationConfig


class TestPhase5TrainingRunner(unittest.TestCase):
    """Test cases for Phase5TrainingRunner integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.trainer = AlzheimerTrainer()
        self.cv_config = CrossValidationConfig(k_folds=3, small_dataset_threshold=30)
        self.phase5_runner = Phase5TrainingRunner(self.trainer, self.cv_config)
    
    def test_phase5_runner_creation(self):
        """Test Phase5TrainingRunner initialization"""
        self.assertIsNotNone(self.phase5_runner)
        self.assertEqual(self.phase5_runner.trainer, self.trainer)
        self.assertEqual(self.phase5_runner.cv_config.k_folds, 3)
        self.assertIsNotNone(self.phase5_runner.cross_validator)
    
    def test_phase5_summary_generation(self):
        """Test Phase 5 summary generation"""
        # Create mock CV results
        mock_cv_results = {
            'dataset_analysis': {
                'n_samples': 100,
                'n_features': 10,
                'n_classes': 2,
                'is_small': False,
                'is_imbalanced': True,
                'class_balance_ratio': 0.05
            },
            'phase_5_1_k_fold': {'mean_scores': {'accuracy': {'test': 0.85}}},
            'phase_5_2_stratified': {'mean_scores': {'accuracy': {'test': 0.87}}},
            'phase_5_3_leave_one_out': None,
            'recommended_strategy': 'stratified_k_fold',
            'summary': {'recommendations': ['Use stratified CV for imbalanced data']}
        }
        
        mock_final_results = {
            'recommended_strategy': 'stratified_k_fold',
            'cv_mean': 0.87,
            'cv_std': 0.05,
            'test_accuracy': 0.88
        }
        
        summary = self.phase5_runner._generate_phase_5_summary(mock_cv_results, mock_final_results)
        
        self.assertIn('phases_completed', summary)
        self.assertIn('dataset_characteristics', summary)
        self.assertIn('phase_5_requirements_met', summary)
        self.assertIn('phase_5_completion_status', summary)
        
        # Check requirements tracking
        requirements = summary['phase_5_requirements_met']
        self.assertTrue(requirements['subphase_5_1_k_fold'])
        self.assertTrue(requirements['subphase_5_2_stratified'])
        self.assertFalse(requirements['subphase_5_3_leave_one_out'])
    
    def test_phase5_with_real_data(self):
        """Test Phase5TrainingRunner with real data"""
        # Load data using trainer
        df = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(df)
        
        # Create a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Test comprehensive cross-validation
        cv_results = self.phase5_runner.cross_validator.comprehensive_cross_validation(model, X, y)
        
        self.assertIsNotNone(cv_results)
        self.assertIn('dataset_analysis', cv_results)
        self.assertIn('phase_5_1_k_fold', cv_results)
        self.assertIn('summary', cv_results)
        
        # Should have at least k-fold results
        self.assertIsNotNone(cv_results['phase_5_1_k_fold'])
    
    def test_different_dataset_scenarios(self):
        """Test Phase 5 with different dataset characteristics"""
        # Small dataset scenario
        X_small, y_small = make_classification(n_samples=20, n_features=5, random_state=42)
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        
        cv_results = self.phase5_runner.cross_validator.comprehensive_cross_validation(model, X_small, y_small)
        
        # Should recommend leave-one-out for small dataset
        self.assertEqual(cv_results['recommended_strategy'], 'leave_one_out')
        self.assertIsNotNone(cv_results['phase_5_3_leave_one_out'])
        
        # Should have executed Phase 5.1 (k-fold) and 5.3 (LOO)
        self.assertIsNotNone(cv_results['phase_5_1_k_fold'])
        self.assertIsNotNone(cv_results['phase_5_3_leave_one_out'])


class TestPhase5Integration(unittest.TestCase):
    """Integration tests for Phase 5 with existing training infrastructure"""
    
    def test_phase5_marks_completion(self):
        """Test that Phase 5 implementation meets all requirements"""
        
        # Test with different dataset types to ensure all subphases can be triggered
        test_cases = [
            # Balanced, adequate size (triggers 5.1 only)
            make_classification(n_samples=100, n_features=8, n_classes=2, random_state=42),
            # Imbalanced (triggers 5.1 and 5.2)
            make_classification(n_samples=100, n_features=8, n_classes=2, weights=[0.95, 0.05], random_state=42),
            # Small dataset (triggers 5.1 and 5.3)
            make_classification(n_samples=20, n_features=5, n_classes=2, random_state=42)
        ]
        
        trainer = AlzheimerTrainer()
        phase5_runner = Phase5TrainingRunner(trainer)
        
        phase_5_1_tested = False
        phase_5_2_tested = False  
        phase_5_3_tested = False
        
        for X, y in test_cases:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            
            cv_results = phase5_runner.cross_validator.comprehensive_cross_validation(model, X, y)
            
            if cv_results['phase_5_1_k_fold']:
                phase_5_1_tested = True
            if cv_results['phase_5_2_stratified']:
                phase_5_2_tested = True
            if cv_results['phase_5_3_leave_one_out']:
                phase_5_3_tested = True
        
        # Verify all three subphases can be executed
        self.assertTrue(phase_5_1_tested, "Phase 5.1 (k-fold CV) should be testable")
        self.assertTrue(phase_5_2_tested, "Phase 5.2 (stratified CV) should be testable") 
        self.assertTrue(phase_5_3_tested, "Phase 5.3 (LOO CV) should be testable")
        
        print(f"✅ Phase 5 Requirements Verification:")
        print(f"   - Subphase 5.1 (k-fold CV): {'✅' if phase_5_1_tested else '❌'}")
        print(f"   - Subphase 5.2 (stratified CV): {'✅' if phase_5_2_tested else '❌'}")
        print(f"   - Subphase 5.3 (LOO CV): {'✅' if phase_5_3_tested else '❌'}")


if __name__ == '__main__':
    unittest.main()