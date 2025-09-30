#!/usr/bin/env python3
"""
Test suite for Phase 10: Final Model & System Validation

This test suite validates the Phase 10 debugging script functionality.
"""

import pytest
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from debug.phase10_final_validation import Phase10FinalValidation


class TestPhase10FinalValidation:
    """Test cases for Phase 10 final validation"""
    
    @pytest.fixture
    def validator(self):
        """Create a Phase10FinalValidation instance"""
        return Phase10FinalValidation(verbose=False, data_source='synthetic')
    
    def test_initialization(self, validator):
        """Test proper initialization of Phase10FinalValidation"""
        assert validator.data_source == 'synthetic'
        assert validator.verbose == False
        assert validator.output_dir.exists()
        assert validator.visualization_dir.exists()
    
    def test_generate_synthetic_data(self, validator):
        """Test synthetic data generation"""
        data, target_col = validator.generate_synthetic_data(n_samples=100)
        
        assert isinstance(data, pd.DataFrame)
        assert target_col == 'target'
        assert len(data) == 100
        assert target_col in data.columns
        assert len(data.columns) == 7  # 6 features + 1 target
        
        # Check data types
        assert data['age'].dtype in [np.float64, float]
        assert data['target'].dtype in [np.int64, int, np.int32]
        
        # Check target has multiple classes
        assert len(data['target'].unique()) > 1
    
    def test_prepare_data_and_models(self, validator):
        """Test data preparation and model training"""
        result = validator.prepare_data_and_models()
        
        assert result == True
        assert validator.X_train is not None
        assert validator.X_test is not None
        assert validator.y_train is not None
        assert validator.y_test is not None
        assert validator.feature_names is not None
        assert validator.class_names is not None
        assert len(validator.trained_models) == 3  # DecisionTree, RandomForest, GradientBoosting
        
        # Verify models are trained
        for name, model_info in validator.trained_models.items():
            assert 'model' in model_info
            assert 'predictions_train' in model_info
            assert 'predictions_test' in model_info
            assert 'accuracy_train' in model_info
            assert 'accuracy_test' in model_info
            assert 0 <= model_info['accuracy_test'] <= 1
    
    def test_subphase_10_1_held_out_validation(self, validator):
        """Test Subphase 10.1: Held-out test data validation"""
        validator.prepare_data_and_models()
        results = validator.subphase_10_1_held_out_validation()
        
        assert 'models_validated' in results
        assert 'validation_metrics' in results
        assert 'generalization_analysis' in results
        assert 'summary' in results
        assert len(results['models_validated']) == 3
        
        # Check validation metrics structure
        for model_name in results['models_validated']:
            assert model_name in results['validation_metrics']
            metrics = results['validation_metrics'][model_name]
            
            assert 'accuracy' in metrics
            assert 'precision_macro' in metrics
            assert 'recall_macro' in metrics
            assert 'f1_macro' in metrics
            assert 'balanced_accuracy' in metrics
            
            # Check metric ranges
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['f1_macro'] <= 1
            
        # Check generalization analysis
        for model_name in results['models_validated']:
            assert model_name in results['generalization_analysis']
            analysis = results['generalization_analysis'][model_name]
            
            assert 'train_accuracy' in analysis
            assert 'test_accuracy' in analysis
            assert 'generalization_gap' in analysis
            assert 'overfitting_detected' in analysis
            assert 'severity' in analysis
            
            assert analysis['severity'] in ['low', 'moderate', 'high']
        
        # Check summary
        assert 'total_models' in results['summary']
        assert 'average_test_accuracy' in results['summary']
        assert 'best_model' in results['summary']
        assert results['summary']['total_models'] == 3
    
    def test_subphase_10_2_end_to_end_pipeline(self, validator):
        """Test Subphase 10.2: End-to-end pipeline tests"""
        validator.prepare_data_and_models()
        results = validator.subphase_10_2_end_to_end_pipeline()
        
        assert 'pipeline_tests' in results
        assert 'test_results' in results
        assert 'summary' in results
        
        # Check test results structure
        assert len(results['test_results']) > 0
        for test in results['test_results']:
            assert 'test' in test
            assert 'status' in test
            assert test['status'] in ['PASSED', 'FAILED']
        
        # Check summary
        summary = results['summary']
        assert 'total_tests' in summary
        assert 'passed' in summary
        assert 'failed' in summary
        assert 'pass_rate' in summary
        assert 'all_tests_passed' in summary
        
        assert summary['total_tests'] == summary['passed'] + summary['failed']
        assert 0 <= summary['pass_rate'] <= 1
        
        # Most tests should pass
        assert summary['passed'] >= summary['total_tests'] * 0.7  # At least 70% pass rate
    
    def test_subphase_10_3_document_findings(self, validator):
        """Test Subphase 10.3: Document findings and next steps"""
        validator.prepare_data_and_models()
        results = validator.subphase_10_3_document_findings()
        
        assert 'documentation_sections' in results
        assert 'key_findings' in results
        assert 'recommendations' in results
        assert 'next_steps' in results
        assert 'model_comparison' in results
        assert 'summary' in results
        
        # Check documentation sections
        assert len(results['documentation_sections']) > 0
        assert 'performance_summary' in results['documentation_sections']
        assert 'model_comparison' in results['documentation_sections']
        assert 'key_findings' in results['documentation_sections']
        
        # Check key findings
        assert len(results['key_findings']) > 0
        for finding in results['key_findings']:
            assert isinstance(finding, str)
            assert len(finding) > 0
        
        # Check recommendations
        assert len(results['recommendations']) > 0
        for rec in results['recommendations']:
            assert isinstance(rec, str)
            assert len(rec) > 0
        
        # Check next steps
        assert len(results['next_steps']) > 0
        for step in results['next_steps']:
            assert isinstance(step, str)
            assert len(step) > 0
        
        # Check model comparison
        assert len(results['model_comparison']) == 3
        for model_name, comparison in results['model_comparison'].items():
            assert 'train_accuracy' in comparison
            assert 'test_accuracy' in comparison
            assert 'overfitting_risk' in comparison
            assert comparison['overfitting_risk'] in ['low', 'moderate', 'high']
        
        # Check summary
        summary = results['summary']
        assert 'total_sections' in summary
        assert 'total_findings' in summary
        assert 'total_recommendations' in summary
        assert 'total_next_steps' in summary
        assert 'documentation_complete' in summary
        assert summary['documentation_complete'] == True
    
    def test_full_phase_10_execution(self, validator):
        """Test complete Phase 10 execution"""
        results = validator.run_phase_10()
        
        # Check top-level structure
        assert 'phase' in results
        assert 'data_source' in results
        assert 'timestamp' in results
        assert 'execution_time_seconds' in results
        assert 'subphase_10_1' in results
        assert 'subphase_10_2' in results
        assert 'subphase_10_3' in results
        assert 'summary' in results
        
        # Check phase name
        assert 'Phase 10' in results['phase']
        
        # Check execution time is reasonable
        assert results['execution_time_seconds'] > 0
        assert results['execution_time_seconds'] < 300  # Less than 5 minutes
        
        # Check all subphases completed
        summary = results['summary']
        assert summary['subphases_completed'] == 3
        assert summary['phase_complete'] == True
        
        # Verify subphase results
        assert len(results['subphase_10_1']['models_validated']) > 0
        assert results['subphase_10_2']['summary']['total_tests'] > 0
        assert len(results['subphase_10_3']['key_findings']) > 0
    
    def test_results_file_generation(self, validator):
        """Test that results are properly saved to file"""
        results = validator.run_phase_10()
        
        # Check that results file was created
        results_file = validator.output_dir / "phase10_results.json"
        assert results_file.exists()
        
        # Verify file can be loaded
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['phase'] == results['phase']
        assert loaded_results['summary']['phase_complete'] == True
    
    def test_model_validation_accuracy(self, validator):
        """Test that model validation produces reasonable accuracy scores"""
        validator.prepare_data_and_models()
        results = validator.subphase_10_1_held_out_validation()
        
        # All models should have reasonable test accuracy (>50% for binary/multi-class)
        for model_name in results['models_validated']:
            metrics = results['validation_metrics'][model_name]
            assert metrics['accuracy'] > 0.5, f"{model_name} has poor accuracy: {metrics['accuracy']}"
            assert metrics['f1_macro'] > 0.3, f"{model_name} has poor F1 score"
    
    def test_pipeline_robustness(self, validator):
        """Test pipeline handles various scenarios"""
        validator.prepare_data_and_models()
        results = validator.subphase_10_2_end_to_end_pipeline()
        
        # Check that critical tests are present
        test_names = [t['test'] for t in results['test_results']]
        assert 'data_loading_preprocessing' in test_names
        assert 'feature_scaling_consistency' in test_names
        assert 'edge_case_handling' in test_names
        
        # Check that model prediction tests exist for each model
        for model_name in validator.trained_models.keys():
            assert f'model_prediction_{model_name}' in test_names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
