#!/usr/bin/env python3
"""
Test suite for Phase 4: Model Architecture Verification Debug Script

This test validates the Phase 4 debugging implementation.
"""

import sys
import os
import tempfile
import pytest
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from debug.phase4_model_architecture_debug import Phase4ModelArchitectureDebugger


class TestPhase4ModelArchitectureDebugger:
    """Test cases for Phase 4 debugging functionality"""
    
    def test_debugger_initialization(self):
        """Test debugger initializes correctly"""
        debugger = Phase4ModelArchitectureDebugger(verbose=False, data_source="synthetic")
        assert debugger.data_source == "synthetic"
        assert debugger.verbose == False
        assert debugger.results == {}
        assert isinstance(debugger.baseline_models, dict)
        assert isinstance(debugger.complex_models, dict)
        assert isinstance(debugger.performance_log, list)

    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        debugger = Phase4ModelArchitectureDebugger()
        data, target_col = debugger.generate_synthetic_data(n_samples=100)
        
        assert data.shape[0] == 100
        assert target_col == 'target'
        assert target_col in data.columns
        assert data.shape[1] == 7  # 6 features + 1 target
        
        # Check data types
        assert data['age'].dtype in ['int64', 'float64']
        assert data['target'].dtype in ['int64', 'int32']

    def test_load_synthetic_data(self):
        """Test loading synthetic data"""
        debugger = Phase4ModelArchitectureDebugger(data_source="synthetic")
        data, target_col = debugger.load_data()
        
        assert data.shape[0] == 1000  # default n_samples
        assert target_col == 'target'
        assert data.shape[1] == 7  # 6 features + 1 target

    def test_subphase_4_1_architecture_analysis(self):
        """Test Subphase 4.1: Architecture Analysis"""
        debugger = Phase4ModelArchitectureDebugger()
        data, target_col = debugger.generate_synthetic_data(n_samples=200)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        results = debugger.subphase_4_1_architecture_analysis(X, y)
        
        # Check result structure
        assert 'data_characteristics' in results
        assert 'architecture_recommendations' in results
        assert 'overfitting_risk_factors' in results
        
        # Check data characteristics
        data_char = results['data_characteristics']
        assert data_char['n_samples'] == 200
        assert data_char['n_features'] == 6
        assert data_char['samples_per_feature_ratio'] == 200/6
        
        # Check recommendations exist
        assert 'complexity' in results['architecture_recommendations']

    def test_subphase_4_2_baseline_models(self):
        """Test Subphase 4.2: Baseline Models"""
        debugger = Phase4ModelArchitectureDebugger()
        data, target_col = debugger.generate_synthetic_data(n_samples=200)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        results = debugger.subphase_4_2_baseline_models(X, y)
        
        # Check that baseline models were trained
        assert 'logistic_regression' in results
        assert 'decision_tree' in results
        
        # Check result structure for each model
        for model_name, model_results in results.items():
            assert 'train_accuracy' in model_results
            assert 'test_accuracy' in model_results
            assert 'cv_mean' in model_results
            assert 'precision' in model_results
            assert 'recall' in model_results
            assert 'f1' in model_results
            
            # Check accuracy is reasonable (between 0 and 1)
            assert 0 <= model_results['train_accuracy'] <= 1
            assert 0 <= model_results['test_accuracy'] <= 1

    def test_subphase_4_3_progressive_complexity(self):
        """Test Subphase 4.3: Progressive Complexity"""
        debugger = Phase4ModelArchitectureDebugger()
        data, target_col = debugger.generate_synthetic_data(n_samples=200)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        results = debugger.subphase_4_3_progressive_complexity(X, y)
        
        # Check that complex models were trained
        expected_models = [
            'random_forest_simple', 'random_forest_moderate', 'random_forest_complex',
            'svm_linear', 'svm_rbf', 'mlp_simple', 'mlp_complex'
        ]
        
        for model_name in expected_models:
            if model_name in results and 'error' not in results[model_name]:
                model_results = results[model_name]
                assert 'train_accuracy' in model_results
                assert 'test_accuracy' in model_results
                assert 'cv_mean' in model_results
                
                # Check accuracy is reasonable
                assert 0 <= model_results['train_accuracy'] <= 1
                assert 0 <= model_results['test_accuracy'] <= 1

    def test_performance_logging(self):
        """Test that performance is logged correctly"""
        debugger = Phase4ModelArchitectureDebugger()
        data, target_col = debugger.generate_synthetic_data(n_samples=100)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Run baseline models to populate performance log
        debugger.subphase_4_2_baseline_models(X, y)
        
        # Check performance log
        assert len(debugger.performance_log) > 0
        
        for entry in debugger.performance_log:
            assert 'model' in entry
            assert 'complexity' in entry
            assert 'train_accuracy' in entry
            assert 'test_accuracy' in entry
            assert 'cv_accuracy' in entry
            assert 'overfitting_score' in entry

    def test_generate_recommendations(self):
        """Test recommendation generation"""
        debugger = Phase4ModelArchitectureDebugger()
        
        # Mock architecture analysis
        architecture_analysis = {
            'overfitting_risk_factors': ['Class imbalance detected'],
            'data_characteristics': {'samples_per_feature_ratio': 50}
        }
        
        # Mock baseline results
        baseline_results = {
            'logistic_regression': {
                'test_accuracy': 0.75,
                'overfitting_score': 0.05
            }
        }
        
        # Mock complexity results
        complexity_results = {
            'random_forest_simple': {
                'test_accuracy': 0.80,
                'overfitting_score': 0.10
            }
        }
        
        recommendations = debugger.generate_recommendations(
            architecture_analysis, baseline_results, complexity_results
        )
        
        # Check recommendation structure
        assert 'best_overall_model' in recommendations
        assert 'best_baseline_model' in recommendations
        assert 'best_complex_model' in recommendations
        assert 'architecture_advice' in recommendations
        assert 'overfitting_warnings' in recommendations
        
        # Check that best models are identified
        assert recommendations['best_overall_model']['name'] == 'random_forest_simple'
        assert recommendations['best_baseline_model']['name'] == 'logistic_regression'

    def test_full_phase_4_run_synthetic(self):
        """Test complete Phase 4 run with synthetic data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create debugger with temp output directory
            debugger = Phase4ModelArchitectureDebugger(data_source="synthetic")
            debugger.output_dir = Path(temp_dir)
            debugger.visualization_dir = debugger.output_dir / "visualizations"
            debugger.visualization_dir.mkdir(exist_ok=True)
            
            # Run Phase 4
            results = debugger.run_phase_4()
            
            # Check results structure
            assert 'timestamp' in results
            assert 'data_source' in results
            assert 'architecture_analysis' in results
            assert 'baseline_results' in results
            assert 'complexity_results' in results
            assert 'recommendations' in results
            assert 'performance_log' in results
            
            # Check that results file was created
            results_file = debugger.output_dir / "phase4_results.json"
            assert results_file.exists()
            
            # Check that results file is valid JSON
            with open(results_file) as f:
                loaded_results = json.load(f)
                assert loaded_results['data_source'] == 'synthetic'

    def test_categorical_data_handling(self):
        """Test handling of categorical data"""
        debugger = Phase4ModelArchitectureDebugger()
        
        # Create mock data with categorical features
        import pandas as pd
        import numpy as np
        
        data = pd.DataFrame({
            'numeric_feature': np.random.random(100),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Test the preprocessing in run_phase_4
        debugger.data_source = "test"
        debugger.load_data = lambda: (data, 'target')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            debugger.output_dir = Path(temp_dir)
            debugger.visualization_dir = debugger.output_dir / "visualizations"
            debugger.visualization_dir.mkdir(exist_ok=True)
            
            # This should not raise an error with categorical data
            results = debugger.run_phase_4()
            assert 'recommendations' in results


if __name__ == "__main__":
    # Run a quick test if executed directly
    test = TestPhase4ModelArchitectureDebugger()
    test.test_debugger_initialization()
    test.test_synthetic_data_generation()
    test.test_load_synthetic_data()
    print("✅ Basic Phase 4 tests passed!")