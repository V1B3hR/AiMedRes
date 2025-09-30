#!/usr/bin/env python3
"""
Test suite for Phase 9: Error Analysis & Edge Cases

This test suite validates the Phase 9 debugging script functionality.
"""

import pytest
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from debug.phase9_error_analysis_edge_cases import Phase9ErrorAnalysis


class TestPhase9ErrorAnalysis:
    """Test cases for Phase 9 error analysis"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a Phase9ErrorAnalysis instance"""
        return Phase9ErrorAnalysis(verbose=False, data_source='synthetic')
    
    def test_initialization(self, analyzer):
        """Test proper initialization of Phase9ErrorAnalysis"""
        assert analyzer.data_source == 'synthetic'
        assert analyzer.verbose == False
        assert analyzer.output_dir.exists()
        assert analyzer.visualization_dir.exists()
    
    def test_generate_synthetic_data(self, analyzer):
        """Test synthetic data generation"""
        data, target_col = analyzer.generate_synthetic_data(n_samples=100)
        
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
    
    def test_prepare_data_and_models(self, analyzer):
        """Test data preparation and model training"""
        result = analyzer.prepare_data_and_models()
        
        assert result == True
        assert analyzer.X_train is not None
        assert analyzer.X_test is not None
        assert analyzer.y_train is not None
        assert analyzer.y_test is not None
        assert analyzer.feature_names is not None
        assert analyzer.class_names is not None
        assert len(analyzer.trained_models) == 3  # DecisionTree, RandomForest, GradientBoosting
        
        # Verify models are trained
        for name, model_info in analyzer.trained_models.items():
            assert 'model' in model_info
            assert 'predictions_train' in model_info
            assert 'predictions_test' in model_info
            assert 'accuracy_train' in model_info
            assert 'accuracy_test' in model_info
            assert 0 <= model_info['accuracy_test'] <= 1
    
    def test_subphase_9_1_misclassified_analysis(self, analyzer):
        """Test Subphase 9.1: Misclassified samples analysis"""
        analyzer.prepare_data_and_models()
        results = analyzer.subphase_9_1_misclassified_analysis()
        
        assert 'models_analyzed' in results
        assert 'misclassification_analysis' in results
        assert 'error_patterns' in results
        assert 'visualizations' in results
        assert len(results['models_analyzed']) == 3
        
        # Check misclassification analysis structure
        for model_name in results['models_analyzed']:
            assert model_name in results['misclassification_analysis']
            analysis = results['misclassification_analysis'][model_name]
            
            assert 'total_misclassified' in analysis
            assert 'error_rate' in analysis
            assert 'error_by_class' in analysis
            assert 'confusion_patterns' in analysis
            assert 'residual_statistics' in analysis
            
            # Check error rate is valid
            assert 0 <= analysis['error_rate'] <= 1
            
            # Check residual statistics
            res_stats = analysis['residual_statistics']
            assert 'mean' in res_stats
            assert 'std' in res_stats
            assert 'min' in res_stats
            assert 'max' in res_stats
        
        # Verify visualizations were created
        for viz_file in results['visualizations']:
            viz_path = analyzer.visualization_dir / viz_file
            assert viz_path.exists(), f"Visualization file not created: {viz_file}"
    
    def test_subphase_9_2_bias_investigation(self, analyzer):
        """Test Subphase 9.2: Bias investigation"""
        analyzer.prepare_data_and_models()
        results = analyzer.subphase_9_2_bias_investigation()
        
        assert 'models_analyzed' in results
        assert 'bias_metrics' in results
        assert 'class_performance' in results
        assert 'prediction_distribution' in results
        assert 'statistical_tests' in results
        assert 'visualizations' in results
        assert len(results['models_analyzed']) == 3
        
        # Check bias metrics structure
        for model_name in results['models_analyzed']:
            assert model_name in results['bias_metrics']
            bias = results['bias_metrics'][model_name]
            
            assert 'demographic_parity_difference' in bias
            assert 'balanced_accuracy' in bias
            assert 'class_imbalance_ratio' in bias
            
            # Check balanced accuracy is valid
            assert 0 <= bias['balanced_accuracy'] <= 1
            
            # Check class performance
            assert model_name in results['class_performance']
            class_perf = results['class_performance'][model_name]
            
            for class_label, metrics in class_perf.items():
                assert 'precision' in metrics
                assert 'recall' in metrics
                assert 'f1_score' in metrics
                assert 'support' in metrics
                
                # Check all metrics are valid
                assert 0 <= metrics['precision'] <= 1
                assert 0 <= metrics['recall'] <= 1
                assert 0 <= metrics['f1_score'] <= 1
                assert metrics['support'] > 0
            
            # Check statistical tests
            assert model_name in results['statistical_tests']
            stats = results['statistical_tests'][model_name]
            
            assert 'chi_square_stat' in stats
            assert 'chi_square_pvalue' in stats
            assert 'prediction_bias_significant' in stats
            assert isinstance(stats['prediction_bias_significant'], bool)
        
        # Verify visualizations were created
        for viz_file in results['visualizations']:
            viz_path = analyzer.visualization_dir / viz_file
            assert viz_path.exists(), f"Visualization file not created: {viz_file}"
    
    def test_subphase_9_3_edge_cases_adversarial(self, analyzer):
        """Test Subphase 9.3: Edge cases and adversarial testing"""
        analyzer.prepare_data_and_models()
        results = analyzer.subphase_9_3_edge_cases_adversarial()
        
        assert 'models_analyzed' in results
        assert 'edge_case_tests' in results
        assert 'adversarial_tests' in results
        assert 'robustness_scores' in results
        assert 'visualizations' in results
        assert len(results['models_analyzed']) == 3
        
        # Check edge case tests
        for model_name in results['models_analyzed']:
            assert model_name in results['edge_case_tests']
            edge_tests = results['edge_case_tests'][model_name]
            
            assert 'n_edge_cases' in edge_tests
            assert 'edge_prediction_distribution' in edge_tests
            assert 'unique_predictions' in edge_tests
            assert edge_tests['n_edge_cases'] > 0
            
            # Check adversarial tests
            assert model_name in results['adversarial_tests']
            adv_tests = results['adversarial_tests'][model_name]
            
            assert 'perturbation_tests' in adv_tests
            assert 'robustness_rate' in adv_tests
            assert 'avg_accuracy_drop' in adv_tests
            assert 'stable_predictions' in adv_tests
            
            # Check perturbation tests structure
            assert len(adv_tests['perturbation_tests']) > 0
            for perturb_test in adv_tests['perturbation_tests']:
                assert 'epsilon' in perturb_test
                assert 'accuracy' in perturb_test
                assert 'consistency_rate' in perturb_test
                assert 'accuracy_drop' in perturb_test
                
                # Check values are valid
                assert 0 <= perturb_test['accuracy'] <= 1
                assert 0 <= perturb_test['consistency_rate'] <= 1
            
            # Check robustness scores
            assert model_name in results['robustness_scores']
            robustness = results['robustness_scores'][model_name]
            
            assert 'overall_score' in robustness
            assert 'adversarial_robustness' in robustness
            assert 'stability_score' in robustness
            
            # Check scores are valid
            assert 0 <= robustness['overall_score'] <= 1
            assert 0 <= robustness['adversarial_robustness'] <= 1
            assert 0 <= robustness['stability_score'] <= 1
        
        # Verify visualizations were created
        for viz_file in results['visualizations']:
            viz_path = analyzer.visualization_dir / viz_file
            assert viz_path.exists(), f"Visualization file not created: {viz_file}"
    
    def test_generate_edge_cases(self, analyzer):
        """Test edge case generation"""
        analyzer.prepare_data_and_models()
        edge_cases = analyzer._generate_edge_cases()
        
        assert isinstance(edge_cases, pd.DataFrame)
        assert len(edge_cases) > 0
        
        # Should have cases for min/max of each feature plus all-min and all-max
        expected_min_cases = len(analyzer.feature_names) * 2 + 2
        assert len(edge_cases) == expected_min_cases
        
        # Check that edge cases have correct features
        assert set(edge_cases.columns) == set(analyzer.feature_names)
    
    def test_adversarial_robustness_testing(self, analyzer):
        """Test adversarial robustness testing"""
        analyzer.prepare_data_and_models()
        
        model_name = list(analyzer.trained_models.keys())[0]
        model_info = analyzer.trained_models[model_name]
        model = model_info['model']
        
        results = analyzer._test_adversarial_robustness(model, model_info)
        
        assert 'perturbation_tests' in results
        assert 'robustness_rate' in results
        assert 'avg_accuracy_drop' in results
        assert 'stable_predictions' in results
        
        # Check that multiple epsilon values were tested
        assert len(results['perturbation_tests']) >= 3
        
        # Check metrics are valid
        assert 0 <= results['robustness_rate'] <= 1
        assert results['avg_accuracy_drop'] >= 0
    
    def test_run_phase_9_complete(self, analyzer):
        """Test complete Phase 9 execution"""
        results = analyzer.run_phase_9()
        
        assert 'phase' in results
        assert results['phase'] == 9
        assert 'timestamp' in results
        assert 'data_source' in results
        assert 'n_models' in results
        assert 'models_trained' in results
        assert 'subphases' in results
        assert 'execution_time_seconds' in results
        assert 'summary' in results
        
        # Check all subphases are present
        assert '9.1_misclassified_analysis' in results['subphases']
        assert '9.2_bias_investigation' in results['subphases']
        assert '9.3_edge_cases_adversarial' in results['subphases']
        
        # Check summary contains key metrics
        summary = results['summary']
        assert 'total_models_analyzed' in summary
        assert 'subphases_completed' in summary
        assert 'total_visualizations' in summary
        assert summary['subphases_completed'] == 3
        
        # Check results file was created
        results_path = analyzer.output_dir / "phase9_results.json"
        assert results_path.exists()
        
        # Verify results file is valid JSON
        with open(results_path, 'r') as f:
            saved_results = json.load(f)
            assert saved_results['phase'] == 9
    
    def test_results_persistence(self, analyzer):
        """Test that results are properly saved"""
        results = analyzer.run_phase_9()
        
        # Check results file
        results_path = analyzer.output_dir / "phase9_results.json"
        assert results_path.exists()
        
        # Load and validate
        with open(results_path, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['phase'] == 9
        assert 'subphases' in loaded_results
        assert len(loaded_results['subphases']) == 3
    
    def test_visualization_creation(self, analyzer):
        """Test that all expected visualizations are created"""
        results = analyzer.run_phase_9()
        
        # Count expected visualizations (3 models × 3 types)
        expected_visualizations = 9
        
        # Check that visualization files exist
        viz_files = list(analyzer.visualization_dir.glob('*.png'))
        phase9_viz_files = [
            f for f in viz_files 
            if 'error_distribution' in f.name or 
               'bias_analysis' in f.name or 
               'adversarial_tests' in f.name
        ]
        
        assert len(phase9_viz_files) >= expected_visualizations, \
            f"Expected at least {expected_visualizations} visualizations, found {len(phase9_viz_files)}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])
