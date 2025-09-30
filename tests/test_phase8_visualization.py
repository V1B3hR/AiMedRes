#!/usr/bin/env python3
"""
Test suite for Phase 8: Model Visualization & Interpretability

This test suite validates the Phase 8 debugging script functionality.
"""

import pytest
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from debug.phase8_model_visualization import Phase8ModelVisualization


class TestPhase8Visualization:
    """Test cases for Phase 8 model visualization"""
    
    @pytest.fixture
    def visualizer(self):
        """Create a Phase8ModelVisualization instance"""
        return Phase8ModelVisualization(verbose=False, data_source='synthetic')
    
    def test_initialization(self, visualizer):
        """Test proper initialization of Phase8ModelVisualization"""
        assert visualizer.data_source == 'synthetic'
        assert visualizer.verbose == False
        assert visualizer.output_dir.exists()
        assert visualizer.visualization_dir.exists()
    
    def test_generate_synthetic_data(self, visualizer):
        """Test synthetic data generation"""
        data, target_col = visualizer.generate_synthetic_data(n_samples=100)
        
        assert isinstance(data, pd.DataFrame)
        assert target_col == 'target'
        assert len(data) == 100
        assert target_col in data.columns
        assert len(data.columns) == 7  # 6 features + 1 target
        
        # Check data types
        assert data['age'].dtype in [np.float64, float]
        assert data['target'].dtype in [np.int64, int]
    
    def test_prepare_data_and_models(self, visualizer):
        """Test data preparation and model training"""
        result = visualizer.prepare_data_and_models()
        
        assert result == True
        assert visualizer.X_train is not None
        assert visualizer.X_test is not None
        assert visualizer.y_train is not None
        assert visualizer.y_test is not None
        assert visualizer.feature_names is not None
        assert len(visualizer.trained_models) == 3  # DecisionTree, RandomForest, GradientBoosting
        
        # Verify models are trained
        for name, model_info in visualizer.trained_models.items():
            assert 'model' in model_info
            assert 'predictions' in model_info
            assert 'accuracy' in model_info
            assert 0 <= model_info['accuracy'] <= 1
    
    def test_feature_importance_subphase(self, visualizer):
        """Test Subphase 8.1: Feature importance generation"""
        visualizer.prepare_data_and_models()
        results = visualizer.subphase_8_1_feature_importance()
        
        assert 'models_analyzed' in results
        assert 'feature_importance_plots' in results
        assert 'top_features_by_model' in results
        assert len(results['models_analyzed']) == 3
        
        # Verify CSV files are created
        for model_name in results['models_analyzed']:
            csv_path = visualizer.visualization_dir / f"feature_importance_{model_name}.csv"
            assert csv_path.exists()
            
            # Verify CSV content
            df = pd.read_csv(csv_path)
            assert 'feature' in df.columns
            assert 'importance' in df.columns
            assert len(df) == len(visualizer.feature_names)
        
        # Verify plot files exist
        for plot_path in results['feature_importance_plots']:
            assert Path(plot_path).exists()
    
    def test_partial_dependence_subphase(self, visualizer):
        """Test Subphase 8.2: Partial dependence plots"""
        visualizer.prepare_data_and_models()
        results = visualizer.subphase_8_2_partial_dependence()
        
        assert 'models_analyzed' in results
        assert 'pdp_plots_generated' in results
        assert 'features_analyzed' in results
        assert len(results['models_analyzed']) == 3
        assert len(results['pdp_plots_generated']) == 6  # 3 1D + 3 2D plots
        
        # Verify plot files exist
        for plot_path in results['pdp_plots_generated']:
            assert Path(plot_path).exists()
    
    def test_confusion_matrices_subphase(self, visualizer):
        """Test Subphase 8.3: Confusion matrices"""
        visualizer.prepare_data_and_models()
        results = visualizer.subphase_8_3_confusion_matrices()
        
        assert 'models_analyzed' in results
        assert 'confusion_matrices' in results
        assert 'classification_reports' in results
        assert len(results['models_analyzed']) == 3
        
        # Verify confusion matrix data
        for model_name in results['models_analyzed']:
            cm_data = results['confusion_matrices'][model_name]
            assert 'matrix' in cm_data
            assert 'normalized' in cm_data
            assert 'plot_path' in cm_data
            assert Path(cm_data['plot_path']).exists()
            
            # Verify classification report
            report = results['classification_reports'][model_name]
            assert 'accuracy' in report
            assert 'macro_avg' in report
            assert 'weighted_avg' in report
            assert 'per_class' in report
    
    def test_full_phase_8_execution(self, visualizer):
        """Test complete Phase 8 execution"""
        results = visualizer.run_phase_8()
        
        # Verify structure
        assert 'timestamp' in results
        assert 'data_source' in results
        assert 'data_shape' in results
        assert 'models_analyzed' in results
        assert 'feature_importance' in results
        assert 'partial_dependence' in results
        assert 'confusion_matrices' in results
        
        # Verify data shape info
        assert results['data_shape']['n_samples'] == 1000
        assert results['data_shape']['n_features'] == 6
        assert results['data_shape']['n_classes'] == 2
        
        # Verify results file is created
        results_path = visualizer.output_dir / 'phase8_results.json'
        assert results_path.exists()
        
        # Verify JSON is valid
        with open(results_path, 'r') as f:
            saved_results = json.load(f)
        assert saved_results['data_source'] == 'synthetic'
    
    def test_visualization_file_count(self, visualizer):
        """Test that all expected visualization files are created"""
        visualizer.run_phase_8()
        
        viz_dir = visualizer.visualization_dir
        
        # Count different types of files
        feature_imp_plots = list(viz_dir.glob('feature_importance_*.png'))
        feature_imp_csvs = list(viz_dir.glob('feature_importance_*.csv'))
        pdp_1d_plots = list(viz_dir.glob('partial_dependence_[!2]*.png'))
        pdp_2d_plots = list(viz_dir.glob('partial_dependence_2d_*.png'))
        cm_plots = list(viz_dir.glob('confusion_matrix_*.png'))
        metrics_plots = list(viz_dir.glob('classification_metrics_*.png'))
        
        # Verify counts (at least, as there may be files from previous runs)
        assert len(feature_imp_plots) >= 3
        assert len(feature_imp_csvs) >= 3
        assert len(pdp_1d_plots) >= 3
        assert len(pdp_2d_plots) >= 3
        assert len(cm_plots) >= 3
        assert len(metrics_plots) >= 3


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
