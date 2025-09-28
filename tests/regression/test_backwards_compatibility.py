"""
Regression tests to ensure backwards compatibility
"""
import pytest
import pandas as pd
import os
import tempfile

from aimedres.training.training import AlzheimerTrainer
from scripts.data_loaders import MockDataLoader


class TestBackwardsCompatibility:
    """Test that existing functionality still works as expected"""
    
    def test_original_training_workflow(self):
        """Test that the original training workflow still works"""
        # This mimics the old way of creating a trainer
        trainer = AlzheimerTrainer()
        
        # Load data (should create test data by default)
        df = trainer.load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'diagnosis' in df.columns
        
        # Preprocess and train
        X, y = trainer.preprocess_data(df)
        results = trainer.train_model(X, y)
        
        # Validate expected structure
        assert isinstance(results, dict)
        assert 'train_accuracy' in results
        assert 'test_accuracy' in results
        assert trainer.model is not None
    
    def test_model_save_load_format(self):
        """Test that model save/load maintains the expected format"""
        trainer = AlzheimerTrainer()
        df = trainer.load_data()
        X, y = trainer.preprocess_data(df)
        trainer.train_model(X, y)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            trainer.save_model(model_path)
            
            # Load in new trainer
            new_trainer = AlzheimerTrainer()
            new_trainer.load_model(model_path)
            
            # Check that all expected attributes are preserved
            assert new_trainer.model is not None
            assert hasattr(new_trainer, 'label_encoder')
            assert hasattr(new_trainer, 'feature_scaler')
            assert hasattr(new_trainer, 'feature_columns')
            assert len(new_trainer.feature_columns) > 0
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_prediction_api_consistency(self, sample_alzheimer_data):
        """Test that prediction API maintains expected behavior"""
        trainer = AlzheimerTrainer()
        df = trainer.load_data()
        X, y = trainer.preprocess_data(df)
        trainer.train_model(X, y)
        
        # Test with standard features
        features = {
            'age': 72,
            'gender': 'F',
            'education_level': 12,
            'mmse_score': 24,
            'cdr_score': 0.5,
            'apoe_genotype': 'E3/E4'
        }
        
        prediction = trainer.predict(features)
        assert isinstance(prediction, str)
        assert prediction in ['Normal', 'MCI', 'Dementia']
    
    def test_feature_importance_structure(self):
        """Test that feature importance maintains expected structure"""
        trainer = AlzheimerTrainer()
        df = trainer.load_data()
        X, y = trainer.preprocess_data(df)
        results = trainer.train_model(X, y)
        
        feature_importance = results['feature_importance']
        assert isinstance(feature_importance, dict)
        assert len(feature_importance) > 0
        
        # Check that all values are numeric
        for feature, importance in feature_importance.items():
            assert isinstance(importance, (int, float))
            assert 0 <= importance <= 1  # Feature importance should be normalized
        
        # Check that it includes expected features
        expected_features = ['age', 'gender', 'education_level', 'mmse_score', 'cdr_score', 'apoe_genotype']
        for feature in expected_features:
            assert feature in feature_importance


class TestTrainingResultsFormat:
    """Test that training results maintain expected format"""
    
    def test_training_results_keys(self):
        """Test that training results contain all expected keys"""
        trainer = AlzheimerTrainer()
        df = trainer.load_data()
        X, y = trainer.preprocess_data(df)
        results = trainer.train_model(X, y)
        
        expected_keys = ['accuracy', 'train_accuracy', 'test_accuracy', 'feature_importance', 'classification_report']
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_classification_report_structure(self):
        """Test that classification report maintains expected structure"""
        trainer = AlzheimerTrainer()
        df = trainer.load_data()
        X, y = trainer.preprocess_data(df)
        results = trainer.train_model(X, y)
        
        report = results['classification_report']
        assert isinstance(report, dict)
        
        # Should contain class-level metrics
        expected_classes = ['Normal', 'MCI', 'Dementia']
        for cls in expected_classes:
            if cls in report:  # Not all classes might be in small test data
                assert 'precision' in report[cls]
                assert 'recall' in report[cls]
                assert 'f1-score' in report[cls]
        
        # Should contain aggregate metrics
        assert 'accuracy' in report
        assert 'macro avg' in report
        assert 'weighted avg' in report