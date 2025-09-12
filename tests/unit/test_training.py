"""
Unit tests for training module
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
import tempfile
import os

from training import AlzheimerTrainer, TrainingConfig
from data_loaders import MockDataLoader
from neuralnet import AliveLoopNode, ResourceRoom


class TestTrainingConfig:
    """Unit tests for TrainingConfig"""
    
    def test_default_initialization(self):
        """Test default configuration"""
        config = TrainingConfig()
        assert config.random_seed == 42
        assert config.test_size == 0.3
        assert config.n_estimators == 100
        assert config.max_depth == 5
        assert config.min_samples_split == 2
    
    def test_custom_initialization(self):
        """Test custom configuration"""
        config = TrainingConfig(
            random_seed=123,
            test_size=0.2,
            n_estimators=50,
            max_depth=10
        )
        assert config.random_seed == 123
        assert config.test_size == 0.2
        assert config.n_estimators == 50
        assert config.max_depth == 10
    
    def test_set_random_seeds(self):
        """Test random seed setting"""
        config = TrainingConfig(random_seed=123)
        config.set_random_seeds()
        
        # Test that numpy random state is set
        np1 = np.random.randint(0, 1000)
        config.set_random_seeds()
        np2 = np.random.randint(0, 1000)
        assert np1 == np2  # Should be deterministic


class TestAlzheimerTrainer:
    """Unit tests for AlzheimerTrainer"""
    
    def test_initialization_default(self):
        """Test default trainer initialization"""
        trainer = AlzheimerTrainer()
        assert trainer.data_loader is None
        assert trainer.config is not None
        assert trainer.model is None
        assert trainer.target_column == 'diagnosis'
        assert isinstance(trainer.feature_columns, list)
        assert len(trainer.feature_columns) == 0
    
    def test_initialization_with_loader(self, sample_alzheimer_data):
        """Test trainer initialization with data loader"""
        mock_loader = MockDataLoader(mock_data=sample_alzheimer_data)
        config = TrainingConfig(random_seed=123)
        
        trainer = AlzheimerTrainer(data_loader=mock_loader, config=config)
        assert trainer.data_loader == mock_loader
        assert trainer.config.random_seed == 123
    
    def test_load_data_with_loader(self, sample_alzheimer_data):
        """Test loading data with provided loader"""
        mock_loader = MockDataLoader(mock_data=sample_alzheimer_data)
        trainer = AlzheimerTrainer(data_loader=mock_loader)
        
        df = trainer.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_alzheimer_data)
        assert mock_loader.load_called
    
    def test_load_data_without_loader(self):
        """Test loading data without loader (creates test data)"""
        trainer = AlzheimerTrainer()
        df = trainer.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'diagnosis' in df.columns
        assert 'age' in df.columns
    
    def test_preprocess_data(self, sample_alzheimer_data):
        """Test data preprocessing"""
        trainer = AlzheimerTrainer()
        X, y = trainer.preprocess_data(sample_alzheimer_data)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
        assert len(X) == len(sample_alzheimer_data)
        assert X.shape[1] > 0  # Should have features
        assert len(trainer.feature_columns) > 0
    
    def test_train_model(self, sample_alzheimer_data):
        """Test model training"""
        trainer = AlzheimerTrainer()
        X, y = trainer.preprocess_data(sample_alzheimer_data)
        results = trainer.train_model(X, y)
        
        # Check that model was created
        assert trainer.model is not None
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'train_accuracy' in results
        assert 'test_accuracy' in results
        assert 'feature_importance' in results
        assert 'classification_report' in results
        
        # Check accuracy types
        assert isinstance(results['train_accuracy'], float)
        assert isinstance(results['test_accuracy'], float)
        assert 0 <= results['train_accuracy'] <= 1
        assert 0 <= results['test_accuracy'] <= 1
    
    def test_train_model_deterministic(self, sample_alzheimer_data):
        """Test that training is deterministic"""
        config = TrainingConfig(random_seed=42)
        
        # Train first model
        trainer1 = AlzheimerTrainer(config=config)
        X, y = trainer1.preprocess_data(sample_alzheimer_data)
        results1 = trainer1.train_model(X, y)
        
        # Train second model with same config
        trainer2 = AlzheimerTrainer(config=config)
        X, y = trainer2.preprocess_data(sample_alzheimer_data)
        results2 = trainer2.train_model(X, y)
        
        # Results should be identical
        assert results1['train_accuracy'] == results2['train_accuracy']
        assert results1['test_accuracy'] == results2['test_accuracy']
    
    def test_save_and_load_model(self, sample_alzheimer_data):
        """Test model saving and loading"""
        trainer = AlzheimerTrainer()
        X, y = trainer.preprocess_data(sample_alzheimer_data)
        trainer.train_model(X, y)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            trainer.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Create new trainer and load model
            new_trainer = AlzheimerTrainer()
            new_trainer.load_model(model_path)
            
            assert new_trainer.model is not None
            assert len(new_trainer.feature_columns) == len(trainer.feature_columns)
            assert new_trainer.feature_columns == trainer.feature_columns
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_predict_with_trained_model(self, sample_alzheimer_data):
        """Test making predictions with trained model"""
        trainer = AlzheimerTrainer()
        X, y = trainer.preprocess_data(sample_alzheimer_data)
        trainer.train_model(X, y)
        
        # Test prediction
        test_features = {
            'age': 72,
            'gender': 'F',
            'education_level': 12,
            'mmse_score': 24,
            'cdr_score': 0.5,
            'apoe_genotype': 'E3/E4'
        }
        
        prediction = trainer.predict(test_features)
        assert isinstance(prediction, str)
        assert prediction in ['Normal', 'MCI', 'Dementia']
    
    def test_predict_without_trained_model(self):
        """Test that prediction fails without trained model"""
        trainer = AlzheimerTrainer()
        test_features = {'age': 72, 'gender': 'F'}
        
        with pytest.raises(ValueError, match="No trained model available"):
            trainer.predict(test_features)
    
    def test_predict_with_missing_features(self, sample_alzheimer_data):
        """Test prediction with missing features"""
        trainer = AlzheimerTrainer()
        X, y = trainer.preprocess_data(sample_alzheimer_data)
        trainer.train_model(X, y)
        
        # Test with incomplete features
        incomplete_features = {'age': 72}
        
        with pytest.raises(ValueError, match="Missing required features"):
            trainer.predict(incomplete_features)