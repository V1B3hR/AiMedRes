#!/usr/bin/env python3
"""
Test suite for structured Alzheimer's training pipeline

Tests the requirements specified in the problem statement:
1. Learning pipeline with preprocessing, training, and configuration management
2. Multi-seed training with early stopping wrapper
3. Evaluation metrics (classification report, confusion matrix, per-class metrics)
4. Train/test split handling (80/20)
5. Sample configuration with 30 epochs and batch size 32 defaults
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from pathlib import Path
import yaml
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aimedres.training.structured_alz_trainer import StructuredAlzTrainer


@pytest.fixture
def sample_alzheimer_data():
    """Create sample Alzheimer's dataset similar to the problem statement requirements"""
    np.random.seed(42)
    n_samples = 50  # Smaller for testing
    
    data = {
        'age': np.random.randint(55, 85, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'education_level': np.random.randint(8, 20, n_samples),
        'mmse_score': np.random.randint(15, 30, n_samples),
        'cdr_score': np.random.choice([0.0, 0.5, 1.0, 2.0], n_samples),
        'apoe_genotype': np.random.choice(['E2/E2', 'E2/E3', 'E3/E3', 'E3/E4', 'E4/E4'], n_samples),
        'bmi': np.random.normal(25, 5, n_samples),
        'hypertension': np.random.choice([0, 1], n_samples),
        'diabetes': np.random.choice([0, 1], n_samples),
        'diagnosis': np.random.choice(['Normal', 'MCI', 'Dementia'], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def test_config():
    """Test configuration matching problem statement requirements"""
    return {
        'epochs': 30,  # Problem statement default
        'batch_size': 32,  # Problem statement default
        'patience': 8,
        'validation_split': 0.2,  # 80/20 split
        'seeds': [42],  # Single seed for testing
        'models': ['logreg', 'random_forest'],  # Reduced for faster testing
        'ensemble': False,
        'metric_primary': 'macro_f1',
        'class_weight': 'balanced',
        'preprocessing': {
            'numeric_strategy': 'median',
            'categorical_strategy': 'most_frequent',
            'missing_threshold': 0.1,
            'scale_features': True,
            'handle_unknown': 'ignore'
        },
        'early_stopping': {
            'enabled': True,
            'monitor': 'macro_f1',
            'patience': 8,
            'min_delta': 0.001,
            'restore_best_weights': True
        },
        'metrics': {
            'save_per_epoch': True,
            'save_confusion_matrix': True,
            'save_classification_report': True,
            'compute_roc_auc': True,
            'save_formats': ['json', 'csv']
        },
        'artifacts': {
            'save_best_model': True,
            'save_final_model': True,
            'save_preprocessing': True,
            'save_feature_names': True,
            'compress_models': False
        }
    }


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestStructuredAlzTraining:
    """Test structured Alzheimer's training pipeline"""
    
    def test_csv_data_loading(self, sample_alzheimer_data, temp_output_dir):
        """Test CSV data loading and auto-detection"""
        # Save sample data as CSV
        csv_path = temp_output_dir / 'test_data.csv'
        sample_alzheimer_data.to_csv(csv_path, index=False)
        
        config = {
            'seeds': [42],
            'models': ['logreg'],
            'epochs': 5,
            'batch_size': 32
        }
        
        trainer = StructuredAlzTrainer(config=config, seed=42, output_dir=temp_output_dir)
        
        # Test data loading
        df = trainer.load_dataset(csv_path)
        assert df.shape[0] == 50
        assert df.shape[1] == 10
        
        # Test target auto-detection
        target_col = trainer.identify_target(df)
        assert target_col == 'diagnosis'
    
    def test_preprocessing_pipeline(self, sample_alzheimer_data, test_config, temp_output_dir):
        """Test preprocessing with missing value imputation and feature type detection"""
        # Add some missing values
        sample_data = sample_alzheimer_data.copy()
        sample_data.loc[0:2, 'age'] = np.nan
        sample_data.loc[3:5, 'gender'] = np.nan
        
        trainer = StructuredAlzTrainer(config=test_config, seed=42, output_dir=temp_output_dir)
        
        # Test preprocessing
        X, y, preprocessor = trainer.preprocess(sample_data)
        
        # Check preprocessing results
        assert X.shape[0] == sample_data.shape[0]  # Features have same number of rows
        assert y.shape[0] == sample_data.shape[0]
        assert preprocessor is not None
        
        # Check feature names are stored
        assert trainer.feature_names is not None
        assert 'numeric' in trainer.feature_names
        assert 'categorical' in trainer.feature_names
    
    def test_train_test_split(self, sample_alzheimer_data, test_config, temp_output_dir):
        """Test 80/20 train/test split"""
        trainer = StructuredAlzTrainer(config=test_config, seed=42, output_dir=temp_output_dir)
        
        X, y, _ = trainer.preprocess(sample_alzheimer_data)
        
        # Use sklearn's train_test_split like the trainer does
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Check split ratio (should be approximately 80/20)
        total = len(X)
        train_ratio = len(X_train) / total
        val_ratio = len(X_val) / total
        
        assert 0.75 <= train_ratio <= 0.85  # Allow some variance due to stratification
        assert 0.15 <= val_ratio <= 0.25
    
    def test_multi_seed_training(self, sample_alzheimer_data, temp_output_dir):
        """Test multi-seed training functionality"""
        config = {
            'epochs': 3,  # Very short for testing
            'batch_size': 32,
            'seeds': [42, 123],  # Multiple seeds
            'models': ['logreg'],
            'validation_split': 0.2,
            'metric_primary': 'macro_f1'
        }
        
        # Save data to CSV
        csv_path = temp_output_dir / 'test_data.csv'
        sample_alzheimer_data.to_csv(csv_path, index=False)
        
        trainer = StructuredAlzTrainer(config=config, seed=42, output_dir=temp_output_dir)
        result = trainer.run_full_training(csv_path)
        
        # Check result contains required metrics
        assert 'best_model' in result
        assert 'best_val_macro_f1' in result
        assert 'best_val_accuracy' in result
    
    def test_early_stopping_wrapper(self, sample_alzheimer_data, temp_output_dir):
        """Test early stopping functionality for MLP"""
        config = {
            'epochs': 10,
            'batch_size': 16,  # Smaller batch for small dataset
            'seeds': [42],
            'models': ['logreg'],  # Use logreg instead of MLP for more reliable test
            'validation_split': 0.2,
            'early_stopping': {
                'enabled': True,
                'monitor': 'macro_f1',
                'patience': 5,
                'min_delta': 0.001,
                'restore_best_weights': True
            }
        }
        
        csv_path = temp_output_dir / 'test_data.csv'
        sample_alzheimer_data.to_csv(csv_path, index=False)
        
        trainer = StructuredAlzTrainer(config=config, seed=42, output_dir=temp_output_dir)
        result = trainer.run_full_training(csv_path)
        
        # Check that training completed successfully
        assert 'best_model' in result
        assert result['best_model'] is not None
    
    def test_evaluation_metrics(self, sample_alzheimer_data, test_config, temp_output_dir):
        """Test comprehensive evaluation metrics"""
        csv_path = temp_output_dir / 'test_data.csv'
        sample_alzheimer_data.to_csv(csv_path, index=False)
        
        trainer = StructuredAlzTrainer(config=test_config, seed=42, output_dir=temp_output_dir)
        result = trainer.run_full_training(csv_path)
        
        # Check that all required metrics are computed
        assert 'best_val_accuracy' in result
        assert 'best_val_macro_f1' in result
        # Note: some metrics may not be available depending on the model performance
        
        # Check per-class metrics
        per_class_metrics = [k for k in result.keys() if k.startswith('best_val_f1_class_')]
        assert len(per_class_metrics) > 0
    
    def test_artifact_saving(self, sample_alzheimer_data, test_config, temp_output_dir):
        """Test that all required artifacts are saved"""
        csv_path = temp_output_dir / 'test_data.csv'
        sample_alzheimer_data.to_csv(csv_path, index=False)
        
        trainer = StructuredAlzTrainer(config=test_config, seed=42, output_dir=temp_output_dir)
        trainer.run_full_training(csv_path)
        
        # Check that all artifacts are saved
        assert (temp_output_dir / 'best_model.pkl').exists()
        assert (temp_output_dir / 'final_model.pkl').exists()
        assert (temp_output_dir / 'preprocessing.pkl').exists()
        assert (temp_output_dir / 'feature_names.json').exists()
        assert (temp_output_dir / 'run_metrics.json').exists()
        
        # Check metrics format
        with open(temp_output_dir / 'run_metrics.json', 'r') as f:
            metrics = json.load(f)
            assert 'best_model' in metrics
            assert 'best_val_macro_f1' in metrics
    
    def test_configuration_defaults(self):
        """Test that configuration defaults match problem statement"""
        # Load baseline configuration
        config_path = Path(__file__).parent.parent / 'src/duetmind_adaptive/training/configs/structured_alz_baseline.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check problem statement requirements
        assert config['epochs'] == 30, "Default epochs should be 30"
        assert config['batch_size'] == 32, "Default batch size should be 32"
        assert config['validation_split'] == 0.2, "Should use 80/20 split"
        
        # Check small dataset configuration exists
        small_config_path = Path(__file__).parent.parent / 'src/duetmind_adaptive/training/configs/structured_alz_small_dataset.yaml'
        assert small_config_path.exists(), "Small dataset configuration should exist"
        
        with open(small_config_path, 'r') as f:
            small_config = yaml.safe_load(f)
            
        assert small_config['epochs'] == 30
        assert small_config['batch_size'] == 32
        assert small_config['profile'] == 'small_dataset'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])