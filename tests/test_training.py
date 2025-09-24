"""
Tests for the training functionality
"""

import unittest
import sys
import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.training import AlzheimerTrainer, TrainingIntegratedAgent, run_training_simulation
from neuralnet import AliveLoopNode, ResourceRoom


class TestAlzheimerTrainer(unittest.TestCase):
    """Test cases for the AlzheimerTrainer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.trainer = AlzheimerTrainer()
        
    def test_create_trainer(self):
        """Test trainer initialization"""
        self.assertIsNotNone(self.trainer)
        self.assertIsNone(self.trainer.model)
        self.assertEqual(self.trainer.target_column, 'diagnosis')
        
    def test_load_data_default(self):
        """Test loading default test data"""
        df = self.trainer.load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('diagnosis', df.columns)
        self.assertIn('age', df.columns)
        
    def test_load_data_from_file(self):
        """Test loading data from a CSV file"""
        # Create temporary CSV file
        test_data = {
            'age': [65, 72],
            'gender': ['M', 'F'],
            'education_level': [16, 12],
            'mmse_score': [28, 24],
            'cdr_score': [0.0, 0.5],
            'apoe_genotype': ['E3/E3', 'E3/E4'],
            'diagnosis': ['Normal', 'MCI']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_test = pd.DataFrame(test_data)
            df_test.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Use the new data loader API
            from data_loaders import CSVDataLoader
            csv_loader = CSVDataLoader(temp_path)
            trainer = AlzheimerTrainer(data_loader=csv_loader)
            df = trainer.load_data()
            self.assertEqual(len(df), 2)
            self.assertIn('diagnosis', df.columns)
        finally:
            os.unlink(temp_path)
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        df = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(df)
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(X), len(y))
        self.assertGreater(X.shape[1], 0)  # Should have features
        
    def test_train_model(self):
        """Test model training"""
        df = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(df)
        results = self.trainer.train_model(X, y)
        
        self.assertIsNotNone(self.trainer.model)
        self.assertIn('train_accuracy', results)
        self.assertIn('test_accuracy', results)
        self.assertIn('feature_importance', results)
        self.assertIsInstance(results['train_accuracy'], float)
        self.assertIsInstance(results['test_accuracy'], float)
        
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        # Train a model first
        df = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(df)
        self.trainer.train_model(X, y)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            self.trainer.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Create new trainer and load model
            new_trainer = AlzheimerTrainer()
            new_trainer.load_model(model_path)
            
            self.assertIsNotNone(new_trainer.model)
            self.assertEqual(len(new_trainer.feature_columns), len(self.trainer.feature_columns))
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_predict(self):
        """Test making predictions"""
        # Train a model first
        df = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(df)
        self.trainer.train_model(X, y)
        
        # Test prediction
        test_features = {
            'age': 72,
            'gender': 'F',
            'education_level': 12,
            'mmse_score': 24,
            'cdr_score': 0.5,
            'apoe_genotype': 'E3/E4'
        }
        
        prediction = self.trainer.predict(test_features)
        self.assertIsInstance(prediction, str)
        self.assertIn(prediction, ['Normal', 'MCI', 'Dementia'])
    
    def test_predict_without_model(self):
        """Test prediction fails without trained model"""
        test_features = {'age': 72, 'gender': 'F'}
        
        with self.assertRaises(ValueError):
            self.trainer.predict(test_features)


class TestTrainingIntegratedAgent(unittest.TestCase):
    """Test cases for the TrainingIntegratedAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.trainer = AlzheimerTrainer()
        self.resource_room = ResourceRoom()
        self.alive_node = AliveLoopNode((0, 0), (0.5, 0), 15.0, node_id=1)
        self.agent = TrainingIntegratedAgent(
            "TestAgent", 
            {"logic": 0.8}, 
            self.alive_node, 
            self.resource_room, 
            self.trainer
        )
    
    def test_create_training_agent(self):
        """Test creating a training-integrated agent"""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.name, "TestAgent")
        self.assertEqual(self.agent.trainer, self.trainer)
    
    def test_enhanced_reasoning_without_model(self):
        """Test enhanced reasoning without trained model"""
        result = self.agent.enhanced_reason_with_ml("Test task")
        self.assertIn('confidence', result)
        self.assertIn('agent', result)
        # Should not have ML prediction without trained model
        self.assertNotIn('ml_prediction', result)
    
    def test_enhanced_reasoning_with_model(self):
        """Test enhanced reasoning with trained model"""
        # Train the model first
        df = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(df)
        self.trainer.train_model(X, y)
        
        # Test enhanced reasoning with patient features
        patient_features = {
            'age': 72,
            'gender': 'F',
            'education_level': 12,
            'mmse_score': 24,
            'cdr_score': 0.5,
            'apoe_genotype': 'E3/E4'
        }
        
        result = self.agent.enhanced_reason_with_ml(
            "Assess patient", 
            patient_features
        )
        
        self.assertIn('ml_prediction', result)
        self.assertIsInstance(result['ml_prediction'], str)
        # Confidence should be boosted when ML prediction is available
        self.assertGreaterEqual(result.get('confidence', 0), 0.5)


class TestTrainingSimulation(unittest.TestCase):
    """Test cases for the full training simulation"""
    
    def test_run_training_simulation(self):
        """Test running the complete training simulation"""
        results, agents = run_training_simulation()
        
        # Check results
        self.assertIn('train_accuracy', results)
        self.assertIn('test_accuracy', results)
        self.assertIn('feature_importance', results)
        
        # Check agents
        self.assertIsInstance(agents, list)
        self.assertGreater(len(agents), 0)
        
        for agent in agents:
            self.assertIsInstance(agent, TrainingIntegratedAgent)
            self.assertIsNotNone(agent.trainer.model)  # Should have trained model


if __name__ == '__main__':
    unittest.main()