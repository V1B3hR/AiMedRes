#!/usr/bin/env python3
"""
Test suite for the machine learning training system in duetmind_adaptive
"""

import unittest
import os
import tempfile
import numpy as np
import pandas as pd
import sys
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training import AlzheimerTrainer, TrainingIntegratedAgent
from labyrinth_adaptive import AliveLoopNode, ResourceRoom


class TestAlzheimerTrainer(unittest.TestCase):
    """Test cases for AlzheimerTrainer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.trainer = AlzheimerTrainer()
        
    def test_load_data_creates_sample_when_file_missing(self):
        """Test that load_data creates sample data when file is missing"""
        trainer = AlzheimerTrainer("nonexistent_file.csv")
        df = trainer.load_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('diagnosis', df.columns)
        self.assertIn('age', df.columns)
        
    def test_create_sample_data_structure(self):
        """Test that _create_sample_data creates proper dataset structure"""
        df = self.trainer._create_sample_data()
        
        self.assertEqual(len(df), 100)
        expected_columns = ['age', 'gender', 'education_level', 'mmse_score', 
                           'cdr_score', 'apoe_genotype', 'diagnosis']
        for col in expected_columns:
            self.assertIn(col, df.columns)
            
        # Test data types and ranges
        self.assertTrue(df['age'].between(50, 90).all())
        self.assertTrue(df['education_level'].between(8, 22).all())
        self.assertTrue(df['mmse_score'].between(0, 30).all())
        self.assertTrue(df['cdr_score'].isin([0.0, 0.5, 1.0, 2.0]).all())
        self.assertTrue(df['gender'].isin(['M', 'F']).all())
        self.assertTrue(df['diagnosis'].isin(['Normal', 'MCI', 'Dementia']).all())
    
    def test_preprocess_data(self):
        """Test data preprocessing functionality"""
        df = self.trainer._create_sample_data()
        X, y = self.trainer.preprocess_data(df)
        
        # Check output shapes
        self.assertEqual(X.shape[0], len(df))
        self.assertEqual(y.shape[0], len(df))
        self.assertEqual(X.shape[1], 6)  # Expected number of features
        
        # Check that features are scaled (mean should be close to 0)
        self.assertAlmostEqual(np.mean(X), 0, delta=0.5)
        
        # Check that target is encoded as integers
        self.assertTrue(np.issubdtype(y.dtype, np.integer))
        
    def test_encode_apoe_risk(self):
        """Test APOE risk encoding"""
        self.assertEqual(self.trainer._encode_apoe_risk('E4/E4'), 3)
        self.assertEqual(self.trainer._encode_apoe_risk('E3/E4'), 2)
        self.assertEqual(self.trainer._encode_apoe_risk('E4/E3'), 2)
        self.assertEqual(self.trainer._encode_apoe_risk('E3/E3'), 1)
        self.assertEqual(self.trainer._encode_apoe_risk('E2/E3'), 0)
        
    def test_train_model_small_dataset(self):
        """Test training with small dataset"""
        # Create small dataset
        data = {
            'age': [65, 70, 75, 80],
            'gender': ['M', 'F', 'M', 'F'],
            'education_level': [12, 14, 16, 18],
            'mmse_score': [28, 24, 20, 16],
            'cdr_score': [0.0, 0.5, 1.0, 2.0],
            'apoe_genotype': ['E3/E3', 'E3/E4', 'E4/E4', 'E4/E4'],
            'diagnosis': ['Normal', 'MCI', 'Dementia', 'Dementia']
        }
        df = pd.DataFrame(data)
        X, y = self.trainer.preprocess_data(df)
        
        results = self.trainer.train_model(X, y)
        
        # Check that training completes and returns expected structure
        self.assertIn('train_accuracy', results)
        self.assertIn('test_accuracy', results)
        self.assertIn('feature_importance', results)
        self.assertIsNotNone(self.trainer.model)
        
    def test_train_model_large_dataset(self):
        """Test training with larger dataset"""
        df = self.trainer._create_sample_data()
        X, y = self.trainer.preprocess_data(df)
        
        results = self.trainer.train_model(X, y)
        
        # Check results structure
        self.assertIn('train_accuracy', results)
        self.assertIn('test_accuracy', results)
        self.assertIn('feature_importance', results)
        self.assertIn('classes', results)
        
        # Check reasonable accuracy
        self.assertGreater(results['test_accuracy'], 0.5)
        
        # Check feature importance
        self.assertEqual(len(results['feature_importance']), len(self.trainer.feature_columns))
        
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        # Train a model first
        df = self.trainer._create_sample_data()
        X, y = self.trainer.preprocess_data(df)
        self.trainer.train_model(X, y)
        
        # Test saving
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            success = self.trainer.save_model(tmp_path)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Test loading
            new_trainer = AlzheimerTrainer()
            load_success = new_trainer.load_model(tmp_path)
            self.assertTrue(load_success)
            self.assertIsNotNone(new_trainer.model)
            self.assertEqual(new_trainer.feature_columns, self.trainer.feature_columns)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_save_model_without_training(self):
        """Test that save_model fails gracefully when no model is trained"""
        success = self.trainer.save_model("test.pkl")
        self.assertFalse(success)
    
    def test_predict(self):
        """Test prediction functionality"""
        # Train a model first
        df = self.trainer._create_sample_data()
        X, y = self.trainer.preprocess_data(df)
        self.trainer.train_model(X, y)
        
        # Make predictions on the same data
        prediction_result = self.trainer.predict(X[:5])  # Predict on first 5 samples
        
        self.assertIn('predictions', prediction_result)
        self.assertIn('probabilities', prediction_result)
        self.assertIn('classes', prediction_result)
        
        self.assertEqual(len(prediction_result['predictions']), 5)
        self.assertEqual(len(prediction_result['probabilities']), 5)
        
        # Check that predictions are valid classes
        valid_classes = ['Normal', 'MCI', 'Dementia']
        for pred in prediction_result['predictions']:
            self.assertIn(pred, valid_classes)
    
    def test_predict_without_model(self):
        """Test that predict fails gracefully when no model is trained"""
        with self.assertRaises(ValueError):
            self.trainer.predict(np.array([[1, 2, 3, 4, 5, 6]]))


class TestTrainingIntegratedAgent(unittest.TestCase):
    """Test cases for TrainingIntegratedAgent class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create trainer and train a model
        self.trainer = AlzheimerTrainer()
        df = self.trainer._create_sample_data()
        X, y = self.trainer.preprocess_data(df)
        self.trainer.train_model(X, y)
        
        # Create agent components
        self.alive_node = AliveLoopNode((0, 0), (0.1, 0.1), initial_energy=15.0)
        self.resource_room = ResourceRoom()
        
        # Create enhanced agent
        self.agent = TrainingIntegratedAgent(
            "TestMedicalAI", 
            {"logic": 0.9, "medical": 0.8}, 
            self.alive_node, 
            self.resource_room, 
            self.trainer
        )
    
    def test_initialization(self):
        """Test TrainingIntegratedAgent initialization"""
        self.assertEqual(self.agent.name, "TestMedicalAI")
        self.assertIsNotNone(self.agent.trainer)
        self.assertIsInstance(self.agent.ml_predictions_cache, dict)
        
    def test_enhanced_reason_without_patient_features(self):
        """Test enhanced reasoning without patient features"""
        result = self.agent.enhanced_reason_with_ml("General medical assessment")
        
        self.assertIn('task', result)
        self.assertIn('traditional_reasoning', result)
        self.assertIn('ml_prediction', result)
        self.assertIn('confidence_combined', result)
        
        # ML prediction should be None when no patient features provided
        self.assertIsNone(result['ml_prediction'])
        
    def test_enhanced_reason_with_patient_features(self):
        """Test enhanced reasoning with patient features"""
        patient_features = {
            'age': 75,
            'gender': 'F',
            'education_level': 12,
            'mmse_score': 22,
            'cdr_score': 1.0,
            'apoe_genotype': 'E3/E4'
        }
        
        result = self.agent.enhanced_reason_with_ml("Assess patient", patient_features)
        
        self.assertIn('task', result)
        self.assertIn('traditional_reasoning', result)
        self.assertIn('ml_prediction', result)
        self.assertIn('confidence_combined', result)
        
        # ML prediction should be present
        self.assertIsNotNone(result['ml_prediction'])
        
        if 'error' not in result['ml_prediction']:
            self.assertIn('prediction', result['ml_prediction'])
            self.assertIn('probabilities', result['ml_prediction'])
            self.assertIn('max_probability', result['ml_prediction'])
    
    def test_extract_feature_vector(self):
        """Test feature vector extraction from patient features"""
        patient_features = {
            'age': 70,
            'gender': 'M',
            'education_level': 16,
            'mmse_score': 25,
            'cdr_score': 0.5,
            'apoe_genotype': 'E3/E4'
        }
        
        feature_vector = self.agent._extract_feature_vector(patient_features)
        
        self.assertEqual(len(feature_vector), len(self.trainer.feature_columns))
        self.assertIsInstance(feature_vector, np.ndarray)
        
    def test_encode_apoe_risk_consistency(self):
        """Test that APOE risk encoding is consistent with trainer"""
        test_genotypes = ['E2/E2', 'E2/E3', 'E3/E3', 'E3/E4', 'E4/E4']
        
        for genotype in test_genotypes:
            agent_encoding = self.agent._encode_apoe_risk(genotype)
            trainer_encoding = self.trainer._encode_apoe_risk(genotype)
            self.assertEqual(agent_encoding, trainer_encoding)
    
    def test_get_ml_insights(self):
        """Test ML insights functionality"""
        insights = self.agent.get_ml_insights()
        
        self.assertIn('model_type', insights)
        self.assertIn('feature_columns', insights)
        self.assertIn('target_classes', insights)
        self.assertIn('training_integrated', insights)
        
        self.assertTrue(insights['training_integrated'])
        self.assertEqual(insights['feature_columns'], self.trainer.feature_columns)
    
    def test_get_ml_insights_without_model(self):
        """Test ML insights when no model is available"""
        # Create agent without trained model
        trainer_without_model = AlzheimerTrainer()
        agent_no_model = TrainingIntegratedAgent(
            "TestAgent", {"logic": 0.8}, self.alive_node, self.resource_room, trainer_without_model
        )
        
        insights = agent_no_model.get_ml_insights()
        self.assertIn('error', insights)


class TestTrainingSystemIntegration(unittest.TestCase):
    """Integration tests for the complete training system"""
    
    def test_complete_workflow(self):
        """Test the complete workflow from data loading to prediction"""
        # Step 1: Create trainer and load data
        trainer = AlzheimerTrainer()
        df = trainer._create_sample_data()
        
        # Step 2: Preprocess data
        X, y = trainer.preprocess_data(df)
        
        # Step 3: Train model
        results = trainer.train_model(X, y)
        self.assertGreater(results['test_accuracy'], 0.5)
        
        # Step 4: Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            success = trainer.save_model(tmp_path)
            self.assertTrue(success)
            
            # Step 5: Create enhanced agent
            alive_node = AliveLoopNode((0, 0), (0.1, 0.1), initial_energy=15.0)
            resource_room = ResourceRoom()
            agent = TrainingIntegratedAgent("IntegrationTestAI", {"logic": 0.9}, 
                                          alive_node, resource_room, trainer)
            
            # Step 6: Test enhanced reasoning
            patient_features = {
                'age': 75, 'gender': 'F', 'education_level': 12,
                'mmse_score': 22, 'cdr_score': 1.0, 'apoe_genotype': 'E3/E4'
            }
            
            result = agent.enhanced_reason_with_ml("Assess patient risk", patient_features)
            
            # Verify integration works
            self.assertIn('traditional_reasoning', result)
            self.assertIn('ml_prediction', result)
            self.assertIn('confidence_combined', result)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_framework_compatibility(self):
        """Test that the training system is compatible with existing framework"""
        # Create components using existing framework
        alive_node = AliveLoopNode((1, 1), (0.2, 0.2), initial_energy=20.0)
        resource_room = ResourceRoom()
        
        # Create trainer
        trainer = AlzheimerTrainer()
        df = trainer._create_sample_data()
        X, y = trainer.preprocess_data(df)
        trainer.train_model(X, y)
        
        # Create enhanced agent
        agent = TrainingIntegratedAgent("FrameworkTestAI", {"analytical": 0.85}, 
                                      alive_node, resource_room, trainer)
        
        # Test that enhanced agent still has all original functionality
        self.assertTrue(hasattr(agent, 'reason'))
        self.assertTrue(hasattr(agent, 'log_event'))
        self.assertTrue(hasattr(agent, 'get_state'))
        self.assertTrue(hasattr(agent, 'teleport_to_resource_room'))
        
        # Test original reasoning still works
        traditional_result = agent.reason("Traditional reasoning test")
        self.assertIn('confidence', traditional_result)
        self.assertIn('insight', traditional_result)
        
        # Test enhanced reasoning adds ML capabilities
        enhanced_result = agent.enhanced_reason_with_ml("Enhanced reasoning test")
        self.assertIn('traditional_reasoning', enhanced_result)
        self.assertIn('enhancement_type', enhanced_result)


def run_training_tests():
    """Run all training system tests"""
    print("=== Running Training System Tests ===\n")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestAlzheimerTrainer,
        TestTrainingIntegratedAgent,
        TestTrainingSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nAll tests {'PASSED' if success else 'FAILED'}!")
    
    return success


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    run_training_tests()