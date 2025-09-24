"""
Integration tests for the full training pipeline
"""
import pytest
import pandas as pd
import tempfile
import os

from training.training import AlzheimerTrainer, TrainingConfig
from scripts.data_loaders import MockDataLoader, CSVDataLoader, create_data_loader


class TestTrainingPipeline:
    """Integration tests for complete training workflow"""
    
    def test_end_to_end_training_with_mock_data(self, sample_alzheimer_data):
        """Test complete training pipeline with mock data"""
        # Setup
        mock_loader = MockDataLoader(mock_data=sample_alzheimer_data)
        config = TrainingConfig(random_seed=42)
        trainer = AlzheimerTrainer(data_loader=mock_loader, config=config)
        
        # Load data
        df = trainer.load_data()
        assert len(df) == len(sample_alzheimer_data)
        assert mock_loader.load_called
        
        # Preprocess
        X, y = trainer.preprocess_data(df)
        assert X.shape[0] == len(sample_alzheimer_data)
        assert len(trainer.feature_columns) > 0
        
        # Train
        results = trainer.train_model(X, y)
        assert trainer.model is not None
        assert 'train_accuracy' in results
        assert 'test_accuracy' in results
        
        # Predict
        test_features = {
            'age': 72,
            'gender': 'F',
            'education_level': 12,
            'mmse_score': 24,
            'cdr_score': 0.5,
            'apoe_genotype': 'E3/E4'
        }
        prediction = trainer.predict(test_features)
        assert prediction in ['Normal', 'MCI', 'Dementia']
    
    def test_end_to_end_training_with_csv_data(self, temp_csv_file):
        """Test complete training pipeline with CSV data"""
        # Setup
        csv_loader = CSVDataLoader(str(temp_csv_file))
        trainer = AlzheimerTrainer(data_loader=csv_loader)
        
        # Load and process data
        df = trainer.load_data()
        X, y = trainer.preprocess_data(df)
        results = trainer.train_model(X, y)
        
        # Verify training completed successfully
        assert trainer.model is not None
        assert results['train_accuracy'] >= 0
        assert results['test_accuracy'] >= 0
    
    def test_model_persistence_workflow(self, sample_alzheimer_data):
        """Test saving and loading trained models"""
        # Train initial model
        mock_loader = MockDataLoader(mock_data=sample_alzheimer_data)
        trainer1 = AlzheimerTrainer(data_loader=mock_loader)
        
        df = trainer1.load_data()
        X, y = trainer1.preprocess_data(df)
        results1 = trainer1.train_model(X, y)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            trainer1.save_model(model_path)
            
            # Load model in new trainer
            trainer2 = AlzheimerTrainer()
            trainer2.load_model(model_path)
            
            # Test predictions are consistent
            test_features = {
                'age': 72,
                'gender': 'F', 
                'education_level': 12,
                'mmse_score': 24,
                'cdr_score': 0.5,
                'apoe_genotype': 'E3/E4'
            }
            
            pred1 = trainer1.predict(test_features)
            pred2 = trainer2.predict(test_features)
            assert pred1 == pred2
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_deterministic_training(self, sample_alzheimer_data):
        """Test that training produces consistent results with same random seed"""
        config = TrainingConfig(random_seed=123)
        
        # First training run
        loader1 = MockDataLoader(mock_data=sample_alzheimer_data)
        trainer1 = AlzheimerTrainer(data_loader=loader1, config=config)
        df1 = trainer1.load_data()
        X1, y1 = trainer1.preprocess_data(df1)
        results1 = trainer1.train_model(X1, y1)
        
        # Second training run with same config
        loader2 = MockDataLoader(mock_data=sample_alzheimer_data)
        trainer2 = AlzheimerTrainer(data_loader=loader2, config=config)
        df2 = trainer2.load_data()
        X2, y2 = trainer2.preprocess_data(df2)
        results2 = trainer2.train_model(X2, y2)
        
        # Results should be identical
        assert results1['train_accuracy'] == results2['train_accuracy']
        assert results1['test_accuracy'] == results2['test_accuracy']
        
        # Feature importance should also be identical
        for feature in results1['feature_importance']:
            assert results1['feature_importance'][feature] == results2['feature_importance'][feature]
    
    def test_data_loader_factory_integration(self, temp_csv_file):
        """Test training with data loader created by factory"""
        # Create loader using factory
        loader = create_data_loader("csv", file_path=str(temp_csv_file))
        trainer = AlzheimerTrainer(data_loader=loader)
        
        # Test full workflow
        df = trainer.load_data()
        X, y = trainer.preprocess_data(df)
        results = trainer.train_model(X, y)
        
        assert trainer.model is not None
        assert 'train_accuracy' in results


class TestTrainingWithNeuralNetwork:
    """Integration tests for training combined with neural network components"""
    
    def test_training_integrated_agent_creation(self, sample_alzheimer_data):
        """Test creating training-integrated agents"""
        from training.training import TrainingIntegratedAgent
        from neuralnet import AliveLoopNode, ResourceRoom
        
        # Setup components
        mock_loader = MockDataLoader(mock_data=sample_alzheimer_data)
        trainer = AlzheimerTrainer(data_loader=mock_loader)
        
        # Train model first
        df = trainer.load_data()
        X, y = trainer.preprocess_data(df)
        trainer.train_model(X, y)
        
        # Create neural network components
        resource_room = ResourceRoom()
        alive_node = AliveLoopNode((0, 0), (0.5, 0), 15.0, node_id=1)
        
        # Create integrated agent
        agent = TrainingIntegratedAgent(
            "TestAgent",
            {"logic": 0.8},
            alive_node,
            resource_room,
            trainer
        )
        
        assert agent.name == "TestAgent"
        assert agent.trainer == trainer
        assert agent.trainer.model is not None
    
    def test_enhanced_reasoning_with_ml(self, sample_alzheimer_data):
        """Test enhanced reasoning with ML predictions"""
        from training.training import TrainingIntegratedAgent, run_training_simulation
        
        # Run training simulation which creates integrated agents
        results, agents = run_training_simulation()
        
        assert len(agents) > 0
        assert 'train_accuracy' in results
        
        # Test enhanced reasoning
        for agent in agents:
            assert hasattr(agent, 'enhanced_reason_with_ml')
            assert agent.trainer.model is not None
            
            # Test reasoning with patient features
            patient_features = {
                'age': 72,
                'gender': 'F',
                'education_level': 12,
                'mmse_score': 24,
                'cdr_score': 0.5,
                'apoe_genotype': 'E3/E4'
            }
            
            result = agent.enhanced_reason_with_ml(
                "Assess patient condition",
                patient_features
            )
            
            assert 'confidence' in result
            assert 'ml_prediction' in result
            assert isinstance(result['ml_prediction'], str)