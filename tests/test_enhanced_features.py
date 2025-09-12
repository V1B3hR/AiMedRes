#!/usr/bin/env python3
"""
Tests for enhanced features: specialized agents, ensemble training, and multi-modal integration
"""

import unittest
import sys
import os
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specialized_medical_agents import (
    RadiologistAgent, NeurologistAgent, PsychiatristAgent, 
    ConsensusManager, create_specialized_medical_team,
    run_multi_step_diagnostic_simulation, create_test_case
)
from enhanced_ensemble_training import (
    EnhancedEnsembleTrainer, AdvancedFeatureEngineering,
    run_enhanced_training_pipeline
)
from multimodal_data_integration import (
    MultiModalDataLoader, DataFusionProcessor, 
    PrivacyPreservingFederatedLearning, MultiModalMedicalAI
)
from training import AlzheimerTrainer
from labyrinth_adaptive import AliveLoopNode, ResourceRoom


class TestSpecializedMedicalAgents(unittest.TestCase):
    """Test cases for specialized medical agents"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.alive_node = AliveLoopNode((0, 0), (0.5, 0), 15.0, node_id=1)
        self.resource_room = ResourceRoom()
        
    def test_create_radiologist_agent(self):
        """Test creating radiologist agent"""
        radiologist = RadiologistAgent(
            "TestRadiologist",
            {"analytical": 0.9, "logical": 0.85},
            self.alive_node,
            self.resource_room
        )
        
        self.assertEqual(radiologist.name, "TestRadiologist")
        self.assertEqual(radiologist.specialization, "radiology")
        self.assertIn("imaging", radiologist.expertise_areas)
        
    def test_create_neurologist_agent(self):
        """Test creating neurologist agent"""
        neurologist = NeurologistAgent(
            "TestNeurologist",
            {"logical": 0.9, "analytical": 0.85},
            self.alive_node,
            self.resource_room
        )
        
        self.assertEqual(neurologist.specialization, "neurology")
        self.assertIn("cognitive_assessment", neurologist.expertise_areas)
        
    def test_create_psychiatrist_agent(self):
        """Test creating psychiatrist agent"""
        psychiatrist = PsychiatristAgent(
            "TestPsychiatrist",
            {"empathy": 0.95, "creative": 0.9},
            self.alive_node,
            self.resource_room
        )
        
        self.assertEqual(psychiatrist.specialization, "psychiatry")
        self.assertIn("behavioral_changes", psychiatrist.expertise_areas)
        
    def test_specialized_assessments(self):
        """Test specialized agent assessments"""
        # Create test patient data
        patient_data = {
            'M/F': 1, 'Age': 75, 'EDUC': 12, 'SES': 3,
            'MMSE': 22, 'CDR': 0.5, 'eTIV': 1500, 'nWBV': 0.72, 'ASF': 1.2
        }
        
        # Test radiologist assessment
        radiologist = RadiologistAgent(
            "TestRadiologist", {"analytical": 0.9},
            self.alive_node, self.resource_room
        )
        
        assessment = radiologist.get_specialized_assessment(patient_data)
        self.assertIn('radiological_findings', assessment)
        self.assertIn('imaging_risk_factors', assessment)
        
        # Test neurologist assessment
        neurologist = NeurologistAgent(
            "TestNeurologist", {"logical": 0.9},
            self.alive_node, self.resource_room
        )
        
        assessment = neurologist.get_specialized_assessment(patient_data)
        self.assertIn('neurological_findings', assessment)
        self.assertIn('cognitive_risk_factors', assessment)
        
    def test_consensus_manager(self):
        """Test consensus building between agents"""
        # Create test agents
        agents = create_specialized_medical_team(self.alive_node, self.resource_room)
        consensus_manager = ConsensusManager()
        
        # Test patient data
        patient_data = {
            'M/F': 1, 'Age': 75, 'EDUC': 12, 'SES': 3,
            'MMSE': 22, 'CDR': 0.5, 'eTIV': 1500, 'nWBV': 0.72, 'ASF': 1.2
        }
        
        # Build consensus
        consensus_result = consensus_manager.build_consensus(agents, patient_data)
        
        self.assertIn('consensus_prediction', consensus_result)
        self.assertIn('consensus_confidence', consensus_result)
        self.assertIn('consensus_metrics', consensus_result)
        self.assertIn('specialist_insights', consensus_result)
        
        # Validate metrics
        metrics = consensus_result['consensus_metrics']
        self.assertIn('agreement_score', metrics)
        self.assertIn('risk_assessment', metrics)
        
    def test_agent_learning(self):
        """Test agent learning from case outcomes"""
        agent = RadiologistAgent(
            "LearningRadiologist", {"analytical": 0.8},
            self.alive_node, self.resource_room
        )
        
        initial_boost = agent.expertise_confidence_boost
        
        # Simulate successful case outcome
        case_data = {'test': 'case'}
        outcome = {'accuracy': 0.9, 'consensus': 'correct'}
        
        agent.learn_from_case(case_data, outcome)
        
        # Should increase confidence boost for good performance
        self.assertGreater(agent.expertise_confidence_boost, initial_boost)
        self.assertEqual(len(agent.case_history), 1)
        
    def test_multi_step_diagnostic_simulation(self):
        """Test multi-step diagnostic simulation"""
        agents = create_specialized_medical_team(self.alive_node, self.resource_room)
        consensus_manager = ConsensusManager()
        
        # Create test cases
        test_cases = []
        for i in range(3):
            case = create_test_case()
            test_cases.append(case)
        
        # Run simulation
        results = run_multi_step_diagnostic_simulation(agents, consensus_manager, test_cases)
        
        self.assertEqual(len(results), len(test_cases))
        
        for result in results:
            self.assertIn('consensus_prediction', result)
            self.assertIn('learning_outcome', result)
            self.assertIn('case_number', result)


class TestEnhancedEnsembleTraining(unittest.TestCase):
    """Test cases for enhanced ensemble training"""
    
    def setUp(self):
        """Set up test fixtures"""
        from data_loaders import MockDataLoader
        self.trainer = AlzheimerTrainer(data_loader=MockDataLoader())
        
    def test_advanced_feature_engineering(self):
        """Test advanced feature engineering"""
        # Create test data
        data = pd.DataFrame({
            'Age': [65, 70, 75],
            'MMSE': [28, 24, 20],
            'EDUC': [16, 12, 10],
            'nWBV': [0.8, 0.75, 0.7],
            'eTIV': [1600, 1500, 1400],
            'SES': [1, 2, 3],
            'CDR': [0.0, 0.5, 1.0],
            'ASF': [1.1, 1.2, 1.3]
        })
        
        y = np.array([0, 1, 1])  # Labels for feature selection
        
        feature_engineer = AdvancedFeatureEngineering()
        enhanced_data = feature_engineer.fit_transform(data, y)
        
        # Should have more features after engineering
        self.assertGreater(enhanced_data.shape[1], data.shape[1])
        
        # Test transform on new data
        new_data = data.iloc[[0]].copy()
        transformed_new = feature_engineer.transform(new_data)
        self.assertEqual(transformed_new.shape[1], enhanced_data.shape[1])
        
    def test_enhanced_ensemble_trainer(self):
        """Test enhanced ensemble trainer"""
        enhanced_trainer = EnhancedEnsembleTrainer(self.trainer)
        
        # Test getting model grid
        model_grid = enhanced_trainer.get_advanced_model_grid()
        self.assertIn('RandomForest', model_grid)
        self.assertIn('GradientBoosting', model_grid)
        self.assertIn('SVM', model_grid)
        
        # Each model should have model and params
        for model_name, config in model_grid.items():
            self.assertIn('model', config)
            self.assertIn('params', config)
            
    def test_comprehensive_cross_validation(self):
        """Test comprehensive cross-validation"""
        enhanced_trainer = EnhancedEnsembleTrainer(self.trainer)
        
        # Create simple test models
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        models = {
            'RF': RandomForestClassifier(n_estimators=10, random_state=42),
            'LR': LogisticRegression(random_state=42)
        }
        
        # Create test data
        X = pd.DataFrame(np.random.random((100, 5)))
        y = np.random.randint(0, 2, 100)
        
        # Run cross-validation
        cv_results = enhanced_trainer.comprehensive_cross_validation(models, X, y)
        
        self.assertEqual(len(cv_results), 2)
        for model_name, results in cv_results.items():
            self.assertIn('accuracy', results)
            self.assertIn('precision', results)
            self.assertIn('recall', results)


class TestMultiModalDataIntegration(unittest.TestCase):
    """Test cases for multi-modal data integration"""
    
    def test_multimodal_data_loader(self):
        """Test multi-modal data loader"""
        config = {'fusion_strategy': 'concatenation'}
        loader = MultiModalDataLoader(config)
        
        # Test loading lung disease dataset (will use mock data)
        data = loader.load_lung_disease_dataset()
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('diagnosis', data.columns)
        
    def test_data_fusion_processor(self):
        """Test data fusion processor"""
        processor = DataFusionProcessor()
        
        # Create test multi-modal data
        modality1 = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': ['A', 'B', 'A']
        })
        
        modality2 = pd.DataFrame({
            'feature3': [7, 8, 9],
            'feature4': [10, 11, 12],
            'target': ['A', 'B', 'A']
        })
        
        data_dict = {'mod1': modality1, 'mod2': modality2}
        
        # Test early fusion
        fused_data = processor.early_fusion(data_dict)
        expected_columns = len(modality1.columns) + len(modality2.columns)
        self.assertGreaterEqual(fused_data.shape[1], expected_columns - 1)  # -1 for duplicate target
        
        # Test late fusion
        late_fusion_result = processor.late_fusion(data_dict, 'target')
        self.assertIn('modality_models', late_fusion_result)
        self.assertIn('combined_predictions', late_fusion_result)
        
    def test_privacy_preserving_federated_learning(self):
        """Test privacy-preserving federated learning"""
        federated_learner = PrivacyPreservingFederatedLearning(privacy_budget=1.0)
        
        # Create test client updates
        client_updates = []
        for i in range(3):
            update = {
                'client_id': i,
                'model_weights': {'weight1': np.random.random(5), 'weight2': np.random.random(3)},
                'num_samples': 100 + i * 20,
                'local_accuracy': 0.8 + np.random.random() * 0.1
            }
            client_updates.append(update)
        
        # Test secure aggregation
        aggregated = federated_learner.secure_aggregation(client_updates)
        
        self.assertIn('aggregated_weights', aggregated)
        self.assertIn('num_clients', aggregated)
        self.assertEqual(aggregated['num_clients'], 3)
        
        # Test differential privacy noise
        gradients = np.array([1.0, 2.0, 3.0])
        noisy_gradients = federated_learner.add_differential_privacy_noise(gradients)
        
        # Should be different due to noise
        self.assertFalse(np.array_equal(gradients, noisy_gradients))
        
    def test_multimodal_medical_ai(self):
        """Test comprehensive multi-modal medical AI system"""
        config = {
            'fusion_strategy': 'concatenation',
            'modalities': []  # No additional modalities for this test
        }
        
        multimodal_ai = MultiModalMedicalAI()
        multimodal_ai.setup_data_integration(config)
        
        # This should work even with just the primary dataset
        self.assertIsNotNone(multimodal_ai.data_loader)


class TestIntegrationWithExistingSystems(unittest.TestCase):
    """Test integration with existing duetmind_adaptive systems"""
    
    def test_enhanced_training_integration(self):
        """Test integration of enhanced training with existing trainer"""
        from data_loaders import MockDataLoader
        
        # Should be able to run enhanced training with existing trainer
        trainer = AlzheimerTrainer(data_loader=MockDataLoader())
        enhanced_trainer = EnhancedEnsembleTrainer(trainer)
        
        # Test that we can access trainer attributes
        self.assertEqual(enhanced_trainer.trainer, trainer)
        
    def test_specialized_agents_integration(self):
        """Test integration of specialized agents with existing components"""
        # Test that specialized agents work with existing simulation components
        alive_node = AliveLoopNode((0, 0), (0.5, 0), 15.0, node_id=1)
        resource_room = ResourceRoom()
        
        # Should be able to create agents using existing components
        agents = create_specialized_medical_team(alive_node, resource_room)
        
        self.assertEqual(len(agents), 3)  # Radiologist, Neurologist, Psychiatrist
        
        # All agents should have required attributes from base classes
        for agent in agents:
            self.hasattr(agent, 'name')
            self.hasattr(agent, 'specialization')
            self.hasattr(agent, 'expertise_areas')


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)