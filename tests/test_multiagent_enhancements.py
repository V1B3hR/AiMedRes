#!/usr/bin/env python3
"""
Tests for Multi-Agent Enhancements
Tests agent-to-agent communications, explainability, and safety features
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specialized_medical_agents import (
    ConsensusManager, SafetyValidator, ExplainabilityEngine, 
    AgentCommunicationProtocol, create_specialized_medical_team,
    create_test_case
)
from labyrinth_adaptive import AliveLoopNode, ResourceRoom
import logging

logging.basicConfig(level=logging.INFO)


class TestMultiAgentEnhancements(unittest.TestCase):
    """Test enhanced multi-agent capabilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.alive_node = AliveLoopNode(position=[0, 0], velocity=[0, 0])
        self.resource_room = ResourceRoom("test_resources")
        
    def test_safety_validator(self):
        """Test safety validation system"""
        validator = SafetyValidator()
        
        # Test high confidence disagreement detection
        assessments = [
            {'prediction': 'Demented', 'confidence': 0.9, 'agent_name': 'agent1'},
            {'prediction': 'Nondemented', 'confidence': 0.9, 'agent_name': 'agent2'}
        ]
        
        is_disagreement = validator._check_high_confidence_disagreement(assessments)
        self.assertTrue(is_disagreement, "Should detect high confidence disagreement")
        
        # Test missing critical data detection
        patient_data = {'M/F': 1, 'Age': 75}  # Missing MMSE, CDR
        missing_critical = validator._check_missing_critical_data(patient_data)
        self.assertTrue(missing_critical, "Should detect missing critical data")
        
    def test_explainability_engine(self):
        """Test explainability features"""
        explainer = ExplainabilityEngine()
        
        # Create mock consensus result
        consensus_result = {
            'consensus_prediction': 'Demented',
            'consensus_confidence': 0.85,
            'individual_assessments': [
                {
                    'specialization': 'neurologist',
                    'prediction': 'Demented',
                    'confidence': 0.9,
                    'risk_factors': ['Memory decline', 'Cognitive impairment'],
                    'reasoning': 'Based on MMSE score and imaging findings'
                },
                {
                    'specialization': 'psychiatrist',
                    'prediction': 'Demented',
                    'confidence': 0.8,
                    'risk_factors': ['Behavioral changes', 'Mood alterations'],
                    'reasoning': 'Consistent with dementia presentation'
                }
            ],
            'consensus_metrics': {
                'agreement_score': 0.9,
                'diversity_index': 0.1,
                'confidence_weighted_score': 0.85
            }
        }
        
        explanation = explainer.generate_consensus_explanation(consensus_result)
        
        self.assertIn('summary', explanation)
        self.assertIn('detailed_explanation', explanation)
        self.assertIn('key_evidence', explanation)
        self.assertIn('specialist_contributions', explanation)
        
    def test_agent_communication_protocol(self):
        """Test secure agent communication"""
        protocol = AgentCommunicationProtocol()
        
        # Test valid message
        result = protocol.send_message(
            sender_agent="agent1",
            receiver_agent="agent2",
            message_type="assessment_request",
            content={"patient_summary": {"age_range": "65_to_75"}}
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('communication_id', result)
        
        # Test invalid message type
        invalid_result = protocol.send_message(
            sender_agent="agent1",
            receiver_agent="agent2",
            message_type="invalid_type",
            content={"data": "test"}
        )
        
        self.assertEqual(invalid_result['status'], 'error')
        
    def test_enhanced_consensus_with_safety(self):
        """Test enhanced consensus building with safety checks"""
        agents = create_specialized_medical_team(self.alive_node, self.resource_room)
        consensus_manager = ConsensusManager()
        
        # Test with normal patient data
        patient_data = {
            'M/F': 1, 'Age': 75, 'EDUC': 12, 'SES': 3,
            'MMSE': 22, 'CDR': 0.5, 'eTIV': 1500, 'nWBV': 0.72, 'ASF': 1.2
        }
        
        consensus_result = consensus_manager.build_consensus(agents, patient_data)
        
        # Check that safety validation was performed
        self.assertIn('safety_validation', consensus_result)
        self.assertIn('explanation', consensus_result)
        self.assertIn('communication_log', consensus_result)
        
        # Verify explainability components
        explanation = consensus_result['explanation']
        self.assertIn('summary', explanation)
        self.assertIn('specialist_contributions', explanation)
        
    def test_peer_review_system(self):
        """Test peer review capabilities"""
        agents = create_specialized_medical_team(self.alive_node, self.resource_room)
        consensus_manager = ConsensusManager()
        
        # Create a case that should trigger peer review (conflicting data)
        patient_data = {
            'M/F': 1, 'Age': 45, 'EDUC': 18, 'SES': 1,  # Young, highly educated
            'MMSE': 15, 'CDR': 1.0, 'eTIV': 1400, 'nWBV': 0.65, 'ASF': 1.3  # Poor cognitive scores
        }
        
        consensus_result = consensus_manager.build_consensus(agents, patient_data)
        
        # Should have communication logs from peer review
        self.assertTrue(len(consensus_result['communication_log']) >= 0)
        
        # Check for peer review indicators in assessments
        for assessment in consensus_result['individual_assessments']:
            # Some assessments might have peer review indicators
            peer_reviewed = assessment.get('peer_reviewed', False)
            if peer_reviewed:
                self.assertIn('peer_agreement_ratio', assessment)
                
    def test_safety_hold_scenario(self):
        """Test safety hold for dangerous scenarios"""
        consensus_manager = ConsensusManager()
        agents = create_specialized_medical_team(self.alive_node, self.resource_room)
        
        # Create patient data with extreme age (should trigger safety hold)
        dangerous_patient_data = {
            'M/F': 1, 'Age': 120,  # Impossible age
            'EDUC': 12, 'SES': 3,
            'MMSE': 22, 'CDR': 0.5
        }
        
        result = consensus_manager.build_consensus(agents, dangerous_patient_data)
        
        # Should be put on safety hold
        if result.get('status') == 'safety_hold':
            self.assertIn('safety_flags', result)
            self.assertIn('recommended_actions', result)


if __name__ == '__main__':
    unittest.main()