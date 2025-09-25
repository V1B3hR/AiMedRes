#!/usr/bin/env python3
"""
Tests for Clinical Scenario Builder
Tests scenario validation, creation, and medical logic validation
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.simulation_dashboard import (
    ClinicalScenarioValidator, ScenarioBuilder, 
    PatientProfile, TimelineEvent, SimulationScenario,
    PatientProfileModel, TimelineEventModel, SimulationScenarioModel
)
from datetime import datetime, timedelta
import tempfile
import logging

logging.basicConfig(level=logging.INFO)


class TestClinicalScenarioBuilder(unittest.TestCase):
    """Test clinical scenario builder functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.scenario_builder = ScenarioBuilder(self.temp_db)
        self.validator = ClinicalScenarioValidator()
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def test_clinical_validator_vitals(self):
        """Test clinical validation of vital signs"""
        # Create scenario with normal vitals
        patient_profile = PatientProfile(
            patient_id="test_001",
            age=65,
            gender="M",
            vitals={
                'systolic_bp': 120,
                'diastolic_bp': 80,
                'heart_rate': 75,
                'temperature': 36.8
            }
        )
        
        scenario = SimulationScenario(
            scenario_id="test_scenario",
            name="Normal Vitals Test",
            description="Test scenario with normal vitals",
            patient_profile=patient_profile
        )
        
        result = self.validator.validate_scenario(scenario)
        self.assertTrue(result['valid'], "Normal vitals should pass validation")
        
    def test_clinical_validator_medications(self):
        """Test medication contraindication validation"""
        # Create scenario with contraindicated medications
        patient_profile = PatientProfile(
            patient_id="test_002",
            age=70,
            gender="F",
            medications=['warfarin', 'ibuprofen']  # Contraindicated combination
        )
        
        scenario = SimulationScenario(
            scenario_id="test_scenario_med",
            name="Medication Test",
            description="Test scenario with medication contraindications",
            patient_profile=patient_profile
        )
        
        result = self.validator.validate_scenario(scenario)
        self.assertFalse(result['valid'], "Contraindicated medications should fail validation")
        self.assertTrue(any('Contraindicated medication combination' in error for error in result['errors']))
        
    def test_clinical_validator_lab_ranges(self):
        """Test laboratory value range validation"""
        # Create scenario with extreme lab values
        patient_profile = PatientProfile(
            patient_id="test_003",
            age=55,
            gender="M",
            lab_values={
                'glucose': 900,  # Extremely high
                'hemoglobin': 2.0,  # Extremely low
                'creatinine': 0.8  # Normal
            }
        )
        
        scenario = SimulationScenario(
            scenario_id="test_scenario_labs",
            name="Lab Values Test",
            description="Test scenario with extreme lab values",
            patient_profile=patient_profile
        )
        
        result = self.validator.validate_scenario(scenario)
        self.assertFalse(result['valid'], "Extreme lab values should fail validation")
        
    def test_timeline_validation(self):
        """Test timeline event validation"""
        # Create scenario with problematic timeline
        patient_profile = PatientProfile(
            patient_id="test_004",
            age=60,
            gender="F"
        )
        
        now = datetime.now()
        timeline_events = [
            TimelineEvent(
                event_id="event_1",
                timestamp=now,
                event_type="medication_change",
                description="Start medication A"
            ),
            TimelineEvent(
                event_id="event_2",
                timestamp=now + timedelta(minutes=30),  # Too close in time
                event_type="medication_change", 
                description="Start medication B"
            )
        ]
        
        scenario = SimulationScenario(
            scenario_id="test_scenario_timeline",
            name="Timeline Test",
            description="Test scenario with problematic timeline",
            patient_profile=patient_profile,
            timeline_events=timeline_events
        )
        
        result = self.validator.validate_scenario(scenario)
        # Should have warnings about medication changes too close together
        self.assertTrue(len(result['recommendations']) > 0)
        
    def test_scenario_creation_success(self):
        """Test successful scenario creation"""
        # Create valid scenario
        patient_profile_data = PatientProfileModel(
            patient_id="test_005",
            age=68,
            gender="M",
            conditions=["hypertension", "diabetes"],
            medications=["metformin", "lisinopril"],
            vitals={
                'systolic_bp': 135,
                'diastolic_bp': 85,
                'heart_rate': 72
            },
            lab_values={
                'glucose': 110,
                'hemoglobin': 13.5
            }
        )
        
        scenario_data = SimulationScenarioModel(
            name="Diabetes Management",
            description="Scenario for testing diabetes management protocols",
            patient_profile=patient_profile_data,
            timeline_events=[],
            parameters={"simulation_speed": 1.0}
        )
        
        result = self.scenario_builder.create_scenario(scenario_data, "test_user")
        
        self.assertEqual(result['status'], 'success')
        self.assertIsNotNone(result['scenario_id'])
        
    def test_scenario_creation_validation_failure(self):
        """Test scenario creation with validation failure"""
        # Create invalid scenario with contraindicated medications
        patient_profile_data = PatientProfileModel(
            patient_id="test_006",
            age=75,
            gender="F",
            medications=["warfarin", "aspirin"],  # Contraindicated
            vitals={'systolic_bp': 140, 'heart_rate': 68}
        )
        
        scenario_data = SimulationScenarioModel(
            name="Invalid Medication Scenario",
            description="Scenario with medication contraindications",
            patient_profile=patient_profile_data
        )
        
        result = self.scenario_builder.create_scenario(scenario_data, "test_user")
        
        self.assertEqual(result['status'], 'validation_failed')
        self.assertIsNone(result['scenario_id'])
        self.assertTrue(len(result['validation_errors']) > 0)
        
    def test_scenario_retrieval(self):
        """Test scenario retrieval from database"""
        # First create a scenario
        patient_profile_data = PatientProfileModel(
            patient_id="test_007",
            age=45,
            gender="F",
            conditions=["asthma"],
            vitals={'respiratory_rate': 18, 'heart_rate': 80}
        )
        
        scenario_data = SimulationScenarioModel(
            name="Asthma Management",
            description="Scenario for asthma treatment protocols",
            patient_profile=patient_profile_data
        )
        
        create_result = self.scenario_builder.create_scenario(scenario_data, "test_user")
        scenario_id = create_result['scenario_id']
        
        # Retrieve the scenario
        retrieved_scenario = self.scenario_builder.get_scenario(scenario_id)
        
        self.assertIsNotNone(retrieved_scenario)
        self.assertEqual(retrieved_scenario.name, "Asthma Management")
        self.assertEqual(retrieved_scenario.patient_profile.age, 45)
        
    def test_scenario_listing(self):
        """Test listing scenarios"""
        # Create a few scenarios
        for i in range(3):
            patient_profile_data = PatientProfileModel(
                patient_id=f"test_00{8+i}",
                age=50 + i * 5,
                gender="M" if i % 2 == 0 else "F"
            )
            
            scenario_data = SimulationScenarioModel(
                name=f"Test Scenario {i+1}",
                description=f"Test scenario number {i+1}",
                patient_profile=patient_profile_data
            )
            
            self.scenario_builder.create_scenario(scenario_data, "test_user")
        
        # List scenarios
        scenarios = self.scenario_builder.list_scenarios()
        
        self.assertGreaterEqual(len(scenarios), 3)
        
        # Check that scenarios have required fields
        for scenario in scenarios:
            self.assertIn('scenario_id', scenario)
            self.assertIn('name', scenario)
            self.assertIn('created_by', scenario)
            
    def test_age_vitals_consistency(self):
        """Test age and vitals consistency validation"""
        # Test pediatric patient with inappropriate vitals
        pediatric_profile = PatientProfile(
            patient_id="test_010",
            age=10,  # Pediatric
            gender="M",
            vitals={'heart_rate': 50}  # Too low for pediatric patient
        )
        
        scenario = SimulationScenario(
            scenario_id="test_pediatric",
            name="Pediatric Test",
            description="Test pediatric vital signs validation",
            patient_profile=pediatric_profile
        )
        
        result = self.validator._validate_age_vitals_consistency(scenario)
        self.assertFalse(result['valid'])
        
        # Test geriatric patient 
        geriatric_profile = PatientProfile(
            patient_id="test_011",
            age=85,  # Geriatric
            gender="F",
            vitals={'systolic_bp': 190}  # High but not invalid for elderly
        )
        
        geriatric_scenario = SimulationScenario(
            scenario_id="test_geriatric",
            name="Geriatric Test",
            description="Test geriatric vital signs validation",
            patient_profile=geriatric_profile
        )
        
        geriatric_result = self.validator._validate_age_vitals_consistency(geriatric_scenario)
        # Should be valid but may have warnings
        self.assertTrue(geriatric_result['valid'])


if __name__ == '__main__':
    unittest.main()