#!/usr/bin/env python3
"""
Test suite for the Web-Based Simulation Dashboard

Tests all core functionality including:
- Scenario creation and management
- Simulation execution and orchestration
- Intervention capabilities
- Metrics collection and monitoring
- Real-time updates and safety monitoring
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

# Import the dashboard components
from simulation_dashboard import (
    ScenarioBuilder, ExecutionOrchestrator, InterventionPanel, MetricsCollector,
    PatientProfile, TimelineEvent, SimulationScenario, SimulationTick,
    PatientProfileModel, TimelineEventModel, SimulationScenarioModel, InterventionRequest
)

class TestScenarioBuilder:
    """Test the Scenario Builder component"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.builder = ScenarioBuilder(self.temp_db.name)
    
    def teardown_method(self):
        """Cleanup after each test method"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_scenario_creation(self):
        """Test creating a new scenario"""
        scenario_data = SimulationScenarioModel(
            name="Test Scenario",
            description="Test description",
            patient_profile=PatientProfileModel(
                patient_id="test_001",
                age=65,
                gender="Female",
                conditions=["Hypertension"],
                medications=["Lisinopril"],
                vitals={"blood_pressure": 140.0},
                lab_values={"glucose": 120.0},
                medical_history=["Diabetes"]
            ),
            timeline_events=[
                TimelineEventModel(
                    event_id="event_1",
                    timestamp=datetime.now(),
                    event_type="medication_change",
                    parameters={"add": ["Metformin"]},
                    description="Added Metformin"
                )
            ]
        )
        
        scenario_id = self.builder.create_scenario(scenario_data, "test_user")
        
        assert scenario_id is not None
        assert isinstance(scenario_id, str)
        assert len(scenario_id) > 0
    
    def test_scenario_retrieval(self):
        """Test retrieving a scenario"""
        # First create a scenario
        scenario_data = SimulationScenarioModel(
            name="Test Retrieval",
            description="Test description",
            patient_profile=PatientProfileModel(
                patient_id="test_002",
                age=70,
                gender="Male"
            )
        )
        
        scenario_id = self.builder.create_scenario(scenario_data, "test_user")
        
        # Then retrieve it
        retrieved_scenario = self.builder.get_scenario(scenario_id)
        
        assert retrieved_scenario is not None
        assert retrieved_scenario.name == "Test Retrieval"
        assert retrieved_scenario.patient_profile.age == 70
        assert retrieved_scenario.patient_profile.gender == "Male"
    
    def test_scenario_listing(self):
        """Test listing scenarios"""
        # Create multiple scenarios
        for i in range(3):
            scenario_data = SimulationScenarioModel(
                name=f"Test Scenario {i}",
                description=f"Description {i}",
                patient_profile=PatientProfileModel(
                    patient_id=f"test_{i:03d}",
                    age=60 + i,
                    gender="Female" if i % 2 == 0 else "Male"
                )
            )
            self.builder.create_scenario(scenario_data, "test_user")
        
        scenarios = self.builder.list_scenarios()
        
        assert len(scenarios) == 3
        assert all('scenario_id' in s for s in scenarios)
        assert all('name' in s for s in scenarios)


class TestExecutionOrchestrator:
    """Test the Execution Orchestrator component"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Mock Redis for testing
        with patch('simulation_dashboard.redis.from_url') as mock_redis:
            mock_redis.return_value = Mock()
            self.orchestrator = ExecutionOrchestrator()
    
    @pytest.mark.asyncio
    async def test_simulation_start(self):
        """Test starting a simulation"""
        # Create a test scenario
        patient_profile = PatientProfile(
            patient_id="test_patient",
            age=75,
            gender="Female",
            conditions=["Mild Cognitive Impairment"]
        )
        
        timeline_events = [
            TimelineEvent(
                event_id="event_1",
                timestamp=datetime.now(),
                event_type="initial_assessment",
                description="Initial assessment"
            )
        ]
        
        scenario = SimulationScenario(
            scenario_id="test_scenario",
            name="Test Scenario",
            description="Test description",
            patient_profile=patient_profile,
            timeline_events=timeline_events
        )
        
        simulation_id = await self.orchestrator.start_simulation(scenario, "test_user")
        
        assert simulation_id is not None
        assert isinstance(simulation_id, str)
        assert simulation_id in self.orchestrator.active_simulations
        
        # Check simulation info
        sim_info = self.orchestrator.active_simulations[simulation_id]
        assert sim_info['scenario_id'] == "test_scenario"
        assert sim_info['user_id'] == "test_user"
        assert sim_info['status'] in ['starting', 'running']
    
    def test_simulation_status(self):
        """Test getting simulation status"""
        # Add a mock simulation
        simulation_id = "test_sim_123"
        self.orchestrator.active_simulations[simulation_id] = {
            'simulation_id': simulation_id,
            'status': 'running',
            'current_tick': 5
        }
        
        status = self.orchestrator.get_simulation_status(simulation_id)
        
        assert status is not None
        assert status['status'] == 'running'
        assert status['current_tick'] == 5
    
    def test_simulation_stop(self):
        """Test stopping a simulation"""
        # Add a mock simulation
        simulation_id = "test_sim_456"
        self.orchestrator.active_simulations[simulation_id] = {
            'simulation_id': simulation_id,
            'status': 'running'
        }
        
        result = self.orchestrator.stop_simulation(simulation_id)
        
        assert result is True
        assert self.orchestrator.active_simulations[simulation_id]['status'] == 'stopping'
    
    def test_apply_event_medication_change(self):
        """Test applying medication change event"""
        patient_profile = PatientProfile(
            patient_id="test",
            age=65,
            gender="Male",
            medications=["Aspirin"]
        )
        
        event = TimelineEvent(
            event_id="med_change",
            timestamp=datetime.now(),
            event_type="medication_change",
            parameters={"add": ["Metformin"], "remove": ["Aspirin"]}
        )
        
        patient_state = self.orchestrator._apply_event(patient_profile, event)
        
        assert "Metformin" in patient_state['medications']
        assert "Aspirin" not in patient_state['medications']
    
    def test_apply_event_vital_change(self):
        """Test applying vital change event"""
        patient_profile = PatientProfile(
            patient_id="test",
            age=65,
            gender="Male",
            vitals={"blood_pressure": 120.0}
        )
        
        event = TimelineEvent(
            event_id="vital_change",
            timestamp=datetime.now(),
            event_type="vital_change",
            parameters={"blood_pressure": 140.0, "heart_rate": 75.0}
        )
        
        patient_state = self.orchestrator._apply_event(patient_profile, event)
        
        assert patient_state['vitals']['blood_pressure'] == 140.0
        assert patient_state['vitals']['heart_rate'] == 75.0
    
    def test_safety_state_assessment(self):
        """Test safety state assessment"""
        # Test normal state
        agent_results = {
            'agent1': {'state': 'completed'},
            'agent2': {'state': 'completed', 'risk_score': 0.3}
        }
        safety_state = self.orchestrator._assess_safety_state(agent_results)
        assert safety_state == 'normal'
        
        # Test warning state
        agent_results = {
            'agent1': {'state': 'completed', 'risk_score': 0.9},
            'agent2': {'state': 'completed'}
        }
        safety_state = self.orchestrator._assess_safety_state(agent_results)
        assert safety_state == 'warning'
        
        # Test critical state
        agent_results = {
            'agent1': {'state': 'error'},
            'agent2': {'state': 'completed'}
        }
        safety_state = self.orchestrator._assess_safety_state(agent_results)
        assert safety_state == 'critical'


class TestInterventionPanel:
    """Test the Intervention Panel component"""
    
    def setup_method(self):
        """Setup for each test method"""
        with patch('simulation_dashboard.redis.from_url') as mock_redis:
            mock_redis.return_value = Mock()
            self.orchestrator = ExecutionOrchestrator()
        self.intervention_panel = InterventionPanel(self.orchestrator)
    
    def test_apply_pause_intervention(self):
        """Test applying pause intervention"""
        # Add a mock simulation
        simulation_id = "test_sim"
        self.orchestrator.active_simulations[simulation_id] = {
            'simulation_id': simulation_id,
            'status': 'running'
        }
        
        request = InterventionRequest(
            scenario_id=simulation_id,
            intervention_type="pause",
            parameters={}
        )
        
        result = self.intervention_panel.apply_intervention(request, "test_user")
        
        assert result['status'] == 'applied'
        assert result['type'] == 'pause'
        assert self.orchestrator.active_simulations[simulation_id]['status'] == 'paused'
    
    def test_apply_force_fallback_intervention(self):
        """Test applying force fallback intervention"""
        simulation_id = "test_sim"
        self.orchestrator.active_simulations[simulation_id] = {
            'simulation_id': simulation_id,
            'status': 'running'
        }
        
        request = InterventionRequest(
            scenario_id=simulation_id,
            intervention_type="force_fallback",
            parameters={}
        )
        
        result = self.intervention_panel.apply_intervention(request, "test_user")
        
        assert result['status'] == 'applied'
        assert self.orchestrator.active_simulations[simulation_id]['force_fallback'] is True
    
    def test_intervention_history(self):
        """Test intervention history tracking"""
        simulation_id = "test_sim"
        self.orchestrator.active_simulations[simulation_id] = {
            'simulation_id': simulation_id,
            'status': 'running'
        }
        
        # Apply multiple interventions
        interventions = ['pause', 'resume', 'stop']
        for intervention_type in interventions:
            request = InterventionRequest(
                scenario_id=simulation_id,
                intervention_type=intervention_type,
                parameters={}
            )
            self.intervention_panel.apply_intervention(request, "test_user")
        
        history = self.intervention_panel.get_intervention_history(simulation_id)
        
        assert len(history) == 3
        assert all(i['simulation_id'] == simulation_id for i in history)
        assert [i['type'] for i in history] == interventions


class TestMetricsCollector:
    """Test the Metrics Collector component"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.metrics_collector = MetricsCollector()
    
    def test_initial_metrics(self):
        """Test initial metrics state"""
        metrics = self.metrics_collector.get_metrics()
        
        assert metrics['total_simulations'] == 0
        assert metrics['active_simulations'] == 0
        assert metrics['completed_simulations'] == 0
        assert metrics['failed_simulations'] == 0
        assert metrics['scenario_throughput'] == 0
    
    def test_metrics_update(self):
        """Test metrics update functionality"""
        # Create mock orchestrator with simulations
        with patch('simulation_dashboard.redis.from_url') as mock_redis:
            mock_redis.return_value = Mock()
            orchestrator = ExecutionOrchestrator()
        
        # Add mock simulations
        orchestrator.active_simulations = {
            'sim1': {'status': 'running'},
            'sim2': {'status': 'completed'},
            'sim3': {'status': 'failed'},
            'sim4': {'status': 'completed'}
        }
        
        self.metrics_collector.update_metrics(orchestrator)
        metrics = self.metrics_collector.get_metrics()
        
        assert metrics['total_simulations'] == 4
        assert metrics['active_simulations'] == 1
        assert metrics['completed_simulations'] == 2
        assert metrics['failed_simulations'] == 1


class TestDataModels:
    """Test the data models and validation"""
    
    def test_patient_profile_creation(self):
        """Test PatientProfile data model"""
        profile = PatientProfile(
            patient_id="test_001",
            age=65,
            gender="Female",
            conditions=["Hypertension", "Diabetes"],
            medications=["Metformin", "Lisinopril"],
            vitals={"blood_pressure": 140.0, "heart_rate": 72.0},
            lab_values={"glucose": 120.0, "cholesterol": 200.0},
            medical_history=["Family history of heart disease"]
        )
        
        assert profile.patient_id == "test_001"
        assert profile.age == 65
        assert profile.gender == "Female"
        assert len(profile.conditions) == 2
        assert len(profile.medications) == 2
        assert "blood_pressure" in profile.vitals
        assert "glucose" in profile.lab_values
    
    def test_timeline_event_creation(self):
        """Test TimelineEvent data model"""
        event = TimelineEvent(
            event_id="event_001",
            timestamp=datetime.now(),
            event_type="medication_change",
            parameters={"add": ["Aspirin"]},
            description="Added aspirin for cardioprotection"
        )
        
        assert event.event_id == "event_001"
        assert event.event_type == "medication_change"
        assert "add" in event.parameters
        assert event.description != ""
    
    def test_simulation_scenario_creation(self):
        """Test SimulationScenario data model"""
        patient_profile = PatientProfile(
            patient_id="test_patient",
            age=70,
            gender="Male"
        )
        
        timeline_events = [
            TimelineEvent(
                event_id="event_1",
                timestamp=datetime.now(),
                event_type="initial_assessment",
                description="Initial assessment"
            )
        ]
        
        scenario = SimulationScenario(
            scenario_id="scenario_001",
            name="Test Scenario",
            description="Test scenario description",
            patient_profile=patient_profile,
            timeline_events=timeline_events,
            created_by="test_user"
        )
        
        assert scenario.scenario_id == "scenario_001"
        assert scenario.name == "Test Scenario"
        assert scenario.patient_profile.age == 70
        assert len(scenario.timeline_events) == 1
        assert scenario.created_by == "test_user"
    
    def test_simulation_tick_creation(self):
        """Test SimulationTick data model"""
        tick = SimulationTick(
            scenario_id="scenario_001",
            tick=5,
            timestamp=datetime.now(),
            agents={"agent1": {"state": "completed"}},
            patient_context_hash="abc123",
            recommendations=[{"code": "RX_ADJUST", "confidence": 0.9}],
            safety_state="normal",
            memory_events=[{"type": "risk_computed"}],
            metrics={"avg_latency_ms": 45.5}
        )
        
        assert tick.scenario_id == "scenario_001"
        assert tick.tick == 5
        assert "agent1" in tick.agents
        assert len(tick.recommendations) == 1
        assert tick.safety_state == "normal"
        assert "avg_latency_ms" in tick.metrics


class TestIntegration:
    """Integration tests for the complete dashboard system"""
    
    def setup_method(self):
        """Setup for integration tests"""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Initialize all components
        self.builder = ScenarioBuilder(self.temp_db.name)
        
        with patch('simulation_dashboard.redis.from_url') as mock_redis:
            mock_redis.return_value = Mock()
            self.orchestrator = ExecutionOrchestrator()
        
        self.intervention_panel = InterventionPanel(self.orchestrator)
        self.metrics_collector = MetricsCollector()
    
    def teardown_method(self):
        """Cleanup after integration tests"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test the complete simulation workflow"""
        # 1. Create a scenario
        scenario_data = SimulationScenarioModel(
            name="Integration Test Scenario",
            description="Complete workflow test",
            patient_profile=PatientProfileModel(
                patient_id="integration_test",
                age=68,
                gender="Female",
                conditions=["Mild Cognitive Impairment"]
            ),
            timeline_events=[
                TimelineEventModel(
                    event_id="event_1",
                    timestamp=datetime.now(),
                    event_type="initial_assessment",
                    description="Initial cognitive assessment"
                )
            ]
        )
        
        scenario_id = self.builder.create_scenario(scenario_data, "integration_user")
        assert scenario_id is not None
        
        # 2. Retrieve the scenario
        scenario = self.builder.get_scenario(scenario_id)
        assert scenario is not None
        assert scenario.name == "Integration Test Scenario"
        
        # 3. Start a simulation
        simulation_id = await self.orchestrator.start_simulation(scenario, "integration_user")
        assert simulation_id is not None
        
        # 4. Apply an intervention
        request = InterventionRequest(
            scenario_id=simulation_id,
            intervention_type="pause",
            parameters={}
        )
        
        intervention_result = self.intervention_panel.apply_intervention(request, "integration_user")
        assert intervention_result['status'] == 'applied'
        
        # 5. Check metrics
        self.metrics_collector.update_metrics(self.orchestrator)
        metrics = self.metrics_collector.get_metrics()
        assert metrics['total_simulations'] >= 1
        
        # 6. Verify intervention history
        history = self.intervention_panel.get_intervention_history(simulation_id)
        assert len(history) >= 1
        assert history[0]['type'] == 'pause'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])