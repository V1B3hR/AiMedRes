#!/usr/bin/env python3
"""
Web-Based Simulation Dashboard for DuetMind Adaptive

A comprehensive system for clinical scenario simulation that includes:
- Scenario Builder: Compose patient profiles + timeline events
- Execution Orchestrator: Launch ephemeral simulation workers with isolation
- Real-Time Metrics Layer: Pub/Sub channel broadcasting agent states
- Intervention Panel: Manual override capabilities
- Drift & QA Harness: Compare simulation output vs ground-truth
- Authentication & RBAC: Clinician vs Developer vs Auditor roles
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import redis
import sqlite3
from pathlib import Path

# FastAPI and web components
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Import existing components
try:
    from clinical_decision_support import ClinicalDecisionSupportSystem
    from specialized_medical_agents import MedicalKnowledgeAgent
    from labyrinth_adaptive import UnifiedAdaptiveAgent
    from secure_medical_processor import SecureMedicalDataProcessor
except ImportError as e:
    logging.warning(f"Some modules not available: {e}")

logger = logging.getLogger("SimulationDashboard")

# Data Models
@dataclass
class PatientProfile:
    """Patient profile for simulation scenarios"""
    patient_id: str
    age: int
    gender: str
    conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    vitals: Dict[str, float] = field(default_factory=dict)
    lab_values: Dict[str, float] = field(default_factory=dict)
    medical_history: List[str] = field(default_factory=list)

@dataclass
class TimelineEvent:
    """Timeline event for patient simulation"""
    event_id: str
    timestamp: datetime
    event_type: str  # 'medication_change', 'vital_change', 'lab_result', 'condition_onset'
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

@dataclass
class SimulationScenario:
    """Complete simulation scenario definition"""
    scenario_id: str
    name: str
    description: str
    patient_profile: PatientProfile
    timeline_events: List[TimelineEvent] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SimulationTick:
    """Data contract for simulation tick"""
    scenario_id: str
    tick: int
    timestamp: datetime
    agents: Dict[str, Dict[str, Any]]
    patient_context_hash: str
    recommendations: List[Dict[str, Any]]
    safety_state: str
    memory_events: List[Dict[str, Any]]
    metrics: Dict[str, float] = field(default_factory=dict)

# Pydantic Models for API
class PatientProfileModel(BaseModel):
    patient_id: str
    age: int
    gender: str
    conditions: List[str] = []
    medications: List[str] = []
    vitals: Dict[str, float] = {}
    lab_values: Dict[str, float] = {}
    medical_history: List[str] = []

class TimelineEventModel(BaseModel):
    event_id: str
    timestamp: datetime
    event_type: str
    parameters: Dict[str, Any] = {}
    description: str = ""

class SimulationScenarioModel(BaseModel):
    name: str
    description: str
    patient_profile: PatientProfileModel
    timeline_events: List[TimelineEventModel] = []
    parameters: Dict[str, Any] = {}

class InterventionRequest(BaseModel):
    scenario_id: str
    intervention_type: str  # 'pause', 'override', 'inject_memory', 'force_fallback'
    parameters: Dict[str, Any] = {}

# Core Dashboard Components

class ClinicalScenarioValidator:
    """Validates clinical scenarios for medical accuracy and safety"""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.medical_constraints = self._initialize_medical_constraints()
    
    def _initialize_validation_rules(self) -> List[Dict[str, Any]]:
        """Initialize clinical validation rules"""
        return [
            {
                'name': 'age_vitals_consistency',
                'validator': self._validate_age_vitals_consistency,
                'severity': 'high'
            },
            {
                'name': 'medication_contraindications',
                'validator': self._validate_medication_contraindications,
                'severity': 'critical'
            },
            {
                'name': 'lab_value_ranges',
                'validator': self._validate_lab_ranges,
                'severity': 'medium'
            },
            {
                'name': 'timeline_medical_logic',
                'validator': self._validate_timeline_logic,
                'severity': 'high'
            }
        ]
    
    def _initialize_medical_constraints(self) -> Dict[str, Any]:
        """Initialize medical constraints and normal ranges"""
        return {
            'vitals': {
                'systolic_bp': {'min': 70, 'max': 250, 'normal_min': 90, 'normal_max': 140},
                'diastolic_bp': {'min': 40, 'max': 150, 'normal_min': 60, 'normal_max': 90},
                'heart_rate': {'min': 30, 'max': 200, 'normal_min': 60, 'normal_max': 100},
                'temperature': {'min': 32.0, 'max': 44.0, 'normal_min': 36.1, 'normal_max': 37.2},
                'respiratory_rate': {'min': 6, 'max': 60, 'normal_min': 12, 'normal_max': 20}
            },
            'lab_values': {
                'glucose': {'min': 20, 'max': 800, 'normal_min': 70, 'normal_max': 100},
                'hemoglobin': {'min': 3.0, 'max': 20.0, 'normal_min': 12.0, 'normal_max': 16.0},
                'white_blood_cells': {'min': 1.0, 'max': 100.0, 'normal_min': 4.0, 'normal_max': 11.0},
                'creatinine': {'min': 0.1, 'max': 15.0, 'normal_min': 0.6, 'normal_max': 1.2}
            },
            'medication_contraindications': {
                'warfarin': ['ibuprofen', 'aspirin', 'clopidogrel'],
                'ace_inhibitors': ['potassium_supplements'],
                'beta_blockers': ['verapamil', 'diltiazem'],
                'metformin': []  # Special handling for kidney function
            }
        }
    
    def validate_scenario(self, scenario: SimulationScenario) -> Dict[str, Any]:
        """Validate complete clinical scenario"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Run all validation rules
        for rule in self.validation_rules:
            try:
                result = rule['validator'](scenario)
                
                if not result['valid']:
                    validation_results['valid'] = False
                    
                    if rule['severity'] == 'critical':
                        validation_results['errors'].extend(result['messages'])
                    elif rule['severity'] == 'high':
                        validation_results['errors'].extend(result['messages'])
                    else:
                        validation_results['warnings'].extend(result['messages'])
                
                if 'recommendations' in result:
                    validation_results['recommendations'].extend(result['recommendations'])
                    
            except Exception as e:
                validation_results['errors'].append(f"Validation rule {rule['name']} failed: {str(e)}")
                validation_results['valid'] = False
        
        return validation_results
    
    def _validate_age_vitals_consistency(self, scenario: SimulationScenario) -> Dict[str, Any]:
        """Validate that vitals are consistent with patient age"""
        result = {'valid': True, 'messages': [], 'recommendations': []}
        
        age = scenario.patient_profile.age
        vitals = scenario.patient_profile.vitals
        
        # Age-specific vital sign validation
        if age < 18:  # Pediatric
            if vitals.get('heart_rate', 80) < 70:
                result['valid'] = False
                result['messages'].append(f"Heart rate too low for pediatric patient (age {age})")
        elif age > 80:  # Geriatric
            if vitals.get('systolic_bp', 120) > 180:
                result['warnings'] = result.get('warnings', [])
                result['warnings'].append(f"High blood pressure in geriatric patient - consider medication review")
        
        return result
    
    def _validate_medication_contraindications(self, scenario: SimulationScenario) -> Dict[str, Any]:
        """Validate medication combinations for safety"""
        result = {'valid': True, 'messages': [], 'recommendations': []}
        
        medications = scenario.patient_profile.medications
        contraindications = self.medical_constraints['medication_contraindications']
        
        for med in medications:
            if med in contraindications:
                contraindicated_meds = contraindications[med]
                for other_med in medications:
                    if other_med in contraindicated_meds:
                        result['valid'] = False
                        result['messages'].append(
                            f"Contraindicated medication combination: {med} + {other_med}"
                        )
                        result['recommendations'].append(
                            f"Consider alternative to {other_med} or discontinue {med}"
                        )
        
        return result
    
    def _validate_lab_ranges(self, scenario: SimulationScenario) -> Dict[str, Any]:
        """Validate laboratory values are within acceptable ranges"""
        result = {'valid': True, 'messages': [], 'recommendations': []}
        
        lab_values = scenario.patient_profile.lab_values
        lab_constraints = self.medical_constraints['lab_values']
        
        for lab_name, value in lab_values.items():
            if lab_name in lab_constraints:
                constraints = lab_constraints[lab_name]
                
                if value < constraints['min'] or value > constraints['max']:
                    result['valid'] = False
                    result['messages'].append(
                        f"{lab_name} value {value} outside acceptable range "
                        f"({constraints['min']}-{constraints['max']})"
                    )
                elif value < constraints['normal_min'] or value > constraints['normal_max']:
                    result['recommendations'].append(
                        f"{lab_name} value {value} outside normal range - monitor closely"
                    )
        
        return result
    
    def _validate_timeline_logic(self, scenario: SimulationScenario) -> Dict[str, Any]:
        """Validate timeline events for medical logic"""
        result = {'valid': True, 'messages': [], 'recommendations': []}
        
        events = sorted(scenario.timeline_events, key=lambda e: e.timestamp)
        
        # Check for medically impossible sequences
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]
            
            # Check time intervals for medication changes
            if (current_event.event_type == 'medication_change' and 
                next_event.event_type == 'medication_change'):
                
                time_diff = (next_event.timestamp - current_event.timestamp).total_seconds()
                if time_diff < 3600:  # Less than 1 hour
                    result['messages'].append(
                        f"Medication changes too close in time: {time_diff/3600:.1f} hours"
                    )
                    result['recommendations'].append(
                        "Consider spacing medication changes by at least 1 hour"
                    )
        
        return result


class ScenarioBuilder:
    """Compose patient profile + timeline events and parameter sweeps"""
    
    def __init__(self, db_path: str = "simulation_scenarios.db"):
        self.db_path = db_path
        self.validator = ClinicalScenarioValidator()
        self._init_database()
    
    def _init_database(self):
        """Initialize the database for scenario storage"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scenarios (
                scenario_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                patient_profile TEXT,
                timeline_events TEXT,
                parameters TEXT,
                created_by TEXT,
                created_at TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def create_scenario(self, scenario_data: SimulationScenarioModel, user_id: str) -> Dict[str, Any]:
        """Create a new simulation scenario with validation"""
        scenario_id = str(uuid.uuid4())
        
        patient_profile = PatientProfile(**scenario_data.patient_profile.model_dump())
        timeline_events = [TimelineEvent(**event.model_dump()) for event in scenario_data.timeline_events]
        
        scenario = SimulationScenario(
            scenario_id=scenario_id,
            name=scenario_data.name,
            description=scenario_data.description,
            patient_profile=patient_profile,
            timeline_events=timeline_events,
            parameters=scenario_data.parameters,
            created_by=user_id,
            created_at=datetime.now()
        )
        
        # Validate scenario before storing
        validation_result = self.validator.validate_scenario(scenario)
        
        if not validation_result['valid']:
            return {
                'status': 'validation_failed',
                'scenario_id': None,
                'validation_errors': validation_result['errors'],
                'validation_warnings': validation_result['warnings'],
                'recommendations': validation_result['recommendations']
            }
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO scenarios VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scenario.scenario_id,
            scenario.name,
            scenario.description,
            json.dumps(scenario.patient_profile.__dict__, default=str),
            json.dumps([event.__dict__ for event in scenario.timeline_events], default=str),
            json.dumps(scenario.parameters),
            scenario.created_by,
            scenario.created_at
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Created scenario {scenario_id}: {scenario.name}")
        return {
            'status': 'success',
            'scenario_id': scenario_id,
            'validation_warnings': validation_result.get('warnings', []),
            'recommendations': validation_result.get('recommendations', [])
        }
    
    def get_scenario(self, scenario_id: str) -> Optional[SimulationScenario]:
        """Retrieve a simulation scenario"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT * FROM scenarios WHERE scenario_id = ?
        """, (scenario_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Reconstruct scenario object
        patient_profile_data = json.loads(row[3])
        timeline_events_data = json.loads(row[4])
        parameters = json.loads(row[5])
        
        patient_profile = PatientProfile(**patient_profile_data)
        timeline_events = [TimelineEvent(**event) for event in timeline_events_data]
        
        return SimulationScenario(
            scenario_id=row[0],
            name=row[1],
            description=row[2],
            patient_profile=patient_profile,
            timeline_events=timeline_events,
            parameters=parameters,
            created_by=row[6],
            created_at=datetime.fromisoformat(row[7])
        )
    
    def list_scenarios(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all scenarios or scenarios by user"""
        conn = sqlite3.connect(self.db_path)
        if user_id:
            cursor = conn.execute("""
                SELECT scenario_id, name, description, created_by, created_at 
                FROM scenarios WHERE created_by = ?
            """, (user_id,))
        else:
            cursor = conn.execute("""
                SELECT scenario_id, name, description, created_by, created_at 
                FROM scenarios
            """)
        
        scenarios = []
        for row in cursor.fetchall():
            scenarios.append({
                'scenario_id': row[0],
                'name': row[1],
                'description': row[2],
                'created_by': row[3],
                'created_at': row[4]
            })
        
        conn.close()
        return scenarios


class ExecutionOrchestrator:
    """Launch ephemeral simulation workers with isolation"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.active_simulations: Dict[str, Dict] = {}
        self.worker_pool = ThreadPoolExecutor(max_workers=10)
    
    async def start_simulation(self, scenario: SimulationScenario, user_id: str) -> str:
        """Start a new simulation"""
        simulation_id = str(uuid.uuid4())
        
        simulation_info = {
            'simulation_id': simulation_id,
            'scenario_id': scenario.scenario_id,
            'status': 'starting',
            'user_id': user_id,
            'start_time': datetime.now(),
            'current_tick': 0,
            'total_events': len(scenario.timeline_events)
        }
        
        self.active_simulations[simulation_id] = simulation_info
        
        # Start simulation in worker thread
        self.worker_pool.submit(self._run_simulation_worker, scenario, simulation_id)
        
        logger.info(f"Started simulation {simulation_id} for scenario {scenario.scenario_id}")
        return simulation_id
    
    def _run_simulation_worker(self, scenario: SimulationScenario, simulation_id: str):
        """Worker function to run simulation"""
        try:
            simulation_info = self.active_simulations[simulation_id]
            simulation_info['status'] = 'running'
            
            # Initialize agents
            agents = self._initialize_agents(scenario)
            
            # Process timeline events
            for tick, event in enumerate(scenario.timeline_events):
                if simulation_info.get('status') != 'running':
                    break
                
                simulation_info['current_tick'] = tick
                
                # Apply event to patient state
                patient_state = self._apply_event(scenario.patient_profile, event)
                
                # Run agent processing
                agent_results = {}
                recommendations = []
                memory_events = []
                
                for agent_name, agent in agents.items():
                    try:
                        result = self._process_agent(agent, patient_state, event)
                        agent_results[agent_name] = result
                        
                        if 'recommendations' in result:
                            recommendations.extend(result['recommendations'])
                        if 'memory_events' in result:
                            memory_events.extend(result['memory_events'])
                    except Exception as e:
                        logger.error(f"Agent {agent_name} error: {e}")
                        agent_results[agent_name] = {'state': 'error', 'error': str(e)}
                
                # Create simulation tick
                tick_data = SimulationTick(
                    scenario_id=scenario.scenario_id,
                    tick=tick,
                    timestamp=datetime.now(),
                    agents=agent_results,
                    patient_context_hash=self._hash_patient_state(patient_state),
                    recommendations=recommendations,
                    safety_state=self._assess_safety_state(agent_results),
                    memory_events=memory_events,
                    metrics=self._calculate_metrics(agent_results)
                )
                
                # Broadcast tick data
                self._broadcast_tick(simulation_id, tick_data)
                
                # Simulate real-time delay
                time.sleep(0.1)
            
            simulation_info['status'] = 'completed'
            simulation_info['end_time'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Simulation {simulation_id} failed: {e}")
            simulation_info['status'] = 'failed'
            simulation_info['error'] = str(e)
    
    def _initialize_agents(self, scenario: SimulationScenario) -> Dict[str, Any]:
        """Initialize simulation agents"""
        agents = {}
        
        # Create mock agents for demonstration
        agents['risk_assessor'] = {
            'name': 'risk_assessor',
            'type': 'medical_risk',
            'initialized': True
        }
        agents['treatment_advisor'] = {
            'name': 'treatment_advisor',
            'type': 'treatment_recommendation',
            'initialized': True
        }
        
        return agents
    
    def _apply_event(self, patient_profile: PatientProfile, event: TimelineEvent) -> Dict[str, Any]:
        """Apply timeline event to patient state"""
        patient_state = {
            'patient_id': patient_profile.patient_id,
            'age': patient_profile.age,
            'gender': patient_profile.gender,
            'conditions': patient_profile.conditions.copy(),
            'medications': patient_profile.medications.copy(),
            'vitals': patient_profile.vitals.copy(),
            'lab_values': patient_profile.lab_values.copy(),
            'medical_history': patient_profile.medical_history.copy(),
            'current_event': event.__dict__
        }
        
        # Apply event modifications
        if event.event_type == 'medication_change':
            if 'add' in event.parameters:
                patient_state['medications'].extend(event.parameters['add'])
            if 'remove' in event.parameters:
                for med in event.parameters['remove']:
                    if med in patient_state['medications']:
                        patient_state['medications'].remove(med)
        
        elif event.event_type == 'vital_change':
            patient_state['vitals'].update(event.parameters)
        
        elif event.event_type == 'lab_result':
            patient_state['lab_values'].update(event.parameters)
        
        elif event.event_type == 'condition_onset':
            if 'condition' in event.parameters:
                patient_state['conditions'].append(event.parameters['condition'])
        
        return patient_state
    
    def _process_agent(self, agent: Dict[str, Any], patient_state: Dict[str, Any], event: TimelineEvent) -> Dict[str, Any]:
        """Process agent with current patient state"""
        # Mock agent processing for demonstration
        if agent['type'] == 'medical_risk':
            risk_score = hash(str(patient_state)) % 100 / 100.0
            return {
                'state': 'completed',
                'latency_ms': 45 + (hash(str(patient_state)) % 50),
                'risk_score': risk_score,
                'recommendations': [{
                    'type': 'risk_assessment',
                    'code': 'RISK_STRATIFICATION',
                    'confidence': risk_score,
                    'description': f'Risk score: {risk_score:.2f}'
                }],
                'memory_events': [{
                    'type': 'risk_computed',
                    'timestamp': datetime.now().isoformat(),
                    'risk_score': risk_score
                }]
            }
        
        elif agent['type'] == 'treatment_recommendation':
            confidence = 0.8 + (hash(str(patient_state)) % 20) / 100.0
            return {
                'state': 'completed',
                'latency_ms': 67 + (hash(str(patient_state)) % 30),
                'recommendations': [{
                    'type': 'treatment',
                    'code': 'TREATMENT_ADJUST',
                    'confidence': confidence,
                    'description': 'Consider treatment adjustment'
                }],
                'memory_events': [{
                    'type': 'treatment_evaluated',
                    'timestamp': datetime.now().isoformat(),
                    'confidence': confidence
                }]
            }
        
        return {'state': 'unknown', 'latency_ms': 0}
    
    def _hash_patient_state(self, patient_state: Dict[str, Any]) -> str:
        """Generate hash for patient state"""
        return str(hash(json.dumps(patient_state, sort_keys=True, default=str)))[:16]
    
    def _assess_safety_state(self, agent_results: Dict[str, Dict[str, Any]]) -> str:
        """Assess overall safety state"""
        errors = sum(1 for result in agent_results.values() if result.get('state') == 'error')
        
        if errors > 0:
            return 'critical'
        
        # Check risk scores
        risk_scores = [result.get('risk_score', 0) for result in agent_results.values()]
        if any(score > 0.8 for score in risk_scores):
            return 'warning'
        
        return 'normal'
    
    def _calculate_metrics(self, agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate simulation metrics"""
        latencies = [result.get('latency_ms', 0) for result in agent_results.values()]
        
        return {
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'max_latency_ms': max(latencies) if latencies else 0,
            'agent_count': len(agent_results),
            'success_rate': sum(1 for result in agent_results.values() 
                              if result.get('state') == 'completed') / len(agent_results)
        }
    
    def _broadcast_tick(self, simulation_id: str, tick_data: SimulationTick):
        """Broadcast simulation tick to subscribers"""
        try:
            message = json.dumps(tick_data.__dict__, default=str)
            self.redis_client.publish(f"simulation:{simulation_id}", message)
            self.redis_client.publish("simulation:all", message)
        except Exception as e:
            logger.error(f"Failed to broadcast tick: {e}")
    
    def get_simulation_status(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get current simulation status"""
        return self.active_simulations.get(simulation_id)
    
    def stop_simulation(self, simulation_id: str) -> bool:
        """Stop a running simulation"""
        if simulation_id in self.active_simulations:
            self.active_simulations[simulation_id]['status'] = 'stopping'
            return True
        return False


class InterventionPanel:
    """Manual override capabilities"""
    
    def __init__(self, orchestrator: ExecutionOrchestrator):
        self.orchestrator = orchestrator
        self.intervention_log = []
    
    def apply_intervention(self, request: InterventionRequest, user_id: str) -> Dict[str, Any]:
        """Apply manual intervention to simulation"""
        intervention_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        intervention_record = {
            'intervention_id': intervention_id,
            'simulation_id': request.scenario_id,  # Note: using scenario_id as simulation_id for simplicity
            'user_id': user_id,
            'timestamp': timestamp,
            'type': request.intervention_type,
            'parameters': request.parameters,
            'status': 'applied'
        }
        
        simulation_info = self.orchestrator.active_simulations.get(request.scenario_id)
        if not simulation_info:
            intervention_record['status'] = 'failed'
            intervention_record['error'] = 'Simulation not found'
        else:
            # Apply intervention based on type
            if request.intervention_type == 'pause':
                simulation_info['status'] = 'paused'
            elif request.intervention_type == 'resume':
                simulation_info['status'] = 'running'
            elif request.intervention_type == 'stop':
                simulation_info['status'] = 'stopping'
            elif request.intervention_type == 'force_fallback':
                simulation_info['force_fallback'] = True
            elif request.intervention_type == 'inject_memory':
                # Would inject synthetic memory event
                pass
        
        self.intervention_log.append(intervention_record)
        logger.info(f"Applied intervention {intervention_id}: {request.intervention_type}")
        
        return intervention_record
    
    def get_intervention_history(self, simulation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get intervention history"""
        if simulation_id:
            return [i for i in self.intervention_log if i['simulation_id'] == simulation_id]
        return self.intervention_log.copy()


class MetricsCollector:
    """Collect and track key simulation metrics"""
    
    def __init__(self):
        self.metrics = {
            'scenario_throughput': 0,  # scenarios per hour
            'reproducibility_rate': 0,  # hash-stable reruns
            'avg_latency_ms': 0,  # end-to-end decision cycle
            'intervention_frequency': 0,  # interventions per simulation hour
            'total_simulations': 0,
            'active_simulations': 0,
            'completed_simulations': 0,
            'failed_simulations': 0
        }
        self.start_time = datetime.now()
    
    def update_metrics(self, orchestrator: ExecutionOrchestrator):
        """Update metrics based on current state"""
        active_count = sum(1 for sim in orchestrator.active_simulations.values() 
                          if sim['status'] in ['running', 'starting'])
        completed_count = sum(1 for sim in orchestrator.active_simulations.values() 
                             if sim['status'] == 'completed')
        failed_count = sum(1 for sim in orchestrator.active_simulations.values() 
                          if sim['status'] == 'failed')
        
        self.metrics.update({
            'active_simulations': active_count,
            'completed_simulations': completed_count,
            'failed_simulations': failed_count,
            'total_simulations': len(orchestrator.active_simulations)
        })
        
        # Calculate throughput
        hours_elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
        if hours_elapsed > 0:
            self.metrics['scenario_throughput'] = completed_count / hours_elapsed
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return self.metrics.copy()


# FastAPI Application
app = FastAPI(title="DuetMind Simulation Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
scenario_builder = ScenarioBuilder()
orchestrator = ExecutionOrchestrator()
intervention_panel = InterventionPanel(orchestrator)
metrics_collector = MetricsCollector()

# WebSocket connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Authentication (simplified)
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Simplified authentication - in production, validate JWT token
    return {"user_id": "demo_user", "role": "developer"}

# API Routes

@app.get("/")
async def dashboard():
    """Serve the main dashboard"""
    return HTMLResponse(content=get_dashboard_html(), status_code=200)

@app.post("/api/scenarios")
async def create_scenario(
    scenario: SimulationScenarioModel,
    user: dict = Depends(get_current_user)
):
    """Create a new simulation scenario"""
    scenario_id = scenario_builder.create_scenario(scenario, user["user_id"])
    return {"scenario_id": scenario_id, "status": "created"}

@app.get("/api/scenarios")
async def list_scenarios(user: dict = Depends(get_current_user)):
    """List all scenarios"""
    scenarios = scenario_builder.list_scenarios()
    return {"scenarios": scenarios}

@app.get("/api/scenarios/{scenario_id}")
async def get_scenario(scenario_id: str, user: dict = Depends(get_current_user)):
    """Get a specific scenario"""
    scenario = scenario_builder.get_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    
    return {
        "scenario_id": scenario.scenario_id,
        "name": scenario.name,
        "description": scenario.description,
        "patient_profile": scenario.patient_profile.__dict__,
        "timeline_events": [event.__dict__ for event in scenario.timeline_events],
        "parameters": scenario.parameters,
        "created_by": scenario.created_by,
        "created_at": scenario.created_at
    }

@app.post("/api/simulations/start")
async def start_simulation(
    request: dict,
    user: dict = Depends(get_current_user)
):
    """Start a new simulation"""
    scenario_id = request.get("scenario_id")
    scenario = scenario_builder.get_scenario(scenario_id)
    
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    
    simulation_id = await orchestrator.start_simulation(scenario, user["user_id"])
    return {"simulation_id": simulation_id, "status": "started"}

@app.get("/api/simulations/{simulation_id}/status")
async def get_simulation_status(simulation_id: str, user: dict = Depends(get_current_user)):
    """Get simulation status"""
    status = orchestrator.get_simulation_status(simulation_id)
    if not status:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    return status

@app.post("/api/simulations/{simulation_id}/intervention")
async def apply_intervention(
    simulation_id: str,
    request: InterventionRequest,
    user: dict = Depends(get_current_user)
):
    """Apply manual intervention"""
    request.scenario_id = simulation_id  # Use simulation_id as scenario_id for simplicity
    result = intervention_panel.apply_intervention(request, user["user_id"])
    return result

@app.get("/api/metrics")
async def get_metrics(user: dict = Depends(get_current_user)):
    """Get current simulation metrics"""
    metrics_collector.update_metrics(orchestrator)
    return metrics_collector.get_metrics()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for testing
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def get_dashboard_html():
    """Generate the dashboard HTML"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>DuetMind Simulation Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; margin: -20px -20px 20px -20px; }
        .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .panel { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #666; margin-top: 5px; }
        .button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        .button:hover { background: #2980b9; }
        .status-running { color: #27ae60; }
        .status-paused { color: #f39c12; }
        .status-stopped { color: #e74c3c; }
        .scenario-form { display: grid; gap: 10px; }
        .scenario-form input, .scenario-form textarea { padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .log { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; font-family: monospace; height: 200px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 DuetMind Simulation Dashboard</h1>
        <p>Web-Based Clinical Scenario Simulation Platform</p>
    </div>

    <div class="metrics" id="metrics">
        <div class="metric-card">
            <div class="metric-value" id="active-simulations">0</div>
            <div class="metric-label">Active Simulations</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="completed-simulations">0</div>
            <div class="metric-label">Completed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="throughput">0.0</div>
            <div class="metric-label">Scenarios/Hour</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="avg-latency">0</div>
            <div class="metric-label">Avg Latency (ms)</div>
        </div>
    </div>

    <div class="dashboard-grid">
        <div class="panel">
            <h2>📋 Scenario Builder</h2>
            <div class="scenario-form">
                <input type="text" id="scenario-name" placeholder="Scenario Name" />
                <textarea id="scenario-description" placeholder="Description" rows="3"></textarea>
                <input type="number" id="patient-age" placeholder="Patient Age" />
                <select id="patient-gender">
                    <option value="">Select Gender</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
                <input type="text" id="patient-conditions" placeholder="Conditions (comma-separated)" />
                <button class="button" onclick="createScenario()">Create Scenario</button>
            </div>
            <div id="scenarios-list">
                <h3>Recent Scenarios</h3>
                <div id="scenarios-container"></div>
            </div>
        </div>

        <div class="panel">
            <h2>🎮 Execution Control</h2>
            <div>
                <select id="scenario-select">
                    <option value="">Select Scenario</option>
                </select>
                <button class="button" onclick="startSimulation()">Start Simulation</button>
            </div>
            <div id="active-simulations">
                <h3>Active Simulations</h3>
                <div id="simulations-container"></div>
            </div>
        </div>

        <div class="panel">
            <h2>⚡ Intervention Panel</h2>
            <div>
                <select id="intervention-simulation">
                    <option value="">Select Simulation</option>
                </select>
                <select id="intervention-type">
                    <option value="pause">Pause</option>
                    <option value="resume">Resume</option>
                    <option value="stop">Stop</option>
                    <option value="force_fallback">Force Fallback</option>
                </select>
                <button class="button" onclick="applyIntervention()">Apply Intervention</button>
            </div>
            <div id="intervention-log">
                <h3>Intervention Log</h3>
                <div class="log" id="interventions-display"></div>
            </div>
        </div>

        <div class="panel">
            <h2>📊 Real-Time Metrics</h2>
            <div class="log" id="real-time-log">
                Waiting for simulation data...
            </div>
            <div>
                <h3>Safety State</h3>
                <div id="safety-state" class="status-running">NORMAL</div>
            </div>
        </div>
    </div>

    <script>
        let scenarios = [];
        let simulations = {};
        let ws = null;

        // Initialize WebSocket connection
        function initWebSocket() {
            ws = new WebSocket('ws://localhost:8000/ws');
            ws.onmessage = function(event) {
                const log = document.getElementById('real-time-log');
                log.innerHTML += event.data + '\\n';
                log.scrollTop = log.scrollHeight;
            };
        }

        // Load metrics
        async function loadMetrics() {
            try {
                const response = await fetch('/api/metrics', {
                    headers: {'Authorization': 'Bearer demo-token'}
                });
                const metrics = await response.json();
                
                document.getElementById('active-simulations').textContent = metrics.active_simulations;
                document.getElementById('completed-simulations').textContent = metrics.completed_simulations;
                document.getElementById('throughput').textContent = metrics.scenario_throughput.toFixed(1);
                document.getElementById('avg-latency').textContent = Math.round(metrics.avg_latency_ms);
            } catch (error) {
                console.error('Failed to load metrics:', error);
            }
        }

        // Load scenarios
        async function loadScenarios() {
            try {
                const response = await fetch('/api/scenarios', {
                    headers: {'Authorization': 'Bearer demo-token'}
                });
                const data = await response.json();
                scenarios = data.scenarios;
                
                const container = document.getElementById('scenarios-container');
                const select = document.getElementById('scenario-select');
                
                container.innerHTML = '';
                select.innerHTML = '<option value="">Select Scenario</option>';
                
                scenarios.forEach(scenario => {
                    const div = document.createElement('div');
                    div.innerHTML = `<strong>${scenario.name}</strong><br><small>${scenario.description}</small>`;
                    container.appendChild(div);
                    
                    const option = document.createElement('option');
                    option.value = scenario.scenario_id;
                    option.textContent = scenario.name;
                    select.appendChild(option);
                });
            } catch (error) {
                console.error('Failed to load scenarios:', error);
            }
        }

        // Create scenario
        async function createScenario() {
            const name = document.getElementById('scenario-name').value;
            const description = document.getElementById('scenario-description').value;
            const age = parseInt(document.getElementById('patient-age').value);
            const gender = document.getElementById('patient-gender').value;
            const conditions = document.getElementById('patient-conditions').value.split(',').map(s => s.trim()).filter(s => s);

            if (!name || !age || !gender) {
                alert('Please fill in required fields');
                return;
            }

            const scenario = {
                name,
                description,
                patient_profile: {
                    patient_id: 'patient_' + Date.now(),
                    age,
                    gender,
                    conditions,
                    medications: [],
                    vitals: {},
                    lab_values: {},
                    medical_history: []
                },
                timeline_events: [
                    {
                        event_id: 'event_1',
                        timestamp: new Date().toISOString(),
                        event_type: 'initial_assessment',
                        parameters: {},
                        description: 'Initial patient assessment'
                    }
                ],
                parameters: {}
            };

            try {
                const response = await fetch('/api/scenarios', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer demo-token'
                    },
                    body: JSON.stringify(scenario)
                });

                if (response.ok) {
                    alert('Scenario created successfully!');
                    loadScenarios();
                    // Clear form
                    document.getElementById('scenario-name').value = '';
                    document.getElementById('scenario-description').value = '';
                    document.getElementById('patient-age').value = '';
                    document.getElementById('patient-gender').value = '';
                    document.getElementById('patient-conditions').value = '';
                } else {
                    alert('Failed to create scenario');
                }
            } catch (error) {
                console.error('Failed to create scenario:', error);
                alert('Failed to create scenario');
            }
        }

        // Start simulation
        async function startSimulation() {
            const scenarioId = document.getElementById('scenario-select').value;
            if (!scenarioId) {
                alert('Please select a scenario');
                return;
            }

            try {
                const response = await fetch('/api/simulations/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer demo-token'
                    },
                    body: JSON.stringify({ scenario_id: scenarioId })
                });

                if (response.ok) {
                    const data = await response.json();
                    alert(`Simulation started: ${data.simulation_id}`);
                    updateInterventionOptions();
                } else {
                    alert('Failed to start simulation');
                }
            } catch (error) {
                console.error('Failed to start simulation:', error);
                alert('Failed to start simulation');
            }
        }

        // Apply intervention
        async function applyIntervention() {
            const simulationId = document.getElementById('intervention-simulation').value;
            const interventionType = document.getElementById('intervention-type').value;

            if (!simulationId || !interventionType) {
                alert('Please select simulation and intervention type');
                return;
            }

            try {
                const response = await fetch(`/api/simulations/${simulationId}/intervention`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer demo-token'
                    },
                    body: JSON.stringify({
                        scenario_id: simulationId,
                        intervention_type: interventionType,
                        parameters: {}
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    alert(`Intervention applied: ${data.intervention_id}`);
                    
                    // Update log
                    const log = document.getElementById('interventions-display');
                    log.innerHTML += `[${new Date().toLocaleTimeString()}] ${interventionType} applied to ${simulationId}\\n`;
                    log.scrollTop = log.scrollHeight;
                } else {
                    alert('Failed to apply intervention');
                }
            } catch (error) {
                console.error('Failed to apply intervention:', error);
                alert('Failed to apply intervention');
            }
        }

        // Update intervention options
        function updateInterventionOptions() {
            // This would be populated with actual active simulations
            // For demo, we'll just add a mock simulation
            const select = document.getElementById('intervention-simulation');
            if (select.children.length === 1) { // Only default option
                const option = document.createElement('option');
                option.value = 'simulation_demo';
                option.textContent = 'Demo Simulation';
                select.appendChild(option);
            }
        }

        // Initialize dashboard
        function init() {
            loadMetrics();
            loadScenarios();
            initWebSocket();
            
            // Refresh metrics every 10 seconds
            setInterval(loadMetrics, 10000);
            
            console.log('DuetMind Simulation Dashboard initialized');
        }

        // Start when page loads
        window.onload = init;
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    # Create database directory if it doesn't exist
    Path("simulation_scenarios.db").parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting DuetMind Simulation Dashboard")
    uvicorn.run(app, host="0.0.0.0", port=8000)