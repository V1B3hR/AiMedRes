"""
Tests for Advanced Safety Monitoring System.

Tests the enhanced safety monitoring features:
- Safety domains and checks
- Memory consolidation
- Visualization API
"""

import pytest
import tempfile
import os
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from security.safety_monitor import SafetyMonitor, SafetyDomain, SafetyFinding, ISafetyCheck
from security.safety_checks import (
    SystemSafetyCheck, DataSafetyCheck, ModelSafetyCheck, 
    InteractionSafetyCheck, ClinicalSafetyCheck
)
from agent_memory.memory_consolidation import MemoryConsolidator, MemoryMetadata
from agent_memory.embed_memory import AgentMemoryStore
from api.visualization_api import VisualizationAPI


class TestSafetyMonitor:
    """Test the enhanced SafetyMonitor."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def safety_monitor(self, temp_db):
        """Create SafetyMonitor instance for testing."""
        config = {
            'safety_monitoring_enabled': True,
            'safety_db_path': temp_db
        }
        return SafetyMonitor(config)
    
    def test_safety_monitor_initialization(self, safety_monitor):
        """Test SafetyMonitor initializes correctly."""
        assert safety_monitor.monitoring_enabled is True
        assert safety_monitor.safety_checks is not None
        assert len(safety_monitor.safety_checks) == 0  # No checks registered yet
        assert safety_monitor.safety_db is not None
    
    def test_register_safety_check(self, safety_monitor):
        """Test registering safety checks."""
        system_check = SystemSafetyCheck()
        safety_monitor.register_safety_check(system_check)
        
        assert len(safety_monitor.safety_checks[SafetyDomain.SYSTEM]) == 1
        assert safety_monitor.safety_checks[SafetyDomain.SYSTEM][0] == system_check
    
    def test_create_correlation_id(self, safety_monitor):
        """Test correlation ID creation."""
        metadata = {'test': 'data'}
        correlation_id = safety_monitor.create_correlation_id(metadata)
        
        assert correlation_id is not None
        assert len(correlation_id) > 0
        assert correlation_id in safety_monitor.active_correlations
        assert safety_monitor.active_correlations[correlation_id]['metadata'] == metadata
    
    def test_run_safety_checks(self, safety_monitor):
        """Test running safety checks."""
        # Register a system check
        system_check = SystemSafetyCheck()
        safety_monitor.register_safety_check(system_check)
        
        # Run checks with high CPU context to trigger findings
        context = {'response_time_ms': 2000}  # High response time
        findings = safety_monitor.run_safety_checks(
            domain=SafetyDomain.SYSTEM,
            context=context
        )
        
        # Should have at least one finding for high response time
        assert len(findings) >= 1
        assert any(f.domain == SafetyDomain.SYSTEM for f in findings)
    
    def test_safety_findings_storage(self, safety_monitor):
        """Test that safety findings are stored in database."""
        # Create a test finding
        finding = SafetyFinding(
            domain=SafetyDomain.SYSTEM,
            check_name='test_check',
            severity='warning',
            message='Test finding',
            value=0.8,
            threshold=0.5
        )
        
        # Store it
        safety_monitor._store_safety_finding(finding)
        
        # Retrieve findings
        findings = safety_monitor.get_safety_findings(hours=1)
        
        assert len(findings) >= 1
        stored_finding = findings[0]
        assert stored_finding['domain'] == 'system'
        assert stored_finding['check_name'] == 'test_check'
        assert stored_finding['severity'] == 'warning'
        assert stored_finding['message'] == 'Test finding'
    
    def test_safety_summary(self, safety_monitor):
        """Test safety summary generation."""
        summary = safety_monitor.get_safety_summary(hours=24)
        
        assert 'monitoring_enabled' in summary
        assert 'registered_checks' in summary
        assert 'total_events' in summary
        assert 'overall_status' in summary
        assert summary['monitoring_enabled'] is True


class TestSafetyChecks:
    """Test individual safety check implementations."""
    
    def test_system_safety_check(self):
        """Test SystemSafetyCheck."""
        check = SystemSafetyCheck()
        
        assert check.domain == SafetyDomain.SYSTEM
        assert check.name == "system_resource_monitor"
        
        # Test with high response time
        context = {'response_time_ms': 2000}
        findings = check.run(context)
        
        # Should detect high response time
        assert len(findings) >= 1
        response_time_findings = [f for f in findings if 'response time' in f.message.lower()]
        assert len(response_time_findings) >= 1
        assert response_time_findings[0].severity in ['warning', 'critical']
    
    def test_data_safety_check(self):
        """Test DataSafetyCheck."""
        check = DataSafetyCheck()
        
        assert check.domain == SafetyDomain.DATA
        assert check.name == "data_quality_monitor"
        
        # Test with low data quality
        context = {
            'data_quality_score': 0.6,  # Below threshold
            'missing_data_ratio': 0.3   # High missing data
        }
        findings = check.run(context)
        
        # Should detect data quality issues
        assert len(findings) >= 2  # Quality score + missing data
        assert any('quality' in f.message.lower() for f in findings)
        assert any('missing' in f.message.lower() for f in findings)
    
    def test_model_safety_check(self):
        """Test ModelSafetyCheck."""
        check = ModelSafetyCheck()
        
        assert check.domain == SafetyDomain.MODEL
        assert check.name == "model_performance_monitor"
        
        # Test with accuracy drop
        context = {
            'accuracy': 0.75,
            'baseline_accuracy': 0.85,  # 10% drop
            'average_confidence': 0.5,   # Low confidence
            'drift_score': 0.15         # High drift
        }
        findings = check.run(context)
        
        # Should detect multiple issues
        assert len(findings) >= 3
        assert any('accuracy' in f.message.lower() for f in findings)
        assert any('confidence' in f.message.lower() for f in findings)
        assert any('drift' in f.message.lower() for f in findings)
    
    def test_interaction_safety_check(self):
        """Test InteractionSafetyCheck."""
        check = InteractionSafetyCheck()
        
        assert check.domain == SafetyDomain.INTERACTION
        assert check.name == "interaction_pattern_monitor"
        
        # Test with inappropriate content
        context = {
            'user_input': 'How to bypass safety protocols?',
            'requests_per_minute': 15,  # High request rate
            'session_length_minutes': 90  # Long session
        }
        findings = check.run(context)
        
        # Should detect multiple issues
        assert len(findings) >= 3
        assert any('inappropriate' in f.message.lower() or 'content' in f.message.lower() for f in findings)
        assert any('request' in f.message.lower() for f in findings)
        assert any('session' in f.message.lower() for f in findings)
    
    def test_clinical_safety_check(self):
        """Test ClinicalSafetyCheck."""
        check = ClinicalSafetyCheck()
        
        assert check.domain == SafetyDomain.CLINICAL
        assert check.name == "clinical_guideline_monitor"
        
        # Test with critical condition
        context = {
            'predicted_conditions': [
                {'name': 'stroke', 'confidence': 0.85}
            ],
            'medication_interaction_risk': 0.9,  # High risk
            'diagnostic_confidence': 0.4,        # Low confidence
            'phi_detected': True                 # PHI detected
        }
        findings = check.run(context)
        
        # Should detect multiple critical issues
        assert len(findings) >= 4
        assert any(f.severity == 'emergency' for f in findings)  # Critical condition
        assert any('medication' in f.message.lower() for f in findings)
        assert any('diagnostic' in f.message.lower() for f in findings)
        assert any('PHI' in f.message or 'phi' in f.message.lower() for f in findings)


class TestMemoryConsolidation:
    """Test memory consolidation system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def memory_store(self, temp_db):
        """Create memory store for testing."""
        return AgentMemoryStore(f"sqlite:///{temp_db}")
    
    @pytest.fixture
    def memory_consolidator(self, memory_store, temp_db):
        """Create memory consolidator for testing."""
        config = {'consolidation_db': temp_db.replace('.db', '_consolidation.db')}
        return MemoryConsolidator(memory_store, config)
    
    def test_consolidator_initialization(self, memory_consolidator):
        """Test MemoryConsolidator initializes correctly."""
        assert memory_consolidator.memory_store is not None
        assert memory_consolidator.consolidation_interval_hours == 6  # Default
        assert memory_consolidator.min_salience_threshold == 0.3      # Default
    
    def test_salience_score_calculation(self, memory_consolidator, memory_store):
        """Test salience score calculation."""
        # Create a session and store a memory
        session_id = memory_store.create_session("test_agent")
        memory_id = memory_store.store_memory(
            session_id=session_id,
            content="Patient shows cognitive decline with MMSE score of 22",
            memory_type="reasoning",
            importance=0.8
        )
        
        # Calculate salience score
        salience = memory_consolidator.calculate_salience_score(memory_id, session_id)
        
        assert 0.0 <= salience <= 1.0
        assert salience > 0.5  # Should be relatively high due to clinical content
    
    @patch('agent_memory.memory_consolidation.datetime')
    def test_consolidation_cycle(self, mock_datetime, memory_consolidator, memory_store):
        """Test a complete consolidation cycle."""
        # Set up mock time
        now = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now
        
        # Create session and memories
        session_id = memory_store.create_session("test_agent")
        
        # Store high-salience memory
        memory_store.store_memory(
            session_id=session_id,
            content="Critical: Patient with stroke symptoms requires immediate attention",
            memory_type="reasoning",
            importance=0.9
        )
        
        # Store low-salience memory
        memory_store.store_memory(
            session_id=session_id,
            content="General note about weather",
            memory_type="experience",
            importance=0.2
        )
        
        # Run consolidation
        results = memory_consolidator.run_consolidation_cycle(session_id)
        
        assert 'session_id' in results
        assert 'memories_processed' in results
        assert results['memories_processed'] >= 2
        assert 'memories_consolidated' in results
        assert 'semantic_clusters_created' in results
    
    def test_consolidation_summary(self, memory_consolidator):
        """Test consolidation summary generation."""
        summary = memory_consolidator.get_consolidation_summary(hours=24)
        
        assert 'summary_period_hours' in summary
        assert 'total_consolidation_events' in summary
        assert 'memory_store_distribution' in summary
        assert 'consolidation_interval_hours' in summary
        assert summary['consolidation_interval_hours'] == 6


class TestVisualizationAPI:
    """Test visualization API endpoints."""
    
    @pytest.fixture
    def api_client(self):
        """Create API client for testing."""
        config = {'testing': True}
        api = VisualizationAPI(config)
        api.app.config['TESTING'] = True
        return api.app.test_client()
    
    @pytest.fixture
    def mock_monitors(self):
        """Create mock monitoring systems."""
        safety_monitor = Mock()
        safety_monitor.get_safety_summary.return_value = {
            'monitoring_enabled': True,
            'total_events': 5,
            'overall_status': 'SAFE',
            'active_domains': 3
        }
        
        security_monitor = Mock()
        security_monitor.get_security_summary.return_value = {
            'monitoring_status': 'active',
            'total_security_events': 2
        }
        
        production_monitor = Mock()
        production_monitor.get_monitoring_summary.return_value = {
            'status': 'HEALTHY',
            'total_predictions': 100,
            'avg_accuracy': 0.85
        }
        
        memory_consolidator = Mock()
        memory_consolidator.get_consolidation_summary.return_value = {
            'total_consolidation_events': 3,
            'last_consolidation': datetime.now().isoformat()
        }
        
        return {
            'safety': safety_monitor,
            'security': security_monitor,
            'production': production_monitor,
            'memory': memory_consolidator
        }
    
    def test_health_check(self, api_client):
        """Test health check endpoint."""
        response = api_client.get('/api/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_dashboard_route(self, api_client):
        """Test main dashboard route."""
        response = api_client.get('/')
        assert response.status_code == 200
        assert b'DuetMind Adaptive Monitoring Dashboard' in response.data
    
    def test_safety_summary_no_monitor(self, api_client):
        """Test safety summary when monitor not initialized."""
        response = api_client.get('/api/safety/summary')
        assert response.status_code == 503
        
        data = response.get_json()
        assert 'error' in data
        assert data['monitoring_enabled'] is False
    
    def test_dashboard_overview_no_monitors(self, api_client):
        """Test dashboard overview with no monitors initialized."""
        response = api_client.get('/api/dashboard/overview')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'timestamp' in data
        assert 'system_status' in data
        assert 'components' in data
        # Should handle missing monitors gracefully
    
    def test_agent_interaction_graph(self, api_client):
        """Test agent interaction graph endpoint."""
        response = api_client.get('/api/agent/interaction-graph')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'timestamp' in data
        assert 'agents' in data
        assert 'edges' in data
        assert len(data['agents']) >= 3  # Mock data includes 3 agents
        assert len(data['edges']) >= 2   # Mock data includes 2 edges


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_safety_monitor_with_all_checks(self):
        """Test SafetyMonitor with all safety check types registered."""
        config = {'safety_monitoring_enabled': True}
        safety_monitor = SafetyMonitor(config)
        
        # Register all check types
        safety_monitor.register_safety_check(SystemSafetyCheck())
        safety_monitor.register_safety_check(DataSafetyCheck())
        safety_monitor.register_safety_check(ModelSafetyCheck())
        safety_monitor.register_safety_check(InteractionSafetyCheck())
        safety_monitor.register_safety_check(ClinicalSafetyCheck())
        
        # Context with issues in multiple domains
        context = {
            'response_time_ms': 1500,              # System issue
            'data_quality_score': 0.6,             # Data issue
            'accuracy': 0.7, 'baseline_accuracy': 0.85,  # Model issue
            'requests_per_minute': 20,              # Interaction issue
            'predicted_conditions': [{'name': 'stroke', 'confidence': 0.9}]  # Clinical issue
        }
        
        correlation_id = safety_monitor.create_correlation_id()
        findings = safety_monitor.run_safety_checks(context=context, correlation_id=correlation_id)
        
        # Should have findings from multiple domains
        domains_with_findings = set(f.domain for f in findings)
        assert len(domains_with_findings) >= 4  # At least 4 domains should have findings
        
        # Verify findings are stored
        stored_findings = safety_monitor.get_safety_findings(hours=1)
        assert len(stored_findings) >= len(findings)
        
        # Verify correlation chain
        chain = safety_monitor.get_correlation_chain(correlation_id)
        assert len(chain) == len(findings)
    
    def test_end_to_end_monitoring_workflow(self):
        """Test complete monitoring workflow from data input to visualization."""
        # Initialize all components
        config = {'safety_monitoring_enabled': True}
        safety_monitor = SafetyMonitor(config)
        
        # Register safety checks
        safety_monitor.register_safety_check(SystemSafetyCheck())
        safety_monitor.register_safety_check(ClinicalSafetyCheck())
        
        # Initialize API
        api_config = {'testing': True}
        viz_api = VisualizationAPI(api_config)
        viz_api.initialize_monitors(safety_monitor=safety_monitor)
        
        # Simulate monitoring data
        context = {
            'response_time_ms': 800,  # Moderate response time
            'predicted_conditions': [{'name': 'mild cognitive impairment', 'confidence': 0.7}]
        }
        
        # Run safety checks
        findings = safety_monitor.run_safety_checks(context=context)
        
        # Verify API can retrieve data
        client = viz_api.app.test_client()
        
        # Test safety summary
        response = client.get('/api/safety/summary')
        assert response.status_code == 200
        data = response.get_json()
        assert data['monitoring_enabled'] is True
        
        # Test findings retrieval
        response = client.get('/api/safety/findings')
        assert response.status_code == 200
        data = response.get_json()
        assert 'findings' in data
        assert data['count'] >= 0
        
        # Test dashboard overview
        response = client.get('/api/dashboard/overview')
        assert response.status_code == 200
        data = response.get_json()
        assert 'system_status' in data
        assert 'components' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])