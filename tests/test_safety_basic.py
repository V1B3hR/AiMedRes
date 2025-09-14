"""
Basic tests for Advanced Safety Monitoring System.

Tests core functionality without complex dependencies.
"""

import pytest
import tempfile
import os
import sys
from unittest.mock import Mock, patch
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security.safety_monitor import SafetyMonitor, SafetyDomain, SafetyFinding
from security.safety_checks import SystemSafetyCheck, DataSafetyCheck
from api.visualization_api import VisualizationAPI


class TestSafetyFinding:
    """Test SafetyFinding data structure."""
    
    def test_safety_finding_creation(self):
        """Test creating a SafetyFinding."""
        finding = SafetyFinding(
            domain=SafetyDomain.SYSTEM,
            check_name='test_check',
            severity='warning',
            message='Test message',
            value=0.8,
            threshold=0.5
        )
        
        assert finding.domain == SafetyDomain.SYSTEM
        assert finding.check_name == 'test_check'
        assert finding.severity == 'warning'
        assert finding.message == 'Test message'
        assert finding.value == 0.8
        assert finding.threshold == 0.5
        assert finding.timestamp is not None
        assert finding.metadata == {}
    
    def test_safety_finding_with_metadata(self):
        """Test SafetyFinding with metadata."""
        metadata = {'source': 'test', 'additional': 'data'}
        finding = SafetyFinding(
            domain=SafetyDomain.CLINICAL,
            check_name='clinical_check',
            severity='critical',
            message='Critical finding',
            metadata=metadata
        )
        
        assert finding.metadata == metadata
        assert finding.domain == SafetyDomain.CLINICAL


class TestSafetyDomains:
    """Test SafetyDomain enum."""
    
    def test_safety_domains_exist(self):
        """Test all expected safety domains exist."""
        expected_domains = ['SYSTEM', 'DATA', 'MODEL', 'INTERACTION', 'CLINICAL']
        
        for domain_name in expected_domains:
            assert hasattr(SafetyDomain, domain_name)
        
        # Test values
        assert SafetyDomain.SYSTEM.value == 'system'
        assert SafetyDomain.DATA.value == 'data'
        assert SafetyDomain.MODEL.value == 'model'
        assert SafetyDomain.INTERACTION.value == 'interaction'
        assert SafetyDomain.CLINICAL.value == 'clinical'


class TestSafetyMonitorBasic:
    """Basic tests for SafetyMonitor without database dependencies."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    def test_safety_monitor_initialization(self, temp_db):
        """Test SafetyMonitor initialization."""
        config = {
            'safety_monitoring_enabled': True,
            'safety_db_path': temp_db
        }
        
        monitor = SafetyMonitor(config)
        
        assert monitor.monitoring_enabled is True
        assert monitor.safety_db == temp_db
        assert monitor.safety_checks is not None
        assert monitor.correlation_chains is not None
        assert monitor.active_correlations is not None
    
    def test_safety_monitor_disabled(self, temp_db):
        """Test SafetyMonitor when disabled."""
        config = {
            'safety_monitoring_enabled': False,
            'safety_db_path': temp_db
        }
        
        monitor = SafetyMonitor(config)
        assert monitor.monitoring_enabled is False
        
        # Should return empty findings when disabled
        findings = monitor.run_safety_checks()
        assert findings == []
    
    def test_register_safety_check(self, temp_db):
        """Test registering safety checks."""
        config = {'safety_monitoring_enabled': True, 'safety_db_path': temp_db}
        monitor = SafetyMonitor(config)
        
        # Create mock safety check
        mock_check = Mock()
        mock_check.domain = SafetyDomain.SYSTEM
        mock_check.name = 'mock_check'
        
        monitor.register_safety_check(mock_check)
        
        assert len(monitor.safety_checks[SafetyDomain.SYSTEM]) == 1
        assert monitor.safety_checks[SafetyDomain.SYSTEM][0] == mock_check
    
    def test_create_correlation_id(self, temp_db):
        """Test correlation ID creation."""
        config = {'safety_monitoring_enabled': True, 'safety_db_path': temp_db}
        monitor = SafetyMonitor(config)
        
        metadata = {'test': 'data'}
        correlation_id = monitor.create_correlation_id(metadata)
        
        assert correlation_id is not None
        assert len(correlation_id) > 0
        assert correlation_id in monitor.active_correlations
        assert monitor.active_correlations[correlation_id]['metadata'] == metadata


class TestSystemSafetyCheck:
    """Test SystemSafetyCheck implementation."""
    
    def test_system_check_properties(self):
        """Test SystemSafetyCheck properties."""
        check = SystemSafetyCheck()
        
        assert check.domain == SafetyDomain.SYSTEM
        assert check.name == "system_resource_monitor"
    
    def test_system_check_no_issues(self):
        """Test SystemSafetyCheck with normal context."""
        check = SystemSafetyCheck()
        context = {'response_time_ms': 100}  # Normal response time
        
        findings = check.run(context)
        
        # Should have no findings for normal metrics
        response_time_findings = [f for f in findings if 'response time' in f.message.lower()]
        assert len(response_time_findings) == 0
    
    def test_system_check_high_response_time(self):
        """Test SystemSafetyCheck with high response time."""
        check = SystemSafetyCheck()
        context = {'response_time_ms': 2000}  # High response time
        
        findings = check.run(context)
        
        # Should detect high response time
        response_time_findings = [f for f in findings if 'response time' in f.message.lower()]
        assert len(response_time_findings) >= 1
        
        finding = response_time_findings[0]
        assert finding.domain == SafetyDomain.SYSTEM
        assert finding.severity in ['warning', 'critical']
        assert finding.value == 2000
    
    @patch('security.safety_checks.psutil.cpu_percent')
    def test_system_check_high_cpu(self, mock_cpu_percent):
        """Test SystemSafetyCheck with high CPU usage."""
        mock_cpu_percent.return_value = 95.0  # Very high CPU
        
        check = SystemSafetyCheck()
        findings = check.run({})
        
        # Should detect high CPU usage
        cpu_findings = [f for f in findings if 'cpu usage' in f.message.lower()]
        assert len(cpu_findings) >= 1
        
        finding = cpu_findings[0]
        assert finding.domain == SafetyDomain.SYSTEM
        assert finding.severity == 'critical'
        assert finding.value == 95.0


class TestDataSafetyCheck:
    """Test DataSafetyCheck implementation."""
    
    def test_data_check_properties(self):
        """Test DataSafetyCheck properties."""
        check = DataSafetyCheck()
        
        assert check.domain == SafetyDomain.DATA
        assert check.name == "data_quality_monitor"
    
    def test_data_check_low_quality(self):
        """Test DataSafetyCheck with low data quality."""
        check = DataSafetyCheck()
        context = {
            'data_quality_score': 0.6,  # Below critical threshold
            'missing_data_ratio': 0.3,  # High missing data
            'duplicate_ratio': 0.2      # High duplicates
        }
        
        findings = check.run(context)
        
        # Should detect multiple data quality issues
        assert len(findings) >= 3
        
        # Check for quality score finding
        quality_findings = [f for f in findings if 'quality score' in f.message.lower()]
        assert len(quality_findings) >= 1
        assert quality_findings[0].severity == 'critical'
        
        # Check for missing data finding
        missing_findings = [f for f in findings if 'missing' in f.message.lower()]
        assert len(missing_findings) >= 1
        
        # Check for duplicate finding
        duplicate_findings = [f for f in findings if 'duplicate' in f.message.lower()]
        assert len(duplicate_findings) >= 1
    
    def test_data_check_schema_violations(self):
        """Test DataSafetyCheck with schema violations."""
        check = DataSafetyCheck()
        context = {
            'schema_violations': ['missing_column', 'invalid_type', 'constraint_violation']
        }
        
        findings = check.run(context)
        
        # Should detect schema violations
        schema_findings = [f for f in findings if 'schema' in f.message.lower()]
        assert len(schema_findings) >= 1
        
        finding = schema_findings[0]
        assert finding.domain == SafetyDomain.DATA
        assert finding.value == 3  # Number of violations


class TestVisualizationAPIBasic:
    """Basic tests for VisualizationAPI."""
    
    def test_api_initialization(self):
        """Test VisualizationAPI initialization."""
        config = {'testing': True}
        api = VisualizationAPI(config)
        
        assert api.config == config
        assert api.app is not None
        assert api.safety_monitor is None  # Not initialized yet
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        config = {'testing': True}
        api = VisualizationAPI(config)
        api.app.config['TESTING'] = True
        
        client = api.app.test_client()
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_dashboard_route(self):
        """Test main dashboard route."""
        config = {'testing': True}
        api = VisualizationAPI(config)
        api.app.config['TESTING'] = True
        
        client = api.app.test_client()
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'DuetMind Adaptive Monitoring Dashboard' in response.data
    
    def test_safety_summary_no_monitor(self):
        """Test safety summary endpoint when monitor not initialized."""
        config = {'testing': True}
        api = VisualizationAPI(config)
        api.app.config['TESTING'] = True
        
        client = api.app.test_client()
        response = client.get('/api/safety/summary')
        
        assert response.status_code == 503
        data = response.get_json()
        assert 'error' in data
        assert data['monitoring_enabled'] is False
    
    def test_dashboard_overview(self):
        """Test dashboard overview endpoint."""
        config = {'testing': True}
        api = VisualizationAPI(config)
        api.app.config['TESTING'] = True
        
        client = api.app.test_client()
        response = client.get('/api/dashboard/overview')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'timestamp' in data
        assert 'system_status' in data
        assert 'components' in data


class TestIntegrationBasic:
    """Basic integration tests."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    def test_safety_monitor_with_system_check(self, temp_db):
        """Test SafetyMonitor with SystemSafetyCheck integration."""
        config = {
            'safety_monitoring_enabled': True,
            'safety_db_path': temp_db
        }
        
        monitor = SafetyMonitor(config)
        system_check = SystemSafetyCheck()
        monitor.register_safety_check(system_check)
        
        # Test with high response time context
        context = {'response_time_ms': 1500}
        correlation_id = monitor.create_correlation_id()
        
        findings = monitor.run_safety_checks(
            domain=SafetyDomain.SYSTEM,
            context=context,
            correlation_id=correlation_id
        )
        
        # Should have findings for high response time
        assert len(findings) >= 1
        assert any(f.domain == SafetyDomain.SYSTEM for f in findings)
        assert any('response time' in f.message.lower() for f in findings)
        
        # Verify correlation tracking
        assert correlation_id in monitor.active_correlations
        chain = monitor.get_correlation_chain(correlation_id)
        assert len(chain) == len(findings)
    
    def test_multiple_safety_checks(self, temp_db):
        """Test multiple safety checks working together."""
        config = {
            'safety_monitoring_enabled': True,
            'safety_db_path': temp_db
        }
        
        monitor = SafetyMonitor(config)
        
        # Register multiple checks
        monitor.register_safety_check(SystemSafetyCheck())
        monitor.register_safety_check(DataSafetyCheck())
        
        # Context with issues in multiple domains
        context = {
            'response_time_ms': 1200,           # System issue
            'data_quality_score': 0.65,        # Data issue
            'missing_data_ratio': 0.2          # Data issue
        }
        
        findings = monitor.run_safety_checks(context=context)
        
        # Should have findings from multiple domains
        domains_with_findings = set(f.domain for f in findings)
        assert SafetyDomain.SYSTEM in domains_with_findings
        assert SafetyDomain.DATA in domains_with_findings
        
        # Verify specific findings
        system_findings = [f for f in findings if f.domain == SafetyDomain.SYSTEM]
        data_findings = [f for f in findings if f.domain == SafetyDomain.DATA]
        
        assert len(system_findings) >= 1
        assert len(data_findings) >= 2  # Quality score + missing data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])