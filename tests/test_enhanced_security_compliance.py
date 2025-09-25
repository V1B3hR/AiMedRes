#!/usr/bin/env python3
"""
Comprehensive tests for enhanced security and compliance systems.

Tests the newly implemented:
- HIPAA-compliant audit logging
- Enhanced medical data encryption
- Performance monitoring for clinical requirements
- AI safety and human oversight systems
- FDA regulatory compliance enhancements
"""

import pytest
import time
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import tempfile
import os

# Import our new modules
from security.hipaa_audit import (
    HIPAAAuditLogger, AccessType, AuditEvent, ComplianceStatus,
    get_audit_logger, audit_phi_access
)
from security.encryption import DataEncryption
from security.performance_monitor import (
    ClinicalPerformanceMonitor, ClinicalPriority, AlertLevel,
    PerformanceThresholds, monitor_performance, get_performance_monitor
)
from security.ai_safety import (
    ClinicalAISafetyMonitor, RiskLevel, ConfidenceLevel, SafetyAction,
    SafetyThresholds, get_safety_monitor
)


class TestHIPAAAuditLogging:
    """Test HIPAA-compliant audit logging system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.audit_logger = HIPAAAuditLogger(
            audit_db_path=self.temp_db.name,
            encryption_key="test_key_123"
        )
    
    def teardown_method(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_phi_access_logging(self):
        """Test PHI access logging with audit trail"""
        # Log PHI access
        audit_id = self.audit_logger.log_phi_access(
            user_id="doctor123",
            user_role="physician",
            patient_id="patient456", 
            access_type=AccessType.READ,
            resource="medical_record",
            purpose="treatment",
            outcome="SUCCESS",
            ip_address="192.168.1.100",
            session_id="session_123"
        )
        
        assert audit_id is not None
        assert len(audit_id) > 0
        
        # Retrieve audit trail
        trail = self.audit_logger.get_audit_trail(patient_id="patient456")
        assert len(trail) == 1
        assert trail[0]['user_id'] == "doctor123"
        assert trail[0]['event_type'] == AuditEvent.PHI_ACCESS.value
        assert trail[0]['outcome'] == "SUCCESS"
    
    def test_clinical_decision_logging(self):
        """Test clinical decision audit logging"""
        decision_data = {
            'diagnosis': 'hypertension',
            'confidence': 0.95,
            'treatment_recommendation': 'medication_adjustment'
        }
        
        audit_id = self.audit_logger.log_clinical_decision(
            user_id="doctor123",
            user_role="physician", 
            patient_id="patient456",
            decision_data=decision_data,
            ai_confidence=0.95,
            human_override=False
        )
        
        assert audit_id is not None
        
        # Check audit trail
        trail = self.audit_logger.get_audit_trail(patient_id="patient456")
        assert len(trail) == 1
        assert trail[0]['event_type'] == AuditEvent.CLINICAL_DECISION.value
        assert 'ai_confidence' in trail[0]['additional_data']
    
    def test_compliance_violation_detection(self):
        """Test compliance violation detection"""
        # Log high-risk access during off-hours
        with patch('security.hipaa_audit.datetime') as mock_datetime:
            # Set time to 2 AM (off-hours)
            mock_datetime.now.return_value = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            audit_id = self.audit_logger.log_phi_access(
                user_id="admin_user",
                user_role="admin",
                patient_id="patient456",
                access_type=AccessType.EXPORT,
                resource="patient_database",
                purpose="system_maintenance",
                ip_address="10.0.0.5"
            )
        
        # Check for compliance violations
        violations = self.audit_logger.get_compliance_violations(resolved=False)
        assert len(violations) > 0
        assert violations[0]['severity'] in ['WARNING', 'VIOLATION']
    
    def test_audit_data_encryption(self):
        """Test that sensitive audit data is encrypted"""
        audit_id = self.audit_logger.log_phi_access(
            user_id="doctor123",
            user_role="physician",
            patient_id="patient456",
            access_type=AccessType.READ,
            resource="sensitive_data",
            purpose="treatment",
            additional_data={
                'sensitive_field': 'confidential_value',
                'patient_ssn': '123-45-6789'
            }
        )
        
        # Verify data is encrypted in database
        import sqlite3
        conn = sqlite3.connect(self.audit_logger.audit_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT additional_data_encrypted FROM hipaa_audit_log WHERE audit_id = ?', (audit_id,))
        encrypted_data = cursor.fetchone()[0]
        conn.close()
        
        # Should be encrypted (not readable as plain text)
        assert b'confidential_value' not in encrypted_data
        assert b'123-45-6789' not in encrypted_data
    
    def test_compliance_report_generation(self):
        """Test compliance report generation"""
        # Generate some test data
        start_date = datetime.now(timezone.utc) - timedelta(hours=1)
        end_date = datetime.now(timezone.utc)
        
        for i in range(5):
            self.audit_logger.log_phi_access(
                user_id=f"user{i}",
                user_role="physician",
                patient_id=f"patient{i}",
                access_type=AccessType.READ,
                resource="medical_record",
                purpose="treatment"
            )
        
        # Generate report
        report = self.audit_logger.generate_compliance_report(start_date, end_date)
        
        assert 'summary' in report
        assert report['summary']['total_events'] == 5
        assert report['summary']['unique_users'] == 5
        assert 'compliance_rate' in report
        assert report['compliance_rate'] >= 0


class TestEnhancedEncryption:
    """Test enhanced medical data encryption"""
    
    def setup_method(self):
        """Set up test environment"""
        self.encryption = DataEncryption(master_password="test_password_123")
    
    def test_phi_data_encryption(self):
        """Test PHI data encryption with compliance logging"""
        phi_data = {
            'patient_id': 'P123456',
            'name': 'John Doe',
            'ssn': '123-45-6789',
            'address': '123 Main St',
            'phone': '555-123-4567',
            'email': 'john@example.com',
            'diagnosis': 'hypertension',
            'treatment': 'medication'
        }
        
        # Encrypt PHI data
        encrypted_data = self.encryption.encrypt_phi_data(
            phi_data=phi_data,
            patient_id='P123456',
            purpose='clinical_analysis',
            audit_log=False  # Disable for test
        )
        
        # Verify sensitive fields are encrypted
        assert encrypted_data['patient_id'] != 'P123456'
        assert encrypted_data['name'] != 'John Doe'
        assert encrypted_data['ssn'] != '123-45-6789'
        
        # Verify non-sensitive fields remain unchanged
        assert encrypted_data['diagnosis'] == 'hypertension'
        assert encrypted_data['treatment'] == 'medication'
        
        # Verify encryption metadata
        assert '_encryption_metadata' in encrypted_data
        metadata = encrypted_data['_encryption_metadata']
        assert 'encrypted_fields' in metadata
        assert 'patient_id' in metadata['encrypted_fields']
        assert 'name' in metadata['encrypted_fields']
    
    def test_phi_data_decryption(self):
        """Test PHI data decryption with audit logging"""
        phi_data = {
            'patient_id': 'P123456',
            'name': 'John Doe',
            'medical_data': 'sensitive medical information'
        }
        
        # Encrypt then decrypt
        encrypted_data = self.encryption.encrypt_phi_data(
            phi_data=phi_data,
            patient_id='P123456',
            audit_log=False
        )
        
        decrypted_data = self.encryption.decrypt_phi_data(
            encrypted_phi_data=encrypted_data,
            patient_id='P123456',
            purpose='treatment',
            user_id='doctor123',
            audit_log=False
        )
        
        # Verify decryption
        assert decrypted_data['patient_id'] == 'P123456'
        assert decrypted_data['name'] == 'John Doe'
        assert decrypted_data['medical_data'] == 'sensitive medical information'
        assert '_encryption_metadata' not in decrypted_data
    
    def test_data_integrity_validation(self):
        """Test data integrity validation"""
        test_data = {
            'field1': 'value1',
            'field2': 'value2',
            '_encryption_metadata': {
                'encryption_timestamp': time.time(),
                'encryption_version': '2.0_hipaa_compliant'
            }
        }
        
        # Test valid data
        assert self.encryption.validate_data_integrity(test_data) == True
        
        # Test invalid data (missing metadata)
        invalid_data = {'field1': 'value1'}
        assert self.encryption.validate_data_integrity(invalid_data) == False


class TestClinicalPerformanceMonitor:
    """Test clinical performance monitoring system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.thresholds = PerformanceThresholds(
            emergency_max_ms=20.0,
            critical_max_ms=50.0,
            urgent_max_ms=100.0
        )
        self.monitor = ClinicalPerformanceMonitor(
            thresholds=self.thresholds,
            enable_audit_integration=False
        )
        self.monitor.start_monitoring()
    
    def teardown_method(self):
        """Clean up test environment"""
        self.monitor.stop_monitoring()
    
    def test_performance_recording(self):
        """Test basic performance recording"""
        # Record a fast operation
        self.monitor.record_operation(
            operation="ai_diagnosis",
            response_time_ms=45.0,
            clinical_priority=ClinicalPriority.CRITICAL,
            success=True,
            user_id="doctor123",
            patient_id="patient456"
        )
        
        # Wait for processing
        time.sleep(0.2)
        
        # Get performance summary
        summary = self.monitor.get_performance_summary(
            priority=ClinicalPriority.CRITICAL,
            hours_back=1
        )
        
        assert summary['total_operations'] == 1
        assert summary['avg_response_time_ms'] == 45.0
        assert summary['violations_count'] == 0
    
    def test_performance_violation_detection(self):
        """Test performance violation detection"""
        alert_triggered = False
        violation_data = None
        
        def alert_callback(alert):
            nonlocal alert_triggered, violation_data
            if alert['alert_type'] == 'PERFORMANCE_VIOLATION':
                alert_triggered = True
                violation_data = alert
        
        self.monitor.add_alert_callback(alert_callback)
        
        # Record slow operation that violates threshold
        self.monitor.record_operation(
            operation="slow_diagnosis",
            response_time_ms=150.0,  # Exceeds 100ms urgent threshold
            clinical_priority=ClinicalPriority.URGENT,
            success=True,
            user_id="doctor123",
            patient_id="patient456"
        )
        
        # Wait for processing
        time.sleep(0.3)
        
        assert alert_triggered == True
        assert violation_data is not None
        assert violation_data['data']['response_time_ms'] == 150.0
    
    def test_clinical_priority_thresholds(self):
        """Test different thresholds for clinical priorities"""
        test_cases = [
            (ClinicalPriority.EMERGENCY, 25.0, True),    # Should violate 20ms threshold
            (ClinicalPriority.CRITICAL, 30.0, False),    # Should not violate 50ms threshold
            (ClinicalPriority.URGENT, 120.0, True),      # Should violate 100ms threshold
            (ClinicalPriority.ROUTINE, 150.0, False),    # Should not violate 200ms threshold
        ]
        
        alerts_received = []
        
        def alert_callback(alert):
            if alert['alert_type'] == 'PERFORMANCE_VIOLATION':
                alerts_received.append(alert)
        
        self.monitor.add_alert_callback(alert_callback)
        
        for priority, response_time, should_violate in test_cases:
            self.monitor.record_operation(
                operation=f"test_{priority.value}",
                response_time_ms=response_time,
                clinical_priority=priority,
                success=True
            )
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check violations
        violations = [a for a in alerts_received if a['alert_type'] == 'PERFORMANCE_VIOLATION']
        expected_violations = sum(1 for _, _, should_violate in test_cases if should_violate)
        
        assert len(violations) == expected_violations
    
    def test_optimization_recommendations(self):
        """Test performance optimization recommendations"""
        # Generate some performance data that should trigger recommendations
        for i in range(10):
            self.monitor.record_operation(
                operation="slow_operation",
                response_time_ms=90.0,  # Near threshold
                clinical_priority=ClinicalPriority.URGENT
            )
        
        time.sleep(0.3)
        
        recommendations = self.monitor.get_optimization_recommendations()
        
        # Should have recommendations for operations approaching threshold
        urgent_recommendations = [r for r in recommendations 
                                if 'URGENT' in r.get('title', '')]
        assert len(urgent_recommendations) > 0
        
        # Check recommendation structure
        for rec in recommendations:
            assert 'type' in rec
            assert 'priority' in rec
            assert 'title' in rec
            assert 'suggested_actions' in rec


class TestAISafetyMonitor:
    """Test AI safety and human oversight system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.safety_thresholds = SafetyThresholds(
            confidence_threshold_high_risk=0.95,
            confidence_threshold_moderate_risk=0.85,
            critical_condition_confidence_threshold=0.98
        )
        self.safety_monitor = ClinicalAISafetyMonitor(
            thresholds=self.safety_thresholds,
            enable_audit_logging=False
        )
        self.safety_monitor.start_monitoring()
    
    def teardown_method(self):
        """Clean up test environment"""
        self.safety_monitor.stop_monitoring()
    
    def test_low_confidence_decision(self):
        """Test handling of low confidence AI decisions"""
        clinical_context = {
            'patient_age': 45,
            'primary_condition': 'chest_pain',
            'condition_severity': 'MODERATE'
        }
        
        ai_recommendation = {
            'primary_recommendation': 'further_testing',
            'confidence': 0.65
        }
        
        decision = self.safety_monitor.assess_ai_decision(
            model_version='v1.0',
            user_id='doctor123',
            patient_id='patient456',
            clinical_context=clinical_context,
            ai_recommendation=ai_recommendation,
            confidence_score=0.65
        )
        
        assert decision.confidence_score == 0.65
        assert decision.computed_risk_level in [RiskLevel.LOW, RiskLevel.MODERATE]
        assert 'LOW_CONFIDENCE' in decision.risk_factors
    
    def test_high_risk_decision_requires_oversight(self):
        """Test that high-risk decisions require human oversight"""
        clinical_context = {
            'patient_age': 8,  # Pediatric patient
            'primary_condition': 'cardiac_arrest',
            'condition_severity': 'CRITICAL',
            'emergency_case': True
        }
        
        ai_recommendation = {
            'primary_recommendation': 'immediate_surgery',
            'treatment_type': 'SURGICAL_INTERVENTION'
        }
        
        decision = self.safety_monitor.assess_ai_decision(
            model_version='v1.0',
            user_id='doctor123',
            patient_id='patient456',
            clinical_context=clinical_context,
            ai_recommendation=ai_recommendation,
            confidence_score=0.88
        )
        
        assert decision.computed_risk_level == RiskLevel.CRITICAL
        assert decision.human_oversight_required == True
        assert decision.safety_action in [SafetyAction.REQUIRE_APPROVAL, SafetyAction.ESCALATE]
        assert 'PEDIATRIC_PATIENT' in decision.risk_factors
        assert 'CRITICAL_CONDITION' in decision.risk_factors
        assert 'EMERGENCY_CASE' in decision.risk_factors
    
    def test_safety_alert_generation(self):
        """Test safety alert generation"""
        alerts_received = []
        
        def alert_callback(alert):
            alerts_received.append(alert)
        
        self.safety_monitor.add_safety_alert_callback(alert_callback)
        
        # Create a very low confidence decision
        clinical_context = {'patient_age': 65, 'condition': 'complex_case'}
        ai_recommendation = {'recommendation': 'uncertain_diagnosis'}
        
        decision = self.safety_monitor.assess_ai_decision(
            model_version='v1.0',
            user_id='doctor123', 
            patient_id='patient456',
            clinical_context=clinical_context,
            ai_recommendation=ai_recommendation,
            confidence_score=0.35  # Very low confidence
        )
        
        time.sleep(0.2)  # Wait for alert processing
        
        # Should trigger low confidence alert
        low_confidence_alerts = [a for a in alerts_received 
                               if a['type'] == 'LOW_CONFIDENCE_ALERT']
        assert len(low_confidence_alerts) > 0
        assert low_confidence_alerts[0]['severity'] == 'HIGH'
    
    def test_human_oversight_workflow(self):
        """Test human oversight request and decision workflow"""
        oversight_requests = []
        
        def oversight_callback(request):
            oversight_requests.append(request)
        
        self.safety_monitor.add_human_oversight_callback(oversight_callback)
        
        # Create high-risk decision requiring oversight
        clinical_context = {
            'patient_age': 75,
            'primary_condition': 'heart_failure', 
            'condition_severity': 'CRITICAL'
        }
        
        ai_recommendation = {
            'primary_recommendation': 'high_risk_medication',
            'treatment_type': 'HIGH_RISK_MEDICATION'
        }
        
        decision = self.safety_monitor.assess_ai_decision(
            model_version='v1.0',
            user_id='doctor123',
            patient_id='patient456',
            clinical_context=clinical_context,
            ai_recommendation=ai_recommendation,
            confidence_score=0.82
        )
        
        time.sleep(0.1)
        
        # Should have generated oversight request
        assert len(oversight_requests) > 0
        assert decision.human_oversight_required == True
        
        # Submit human decision
        human_decision = {
            'final_decision': 'APPROVED_WITH_MODIFICATIONS',
            'modifications': 'Reduce dosage by 50%',
            'rationale': 'Patient age and comorbidities warrant caution'
        }
        
        success = self.safety_monitor.submit_human_decision(
            decision_id=decision.decision_id,
            reviewer_id='senior_physician_123',
            human_decision=human_decision,
            safety_notes='Approved with reduced risk modifications'
        )
        
        assert success == True
        assert decision.human_decision == human_decision
        assert decision.final_outcome == 'APPROVED_WITH_MODIFICATIONS'
    
    def test_safety_summary_generation(self):
        """Test safety monitoring summary generation"""
        # Generate various types of decisions
        test_cases = [
            (0.95, RiskLevel.LOW, False),
            (0.75, RiskLevel.MODERATE, True),
            (0.45, RiskLevel.HIGH, True),
            (0.92, RiskLevel.LOW, False),
            (0.88, RiskLevel.MODERATE, True)
        ]
        
        for confidence, expected_min_risk, should_need_oversight in test_cases:
            self.safety_monitor.assess_ai_decision(
                model_version='v1.0',
                user_id='doctor123',
                patient_id=f'patient_{confidence}',
                clinical_context={'patient_age': 45},
                ai_recommendation={'recommendation': 'test'},
                confidence_score=confidence
            )
        
        time.sleep(0.2)
        
        # Get safety summary
        summary = self.safety_monitor.get_safety_summary(hours_back=1)
        
        assert summary['total_decisions'] == len(test_cases)
        assert summary['human_oversight_decisions'] > 0
        assert summary['average_confidence'] > 0
        assert 'safety_status' in summary
        assert summary['safety_status'] in ['NORMAL', 'MONITORED', 'ELEVATED', 'CONCERNING']
    
    def test_risk_level_calculation(self):
        """Test risk level calculation logic"""
        # Test critical condition with low confidence
        critical_context = {
            'patient_age': 5,  # Pediatric
            'condition_severity': 'CRITICAL',
            'emergency_case': True
        }
        
        decision = self.safety_monitor.assess_ai_decision(
            model_version='v1.0',
            user_id='doctor123',
            patient_id='patient456',
            clinical_context=critical_context,
            ai_recommendation={'recommendation': 'emergency_treatment'},
            confidence_score=0.70
        )
        
        assert decision.computed_risk_level == RiskLevel.CRITICAL
        
        # Test normal case with high confidence
        normal_context = {
            'patient_age': 35,
            'condition_severity': 'MILD'
        }
        
        decision2 = self.safety_monitor.assess_ai_decision(
            model_version='v1.0',
            user_id='doctor123',
            patient_id='patient789',
            clinical_context=normal_context,
            ai_recommendation={'recommendation': 'routine_treatment'},
            confidence_score=0.95
        )
        
        assert decision2.computed_risk_level in [RiskLevel.MINIMAL, RiskLevel.LOW]


class TestIntegratedSystems:
    """Test integration between all security and compliance systems"""
    
    def test_performance_and_safety_integration(self):
        """Test integration between performance monitor and safety monitor"""
        # This would test that performance violations are properly
        # integrated with safety assessments
        pass
    
    def test_audit_logging_integration(self):
        """Test that all systems properly integrate with HIPAA audit logging"""
        # This would test end-to-end audit trail from all systems
        pass
    
    def test_regulatory_compliance_integration(self):
        """Test integration with FDA regulatory compliance tracking"""
        # This would test that all safety and performance data
        # is properly captured for regulatory submissions
        pass


if __name__ == "__main__":
    # Run specific test classes
    pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])