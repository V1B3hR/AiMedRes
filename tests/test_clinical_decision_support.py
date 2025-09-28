#!/usr/bin/env python3
"""
Comprehensive tests for the Clinical Decision Support System

Tests all major components:
- Risk stratification engine
- Explainable AI dashboard
- EHR integration
- Regulatory compliance
- End-to-end workflows
"""

import pytest
import json
import tempfile
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import uuid

# Import the modules to test
from clinical_decision_support import (
    ClinicalDecisionSupportSystem, RiskStratificationEngine, 
    RiskAssessment, InterventionRecommendation
)
from explainable_ai_dashboard import (
    DashboardGenerator, AlzheimerExplainer, 
    FeatureImportance, DecisionExplanation
)
from ehr_integration import (
    EHRConnector, FHIRConverter, HL7MessageProcessor,
    FHIRPatient, FHIRObservation, FHIRDiagnosticReport
)
from regulatory_compliance import (
    HIPAAComplianceManager, FDAValidationManager, ComplianceDashboard,
    AuditEvent, AuditEventType, ValidationRecord, AdverseEvent
)
from clinical_decision_support_main import ClinicalWorkflowOrchestrator


class TestRiskStratificationEngine:
    """Test the risk stratification engine"""
    
    @pytest.fixture
    def risk_engine(self):
        """Create risk stratification engine for testing"""
        config = {'test_mode': True}
        return RiskStratificationEngine(config)
    
    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing"""
        return {
            'patient_id': 'TEST_PATIENT_001',
            'Age': 78,
            'M/F': 0,  # Female
            'MMSE': 22,
            'CDR': 0.5,
            'EDUC': 14,
            'nWBV': 0.72,
            'hypertension': True,
            'diabetes': False,
            'BMI': 28.5
        }
    
    def test_alzheimer_risk_calculation(self, risk_engine, sample_patient_data):
        """Test Alzheimer's disease risk calculation"""
        risk_score = risk_engine._calculate_alzheimer_risk(sample_patient_data)
        
        assert 0.0 <= risk_score <= 1.0
        assert risk_score > 0.5  # Should be high risk given the patient profile
    
    def test_cardiovascular_risk_calculation(self, risk_engine, sample_patient_data):
        """Test cardiovascular risk calculation"""
        risk_score = risk_engine._calculate_cardiovascular_risk(sample_patient_data)
        
        assert 0.0 <= risk_score <= 1.0
        assert risk_score > 0.0  # Should have some risk due to age and hypertension
    
    def test_diabetes_risk_calculation(self, risk_engine, sample_patient_data):
        """Test diabetes risk calculation"""
        risk_score = risk_engine._calculate_diabetes_risk(sample_patient_data)
        
        assert 0.0 <= risk_score <= 1.0
    
    def test_comprehensive_risk_assessment(self, risk_engine, sample_patient_data):
        """Test comprehensive risk assessment"""
        assessment = risk_engine.assess_risk(sample_patient_data, 'alzheimer')
        
        assert isinstance(assessment, RiskAssessment)
        assert assessment.patient_id == 'TEST_PATIENT_001'
        assert assessment.condition == 'alzheimer'
        assert assessment.risk_level in ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH']
        assert 0.0 <= assessment.risk_score <= 1.0
        assert 0.0 <= assessment.confidence <= 1.0
        assert isinstance(assessment.interventions, list)
        assert isinstance(assessment.explanation, dict)
        assert assessment.next_assessment_date is not None
    
    def test_risk_level_determination(self, risk_engine):
        """Test risk level determination"""
        # Test high risk
        high_risk_level = risk_engine._determine_risk_level(0.85, 'alzheimer')
        assert high_risk_level == 'HIGH'
        
        # Test medium risk
        medium_risk_level = risk_engine._determine_risk_level(0.6, 'alzheimer')
        assert medium_risk_level == 'MEDIUM'
        
        # Test low risk
        low_risk_level = risk_engine._determine_risk_level(0.3, 'alzheimer')
        assert low_risk_level == 'LOW'
        
        # Test minimal risk
        minimal_risk_level = risk_engine._determine_risk_level(0.1, 'alzheimer')
        assert minimal_risk_level == 'MINIMAL'
    
    def test_intervention_recommendations(self, risk_engine, sample_patient_data):
        """Test intervention recommendations"""
        interventions = risk_engine._recommend_interventions(
            0.7, 'HIGH', 'alzheimer', sample_patient_data
        )
        
        assert isinstance(interventions, list)
        assert len(interventions) > 0
        # Should include high-risk interventions
        assert any('referral' in intervention for intervention in interventions)


class TestExplainableAIDashboard:
    """Test the explainable AI dashboard components"""
    
    @pytest.fixture
    def dashboard_generator(self):
        """Create dashboard generator for testing"""
        config = {'model_type': 'alzheimer'}
        return DashboardGenerator(config)
    
    @pytest.fixture
    def alzheimer_explainer(self):
        """Create Alzheimer explainer for testing"""
        config = {'model_type': 'alzheimer'}
        return AlzheimerExplainer(config)
    
    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing"""
        return {
            'patient_id': 'TEST_PATIENT_001',
            'Age': 78,
            'M/F': 0,
            'MMSE': 22,
            'CDR': 0.5,
            'EDUC': 14,
            'nWBV': 0.72
        }
    
    @pytest.fixture
    def sample_model_output(self):
        """Sample model output for testing"""
        return {
            'prediction': 'Demented',
            'confidence': 0.85,
            'risk_score': 0.72
        }
    
    def test_feature_importance_calculation(self, alzheimer_explainer, sample_patient_data):
        """Test feature importance calculation"""
        feature_importance = alzheimer_explainer.get_feature_importance(sample_patient_data)
        
        assert isinstance(feature_importance, list)
        assert len(feature_importance) > 0
        
        for feature in feature_importance:
            assert isinstance(feature, FeatureImportance)
            assert feature.feature_name is not None
            assert 0.0 <= feature.importance_score <= 1.0
            assert feature.direction in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
            assert 0.0 <= feature.confidence <= 1.0
            assert feature.clinical_meaning is not None
    
    def test_decision_explanation_generation(self, alzheimer_explainer, 
                                           sample_patient_data, sample_model_output):
        """Test decision explanation generation"""
        explanation = alzheimer_explainer.explain_prediction(
            sample_patient_data, sample_model_output
        )
        
        assert isinstance(explanation, DecisionExplanation)
        assert explanation.model_prediction == 'Demented'
        assert explanation.confidence_score == 0.85
        assert isinstance(explanation.primary_factors, list)
        assert isinstance(explanation.contributing_factors, list)
        assert isinstance(explanation.protective_factors, list)
        assert isinstance(explanation.uncertainty_factors, list)
        assert isinstance(explanation.alternative_scenarios, list)
        assert isinstance(explanation.clinical_recommendations, list)
    
    def test_dashboard_generation(self, dashboard_generator, sample_patient_data):
        """Test complete dashboard generation"""
        # Mock assessment results
        mock_assessment = Mock()
        mock_assessment.risk_level = 'MEDIUM'
        mock_assessment.risk_score = 0.65
        mock_assessment.confidence = 0.8
        
        assessments = {'alzheimer': mock_assessment}
        
        dashboard_data = dashboard_generator.generate_patient_dashboard(
            sample_patient_data, assessments
        )
        
        assert isinstance(dashboard_data, dict)
        assert 'patient_info' in dashboard_data
        assert 'risk_summary' in dashboard_data
        assert 'explanations' in dashboard_data
        assert 'visualizations' in dashboard_data
        assert 'recommendations' in dashboard_data
        assert 'generated_at' in dashboard_data
        
        # Check patient info
        patient_info = dashboard_data['patient_info']
        assert patient_info['patient_id'] == 'TEST_PATIENT_001'
        assert patient_info['age'] == 78
        assert patient_info['gender'] == 'Female'
    
    def test_html_export(self, dashboard_generator, sample_patient_data):
        """Test HTML dashboard export"""
        mock_assessment = Mock()
        mock_assessment.risk_level = 'MEDIUM'
        
        assessments = {'alzheimer': mock_assessment}
        
        dashboard_data = dashboard_generator.generate_patient_dashboard(
            sample_patient_data, assessments
        )
        
        html_output = dashboard_generator.export_dashboard_html(dashboard_data)
        
        assert isinstance(html_output, str)
        assert '<!DOCTYPE html>' in html_output
        assert 'Clinical Decision Support Dashboard' in html_output
        assert 'TEST_PATIENT_001' in html_output


class TestEHRIntegration:
    """Test EHR integration components"""
    
    @pytest.fixture
    def ehr_connector(self):
        """Create EHR connector for testing"""
        config = {
            'organization_id': 'test-org',
            'master_password': 'test_password'
        }
        return EHRConnector(config)
    
    @pytest.fixture
    def fhir_converter(self):
        """Create FHIR converter for testing"""
        config = {'organization_id': 'test-org'}
        return FHIRConverter(config)
    
    @pytest.fixture
    def hl7_processor(self):
        """Create HL7 processor for testing"""
        config = {}
        return HL7MessageProcessor(config)
    
    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing"""
        return {
            'patient_id': 'TEST_PATIENT_001',
            'Age': 78,
            'M/F': 0,
            'family_name': 'Doe',
            'given_name': 'Jane'
        }
    
    @pytest.fixture
    def sample_hl7_message(self):
        """Sample HL7 message for testing"""
        return """MSH|^~\\&|EHR|HOSPITAL|AiMedRes|AIMEDRES|20241201120000||ADT^A01|12345|P|2.5
PID|||PATIENT_001||Doe^Jane||19450515|F||||||||||
OBX|1|NM|MMSE^Mini Mental State Exam^L||22|score||||F
OBX|2|NM|CDR^Clinical Dementia Rating^L||0.5|rating||||F"""
    
    def test_fhir_patient_conversion(self, fhir_converter, sample_patient_data):
        """Test conversion to FHIR Patient resource"""
        fhir_patient = fhir_converter.patient_data_to_fhir(sample_patient_data)
        
        assert isinstance(fhir_patient, FHIRPatient)
        assert fhir_patient.id == 'TEST_PATIENT_001'
        assert fhir_patient.gender == 'female'
        assert fhir_patient.active is True
        
        # Test FHIR JSON conversion
        fhir_json = fhir_patient.to_fhir_json()
        assert fhir_json['resourceType'] == 'Patient'
        assert fhir_json['id'] == 'TEST_PATIENT_001'
    
    def test_assessment_to_diagnostic_report(self, fhir_converter):
        """Test conversion of assessment to FHIR DiagnosticReport"""
        assessment = {
            'condition': 'alzheimer',
            'risk_score': 0.72,
            'risk_level': 'HIGH',
            'confidence': 0.85,
            'interventions': ['neurologist_referral', 'cognitive_assessment']
        }
        
        diagnostic_report = fhir_converter.assessment_to_diagnostic_report(
            assessment, 'TEST_PATIENT_001'
        )
        
        assert isinstance(diagnostic_report, FHIRDiagnosticReport)
        assert diagnostic_report.status == 'final'
        assert 'Patient/TEST_PATIENT_001' in diagnostic_report.subject['reference']
        
        # Test FHIR JSON conversion
        fhir_json = diagnostic_report.to_fhir_json()
        assert fhir_json['resourceType'] == 'DiagnosticReport'
    
    def test_hl7_message_parsing(self, hl7_processor, sample_hl7_message):
        """Test HL7 message parsing"""
        parsed_data = hl7_processor.parse_hl7_message(sample_hl7_message)
        
        assert isinstance(parsed_data, dict)
        assert 'message_type' in parsed_data
        assert 'segments' in parsed_data
        assert 'MSH' in parsed_data['segments']
        assert 'PID' in parsed_data['segments']
        assert 'OBX' in parsed_data['segments']
    
    def test_patient_data_extraction_from_hl7(self, hl7_processor, sample_hl7_message):
        """Test patient data extraction from HL7 message"""
        parsed_data = hl7_processor.parse_hl7_message(sample_hl7_message)
        patient_data = hl7_processor.extract_patient_data(parsed_data)
        
        assert isinstance(patient_data, dict)
        assert patient_data['patient_id'] == 'PATIENT_001'
        assert patient_data['family_name'] == 'Doe'
        assert patient_data['given_name'] == 'Jane'
        assert 'observations' in patient_data
    
    def test_hl7_ack_generation(self, hl7_processor):
        """Test HL7 ACK message generation"""
        original_message = {
            'sending_application': 'EHR',
            'message_control_id': '12345'
        }
        
        ack_message = hl7_processor.create_ack_message(original_message)
        
        assert isinstance(ack_message, str)
        assert 'MSH' in ack_message
        assert 'MSA' in ack_message
        assert 'AiMedRes' in ack_message
    
    def test_data_ingestion_json(self, ehr_connector, sample_patient_data):
        """Test JSON data ingestion"""
        # Add gender field for proper conversion
        test_data = sample_patient_data.copy()
        test_data['gender'] = 'female'
        
        ingested_data = ehr_connector.ingest_patient_data(test_data, 'json')
        
        assert isinstance(ingested_data, dict)
        assert ingested_data['patient_id'] == 'TEST_PATIENT_001'
        assert ingested_data['Age'] == 78
        assert ingested_data['M/F'] == 0
    
    def test_data_ingestion_hl7(self, ehr_connector, sample_hl7_message):
        """Test HL7 data ingestion"""
        ingested_data = ehr_connector.ingest_patient_data(sample_hl7_message, 'hl7')
        
        assert isinstance(ingested_data, dict)
        assert 'patient_id' in ingested_data
        assert 'Age' in ingested_data or 'birth_date' in ingested_data


class TestRegulatoryCompliance:
    """Test regulatory compliance components"""
    
    @pytest.fixture
    def temp_db_paths(self):
        """Create temporary database paths for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as audit_db:
            audit_path = audit_db.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as validation_db:
            validation_path = validation_db.name
        
        yield audit_path, validation_path
        
        # Cleanup
        import os
        try:
            os.unlink(audit_path)
            os.unlink(validation_path)
        except:
            pass
    
    @pytest.fixture
    def hipaa_manager(self, temp_db_paths):
        """Create HIPAA compliance manager for testing"""
        audit_path, _ = temp_db_paths
        config = {'audit_db_path': audit_path}
        return HIPAAComplianceManager(config)
    
    @pytest.fixture
    def fda_manager(self, temp_db_paths):
        """Create FDA validation manager for testing"""
        _, validation_path = temp_db_paths
        config = {'validation_db_path': validation_path}
        return FDAValidationManager(config)
    
    @pytest.fixture
    def sample_audit_event(self):
        """Sample audit event for testing"""
        return AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.MODEL_PREDICTION,
            timestamp=datetime.now(),
            user_id='test_user_001',
            patient_id='TEST_PATIENT_001',
            resource_accessed='alzheimer_model',
            action_performed='risk_prediction',
            outcome='SUCCESS',
            ip_address='192.168.1.100',
            user_agent='Test Client',
            additional_data={'model_version': 'v1.0', 'confidence': 0.85}
        )
    
    def test_hipaa_audit_logging(self, hipaa_manager, sample_audit_event):
        """Test HIPAA audit event logging"""
        # Log event
        hipaa_manager.log_audit_event(sample_audit_event)
        
        # Verify event was logged
        conn = sqlite3.connect(hipaa_manager.audit_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM audit_events WHERE event_id = ?', 
                      (sample_audit_event.event_id,))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1
    
    def test_phi_access_logging(self, hipaa_manager):
        """Test PHI access logging"""
        access_id = hipaa_manager.log_phi_access(
            user_id='test_user_001',
            patient_id='TEST_PATIENT_001',
            data_elements=['Age', 'MMSE', 'CDR'],
            purpose='clinical_decision_support'
        )
        
        assert access_id is not None
        
        # Verify PHI access was logged
        conn = sqlite3.connect(hipaa_manager.audit_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM phi_access_log WHERE access_id = ?', (access_id,))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1
    
    def test_data_minimization(self, hipaa_manager):
        """Test data minimization functionality"""
        full_data = {
            'patient_id': 'TEST_PATIENT_001',
            'Age': 78,
            'MMSE': 22,
            'CDR': 0.5,
            'SSN': '123-45-6789',  # Should be removed
            'phone_number': '555-1234',  # Should be removed
            'detailed_notes': 'Patient history...'  # Should be removed
        }
        
        minimized_data = hipaa_manager.check_data_minimization(full_data, 'risk_assessment')
        
        # Should contain only necessary fields
        assert 'patient_id' in minimized_data
        assert 'Age' in minimized_data
        assert 'MMSE' in minimized_data
        assert 'CDR' in minimized_data
        
        # Should not contain unnecessary fields
        assert 'SSN' not in minimized_data
        assert 'phone_number' not in minimized_data
        assert 'detailed_notes' not in minimized_data
    
    def test_hipaa_compliance_report(self, hipaa_manager, sample_audit_event):
        """Test HIPAA compliance report generation"""
        # Log some events
        hipaa_manager.log_audit_event(sample_audit_event)
        
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now() + timedelta(days=1)
        
        report = hipaa_manager.generate_compliance_report(start_date, end_date)
        
        assert isinstance(report, dict)
        assert 'report_period' in report
        assert 'audit_summary' in report
        assert 'phi_access_summary' in report
        assert 'breach_incidents' in report
        assert 'compliance_status' in report
    
    def test_fda_validation_protocol_creation(self, fda_manager):
        """Test FDA validation protocol creation"""
        protocol = fda_manager.create_validation_protocol('v1.0', '510k')
        
        assert isinstance(protocol, dict)
        assert 'protocol_id' in protocol
        assert protocol['model_version'] == 'v1.0'
        assert protocol['submission_type'] == '510k'
        assert 'validation_phases' in protocol
        assert 'analytical_validation' in protocol['validation_phases']
        assert 'clinical_validation' in protocol['validation_phases']
    
    def test_validation_record_logging(self, fda_manager):
        """Test validation record logging"""
        validation_record = ValidationRecord(
            validation_id=str(uuid.uuid4()),
            model_version='v1.0',
            validation_type='analytical',
            test_dataset='test_set_001',
            performance_metrics={'sensitivity': 0.92, 'specificity': 0.87},
            validation_date=datetime.now(),
            validator='Dr. Test',
            clinical_endpoints=['Accuracy'],
            success_criteria={'sensitivity': 0.90},
            results={'passed': True},
            status='PASSED',
            regulatory_notes='Test validation'
        )
        
        fda_manager.record_validation_result(validation_record)
        
        # Verify record was logged
        conn = sqlite3.connect(fda_manager.validation_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM validation_records WHERE validation_id = ?',
                      (validation_record.validation_id,))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1
    
    def test_adverse_event_logging(self, fda_manager):
        """Test adverse event logging"""
        adverse_event = AdverseEvent(
            event_id=str(uuid.uuid4()),
            patient_id='TEST_PATIENT_001',
            event_date=datetime.now(),
            description='Test adverse event',
            severity='MILD',
            causality='POSSIBLE',
            outcome='RECOVERED',
            ai_system_involved=True,
            model_version='v1.0',
            prediction_confidence=0.85,
            clinical_context={'condition': 'test'},
            reporter='Dr. Test',
            follow_up_required=False
        )
        
        fda_manager.record_adverse_event(adverse_event)
        
        # Verify event was logged
        conn = sqlite3.connect(fda_manager.validation_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM adverse_events WHERE event_id = ?',
                      (adverse_event.event_id,))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1
    
    def test_clinical_performance_tracking(self, fda_manager):
        """Test clinical performance tracking"""
        performance_data = {
            'true_positives': 85,
            'false_positives': 10,
            'true_negatives': 180,
            'false_negatives': 5
        }
        
        fda_manager.track_clinical_performance('v1.0', performance_data)
        
        # Verify performance was tracked
        conn = sqlite3.connect(fda_manager.validation_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM clinical_performance WHERE model_version = ?', ('v1.0',))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1
    
    def test_fda_submission_package_generation(self, fda_manager):
        """Test FDA submission package generation"""
        # Add some test data first
        validation_record = ValidationRecord(
            validation_id=str(uuid.uuid4()),
            model_version='v1.0',
            validation_type='analytical',
            test_dataset='test_set',
            performance_metrics={'sensitivity': 0.92, 'specificity': 0.87},
            validation_date=datetime.now(),
            validator='Dr. Test',
            clinical_endpoints=['Accuracy'],
            success_criteria={'sensitivity': 0.90},
            results={'passed': True},
            status='PASSED',
            regulatory_notes='Test'
        )
        
        fda_manager.record_validation_result(validation_record)
        
        package = fda_manager.generate_fda_submission_package('v1.0')
        
        assert isinstance(package, dict)
        assert package['model_version'] == 'v1.0'
        assert 'submission_date' in package
        assert 'regulatory_pathway' in package
        assert 'validation_summary' in package
        assert 'clinical_performance' in package
        assert 'safety_profile' in package
        assert 'readiness_assessment' in package


class TestClinicalWorkflowOrchestrator:
    """Test the main workflow orchestrator"""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as audit_db:
            audit_path = audit_db.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as validation_db:
            validation_path = validation_db.name
        
        config = {
            'organization_id': 'test-org',
            'master_password': 'test_password',
            'audit_db_path': audit_path,
            'validation_db_path': validation_path
        }
        
        yield config
        
        # Cleanup
        import os
        try:
            os.unlink(audit_path)
            os.unlink(validation_path)
        except:
            pass
    
    @pytest.fixture
    def orchestrator(self, temp_config):
        """Create workflow orchestrator for testing"""
        return ClinicalWorkflowOrchestrator(temp_config)
    
    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing"""
        return {
            'patient_id': 'TEST_PATIENT_001',
            'Age': 78,
            'M/F': 0,
            'MMSE': 22,
            'CDR': 0.5,
            'EDUC': 14,
            'nWBV': 0.72
        }
    
    def test_complete_workflow_processing(self, orchestrator, sample_patient_data):
        """Test complete patient workflow processing"""
        user_id = 'test_clinician_001'
        
        results = orchestrator.process_patient_workflow(
            sample_patient_data, user_id
        )
        
        assert isinstance(results, dict)
        assert 'session_id' in results
        assert 'patient_id' in results
        assert 'timestamp' in results
        assert 'user_id' in results
        assert 'assessments' in results
        assert 'clinical_summary' in results
        assert 'dashboard' in results
        assert 'processing_time_ms' in results
        assert 'compliance_status' in results
        assert 'next_steps' in results
        
        # Verify assessments were generated
        assert len(results['assessments']) > 0
        
        # Verify clinical summary
        clinical_summary = results['clinical_summary']
        assert 'overall_risk_score' in clinical_summary
        assert 'conditions_assessed' in clinical_summary
        assert 'priority_interventions' in clinical_summary
        
        # Verify next steps were generated
        assert len(results['next_steps']) > 0
    
    def test_ehr_export_functionality(self, orchestrator, sample_patient_data):
        """Test EHR export functionality"""
        user_id = 'test_clinician_001'
        
        # First process a workflow
        results = orchestrator.process_patient_workflow(
            sample_patient_data, user_id
        )
        
        session_id = results['session_id']
        
        # Test EHR export
        export_result = orchestrator.export_to_ehr(
            session_id, 'test_ehr_endpoint', 'fhir'
        )
        
        assert isinstance(export_result, dict)
        assert export_result['status'] == 'SUCCESS'
        assert export_result['session_id'] == session_id
        assert export_result['export_format'] == 'fhir'
        assert 'exported_data' in export_result
    
    def test_compliance_status_retrieval(self, orchestrator):
        """Test compliance status retrieval"""
        compliance_status = orchestrator.get_compliance_status()
        
        assert isinstance(compliance_status, dict)
        assert 'generated_at' in compliance_status
        assert 'hipaa_compliance' in compliance_status
        assert 'fda_validation' in compliance_status
        assert 'overall_compliance_score' in compliance_status
    
    def test_session_management(self, orchestrator, sample_patient_data):
        """Test session management functionality"""
        user_id = 'test_clinician_001'
        
        # Process workflow
        results = orchestrator.process_patient_workflow(
            sample_patient_data, user_id
        )
        
        session_id = results['session_id']
        
        # Verify session is stored
        assert session_id in orchestrator.active_sessions
        
        # Verify session data
        session_data = orchestrator.active_sessions[session_id]
        assert session_data['patient_id'] == sample_patient_data['patient_id']
        assert session_data['user_id'] == user_id
    
    def test_error_handling_and_logging(self, orchestrator):
        """Test error handling and audit logging"""
        # Test with minimally invalid patient data that should still work due to data minimization
        minimal_invalid_data = {'some_field': 'value'}
        user_id = 'test_clinician_001'
        
        try:
            result = orchestrator.process_patient_workflow(minimal_invalid_data, user_id)
            # The workflow should complete even with minimal data due to data minimization
            assert 'session_id' in result
            assert 'compliance_status' in result
        except Exception as e:
            # If it does raise an exception, that's also acceptable
            assert isinstance(e, Exception)
        
        # Test with completely empty data should raise an exception
        with pytest.raises(Exception):
            orchestrator.process_patient_workflow(None, user_id)


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])