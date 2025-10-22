#!/usr/bin/env python3
"""
Test suite for FDA Pre-Submission Framework implementation
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import json

from aimedres.compliance.regulatory import FDAValidationManager, ValidationRecord, AdverseEvent


class TestFDAPreSubmissionFramework:
    """Test suite for FDA pre-submission framework functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.config = {
            'validation_db_path': self.temp_db.name
        }
        self.fda_manager = FDAValidationManager(self.config)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_enhanced_submission_package_generation(self):
        """Test enhanced FDA submission package generation."""
        model_version = "test_v1.0"
        
        # Generate submission package
        package = self.fda_manager.generate_fda_submission_package(model_version)
        
        # Verify package structure
        assert isinstance(package, dict)
        assert len(package) >= 15  # Should have many sections
        
        # Check core sections
        required_sections = [
            'model_version', 'submission_date', 'regulatory_pathway',
            'device_description', 'intended_use_statement',
            'predicate_device_comparison', 'software_documentation',
            'clinical_validation_report', 'labeling_information',
            'pre_submission_checklist', 'pre_submission_meeting_package'
        ]
        
        for section in required_sections:
            assert section in package, f"Missing section: {section}"
        
        # Verify submission readiness score
        readiness = package['submission_readiness_score']
        assert 'total_score' in readiness
        assert 'readiness_status' in readiness
        assert 'recommendations' in readiness
        assert 'next_steps' in readiness
        assert readiness['max_score'] == 100
    
    def test_pre_submission_checklist(self):
        """Test pre-submission checklist generation."""
        model_version = "test_v1.0"
        package = self.fda_manager.generate_fda_submission_package(model_version)
        
        checklist = package['pre_submission_checklist']
        
        # Verify checklist structure
        assert 'documentation_completeness' in checklist
        assert 'validation_requirements' in checklist
        assert 'safety_requirements' in checklist
        assert 'regulatory_requirements' in checklist
        assert 'overall_completeness' in checklist
        
        # Verify overall completeness calculation
        overall = checklist['overall_completeness']
        assert 'percentage' in overall
        assert 'completed_items' in overall
        assert 'total_items' in overall
        assert 'ready_for_submission' in overall
        assert 0 <= overall['percentage'] <= 100
    
    def test_fda_consultation_request_generation(self):
        """Test FDA Q-Sub consultation request generation."""
        model_version = "test_v1.0"
        
        consultation = self.fda_manager.generate_fda_consultation_request(model_version)
        
        # Verify consultation request structure
        required_sections = [
            'submission_type', 'meeting_type', 'device_information',
            'sponsor_information', 'background_summary', 'specific_questions',
            'meeting_objectives', 'supporting_documents_list', 'timeline',
            'regulatory_strategy'
        ]
        
        for section in required_sections:
            assert section in consultation, f"Missing section: {section}"
        
        # Verify specific questions
        questions = consultation['specific_questions']
        assert len(questions) >= 4
        for question in questions:
            assert 'question_id' in question
            assert 'category' in question
            assert 'question' in question
            assert 'rationale' in question
        
        # Verify timeline
        timeline = consultation['timeline']
        assert 'q_sub_submission_date' in timeline
        assert 'planned_510k_submission' in timeline
    
    def test_continuous_validation_monitoring(self):
        """Test continuous validation monitoring functionality."""
        model_version = "test_v1.0"
        
        # Test continuous validation monitoring
        monitoring = self.fda_manager.monitor_continuous_validation(model_version)
        
        # Verify monitoring structure
        required_sections = [
            'model_version', 'assessment_date', 'validation_activity',
            'performance_monitoring', 'safety_surveillance', 'system_usage',
            'compliance_status', 'recommendations'
        ]
        
        for section in required_sections:
            assert section in monitoring, f"Missing section: {section}"
        
        # Verify compliance status
        compliance = monitoring['compliance_status']
        assert 'validation_compliance' in compliance
        assert 'monitoring_compliance' in compliance
        assert 'safety_compliance' in compliance
        assert 'overall_compliance' in compliance
        assert isinstance(compliance['overall_compliance'], bool)
        
        # Verify recommendations are provided
        assert isinstance(monitoring['recommendations'], list)
        assert len(monitoring['recommendations']) > 0
    
    def test_enhanced_readiness_scoring(self):
        """Test enhanced FDA submission readiness scoring."""
        model_version = "test_v1.0"
        
        # Add some validation records to improve score
        validation_record = ValidationRecord(
            validation_id="test_val_001",
            model_version=model_version,
            validation_type="analytical",
            test_dataset="test_dataset",
            performance_metrics={'sensitivity': 0.95, 'specificity': 0.90},
            validation_date=datetime.now(),
            validator="Test Validator",
            clinical_endpoints=["Test accuracy"],
            success_criteria={'sensitivity': 0.90, 'specificity': 0.85},
            results={'passed': True},
            status="PASSED",
            regulatory_notes="Test validation"
        )
        
        self.fda_manager.record_validation_result(validation_record)
        
        # Generate package with validation data
        package = self.fda_manager.generate_fda_submission_package(model_version)
        readiness = package['submission_readiness_score']
        
        # Verify enhanced scoring
        assert readiness['total_score'] > 40  # Should be higher with validation data
        assert len(readiness['score_components']) == 6  # Enhanced component count
        assert 'documentation_completeness' in readiness['score_components']
        assert 'validation_completeness' in readiness['score_components']
        assert 'performance_evidence' in readiness['score_components']
        assert 'safety_documentation' in readiness['score_components']
        assert 'quality_assurance' in readiness['score_components']
        assert 'regulatory_compliance' in readiness['score_components']
    
    def test_device_description_generation(self):
        """Test FDA-compliant device description generation."""
        model_version = "test_v1.0"
        package = self.fda_manager.generate_fda_submission_package(model_version)
        
        device_desc = package['device_description']
        
        # Verify device description structure
        required_fields = [
            'device_name', 'device_classification', 'intended_use',
            'device_category', 'technology_description', 'hardware_requirements',
            'interoperability', 'deployment_model'
        ]
        
        for field in required_fields:
            assert field in device_desc, f"Missing field: {field}"
            assert device_desc[field], f"Empty field: {field}"
    
    def test_software_documentation_generation(self):
        """Test comprehensive software documentation generation."""
        model_version = "test_v1.0"
        package = self.fda_manager.generate_fda_submission_package(model_version)
        
        software_docs = package['software_documentation']
        
        # Verify software documentation structure
        required_sections = [
            'software_lifecycle_processes', 'risk_management',
            'verification_validation', 'configuration_management'
        ]
        
        for section in required_sections:
            assert section in software_docs, f"Missing section: {section}"
            assert isinstance(software_docs[section], dict)
    
    def test_clinical_validation_report_generation(self):
        """Test clinical validation report generation."""
        model_version = "test_v1.0"
        package = self.fda_manager.generate_fda_submission_package(model_version)
        
        clinical_report = package['clinical_validation_report']
        
        # Verify clinical report structure
        required_sections = [
            'study_design', 'statistical_analysis',
            'results_summary', 'clinical_significance'
        ]
        
        for section in required_sections:
            assert section in clinical_report, f"Missing section: {section}"
            assert isinstance(clinical_report[section], dict)
    
    def test_labeling_information_generation(self):
        """Test FDA-compliant labeling information generation."""
        model_version = "test_v1.0"
        package = self.fda_manager.generate_fda_submission_package(model_version)
        
        labeling = package['labeling_information']
        
        # Verify labeling structure
        required_sections = [
            'device_labeling', 'indications_for_use',
            'warnings_precautions', 'user_instructions'
        ]
        
        for section in required_sections:
            assert section in labeling, f"Missing section: {section}"
            assert isinstance(labeling[section], dict)
    
    def test_pre_submission_meeting_materials(self):
        """Test pre-submission meeting materials preparation."""
        model_version = "test_v1.0"
        package = self.fda_manager.generate_fda_submission_package(model_version)
        
        meeting_package = package['pre_submission_meeting_package']
        
        # Verify meeting package structure
        required_sections = [
            'meeting_request', 'supporting_documents',
            'meeting_timeline', 'post_meeting_plan'
        ]
        
        for section in required_sections:
            assert section in meeting_package, f"Missing section: {section}"
            assert isinstance(meeting_package[section], dict)
        
        # Verify meeting request details
        meeting_request = meeting_package['meeting_request']
        assert 'meeting_type' in meeting_request
        assert 'requested_topics' in meeting_request
        assert 'specific_questions' in meeting_request
        assert len(meeting_request['requested_topics']) >= 4
        assert len(meeting_request['specific_questions']) >= 4


def test_fda_framework_integration():
    """Integration test for FDA framework components."""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    try:
        config = {'validation_db_path': temp_db.name}
        fda_manager = FDAValidationManager(config)
        model_version = "integration_test_v1.0"
        
        # Test complete workflow
        package = fda_manager.generate_fda_submission_package(model_version)
        consultation = fda_manager.generate_fda_consultation_request(model_version)
        monitoring = fda_manager.monitor_continuous_validation(model_version)
        
        # Verify integration
        assert package['model_version'] == model_version
        assert consultation['device_information']['device_name'].endswith(model_version)
        assert monitoring['model_version'] == model_version
        
        # Verify readiness assessment consistency
        package_readiness = package['submission_readiness_score']['readiness_status']
        assert package_readiness in ['READY_FOR_SUBMISSION', 'NEARLY_READY', 'NEEDS_IMPROVEMENT', 'NOT_READY']
        
    finally:
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])