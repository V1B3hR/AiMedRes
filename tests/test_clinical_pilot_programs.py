#!/usr/bin/env python3
"""
Test suite for Clinical Pilot Programs (P8B)

Tests partnership management, validation studies, case validation,
workflow optimization, and pilot metrics.
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aimedres.clinical.clinical_pilot_programs import (
    ClinicalPilotManager,
    InstitutionalPartnership,
    ValidationStudy,
    CaseValidation,
    WorkflowOptimization,
    PartnershipStatus,
    ValidationStudyPhase,
    CaseValidationStatus,
    create_clinical_pilot_manager
)


class TestInstitutionalPartnership:
    """Test institutional partnership functionality"""
    
    def test_partnership_creation(self):
        """Test creating a partnership"""
        partnership = InstitutionalPartnership(
            institution_name="Test Hospital",
            contact_person="Dr. Test",
            contact_email="test@hospital.org"
        )
        
        assert partnership.institution_name == "Test Hospital"
        assert partnership.status == PartnershipStatus.PENDING
        assert partnership.actual_case_count == 0

    def test_partnership_to_dict(self):
        """Test partnership serialization"""
        partnership = InstitutionalPartnership(
            institution_name="Test Hospital",
            contact_person="Dr. Test",
            contact_email="test@hospital.org"
        )
        
        data = partnership.to_dict()
        assert 'partnership_id' in data
        assert data['institution_name'] == "Test Hospital"
        assert data['status'] == 'pending'


class TestValidationStudy:
    """Test validation study functionality"""
    
    def test_study_creation(self):
        """Test creating a validation study"""
        study = ValidationStudy(
            study_name="Test Study",
            partnership_id="test_partnership_123",
            target_sample_size=100
        )
        
        assert study.study_name == "Test Study"
        assert study.phase == ValidationStudyPhase.DESIGN
        assert study.target_sample_size == 100

    def test_power_calculation(self):
        """Test statistical power calculation"""
        study = ValidationStudy(
            study_name="Test Study",
            target_sample_size=1000
        )
        
        power_analysis = study.calculate_power(effect_size=0.3)
        
        assert 'estimated_power' in power_analysis
        assert 'target_sample_size' in power_analysis
        assert power_analysis['target_sample_size'] == 1000
        assert 0 <= power_analysis['estimated_power'] <= 1.0

    def test_update_metrics(self):
        """Test updating study metrics"""
        study = ValidationStudy(study_name="Test Study")
        
        study.update_metrics({
            'accuracy': 0.95,
            'precision': 0.92
        })
        
        assert study.metrics['accuracy'] == 0.95
        assert study.metrics['precision'] == 0.92


class TestCaseValidation:
    """Test case validation functionality"""
    
    def test_case_creation(self):
        """Test creating a case validation"""
        case = CaseValidation(
            study_id="study_123",
            patient_id="patient_001",
            ai_prediction={'diagnosis': 'alzheimers', 'confidence': 0.85}
        )
        
        assert case.study_id == "study_123"
        assert case.status == CaseValidationStatus.PENDING
        assert case.agreement is None

    def test_case_validation_agreement(self):
        """Test case validation with agreement"""
        case = CaseValidation(
            study_id="study_123",
            patient_id="patient_001",
            ai_prediction={'diagnosis': 'alzheimers'}
        )
        
        case.validate_case(
            ground_truth={'diagnosis': 'alzheimers'},
            feedback={'satisfaction': 'high'}
        )
        
        assert case.status == CaseValidationStatus.COMPLETED
        assert case.agreement == True
        assert case.validated_at is not None

    def test_case_validation_disagreement(self):
        """Test case validation with disagreement"""
        case = CaseValidation(
            study_id="study_123",
            patient_id="patient_001",
            ai_prediction={'diagnosis': 'alzheimers'}
        )
        
        case.validate_case(
            ground_truth={'diagnosis': 'parkinsons'},
            feedback={'satisfaction': 'medium', 'issue': 'misdiagnosis'}
        )
        
        assert case.status == CaseValidationStatus.COMPLETED
        assert case.agreement == False


class TestWorkflowOptimization:
    """Test workflow optimization tracking"""
    
    def test_optimization_creation(self):
        """Test creating workflow optimization"""
        opt = WorkflowOptimization(
            category="ui",
            issue_description="Button placement confusing",
            severity="medium",
            affected_users=5
        )
        
        assert opt.category == "ui"
        assert opt.severity == "medium"
        assert opt.priority_score == 0.0

    def test_priority_calculation(self):
        """Test priority score calculation"""
        opt = WorkflowOptimization(
            category="workflow",
            issue_description="Slow loading time",
            frequency=10,
            severity="high",
            affected_users=20
        )
        
        opt.calculate_priority()
        
        assert opt.priority_score > 0
        # Priority should be influenced by all factors
        assert opt.priority_score == (10 * 0.3) + (3 * 0.4) + (20 * 0.3)


class TestClinicalPilotManager:
    """Test clinical pilot manager functionality"""
    
    def test_manager_creation(self):
        """Test creating pilot manager"""
        manager = create_clinical_pilot_manager()
        assert manager is not None
        assert len(manager.partnerships) == 0
        assert len(manager.studies) == 0

    def test_create_partnership(self):
        """Test creating partnership through manager"""
        manager = create_clinical_pilot_manager()
        
        partnership = manager.create_partnership(
            institution_name="Test Hospital",
            contact_person="Dr. Test",
            contact_email="test@hospital.org",
            target_case_count=200,
            specialties=["Neurology"]
        )
        
        assert len(manager.partnerships) == 1
        assert partnership.institution_name == "Test Hospital"
        assert partnership.specialties == ["Neurology"]

    def test_activate_partnership(self):
        """Test activating partnership"""
        manager = create_clinical_pilot_manager()
        
        partnership = manager.create_partnership(
            institution_name="Test Hospital",
            contact_person="Dr. Test",
            contact_email="test@hospital.org"
        )
        
        manager.activate_partnership(partnership.partnership_id)
        
        activated = manager.partnerships[partnership.partnership_id]
        assert activated.status == PartnershipStatus.ACTIVE
        assert activated.agreement_signed_date is not None

    def test_create_validation_study(self):
        """Test creating validation study"""
        manager = create_clinical_pilot_manager()
        
        partnership = manager.create_partnership(
            institution_name="Test Hospital",
            contact_person="Dr. Test",
            contact_email="test@hospital.org"
        )
        
        study = manager.create_validation_study(
            study_name="Test Study",
            partnership_id=partnership.partnership_id,
            target_sample_size=500
        )
        
        assert len(manager.studies) == 1
        assert study.target_sample_size == 500
        assert 'estimated_power' in study.power_analysis

    def test_add_case_validation(self):
        """Test adding case validation"""
        manager = create_clinical_pilot_manager()
        
        partnership = manager.create_partnership(
            institution_name="Test Hospital",
            contact_person="Dr. Test",
            contact_email="test@hospital.org"
        )
        
        study = manager.create_validation_study(
            study_name="Test Study",
            partnership_id=partnership.partnership_id
        )
        
        case = manager.add_case_validation(
            study_id=study.study_id,
            patient_id="patient_001",
            ai_prediction={'diagnosis': 'alzheimers'}
        )
        
        assert len(manager.cases) == 1
        assert study.current_sample_size == 1

    def test_validate_case(self):
        """Test validating a case"""
        manager = create_clinical_pilot_manager()
        
        partnership = manager.create_partnership(
            institution_name="Test Hospital",
            contact_person="Dr. Test",
            contact_email="test@hospital.org"
        )
        
        study = manager.create_validation_study(
            study_name="Test Study",
            partnership_id=partnership.partnership_id
        )
        
        case = manager.add_case_validation(
            study_id=study.study_id,
            patient_id="patient_001",
            ai_prediction={'diagnosis': 'alzheimers'}
        )
        
        manager.validate_case(
            case_id=case.case_id,
            ground_truth={'diagnosis': 'alzheimers'},
            clinician_feedback={'satisfaction': 'high'}
        )
        
        validated_case = manager.cases[case.case_id]
        assert validated_case.status == CaseValidationStatus.COMPLETED
        assert validated_case.agreement == True

    def test_add_workflow_optimization(self):
        """Test adding workflow optimization"""
        manager = create_clinical_pilot_manager()
        
        opt = manager.add_workflow_optimization(
            category="ui",
            issue_description="Confusing interface",
            severity="medium",
            affected_users=5
        )
        
        assert len(manager.optimizations) == 1
        assert opt.priority_score > 0

    def test_get_pilot_metrics_empty(self):
        """Test getting metrics from empty pilot"""
        manager = create_clinical_pilot_manager()
        metrics = manager.get_pilot_metrics()
        
        assert metrics['partnerships']['total'] == 0
        assert metrics['studies']['total'] == 0
        assert metrics['cases']['total'] == 0

    def test_get_pilot_metrics_with_data(self):
        """Test getting metrics with data"""
        manager = create_clinical_pilot_manager()
        
        # Create partnership and study
        partnership = manager.create_partnership(
            institution_name="Test Hospital",
            contact_person="Dr. Test",
            contact_email="test@hospital.org"
        )
        manager.activate_partnership(partnership.partnership_id)
        
        study = manager.create_validation_study(
            study_name="Test Study",
            partnership_id=partnership.partnership_id,
            target_sample_size=1000
        )
        
        # Add and validate cases
        for i in range(10):
            case = manager.add_case_validation(
                study_id=study.study_id,
                patient_id=f"patient_{i:03d}",
                ai_prediction={'diagnosis': 'alzheimers'}
            )
            
            if i % 2 == 0:
                manager.validate_case(
                    case_id=case.case_id,
                    ground_truth={'diagnosis': 'alzheimers'},
                    clinician_feedback={'satisfaction': 'high'}
                )
        
        metrics = manager.get_pilot_metrics()
        
        assert metrics['partnerships']['total'] == 1
        assert metrics['partnerships']['active'] == 1
        assert metrics['studies']['total'] == 1
        assert metrics['cases']['total'] == 10
        assert metrics['cases']['completed'] == 5
        assert metrics['cases']['completion_rate'] == 0.5
        assert metrics['cases']['agreement_rate'] == 1.0

    def test_get_study_report(self):
        """Test generating study report"""
        manager = create_clinical_pilot_manager()
        
        partnership = manager.create_partnership(
            institution_name="Test Hospital",
            contact_person="Dr. Test",
            contact_email="test@hospital.org"
        )
        
        study = manager.create_validation_study(
            study_name="Test Study",
            partnership_id=partnership.partnership_id,
            target_sample_size=100
        )
        
        # Add cases
        for i in range(10):
            case = manager.add_case_validation(
                study_id=study.study_id,
                patient_id=f"patient_{i:03d}",
                ai_prediction={'diagnosis': 'alzheimers'}
            )
            
            manager.validate_case(
                case_id=case.case_id,
                ground_truth={'diagnosis': 'alzheimers'},
                clinician_feedback={'satisfaction': 'high'}
            )
        
        report = manager.get_study_report(study.study_id)
        
        assert report['study_name'] == "Test Study"
        assert report['sample_size']['current'] == 10
        assert report['sample_size']['completed'] == 10
        assert report['agreement_metrics']['agreement_rate'] == 1.0

    def test_export_pilot_data(self):
        """Test exporting pilot data"""
        manager = create_clinical_pilot_manager()
        
        # Create some data
        partnership = manager.create_partnership(
            institution_name="Test Hospital",
            contact_person="Dr. Test",
            contact_email="test@hospital.org"
        )
        
        exported = manager.export_pilot_data(format='json')
        
        assert 'partnerships' in exported
        assert 'studies' in exported
        assert 'export_timestamp' in exported

    def test_1000_case_target_tracking(self):
        """Test tracking progress toward 1000 case target"""
        manager = create_clinical_pilot_manager()
        
        partnership = manager.create_partnership(
            institution_name="Test Hospital",
            contact_person="Dr. Test",
            contact_email="test@hospital.org"
        )
        
        study = manager.create_validation_study(
            study_name="Large Study",
            partnership_id=partnership.partnership_id,
            target_sample_size=1000
        )
        
        # Simulate 100 cases
        for i in range(100):
            case = manager.add_case_validation(
                study_id=study.study_id,
                patient_id=f"patient_{i:04d}",
                ai_prediction={'diagnosis': 'alzheimers'}
            )
            manager.validate_case(
                case_id=case.case_id,
                ground_truth={'diagnosis': 'alzheimers'},
                clinician_feedback={'satisfaction': 'high'}
            )
        
        metrics = manager.get_pilot_metrics()
        
        assert metrics['target_progress']['1000_case_validation'] == "100/1000"
        assert metrics['target_progress']['percentage'] == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
