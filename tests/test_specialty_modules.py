"""
Test Suite for Specialty Clinical Modules (P13)

Tests for:
- Pediatric adaptation (age normative baselines)
- Geriatric care (polypharmacy risk modeling)
- Emergency department triage (low-latency heuristics)
- Telemedicine integration (session context sync)
"""

import pytest
import time
from datetime import datetime, timedelta

from aimedres.clinical.specialty_modules import (
    PediatricModule,
    GeriatricModule,
    EmergencyTriageModule,
    TelemedicineModule,
    create_pediatric_module,
    create_geriatric_module,
    create_emergency_triage_module,
    create_telemedicine_module,
    AgeGroup,
    TriagePriority
)


class TestPediatricModule:
    """Tests for pediatric clinical module."""
    
    def test_initialization(self):
        """Test pediatric module initialization."""
        module = create_pediatric_module()
        assert module is not None
        assert len(module._baselines) >= 5
    
    def test_age_group_classification(self):
        """Test age group determination."""
        module = create_pediatric_module()
        
        assert module.get_age_group(14) == AgeGroup.NEONATE  # 14 days
        assert module.get_age_group(60) == AgeGroup.INFANT  # 2 months
        assert module.get_age_group(365) == AgeGroup.INFANT  # 1 year
        assert module.get_age_group(730) == AgeGroup.TODDLER  # 2 years
        assert module.get_age_group(365 * 5) == AgeGroup.CHILD  # 5 years
        assert module.get_age_group(365 * 15) == AgeGroup.ADOLESCENT  # 15 years
        assert module.get_age_group(365 * 30) == AgeGroup.YOUNG_ADULT  # 30 years
        assert module.get_age_group(365 * 70) == AgeGroup.SENIOR  # 70 years
    
    def test_vital_signs_assessment_normal(self):
        """Test vital signs assessment with normal values."""
        module = create_pediatric_module()
        
        # Test neonate with normal vitals
        vital_signs = {
            "heart_rate": 120,
            "respiratory_rate": 40,
            "blood_pressure_systolic": 75,
            "temperature": 37.0
        }
        
        assessment = module.assess_vital_signs(
            age_days=14,  # Neonate
            vital_signs=vital_signs
        )
        
        assert assessment["age_group"] == "neonate"
        assert assessment["severity"] == "normal"
        assert len(assessment["abnormal_flags"]) == 0
        assert all(v["status"] == "normal" for v in assessment["vital_signs"].values())
    
    def test_vital_signs_assessment_abnormal(self):
        """Test vital signs assessment with abnormal values."""
        module = create_pediatric_module()
        
        # Test child with abnormal vitals
        vital_signs = {
            "heart_rate": 150,  # Too high for child
            "respiratory_rate": 35,  # Too high
            "blood_pressure_systolic": 80,
            "temperature": 39.0  # Fever
        }
        
        assessment = module.assess_vital_signs(
            age_days=365 * 8,  # 8-year-old child
            vital_signs=vital_signs
        )
        
        assert assessment["age_group"] == "child"
        assert len(assessment["abnormal_flags"]) >= 2
        assert assessment["severity"] in ["concerning", "critical"]
    
    def test_vital_signs_critical_severity(self):
        """Test critical severity detection."""
        module = create_pediatric_module()
        
        # Multiple abnormal vitals
        vital_signs = {
            "heart_rate": 50,  # Too low
            "respiratory_rate": 8,  # Too low
            "blood_pressure_systolic": 60,  # Too low
            "temperature": 35.0  # Hypothermia
        }
        
        assessment = module.assess_vital_signs(
            age_days=365 * 10,
            vital_signs=vital_signs
        )
        
        assert len(assessment["abnormal_flags"]) >= 3
        assert assessment["severity"] == "critical"
    
    def test_developmental_assessment_complete(self):
        """Test developmental assessment with all milestones."""
        module = create_pediatric_module()
        
        assessment = module.get_developmental_assessment(
            age_days=365,  # 1 year old
            achieved_milestones=["head_control", "sitting", "babbling", "crawling"]
        )
        
        assert assessment["age_group"] == "infant"
        assert assessment["completion_rate"] == 1.0
        assert len(assessment["missing_milestones"]) == 0
        assert assessment["concern_level"] == "none"
    
    def test_developmental_assessment_delayed(self):
        """Test developmental assessment with delays."""
        module = create_pediatric_module()
        
        assessment = module.get_developmental_assessment(
            age_days=365,  # 1 year old
            achieved_milestones=["head_control"]  # Missing most milestones
        )
        
        assert assessment["age_group"] == "infant"
        assert assessment["completion_rate"] < 0.5
        assert len(assessment["missing_milestones"]) >= 2
        assert assessment["concern_level"] == "high"
    
    def test_age_normative_baselines(self):
        """Test that age-normative baselines exist for all age groups."""
        module = create_pediatric_module()
        
        pediatric_groups = [
            AgeGroup.NEONATE,
            AgeGroup.INFANT,
            AgeGroup.TODDLER,
            AgeGroup.CHILD,
            AgeGroup.ADOLESCENT
        ]
        
        for age_group in pediatric_groups:
            assert age_group in module._baselines
            baseline = module._baselines[age_group]
            assert len(baseline.vital_signs) >= 4
            assert len(baseline.developmental_milestones) >= 1
            assert len(baseline.growth_percentiles) >= 2


class TestGeriatricModule:
    """Tests for geriatric care module."""
    
    def test_initialization(self):
        """Test geriatric module initialization."""
        module = create_geriatric_module()
        assert module is not None
        assert len(module._drug_interactions) > 0
    
    def test_polypharmacy_risk_low(self):
        """Test low polypharmacy risk assessment."""
        module = create_geriatric_module()
        
        medications = [
            {"name": "aspirin", "class": "antiplatelet"},
            {"name": "lisinopril", "class": "ace_inhibitor"}
        ]
        
        profile = module.assess_polypharmacy_risk(
            patient_id="patient_001",
            age=70,
            medications=medications,
            comorbidities=["hypertension"]
        )
        
        assert profile.patient_id == "patient_001"
        assert profile.age == 70
        assert profile.polypharmacy_risk < 0.5
        assert len(profile.medications) == 2
    
    def test_polypharmacy_risk_high(self):
        """Test high polypharmacy risk assessment."""
        module = create_geriatric_module()
        
        medications = [
            {"name": "warfarin", "class": "anticoagulant"},
            {"name": "aspirin", "class": "antiplatelet"},
            {"name": "metformin", "class": "antidiabetic"},
            {"name": "lisinopril", "class": "ace_inhibitor"},
            {"name": "atorvastatin", "class": "statin"},
            {"name": "omeprazole", "class": "ppi"},
            {"name": "furosemide", "class": "diuretic"},
            {"name": "levothyroxine", "class": "thyroid"},
            {"name": "prednisone", "class": "corticosteroid"},
            {"name": "digoxin", "class": "cardiac_glycoside"},
            {"name": "amiodarone", "class": "antiarrhythmic"}
        ]
        
        comorbidities = [
            "diabetes", "hypertension", "atrial_fibrillation",
            "heart_failure", "osteoarthritis"
        ]
        
        profile = module.assess_polypharmacy_risk(
            patient_id="patient_002",
            age=82,
            medications=medications,
            comorbidities=comorbidities
        )
        
        assert profile.polypharmacy_risk > 0.6
        assert len(profile.drug_interactions) > 0
        assert len(profile.recommended_interventions) >= 2
        assert "comprehensive_medication_review" in profile.recommended_interventions
    
    def test_drug_interaction_detection(self):
        """Test drug interaction detection."""
        module = create_geriatric_module()
        
        medications = [
            {"name": "warfarin", "class": "anticoagulant"},
            {"name": "aspirin", "class": "antiplatelet"},  # Interacts with warfarin
            {"name": "ibuprofen", "class": "nsaid"}  # Also interacts with warfarin
        ]
        
        profile = module.assess_polypharmacy_risk(
            patient_id="patient_003",
            age=75,
            medications=medications,
            comorbidities=[]
        )
        
        assert len(profile.drug_interactions) >= 2
        interaction_drugs = [
            (i["drug1"], i["drug2"]) for i in profile.drug_interactions
        ]
        assert ("warfarin", "aspirin") in interaction_drugs
        assert ("warfarin", "ibuprofen") in interaction_drugs
    
    def test_frailty_assessment(self):
        """Test frailty score calculation."""
        module = create_geriatric_module()
        
        # Young senior with few comorbidities
        profile_young = module.assess_polypharmacy_risk(
            patient_id="patient_004",
            age=68,
            medications=[{"name": "aspirin", "class": "antiplatelet"}],
            comorbidities=["hypertension"]
        )
        
        # Elderly with multiple comorbidities
        profile_elderly = module.assess_polypharmacy_risk(
            patient_id="patient_005",
            age=88,
            medications=[],
            comorbidities=[
                "diabetes", "heart_failure", "copd", "ckd", "dementia"
            ]
        )
        
        assert profile_elderly.frailty_score > profile_young.frailty_score
    
    def test_fall_risk_assessment(self):
        """Test fall risk assessment."""
        module = create_geriatric_module()
        
        high_risk_meds = [
            {"name": "diazepam", "class": "benzodiazepines"},
            {"name": "quetiapine", "class": "antipsychotics"},
            {"name": "zolpidem", "class": "sedatives"}
        ]
        
        profile = module.assess_polypharmacy_risk(
            patient_id="patient_006",
            age=80,
            medications=high_risk_meds,
            comorbidities=["osteoporosis"]
        )
        
        assert profile.fall_risk_score > 0.4
        assert "fall_prevention_program" in profile.recommended_interventions


class TestEmergencyTriageModule:
    """Tests for emergency department triage module."""
    
    def test_initialization(self):
        """Test emergency triage module initialization."""
        module = create_emergency_triage_module()
        assert module is not None
        assert len(module._vital_sign_thresholds) > 0
    
    def test_triage_immediate_priority(self):
        """Test immediate priority triage."""
        module = create_emergency_triage_module()
        
        assessment = module.triage_assessment(
            patient_id="patient_001",
            chief_complaint="severe chest pain with difficulty breathing",
            vital_signs={
                "heart_rate": 45,  # Critically low
                "respiratory_rate": 32,
                "blood_pressure_systolic": 85,
                "oxygen_saturation": 88
            },
            pain_level=10
        )
        
        assert assessment.triage_priority == TriagePriority.IMMEDIATE
        assert assessment.estimated_wait_time_min == 0
        assert len(assessment.red_flags) > 0
        assert "trauma_bay" in assessment.required_resources
    
    def test_triage_urgent_priority(self):
        """Test urgent priority triage."""
        module = create_emergency_triage_module()
        
        assessment = module.triage_assessment(
            patient_id="patient_002",
            chief_complaint="moderate abdominal pain",
            vital_signs={
                "heart_rate": 115,
                "respiratory_rate": 24,
                "blood_pressure_systolic": 105,
                "oxygen_saturation": 94
            },
            pain_level=8
        )
        
        assert assessment.triage_priority == TriagePriority.URGENT
        assert assessment.estimated_wait_time_min <= 30
        assert "physician" in assessment.required_resources
    
    def test_triage_non_urgent_priority(self):
        """Test non-urgent priority triage."""
        module = create_emergency_triage_module()
        
        assessment = module.triage_assessment(
            patient_id="patient_003",
            chief_complaint="minor cold symptoms",
            vital_signs={
                "heart_rate": 75,
                "respiratory_rate": 16,
                "blood_pressure_systolic": 120,
                "oxygen_saturation": 98,
                "temperature": 37.0
            },
            pain_level=1
        )
        
        assert assessment.triage_priority == TriagePriority.NON_URGENT
        assert assessment.estimated_wait_time_min >= 120
    
    def test_low_latency_assessment(self):
        """Test low-latency triage (P13 requirement)."""
        module = create_emergency_triage_module()
        
        start_time = time.time()
        
        # Perform 100 triage assessments
        for i in range(100):
            assessment = module.triage_assessment(
                patient_id=f"patient_{i}",
                chief_complaint="test complaint",
                vital_signs={
                    "heart_rate": 80,
                    "respiratory_rate": 16,
                    "blood_pressure_systolic": 120,
                    "oxygen_saturation": 98
                },
                pain_level=3
            )
            assert assessment.assessment_time_ms < 100  # < 100ms per assessment
        
        elapsed = time.time() - start_time
        avg_time_ms = (elapsed * 1000) / 100
        
        print(f"\nAverage triage time: {avg_time_ms:.2f}ms")
        
        # Should complete in <5ms per assessment on average
        assert avg_time_ms < 10
    
    def test_red_flag_detection(self):
        """Test detection of critical red flags."""
        module = create_emergency_triage_module()
        
        red_flag_complaints = [
            "chest pain and shortness of breath",
            "difficulty breathing severely",
            "altered mental status confused",
            "severe bleeding from head",
            "stroke symptoms weakness"
        ]
        
        for complaint in red_flag_complaints:
            assessment = module.triage_assessment(
                patient_id="test_patient",
                chief_complaint=complaint,
                vital_signs={
                    "heart_rate": 80,
                    "respiratory_rate": 16,
                    "blood_pressure_systolic": 120,
                    "oxygen_saturation": 98
                },
                pain_level=5
            )
            
            assert len(assessment.red_flags) > 0
            assert assessment.triage_priority in [
                TriagePriority.IMMEDIATE,
                TriagePriority.URGENT
            ]
    
    def test_pain_level_influence(self):
        """Test that pain level influences triage priority."""
        module = create_emergency_triage_module()
        
        # Low pain
        low_pain = module.triage_assessment(
            patient_id="patient_low",
            chief_complaint="general discomfort",
            vital_signs={
                "heart_rate": 80,
                "respiratory_rate": 16,
                "blood_pressure_systolic": 120,
                "oxygen_saturation": 98
            },
            pain_level=2
        )
        
        # High pain
        high_pain = module.triage_assessment(
            patient_id="patient_high",
            chief_complaint="severe pain",
            vital_signs={
                "heart_rate": 80,
                "respiratory_rate": 16,
                "blood_pressure_systolic": 120,
                "oxygen_saturation": 98
            },
            pain_level=9
        )
        
        assert high_pain.triage_priority.value < low_pain.triage_priority.value


class TestTelemedicineModule:
    """Tests for telemedicine integration module."""
    
    def test_initialization(self):
        """Test telemedicine module initialization."""
        module = create_telemedicine_module()
        assert module is not None
        assert len(module.active_sessions) == 0
    
    def test_start_session(self):
        """Test starting a telemedicine session."""
        module = create_telemedicine_module()
        
        session = module.start_session(
            patient_id="patient_001",
            provider_id="doctor_001",
            session_type="video",
            chief_complaint="follow-up visit"
        )
        
        assert session.session_id is not None
        assert session.patient_id == "patient_001"
        assert session.provider_id == "doctor_001"
        assert session.session_type == "video"
        assert session.start_time is not None
        assert session.end_time is None
    
    def test_sync_clinical_context(self):
        """Test clinical context synchronization."""
        module = create_telemedicine_module()
        
        session = module.start_session(
            patient_id="patient_002",
            provider_id="doctor_002",
            session_type="video"
        )
        
        clinical_data = {
            "vital_signs": {"heart_rate": 75, "blood_pressure": "120/80"},
            "symptoms": ["headache", "fatigue"],
            "current_medications": ["aspirin", "lisinopril"]
        }
        
        success = module.sync_clinical_context(session.session_id, clinical_data)
        
        assert success is True
        
        # Verify data was synced
        updated_session = module.active_sessions[session.session_id]
        assert "vital_signs" in updated_session.clinical_context
        assert "symptoms" in updated_session.clinical_context
    
    def test_add_assessment(self):
        """Test adding clinical assessments to session."""
        module = create_telemedicine_module()
        
        session = module.start_session(
            patient_id="patient_003",
            provider_id="doctor_003"
        )
        
        assessment1 = {
            "type": "initial",
            "findings": "patient reports headache for 3 days",
            "severity": "moderate"
        }
        
        assessment2 = {
            "type": "diagnosis",
            "diagnosis": "tension headache",
            "confidence": 0.85
        }
        
        success1 = module.add_assessment(session.session_id, assessment1)
        success2 = module.add_assessment(session.session_id, assessment2)
        
        assert success1 is True
        assert success2 is True
        
        updated_session = module.active_sessions[session.session_id]
        assert len(updated_session.assessments) == 2
    
    def test_end_session(self):
        """Test ending a telemedicine session."""
        module = create_telemedicine_module()
        
        session = module.start_session(
            patient_id="patient_004",
            provider_id="doctor_004"
        )
        
        session_id = session.session_id
        
        # Add some content
        module.sync_clinical_context(
            session_id,
            {"symptoms": ["cough"]}
        )
        
        # End session
        completed = module.end_session(
            session_id,
            session_notes="Patient advised rest and fluids",
            follow_up_required=True
        )
        
        assert completed is not None
        assert completed.end_time is not None
        assert completed.session_notes == "Patient advised rest and fluids"
        assert completed.follow_up_required is True
        
        # Session should be removed from active sessions
        assert session_id not in module.active_sessions
    
    def test_get_session_summary(self):
        """Test getting session summary."""
        module = create_telemedicine_module()
        
        session = module.start_session(
            patient_id="patient_005",
            provider_id="doctor_005",
            chief_complaint="routine checkup"
        )
        
        # Add some assessments
        module.add_assessment(session.session_id, {"type": "initial"})
        module.add_assessment(session.session_id, {"type": "followup"})
        
        summary = module.get_session_summary(session.session_id)
        
        assert summary is not None
        assert summary["session_id"] == session.session_id
        assert summary["patient_id"] == "patient_005"
        assert summary["provider_id"] == "doctor_005"
        assert summary["assessments_count"] == 2
        assert summary["status"] == "active"
        assert "duration_seconds" in summary
    
    def test_multiple_concurrent_sessions(self):
        """Test handling multiple concurrent sessions."""
        module = create_telemedicine_module()
        
        # Start 10 concurrent sessions
        sessions = []
        for i in range(10):
            session = module.start_session(
                patient_id=f"patient_{i}",
                provider_id=f"doctor_{i % 3}",  # 3 doctors
                session_type="video"
            )
            sessions.append(session)
        
        assert len(module.active_sessions) == 10
        
        # End half of them
        for i in range(5):
            module.end_session(sessions[i].session_id)
        
        assert len(module.active_sessions) == 5
    
    def test_session_context_sync_performance(self):
        """Test performance of clinical context synchronization."""
        module = create_telemedicine_module()
        
        session = module.start_session(
            patient_id="patient_perf",
            provider_id="doctor_perf"
        )
        
        # Sync context 100 times
        start_time = time.time()
        for i in range(100):
            module.sync_clinical_context(
                session.session_id,
                {f"data_point_{i}": f"value_{i}"}
            )
        elapsed = time.time() - start_time
        
        avg_time_ms = (elapsed * 1000) / 100
        print(f"\nAverage sync time: {avg_time_ms:.2f}ms")
        
        # Should be very fast (< 1ms per sync)
        assert avg_time_ms < 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
