#!/usr/bin/env python3
"""
Test suite for FDA Regulatory Pathway Planning (P9)

Tests device classification, pre-submission packages, clinical evidence dossiers,
QMS documentation, and pathway planning functionality.
"""

import pytest
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aimedres.compliance.fda_pathway_planning import (
    FDAPathwayPlanner,
    DeviceClassificationAnalysis,
    PreSubmissionPackage,
    ClinicalEvidenceItem,
    ClinicalEvidenceDossier,
    QMSDocument,
    QMSSkeleton,
    RiskCategory,
    SoftwareLevel,
    QSubStatus,
    EvidenceType,
    create_fda_pathway_planner
)


class TestDeviceClassificationAnalysis:
    """Test device classification functionality"""
    
    def test_classification_creation(self):
        """Test creating classification analysis"""
        analysis = DeviceClassificationAnalysis(
            device_name="Test Device",
            intended_use="Diagnostic support",
            indications_for_use="For clinical use"
        )
        
        assert analysis.device_name == "Test Device"
        assert analysis.risk_category == RiskCategory.MODERATE

    def test_low_risk_analysis(self):
        """Test low risk classification"""
        analysis = DeviceClassificationAnalysis(
            device_name="Test Device",
            patient_population="adult",
            clinical_decision_type="monitoring",
            autonomy_level="advisory",
            critical_decision_impact=False
        )
        
        risk_analysis = analysis.analyze_risk()
        
        assert risk_analysis['risk_category'] in ['low', 'moderate']
        assert 'risk_score' in risk_analysis

    def test_high_risk_analysis(self):
        """Test high risk classification"""
        analysis = DeviceClassificationAnalysis(
            device_name="Critical Device",
            patient_population="pediatric",
            clinical_decision_type="treatment_planning",
            autonomy_level="autonomous",
            critical_decision_impact=True
        )
        
        risk_analysis = analysis.analyze_risk()
        
        assert analysis.risk_category == RiskCategory.HIGH
        assert analysis.proposed_classification == "Class III"
        assert analysis.recommended_pathway == "pma"

    def test_moderate_risk_analysis(self):
        """Test moderate risk classification"""
        analysis = DeviceClassificationAnalysis(
            device_name="Diagnostic Device",
            patient_population="adult",
            clinical_decision_type="diagnostic_support",
            autonomy_level="advisory",
            critical_decision_impact=False
        )
        
        risk_analysis = analysis.analyze_risk()
        
        assert analysis.risk_category == RiskCategory.MODERATE
        assert analysis.proposed_classification == "Class II"
        assert analysis.recommended_pathway == "510k"

    def test_mitigation_strategies(self):
        """Test mitigation strategy generation"""
        analysis = DeviceClassificationAnalysis(
            device_name="High Risk Device",
            critical_decision_impact=True,
            autonomy_level="autonomous"
        )
        
        analysis.analyze_risk()
        
        assert len(analysis.mitigation_strategies) > 0
        assert any("oversight" in strategy.lower() 
                  for strategy in analysis.mitigation_strategies)


class TestPreSubmissionPackage:
    """Test pre-submission package functionality"""
    
    def test_qsub_creation(self):
        """Test creating Q-Sub package"""
        package = PreSubmissionPackage(
            submission_title="Test Q-Sub",
            device_name="Test Device"
        )
        
        assert package.submission_title == "Test Q-Sub"
        assert package.status == QSubStatus.PLANNING

    def test_add_regulatory_question(self):
        """Test adding regulatory questions"""
        package = PreSubmissionPackage(device_name="Test Device")
        
        package.add_regulatory_question("Is the pathway appropriate?")
        package.add_regulatory_question("What testing is needed?")
        
        assert len(package.regulatory_questions) == 2

    def test_add_fda_feedback(self):
        """Test recording FDA feedback"""
        package = PreSubmissionPackage(device_name="Test Device")
        
        package.add_fda_feedback("pathway", "510(k) pathway is appropriate")
        
        assert 'pathway' in package.fda_feedback
        assert package.status == QSubStatus.FEEDBACK_RECEIVED


class TestClinicalEvidenceItem:
    """Test clinical evidence item functionality"""
    
    def test_evidence_creation(self):
        """Test creating evidence item"""
        evidence = ClinicalEvidenceItem(
            evidence_type=EvidenceType.CLINICAL,
            title="Test Study",
            description="Clinical validation study"
        )
        
        assert evidence.evidence_type == EvidenceType.CLINICAL
        assert evidence.title == "Test Study"

    def test_completeness_assessment(self):
        """Test evidence completeness assessment"""
        evidence = ClinicalEvidenceItem(
            evidence_type=EvidenceType.CLINICAL,
            title="Complete Study",
            description="Full description",
            study_design="Prospective",
            sample_size=100,
            endpoints=["Accuracy", "Precision"],
            statistical_significance=True,
            clinical_significance=True
        )
        
        evidence.results = {'accuracy': 0.95}
        score = evidence.assess_completeness()
        
        assert score > 0.5
        assert evidence.completeness_score == score

    def test_incomplete_evidence(self):
        """Test incomplete evidence assessment"""
        evidence = ClinicalEvidenceItem(
            evidence_type=EvidenceType.ANALYTICAL,
            title="Minimal Evidence"
        )
        
        score = evidence.assess_completeness()
        
        assert score < 0.5


class TestClinicalEvidenceDossier:
    """Test clinical evidence dossier functionality"""
    
    def test_dossier_creation(self):
        """Test creating evidence dossier"""
        dossier = ClinicalEvidenceDossier(device_name="Test Device")
        
        assert dossier.device_name == "Test Device"
        assert len(dossier.evidence_items) == 0

    def test_add_evidence(self):
        """Test adding evidence to dossier"""
        dossier = ClinicalEvidenceDossier(device_name="Test Device")
        
        evidence = ClinicalEvidenceItem(
            evidence_type=EvidenceType.CLINICAL,
            title="Test Study",
            sample_size=100
        )
        
        dossier.add_evidence(evidence)
        
        assert len(dossier.evidence_items) == 1
        assert evidence.completeness_score > 0

    def test_gap_analysis_missing_evidence(self):
        """Test gap analysis with missing evidence types"""
        dossier = ClinicalEvidenceDossier(device_name="Test Device")
        
        # Only add clinical evidence
        evidence = ClinicalEvidenceItem(
            evidence_type=EvidenceType.CLINICAL,
            title="Clinical Study"
        )
        dossier.add_evidence(evidence)
        
        gap_analysis = dossier.perform_gap_analysis()
        
        assert len(gap_analysis['gaps']) > 0
        assert any('analytical' in gap.lower() for gap in gap_analysis['gaps'])

    def test_gap_analysis_complete_dossier(self):
        """Test gap analysis with complete evidence"""
        dossier = ClinicalEvidenceDossier(device_name="Test Device")
        
        # Add all required evidence types
        for evidence_type in [EvidenceType.ANALYTICAL, EvidenceType.CLINICAL, EvidenceType.PERFORMANCE]:
            evidence = ClinicalEvidenceItem(
                evidence_type=evidence_type,
                title=f"{evidence_type.value} Study",
                description="Complete study",
                study_design="Prospective",
                sample_size=200,
                endpoints=["Primary"],
                statistical_significance=True,
                clinical_significance=True
            )
            evidence.results = {'result': 'positive'}
            dossier.add_evidence(evidence)
        
        gap_analysis = dossier.perform_gap_analysis()
        
        assert gap_analysis['average_completeness'] > 0.7
        assert gap_analysis['readiness_percentage'] > 0

    def test_low_quality_evidence_detection(self):
        """Test detection of low quality evidence"""
        dossier = ClinicalEvidenceDossier(device_name="Test Device")
        
        # Add low quality evidence
        evidence = ClinicalEvidenceItem(
            evidence_type=EvidenceType.CLINICAL,
            title="Minimal Study"
        )
        dossier.add_evidence(evidence)
        
        gap_analysis = dossier.perform_gap_analysis()
        
        assert any('improvement' in gap.lower() or 'completeness' in gap.lower() 
                  for gap in gap_analysis['gaps'])


class TestQMSDocument:
    """Test QMS document functionality"""
    
    def test_qms_doc_creation(self):
        """Test creating QMS document"""
        doc = QMSDocument(
            doc_type="SOP",
            title="Test SOP",
            purpose="Test purpose",
            scope="Test scope"
        )
        
        assert doc.doc_type == "SOP"
        assert doc.status == "draft"

    def test_qms_doc_to_dict(self):
        """Test QMS document serialization"""
        doc = QMSDocument(
            doc_type="SOP",
            title="Test SOP"
        )
        
        data = doc.to_dict()
        
        assert 'doc_id' in data
        assert data['doc_type'] == "SOP"


class TestQMSSkeleton:
    """Test QMS skeleton functionality"""
    
    def test_qms_skeleton_creation(self):
        """Test creating QMS skeleton"""
        skeleton = QMSSkeleton()
        
        assert len(skeleton.documents) == 0

    def test_initialize_standard_sops(self):
        """Test initializing standard SOPs"""
        skeleton = QMSSkeleton()
        skeleton.initialize_standard_sops()
        
        assert len(skeleton.documents) >= 5
        
        # Check for key SOPs
        titles = [doc.title for doc in skeleton.documents]
        assert "Data Management and Governance" in titles
        assert "Model Change Control" in titles
        assert "Post-Market Surveillance" in titles

    def test_get_document_by_title(self):
        """Test getting document by title"""
        skeleton = QMSSkeleton()
        skeleton.initialize_standard_sops()
        
        doc = skeleton.get_document_by_title("Data Management and Governance")
        
        assert doc is not None
        assert doc.doc_type == "SOP"

    def test_qms_to_dict(self):
        """Test QMS skeleton serialization"""
        skeleton = QMSSkeleton()
        skeleton.initialize_standard_sops()
        
        data = skeleton.to_dict()
        
        assert 'total_documents' in data
        assert data['total_documents'] >= 5
        assert 'by_status' in data


class TestFDAPathwayPlanner:
    """Test FDA pathway planner functionality"""
    
    def test_planner_creation(self):
        """Test creating pathway planner"""
        planner = create_fda_pathway_planner()
        
        assert planner is not None
        assert len(planner.classifications) == 0

    def test_create_classification_analysis(self):
        """Test creating classification through planner"""
        planner = create_fda_pathway_planner()
        
        analysis = planner.create_classification_analysis(
            device_name="Test Device",
            intended_use="Diagnostic support",
            indications_for_use="For clinical use"
        )
        
        assert len(planner.classifications) == 1
        assert analysis.device_name == "Test Device"
        assert len(analysis.mitigation_strategies) > 0

    def test_create_presubmission_package(self):
        """Test creating pre-submission package"""
        planner = create_fda_pathway_planner()
        
        # First create classification
        analysis = planner.create_classification_analysis(
            device_name="Test Device",
            intended_use="Diagnostic",
            indications_for_use="Clinical use"
        )
        
        # Then create Q-Sub
        package = planner.create_presubmission_package(
            device_name="Test Device",
            classification_id=analysis.classification_id
        )
        
        assert len(planner.qsub_packages) == 1
        assert len(package.regulatory_questions) > 0

    def test_create_evidence_dossier(self):
        """Test creating evidence dossier"""
        planner = create_fda_pathway_planner()
        
        dossier = planner.create_evidence_dossier("Test Device")
        
        assert len(planner.evidence_dossiers) == 1
        assert dossier.device_name == "Test Device"

    def test_add_evidence_to_dossier(self):
        """Test adding evidence to dossier"""
        planner = create_fda_pathway_planner()
        
        dossier = planner.create_evidence_dossier("Test Device")
        
        evidence = planner.add_evidence_to_dossier(
            dossier_id=dossier.dossier_id,
            evidence_type=EvidenceType.CLINICAL,
            title="Clinical Study",
            description="Validation study",
            study_design="Prospective",
            sample_size=100
        )
        
        assert evidence is not None
        assert len(dossier.evidence_items) == 1

    def test_create_qms_skeleton(self):
        """Test creating QMS skeleton"""
        planner = create_fda_pathway_planner()
        
        skeleton = planner.create_qms_skeleton()
        
        assert len(planner.qms_skeletons) == 1
        assert len(skeleton.documents) >= 5

    def test_get_pathway_status_empty(self):
        """Test getting status from empty planner"""
        planner = create_fda_pathway_planner()
        
        status = planner.get_pathway_status()
        
        assert status['classifications']['total'] == 0
        assert status['presubmissions']['total'] == 0
        assert status['overall_readiness'] == 0.0

    def test_get_pathway_status_with_data(self):
        """Test getting status with data"""
        planner = create_fda_pathway_planner()
        
        # Create classification
        analysis = planner.create_classification_analysis(
            device_name="Test Device",
            intended_use="Diagnostic",
            indications_for_use="Clinical use"
        )
        
        # Create Q-Sub
        planner.create_presubmission_package(
            device_name="Test Device",
            classification_id=analysis.classification_id
        )
        
        # Create evidence dossier with evidence
        dossier = planner.create_evidence_dossier("Test Device")
        planner.add_evidence_to_dossier(
            dossier_id=dossier.dossier_id,
            evidence_type=EvidenceType.CLINICAL,
            title="Study",
            description="Complete",
            sample_size=100
        )
        
        # Create QMS
        planner.create_qms_skeleton()
        
        status = planner.get_pathway_status()
        
        assert status['classifications']['total'] == 1
        assert status['presubmissions']['total'] == 1
        assert status['evidence_dossiers']['total'] == 1
        assert status['qms']['total_skeletons'] == 1
        assert status['overall_readiness'] > 0

    def test_overall_readiness_calculation(self):
        """Test overall readiness calculation"""
        planner = create_fda_pathway_planner()
        
        # Add complete data
        analysis = planner.create_classification_analysis(
            device_name="Test Device",
            intended_use="Diagnostic",
            indications_for_use="Clinical use"
        )
        
        package = planner.create_presubmission_package(
            device_name="Test Device",
            classification_id=analysis.classification_id
        )
        
        # Mark Q-Sub as completed
        package.status = QSubStatus.COMPLETED
        
        dossier = planner.create_evidence_dossier("Test Device")
        
        # Add high-quality evidence
        for evidence_type in [EvidenceType.ANALYTICAL, EvidenceType.CLINICAL, EvidenceType.PERFORMANCE]:
            evidence = planner.add_evidence_to_dossier(
                dossier_id=dossier.dossier_id,
                evidence_type=evidence_type,
                title=f"{evidence_type.value} Study",
                description="Complete",
                study_design="Prospective",
                sample_size=200
            )
            evidence.endpoints = ["Primary"]
            evidence.results = {"result": "positive"}
            evidence.statistical_significance = True
            evidence.clinical_significance = True
            evidence.assess_completeness()
        
        dossier.perform_gap_analysis()
        
        skeleton = planner.create_qms_skeleton()
        for doc in skeleton.documents:
            doc.status = "approved"
        
        status = planner.get_pathway_status()
        
        assert status['overall_readiness'] > 0.5

    def test_export_pathway_plan(self):
        """Test exporting pathway plan"""
        planner = create_fda_pathway_planner()
        
        # Create some data
        planner.create_classification_analysis(
            device_name="Test Device",
            intended_use="Diagnostic",
            indications_for_use="Clinical use"
        )
        
        exported = planner.export_pathway_plan(format='json')
        
        assert 'classifications' in exported
        assert 'pathway_status' in exported
        assert 'export_timestamp' in exported

    def test_comprehensive_workflow(self):
        """Test complete FDA pathway workflow"""
        planner = create_fda_pathway_planner()
        
        # Step 1: Device Classification
        classification = planner.create_classification_analysis(
            device_name="AiMedRes System",
            intended_use="Multi-condition diagnostic support",
            indications_for_use="For use in clinical settings",
            patient_population="adult",
            clinical_decision_type="diagnostic_support",
            autonomy_level="advisory"
        )
        
        assert classification.recommended_pathway in ['510k', 'de_novo', 'pma']
        
        # Step 2: Pre-Submission
        qsub = planner.create_presubmission_package(
            device_name="AiMedRes System",
            classification_id=classification.classification_id
        )
        
        assert len(qsub.regulatory_questions) > 0
        
        # Step 3: Evidence Dossier
        dossier = planner.create_evidence_dossier("AiMedRes System")
        
        for evidence_type in [EvidenceType.ANALYTICAL, EvidenceType.CLINICAL, EvidenceType.PERFORMANCE]:
            planner.add_evidence_to_dossier(
                dossier_id=dossier.dossier_id,
                evidence_type=evidence_type,
                title=f"{evidence_type.value} Evidence",
                description="Complete evidence",
                sample_size=100
            )
        
        gap_analysis = dossier.perform_gap_analysis()
        assert gap_analysis['total_evidence_items'] == 3
        
        # Step 4: QMS Skeleton
        qms = planner.create_qms_skeleton()
        assert len(qms.documents) >= 5
        
        # Step 5: Overall Status
        status = planner.get_pathway_status()
        
        assert status['classifications']['total'] == 1
        assert status['presubmissions']['total'] == 1
        assert status['evidence_dossiers']['total'] == 1
        assert status['qms']['total_skeletons'] == 1
        assert status['overall_readiness'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
