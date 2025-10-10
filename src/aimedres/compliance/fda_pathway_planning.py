#!/usr/bin/env python3
"""
FDA Regulatory Pathway Planning Module (P9)

Comprehensive FDA pathway planning including device classification, pre-submission
consultation materials, clinical evidence dossier compilation, and QMS documentation.

Key Features:
- Device/software classification with risk categorization
- Pre-submission (Q-Sub) briefing documentation and meeting scheduling
- Clinical evidence dossier structure and gap analysis
- QMS documentation skeleton (SOPs for data management, model change control, post-market surveillance)
- Regulatory pathway decision support
- Timeline and milestone tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import uuid
import json

logger = logging.getLogger(__name__)


class RiskCategory(Enum):
    """FDA risk categorization for medical devices"""
    LOW = "low"  # Class I - Minimal risk
    MODERATE = "moderate"  # Class II - Moderate risk
    HIGH = "high"  # Class III - High risk


class SoftwareLevel(Enum):
    """Software level of concern"""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class QSubStatus(Enum):
    """Pre-submission consultation status"""
    PLANNING = "planning"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    FEEDBACK_RECEIVED = "feedback_received"
    MEETING_SCHEDULED = "meeting_scheduled"
    COMPLETED = "completed"


class EvidenceType(Enum):
    """Types of clinical evidence"""
    ANALYTICAL = "analytical"
    CLINICAL = "clinical"
    PERFORMANCE = "performance"
    USABILITY = "usability"
    REAL_WORLD = "real_world_evidence"


@dataclass
class DeviceClassificationAnalysis:
    """Device classification analysis and risk categorization"""
    classification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    device_name: str = ""
    intended_use: str = ""
    indications_for_use: str = ""
    risk_category: RiskCategory = RiskCategory.MODERATE
    software_level: SoftwareLevel = SoftwareLevel.MODERATE
    
    # Classification factors
    patient_population: str = "adult"
    clinical_decision_type: str = "diagnostic_support"  # diagnostic_support, treatment_planning, monitoring
    autonomy_level: str = "advisory"  # advisory, semi_autonomous, autonomous
    critical_decision_impact: bool = False
    
    # Risk factors
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    # Regulatory pathway recommendation
    recommended_pathway: str = "510k"  # 510k, de_novo, pma
    pathway_justification: str = ""
    
    # Product code
    product_code: str = "DQO"  # Generic AI/ML software
    proposed_classification: str = "Class II"
    
    created_at: datetime = field(default_factory=datetime.now)

    def analyze_risk(self) -> Dict[str, Any]:
        """Analyze risk factors and determine classification"""
        risk_score = 0
        risk_details = []
        
        # Patient population risk
        if "pediatric" in self.patient_population.lower() or "elderly" in self.patient_population.lower():
            risk_score += 2
            risk_details.append("Vulnerable patient population")
        
        # Decision type risk
        if "treatment" in self.clinical_decision_type:
            risk_score += 3
            risk_details.append("Treatment decisions carry higher risk")
        elif "diagnostic" in self.clinical_decision_type:
            risk_score += 2
            risk_details.append("Diagnostic decisions require validation")
        
        # Autonomy risk
        if self.autonomy_level == "autonomous":
            risk_score += 4
            risk_details.append("Autonomous decisions require extensive validation")
        elif self.autonomy_level == "semi_autonomous":
            risk_score += 2
            risk_details.append("Semi-autonomous decisions need oversight")
        
        # Critical impact
        if self.critical_decision_impact:
            risk_score += 5
            risk_details.append("Critical impact to patient safety")
        
        # Determine risk category
        if risk_score >= 10:
            self.risk_category = RiskCategory.HIGH
            self.proposed_classification = "Class III"
            self.recommended_pathway = "pma"
        elif risk_score >= 5:
            self.risk_category = RiskCategory.MODERATE
            self.proposed_classification = "Class II"
            self.recommended_pathway = "510k"
        else:
            self.risk_category = RiskCategory.LOW
            self.proposed_classification = "Class I"
            self.recommended_pathway = "exempt"
        
        self.risk_factors = risk_details
        
        return {
            'risk_score': risk_score,
            'risk_category': self.risk_category.value,
            'risk_factors': risk_details,
            'proposed_classification': self.proposed_classification,
            'recommended_pathway': self.recommended_pathway
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'classification_id': self.classification_id,
            'device_name': self.device_name,
            'intended_use': self.intended_use,
            'risk_category': self.risk_category.value,
            'software_level': self.software_level.value,
            'recommended_pathway': self.recommended_pathway,
            'proposed_classification': self.proposed_classification,
            'product_code': self.product_code,
            'risk_factors': self.risk_factors,
            'mitigation_strategies': self.mitigation_strategies,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class PreSubmissionPackage:
    """Pre-submission (Q-Sub) consultation package"""
    qsub_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submission_title: str = ""
    status: QSubStatus = QSubStatus.PLANNING
    
    # Device information
    device_name: str = ""
    device_description: str = ""
    classification_analysis: Optional[DeviceClassificationAnalysis] = None
    
    # Questions for FDA
    regulatory_questions: List[str] = field(default_factory=list)
    testing_questions: List[str] = field(default_factory=list)
    clinical_study_questions: List[str] = field(default_factory=list)
    
    # Supporting documents
    supporting_documents: List[str] = field(default_factory=list)
    
    # Submission tracking
    submission_date: Optional[datetime] = None
    target_meeting_date: Optional[datetime] = None
    actual_meeting_date: Optional[datetime] = None
    fda_feedback: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)

    def add_regulatory_question(self, question: str):
        """Add regulatory question for FDA"""
        self.regulatory_questions.append(question)

    def add_fda_feedback(self, topic: str, feedback: str):
        """Record FDA feedback"""
        self.fda_feedback[topic] = {
            'feedback': feedback,
            'received_at': datetime.now().isoformat()
        }
        self.status = QSubStatus.FEEDBACK_RECEIVED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'qsub_id': self.qsub_id,
            'submission_title': self.submission_title,
            'status': self.status.value,
            'device_name': self.device_name,
            'regulatory_questions': self.regulatory_questions,
            'testing_questions': self.testing_questions,
            'clinical_study_questions': self.clinical_study_questions,
            'fda_feedback': self.fda_feedback,
            'created_at': self.created_at.isoformat()
        }
        
        if self.submission_date:
            result['submission_date'] = self.submission_date.isoformat()
        if self.target_meeting_date:
            result['target_meeting_date'] = self.target_meeting_date.isoformat()
        if self.actual_meeting_date:
            result['actual_meeting_date'] = self.actual_meeting_date.isoformat()
            
        return result


@dataclass
class ClinicalEvidenceItem:
    """Individual piece of clinical evidence"""
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    evidence_type: EvidenceType = EvidenceType.CLINICAL
    title: str = ""
    description: str = ""
    study_design: str = ""
    sample_size: int = 0
    endpoints: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    statistical_significance: bool = False
    clinical_significance: bool = False
    limitations: List[str] = field(default_factory=list)
    publication_status: str = "unpublished"
    reference: str = ""
    completeness_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def assess_completeness(self) -> float:
        """Assess evidence completeness"""
        score = 0.0
        
        if self.title: score += 0.1
        if self.description: score += 0.1
        if self.study_design: score += 0.2
        if self.sample_size > 0: score += 0.1
        if self.endpoints: score += 0.1
        if self.results: score += 0.2
        if self.statistical_significance: score += 0.1
        if self.clinical_significance: score += 0.1
        
        self.completeness_score = score
        return score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'evidence_id': self.evidence_id,
            'evidence_type': self.evidence_type.value,
            'title': self.title,
            'study_design': self.study_design,
            'sample_size': self.sample_size,
            'endpoints': self.endpoints,
            'statistical_significance': self.statistical_significance,
            'clinical_significance': self.clinical_significance,
            'completeness_score': self.completeness_score,
            'publication_status': self.publication_status,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ClinicalEvidenceDossier:
    """Clinical evidence dossier compilation"""
    dossier_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    device_name: str = ""
    evidence_items: List[ClinicalEvidenceItem] = field(default_factory=list)
    evidence_gaps: List[str] = field(default_factory=list)
    completeness_assessment: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_evidence(self, evidence: ClinicalEvidenceItem):
        """Add evidence item to dossier"""
        evidence.assess_completeness()
        self.evidence_items.append(evidence)

    def perform_gap_analysis(self) -> Dict[str, Any]:
        """Perform gap analysis on evidence"""
        gaps = []
        evidence_by_type = {}
        
        # Group evidence by type
        for item in self.evidence_items:
            etype = item.evidence_type.value
            if etype not in evidence_by_type:
                evidence_by_type[etype] = []
            evidence_by_type[etype].append(item)
        
        # Check for required evidence types
        required_types = [EvidenceType.ANALYTICAL, EvidenceType.CLINICAL, EvidenceType.PERFORMANCE]
        for req_type in required_types:
            if req_type.value not in evidence_by_type:
                gaps.append(f"Missing {req_type.value} evidence")
        
        # Check evidence quality
        low_quality_count = sum(1 for item in self.evidence_items 
                               if item.completeness_score < 0.7)
        if low_quality_count > 0:
            gaps.append(f"{low_quality_count} evidence items need improvement (completeness < 70%)")
        
        # Check sample sizes for clinical evidence
        clinical_evidence = evidence_by_type.get(EvidenceType.CLINICAL.value, [])
        if clinical_evidence:
            total_sample_size = sum(item.sample_size for item in clinical_evidence)
            if total_sample_size < 100:
                gaps.append(f"Clinical evidence sample size too small ({total_sample_size} < 100)")
        
        self.evidence_gaps = gaps
        
        # Calculate overall completeness
        avg_completeness = (sum(item.completeness_score for item in self.evidence_items) / 
                           len(self.evidence_items)) if self.evidence_items else 0.0
        
        self.completeness_assessment = {
            'total_evidence_items': len(self.evidence_items),
            'evidence_by_type': {k: len(v) for k, v in evidence_by_type.items()},
            'average_completeness': avg_completeness,
            'gaps_identified': len(gaps),
            'gaps': gaps,
            'readiness_percentage': (avg_completeness * 100) if avg_completeness >= 0.7 else 0.0
        }
        
        return self.completeness_assessment

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'dossier_id': self.dossier_id,
            'device_name': self.device_name,
            'evidence_items': [item.to_dict() for item in self.evidence_items],
            'evidence_gaps': self.evidence_gaps,
            'completeness_assessment': self.completeness_assessment,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class QMSDocument:
    """Quality Management System document"""
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_type: str = ""  # SOP, Policy, Procedure, Work Instruction
    title: str = ""
    purpose: str = ""
    scope: str = ""
    responsibilities: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    version: str = "1.0"
    status: str = "draft"  # draft, review, approved, implemented
    created_at: datetime = field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'doc_id': self.doc_id,
            'doc_type': self.doc_type,
            'title': self.title,
            'purpose': self.purpose,
            'scope': self.scope,
            'responsibilities': self.responsibilities,
            'version': self.version,
            'status': self.status,
            'created_at': self.created_at.isoformat()
        }
        
        if self.approved_at:
            result['approved_at'] = self.approved_at.isoformat()
            
        return result


@dataclass
class QMSSkeleton:
    """Quality Management System documentation skeleton"""
    qms_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    documents: List[QMSDocument] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def initialize_standard_sops(self):
        """Initialize standard SOPs for medical AI device"""
        standard_sops = [
            {
                'doc_type': 'SOP',
                'title': 'Data Management and Governance',
                'purpose': 'Define procedures for medical data acquisition, storage, and management',
                'scope': 'All clinical data used in AI model training and validation',
                'responsibilities': ['Data Manager', 'Quality Assurance', 'IT Security'],
                'procedures': [
                    'Data acquisition from clinical sources',
                    'Data validation and quality checks',
                    'Data storage and backup procedures',
                    'Data access controls and audit logging',
                    'Data retention and disposal'
                ]
            },
            {
                'doc_type': 'SOP',
                'title': 'Model Change Control',
                'purpose': 'Define procedures for AI model updates and version control',
                'scope': 'All AI model changes including training, validation, and deployment',
                'responsibilities': ['ML Engineer', 'Quality Assurance', 'Regulatory Affairs'],
                'procedures': [
                    'Change request and approval process',
                    'Model retraining procedures',
                    'Validation testing requirements',
                    'Version control and documentation',
                    'Deployment approval workflow',
                    'Rollback procedures'
                ]
            },
            {
                'doc_type': 'SOP',
                'title': 'Post-Market Surveillance',
                'purpose': 'Define procedures for ongoing monitoring of device performance',
                'scope': 'All deployed AI models in clinical use',
                'responsibilities': ['Clinical Operations', 'Quality Assurance', 'Regulatory Affairs'],
                'procedures': [
                    'Performance monitoring and metrics',
                    'Adverse event reporting',
                    'Complaint handling procedures',
                    'Periodic safety updates',
                    'Corrective and preventive actions (CAPA)',
                    'Regulatory reporting requirements'
                ]
            },
            {
                'doc_type': 'SOP',
                'title': 'Software Validation and Verification',
                'purpose': 'Define procedures for software testing and validation',
                'scope': 'All software components of the AI medical device',
                'responsibilities': ['Software QA', 'Development Team', 'Clinical Validation'],
                'procedures': [
                    'Unit testing requirements',
                    'Integration testing procedures',
                    'System testing protocols',
                    'Clinical validation studies',
                    'User acceptance testing',
                    'Traceability matrix maintenance'
                ]
            },
            {
                'doc_type': 'SOP',
                'title': 'Risk Management',
                'purpose': 'Define procedures for risk identification and mitigation',
                'scope': 'All aspects of AI device development and deployment',
                'responsibilities': ['Risk Manager', 'Quality Assurance', 'Clinical Team'],
                'procedures': [
                    'Risk identification and assessment',
                    'Risk control measures',
                    'Risk-benefit analysis',
                    'Residual risk evaluation',
                    'Risk management file maintenance'
                ]
            }
        ]
        
        for sop_data in standard_sops:
            doc = QMSDocument(**sop_data)
            self.documents.append(doc)

    def get_document_by_title(self, title: str) -> Optional[QMSDocument]:
        """Get document by title"""
        for doc in self.documents:
            if doc.title == title:
                return doc
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'qms_id': self.qms_id,
            'documents': [doc.to_dict() for doc in self.documents],
            'total_documents': len(self.documents),
            'by_status': {
                'draft': sum(1 for d in self.documents if d.status == 'draft'),
                'review': sum(1 for d in self.documents if d.status == 'review'),
                'approved': sum(1 for d in self.documents if d.status == 'approved')
            },
            'created_at': self.created_at.isoformat()
        }


class FDAPathwayPlanner:
    """Manages FDA regulatory pathway planning"""
    
    def __init__(self):
        self.classifications: Dict[str, DeviceClassificationAnalysis] = {}
        self.qsub_packages: Dict[str, PreSubmissionPackage] = {}
        self.evidence_dossiers: Dict[str, ClinicalEvidenceDossier] = {}
        self.qms_skeletons: Dict[str, QMSSkeleton] = {}
        logger.info("FDA Pathway Planner initialized")

    def create_classification_analysis(self,
                                      device_name: str,
                                      intended_use: str,
                                      indications_for_use: str,
                                      patient_population: str = "adult",
                                      clinical_decision_type: str = "diagnostic_support",
                                      autonomy_level: str = "advisory",
                                      critical_decision_impact: bool = False) -> DeviceClassificationAnalysis:
        """Create device classification analysis"""
        analysis = DeviceClassificationAnalysis(
            device_name=device_name,
            intended_use=intended_use,
            indications_for_use=indications_for_use,
            patient_population=patient_population,
            clinical_decision_type=clinical_decision_type,
            autonomy_level=autonomy_level,
            critical_decision_impact=critical_decision_impact
        )
        
        # Perform risk analysis
        risk_analysis = analysis.analyze_risk()
        
        # Add mitigation strategies based on risk
        if analysis.risk_category == RiskCategory.HIGH:
            analysis.mitigation_strategies = [
                "Extensive clinical validation required",
                "Human oversight mandatory for all decisions",
                "Real-time performance monitoring",
                "Regular safety audits"
            ]
        elif analysis.risk_category == RiskCategory.MODERATE:
            analysis.mitigation_strategies = [
                "Clinical validation with appropriate sample size",
                "Human review of high-risk cases",
                "Performance monitoring in production",
                "Periodic safety reviews"
            ]
        
        self.classifications[analysis.classification_id] = analysis
        logger.info(f"Created classification analysis for {device_name}: {analysis.recommended_pathway}")
        
        return analysis

    def create_presubmission_package(self,
                                    device_name: str,
                                    classification_id: str) -> PreSubmissionPackage:
        """Create pre-submission consultation package"""
        classification = self.classifications.get(classification_id)
        
        package = PreSubmissionPackage(
            submission_title=f"Pre-Submission for {device_name}",
            device_name=device_name,
            classification_analysis=classification
        )
        
        # Add standard regulatory questions
        standard_questions = [
            "Is the proposed regulatory pathway appropriate for this device?",
            "What testing should be included in the premarket submission?",
            "Are the proposed indications for use appropriate?",
            "What clinical data is needed to support the submission?",
            "Are there any special controls that should be considered?"
        ]
        
        for q in standard_questions:
            package.add_regulatory_question(q)
        
        self.qsub_packages[package.qsub_id] = package
        logger.info(f"Created pre-submission package for {device_name}")
        
        return package

    def create_evidence_dossier(self, device_name: str) -> ClinicalEvidenceDossier:
        """Create clinical evidence dossier"""
        dossier = ClinicalEvidenceDossier(device_name=device_name)
        self.evidence_dossiers[dossier.dossier_id] = dossier
        logger.info(f"Created evidence dossier for {device_name}")
        return dossier

    def add_evidence_to_dossier(self,
                               dossier_id: str,
                               evidence_type: EvidenceType,
                               title: str,
                               description: str,
                               study_design: str = "",
                               sample_size: int = 0) -> Optional[ClinicalEvidenceItem]:
        """Add evidence to dossier"""
        if dossier_id not in self.evidence_dossiers:
            return None
        
        evidence = ClinicalEvidenceItem(
            evidence_type=evidence_type,
            title=title,
            description=description,
            study_design=study_design,
            sample_size=sample_size
        )
        
        self.evidence_dossiers[dossier_id].add_evidence(evidence)
        logger.info(f"Added {evidence_type.value} evidence to dossier {dossier_id}")
        
        return evidence

    def create_qms_skeleton(self) -> QMSSkeleton:
        """Create QMS documentation skeleton"""
        skeleton = QMSSkeleton()
        skeleton.initialize_standard_sops()
        
        self.qms_skeletons[skeleton.qms_id] = skeleton
        logger.info(f"Created QMS skeleton with {len(skeleton.documents)} SOPs")
        
        return skeleton

    def get_pathway_status(self) -> Dict[str, Any]:
        """Get overall FDA pathway planning status"""
        # Calculate progress metrics
        classifications_complete = len(self.classifications)
        qsubs_complete = sum(1 for p in self.qsub_packages.values() 
                            if p.status == QSubStatus.COMPLETED)
        
        # Evidence dossier readiness
        dossiers_ready = 0
        for dossier in self.evidence_dossiers.values():
            if dossier.completeness_assessment:
                if dossier.completeness_assessment.get('average_completeness', 0) >= 0.7:
                    dossiers_ready += 1
        
        # QMS completion
        qms_approved = 0
        total_qms_docs = 0
        for skeleton in self.qms_skeletons.values():
            total_qms_docs += len(skeleton.documents)
            qms_approved += sum(1 for d in skeleton.documents if d.status == 'approved')
        
        return {
            'classifications': {
                'total': classifications_complete,
                'by_risk': {
                    risk.value: sum(1 for c in self.classifications.values() 
                                   if c.risk_category == risk)
                    for risk in RiskCategory
                },
                'by_pathway': {}
            },
            'presubmissions': {
                'total': len(self.qsub_packages),
                'completed': qsubs_complete,
                'by_status': {
                    status.value: sum(1 for p in self.qsub_packages.values() 
                                     if p.status == status)
                    for status in QSubStatus
                }
            },
            'evidence_dossiers': {
                'total': len(self.evidence_dossiers),
                'ready': dossiers_ready,
                'readiness_percentage': (dossiers_ready / len(self.evidence_dossiers) * 100) 
                                       if self.evidence_dossiers else 0.0
            },
            'qms': {
                'total_skeletons': len(self.qms_skeletons),
                'total_documents': total_qms_docs,
                'approved_documents': qms_approved,
                'completion_percentage': (qms_approved / total_qms_docs * 100) 
                                        if total_qms_docs > 0 else 0.0
            },
            'overall_readiness': self._calculate_overall_readiness()
        }

    def _calculate_overall_readiness(self) -> float:
        """Calculate overall FDA pathway readiness"""
        scores = []
        
        # Classification completion
        if self.classifications:
            scores.append(1.0)  # 100% if any classifications exist
        else:
            scores.append(0.0)
        
        # Q-Sub completion
        if self.qsub_packages:
            qsub_score = sum(1 for p in self.qsub_packages.values() 
                            if p.status == QSubStatus.COMPLETED) / len(self.qsub_packages)
            scores.append(qsub_score)
        else:
            scores.append(0.0)
        
        # Evidence dossier readiness
        if self.evidence_dossiers:
            evidence_scores = []
            for dossier in self.evidence_dossiers.values():
                if dossier.completeness_assessment:
                    evidence_scores.append(
                        dossier.completeness_assessment.get('average_completeness', 0)
                    )
            if evidence_scores:
                scores.append(sum(evidence_scores) / len(evidence_scores))
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)
        
        # QMS completion
        if self.qms_skeletons:
            total_docs = sum(len(s.documents) for s in self.qms_skeletons.values())
            approved_docs = sum(
                sum(1 for d in s.documents if d.status == 'approved')
                for s in self.qms_skeletons.values()
            )
            scores.append(approved_docs / total_docs if total_docs > 0 else 0.0)
        else:
            scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0

    def export_pathway_plan(self, format: str = 'json') -> str:
        """Export FDA pathway planning data"""
        data = {
            'classifications': [c.to_dict() for c in self.classifications.values()],
            'presubmissions': [p.to_dict() for p in self.qsub_packages.values()],
            'evidence_dossiers': [d.to_dict() for d in self.evidence_dossiers.values()],
            'qms_skeletons': [s.to_dict() for s in self.qms_skeletons.values()],
            'pathway_status': self.get_pathway_status(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        if format == 'json':
            return json.dumps(data, indent=2)
        else:
            return str(data)


def create_fda_pathway_planner() -> FDAPathwayPlanner:
    """Factory function to create FDA pathway planner"""
    return FDAPathwayPlanner()


# Example usage demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create planner
    planner = create_fda_pathway_planner()
    
    # Create classification analysis
    classification = planner.create_classification_analysis(
        device_name="AiMedRes Multi-Condition Diagnostic Support System",
        intended_use="To provide diagnostic decision support for multiple neurological conditions",
        indications_for_use="For use by licensed healthcare professionals in clinical settings",
        patient_population="adult",
        clinical_decision_type="diagnostic_support",
        autonomy_level="advisory",
        critical_decision_impact=False
    )
    
    print("\n=== Device Classification ===")
    print(f"Risk Category: {classification.risk_category.value}")
    print(f"Recommended Pathway: {classification.recommended_pathway}")
    print(f"Proposed Classification: {classification.proposed_classification}")
    
    # Create pre-submission package
    qsub = planner.create_presubmission_package(
        device_name="AiMedRes",
        classification_id=classification.classification_id
    )
    
    print(f"\n=== Pre-Submission Package ===")
    print(f"Q-Sub ID: {qsub.qsub_id}")
    print(f"Regulatory Questions: {len(qsub.regulatory_questions)}")
    
    # Create evidence dossier
    dossier = planner.create_evidence_dossier("AiMedRes")
    
    # Add sample evidence
    planner.add_evidence_to_dossier(
        dossier_id=dossier.dossier_id,
        evidence_type=EvidenceType.CLINICAL,
        title="Multi-Center Clinical Validation Study",
        description="Prospective validation across 5 clinical sites",
        study_design="Prospective observational",
        sample_size=1000
    )
    
    # Perform gap analysis
    gap_analysis = dossier.perform_gap_analysis()
    print(f"\n=== Evidence Dossier Gap Analysis ===")
    print(json.dumps(gap_analysis, indent=2))
    
    # Create QMS skeleton
    qms = planner.create_qms_skeleton()
    print(f"\n=== QMS Skeleton ===")
    print(f"Total SOPs: {len(qms.documents)}")
    for doc in qms.documents:
        print(f"  - {doc.title}")
    
    # Overall status
    status = planner.get_pathway_status()
    print(f"\n=== Overall FDA Pathway Status ===")
    print(json.dumps(status, indent=2))
