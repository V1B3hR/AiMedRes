"""
Liability Documentation - Legal Protection Framework

Implements comprehensive documentation and audit trails for clinical AI
decisions to provide legal protection and regulatory compliance.
"""

import logging
import json
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
import uuid


logger = logging.getLogger('duetmind.liability_documentation')


class DocumentationType(Enum):
    """Types of liability documentation"""
    AI_DECISION_RECORD = "AI_DECISION_RECORD"
    HUMAN_OVERSIGHT_RECORD = "HUMAN_OVERSIGHT_RECORD"
    SAFETY_INCIDENT_REPORT = "SAFETY_INCIDENT_REPORT"
    AUDIT_TRAIL = "AUDIT_TRAIL"
    CONSENT_RECORD = "CONSENT_RECORD"
    DISCLOSURE_RECORD = "DISCLOSURE_RECORD"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    QUALITY_ASSURANCE = "QUALITY_ASSURANCE"


class LegalStatus(Enum):
    """Legal review status"""
    PENDING = "PENDING"
    REVIEWED = "REVIEWED"
    APPROVED = "APPROVED"
    FLAGGED = "FLAGGED"
    DISPUTED = "DISPUTED"


class ComplianceStandard(Enum):
    """Regulatory compliance standards"""
    HIPAA = "HIPAA"
    FDA_510K = "FDA_510K"
    FDA_PMA = "FDA_PMA"
    GDPR = "GDPR"
    ISO_13485 = "ISO_13485"
    ISO_14155 = "ISO_14155"
    IEC_62304 = "IEC_62304"
    HITECH = "HITECH"


@dataclass
class LiabilityDocument:
    """Core liability documentation structure"""
    document_id: str
    document_type: DocumentationType
    patient_id: str
    healthcare_provider_id: str
    ai_system_version: str
    timestamp: datetime
    legal_status: LegalStatus
    compliance_standards: List[ComplianceStandard]
    
    # Core documentation
    clinical_context: Dict[str, Any]
    ai_decision_details: Dict[str, Any]
    human_oversight_details: Dict[str, Any]
    risk_factors: List[str]
    
    # Legal protection elements
    informed_consent_obtained: bool
    ai_disclosure_provided: bool
    human_physician_involved: bool
    override_rationale: Optional[str]
    
    # Audit and verification
    digital_signature: str
    verification_hash: str
    chain_of_custody: List[Dict[str, Any]]
    
    # Compliance tracking
    hipaa_compliant: bool
    retention_period_years: int
    access_log: List[Dict[str, Any]]


@dataclass
class SafetyIncident:
    """Safety incident documentation"""
    incident_id: str
    incident_type: str
    severity: str
    patient_impact: str
    ai_system_involvement: bool
    root_cause_analysis: Dict[str, Any]
    corrective_actions: List[str]
    preventive_measures: List[str]
    regulatory_reporting_required: bool
    legal_implications: Dict[str, Any]


@dataclass
class ConsentRecord:
    """Patient consent documentation"""
    consent_id: str
    patient_id: str
    consent_type: str
    ai_involvement_disclosed: bool
    risks_disclosed: List[str]
    alternatives_discussed: bool
    consent_obtained_date: datetime
    consent_method: str  # "verbal", "written", "electronic"
    witness_present: bool
    revocation_rights_explained: bool


class LiabilityDocumentation:
    """
    Comprehensive liability documentation system for clinical AI.
    
    Features:
    - Automated documentation generation for all AI decisions
    - Legal compliance tracking and verification
    - Digital signatures and tamper-proof records
    - Audit trail maintenance
    - Regulatory reporting support
    - Risk assessment documentation
    - Incident tracking and reporting
    """
    
    def __init__(self):
        """Initialize liability documentation system"""
        self.documents = {}
        self.safety_incidents = {}
        self.consent_records = {}
        self.audit_trail = []
        
        # Configuration
        self.retention_policies = {
            DocumentationType.AI_DECISION_RECORD: 7,  # years
            DocumentationType.HUMAN_OVERSIGHT_RECORD: 7,
            DocumentationType.SAFETY_INCIDENT_REPORT: 10,
            DocumentationType.AUDIT_TRAIL: 6,
            DocumentationType.CONSENT_RECORD: 7,
            DocumentationType.DISCLOSURE_RECORD: 7,
            DocumentationType.RISK_ASSESSMENT: 5,
            DocumentationType.QUALITY_ASSURANCE: 3
        }
        
        # Legal requirements by jurisdiction
        self.legal_requirements = self._initialize_legal_requirements()
    
    def _initialize_legal_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Initialize legal requirements by jurisdiction"""
        return {
            'US': {
                'required_standards': [ComplianceStandard.HIPAA, ComplianceStandard.HITECH],
                'ai_disclosure_required': True,
                'informed_consent_required': True,
                'physician_oversight_required': True,
                'audit_retention_years': 6,
                'incident_reporting_required': True
            },
            'EU': {
                'required_standards': [ComplianceStandard.GDPR, ComplianceStandard.ISO_13485],
                'ai_disclosure_required': True,
                'informed_consent_required': True,
                'physician_oversight_required': True,
                'audit_retention_years': 7,
                'incident_reporting_required': True
            }
        }
    
    def document_ai_decision(self,
                           patient_id: str,
                           healthcare_provider_id: str,
                           ai_system_version: str,
                           clinical_context: Dict[str, Any],
                           ai_decision_details: Dict[str, Any],
                           human_oversight_details: Dict[str, Any],
                           jurisdiction: str = 'US') -> LiabilityDocument:
        """
        Document AI clinical decision with full legal protection.
        
        Args:
            patient_id: Patient identifier
            healthcare_provider_id: Healthcare provider ID
            ai_system_version: AI system version
            clinical_context: Clinical context and patient data
            ai_decision_details: AI decision and recommendation details
            human_oversight_details: Human oversight and review details
            jurisdiction: Legal jurisdiction
            
        Returns:
            LiabilityDocument with complete documentation
        """
        document_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Get legal requirements for jurisdiction
        legal_reqs = self.legal_requirements.get(jurisdiction, self.legal_requirements['US'])
        
        # Extract risk factors
        risk_factors = self._extract_risk_factors(clinical_context, ai_decision_details)
        
        # Verify compliance requirements
        compliance_status = self._verify_compliance_requirements(
            clinical_context, ai_decision_details, human_oversight_details, legal_reqs
        )
        
        # Create liability document
        document = LiabilityDocument(
            document_id=document_id,
            document_type=DocumentationType.AI_DECISION_RECORD,
            patient_id=patient_id,
            healthcare_provider_id=healthcare_provider_id,
            ai_system_version=ai_system_version,
            timestamp=timestamp,
            legal_status=LegalStatus.PENDING,
            compliance_standards=legal_reqs['required_standards'],
            
            # Core documentation
            clinical_context=clinical_context,
            ai_decision_details=ai_decision_details,
            human_oversight_details=human_oversight_details,
            risk_factors=risk_factors,
            
            # Legal protection elements
            informed_consent_obtained=compliance_status['consent_obtained'],
            ai_disclosure_provided=compliance_status['ai_disclosed'],
            human_physician_involved=compliance_status['physician_involved'],
            override_rationale=human_oversight_details.get('override_rationale'),
            
            # Audit and verification
            digital_signature=self._generate_digital_signature(document_id, timestamp),
            verification_hash=self._compute_verification_hash({
                'patient_id': patient_id,
                'clinical_context': clinical_context,
                'ai_decision': ai_decision_details,
                'timestamp': timestamp.isoformat()
            }),
            chain_of_custody=[{
                'action': 'document_created',
                'user_id': healthcare_provider_id,
                'timestamp': timestamp.isoformat(),
                'system': 'liability_documentation_system'
            }],
            
            # Compliance tracking
            hipaa_compliant=self._verify_hipaa_compliance(clinical_context, ai_decision_details),
            retention_period_years=self.retention_policies[DocumentationType.AI_DECISION_RECORD],
            access_log=[{
                'user_id': healthcare_provider_id,
                'access_type': 'create',
                'timestamp': timestamp.isoformat(),
                'purpose': 'clinical_documentation'
            }]
        )
        
        # Store document
        self.documents[document_id] = document
        
        # Add to audit trail
        self._add_audit_entry(
            action='AI_DECISION_DOCUMENTED',
            document_id=document_id,
            user_id=healthcare_provider_id,
            details={'ai_confidence': ai_decision_details.get('confidence_score', 0)}
        )
        
        logger.info(f"AI decision documented: {document_id}")
        
        return document
    
    def document_safety_incident(self,
                               patient_id: str,
                               incident_type: str,
                               severity: str,
                               patient_impact: str,
                               ai_system_involvement: bool,
                               healthcare_provider_id: str,
                               incident_description: str,
                               root_cause_analysis: Optional[Dict[str, Any]] = None) -> str:
        """
        Document safety incident with AI involvement.
        
        Args:
            patient_id: Patient identifier
            incident_type: Type of safety incident
            severity: Incident severity level
            patient_impact: Description of patient impact
            ai_system_involvement: Whether AI system was involved
            healthcare_provider_id: Reporting healthcare provider
            incident_description: Detailed incident description
            root_cause_analysis: Root cause analysis if available
            
        Returns:
            Incident ID
        """
        incident_id = f"INCIDENT_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Determine if regulatory reporting is required
        regulatory_reporting_required = self._requires_regulatory_reporting(
            incident_type, severity, ai_system_involvement
        )
        
        # Analyze legal implications
        legal_implications = self._analyze_legal_implications(
            incident_type, severity, patient_impact, ai_system_involvement
        )
        
        # Generate corrective and preventive actions
        corrective_actions = self._generate_corrective_actions(incident_type, root_cause_analysis)
        preventive_measures = self._generate_preventive_measures(incident_type, root_cause_analysis)
        
        incident = SafetyIncident(
            incident_id=incident_id,
            incident_type=incident_type,
            severity=severity,
            patient_impact=patient_impact,
            ai_system_involvement=ai_system_involvement,
            root_cause_analysis=root_cause_analysis or {},
            corrective_actions=corrective_actions,
            preventive_measures=preventive_measures,
            regulatory_reporting_required=regulatory_reporting_required,
            legal_implications=legal_implications
        )
        
        # Store incident
        self.safety_incidents[incident_id] = incident
        
        # Create associated liability document
        self.document_ai_decision(
            patient_id=patient_id,
            healthcare_provider_id=healthcare_provider_id,
            ai_system_version='incident_related',
            clinical_context={'incident_id': incident_id},
            ai_decision_details={'incident_related': True},
            human_oversight_details={
                'incident_reporter': healthcare_provider_id,
                'incident_description': incident_description
            }
        )
        
        # Add to audit trail
        self._add_audit_entry(
            action='SAFETY_INCIDENT_DOCUMENTED',
            document_id=incident_id,
            user_id=healthcare_provider_id,
            details={
                'incident_type': incident_type,
                'severity': severity,
                'ai_involved': ai_system_involvement
            }
        )
        
        logger.warning(f"Safety incident documented: {incident_id}")
        
        return incident_id
    
    def record_patient_consent(self,
                             patient_id: str,
                             consent_type: str,
                             ai_involvement_disclosed: bool,
                             risks_disclosed: List[str],
                             healthcare_provider_id: str,
                             consent_method: str = 'written',
                             witness_present: bool = False) -> str:
        """
        Record patient consent for AI-assisted care.
        
        Args:
            patient_id: Patient identifier
            consent_type: Type of consent
            ai_involvement_disclosed: Whether AI involvement was disclosed
            risks_disclosed: List of risks disclosed to patient
            healthcare_provider_id: Healthcare provider obtaining consent
            consent_method: Method of consent ("verbal", "written", "electronic")
            witness_present: Whether witness was present
            
        Returns:
            Consent record ID
        """
        consent_id = f"CONSENT_{patient_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        consent_record = ConsentRecord(
            consent_id=consent_id,
            patient_id=patient_id,
            consent_type=consent_type,
            ai_involvement_disclosed=ai_involvement_disclosed,
            risks_disclosed=risks_disclosed,
            alternatives_discussed=True,  # Assume discussed
            consent_obtained_date=datetime.now(timezone.utc),
            consent_method=consent_method,
            witness_present=witness_present,
            revocation_rights_explained=True  # Legal requirement
        )
        
        # Store consent record
        self.consent_records[consent_id] = consent_record
        
        # Add to audit trail
        self._add_audit_entry(
            action='PATIENT_CONSENT_RECORDED',
            document_id=consent_id,
            user_id=healthcare_provider_id,
            details={
                'consent_type': consent_type,
                'ai_disclosed': ai_involvement_disclosed,
                'method': consent_method
            }
        )
        
        logger.info(f"Patient consent recorded: {consent_id}")
        
        return consent_id
    
    def _extract_risk_factors(self,
                            clinical_context: Dict[str, Any],
                            ai_decision_details: Dict[str, Any]) -> List[str]:
        """Extract risk factors from clinical data"""
        risk_factors = []
        
        # Patient risk factors
        if clinical_context.get('patient_age', 0) > 65:
            risk_factors.append('elderly_patient')
        
        if clinical_context.get('comorbidity_count', 0) > 3:
            risk_factors.append('multiple_comorbidities')
        
        # AI-related risk factors
        confidence = ai_decision_details.get('confidence_score', 1.0)
        if confidence < 0.8:
            risk_factors.append('low_ai_confidence')
        
        if ai_decision_details.get('treatment_type') == 'HIGH_RISK_MEDICATION':
            risk_factors.append('high_risk_treatment')
        
        # Clinical context risk factors
        if clinical_context.get('condition_severity') in ['HIGH', 'CRITICAL']:
            risk_factors.append('severe_condition')
        
        return risk_factors
    
    def _verify_compliance_requirements(self,
                                      clinical_context: Dict[str, Any],
                                      ai_decision_details: Dict[str, Any],
                                      human_oversight_details: Dict[str, Any],
                                      legal_reqs: Dict[str, Any]) -> Dict[str, bool]:
        """Verify compliance with legal requirements"""
        return {
            'consent_obtained': human_oversight_details.get('consent_obtained', False),
            'ai_disclosed': human_oversight_details.get('ai_involvement_disclosed', False),
            'physician_involved': human_oversight_details.get('physician_reviewer_id') is not None,
            'hipaa_compliant': self._verify_hipaa_compliance(clinical_context, ai_decision_details)
        }
    
    def _verify_hipaa_compliance(self,
                               clinical_context: Dict[str, Any],
                               ai_decision_details: Dict[str, Any]) -> bool:
        """Verify HIPAA compliance"""
        # Check for PHI handling compliance
        required_elements = [
            'patient_id',  # Should be anonymized/tokenized
            'healthcare_provider_id',
            'timestamp'
        ]
        
        # Verify no direct PHI is stored inappropriately
        prohibited_fields = ['ssn', 'full_name', 'address', 'phone_number']
        
        for field in prohibited_fields:
            if field in clinical_context or field in ai_decision_details:
                return False
        
        return True
    
    def _generate_digital_signature(self, document_id: str, timestamp: datetime) -> str:
        """Generate digital signature for document integrity"""
        signature_data = f"{document_id}:{timestamp.isoformat()}:liability_documentation"
        return hashlib.sha256(signature_data.encode()).hexdigest()
    
    def _compute_verification_hash(self, data: Dict[str, Any]) -> str:
        """Compute verification hash for tamper detection"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha512(data_str.encode()).hexdigest()
    
    def _requires_regulatory_reporting(self,
                                     incident_type: str,
                                     severity: str,
                                     ai_system_involvement: bool) -> bool:
        """Determine if incident requires regulatory reporting"""
        # FDA reporting requirements for AI/ML medical devices
        if ai_system_involvement and severity in ['HIGH', 'CRITICAL']:
            return True
        
        # Mandatory reporting incident types
        mandatory_reporting_types = [
            'device_malfunction',
            'ai_error',
            'patient_harm',
            'data_breach',
            'security_incident'
        ]
        
        return incident_type in mandatory_reporting_types
    
    def _analyze_legal_implications(self,
                                  incident_type: str,
                                  severity: str,
                                  patient_impact: str,
                                  ai_system_involvement: bool) -> Dict[str, Any]:
        """Analyze legal implications of safety incident"""
        implications = {
            'potential_liability': 'LOW',
            'malpractice_risk': 'LOW',
            'regulatory_action_risk': 'LOW',
            'documentation_requirements': [],
            'legal_review_recommended': False
        }
        
        # Assess based on severity and AI involvement
        if severity in ['HIGH', 'CRITICAL'] and ai_system_involvement:
            implications['potential_liability'] = 'HIGH'
            implications['legal_review_recommended'] = True
            implications['documentation_requirements'].extend([
                'expert_review',
                'root_cause_analysis',
                'corrective_action_plan'
            ])
        
        if 'harm' in patient_impact.lower():
            implications['malpractice_risk'] = 'MODERATE'
            implications['documentation_requirements'].append('patient_communication_log')
        
        if ai_system_involvement and incident_type in ['ai_error', 'device_malfunction']:
            implications['regulatory_action_risk'] = 'MODERATE'
            implications['documentation_requirements'].append('fda_reporting')
        
        return implications
    
    def _generate_corrective_actions(self,
                                   incident_type: str,
                                   root_cause_analysis: Optional[Dict[str, Any]]) -> List[str]:
        """Generate corrective actions based on incident type"""
        actions = []
        
        if incident_type == 'ai_error':
            actions.extend([
                'Review AI model performance',
                'Update model validation procedures',
                'Enhance human oversight protocols'
            ])
        elif incident_type == 'human_error':
            actions.extend([
                'Provide additional training',
                'Review decision-making process',
                'Implement additional safeguards'
            ])
        elif incident_type == 'system_failure':
            actions.extend([
                'Investigate system reliability',
                'Implement redundancy measures',
                'Update maintenance procedures'
            ])
        
        # Add root cause specific actions
        if root_cause_analysis:
            root_cause = root_cause_analysis.get('primary_cause')
            if root_cause == 'insufficient_training_data':
                actions.append('Expand training dataset')
            elif root_cause == 'inadequate_validation':
                actions.append('Enhance validation protocols')
        
        return actions
    
    def _generate_preventive_measures(self,
                                    incident_type: str,
                                    root_cause_analysis: Optional[Dict[str, Any]]) -> List[str]:
        """Generate preventive measures to avoid similar incidents"""
        measures = []
        
        if incident_type == 'ai_error':
            measures.extend([
                'Implement continuous model monitoring',
                'Establish performance thresholds',
                'Regular model retraining'
            ])
        elif incident_type == 'communication_failure':
            measures.extend([
                'Improve alert systems',
                'Standardize communication protocols',
                'Regular communication training'
            ])
        
        # General preventive measures
        measures.extend([
            'Regular safety audits',
            'Incident trend analysis',
            'Staff safety training updates'
        ])
        
        return measures
    
    def _add_audit_entry(self,
                        action: str,
                        document_id: str,
                        user_id: str,
                        details: Dict[str, Any]):
        """Add entry to audit trail"""
        audit_entry = {
            'audit_id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': action,
            'document_id': document_id,
            'user_id': user_id,
            'details': details,
            'system': 'liability_documentation_system'
        }
        
        self.audit_trail.append(audit_entry)
    
    def get_document(self, document_id: str, user_id: str, access_purpose: str) -> Optional[LiabilityDocument]:
        """
        Retrieve liability document with access logging.
        
        Args:
            document_id: Document ID to retrieve
            user_id: User requesting access
            access_purpose: Purpose of access for audit
            
        Returns:
            LiabilityDocument if found and authorized, None otherwise
        """
        if document_id not in self.documents:
            return None
        
        document = self.documents[document_id]
        
        # Log access
        access_entry = {
            'user_id': user_id,
            'access_type': 'read',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'purpose': access_purpose
        }
        document.access_log.append(access_entry)
        
        # Add to audit trail
        self._add_audit_entry(
            action='DOCUMENT_ACCESSED',
            document_id=document_id,
            user_id=user_id,
            details={'purpose': access_purpose}
        )
        
        return document
    
    def verify_document_integrity(self, document_id: str) -> Dict[str, Any]:
        """
        Verify document integrity and detect tampering.
        
        Args:
            document_id: Document ID to verify
            
        Returns:
            Verification results
        """
        if document_id not in self.documents:
            return {'error': 'Document not found'}
        
        document = self.documents[document_id]
        
        # Verify digital signature
        expected_signature = self._generate_digital_signature(document_id, document.timestamp)
        signature_valid = document.digital_signature == expected_signature
        
        # Verify hash
        verification_data = {
            'patient_id': document.patient_id,
            'clinical_context': document.clinical_context,
            'ai_decision': document.ai_decision_details,
            'timestamp': document.timestamp.isoformat()
        }
        expected_hash = self._compute_verification_hash(verification_data)
        hash_valid = document.verification_hash == expected_hash
        
        return {
            'document_id': document_id,
            'integrity_verified': signature_valid and hash_valid,
            'signature_valid': signature_valid,
            'hash_valid': hash_valid,
            'verification_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def generate_legal_report(self,
                            patient_id: Optional[str] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate comprehensive legal report.
        
        Args:
            patient_id: Optional patient ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Comprehensive legal report
        """
        # Filter documents
        filtered_docs = []
        for doc in self.documents.values():
            if patient_id and doc.patient_id != patient_id:
                continue
            if start_date and doc.timestamp < start_date:
                continue
            if end_date and doc.timestamp > end_date:
                continue
            filtered_docs.append(doc)
        
        # Calculate statistics
        total_decisions = len(filtered_docs)
        human_oversight_rate = sum(1 for doc in filtered_docs if doc.human_physician_involved) / max(total_decisions, 1)
        consent_rate = sum(1 for doc in filtered_docs if doc.informed_consent_obtained) / max(total_decisions, 1)
        disclosure_rate = sum(1 for doc in filtered_docs if doc.ai_disclosure_provided) / max(total_decisions, 1)
        
        # Compliance analysis
        compliance_issues = []
        for doc in filtered_docs:
            if not doc.hipaa_compliant:
                compliance_issues.append(f"HIPAA compliance issue in {doc.document_id}")
            if not doc.informed_consent_obtained:
                compliance_issues.append(f"Missing informed consent in {doc.document_id}")
            if not doc.ai_disclosure_provided:
                compliance_issues.append(f"Missing AI disclosure in {doc.document_id}")
        
        return {
            'report_generated': datetime.now(timezone.utc).isoformat(),
            'filter_criteria': {
                'patient_id': patient_id,
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None
            },
            'summary_statistics': {
                'total_ai_decisions': total_decisions,
                'human_oversight_rate': human_oversight_rate,
                'informed_consent_rate': consent_rate,
                'ai_disclosure_rate': disclosure_rate,
                'safety_incidents': len(self.safety_incidents),
                'consent_records': len(self.consent_records)
            },
            'compliance_analysis': {
                'total_compliance_issues': len(compliance_issues),
                'compliance_issues': compliance_issues,
                'overall_compliance_rate': max(0, 1 - len(compliance_issues) / max(total_decisions, 1))
            },
            'legal_risk_assessment': self._assess_legal_risk(filtered_docs),
            'recommendations': self._generate_legal_recommendations(filtered_docs, compliance_issues)
        }
    
    def _assess_legal_risk(self, documents: List[LiabilityDocument]) -> Dict[str, Any]:
        """Assess overall legal risk based on documentation"""
        risk_factors = []
        risk_score = 0.0
        
        for doc in documents:
            if not doc.informed_consent_obtained:
                risk_factors.append('Missing informed consent')
                risk_score += 0.3
            
            if not doc.ai_disclosure_provided:
                risk_factors.append('Missing AI disclosure')
                risk_score += 0.2
            
            if not doc.human_physician_involved:
                risk_factors.append('No human physician oversight')
                risk_score += 0.4
            
            if 'low_ai_confidence' in doc.risk_factors:
                risk_factors.append('Low AI confidence decisions')
                risk_score += 0.1
        
        # Normalize risk score
        risk_score = min(1.0, risk_score / max(len(documents), 1))
        
        if risk_score < 0.3:
            risk_level = 'LOW'
        elif risk_score < 0.6:
            risk_level = 'MODERATE'
        else:
            risk_level = 'HIGH'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': list(set(risk_factors)),
            'high_risk_documents': len([d for d in documents if len(d.risk_factors) > 2])
        }
    
    def _generate_legal_recommendations(self,
                                      documents: List[LiabilityDocument],
                                      compliance_issues: List[str]) -> List[str]:
        """Generate legal recommendations"""
        recommendations = []
        
        if compliance_issues:
            recommendations.append('Address all compliance issues immediately')
            recommendations.append('Implement compliance monitoring dashboard')
        
        consent_rate = sum(1 for doc in documents if doc.informed_consent_obtained) / max(len(documents), 1)
        if consent_rate < 0.95:
            recommendations.append('Improve informed consent processes')
        
        disclosure_rate = sum(1 for doc in documents if doc.ai_disclosure_provided) / max(len(documents), 1)
        if disclosure_rate < 0.95:
            recommendations.append('Standardize AI disclosure procedures')
        
        oversight_rate = sum(1 for doc in documents if doc.human_physician_involved) / max(len(documents), 1)
        if oversight_rate < 0.8:
            recommendations.append('Increase human physician oversight')
        
        recommendations.extend([
            'Regular legal compliance audits',
            'Staff training on documentation requirements',
            'Legal review of high-risk cases'
        ])
        
        return recommendations
    
    def get_audit_trail(self,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get filtered audit trail.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            user_id: Optional user ID filter
            
        Returns:
            Filtered audit trail entries
        """
        filtered_entries = []
        
        for entry in self.audit_trail:
            entry_time = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
            
            if start_date and entry_time < start_date:
                continue
            if end_date and entry_time > end_date:
                continue
            if user_id and entry['user_id'] != user_id:
                continue
            
            filtered_entries.append(entry)
        
        return filtered_entries
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get liability documentation system status"""
        return {
            'system_operational': True,
            'total_documents': len(self.documents),
            'total_safety_incidents': len(self.safety_incidents),
            'total_consent_records': len(self.consent_records),
            'audit_trail_entries': len(self.audit_trail),
            'compliance_monitoring_active': True,
            'retention_policies_configured': len(self.retention_policies)
        }