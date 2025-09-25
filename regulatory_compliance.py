#!/usr/bin/env python3
"""
Regulatory Compliance Module for Clinical Decision Support System

Provides comprehensive compliance frameworks for medical AI systems:
- HIPAA compliance and audit logging
- FDA validation pathway support
- Clinical trial data management
- Regulatory reporting dashboards
- Quality assurance workflows
- Risk management protocols

This module ensures adherence to healthcare regulatory requirements.
"""

import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import uuid
from abc import ABC, abstractmethod

# Import existing security components
from secure_medical_processor import SecureMedicalDataProcessor

logger = logging.getLogger("RegulatoryCompliance")


class ComplianceFramework(Enum):
    """Supported regulatory compliance frameworks"""
    HIPAA = "HIPAA"
    FDA_510K = "FDA_510K"
    FDA_DE_NOVO = "FDA_DE_NOVO"
    GDPR = "GDPR"
    ISO_13485 = "ISO_13485"
    ISO_14155 = "ISO_14155"  # Clinical investigation of medical devices
    ICH_GCP = "ICH_GCP"     # Good Clinical Practice


class AuditEventType(Enum):
    """Types of audit events for compliance tracking"""
    DATA_ACCESS = "data_access"
    MODEL_PREDICTION = "model_prediction"
    DATA_EXPORT = "data_export"
    SYSTEM_LOGIN = "system_login"
    CONFIGURATION_CHANGE = "configuration_change"
    MODEL_TRAINING = "model_training"
    VALIDATION_TEST = "validation_test"
    ADVERSE_EVENT = "adverse_event"
    SECURITY_INCIDENT = "security_incident"


@dataclass
class AuditEvent:
    """Audit event record structure"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    patient_id: Optional[str]
    resource_accessed: str
    action_performed: str
    outcome: str  # 'SUCCESS', 'FAILURE', 'WARNING'
    ip_address: Optional[str]
    user_agent: Optional[str]
    additional_data: Dict[str, Any]
    risk_level: str = 'LOW'  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'


@dataclass
class ValidationRecord:
    """Model validation record for FDA compliance"""
    validation_id: str
    model_version: str
    validation_type: str  # 'analytical', 'clinical', 'usability'
    test_dataset: str
    performance_metrics: Dict[str, float]
    validation_date: datetime
    validator: str
    clinical_endpoints: List[str]
    success_criteria: Dict[str, Any]
    results: Dict[str, Any]
    status: str  # 'PENDING', 'PASSED', 'FAILED', 'CONDITIONAL'
    regulatory_notes: str


@dataclass
class AdverseEvent:
    """Adverse event record for safety monitoring"""
    event_id: str
    patient_id: str
    event_date: datetime
    description: str
    severity: str  # 'MILD', 'MODERATE', 'SEVERE'
    causality: str  # 'UNRELATED', 'POSSIBLE', 'PROBABLE', 'DEFINITE'
    outcome: str   # 'RECOVERED', 'ONGOING', 'UNKNOWN'
    ai_system_involved: bool
    model_version: Optional[str]
    prediction_confidence: Optional[float]
    clinical_context: Dict[str, Any]
    reporter: str
    follow_up_required: bool


class HIPAAComplianceManager:
    """
    HIPAA compliance management for protected health information.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audit_db_path = config.get('audit_db_path', 'compliance_audit.db')
        self._init_audit_database()
        
    def _init_audit_database(self):
        """Initialize audit database schema"""
        
        conn = sqlite3.connect(self.audit_db_path)
        cursor = conn.cursor()
        
        # Audit events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                patient_id TEXT,
                resource_accessed TEXT NOT NULL,
                action_performed TEXT NOT NULL,
                outcome TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                additional_data TEXT,
                risk_level TEXT DEFAULT 'LOW'
            )
        ''')
        
        # PHI access log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS phi_access_log (
                access_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                data_elements TEXT NOT NULL,
                purpose TEXT NOT NULL,
                authorization_basis TEXT NOT NULL,
                minimum_necessary BOOLEAN DEFAULT TRUE,
                retention_period TEXT
            )
        ''')
        
        # Breach incidents
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS breach_incidents (
                incident_id TEXT PRIMARY KEY,
                discovery_date TEXT NOT NULL,
                incident_date TEXT NOT NULL,
                affected_individuals INTEGER NOT NULL,
                types_of_phi TEXT NOT NULL,
                description TEXT NOT NULL,
                containment_actions TEXT,
                notification_status TEXT DEFAULT 'PENDING',
                regulatory_reporting_required BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_audit_event(self, event: AuditEvent):
        """Log audit event for HIPAA compliance"""
        
        conn = sqlite3.connect(self.audit_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_events (
                event_id, event_type, timestamp, user_id, patient_id,
                resource_accessed, action_performed, outcome, ip_address,
                user_agent, additional_data, risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            event.event_type.value,
            event.timestamp.isoformat(),
            event.user_id,
            event.patient_id,
            event.resource_accessed,
            event.action_performed,
            event.outcome,
            event.ip_address,
            event.user_agent,
            json.dumps(event.additional_data),
            event.risk_level
        ))
        
        conn.commit()
        conn.close()
        
        # Check for high-risk events
        if event.risk_level in ['HIGH', 'CRITICAL']:
            self._trigger_security_alert(event)
    
    def log_phi_access(self, user_id: str, patient_id: str, 
                      data_elements: List[str], purpose: str,
                      authorization_basis: str = 'treatment'):
        """Log PHI access for minimum necessary compliance"""
        
        access_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.audit_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO phi_access_log (
                access_id, timestamp, user_id, patient_id, data_elements,
                purpose, authorization_basis, minimum_necessary, retention_period
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            access_id,
            datetime.now().isoformat(),
            user_id,
            patient_id,
            json.dumps(data_elements),
            purpose,
            authorization_basis,
            True,  # Assume minimum necessary by default
            '6 years'  # Standard HIPAA retention period
        ))
        
        conn.commit()
        conn.close()
        
        return access_id
    
    def check_data_minimization(self, requested_data: Dict[str, Any], 
                               purpose: str) -> Dict[str, Any]:
        """Check and enforce data minimization principles"""
        
        # Define minimum necessary data sets for different purposes
        minimal_datasets = {
            'risk_assessment': [
                'patient_id', 'Age', 'M/F', 'MMSE', 'CDR', 'EDUC'
            ],
            'treatment_planning': [
                'patient_id', 'Age', 'M/F', 'MMSE', 'CDR', 'current_medications'
            ],
            'research': [
                'Age', 'M/F', 'MMSE', 'CDR', 'EDUC'  # No patient_id for research
            ]
        }
        
        required_fields = minimal_datasets.get(purpose, list(requested_data.keys()))
        
        # Filter to minimum necessary
        minimized_data = {
            key: value for key, value in requested_data.items()
            if key in required_fields
        }
        
        # Log data minimization decision
        audit_event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.DATA_ACCESS,
            timestamp=datetime.now(),
            user_id=None,
            patient_id=requested_data.get('patient_id'),
            resource_accessed='patient_data',
            action_performed='data_minimization_check',
            outcome='SUCCESS',
            ip_address=None,
            user_agent=None,
            additional_data={
                'original_fields': list(requested_data.keys()),
                'minimized_fields': list(minimized_data.keys()),
                'purpose': purpose
            }
        )
        
        self.log_audit_event(audit_event)
        
        return minimized_data
    
    def generate_compliance_report(self, start_date: datetime, 
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate HIPAA compliance report"""
        
        conn = sqlite3.connect(self.audit_db_path)
        cursor = conn.cursor()
        
        # Get audit events in date range
        cursor.execute('''
            SELECT event_type, outcome, risk_level, COUNT(*) as count
            FROM audit_events
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY event_type, outcome, risk_level
        ''', (start_date.isoformat(), end_date.isoformat()))
        
        audit_summary = cursor.fetchall()
        
        # Get PHI access statistics
        cursor.execute('''
            SELECT authorization_basis, COUNT(*) as count
            FROM phi_access_log
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY authorization_basis
        ''', (start_date.isoformat(), end_date.isoformat()))
        
        phi_access_summary = cursor.fetchall()
        
        # Get breach incidents
        cursor.execute('''
            SELECT * FROM breach_incidents
            WHERE incident_date BETWEEN ? AND ?
        ''', (start_date.isoformat(), end_date.isoformat()))
        
        breach_incidents = cursor.fetchall()
        
        conn.close()
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'audit_summary': [
                {
                    'event_type': row[0],
                    'outcome': row[1],
                    'risk_level': row[2],
                    'count': row[3]
                }
                for row in audit_summary
            ],
            'phi_access_summary': [
                {
                    'authorization_basis': row[0],
                    'count': row[1]
                }
                for row in phi_access_summary
            ],
            'breach_incidents': len(breach_incidents),
            'compliance_status': 'COMPLIANT' if len(breach_incidents) == 0 else 'UNDER_REVIEW'
        }
    
    def _trigger_security_alert(self, event: AuditEvent):
        """Trigger security alert for high-risk events"""
        
        alert_data = {
            'alert_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'event_id': event.event_id,
            'risk_level': event.risk_level,
            'description': f"High-risk event detected: {event.action_performed}",
            'requires_investigation': True
        }
        
        logger.warning(f"Security alert triggered: {alert_data}")
        
        # In production, this would send alerts to security team


class FDAValidationManager:
    """
    FDA validation pathway management for medical AI systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_db_path = config.get('validation_db_path', 'fda_validation.db')
        self._init_validation_database()
        
    def _init_validation_database(self):
        """Initialize FDA validation database schema"""
        
        conn = sqlite3.connect(self.validation_db_path)
        cursor = conn.cursor()
        
        # Validation records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_records (
                validation_id TEXT PRIMARY KEY,
                model_version TEXT NOT NULL,
                validation_type TEXT NOT NULL,
                test_dataset TEXT NOT NULL,
                performance_metrics TEXT NOT NULL,
                validation_date TEXT NOT NULL,
                validator TEXT NOT NULL,
                clinical_endpoints TEXT NOT NULL,
                success_criteria TEXT NOT NULL,
                results TEXT NOT NULL,
                status TEXT NOT NULL,
                regulatory_notes TEXT
            )
        ''')
        
        # Adverse events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adverse_events (
                event_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                event_date TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT NOT NULL,
                causality TEXT NOT NULL,
                outcome TEXT NOT NULL,
                ai_system_involved BOOLEAN NOT NULL,
                model_version TEXT,
                prediction_confidence REAL,
                clinical_context TEXT NOT NULL,
                reporter TEXT NOT NULL,
                follow_up_required BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Clinical performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clinical_performance (
                record_id TEXT PRIMARY KEY,
                model_version TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                patient_count INTEGER NOT NULL,
                true_positives INTEGER NOT NULL,
                false_positives INTEGER NOT NULL,
                true_negatives INTEGER NOT NULL,
                false_negatives INTEGER NOT NULL,
                sensitivity REAL NOT NULL,
                specificity REAL NOT NULL,
                ppv REAL NOT NULL,
                npv REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_validation_protocol(self, model_version: str, 
                                 submission_type: str = '510k') -> Dict[str, Any]:
        """Create FDA validation protocol"""
        
        protocol_templates = {
            '510k': {
                'analytical_validation': [
                    'Algorithm performance testing',
                    'Robustness testing',
                    'Cybersecurity analysis',
                    'Software verification'
                ],
                'clinical_validation': [
                    'Clinical accuracy study',
                    'Clinical utility demonstration',
                    'User interface validation'
                ],
                'required_endpoints': [
                    'Sensitivity ≥ 90%',
                    'Specificity ≥ 85%',
                    'PPV ≥ 80%',
                    'NPV ≥ 90%'
                ]
            },
            'de_novo': {
                'analytical_validation': [
                    'Comprehensive algorithm validation',
                    'Multi-site testing',
                    'Edge case analysis',
                    'Bias evaluation'
                ],
                'clinical_validation': [
                    'Pivotal clinical study',
                    'Real-world evidence',
                    'Long-term safety monitoring'
                ],
                'required_endpoints': [
                    'Primary effectiveness endpoint',
                    'Safety endpoints',
                    'Quality of life measures'
                ]
            }
        }
        
        template = protocol_templates.get(submission_type.lower(), protocol_templates['510k'])
        
        protocol = {
            'protocol_id': str(uuid.uuid4()),
            'model_version': model_version,
            'submission_type': submission_type,
            'created_date': datetime.now().isoformat(),
            'validation_phases': template,
            'status': 'DRAFT',
            'regulatory_pathway': self._determine_regulatory_pathway(model_version)
        }
        
        return protocol
    
    def record_validation_result(self, validation_record: ValidationRecord):
        """Record validation test results"""
        
        conn = sqlite3.connect(self.validation_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO validation_records (
                validation_id, model_version, validation_type, test_dataset,
                performance_metrics, validation_date, validator, clinical_endpoints,
                success_criteria, results, status, regulatory_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            validation_record.validation_id,
            validation_record.model_version,
            validation_record.validation_type,
            validation_record.test_dataset,
            json.dumps(validation_record.performance_metrics),
            validation_record.validation_date.isoformat(),
            validation_record.validator,
            json.dumps(validation_record.clinical_endpoints),
            json.dumps(validation_record.success_criteria),
            json.dumps(validation_record.results),
            validation_record.status,
            validation_record.regulatory_notes
        ))
        
        conn.commit()
        conn.close()
        
        # Check if validation meets FDA requirements
        self._evaluate_validation_compliance(validation_record)
    
    def record_adverse_event(self, adverse_event: AdverseEvent):
        """Record adverse event for FDA safety monitoring"""
        
        conn = sqlite3.connect(self.validation_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO adverse_events (
                event_id, patient_id, event_date, description, severity,
                causality, outcome, ai_system_involved, model_version,
                prediction_confidence, clinical_context, reporter, follow_up_required
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            adverse_event.event_id,
            adverse_event.patient_id,
            adverse_event.event_date.isoformat(),
            adverse_event.description,
            adverse_event.severity,
            adverse_event.causality,
            adverse_event.outcome,
            adverse_event.ai_system_involved,
            adverse_event.model_version,
            adverse_event.prediction_confidence,
            json.dumps(adverse_event.clinical_context),
            adverse_event.reporter,
            adverse_event.follow_up_required
        ))
        
        conn.commit()
        conn.close()
        
        # Check if reportable to FDA
        self._assess_fda_reporting_requirement(adverse_event)
    
    def track_clinical_performance(self, model_version: str, 
                                 performance_data: Dict[str, Any]):
        """Track ongoing clinical performance for post-market surveillance"""
        
        # Calculate performance metrics
        tp = performance_data['true_positives']
        fp = performance_data['false_positives']
        tn = performance_data['true_negatives']
        fn = performance_data['false_negatives']
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        conn = sqlite3.connect(self.validation_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO clinical_performance (
                record_id, model_version, timestamp, patient_count,
                true_positives, false_positives, true_negatives, false_negatives,
                sensitivity, specificity, ppv, npv
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            model_version,
            datetime.now().isoformat(),
            tp + fp + tn + fn,
            tp, fp, tn, fn,
            sensitivity, specificity, ppv, npv
        ))
        
        conn.commit()
        conn.close()
        
        # Check for performance degradation
        self._monitor_performance_drift(model_version, sensitivity, specificity)
    
    def generate_fda_submission_package(self, model_version: str) -> Dict[str, Any]:
        """Generate FDA submission package documentation"""
        
        conn = sqlite3.connect(self.validation_db_path)
        cursor = conn.cursor()
        
        # Get all validation records for this model
        cursor.execute('''
            SELECT * FROM validation_records WHERE model_version = ?
        ''', (model_version,))
        
        validation_records = cursor.fetchall()
        
        # Get clinical performance data
        cursor.execute('''
            SELECT * FROM clinical_performance WHERE model_version = ?
            ORDER BY timestamp DESC LIMIT 10
        ''', (model_version,))
        
        performance_records = cursor.fetchall()
        
        # Get adverse events
        cursor.execute('''
            SELECT * FROM adverse_events WHERE model_version = ?
        ''', (model_version,))
        
        adverse_events = cursor.fetchall()
        
        conn.close()
        
        submission_package = {
            'model_version': model_version,
            'submission_date': datetime.now().isoformat(),
            'regulatory_pathway': self._determine_regulatory_pathway(model_version),
            'validation_summary': {
                'total_validations': len(validation_records),
                'passed_validations': len([r for r in validation_records if r[10] == 'PASSED']),
                'validation_types': list(set([r[2] for r in validation_records]))
            },
            'clinical_performance': {
                'total_patients': sum([r[3] for r in performance_records]),
                'average_sensitivity': sum([r[8] for r in performance_records]) / len(performance_records) if performance_records else 0,
                'average_specificity': sum([r[9] for r in performance_records]) / len(performance_records) if performance_records else 0
            },
            'safety_profile': {
                'total_adverse_events': len(adverse_events),
                'serious_adverse_events': len([ae for ae in adverse_events if ae[6] == 'SERIOUS']),
                'resolved_events': len([ae for ae in adverse_events if ae[8] is not None])
            },
            'quality_metrics': self._calculate_quality_metrics(validation_records),
            'risk_assessment': self._generate_risk_assessment(validation_records, adverse_events),
            'submission_readiness_score': self._calculate_submission_readiness_score(validation_records, performance_records, adverse_events)
        }
        
        logger.info(f"Generated FDA submission package for model {model_version}")
        return submission_package
    
    def _calculate_quality_metrics(self, validation_records: List) -> Dict[str, Any]:
        """Calculate quality metrics for FDA submission."""
        if not validation_records:
            return {
                'validation_completion_rate': 0,
                'average_accuracy': 0,
                'consistency_score': 0
            }
        
        passed_validations = sum(1 for r in validation_records if r[10] == 'PASSED')
        total_validations = len(validation_records)
        
        # Calculate average metrics if available
        accuracy_values = [r[8] for r in validation_records if r[8] is not None]
        avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0
        
        return {
            'validation_completion_rate': (passed_validations / total_validations * 100) if total_validations > 0 else 0,
            'average_accuracy': avg_accuracy * 100,  # Convert to percentage
            'consistency_score': min(95.0, avg_accuracy * 105),  # Derived consistency metric
            'total_test_cases': total_validations,
            'passed_test_cases': passed_validations,
            'failed_test_cases': total_validations - passed_validations
        }
    
    def _generate_risk_assessment(self, validation_records: List, adverse_events: List) -> Dict[str, Any]:
        """Generate comprehensive risk assessment for FDA submission."""
        risk_factors = []
        risk_level = "LOW"
        
        # Assess validation failure rate
        if validation_records:
            failure_rate = sum(1 for r in validation_records if r[10] == 'FAILED') / len(validation_records)
            if failure_rate > 0.1:  # More than 10% failure rate
                risk_factors.append("High validation failure rate")
                risk_level = "MEDIUM"
            if failure_rate > 0.2:  # More than 20% failure rate
                risk_level = "HIGH"
        
        # Assess adverse events
        serious_events = len([ae for ae in adverse_events if ae[6] == 'SERIOUS'])
        if serious_events > 0:
            risk_factors.append(f"{serious_events} serious adverse events reported")
            risk_level = "HIGH" if serious_events > 5 else "MEDIUM"
        
        # Calculate overall risk score
        base_score = 100
        for _ in risk_factors:
            base_score -= 15
        
        risk_score = max(0, min(100, base_score))
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'mitigation_strategies': self._get_mitigation_strategies(risk_factors),
            'regulatory_concerns': self._identify_regulatory_concerns(validation_records, adverse_events)
        }
    
    def _get_mitigation_strategies(self, risk_factors: List[str]) -> List[str]:
        """Get mitigation strategies for identified risk factors."""
        strategies = []
        
        if any('failure rate' in factor for factor in risk_factors):
            strategies.extend([
                'Implement additional validation testing',
                'Enhance model training with diverse datasets',
                'Establish stricter quality control processes'
            ])
        
        if any('adverse events' in factor for factor in risk_factors):
            strategies.extend([
                'Implement enhanced safety monitoring',
                'Establish rapid response protocols',
                'Increase physician oversight requirements'
            ])
        
        return strategies
    
    def _identify_regulatory_concerns(self, validation_records: List, adverse_events: List) -> List[str]:
        """Identify potential regulatory concerns."""
        concerns = []
        
        if not validation_records:
            concerns.append("Insufficient validation data for regulatory review")
        
        unresolved_events = [ae for ae in adverse_events if ae[8] is None]
        if unresolved_events:
            concerns.append(f"{len(unresolved_events)} unresolved adverse events")
        
        return concerns
    
    def _calculate_submission_readiness_score(self, validation_records: List, performance_records: List, adverse_events: List) -> Dict[str, Any]:
        """Calculate FDA submission readiness score."""
        score_components = {
            'validation_completeness': 0,
            'performance_evidence': 0,
            'safety_documentation': 0,
            'quality_assurance': 0
        }
        
        # Validation completeness (0-25 points)
        if validation_records:
            passed_rate = sum(1 for r in validation_records if r[10] == 'PASSED') / len(validation_records)
            score_components['validation_completeness'] = min(25, passed_rate * 30)
        
        # Performance evidence (0-25 points)
        if performance_records:
            score_components['performance_evidence'] = min(25, len(performance_records) * 2.5)
        
        # Safety documentation (0-25 points)
        if len(adverse_events) == 0:
            score_components['safety_documentation'] = 25
        else:
            resolved_rate = sum(1 for ae in adverse_events if ae[8] is not None) / len(adverse_events)
            score_components['safety_documentation'] = resolved_rate * 25
        
        # Quality assurance (0-25 points)
        if validation_records:
            score_components['quality_assurance'] = 20  # Base score for having QA processes
        
        total_score = sum(score_components.values())
        
        # Determine readiness status
        if total_score >= 80:
            readiness_status = "READY_FOR_SUBMISSION"
        elif total_score >= 60:
            readiness_status = "NEARLY_READY"
        elif total_score >= 40:
            readiness_status = "NEEDS_IMPROVEMENT"
        else:
            readiness_status = "NOT_READY"
        
        return {
            'total_score': round(total_score, 1),
            'max_score': 100,
            'score_components': score_components,
            'readiness_status': readiness_status,
            'recommendations': self._get_readiness_recommendations(score_components, total_score)
        }
    
    def _get_readiness_recommendations(self, score_components: Dict[str, float], total_score: float) -> List[str]:
        """Get recommendations for improving FDA submission readiness."""
        recommendations = []
        
        if score_components['validation_completeness'] < 20:
            recommendations.append("Increase validation testing coverage and pass rate")
        
        if score_components['performance_evidence'] < 20:
            recommendations.append("Collect more clinical performance data")
        
        if score_components['safety_documentation'] < 20:
            recommendations.append("Address outstanding adverse events and improve safety documentation")
        
        if score_components['quality_assurance'] < 15:
            recommendations.append("Implement comprehensive quality assurance processes")
        
        if total_score < 80:
            recommendations.append("Consider pre-submission meeting with FDA to discuss requirements")
        
        return recommendations
    
    def _determine_regulatory_pathway(self, model_version: str) -> str:
        """Determine appropriate FDA regulatory pathway"""
        
        # This would be based on the specific model and intended use
        # For now, assume 510(k) pathway for most medical AI
        return "510(k)"
    
    def _evaluate_validation_compliance(self, validation_record: ValidationRecord):
        """Evaluate if validation meets FDA requirements"""
        
        metrics = validation_record.performance_metrics
        
        # Check minimum performance thresholds
        min_sensitivity = 0.90
        min_specificity = 0.85
        
        if (metrics.get('sensitivity', 0) >= min_sensitivity and 
            metrics.get('specificity', 0) >= min_specificity):
            logger.info(f"Validation {validation_record.validation_id} meets FDA thresholds")
        else:
            logger.warning(f"Validation {validation_record.validation_id} below FDA thresholds")
    
    def _assess_fda_reporting_requirement(self, adverse_event: AdverseEvent):
        """Assess if adverse event requires FDA reporting"""
        
        # FDA requires reporting of serious adverse events within 24-48 hours
        reportable = (
            adverse_event.severity == 'SEVERE' and
            adverse_event.ai_system_involved and
            adverse_event.causality in ['PROBABLE', 'DEFINITE']
        )
        
        if reportable:
            logger.critical(f"Adverse event {adverse_event.event_id} requires FDA reporting")
            # In production, trigger automatic reporting workflow
    
    def _monitor_performance_drift(self, model_version: str, 
                                 current_sensitivity: float, current_specificity: float):
        """Monitor for clinically significant performance drift"""
        
        # Get historical performance baselines
        conn = sqlite3.connect(self.validation_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT AVG(sensitivity), AVG(specificity) FROM clinical_performance 
            WHERE model_version = ? AND timestamp < ?
        ''', (model_version, (datetime.now() - timedelta(days=30)).isoformat()))
        
        baseline = cursor.fetchone()
        conn.close()
        
        if baseline and baseline[0] and baseline[1]:
            sensitivity_drift = abs(current_sensitivity - baseline[0])
            specificity_drift = abs(current_specificity - baseline[1])
            
            # Alert if drift > 5%
            if sensitivity_drift > 0.05 or specificity_drift > 0.05:
                logger.warning(f"Performance drift detected for {model_version}")
    
    def _assess_submission_readiness(self, validation_records: List, 
                                   adverse_events: List) -> Dict[str, Any]:
        """Assess readiness for FDA submission"""
        
        readiness_criteria = {
            'analytical_validation_complete': False,
            'clinical_validation_complete': False,
            'safety_profile_acceptable': True,
            'documentation_complete': False
        }
        
        # Check validation completeness
        validation_types = [r[2] for r in validation_records]
        if 'analytical' in validation_types:
            readiness_criteria['analytical_validation_complete'] = True
        if 'clinical' in validation_types:
            readiness_criteria['clinical_validation_complete'] = True
        
        # Check safety profile
        severe_events = [e for e in adverse_events if e[4] == 'SEVERE']
        if len(severe_events) > 5:  # Arbitrary threshold
            readiness_criteria['safety_profile_acceptable'] = False
        
        # Overall readiness
        overall_ready = all([
            readiness_criteria['analytical_validation_complete'],
            readiness_criteria['clinical_validation_complete'],
            readiness_criteria['safety_profile_acceptable']
        ])
        
        readiness_criteria['overall_ready'] = overall_ready
        
        return readiness_criteria


class ComplianceDashboard:
    """
    Regulatory compliance dashboard for monitoring and reporting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hipaa_manager = HIPAAComplianceManager(config)
        self.fda_manager = FDAValidationManager(config)
        
    def generate_compliance_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive compliance dashboard"""
        
        # Get current date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # HIPAA compliance status
        hipaa_report = self.hipaa_manager.generate_compliance_report(start_date, end_date)
        
        # FDA validation status
        fda_package = self.fda_manager.generate_fda_submission_package('v1.0')
        
        dashboard = {
            'generated_at': datetime.now().isoformat(),
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'hipaa_compliance': {
                'status': hipaa_report['compliance_status'],
                'breach_incidents': hipaa_report['breach_incidents'],
                'audit_events': len(hipaa_report['audit_summary']),
                'phi_accesses': sum([item['count'] for item in hipaa_report['phi_access_summary']])
            },
            'fda_validation': {
                'model_version': fda_package['model_version'],
                'submission_ready': fda_package['readiness_assessment']['overall_ready'],
                'validation_progress': fda_package['validation_summary'],
                'safety_profile': fda_package['safety_profile']
            },
            'overall_compliance_score': self._calculate_compliance_score(hipaa_report, fda_package),
            'action_items': self._generate_action_items(hipaa_report, fda_package)
        }
        
        return dashboard
    
    def _calculate_compliance_score(self, hipaa_report: Dict[str, Any], 
                                  fda_package: Dict[str, Any]) -> float:
        """Calculate overall compliance score (0-100)"""
        
        score = 100.0
        
        # HIPAA deductions
        if hipaa_report['compliance_status'] != 'COMPLIANT':
            score -= 30
        if hipaa_report['breach_incidents'] > 0:
            score -= 20
        
        # FDA deductions
        if not fda_package['readiness_assessment']['overall_ready']:
            score -= 25
        if fda_package['safety_profile']['severe_events'] > 0:
            score -= 15
        
        return max(0.0, score)
    
    def _generate_action_items(self, hipaa_report: Dict[str, Any], 
                             fda_package: Dict[str, Any]) -> List[str]:
        """Generate compliance action items"""
        
        action_items = []
        
        # HIPAA action items
        if hipaa_report['compliance_status'] != 'COMPLIANT':
            action_items.append("Address HIPAA compliance issues")
        if hipaa_report['breach_incidents'] > 0:
            action_items.append("Complete breach incident investigation")
        
        # FDA action items
        if not fda_package['readiness_assessment']['analytical_validation_complete']:
            action_items.append("Complete analytical validation testing")
        if not fda_package['readiness_assessment']['clinical_validation_complete']:
            action_items.append("Complete clinical validation studies")
        if fda_package['safety_profile']['severe_events'] > 0:
            action_items.append("Review and address severe adverse events")
        
        return action_items


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = {
        'audit_db_path': '/tmp/compliance_audit.db',
        'validation_db_path': '/tmp/fda_validation.db'
    }
    
    # Create compliance managers
    hipaa_manager = HIPAAComplianceManager(config)
    fda_manager = FDAValidationManager(config)
    
    # Test HIPAA audit logging
    audit_event = AuditEvent(
        event_id=str(uuid.uuid4()),
        event_type=AuditEventType.MODEL_PREDICTION,
        timestamp=datetime.now(),
        user_id='clinician_001',
        patient_id='PATIENT_001',
        resource_accessed='alzheimer_model',
        action_performed='risk_prediction',
        outcome='SUCCESS',
        ip_address='192.168.1.100',
        user_agent='Clinical Dashboard v1.0',
        additional_data={'model_version': 'v1.0', 'confidence': 0.85}
    )
    
    hipaa_manager.log_audit_event(audit_event)
    print("=== Regulatory Compliance Test ===")
    print("HIPAA audit event logged successfully")
    
    # Test FDA validation recording
    validation_record = ValidationRecord(
        validation_id=str(uuid.uuid4()),
        model_version='v1.0',
        validation_type='analytical',
        test_dataset='validation_set_001',
        performance_metrics={'sensitivity': 0.92, 'specificity': 0.87, 'auc': 0.94},
        validation_date=datetime.now(),
        validator='Dr. Smith',
        clinical_endpoints=['Alzheimer detection accuracy'],
        success_criteria={'sensitivity': 0.90, 'specificity': 0.85},
        results={'passed': True, 'notes': 'Exceeds minimum requirements'},
        status='PASSED',
        regulatory_notes='Meets FDA analytical validation requirements'
    )
    
    fda_manager.record_validation_result(validation_record)
    print("FDA validation record created successfully")
    
    # Generate compliance dashboard
    dashboard = ComplianceDashboard(config)
    compliance_report = dashboard.generate_compliance_dashboard()
    
    print(f"\nCompliance Dashboard Generated:")
    print(f"Overall Compliance Score: {compliance_report['overall_compliance_score']}")
    print(f"HIPAA Status: {compliance_report['hipaa_compliance']['status']}")
    print(f"FDA Submission Ready: {compliance_report['fda_validation']['submission_ready']}")
    print(f"Action Items: {len(compliance_report['action_items'])}")