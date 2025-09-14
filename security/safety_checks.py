"""
Concrete safety check implementations for DuetMind Adaptive.

Provides domain-specific safety checks:
- SystemSafetyCheck: Resource usage, performance, technical issues
- DataSafetyCheck: Data quality, schema validation, completeness
- ModelSafetyCheck: Confidence drift, calibration, accuracy
- InteractionSafetyCheck: Conversation rules, user interaction patterns
- ClinicalSafetyCheck: Guideline adherence, medical accuracy
"""

import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import numpy as np
import re

from .safety_monitor import ISafetyCheck, SafetyDomain, SafetyFinding

logger = logging.getLogger('duetmind.safety.checks')


class SystemSafetyCheck(ISafetyCheck):
    """Safety check for system resources and performance."""
    
    @property
    def domain(self) -> SafetyDomain:
        return SafetyDomain.SYSTEM
    
    @property
    def name(self) -> str:
        return "system_resource_monitor"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.cpu_warning_threshold = self.config.get('cpu_warning_threshold', 80.0)
        self.cpu_critical_threshold = self.config.get('cpu_critical_threshold', 90.0)
        self.memory_warning_threshold = self.config.get('memory_warning_threshold', 80.0)
        self.memory_critical_threshold = self.config.get('memory_critical_threshold', 90.0)
        self.disk_warning_threshold = self.config.get('disk_warning_threshold', 85.0)
        self.disk_critical_threshold = self.config.get('disk_critical_threshold', 95.0)
        self.response_time_warning_ms = self.config.get('response_time_warning_ms', 1000.0)
        self.response_time_critical_ms = self.config.get('response_time_critical_ms', 3000.0)
    
    def run(self, context: Dict[str, Any], correlation_id: Optional[str] = None) -> List[SafetyFinding]:
        findings = []
        
        try:
            # CPU usage check
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent >= self.cpu_critical_threshold:
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity='critical',
                    message=f'High CPU usage: {cpu_percent:.1f}%',
                    value=cpu_percent,
                    threshold=self.cpu_critical_threshold,
                    correlation_id=correlation_id,
                    metadata={'metric': 'cpu_usage', 'unit': 'percent'}
                ))
            elif cpu_percent >= self.cpu_warning_threshold:
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity='warning',
                    message=f'Elevated CPU usage: {cpu_percent:.1f}%',
                    value=cpu_percent,
                    threshold=self.cpu_warning_threshold,
                    correlation_id=correlation_id,
                    metadata={'metric': 'cpu_usage', 'unit': 'percent'}
                ))
            
            # Memory usage check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            if memory_percent >= self.memory_critical_threshold:
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity='critical',
                    message=f'High memory usage: {memory_percent:.1f}%',
                    value=memory_percent,
                    threshold=self.memory_critical_threshold,
                    correlation_id=correlation_id,
                    metadata={'metric': 'memory_usage', 'unit': 'percent', 'available_mb': memory.available // (1024*1024)}
                ))
            elif memory_percent >= self.memory_warning_threshold:
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity='warning',
                    message=f'Elevated memory usage: {memory_percent:.1f}%',
                    value=memory_percent,
                    threshold=self.memory_warning_threshold,
                    correlation_id=correlation_id,
                    metadata={'metric': 'memory_usage', 'unit': 'percent', 'available_mb': memory.available // (1024*1024)}
                ))
            
            # Disk usage check
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent >= self.disk_critical_threshold:
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity='critical',
                    message=f'High disk usage: {disk_percent:.1f}%',
                    value=disk_percent,
                    threshold=self.disk_critical_threshold,
                    correlation_id=correlation_id,
                    metadata={'metric': 'disk_usage', 'unit': 'percent', 'free_gb': disk.free // (1024*1024*1024)}
                ))
            elif disk_percent >= self.disk_warning_threshold:
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity='warning',
                    message=f'Elevated disk usage: {disk_percent:.1f}%',
                    value=disk_percent,
                    threshold=self.disk_warning_threshold,
                    correlation_id=correlation_id,
                    metadata={'metric': 'disk_usage', 'unit': 'percent', 'free_gb': disk.free // (1024*1024*1024)}
                ))
            
            # Response time check (if provided in context)
            response_time_ms = context.get('response_time_ms')
            if response_time_ms is not None:
                if response_time_ms >= self.response_time_critical_ms:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='critical',
                        message=f'High response time: {response_time_ms:.1f}ms',
                        value=response_time_ms,
                        threshold=self.response_time_critical_ms,
                        correlation_id=correlation_id,
                        metadata={'metric': 'response_time', 'unit': 'milliseconds'}
                    ))
                elif response_time_ms >= self.response_time_warning_ms:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='warning',
                        message=f'Elevated response time: {response_time_ms:.1f}ms',
                        value=response_time_ms,
                        threshold=self.response_time_warning_ms,
                        correlation_id=correlation_id,
                        metadata={'metric': 'response_time', 'unit': 'milliseconds'}
                    ))
        
        except Exception as e:
            logger.error(f"Error in system safety check: {e}")
            findings.append(SafetyFinding(
                domain=self.domain,
                check_name=self.name,
                severity='warning',
                message=f'System monitoring error: {str(e)}',
                correlation_id=correlation_id,
                metadata={'error': str(e)}
            ))
        
        return findings


class DataSafetyCheck(ISafetyCheck):
    """Safety check for data quality and integrity."""
    
    @property
    def domain(self) -> SafetyDomain:
        return SafetyDomain.DATA
    
    @property
    def name(self) -> str:
        return "data_quality_monitor"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.missing_data_warning_threshold = self.config.get('missing_data_warning_threshold', 0.1)  # 10%
        self.missing_data_critical_threshold = self.config.get('missing_data_critical_threshold', 0.25)  # 25%
        self.duplicate_warning_threshold = self.config.get('duplicate_warning_threshold', 0.05)  # 5%
        self.duplicate_critical_threshold = self.config.get('duplicate_critical_threshold', 0.15)  # 15%
    
    def run(self, context: Dict[str, Any], correlation_id: Optional[str] = None) -> List[SafetyFinding]:
        findings = []
        
        try:
            # Check for data quality metrics in context
            data_quality_score = context.get('data_quality_score')
            if data_quality_score is not None:
                if data_quality_score < 0.7:  # Below 70%
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='critical',
                        message=f'Low data quality score: {data_quality_score:.3f}',
                        value=data_quality_score,
                        threshold=0.7,
                        correlation_id=correlation_id,
                        metadata={'metric': 'data_quality_score'}
                    ))
                elif data_quality_score < 0.85:  # Below 85%
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='warning',
                        message=f'Moderate data quality score: {data_quality_score:.3f}',
                        value=data_quality_score,
                        threshold=0.85,
                        correlation_id=correlation_id,
                        metadata={'metric': 'data_quality_score'}
                    ))
            
            # Check for missing data ratio
            missing_data_ratio = context.get('missing_data_ratio')
            if missing_data_ratio is not None:
                if missing_data_ratio >= self.missing_data_critical_threshold:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='critical',
                        message=f'High missing data ratio: {missing_data_ratio:.3f}',
                        value=missing_data_ratio,
                        threshold=self.missing_data_critical_threshold,
                        correlation_id=correlation_id,
                        metadata={'metric': 'missing_data_ratio'}
                    ))
                elif missing_data_ratio >= self.missing_data_warning_threshold:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='warning',
                        message=f'Elevated missing data ratio: {missing_data_ratio:.3f}',
                        value=missing_data_ratio,
                        threshold=self.missing_data_warning_threshold,
                        correlation_id=correlation_id,
                        metadata={'metric': 'missing_data_ratio'}
                    ))
            
            # Check for duplicate data ratio
            duplicate_ratio = context.get('duplicate_ratio')
            if duplicate_ratio is not None:
                if duplicate_ratio >= self.duplicate_critical_threshold:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='critical',
                        message=f'High duplicate data ratio: {duplicate_ratio:.3f}',
                        value=duplicate_ratio,
                        threshold=self.duplicate_critical_threshold,
                        correlation_id=correlation_id,
                        metadata={'metric': 'duplicate_ratio'}
                    ))
                elif duplicate_ratio >= self.duplicate_warning_threshold:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='warning',
                        message=f'Elevated duplicate data ratio: {duplicate_ratio:.3f}',
                        value=duplicate_ratio,
                        threshold=self.duplicate_warning_threshold,
                        correlation_id=correlation_id,
                        metadata={'metric': 'duplicate_ratio'}
                    ))
            
            # Schema validation check
            schema_violations = context.get('schema_violations', [])
            if schema_violations:
                severity = 'critical' if len(schema_violations) > 5 else 'warning'
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity=severity,
                    message=f'Schema violations detected: {len(schema_violations)} issues',
                    value=len(schema_violations),
                    correlation_id=correlation_id,
                    metadata={'metric': 'schema_violations', 'violations': schema_violations[:5]}  # Include first 5
                ))
        
        except Exception as e:
            logger.error(f"Error in data safety check: {e}")
            findings.append(SafetyFinding(
                domain=self.domain,
                check_name=self.name,
                severity='warning',
                message=f'Data quality monitoring error: {str(e)}',
                correlation_id=correlation_id,
                metadata={'error': str(e)}
            ))
        
        return findings


class ModelSafetyCheck(ISafetyCheck):
    """Safety check for model performance and confidence."""
    
    @property
    def domain(self) -> SafetyDomain:
        return SafetyDomain.MODEL
    
    @property
    def name(self) -> str:
        return "model_performance_monitor"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.accuracy_warning_drop = self.config.get('accuracy_warning_drop', 0.05)  # 5% drop
        self.accuracy_critical_drop = self.config.get('accuracy_critical_drop', 0.10)  # 10% drop
        self.confidence_warning_threshold = self.config.get('confidence_warning_threshold', 0.6)
        self.confidence_critical_threshold = self.config.get('confidence_critical_threshold', 0.4)
        self.drift_warning_threshold = self.config.get('drift_warning_threshold', 0.1)
        self.drift_critical_threshold = self.config.get('drift_critical_threshold', 0.2)
    
    def run(self, context: Dict[str, Any], correlation_id: Optional[str] = None) -> List[SafetyFinding]:
        findings = []
        
        try:
            # Model accuracy check
            current_accuracy = context.get('accuracy')
            baseline_accuracy = context.get('baseline_accuracy', 0.85)  # Default baseline
            
            if current_accuracy is not None:
                accuracy_drop = baseline_accuracy - current_accuracy
                if accuracy_drop >= self.accuracy_critical_drop:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='critical',
                        message=f'Significant accuracy drop: {accuracy_drop:.3f} (current: {current_accuracy:.3f})',
                        value=accuracy_drop,
                        threshold=self.accuracy_critical_drop,
                        correlation_id=correlation_id,
                        metadata={'metric': 'accuracy_drop', 'current_accuracy': current_accuracy, 'baseline_accuracy': baseline_accuracy}
                    ))
                elif accuracy_drop >= self.accuracy_warning_drop:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='warning',
                        message=f'Moderate accuracy drop: {accuracy_drop:.3f} (current: {current_accuracy:.3f})',
                        value=accuracy_drop,
                        threshold=self.accuracy_warning_drop,
                        correlation_id=correlation_id,
                        metadata={'metric': 'accuracy_drop', 'current_accuracy': current_accuracy, 'baseline_accuracy': baseline_accuracy}
                    ))
            
            # Model confidence check
            avg_confidence = context.get('average_confidence')
            if avg_confidence is not None:
                if avg_confidence <= self.confidence_critical_threshold:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='critical',
                        message=f'Very low model confidence: {avg_confidence:.3f}',
                        value=avg_confidence,
                        threshold=self.confidence_critical_threshold,
                        correlation_id=correlation_id,
                        metadata={'metric': 'average_confidence'}
                    ))
                elif avg_confidence <= self.confidence_warning_threshold:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='warning',
                        message=f'Low model confidence: {avg_confidence:.3f}',
                        value=avg_confidence,
                        threshold=self.confidence_warning_threshold,
                        correlation_id=correlation_id,
                        metadata={'metric': 'average_confidence'}
                    ))
            
            # Drift detection check
            drift_score = context.get('drift_score')
            if drift_score is not None:
                if drift_score >= self.drift_critical_threshold:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='critical',
                        message=f'Significant model drift detected: {drift_score:.3f}',
                        value=drift_score,
                        threshold=self.drift_critical_threshold,
                        correlation_id=correlation_id,
                        metadata={'metric': 'drift_score', 'recommendation': 'model_retrain'}
                    ))
                elif drift_score >= self.drift_warning_threshold:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='warning',
                        message=f'Model drift detected: {drift_score:.3f}',
                        value=drift_score,
                        threshold=self.drift_warning_threshold,
                        correlation_id=correlation_id,
                        metadata={'metric': 'drift_score', 'recommendation': 'monitor_closely'}
                    ))
            
            # Calibration check
            calibration_error = context.get('calibration_error')
            if calibration_error is not None:
                if calibration_error >= 0.15:  # 15% calibration error
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='critical',
                        message=f'High model calibration error: {calibration_error:.3f}',
                        value=calibration_error,
                        threshold=0.15,
                        correlation_id=correlation_id,
                        metadata={'metric': 'calibration_error', 'recommendation': 'recalibrate_model'}
                    ))
                elif calibration_error >= 0.08:  # 8% calibration error
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='warning',
                        message=f'Moderate model calibration error: {calibration_error:.3f}',
                        value=calibration_error,
                        threshold=0.08,
                        correlation_id=correlation_id,
                        metadata={'metric': 'calibration_error', 'recommendation': 'monitor_calibration'}
                    ))
        
        except Exception as e:
            logger.error(f"Error in model safety check: {e}")
            findings.append(SafetyFinding(
                domain=self.domain,
                check_name=self.name,
                severity='warning',
                message=f'Model monitoring error: {str(e)}',
                correlation_id=correlation_id,
                metadata={'error': str(e)}
            ))
        
        return findings


class InteractionSafetyCheck(ISafetyCheck):
    """Safety check for user interaction patterns and conversation rules."""
    
    @property
    def domain(self) -> SafetyDomain:
        return SafetyDomain.INTERACTION
    
    @property
    def name(self) -> str:
        return "interaction_pattern_monitor"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_session_length_minutes = self.config.get('max_session_length_minutes', 60)
        self.rapid_request_threshold = self.config.get('rapid_request_threshold', 10)  # requests per minute
        self.inappropriate_content_patterns = self.config.get('inappropriate_content_patterns', [
            r'(?i)(kill|die|suicide|harm)',  # Self-harm language
            r'(?i)(hack|exploit|bypass)',    # Security bypass attempts
            r'(?i)(override|ignore\s+safety)',  # Safety override attempts
        ])
    
    def run(self, context: Dict[str, Any], correlation_id: Optional[str] = None) -> List[SafetyFinding]:
        findings = []
        
        try:
            # Session length check
            session_length_minutes = context.get('session_length_minutes')
            if session_length_minutes is not None and session_length_minutes > self.max_session_length_minutes:
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity='warning',
                    message=f'Long session detected: {session_length_minutes:.1f} minutes',
                    value=session_length_minutes,
                    threshold=self.max_session_length_minutes,
                    correlation_id=correlation_id,
                    metadata={'metric': 'session_length', 'recommendation': 'suggest_break'}
                ))
            
            # Rapid request pattern check
            request_rate = context.get('requests_per_minute')
            if request_rate is not None and request_rate > self.rapid_request_threshold:
                severity = 'critical' if request_rate > self.rapid_request_threshold * 2 else 'warning'
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity=severity,
                    message=f'Rapid request pattern: {request_rate:.1f} requests/minute',
                    value=request_rate,
                    threshold=self.rapid_request_threshold,
                    correlation_id=correlation_id,
                    metadata={'metric': 'request_rate', 'recommendation': 'rate_limit'}
                ))
            
            # Content safety check
            user_input = context.get('user_input', '')
            if user_input:
                for pattern in self.inappropriate_content_patterns:
                    if re.search(pattern, user_input):
                        findings.append(SafetyFinding(
                            domain=self.domain,
                            check_name=self.name,
                            severity='critical',
                            message=f'Inappropriate content detected in user input',
                            correlation_id=correlation_id,
                            metadata={'metric': 'content_safety', 'pattern_matched': pattern, 'recommendation': 'block_request'}
                        ))
                        break  # Only report first match
            
            # Conversation flow check
            conversation_coherence = context.get('conversation_coherence')
            if conversation_coherence is not None and conversation_coherence < 0.3:
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity='warning',
                    message=f'Low conversation coherence: {conversation_coherence:.3f}',
                    value=conversation_coherence,
                    threshold=0.3,
                    correlation_id=correlation_id,
                    metadata={'metric': 'conversation_coherence', 'recommendation': 'clarify_intent'}
                ))
            
            # Repeated request check
            repeated_requests = context.get('repeated_request_count')
            if repeated_requests is not None and repeated_requests > 3:
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity='warning',
                    message=f'Repeated similar requests: {repeated_requests} times',
                    value=repeated_requests,
                    threshold=3,
                    correlation_id=correlation_id,
                    metadata={'metric': 'repeated_requests', 'recommendation': 'provide_alternative'}
                ))
        
        except Exception as e:
            logger.error(f"Error in interaction safety check: {e}")
            findings.append(SafetyFinding(
                domain=self.domain,
                check_name=self.name,
                severity='warning',
                message=f'Interaction monitoring error: {str(e)}',
                correlation_id=correlation_id,
                metadata={'error': str(e)}
            ))
        
        return findings


class ClinicalSafetyCheck(ISafetyCheck):
    """Safety check for clinical guideline adherence and medical accuracy."""
    
    @property
    def domain(self) -> SafetyDomain:
        return SafetyDomain.CLINICAL
    
    @property
    def name(self) -> str:
        return "clinical_guideline_monitor"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.critical_conditions = self.config.get('critical_conditions', [
            'stroke', 'heart attack', 'cardiac arrest', 'severe bleeding',
            'anaphylaxis', 'respiratory failure', 'sepsis'
        ])
        self.medication_interaction_threshold = self.config.get('medication_interaction_threshold', 0.8)
        self.diagnostic_confidence_threshold = self.config.get('diagnostic_confidence_threshold', 0.7)
    
    def run(self, context: Dict[str, Any], correlation_id: Optional[str] = None) -> List[SafetyFinding]:
        findings = []
        
        try:
            # Critical condition detection
            predicted_conditions = context.get('predicted_conditions', [])
            for condition in predicted_conditions:
                condition_name = condition.get('name', '').lower()
                confidence = condition.get('confidence', 0.0)
                
                if any(critical in condition_name for critical in self.critical_conditions):
                    severity = 'emergency' if confidence > 0.8 else 'critical'
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity=severity,
                        message=f'Critical condition predicted: {condition_name} (confidence: {confidence:.3f})',
                        value=confidence,
                        correlation_id=correlation_id,
                        metadata={
                            'metric': 'critical_condition',
                            'condition': condition_name,
                            'recommendation': 'immediate_medical_attention'
                        }
                    ))
            
            # Medication interaction check
            medication_risk_score = context.get('medication_interaction_risk')
            if medication_risk_score is not None and medication_risk_score >= self.medication_interaction_threshold:
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity='critical',
                    message=f'High medication interaction risk: {medication_risk_score:.3f}',
                    value=medication_risk_score,
                    threshold=self.medication_interaction_threshold,
                    correlation_id=correlation_id,
                    metadata={'metric': 'medication_interaction', 'recommendation': 'pharmacist_consultation'}
                ))
            
            # Diagnostic confidence check
            diagnostic_confidence = context.get('diagnostic_confidence')
            if diagnostic_confidence is not None and diagnostic_confidence < self.diagnostic_confidence_threshold:
                severity = 'critical' if diagnostic_confidence < 0.5 else 'warning'
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity=severity,
                    message=f'Low diagnostic confidence: {diagnostic_confidence:.3f}',
                    value=diagnostic_confidence,
                    threshold=self.diagnostic_confidence_threshold,
                    correlation_id=correlation_id,
                    metadata={'metric': 'diagnostic_confidence', 'recommendation': 'additional_testing'}
                ))
            
            # Guideline deviation check
            guideline_adherence_score = context.get('guideline_adherence_score')
            if guideline_adherence_score is not None and guideline_adherence_score < 0.8:
                severity = 'critical' if guideline_adherence_score < 0.6 else 'warning'
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity=severity,
                    message=f'Low guideline adherence: {guideline_adherence_score:.3f}',
                    value=guideline_adherence_score,
                    threshold=0.8,
                    correlation_id=correlation_id,
                    metadata={'metric': 'guideline_adherence', 'recommendation': 'clinical_review'}
                ))
            
            # PHI detection check
            phi_detected = context.get('phi_detected', False)
            if phi_detected:
                findings.append(SafetyFinding(
                    domain=self.domain,
                    check_name=self.name,
                    severity='critical',
                    message='Protected Health Information (PHI) detected in output',
                    correlation_id=correlation_id,
                    metadata={'metric': 'phi_leakage', 'recommendation': 'sanitize_output'}
                ))
            
            # Age-inappropriate recommendations
            patient_age = context.get('patient_age')
            recommendations = context.get('recommendations', [])
            if patient_age is not None and recommendations:
                age_appropriate = context.get('age_appropriate_recommendations', True)
                if not age_appropriate:
                    findings.append(SafetyFinding(
                        domain=self.domain,
                        check_name=self.name,
                        severity='warning',
                        message=f'Age-inappropriate recommendations for {patient_age}-year-old patient',
                        correlation_id=correlation_id,
                        metadata={'metric': 'age_appropriateness', 'patient_age': patient_age, 'recommendation': 'age_specific_review'}
                    ))
        
        except Exception as e:
            logger.error(f"Error in clinical safety check: {e}")
            findings.append(SafetyFinding(
                domain=self.domain,
                check_name=self.name,
                severity='warning',
                message=f'Clinical monitoring error: {str(e)}',
                correlation_id=correlation_id,
                metadata={'error': str(e)}
            ))
        
        return findings