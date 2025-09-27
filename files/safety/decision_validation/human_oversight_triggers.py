"""
Human Oversight Triggers - When to Require Human Review

Implements intelligent triggering mechanisms for human oversight
based on risk assessment, confidence levels, and clinical context.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

from .clinical_confidence_scorer import ConfidenceMetrics, ConfidenceLevel


logger = logging.getLogger('duetmind.human_oversight_triggers')


class OversightUrgency(Enum):
    """Urgency levels for human oversight requests"""
    ROUTINE = "ROUTINE"          # Normal review queue
    PRIORITY = "PRIORITY"        # Expedited review
    URGENT = "URGENT"           # Immediate attention required
    EMERGENCY = "EMERGENCY"      # Critical patient safety issue


class TriggerReason(Enum):
    """Reasons for triggering human oversight"""
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    CRITICAL_CONDITION = "CRITICAL_CONDITION"
    NOVEL_CASE = "NOVEL_CASE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    REGULATORY_REQUIREMENT = "REGULATORY_REQUIREMENT"
    PATIENT_REQUEST = "PATIENT_REQUEST"
    SYSTEM_ANOMALY = "SYSTEM_ANOMALY"
    BIAS_DETECTED = "BIAS_DETECTED"
    ADVERSE_HISTORY = "ADVERSE_HISTORY"


@dataclass
class OversightTrigger:
    """Details of why human oversight was triggered"""
    trigger_id: str
    trigger_reason: TriggerReason
    urgency: OversightUrgency
    confidence_threshold_violated: Optional[float]
    clinical_risk_factors: List[str]
    explanation: str
    required_reviewer_qualifications: List[str]
    estimated_review_time_minutes: int
    escalation_path: List[str]
    timestamp: datetime


@dataclass
class ReviewerRequirements:
    """Requirements for human reviewers"""
    minimum_qualifications: List[str]
    specialty_required: Optional[str]
    experience_level: str  # "JUNIOR", "SENIOR", "ATTENDING", "SPECIALIST"
    certification_required: List[str]
    dual_review_required: bool


class HumanOversightTriggers:
    """
    Intelligent system for determining when human oversight is required
    for AI clinical decisions.
    
    Features:
    - Multi-dimensional risk assessment
    - Confidence-based thresholds
    - Clinical context awareness
    - Regulatory compliance triggers
    - Workload-aware reviewer assignment
    """
    
    def __init__(self, 
                 confidence_thresholds: Optional[Dict[str, float]] = None,
                 specialty_requirements: Optional[Dict[str, List[str]]] = None):
        """
        Initialize human oversight trigger system.
        
        Args:
            confidence_thresholds: Custom confidence thresholds for different scenarios
            specialty_requirements: Specialty requirements for different condition types
        """
        # Default confidence thresholds
        self.confidence_thresholds = confidence_thresholds or {
            'emergency': 0.98,
            'critical': 0.95,
            'high_risk': 0.90,
            'moderate_risk': 0.80,
            'routine': 0.70
        }
        
        # Specialty requirements
        self.specialty_requirements = specialty_requirements or {
            'cardiology': ['cardiology', 'internal_medicine'],
            'neurology': ['neurology', 'internal_medicine'],
            'oncology': ['oncology', 'hematology'],
            'emergency': ['emergency_medicine', 'internal_medicine'],
            'pediatrics': ['pediatrics'],
            'surgery': ['surgery', 'general_surgery']
        }
        
        # Regulatory triggers
        self.regulatory_triggers = {
            'high_risk_device': True,
            'controlled_substances': True,
            'experimental_protocols': True,
            'vulnerable_populations': True
        }
        
        # Reviewer workload tracking
        self.reviewer_workload = {}
        
        # Historical trigger patterns
        self.trigger_history = []
    
    def should_trigger_oversight(self,
                               confidence_metrics: ConfidenceMetrics,
                               clinical_context: Dict[str, Any],
                               ai_recommendation: Dict[str, Any],
                               model_metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[OversightTrigger]]:
        """
        Determine if human oversight should be triggered.
        
        Args:
            confidence_metrics: Confidence assessment from clinical confidence scorer
            clinical_context: Clinical context and patient data
            ai_recommendation: AI recommendation details
            model_metadata: Additional model information
            
        Returns:
            Tuple of (should_trigger, trigger_details)
        """
        trigger_reasons = []
        urgency = OversightUrgency.ROUTINE
        clinical_risk_factors = []
        
        # Check confidence-based triggers
        confidence_trigger = self._check_confidence_triggers(
            confidence_metrics, clinical_context
        )
        if confidence_trigger:
            trigger_reasons.append(confidence_trigger['reason'])
            if confidence_trigger['urgency'].value > urgency.value:
                urgency = confidence_trigger['urgency']
        
        # Check clinical risk factors
        risk_triggers = self._check_clinical_risk_triggers(
            clinical_context, ai_recommendation
        )
        trigger_reasons.extend(risk_triggers['reasons'])
        clinical_risk_factors.extend(risk_triggers['risk_factors'])
        if risk_triggers['urgency'].value > urgency.value:
            urgency = risk_triggers['urgency']
        
        # Check regulatory triggers
        regulatory_trigger = self._check_regulatory_triggers(
            clinical_context, ai_recommendation, model_metadata
        )
        if regulatory_trigger:
            trigger_reasons.append(regulatory_trigger['reason'])
            if regulatory_trigger['urgency'].value > urgency.value:
                urgency = regulatory_trigger['urgency']
        
        # Check system anomaly triggers
        anomaly_trigger = self._check_system_anomaly_triggers(
            confidence_metrics, model_metadata
        )
        if anomaly_trigger:
            trigger_reasons.append(anomaly_trigger['reason'])
            if anomaly_trigger['urgency'].value > urgency.value:
                urgency = anomaly_trigger['urgency']
        
        # Determine if oversight should be triggered
        should_trigger = len(trigger_reasons) > 0
        
        if should_trigger:
            # Create oversight trigger
            trigger = OversightTrigger(
                trigger_id=f"oversight_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                trigger_reason=trigger_reasons[0],  # Primary reason
                urgency=urgency,
                confidence_threshold_violated=self._get_violated_threshold(
                    confidence_metrics, clinical_context
                ),
                clinical_risk_factors=clinical_risk_factors,
                explanation=self._generate_trigger_explanation(
                    trigger_reasons, clinical_risk_factors, confidence_metrics
                ),
                required_reviewer_qualifications=self._determine_reviewer_requirements(
                    clinical_context, urgency
                ),
                estimated_review_time_minutes=self._estimate_review_time(
                    clinical_context, urgency, len(trigger_reasons)
                ),
                escalation_path=self._determine_escalation_path(urgency),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Store trigger history
            self.trigger_history.append(trigger)
            
            return True, trigger
        
        return False, None
    
    def _check_confidence_triggers(self,
                                 confidence_metrics: ConfidenceMetrics,
                                 clinical_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if confidence levels trigger oversight"""
        # Determine required confidence threshold based on clinical context
        severity = clinical_context.get('condition_severity', 'MODERATE')
        
        required_threshold = self.confidence_thresholds.get('routine', 0.70)
        if severity == 'CRITICAL':
            required_threshold = self.confidence_thresholds.get('critical', 0.95)
        elif severity == 'HIGH':
            required_threshold = self.confidence_thresholds.get('high_risk', 0.90)
        elif clinical_context.get('emergency_case', False):
            required_threshold = self.confidence_thresholds.get('emergency', 0.98)
        
        # Check if confidence is below threshold
        if confidence_metrics.risk_adjusted_confidence < required_threshold:
            urgency = OversightUrgency.ROUTINE
            if confidence_metrics.confidence_level in [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW]:
                if severity in ['CRITICAL', 'HIGH']:
                    urgency = OversightUrgency.URGENT
                else:
                    urgency = OversightUrgency.PRIORITY
            
            return {
                'reason': TriggerReason.LOW_CONFIDENCE,
                'urgency': urgency,
                'threshold': required_threshold
            }
        
        # Check uncertainty levels
        if confidence_metrics.uncertainty_score > 0.7:
            return {
                'reason': TriggerReason.HIGH_UNCERTAINTY,
                'urgency': OversightUrgency.PRIORITY,
                'threshold': 0.7
            }
        
        return None
    
    def _check_clinical_risk_triggers(self,
                                    clinical_context: Dict[str, Any],
                                    ai_recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Check clinical risk factors that trigger oversight"""
        reasons = []
        risk_factors = []
        urgency = OversightUrgency.ROUTINE
        
        # Critical condition triggers
        if clinical_context.get('condition_severity') == 'CRITICAL':
            reasons.append(TriggerReason.CRITICAL_CONDITION)
            risk_factors.append('Critical condition severity')
            urgency = OversightUrgency.URGENT
        
        # Novel or rare cases
        if clinical_context.get('rare_condition', False):
            reasons.append(TriggerReason.NOVEL_CASE)
            risk_factors.append('Rare medical condition')
            urgency = max(urgency, OversightUrgency.PRIORITY)
        
        # Multiple comorbidities
        comorbidities = clinical_context.get('comorbidity_count', 0)
        if comorbidities > 5:
            reasons.append(TriggerReason.CONFLICTING_EVIDENCE)
            risk_factors.append(f'Multiple comorbidities ({comorbidities})')
            urgency = max(urgency, OversightUrgency.PRIORITY)
        
        # Vulnerable populations
        patient_age = clinical_context.get('patient_age', 50)
        if patient_age < 18 or patient_age > 80:
            risk_factors.append(f'Vulnerable population (age {patient_age})')
            urgency = max(urgency, OversightUrgency.PRIORITY)
        
        # High-risk medications or procedures
        if ai_recommendation.get('treatment_type') in [
            'HIGH_RISK_MEDICATION', 'SURGICAL_INTERVENTION', 'EXPERIMENTAL_TREATMENT'
        ]:
            reasons.append(TriggerReason.REGULATORY_REQUIREMENT)
            risk_factors.append(f"High-risk treatment: {ai_recommendation.get('treatment_type')}")
            urgency = max(urgency, OversightUrgency.URGENT)
        
        # Patient history of adverse reactions
        if clinical_context.get('adverse_reaction_history', False):
            reasons.append(TriggerReason.ADVERSE_HISTORY)
            risk_factors.append('History of adverse reactions')
            urgency = max(urgency, OversightUrgency.PRIORITY)
        
        return {
            'reasons': reasons,
            'risk_factors': risk_factors,
            'urgency': urgency
        }
    
    def _check_regulatory_triggers(self,
                                 clinical_context: Dict[str, Any],
                                 ai_recommendation: Dict[str, Any],
                                 model_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check regulatory compliance triggers"""
        # FDA high-risk device requirements
        if model_metadata and model_metadata.get('fda_risk_class') == 'III':
            return {
                'reason': TriggerReason.REGULATORY_REQUIREMENT,
                'urgency': OversightUrgency.PRIORITY
            }
        
        # Controlled substance prescriptions
        if ai_recommendation.get('controlled_substance', False):
            return {
                'reason': TriggerReason.REGULATORY_REQUIREMENT,
                'urgency': OversightUrgency.URGENT
            }
        
        # Patient explicit request for human review
        if clinical_context.get('patient_requests_human_review', False):
            return {
                'reason': TriggerReason.PATIENT_REQUEST,
                'urgency': OversightUrgency.PRIORITY
            }
        
        return None
    
    def _check_system_anomaly_triggers(self,
                                     confidence_metrics: ConfidenceMetrics,
                                     model_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check for system anomalies that require oversight"""
        # Model performance degradation
        if model_metadata and model_metadata.get('model_performance_alert', False):
            return {
                'reason': TriggerReason.SYSTEM_ANOMALY,
                'urgency': OversightUrgency.URGENT
            }
        
        # Bias detection alert
        if model_metadata and model_metadata.get('bias_alert', False):
            return {
                'reason': TriggerReason.BIAS_DETECTED,
                'urgency': OversightUrgency.PRIORITY
            }
        
        return None
    
    def _get_violated_threshold(self,
                              confidence_metrics: ConfidenceMetrics,
                              clinical_context: Dict[str, Any]) -> Optional[float]:
        """Get the confidence threshold that was violated"""
        severity = clinical_context.get('condition_severity', 'MODERATE')
        
        if severity == 'CRITICAL':
            threshold = self.confidence_thresholds.get('critical', 0.95)
        elif severity == 'HIGH':
            threshold = self.confidence_thresholds.get('high_risk', 0.90)
        else:
            threshold = self.confidence_thresholds.get('routine', 0.70)
        
        if confidence_metrics.risk_adjusted_confidence < threshold:
            return threshold
        
        return None
    
    def _generate_trigger_explanation(self,
                                    trigger_reasons: List[TriggerReason],
                                    clinical_risk_factors: List[str],
                                    confidence_metrics: ConfidenceMetrics) -> str:
        """Generate human-readable explanation for oversight trigger"""
        explanation_parts = []
        
        # Primary trigger reason
        primary_reason = trigger_reasons[0]
        if primary_reason == TriggerReason.LOW_CONFIDENCE:
            explanation_parts.append(
                f"AI confidence level ({confidence_metrics.confidence_level.value}) "
                f"below required threshold for this clinical scenario"
            )
        elif primary_reason == TriggerReason.HIGH_UNCERTAINTY:
            explanation_parts.append(
                f"High uncertainty detected (score: {confidence_metrics.uncertainty_score:.2f})"
            )
        elif primary_reason == TriggerReason.CRITICAL_CONDITION:
            explanation_parts.append("Critical patient condition requires mandatory human review")
        elif primary_reason == TriggerReason.REGULATORY_REQUIREMENT:
            explanation_parts.append("Regulatory compliance requires human oversight")
        
        # Clinical risk factors
        if clinical_risk_factors:
            explanation_parts.append(f"Clinical risk factors: {', '.join(clinical_risk_factors)}")
        
        return ". ".join(explanation_parts)
    
    def _determine_reviewer_requirements(self,
                                       clinical_context: Dict[str, Any],
                                       urgency: OversightUrgency) -> List[str]:
        """Determine required qualifications for reviewer"""
        requirements = ['licensed_physician']
        
        # Specialty requirements
        condition = clinical_context.get('primary_condition', '')
        for specialty, conditions in self.specialty_requirements.items():
            if condition in conditions:
                requirements.append(f'specialty_{specialty}')
                break
        
        # Experience requirements based on urgency
        if urgency == OversightUrgency.EMERGENCY:
            requirements.append('attending_level')
        elif urgency == OversightUrgency.URGENT:
            requirements.append('senior_level')
        
        # Dual review for high-risk cases
        if clinical_context.get('condition_severity') == 'CRITICAL':
            requirements.append('dual_review_required')
        
        return requirements
    
    def _estimate_review_time(self,
                            clinical_context: Dict[str, Any],
                            urgency: OversightUrgency,
                            num_triggers: int) -> int:
        """Estimate time required for human review in minutes"""
        base_time = 5  # Base review time
        
        # Complexity factors
        complexity_multiplier = 1.0
        if clinical_context.get('condition_severity') == 'CRITICAL':
            complexity_multiplier *= 2.0
        if clinical_context.get('comorbidity_count', 0) > 3:
            complexity_multiplier *= 1.5
        if num_triggers > 2:
            complexity_multiplier *= 1.3
        
        # Urgency factors
        urgency_multiplier = {
            OversightUrgency.ROUTINE: 1.0,
            OversightUrgency.PRIORITY: 0.8,
            OversightUrgency.URGENT: 0.6,
            OversightUrgency.EMERGENCY: 0.4
        }.get(urgency, 1.0)
        
        estimated_time = int(base_time * complexity_multiplier * urgency_multiplier)
        return max(2, min(30, estimated_time))  # 2-30 minute range
    
    def _determine_escalation_path(self, urgency: OversightUrgency) -> List[str]:
        """Determine escalation path based on urgency"""
        if urgency == OversightUrgency.EMERGENCY:
            return ['attending_physician', 'department_chief', 'medical_director']
        elif urgency == OversightUrgency.URGENT:
            return ['senior_physician', 'attending_physician']
        elif urgency == OversightUrgency.PRIORITY:
            return ['available_physician', 'senior_physician']
        else:
            return ['available_physician']
    
    def get_reviewer_workload(self, reviewer_id: str) -> Dict[str, Any]:
        """Get current workload for a specific reviewer"""
        return self.reviewer_workload.get(reviewer_id, {
            'active_reviews': 0,
            'avg_review_time': 0,
            'specialties': [],
            'availability': 'AVAILABLE'
        })
    
    def update_reviewer_workload(self, reviewer_id: str, workload_data: Dict[str, Any]):
        """Update reviewer workload information"""
        self.reviewer_workload[reviewer_id] = workload_data
    
    def get_trigger_statistics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get statistics on recent oversight triggers"""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours_back * 3600)
        recent_triggers = [
            t for t in self.trigger_history 
            if t.timestamp.timestamp() > cutoff
        ]
        
        if not recent_triggers:
            return {'message': 'No recent triggers'}
        
        # Calculate statistics
        reason_counts = {}
        urgency_counts = {}
        
        for trigger in recent_triggers:
            reason = trigger.trigger_reason.value
            urgency = trigger.urgency.value
            
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
        
        return {
            'total_triggers': len(recent_triggers),
            'trigger_reasons': reason_counts,
            'urgency_distribution': urgency_counts,
            'avg_estimated_review_time': sum(
                t.estimated_review_time_minutes for t in recent_triggers
            ) / len(recent_triggers)
        }