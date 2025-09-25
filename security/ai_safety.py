#!/usr/bin/env python3
"""
AI Safety and Human Oversight System for Clinical Decision Support

Implements comprehensive safety protocols and human-in-the-loop validation
for medical AI systems. Features include:

- Real-time AI decision confidence assessment
- Human oversight triggers for high-risk scenarios
- Clinical decision validation workflows
- Safety monitoring and alert systems
- Bias detection and mitigation
- Emergency escalation protocols
"""

import json
import time
import logging
import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import statistics

logger = logging.getLogger('duetmind.ai_safety')


class RiskLevel(Enum):
    """Risk levels for AI decisions"""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ConfidenceLevel(Enum):
    """AI confidence levels"""
    VERY_LOW = "VERY_LOW"      # 0-40%
    LOW = "LOW"                # 40-60%
    MODERATE = "MODERATE"      # 60-80%
    HIGH = "HIGH"              # 80-95%
    VERY_HIGH = "VERY_HIGH"    # 95-100%


class SafetyAction(Enum):
    """Safety actions that can be taken"""
    PROCEED = "PROCEED"
    REQUIRE_REVIEW = "REQUIRE_REVIEW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    ESCALATE = "ESCALATE"
    BLOCK = "BLOCK"


class BiasType(Enum):
    """Types of bias to detect"""
    DEMOGRAPHIC = "DEMOGRAPHIC"
    SOCIOECONOMIC = "SOCIOECONOMIC"
    GEOGRAPHIC = "GEOGRAPHIC"
    TEMPORAL = "TEMPORAL"
    SELECTION = "SELECTION"
    CONFIRMATION = "CONFIRMATION"


@dataclass
class AIDecision:
    """Structure for AI decision data"""
    decision_id: str
    timestamp: datetime
    model_version: str
    user_id: str
    patient_id: str
    clinical_context: Dict[str, Any]
    ai_recommendation: Dict[str, Any]
    confidence_score: float
    uncertainty_measures: Dict[str, float]
    risk_factors: List[str]
    computed_risk_level: RiskLevel
    safety_action: SafetyAction
    human_oversight_required: bool
    reviewer_id: Optional[str] = None
    human_decision: Optional[Dict[str, Any]] = None
    final_outcome: Optional[str] = None
    safety_notes: Optional[str] = None


@dataclass
class SafetyThresholds:
    """Safety thresholds configuration"""
    confidence_threshold_high_risk: float = 0.95
    confidence_threshold_moderate_risk: float = 0.85
    confidence_threshold_low_risk: float = 0.70
    
    # Risk-based thresholds
    critical_condition_confidence_threshold: float = 0.98
    emergency_confidence_threshold: float = 0.95
    
    # Bias detection thresholds
    demographic_disparity_threshold: float = 0.1
    outcome_disparity_threshold: float = 0.05
    
    # Human oversight requirements
    high_risk_human_approval_required: bool = True
    emergency_dual_approval_required: bool = True
    
    # Performance thresholds
    max_decision_time_ms: float = 100.0
    max_queue_size: int = 1000


class ClinicalAISafetyMonitor:
    """
    Comprehensive AI safety and human oversight system for clinical applications.
    
    Monitors AI decisions in real-time, assesses safety and confidence,
    and ensures appropriate human oversight for high-risk scenarios.
    """
    
    def __init__(self, 
                 thresholds: Optional[SafetyThresholds] = None,
                 enable_bias_detection: bool = True,
                 enable_audit_logging: bool = True):
        """
        Initialize AI Safety Monitor.
        
        Args:
            thresholds: Safety threshold configuration
            enable_bias_detection: Enable bias detection and monitoring
            enable_audit_logging: Enable HIPAA audit logging integration
        """
        self.thresholds = thresholds or SafetyThresholds()
        self.enable_bias_detection = enable_bias_detection
        self.enable_audit_logging = enable_audit_logging
        
        # Decision tracking
        self.active_decisions = {}
        self.decision_history = deque(maxlen=10000)
        self.pending_reviews = {}
        
        # Safety monitoring
        self.safety_alerts = {}
        self.bias_monitoring = {}
        self.performance_metrics = defaultdict(list)
        
        # Human oversight
        self.oversight_queue = deque()
        self.available_reviewers = set()
        self.reviewer_workload = defaultdict(int)
        
        # Threading
        self._lock = threading.RLock()
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Callbacks
        self.safety_alert_callbacks = []
        self.human_oversight_callbacks = []
        
        logger.info("Clinical AI Safety Monitor initialized")
    
    def start_monitoring(self):
        """Start background safety monitoring."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="AISafetyMonitor"
            )
            self._monitor_thread.start()
            logger.info("AI Safety monitoring started")
    
    def stop_monitoring(self):
        """Stop background safety monitoring."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)
        logger.info("AI Safety monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Check for stale pending reviews
                self._check_pending_reviews()
                
                # Monitor bias patterns
                if self.enable_bias_detection:
                    self._monitor_bias_patterns()
                
                # Check system performance
                self._check_safety_performance()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(1.0)  # 1 second monitoring interval
                
            except Exception as e:
                logger.error(f"Error in safety monitoring loop: {e}")
    
    def assess_ai_decision(self,
                          model_version: str,
                          user_id: str,
                          patient_id: str,
                          clinical_context: Dict[str, Any],
                          ai_recommendation: Dict[str, Any],
                          confidence_score: float,
                          model_outputs: Dict[str, Any] = None) -> AIDecision:
        """
        Assess AI decision safety and determine required oversight.
        
        Args:
            model_version: Version of the AI model
            user_id: ID of the requesting user
            patient_id: Patient identifier
            clinical_context: Clinical context and patient data
            ai_recommendation: AI recommendation/prediction
            confidence_score: Model confidence score (0-1)
            model_outputs: Raw model outputs for analysis
            
        Returns:
            AIDecision object with safety assessment
        """
        decision_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Calculate uncertainty measures
        uncertainty_measures = self._calculate_uncertainty_measures(
            confidence_score, model_outputs or {}
        )
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            clinical_context, ai_recommendation, confidence_score
        )
        
        # Compute risk level
        risk_level = self._compute_risk_level(
            clinical_context, confidence_score, risk_factors, ai_recommendation
        )
        
        # Determine safety action
        safety_action, human_oversight_required = self._determine_safety_action(
            risk_level, confidence_score, clinical_context, risk_factors
        )
        
        # Create decision record
        decision = AIDecision(
            decision_id=decision_id,
            timestamp=timestamp,
            model_version=model_version,
            user_id=user_id,
            patient_id=patient_id,
            clinical_context=clinical_context,
            ai_recommendation=ai_recommendation,
            confidence_score=confidence_score,
            uncertainty_measures=uncertainty_measures,
            risk_factors=risk_factors,
            computed_risk_level=risk_level,
            safety_action=safety_action,
            human_oversight_required=human_oversight_required
        )
        
        # Store decision
        with self._lock:
            self.active_decisions[decision_id] = decision
            self.decision_history.append(decision)
        
        # Handle human oversight if required
        if human_oversight_required:
            self._request_human_oversight(decision)
        
        # Log audit trail if enabled
        if self.enable_audit_logging:
            self._log_safety_audit(decision)
        
        # Check for safety alerts
        self._check_safety_alerts(decision)
        
        return decision
    
    def _calculate_uncertainty_measures(self, 
                                       confidence_score: float,
                                       model_outputs: Dict[str, Any]) -> Dict[str, float]:
        """Calculate uncertainty measures from model outputs."""
        uncertainty_measures = {
            'confidence_uncertainty': 1.0 - confidence_score,
            'prediction_variance': 0.0,
            'epistemic_uncertainty': 0.0,
            'aleatoric_uncertainty': 0.0
        }
        
        # Extract variance/uncertainty from model outputs if available
        if 'prediction_variance' in model_outputs:
            uncertainty_measures['prediction_variance'] = float(model_outputs['prediction_variance'])
        
        if 'dropout_variance' in model_outputs:
            uncertainty_measures['epistemic_uncertainty'] = float(model_outputs['dropout_variance'])
        
        if 'output_entropy' in model_outputs:
            uncertainty_measures['aleatoric_uncertainty'] = float(model_outputs['output_entropy'])
        
        return uncertainty_measures
    
    def _identify_risk_factors(self,
                              clinical_context: Dict[str, Any],
                              ai_recommendation: Dict[str, Any],
                              confidence_score: float) -> List[str]:
        """Identify risk factors in AI decision."""
        risk_factors = []
        
        # Low confidence
        if confidence_score < self.thresholds.confidence_threshold_low_risk:
            risk_factors.append("LOW_CONFIDENCE")
        
        # Critical condition detection
        if clinical_context.get('condition_severity') == 'CRITICAL':
            risk_factors.append("CRITICAL_CONDITION")
        
        if clinical_context.get('emergency_case', False):
            risk_factors.append("EMERGENCY_CASE")
        
        # High-risk medications or treatments
        if ai_recommendation.get('treatment_type') in ['HIGH_RISK_MEDICATION', 'SURGICAL_INTERVENTION']:
            risk_factors.append("HIGH_RISK_TREATMENT")
        
        # Pediatric or geriatric patients
        patient_age = clinical_context.get('patient_age', 0)
        if patient_age < 18:
            risk_factors.append("PEDIATRIC_PATIENT")
        elif patient_age > 75:
            risk_factors.append("GERIATRIC_PATIENT")
        
        # Complex medical history
        if len(clinical_context.get('comorbidities', [])) > 3:
            risk_factors.append("COMPLEX_MEDICAL_HISTORY")
        
        # Rare conditions
        if clinical_context.get('condition_prevalence', 1.0) < 0.01:
            risk_factors.append("RARE_CONDITION")
        
        return risk_factors
    
    def _compute_risk_level(self,
                           clinical_context: Dict[str, Any],
                           confidence_score: float,
                           risk_factors: List[str],
                           ai_recommendation: Dict[str, Any]) -> RiskLevel:
        """Compute overall risk level for AI decision."""
        
        # Start with minimal risk
        risk_score = 0
        
        # Risk from low confidence
        if confidence_score < 0.5:
            risk_score += 3
        elif confidence_score < 0.7:
            risk_score += 2
        elif confidence_score < 0.85:
            risk_score += 1
        
        # Risk from factors
        high_risk_factors = ['CRITICAL_CONDITION', 'EMERGENCY_CASE', 'HIGH_RISK_TREATMENT']
        medium_risk_factors = ['PEDIATRIC_PATIENT', 'GERIATRIC_PATIENT', 'COMPLEX_MEDICAL_HISTORY']
        
        for factor in risk_factors:
            if factor in high_risk_factors:
                risk_score += 2
            elif factor in medium_risk_factors:
                risk_score += 1
            else:
                risk_score += 0.5
        
        # Determine risk level
        if risk_score >= 5:
            return RiskLevel.CRITICAL
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MODERATE
        elif risk_score >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _determine_safety_action(self,
                                risk_level: RiskLevel,
                                confidence_score: float,
                                clinical_context: Dict[str, Any],
                                risk_factors: List[str]) -> Tuple[SafetyAction, bool]:
        """Determine appropriate safety action."""
        
        human_oversight_required = False
        
        # Critical risk - always requires human approval
        if risk_level == RiskLevel.CRITICAL:
            if 'EMERGENCY_CASE' in risk_factors and self.thresholds.emergency_dual_approval_required:
                return SafetyAction.ESCALATE, True
            return SafetyAction.REQUIRE_APPROVAL, True
        
        # High risk - requires review or approval
        elif risk_level == RiskLevel.HIGH:
            if confidence_score < self.thresholds.confidence_threshold_high_risk:
                return SafetyAction.REQUIRE_APPROVAL, True
            else:
                return SafetyAction.REQUIRE_REVIEW, True
        
        # Moderate risk - may require review
        elif risk_level == RiskLevel.MODERATE:
            if confidence_score < self.thresholds.confidence_threshold_moderate_risk:
                return SafetyAction.REQUIRE_REVIEW, True
            else:
                return SafetyAction.PROCEED, False
        
        # Low risk - generally can proceed
        elif risk_level == RiskLevel.LOW:
            if confidence_score < self.thresholds.confidence_threshold_low_risk:
                human_oversight_required = True
            return SafetyAction.PROCEED, human_oversight_required
        
        # Minimal risk - proceed without oversight
        else:
            return SafetyAction.PROCEED, False
    
    def _request_human_oversight(self, decision: AIDecision):
        """Request human oversight for AI decision."""
        oversight_request = {
            'decision_id': decision.decision_id,
            'timestamp': decision.timestamp.isoformat(),
            'urgency': self._calculate_oversight_urgency(decision),
            'required_action': decision.safety_action.value,
            'clinical_summary': self._generate_clinical_summary(decision),
            'risk_summary': self._generate_risk_summary(decision)
        }
        
        with self._lock:
            self.pending_reviews[decision.decision_id] = oversight_request
            self.oversight_queue.append(oversight_request)
        
        # Notify human oversight callbacks
        for callback in self.human_oversight_callbacks:
            try:
                callback(oversight_request)
            except Exception as e:
                logger.error(f"Error in human oversight callback: {e}")
        
        logger.info(f"Human oversight requested for decision {decision.decision_id}")
    
    def _calculate_oversight_urgency(self, decision: AIDecision) -> str:
        """Calculate urgency level for human oversight."""
        if decision.computed_risk_level == RiskLevel.CRITICAL:
            return "IMMEDIATE"
        elif decision.computed_risk_level == RiskLevel.HIGH:
            return "URGENT"
        elif decision.computed_risk_level == RiskLevel.MODERATE:
            return "ROUTINE"
        else:
            return "LOW"
    
    def _generate_clinical_summary(self, decision: AIDecision) -> str:
        """Generate clinical summary for human reviewers."""
        patient_age = decision.clinical_context.get('patient_age', 'unknown')
        condition = decision.clinical_context.get('primary_condition', 'unspecified')
        recommendation = decision.ai_recommendation.get('primary_recommendation', 'unspecified')
        
        return f"Patient (age {patient_age}) with {condition}. AI recommends: {recommendation}. Confidence: {decision.confidence_score:.2f}"
    
    def _generate_risk_summary(self, decision: AIDecision) -> str:
        """Generate risk summary for human reviewers."""
        risk_summary = f"Risk Level: {decision.computed_risk_level.value}. "
        if decision.risk_factors:
            risk_summary += f"Risk Factors: {', '.join(decision.risk_factors[:3])}."
        return risk_summary
    
    def submit_human_decision(self,
                             decision_id: str,
                             reviewer_id: str,
                             human_decision: Dict[str, Any],
                             safety_notes: str = None) -> bool:
        """
        Submit human oversight decision.
        
        Args:
            decision_id: ID of the AI decision
            reviewer_id: ID of the human reviewer
            human_decision: Human decision and rationale
            safety_notes: Additional safety notes
            
        Returns:
            True if decision was successfully recorded
        """
        with self._lock:
            if decision_id not in self.active_decisions:
                logger.error(f"Decision {decision_id} not found")
                return False
            
            decision = self.active_decisions[decision_id]
            decision.reviewer_id = reviewer_id
            decision.human_decision = human_decision
            decision.safety_notes = safety_notes
            decision.final_outcome = human_decision.get('final_decision', 'UNKNOWN')
            
            # Remove from pending reviews
            self.pending_reviews.pop(decision_id, None)
            
            # Update reviewer workload
            self.reviewer_workload[reviewer_id] += 1
        
        # Log audit if enabled
        if self.enable_audit_logging:
            self._log_human_decision_audit(decision)
        
        logger.info(f"Human decision submitted for {decision_id} by {reviewer_id}")
        return True
    
    def _log_safety_audit(self, decision: AIDecision):
        """Log AI safety decision to audit system."""
        try:
            from .hipaa_audit import get_audit_logger
            
            audit_logger = get_audit_logger()
            audit_logger.log_clinical_decision(
                user_id=decision.user_id,
                user_role="clinician",
                patient_id=decision.patient_id,
                decision_data={
                    'model_version': decision.model_version,
                    'ai_recommendation': decision.ai_recommendation,
                    'risk_level': decision.computed_risk_level.value,
                    'safety_action': decision.safety_action.value,
                    'risk_factors': decision.risk_factors
                },
                ai_confidence=decision.confidence_score,
                human_override=decision.human_oversight_required
            )
        except ImportError:
            logger.warning("HIPAA audit logging not available for AI safety decision")
    
    def _log_human_decision_audit(self, decision: AIDecision):
        """Log human oversight decision to audit system."""
        try:
            from .hipaa_audit import get_audit_logger
            
            audit_logger = get_audit_logger()
            audit_logger._log_audit_event(
                event_type="CLINICAL_DECISION",
                user_id=decision.reviewer_id or "unknown_reviewer",
                user_role="clinical_reviewer", 
                patient_id=decision.patient_id,
                resource="human_oversight_decision",
                purpose="ai_decision_validation",
                outcome=decision.final_outcome or "REVIEWED",
                additional_data={
                    'original_decision_id': decision.decision_id,
                    'ai_confidence': decision.confidence_score,
                    'human_decision': decision.human_decision,
                    'safety_notes': decision.safety_notes
                }
            )
        except ImportError:
            logger.warning("HIPAA audit logging not available for human oversight decision")
    
    def _check_safety_alerts(self, decision: AIDecision):
        """Check if decision triggers any safety alerts."""
        alerts = []
        
        # Very low confidence alert
        if decision.confidence_score < 0.5:
            alerts.append({
                'type': 'LOW_CONFIDENCE_ALERT',
                'message': f'Very low AI confidence ({decision.confidence_score:.2f}) for patient {decision.patient_id}',
                'severity': 'HIGH'
            })
        
        # Critical risk alert
        if decision.computed_risk_level == RiskLevel.CRITICAL:
            alerts.append({
                'type': 'CRITICAL_RISK_ALERT',
                'message': f'Critical risk AI decision requires immediate review for patient {decision.patient_id}',
                'severity': 'CRITICAL'
            })
        
        # Multiple risk factors alert
        if len(decision.risk_factors) >= 3:
            alerts.append({
                'type': 'MULTIPLE_RISK_FACTORS',
                'message': f'AI decision has {len(decision.risk_factors)} risk factors for patient {decision.patient_id}',
                'severity': 'MEDIUM'
            })
        
        # Trigger alerts
        for alert in alerts:
            self._trigger_safety_alert(alert, decision)
    
    def _trigger_safety_alert(self, alert: Dict[str, Any], decision: AIDecision):
        """Trigger safety alert."""
        alert_id = f"safety_alert_{int(time.time() * 1000)}"
        
        full_alert = {
            'alert_id': alert_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'decision_id': decision.decision_id,
            'patient_id': decision.patient_id,
            'user_id': decision.user_id,
            **alert
        }
        
        with self._lock:
            self.safety_alerts[alert_id] = full_alert
        
        # Call alert callbacks
        for callback in self.safety_alert_callbacks:
            try:
                callback(full_alert)
            except Exception as e:
                logger.error(f"Error in safety alert callback: {e}")
        
        # Log based on severity
        if alert['severity'] == 'CRITICAL':
            logger.error(f"CRITICAL SAFETY ALERT: {alert['message']}")
        elif alert['severity'] == 'HIGH':
            logger.warning(f"HIGH SAFETY ALERT: {alert['message']}")
        else:
            logger.info(f"SAFETY ALERT: {alert['message']}")
    
    def _check_pending_reviews(self):
        """Check for stale pending reviews."""
        now = datetime.now(timezone.utc)
        stale_threshold = timedelta(hours=2)  # 2 hours
        
        with self._lock:
            stale_reviews = []
            for decision_id, request in self.pending_reviews.items():
                request_time = datetime.fromisoformat(request['timestamp'].replace('Z', '+00:00'))
                if now - request_time > stale_threshold:
                    stale_reviews.append((decision_id, request))
            
            # Alert on stale reviews
            for decision_id, request in stale_reviews:
                self._trigger_safety_alert({
                    'type': 'STALE_REVIEW_ALERT',
                    'message': f'Pending review for decision {decision_id} is stale (>{stale_threshold.total_seconds()/3600}h old)',
                    'severity': 'HIGH'
                }, self.active_decisions.get(decision_id))
    
    def _monitor_bias_patterns(self):
        """Monitor for bias patterns in AI decisions."""
        if not self.enable_bias_detection:
            return
        
        # This would implement sophisticated bias detection
        # For now, just a placeholder
        recent_decisions = list(self.decision_history)[-100:]  # Last 100 decisions
        
        if len(recent_decisions) < 20:
            return  # Need minimum data
        
        # Check for demographic bias patterns
        self._check_demographic_bias(recent_decisions)
        
        # Check for outcome disparities
        self._check_outcome_disparities(recent_decisions)
    
    def _check_demographic_bias(self, decisions: List[AIDecision]):
        """Check for demographic bias in AI decisions."""
        # Group decisions by demographic factors
        age_groups = defaultdict(list)
        gender_groups = defaultdict(list)
        
        for decision in decisions:
            patient_age = decision.clinical_context.get('patient_age', 0)
            patient_gender = decision.clinical_context.get('patient_gender', 'unknown')
            
            if patient_age < 18:
                age_groups['pediatric'].append(decision)
            elif patient_age > 65:
                age_groups['geriatric'].append(decision)
            else:
                age_groups['adult'].append(decision)
            
            gender_groups[patient_gender].append(decisions)
        
        # Check for confidence disparities
        for group, group_decisions in age_groups.items():
            if len(group_decisions) > 5:
                avg_confidence = statistics.mean([d.confidence_score for d in group_decisions])
                overall_avg = statistics.mean([d.confidence_score for d in decisions])
                
                if abs(avg_confidence - overall_avg) > self.thresholds.demographic_disparity_threshold:
                    self._trigger_bias_alert(f"Confidence disparity detected for {group} age group")
    
    def _check_outcome_disparities(self, decisions: List[AIDecision]):
        """Check for outcome disparities that might indicate bias."""
        # This would implement more sophisticated bias detection
        # For now, just a placeholder
        pass
    
    def _trigger_bias_alert(self, message: str):
        """Trigger bias detection alert."""
        bias_alert = {
            'type': 'BIAS_DETECTION_ALERT',
            'message': message,
            'severity': 'MEDIUM',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.warning(f"BIAS ALERT: {message}")
        
        for callback in self.safety_alert_callbacks:
            try:
                callback(bias_alert)
            except Exception as e:
                logger.error(f"Error in bias alert callback: {e}")
    
    def _check_safety_performance(self):
        """Check overall safety system performance."""
        with self._lock:
            # Check queue sizes
            if len(self.oversight_queue) > self.thresholds.max_queue_size:
                self._trigger_safety_alert({
                    'type': 'OVERSIGHT_QUEUE_OVERLOAD',
                    'message': f'Oversight queue has {len(self.oversight_queue)} pending items',
                    'severity': 'HIGH'
                }, None)
            
            # Check for decision processing delays
            recent_decisions = [d for d in self.decision_history if 
                             datetime.now(timezone.utc) - d.timestamp < timedelta(minutes=10)]
            
            if recent_decisions:
                avg_processing_time = statistics.mean([
                    1000  # Placeholder - would calculate actual processing time
                    for d in recent_decisions
                ])
                
                if avg_processing_time > self.thresholds.max_decision_time_ms:
                    self._trigger_safety_alert({
                        'type': 'SLOW_DECISION_PROCESSING',
                        'message': f'Average decision processing time is {avg_processing_time:.1f}ms',
                        'severity': 'MEDIUM'
                    }, None)
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory issues."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        with self._lock:
            # Clean up completed decisions
            completed_decisions = [
                decision_id for decision_id, decision in self.active_decisions.items()
                if decision.timestamp < cutoff_time and decision.final_outcome
            ]
            
            for decision_id in completed_decisions:
                del self.active_decisions[decision_id]
            
            # Clean up old alerts
            old_alerts = [
                alert_id for alert_id, alert in self.safety_alerts.items()
                if datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00')) < cutoff_time
            ]
            
            for alert_id in old_alerts:
                del self.safety_alerts[alert_id]
    
    def add_safety_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for safety alerts."""
        self.safety_alert_callbacks.append(callback)
    
    def add_human_oversight_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for human oversight requests."""
        self.human_oversight_callbacks.append(callback)
    
    def get_safety_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get safety monitoring summary."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        with self._lock:
            recent_decisions = [d for d in self.decision_history if d.timestamp >= cutoff_time]
            
            if not recent_decisions:
                return {
                    'summary_period_hours': hours_back,
                    'total_decisions': 0,
                    'safety_status': 'NO_DATA'
                }
            
            # Calculate summary statistics
            high_risk_decisions = sum(1 for d in recent_decisions 
                                    if d.computed_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
            
            human_oversight_decisions = sum(1 for d in recent_decisions if d.human_oversight_required)
            
            avg_confidence = statistics.mean([d.confidence_score for d in recent_decisions])
            
            pending_reviews_count = len(self.pending_reviews)
            
            active_alerts = len([a for a in self.safety_alerts.values() 
                               if datetime.fromisoformat(a['timestamp'].replace('Z', '+00:00')) >= cutoff_time])
            
            return {
                'summary_period_hours': hours_back,
                'total_decisions': len(recent_decisions),
                'high_risk_decisions': high_risk_decisions,
                'human_oversight_decisions': human_oversight_decisions,
                'human_oversight_rate_percent': (human_oversight_decisions / len(recent_decisions)) * 100,
                'average_confidence': avg_confidence,
                'pending_reviews': pending_reviews_count,
                'active_safety_alerts': active_alerts,
                'safety_status': self._calculate_safety_status(recent_decisions, pending_reviews_count, active_alerts)
            }
    
    def _calculate_safety_status(self, decisions: List[AIDecision], pending_reviews: int, active_alerts: int) -> str:
        """Calculate overall safety status."""
        if not decisions:
            return 'NO_DATA'
        
        high_risk_rate = sum(1 for d in decisions 
                           if d.computed_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]) / len(decisions)
        
        if active_alerts > 5 or pending_reviews > 20:
            return 'CONCERNING'
        elif high_risk_rate > 0.2 or active_alerts > 2:
            return 'ELEVATED'
        elif high_risk_rate > 0.1 or pending_reviews > 5:
            return 'MONITORED'
        else:
            return 'NORMAL'


# Global safety monitor instance
_global_safety_monitor = None

def get_safety_monitor() -> ClinicalAISafetyMonitor:
    """Get global AI safety monitor instance."""
    global _global_safety_monitor
    if _global_safety_monitor is None:
        _global_safety_monitor = ClinicalAISafetyMonitor()
        _global_safety_monitor.start_monitoring()
    return _global_safety_monitor