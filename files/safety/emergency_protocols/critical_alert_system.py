"""
Critical Alert System - Life-threatening Condition Alerts

Implements immediate alerting system for life-threatening conditions
with multi-channel notifications and escalation protocols.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone, timedelta
import json
from collections import deque


logger = logging.getLogger('duetmind.critical_alert_system')


class AlertSeverity(Enum):
    """Severity levels for critical alerts"""
    EMERGENCY = "EMERGENCY"      # Immediate life threat - <2 minutes response
    URGENT = "URGENT"           # Serious condition - <15 minutes response
    HIGH = "HIGH"               # Significant risk - <1 hour response
    MODERATE = "MODERATE"       # Elevated concern - <4 hours response


class AlertType(Enum):
    """Types of critical medical alerts"""
    CARDIAC_ARREST = "CARDIAC_ARREST"
    STROKE = "STROKE"
    SEPSIS = "SEPSIS"
    RESPIRATORY_FAILURE = "RESPIRATORY_FAILURE"
    ANAPHYLAXIS = "ANAPHYLAXIS"
    HEMORRHAGE = "HEMORRHAGE"
    DIABETIC_EMERGENCY = "DIABETIC_EMERGENCY"
    DRUG_OVERDOSE = "DRUG_OVERDOSE"
    TRAUMA = "TRAUMA"
    PSYCHIATRIC_EMERGENCY = "PSYCHIATRIC_EMERGENCY"
    VITAL_SIGNS_CRITICAL = "VITAL_SIGNS_CRITICAL"
    AI_SAFETY_ALERT = "AI_SAFETY_ALERT"


class NotificationChannel(Enum):
    """Available notification channels"""
    SMS = "SMS"
    EMAIL = "EMAIL"
    PHONE_CALL = "PHONE_CALL"
    PAGER = "PAGER"
    MOBILE_APP = "MOBILE_APP"
    HOSPITAL_PA = "HOSPITAL_PA"
    EMERGENCY_SYSTEM = "EMERGENCY_SYSTEM"


class AlertStatus(Enum):
    """Status of alert"""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESPONDED = "RESPONDED"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    CANCELLED = "CANCELLED"


@dataclass
class CriticalAlert:
    """Critical alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    patient_id: str
    patient_name: str
    location: str
    condition_description: str
    ai_recommendation: Dict[str, Any]
    vital_signs: Dict[str, Any]
    clinical_context: Dict[str, Any]
    required_response_time: int  # minutes
    notification_channels: List[NotificationChannel]
    assigned_responders: List[str]
    escalation_path: List[str]
    status: AlertStatus
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


@dataclass
class ResponderInfo:
    """Information about emergency responders"""
    responder_id: str
    name: str
    role: str
    specialties: List[str]
    contact_info: Dict[str, str]
    availability_status: str
    location: str
    response_time_avg: int  # minutes


class CriticalAlertSystem:
    """
    Critical alert system for life-threatening medical conditions.
    
    Features:
    - Real-time condition monitoring and alert generation
    - Multi-channel notification system
    - Automatic escalation protocols
    - Response time tracking
    - Integration with hospital emergency systems
    - AI safety alert integration
    """
    
    def __init__(self):
        """Initialize critical alert system"""
        self.active_alerts = {}
        self.alert_history = []
        self.responders = {}
        self.notification_callbacks = {}
        
        # Configuration
        self.max_response_times = {
            AlertSeverity.EMERGENCY: 2,
            AlertSeverity.URGENT: 15,
            AlertSeverity.HIGH: 60,
            AlertSeverity.MODERATE: 240
        }
        
        # Critical condition patterns
        self.critical_patterns = self._initialize_critical_patterns()
        
        # Initialize responder database
        self._initialize_responders()
    
    def _initialize_critical_patterns(self) -> Dict[AlertType, Dict[str, Any]]:
        """Initialize patterns for detecting critical conditions"""
        return {
            AlertType.CARDIAC_ARREST: {
                'vital_signs': {
                    'heart_rate': {'min': 0, 'max': 30},
                    'blood_pressure_systolic': {'min': 0, 'max': 60}
                },
                'symptoms': ['chest_pain', 'unconsciousness', 'no_pulse'],
                'severity': AlertSeverity.EMERGENCY
            },
            AlertType.STROKE: {
                'symptoms': ['facial_drooping', 'arm_weakness', 'speech_difficulty'],
                'time_critical': True,
                'severity': AlertSeverity.EMERGENCY
            },
            AlertType.SEPSIS: {
                'vital_signs': {
                    'temperature': {'min': 100.4, 'max': 999},
                    'heart_rate': {'min': 90, 'max': 999},
                    'white_blood_cell_count': {'min': 12000, 'max': 999999}
                },
                'severity': AlertSeverity.URGENT
            },
            AlertType.RESPIRATORY_FAILURE: {
                'vital_signs': {
                    'respiratory_rate': {'min': 0, 'max': 8},
                    'oxygen_saturation': {'min': 0, 'max': 88}
                },
                'severity': AlertSeverity.EMERGENCY
            },
            AlertType.ANAPHYLAXIS: {
                'symptoms': ['difficulty_breathing', 'swelling', 'severe_allergic_reaction'],
                'vital_signs': {
                    'blood_pressure_systolic': {'min': 0, 'max': 90}
                },
                'severity': AlertSeverity.EMERGENCY
            }
        }
    
    def _initialize_responders(self):
        """Initialize emergency responder database"""
        # This would typically load from a database
        # For now, initialize with sample responders
        sample_responders = [
            ResponderInfo(
                responder_id="emergency_physician_001",
                name="Dr. Emergency",
                role="Emergency Physician",
                specialties=["emergency_medicine", "trauma"],
                contact_info={"phone": "+1-555-0101", "pager": "1001"},
                availability_status="AVAILABLE",
                location="ER_Station_1",
                response_time_avg=3
            ),
            ResponderInfo(
                responder_id="charge_nurse_001",
                name="Nurse Manager",
                role="Charge Nurse",
                specialties=["critical_care", "emergency"],
                contact_info={"phone": "+1-555-0102", "pager": "1002"},
                availability_status="AVAILABLE",
                location="ER_Station_2",
                response_time_avg=2
            )
        ]
        
        for responder in sample_responders:
            self.responders[responder.responder_id] = responder
    
    def detect_critical_condition(self,
                                patient_id: str,
                                patient_name: str,
                                location: str,
                                vital_signs: Dict[str, Any],
                                symptoms: List[str],
                                clinical_context: Dict[str, Any],
                                ai_recommendation: Dict[str, Any]) -> Optional[CriticalAlert]:
        """
        Detect critical conditions from patient data and AI recommendations.
        
        Args:
            patient_id: Patient identifier
            patient_name: Patient name
            location: Patient location
            vital_signs: Current vital signs
            symptoms: Reported symptoms
            clinical_context: Clinical context
            ai_recommendation: AI system recommendation
            
        Returns:
            CriticalAlert if critical condition detected, None otherwise
        """
        # Check for critical patterns
        detected_conditions = []
        
        for alert_type, pattern in self.critical_patterns.items():
            if self._matches_critical_pattern(vital_signs, symptoms, pattern):
                detected_conditions.append((alert_type, pattern))
        
        # Check AI recommendation for safety alerts
        if ai_recommendation.get('safety_alert', False):
            detected_conditions.append((AlertType.AI_SAFETY_ALERT, {
                'severity': AlertSeverity.URGENT,
                'ai_triggered': True
            }))
        
        # If no critical conditions detected, return None
        if not detected_conditions:
            return None
        
        # Select most severe condition
        most_severe = max(detected_conditions, 
                         key=lambda x: self._get_severity_priority(x[1].get('severity', AlertSeverity.MODERATE)))
        
        alert_type, pattern = most_severe
        severity = pattern.get('severity', AlertSeverity.MODERATE)
        
        # Create critical alert
        alert = self._create_critical_alert(
            alert_type=alert_type,
            severity=severity,
            patient_id=patient_id,
            patient_name=patient_name,
            location=location,
            vital_signs=vital_signs,
            symptoms=symptoms,
            clinical_context=clinical_context,
            ai_recommendation=ai_recommendation
        )
        
        # Store active alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Trigger immediate notifications
        asyncio.create_task(self._send_critical_notifications(alert))
        
        # Start escalation timer
        asyncio.create_task(self._monitor_response_time(alert))
        
        logger.critical(f"Critical alert generated: {alert.alert_id} - {alert_type.value}")
        
        return alert
    
    def _matches_critical_pattern(self,
                                vital_signs: Dict[str, Any],
                                symptoms: List[str],
                                pattern: Dict[str, Any]) -> bool:
        """Check if patient data matches critical pattern"""
        # Check vital signs
        if 'vital_signs' in pattern:
            for vital, ranges in pattern['vital_signs'].items():
                if vital in vital_signs:
                    value = vital_signs[vital]
                    if isinstance(value, (int, float)):
                        if 'min' in ranges and value < ranges['min']:
                            return True
                        if 'max' in ranges and value > ranges['max']:
                            return True
        
        # Check symptoms
        if 'symptoms' in pattern:
            required_symptoms = pattern['symptoms']
            if any(symptom in symptoms for symptom in required_symptoms):
                return True
        
        return False
    
    def _get_severity_priority(self, severity: AlertSeverity) -> int:
        """Get numerical priority for severity level"""
        priority_map = {
            AlertSeverity.EMERGENCY: 4,
            AlertSeverity.URGENT: 3,
            AlertSeverity.HIGH: 2,
            AlertSeverity.MODERATE: 1
        }
        return priority_map.get(severity, 0)
    
    def _create_critical_alert(self,
                             alert_type: AlertType,
                             severity: AlertSeverity,
                             patient_id: str,
                             patient_name: str,
                             location: str,
                             vital_signs: Dict[str, Any],
                             symptoms: List[str],
                             clinical_context: Dict[str, Any],
                             ai_recommendation: Dict[str, Any]) -> CriticalAlert:
        """Create critical alert object"""
        alert_id = f"CRITICAL_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{alert_type.value}"
        
        # Determine notification channels based on severity
        channels = self._determine_notification_channels(severity)
        
        # Assign responders based on alert type and severity
        assigned_responders = self._assign_responders(alert_type, severity, location)
        
        # Create escalation path
        escalation_path = self._create_escalation_path(severity, alert_type)
        
        # Generate condition description
        condition_description = self._generate_condition_description(
            alert_type, vital_signs, symptoms, ai_recommendation
        )
        
        return CriticalAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            patient_id=patient_id,
            patient_name=patient_name,
            location=location,
            condition_description=condition_description,
            ai_recommendation=ai_recommendation,
            vital_signs=vital_signs,
            clinical_context=clinical_context,
            required_response_time=self.max_response_times[severity],
            notification_channels=channels,
            assigned_responders=assigned_responders,
            escalation_path=escalation_path,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(timezone.utc)
        )
    
    def _determine_notification_channels(self, severity: AlertSeverity) -> List[NotificationChannel]:
        """Determine notification channels based on severity"""
        if severity == AlertSeverity.EMERGENCY:
            return [
                NotificationChannel.PHONE_CALL,
                NotificationChannel.PAGER,
                NotificationChannel.MOBILE_APP,
                NotificationChannel.HOSPITAL_PA,
                NotificationChannel.EMERGENCY_SYSTEM
            ]
        elif severity == AlertSeverity.URGENT:
            return [
                NotificationChannel.PAGER,
                NotificationChannel.MOBILE_APP,
                NotificationChannel.SMS,
                NotificationChannel.EMAIL
            ]
        else:
            return [
                NotificationChannel.MOBILE_APP,
                NotificationChannel.EMAIL
            ]
    
    def _assign_responders(self,
                         alert_type: AlertType,
                         severity: AlertSeverity,
                         location: str) -> List[str]:
        """Assign appropriate responders based on alert characteristics"""
        assigned = []
        
        # Find available responders
        available_responders = [
            r for r in self.responders.values()
            if r.availability_status == "AVAILABLE"
        ]
        
        # Sort by response time and specialization
        if alert_type in [AlertType.CARDIAC_ARREST, AlertType.RESPIRATORY_FAILURE]:
            # Prioritize emergency physicians and critical care specialists
            emergency_responders = [
                r for r in available_responders
                if "emergency_medicine" in r.specialties or "critical_care" in r.specialties
            ]
            assigned.extend([r.responder_id for r in emergency_responders[:2]])
        
        # Always include charge nurse for coordination
        charge_nurses = [
            r for r in available_responders
            if r.role == "Charge Nurse"
        ]
        if charge_nurses:
            assigned.append(charge_nurses[0].responder_id)
        
        # For emergency severity, assign additional responders
        if severity == AlertSeverity.EMERGENCY and len(assigned) < 3:
            additional_responders = [
                r for r in available_responders
                if r.responder_id not in assigned
            ][:3-len(assigned)]
            assigned.extend([r.responder_id for r in additional_responders])
        
        return assigned
    
    def _create_escalation_path(self, 
                              severity: AlertSeverity, 
                              alert_type: AlertType) -> List[str]:
        """Create escalation path for alert"""
        if severity == AlertSeverity.EMERGENCY:
            return [
                "attending_physician",
                "department_chief",
                "hospital_administrator",
                "medical_director"
            ]
        elif severity == AlertSeverity.URGENT:
            return [
                "senior_physician",
                "attending_physician",
                "department_chief"
            ]
        else:
            return [
                "attending_physician"
            ]
    
    def _generate_condition_description(self,
                                      alert_type: AlertType,
                                      vital_signs: Dict[str, Any],
                                      symptoms: List[str],
                                      ai_recommendation: Dict[str, Any]) -> str:
        """Generate human-readable condition description"""
        description_parts = [f"CRITICAL ALERT: {alert_type.value.replace('_', ' ').title()}"]
        
        # Add vital signs information
        if vital_signs:
            critical_vitals = []
            for vital, value in vital_signs.items():
                critical_vitals.append(f"{vital}: {value}")
            
            if critical_vitals:
                description_parts.append(f"Vital Signs: {', '.join(critical_vitals)}")
        
        # Add symptoms
        if symptoms:
            description_parts.append(f"Symptoms: {', '.join(symptoms)}")
        
        # Add AI recommendation summary
        if ai_recommendation:
            recommendation = ai_recommendation.get('primary_recommendation', 'Unknown')
            confidence = ai_recommendation.get('confidence_score', 0)
            description_parts.append(f"AI Recommendation: {recommendation} (Confidence: {confidence:.2f})")
        
        return " | ".join(description_parts)
    
    async def _send_critical_notifications(self, alert: CriticalAlert):
        """Send notifications through all specified channels"""
        notification_tasks = []
        
        for channel in alert.notification_channels:
            for responder_id in alert.assigned_responders:
                task = asyncio.create_task(
                    self._send_notification(channel, responder_id, alert)
                )
                notification_tasks.append(task)
        
        # Wait for all notifications to be sent
        await asyncio.gather(*notification_tasks, return_exceptions=True)
        
        logger.info(f"Notifications sent for alert {alert.alert_id}")
    
    async def _send_notification(self,
                               channel: NotificationChannel,
                               responder_id: str,
                               alert: CriticalAlert):
        """Send individual notification"""
        if channel not in self.notification_callbacks:
            logger.warning(f"No callback registered for notification channel: {channel.value}")
            return
        
        responder = self.responders.get(responder_id)
        if not responder:
            logger.error(f"Responder not found: {responder_id}")
            return
        
        try:
            callback = self.notification_callbacks[channel]
            await callback(responder, alert)
            logger.info(f"Notification sent via {channel.value} to {responder_id}")
        except Exception as e:
            logger.error(f"Failed to send notification via {channel.value} to {responder_id}: {e}")
    
    async def _monitor_response_time(self, alert: CriticalAlert):
        """Monitor response time and escalate if necessary"""
        # Wait for required response time
        await asyncio.sleep(alert.required_response_time * 60)  # Convert to seconds
        
        # Check if alert has been acknowledged
        current_alert = self.active_alerts.get(alert.alert_id)
        if not current_alert or current_alert.status != AlertStatus.ACTIVE:
            return  # Alert already handled
        
        # Escalate alert
        await self._escalate_alert(alert)
    
    async def _escalate_alert(self, alert: CriticalAlert):
        """Escalate alert to next level"""
        logger.warning(f"Escalating alert {alert.alert_id} due to no response")
        
        # Update alert status
        alert.status = AlertStatus.ESCALATED
        
        # Notify escalation path
        for escalation_contact in alert.escalation_path:
            # Send escalation notifications
            await self._send_escalation_notification(escalation_contact, alert)
        
        # If emergency alert, also trigger hospital-wide alarm
        if alert.severity == AlertSeverity.EMERGENCY:
            await self._trigger_hospital_wide_alarm(alert)
    
    async def _send_escalation_notification(self, contact: str, alert: CriticalAlert):
        """Send escalation notification"""
        # This would integrate with hospital escalation systems
        logger.critical(f"ESCALATION: Alert {alert.alert_id} escalated to {contact}")
    
    async def _trigger_hospital_wide_alarm(self, alert: CriticalAlert):
        """Trigger hospital-wide emergency alarm"""
        logger.critical(f"HOSPITAL WIDE ALARM: {alert.alert_id} - {alert.condition_description}")
    
    def acknowledge_alert(self, alert_id: str, responder_id: str) -> bool:
        """Acknowledge alert by responder"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(timezone.utc)
        
        logger.info(f"Alert {alert_id} acknowledged by {responder_id}")
        return True
    
    def respond_to_alert(self, alert_id: str, responder_id: str, response_notes: str) -> bool:
        """Mark alert as responded to"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESPONDED
        alert.responded_at = datetime.now(timezone.utc)
        
        logger.info(f"Alert {alert_id} responded to by {responder_id}")
        return True
    
    def resolve_alert(self, alert_id: str, responder_id: str, resolution_notes: str) -> bool:
        """Resolve alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc)
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert {alert_id} resolved by {responder_id}")
        return True
    
    def register_notification_callback(self,
                                     channel: NotificationChannel,
                                     callback: Callable[[ResponderInfo, CriticalAlert], None]):
        """Register callback for notification channel"""
        self.notification_callbacks[channel] = callback
    
    def get_active_alerts(self) -> List[CriticalAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_statistics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get alert statistics for specified time period"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.created_at >= cutoff
        ]
        
        if not recent_alerts:
            return {'message': 'No recent alerts'}
        
        # Calculate statistics
        total_alerts = len(recent_alerts)
        resolved_alerts = sum(1 for alert in recent_alerts if alert.status == AlertStatus.RESOLVED)
        
        # Response time statistics
        response_times = []
        for alert in recent_alerts:
            if alert.responded_at and alert.created_at:
                response_time = (alert.responded_at - alert.created_at).total_seconds() / 60
                response_times.append(response_time)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Alert type distribution
        alert_type_counts = {}
        severity_counts = {}
        
        for alert in recent_alerts:
            alert_type = alert.alert_type.value
            severity = alert.severity.value
            
            alert_type_counts[alert_type] = alert_type_counts.get(alert_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': len(self.active_alerts),
            'resolved_alerts': resolved_alerts,
            'resolution_rate': resolved_alerts / total_alerts if total_alerts > 0 else 0,
            'average_response_time_minutes': avg_response_time,
            'alert_type_distribution': alert_type_counts,
            'severity_distribution': severity_counts,
            'emergency_alerts': severity_counts.get('EMERGENCY', 0)
        }
    
    def update_responder_availability(self, responder_id: str, status: str):
        """Update responder availability status"""
        if responder_id in self.responders:
            self.responders[responder_id].availability_status = status
            logger.info(f"Updated responder {responder_id} availability to {status}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        available_responders = sum(
            1 for r in self.responders.values()
            if r.availability_status == "AVAILABLE"
        )
        
        return {
            'system_operational': True,
            'active_alerts': len(self.active_alerts),
            'total_responders': len(self.responders),
            'available_responders': available_responders,
            'notification_channels_configured': len(self.notification_callbacks),
            'last_alert_time': max(
                (alert.created_at for alert in self.alert_history),
                default=None
            )
        }