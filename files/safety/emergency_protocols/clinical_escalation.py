"""
Clinical Escalation - Automatic Physician Notification

Implements intelligent clinical escalation protocols with automatic
physician notification based on severity, urgency, and clinical context.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone, timedelta
import json


logger = logging.getLogger('duetmind.clinical_escalation')


class EscalationLevel(Enum):
    """Levels of clinical escalation"""
    ROUTINE = "ROUTINE"           # Standard workflow
    EXPEDITED = "EXPEDITED"       # Faster than normal
    URGENT = "URGENT"            # Immediate attention needed
    EMERGENCY = "EMERGENCY"       # Life-threatening situation
    CODE_BLUE = "CODE_BLUE"      # Cardiac/respiratory arrest


class PhysicianRole(Enum):
    """Types of physicians in escalation hierarchy"""
    RESIDENT = "RESIDENT"
    ATTENDING = "ATTENDING"
    SENIOR_ATTENDING = "SENIOR_ATTENDING"
    DEPARTMENT_CHIEF = "DEPARTMENT_CHIEF"
    MEDICAL_DIRECTOR = "MEDICAL_DIRECTOR"
    SPECIALIST = "SPECIALIST"
    HOSPITALIST = "HOSPITALIST"


class NotificationMethod(Enum):
    """Methods for physician notification"""
    SECURE_MESSAGE = "SECURE_MESSAGE"
    PAGER = "PAGER"
    PHONE_CALL = "PHONE_CALL"
    MOBILE_APP = "MOBILE_APP"
    EMAIL = "EMAIL"
    IN_PERSON = "IN_PERSON"


class EscalationStatus(Enum):
    """Status of escalation"""
    PENDING = "PENDING"
    NOTIFIED = "NOTIFIED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESPONDED = "RESPONDED"
    RESOLVED = "RESOLVED"
    ESCALATED_FURTHER = "ESCALATED_FURTHER"
    TIMED_OUT = "TIMED_OUT"


@dataclass
class Physician:
    """Physician information for escalation"""
    physician_id: str
    name: str
    role: PhysicianRole
    specialties: List[str]
    contact_methods: Dict[NotificationMethod, str]
    availability_status: str  # "AVAILABLE", "BUSY", "OFF_DUTY", "ON_CALL"
    current_location: str
    response_time_avg: int  # average response time in minutes
    patient_load: int
    escalation_preferences: Dict[str, Any]


@dataclass
class EscalationEvent:
    """Clinical escalation event"""
    escalation_id: str
    patient_id: str
    patient_name: str
    escalation_level: EscalationLevel
    trigger_reason: str
    clinical_summary: str
    ai_recommendation: Dict[str, Any]
    vital_signs: Dict[str, Any]
    urgency_score: float
    assigned_physician: str
    notification_methods: List[NotificationMethod]
    response_deadline: datetime
    escalation_path: List[str]
    status: EscalationStatus
    created_at: datetime
    notified_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


class ClinicalEscalation:
    """
    Intelligent clinical escalation system with automatic physician notification.
    
    Features:
    - Smart physician assignment based on specialty and availability
    - Multi-channel notification with failover
    - Automatic escalation if no response within deadline
    - Integration with hospital paging and communication systems
    - Response time tracking and optimization
    - Emergency protocol activation
    """
    
    def __init__(self):
        """Initialize clinical escalation system"""
        self.physicians = {}
        self.active_escalations = {}
        self.escalation_history = []
        self.notification_callbacks = {}
        
        # Configuration
        self.response_time_limits = {
            EscalationLevel.ROUTINE: 240,      # 4 hours
            EscalationLevel.EXPEDITED: 60,     # 1 hour
            EscalationLevel.URGENT: 15,        # 15 minutes
            EscalationLevel.EMERGENCY: 5,      # 5 minutes
            EscalationLevel.CODE_BLUE: 2       # 2 minutes
        }
        
        # Escalation rules
        self.escalation_rules = self._initialize_escalation_rules()
        
        # Initialize physician directory
        self._initialize_physicians()
    
    def _initialize_escalation_rules(self) -> Dict[str, Any]:
        """Initialize clinical escalation rules"""
        return {
            'vital_signs_critical': {
                'conditions': {
                    'heart_rate': {'min': 40, 'max': 150},
                    'blood_pressure_systolic': {'min': 80, 'max': 180},
                    'oxygen_saturation': {'min': 88, 'max': 100},
                    'temperature': {'min': 96.0, 'max': 103.0}
                },
                'escalation_level': EscalationLevel.URGENT,
                'required_specialties': ['emergency_medicine', 'internal_medicine']
            },
            'cardiac_emergency': {
                'conditions': {
                    'symptoms': ['chest_pain', 'shortness_of_breath'],
                    'ecg_abnormal': True
                },
                'escalation_level': EscalationLevel.EMERGENCY,
                'required_specialties': ['cardiology', 'emergency_medicine']
            },
            'stroke_alert': {
                'conditions': {
                    'symptoms': ['facial_drooping', 'arm_weakness', 'speech_difficulty'],
                    'time_window': 180  # minutes
                },
                'escalation_level': EscalationLevel.EMERGENCY,
                'required_specialties': ['neurology', 'emergency_medicine']
            },
            'sepsis_criteria': {
                'conditions': {
                    'temperature': {'min': 100.4, 'max': 999},
                    'heart_rate': {'min': 90, 'max': 999},
                    'altered_mental_status': True
                },
                'escalation_level': EscalationLevel.URGENT,
                'required_specialties': ['infectious_disease', 'internal_medicine']
            },
            'pediatric_emergency': {
                'conditions': {
                    'patient_age': {'min': 0, 'max': 18},
                    'any_critical_vital': True
                },
                'escalation_level': EscalationLevel.URGENT,
                'required_specialties': ['pediatrics', 'emergency_medicine']
            }
        }
    
    def _initialize_physicians(self):
        """Initialize physician directory"""
        sample_physicians = [
            Physician(
                physician_id="attending_001",
                name="Dr. Sarah Johnson",
                role=PhysicianRole.ATTENDING,
                specialties=["emergency_medicine", "internal_medicine"],
                contact_methods={
                    NotificationMethod.PAGER: "5001",
                    NotificationMethod.PHONE_CALL: "+1-555-0201",
                    NotificationMethod.MOBILE_APP: "dr.johnson@hospital.com"
                },
                availability_status="AVAILABLE",
                current_location="ER_A",
                response_time_avg=8,
                patient_load=3,
                escalation_preferences={'preferred_contact': 'PAGER'}
            ),
            Physician(
                physician_id="cardiologist_001",
                name="Dr. Michael Chen",
                role=PhysicianRole.SPECIALIST,
                specialties=["cardiology"],
                contact_methods={
                    NotificationMethod.PAGER: "5002",
                    NotificationMethod.PHONE_CALL: "+1-555-0202",
                    NotificationMethod.SECURE_MESSAGE: "dr.chen@hospital.com"
                },
                availability_status="ON_CALL",
                current_location="CARDIOLOGY_UNIT",
                response_time_avg=12,
                patient_load=2,
                escalation_preferences={'preferred_contact': 'PHONE_CALL'}
            ),
            Physician(
                physician_id="department_chief_001",
                name="Dr. Lisa Rodriguez",
                role=PhysicianRole.DEPARTMENT_CHIEF,
                specialties=["emergency_medicine", "administration"],
                contact_methods={
                    NotificationMethod.PHONE_CALL: "+1-555-0203",
                    NotificationMethod.PAGER: "5003",
                    NotificationMethod.MOBILE_APP: "dr.rodriguez@hospital.com"
                },
                availability_status="AVAILABLE",
                current_location="ADMIN_OFFICE",
                response_time_avg=15,
                patient_load=0,
                escalation_preferences={'preferred_contact': 'PHONE_CALL'}
            )
        ]
        
        for physician in sample_physicians:
            self.physicians[physician.physician_id] = physician
    
    def trigger_escalation(self,
                         patient_id: str,
                         patient_name: str,
                         clinical_context: Dict[str, Any],
                         ai_recommendation: Dict[str, Any],
                         trigger_reason: str,
                         vital_signs: Optional[Dict[str, Any]] = None) -> Optional[EscalationEvent]:
        """
        Trigger clinical escalation based on patient condition.
        
        Args:
            patient_id: Patient identifier
            patient_name: Patient name
            clinical_context: Clinical context and history
            ai_recommendation: AI system recommendation
            trigger_reason: Reason for escalation
            vital_signs: Current vital signs
            
        Returns:
            EscalationEvent if escalation triggered, None otherwise
        """
        # Determine escalation level
        escalation_level = self._determine_escalation_level(
            clinical_context, ai_recommendation, vital_signs, trigger_reason
        )
        
        if escalation_level == EscalationLevel.ROUTINE:
            return None  # No escalation needed
        
        # Calculate urgency score
        urgency_score = self._calculate_urgency_score(
            clinical_context, ai_recommendation, vital_signs, escalation_level
        )
        
        # Find appropriate physician
        assigned_physician = self._assign_physician(
            clinical_context, escalation_level, urgency_score
        )
        
        if not assigned_physician:
            logger.error("No available physician for escalation")
            return None
        
        # Create escalation event
        escalation = self._create_escalation_event(
            patient_id=patient_id,
            patient_name=patient_name,
            escalation_level=escalation_level,
            trigger_reason=trigger_reason,
            clinical_context=clinical_context,
            ai_recommendation=ai_recommendation,
            vital_signs=vital_signs or {},
            urgency_score=urgency_score,
            assigned_physician=assigned_physician
        )
        
        # Store escalation
        self.active_escalations[escalation.escalation_id] = escalation
        self.escalation_history.append(escalation)
        
        # Send notifications
        asyncio.create_task(self._send_escalation_notifications(escalation))
        
        # Start response monitoring
        asyncio.create_task(self._monitor_escalation_response(escalation))
        
        logger.info(f"Clinical escalation triggered: {escalation.escalation_id}")
        
        return escalation
    
    def _determine_escalation_level(self,
                                  clinical_context: Dict[str, Any],
                                  ai_recommendation: Dict[str, Any],
                                  vital_signs: Optional[Dict[str, Any]],
                                  trigger_reason: str) -> EscalationLevel:
        """Determine appropriate escalation level"""
        # Check for emergency conditions
        if self._is_code_blue_condition(clinical_context, vital_signs):
            return EscalationLevel.CODE_BLUE
        
        # Check escalation rules
        for rule_name, rule in self.escalation_rules.items():
            if self._matches_escalation_rule(rule, clinical_context, vital_signs):
                return rule['escalation_level']
        
        # Check AI recommendation urgency
        ai_urgency = ai_recommendation.get('urgency', 'routine')
        if ai_urgency == 'emergency':
            return EscalationLevel.EMERGENCY
        elif ai_urgency == 'urgent':
            return EscalationLevel.URGENT
        elif ai_urgency == 'expedited':
            return EscalationLevel.EXPEDITED
        
        # Check trigger reason
        emergency_triggers = ['cardiac_arrest', 'respiratory_failure', 'stroke', 'severe_bleeding']
        urgent_triggers = ['sepsis', 'severe_pain', 'altered_mental_status']
        
        if any(trigger in trigger_reason.lower() for trigger in emergency_triggers):
            return EscalationLevel.EMERGENCY
        elif any(trigger in trigger_reason.lower() for trigger in urgent_triggers):
            return EscalationLevel.URGENT
        
        return EscalationLevel.EXPEDITED
    
    def _is_code_blue_condition(self,
                              clinical_context: Dict[str, Any],
                              vital_signs: Optional[Dict[str, Any]]) -> bool:
        """Check if condition warrants code blue"""
        if not vital_signs:
            return False
        
        # Cardiac arrest indicators
        if vital_signs.get('heart_rate', 70) < 30:
            return True
        
        # Respiratory arrest indicators
        if vital_signs.get('respiratory_rate', 16) < 6:
            return True
        
        # Severe hypotension
        if vital_signs.get('blood_pressure_systolic', 120) < 60:
            return True
        
        # Unresponsive patient
        if clinical_context.get('consciousness_level') == 'unresponsive':
            return True
        
        return False
    
    def _matches_escalation_rule(self,
                               rule: Dict[str, Any],
                               clinical_context: Dict[str, Any],
                               vital_signs: Optional[Dict[str, Any]]) -> bool:
        """Check if patient data matches escalation rule"""
        conditions = rule.get('conditions', {})
        
        # Check vital signs conditions
        if vital_signs:
            for vital, ranges in conditions.items():
                if vital in vital_signs and isinstance(ranges, dict):
                    value = vital_signs[vital]
                    if isinstance(value, (int, float)):
                        if 'min' in ranges and value < ranges['min']:
                            return True
                        if 'max' in ranges and value > ranges['max']:
                            return True
        
        # Check symptom conditions
        if 'symptoms' in conditions:
            patient_symptoms = clinical_context.get('symptoms', [])
            required_symptoms = conditions['symptoms']
            if any(symptom in patient_symptoms for symptom in required_symptoms):
                return True
        
        # Check age conditions
        if 'patient_age' in conditions:
            patient_age = clinical_context.get('patient_age', 50)
            age_range = conditions['patient_age']
            if age_range['min'] <= patient_age <= age_range['max']:
                return True
        
        return False
    
    def _calculate_urgency_score(self,
                               clinical_context: Dict[str, Any],
                               ai_recommendation: Dict[str, Any],
                               vital_signs: Optional[Dict[str, Any]],
                               escalation_level: EscalationLevel) -> float:
        """Calculate numerical urgency score (0-1)"""
        base_score = {
            EscalationLevel.ROUTINE: 0.2,
            EscalationLevel.EXPEDITED: 0.4,
            EscalationLevel.URGENT: 0.7,
            EscalationLevel.EMERGENCY: 0.9,
            EscalationLevel.CODE_BLUE: 1.0
        }.get(escalation_level, 0.5)
        
        # Adjust based on AI confidence
        ai_confidence = ai_recommendation.get('confidence_score', 0.7)
        if ai_confidence < 0.5:
            base_score += 0.1  # Low confidence increases urgency
        
        # Adjust based on patient risk factors
        risk_factors = clinical_context.get('risk_factors', [])
        high_risk_factors = ['elderly', 'immunocompromised', 'cardiac_history', 'diabetes']
        risk_adjustment = len([rf for rf in risk_factors if rf in high_risk_factors]) * 0.05
        
        # Adjust based on vital sign stability
        if vital_signs:
            stability_score = self._calculate_vital_signs_stability(vital_signs)
            base_score += (1.0 - stability_score) * 0.2
        
        return min(1.0, max(0.0, base_score + risk_adjustment))
    
    def _calculate_vital_signs_stability(self, vital_signs: Dict[str, Any]) -> float:
        """Calculate vital signs stability score (0-1, 1 = stable)"""
        normal_ranges = {
            'heart_rate': (60, 100),
            'blood_pressure_systolic': (90, 140),
            'blood_pressure_diastolic': (60, 90),
            'respiratory_rate': (12, 20),
            'oxygen_saturation': (95, 100),
            'temperature': (97.0, 99.5)
        }
        
        stability_scores = []
        
        for vital, (normal_min, normal_max) in normal_ranges.items():
            if vital in vital_signs:
                value = vital_signs[vital]
                if isinstance(value, (int, float)):
                    if normal_min <= value <= normal_max:
                        stability_scores.append(1.0)
                    else:
                        # Calculate how far outside normal range
                        if value < normal_min:
                            deviation = (normal_min - value) / normal_min
                        else:
                            deviation = (value - normal_max) / normal_max
                        
                        stability = max(0.0, 1.0 - deviation)
                        stability_scores.append(stability)
        
        return sum(stability_scores) / len(stability_scores) if stability_scores else 0.5
    
    def _assign_physician(self,
                        clinical_context: Dict[str, Any],
                        escalation_level: EscalationLevel,
                        urgency_score: float) -> Optional[str]:
        """Assign appropriate physician for escalation"""
        # Get required specialties
        required_specialties = self._get_required_specialties(clinical_context, escalation_level)
        
        # Find available physicians with required specialties
        suitable_physicians = []
        
        for physician_id, physician in self.physicians.items():
            if physician.availability_status in ["AVAILABLE", "ON_CALL"]:
                # Check specialty match
                if not required_specialties or any(
                    spec in physician.specialties for spec in required_specialties
                ):
                    suitable_physicians.append(physician)
        
        if not suitable_physicians:
            # Fallback to any available physician
            suitable_physicians = [
                p for p in self.physicians.values()
                if p.availability_status in ["AVAILABLE", "ON_CALL"]
            ]
        
        if not suitable_physicians:
            return None
        
        # Rank physicians by suitability
        ranked_physicians = self._rank_physicians(
            suitable_physicians, clinical_context, escalation_level, urgency_score
        )
        
        return ranked_physicians[0].physician_id if ranked_physicians else None
    
    def _get_required_specialties(self,
                                clinical_context: Dict[str, Any],
                                escalation_level: EscalationLevel) -> List[str]:
        """Get required physician specialties"""
        # Check escalation rules for specialty requirements
        for rule_name, rule in self.escalation_rules.items():
            if rule.get('escalation_level') == escalation_level:
                if self._matches_escalation_rule(rule, clinical_context, {}):
                    return rule.get('required_specialties', [])
        
        # Default specialty requirements by escalation level
        if escalation_level in [EscalationLevel.EMERGENCY, EscalationLevel.CODE_BLUE]:
            return ['emergency_medicine']
        elif escalation_level == EscalationLevel.URGENT:
            return ['internal_medicine', 'emergency_medicine']
        
        return []
    
    def _rank_physicians(self,
                       physicians: List[Physician],
                       clinical_context: Dict[str, Any],
                       escalation_level: EscalationLevel,
                       urgency_score: float) -> List[Physician]:
        """Rank physicians by suitability for escalation"""
        def calculate_physician_score(physician: Physician) -> float:
            score = 0.0
            
            # Availability score
            if physician.availability_status == "AVAILABLE":
                score += 30
            elif physician.availability_status == "ON_CALL":
                score += 20
            
            # Response time score (lower is better)
            max_response_time = 30
            response_score = max(0, (max_response_time - physician.response_time_avg) / max_response_time * 20)
            score += response_score
            
            # Patient load score (lower load is better)
            max_load = 10
            load_score = max(0, (max_load - physician.patient_load) / max_load * 15)
            score += load_score
            
            # Role hierarchy score
            role_scores = {
                PhysicianRole.RESIDENT: 5,
                PhysicianRole.ATTENDING: 15,
                PhysicianRole.SENIOR_ATTENDING: 20,
                PhysicianRole.SPECIALIST: 25,
                PhysicianRole.DEPARTMENT_CHIEF: 30,
                PhysicianRole.MEDICAL_DIRECTOR: 35
            }
            score += role_scores.get(physician.role, 10)
            
            # Specialty match bonus
            required_specialties = self._get_required_specialties(clinical_context, escalation_level)
            specialty_matches = sum(
                1 for spec in required_specialties
                if spec in physician.specialties
            )
            score += specialty_matches * 10
            
            return score
        
        # Sort by score (highest first)
        return sorted(physicians, key=calculate_physician_score, reverse=True)
    
    def _create_escalation_event(self,
                               patient_id: str,
                               patient_name: str,
                               escalation_level: EscalationLevel,
                               trigger_reason: str,
                               clinical_context: Dict[str, Any],
                               ai_recommendation: Dict[str, Any],
                               vital_signs: Dict[str, Any],
                               urgency_score: float,
                               assigned_physician: str) -> EscalationEvent:
        """Create escalation event"""
        escalation_id = f"ESC_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{patient_id}"
        
        # Generate clinical summary
        clinical_summary = self._generate_clinical_summary(
            patient_name, clinical_context, ai_recommendation, vital_signs, trigger_reason
        )
        
        # Determine notification methods
        physician = self.physicians[assigned_physician]
        notification_methods = self._determine_notification_methods(physician, escalation_level)
        
        # Calculate response deadline
        response_time_limit = self.response_time_limits[escalation_level]
        response_deadline = datetime.now(timezone.utc) + timedelta(minutes=response_time_limit)
        
        # Create escalation path
        escalation_path = self._create_escalation_path(physician, escalation_level)
        
        return EscalationEvent(
            escalation_id=escalation_id,
            patient_id=patient_id,
            patient_name=patient_name,
            escalation_level=escalation_level,
            trigger_reason=trigger_reason,
            clinical_summary=clinical_summary,
            ai_recommendation=ai_recommendation,
            vital_signs=vital_signs,
            urgency_score=urgency_score,
            assigned_physician=assigned_physician,
            notification_methods=notification_methods,
            response_deadline=response_deadline,
            escalation_path=escalation_path,
            status=EscalationStatus.PENDING,
            created_at=datetime.now(timezone.utc)
        )
    
    def _generate_clinical_summary(self,
                                 patient_name: str,
                                 clinical_context: Dict[str, Any],
                                 ai_recommendation: Dict[str, Any],
                                 vital_signs: Dict[str, Any],
                                 trigger_reason: str) -> str:
        """Generate clinical summary for physician"""
        summary_parts = [
            f"CLINICAL ESCALATION - Patient: {patient_name}",
            f"Reason: {trigger_reason}"
        ]
        
        # Add patient demographics
        age = clinical_context.get('patient_age')
        if age:
            summary_parts.append(f"Age: {age}")
        
        # Add vital signs
        if vital_signs:
            vital_summary = ", ".join([
                f"{vital}: {value}" for vital, value in vital_signs.items()
                if isinstance(value, (int, float))
            ])
            if vital_summary:
                summary_parts.append(f"Vitals: {vital_summary}")
        
        # Add symptoms
        symptoms = clinical_context.get('symptoms', [])
        if symptoms:
            summary_parts.append(f"Symptoms: {', '.join(symptoms)}")
        
        # Add AI recommendation
        ai_rec = ai_recommendation.get('primary_recommendation', 'Unknown')
        ai_conf = ai_recommendation.get('confidence_score', 0)
        summary_parts.append(f"AI Recommendation: {ai_rec} (Confidence: {ai_conf:.2f})")
        
        return " | ".join(summary_parts)
    
    def _determine_notification_methods(self,
                                      physician: Physician,
                                      escalation_level: EscalationLevel) -> List[NotificationMethod]:
        """Determine notification methods based on escalation level"""
        if escalation_level == EscalationLevel.CODE_BLUE:
            # Use all available methods for code blue
            return list(physician.contact_methods.keys())
        elif escalation_level == EscalationLevel.EMERGENCY:
            # Use immediate methods
            return [
                method for method in [NotificationMethod.PHONE_CALL, NotificationMethod.PAGER, NotificationMethod.MOBILE_APP]
                if method in physician.contact_methods
            ]
        elif escalation_level == EscalationLevel.URGENT:
            # Use fast methods
            return [
                method for method in [NotificationMethod.PAGER, NotificationMethod.MOBILE_APP, NotificationMethod.SECURE_MESSAGE]
                if method in physician.contact_methods
            ]
        else:
            # Use preferred method
            preferred = physician.escalation_preferences.get('preferred_contact')
            if preferred and NotificationMethod(preferred) in physician.contact_methods:
                return [NotificationMethod(preferred)]
            else:
                return [list(physician.contact_methods.keys())[0]]
    
    def _create_escalation_path(self,
                              physician: Physician,
                              escalation_level: EscalationLevel) -> List[str]:
        """Create escalation path if primary physician doesn't respond"""
        escalation_path = []
        
        # Start with supervisor or senior physician
        if physician.role == PhysicianRole.RESIDENT:
            escalation_path.extend(["attending_001", "department_chief_001"])
        elif physician.role == PhysicianRole.ATTENDING:
            escalation_path.append("department_chief_001")
        
        # For emergency situations, escalate to department chief and medical director
        if escalation_level in [EscalationLevel.EMERGENCY, EscalationLevel.CODE_BLUE]:
            if "department_chief_001" not in escalation_path:
                escalation_path.append("department_chief_001")
            escalation_path.append("medical_director_001")
        
        return escalation_path
    
    async def _send_escalation_notifications(self, escalation: EscalationEvent):
        """Send notifications to assigned physician"""
        physician = self.physicians[escalation.assigned_physician]
        
        notification_tasks = []
        for method in escalation.notification_methods:
            task = asyncio.create_task(
                self._send_notification(method, physician, escalation)
            )
            notification_tasks.append(task)
        
        # Wait for all notifications to be sent
        await asyncio.gather(*notification_tasks, return_exceptions=True)
        
        escalation.status = EscalationStatus.NOTIFIED
        escalation.notified_at = datetime.now(timezone.utc)
        
        logger.info(f"Escalation notifications sent: {escalation.escalation_id}")
    
    async def _send_notification(self,
                               method: NotificationMethod,
                               physician: Physician,
                               escalation: EscalationEvent):
        """Send individual notification"""
        if method not in self.notification_callbacks:
            logger.warning(f"No callback registered for notification method: {method.value}")
            return
        
        contact_info = physician.contact_methods.get(method)
        if not contact_info:
            logger.warning(f"No contact info for {method.value} for physician {physician.physician_id}")
            return
        
        try:
            callback = self.notification_callbacks[method]
            await callback(physician, escalation, contact_info)
            logger.info(f"Notification sent via {method.value} to {physician.physician_id}")
        except Exception as e:
            logger.error(f"Failed to send notification via {method.value}: {e}")
    
    async def _monitor_escalation_response(self, escalation: EscalationEvent):
        """Monitor escalation response and handle timeouts"""
        # Wait for response deadline
        time_to_wait = (escalation.response_deadline - datetime.now(timezone.utc)).total_seconds()
        if time_to_wait > 0:
            await asyncio.sleep(time_to_wait)
        
        # Check if escalation has been handled
        current_escalation = self.active_escalations.get(escalation.escalation_id)
        if not current_escalation or current_escalation.status in [
            EscalationStatus.RESPONDED, EscalationStatus.RESOLVED
        ]:
            return  # Already handled
        
        # Escalate further
        await self._escalate_further(escalation)
    
    async def _escalate_further(self, escalation: EscalationEvent):
        """Escalate to next level if no response"""
        logger.warning(f"Escalating further due to no response: {escalation.escalation_id}")
        
        escalation.status = EscalationStatus.ESCALATED_FURTHER
        
        # Notify next in escalation path
        if escalation.escalation_path:
            next_physician_id = escalation.escalation_path[0]
            escalation.escalation_path = escalation.escalation_path[1:]
            
            if next_physician_id in self.physicians:
                escalation.assigned_physician = next_physician_id
                escalation.status = EscalationStatus.PENDING
                
                # Send notifications to new physician
                await self._send_escalation_notifications(escalation)
                
                # Continue monitoring
                asyncio.create_task(self._monitor_escalation_response(escalation))
            else:
                logger.error(f"Next physician not found: {next_physician_id}")
        else:
            # No more escalation path - mark as timed out
            escalation.status = EscalationStatus.TIMED_OUT
            logger.critical(f"Escalation timed out with no response: {escalation.escalation_id}")
    
    def acknowledge_escalation(self, escalation_id: str, physician_id: str) -> bool:
        """Acknowledge escalation by physician"""
        if escalation_id not in self.active_escalations:
            return False
        
        escalation = self.active_escalations[escalation_id]
        escalation.status = EscalationStatus.ACKNOWLEDGED
        escalation.acknowledged_at = datetime.now(timezone.utc)
        
        logger.info(f"Escalation acknowledged: {escalation_id} by {physician_id}")
        return True
    
    def respond_to_escalation(self, 
                            escalation_id: str, 
                            physician_id: str, 
                            response_notes: str) -> bool:
        """Mark escalation as responded to"""
        if escalation_id not in self.active_escalations:
            return False
        
        escalation = self.active_escalations[escalation_id]
        escalation.status = EscalationStatus.RESPONDED
        escalation.responded_at = datetime.now(timezone.utc)
        
        logger.info(f"Escalation responded to: {escalation_id} by {physician_id}")
        return True
    
    def resolve_escalation(self, 
                         escalation_id: str, 
                         physician_id: str, 
                         resolution_notes: str) -> bool:
        """Resolve escalation"""
        if escalation_id not in self.active_escalations:
            return False
        
        escalation = self.active_escalations[escalation_id]
        escalation.status = EscalationStatus.RESOLVED
        escalation.resolution_notes = resolution_notes
        
        # Remove from active escalations
        del self.active_escalations[escalation_id]
        
        logger.info(f"Escalation resolved: {escalation_id} by {physician_id}")
        return True
    
    def register_notification_callback(self,
                                     method: NotificationMethod,
                                     callback: Callable):
        """Register callback for notification method"""
        self.notification_callbacks[method] = callback
    
    def get_active_escalations(self) -> List[EscalationEvent]:
        """Get all active escalations"""
        return list(self.active_escalations.values())
    
    def get_escalation_statistics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get escalation statistics"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        recent_escalations = [
            e for e in self.escalation_history
            if e.created_at >= cutoff
        ]
        
        if not recent_escalations:
            return {'message': 'No recent escalations'}
        
        # Calculate response times
        response_times = []
        for escalation in recent_escalations:
            if escalation.responded_at and escalation.notified_at:
                response_time = (escalation.responded_at - escalation.notified_at).total_seconds() / 60
                response_times.append(response_time)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Count by level and status
        level_counts = {}
        status_counts = {}
        
        for escalation in recent_escalations:
            level = escalation.escalation_level.value
            status = escalation.status.value
            
            level_counts[level] = level_counts.get(level, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_escalations': len(recent_escalations),
            'active_escalations': len(self.active_escalations),
            'level_distribution': level_counts,
            'status_distribution': status_counts,
            'average_response_time_minutes': avg_response_time,
            'emergency_escalations': level_counts.get('EMERGENCY', 0) + level_counts.get('CODE_BLUE', 0)
        }
    
    def update_physician_availability(self, physician_id: str, status: str):
        """Update physician availability"""
        if physician_id in self.physicians:
            self.physicians[physician_id].availability_status = status
            logger.info(f"Updated physician {physician_id} availability to {status}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get escalation system status"""
        available_physicians = sum(
            1 for p in self.physicians.values()
            if p.availability_status in ["AVAILABLE", "ON_CALL"]
        )
        
        return {
            'system_operational': True,
            'active_escalations': len(self.active_escalations),
            'total_physicians': len(self.physicians),
            'available_physicians': available_physicians,
            'notification_methods_configured': len(self.notification_callbacks)
        }