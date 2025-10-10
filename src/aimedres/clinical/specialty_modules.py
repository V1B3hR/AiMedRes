"""
Specialty Clinical Modules (P13)

Implements specialized clinical adaptations for:
- Pediatric care (age normative baselines)
- Geriatric care (polypharmacy risk modeling)
- Emergency department triage (low-latency heuristics)
- Telemedicine integration (session context sync)

These modules provide age-specific and context-specific clinical
decision support optimized for different care settings.
"""

import time
import uuid
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from collections import defaultdict

logger = logging.getLogger('aimedres.clinical.specialty_modules')


class AgeGroup(Enum):
    """Patient age groups for specialized care."""
    NEONATE = "neonate"  # 0-28 days
    INFANT = "infant"  # 29 days - 1 year
    TODDLER = "toddler"  # 1-3 years
    CHILD = "child"  # 3-12 years
    ADOLESCENT = "adolescent"  # 12-18 years
    YOUNG_ADULT = "young_adult"  # 18-40 years
    MIDDLE_ADULT = "middle_adult"  # 40-65 years
    SENIOR = "senior"  # 65-80 years
    ELDERLY = "elderly"  # 80+ years


class TriagePriority(Enum):
    """Emergency department triage levels."""
    IMMEDIATE = 1  # Life-threatening
    URGENT = 2  # Serious, not life-threatening
    SEMI_URGENT = 3  # Moderately serious
    STANDARD = 4  # Non-urgent
    NON_URGENT = 5  # Minor conditions


@dataclass
class PediatricBaseline:
    """Age-normative baselines for pediatric patients."""
    age_group: AgeGroup
    vital_signs: Dict[str, Tuple[float, float]]  # name -> (min, max)
    developmental_milestones: List[str]
    growth_percentiles: Dict[str, float]  # height, weight, head_circumference
    vaccination_schedule: List[str]
    risk_factors: List[str]


@dataclass
class GeriatricProfile:
    """Geriatric patient assessment profile."""
    patient_id: str
    age: int
    medications: List[Dict[str, Any]]
    comorbidities: List[str]
    frailty_score: float  # 0-1, higher = more frail
    fall_risk_score: float  # 0-1, higher = more risk
    cognitive_status: str  # normal, mild_impairment, moderate, severe
    polypharmacy_risk: float  # 0-1, higher = more risk
    drug_interactions: List[Dict[str, Any]]
    recommended_interventions: List[str]


@dataclass
class TriageAssessment:
    """Emergency department triage assessment."""
    patient_id: str
    chief_complaint: str
    vital_signs: Dict[str, float]
    pain_level: int  # 0-10
    triage_priority: TriagePriority
    estimated_wait_time_min: int
    required_resources: List[str]
    red_flags: List[str]
    assessment_time_ms: float
    rationale: str


@dataclass
class TelemedicineSession:
    """Telemedicine consultation session."""
    session_id: str
    patient_id: str
    provider_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    session_type: str = "video"  # video, phone, chat
    chief_complaint: str = ""
    clinical_context: Dict[str, Any] = field(default_factory=dict)
    assessments: List[Dict[str, Any]] = field(default_factory=list)
    prescriptions: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_required: bool = False
    session_notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class PediatricModule:
    """
    Pediatric clinical decision support module.
    
    Provides age-normative baselines and specialized assessments
    for pediatric patients.
    """
    
    def __init__(self):
        """Initialize pediatric module with age-normative data."""
        self._baselines = self._initialize_baselines()
        logger.info("Pediatric module initialized")
    
    def _initialize_baselines(self) -> Dict[AgeGroup, PediatricBaseline]:
        """Initialize age-normative baselines for each age group."""
        baselines = {
            AgeGroup.NEONATE: PediatricBaseline(
                age_group=AgeGroup.NEONATE,
                vital_signs={
                    "heart_rate": (100, 160),
                    "respiratory_rate": (30, 60),
                    "blood_pressure_systolic": (60, 90),
                    "temperature": (36.5, 37.5)
                },
                developmental_milestones=[
                    "reflexes", "feeding", "sleep_patterns"
                ],
                growth_percentiles={
                    "weight": 3.4,  # kg
                    "length": 50.0,  # cm
                    "head_circumference": 35.0  # cm
                },
                vaccination_schedule=["HepB"],
                risk_factors=["jaundice", "feeding_difficulties", "infection"]
            ),
            AgeGroup.INFANT: PediatricBaseline(
                age_group=AgeGroup.INFANT,
                vital_signs={
                    "heart_rate": (100, 150),
                    "respiratory_rate": (25, 50),
                    "blood_pressure_systolic": (70, 100),
                    "temperature": (36.5, 37.5)
                },
                developmental_milestones=[
                    "head_control", "sitting", "babbling", "crawling"
                ],
                growth_percentiles={
                    "weight": 9.0,
                    "length": 72.0,
                    "head_circumference": 45.0
                },
                vaccination_schedule=["DTaP", "IPV", "PCV13", "Hib", "RV"],
                risk_factors=["developmental_delay", "failure_to_thrive"]
            ),
            AgeGroup.TODDLER: PediatricBaseline(
                age_group=AgeGroup.TODDLER,
                vital_signs={
                    "heart_rate": (90, 140),
                    "respiratory_rate": (20, 40),
                    "blood_pressure_systolic": (80, 110),
                    "temperature": (36.5, 37.5)
                },
                developmental_milestones=[
                    "walking", "speech", "social_interaction", "fine_motor"
                ],
                growth_percentiles={
                    "weight": 12.5,
                    "length": 85.0,
                    "head_circumference": 48.0
                },
                vaccination_schedule=["MMR", "Varicella", "HepA"],
                risk_factors=["speech_delay", "behavioral_issues"]
            ),
            AgeGroup.CHILD: PediatricBaseline(
                age_group=AgeGroup.CHILD,
                vital_signs={
                    "heart_rate": (70, 120),
                    "respiratory_rate": (18, 30),
                    "blood_pressure_systolic": (85, 120),
                    "temperature": (36.5, 37.5)
                },
                developmental_milestones=[
                    "reading", "writing", "complex_motor", "social_skills"
                ],
                growth_percentiles={
                    "weight": 23.0,
                    "length": 122.0,
                    "head_circumference": 50.0
                },
                vaccination_schedule=["DTaP", "IPV", "MMR", "Varicella"],
                risk_factors=["learning_disability", "obesity", "asthma"]
            ),
            AgeGroup.ADOLESCENT: PediatricBaseline(
                age_group=AgeGroup.ADOLESCENT,
                vital_signs={
                    "heart_rate": (60, 100),
                    "respiratory_rate": (12, 20),
                    "blood_pressure_systolic": (95, 135),
                    "temperature": (36.5, 37.5)
                },
                developmental_milestones=[
                    "abstract_thinking", "identity_formation", "independence"
                ],
                growth_percentiles={
                    "weight": 52.0,
                    "length": 160.0,
                    "head_circumference": 55.0
                },
                vaccination_schedule=["Tdap", "MenACWY", "HPV"],
                risk_factors=["mental_health", "substance_use", "risky_behavior"]
            )
        }
        return baselines
    
    def get_age_group(self, age_days: int) -> AgeGroup:
        """
        Determine age group from age in days.
        
        Args:
            age_days: Patient age in days
            
        Returns:
            Appropriate AgeGroup
        """
        if age_days <= 28:
            return AgeGroup.NEONATE
        elif age_days <= 365:
            return AgeGroup.INFANT
        elif age_days <= 365 * 3:
            return AgeGroup.TODDLER
        elif age_days <= 365 * 12:
            return AgeGroup.CHILD
        elif age_days <= 365 * 18:
            return AgeGroup.ADOLESCENT
        elif age_days <= 365 * 40:
            return AgeGroup.YOUNG_ADULT
        elif age_days <= 365 * 65:
            return AgeGroup.MIDDLE_ADULT
        elif age_days <= 365 * 80:
            return AgeGroup.SENIOR
        else:
            return AgeGroup.ELDERLY
    
    def assess_vital_signs(
        self,
        age_days: int,
        vital_signs: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Assess vital signs against age-normative baselines.
        
        Args:
            age_days: Patient age in days
            vital_signs: Current vital signs
            
        Returns:
            Assessment with flags for abnormal values
        """
        age_group = self.get_age_group(age_days)
        
        if age_group not in self._baselines:
            return {"error": f"No baseline for age group: {age_group}"}
        
        baseline = self._baselines[age_group]
        assessment = {
            "age_group": age_group.value,
            "vital_signs": {},
            "abnormal_flags": [],
            "severity": "normal"
        }
        
        for sign_name, sign_value in vital_signs.items():
            if sign_name in baseline.vital_signs:
                min_val, max_val = baseline.vital_signs[sign_name]
                
                if sign_value < min_val:
                    status = "low"
                    assessment["abnormal_flags"].append(f"{sign_name}_low")
                elif sign_value > max_val:
                    status = "high"
                    assessment["abnormal_flags"].append(f"{sign_name}_high")
                else:
                    status = "normal"
                
                assessment["vital_signs"][sign_name] = {
                    "value": sign_value,
                    "baseline": (min_val, max_val),
                    "status": status
                }
        
        # Determine overall severity
        if len(assessment["abnormal_flags"]) >= 3:
            assessment["severity"] = "critical"
        elif len(assessment["abnormal_flags"]) >= 2:
            assessment["severity"] = "concerning"
        elif len(assessment["abnormal_flags"]) >= 1:
            assessment["severity"] = "monitor"
        
        return assessment
    
    def get_developmental_assessment(
        self,
        age_days: int,
        achieved_milestones: List[str]
    ) -> Dict[str, Any]:
        """
        Assess developmental progress.
        
        Args:
            age_days: Patient age in days
            achieved_milestones: List of achieved milestones
            
        Returns:
            Developmental assessment
        """
        age_group = self.get_age_group(age_days)
        
        if age_group not in self._baselines:
            return {"error": f"No baseline for age group: {age_group}"}
        
        baseline = self._baselines[age_group]
        expected_milestones = set(baseline.developmental_milestones)
        achieved = set(achieved_milestones)
        
        missing_milestones = expected_milestones - achieved
        
        assessment = {
            "age_group": age_group.value,
            "expected_milestones": list(expected_milestones),
            "achieved_milestones": list(achieved),
            "missing_milestones": list(missing_milestones),
            "completion_rate": len(achieved) / len(expected_milestones) if expected_milestones else 1.0,
            "concern_level": "none"
        }
        
        # Determine concern level
        if len(missing_milestones) >= len(expected_milestones) * 0.5:
            assessment["concern_level"] = "high"
        elif len(missing_milestones) >= len(expected_milestones) * 0.25:
            assessment["concern_level"] = "moderate"
        elif len(missing_milestones) > 0:
            assessment["concern_level"] = "low"
        
        return assessment


class GeriatricModule:
    """
    Geriatric clinical decision support module.
    
    Provides polypharmacy risk assessment and specialized
    geriatric care recommendations.
    """
    
    def __init__(self):
        """Initialize geriatric module."""
        self._drug_interactions = self._load_drug_interactions()
        logger.info("Geriatric module initialized")
    
    def _load_drug_interactions(self) -> Dict[str, List[str]]:
        """
        Load common drug interaction database.
        
        Returns:
            Dict mapping drug names to list of interacting drugs
        """
        # Simplified interaction database
        return {
            "warfarin": ["aspirin", "ibuprofen", "omeprazole"],
            "metformin": ["alcohol", "contrast_dye"],
            "lisinopril": ["potassium", "nsaids"],
            "atorvastatin": ["grapefruit", "gemfibrozil"],
            "digoxin": ["quinidine", "verapamil", "amiodarone"],
            "levothyroxine": ["calcium", "iron", "ppi"],
            "prednisone": ["nsaids", "warfarin"],
            "furosemide": ["digoxin", "lithium"]
        }
    
    def assess_polypharmacy_risk(
        self,
        patient_id: str,
        age: int,
        medications: List[Dict[str, Any]],
        comorbidities: List[str]
    ) -> GeriatricProfile:
        """
        Assess polypharmacy risk for geriatric patient.
        
        Args:
            patient_id: Patient identifier
            age: Patient age
            medications: List of current medications
            comorbidities: List of comorbid conditions
            
        Returns:
            GeriatricProfile with risk assessment
        """
        # Calculate medication count risk
        med_count = len(medications)
        med_risk = min(med_count / 10.0, 1.0)  # Risk increases with count
        
        # Identify drug interactions
        drug_interactions = []
        med_names = [med.get("name", "").lower() for med in medications]
        
        for med in med_names:
            if med in self._drug_interactions:
                for interacting_drug in self._drug_interactions[med]:
                    if interacting_drug in med_names:
                        drug_interactions.append({
                            "drug1": med,
                            "drug2": interacting_drug,
                            "severity": "moderate",
                            "recommendation": "monitor_closely"
                        })
        
        # Calculate interaction risk
        interaction_risk = min(len(drug_interactions) / 5.0, 1.0)
        
        # Calculate comorbidity burden
        comorbidity_risk = min(len(comorbidities) / 5.0, 1.0)
        
        # Overall polypharmacy risk (weighted average)
        polypharmacy_risk = (
            0.4 * med_risk +
            0.4 * interaction_risk +
            0.2 * comorbidity_risk
        )
        
        # Calculate frailty score (simplified)
        frailty_score = min((age - 65) / 35.0 + comorbidity_risk * 0.3, 1.0)
        
        # Fall risk assessment
        high_risk_meds = ["benzodiazepines", "antipsychotics", "sedatives"]
        med_keywords = [med.get("class", "").lower() for med in medications]
        fall_risk_from_meds = sum(1 for risk_med in high_risk_meds if any(risk_med in m for m in med_keywords))
        fall_risk_score = min((fall_risk_from_meds / 3.0 + frailty_score * 0.5) / 2.0, 1.0)
        
        # Generate recommendations
        recommendations = []
        if polypharmacy_risk > 0.7:
            recommendations.append("comprehensive_medication_review")
        if len(drug_interactions) > 0:
            recommendations.append("pharmacist_consultation")
        if frailty_score > 0.6:
            recommendations.append("geriatric_assessment")
        if fall_risk_score > 0.5:
            recommendations.append("fall_prevention_program")
        if med_count > 10:
            recommendations.append("deprescribing_evaluation")
        
        # Cognitive status (simplified assessment)
        cognitive_status = "normal"
        if age > 80 and comorbidity_risk > 0.6:
            cognitive_status = "mild_impairment"
        elif age > 85:
            cognitive_status = "monitor"
        
        profile = GeriatricProfile(
            patient_id=patient_id,
            age=age,
            medications=medications,
            comorbidities=comorbidities,
            frailty_score=frailty_score,
            fall_risk_score=fall_risk_score,
            cognitive_status=cognitive_status,
            polypharmacy_risk=polypharmacy_risk,
            drug_interactions=drug_interactions,
            recommended_interventions=recommendations
        )
        
        return profile


class EmergencyTriageModule:
    """
    Emergency department triage module.
    
    Provides rapid, low-latency clinical triage assessments
    using heuristic rules optimized for ED workflows.
    """
    
    def __init__(self):
        """Initialize emergency triage module."""
        self._vital_sign_thresholds = self._initialize_thresholds()
        logger.info("Emergency triage module initialized")
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize vital sign thresholds for triage."""
        return {
            "critical": {
                "heart_rate": (50, 140),
                "respiratory_rate": (8, 30),
                "blood_pressure_systolic": (90, 180),
                "oxygen_saturation": (90, 100),
                "temperature": (35.0, 39.5)
            },
            "urgent": {
                "heart_rate": (60, 120),
                "respiratory_rate": (10, 25),
                "blood_pressure_systolic": (100, 160),
                "oxygen_saturation": (92, 100),
                "temperature": (35.5, 38.5)
            }
        }
    
    def triage_assessment(
        self,
        patient_id: str,
        chief_complaint: str,
        vital_signs: Dict[str, float],
        pain_level: int,
        symptoms: Optional[List[str]] = None
    ) -> TriageAssessment:
        """
        Perform rapid triage assessment.
        
        Args:
            patient_id: Patient identifier
            chief_complaint: Patient's main complaint
            vital_signs: Current vital signs
            pain_level: Pain level (0-10)
            symptoms: Additional symptoms
            
        Returns:
            TriageAssessment with priority and recommendations
        """
        start_time = time.time()
        
        symptoms = symptoms or []
        red_flags = []
        required_resources = []
        
        # Check for immediate red flags
        immediate_keywords = [
            "chest pain", "difficulty breathing", "altered mental status",
            "unresponsive", "severe bleeding", "stroke symptoms",
            "seizure", "severe trauma"
        ]
        
        complaint_lower = chief_complaint.lower()
        for keyword in immediate_keywords:
            if keyword in complaint_lower:
                red_flags.append(keyword)
        
        # Assess vital signs
        critical_vitals = self._vital_sign_thresholds["critical"]
        urgent_vitals = self._vital_sign_thresholds["urgent"]
        
        vital_criticality = "normal"
        for sign_name, sign_value in vital_signs.items():
            if sign_name in critical_vitals:
                min_val, max_val = critical_vitals[sign_name]
                if sign_value < min_val or sign_value > max_val:
                    vital_criticality = "critical"
                    red_flags.append(f"{sign_name}_critical")
                    break
            
            if sign_name in urgent_vitals:
                min_val, max_val = urgent_vitals[sign_name]
                if sign_value < min_val or sign_value > max_val:
                    if vital_criticality != "critical":
                        vital_criticality = "urgent"
        
        # Determine triage priority
        if red_flags or vital_criticality == "critical":
            priority = TriagePriority.IMMEDIATE
            wait_time = 0
            required_resources = ["trauma_bay", "rapid_response_team", "lab_stat"]
        elif vital_criticality == "urgent" or pain_level >= 8:
            priority = TriagePriority.URGENT
            wait_time = 15
            required_resources = ["exam_room", "nurse", "physician"]
        elif pain_level >= 5 or "moderate" in complaint_lower:
            priority = TriagePriority.SEMI_URGENT
            wait_time = 60
            required_resources = ["exam_room", "nurse"]
        elif pain_level >= 2:
            priority = TriagePriority.STANDARD
            wait_time = 120
            required_resources = ["exam_room"]
        else:
            priority = TriagePriority.NON_URGENT
            wait_time = 240
            required_resources = ["waiting_area"]
        
        # Generate rationale
        rationale_parts = []
        if red_flags:
            rationale_parts.append(f"Red flags detected: {', '.join(red_flags[:3])}")
        if vital_criticality != "normal":
            rationale_parts.append(f"Vital signs: {vital_criticality}")
        if pain_level >= 7:
            rationale_parts.append(f"High pain level: {pain_level}/10")
        
        rationale = "; ".join(rationale_parts) if rationale_parts else "Standard triage criteria"
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        assessment = TriageAssessment(
            patient_id=patient_id,
            chief_complaint=chief_complaint,
            vital_signs=vital_signs,
            pain_level=pain_level,
            triage_priority=priority,
            estimated_wait_time_min=wait_time,
            required_resources=required_resources,
            red_flags=red_flags,
            assessment_time_ms=elapsed_ms,
            rationale=rationale
        )
        
        logger.debug(
            f"Triage completed for {patient_id}: "
            f"Priority={priority.name}, Time={elapsed_ms:.1f}ms"
        )
        
        return assessment


class TelemedicineModule:
    """
    Telemedicine integration module.
    
    Provides session management and clinical context synchronization
    for telehealth consultations.
    """
    
    def __init__(self):
        """Initialize telemedicine module."""
        self.active_sessions: Dict[str, TelemedicineSession] = {}
        self.session_lock = threading.RLock()
        logger.info("Telemedicine module initialized")
    
    def start_session(
        self,
        patient_id: str,
        provider_id: str,
        session_type: str = "video",
        chief_complaint: str = ""
    ) -> TelemedicineSession:
        """
        Start a new telemedicine session.
        
        Args:
            patient_id: Patient identifier
            provider_id: Provider identifier
            session_type: Type of session (video, phone, chat)
            chief_complaint: Patient's chief complaint
            
        Returns:
            Created TelemedicineSession
        """
        session_id = str(uuid.uuid4())
        
        session = TelemedicineSession(
            session_id=session_id,
            patient_id=patient_id,
            provider_id=provider_id,
            start_time=datetime.now(),
            session_type=session_type,
            chief_complaint=chief_complaint
        )
        
        with self.session_lock:
            self.active_sessions[session_id] = session
        
        logger.info(
            f"Telemedicine session started: {session_id} "
            f"({session_type})"
        )
        
        return session
    
    def sync_clinical_context(
        self,
        session_id: str,
        clinical_data: Dict[str, Any]
    ) -> bool:
        """
        Synchronize clinical context during session.
        
        Args:
            session_id: Session identifier
            clinical_data: Clinical data to sync
            
        Returns:
            True if successful
        """
        with self.session_lock:
            if session_id not in self.active_sessions:
                logger.error(f"Session not found: {session_id}")
                return False
            
            session = self.active_sessions[session_id]
            session.clinical_context.update(clinical_data)
            
            logger.debug(f"Clinical context synced for session {session_id}")
            
            return True
    
    def add_assessment(
        self,
        session_id: str,
        assessment: Dict[str, Any]
    ) -> bool:
        """Add clinical assessment to session."""
        with self.session_lock:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            assessment["timestamp"] = datetime.now().isoformat()
            session.assessments.append(assessment)
            
            return True
    
    def end_session(
        self,
        session_id: str,
        session_notes: str = "",
        follow_up_required: bool = False
    ) -> Optional[TelemedicineSession]:
        """
        End a telemedicine session.
        
        Args:
            session_id: Session identifier
            session_notes: Clinical notes
            follow_up_required: Whether follow-up is needed
            
        Returns:
            Completed TelemedicineSession or None
        """
        with self.session_lock:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            session.end_time = datetime.now()
            session.session_notes = session_notes
            session.follow_up_required = follow_up_required
            
            # Remove from active sessions
            completed_session = self.active_sessions.pop(session_id)
            
            duration = (session.end_time - session.start_time).total_seconds()
            logger.info(
                f"Telemedicine session ended: {session_id} "
                f"(duration={duration:.1f}s)"
            )
            
            return completed_session
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of session (active or completed)."""
        with self.session_lock:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            
            return {
                "session_id": session.session_id,
                "patient_id": session.patient_id,
                "provider_id": session.provider_id,
                "session_type": session.session_type,
                "start_time": session.start_time.isoformat(),
                "duration_seconds": (
                    (datetime.now() - session.start_time).total_seconds()
                    if not session.end_time
                    else (session.end_time - session.start_time).total_seconds()
                ),
                "chief_complaint": session.chief_complaint,
                "assessments_count": len(session.assessments),
                "status": "active" if not session.end_time else "completed"
            }


# Factory functions

def create_pediatric_module() -> PediatricModule:
    """Create a PediatricModule instance."""
    return PediatricModule()


def create_geriatric_module() -> GeriatricModule:
    """Create a GeriatricModule instance."""
    return GeriatricModule()


def create_emergency_triage_module() -> EmergencyTriageModule:
    """Create an EmergencyTriageModule instance."""
    return EmergencyTriageModule()


def create_telemedicine_module() -> TelemedicineModule:
    """Create a TelemedicineModule instance."""
    return TelemedicineModule()
