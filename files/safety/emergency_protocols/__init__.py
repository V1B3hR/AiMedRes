"""
Emergency Protocols Module

Critical emergency response and safety protocols for clinical AI systems.
"""

from .critical_alert_system import CriticalAlertSystem, CriticalAlert, AlertSeverity, AlertType
from .fail_safe_mechanisms import FailSafeMechanisms, FailureEvent, FailureType, SystemState
from .clinical_escalation import ClinicalEscalation, EscalationEvent, EscalationLevel, Physician
from .liability_documentation import LiabilityDocumentation, LiabilityDocument, SafetyIncident

__all__ = [
    'CriticalAlertSystem',
    'CriticalAlert',
    'AlertSeverity',
    'AlertType',
    'FailSafeMechanisms',
    'FailureEvent',
    'FailureType',
    'SystemState',
    'ClinicalEscalation',
    'EscalationEvent',
    'EscalationLevel',
    'Physician',
    'LiabilityDocumentation',
    'LiabilityDocument',
    'SafetyIncident'
]