"""
Agents Module

Specialized medical agents for enhanced multi-agent medical simulation.
"""

from .specialized_medical_agents import (
    ConsensusManager,
    ExplainabilityEngine,
    MedicalKnowledgeAgent,
    NeurologistAgent,
    RadiologistAgent,
    SpecializedMedicalAgent,
)

__all__ = [
    "MedicalKnowledgeAgent",
    "SpecializedMedicalAgent",
    "RadiologistAgent",
    "NeurologistAgent",
    "ConsensusManager",
    "ExplainabilityEngine",
]
