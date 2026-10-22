"""
Agents Module (Quarantined Simulation Prototype).

Contains heuristic multi-agent medical simulation logic.
Conforms to Phase 3.1 of the Consolidated Roadmap:
Quarantined from production inference pipelines. For cognitive simulations,
refer to `aimedres.simulation`.
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
