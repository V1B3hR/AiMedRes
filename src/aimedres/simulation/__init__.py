"""
Quarantined Simulation Research & Cognitive Multi-Agent Package.

WARNING & GOVERNANCE NOTICE:
This module contains experimental heuristics, cognitive simulations,
and synthetic multi-agent diagnostic interactions.
- NOT APPROVED OR VALIDATED CLINICAL REASONING.
- NOT A REGULATORY-CLEARED MEDICAL DEVICE (FDA SaMD / CE-MDR).
- STRICTLY QUARANTINED FROM PRODUCTION INFERENCE BLUEPRINTS.
- INTENDED FOR COGNITIVE SIMULATION & COMPUTATIONAL ARCHITECTURE RESEARCH ONLY.
"""

from __future__ import annotations

import logging
import os
import warnings

logger = logging.getLogger(__name__)

# Issue clear runtime notice when simulation package is imported
_GOVERNANCE_NOTICE = (
    "EXPERIMENTAL NOTICE: `aimedres.simulation` contains heuristic agent simulations "
    "and unvalidated cognitive prototypes. It is quarantined from production clinical inference."
)
warnings.warn(_GOVERNANCE_NOTICE, UserWarning, stacklevel=2)

from ..agents.specialized_medical_agents import (
    ConsensusManager,
    ExplainabilityEngine,
    MedicalKnowledgeAgent,
    NeurologistAgent,
    RadiologistAgent,
    SpecializedMedicalAgent,
)
from ..core.agent import DuetMindAgent
from ..core.neural_network import AdaptiveNeuralNetwork

__all__ = [
    "MedicalKnowledgeAgent",
    "SpecializedMedicalAgent",
    "RadiologistAgent",
    "NeurologistAgent",
    "ConsensusManager",
    "ExplainabilityEngine",
    "AdaptiveNeuralNetwork",
    "DuetMindAgent",
    "assert_simulation_allowed",
]


def assert_simulation_allowed() -> None:
    """
    Enforce architectural quarantine.
    Fails closed if invoked in production mode with strict clinical boundaries.
    """
    env = os.environ.get("AIMEDRES_ENV", "development").lower()
    strict = os.environ.get("STRICT_CLINICAL_ISOLATION", "false").lower() in ("true", "1")

    if env == "production" and strict:
        raise RuntimeError(
            "CRITICAL ARCHITECTURAL VIOLATION: Attempted to execute unvalidated heuristic "
            "simulation agent in production clinical mode. Simulation modules are quarantined."
        )
