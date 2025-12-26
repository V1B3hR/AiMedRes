"""
Agent Memory Module

Advanced memory consolidation system for DuetMind Adaptive.
"""

# Import new population insights separately to avoid legacy dependencies
from .population_insights import (
    CohortType,
    PopulationInsightsEngine,
    TrendDirection,
    create_population_insights_engine,
)

# Legacy imports (may have pydantic compatibility issues)
try:
    from .embed_memory import AgentMemoryStore, MemoryType
    from .memory_consolidation import ConsolidationEvent, MemoryConsolidator, MemoryStore

    LEGACY_AVAILABLE = True
except Exception:
    LEGACY_AVAILABLE = False

__all__ = [
    "PopulationInsightsEngine",
    "create_population_insights_engine",
    "CohortType",
    "TrendDirection",
]

if LEGACY_AVAILABLE:
    __all__.extend(
        [
            "MemoryConsolidator",
            "MemoryStore",
            "ConsolidationEvent",
            "AgentMemoryStore",
            "MemoryType",
        ]
    )
