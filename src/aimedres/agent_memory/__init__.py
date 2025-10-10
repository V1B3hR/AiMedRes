"""
Agent Memory Module

Advanced memory consolidation system for DuetMind Adaptive.
"""

# Import new population insights separately to avoid legacy dependencies
from .population_insights import (
    PopulationInsightsEngine,
    create_population_insights_engine,
    CohortType,
    TrendDirection
)

# Legacy imports (may have pydantic compatibility issues)
try:
    from .memory_consolidation import MemoryConsolidator, MemoryStore, ConsolidationEvent
    from .embed_memory import AgentMemoryStore, MemoryType
    LEGACY_AVAILABLE = True
except Exception:
    LEGACY_AVAILABLE = False

__all__ = [
    'PopulationInsightsEngine',
    'create_population_insights_engine',
    'CohortType',
    'TrendDirection',
]

if LEGACY_AVAILABLE:
    __all__.extend([
        'MemoryConsolidator',
        'MemoryStore',
        'ConsolidationEvent',
        'AgentMemoryStore',
        'MemoryType',
    ])
