"""
Agent Memory Module

Advanced memory consolidation system for DuetMind Adaptive.
"""

from .memory_consolidation import MemoryConsolidator, MemoryStore, ConsolidationEvent
from .embed_memory import AgentMemoryStore, MemoryType

__all__ = [
    'MemoryConsolidator',
    'MemoryStore',
    'ConsolidationEvent',
    'AgentMemoryStore',
    'MemoryType',
]
