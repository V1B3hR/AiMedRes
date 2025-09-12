"""
Audit Event System for DuetMind Adaptive MLOps.
Provides immutable audit trail with hash chaining for integrity verification.
"""

from .event_chain import AuditEventChain

__all__ = ['AuditEventChain']