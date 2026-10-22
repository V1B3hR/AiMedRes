"""
Service Model Registry for registered clinical inference models.
Conforms to Phase 3.1 & 3.2 of the Consolidated Roadmap.
"""

from ..api.model_routes import ModelRegistry, model_registry

__all__ = ["ModelRegistry", "model_registry"]
