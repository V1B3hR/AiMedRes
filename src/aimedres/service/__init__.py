"""
AiMedRes Service Domain (Production Inference & Model Serving).
Conforms to Phase 3.1 & 3.2 of the Consolidated Roadmap.

Provides clean separation of production inference services, model registry,
and security middleware from exploratory research and simulation scripts.
"""

from .registry import ModelRegistry
from .server import SecureAPIServer, create_app

__all__ = [
    "SecureAPIServer",
    "ModelRegistry",
    "create_app",
]
