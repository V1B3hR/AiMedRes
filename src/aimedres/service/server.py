"""
Production Inference Service Server.
Conforms to Phase 3.1 & 3.2 of the Consolidated Roadmap.

Encapsulates Flask application factory, security middleware, and registered clinical blueprints.
"""

from typing import Any, Optional

from flask import Flask

from ..api.server import SecureAPIServer
from ..core.config import DuetMindConfig


def create_app(config: Optional[DuetMindConfig] = None) -> Flask:
    """
    Factory creating a production-hardened Flask application instance.
    """
    server = SecureAPIServer(config=config)
    return server.app


__all__ = ["SecureAPIServer", "create_app"]
