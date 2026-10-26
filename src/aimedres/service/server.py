"""
Maintained research API server.

Encapsulates the Flask application factory and security middleware. It does not
expose clinical inference, EHR, or prototype integration endpoints.
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
