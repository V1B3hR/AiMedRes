"""
AiMedRes - Advanced AI Medical Research Assistant

A secure, production-ready AI system combining Adaptive Neural Networks
with intelligent healthcare analytics for medical and research applications.

Main Components:
- Core: Neural network engine and adaptive systems
- API: Secure REST API and web interfaces
- Training: ML training pipelines and data processing
- Security: Authentication, encryption, and compliance
- Utils: Shared utilities and helper functions
"""

__version__ = "1.0.0"
__author__ = "AiMedRes Team"

try:
    from .core.agent import DuetMindAgent
    from .core.config import DuetMindConfig
    from .core.neural_network import AdaptiveNeuralNetwork
    from .security.auth import SecureAuthManager
    from .security.validation import InputValidator

    __all__ = [
        "AdaptiveNeuralNetwork",
        "DuetMindAgent",
        "DuetMindConfig",
        "SecureAuthManager",
        "InputValidator",
    ]
except ImportError:
    # Graceful fallback when web/api dependencies (like Flask) are not installed in pure ML environments
    __all__ = []
