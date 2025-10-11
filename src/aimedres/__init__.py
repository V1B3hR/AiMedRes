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

__version__ = "0.2.0"
__author__ = "AiMedRes Team"

# Lazy imports to avoid loading heavy dependencies on import
def __getattr__(name):
    """Lazy load modules on first access."""
    if name == "AdaptiveNeuralNetwork":
        from .core.neural_network import AdaptiveNeuralNetwork
        return AdaptiveNeuralNetwork
    elif name == "DuetMindAgent":
        from .core.agent import DuetMindAgent
        return DuetMindAgent
    elif name == "DuetMindConfig":
        from .core.config import DuetMindConfig
        return DuetMindConfig
    elif name == "SecureAuthManager":
        from .security.auth import SecureAuthManager
        return SecureAuthManager
    elif name == "InputValidator":
        from .security.validation import InputValidator
        return InputValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AdaptiveNeuralNetwork", 
    "DuetMindAgent", 
    "DuetMindConfig",
    "SecureAuthManager",
    "InputValidator",
]