"""
Command-line interface modules.

This package contains CLI commands for training, serving, and demo operations.
"""

from .serve import main as serve_cli
from .train import main as train_cli

__all__ = ["train_cli", "serve_cli"]
