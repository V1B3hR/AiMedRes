"""
Command-line interface modules.

This package contains CLI commands for training, serving, and demo operations.
"""

from .train import main as train_cli


def serve_cli(argv=None):
    """Lazy compatibility wrapper for the maintained serve command."""
    from .secure_serve import main

    return main(argv)


__all__ = ["train_cli", "serve_cli"]
