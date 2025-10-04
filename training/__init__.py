"""
Compatibility shim for training module.

This module has been moved to src/aimedres/training/.
This shim provides backward compatibility for existing imports.
"""

import warnings

warnings.warn(
    "Importing from 'training' is deprecated. "
    "Use 'from aimedres.training import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location
from aimedres.training import *  # noqa: F401, F403
