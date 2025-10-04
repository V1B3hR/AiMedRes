"""
Compatibility shim for agent_memory module.

This module has been moved to src/aimedres/agent_memory/.
This shim provides backward compatibility for existing imports.
"""

import warnings

warnings.warn(
    "Importing from 'agent_memory' is deprecated. "
    "Use 'from aimedres.agent_memory import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location
from aimedres.agent_memory import *  # noqa: F401, F403
