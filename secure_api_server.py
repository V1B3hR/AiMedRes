#!/usr/bin/env python3
"""
Backward compatibility wrapper for secure_api_server.py

This file has moved to src/aimedres/cli/secure_serve.py
This wrapper ensures existing code continues to work.
"""

import sys
import os
import warnings

# Warn about deprecation
warnings.warn(
    "Running secure_api_server.py from the root directory is deprecated. "
    "Use the 'aimedres serve' or 'aimedres-serve' command instead. "
    "This compatibility wrapper will be removed in version 2.0.0",
    DeprecationWarning,
    stacklevel=2
)

# Run the script directly
if __name__ == '__main__':
    source_root = os.path.join(os.path.dirname(__file__), 'src')
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    from aimedres.cli.secure_serve import main as serve_main

    sys.exit(serve_main(sys.argv[1:]))
