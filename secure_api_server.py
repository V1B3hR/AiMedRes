#!/usr/bin/env python3
"""
Backward compatibility wrapper for secure_api_server.py

This file has moved to src/aimedres/cli/serve.py
This wrapper ensures existing code continues to work.
"""

import sys
import os
import warnings
import subprocess

# Warn about deprecation
warnings.warn(
    "Running secure_api_server.py from the root directory is deprecated. "
    "Use 'aimedres serve' or 'aimedres-serve' command instead. "
    "This compatibility wrapper will be removed in version 2.0.0",
    DeprecationWarning,
    stacklevel=2
)

# Run the script directly
if __name__ == '__main__':
    serve_path = os.path.join(os.path.dirname(__file__), 'src', 'aimedres', 'cli', 'serve.py')
    result = subprocess.run([sys.executable, serve_path] + sys.argv[1:])
    sys.exit(result.returncode)
