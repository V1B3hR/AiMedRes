#!/usr/bin/env python3
"""
Backward compatibility wrapper for main.py

This file has moved to src/aimedres/__main__.py
This wrapper ensures existing code continues to work.
"""

import sys
import os
import warnings
import subprocess

# Warn about deprecation
warnings.warn(
    "Running main.py from the root directory is deprecated. "
    "Use 'python -m aimedres' or 'aimedres' command instead. "
    "This compatibility wrapper will be removed in version 2.0.0",
    DeprecationWarning,
    stacklevel=2
)

# Run the script directly
if __name__ == "__main__":
    main_path = os.path.join(os.path.dirname(__file__), 'src', 'aimedres', '__main__.py')
    result = subprocess.run([sys.executable, main_path] + sys.argv[1:])
    sys.exit(result.returncode)
