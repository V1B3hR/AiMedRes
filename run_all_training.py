#!/usr/bin/env python3
"""
Backward compatibility wrapper for run_all_training.py

This file has moved to src/aimedres/cli/train.py
This wrapper ensures existing code continues to work.
"""

import sys
import os
import warnings
import subprocess

# Warn about deprecation
warnings.warn(
    "Running run_all_training.py from the root directory is deprecated. "
    "Use 'aimedres train' or 'aimedres-train' command instead. "
    "This compatibility wrapper will be removed in version 2.0.0",
    DeprecationWarning,
    stacklevel=2
)

# Run the script directly
if __name__ == "__main__":
    train_path = os.path.join(os.path.dirname(__file__), 'src', 'aimedres', 'cli', 'train.py')
    result = subprocess.run([sys.executable, train_path] + sys.argv[1:])
    sys.exit(result.returncode)
