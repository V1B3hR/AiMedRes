#!/usr/bin/env python3
"""
Core Training Orchestrator for AiMedRes

This script provides a convenient wrapper to run all medical AI training
from the core module directory. It delegates to the main run_all_training.py
at the repository root.

Usage from core directory:
    cd /path/to/AiMedRes/src/aimedres/core
    python run_all_training.py [OPTIONS]

Examples:
    # List all available training jobs
    python run_all_training.py --list

    # Run all training with default settings
    python run_all_training.py

    # Run with custom parameters
    python run_all_training.py --epochs 50 --folds 5

    # Run in parallel mode
    python run_all_training.py --parallel --max-workers 6

    # Run specific models only
    python run_all_training.py --only als alzheimers parkinsons
"""

import sys
from pathlib import Path


def main():
    # Determine repository root (3 levels up from core: core -> aimedres -> src -> root)
    core_dir = Path(__file__).resolve().parent
    repo_root = core_dir.parent.parent.parent

    # Add repository root to Python path so imports work correctly
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import and execute the main training orchestrator
    # The main script expects to be run from the repository root
    # so we need to change directory before executing
    import os

    import run_all_training

    original_dir = os.getcwd()
    os.chdir(repo_root)

    try:
        # Execute the main training orchestrator
        exit_code = run_all_training.main()
        sys.exit(exit_code)
    finally:
        # Restore original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
