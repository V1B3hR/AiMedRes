#!/usr/bin/env python3
"""
Run the adaptive labyrinth simulation for the DuetMind Adaptive System
This script runs the multi-agent adaptive simulation
"""

from main import run_simulation
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    print("=== DuetMind Adaptive System - Labyrinth Simulation ===")
    success = run_simulation()
    if success:
        print("✅ Simulation completed successfully!")
        sys.exit(0)
    else:
        print("❌ Simulation failed!")
        sys.exit(1)