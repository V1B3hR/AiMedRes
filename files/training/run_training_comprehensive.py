#!/usr/bin/env python3
"""
Run comprehensive training for the DuetMind Adaptive System
This script provides comprehensive training of the neural network components
"""

from main import run_comprehensive_training
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    print("=== DuetMind Adaptive System - Comprehensive Training ===")
    success = run_comprehensive_training()
    if success:
        print("✅ Comprehensive training completed successfully!")
        sys.exit(0)
    else:
        print("❌ Training failed!")
        sys.exit(1)