#!/usr/bin/env python3
"""
Backward-compatibility forwarder for Parkinson's disease training pipeline.
The canonical, leakage-safe implementation resides in:
  src/aimedres/training/train_parkinsons.py
"""

import sys
from pathlib import Path

# Ensure src is on sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from aimedres.training.train_parkinsons import (
    ParkinsonTrainingPipeline,
    ParkinsonDataset,
    ParkinsonNeuralNetwork,
    main,
)

__all__ = [
    "ParkinsonTrainingPipeline",
    "ParkinsonDataset",
    "ParkinsonNeuralNetwork",
    "main",
]

if __name__ == "__main__":
    main()
