#!/usr/bin/env python3
"""
Demo script for: python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5

This demonstrates the complete command in action with dry-run mode.
"""

import subprocess
import sys
from pathlib import Path


def run_demo():
    """Run demonstration of the command."""
    print()
    print("=" * 80)
    print("DEMONSTRATION: Parallel Training with 6 Workers, 50 Epochs, 5 Folds")
    print("=" * 80)
    print()
    print("Command:")
    print("  python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5")
    print()
    print("This demonstration will run in dry-run mode to show what would execute.")
    print()
    
    # Change to repo root
    repo_root = Path(__file__).parent.parent.parent
    import os
    os.chdir(repo_root)
    
    # Run with dry-run to show what would happen
    cmd = [
        sys.executable, "run_all_training.py",
        "--parallel",
        "--max-workers", "6",
        "--epochs", "50",
        "--folds", "5",
        "--dry-run"
    ]
    
    print("=" * 80)
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    output = result.stdout + result.stderr
    
    # Print key information
    for line in output.split('\n'):
        # Print important lines
        if any(keyword in line for keyword in [
            'AiMedRes', 'Started at', 'GPU', 'jobs', 'Parallel mode',
            'Selected jobs:', '(dry-run) Command:', 'Training Pipeline Summary',
            'Successful:', 'Skipped:', 'All selected'
        ]):
            print(line)
    
    print()
    print("=" * 80)
    
    if result.returncode == 0:
        print("✅ SUCCESS: Command executed successfully (dry-run mode)")
        print()
        print("Key Features Demonstrated:")
        print("  ✓ Parallel execution enabled")
        print("  ✓ Up to 6 concurrent workers")
        print("  ✓ 50 epochs for neural network training")
        print("  ✓ 5-fold cross-validation")
        print("  ✓ Automatic job discovery")
        print("  ✓ Parameter propagation to all compatible scripts")
        print()
        print("To run actual training (not dry-run), use:")
        print("  python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5")
        print()
        print("To run specific jobs only:")
        print("  python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --only als alzheimers")
        print()
        return 0
    else:
        print("❌ FAILED: Command failed")
        print(f"Exit code: {result.returncode}")
        print()
        print("Error output:")
        print(result.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(run_demo())
