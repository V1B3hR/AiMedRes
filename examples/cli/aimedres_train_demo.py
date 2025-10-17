#!/usr/bin/env python3
"""
Demo script for the aimedres CLI train command:
  aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128

This demonstrates the modern CLI interface for training medical AI models.
"""

import subprocess
import sys
from pathlib import Path


def run_demo():
    """Run demonstration of the aimedres CLI train command."""
    print()
    print("=" * 80)
    print("DEMONSTRATION: aimedres CLI Train Command")
    print("=" * 80)
    print()
    print("Command:")
    print("  aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128")
    print()
    print("This demonstration will run in dry-run mode to show what would execute.")
    print()
    
    # Change to repo root
    repo_root = Path(__file__).parent.parent.parent
    
    # Run with dry-run to show what would happen
    cmd = [
        "./aimedres", "train",
        "--parallel",
        "--max-workers", "6",
        "--epochs", "50",
        "--folds", "5",
        "--batch", "128",
        "--dry-run"
    ]
    
    print("=" * 80)
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=repo_root)
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
        print("  ✓ Modern aimedres CLI interface")
        print("  ✓ Parallel execution enabled")
        print("  ✓ Up to 6 concurrent workers")
        print("  ✓ 50 epochs for neural network training")
        print("  ✓ 5-fold cross-validation")
        print("  ✓ Batch size of 128")
        print("  ✓ Automatic job discovery")
        print("  ✓ Parameter propagation to all compatible scripts")
        print()
        print("Usage Examples:")
        print()
        print("1. Run all training jobs with specified parameters:")
        print("   aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128")
        print()
        print("2. Run specific jobs only:")
        print("   aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128 \\")
        print("                  --only als alzheimers parkinsons")
        print()
        print("3. List all available training jobs:")
        print("   aimedres train --list")
        print()
        print("4. Run in dry-run mode to preview commands:")
        print("   aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128 \\")
        print("                  --dry-run")
        print()
        print("5. Sequential execution (no parallelism):")
        print("   aimedres train --epochs 50 --folds 5 --batch 128")
        print()
        print("Backward Compatibility:")
        print("  The old command format is still supported:")
        print("  python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128")
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
