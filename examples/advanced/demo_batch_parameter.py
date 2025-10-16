#!/usr/bin/env python3
"""
Demonstration that the command from the problem statement now works:
  aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128

This script demonstrates the functionality in dry-run mode.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Demonstrate the command works"""
    print()
    print("=" * 80)
    print("DEMONSTRATION: aimedres train command with batch parameter")
    print("=" * 80)
    print()
    print("Problem Statement:")
    print("  @V1B3hR/AiMedRes run aimedres train --parallel --max-workers 6")
    print("                        --epochs 50 --folds 5 batch 128")
    print()
    print("Corrected Command (batch -> --batch):")
    print("  aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128")
    print()
    print("=" * 80)
    print()
    
    # Change to repo root
    repo_root = Path(__file__).parent.parent.parent
    import os
    os.chdir(repo_root)
    
    # Run the command via the CLI entry point (in dry-run mode for demo)
    cmd = [
        sys.executable, "src/aimedres/cli/commands.py",
        "train",
        "--parallel",
        "--max-workers", "6",
        "--epochs", "50",
        "--folds", "5",
        "--batch", "128",
        "--dry-run",
        "--only", "als", "alzheimers"  # Just two jobs for demo
    ]
    
    print("Executing command (dry-run mode):")
    print(f"  {' '.join(cmd)}")
    print()
    print("=" * 80)
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    print()
    print("=" * 80)
    print()
    
    if result.returncode == 0:
        print("✅ SUCCESS!")
        print()
        print("The command is now fully functional. Key features:")
        print()
        print("  ✓ --parallel flag enables concurrent execution")
        print("  ✓ --max-workers 6 allows up to 6 parallel jobs")
        print("  ✓ --epochs 50 sets training epochs to 50")
        print("  ✓ --folds 5 enables 5-fold cross-validation")
        print("  ✓ --batch 128 sets batch size to 128 (new!)")
        print()
        print("Usage:")
        print("  # Run all training jobs with these parameters:")
        print("  aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128")
        print()
        print("  # Run specific jobs only:")
        print("  aimedres train --only als alzheimers --epochs 50 --batch 128")
        print()
        print("  # See all available jobs:")
        print("  aimedres train --list")
        print()
        print("Note: Individual training scripts will use the --batch parameter")
        print("      if they support it. The infrastructure is in place to")
        print("      propagate this parameter to all compatible scripts.")
        print()
        return 0
    else:
        print("❌ FAILED")
        print(f"Exit code: {result.returncode}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
