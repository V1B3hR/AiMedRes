#!/usr/bin/env python3
"""
Demo script to verify the parallel training command with custom parameters.
Specifically tests: python run_all_training.py --parallel --max-workers 6 --epochs 80 --folds 5 --dry-run
"""

import sys
import subprocess
from pathlib import Path


def run_command_verification():
    """Verify the command from the problem statement works correctly"""
    print("=" * 80)
    print("Verifying Command: python run_all_training.py --parallel --max-workers 6 --epochs 80 --folds 5 --dry-run")
    print("=" * 80)
    print()
    
    # The exact command from the problem statement
    cmd = [
        sys.executable,
        "run_all_training.py",
        "--parallel",
        "--max-workers", "6",
        "--epochs", "80",
        "--folds", "5",
        "--dry-run"
    ]
    
    print("Command to execute:")
    print(f"  {' '.join(cmd)}")
    print()
    print("Running command...")
    print("-" * 80)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print("-" * 80)
    print()
    
    # Verify expected behaviors
    output = result.stdout + result.stderr
    checks = {
        "Exit code is 0": result.returncode == 0,
        "Parallel mode enabled": "Parallel mode enabled" in output,
        "Epochs 80 in commands": "--epochs 80" in output,
        "Folds 5 in commands": "--folds 5" in output,
        "Dry run mode active": "(dry-run) Command:" in output,
        "Multiple jobs discovered": "Selected jobs:" in output,
    }
    
    print("Verification Checks:")
    print()
    all_passed = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False
    
    print()
    print("=" * 80)
    
    if all_passed:
        print("✅ SUCCESS: All verification checks passed!")
        print()
        print("The command works correctly with:")
        print("  - Parallel execution enabled")
        print("  - 6 max workers")
        print("  - 80 epochs for training")
        print("  - 5 cross-validation folds")
        print("  - Dry-run mode (preview only)")
        return 0
    else:
        print("❌ FAILURE: Some verification checks failed!")
        return 1


def main():
    """Main entry point"""
    # Change to repository root
    repo_root = Path(__file__).parent
    import os
    os.chdir(repo_root)
    
    return run_command_verification()


if __name__ == "__main__":
    sys.exit(main())
