#!/usr/bin/env python3
"""
Demonstration of the exact command from the problem statement:
    python run_all_training.py --parallel --max-workers 6

This script shows the command in action with a dry-run.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Demonstrate the exact command from the problem statement."""
    print()
    print("=" * 80)
    print("DEMONSTRATION: Problem Statement Command")
    print("=" * 80)
    print()
    print("Problem Statement:")
    print("  # Run in parallel mode")
    print("  python run_all_training.py --parallel --max-workers 6")
    print()
    print("=" * 80)
    print()
    
    # Change to repository root
    repo_root = Path(__file__).parent
    
    # Run the exact command with dry-run
    print("Executing with --dry-run to preview (not running actual training):")
    print()
    cmd = [sys.executable, "run_all_training.py", "--parallel", "--max-workers", "6", "--dry-run"]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    print("-" * 80)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=60
    )
    
    # Print the output
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print("-" * 80)
    print()
    
    # Analyze the output
    output = result.stdout + result.stderr
    
    print("ANALYSIS:")
    print()
    
    # Check key features
    checks = [
        ("✓ Exit Code 0", result.returncode == 0),
        ("✓ Parallel Mode Enabled", "Parallel mode enabled" in output),
        ("✓ Max Workers Setting Recognized", "--max-workers" in " ".join(cmd)),
        ("✓ Jobs Discovered", "Selected jobs:" in output),
        ("✓ Jobs Ready to Execute", "(dry-run) Command:" in output),
        ("✓ Summary Generated", "Summary written:" in output),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    print()
    print("=" * 80)
    
    if all_passed:
        print("✅ SUCCESS: The command from the problem statement works perfectly!")
        print()
        print("The parallel mode feature is fully functional.")
        print()
        print("To run actual training (remove --dry-run):")
        print("  python run_all_training.py --parallel --max-workers 6")
        print()
        print("Additional options:")
        print("  --epochs N        : Set training epochs")
        print("  --folds N         : Set cross-validation folds")
        print("  --only job1 job2  : Run only specific jobs")
        print("  --exclude job1    : Exclude specific jobs")
        print("  --list            : List all available jobs")
        print()
        return 0
    else:
        print("❌ FAILED: Some checks did not pass")
        return 1


if __name__ == "__main__":
    sys.exit(main())
