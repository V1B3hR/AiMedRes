#!/usr/bin/env python3
"""
Verification script for core training orchestrator
Tests that full training can be run from /core directory
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def run_test(
    description: str, command: Sequence[str], check_output: Optional[Sequence[str]] = None
) -> bool:
    """Run a test command and verify output"""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"{'='*70}")
    print(f"Command: {subprocess.list2cmdline(list(command))}")
    print()

    result = subprocess.run(list(command), shell=False, capture_output=True, text=True, timeout=60)

    output = result.stdout + result.stderr
    print(output[:1000] if len(output) > 1000 else output)

    if result.returncode != 0:
        print(f"❌ FAILED: Command returned exit code {result.returncode}")
        return False

    if check_output:
        for expected in check_output:
            if expected not in output:
                print(f"❌ FAILED: Expected output '{expected}' not found")
                return False

    print("✅ PASSED")
    return True


def main():
    """Run all verification tests"""
    print("\n" + "=" * 70)
    print("  VERIFICATION: Core Training Orchestrator")
    print("=" * 70)
    print("\nThis script verifies that full training can be run from /core")

    core_dir = Path(__file__).resolve().parent

    # Change to core directory
    import os

    os.chdir(core_dir)
    print(f"\nWorking directory: {os.getcwd()}")

    tests = []

    # Test 1: List all 6 models
    tests.append(
        run_test(
            "List all 6 core models",
            [sys.executable, "run_all_training.py", "--list", "--no-auto-discover"],
            check_output=[
                "als: ALS",
                "alzheimers: Alzheimer",
                "parkinsons: Parkinson",
                "brain_mri: Brain MRI",
                "cardiovascular: Cardiovascular",
                "diabetes: Diabetes",
            ],
        )
    )

    # Test 2: Dry run with custom parameters
    tests.append(
        run_test(
            "Dry run with custom epochs and folds",
            [
                sys.executable,
                "run_all_training.py",
                "--dry-run",
                "--epochs",
                "50",
                "--folds",
                "5",
                "--only",
                "als",
                "--no-auto-discover",
            ],
            check_output=["dry-run", "--epochs 50", "--folds 5", "als_comprehensive_results"],
        )
    )

    # Test 3: Parallel mode
    tests.append(
        run_test(
            "Parallel mode with 6 workers",
            [
                sys.executable,
                "run_all_training.py",
                "--dry-run",
                "--parallel",
                "--max-workers",
                "6",
                "--no-auto-discover",
            ],
            check_output=["Parallel mode enabled", "als", "alzheimers", "parkinsons"],
        )
    )

    # Test 4: Filter specific models
    tests.append(
        run_test(
            "Filter to specific models only",
            [
                sys.executable,
                "run_all_training.py",
                "--dry-run",
                "--only",
                "als",
                "alzheimers",
                "--no-auto-discover",
            ],
            check_output=["Selected jobs: 2", "als", "alzheimers"],
        )
    )

    # Summary
    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)

    passed = sum(tests)
    total = len(tests)

    print(f"\n✅ Passed: {passed}/{total}")
    if passed < total:
        print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! Full training can be run from /core")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
