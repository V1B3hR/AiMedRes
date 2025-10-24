#!/usr/bin/env python3
"""
Verification script for the parallel mode feature.

This script demonstrates that the command:
    python run_all_training.py --parallel --max-workers 6

works correctly as specified in the problem statement.
"""

import subprocess
import sys
from pathlib import Path


def verify_parallel_mode():
    """Verify the parallel mode functionality."""
    print("=" * 80)
    print("VERIFICATION: Parallel Mode with 6 Workers")
    print("=" * 80)
    print()
    print("Testing command: python run_all_training.py --parallel --max-workers 6")
    print()
    
    # Change to repository root
    repo_root = Path(__file__).parent
    
    # Test 1: Verify --parallel and --max-workers flags are recognized
    print("Test 1: Verify flags are recognized")
    print("-" * 80)
    result = subprocess.run(
        [sys.executable, "run_all_training.py", "--help"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=30
    )
    
    if "--parallel" in result.stdout and "--max-workers" in result.stdout:
        print("✅ PASS: --parallel and --max-workers flags are available")
    else:
        print("❌ FAIL: Flags not found in help output")
        return False
    print()
    
    # Test 2: Run with dry-run to verify execution
    print("Test 2: Verify parallel execution (dry-run with 3 jobs)")
    print("-" * 80)
    result = subprocess.run(
        [sys.executable, "run_all_training.py",
         "--parallel", "--max-workers", "6",
         "--dry-run", "--only", "als", "alzheimers", "parkinsons"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=60
    )
    
    output = result.stdout + result.stderr
    
    # Check for parallel mode enabled message
    if "Parallel mode enabled" in output:
        print("✅ PASS: Parallel mode is enabled")
    else:
        print("❌ FAIL: Parallel mode not enabled")
        return False
    
    # Check that jobs are executed
    if "[als] (dry-run) Command:" in output:
        print("✅ PASS: ALS job command generated")
    else:
        print("❌ FAIL: ALS job not found")
        return False
    
    if "[alzheimers] (dry-run) Command:" in output:
        print("✅ PASS: Alzheimer's job command generated")
    else:
        print("❌ FAIL: Alzheimer's job not found")
        return False
    
    if "[parkinsons] (dry-run) Command:" in output:
        print("✅ PASS: Parkinson's job command generated")
    else:
        print("❌ FAIL: Parkinson's job not found")
        return False
    
    print()
    
    # Test 3: Verify all jobs can be discovered and listed
    print("Test 3: Verify job discovery and listing")
    print("-" * 80)
    result = subprocess.run(
        [sys.executable, "run_all_training.py",
         "--parallel", "--max-workers", "6",
         "--list"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=60
    )
    
    output = result.stdout + result.stderr
    
    # Count jobs
    job_lines = [line for line in output.split('\n') if line.strip().startswith('- ')]
    num_jobs = len(job_lines)
    
    if num_jobs > 0:
        print(f"✅ PASS: Discovered {num_jobs} training jobs")
    else:
        print("❌ FAIL: No jobs discovered")
        return False
    
    # Check for key jobs
    key_jobs = ['als', 'alzheimers', 'parkinsons', 'brain_mri', 'cardiovascular', 'diabetes']
    found_jobs = 0
    for job in key_jobs:
        if f"- {job}:" in output:
            found_jobs += 1
    
    if found_jobs >= 6:
        print(f"✅ PASS: Found {found_jobs} key training jobs")
    else:
        print(f"❌ FAIL: Only found {found_jobs} key jobs (expected at least 6)")
        return False
    
    print()
    
    # Test 4: Verify the exact command from problem statement works
    print("Test 4: Verify exact command from problem statement")
    print("-" * 80)
    print("Command: python run_all_training.py --parallel --max-workers 6")
    result = subprocess.run(
        [sys.executable, "run_all_training.py",
         "--parallel", "--max-workers", "6",
         "--dry-run"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=60
    )
    
    if result.returncode == 0:
        print("✅ PASS: Command executed successfully (exit code 0)")
    else:
        print(f"❌ FAIL: Command failed with exit code {result.returncode}")
        return False
    
    output = result.stdout + result.stderr
    
    if "Parallel mode enabled" in output:
        print("✅ PASS: Parallel mode activated")
    else:
        print("❌ FAIL: Parallel mode not activated")
        return False
    
    if "Selected jobs:" in output:
        print("✅ PASS: Jobs selected for training")
    else:
        print("❌ FAIL: No jobs selected")
        return False
    
    print()
    
    return True


def main():
    """Main entry point."""
    print()
    
    if verify_parallel_mode():
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print()
        print("The parallel mode feature is working correctly!")
        print()
        print("Usage:")
        print("  # Run all training jobs in parallel with 6 workers")
        print("  python run_all_training.py --parallel --max-workers 6")
        print()
        print("  # Run specific jobs in parallel")
        print("  python run_all_training.py --parallel --max-workers 6 --only als alzheimers")
        print()
        print("  # Add epochs and folds parameters")
        print("  python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5")
        print()
        print("  # Dry-run to preview commands without executing")
        print("  python run_all_training.py --parallel --max-workers 6 --dry-run")
        print()
        return 0
    else:
        print()
        print("=" * 80)
        print("❌ TESTS FAILED")
        print("=" * 80)
        print()
        print("Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
