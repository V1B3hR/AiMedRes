#!/usr/bin/env python3
"""
Integration test for the aimedres CLI train command.

This test verifies that the command:
  aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128
works correctly.
"""

import sys
import subprocess
from pathlib import Path


def test_aimedres_train_basic():
    """Test basic aimedres train command with dry-run."""
    print("=" * 80)
    print("Test 1: Basic aimedres train with all parameters")
    print("=" * 80)
    
    result = subprocess.run(
        ["./aimedres", "train", 
         "--parallel", "--max-workers", "6",
         "--epochs", "50", "--folds", "5", "--batch", "128",
         "--dry-run", "--only", "als"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=Path(__file__).parent.parent.parent
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    output = result.stdout + result.stderr
    
    # Note: Parallel mode message only appears when len(jobs) > 1
    # For single job, the parallel flag is accepted but not shown
    print("✓ Command accepted --parallel and --max-workers flags")
    
    # Verify all parameters are in the command
    if "--epochs 50" not in output:
        print("✗ FAILED: --epochs 50 not in command")
        return False
    print("✓ --epochs 50 parameter applied")
    
    if "--folds 5" not in output:
        print("✗ FAILED: --folds 5 not in command")
        return False
    print("✓ --folds 5 parameter applied")
    
    if "--batch-size 128" not in output:
        print("✗ FAILED: --batch-size 128 not in command")
        return False
    print("✓ --batch 128 parameter applied (as --batch-size)")
    
    print("✓ Test 1 PASSED")
    return True


def test_aimedres_train_help():
    """Test that aimedres train --help works."""
    print()
    print("=" * 80)
    print("Test 2: aimedres train --help")
    print("=" * 80)
    
    result = subprocess.run(
        ["./aimedres", "train", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).parent.parent.parent
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        return False
    
    output = result.stdout + result.stderr
    
    # Verify help text contains all expected parameters
    required_params = ["--parallel", "--max-workers", "--epochs", "--folds", "--batch"]
    for param in required_params:
        if param not in output:
            print(f"✗ FAILED: {param} not in help text")
            return False
        print(f"✓ {param} in help text")
    
    print("✓ Test 2 PASSED")
    return True


def test_aimedres_train_list():
    """Test that aimedres train --list works with parameters."""
    print()
    print("=" * 80)
    print("Test 3: aimedres train --list with parameters")
    print("=" * 80)
    
    result = subprocess.run(
        ["./aimedres", "train",
         "--parallel", "--max-workers", "6",
         "--epochs", "50", "--folds", "5", "--batch", "128",
         "--list"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=Path(__file__).parent.parent.parent
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        return False
    
    output = result.stdout + result.stderr
    
    # Verify key jobs are listed
    key_jobs = ['als', 'alzheimers', 'parkinsons']
    for job in key_jobs:
        if f"- {job}:" in output.lower():
            print(f"✓ {job.upper()} job discovered")
        else:
            print(f"✗ FAILED: {job.upper()} job not found")
            return False
    
    print("✓ Test 3 PASSED")
    return True


def test_aimedres_train_multiple_jobs():
    """Test aimedres train with multiple jobs in parallel."""
    print()
    print("=" * 80)
    print("Test 4: aimedres train with multiple jobs in parallel")
    print("=" * 80)
    
    result = subprocess.run(
        ["./aimedres", "train",
         "--parallel", "--max-workers", "6",
         "--epochs", "50", "--folds", "5", "--batch", "128",
         "--dry-run", "--only", "als", "alzheimers", "parkinsons"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=Path(__file__).parent.parent.parent
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        return False
    
    output = result.stdout + result.stderr
    
    # Count job commands
    job_commands = 0
    for job in ["als", "alzheimers", "parkinsons"]:
        if f"[{job}] (dry-run) Command:" in output:
            job_commands += 1
            print(f"✓ {job.upper()} job command generated")
    
    if job_commands != 3:
        print(f"✗ FAILED: Expected 3 job commands, got {job_commands}")
        return False
    
    # Verify epochs parameter is applied to all compatible jobs
    epochs_count = output.count("--epochs 50")
    if epochs_count < 3:
        print(f"✗ FAILED: --epochs 50 should appear at least 3 times, found {epochs_count}")
        return False
    print(f"✓ --epochs 50 applied to all compatible jobs ({epochs_count} times)")
    
    print("✓ Test 4 PASSED")
    return True


def main():
    """Run all tests."""
    print()
    print("=" * 80)
    print("INTEGRATION TEST: aimedres train CLI command")
    print("=" * 80)
    print()
    
    tests = [
        test_aimedres_train_help,
        test_aimedres_train_basic,
        test_aimedres_train_list,
        test_aimedres_train_multiple_jobs,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test raised exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print()
        print("✅ SUCCESS: All tests passed")
        print()
        print("The aimedres CLI train command is working correctly:")
        print("  aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128")
        print()
        return 0
    else:
        print()
        print("❌ FAILED: Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
