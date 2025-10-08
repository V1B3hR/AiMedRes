#!/usr/bin/env python3
"""
Test script to verify the --sample parameter support in run_all_training.py
"""

import sys
import subprocess
from pathlib import Path


def test_sample_parameter_in_help():
    """Test that --sample parameter appears in help"""
    print("Test 1: Check --sample parameter in help...")
    result = subprocess.run(
        [sys.executable, "run_all_training.py", "--help"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ Failed with code {result.returncode}")
        print(result.stderr)
        return False
    
    output = result.stdout
    
    if "--sample SAMPLE" in output:
        print("✓ --sample parameter found in help")
    else:
        print("✗ --sample parameter not found in help")
        return False
    
    if "Global default sample size (if supported)" in output:
        print("✓ Help text for --sample is correct")
    else:
        print("✗ Help text for --sample is missing or incorrect")
        return False
    
    print("✓ Test passed: --sample parameter in help\n")
    return True


def test_sample_parameter_accepted():
    """Test that --sample parameter is accepted by the argument parser"""
    print("Test 2: Check --sample parameter is accepted...")
    result = subprocess.run(
        [sys.executable, "run_all_training.py", "--dry-run", 
         "--sample", "3000", "--only", "als"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ Failed with code {result.returncode}")
        print(result.stderr)
        return False
    
    output = result.stdout + result.stderr
    
    # The command should execute without error
    print("✓ Command accepted --sample parameter without error")
    
    print("✓ Test passed: --sample parameter accepted\n")
    return True


def test_full_command_from_problem_statement():
    """Test the exact command from the problem statement"""
    print("Test 3: Test full command from problem statement...")
    result = subprocess.run(
        [sys.executable, "run_all_training.py", 
         "--parallel", "--max-workers", "6", 
         "--epochs", "70", "--folds", "5", "--sample", "3000",
         "--dry-run", "--only", "als", "alzheimers"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ Failed with code {result.returncode}")
        print(result.stderr)
        return False
    
    output = result.stdout + result.stderr
    
    # Check parallel mode is enabled
    if "Parallel mode enabled" in output:
        print("✓ Parallel mode enabled")
    else:
        print("✗ Parallel mode not enabled")
        return False
    
    # Check epochs and folds are in the commands
    if "--epochs 70" in output:
        print("✓ Epochs parameter (70) applied")
    else:
        print("✗ Epochs parameter not found")
        return False
    
    if "--folds 5" in output:
        print("✓ Folds parameter (5) applied")
    else:
        print("✗ Folds parameter not found")
        return False
    
    # Check that commands are generated
    if "[als] (dry-run) Command:" in output:
        print("✓ ALS job command generated")
    else:
        print("✗ ALS job command not generated")
        return False
    
    if "[alzheimers] (dry-run) Command:" in output:
        print("✓ Alzheimer's job command generated")
    else:
        print("✗ Alzheimer's job command not generated")
        return False
    
    print("✓ Test passed: Full command executes successfully\n")
    return True


def test_sample_flag_detection():
    """Test that the sample flag is properly detected in job listings"""
    print("Test 4: Check sample flag in job listings...")
    result = subprocess.run(
        [sys.executable, "run_all_training.py", "--list", "--only", "als"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ Failed with code {result.returncode}")
        print(result.stderr)
        return False
    
    output = result.stdout + result.stderr
    
    # Check that sample flag is shown in the output
    if "sample=" in output:
        print("✓ Sample flag shown in job listing")
    else:
        print("✗ Sample flag not shown in job listing")
        return False
    
    print("✓ Test passed: Sample flag detection works\n")
    return True


def main():
    """Run all tests"""
    print("=" * 80)
    print("Testing --sample Parameter Support")
    print("=" * 80)
    print()
    
    # Change to repository root
    repo_root = Path(__file__).parent
    import os
    os.chdir(repo_root)
    
    tests = [
        test_sample_parameter_in_help,
        test_sample_parameter_accepted,
        test_full_command_from_problem_statement,
        test_sample_flag_detection,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}\n")
    
    print("=" * 80)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("\n✓ All tests passed! --sample parameter is working correctly.")
        print("\nThe command from the problem statement now works:")
        print("  python run_all_training.py --parallel --max-workers 6 --epochs 70 --folds 5 --sample 3000")
        print("\nNote:")
        print("  - The orchestrator accepts and processes the --sample parameter")
        print("  - It will be passed to training scripts that support it")
        print("  - Currently, default training scripts don't support --sample")
        print("  - Use --extra-arg to force parameters to all scripts if needed")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
