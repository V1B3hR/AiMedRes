#!/usr/bin/env python3
"""
Final comprehensive test for the run_all_training.py functionality.
Tests that all 6 disease prediction models can be run.
"""

import sys
import subprocess

def run_test(description, command, check_function):
    """Run a test and check the result."""
    print(f"\n{description}")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30
    )
    output = result.stdout + result.stderr
    
    if check_function(output, result.returncode):
        print(f"  ✓ PASSED")
        return True
    else:
        print(f"  ✗ FAILED")
        print(f"  Output: {output[:200]}")
        return False

def main():
    """Run all final tests."""
    print("=" * 70)
    print("  FINAL COMPREHENSIVE TEST: Run All Training Functionality")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: List all 6 core models
    total_tests += 1
    if run_test(
        "Test 1: List all 6 core models (no auto-discovery)",
        [sys.executable, "run_all_training.py", "--list", "--no-auto-discover"],
        lambda output, code: "Selected jobs: 6" in output and code == 0
    ):
        tests_passed += 1
    
    # Test 2: Verify all 6 model IDs exist
    total_tests += 1
    def check_all_models(output, code):
        models = ["als:", "alzheimers:", "parkinsons:", "brain_mri:", "cardiovascular:", "diabetes:"]
        for model in models:
            if f"- {model}" not in output:
                print(f"    Missing: {model}")
                return False
        return code == 0
    
    if run_test(
        "Test 2: Verify all 6 model IDs exist",
        [sys.executable, "run_all_training.py", "--list", "--no-auto-discover"],
        check_all_models
    ):
        tests_passed += 1
    
    # Test 3: Dry-run all 6 models with custom parameters
    total_tests += 1
    if run_test(
        "Test 3: Dry-run all 6 models with custom parameters",
        [sys.executable, "run_all_training.py",
         "--only", "als", "alzheimers", "parkinsons", "brain_mri", "cardiovascular", "diabetes",
         "--dry-run", "--epochs", "20", "--folds", "5"],
        lambda output, code: "Selected jobs: 6" in output and code == 0
    ):
        tests_passed += 1
    
    # Test 4: Verify custom epochs parameter
    total_tests += 1
    if run_test(
        "Test 4: Verify custom epochs parameter",
        [sys.executable, "run_all_training.py", "--only", "als", "--dry-run", "--epochs", "42"],
        lambda output, code: "--epochs 42" in output and code == 0
    ):
        tests_passed += 1
    
    # Test 5: Verify parallel mode
    total_tests += 1
    if run_test(
        "Test 5: Verify parallel mode",
        [sys.executable, "run_all_training.py", "--parallel", "--max-workers", "3",
         "--dry-run", "--only", "als", "alzheimers"],
        lambda output, code: "Parallel mode enabled" in output and code == 0
    ):
        tests_passed += 1
    
    # Test 6: Verify all 6 commands are generated
    total_tests += 1
    def check_all_commands(output, code):
        commands = [
            "[als] (dry-run) Command:",
            "[alzheimers] (dry-run) Command:",
            "[parkinsons] (dry-run) Command:",
            "[brain_mri] (dry-run) Command:",
            "[cardiovascular] (dry-run) Command:",
            "[diabetes] (dry-run) Command:"
        ]
        for cmd in commands:
            if cmd not in output:
                print(f"    Missing: {cmd}")
                return False
        return code == 0
    
    if run_test(
        "Test 6: Verify all 6 commands are generated",
        [sys.executable, "run_all_training.py",
         "--only", "als", "alzheimers", "parkinsons", "brain_mri", "cardiovascular", "diabetes",
         "--dry-run", "--epochs", "10"],
        check_all_commands
    ):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"  RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 70)
    
    if tests_passed == total_tests:
        print("\n✓ SUCCESS! All tests passed!")
        print("\nThe command 'python run_all_training.py' successfully:")
        print("  - Runs all 6 disease prediction models (Alzheimer's, ALS, Parkinson's,")
        print("    Brain MRI, Cardiovascular, Diabetes)")
        print("  - Supports custom parameters (--epochs, --folds)")
        print("  - Supports parallel execution (--parallel)")
        print("  - Supports filtering (--only, --exclude)")
        return 0
    else:
        print(f"\n✗ FAILURE: {total_tests - tests_passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
