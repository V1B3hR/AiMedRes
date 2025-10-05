#!/usr/bin/env python3
"""
Test script to verify the training orchestrator functionality.
This tests the "Run training for all" requirement.
"""

import sys
import subprocess
from pathlib import Path

def test_orchestrator_list():
    """Test that the orchestrator can list all discovered jobs"""
    print("Test 1: List all training jobs...")
    result = subprocess.run(
        [sys.executable, "run_all_training.py", "--list"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ Failed with code {result.returncode}")
        print(result.stderr)
        return False
    
    # Check both stdout and stderr (logging may go to either)
    output = result.stdout + result.stderr
    if "script=src/aimedres/training/train_als.py" in output:
        print("✓ Found ALS training script (canonical location)")
    else:
        print("✗ ALS training script not found in canonical location")
        return False
    
    if "script=src/aimedres/training/train_alzheimers.py" in output:
        print("✓ Found Alzheimer's training script (canonical location)")
    else:
        print("✗ Alzheimer's training script not found in canonical location")
        return False
    
    # Check that legacy scripts are NOT discovered
    if "script=training/train_alzheimers.py" in output or "script=files/training/train_alzheimers.py" in output:
        print("✗ Legacy training scripts should not be discovered")
        return False
    else:
        print("✓ Legacy scripts correctly skipped")
    
    # Check for proper flag detection
    if "epochs=True folds=True outdir=True" in output:
        print("✓ Flags properly detected for canonical scripts")
    else:
        print("✗ Flags not properly detected")
        return False
    
    print("✓ Test passed: Orchestrator can list all training jobs\n")
    return True


def test_orchestrator_dry_run():
    """Test that the orchestrator generates correct commands"""
    print("Test 2: Dry run with parameters...")
    result = subprocess.run(
        [sys.executable, "run_all_training.py", "--dry-run", 
         "--epochs", "10", "--folds", "3", "--only", "als"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ Failed with code {result.returncode}")
        print(result.stderr)
        return False
    
    output = result.stdout + result.stderr
    
    # Check command includes correct parameters
    if "--epochs 10" in output and "--folds 3" in output:
        print("✓ Command includes specified epochs and folds")
    else:
        print("✗ Command missing specified parameters")
        return False
    
    # Check command uses canonical script
    if "src/aimedres/training/train_als.py" in output:
        print("✓ Command uses canonical script location")
    else:
        print("✗ Command not using canonical script location")
        return False
    
    # Check output directory is included
    if "--output-dir" in output:
        print("✓ Command includes output directory")
    else:
        print("✗ Command missing output directory")
        return False
    
    print("✓ Test passed: Orchestrator generates correct commands\n")
    return True


def test_orchestrator_parallel():
    """Test that parallel mode is recognized"""
    print("Test 3: Parallel execution mode...")
    result = subprocess.run(
        [sys.executable, "run_all_training.py", "--dry-run", 
         "--parallel", "--max-workers", "2", "--only", "als", "alzheimers"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ Failed with code {result.returncode}")
        print(result.stderr)
        return False
    
    output = result.stdout + result.stderr
    
    if "Parallel mode enabled" in output:
        print("✓ Parallel mode recognized")
    else:
        print("✗ Parallel mode not recognized")
        return False
    
    print("✓ Test passed: Parallel mode works\n")
    return True


def test_orchestrator_filtering():
    """Test that job filtering works"""
    print("Test 4: Job filtering...")
    
    # Test --only filter
    result = subprocess.run(
        [sys.executable, "run_all_training.py", "--list", "--only", "als", "alzheimers"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ Failed with code {result.returncode}")
        return False
    
    output = result.stdout + result.stderr
    
    # Should include als and alzheimers
    if "als:" in output.lower() and "alzheimers:" in output.lower():
        print("✓ --only filter includes specified jobs")
    else:
        print("✗ --only filter not working correctly")
        return False
    
    # Should not include others like diabetes or parkinsons (unless they're in the filter)
    # Count the number of job lines (they start with "- ")
    job_lines = [line for line in output.split('\n') if line.strip().startswith('- ')]
    if len(job_lines) <= 4:  # Should be around 2-3 jobs (als, alzheimers, maybe duplicates)
        print(f"✓ Filter reduced jobs to {len(job_lines)} (expected ~2-3)")
    else:
        print(f"✗ Too many jobs after filter: {len(job_lines)}")
        return False
    
    print("✓ Test passed: Job filtering works\n")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Training Orchestrator - 'Run training for all'")
    print("=" * 60)
    print()
    
    # Change to repository root
    repo_root = Path(__file__).parent
    import os
    os.chdir(repo_root)
    
    tests = [
        test_orchestrator_list,
        test_orchestrator_dry_run,
        test_orchestrator_parallel,
        test_orchestrator_filtering,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}\n")
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✓ All tests passed! Training orchestrator is ready.")
        print("\nThe 'Run training for all' functionality is working:")
        print("  - Discovers all training scripts from src/aimedres/training/")
        print("  - Skips legacy duplicate scripts")
        print("  - Properly detects command-line flags")
        print("  - Generates correct commands with parameters")
        print("  - Supports parallel execution")
        print("  - Supports filtering and selection")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
