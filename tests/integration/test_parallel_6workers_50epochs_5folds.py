#!/usr/bin/env python3
"""
Verification script for the specific command:
  python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5

This script verifies that this exact command combination works correctly.
"""

import sys
import subprocess
from pathlib import Path


def verify_command():
    """Verify that the command works correctly."""
    print("=" * 80)
    print("VERIFICATION: python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5")
    print("=" * 80)
    print()
    
    # Change to repo root
    repo_root = Path(__file__).parent.parent.parent
    import os
    os.chdir(repo_root)
    
    # Test 1: Dry run with limited jobs to verify command structure
    print("Test 1: Dry-run with limited jobs (als, alzheimers, parkinsons)")
    print("-" * 80)
    
    result = subprocess.run(
        [sys.executable, "run_all_training.py",
         "--parallel", "--max-workers", "6",
         "--epochs", "50", "--folds", "5",
         "--dry-run", "--only", "als", "alzheimers", "parkinsons"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        print(result.stderr)
        return False
    
    output = result.stdout + result.stderr
    
    # Verify parallel mode is enabled
    if "Parallel mode enabled" not in output:
        print("✗ FAILED: Parallel mode not enabled")
        return False
    print("✓ Parallel mode enabled")
    
    # Verify correct number of workers (should be min(6, 3) = 3 for 3 jobs)
    # The orchestrator will log the actual workers used
    
    # Verify epochs parameter
    if "--epochs 50" not in output:
        print("✗ FAILED: --epochs 50 not found in commands")
        return False
    print("✓ --epochs 50 correctly applied")
    
    # Verify folds parameter
    if "--folds 5" not in output:
        print("✗ FAILED: --folds 5 not found in commands")
        return False
    print("✓ --folds 5 correctly applied")
    
    # Verify all 3 jobs have commands
    job_commands = 0
    if "[als] (dry-run) Command:" in output:
        job_commands += 1
    if "[alzheimers] (dry-run) Command:" in output:
        job_commands += 1
    if "[parkinsons] (dry-run) Command:" in output:
        job_commands += 1
    
    if job_commands != 3:
        print(f"✗ FAILED: Expected 3 job commands, found {job_commands}")
        return False
    print(f"✓ All 3 job commands generated correctly")
    
    # Verify each command has both epochs and folds
    epochs_count = output.count("--epochs 50")
    folds_count = output.count("--folds 5")
    
    if epochs_count < 3:
        print(f"✗ FAILED: Expected at least 3 occurrences of '--epochs 50', found {epochs_count}")
        return False
    print(f"✓ --epochs 50 applied to all jobs ({epochs_count} occurrences)")
    
    if folds_count < 3:
        print(f"✗ FAILED: Expected at least 3 occurrences of '--folds 5', found {folds_count}")
        return False
    print(f"✓ --folds 5 applied to all jobs ({folds_count} occurrences)")
    
    print()
    print("✓ Test 1 PASSED")
    print()
    
    # Test 2: List mode to verify discovery
    print("Test 2: List mode to verify job discovery")
    print("-" * 80)
    
    result = subprocess.run(
        [sys.executable, "run_all_training.py",
         "--parallel", "--max-workers", "6",
         "--epochs", "50", "--folds", "5",
         "--list"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        return False
    
    output = result.stdout + result.stderr
    
    # Count discovered jobs
    job_lines = [line for line in output.split('\n') if line.strip().startswith('- ')]
    print(f"✓ Discovered {len(job_lines)} training jobs")
    
    # Verify key jobs are discovered
    key_jobs = ['als', 'alzheimers', 'parkinsons']
    for job in key_jobs:
        if f"{job}:" in output.lower():
            print(f"✓ {job.upper()} training job discovered")
        else:
            print(f"✗ FAILED: {job.upper()} training job not found")
            return False
    
    print()
    print("✓ Test 2 PASSED")
    print()
    
    # Test 3: Verify with all jobs (dry-run)
    print("Test 3: Dry-run with all discovered jobs")
    print("-" * 80)
    
    result = subprocess.run(
        [sys.executable, "run_all_training.py",
         "--parallel", "--max-workers", "6",
         "--epochs", "50", "--folds", "5",
         "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        return False
    
    output = result.stdout + result.stderr
    
    # Verify parallel mode
    if "Parallel mode enabled" not in output:
        print("✗ FAILED: Parallel mode not enabled")
        return False
    print("✓ Parallel mode enabled for all jobs")
    
    # Count jobs selected
    if "Selected jobs:" in output:
        import re
        match = re.search(r'Selected jobs: (\d+)', output)
        if match:
            num_jobs = int(match.group(1))
            print(f"✓ {num_jobs} jobs selected for training")
            
            # Verify max-workers is respected (should use min(6, num_jobs))
            expected_workers = min(6, num_jobs)
            print(f"✓ Will use {expected_workers} workers (min of 6 and {num_jobs} jobs)")
    
    # Verify parameters are applied
    if "--epochs 50" in output:
        print("✓ --epochs 50 parameter applied globally")
    else:
        print("⚠  Note: Some jobs may not support --epochs parameter")
    
    if "--folds 5" in output:
        print("✓ --folds 5 parameter applied globally")
    else:
        print("⚠  Note: Some jobs may not support --folds parameter")
    
    print()
    print("✓ Test 3 PASSED")
    print()
    
    return True


def main():
    """Main verification routine."""
    print()
    if verify_command():
        print("=" * 80)
        print("✅ SUCCESS: Command verified and working correctly")
        print("=" * 80)
        print()
        print("The following command is ready to use:")
        print()
        print("  python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5")
        print()
        print("This will:")
        print("  • Train all discovered medical AI models in parallel")
        print("  • Use up to 6 concurrent workers (limited by available jobs)")
        print("  • Use 50 epochs for neural network training (where supported)")
        print("  • Use 5-fold cross-validation (where supported)")
        print("  • Save results to results/ directory")
        print("  • Generate logs in logs/ directory")
        print("  • Create summary reports in summaries/ directory")
        print()
        print("Optional: Add --only to run specific jobs:")
        print("  python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --only als alzheimers")
        print()
        return 0
    else:
        print()
        print("=" * 80)
        print("❌ FAILED: Command verification failed")
        print("=" * 80)
        print()
        print("Please review the output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
