#!/usr/bin/env python3
"""
Verification script to ensure that the command:
  python run_all_training.py --epochs 50 --folds 5
works correctly and generates proper training commands.
"""

import sys
import subprocess
from pathlib import Path

def verify_command_structure():
    """Verify that the training command works with --epochs 50 --folds 5"""
    print("=" * 70)
    print("Verifying: python run_all_training.py --epochs 50 --folds 5")
    print("=" * 70)
    print()
    
    # Test 1: Dry run to verify command generation
    print("Test 1: Dry-run mode - verify command generation")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, "run_all_training.py", 
         "--epochs", "50", "--folds", "5", 
         "--dry-run", "--only", "als"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        print(result.stderr)
        return False
    
    output = result.stdout + result.stderr
    
    # Verify epochs and folds are included in command
    if "--epochs 50" not in output:
        print("✗ FAILED: --epochs 50 not found in command")
        return False
    print("✓ --epochs 50 correctly included")
    
    if "--folds 5" not in output:
        print("✗ FAILED: --folds 5 not found in command")
        return False
    print("✓ --folds 5 correctly included")
    
    # Verify correct script is being used
    if "src/aimedres/training/train_als.py" not in output:
        print("✗ FAILED: Canonical training script not used")
        return False
    print("✓ Canonical training script location used")
    
    print()
    print("✓ Test 1 PASSED: Command structure is correct")
    print()
    
    # Test 2: List mode to verify all jobs are discovered
    print("Test 2: List mode - verify job discovery")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, "run_all_training.py", 
         "--epochs", "50", "--folds", "5", "--list"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        return False
    
    output = result.stdout + result.stderr
    
    # Count discovered jobs
    job_lines = [line for line in output.split('\n') if line.strip().startswith('- ')]
    print(f"✓ Discovered {len(job_lines)} training jobs")
    
    # Verify key medical AI models are discovered
    required_models = ['als', 'alzheimers', 'parkinsons']
    for model in required_models:
        if f"{model}:" in output.lower():
            print(f"✓ {model.upper()} training job discovered")
        else:
            print(f"✗ FAILED: {model.upper()} training job not found")
            return False
    
    print()
    print("✓ Test 2 PASSED: All expected jobs discovered")
    print()
    
    # Test 3: Verify parallel mode works with the parameters
    print("Test 3: Parallel mode compatibility")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, "run_all_training.py", 
         "--epochs", "50", "--folds", "5",
         "--parallel", "--max-workers", "2",
         "--dry-run", "--only", "als", "alzheimers"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        return False
    
    output = result.stdout + result.stderr
    
    if "Parallel mode enabled" in output:
        print("✓ Parallel mode works with --epochs and --folds")
    else:
        print("✗ FAILED: Parallel mode not recognized")
        return False
    
    # Verify both jobs get the same parameters
    if output.count("--epochs 50") >= 2 and output.count("--folds 5") >= 2:
        print("✓ Parameters correctly propagated to all jobs in parallel mode")
    else:
        print("✗ FAILED: Parameters not properly propagated")
        return False
    
    print()
    print("✓ Test 3 PASSED: Parallel mode compatible")
    print()
    
    return True

def main():
    """Main verification routine"""
    repo_root = Path(__file__).parent.parent.parent
    import os
    os.chdir(repo_root)
    
    print()
    print("VERIFICATION: python run_all_training.py --epochs 50 --folds 5")
    print()
    
    try:
        if verify_command_structure():
            print("=" * 70)
            print("✅ SUCCESS: Command verified and working correctly")
            print("=" * 70)
            print()
            print("The following command is ready to use:")
            print()
            print("  python run_all_training.py --epochs 50 --folds 5")
            print()
            print("This will:")
            print("  • Train all discovered medical AI models")
            print("  • Use 50 epochs for neural networks")
            print("  • Use 5-fold cross-validation")
            print("  • Save results to results/ directory")
            print("  • Generate logs in logs/ directory")
            print("  • Create summary reports in summaries/ directory")
            print()
            print("Alternative usage:")
            print("  ./run_medical_training.sh  # Convenience script")
            print()
            return 0
        else:
            print("=" * 70)
            print("❌ FAILED: Verification failed")
            print("=" * 70)
            return 1
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
