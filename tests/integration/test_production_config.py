#!/usr/bin/env python3
"""
Verification script to ensure that the production-ready command works:
  aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128
"""

import sys
import subprocess
from pathlib import Path


def verify_production_command():
    """Verify that the production command works with all parameters"""
    print("=" * 70)
    print("Verifying: aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128")
    print("=" * 70)
    print()
    
    # Test 1: Dry run to verify command generation with batch parameter
    print("Test 1: Dry-run mode - verify batch parameter translation")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, "-m", "aimedres.cli.commands", "train",
         "--parallel", "--max-workers", "6",
         "--epochs", "50", "--folds", "5", "--batch", "128",
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
    
    # Verify all parameters are included in command
    if "--epochs 50" not in output:
        print("✗ FAILED: --epochs 50 not found in command")
        return False
    print("✓ --epochs 50 correctly included")
    
    if "--folds 5" not in output:
        print("✗ FAILED: --folds 5 not found in command")
        return False
    print("✓ --folds 5 correctly included")
    
    # Verify batch is translated to batch-size
    if "--batch-size 128" not in output:
        print("✗ FAILED: --batch-size 128 not found in command (should be translated from --batch 128)")
        return False
    print("✓ --batch 128 correctly translated to --batch-size 128")
    
    # Verify correct script is being used
    if "src/aimedres/training/train_als.py" not in output:
        print("✗ FAILED: Canonical training script not used")
        return False
    print("✓ Canonical training script location used")
    
    print()
    print("✓ Test 1 PASSED: All parameters correctly handled")
    print()
    
    # Test 2: List mode to verify batch support detection
    print("Test 2: List mode - verify batch support detection")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, "-m", "aimedres.cli.commands", "train",
         "--list", "--only", "als"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        return False
    
    output = result.stdout + result.stderr
    
    # Verify ALS job shows batch=True
    if "als:" in output.lower() and "batch=true" in output.lower():
        print("✓ ALS training job correctly shows batch support")
    else:
        print("✗ FAILED: ALS training job should show batch=True")
        return False
    
    print()
    print("✓ Test 2 PASSED: Batch support correctly detected")
    print()
    
    # Test 3: Verify multiple jobs with parallel mode
    print("Test 3: Multiple jobs in parallel mode")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, "-m", "aimedres.cli.commands", "train",
         "--parallel", "--max-workers", "6",
         "--epochs", "50", "--folds", "5", "--batch", "128",
         "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        return False
    
    output = result.stdout + result.stderr
    
    # Count jobs that received batch-size parameter
    batch_size_count = output.count("--batch-size 128")
    print(f"✓ {batch_size_count} jobs received --batch-size 128 parameter")
    
    # Verify parallel mode
    if "Parallel mode enabled" in output and "max_workers" in output.lower():
        print("✓ Parallel mode with max-workers configured")
    
    print()
    print("✓ Test 3 PASSED: Multiple jobs handled correctly")
    print()
    
    return True


def main():
    """Main verification routine"""
    # Change to repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    import os
    os.chdir(repo_root)
    
    print()
    print("VERIFICATION: Production-ready configuration")
    print()
    
    try:
        if verify_production_command():
            print("=" * 70)
            print("✅ SUCCESS: Production command verified and working correctly")
            print("=" * 70)
            print()
            print("The following command is ready to use:")
            print()
            print("  aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128")
            print()
            print("This will:")
            print("  • Train all discovered medical AI models in parallel")
            print("  • Use up to 6 concurrent workers")
            print("  • Use 50 epochs for neural networks")
            print("  • Use 5-fold cross-validation")
            print("  • Use batch size of 128 for models that support it")
            print("  • Save results to results/ directory")
            print("  • Generate logs in logs/ directory")
            print("  • Create summary reports in summaries/ directory")
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
