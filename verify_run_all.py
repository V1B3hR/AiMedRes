#!/usr/bin/env python3
"""
Comprehensive verification script for "run all" training functionality.
This script verifies that the training orchestrator can discover and run all training scripts.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description, check_output=None):
    """Run a command and verify its output."""
    print(f"\n{'='*70}")
    print(f"🔍 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout + result.stderr
        print(output)
        
        if result.returncode != 0:
            print(f"❌ Command failed with exit code {result.returncode}")
            return False
        
        if check_output:
            for check in check_output:
                if check not in output:
                    print(f"❌ Expected output not found: {check}")
                    return False
        
        print(f"✅ {description} - SUCCESS")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Command failed with error: {e}")
        return False


def main():
    """Run comprehensive verification."""
    print("="*70)
    print("🏥 AiMedRes Training Orchestrator - 'Run All' Verification")
    print("="*70)
    print()
    print("This script verifies that the training orchestrator is fully")
    print("functional and ready to 'run all' medical AI training scripts.")
    print()
    
    # Change to repository root
    repo_root = Path(__file__).parent
    import os
    os.chdir(repo_root)
    
    tests = [
        {
            "cmd": [sys.executable, "test_run_all_training.py"],
            "description": "Run automated test suite",
            "check": ["4/4 tests passed"]
        },
        {
            "cmd": [sys.executable, "run_all_training.py", "--list"],
            "description": "List all discovered training jobs",
            "check": [
                "src/aimedres/training/train_als.py",
                "src/aimedres/training/train_alzheimers.py",
                "src/aimedres/training/train_parkinsons.py"
            ]
        },
        {
            "cmd": [sys.executable, "run_all_training.py", "--dry-run", "--epochs", "5", "--folds", "2"],
            "description": "Dry-run with custom parameters (preview commands)",
            "check": ["--epochs 5", "--folds 2"]
        },
        {
            "cmd": [sys.executable, "run_all_training.py", "--dry-run", "--parallel", "--max-workers", "2", "--only", "als", "alzheimers"],
            "description": "Dry-run parallel mode with filtering",
            "check": ["Parallel mode enabled", "train_als.py"]
        },
        {
            "cmd": [sys.executable, "run_all_training.py", "--list", "--only", "als", "alzheimers"],
            "description": "Filter specific jobs",
            "check": ["als:", "alzheimers:"]
        },
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if run_command(test["cmd"], test["description"], test.get("check")):
            passed += 1
        else:
            print(f"⚠️ Test failed, continuing...")
    
    print("\n" + "="*70)
    print(f"📊 Verification Results: {passed}/{total} checks passed")
    print("="*70)
    
    if passed == total:
        print("\n✅ SUCCESS: All verification checks passed!")
        print("\n🎉 The 'Run All' training functionality is fully operational!")
        print("\nYou can now:")
        print("  1. Run all training:         python run_all_training.py")
        print("  2. Run with parameters:      python run_all_training.py --epochs 50 --folds 5")
        print("  3. Run in parallel:          python run_all_training.py --parallel --max-workers 4")
        print("  4. Run specific models:      python run_all_training.py --only als alzheimers")
        print("  5. Use convenience script:   ./run_medical_training.sh")
        print("\n📚 See IMPLEMENTATION_SUMMARY.md for complete documentation")
        return 0
    else:
        print(f"\n⚠️ WARNING: {total - passed} verification check(s) failed")
        print("The system may still work, but some features might have issues.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
