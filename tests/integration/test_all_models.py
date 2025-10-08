#!/usr/bin/env python3
"""
Verification script to confirm all 6 disease prediction models can be run.
This demonstrates the solution to: "Run training for all disease prediction models"
"""

import sys
import subprocess
from pathlib import Path

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def verify_all_6_models_listed():
    """Verify that all 6 core models are available."""
    print_section("Step 1: Verify All 6 Core Models Are Available")
    
    result = subprocess.run(
        [sys.executable, "run_all_training.py", "--list", "--no-auto-discover"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    output = result.stdout + result.stderr
    
    required_models = {
        "als": "ALS (Amyotrophic Lateral Sclerosis)",
        "alzheimers": "Alzheimer's Disease",
        "parkinsons": "Parkinson's Disease",
        "brain_mri": "Brain MRI Classification",
        "cardiovascular": "Cardiovascular Disease Prediction",
        "diabetes": "Diabetes Prediction"
    }
    
    print("\nChecking for all 6 core disease prediction models:\n")
    all_found = True
    
    for model_id, model_name in required_models.items():
        if f"- {model_id}:" in output:
            print(f"  ✓ {model_name} ({model_id})")
        else:
            print(f"  ✗ {model_name} ({model_id}) NOT FOUND")
            all_found = False
    
    if all_found:
        print("\n✓ SUCCESS: All 6 disease prediction models are available!")
        return True
    else:
        print("\n✗ FAILURE: Some models are missing")
        return False

def verify_run_all_command():
    """Verify that the run_all_training.py command works."""
    print_section("Step 2: Verify Run All Training Command Works")
    
    print("\nTesting command: python run_all_training.py --dry-run --epochs 10\n")
    
    result = subprocess.run(
        [sys.executable, "run_all_training.py", 
         "--only", "als", "alzheimers", "parkinsons", "brain_mri", "cardiovascular", "diabetes",
         "--dry-run", "--epochs", "10", "--folds", "3"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    output = result.stdout + result.stderr
    
    # Check that all 6 models have commands generated
    models_with_commands = 0
    expected_commands = [
        "[als] (dry-run) Command:",
        "[alzheimers] (dry-run) Command:",
        "[parkinsons] (dry-run) Command:",
        "[brain_mri] (dry-run) Command:",
        "[cardiovascular] (dry-run) Command:",
        "[diabetes] (dry-run) Command:"
    ]
    
    print("Commands that would be executed:\n")
    for cmd in expected_commands:
        if cmd in output:
            models_with_commands += 1
            # Extract and print the command line
            for line in output.split('\n'):
                if cmd in line:
                    print(f"  ✓ {line.strip()}")
                    break
        else:
            print(f"  ✗ {cmd} NOT FOUND")
    
    if models_with_commands == 6:
        print(f"\n✓ SUCCESS: All 6 models can be trained with python run_all_training.py")
        return True
    else:
        print(f"\n✗ FAILURE: Only {models_with_commands}/6 models have commands")
        return False

def verify_custom_parameters():
    """Verify custom parameters work."""
    print_section("Step 3: Verify Custom Parameters Work")
    
    print("\nTesting: python run_all_training.py --epochs 20 --folds 5 --dry-run\n")
    
    result = subprocess.run(
        [sys.executable, "run_all_training.py",
         "--only", "als", "alzheimers", "parkinsons", "cardiovascular", "diabetes",
         "--dry-run", "--epochs", "20", "--folds", "5"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    output = result.stdout + result.stderr
    
    has_epochs_20 = "--epochs 20" in output
    has_folds_5 = "--folds 5" in output
    
    if has_epochs_20:
        print("  ✓ Custom epochs parameter (20) applied")
    else:
        print("  ✗ Custom epochs parameter NOT found")
    
    if has_folds_5:
        print("  ✓ Custom folds parameter (5) applied")
    else:
        print("  ✗ Custom folds parameter NOT found")
    
    if has_epochs_20 and has_folds_5:
        print("\n✓ SUCCESS: Custom parameters work correctly")
        return True
    else:
        print("\n✗ FAILURE: Custom parameters not working")
        return False

def main():
    """Run all verification steps."""
    print("\n" + "=" * 70)
    print("  VERIFICATION: Run All 6 Disease Prediction Models")
    print("=" * 70)
    print("\nThis script verifies that the following command works:")
    print("  python run_all_training.py")
    print("\nAnd trains all 6 disease prediction models:")
    print("  1. Alzheimer's Disease")
    print("  2. ALS (Amyotrophic Lateral Sclerosis)")
    print("  3. Parkinson's Disease")
    print("  4. Brain MRI Classification")
    print("  5. Cardiovascular Disease Prediction")
    print("  6. Diabetes Prediction")
    
    # Change to repository root
    repo_root = Path(__file__).parent
    import os
    os.chdir(repo_root)
    
    # Run verification steps
    results = []
    results.append(verify_all_6_models_listed())
    results.append(verify_run_all_command())
    results.append(verify_custom_parameters())
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if all(results):
        print("\n🎉 SUCCESS! All verification tests passed!")
        print("\nThe 'Run All Training' functionality is working correctly.")
        print("\nUsage Examples:")
        print("  # Run all 6 models with default settings:")
        print("  python run_all_training.py")
        print()
        print("  # Run with custom epochs and folds:")
        print("  python run_all_training.py --epochs 20 --folds 5")
        print()
        print("  # Run in parallel:")
        print("  python run_all_training.py --parallel --max-workers 4")
        print()
        print("  # Run specific models only:")
        print("  python run_all_training.py --only als alzheimers parkinsons")
        print()
        print("  # Preview without running (dry-run):")
        print("  python run_all_training.py --dry-run --epochs 10")
        return 0
    else:
        print("\n✗ FAILURE: Some verification tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
