#!/usr/bin/env python3
"""
Test script to verify that ALL 7 core medical AI models are included in the orchestrator.
This addresses the requirement: "Run training for ALL medical AI models using the orchestrator"
"""

import sys
import subprocess
from pathlib import Path


def test_all_seven_models_included():
    """Test that all 7 core medical AI models are included in default_jobs"""
    print("Test: Verify all 7 core medical AI models are included...")
    
    # Run with --no-auto-discover to only get the built-in default jobs
    result = subprocess.run(
        [sys.executable, "src/aimedres/cli/train.py", "--list", "--no-auto-discover"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ Failed with code {result.returncode}")
        print(result.stderr)
        return False
    
    output = result.stdout + result.stderr
    
    # Define the 7 core medical AI models we expect
    expected_models = {
        "als": "src/aimedres/training/train_als.py",
        "alzheimers": "src/aimedres/training/train_alzheimers.py",
        "parkinsons": "src/aimedres/training/train_parkinsons.py",
        "brain_mri": "src/aimedres/training/train_brain_mri.py",
        "cardiovascular": "src/aimedres/training/train_cardiovascular.py",
        "diabetes": "src/aimedres/training/train_diabetes.py",
        "specialized_agents": "src/aimedres/training/train_specialized_agents.py",
    }
    
    # Count jobs from the output
    job_lines = [line for line in output.split('\n') if line.strip().startswith('- ')]
    
    if len(job_lines) != 7:
        print(f"✗ Expected 7 default jobs, but found {len(job_lines)}")
        print(f"  Jobs found: {job_lines}")
        return False
    
    print(f"✓ Found 7 default jobs (expected count)")
    
    # Verify each model is present
    missing_models = []
    for model_id, script_path in expected_models.items():
        found = False
        for line in job_lines:
            if f"{model_id}:" in line and f"script={script_path}" in line:
                found = True
                print(f"✓ Found {model_id}: {script_path}")
                break
        
        if not found:
            missing_models.append((model_id, script_path))
    
    if missing_models:
        print(f"✗ Missing models in default_jobs:")
        for model_id, script_path in missing_models:
            print(f"  - {model_id}: {script_path}")
        return False
    
    print("✓ All 7 core medical AI models are included in default_jobs")
    print("✓ Test passed: ALL medical AI models are available for training\n")
    return True


def test_specialized_agents_configuration():
    """Test that specialized_agents job is properly configured"""
    print("Test: Verify specialized_agents job configuration...")
    
    result = subprocess.run(
        [sys.executable, "src/aimedres/cli/train.py", "--dry-run", 
         "--only", "specialized_agents", "--epochs", "10", "--folds", "3"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ Failed with code {result.returncode}")
        print(result.stderr)
        return False
    
    output = result.stdout + result.stderr
    
    # Check that the command is correct
    if "src/aimedres/training/train_specialized_agents.py" not in output:
        print("✗ Specialized agents script path not found")
        return False
    print("✓ Correct script path")
    
    if "--epochs 10" not in output:
        print("✗ Epochs parameter not found")
        return False
    print("✓ Epochs parameter included")
    
    if "--folds 3" not in output:
        print("✗ Folds parameter not found")
        return False
    print("✓ Folds parameter included")
    
    if "--output-dir" not in output:
        print("✗ Output directory parameter not found")
        return False
    print("✓ Output directory parameter included")
    
    if "specialized_agents_comprehensive_results" not in output:
        print("✗ Expected output directory name not found")
        return False
    print("✓ Correct output directory name")
    
    print("✓ Test passed: Specialized agents job is properly configured\n")
    return True


def test_all_seven_models_dry_run():
    """Test that all 7 models can generate valid commands"""
    print("Test: Verify all 7 models generate valid commands...")
    
    result = subprocess.run(
        [sys.executable, "src/aimedres/cli/train.py", "--dry-run", 
         "--no-auto-discover", "--epochs", "10", "--folds", "3"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"✗ Failed with code {result.returncode}")
        print(result.stderr)
        return False
    
    output = result.stdout + result.stderr
    
    # Check that we selected 7 jobs
    if "Selected jobs: 7" not in output:
        print("✗ Expected 7 selected jobs")
        return False
    print("✓ All 7 jobs selected")
    
    # Check that commands were generated for each job
    expected_job_ids = ["als", "alzheimers", "parkinsons", "brain_mri", 
                       "cardiovascular", "diabetes", "specialized_agents"]
    
    missing_commands = []
    for job_id in expected_job_ids:
        if f"[{job_id}] (dry-run) Command:" not in output:
            missing_commands.append(job_id)
        else:
            print(f"✓ Command generated for {job_id}")
    
    if missing_commands:
        print(f"✗ Missing commands for: {', '.join(missing_commands)}")
        return False
    
    print("✓ Test passed: All 7 models generate valid commands\n")
    return True


def main():
    """Run all tests"""
    print("=" * 80)
    print("Testing ALL 7 Core Medical AI Models - Complete Coverage")
    print("=" * 80)
    print()
    
    # Change to repository root
    repo_root = Path(__file__).parent.parent.parent
    import os
    os.chdir(repo_root)
    
    tests = [
        test_all_seven_models_included,
        test_specialized_agents_configuration,
        test_all_seven_models_dry_run,
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
        print("\n✓ All tests passed! ALL 7 medical AI models are ready for training.")
        print("\nThe complete set of medical AI models includes:")
        print("  1. ALS (Amyotrophic Lateral Sclerosis)")
        print("  2. Alzheimer's Disease")
        print("  3. Parkinson's Disease")
        print("  4. Brain MRI Classification")
        print("  5. Cardiovascular Disease Prediction")
        print("  6. Diabetes Prediction")
        print("  7. Specialized Medical Agents")
        print("\nAll models can be trained using: python src/aimedres/cli/train.py")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
