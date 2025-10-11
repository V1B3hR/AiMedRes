#!/usr/bin/env python3
"""
End-to-End Validation Script for AiMedRes v0.2.0

Tests basic functionality to ensure the repository is properly set up
and ready for external use.
"""

import sys
import importlib

def test_import(module_name, description):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {description}: {module_name} - {e}")
        return False

def main():
    """Run validation tests."""
    print("="*60)
    print("AiMedRes v0.2.0 - End-to-End Validation")
    print("="*60)
    print()
    
    tests_passed = 0
    tests_total = 0
    
    # Test core imports
    print("1. Testing Core Imports:")
    print("-" * 40)
    
    test_modules = [
        ("aimedres.training.train_alzheimers", "Alzheimer's Training"),
        ("aimedres.training.train_als", "ALS Training"),
        ("aimedres.training.train_diabetes", "Diabetes Training"),
        ("aimedres.training.train_cardiovascular", "Cardiovascular Training"),
        ("aimedres.training.train_parkinsons", "Parkinson's Training"),
        ("aimedres.training.train_brain_mri", "Brain MRI Training"),
        ("aimedres.agent_memory.memory_consolidation", "Memory Consolidation"),
        ("aimedres.agents.specialized_medical_agents", "Specialized Agents"),
    ]
    
    for module, desc in test_modules:
        tests_total += 1
        if test_import(module, desc):
            tests_passed += 1
    
    print()
    print("2. Testing Training Pipeline Instantiation:")
    print("-" * 40)
    
    try:
        from aimedres.training.train_alzheimers import AlzheimerTrainingPipeline
        pipeline = AlzheimerTrainingPipeline()
        print(f"✓ Created AlzheimerTrainingPipeline: {pipeline.__class__.__name__}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to create pipeline: {e}")
    tests_total += 1
    
    print()
    print("3. Testing Entry Points:")
    print("-" * 40)
    
    entry_points = [
        "run_all_training.py",
        "main.py",
        "run_alzheimer_training.py"
    ]
    
    for ep in entry_points:
        try:
            with open(ep, 'r') as f:
                # Just check if file exists and is readable
                content = f.read(100)
                if content:
                    print(f"✓ Entry point exists: {ep}")
                    tests_passed += 1
        except FileNotFoundError:
            print(f"✗ Entry point missing: {ep}")
        tests_total += 1
    
    # Summary
    print()
    print("="*60)
    print(f"Validation Complete: {tests_passed}/{tests_total} tests passed")
    print("="*60)
    
    if tests_passed == tests_total:
        print("✓ All validation tests passed! Repository is ready.")
        return 0
    else:
        print(f"⚠ {tests_total - tests_passed} test(s) failed. Please review.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
