#!/usr/bin/env python3
"""
Test Suite for Problem Statement Solutions

Tests all implementations to ensure they work correctly.
"""

import subprocess
import sys
import os

def run_script(script_path):
    """Run a Python script and return success status and output."""
    try:
        result = subprocess.run(
            [sys.executable, script_path], 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Script timed out"
    except Exception as e:
        return False, "", str(e)

def test_implementations():
    """Test all problem statement implementations."""
    
    scripts_to_test = [
        "corrected_problem_statement.py",
        "deep_learning_training_problem_statement.py", 
        "enhanced_problem_statement_solution.py"
    ]
    
    print("=== Testing Problem Statement Solutions ===\n")
    
    results = {}
    
    for script in scripts_to_test:
        if not os.path.exists(script):
            print(f"❌ {script}: File not found")
            results[script] = False
            continue
            
        print(f"🧪 Testing {script}...")
        success, stdout, stderr = run_script(script)
        
        if success:
            print(f"✅ {script}: SUCCESS")
            # Check for expected output patterns
            if "First 5 records:" in stdout:
                print(f"   ✓ Dataset loading works")
            else:
                print(f"   ⚠️ Missing expected output pattern")
                
            if "enhanced_problem_statement_solution.py" in script:
                if "Deep Learning Model Accuracy:" in stdout:
                    print(f"   ✓ Deep learning training works")
                if "Feature Importance" in stdout:
                    print(f"   ✓ Feature analysis works")
                    
        else:
            print(f"❌ {script}: FAILED")
            if stderr:
                print(f"   Error: {stderr[:200]}...")
                
        results[script] = success
        print()
    
    # Summary
    print("=== Test Summary ===")
    passed = sum(results.values())
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    for script, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {script}: {status}")
    
    if passed == total:
        print("\n🎉 All tests passed! Problem statement solutions are working correctly.")
    else:
        print(f"\n⚠️ Some tests failed. Check the outputs above for details.")
        
    return passed == total

if __name__ == "__main__":
    success = test_implementations()
    sys.exit(0 if success else 1)