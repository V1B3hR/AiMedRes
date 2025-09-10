#!/usr/bin/env python3
"""
Test script for enhanced training capabilities
"""

import subprocess
import sys
import time

def test_training_mode(mode_name):
    """Test a specific training mode"""
    print(f"\n{'='*60}")
    print(f"Testing {mode_name.upper()} training mode...")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run([
            sys.executable, 
            "full_training.py", 
            "--mode", mode_name
        ], capture_output=True, text=True, timeout=180)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {mode_name.upper()} PASSED in {elapsed:.1f}s")
            return True
        else:
            print(f"❌ {mode_name.upper()} FAILED in {elapsed:.1f}s")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {mode_name.upper()} TIMEOUT after 180s")
        return False
    except Exception as e:
        print(f"💥 {mode_name.upper()} ERROR: {e}")
        return False

def main():
    """Test all enhanced training modes"""
    print("🧠 DuetMind Adaptive - Enhanced Training Test Suite")
    print("🎯 Testing all enhanced training capabilities...")
    
    # Test modes to validate
    test_modes = ["basic", "extended", "advanced", "simulation"]
    
    results = {}
    total_start = time.time()
    
    for mode in test_modes:
        results[mode] = test_training_mode(mode)
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 ENHANCED TRAINING TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for mode, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {mode.upper()}: {status}")
    
    success_rate = passed / total
    print(f"\n📊 Success Rate: {success_rate*100:.0f}% ({passed}/{total} modes)")
    print(f"⏱️  Total Time: {total_elapsed:.1f}s")
    
    if success_rate >= 0.75:
        print("\n🎉 ENHANCED TRAINING CAPABILITIES WORKING WELL!")
        return True
    else:
        print("\n💔 SOME ENHANCED TRAINING CAPABILITIES NEED ATTENTION")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)