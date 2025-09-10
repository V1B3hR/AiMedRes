#!/usr/bin/env python3
"""
Test Comprehensive Training + Simulation System
===============================================

This test validates that the comprehensive training + simulation system
works correctly and meets all the problem statement requirements:

- Comprehensive training + simulation (medical AI context)
- Imports labyrinth components and runs the adaptive simulation
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def test_labyrinth_imports():
    """Test that labyrinth components can be imported successfully"""
    print("🧪 Testing labyrinth component imports...")
    
    try:
        # Test importing labyrinth components
        result = subprocess.run([
            sys.executable, 
            "-c", 
            "from labyrinth_adaptive import UnifiedAdaptiveAgent, AliveLoopNode, ResourceRoom, NetworkMetrics, MazeMaster, CapacitorInSpace; print('✅ Labyrinth components imported successfully')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Labyrinth imports: PASSED")
            return True
        else:
            print(f"❌ Labyrinth imports: FAILED - {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Labyrinth imports: FAILED - {e}")
        return False

def test_simulation_functionality():
    """Test that the adaptive simulation runs correctly"""
    print("🧪 Testing adaptive simulation...")
    
    try:
        result = subprocess.run([
            sys.executable, 
            "run_simulation.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and "✅ Simulation completed successfully!" in result.stdout:
            print("✅ Adaptive simulation: PASSED")
            return True
        else:
            print(f"❌ Adaptive simulation: FAILED - {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Adaptive simulation: FAILED - {e}")
        return False

def test_comprehensive_training():
    """Test comprehensive training functionality"""
    print("🧪 Testing comprehensive training...")
    
    try:
        result = subprocess.run([
            sys.executable, 
            "full_training.py",
            "--mode", "basic"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and "✅ Basic Training Completed!" in result.stdout:
            print("✅ Comprehensive training: PASSED")
            return True
        else:
            print(f"❌ Comprehensive training: FAILED - {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Comprehensive training: FAILED - {e}")
        return False

def test_medical_ai_context():
    """Test medical AI context integration"""
    print("🧪 Testing medical AI context...")
    
    try:
        result = subprocess.run([
            sys.executable, 
            "full_training.py",
            "--mode", "simulation"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and "ML prediction:" in result.stdout:
            print("✅ Medical AI context: PASSED")
            return True
        else:
            print(f"❌ Medical AI context: FAILED")
            print(f"Return code: {result.returncode}")
            if "ML prediction:" not in result.stdout:
                print("ML prediction not found in output")
            return False
    except Exception as e:
        print(f"❌ Medical AI context: FAILED - {e}")
        return False

def test_comprehensive_system():
    """Test the complete comprehensive training + simulation system"""
    print("🧪 Testing comprehensive training + simulation system...")
    
    try:
        result = subprocess.run([
            sys.executable, 
            "comprehensive_training_simulation.py"
        ], capture_output=True, text=True, timeout=120)
        
        success_indicators = [
            "✅ Training accuracy:",
            "✅ Simulation health score:",
            "✅ Labyrinth components imported and functional",
            "✅ Medical AI context fully integrated"
        ]
        
        if result.returncode == 0 and all(indicator in result.stdout for indicator in success_indicators):
            print("✅ Comprehensive system: PASSED")
            return True
        else:
            print(f"❌ Comprehensive system: FAILED")
            print(f"stdout: {result.stdout[-500:]}")  # Last 500 chars
            print(f"stderr: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Comprehensive system: FAILED - {e}")
        return False

def test_main_integration():
    """Test main.py comprehensive mode integration"""
    print("🧪 Testing main.py comprehensive mode...")
    
    try:
        result = subprocess.run([
            sys.executable, 
            "main.py",
            "--mode", "comprehensive"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and "Operation completed successfully!" in result.stdout:
            print("✅ Main integration: PASSED")
            return True
        else:
            print(f"❌ Main integration: FAILED - {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Main integration: FAILED - {e}")
        return False

def main():
    """Run all tests for the comprehensive training + simulation system"""
    print("🔬 COMPREHENSIVE TRAINING + SIMULATION TEST SUITE")
    print("=" * 60)
    print("Testing problem statement requirements:")
    print("- Comprehensive training + simulation (medical AI context)")
    print("- Imports labyrinth components and runs adaptive simulation")
    print("=" * 60)
    
    # Track test results
    tests = [
        ("Labyrinth Imports", test_labyrinth_imports),
        ("Simulation Functionality", test_simulation_functionality),
        ("Comprehensive Training", test_comprehensive_training),
        ("Medical AI Context", test_medical_ai_context),
        ("Comprehensive System", test_comprehensive_system),
        ("Main Integration", test_main_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
        print()
    
    # Final summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    success_rate = passed / total
    if success_rate == 1.0:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ Problem statement requirements fully implemented:")
        print("   - Comprehensive training + simulation (medical AI context)")
        print("   - Labyrinth components imported and functional")
        print("   - Adaptive simulation running correctly")
        return True
    elif success_rate >= 0.5:
        print(f"⚠️  PARTIAL SUCCESS ({passed}/{total} tests passed)")
        print("Some components may need additional work")
        return False
    else:
        print(f"❌ TESTS FAILED ({passed}/{total} tests passed)")
        print("Significant issues need to be addressed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)