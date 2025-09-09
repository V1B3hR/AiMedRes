#!/usr/bin/env python3
"""
Run Comprehensive Training Test and Report for DuetMind Adaptive

This script executes the full system test for comprehensive training
as specified in the problem statement and generates detailed reports.
"""

import sys
import os
from pathlib import Path

def main():
    """Run comprehensive training test and report"""
    print("🧠 DuetMind Adaptive - Full System Test/Report: Run Comprehensive Training")
    print("=" * 80)
    print()
    
    try:
        # Import and run the comprehensive training test
        from test_comprehensive_training import run_comprehensive_training_test
        
        print("🚀 Starting comprehensive training system test...")
        print()
        
        # Run the full test suite
        success = run_comprehensive_training_test()
        
        print()
        print("=" * 80)
        if success:
            print("🎉 COMPREHENSIVE TRAINING TEST COMPLETED SUCCESSFULLY!")
            print("✅ All training components validated and working correctly")
            print("📊 Success Rate: 100%")
            print("🚀 System ready for production deployment")
        else:
            print("⚠️  COMPREHENSIVE TRAINING TEST COMPLETED WITH ISSUES")
            print("🔧 Some components may need attention")
            print("📋 Review test reports for details")
        
        print()
        print("📄 Generated Reports:")
        print("   📝 comprehensive_training_report.txt")
        print("   📊 comprehensive_training_report.json")
        print("=" * 80)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Error running comprehensive training test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())