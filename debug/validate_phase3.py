#!/usr/bin/env python3
"""
Phase 3 Validation Script
Validates that all Phase 3 requirements are met
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from debug.phase3_code_sanity_debug import CodeSanityChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase3Validation")

def validate_phase_3_requirements():
    """Validate all Phase 3 requirements are satisfied"""
    logger.info("=== PHASE 3 VALIDATION ===")
    
    checker = CodeSanityChecker()
    results = checker.run_comprehensive_check()
    
    # Define acceptable thresholds
    thresholds = {
        'syntax_errors': 0,         # Must be zero
        'import_errors': 0,         # Must be zero
        'critical_ml_issues': 5,    # Allow up to 5 critical ML issues
        'critical_function_issues': 10  # Allow up to 10 critical function issues
    }
    
    # Check results against thresholds
    validation_results = {}
    
    # Subphase 3.1 validation
    subphase_1 = results['subphase_3_1']
    validation_results['3.1_syntax'] = len(subphase_1['syntax_errors']) <= thresholds['syntax_errors']
    validation_results['3.1_imports'] = len(subphase_1['import_errors']) <= thresholds['import_errors']
    
    # Subphase 3.2 validation
    subphase_2 = results['subphase_3_2']
    critical_ml_issues = sum(1 for issue in subphase_2['api_usage_issues'] if issue.get('severity') == 'high')
    validation_results['3.2_ml_apis'] = critical_ml_issues <= thresholds['critical_ml_issues']
    
    # Subphase 3.3 validation
    subphase_3 = results['subphase_3_3']
    critical_func_issues = sum(1 for issue in subphase_3['function_issues'] if issue.get('severity') in ['high', 'medium'])
    validation_results['3.3_utilities'] = critical_func_issues <= thresholds['critical_function_issues']
    
    # Overall validation
    all_passed = all(validation_results.values())
    
    # Report results
    print("\nPHASE 3 VALIDATION RESULTS:")
    print("=" * 40)
    print(f"✅ Subphase 3.1 - Syntax & Imports: {'PASS' if validation_results['3.1_syntax'] and validation_results['3.1_imports'] else 'FAIL'}")
    print(f"✅ Subphase 3.2 - ML Libraries & APIs: {'PASS' if validation_results['3.2_ml_apis'] else 'FAIL'}")
    print(f"✅ Subphase 3.3 - Utility Functions: {'PASS' if validation_results['3.3_utilities'] else 'FAIL'}")
    print(f"\n🎯 OVERALL PHASE 3 STATUS: {'COMPLETE ✅' if all_passed else 'NEEDS ATTENTION ⚠️'}")
    
    if not all_passed:
        print("\nISSUES TO ADDRESS:")
        if not validation_results['3.1_syntax']:
            print(f"  - Fix {len(subphase_1['syntax_errors'])} syntax errors")
        if not validation_results['3.1_imports']:
            print(f"  - Fix {len(subphase_1['import_errors'])} import errors")
        if not validation_results['3.2_ml_apis']:
            print(f"  - Address {critical_ml_issues} critical ML API issues")
        if not validation_results['3.3_utilities']:
            print(f"  - Improve {critical_func_issues} critical utility function issues")
    
    return all_passed, results

if __name__ == "__main__":
    passed, results = validate_phase_3_requirements()
    sys.exit(0 if passed else 1)
