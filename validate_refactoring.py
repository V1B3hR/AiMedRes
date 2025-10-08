#!/usr/bin/env python3
"""
Validation script for architecture refactoring.

This script tests that the new module structure is working correctly
and that all imports can be resolved.
"""

import sys
import warnings

def test_core_modules():
    """Test that core modules can be imported from new locations"""
    print("Testing core modules...")
    
    try:
        # Test constants
        from aimedres.core import constants
        print("✓ aimedres.core.constants")
    except ImportError as e:
        print(f"✗ aimedres.core.constants: {e}")
        return False
    
    try:
        # Test labyrinth
        from aimedres.core import labyrinth
        print("✓ aimedres.core.labyrinth")
    except ImportError as e:
        print(f"✗ aimedres.core.labyrinth: {e}")
        return False
    
    try:
        # Test production_agent
        from aimedres.core import production_agent
        print("✓ aimedres.core.production_agent")
    except ImportError as e:
        print(f"✗ aimedres.core.production_agent: {e}")
        return False
    
    try:
        # Test cognitive_engine
        from aimedres.core import cognitive_engine
        print("✓ aimedres.core.cognitive_engine")
    except ImportError as e:
        print(f"✗ aimedres.core.cognitive_engine: {e}")
        return False
    
    return True

def test_utils_modules():
    """Test that utils modules can be imported from new locations"""
    print("\nTesting utils modules...")
    
    try:
        from aimedres.utils import helpers
        print("✓ aimedres.utils.helpers")
    except ImportError as e:
        print(f"✗ aimedres.utils.helpers: {e}")
        return False
    
    try:
        from aimedres.utils import data_loaders
        print("✓ aimedres.utils.data_loaders")
    except ImportError as e:
        print(f"✗ aimedres.utils.data_loaders: {e}")
        return False
    
    return True

def test_clinical_modules():
    """Test that clinical modules can be imported from new locations"""
    print("\nTesting clinical modules...")
    
    try:
        from aimedres.clinical import decision_support
        print("✓ aimedres.clinical.decision_support")
    except ImportError as e:
        print(f"✗ aimedres.clinical.decision_support: {e}")
        return False
    
    try:
        from aimedres.clinical import parkinsons_als
        print("✓ aimedres.clinical.parkinsons_als")
    except ImportError as e:
        print(f"✗ aimedres.clinical.parkinsons_als: {e}")
        return False
    
    return True

def test_compliance_modules():
    """Test that compliance modules can be imported from new locations"""
    print("\nTesting compliance modules...")
    
    try:
        from aimedres.compliance import fda
        print("✓ aimedres.compliance.fda")
    except ImportError as e:
        print(f"✗ aimedres.compliance.fda: {e}")
        return False
    
    try:
        from aimedres.compliance import regulatory
        print("✓ aimedres.compliance.regulatory")
    except ImportError as e:
        print(f"✗ aimedres.compliance.regulatory: {e}")
        return False
    
    try:
        from aimedres.compliance import gdpr
        print("✓ aimedres.compliance.gdpr")
    except ImportError as e:
        print(f"✗ aimedres.compliance.gdpr: {e}")
        return False
    
    return True

def test_integration_modules():
    """Test that integration modules can be imported from new locations"""
    print("\nTesting integration modules...")
    
    try:
        from aimedres.integration import ehr
        print("✓ aimedres.integration.ehr")
    except ImportError as e:
        print(f"✗ aimedres.integration.ehr: {e}")
        return False
    
    try:
        from aimedres.integration import multimodal
        print("✓ aimedres.integration.multimodal")
    except ImportError as e:
        print(f"✗ aimedres.integration.multimodal: {e}")
        return False
    
    return True

def test_dashboard_modules():
    """Test that dashboard modules can be imported from new locations"""
    print("\nTesting dashboard modules...")
    
    try:
        from aimedres.dashboards import explainable_ai
        print("✓ aimedres.dashboards.explainable_ai")
    except ImportError as e:
        print(f"✗ aimedres.dashboards.explainable_ai: {e}")
        return False
    
    try:
        from aimedres.dashboards import data_quality
        print("✓ aimedres.dashboards.data_quality")
    except ImportError as e:
        print(f"✗ aimedres.dashboards.data_quality: {e}")
        return False
    
    return True

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("Architecture Refactoring Validation")
    print("=" * 60)
    
    all_passed = True
    
    # Suppress deprecation warnings for this validation
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Test all modules
    all_passed &= test_core_modules()
    all_passed &= test_utils_modules()
    all_passed &= test_clinical_modules()
    all_passed &= test_compliance_modules()
    all_passed &= test_integration_modules()
    all_passed &= test_dashboard_modules()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All validation tests passed!")
        print("=" * 60)
        return 0
    else:
        print("❌ Some validation tests failed!")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
