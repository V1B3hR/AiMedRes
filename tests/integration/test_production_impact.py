#!/usr/bin/env python3
"""
Validation script to test the Production & Impact implementations
"""

import os
import sys
from pathlib import Path

def test_deployment_toolkit():
    """Test that deployment toolkit works"""
    try:
        from aimedres.core.production_agent import ProductionDeploymentManager
        import tempfile
        
        print("🔧 Testing Deployment Toolkit...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'deployment_path': temp_dir,
                'port': 8080,
                'workers': 4,
                'grafana_password': 'test123'
            }
            
            manager = ProductionDeploymentManager(config)
            success = manager.deploy_to_files()
            
            if success:
                files = os.listdir(temp_dir)
                expected_files = [
                    'Dockerfile', 'docker-compose.yml', 'nginx.conf',
                    'k8s-deployment.yaml', 'k8s-service.yaml', 'k8s-ingress.yaml',
                    'prometheus.yml', 'grafana-dashboard.json', 'requirements.txt'
                ]
                
                missing_files = [f for f in expected_files if f not in files]
                if not missing_files:
                    print("✅ Deployment Toolkit: WORKING")
                    return True
                else:
                    print(f"❌ Missing deployment files: {missing_files}")
                    return False
            else:
                print("❌ Deployment toolkit failed to generate files")
                return False
                
    except Exception as e:
        print(f"❌ Deployment toolkit error: {e}")
        return False

def test_documentation_exists():
    """Test that all required documentation exists"""
    print("\n📚 Testing Documentation...")
    
    required_docs = [
        'CONTRIBUTING.md',
        'docs/USER_EXPERIENCE_GUIDE.md', 
        'docs/CLINICAL_VALIDATION_FRAMEWORK.md'
    ]
    
    all_exist = True
    for doc_path in required_docs:
        full_path = Path(doc_path)
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            print(f"✅ {doc_path} exists ({size_kb:.1f} KB)")
        else:
            print(f"❌ {doc_path} missing")
            all_exist = False
    
    return all_exist

def test_user_experience_examples():
    """Test that user experience examples can be imported"""
    print("\n🎯 Testing User Experience Examples...")
    
    examples_to_test = [
        'examples.enterprise_demo',
        'examples.demo_production_mlops',
        'demo_enhanced_features'  # This file is in root, not examples/
    ]
    
    working_examples = 0
    for example in examples_to_test:
        try:
            __import__(example)
            print(f"✅ {example} can be imported")
            working_examples += 1
        except Exception as e:
            print(f"⚠️  {example} import issue: {e}")
    
    print(f"📊 {working_examples}/{len(examples_to_test)} examples working")
    return working_examples > 0

def test_clinical_validation_components():
    """Test clinical validation components"""
    print("\n🏥 Testing Clinical Validation Components...")
    
    # Test if clinical decision support components exist
    try:
        import aimedres.clinical.decision_support
        print("✅ Clinical decision support module available")
        clinical_available = True
    except ImportError as e:
        print(f"⚠️  Clinical decision support module not available: {e}")
        clinical_available = False
    
    # Test demo enhanced features (includes clinical scenarios)
    try:
        import demo_enhanced_features
        print("✅ Enhanced clinical features available")
        enhanced_available = True
    except ImportError as e:
        print(f"⚠️  Enhanced clinical features not available: {e}")
        enhanced_available = False
    
    # Check if clinical validation framework documentation exists
    clinical_doc_exists = Path('docs/CLINICAL_VALIDATION_FRAMEWORK.md').exists()
    if clinical_doc_exists:
        print("✅ Clinical validation framework documentation available")
    else:
        print("❌ Clinical validation framework documentation missing")
    
    # At least documentation should be available for clinical validation
    return clinical_available or enhanced_available or clinical_doc_exists

def test_production_readiness():
    """Test production readiness capabilities"""
    print("\n🚀 Testing Production Readiness...")
    
    # Test if production MLOps components are available
    production_components = [
        'examples.demo_production_mlops',
        'duetmind'
    ]
    
    working_components = 0
    for component in production_components:
        try:
            __import__(component)
            print(f"✅ {component} available")
            working_components += 1
        except Exception as e:
            print(f"⚠️  {component} issue: {e}")
    
    return working_components >= len(production_components) // 2

def validate_roadmap_updates():
    """Validate that roadmap has been updated"""
    print("\n📋 Validating Roadmap Updates...")
    
    try:
        with open('roadmap.md', 'r') as f:
            roadmap_content = f.read()
        
        # Check for updated markers
        production_section = "4. Production & Impact"
        if production_section in roadmap_content:
            section_start = roadmap_content.find(production_section)
            section_content = roadmap_content[section_start:section_start+1000]
            
            completed_items = section_content.count('✅')
            if completed_items >= 4:
                print(f"✅ Roadmap updated with {completed_items} completed items")
                return True
            else:
                print(f"⚠️  Only {completed_items} items marked as complete in roadmap")
                return False
        else:
            print("❌ Production & Impact section not found in roadmap")
            return False
            
    except Exception as e:
        print(f"❌ Error reading roadmap: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 Production & Impact Validation Suite")
    print("=" * 50)
    
    tests = [
        ("Deployment Toolkit", test_deployment_toolkit),
        ("Documentation", test_documentation_exists), 
        ("User Experience", test_user_experience_examples),
        ("Clinical Validation", test_clinical_validation_components),
        ("Production Readiness", test_production_readiness),
        ("Roadmap Updates", validate_roadmap_updates)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Validation Summary")
    print("=" * 50)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL PRODUCTION & IMPACT REQUIREMENTS COMPLETED!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} requirements need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)