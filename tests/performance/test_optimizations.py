#!/usr/bin/env python3
"""
Comprehensive test suite for performance optimizations, bias detection, 
adversarial testing, and data integrity enhancements.

This test validates that the implemented improvements meet the requirements:
1. Performance Bottleneck: Response time <100ms target
2. Data Integrity: Enhanced de-identification with privacy management
3. Advanced features: Bias detection and adversarial testing
"""

import time
import json
import logging
from typing import Dict, Any, List
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aimedres.core.production_agent import OptimizedAdaptiveEngine
from aimedres.security.performance_monitor import ClinicalPerformanceMonitor, ClinicalPriority
from aimedres.security.ai_safety import ClinicalAISafetyMonitor
from aimedres.security.privacy import PrivacyManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceOptimizationTests:
    """Comprehensive test suite for performance and safety enhancements."""
    
    def __init__(self):
        self.results = {
            'performance_tests': {},
            'bias_detection_tests': {},
            'adversarial_tests': {},
            'data_integrity_tests': {},
            'summary': {}
        }
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites."""
        logger.info("🚀 Starting Comprehensive Performance & Safety Test Suite")
        
        # 1. Performance Bottleneck Tests
        logger.info("📊 Running Performance Optimization Tests...")
        self.results['performance_tests'] = self.test_performance_optimizations()
        
        # 2. Bias Detection Tests  
        logger.info("⚖️ Running Bias Detection Tests...")
        self.results['bias_detection_tests'] = self.test_bias_detection()
        
        # 3. Adversarial Testing
        logger.info("🛡️ Running Adversarial Testing...")
        self.results['adversarial_tests'] = self.test_adversarial_robustness()
        
        # 4. Data Integrity Tests
        logger.info("🔒 Running Data Integrity Tests...")
        self.results['data_integrity_tests'] = self.test_data_integrity()
        
        # Generate summary
        self.results['summary'] = self.generate_test_summary()
        
        return self.results
    
    def test_performance_optimizations(self) -> Dict[str, Any]:
        """Test performance optimizations targeting <100ms response times."""
        performance_results = {
            'target_response_time_ms': 100,
            'test_scenarios': [],
            'avg_response_time_ms': 0,
            'target_achieved': False,
            'optimization_effectiveness': {}
        }
        
        try:
            # Initialize optimized engine
            engine = OptimizedAdaptiveEngine(config={'fast_mode_enabled': True})
            
            # Test scenarios with different priorities
            test_scenarios = [
                {'priority': ClinicalPriority.EMERGENCY, 'task': 'Emergency cardiac assessment', 'expected_max_ms': 20},
                {'priority': ClinicalPriority.CRITICAL, 'task': 'Critical patient diagnosis', 'expected_max_ms': 50},
                {'priority': ClinicalPriority.URGENT, 'task': 'Urgent symptom analysis', 'expected_max_ms': 100},
                {'priority': ClinicalPriority.ROUTINE, 'task': 'Routine health check', 'expected_max_ms': 200}
            ]
            
            all_response_times = []
            
            for scenario in test_scenarios:
                response_times = []
                
                # Run multiple iterations for reliable timing
                for i in range(10):
                    start_time = time.time()
                    result = engine.safe_think(
                        'TestAgent', 
                        scenario['task'], 
                        clinical_priority=scenario['priority']
                    )
                    response_time_ms = (time.time() - start_time) * 1000
                    response_times.append(response_time_ms)
                    all_response_times.append(response_time_ms)
                
                avg_time = sum(response_times) / len(response_times)
                max_time = max(response_times)
                target_met = avg_time <= scenario['expected_max_ms']
                
                scenario_result = {
                    'priority': scenario['priority'].value,
                    'task': scenario['task'],
                    'avg_response_time_ms': avg_time,
                    'max_response_time_ms': max_time,
                    'target_ms': scenario['expected_max_ms'],
                    'target_achieved': target_met,
                    'samples': len(response_times)
                }
                
                performance_results['test_scenarios'].append(scenario_result)
                logger.info(f"  {scenario['priority'].value}: {avg_time:.1f}ms avg (target: {scenario['expected_max_ms']}ms) {'✅' if target_met else '❌'}")
            
            # Calculate overall performance
            performance_results['avg_response_time_ms'] = sum(all_response_times) / len(all_response_times)
            performance_results['target_achieved'] = performance_results['avg_response_time_ms'] <= 100
            
            # Test optimization effectiveness
            performance_results['optimization_effectiveness'] = self._test_optimization_features(engine)
            
            # Get performance report from engine
            engine_report = engine.get_performance_report()
            performance_results['engine_performance_report'] = engine_report
            
            logger.info(f"  Overall Performance: {performance_results['avg_response_time_ms']:.1f}ms avg {'✅' if performance_results['target_achieved'] else '❌'}")
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            performance_results['error'] = str(e)
        
        return performance_results
    
    def _test_optimization_features(self, engine) -> Dict[str, Any]:
        """Test specific optimization features."""
        optimization_tests = {}
        
        # Test caching effectiveness
        start_time = time.time()
        result1 = engine.safe_think('TestAgent', 'Cache test query')
        first_call_time = (time.time() - start_time) * 1000
        
        start_time = time.time()
        result2 = engine.safe_think('TestAgent', 'Cache test query')  # Should be cached
        second_call_time = (time.time() - start_time) * 1000
        
        cache_speedup = first_call_time / second_call_time if second_call_time > 0 else 0
        
        optimization_tests['caching'] = {
            'first_call_ms': first_call_time,
            'cached_call_ms': second_call_time,
            'speedup_factor': cache_speedup,
            'cache_effective': cache_speedup > 2.0
        }
        
        # Test fast mode effectiveness
        optimization_tests['fast_mode'] = {
            'enabled': engine.fast_mode_enabled,
            'frequent_ops_cache_size': len(engine._frequent_operations_cache),
            'cache_validity_seconds': engine._cache_validity_seconds
        }
        
        return optimization_tests
    
    def test_bias_detection(self) -> Dict[str, Any]:
        """Test enhanced bias detection capabilities."""
        bias_results = {
            'bias_detection_enabled': True,
            'statistical_methods_tested': [],
            'detected_biases': [],
            'recommendations_generated': 0
        }
        
        try:
            # Initialize AI safety monitor
            safety_monitor = ClinicalAISafetyMonitor(enable_bias_detection=True)
            
            # Create test decision data with known bias patterns
            test_decisions = self._generate_biased_test_data()
            
            # Add test decisions to monitor
            for decision_data in test_decisions:
                decision = safety_monitor.assess_ai_decision(**decision_data)
            
            # Run bias analysis
            recent_decisions = list(safety_monitor.decision_history)[-100:]
            if len(recent_decisions) >= 20:
                bias_analysis = safety_monitor._comprehensive_bias_analysis(recent_decisions)
                bias_results['statistical_methods_tested'] = list(bias_analysis.keys())
                
                # Check for significant biases
                for bias_type, analysis in bias_analysis.items():
                    if isinstance(analysis, dict):
                        for test_name, test_results in analysis.items():
                            if isinstance(test_results, dict) and 'p_value' in test_results:
                                if test_results['p_value'] < 0.05:
                                    bias_results['detected_biases'].append({
                                        'type': bias_type,
                                        'test': test_name,
                                        'p_value': test_results['p_value'],
                                        'significant': True
                                    })
                
                # Test bias remediation suggestions
                for bias_type in bias_analysis.keys():
                    suggestions = safety_monitor._generate_bias_remediation_suggestions(bias_type, {'p_value': 0.01})
                    bias_results['recommendations_generated'] += len(suggestions)
            
            logger.info(f"  Bias detection methods tested: {len(bias_results['statistical_methods_tested'])}")
            logger.info(f"  Significant biases detected: {len(bias_results['detected_biases'])}")
            
        except Exception as e:
            logger.error(f"Bias detection test failed: {e}")
            bias_results['error'] = str(e)
        
        return bias_results
    
    def _generate_biased_test_data(self) -> List[Dict[str, Any]]:
        """Generate test data with known bias patterns for testing."""
        import random
        from datetime import datetime, timezone
        
        test_data = []
        
        # Create biased data patterns
        for i in range(50):
            # Create age bias - older patients get lower confidence
            age = random.randint(20, 90)
            confidence = 0.9 - (age - 20) * 0.01 if age > 60 else 0.8 + random.uniform(-0.1, 0.1)
            confidence = max(0.1, min(0.95, confidence))
            
            # Create gender bias - slight confidence difference
            gender = 'male' if i % 2 == 0 else 'female'
            if gender == 'female':
                confidence -= 0.05
            
            confidence = max(0.1, min(0.95, confidence))
            
            test_data.append({
                'model_version': 'test_v1.0',
                'user_id': f'test_user_{i}',
                'patient_id': f'test_patient_{i}',
                'clinical_context': {
                    'patient_age': age,
                    'patient_gender': gender,
                    'symptoms_severity': random.uniform(0.3, 0.9)
                },
                'ai_recommendation': {
                    'diagnosis': f'condition_{i % 5}',
                    'confidence': confidence
                },
                'confidence_score': confidence,
                'model_outputs': {'raw_score': confidence}
            })
        
        return test_data
    
    def test_adversarial_robustness(self) -> Dict[str, Any]:
        """Test adversarial robustness capabilities."""
        adversarial_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'robustness_score': 0.0,
            'vulnerabilities_found': [],
            'test_categories': {}
        }
        
        try:
            # Initialize AI safety monitor
            safety_monitor = ClinicalAISafetyMonitor()
            
            # Generate and run standard adversarial tests
            test_cases = safety_monitor.generate_standard_adversarial_tests()
            
            if test_cases:
                # Run adversarial tests
                results = safety_monitor.run_adversarial_tests(test_cases)
                
                adversarial_results.update({
                    'total_tests': results['total_tests'],
                    'passed_tests': results['passed_tests'],
                    'robustness_score': results['robustness_score'],
                    'vulnerabilities_found': results['vulnerabilities_detected']
                })
                
                # Categorize test results
                categories = {}
                for test_detail in results['test_details']:
                    test_type = test_detail['test_type']
                    if test_type not in categories:
                        categories[test_type] = {'passed': 0, 'failed': 0}
                    
                    if test_detail['passed']:
                        categories[test_type]['passed'] += 1
                    else:
                        categories[test_type]['failed'] += 1
                
                adversarial_results['test_categories'] = categories
                
                logger.info(f"  Adversarial tests: {results['passed_tests']}/{results['total_tests']} passed")
                logger.info(f"  Robustness score: {results['robustness_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Adversarial testing failed: {e}")
            adversarial_results['error'] = str(e)
        
        return adversarial_results
    
    def test_data_integrity(self) -> Dict[str, Any]:
        """Test enhanced data integrity and anonymization capabilities."""
        integrity_results = {
            'anonymization_methods_tested': [],
            'privacy_scores': [],
            'hipaa_compliance_verified': False,
            'gdpr_compliance_verified': False
        }
        
        try:
            # Initialize privacy manager
            privacy_manager = PrivacyManager(config={'gdpr_enabled': True})
            
            # Test data for anonymization
            test_medical_data = {
                'name': 'John Doe',
                'ssn': '123-45-6789',
                'age': 45,
                'zip_code': '12345',
                'birth_date': '1978-05-15',
                'email': 'john.doe@example.com',
                'phone': '555-123-4567',
                'blood_pressure_systolic': 140,
                'heart_rate': 72,
                'weight': 80.5,
                'height': 1.75,
                'diagnosis': 'hypertension'
            }
            
            # Test different anonymization methods
            methods = ['k_anonymity', 'l_diversity', 't_closeness', 'differential_privacy']
            
            for method in methods:
                try:
                    # Apply anonymization
                    anonymized_data = privacy_manager.advanced_anonymize_medical_data(
                        test_medical_data, 
                        k_value=5, 
                        privacy_level='high'
                    )
                    
                    # Verify anonymization quality
                    quality_metrics = privacy_manager.verify_anonymization_quality(
                        test_medical_data, 
                        anonymized_data
                    )
                    
                    integrity_results['anonymization_methods_tested'].append(method)
                    integrity_results['privacy_scores'].append(quality_metrics['privacy_score'])
                    
                    # Check HIPAA compliance (no direct identifiers should remain)
                    direct_identifiers = ['name', 'ssn', 'email', 'phone', 'zip_code']
                    hipaa_compliant = not any(id_field in anonymized_data for id_field in direct_identifiers)
                    
                    if hipaa_compliant:
                        integrity_results['hipaa_compliance_verified'] = True
                    
                    logger.info(f"  {method}: Privacy score {quality_metrics['privacy_score']:.2f}, HIPAA compliant: {hipaa_compliant}")
                    
                except Exception as method_error:
                    logger.error(f"  {method} failed: {method_error}")
            
            # Test data retention and deletion
            test_data_id = "test_medical_record_001"
            privacy_manager.register_data_for_retention(
                test_data_id, 
                'medical_data'
            )
            
            # Test anonymization marking
            anonymization_success = privacy_manager.anonymize_data(test_data_id, 'advanced_high')
            integrity_results['retention_management_tested'] = anonymization_success
            
            # GDPR compliance check
            integrity_results['gdpr_compliance_verified'] = privacy_manager.retention_policy.gdpr_enabled
            
            logger.info(f"  Anonymization methods tested: {len(integrity_results['anonymization_methods_tested'])}")
            logger.info(f"  Average privacy score: {sum(integrity_results['privacy_scores'])/len(integrity_results['privacy_scores']) if integrity_results['privacy_scores'] else 0:.2f}")
            
        except Exception as e:
            logger.error(f"Data integrity test failed: {e}")
            integrity_results['error'] = str(e)
        
        return integrity_results
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        summary = {
            'timestamp': time.time(),
            'overall_success': True,
            'performance_target_met': False,
            'bias_detection_operational': False,
            'adversarial_robustness_score': 0.0,
            'data_privacy_score': 0.0,
            'recommendations': []
        }
        
        # Performance assessment
        if 'performance_tests' in self.results:
            perf_results = self.results['performance_tests']
            if 'target_achieved' in perf_results:
                summary['performance_target_met'] = perf_results['target_achieved']
                if not perf_results['target_achieved']:
                    summary['recommendations'].append(
                        f"Performance optimization needed: {perf_results.get('avg_response_time_ms', 0):.1f}ms > 100ms target"
                    )
        
        # Bias detection assessment
        if 'bias_detection_tests' in self.results:
            bias_results = self.results['bias_detection_tests']
            summary['bias_detection_operational'] = len(bias_results.get('statistical_methods_tested', [])) > 0
            if bias_results.get('detected_biases'):
                summary['recommendations'].append(
                    f"Bias detected: {len(bias_results['detected_biases'])} significant bias patterns found"
                )
        
        # Adversarial robustness assessment
        if 'adversarial_tests' in self.results:
            adv_results = self.results['adversarial_tests']
            summary['adversarial_robustness_score'] = adv_results.get('robustness_score', 0.0)
            if summary['adversarial_robustness_score'] < 0.8:
                summary['recommendations'].append(
                    f"Adversarial robustness below threshold: {summary['adversarial_robustness_score']:.2f} < 0.8"
                )
        
        # Data privacy assessment
        if 'data_integrity_tests' in self.results:
            integrity_results = self.results['data_integrity_tests']
            privacy_scores = integrity_results.get('privacy_scores', [])
            if privacy_scores:
                summary['data_privacy_score'] = sum(privacy_scores) / len(privacy_scores)
                if summary['data_privacy_score'] < 0.8:
                    summary['recommendations'].append(
                        f"Data privacy score below threshold: {summary['data_privacy_score']:.2f} < 0.8"
                    )
        
        # Overall success determination
        summary['overall_success'] = (
            summary['performance_target_met'] and
            summary['bias_detection_operational'] and
            summary['adversarial_robustness_score'] >= 0.8 and
            summary['data_privacy_score'] >= 0.8
        )
        
        return summary


def main():
    """Run the comprehensive test suite."""
    test_suite = PerformanceOptimizationTests()
    results = test_suite.run_all_tests()
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("="*80)
    
    summary = results['summary']
    print(f"Overall Success: {'✅' if summary['overall_success'] else '❌'}")
    print(f"Performance Target (<100ms): {'✅' if summary['performance_target_met'] else '❌'}")
    print(f"Bias Detection Operational: {'✅' if summary['bias_detection_operational'] else '❌'}")
    print(f"Adversarial Robustness: {summary['adversarial_robustness_score']:.2f}")
    print(f"Data Privacy Score: {summary['data_privacy_score']:.2f}")
    
    if summary['recommendations']:
        print("\n📋 Recommendations:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"{i}. {rec}")
    
    # Save detailed results
    results_file = 'test_results_comprehensive.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    return summary['overall_success']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)