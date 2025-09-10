#!/usr/bin/env python3
"""
Comprehensive Training Test and Report Generator for duetmind_adaptive

This test validates the complete training pipeline and generates a detailed report
of all training components, performance metrics, and system integration.
"""

import unittest
import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Ensure we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

class ComprehensiveTrainingTest(unittest.TestCase):
    """Test suite for comprehensive training system validation"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_start_time = time.time()
        self.test_results = {}
        self.training_modes = ["basic", "kaggle", "simulation", "comprehensive"]
        
    def test_basic_training_mode(self):
        """Test basic training mode functionality"""
        print("🔬 Testing Basic Training Mode...")
        
        try:
            result = subprocess.run([
                sys.executable, "full_training.py", "--mode", "basic"
            ], capture_output=True, text=True, timeout=120, cwd=os.getcwd())
            
            # Check successful execution
            self.assertEqual(result.returncode, 0, f"Basic training failed: {result.stderr}")
            
            # Check for success indicators in output
            self.assertIn("✅ Basic Training Completed!", result.stdout)
            self.assertIn("🎊 SUCCESS: Basic training completed!", result.stdout)
            
            # Check that model file was created
            self.assertTrue(os.path.exists("basic_alzheimer_model.pkl"))
            
            # Extract metrics from output
            lines = result.stdout.split('\n')
            training_accuracy = None
            test_accuracy = None
            
            for line in lines:
                if "Training Accuracy:" in line:
                    training_accuracy = float(line.split(":")[1].strip())
                elif "Test Accuracy:" in line:
                    test_accuracy = float(line.split(":")[1].strip())
            
            # Validate accuracy metrics
            self.assertIsNotNone(training_accuracy, "Training accuracy not found")
            self.assertIsNotNone(test_accuracy, "Test accuracy not found") 
            self.assertGreaterEqual(training_accuracy, 0.8, "Training accuracy too low")
            self.assertGreaterEqual(test_accuracy, 0.7, "Test accuracy too low")
            
            self.test_results["basic"] = {
                "status": "PASSED",
                "training_accuracy": training_accuracy,
                "test_accuracy": test_accuracy,
                "model_file": "basic_alzheimer_model.pkl"
            }
            
            print(f"✅ Basic Training: PASSED (Train: {training_accuracy:.3f}, Test: {test_accuracy:.3f})")
            
        except Exception as e:
            self.test_results["basic"] = {"status": "FAILED", "error": str(e)}
            self.fail(f"Basic training test failed: {e}")
    
    def test_kaggle_training_mode(self):
        """Test Kaggle dataset training mode"""
        print("🌐 Testing Kaggle Training Mode...")
        
        try:
            result = subprocess.run([
                sys.executable, "full_training.py", "--mode", "kaggle"
            ], capture_output=True, text=True, timeout=180, cwd=os.getcwd())
            
            # Check successful execution
            self.assertEqual(result.returncode, 0, f"Kaggle training failed: {result.stderr}")
            
            # Check for success indicators
            self.assertIn("✅ Kaggle Dataset Training Completed!", result.stdout)
            self.assertIn("🎊 SUCCESS: Kaggle training completed!", result.stdout)
            
            # Check for model file creation
            expected_models = ["alzheimer_mri_model.pkl", "files/training/alzheimer_mri_model.pkl"]
            model_found = any(os.path.exists(model) for model in expected_models)
            self.assertTrue(model_found, "Kaggle training model not found")
            
            # Extract accuracy from output
            accuracy = None
            lines = result.stdout.split('\n')
            for line in lines:
                if "Model accuracy:" in line:
                    accuracy = float(line.split(":")[1].strip())
                    break
            
            self.assertIsNotNone(accuracy, "Model accuracy not found in output")
            self.assertGreaterEqual(accuracy, 0.7, "Kaggle model accuracy too low")
            
            self.test_results["kaggle"] = {
                "status": "PASSED",
                "accuracy": accuracy,
                "model_files": [model for model in expected_models if os.path.exists(model)]
            }
            
            print(f"✅ Kaggle Training: PASSED (Accuracy: {accuracy:.3f})")
            
        except Exception as e:
            self.test_results["kaggle"] = {"status": "FAILED", "error": str(e)}
            self.fail(f"Kaggle training test failed: {e}")
    
    def test_simulation_training_mode(self):
        """Test agent simulation training mode"""
        print("🤖 Testing Simulation Training Mode...")
        
        try:
            result = subprocess.run([
                sys.executable, "full_training.py", "--mode", "simulation"
            ], capture_output=True, text=True, timeout=120, cwd=os.getcwd())
            
            # Check successful execution
            self.assertEqual(result.returncode, 0, f"Simulation training failed: {result.stderr}")
            
            # Check for success indicators
            self.assertIn("✅ Simulation Training Completed!", result.stdout)
            self.assertIn("🎊 SUCCESS: Simulation training completed!", result.stdout)
            
            # Check for agent creation
            self.assertIn("👥 Trained", result.stdout)
            self.assertIn("AI agents with ML capabilities", result.stdout)
            
            # Extract agent count
            lines = result.stdout.split('\n')
            agent_count = 0
            for line in lines:
                if "👥 Trained" in line and "AI agents" in line:
                    try:
                        agent_count = int(line.split("👥 Trained")[1].split("AI agents")[0].strip())
                    except:
                        agent_count = 0
                    break
            
            self.assertGreater(agent_count, 0, "No AI agents were created")
            
            self.test_results["simulation"] = {
                "status": "PASSED",
                "agent_count": agent_count,
                "model_file": "alzheimer_model.pkl" if os.path.exists("alzheimer_model.pkl") else None
            }
            
            print(f"✅ Simulation Training: PASSED ({agent_count} agents created)")
            
        except Exception as e:
            self.test_results["simulation"] = {"status": "FAILED", "error": str(e)}
            self.fail(f"Simulation training test failed: {e}")
    
    def test_comprehensive_training_mode(self):
        """Test full comprehensive training mode"""
        print("🚀 Testing Comprehensive Training Mode...")
        
        try:
            result = subprocess.run([
                sys.executable, "full_training.py", "--mode", "comprehensive"
            ], capture_output=True, text=True, timeout=300, cwd=os.getcwd())
            
            # Check successful execution
            self.assertEqual(result.returncode, 0, f"Comprehensive training failed: {result.stderr}")
            
            # Check for overall success
            self.assertIn("🎉 FULL TRAINING COMPLETED SUCCESSFULLY!", result.stdout)
            self.assertIn("📊 Success Rate: 100%", result.stdout)
            self.assertIn("🎊 SUCCESS: Comprehensive training completed!", result.stdout)
            
            # Check that all components passed
            self.assertIn("BASIC: ✅ PASSED", result.stdout)
            self.assertIn("KAGGLE: ✅ PASSED", result.stdout)
            self.assertIn("SIMULATION: ✅ PASSED", result.stdout)
            
            # Check for model files
            expected_models = [
                "basic_alzheimer_model.pkl",
                "alzheimer_mri_model.pkl",
                "alzheimer_model.pkl"
            ]
            
            created_models = [model for model in expected_models if os.path.exists(model)]
            self.assertGreater(len(created_models), 0, "No models were created")
            
            self.test_results["comprehensive"] = {
                "status": "PASSED",
                "success_rate": "100%",
                "components_passed": ["basic", "kaggle", "simulation"],
                "models_created": created_models
            }
            
            print(f"✅ Comprehensive Training: PASSED (All components successful)")
            
        except Exception as e:
            self.test_results["comprehensive"] = {"status": "FAILED", "error": str(e)}
            self.fail(f"Comprehensive training test failed: {e}")
    
    def test_training_module_imports(self):
        """Test that all training modules can be imported correctly"""
        print("📦 Testing Training Module Imports...")
        
        try:
            # Test basic imports
            from files.files.training.training import AlzheimerTrainer, TrainingIntegratedAgent, run_training_simulation
            
            # Test trainer instantiation
            trainer = AlzheimerTrainer()
            self.assertIsNotNone(trainer)
            
            # Test data loading
            df = trainer.load_data()
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            
            print("✅ Module Imports: PASSED")
            
            self.test_results["imports"] = {
                "status": "PASSED",
                "modules": ["AlzheimerTrainer", "TrainingIntegratedAgent", "run_training_simulation"]
            }
            
        except Exception as e:
            self.test_results["imports"] = {"status": "FAILED", "error": str(e)}
            self.fail(f"Module import test failed: {e}")
    
    def test_model_persistence(self):
        """Test that trained models can be saved and loaded"""
        print("💾 Testing Model Persistence...")
        
        try:
            from files.files.training.training import AlzheimerTrainer
            
            # Create trainer and train model
            trainer = AlzheimerTrainer()
            df = trainer.load_data()
            X, y = trainer.preprocess_data(df)
            results = trainer.train_model(X, y)
            
            # Save model
            test_model_path = "test_persistence_model.pkl"
            trainer.save_model(test_model_path)
            
            # Verify file was created
            self.assertTrue(os.path.exists(test_model_path))
            
            # Load model in new trainer instance
            new_trainer = AlzheimerTrainer()
            new_trainer.load_model(test_model_path)
            
            # Verify model was loaded
            self.assertIsNotNone(new_trainer.model)
            self.assertEqual(new_trainer.feature_columns, trainer.feature_columns)
            
            # Clean up
            if os.path.exists(test_model_path):
                os.remove(test_model_path)
            
            print("✅ Model Persistence: PASSED")
            
            self.test_results["persistence"] = {
                "status": "PASSED",
                "test_accuracy": results["test_accuracy"]
            }
            
        except Exception as e:
            self.test_results["persistence"] = {"status": "FAILED", "error": str(e)}
            self.fail(f"Model persistence test failed: {e}")


class ComprehensiveTrainingReport:
    """Generate comprehensive training test report"""
    
    def __init__(self, test_results: Dict[str, Any]):
        self.test_results = test_results
        self.timestamp = datetime.now()
    
    def generate_text_report(self) -> str:
        """Generate human-readable text report"""
        report_lines = [
            "=" * 80,
            "📋 DUETMIND ADAPTIVE - COMPREHENSIVE TRAINING TEST REPORT",
            "=" * 80,
            f"🕐 Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"🧪 Test Environment: Python {sys.version.split()[0]}",
            "",
            "📊 TRAINING COMPONENT RESULTS:",
            "-" * 40
        ]
        
        # Component results
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in self.test_results.items():
            total_tests += 1
            status = result.get("status", "UNKNOWN")
            
            if status == "PASSED":
                passed_tests += 1
                icon = "✅"
            else:
                icon = "❌"
            
            report_lines.append(f"{icon} {test_name.upper()}: {status}")
            
            # Add specific metrics
            if test_name == "basic" and status == "PASSED":
                report_lines.append(f"   📈 Training Accuracy: {result.get('training_accuracy', 'N/A'):.3f}")
                report_lines.append(f"   📈 Test Accuracy: {result.get('test_accuracy', 'N/A'):.3f}")
                
            elif test_name == "kaggle" and status == "PASSED":
                report_lines.append(f"   📈 Model Accuracy: {result.get('accuracy', 'N/A'):.3f}")
                report_lines.append(f"   💾 Models: {len(result.get('model_files', []))} created")
                
            elif test_name == "simulation" and status == "PASSED":
                report_lines.append(f"   🤖 AI Agents: {result.get('agent_count', 0)} created")
                
            elif test_name == "comprehensive" and status == "PASSED":
                report_lines.append(f"   📊 Success Rate: {result.get('success_rate', 'N/A')}")
                report_lines.append(f"   🔧 Components: {len(result.get('components_passed', []))} passed")
                report_lines.append(f"   💾 Models: {len(result.get('models_created', []))} created")
                
            elif test_name == "persistence" and status == "PASSED":
                report_lines.append(f"   📈 Test Accuracy: {result.get('test_accuracy', 'N/A'):.3f}")
            
            if status == "FAILED" and "error" in result:
                report_lines.append(f"   ❌ Error: {result['error']}")
            
            report_lines.append("")
        
        # Summary
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report_lines.extend([
            "📈 OVERALL SUMMARY:",
            "-" * 40,
            f"🧪 Total Tests: {total_tests}",
            f"✅ Passed: {passed_tests}",
            f"❌ Failed: {total_tests - passed_tests}",
            f"📊 Success Rate: {success_rate:.1f}%",
            "",
        ])
        
        if success_rate == 100:
            report_lines.extend([
                "🎉 COMPREHENSIVE TRAINING TEST: FULLY SUCCESSFUL!",
                "✨ All training components are working correctly.",
                "🚀 System is ready for production deployment.",
            ])
        elif success_rate >= 80:
            report_lines.extend([
                "⚠️  COMPREHENSIVE TRAINING TEST: MOSTLY SUCCESSFUL",
                "🔧 Some components may need attention.",
                "📋 Review failed tests and address issues.",
            ])
        else:
            report_lines.extend([
                "💥 COMPREHENSIVE TRAINING TEST: NEEDS ATTENTION",
                "🛠️  Multiple components require fixes.",
                "🔍 Detailed investigation recommended.",
            ])
        
        report_lines.extend([
            "",
            "=" * 80,
            "📋 End of Report",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def generate_json_report(self) -> Dict[str, Any]:
        """Generate machine-readable JSON report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result.get("status") == "PASSED")
        
        return {
            "report_metadata": {
                "timestamp": self.timestamp.isoformat(),
                "python_version": sys.version.split()[0],
                "test_type": "comprehensive_training"
            },
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            "component_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [name for name, result in self.test_results.items() 
                       if result.get("status") == "FAILED"]
        
        if not failed_tests:
            recommendations.append("System is fully operational and ready for production use.")
            recommendations.append("Consider setting up automated testing for continuous validation.")
        else:
            for failed_test in failed_tests:
                if failed_test == "basic":
                    recommendations.append("Review basic training pipeline and test data generation.")
                elif failed_test == "kaggle":
                    recommendations.append("Check Kaggle API credentials and network connectivity.")
                elif failed_test == "simulation":
                    recommendations.append("Verify agent simulation framework and dependencies.")
                elif failed_test == "comprehensive":
                    recommendations.append("Review comprehensive training orchestration logic.")
        
        return recommendations
    
    def save_reports(self, base_filename: str = "comprehensive_training_report"):
        """Save both text and JSON reports to files"""
        # Text report
        text_report = self.generate_text_report()
        text_filename = f"{base_filename}.txt"
        with open(text_filename, 'w') as f:
            f.write(text_report)
        
        # JSON report
        json_report = self.generate_json_report()
        json_filename = f"{base_filename}.json"
        with open(json_filename, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        return text_filename, json_filename


def run_comprehensive_training_test():
    """Run comprehensive training test suite and generate report"""
    print("🧠 DuetMind Adaptive - Comprehensive Training Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(ComprehensiveTrainingTest)
    
    # Custom test result collector
    class TestResultCollector(unittest.TextTestRunner):
        def run(self, test):
            result = super().run(test)
            
            # Extract test results from the test instance
            test_results = {}
            for test_case in result.testsRun:
                # This is a simplified approach - in practice, you'd want more robust result collection
                pass
            
            return result
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Collect results (this is a simplified version - we'll get results from the test methods)
    # In a real implementation, you'd want to use a custom test result class
    # For now, we'll run the tests manually to collect results properly
    
    test_instance = ComprehensiveTrainingTest()
    test_instance.setUp()
    
    collected_results = {}
    
    # Run each test and collect results
    test_methods = [
        "test_training_module_imports",
        "test_model_persistence", 
        "test_basic_training_mode",
        "test_kaggle_training_mode",
        "test_simulation_training_mode",
        "test_comprehensive_training_mode"
    ]
    
    for method_name in test_methods:
        try:
            print(f"\n🧪 Running {method_name}...")
            method = getattr(test_instance, method_name)
            method()
            print(f"✅ {method_name}: PASSED")
        except Exception as e:
            print(f"❌ {method_name}: FAILED - {e}")
            # Update test_instance.test_results if needed
    
    # Generate and save reports
    report_generator = ComprehensiveTrainingReport(test_instance.test_results)
    
    # Print text report
    text_report = report_generator.generate_text_report()
    print("\n" + text_report)
    
    # Save reports
    text_file, json_file = report_generator.save_reports()
    print(f"\n📄 Reports saved:")
    print(f"   📝 Text Report: {text_file}")
    print(f"   📊 JSON Report: {json_file}")
    
    # Return success status
    total_tests = len(test_instance.test_results)
    passed_tests = sum(1 for result in test_instance.test_results.values() 
                      if result.get("status") == "PASSED")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    return success_rate >= 80  # Consider 80%+ as success


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    success = run_comprehensive_training_test()
    sys.exit(0 if success else 1)