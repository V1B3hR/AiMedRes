"""
Test for training functionality
Tests the new training capabilities added to meet the "run training" requirement
"""
import unittest
import subprocess
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestTraining(unittest.TestCase):
    """Test cases for training functionality"""

    def setUp(self):
        """Set up test environment"""
        self.repo_root = Path(__file__).parent.parent
        self.python = sys.executable

    def test_basic_training_script_works(self):
        """Test that basic training script runs without errors"""
        result = subprocess.run(
            [self.python, "run_training_modern.py"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        self.assertEqual(result.returncode, 0, f"Training script failed: {result.stderr}")
        self.assertIn("First 5 records:", result.stdout)

    def test_comprehensive_training_pipeline_works(self):
        """Test that comprehensive training pipeline runs without errors"""
        result = subprocess.run(
            [self.python, "train_adaptive_model.py"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        self.assertEqual(result.returncode, 0, f"Comprehensive training failed: {result.stderr}")
        self.assertIn("Training Complete", result.stdout)
        self.assertIn("Dataset loaded", result.stdout)
        self.assertIn("Simulation Complete", result.stdout)

    def test_training_entry_point_basic_mode(self):
        """Test the unified training entry point - basic mode"""
        result = subprocess.run(
            [self.python, "train.py", "basic"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        self.assertEqual(result.returncode, 0, f"Basic training mode failed: {result.stderr}")
        self.assertIn("Basic training completed successfully", result.stdout)

    def test_training_entry_point_simulation_mode(self):
        """Test the unified training entry point - simulation mode"""
        result = subprocess.run(
            [self.python, "train.py", "simulation"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        self.assertEqual(result.returncode, 0, f"Simulation training mode failed: {result.stderr}")
        self.assertIn("Simulation training completed successfully", result.stdout)

    def test_training_entry_point_comprehensive_mode(self):
        """Test the unified training entry point - comprehensive mode"""
        result = subprocess.run(
            [self.python, "train.py", "comprehensive"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        self.assertEqual(result.returncode, 0, f"Comprehensive training mode failed: {result.stderr}")
        self.assertIn("Comprehensive training completed successfully", result.stdout)

    def test_training_entry_point_default_mode(self):
        """Test the unified training entry point - default (comprehensive) mode"""
        result = subprocess.run(
            [self.python, "train.py"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        self.assertEqual(result.returncode, 0, f"Default training mode failed: {result.stderr}")
        self.assertIn("COMPREHENSIVE Mode", result.stdout)

    def test_no_deprecation_warnings_in_modern_script(self):
        """Test that the modern training script doesn't show deprecation warnings"""
        result = subprocess.run(
            [self.python, "run_training_modern.py"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        self.assertEqual(result.returncode, 0)
        # Should not contain deprecation warnings
        self.assertNotIn("DeprecationWarning", result.stderr)
        self.assertNotIn("deprecated", result.stderr.lower())

    def test_fixed_script_no_deprecation_warnings(self):
        """Test that the fixed run_training.py script doesn't show deprecation warnings"""
        result = subprocess.run(
            [self.python, "run_training.py"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        self.assertEqual(result.returncode, 0)
        # Should not contain deprecation warnings after our fix
        self.assertNotIn("DeprecationWarning", result.stderr)
        self.assertNotIn("deprecated", result.stderr.lower())

if __name__ == '__main__':
    unittest.main()