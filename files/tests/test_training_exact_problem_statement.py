#!/usr/bin/env python3
"""
Test for the exact problem statement implementation
"""

import sys
import os
import tempfile
import unittest
from unittest.mock import patch

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestTrainingExactProblemStatement(unittest.TestCase):
    """Test cases for the exact problem statement implementation"""
    
    def test_kagglehub_imports(self):
        """Test that kagglehub imports work correctly"""
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        self.assertTrue(hasattr(kagglehub, 'load_dataset'))
        self.assertTrue(hasattr(kagglehub, 'dataset_load'))
        self.assertTrue(hasattr(kagglehub, 'dataset_download'))
        
    def test_training_exact_problem_statement_execution(self):
        """Test that the exact problem statement script can be executed"""
        import subprocess
        import sys
        
        # Run the training script and capture output
        result = subprocess.run([
            sys.executable, 
            'training_exact_problem_statement.py'
        ], 
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True, 
        text=True,
        timeout=180
        )
        
        # Check that the script ran successfully
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        
        # Check that it printed the expected output
        self.assertIn("First 5 records:", result.stdout)
        
        # Check that it loaded data with expected structure
        lines = result.stdout.strip().split('\n')
        # Should have multiple lines of output including the dataframe
        self.assertGreater(len(lines), 3, "Expected multiple lines of output")

if __name__ == '__main__':
    unittest.main()