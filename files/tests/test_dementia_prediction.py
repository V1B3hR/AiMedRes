#!/usr/bin/env python3
"""
Test for the dementia prediction dataset loading functionality.
Tests the exact problem statement implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd

class TestDementiaPredictionDataset(unittest.TestCase):
    """Test dementia prediction dataset loading"""
    
    def test_dataset_loading(self):
        """Test that the dataset loads correctly"""
        # Import and run the problem statement implementation
        import subprocess
        import tempfile
        
        # Run the training script and capture output
        result = subprocess.run([
            'python3', 
            'files/training/problem_statement_exact.py'
        ], capture_output=True, text=True, cwd='/home/runner/work/duetmind_adaptive/duetmind_adaptive')
        
        # Check that script ran successfully
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")
        
        # Check that output contains expected data
        self.assertIn("First 5 records:", result.stdout)
        self.assertIn("Subject ID", result.stdout)
        self.assertIn("Group", result.stdout)
        
    def test_dataset_structure(self):
        """Test that the loaded dataset has expected structure"""
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        
        # Auto-detect file
        dataset_path = kagglehub.dataset_download("shashwatwork/dementia-prediction-dataset")
        files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        file_path = files[0] if files else "dementia_dataset.csv"
        
        # Load dataset
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "shashwatwork/dementia-prediction-dataset",
            file_path
        )
        
        # Test dataset properties
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0, "Dataset should not be empty")
        self.assertIn("Group", df.columns, "Dataset should have Group column")
        self.assertIn("Subject ID", df.columns, "Dataset should have Subject ID column")

if __name__ == '__main__':
    unittest.main()