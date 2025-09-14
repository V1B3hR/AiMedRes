#!/usr/bin/env python3
"""
Test for the dementia prediction dataset loading functionality.
Tests the exact problem statement implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))

import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

class TestDementiaPredictionDataset(unittest.TestCase):
    """Test dementia prediction dataset loading"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data similar to conftest.py fixture
        self.sample_data = pd.DataFrame({
            'Subject ID': ['S001', 'S002', 'S003', 'S004', 'S005'],
            'Group': ['Demented', 'Nondemented', 'Demented', 'Converted', 'Nondemented'],
            'Age': [87, 75, 88, 75, 71],
            'EDUC': [14, 12, 18, 12, 12],
            'SES': [2, 1, 3, 2, 2],
            'MMSE': [27, 29, 20, 23, 28],
            'CDR': [0.5, 0, 1, 0.5, 0],
            'eTIV': [1987, 1678, 1215, 1689, 1357],
            'nWBV': [0.696, 0.736, 0.710, 0.712, 0.748],
            'ASF': [1.046, 0.963, 1.365, 0.978, 1.215]
        })
    
    def test_dataset_loading(self):
        """Test that the dataset loads correctly"""
        # Import and run the problem statement implementation
        import subprocess
        import tempfile
        
        # Run the training script and capture output
        result = subprocess.run([
            'python3', 
            'training/problem_statement_exact.py'
        ], capture_output=True, text=True, cwd='/home/runner/work/duetmind_adaptive/duetmind_adaptive')
        
        # Check that script ran successfully
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")
        
        # Check that output contains expected data
        self.assertIn("First 5 records:", result.stdout)
        self.assertIn("Subject ID", result.stdout)
        self.assertIn("Group", result.stdout)
        
    @patch('builtins.__import__')
    def test_dataset_structure(self, mock_import):
        """Test that the loaded dataset has expected structure (mocked)"""
        # Setup mock kagglehub module
        mock_kagglehub = MagicMock()
        mock_kagglehub.dataset_download.return_value = "/mock/path/dataset"
        mock_kagglehub.load_dataset.return_value = self.sample_data
        mock_kagglehub.KaggleDatasetAdapter.PANDAS = "pandas"
        
        # Mock os.listdir to return a CSV file
        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = ['dementia_dataset.csv']
            
            def mock_import_func(name, *args, **kwargs):
                if name == 'kagglehub':
                    return mock_kagglehub
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = mock_import_func
            
            # Now run the test with mocked kagglehub
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
            
            # Auto-detect file (mocked)
            dataset_path = kagglehub.dataset_download("shashwatwork/dementia-prediction-dataset")
            files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
            file_path = files[0] if files else "dementia_dataset.csv"
            
            # Load dataset (mocked)
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
            
            # Verify mocks were called correctly
            mock_kagglehub.dataset_download.assert_called_once_with("shashwatwork/dementia-prediction-dataset")
            mock_kagglehub.load_dataset.assert_called_once()

if __name__ == '__main__':
    unittest.main()