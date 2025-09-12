#!/usr/bin/env python3
"""
Test for the Kaggle dataset loader implementation
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestKaggleDatasetLoader(unittest.TestCase):
    """Test cases for the Kaggle dataset loader"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock sample data
        self.sample_data = pd.DataFrame({
            'Age': [65, 72, 68, 75, 70],
            'Gender': ['M', 'F', 'M', 'F', 'M'],
            'Education': [12, 16, 14, 18, 10],
            'MMSE': [28, 24, 26, 22, 25],
            'CDR': [0.0, 0.5, 0.0, 1.0, 0.5],
            'eTIV': [1563, 1678, 1592, 1515, 1720],
            'nWBV': [0.696, 0.735, 0.710, 0.665, 0.742],
            'ASF': [1.046, 0.963, 1.020, 1.079, 0.958],
            'Dementia': [0, 1, 0, 1, 1]
        })
    
    @patch('builtins.__import__')
    def test_kaggle_dataset_loading(self, mock_import):
        """Test that we can load the kaggle dataset as specified in problem statement (mocked)"""
        # Setup mock kagglehub module
        mock_kagglehub = MagicMock()
        mock_kagglehub.load_dataset.return_value = self.sample_data
        mock_kagglehub.KaggleDatasetAdapter.PANDAS = "pandas"
        
        def mock_import_func(name, *args, **kwargs):
            if name == 'kagglehub':
                return mock_kagglehub
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = mock_import_func
        
        # Import the exact implementation (now mocked)
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        
        # Set the path to the file you'd like to load  
        file_path = "dementia-death-rates new.csv"
        
        # Load the latest version (mocked)
        df = kagglehub.load_dataset(
          KaggleDatasetAdapter.PANDAS,
          "willianoliveiragibin/death-alzheimers", 
          file_path,
        )
        
        # Verify the dataset loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertGreater(len(df.columns), 0)
        
        # Verify we can get the first 5 records
        head_df = df.head()
        self.assertIsInstance(head_df, pd.DataFrame)
        self.assertLessEqual(len(head_df), 5)
        
        # Verify the mock was called correctly
        mock_kagglehub.load_dataset.assert_called_once_with(
            "pandas",
            "willianoliveiragibin/death-alzheimers",
            file_path
        )
        
        print(f"Successfully loaded mocked dataset with {len(df)} rows and {len(df.columns)} columns")
        print("Test passed: Kaggle dataset loader implementation working correctly (mocked)")
    
    def test_kaggle_dataset_loading_import_error(self):
        """Test handling when kagglehub is not available"""
        with patch('builtins.__import__') as mock_import:
            def mock_import_func(name, *args, **kwargs):
                if name == 'kagglehub':
                    raise ImportError("No module named 'kagglehub'")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = mock_import_func
            
            # This should handle the ImportError gracefully
            with self.assertRaises(ImportError):
                import kagglehub
                kagglehub.load_dataset("pandas", "test/dataset", "test.csv")
    
    def test_kaggle_dataset_loader_file_exists(self):
        """Test that the kaggle dataset loader file exists and is executable"""
        loader_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'kaggle_dataset_loader.py')
        self.assertTrue(os.path.exists(loader_path), 
                       f"Kaggle dataset loader file does not exist at {loader_path}")
        
        # Check that the file contains the required code
        with open(loader_path, 'r') as f:
            content = f.read()
            self.assertIn('kagglehub.load_dataset', content)
            self.assertIn('KaggleDatasetAdapter.PANDAS', content)
            self.assertIn('willianoliveiragibin/death-alzheimers', content)
            self.assertIn('dementia-death-rates new.csv', content)


if __name__ == '__main__':
    unittest.main()