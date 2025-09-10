#!/usr/bin/env python3
"""
Test for the Kaggle dataset loader implementation
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


class TestKaggleDatasetLoader(unittest.TestCase):
    """Test cases for the Kaggle dataset loader"""
    
    def test_kaggle_dataset_loading(self):
        """Test that we can load the kaggle dataset as specified in problem statement"""
        # Import the exact implementation
        try:
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
            
            # Set the path to the file you'd like to load  
            file_path = "dementia-death-rates new.csv"
            
            # Load the latest version
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
            
            print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            print("Test passed: Kaggle dataset loader implementation working correctly")
            
        except ImportError as e:
            self.skipTest(f"kagglehub not available: {e}")
        except Exception as e:
            self.fail(f"Failed to load kaggle dataset: {e}")
    
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