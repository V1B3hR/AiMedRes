#!/usr/bin/env python3
"""
Test for the training problem statement implementation
"""

import sys
import os
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestTrainingProblemStatement(unittest.TestCase):
    """Test cases for the problem statement implementation"""
    
    def test_kagglehub_imports(self):
        """Test that kagglehub imports work correctly"""
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        self.assertTrue(hasattr(kagglehub, 'dataset_download'))
        self.assertTrue(hasattr(KaggleDatasetAdapter, 'PANDAS'))
        
    def test_training_problem_statement_execution(self):
        """Test that the problem statement script can be executed"""
        import subprocess
        import sys
        
        # Run the training script and capture output
        result = subprocess.run([
            sys.executable, 
            'training_problem_statement.py'
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
        self.assertIn("Dataset summary:", result.stdout)
        self.assertIn("Total records:", result.stdout)
        
        # Check that it shows the expected dataset structure
        self.assertIn("Classes:", result.stdout)
        self.assertIn("Split distribution:", result.stdout)
        
        # Check for the expected classes in Alzheimer's MRI dataset
        output = result.stdout
        self.assertIn("No Impairment", output)
        self.assertIn("Mild Impairment", output)
        self.assertIn("Moderate Impairment", output)
        self.assertIn("Very Mild Impairment", output)

    def test_dataframe_creation_from_images(self):
        """Test the dataframe creation logic"""
        # Import the function from the script
        import importlib.util
        script_path = os.path.join(os.path.dirname(__file__), 'training_problem_statement.py')
        spec = importlib.util.spec_from_file_location("training_problem_statement", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Create a mock dataset structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock dataset structure
            combined_dir = os.path.join(temp_dir, "Combined Dataset")
            os.makedirs(combined_dir)
            
            for split in ["train", "test"]:
                split_dir = os.path.join(combined_dir, split)
                os.makedirs(split_dir)
                
                for class_name in ["No Impairment", "Mild Impairment"]:
                    class_dir = os.path.join(split_dir, class_name)
                    os.makedirs(class_dir)
                    
                    # Create mock image files
                    for i in range(2):
                        image_path = os.path.join(class_dir, f"image_{i}.jpg")
                        with open(image_path, 'w') as f:
                            f.write("mock image")
            
            # Test the function
            df = module.create_dataframe_from_images(temp_dir)
            
            # Verify the DataFrame structure
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            
            expected_columns = ['image_id', 'image_filename', 'image_path', 'class', 'split', 'class_encoded']
            for col in expected_columns:
                self.assertIn(col, df.columns)
            
            # Check that we have the expected splits and classes
            self.assertTrue(set(df['split'].unique()).issubset({'train', 'test'}))
            self.assertTrue(set(df['class'].unique()).issubset({'No Impairment', 'Mild Impairment'}))

if __name__ == '__main__':
    unittest.main()