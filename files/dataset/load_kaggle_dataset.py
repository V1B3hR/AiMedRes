#!/usr/bin/env python3
"""
Kaggle Dataset Loader for duetmind_adaptive
Loads the Alzheimer features dataset using kagglehub
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

# Set the path to the file you'd like to load
# For the alzheimer-features dataset, try common file names
file_path = ""

def load_alzheimer_dataset():
    """Load the Alzheimer features dataset with automatic file detection."""
    
    # Common file names for Alzheimer datasets
    common_files = [
        "",  # Try loading the entire dataset first
        "alzheimer.csv",
        "alzheimers.csv", 
        "data.csv",
        "alzheimer_features.csv",
        "alzheimer_data.csv"
    ]
    
    dataset_handle = "brsdincer/alzheimer-features"
    
    for file_name in common_files:
        try:
            print(f"Attempting to load file: '{file_name}' from dataset...")
            
            # Load the latest version using modern API
            df = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                dataset_handle,
                file_name,
                # Provide any additional arguments like 
                # sql_query or pandas_kwargs. See the 
                # documentation for more information:
                # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
            )
            
            print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            print("Column names:", df.columns.tolist())
            print("First 5 records:")
            print(df.head())
            return df
            
        except Exception as e:
            print(f"Failed to load '{file_name}': {e}")
            continue
    
    print("Could not load any file from the dataset. Please check the dataset structure.")
    return None

if __name__ == "__main__":
    # Load the dataset
    df = load_alzheimer_dataset()
    
    if df is not None:
        print("\nDataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
    else:
        print("\nFailed to load the dataset.")