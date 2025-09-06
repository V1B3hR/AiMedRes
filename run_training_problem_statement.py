#!/usr/bin/env python3
"""
Run training - Exact Problem Statement Implementation
This implements the exact code from the problem statement with the correct file_path
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "oasis_longitudinal.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "jboysen/mri-and-alzheimers",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())

# Additional training functionality
if __name__ == "__main__":
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nDataset info:")
    print(df.info())
    print("\nDataset statistics:")
    print(df.describe())
    print("\n✓ Problem statement implementation successful!")
    print("✓ Training data loaded and ready for processing!")