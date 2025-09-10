#!/usr/bin/env python3
"""
Run training - Problem Statement Implementation (Clean Version)
This implements the exact code from the problem statement with deprecation warning suppressed
"""

import warnings

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Suppress the deprecation warning for the exact problem statement compliance
warnings.filterwarnings('ignore', category=DeprecationWarning)

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
    print(f"\nDataset successfully loaded!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Basic data exploration
    print(f"\nTarget variable distribution (Group):")
    print(df['Group'].value_counts())
    
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    print("\n✓ Problem statement implementation successful!")
    print("✓ Training data loaded and ready for machine learning!")