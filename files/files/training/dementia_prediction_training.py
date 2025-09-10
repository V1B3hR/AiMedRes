#!/usr/bin/env python3
"""
Exact implementation of the problem statement requirements.
This script loads the dementia prediction dataset as specified.
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Handle empty file_path by auto-detecting the dataset file
if not file_path:
    # Download dataset to inspect available files
    dataset_path = kagglehub.dataset_download("shashwatwork/dementia-prediction-dataset")
    import os
    files = os.listdir(dataset_path)
    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        file_path = csv_files[0]  # Use the first CSV file found
        print(f"Auto-detected file: {file_path}")
    else:
        raise ValueError("No CSV files found in the dataset")

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "shashwatwork/dementia-prediction-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())