#!/usr/bin/env python3
"""
Direct implementation matching the exact problem statement format.
This preserves the original API call as requested with necessary file detection.
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Auto-detect file when empty (required for kagglehub to work)
if not file_path:
    import os
    dataset_path = kagglehub.dataset_download("shashwatwork/dementia-prediction-dataset")
    files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    file_path = files[0] if files else "dementia_dataset.csv"

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