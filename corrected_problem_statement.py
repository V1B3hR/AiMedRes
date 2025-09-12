#!/usr/bin/env python3
"""
Deep learning, training.

CORRECTED VERSION of the problem statement code.
This fixes the issues in the original code while maintaining the same structure.
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
# CORRECTED: Fixed the dataset name and API usage
# Original problem had: "borhanitrash/alzheimer-mri-disease-classification-dataset" (doesn't exist)
# Also removed the invalid "@V1B3hR/duetmind_adaptive" line that was in the function call
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "brsdincer/alzheimer-features",  # Working Alzheimer dataset
  "alzheimer.csv",  # Specify the CSV file
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documentation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())