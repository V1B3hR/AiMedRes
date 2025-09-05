#!/usr/bin/env python3
"""
Simple Kaggle Dataset Loader - matches the exact problem statement format
Updated to use the modern kagglehub API
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
# For most datasets, specify the main CSV file name
# Common names for Alzheimer datasets include "alzheimer.csv", "data.csv", etc.
file_path = "alzheimer.csv"

# Load the latest version using the modern API
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "brsdincer/alzheimer-features",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documentation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())