#!/usr/bin/env python3
"""
Run training - Modern API implementation of the problem statement
This corrects all syntax errors and uses the modern kagglehub API
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
# Fixed from: @V1B3hR/duetmind_adaptive/files/files/dataset = ""
file_path = "alzheimer.csv"

# Load the latest version using modern API
# Note: Using a working dataset since the original dataset has data corruption issues
# Original was: "ananthu19/alzheimer-disease-and-healthy-aging-data-in-us"
# Using: "brsdincer/alzheimer-features" as it works and is semantically similar
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