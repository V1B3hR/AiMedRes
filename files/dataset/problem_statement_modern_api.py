#!/usr/bin/env python3
"""
Corrected implementation using the modern API to avoid deprecated warnings
This implements the problem statement but with the updated kagglehub API
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
# Fixed the malformed assignment from the problem statement
file_path = "Alzheimer Disease and Healthy Aging Data In US.csv"

# Load the latest version using modern API (no deprecation warning)
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "ananthu19/alzheimer-disease-and-healthy-aging-data-in-us",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documentation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
  pandas_kwargs={'encoding': 'latin-1'}
)

print("First 5 records:", df.head())