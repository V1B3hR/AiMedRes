#!/usr/bin/env python3
"""
Corrected implementation of the exact problem statement
This fixes the syntax errors and uses the dataset specified in the problem statement
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
# Fixed the malformed assignment from the problem statement
file_path = "Alzheimer Disease and Healthy Aging Data In US.csv"

# Load the latest version
# Using the dataset name from the problem statement
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "ananthu19/alzheimer-disease-and-healthy-aging-data-in-us",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())