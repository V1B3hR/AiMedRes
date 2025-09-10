#!/usr/bin/env python3
"""
Exact implementation of the problem statement requirements.
This script loads the Alzheimer's disease and healthy aging dataset exactly as specified.
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import warnings

# Set the path to the file you'd like to load
file_path = ""

# Handle empty file_path by auto-detecting the CSV file
if not file_path:
    import os
    dataset_path = kagglehub.dataset_download("daniilkrasnoproshin/alzheimers-disease-and-healthy-aging-data")
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    if csv_files:
        file_path = csv_files[0]

# Load the latest version
# Suppress deprecation warning for the exact API call from problem statement
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    df = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "daniilkrasnoproshin/alzheimers-disease-and-healthy-aging-data",
      file_path,
      # Provide any additional arguments like 
      # sql_query or pandas_kwargs. See the 
      # documenation for more information:
      # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

print("First 5 records:", df.head())