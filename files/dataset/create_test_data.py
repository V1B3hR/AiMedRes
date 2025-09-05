#!/usr/bin/env python3
"""
Test script to simulate Alzheimer dataset structure for testing
"""

import pandas as pd
import os

def create_test_alzheimer_data():
    """Create a test Alzheimer dataset for validation."""
    
    # Create sample data that would be typical for an Alzheimer features dataset
    data = {
        'age': [65, 72, 58, 81, 69, 75, 63, 79],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'education_level': [16, 12, 18, 14, 20, 10, 15, 13],
        'mmse_score': [28, 24, 29, 20, 26, 18, 27, 22],
        'cdr_score': [0.0, 0.5, 0.0, 1.0, 0.0, 2.0, 0.5, 1.0],
        'apoe_genotype': ['E3/E3', 'E3/E4', 'E2/E3', 'E4/E4', 'E3/E3', 'E3/E4', 'E2/E3', 'E3/E4'],
        'diagnosis': ['Normal', 'MCI', 'Normal', 'Dementia', 'Normal', 'Dementia', 'MCI', 'Dementia']
    }
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    df.to_csv('alzheimer_features_test.csv', index=False)
    print(f"Created test dataset with {len(df)} rows and {len(df.columns)} columns")
    print("Columns:", df.columns.tolist())
    print("First 5 records:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    create_test_alzheimer_data()