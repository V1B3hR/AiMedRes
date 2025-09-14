#!/usr/bin/env python3
"""
Updated to use the correct dataset from the problem statement.
This script loads the exact dataset specified: lukechugh/best-alzheimer-mri-dataset-99-accuracy
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os
import warnings

# Set the path to the file you'd like to load
file_path = ""

# Since the dataset contains images not CSV, we'll create a DataFrame from the structure
def create_dataframe_from_images(dataset_path):
    """Create a DataFrame from the image dataset structure"""
    data = []
    combined_path = os.path.join(dataset_path, "Combined Dataset")
    
    for split in ["train", "test"]:
        split_path = os.path.join(combined_path, split)
        if os.path.exists(split_path):
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    # Count images in each class
                    image_files = [f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    
                    for i, image_file in enumerate(image_files):
                        # Create a record for each image with metadata
                        data.append({
                            'image_id': f"{split}_{class_name}_{i+1:04d}",
                            'image_filename': image_file,
                            'image_path': os.path.join(split, class_name, image_file),
                            'class': class_name,
                            'split': split,
                            'class_encoded': ['No Impairment', 'Very Mild Impairment', 
                                            'Mild Impairment', 'Moderate Impairment'].index(class_name) if class_name in 
                                            ['No Impairment', 'Very Mild Impairment', 'Mild Impairment', 'Moderate Impairment'] else -1
                        })
    
    return pd.DataFrame(data)

# Download the dataset and create DataFrame
try:
    dataset_path = kagglehub.dataset_download("lukechugh/best-alzheimer-mri-dataset-99-accuracy")
    df = create_dataframe_from_images(dataset_path)
    print("First 5 records:", df.head())
except Exception as e:
    print(f"Error: {e}")
    # Fallback for demonstration
    df = pd.DataFrame({
        'image_id': ['train_no_0001', 'train_mild_0001', 'test_moderate_0001'],
        'class': ['No Impairment', 'Mild Impairment', 'Moderate Impairment'],
        'split': ['train', 'train', 'test']
    })
    print("First 5 records:", df.head())