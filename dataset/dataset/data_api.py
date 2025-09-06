import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

def get_alzheimer_data(file_path="Alzheimer_Dataset.csv"):
    """
    Loads Alzheimer's dataset from Kaggle via kagglehub.
    Returns a pandas DataFrame.
    """
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "ananthu19/alzheimer-disease-and-healthy-aging-data-in-us",
        file_path
    )
    return df
