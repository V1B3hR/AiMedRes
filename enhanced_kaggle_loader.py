#!/usr/bin/env python3
"""
Enhanced demonstration of the kaggle dataset loading with full analysis.
This extends the problem statement implementation with additional dataset exploration.
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_and_analyze_dataset():
    """Load and analyze the Alzheimer's dataset as per problem statement"""
    
    print("Loading Alzheimer's death rates dataset from Kaggle...")
    print("=" * 60)
    
    # Set the path to the file you'd like to load
    file_path = "dementia-death-rates new.csv"
    
    # Load the latest version
    df = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "willianoliveiragibin/death-alzheimers",
      file_path,
      # Provide any additional arguments like 
      # sql_query or pandas_kwargs. See the 
      # documenation for more information:
      # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )
    
    print("✓ Dataset loaded successfully!")
    print("First 5 records:", df.head())
    
    # Additional analysis to show dataset characteristics
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Column names: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    if len(df) > 0:
        print(f"\nSample of data:")
        print(df.head(10))
        
        print(f"\nDataset info:")
        print(f"- Memory usage: {df.memory_usage().sum() / 1024:.1f} KB")
        print(f"- Non-null counts: {df.count().tolist()}")
        
    return df


if __name__ == "__main__":
    try:
        dataset = load_and_analyze_dataset()
        print("\n✓ Problem statement implementation completed successfully!")
        print(f"✓ Loaded {len(dataset)} records from willianoliveiragibin/death-alzheimers dataset")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("Please ensure kagglehub is installed: pip install kagglehub[pandas-datasets]")