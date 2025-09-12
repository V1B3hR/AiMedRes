#!/usr/bin/env python3
"""
Raw data ingestion pipeline for DuetMind Adaptive MLOps.
Downloads or stages raw Alzheimer datasets into data/raw/.
"""

import logging
import os
import pandas as pd
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_file: str = "params.yaml") -> dict:
    """Load parameters from YAML file."""
    with open(params_file, 'r') as f:
        return yaml.safe_load(f)


def create_sample_alzheimer_data(output_path: str, n_samples: int = 1000) -> None:
    """
    Create sample Alzheimer dataset for demonstration purposes.
    In production, this would connect to actual data sources.
    """
    import numpy as np
    
    np.random.seed(42)
    
    # Generate synthetic Alzheimer's data
    data = {
        'patient_id': range(n_samples),
        'age': np.random.normal(75, 10, n_samples).astype(int),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'education_level': np.random.randint(6, 20, n_samples),
        'mmse_score': np.random.normal(25, 5, n_samples).clip(0, 30),
        'apoe_genotype': np.random.choice(['APOE2', 'APOE3', 'APOE4'], n_samples),
        'family_history': np.random.choice([0, 1], n_samples),
        'cognitive_test_1': np.random.normal(100, 15, n_samples),
        'cognitive_test_2': np.random.normal(50, 10, n_samples),
        'brain_volume': np.random.normal(1200, 100, n_samples),
    }
    
    # Create target variable (diagnosis) based on features
    diagnosis_prob = (
        0.1 +  # baseline risk
        0.002 * (data['age'] - 65) +  # age factor
        0.01 * (30 - data['mmse_score']) +  # cognitive score factor
        0.1 * (data['apoe_genotype'] == 'APOE4').astype(int) +  # genetic factor
        0.05 * data['family_history']  # family history factor
    )
    
    data['diagnosis'] = np.random.binomial(1, np.clip(diagnosis_prob, 0, 1), n_samples)
    
    df = pd.DataFrame(data)
    
    # Save raw data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample dataset with {len(df)} samples at {output_path}")
    
    return df


def ingest_raw_data(params: dict) -> None:
    """
    Main data ingestion function.
    In production, this would handle multiple data sources.
    """
    raw_data_path = params['data']['raw']
    
    # Create raw data directory
    Path(raw_data_path).mkdir(parents=True, exist_ok=True)
    
    # For demonstration, create sample data
    # In production, this would fetch from actual data sources
    sample_file = os.path.join(raw_data_path, 'alzheimer_sample.csv')
    
    if not os.path.exists(sample_file):
        logger.info("Creating sample Alzheimer dataset...")
        create_sample_alzheimer_data(sample_file)
    else:
        logger.info(f"Raw data already exists at {sample_file}")
    
    # Validate raw data
    df = pd.read_csv(sample_file)
    logger.info(f"Raw data validation: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


if __name__ == "__main__":
    # Load parameters
    params = load_params()
    
    # Ingest raw data
    ingest_raw_data(params)
    
    logger.info("Raw data ingestion completed successfully!")