#!/usr/bin/env python3
"""
Feature engineering pipeline for DuetMind Adaptive MLOps.
Creates canonical Parquet feature set in data/processed/.
"""

import logging
import os
import pandas as pd
import yaml
import hashlib
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_file: str = "params.yaml") -> dict:
    """Load parameters from YAML file."""
    with open(params_file, 'r') as f:
        return yaml.safe_load(f)


def compute_feature_hash(df: pd.DataFrame) -> str:
    """
    Compute deterministic hash of feature schema.
    Based on column names and dtypes for consistency checks.
    """
    # Create deterministic representation of schema
    schema_items = [(col, str(df[col].dtype)) for col in sorted(df.columns)]
    schema_str = str(schema_items)
    
    # Compute SHA256 hash
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def engineer_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Perform feature engineering transformations.
    """
    logger.info("Starting feature engineering...")
    
    # Copy dataframe to avoid modifying original
    features_df = df.copy()
    
    # Age-related features
    features_df['age_group'] = pd.cut(features_df['age'], 
                                     bins=[0, 65, 75, 85, 150], 
                                     labels=['young', 'middle', 'older', 'elderly'],
                                     include_lowest=True)
    
    # Cognitive score normalization
    features_df['mmse_normalized'] = (features_df['mmse_score'] - features_df['mmse_score'].mean()) / features_df['mmse_score'].std()
    
    # Risk score combination
    features_df['cognitive_composite'] = (
        features_df['cognitive_test_1'] * 0.6 + 
        features_df['cognitive_test_2'] * 0.4
    ) / 100
    
    # Binary encoding for categorical variables
    features_df['is_apoe4'] = (features_df['apoe_genotype'] == 'APOE4').astype(int)
    features_df['is_male'] = (features_df['gender'] == 'M').astype(int)
    
    # Interaction features
    features_df['age_mmse_interaction'] = features_df['age'] * features_df['mmse_score']
    features_df['education_cognitive_interaction'] = features_df['education_level'] * features_df['cognitive_composite']
    
    logger.info(f"Feature engineering completed. Shape: {features_df.shape}")
    return features_df


def split_features_labels(df: pd.DataFrame, params: dict) -> tuple:
    """
    Split dataframe into features and labels.
    """
    target_col = params['features']['target_column']
    
    # Separate features and target
    labels_df = df[[target_col]].copy()
    features_df = df.drop(columns=[target_col]).copy()
    
    logger.info(f"Features shape: {features_df.shape}")
    logger.info(f"Labels shape: {labels_df.shape}")
    
    return features_df, labels_df


def save_processed_data(features_df: pd.DataFrame, labels_df: pd.DataFrame, params: dict) -> dict:
    """
    Save processed features and labels as Parquet files.
    """
    processed_path = params['data']['processed']
    Path(processed_path).mkdir(parents=True, exist_ok=True)
    
    # Save features
    features_file = params['data']['features']
    features_df.to_parquet(features_file, index=False)
    logger.info(f"Saved features to {features_file}")
    
    # Save labels
    labels_file = params['data']['labels']
    labels_df.to_parquet(labels_file, index=False)
    logger.info(f"Saved labels to {labels_file}")
    
    # Compute feature hash for consistency
    feature_hash = compute_feature_hash(features_df)
    logger.info(f"Feature schema hash: {feature_hash}")
    
    return {
        'features_file': features_file,
        'labels_file': labels_file,
        'feature_hash': feature_hash,
        'n_features': len(features_df.columns),
        'n_samples': len(features_df)
    }


def build_features(params: dict) -> dict:
    """
    Main feature building function.
    """
    # Load raw data
    raw_path = os.path.join(params['data']['raw'], 'alzheimer_sample.csv')
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run ingest_raw.py first.")
    
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    
    # Engineer features
    engineered_df = engineer_features(df, params)
    
    # Split features and labels
    features_df, labels_df = split_features_labels(engineered_df, params)
    
    # Save processed data
    metadata = save_processed_data(features_df, labels_df, params)
    
    return metadata


if __name__ == "__main__":
    # Load parameters
    params = load_params()
    
    # Build features
    metadata = build_features(params)
    
    logger.info("Feature building completed successfully!")
    logger.info(f"Metadata: {metadata}")