#!/usr/bin/env python3
"""
Schema validation contracts using Pandera for DuetMind Adaptive MLOps.
Defines and validates data schemas for consistency checks.
"""

import logging
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from typing import Union
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Raw data schema
RAW_DATA_SCHEMA = DataFrameSchema({
    "patient_id": Column(pa.Int64, checks=[
        Check.greater_than_or_equal_to(0),
        Check(lambda x: x.is_unique, error="Patient IDs must be unique")
    ]),
    "age": Column(pa.Int64, checks=[
        Check.between(18, 120, include_min=True, include_max=True)
    ]),
    "gender": Column(pa.String, checks=[
        Check.isin(['M', 'F'])
    ]),
    "education_level": Column(pa.Int64, checks=[
        Check.between(0, 25, include_min=True, include_max=True)
    ]),
    "mmse_score": Column(pa.Float64, checks=[
        Check.between(0, 30, include_min=True, include_max=True)
    ]),
    "apoe_genotype": Column(pa.String, checks=[
        Check.isin(['APOE2', 'APOE3', 'APOE4'])
    ]),
    "family_history": Column(pa.Int64, checks=[
        Check.isin([0, 1])
    ]),
    "cognitive_test_1": Column(pa.Float64, checks=[
        Check.greater_than(0)
    ]),
    "cognitive_test_2": Column(pa.Float64, checks=[
        Check.greater_than(0)  
    ]),
    "brain_volume": Column(pa.Float64, checks=[
        Check.greater_than(0)
    ]),
    "diagnosis": Column(pa.Int64, checks=[
        Check.isin([0, 1])
    ])
}, strict=True, coerce=True)


# Features schema (after feature engineering)
FEATURES_SCHEMA = DataFrameSchema({
    "patient_id": Column(pa.Int64),
    "age": Column(pa.Int64, checks=[Check.between(18, 120)]),
    "gender": Column(pa.String),
    "education_level": Column(pa.Int64),
    "mmse_score": Column(pa.Float64),
    "apoe_genotype": Column(pa.String),
    "family_history": Column(pa.Int64),
    "cognitive_test_1": Column(pa.Float64),
    "cognitive_test_2": Column(pa.Float64),
    "brain_volume": Column(pa.Float64),
    
    # Engineered features
    "age_group": Column(pa.Category),
    "mmse_normalized": Column(pa.Float64),
    "cognitive_composite": Column(pa.Float64),
    "is_apoe4": Column(pa.Int64, checks=[Check.isin([0, 1])]),
    "is_male": Column(pa.Int64, checks=[Check.isin([0, 1])]),
    "age_mmse_interaction": Column(pa.Float64),
    "education_cognitive_interaction": Column(pa.Float64)
}, coerce=True)


# Labels schema
LABELS_SCHEMA = DataFrameSchema({
    "diagnosis": Column(pa.Int64, checks=[
        Check.isin([0, 1])
    ])
}, strict=True, coerce=True)


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate raw data against expected schema.
    
    Args:
        df: Raw dataframe to validate
        
    Returns:
        Validated dataframe
        
    Raises:
        pandera.errors.SchemaError: If validation fails
    """
    logger.info("Validating raw data schema...")
    
    try:
        validated_df = RAW_DATA_SCHEMA.validate(df)
        logger.info(f"Raw data validation passed for {len(validated_df)} rows")
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error(f"Raw data validation failed: {e}")
        raise


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate features dataframe against expected schema.
    
    Args:
        df: Features dataframe to validate
        
    Returns:
        Validated dataframe
        
    Raises:
        pandera.errors.SchemaError: If validation fails
    """
    logger.info("Validating features schema...")
    
    try:
        validated_df = FEATURES_SCHEMA.validate(df)
        logger.info(f"Features validation passed for {len(validated_df)} rows")
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error(f"Features validation failed: {e}")
        raise


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate labels dataframe against expected schema.
    
    Args:
        df: Labels dataframe to validate
        
    Returns:
        Validated dataframe
        
    Raises:
        pandera.errors.SchemaError: If validation fails
    """
    logger.info("Validating labels schema...")
    
    try:
        validated_df = LABELS_SCHEMA.validate(df)
        logger.info(f"Labels validation passed for {len(validated_df)} rows")
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error(f"Labels validation failed: {e}")
        raise


def validate_data_quality(df: pd.DataFrame, name: str = "data") -> dict:
    """
    Perform additional data quality checks beyond schema validation.
    
    Args:
        df: Dataframe to check
        name: Name for logging purposes
        
    Returns:
        Dictionary with quality metrics
    """
    logger.info(f"Performing data quality checks for {name}...")
    
    quality_metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    # Check for missing values by column
    missing_by_column = df.isnull().sum()
    if missing_by_column.any():
        quality_metrics['missing_by_column'] = missing_by_column[missing_by_column > 0].to_dict()
        logger.warning(f"Missing values found: {quality_metrics['missing_by_column']}")
    
    # Check for duplicates
    if quality_metrics['duplicate_rows'] > 0:
        logger.warning(f"Found {quality_metrics['duplicate_rows']} duplicate rows")
    
    # Check data types
    dtypes_info = df.dtypes.value_counts().to_dict()
    quality_metrics['column_types'] = {str(k): v for k, v in dtypes_info.items()}
    
    logger.info(f"Data quality summary for {name}: {quality_metrics}")
    return quality_metrics


def validate_file(file_path: str, schema_type: str = "auto") -> bool:
    """
    Validate a data file against appropriate schema.
    
    Args:
        file_path: Path to data file (CSV or Parquet)
        schema_type: Type of schema to use ('raw', 'features', 'labels', 'auto')
        
    Returns:
        True if validation passes
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If schema_type is invalid
        pandera.errors.SchemaError: If validation fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Auto-detect schema type if needed
    if schema_type == "auto":
        if "diagnosis" in df.columns and len(df.columns) > 5:
            if any(col.startswith(('age_group', 'mmse_normalized')) for col in df.columns):
                schema_type = "features"
            else:
                schema_type = "raw"
        elif "diagnosis" in df.columns and len(df.columns) <= 2:
            schema_type = "labels"
        else:
            logger.warning("Could not auto-detect schema type, defaulting to raw")
            schema_type = "raw"
    
    # Validate based on schema type
    if schema_type == "raw":
        validate_raw_data(df)
    elif schema_type == "features":
        validate_features(df)
    elif schema_type == "labels":
        validate_labels(df)
    else:
        raise ValueError(f"Invalid schema_type: {schema_type}")
    
    # Additional quality checks
    validate_data_quality(df, os.path.basename(file_path))
    
    logger.info(f"Validation completed successfully for {file_path}")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate data files against schemas")
    parser.add_argument("file_path", help="Path to data file to validate")
    parser.add_argument("--schema", choices=["raw", "features", "labels", "auto"], 
                       default="auto", help="Schema type to use for validation")
    
    args = parser.parse_args()
    
    try:
        validate_file(args.file_path, args.schema)
        print(f"✓ Validation passed for {args.file_path}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        exit(1)