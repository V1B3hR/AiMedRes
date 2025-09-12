#!/usr/bin/env python3
"""
Feature hashing utility for DuetMind Adaptive MLOps.
Computes deterministic hashes for feature schema consistency checks.
"""

import hashlib
import pandas as pd
import argparse
import logging
import os
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_feature_hash(df: pd.DataFrame, algorithm: str = 'sha256') -> str:
    """
    Compute deterministic hash of feature schema.
    
    Args:
        df: DataFrame to hash
        algorithm: Hash algorithm to use ('sha256', 'md5', 'sha1')
        
    Returns:
        Hex digest of the hash (truncated to 16 characters for sha256)
    """
    # Create deterministic representation of schema
    # Sort columns to ensure deterministic ordering
    schema_items = []
    
    for col in sorted(df.columns):
        dtype_str = str(df[col].dtype)
        # Normalize dtype representation
        if dtype_str.startswith('int'):
            dtype_str = 'int'
        elif dtype_str.startswith('float'):
            dtype_str = 'float'
        elif dtype_str in ['object', 'string']:
            dtype_str = 'string'
        elif dtype_str == 'category':
            dtype_str = 'category'
        elif dtype_str.startswith('datetime'):
            dtype_str = 'datetime'
        elif dtype_str == 'bool':
            dtype_str = 'bool'
            
        schema_items.append((col, dtype_str))
    
    # Create string representation
    schema_str = str(schema_items)
    logger.debug(f"Schema string: {schema_str}")
    
    # Compute hash
    if algorithm == 'sha256':
        hash_obj = hashlib.sha256(schema_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # Truncate to 16 chars
    elif algorithm == 'md5':
        hash_obj = hashlib.md5(schema_str.encode('utf-8'))
        return hash_obj.hexdigest()
    elif algorithm == 'sha1':
        hash_obj = hashlib.sha1(schema_str.encode('utf-8'))
        return hash_obj.hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def compute_feature_hash_from_file(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Compute feature hash from a data file.
    
    Args:
        file_path: Path to CSV or Parquet file
        algorithm: Hash algorithm to use
        
    Returns:
        Feature hash string
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format. Use .csv or .parquet: {file_path}")
    
    return compute_feature_hash(df, algorithm)


def get_schema_summary(df: pd.DataFrame) -> Dict:
    """
    Get detailed schema summary for a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with schema information
    """
    summary = {
        'num_columns': len(df.columns),
        'num_rows': len(df),
        'columns': {},
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'null_count': df[col].isnull().sum(),
            'null_percentage': df[col].isnull().sum() / len(df) * 100,
            'unique_values': df[col].nunique()
        }
        
        # Add sample values for small cardinality columns
        if col_info['unique_values'] <= 10:
            col_info['unique_values_list'] = df[col].unique().tolist()
        
        # Add basic stats for numeric columns
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            col_info['min'] = float(df[col].min()) if not df[col].isnull().all() else None
            col_info['max'] = float(df[col].max()) if not df[col].isnull().all() else None
            col_info['mean'] = float(df[col].mean()) if not df[col].isnull().all() else None
        
        summary['columns'][col] = col_info
    
    return summary


def compare_schemas(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
    """
    Compare schemas of two DataFrames.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        
    Returns:
        Dictionary with comparison results
    """
    hash1 = compute_feature_hash(df1)
    hash2 = compute_feature_hash(df2)
    
    comparison = {
        'schemas_match': hash1 == hash2,
        'hash1': hash1,
        'hash2': hash2,
        'columns_only_in_df1': set(df1.columns) - set(df2.columns),
        'columns_only_in_df2': set(df2.columns) - set(df1.columns),
        'common_columns': set(df1.columns) & set(df2.columns),
        'dtype_mismatches': {}
    }
    
    # Check dtype mismatches in common columns
    for col in comparison['common_columns']:
        if str(df1[col].dtype) != str(df2[col].dtype):
            comparison['dtype_mismatches'][col] = {
                'df1_dtype': str(df1[col].dtype),
                'df2_dtype': str(df2[col].dtype)
            }
    
    return comparison


def main():
    """Main CLI interface for feature hashing."""
    parser = argparse.ArgumentParser(description="Compute feature hashes for data consistency checks")
    parser.add_argument("file_path", help="Path to data file (CSV or Parquet)")
    parser.add_argument("--algorithm", choices=['sha256', 'md5', 'sha1'], 
                       default='sha256', help="Hash algorithm to use")
    parser.add_argument("--summary", action='store_true', 
                       help="Show detailed schema summary")
    parser.add_argument("--compare", help="Compare with another file")
    
    args = parser.parse_args()
    
    try:
        # Compute hash for main file
        feature_hash = compute_feature_hash_from_file(args.file_path, args.algorithm)
        print(f"Feature hash ({args.algorithm}): {feature_hash}")
        
        if args.summary:
            # Load and analyze data
            if args.file_path.endswith('.csv'):
                df = pd.read_csv(args.file_path)
            else:
                df = pd.read_parquet(args.file_path)
            
            summary = get_schema_summary(df)
            print(f"\nSchema Summary:")
            print(f"  Rows: {summary['num_rows']:,}")
            print(f"  Columns: {summary['num_columns']}")
            print(f"  Memory Usage: {summary['memory_usage_mb']:.2f} MB")
            print(f"\nColumn Details:")
            
            for col, info in summary['columns'].items():
                print(f"  {col}:")
                print(f"    Type: {info['dtype']}")
                print(f"    Nulls: {info['null_count']} ({info['null_percentage']:.1f}%)")
                print(f"    Unique: {info['unique_values']}")
                
                if 'unique_values_list' in info:
                    print(f"    Values: {info['unique_values_list']}")
                if 'min' in info:
                    print(f"    Range: {info['min']:.3f} - {info['max']:.3f}")
        
        if args.compare:
            # Compare with another file
            if args.file_path.endswith('.csv'):
                df1 = pd.read_csv(args.file_path)
            else:
                df1 = pd.read_parquet(args.file_path)
            
            if args.compare.endswith('.csv'):
                df2 = pd.read_csv(args.compare)
            else:
                df2 = pd.read_parquet(args.compare)
            
            comparison = compare_schemas(df1, df2)
            
            print(f"\nSchema Comparison:")
            print(f"  Schemas Match: {comparison['schemas_match']}")
            print(f"  Hash 1: {comparison['hash1']}")
            print(f"  Hash 2: {comparison['hash2']}")
            
            if comparison['columns_only_in_df1']:
                print(f"  Columns only in file 1: {list(comparison['columns_only_in_df1'])}")
            if comparison['columns_only_in_df2']:
                print(f"  Columns only in file 2: {list(comparison['columns_only_in_df2'])}")
            if comparison['dtype_mismatches']:
                print(f"  Type mismatches: {comparison['dtype_mismatches']}")
                
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        exit(1)


if __name__ == "__main__":
    main()