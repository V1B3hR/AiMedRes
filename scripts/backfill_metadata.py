#!/usr/bin/env python3
"""
Metadata backfill script for DuetMind Adaptive MLOps.
Registers existing Parquet assets and feature hash into Postgres.
"""

import logging
import os
import pandas as pd
import yaml
from pathlib import Path
from sqlalchemy import create_engine, text
from datetime import datetime
import hashlib
import json
from typing import Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_file: str = "params.yaml") -> dict:
    """Load parameters from YAML file."""
    with open(params_file, 'r') as f:
        return yaml.safe_load(f)


def get_database_connection(params: dict):
    """Create database connection."""
    db_config = params['database']
    connection_string = f"postgresql://{db_config.get('user', 'duetmind')}:{db_config.get('password', 'duetmind_secret')}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    try:
        engine = create_engine(connection_string)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Successfully connected to database: {db_config['database']}")
        return engine
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


def compute_feature_hash(df: pd.DataFrame) -> str:
    """Compute deterministic hash of feature schema."""
    schema_items = [(col, str(df[col].dtype)) for col in sorted(df.columns)]
    schema_str = str(schema_items)
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def get_file_info(file_path: str) -> Dict:
    """Get file metadata (size, row count, etc.)."""
    if not os.path.exists(file_path):
        return {}
    
    file_stats = os.stat(file_path)
    file_info = {
        'size_bytes': file_stats.st_size,
        'created_at': datetime.fromtimestamp(file_stats.st_ctime),
        'modified_at': datetime.fromtimestamp(file_stats.st_mtime)
    }
    
    # Get row and column count for data files
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            return file_info
        
        file_info.update({
            'row_count': len(df),
            'column_count': len(df.columns),
            'schema_hash': compute_feature_hash(df),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        })
        
    except Exception as e:
        logger.warning(f"Could not analyze file {file_path}: {e}")
    
    return file_info


def register_dataset(engine, name: str, description: str, source_type: str, 
                    file_path: str, file_format: str, metadata: Dict) -> Optional[int]:
    """Register a dataset in the catalog."""
    
    with engine.connect() as conn:
        # Check if dataset already exists
        result = conn.execute(
            text("SELECT id FROM dataset_catalog WHERE name = :name"),
            {"name": name}
        )
        existing = result.fetchone()
        
        if existing:
            dataset_id = existing[0]
            logger.info(f"Dataset '{name}' already exists with ID {dataset_id}")
            
            # Update existing record
            conn.execute(
                text("""
                UPDATE dataset_catalog 
                SET description = :description,
                    file_path = :file_path,
                    schema_hash = :schema_hash,
                    size_bytes = :size_bytes,
                    row_count = :row_count,
                    column_count = :column_count,
                    updated_at = NOW(),
                    metadata_json = :metadata_json
                WHERE id = :id
                """),
                {
                    "description": description,
                    "file_path": file_path,
                    "schema_hash": metadata.get('schema_hash'),
                    "size_bytes": metadata.get('size_bytes'),
                    "row_count": metadata.get('row_count'),
                    "column_count": metadata.get('column_count'),
                    "metadata_json": json.dumps(metadata),
                    "id": dataset_id
                }
            )
            conn.commit()
            logger.info(f"Updated dataset '{name}'")
            
        else:
            # Insert new record
            result = conn.execute(
                text("""
                INSERT INTO dataset_catalog 
                (name, description, source_type, file_path, file_format, 
                 schema_hash, size_bytes, row_count, column_count, metadata_json)
                VALUES 
                (:name, :description, :source_type, :file_path, :file_format,
                 :schema_hash, :size_bytes, :row_count, :column_count, :metadata_json)
                RETURNING id
                """),
                {
                    "name": name,
                    "description": description,
                    "source_type": source_type,
                    "file_path": file_path,
                    "file_format": file_format,
                    "schema_hash": metadata.get('schema_hash'),
                    "size_bytes": metadata.get('size_bytes'),
                    "row_count": metadata.get('row_count'),
                    "column_count": metadata.get('column_count'),
                    "metadata_json": json.dumps(metadata)
                }
            )
            conn.commit()
            dataset_id = result.fetchone()[0]
            logger.info(f"Registered new dataset '{name}' with ID {dataset_id}")
        
        return dataset_id


def register_feature_view(engine, name: str, description: str, feature_hash: str, 
                         source_dataset_id: int, feature_list: Dict, 
                         transformation_logic: str) -> Optional[int]:
    """Register a feature view."""
    
    with engine.connect() as conn:
        # Check if feature view already exists
        result = conn.execute(
            text("SELECT id FROM feature_view WHERE name = :name"),
            {"name": name}
        )
        existing = result.fetchone()
        
        if existing:
            feature_view_id = existing[0]
            logger.info(f"Feature view '{name}' already exists with ID {feature_view_id}")
            
            # Update existing record
            conn.execute(
                text("""
                UPDATE feature_view 
                SET description = :description,
                    feature_hash = :feature_hash,
                    source_dataset_id = :source_dataset_id,
                    feature_list = :feature_list,
                    transformation_logic = :transformation_logic,
                    updated_at = NOW()
                WHERE id = :id
                """),
                {
                    "description": description,
                    "feature_hash": feature_hash,
                    "source_dataset_id": source_dataset_id,
                    "feature_list": json.dumps(feature_list),
                    "transformation_logic": transformation_logic,
                    "id": feature_view_id
                }
            )
            conn.commit()
            logger.info(f"Updated feature view '{name}'")
            
        else:
            # Insert new record
            result = conn.execute(
                text("""
                INSERT INTO feature_view 
                (name, description, feature_hash, source_dataset_id, 
                 feature_list, transformation_logic)
                VALUES 
                (:name, :description, :feature_hash, :source_dataset_id,
                 :feature_list, :transformation_logic)
                RETURNING id
                """),
                {
                    "name": name,
                    "description": description,
                    "feature_hash": feature_hash,
                    "source_dataset_id": source_dataset_id,
                    "feature_list": json.dumps(feature_list),
                    "transformation_logic": transformation_logic
                }
            )
            conn.commit()
            feature_view_id = result.fetchone()[0]
            logger.info(f"Registered new feature view '{name}' with ID {feature_view_id}")
        
        return feature_view_id


def backfill_datasets(engine, params: dict) -> Dict[str, int]:
    """Backfill existing datasets."""
    dataset_ids = {}
    
    # Register raw dataset
    raw_file = os.path.join(params['data']['raw'], 'alzheimer_sample.csv')
    if os.path.exists(raw_file):
        metadata = get_file_info(raw_file)
        dataset_id = register_dataset(
            engine, 
            name='alzheimer_raw',
            description='Raw Alzheimer dataset with patient demographics and cognitive scores',
            source_type='raw',
            file_path=raw_file,
            file_format='csv',
            metadata=metadata
        )
        dataset_ids['raw'] = dataset_id
    
    # Register processed features
    features_file = params['data']['features']
    if os.path.exists(features_file):
        metadata = get_file_info(features_file)
        dataset_id = register_dataset(
            engine,
            name='alzheimer_features',
            description='Engineered features for Alzheimer prediction model',
            source_type='processed',
            file_path=features_file,
            file_format='parquet',
            metadata=metadata
        )
        dataset_ids['features'] = dataset_id
    
    # Register labels
    labels_file = params['data']['labels']
    if os.path.exists(labels_file):
        metadata = get_file_info(labels_file)
        dataset_id = register_dataset(
            engine,
            name='alzheimer_labels',
            description='Target labels for Alzheimer prediction model',
            source_type='processed',
            file_path=labels_file,
            file_format='parquet',
            metadata=metadata
        )
        dataset_ids['labels'] = dataset_id
    
    return dataset_ids


def backfill_feature_views(engine, dataset_ids: Dict[str, int], params: dict) -> Dict[str, int]:
    """Backfill feature views."""
    feature_view_ids = {}
    
    # Create feature view for processed features
    if 'features' in dataset_ids:
        features_file = params['data']['features']
        df = pd.read_parquet(features_file)
        feature_hash = compute_feature_hash(df)
        
        feature_list = {
            'features': [
                {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'description': f"Feature: {col}"
                } 
                for col in df.columns
            ],
            'count': len(df.columns)
        }
        
        transformation_logic = """
        Feature engineering pipeline:
        1. Age grouping (young, middle, older, elderly)
        2. MMSE score normalization
        3. Cognitive composite score calculation
        4. Binary encoding for categorical variables (APOE4, gender)
        5. Interaction features (age*MMSE, education*cognitive)
        """
        
        feature_view_id = register_feature_view(
            engine,
            name='alzheimer_features_v1',
            description='Engineered features for Alzheimer prediction - version 1',
            feature_hash=feature_hash,
            source_dataset_id=dataset_ids['features'],
            feature_list=feature_list,
            transformation_logic=transformation_logic
        )
        feature_view_ids['alzheimer_features_v1'] = feature_view_id
    
    return feature_view_ids


def backfill_model_registry(engine, feature_view_ids: Dict[str, int], params: dict):
    """Backfill model registry from existing MLflow runs."""
    
    with engine.connect() as conn:
        # Register the model in our custom registry
        result = conn.execute(
            text("SELECT id FROM model_registry WHERE name = :name"),
            {"name": "alzheimer_classifier"}
        )
        existing = result.fetchone()
        
        if not existing:
            result = conn.execute(
                text("""
                INSERT INTO model_registry 
                (name, description, model_type, framework)
                VALUES 
                (:name, :description, :model_type, :framework)
                RETURNING id
                """),
                {
                    "name": "alzheimer_classifier",
                    "description": "Random Forest classifier for Alzheimer's disease prediction",
                    "model_type": "classification",
                    "framework": "sklearn"
                }
            )
            conn.commit()
            model_id = result.fetchone()[0]
            logger.info(f"Registered model 'alzheimer_classifier' with ID {model_id}")
        else:
            model_id = existing[0]
            logger.info(f"Model 'alzheimer_classifier' already exists with ID {model_id}")


def main():
    """Main backfill function."""
    logger.info("Starting metadata backfill process...")
    
    # Load parameters
    params = load_params()
    
    # Connect to database
    try:
        engine = get_database_connection(params)
    except Exception as e:
        logger.error(f"Cannot connect to database. Make sure PostgreSQL is running: {e}")
        logger.info("You can start the infrastructure with: make infra-up")
        return
    
    # Backfill datasets
    logger.info("Backfilling datasets...")
    dataset_ids = backfill_datasets(engine, params)
    logger.info(f"Registered {len(dataset_ids)} datasets: {list(dataset_ids.keys())}")
    
    # Backfill feature views
    logger.info("Backfilling feature views...")
    feature_view_ids = backfill_feature_views(engine, dataset_ids, params)
    logger.info(f"Registered {len(feature_view_ids)} feature views: {list(feature_view_ids.keys())}")
    
    # Backfill model registry
    logger.info("Backfilling model registry...")
    backfill_model_registry(engine, feature_view_ids, params)
    
    logger.info("Metadata backfill completed successfully!")
    logger.info("You can view the data in Adminer at: http://localhost:8080")


if __name__ == "__main__":
    main()