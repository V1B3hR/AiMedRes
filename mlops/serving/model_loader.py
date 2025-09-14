#!/usr/bin/env python3
"""
Model Loader for Production Serving.
Handles loading models from MLflow and local storage with versioning.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import os
from pathlib import Path
import joblib
import pickle
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and management of model versions."""
    
    def __init__(self, mlflow_tracking_uri: str = "sqlite:///mlflow.db"):
        """Initialize model loader."""
        self.mlflow_tracking_uri = mlflow_tracking_uri
        
        # Setup MLflow client
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_client = MlflowClient()
        
        # Cache for loaded models
        self.model_cache = {}
        
        logger.info("Model loader initialized")
    
    def get_latest_model_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model."""
        try:
            # Try MLflow model registry first
            try:
                latest_versions = self.mlflow_client.get_latest_versions(
                    model_name, 
                    stages=["Production", "Staging", "None"]
                )
                
                if latest_versions:
                    # Return the highest version number
                    latest_version = max(latest_versions, key=lambda v: int(v.version))
                    logger.info(f"Latest model version from MLflow: {latest_version.version}")
                    return latest_version.version
                    
            except MlflowException as e:
                logger.warning(f"MLflow model registry error: {e}")
            
            # Fallback: get latest run from experiment
            try:
                experiment = self.mlflow_client.get_experiment_by_name(f"{model_name}_experiment")
                if not experiment:
                    # Try alternative naming
                    experiment = self.mlflow_client.get_experiment_by_name("duetmind_alzheimer_prediction")
                
                if experiment:
                    runs = self.mlflow_client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"],
                        max_results=1
                    )
                    
                    if runs:
                        run_id = runs[0].info.run_id
                        logger.info(f"Latest model from experiment run: {run_id}")
                        return run_id
                        
            except Exception as e:
                logger.warning(f"Error getting latest run: {e}")
            
            # Final fallback: check local models directory
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob(f"{model_name}*.pkl")) + list(models_dir.glob(f"{model_name}*.joblib"))
                if model_files:
                    # Return the most recently modified file
                    latest_file = max(model_files, key=lambda f: f.stat().st_mtime)
                    logger.info(f"Latest local model: {latest_file}")
                    return str(latest_file)
            
            logger.warning(f"No versions found for model: {model_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest model version: {e}")
            return None
    
    def load_model_version(self, model_name: str, version: str) -> Optional[Any]:
        """Load a specific model version."""
        cache_key = f"{model_name}:{version}"
        
        # Check cache first
        if cache_key in self.model_cache:
            logger.info(f"Loading model from cache: {cache_key}")
            return self.model_cache[cache_key]
        
        model = None
        
        try:
            # Method 1: Try MLflow model registry
            try:
                model_uri = f"models:/{model_name}/{version}"
                model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Loaded model from MLflow registry: {model_uri}")
            except MlflowException as e:
                logger.warning(f"MLflow registry load failed: {e}")
            
            # Method 2: Try MLflow run artifacts
            if model is None:
                try:
                    # If version looks like a run_id (hex string)
                    if len(version) == 32 and all(c in '0123456789abcdef' for c in version.lower()):
                        model_uri = f"runs:/{version}/model"
                        model = mlflow.sklearn.load_model(model_uri)
                        logger.info(f"Loaded model from MLflow run: {model_uri}")
                except Exception as e:
                    logger.warning(f"MLflow run load failed: {e}")
            
            # Method 3: Try local file loading
            if model is None:
                local_path = Path(version)
                if local_path.exists():
                    model = self._load_local_model(local_path)
                    logger.info(f"Loaded model from local file: {local_path}")
                else:
                    # Try models directory
                    models_dir = Path("models")
                    potential_paths = [
                        models_dir / f"{model_name}_{version}.pkl",
                        models_dir / f"{model_name}_{version}.joblib",
                        models_dir / f"{version}.pkl",
                        models_dir / f"{version}.joblib",
                        models_dir / version
                    ]
                    
                    for path in potential_paths:
                        if path.exists():
                            model = self._load_local_model(path)
                            if model:
                                logger.info(f"Loaded model from: {path}")
                                break
            
            # Cache the loaded model
            if model is not None:
                self.model_cache[cache_key] = model
                logger.info(f"Cached model: {cache_key}")
            else:
                logger.error(f"Failed to load model: {model_name}:{version}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}:{version}: {e}")
            return None
    
    def _load_local_model(self, model_path: Path) -> Optional[Any]:
        """Load model from local file."""
        try:
            if model_path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            elif model_path.suffix == '.joblib':
                return joblib.load(model_path)
            else:
                # Try both methods
                try:
                    return joblib.load(model_path)
                except:
                    with open(model_path, 'rb') as f:
                        return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading local model {model_path}: {e}")
            return None
    
    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all available versions of a model."""
        versions = []
        
        try:
            # MLflow registry versions
            try:
                registered_versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
                for version in registered_versions:
                    versions.append({
                        'version': version.version,
                        'stage': version.current_stage,
                        'source': 'mlflow_registry',
                        'created_at': version.creation_timestamp,
                        'run_id': version.run_id
                    })
            except Exception as e:
                logger.warning(f"Error listing MLflow registry versions: {e}")
            
            # MLflow experiment runs
            try:
                experiment = self.mlflow_client.get_experiment_by_name(f"{model_name}_experiment")
                if not experiment:
                    experiment = self.mlflow_client.get_experiment_by_name("duetmind_alzheimer_prediction")
                
                if experiment:
                    runs = self.mlflow_client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"]
                    )
                    
                    for run in runs:
                        versions.append({
                            'version': run.info.run_id,
                            'stage': 'experiment_run',
                            'source': 'mlflow_experiment',
                            'created_at': run.info.start_time,
                            'run_id': run.info.run_id,
                            'metrics': run.data.metrics
                        })
            except Exception as e:
                logger.warning(f"Error listing experiment runs: {e}")
            
            # Local model files
            models_dir = Path("models")
            if models_dir.exists():
                for model_file in models_dir.glob("*.pkl"):
                    versions.append({
                        'version': model_file.name,
                        'stage': 'local_file',
                        'source': 'local_storage',
                        'created_at': model_file.stat().st_mtime,
                        'file_path': str(model_file)
                    })
                
                for model_file in models_dir.glob("*.joblib"):
                    versions.append({
                        'version': model_file.name,
                        'stage': 'local_file',
                        'source': 'local_storage',
                        'created_at': model_file.stat().st_mtime,
                        'file_path': str(model_file)
                    })
        
        except Exception as e:
            logger.error(f"Error listing model versions: {e}")
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda v: v.get('created_at', 0), reverse=True)
        
        return versions
    
    def get_model_metadata(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get metadata for a specific model version."""
        try:
            # Try MLflow registry first
            try:
                model_version = self.mlflow_client.get_model_version(model_name, version)
                return {
                    'name': model_version.name,
                    'version': model_version.version,
                    'stage': model_version.current_stage,
                    'description': model_version.description,
                    'created_at': model_version.creation_timestamp,
                    'last_updated': model_version.last_updated_timestamp,
                    'run_id': model_version.run_id,
                    'source': 'mlflow_registry'
                }
            except MlflowException:
                pass
            
            # Try experiment run
            if len(version) == 32:  # Looks like run_id
                try:
                    run = self.mlflow_client.get_run(version)
                    return {
                        'name': model_name,
                        'version': version,
                        'stage': 'experiment_run',
                        'created_at': run.info.start_time,
                        'end_time': run.info.end_time,
                        'status': run.info.status,
                        'metrics': run.data.metrics,
                        'params': run.data.params,
                        'tags': run.data.tags,
                        'source': 'mlflow_experiment'
                    }
                except Exception:
                    pass
            
            # Local file metadata
            local_path = Path(version)
            if local_path.exists():
                stat = local_path.stat()
                return {
                    'name': model_name,
                    'version': version,
                    'stage': 'local_file',
                    'file_path': str(local_path),
                    'file_size': stat.st_size,
                    'created_at': stat.st_ctime,
                    'modified_at': stat.st_mtime,
                    'source': 'local_storage'
                }
            
            return {
                'name': model_name,
                'version': version,
                'error': 'Model not found',
                'source': 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Error getting model metadata: {e}")
            return {
                'name': model_name,
                'version': version,
                'error': str(e),
                'source': 'error'
            }
    
    def validate_model(self, model: Any) -> Dict[str, Any]:
        """Validate that a model is working correctly."""
        validation_result = {
            'is_valid': False,
            'has_predict': False,
            'has_predict_proba': False,
            'model_type': str(type(model)),
            'errors': []
        }
        
        try:
            # Check if model has predict method
            if hasattr(model, 'predict'):
                validation_result['has_predict'] = True
                
                # Test prediction with dummy data
                try:
                    dummy_data = np.array([[0.5, 65, 12, 2, 25, 0.5, 1500, 0.75, 1.2]])
                    prediction = model.predict(dummy_data)
                    validation_result['sample_prediction'] = prediction.tolist()
                except Exception as e:
                    validation_result['errors'].append(f"Predict method error: {str(e)}")
            else:
                validation_result['errors'].append("Model does not have predict method")
            
            # Check if model has predict_proba method
            if hasattr(model, 'predict_proba'):
                validation_result['has_predict_proba'] = True
                
                try:
                    dummy_data = np.array([[0.5, 65, 12, 2, 25, 0.5, 1500, 0.75, 1.2]])
                    probabilities = model.predict_proba(dummy_data)
                    validation_result['sample_probabilities'] = probabilities.tolist()
                except Exception as e:
                    validation_result['errors'].append(f"Predict_proba method error: {str(e)}")
            
            # Mark as valid if no errors and has predict method
            validation_result['is_valid'] = (
                validation_result['has_predict'] and 
                len(validation_result['errors']) == 0
            )
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def clear_cache(self):
        """Clear the model cache."""
        self.model_cache.clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models."""
        return {
            'cached_models': list(self.model_cache.keys()),
            'cache_size': len(self.model_cache)
        }


if __name__ == "__main__":
    # Demo usage
    print("Model Loader Demo")
    
    # Create model loader
    loader = ModelLoader()
    
    # List available models
    model_name = "alzheimer_classifier"
    versions = loader.list_model_versions(model_name)
    print(f"Available versions for {model_name}: {len(versions)}")
    
    for version in versions[:3]:  # Show first 3 versions
        print(f"  Version: {version['version']}, Source: {version['source']}")
    
    # Get latest version
    latest_version = loader.get_latest_model_version(model_name)
    print(f"Latest version: {latest_version}")
    
    if latest_version:
        # Load model
        model = loader.load_model_version(model_name, latest_version)
        if model:
            print("Model loaded successfully")
            
            # Validate model
            validation = loader.validate_model(model)
            print(f"Model validation: {validation['is_valid']}")
            
            # Get metadata
            metadata = loader.get_model_metadata(model_name, latest_version)
            print(f"Model metadata: {metadata.get('source', 'unknown')}")
        else:
            print("Failed to load model")
    
    # Cache info
    cache_info = loader.get_cache_info()
    print(f"Cache info: {cache_info}")
    
    print("Demo completed")