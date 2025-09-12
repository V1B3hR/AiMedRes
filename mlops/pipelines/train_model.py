#!/usr/bin/env python3
"""
Model training pipeline for DuetMind Adaptive MLOps.
Trains model, logs to MLflow, produces artifact references.
"""

import logging
import os
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import hashlib
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_file: str = "params.yaml") -> dict:
    """Load parameters from YAML file."""
    with open(params_file, 'r') as f:
        return yaml.safe_load(f)


def compute_feature_hash(df: pd.DataFrame) -> str:
    """Compute deterministic hash of feature schema."""
    schema_items = [(col, str(df[col].dtype)) for col in sorted(df.columns)]
    schema_str = str(schema_items)
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def load_processed_data(params: dict) -> tuple:
    """Load processed features and labels."""
    features_file = params['data']['features']
    labels_file = params['data']['labels']
    
    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        raise FileNotFoundError("Processed data not found. Run build_features.py first.")
    
    logger.info(f"Loading features from {features_file}")
    features_df = pd.read_parquet(features_file)
    
    logger.info(f"Loading labels from {labels_file}")
    labels_df = pd.read_parquet(labels_file)
    
    return features_df, labels_df


def preprocess_features(features_df: pd.DataFrame) -> tuple:
    """
    Preprocess features for training.
    Handle categorical variables and scaling.
    """
    processed_df = features_df.copy()
    encoders = {}
    
    # Handle categorical columns
    categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        logger.info(f"Encoding categorical column: {col}")
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col].astype(str))
        encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numerical_cols) > 0:
        processed_df[numerical_cols] = scaler.fit_transform(processed_df[numerical_cols])
        logger.info(f"Scaled {len(numerical_cols)} numerical columns")
    
    return processed_df, encoders, scaler


def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> RandomForestClassifier:
    """Train the machine learning model."""
    model_params = params['model']
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=model_params['n_estimators'],
        max_depth=model_params['max_depth'],
        random_state=model_params['random_state'],
        n_jobs=-1
    )
    
    logger.info("Training RandomForest model...")
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model performance."""
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'test_samples': len(y_test)
    }
    
    # ROC AUC if probabilities available
    if y_pred_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            metrics['roc_auc'] = auc
        except ValueError:
            logger.warning("Could not compute ROC AUC")
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


def save_model_artifacts(model, encoders, scaler, feature_hash: str, params: dict) -> dict:
    """Save model and preprocessing artifacts."""
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / f"model_{feature_hash}.pkl"
    joblib.dump(model, model_path)
    
    # Save preprocessors
    encoders_path = models_dir / f"encoders_{feature_hash}.pkl"
    joblib.dump(encoders, encoders_path)
    
    scaler_path = models_dir / f"scaler_{feature_hash}.pkl"
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"Saved model artifacts with hash {feature_hash}")
    
    return {
        'model_path': str(model_path),
        'encoders_path': str(encoders_path),
        'scaler_path': str(scaler_path),
        'feature_hash': feature_hash
    }


def setup_mlflow(params: dict) -> None:
    """Setup MLflow tracking."""
    mlflow_config = params['mlflow']
    
    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    
    # Set experiment
    mlflow.set_experiment(mlflow_config['experiment_name'])
    
    logger.info(f"MLflow tracking URI: {mlflow_config['tracking_uri']}")
    logger.info(f"MLflow experiment: {mlflow_config['experiment_name']}")


def train_pipeline(params: dict) -> dict:
    """
    Main training pipeline function.
    """
    # Setup MLflow
    setup_mlflow(params)
    
    # Load processed data
    features_df, labels_df = load_processed_data(params)
    
    # Compute feature hash for consistency
    feature_hash = compute_feature_hash(features_df)
    logger.info(f"Feature schema hash: {feature_hash}")
    
    # Preprocess features
    processed_features, encoders, scaler = preprocess_features(features_df)
    
    # Extract labels
    target_col = params['features']['target_column']
    y = labels_df[target_col].values
    
    # Train-test split
    test_size = params['model']['test_size']
    random_state = params['model']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        processed_features, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params['model'])
        mlflow.log_param("feature_hash", feature_hash)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        
        # Train model
        model = train_model(X_train, y_train, params)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        if 'roc_auc' in metrics:
            mlflow.log_metric("roc_auc", metrics['roc_auc'])
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="alzheimer_classifier"
        )
        
        # Save local artifacts
        artifacts = save_model_artifacts(model, encoders, scaler, feature_hash, params)
        
        # Log artifact paths
        for key, path in artifacts.items():
            mlflow.log_param(f"artifact_{key}", path)
        
        # Get MLflow run info
        run_info = mlflow.active_run().info
        
        logger.info(f"MLflow run completed: {run_info.run_id}")
        
        return {
            'run_id': run_info.run_id,
            'feature_hash': feature_hash,
            'metrics': metrics,
            'artifacts': artifacts
        }


if __name__ == "__main__":
    # Load parameters
    params = load_params()
    
    # Run training pipeline
    result = train_pipeline(params)
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Results: {result}")