#!/usr/bin/env python3
"""
Baseline Model Training Pipeline

Trains LightGBM and XGBoost baseline models on extracted imaging features.
Includes MLflow logging and basic drift detection setup.
"""

import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import traceback
from typing import Dict, Any, Tuple, List
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import IsolationForest

# MLflow
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.sklearn

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from params_imaging.yaml."""
    config_path = Path('params_imaging.yaml')
    if not config_path.exists():
        # Fall back to params.yaml
        config_path = Path('params.yaml')
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_features_data(config: Dict[str, Any], logger: logging.Logger) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare features and labels for training."""
    
    # Try to load imaging features
    features_dir = Path(config.get('data', {}).get('features_dir', 'outputs/imaging/features'))
    features_path = features_dir / 'features.parquet'
    
    if not features_path.exists():
        features_path = features_dir / 'features.csv'
    
    if not features_path.exists():
        # Fall back to traditional ML features
        logger.warning("No imaging features found, trying traditional ML features")
        processed_dir = Path(config.get('data', {}).get('processed', 'data/processed'))
        features_path = processed_dir / 'features.parquet'
        
        if not features_path.exists():
            raise FileNotFoundError(f"No features found in {features_dir} or {processed_dir}")
    
    # Load features
    logger.info(f"Loading features from {features_path}")
    
    if features_path.suffix == '.parquet':
        features_df = pd.read_parquet(features_path)
    else:
        features_df = pd.read_csv(features_path)
    
    logger.info(f"Loaded features: {features_df.shape}")
    
    # Create synthetic labels if not present
    if 'label' not in features_df.columns and 'target' not in features_df.columns:
        logger.info("No labels found, creating synthetic labels based on features")
        
        # Create binary classification labels based on feature patterns
        # This is for demonstration - in practice, you'd have real labels
        
        # Use volume-based features if available
        volume_cols = [col for col in features_df.columns if 'volume' in col.lower()]
        if volume_cols:
            # Use total brain volume percentile as synthetic label
            volume_col = volume_cols[0]
            threshold = features_df[volume_col].quantile(0.3)  # Bottom 30% as "pathological"
            labels = (features_df[volume_col] < threshold).astype(int)
        else:
            # Use QC-based synthetic labels
            qc_cols = [col for col in features_df.columns if col.startswith('qc_')]
            if qc_cols:
                # Use SNR or quality metrics
                snr_cols = [col for col in qc_cols if 'snr' in col.lower()]
                if snr_cols:
                    threshold = features_df[snr_cols[0]].quantile(0.3)
                    labels = (features_df[snr_cols[0]] < threshold).astype(int)
                else:
                    # Random labels for demo
                    labels = pd.Series(np.random.choice([0, 1], size=len(features_df), p=[0.7, 0.3]))
            else:
                # Random labels for demo
                labels = pd.Series(np.random.choice([0, 1], size=len(features_df), p=[0.7, 0.3]))
        
        features_df['synthetic_label'] = labels
        target_col = 'synthetic_label'
        
    else:
        target_col = 'label' if 'label' in features_df.columns else 'target'
    
    # Prepare features and labels
    y = features_df[target_col]
    
    # Remove non-feature columns
    exclude_cols = [target_col, 'image_id', 'image_path', 'subject_id']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    X = features_df[feature_cols]
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove constant features
    constant_features = X.columns[X.nunique() <= 1]
    if len(constant_features) > 0:
        logger.info(f"Removing {len(constant_features)} constant features")
        X = X.drop(columns=constant_features)
    
    # Remove highly correlated features (optional)
    if len(X.columns) > 50:  # Only if we have many features
        logger.info("Removing highly correlated features")
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > 0.95)
        ]
        X = X.drop(columns=high_corr_features)
        logger.info(f"Removed {len(high_corr_features)} highly correlated features")
    
    logger.info(f"Final feature matrix: {X.shape}")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")
    
    return X, y


def train_lightgbm_model(
    X: pd.DataFrame, 
    y: pd.Series, 
    config: Dict[str, Any],
    logger: logging.Logger
) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    """Train LightGBM baseline model."""
    
    logger.info("Training LightGBM model")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'random_state': 42,
        'n_estimators': 100,
        'verbose': -1
    }
    
    # Train model
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    metrics['cv_accuracy_mean'] = cv_scores.mean()
    metrics['cv_accuracy_std'] = cv_scores.std()
    
    logger.info(f"LightGBM Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"LightGBM ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return model, metrics


def train_xgboost_model(
    X: pd.DataFrame, 
    y: pd.Series, 
    config: Dict[str, Any],
    logger: logging.Logger
) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
    """Train XGBoost baseline model."""
    
    logger.info("Training XGBoost model")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'random_state': 42,
        'eval_metric': 'logloss',
        'verbosity': 0
    }
    
    # Train model
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    metrics['cv_accuracy_mean'] = cv_scores.mean()
    metrics['cv_accuracy_std'] = cv_scores.std()
    
    logger.info(f"XGBoost Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"XGBoost ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return model, metrics


def setup_drift_detection(
    X: pd.DataFrame, 
    logger: logging.Logger
) -> Tuple[IsolationForest, Dict[str, Any]]:
    """Setup basic drift detection for volumetric features."""
    
    logger.info("Setting up drift detection baseline")
    
    # Focus on key volumetric features for drift detection
    drift_features = []
    
    # Select important features for drift monitoring
    volume_features = [col for col in X.columns if 'volume' in col.lower()]
    intensity_features = [col for col in X.columns if 'intensity' in col.lower() or 'mean' in col.lower()]
    qc_features = [col for col in X.columns if col.startswith('qc_')]
    
    # Take top features from each category
    drift_features.extend(volume_features[:3])  # Top 3 volume features
    drift_features.extend(intensity_features[:2])  # Top 2 intensity features  
    drift_features.extend(qc_features[:2])  # Top 2 QC features
    
    # Ensure we have some features
    if not drift_features:
        drift_features = X.columns[:min(5, len(X.columns))].tolist()
    
    drift_features = [f for f in drift_features if f in X.columns]
    
    logger.info(f"Selected {len(drift_features)} features for drift detection: {drift_features}")
    
    # Train isolation forest on reference data
    X_drift = X[drift_features].fillna(X[drift_features].median())
    
    drift_detector = IsolationForest(
        contamination=0.1,  # Expect 10% outliers
        random_state=42,
        n_estimators=100
    )
    
    drift_detector.fit(X_drift)
    
    # Calculate baseline statistics
    baseline_stats = {
        'feature_means': X_drift.mean().to_dict(),
        'feature_stds': X_drift.std().to_dict(),
        'feature_mins': X_drift.min().to_dict(),
        'feature_maxs': X_drift.max().to_dict(),
        'drift_features': drift_features,
        'n_samples': len(X_drift)
    }
    
    logger.info("Drift detection baseline established")
    
    return drift_detector, baseline_stats


def save_models_and_artifacts(
    lgb_model, lgb_metrics,
    xgb_model, xgb_metrics,
    drift_detector, drift_stats,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """Save models and related artifacts."""
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LightGBM model
    lgb_model_path = models_dir / 'lightgbm_baseline.pkl'
    with open(lgb_model_path, 'wb') as f:
        pickle.dump(lgb_model, f)
    
    # Save XGBoost model
    xgb_model_path = models_dir / 'xgboost_baseline.pkl'
    with open(xgb_model_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    
    # Save drift detector
    drift_detector_path = models_dir / 'drift_detector.pkl'
    with open(drift_detector_path, 'wb') as f:
        pickle.dump(drift_detector, f)
    
    # Save drift baseline stats
    drift_stats_path = models_dir / 'drift_baseline_stats.json'
    with open(drift_stats_path, 'w') as f:
        json.dump(drift_stats, f, indent=2)
    
    # Save model comparison
    model_comparison = {
        'lightgbm': lgb_metrics,
        'xgboost': xgb_metrics,
        'best_model': 'lightgbm' if lgb_metrics['roc_auc'] > xgb_metrics['roc_auc'] else 'xgboost',
        'training_timestamp': str(pd.Timestamp.now())
    }
    
    comparison_path = models_dir / 'model_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(model_comparison, f, indent=2)
    
    logger.info(f"Models saved to {models_dir}")
    logger.info(f"Best model: {model_comparison['best_model']}")


def main():
    """Main model training pipeline."""
    logger = setup_logging()
    logger.info("Starting baseline model training pipeline")
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Setup MLflow
        experiment_name = "imaging_baseline_models"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name="baseline_models_training"):
            
            # Load features and labels
            X, y = load_features_data(config, logger)
            
            # Log dataset info
            mlflow.log_metrics({
                'n_samples': len(X),
                'n_features': len(X.columns),
                'n_positive_labels': int(y.sum()),
                'n_negative_labels': int(len(y) - y.sum()),
                'class_balance': float(y.mean())
            })
            
            # Train LightGBM model
            lgb_model, lgb_metrics = train_lightgbm_model(X, y, config, logger)
            
            # Log LightGBM metrics
            for metric_name, value in lgb_metrics.items():
                mlflow.log_metric(f"lightgbm_{metric_name}", value)
            
            # Train XGBoost model
            xgb_model, xgb_metrics = train_xgboost_model(X, y, config, logger)
            
            # Log XGBoost metrics
            for metric_name, value in xgb_metrics.items():
                mlflow.log_metric(f"xgboost_{metric_name}", value)
            
            # Setup drift detection
            drift_detector, drift_stats = setup_drift_detection(X, logger)
            
            # Log drift detection info
            mlflow.log_metrics({
                'drift_features_count': len(drift_stats['drift_features']),
                'drift_baseline_samples': drift_stats['n_samples']
            })
            
            # Save models and artifacts
            save_models_and_artifacts(
                lgb_model, lgb_metrics,
                xgb_model, xgb_metrics, 
                drift_detector, drift_stats,
                config, logger
            )
            
            # Log models to MLflow
            mlflow.lightgbm.log_model(lgb_model, "lightgbm_model")
            mlflow.xgboost.log_model(xgb_model, "xgboost_model")
            mlflow.sklearn.log_model(drift_detector, "drift_detector")
            
            # Log artifacts
            mlflow.log_artifacts("models", artifact_path="models")
            
            logger.info("Baseline model training completed successfully")
            logger.info(f"LightGBM ROC-AUC: {lgb_metrics['roc_auc']:.4f}")
            logger.info(f"XGBoost ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
            
    except Exception as e:
        logger.error(f"Model training pipeline failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()