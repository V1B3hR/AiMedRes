"""
Structured Alzheimer's Disease Training Pipeline
===============================================

A comprehensive training system for Alzheimer's disease prediction using structured data,
with support for multiple models, early stopping, multi-seed training, and ensemble methods.
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
import time
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Union
from collections import defaultdict
import random

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, roc_auc_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

logger = logging.getLogger('StructuredAlzTrainer')


@dataclass
class EarlyStopState:
    """State tracking for early stopping in MLP training"""
    best_score: float = -1.0
    best_epoch: int = -1
    epochs_no_improve: int = 0
    best_weights: Optional[Tuple] = None


class EarlyStoppingMLP:
    """
    Wrapper for MLPClassifier that implements early stopping with partial_fit
    Supports patience, tracks best macro_f1, and restores best weights
    """
    
    def __init__(self, mlp_classifier: MLPClassifier, patience: int = 8, 
                 min_delta: float = 1e-6, monitor: str = 'macro_f1'):
        self.mlp = mlp_classifier
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.early_stop_state = EarlyStopState()
        
    def fit(self, X_train, y_train, X_val, y_val, epochs: int = 80, batch_size: int = 64):
        """
        Train MLP with early stopping using batch processing
        """
        n_samples = X_train.shape[0]
        indices = np.arange(n_samples)
        epoch_metrics = []
        
        # Get unique classes for partial_fit
        classes = np.unique(np.concatenate([y_train, y_val]))
        
        logger.info(f"Starting MLP training with early stopping (patience={self.patience})")
        
        for epoch in range(1, epochs + 1):
            # Shuffle data for this epoch
            np.random.shuffle(indices)
            
            # Train in batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                if epoch == 1 and start_idx == 0:
                    # First call to partial_fit needs classes parameter
                    self.mlp.partial_fit(X_batch, y_batch, classes=classes)
                else:
                    self.mlp.partial_fit(X_batch, y_batch)
            
            # Evaluate on validation set
            y_pred_val = self.mlp.predict(X_val)
            val_acc = accuracy_score(y_val, y_pred_val)
            val_f1 = f1_score(y_val, y_pred_val, average='macro')
            
            # Choose metric to monitor
            current_score = val_f1 if self.monitor == 'macro_f1' else val_acc
            
            epoch_metrics.append({
                'epoch': epoch,
                'val_accuracy': val_acc,
                'val_macro_f1': val_f1,
                'monitored_metric': current_score
            })
            
            # Check for improvement
            if current_score > self.early_stop_state.best_score + self.min_delta:
                self.early_stop_state.best_score = current_score
                self.early_stop_state.best_epoch = epoch
                self.early_stop_state.epochs_no_improve = 0
                # Save best weights
                self.early_stop_state.best_weights = (
                    [w.copy() for w in self.mlp.coefs_],
                    [b.copy() for b in self.mlp.intercepts_]
                )
            else:
                self.early_stop_state.epochs_no_improve += 1
                
            logger.debug(f"Epoch {epoch}: val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, "
                        f"no_improve={self.early_stop_state.epochs_no_improve}")
                
            # Early stopping check
            if self.early_stop_state.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping at epoch {epoch} (best: {self.early_stop_state.best_epoch})")
                break
        
        # Restore best weights
        if self.early_stop_state.best_weights:
            self.mlp.coefs_, self.mlp.intercepts_ = self.early_stop_state.best_weights
            logger.info(f"Restored best weights from epoch {self.early_stop_state.best_epoch}")
        
        return epoch_metrics
    
    def predict(self, X):
        return self.mlp.predict(X)
    
    def predict_proba(self, X):
        return self.mlp.predict_proba(X)


class StructuredAlzTrainer:
    """
    Main trainer class for structured Alzheimer's disease prediction
    
    Supports multiple models (logistic regression, random forest, MLP),
    ensemble methods, early stopping, multi-seed training, and comprehensive
    metrics tracking with artifact persistence.
    """
    
    def __init__(self, config: Dict[str, Any], seed: int, output_dir: Path, 
                 target_column: Optional[str] = None):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration dictionary
            seed: Random seed for reproducibility 
            output_dir: Directory to save artifacts
            target_column: Name of target column (auto-detected if None)
        """
        self.config = config
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_column = target_column
        
        # Initialize components
        self.models = {}
        self.preprocessor = None
        self.feature_names = None
        self.label_encoder = None
        self.trained_models = {}
        self.epoch_metrics = []
        
        # Configuration shortcuts
        self.primary_metric = config.get('metric_primary', 'macro_f1')
        self.class_weight = config.get('class_weight', 'balanced')
        self.batch_size = config.get('batch_size', 64)
        self.epochs = config.get('epochs', 80)
        self.patience = config.get('patience', 8)
        self.validation_split = config.get('validation_split', 0.2)
        
        logger.info(f"StructuredAlzTrainer initialized (seed={seed}, output_dir={output_dir})")

    def set_seeds(self, seed: Optional[int] = None):
        """Set random seeds for reproducibility"""
        if seed is None:
            seed = self.seed
            
        random.seed(seed)
        np.random.seed(seed)
        
        # Set scikit-learn random states in model parameters
        for model_name in self.config.get('models', []):
            if model_name in self.config.get('model_params', {}):
                self.config['model_params'][model_name]['random_state'] = seed
                
        logger.debug(f"Set random seeds to {seed}")

    def load_dataset(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load dataset from file
        
        Args:
            path: Path to dataset file (CSV, parquet supported)
            
        Returns:
            Loaded DataFrame
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
            
        logger.info(f"Loading dataset from {path}")
        
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
        elif path.suffix.lower() == '.parquet':
            df = pd.read_parquet(path)
        else:
            # Try CSV as default
            df = pd.read_csv(path)
            
        logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def identify_target(self, df: pd.DataFrame) -> str:
        """
        Identify target column in dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of target column
            
        Raises:
            ValueError: If target column cannot be identified
        """
        if self.target_column and self.target_column in df.columns:
            logger.info(f"Using specified target column: {self.target_column}")
            return self.target_column
            
        # Search for common target column names
        candidates = [
            'diagnosis', 'Diagnosis', 'DIAGNOSIS',
            'target', 'Target', 'TARGET',
            'label', 'Label', 'LABEL',
            'class', 'Class', 'CLASS',
            'Group', 'group', 'GROUP'
        ]
        
        for candidate in candidates:
            if candidate in df.columns:
                logger.info(f"Auto-detected target column: {candidate}")
                return candidate
                
        raise ValueError(
            f"Could not auto-detect target column. Available columns: {list(df.columns)}. "
            "Please specify --target-column"
        )

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
        """
        Preprocess dataset with feature engineering and target separation
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, target, preprocessor)
        """
        logger.info("Starting data preprocessing")
        
        # Identify target column
        target_col = self.identify_target(df)
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
        
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        # Check for missing values
        missing_pct = (X.isnull().sum() / len(X) * 100)
        high_missing = missing_pct[missing_pct > self.config.get('preprocessing', {}).get('missing_threshold', 10)]
        
        if len(high_missing) > 0:
            logger.warning(f"High missing values detected:\n{high_missing}")
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Feature types: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
        
        # Build preprocessing pipeline
        transformers = []
        
        if numeric_cols:
            numeric_pipeline = Pipeline([
                ('impute', SimpleImputer(
                    strategy=self.config.get('preprocessing', {}).get('numeric_strategy', 'median')
                )),
                ('scale', StandardScaler())
            ])
            transformers.append(('numeric', numeric_pipeline, numeric_cols))
            
        if categorical_cols:
            categorical_pipeline = Pipeline([
                ('impute', SimpleImputer(
                    strategy=self.config.get('preprocessing', {}).get('categorical_strategy', 'most_frequent')
                )),
                ('onehot', OneHotEncoder(
                    handle_unknown=self.config.get('preprocessing', {}).get('handle_unknown', 'ignore'),
                    sparse_output=False
                ))
            ])
            transformers.append(('categorical', categorical_pipeline, categorical_cols))
        
        if not transformers:
            raise ValueError("No valid features found for preprocessing")
            
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        
        # Store feature information
        self.feature_names = {
            'numeric': numeric_cols,
            'categorical': categorical_cols
        }
        
        logger.info("Preprocessing pipeline created successfully")
        
        return X, y, preprocessor

    def build_models(self, n_classes: int) -> Dict[str, Any]:
        """
        Build model dictionary based on configuration
        
        Args:
            n_classes: Number of target classes
            
        Returns:
            Dictionary of model instances
        """
        logger.info("Building models")
        
        models = {}
        class_weight = self.class_weight if self.class_weight != 'None' else None
        model_params = self.config.get('model_params', {})
        
        for model_name in self.config.get('models', []):
            params = model_params.get(model_name, {}).copy()
            
            if model_name == 'logreg':
                if class_weight:
                    params['class_weight'] = class_weight
                models['logreg'] = LogisticRegression(**params)
                
            elif model_name == 'random_forest':
                if class_weight:
                    params['class_weight'] = class_weight  
                models['random_forest'] = RandomForestClassifier(**params)
                
            elif model_name == 'mlp':
                # MLP doesn't support class_weight directly
                models['mlp'] = MLPClassifier(
                    max_iter=1,  # We'll use partial_fit
                    warm_start=True,
                    **params
                )
            else:
                logger.warning(f"Unknown model type: {model_name}")
        
        # Add ensemble if requested
        if self.config.get('ensemble', False):
            ensemble_params = self.config.get('ensemble_params', {})
            estimator_names = ensemble_params.get('estimators', ['logreg', 'random_forest', 'mlp'])
            
            estimators = []
            for name in estimator_names:
                if name in models:
                    estimators.append((name, models[name]))
                    
            if len(estimators) >= 2:
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting=ensemble_params.get('voting', 'soft'),
                    weights=ensemble_params.get('weights', None),
                    n_jobs=ensemble_params.get('n_jobs', 1)
                )
                models['ensemble'] = ensemble
                logger.info(f"Added ensemble with {len(estimators)} estimators")
            else:
                logger.warning("Not enough models for ensemble, skipping")
        
        self.models = models
        logger.info(f"Built {len(models)} models: {list(models.keys())}")
        
        return models

    def train_model(self, model_name: str, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """
        Train a single model with appropriate method
        
        Args:
            model_name: Name of model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Dictionary containing training results and metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in built models")
            
        model = self.models[model_name]
        logger.info(f"Training model: {model_name}")
        
        start_time = time.time()
        
        if model_name == 'mlp':
            # Use early stopping wrapper for MLP
            early_stopping_config = self.config.get('early_stopping', {})
            mlp_wrapper = EarlyStoppingMLP(
                model,
                patience=early_stopping_config.get('patience', self.patience),
                min_delta=early_stopping_config.get('min_delta', 1e-6),
                monitor=early_stopping_config.get('monitor', self.primary_metric)
            )
            
            epoch_metrics = mlp_wrapper.fit(
                X_train, y_train, X_val, y_val,
                epochs=self.epochs,
                batch_size=self.batch_size
            )
            
            self.epoch_metrics.extend(epoch_metrics)
            trained_model = mlp_wrapper.mlp
            
        else:
            # Standard scikit-learn fit for other models
            model.fit(X_train, y_train)
            trained_model = model
        
        training_time = time.time() - start_time
        
        # Evaluate on validation set
        val_metrics = self.evaluate(trained_model, X_val, y_val, prefix='val_')
        
        # Evaluate on training set
        train_metrics = self.evaluate(trained_model, X_train, y_train, prefix='train_')
        
        # Combine results
        results = {
            'model_name': model_name,
            'training_time': training_time,
            **train_metrics,
            **val_metrics
        }
        
        self.trained_models[model_name] = trained_model
        
        logger.info(f"Completed training {model_name}: "
                   f"val_acc={val_metrics.get('val_accuracy', 0):.4f}, "
                   f"val_f1={val_metrics.get('val_macro_f1', 0):.4f}")
        
        return results

    def evaluate(self, model, X, y, prefix: str = '') -> Dict[str, float]:
        """
        Comprehensive evaluation of model performance
        
        Args:
            model: Trained model to evaluate
            X, y: Data and targets for evaluation
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of computed metrics
        """
        y_pred = model.predict(X)
        
        metrics = {
            f'{prefix}accuracy': accuracy_score(y, y_pred),
            f'{prefix}macro_f1': f1_score(y, y_pred, average='macro'),
            f'{prefix}weighted_f1': f1_score(y, y_pred, average='weighted')
        }
        
        # Per-class F1 scores
        per_class_f1 = f1_score(y, y_pred, average=None)
        unique_classes = np.unique(y)
        
        for i, class_label in enumerate(unique_classes):
            metrics[f'{prefix}f1_class_{class_label}'] = per_class_f1[i]
        
        # ROC AUC (if applicable)
        try:
            if len(unique_classes) == 2:
                # Binary classification
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)[:, 1]
                    metrics[f'{prefix}roc_auc'] = roc_auc_score(y, y_proba)
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X)
                    metrics[f'{prefix}roc_auc'] = roc_auc_score(y, y_scores)
            elif len(unique_classes) > 2:
                # Multi-class classification - use OneVsRest if possible
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)
                    metrics[f'{prefix}roc_auc_ovr'] = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
        except Exception as e:
            logger.debug(f"Could not compute ROC AUC: {e}")
        
        return metrics

    def save_artifacts(self, best_model_name: str, final_model_name: str, 
                      run_metrics: Dict[str, Any]) -> None:
        """
        Save all training artifacts including models, metrics, and reports
        
        Args:
            best_model_name: Name of best performing model
            final_model_name: Name of final model to save
            run_metrics: Dictionary of run metrics to save
        """
        logger.info("Saving training artifacts")
        
        artifacts_config = self.config.get('artifacts', {})
        
        # Save best model
        if artifacts_config.get('save_best_model', True) and best_model_name in self.trained_models:
            best_model_path = self.output_dir / 'best_model.pkl'
            joblib.dump(self.trained_models[best_model_name], best_model_path)
            logger.info(f"Saved best model ({best_model_name}) to {best_model_path}")
        
        # Save final model
        if artifacts_config.get('save_final_model', True) and final_model_name in self.trained_models:
            final_model_path = self.output_dir / 'final_model.pkl'
            joblib.dump(self.trained_models[final_model_name], final_model_path)
            logger.info(f"Saved final model ({final_model_name}) to {final_model_path}")
        
        # Save preprocessor
        if artifacts_config.get('save_preprocessing', True) and self.preprocessor is not None:
            preprocessing_path = self.output_dir / 'preprocessing.pkl'
            joblib.dump(self.preprocessor, preprocessing_path)
            logger.info(f"Saved preprocessor to {preprocessing_path}")
        
        # Save feature names
        if artifacts_config.get('save_feature_names', True) and self.feature_names is not None:
            feature_names_path = self.output_dir / 'feature_names.json'
            with open(feature_names_path, 'w') as f:
                json.dump(self.feature_names, f, indent=2)
        
        # Save metrics
        metrics_config = self.config.get('metrics', {})
        
        if 'json' in metrics_config.get('save_formats', ['json']):
            metrics_path = self.output_dir / 'run_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(run_metrics, f, indent=2, default=str)
        
        # Save epoch metrics for MLP
        if self.epoch_metrics and metrics_config.get('save_per_epoch', True):
            if 'csv' in metrics_config.get('save_formats', ['json']):
                epoch_df = pd.DataFrame(self.epoch_metrics)
                epoch_df.to_csv(self.output_dir / 'epoch_metrics.csv', index=False)
            
            if 'json' in metrics_config.get('save_formats', ['json']):
                with open(self.output_dir / 'epoch_metrics.json', 'w') as f:
                    json.dump(self.epoch_metrics, f, indent=2)
        
        logger.info(f"Artifacts saved to {self.output_dir}")

    def run_full_training(self, data_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Execute complete training pipeline
        
        Args:
            data_path: Path to dataset (overrides config if provided)
            
        Returns:
            Dictionary containing complete training results
        """
        logger.info(f"Starting full training pipeline (seed={self.seed})")
        
        # Set seeds for reproducibility
        self.set_seeds(self.seed)
        
        # Load dataset
        if data_path is None:
            data_path = self.config.get('data_path', '')
            
        if not data_path or not Path(data_path).exists():
            raise ValueError(f"Data path not provided or does not exist: {data_path}")
            
        df = self.load_dataset(data_path)
        
        # Preprocess data
        X, y, preprocessor = self.preprocess(df)
        self.preprocessor = preprocessor
        
        # Fit preprocessor and transform data
        X_transformed = preprocessor.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_transformed, y,
            test_size=self.validation_split,
            random_state=self.seed,
            stratify=y
        )
        
        logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}")
        
        # Build models
        n_classes = len(np.unique(y))
        models = self.build_models(n_classes)
        
        # Train all models
        all_results = {}
        best_model_name = None
        best_score = -1.0
        
        for model_name in models.keys():
            try:
                result = self.train_model(model_name, X_train, y_train, X_val, y_val)
                all_results[model_name] = result
                
                # Track best model by primary metric
                current_score = result.get(f'val_{self.primary_metric}', 0)
                if current_score > best_score:
                    best_score = current_score
                    best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Failed to train model {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
        
        # Prepare final results
        run_metrics = {
            'seed': self.seed,
            'config_profile': self.config.get('profile', 'unknown'),
            'dataset_shape': df.shape,
            'n_classes': n_classes,
            'best_model': best_model_name,
            f'best_val_{self.primary_metric}': best_score,
            'model_results': all_results,
            'training_completed_at': time.time()
        }
        
        # Add best model's key metrics to top level
        if best_model_name and best_model_name in all_results:
            best_results = all_results[best_model_name]
            for key, value in best_results.items():
                if key.startswith('val_'):
                    run_metrics[f'best_{key}'] = value
        
        # Save artifacts
        final_model_name = best_model_name or list(models.keys())[0] if models else None
        if final_model_name:
            self.save_artifacts(best_model_name, final_model_name, run_metrics)
        
        logger.info(f"Training pipeline completed. Best model: {best_model_name} "
                   f"({self.primary_metric}={best_score:.4f})")
        
        return run_metrics