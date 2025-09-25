"""
AutoML Integration for DuetMind Adaptive

Implements automated hyperparameter optimization and model selection
using Optuna for efficient search across parameter space.
"""

import logging
import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime
import pickle
import json
from pathlib import Path

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

logger = logging.getLogger(__name__)

class AutoMLOptimizer:
    """
    AutoML system for automated hyperparameter optimization and model selection.
    
    Features:
    - Multi-algorithm support (RandomForest, LogisticRegression, XGBoost)
    - Bayesian optimization with Optuna
    - Cross-validation based evaluation
    - Automated model selection
    - Hyperparameter importance analysis
    """
    
    def __init__(self,
                 objective_metric: str = 'roc_auc',
                 n_trials: int = 100,
                 timeout: int = 3600,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize AutoML optimizer.
        
        Args:
            objective_metric: Metric to optimize ('roc_auc', 'accuracy', 'f1')
            n_trials: Maximum number of optimization trials
            timeout: Timeout in seconds for optimization
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.objective_metric = objective_metric
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Supported algorithms and their parameter spaces
        self.algorithms = {
            'random_forest': self._optimize_random_forest,
            'logistic_regression': self._optimize_logistic_regression
        }
        
        # Try to import XGBoost
        try:
            import xgboost as xgb
            self.algorithms['xgboost'] = self._optimize_xgboost
            self.xgb_available = True
        except ImportError:
            self.xgb_available = False
            logger.warning("XGBoost not available, skipping in AutoML")
        
        self.study = None
        self.best_model = None
        self.best_params = None
        self.optimization_history = []
        
    def optimize(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None,
                 algorithms: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run AutoML optimization to find best model and hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            algorithms: List of algorithms to try (default: all available)
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting AutoML optimization...")
        
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        # Validate algorithms
        invalid_algs = set(algorithms) - set(self.algorithms.keys())
        if invalid_algs:
            raise ValueError(f"Invalid algorithms: {invalid_algs}")
        
        # Store data for optimization
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.selected_algorithms = algorithms
        
        # Create study
        study_name = f"automl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[self._log_callback]
        )
        
        # Extract best results
        best_trial = self.study.best_trial
        self.best_params = best_trial.params
        
        # Train final model with best parameters
        algorithm = self.best_params['algorithm']
        model_params = {k: v for k, v in self.best_params.items() if k != 'algorithm'}
        self.best_model = self._create_model(algorithm, model_params)
        self.best_model.fit(X_train, y_train)
        
        # Generate comprehensive results
        results = self._generate_results()
        
        logger.info(f"AutoML optimization completed. Best {self.objective_metric}: {self.study.best_value:.4f}")
        logger.info(f"Best algorithm: {self.best_params['algorithm']}")
        
        return results
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value to maximize
        """
        # Select algorithm
        algorithm = trial.suggest_categorical('algorithm', self.selected_algorithms)
        
        # Get algorithm-specific parameters
        params = self.algorithms[algorithm](trial)
        
        # Create and evaluate model
        model = self._create_model(algorithm, params)
        
        try:
            if self.X_val is not None and self.y_val is not None:
                # Use validation set if provided
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict_proba(self.X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(self.X_val)
                score = self._calculate_metric(self.y_val, y_pred)
            else:
                # Use cross-validation
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=self.objective_metric)
                score = scores.mean()
                
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
        
        return score
    
    def _optimize_random_forest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Optimize RandomForest hyperparameters."""
        return {
            'n_estimators': trial.suggest_int('rf_n_estimators', 10, 300),
            'max_depth': trial.suggest_int('rf_max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
            'random_state': self.random_state
        }
    
    def _optimize_logistic_regression(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Optimize LogisticRegression hyperparameters."""
        return {
            'C': trial.suggest_float('lr_C', 0.01, 100.0, log=True),
            'penalty': trial.suggest_categorical('lr_penalty', ['l1', 'l2', 'elasticnet']),
            'solver': trial.suggest_categorical('lr_solver', ['liblinear', 'saga']),
            'max_iter': trial.suggest_int('lr_max_iter', 100, 1000),
            'random_state': self.random_state
        }
    
    def _optimize_xgboost(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters."""
        return {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 10, 300),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.0, 10.0),
            'random_state': self.random_state
        }
    
    def _create_model(self, algorithm: str, params: Dict[str, Any]):
        """Create model instance with given parameters."""
        if algorithm == 'random_forest':
            return RandomForestClassifier(**params)
        elif algorithm == 'logistic_regression':
            return LogisticRegression(**params)
        elif algorithm == 'xgboost' and self.xgb_available:
            import xgboost as xgb
            return xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the specified metric."""
        if self.objective_metric == 'accuracy':
            y_pred_binary = (y_pred > 0.5).astype(int) if len(np.unique(y_pred)) > 2 else y_pred
            return accuracy_score(y_true, y_pred_binary)
        elif self.objective_metric == 'roc_auc':
            return roc_auc_score(y_true, y_pred)
        elif self.objective_metric == 'f1':
            y_pred_binary = (y_pred > 0.5).astype(int) if len(np.unique(y_pred)) > 2 else y_pred
            return f1_score(y_true, y_pred_binary)
        else:
            raise ValueError(f"Unknown metric: {self.objective_metric}")
    
    def _log_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback to log optimization progress."""
        self.optimization_history.append({
            'trial': trial.number,
            'value': trial.value,
            'params': trial.params,
            'datetime': datetime.now().isoformat()
        })
        
        if trial.number % 10 == 0:
            logger.info(f"Trial {trial.number}: {self.objective_metric}={trial.value:.4f}")
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive optimization results."""
        results = {
            'best_score': self.study.best_value,
            'best_params': self.best_params,
            'best_algorithm': self.best_params['algorithm'],
            'n_trials': len(self.study.trials),
            'optimization_time': sum(t.duration.total_seconds() for t in self.study.trials if t.duration),
            'study_name': self.study.study_name,
            'objective_metric': self.objective_metric
        }
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            results['param_importance'] = importance
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            results['param_importance'] = {}
        
        # Optimization history
        results['optimization_history'] = self.optimization_history
        
        # Trial statistics
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            scores = [t.value for t in completed_trials]
            results['trial_statistics'] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            }
        
        return results
    
    def save_study(self, filepath: str):
        """Save the complete study for later analysis."""
        study_data = {
            'study': self.study,
            'best_model': self.best_model,
            'results': self._generate_results()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(study_data, f)
        
        logger.info(f"AutoML study saved to {filepath}")
    
    def load_study(self, filepath: str):
        """Load a previously saved study."""
        with open(filepath, 'rb') as f:
            study_data = pickle.load(f)
        
        self.study = study_data['study']
        self.best_model = study_data['best_model']
        self.best_params = self.study.best_params
        
        logger.info(f"AutoML study loaded from {filepath}")

def create_automl_optimizer(**kwargs) -> AutoMLOptimizer:
    """Factory function to create AutoML optimizer."""
    return AutoMLOptimizer(**kwargs)

def run_automl_pipeline(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: Optional[np.ndarray] = None,
                       y_val: Optional[np.ndarray] = None,
                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to run complete AutoML pipeline.
    
    Args:
        X_train: Training features
        y_train: Training targets  
        X_val: Validation features
        y_val: Validation targets
        config: AutoML configuration
        
    Returns:
        AutoML results dictionary
    """
    if config is None:
        config = {}
    
    optimizer = AutoMLOptimizer(**config)
    results = optimizer.optimize(X_train, y_train, X_val, y_val)
    
    return {
        'optimizer': optimizer,
        'results': results
    }