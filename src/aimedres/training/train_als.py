#!/usr/bin/env python3
"""
Advanced ALS (Amyotrophic Lateral Sclerosis) Progression Prediction Pipeline

Features:
- Unified configuration via dataclass
- Reproducibility seeding (numpy, random, torch)
- Automatic target & task detection (regression/classification)
- Classic ML model suite + optional hyperparameter search
- Neural network (PyTorch MLP) with:
  * Device auto-selection (CUDA / MPS / CPU)
  * Mixed precision (if CUDA)
  * Early stopping & ReduceLROnPlateau
  * Gradient clipping
- Feature engineering & preprocessing with ColumnTransformer
- One-hot feature name recovery
- Optional SHAP + tree feature importances
- Comprehensive logging & structured metrics export
- Flexible CLI toggles
- Graceful degradation if optional libs absent
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import pickle
import random
import logging
import argparse
import warnings
import importlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_validate,
    train_test_split,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score
)

# Silence common spurious warnings (e.g., sklearn future warnings)
warnings.filterwarnings("ignore")

# ------------------------- Logging Setup ------------------------- #
def init_logger(log_file: str = "als_training.log", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("ALS_PIPELINE")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

logger = init_logger()


# ------------------------- Optional Imports ------------------------- #
def _try_import(name: str):
    try:
        module = importlib.import_module(name)
        return module, True
    except ImportError:
        return None, False

xgb, XGBOOST_AVAILABLE = _try_import("xgboost")
torch, PYTORCH_AVAILABLE = _try_import("torch")
if PYTORCH_AVAILABLE:
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

shap, SHAP_AVAILABLE = _try_import("shap")
kagglehub, KAGGLEHUB_AVAILABLE = _try_import("kagglehub")
plt, MPL_AVAILABLE = _try_import("matplotlib.pyplot")
sns, SEABORN_AVAILABLE = _try_import("seaborn")


# ... (imports and previous code remain unchanged)

# ------------------------- Configuration Dataclass ------------------------- #
@dataclass
class TrainingConfig:
    output_dir: str = "als_outputs"
    dataset_choice: str = "als-progression"    # or 'bram-als'
    task_type: str = "auto"                    # 'regression' | 'classification' | 'auto'
    target_column: Optional[str] = None

    # Training parameters
    epochs: int = 50           # <--- Changed from 80 to 50
    batch_size: int = 64       # <--- Changed from 32 to 64
    folds: int = 5
    random_seed: int = 42

    # Neural network toggles
    nn_enabled: bool = True
    classical_enabled: bool = True
    lr: float = 1e-3
    early_stopping_patience: int = 15
    lr_reduction_patience: int = 8
    lr_reduction_factor: float = 0.5
    grad_clip_norm: float = 5.0

    # Hyperparameter search
    use_hyperparam_search: bool = False
    hyperparam_iter: int = 20

    # Optional analysis
    compute_shap: bool = False
    plot_importance: bool = True
    save_feature_matrix: bool = False
    remove_constant_features: bool = True

    # Model toggles
    include_xgboost: bool = True
    include_svm: bool = True
    include_gradient_boosting: bool = True

    # Device preference
    prefer_gpu: bool = True

    # Logging verbosity
    verbose: bool = False

    # Internal metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

# ... (rest of the file remains unchanged)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Advanced ALS Progression Prediction Training Pipeline")
    p.add_argument('--data-path', type=str, default=None, help='Path to CSV (optional)')
    p.add_argument('--output-dir', type=str, default='als_outputs')
    p.add_argument('--dataset-choice', choices=['als-progression', 'bram-als'], default='als-progression')
    p.add_argument('--task-type', choices=['regression', 'classification', 'auto'], default='auto')
    p.add_argument('--target-column', type=str, default=None)

    p.add_argument('--epochs', type=int, default=50)      # <--- Changed default from 80 to 50
    p.add_argument('--batch-size', type=int, default=64)  # <--- Changed default from 32 to 64
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--disable-nn', action='store_true')
    p.add_argument('--disable-classic', action='store_true')

    p.add_argument('--use-hyperparam-search', action='store_true')
    p.add_argument('--hyperparam-iter', type=int, default=20)

    p.add_argument('--no-xgboost', action='store_true')
    p.add_argument('--no-svm', action='store_true')
    p.add_argument('--no-gb', action='store_true')

    p.add_argument('--compute-shap', action='store_true')
    p.add_argument('--no-importance-plot', action='store_true')
    p.add_argument('--save-feature-matrix', action='store_true')

    p.add_argument('--verbose', action='store_true')
    return p

# ... (rest of the file remains unchanged)

# ------------------------- Utility Functions ------------------------- #
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if PYTORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def rmsle(y_true, y_pred):
    return math.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


# ------------------------- Neural Network Modules ------------------------- #
if PYTORCH_AVAILABLE:

    class ALSMLPRegressor(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.30),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.30),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.network(x)

    class ALSMLPClassifier(nn.Module):
        def __init__(self, input_size: int, num_classes: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.35),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.35),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.network(x)


# ------------------------- Core Pipeline ------------------------- #
class ALSTrainingPipeline:
    def __init__(self, config: TrainingConfig):
        self.cfg = config
        set_global_seed(self.cfg.random_seed)

        # Output directories
        self.output_dir = Path(self.cfg.output_dir)
        (self.output_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "preprocessors").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "artifacts").mkdir(exist_ok=True)

        # Data / artifacts
        self.data: Optional[pd.DataFrame] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names_raw: List[str] = []
        self.encoded_feature_names: Optional[List[str]] = None

        self.X_processed: Optional[np.ndarray] = None
        self.y_processed: Optional[np.ndarray] = None
        self.task_type: Optional[str] = None

        # Results
        self.regression_results: Dict[str, Dict[str, float]] = {}
        self.classification_results: Dict[str, Dict[str, float]] = {}
        self.neural_results: Dict[str, float] = {}

        if self.cfg.verbose:
            logger.setLevel(logging.DEBUG)

        logger.info("Initialized Advanced ALS Training Pipeline")
        logger.debug(f"Config:\n{self.cfg.to_json()}")

    # --------------------- Data Loading --------------------- #
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        if data_path and os.path.isfile(data_path):
            logger.info(f"Loading local dataset: {data_path}")
            self.data = pd.read_csv(data_path)
            return self.data

        if KAGGLEHUB_AVAILABLE:
            try:
                logger.info(f"Downloading dataset from Kaggle: {self.cfg.dataset_choice}")
                if self.cfg.dataset_choice == "als-progression":
                    dpath = kagglehub.dataset_download("daniilkrasnoproshin/amyotrophic-lateral-sclerosis-als")
                else:
                    dpath = kagglehub.dataset_download("mpwolke/cusersmarildownloadsbramcsv")
                csvs = list(Path(dpath).glob("*.csv"))
                if not csvs:
                    raise FileNotFoundError("No CSV files discovered in Kaggle payload.")
                self.data = pd.read_csv(csvs[0])
                logger.info(f"Loaded Kaggle dataset: {csvs[0].name} shape={self.data.shape}")
                return self.data
            except Exception as e:
                logger.warning(f"Kaggle download failed: {e}. Using synthetic fallback.")

        logger.warning("Using synthetic ALS dataset (for dev/testing).")
        self.data = self._create_sample_als_data()
        logger.info(f"Synthetic dataset shape={self.data.shape}")
        return self.data

    def _create_sample_als_data(self, n_samples: int = 800) -> pd.DataFrame:
        np.random.seed(self.cfg.random_seed)
        data = {
            'ALSFRS_R_Total': np.random.randint(0, 48, n_samples),
            'ALSFRS_Speech': np.random.randint(0, 4, n_samples),
            'ALSFRS_Salivation': np.random.randint(0, 4, n_samples),
            'ALSFRS_Swallowing': np.random.randint(0, 4, n_samples),
            'ALSFRS_Handwriting': np.random.randint(0, 4, n_samples),
            'ALSFRS_Walking': np.random.randint(0, 4, n_samples),
            'ALSFRS_Breathing': np.random.randint(0, 4, n_samples),
            'FVC_percent': np.random.uniform(20, 100, n_samples),
            'FVC_liters': np.random.uniform(1.0, 5.5, n_samples),
            'Age_at_onset': np.random.uniform(38, 82, n_samples),
            'Disease_duration_months': np.random.exponential(20, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Site_of_onset': np.random.choice(['Bulbar', 'Limb'], n_samples, p=[0.3, 0.7]),
            'Creatinine': np.random.uniform(0.4, 2.2, n_samples),
            'Albumin': np.random.uniform(3.1, 5.1, n_samples),
            'ALT': np.random.uniform(8, 65, n_samples),
            'BMI': np.random.normal(25, 4.5, n_samples),
            'Weight_kg': np.random.normal(72, 13, n_samples),
            'Height_cm': np.random.normal(170, 9, n_samples),
            'Riluzole': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'Edaravone': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'NIV_usage': np.random.choice([0, 1], n_samples, p=[0.68, 0.32]),
        }
        df = pd.DataFrame(data)
        progression = (
            (48 - df['ALSFRS_R_Total']) / 48 * 0.42 +
            (100 - df['FVC_percent']) / 100 * 0.28 +
            np.clip(df['Disease_duration_months'] / 72, 0, 1) * 0.21 +
            np.random.normal(0, 0.08, n_samples)
        )
        df['Progression_Rate'] = np.clip(progression, 0, 1)
        df['Fast_Progression'] = (df['Progression_Rate'] > 0.6).astype(int)
        base_survival = np.random.exponential(40, n_samples)
        survival_mod = 1 - df['Progression_Rate'] * 0.7
        df['Survival_Months'] = np.clip(base_survival * survival_mod, 4, 132)
        df['Subject_ID'] = [f'ALS_{i:05d}' for i in range(n_samples)]
        return df

    # --------------------- Preprocessing --------------------- #
    def preprocess(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data().")
        df = self.data.copy()
        target = self.cfg.target_column

        # Auto-detect target
        if target is None:
            if self.cfg.task_type in ("regression", "auto"):
                for col in ['Progression_Rate', 'ALSFRS_R_Total', 'FVC_percent', 'Survival_Months']:
                    if col in df.columns:
                        target = col
                        self.task_type = 'regression'
                        break
            if target is None:
                for col in ['Fast_Progression', 'status', 'class', 'label', 'target']:
                    if col in df.columns:
                        target = col
                        self.task_type = 'classification'
                        break
            if target is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                target = numeric_cols[-1]
                self.task_type = 'regression'
                logger.warning(f"Defaulting target to {target} (regression).")
        else:
            if self.cfg.task_type == 'auto':
                if target in ['Fast_Progression', 'status', 'class', 'label', 'target']:
                    self.task_type = 'classification'
                elif len(df[target].unique()) <= 10:
                    self.task_type = 'classification'
                else:
                    self.task_type = 'regression'
            else:
                self.task_type = self.cfg.task_type

        logger.info(f"Target: {target} | Task: {self.task_type}")

        non_feature_cols = {'Subject_ID', 'subject_id', 'patient_id', 'id', 'name'}
        drop_cols = [c for c in non_feature_cols if c in df.columns and c != target]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
            logger.debug(f"Dropped non-feature columns: {drop_cols}")

        y = df[target]
        X = df.drop(columns=[target])
        self.feature_names_raw = list(X.columns)

        # Encode target
        if self.task_type == 'classification':
            if y.dtype == object:
                self.label_encoder = LabelEncoder()
                y_processed = self.label_encoder.fit_transform(y)
                logger.info(f"Label classes: {list(self.label_encoder.classes_)}")
            else:
                y_processed = y.astype(int).values
        else:
            y_processed = y.astype(float).values

        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Remove constant features
        if self.cfg.remove_constant_features:
            const_cols = [c for c in num_cols if X[c].nunique() <= 1]
            if const_cols:
                X.drop(columns=const_cols, inplace=True)
                num_cols = [c for c in num_cols if c not in const_cols]
                logger.debug(f"Removed constant features: {const_cols}")

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
            ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]) if cat_cols else None

        transformers = []
        if num_cols:
            transformers.append(("num", numeric_transformer, num_cols))
        if cat_cols:
            transformers.append(("cat", categorical_transformer, cat_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        X_processed = self.preprocessor.fit_transform(X)

        # Extract feature names
        feature_names: List[str] = []
        if num_cols:
            feature_names.extend(num_cols)
        if cat_cols:
            ohe: OneHotEncoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            try:
                cat_features = ohe.get_feature_names_out(cat_cols).tolist()
            except AttributeError:
                # Compatibility fallback (older sklearn)
                cat_features = []
            feature_names.extend(cat_features)

        self.encoded_feature_names = feature_names
        self.X_processed = X_processed
        self.y_processed = y_processed

        logger.info(f"Preprocessing complete. Feature matrix: {X_processed.shape}")
        if self.task_type == 'classification':
            counts = np.bincount(y_processed)
            logger.info(f"Class distribution: {counts.tolist()}")
        else:
            logger.info(f"Target mean={y_processed.mean():.4f} std={y_processed.std():.4f}")

        if self.cfg.save_feature_matrix:
            np.save(self.output_dir / "artifacts" / "X_processed.npy", X_processed)
            np.save(self.output_dir / "artifacts" / "y.npy", y_processed)

    # --------------------- Model Registries --------------------- #
    def _get_regression_models(self) -> Dict[str, Any]:
        models: Dict[str, Any] = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=self.cfg.random_seed),
            'RandomForest': RandomForestRegressor(
                n_estimators=160, random_state=self.cfg.random_seed, n_jobs=-1
            ),
            'SVR': SVR(kernel='rbf', C=10, gamma='scale')
        }
        if self.cfg.include_gradient_boosting:
            models['GradientBoosting'] = GradientBoostingRegressor(random_state=self.cfg.random_seed)
        if self.cfg.include_xgboost and XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(
                random_state=self.cfg.random_seed,
                n_estimators=250,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric='rmse'
            )
        return models

    def _get_classification_models(self) -> Dict[str, Any]:
        models: Dict[str, Any] = {
            'LogisticRegression': LogisticRegression(
                random_state=self.cfg.random_seed, max_iter=2000
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200, random_state=self.cfg.random_seed, n_jobs=-1
            )
        }
        if self.cfg.include_svm:
            models['SVM'] = SVC(
                probability=True, kernel='rbf', C=10, gamma='scale', random_state=self.cfg.random_seed
            )
        if self.cfg.include_xgboost and XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                random_state=self.cfg.random_seed,
                n_estimators=350,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric='logloss',
                use_label_encoder=False
            )
        return models

    # --------------------- Hyperparameter Search --------------------- #
    def _maybe_hyperparam_search(self, model_name: str, model, task: str):
        if not self.cfg.use_hyperparam_search:
            return model
        try:
            logger.info(f"Hyperparameter search for {model_name} ({task})...")
            if task == 'regression':
                param_dist = {
                    'RandomForest': {
                        'n_estimators': [100, 160, 220, 300],
                        'max_depth': [None, 6, 8, 10],
                        'min_samples_split': [2, 5, 10]
                    },
                    'Ridge': {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]},
                    'SVR': {'C': [1, 5, 10, 20], 'gamma': ['scale', 'auto']},
                    'XGBoost': {
                        'n_estimators': [200, 300, 400],
                        'max_depth': [4, 5, 6],
                        'learning_rate': [0.03, 0.05, 0.07]
                    }
                }.get(model_name, {})
                scoring = 'neg_root_mean_squared_error'
            else:
                param_dist = {
                    'RandomForest': {
                        'n_estimators': [150, 200, 300],
                        'max_depth': [None, 6, 8, 10],
                        'min_samples_split': [2, 5, 10]
                    },
                    'LogisticRegression': {
                        'C': [0.1, 0.5, 1.0, 2.0]
                    },
                    'SVM': {'C': [1, 5, 10, 15], 'gamma': ['scale', 'auto']},
                    'XGBoost': {
                        'n_estimators': [250, 350, 450],
                        'max_depth': [4, 5, 6],
                        'learning_rate': [0.03, 0.05, 0.07]
                    }
                }.get(model_name, {})
                scoring = 'f1_macro'

            if not param_dist:
                return model

            search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=min(self.cfg.hyperparam_iter, sum(len(v) for v in param_dist.values())),
                scoring=scoring,
                cv=3,
                random_state=self.cfg.random_seed,
                n_jobs=-1
            )
            search.fit(self.X_processed, self.y_processed)
            logger.info(f"Best params for {model_name}: {search.best_params_}")
            return search.best_estimator_
        except Exception as e:
            logger.warning(f"Hyperparam search failed for {model_name}: {e}")
            return model

    # --------------------- Classic Regression Training --------------------- #
    def train_regression(self) -> Dict[str, Dict[str, float]]:
        if self.task_type != 'regression':
            logger.info("Skipping regression (task type not regression).")
            return {}
        models = self._get_regression_models()
        cv = KFold(n_splits=self.cfg.folds, shuffle=True, random_state=self.cfg.random_seed)
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        results: Dict[str, Dict[str, float]] = {}

        for name, model in models.items():
            logger.info(f"[Regression] {name}")
            model = self._maybe_hyperparam_search(name, model, 'regression')
            try:
                cv_res = cross_validate(model, self.X_processed, self.y_processed, cv=cv, scoring=scoring)
                mse_mean = -cv_res['test_neg_mean_squared_error'].mean()
                mae_mean = -cv_res['test_neg_mean_absolute_error'].mean()
                r2_mean = cv_res['test_r2'].mean()
                results[name] = {
                    'rmse_mean': math.sqrt(mse_mean),
                    'mse_mean': mse_mean,
                    'mae_mean': mae_mean,
                    'r2_mean': r2_mean,
                    'r2_std': cv_res['test_r2'].std()
                }
                model.fit(self.X_processed, self.y_processed)
                self._save_model(model, f"regression_{name}.pkl")
                logger.info(f"{name} -> RMSE={results[name]['rmse_mean']:.4f} R2={r2_mean:.4f}")
            except Exception as e:
                logger.error(f"Regression model {name} failed: {e}")
        self.regression_results = results
        return results

    # --------------------- Classic Classification Training --------------------- #
    def train_classification(self) -> Dict[str, Dict[str, float]]:
        if self.task_type != 'classification':
            logger.info("Skipping classification (task type not classification).")
            return {}
        models = self._get_classification_models()
        cv = StratifiedKFold(n_splits=self.cfg.folds, shuffle=True, random_state=self.cfg.random_seed)
        scoring = ['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc']
        results: Dict[str, Dict[str, float]] = {}

        for name, model in models.items():
            logger.info(f"[Classification] {name}")
            model = self._maybe_hyperparam_search(name, model, 'classification')
            try:
                cv_res = cross_validate(model, self.X_processed, self.y_processed, cv=cv, scoring=scoring)
                results[name] = {
                    'accuracy_mean': cv_res['test_accuracy'].mean(),
                    'balanced_accuracy_mean': cv_res['test_balanced_accuracy'].mean(),
                    'f1_macro_mean': cv_res['test_f1_macro'].mean(),
                    'roc_auc_mean': cv_res['test_roc_auc'].mean()
                }
                model.fit(self.X_processed, self.y_processed)
                self._save_model(model, f"classification_{name}.pkl")
                logger.info(f"{name} -> Acc={results[name]['accuracy_mean']:.4f} F1={results[name]['f1_macro_mean']:.4f}")
            except Exception as e:
                logger.error(f"Classification model {name} failed: {e}")
        self.classification_results = results
        return results

    # --------------------- Neural Network Training --------------------- #
    def train_neural(self) -> Dict[str, float]:
        if not self.cfg.nn_enabled:
            logger.info("Neural network disabled by config.")
            return {}
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not installed - skipping neural network.")
            return {}
        if self.X_processed is None:
            raise ValueError("Call preprocess() before train_neural().")

        device = self._select_device()
        logger.info(f"NN Device: {device}")

        X_tensor = torch.tensor(self.X_processed, dtype=torch.float32)
        if self.task_type == 'classification':
            y_tensor = torch.tensor(self.y_processed, dtype=torch.long)
            stratify = self.y_processed
        else:
            y_tensor = torch.tensor(self.y_processed, dtype=torch.float32)
            stratify = None

        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2,
            random_state=self.cfg.random_seed,
            stratify=stratify if self.task_type == 'classification' else None
        )

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.cfg.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.cfg.batch_size)

        input_size = X_tensor.shape[1]
        if self.task_type == 'classification':
            num_classes = int(np.unique(self.y_processed).size)
            model = ALSMLPClassifier(input_size, num_classes)
            criterion = nn.CrossEntropyLoss()
        else:
            model = ALSMLPRegressor(input_size)
            criterion = nn.MSELoss()

        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.cfg.lr_reduction_patience,
            factor=self.cfg.lr_reduction_factor
        )

        scaler = torch.cuda.amp.GradScaler() if (torch.cuda.is_available()) else None
        best_metric = float('inf') if self.task_type == 'regression' else 0.0
        patience_counter = 0

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                if scaler:
                    with torch.cuda.amp.autocast():
                        out = model(xb)
                        if self.task_type == 'regression':
                            out = out.squeeze()
                        loss = criterion(out, yb)
                    scaler.scale(loss).backward()
                    if self.cfg.grad_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(xb)
                    if self.task_type == 'regression':
                        out = out.squeeze()
                    loss = criterion(out, yb)
                    loss.backward()
                    if self.cfg.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)
                    optimizer.step()
                running_loss += loss.item()

            val_metric = self._evaluate_nn(model, val_loader, criterion, device)
            scheduler.step(val_metric if self.task_type == 'regression' else -val_metric)

            improved = (val_metric < best_metric) if self.task_type == 'regression' else (val_metric > best_metric)
            if improved:
                best_metric = val_metric
                patience_counter = 0
                torch.save(model.state_dict(), self.output_dir / "models" / "neural_best.pth")
            else:
                patience_counter += 1

            if epoch == 1 or epoch % 10 == 0:
                metric_name = "RMSE" if self.task_type == 'regression' else "Accuracy"
                logger.info(f"[NN] Epoch {epoch}/{self.cfg.epochs} loss={running_loss/len(train_loader):.4f} "
                            f"val_{metric_name}={val_metric:.4f}")

            if patience_counter >= self.cfg.early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

        # Load best checkpoint
        model.load_state_dict(torch.load(self.output_dir / "models" / "neural_best.pth"))
        final = self._final_nn_eval(model, X_val, y_val, device)
        self.neural_results = final
        self._save_json(final, self.output_dir / "metrics" / "neural_results.json")
        return final

    def _select_device(self) -> str:
        if not PYTORCH_AVAILABLE:
            return 'cpu'
        if self.cfg.prefer_gpu and torch.cuda.is_available():
            return 'cuda'
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _evaluate_nn(self, model, loader, criterion, device):
        model.eval()
        correct = 0
        total = 0
        preds_all = []
        targets_all = []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                if self.task_type == 'classification':
                    _, pred = torch.max(outputs, 1)
                    total += yb.size(0)
                    correct += (pred == yb).sum().item()
                else:
                    outputs = outputs.squeeze()
                    preds_all.extend(outputs.detach().cpu().numpy())
                    targets_all.extend(yb.detach().cpu().numpy())
        if self.task_type == 'classification':
            return correct / max(total, 1)
        else:
            return math.sqrt(mean_squared_error(targets_all, preds_all))

    def _final_nn_eval(self, model, X_val, y_val, device):
        model.eval()
        with torch.no_grad():
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            outputs = model(X_val)
            if self.task_type == 'classification':
                _, preds = torch.max(outputs, 1)
                acc = (preds == y_val).float().mean().item()
                f1 = f1_score(y_val.cpu().numpy(), preds.cpu().numpy(), average='macro')
                return {
                    'best_validation_accuracy': acc,
                    'validation_f1_macro': f1
                }
            else:
                outputs = outputs.squeeze()
                preds = outputs.cpu().numpy()
                truth = y_val.cpu().numpy()
                rmse = math.sqrt(mean_squared_error(truth, preds))
                mae = mean_absolute_error(truth, preds)
                r2 = r2_score(truth, preds)
                return {
                    'best_validation_rmse': rmse,
                    'validation_mae': mae,
                    'validation_r2': r2
                }

    # --------------------- Feature Importance & SHAP --------------------- #
    def compute_feature_importance(self):
        if self.X_processed is None or self.y_processed is None:
            return
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        # Reload candidate tree models
        candidates = []
        for base in ['RandomForest', 'XGBoost', 'GradientBoosting']:
            fname = f"{'regression' if self.task_type=='regression' else 'classification'}_{base}.pkl"
            path = self.output_dir / "models" / fname
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                    candidates.append((base, model))
                except Exception:
                    pass

        for name, model in candidates:
            if hasattr(model, 'feature_importances_'):
                try:
                    fi = model.feature_importances_
                    df_imp = pd.DataFrame({
                        'feature': self.encoded_feature_names,
                        'importance': fi
                    }).sort_values('importance', ascending=False)
                    df_imp.to_csv(vis_dir / f"feature_importance_{name}.csv", index=False)
                    if MPL_AVAILABLE and SEABORN_AVAILABLE and self.cfg.plot_importance:
                        plt.figure(figsize=(8, 8))
                        top_df = df_imp.head(25)
                        sns.barplot(data=top_df, x='importance', y='feature', palette='viridis')
                        plt.title(f"Top Feature Importances ({name})")
                        plt.tight_layout()
                        plt.savefig(vis_dir / f"feature_importance_{name}.png")
                        plt.close()
                    logger.info(f"Saved feature importance for {name}")
                except Exception as e:
                    logger.warning(f"Failed importance for {name}: {e}")

        if self.cfg.compute_shap and SHAP_AVAILABLE and candidates:
            name, model = candidates[0]
            logger.info(f"Computing SHAP values for {name} (subset for performance)...")
            try:
                if 'xgb' in str(type(model)).lower() or hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                    sample = self.X_processed[: min(1000, self.X_processed.shape[0])]
                    shap_values = explainer.shap_values(sample)
                    shap.summary_plot(shap_values, sample, feature_names=self.encoded_feature_names, show=False)
                    if MPL_AVAILABLE:
                        plt.tight_layout()
                        plt.savefig(vis_dir / f"shap_summary_{name}.png")
                        plt.close()
                        logger.info("Saved SHAP summary plot.")
                else:
                    logger.warning("Model type unsupported for SHAP TreeExplainer.")
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")
        elif self.cfg.compute_shap and not SHAP_AVAILABLE:
            logger.warning("SHAP requested but shap not installed.")

    # --------------------- Persistence & Reporting --------------------- #
    def _save_model(self, model, filename: str):
        path = self.output_dir / "models" / filename
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def save_preprocessing(self):
        if self.preprocessor is not None:
            with open(self.output_dir / "preprocessors" / "preprocessor.pkl", 'wb') as f:
                pickle.dump(self.preprocessor, f)
        if self.label_encoder is not None:
            with open(self.output_dir / "preprocessors" / "label_encoder.pkl", 'wb') as f:
                pickle.dump(self.label_encoder, f)

    def _save_json(self, data: Dict[str, Any], path: Path):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def export_metrics(self):
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'config': asdict(self.cfg),
            'task_type': self.task_type,
            'regression_results': self.regression_results,
            'classification_results': self.classification_results,
            'neural_results': self.neural_results,
            'feature_count': len(self.encoded_feature_names) if self.encoded_feature_names else None
        }
        self._save_json(report, self.output_dir / "metrics" / "training_report.json")

        # Flat metrics CSV
        rows = []
        for model, metrics in self.regression_results.items():
            r = {'model': model, 'type': 'regression'}
            r.update(metrics)
            rows.append(r)
        for model, metrics in self.classification_results.items():
            r = {'model': model, 'type': 'classification'}
            r.update(metrics)
            rows.append(r)
        if rows:
            pd.DataFrame(rows).to_csv(self.output_dir / "metrics" / "model_metrics.csv", index=False)

        # Human summary
        with open(self.output_dir / "metrics" / "summary.txt", 'w') as f:
            f.write("ALS TRAINING SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Task Type: {self.task_type}\n")
            if self.regression_results:
                best_r = min(self.regression_results.items(), key=lambda x: x[1]['rmse_mean'])
                f.write(f"Best Regression: {best_r[0]} RMSE={best_r[1]['rmse_mean']:.4f} R2={best_r[1]['r2_mean']:.4f}\n")
            if self.classification_results:
                best_c = max(self.classification_results.items(), key=lambda x: x[1]['accuracy_mean'])
                f.write(f"Best Classification: {best_c[0]} Acc={best_c[1]['accuracy_mean']:.4f} "
                        f"F1={best_c[1]['f1_macro_mean']:.4f}\n")
            if self.neural_results:
                if self.task_type == 'classification':
                    f.write(f"Neural Best Acc: {self.neural_results.get('best_validation_accuracy')}\n")
                else:
                    f.write(f"Neural Best RMSE: {self.neural_results.get('best_validation_rmse')}\n")

    # --------------------- Orchestration --------------------- #
    def run(self, data_path: Optional[str] = None):
        start = time.time()
        logger.info("=== ALS Pipeline: START ===")
        self.load_data(data_path)
        self.preprocess()

        if self.cfg.classical_enabled:
            if self.task_type == 'regression':
                self.train_regression()
            else:
                self.train_classification()

        if self.cfg.nn_enabled:
            self.train_neural()

        self.save_preprocessing()
        self.compute_feature_importance()
        self.export_metrics()

        logger.info(f"=== ALS Pipeline: DONE ({(time.time()-start)/60:.2f} min) ===")

        return {
            'task_type': self.task_type,
            'regression_results': self.regression_results,
            'classification_results': self.classification_results,
            'neural_results': self.neural_results
        }


# ------------------------- CLI ------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Advanced ALS Progression Prediction Training Pipeline")
    p.add_argument('--data-path', type=str, default=None, help='Path to CSV (optional)')
    p.add_argument('--output-dir', type=str, default='als_outputs')
    p.add_argument('--dataset-choice', choices=['als-progression', 'bram-als'], default='als-progression')
    p.add_argument('--task-type', choices=['regression', 'classification', 'auto'], default='auto')
    p.add_argument('--target-column', type=str, default=None)

    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--disable-nn', action='store_true')
    p.add_argument('--disable-classic', action='store_true')

    p.add_argument('--use-hyperparam-search', action='store_true')
    p.add_argument('--hyperparam-iter', type=int, default=20)

    p.add_argument('--no-xgboost', action='store_true')
    p.add_argument('--no-svm', action='store_true')
    p.add_argument('--no-gb', action='store_true')

    p.add_argument('--compute-shap', action='store_true')
    p.add_argument('--no-importance-plot', action='store_true')
    p.add_argument('--save-feature-matrix', action='store_true')

    p.add_argument('--verbose', action='store_true')
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = TrainingConfig(
        output_dir=args.output_dir,
        dataset_choice=args.dataset_choice,
        task_type=args.task_type,
        target_column=args.target_column,
        epochs=args.epochs,
        batch_size=args.batch_size,
        folds=args.folds,
        random_seed=args.seed,
        nn_enabled=not args.disable_nn,
        classical_enabled=not args.disable_classic,
        use_hyperparam_search=args.use_hyperparam_search,
        hyperparam_iter=args.hyperparam_iter,
        include_xgboost=not args.no_xgboost,
        include_svm=not args.no_svm,
        include_gradient_boosting=not args.no_gb,
        compute_shap=args.compute_shap,
        plot_importance=not args.no_importance_plot,
        save_feature_matrix=args.save_feature_matrix,
        verbose=args.verbose
    )

    pipeline = ALSTrainingPipeline(cfg)
    try:
        results = pipeline.run(data_path=args.data_path)
        print("\n=== Training Complete ===")
        print(f"Task: {results['task_type']}")
        if results['regression_results']:
            best_r = min(results['regression_results'].items(), key=lambda x: x[1]['rmse_mean'])
            print(f"Best Regression: {best_r[0]} RMSE={best_r[1]['rmse_mean']:.4f} R2={best_r[1]['r2_mean']:.4f}")
        if results['classification_results']:
            best_c = max(results['classification_results'].items(), key=lambda x: x[1]['accuracy_mean'])
            print(f"Best Classification: {best_c[0]} Acc={best_c[1]['accuracy_mean']:.4f} "
                  f"F1={best_c[1]['f1_macro_mean']:.4f}")
        if results['neural_results']:
            print(f"Neural Results: {results['neural_results']}")
        print(f"Outputs saved to: {cfg.output_dir}")
    except Exception as e:
        logger.exception(f"Pipeline failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
