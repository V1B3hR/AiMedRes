#!/usr/bin/env python3
"""
ALS (Amyotrophic Lateral Sclerosis) Progression Prediction Training Pipeline

This script implements a comprehensive machine learning pipeline for ALS progression prediction
using the Kaggle datasets:
- https://www.kaggle.com/datasets/daniilkrasnoproshin/amyotrophic-lateral-sclerosis-als
- https://www.kaggle.com/datasets/mpwolke/cusersmarildownloadsbramcsv

Features:
- Downloads datasets using kagglehub
- Comprehensive data preprocessing  
- Training of both regression and classification models with 5-fold cross-validation
- Tabular neural network training (MLP) 
- Detailed metrics reporting for both tasks
- Model and preprocessing pipeline persistence
"""

import os
import sys
import logging
import warnings
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, 
                            classification_report, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Advanced ML Libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Neural Network Libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Data Loading
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('als_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ALSMLPRegressor(nn.Module):
    """
    Multi-layer perceptron optimized for ALS progression prediction (regression)
    """
    
    def __init__(self, input_size: int):
        super(ALSMLPRegressor, self).__init__()
        
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second hidden layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Third hidden layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer (single output for regression)
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


class ALSMLPClassifier(nn.Module):
    """
    Multi-layer perceptron optimized for ALS classification
    """
    
    def __init__(self, input_size: int, num_classes: int = 2):
        super(ALSMLPClassifier, self).__init__()
        
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second hidden layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Third hidden layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class ALSTrainingPipeline:
    """
    Complete training pipeline for ALS progression prediction
    """
    
    def __init__(self, output_dir: str = "als_outputs"):
        """
        Initialize the training pipeline
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "preprocessors").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        # Initialize components
        self.data = None
        self.preprocessor = None
        self.label_encoder = None
        self.X_processed = None
        self.y_processed = None
        self.feature_names = None
        self.task_type = None  # 'regression' or 'classification'
        
        # Results storage
        self.regression_results = {}
        self.classification_results = {}
        self.neural_results = {}
        
        logger.info(f"Initialized ALS Training Pipeline. Outputs will be saved to {self.output_dir}")

    def load_data(self, data_path: str = None, dataset_choice: str = "als-progression") -> pd.DataFrame:
        """
        Load ALS disease dataset
        
        Args:
            data_path: Path to local dataset CSV (optional)
            dataset_choice: Which Kaggle dataset to use ("als-progression" or "bram-als")
        
        Returns:
            Loaded DataFrame
        """
        if data_path and os.path.exists(data_path):
            logger.info(f"Loading data from local path: {data_path}")
            self.data = pd.read_csv(data_path)
        elif KAGGLEHUB_AVAILABLE:
            logger.info(f"Downloading ALS dataset from Kaggle ({dataset_choice})...")
            try:
                if dataset_choice == "als-progression":
                    # Primary ALS dataset
                    dataset_path = kagglehub.dataset_download("daniilkrasnoproshin/amyotrophic-lateral-sclerosis-als")
                    csv_files = list(Path(dataset_path).glob("*.csv"))
                    if csv_files:
                        self.data = pd.read_csv(csv_files[0])
                        logger.info(f"Loaded dataset from: {csv_files[0]}")
                    else:
                        raise FileNotFoundError("No CSV files found in downloaded dataset")
                        
                elif dataset_choice == "bram-als":
                    # Alternative ALS dataset
                    dataset_path = kagglehub.dataset_download("mpwolke/cusersmarildownloadsbramcsv")
                    csv_files = list(Path(dataset_path).glob("*.csv"))
                    if csv_files:
                        self.data = pd.read_csv(csv_files[0])
                        logger.info(f"Loaded dataset from: {csv_files[0]}")
                    else:
                        raise FileNotFoundError("No CSV files found in downloaded dataset")
                        
            except Exception as e:
                logger.warning(f"Failed to download from Kaggle: {e}")
                logger.info("Creating sample ALS dataset for demonstration")
                self.data = self._create_sample_als_data()
        else:
            logger.warning("Kagglehub not available, creating sample dataset")
            self.data = self._create_sample_als_data()
        
        logger.info(f"Dataset loaded successfully. Shape: {self.data.shape}")
        logger.info(f"Columns: {list(self.data.columns)}")
        
        return self.data

    def _create_sample_als_data(self) -> pd.DataFrame:
        """Create sample ALS dataset for demonstration"""
        np.random.seed(42)
        
        n_samples = 1000
        
        # ALS-specific features based on clinical assessments
        data = {
            # ALSFRS-R (ALS Functional Rating Scale - Revised) components
            'ALSFRS_R_Total': np.random.randint(0, 48, n_samples),  # Total score 0-48
            'ALSFRS_Speech': np.random.randint(0, 4, n_samples),
            'ALSFRS_Salivation': np.random.randint(0, 4, n_samples),
            'ALSFRS_Swallowing': np.random.randint(0, 4, n_samples),
            'ALSFRS_Handwriting': np.random.randint(0, 4, n_samples),
            'ALSFRS_Walking': np.random.randint(0, 4, n_samples),
            'ALSFRS_Breathing': np.random.randint(0, 4, n_samples),
            
            # Vital capacity and respiratory function
            'FVC_percent': np.random.uniform(20, 100, n_samples),  # Forced Vital Capacity %
            'FVC_liters': np.random.uniform(1.0, 5.0, n_samples),
            
            # Demographics and disease characteristics
            'Age_at_onset': np.random.uniform(40, 80, n_samples),
            'Disease_duration_months': np.random.exponential(18, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Site_of_onset': np.random.choice(['Bulbar', 'Limb'], n_samples, p=[0.25, 0.75]),
            
            # Laboratory values
            'Creatinine': np.random.uniform(0.5, 2.0, n_samples),
            'Albumin': np.random.uniform(3.0, 5.0, n_samples),
            'ALT': np.random.uniform(10, 60, n_samples),
            
            # BMI and physical measurements
            'BMI': np.random.normal(25, 4, n_samples),
            'Weight_kg': np.random.normal(70, 15, n_samples),
            'Height_cm': np.random.normal(170, 10, n_samples),
            
            # Treatment indicators
            'Riluzole': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'Edaravone': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'NIV_usage': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Non-invasive ventilation
        }
        
        df = pd.DataFrame(data)
        
        # Create progression rate (primary target for regression)
        # Lower ALSFRS-R, lower FVC, and longer disease duration indicate faster progression
        progression_score = (
            (48 - df['ALSFRS_R_Total']) / 48 * 0.4 +
            (100 - df['FVC_percent']) / 100 * 0.3 +
            np.clip(df['Disease_duration_months'] / 60, 0, 1) * 0.2 +
            np.random.normal(0, 0.1, n_samples)
        )
        df['Progression_Rate'] = np.clip(progression_score, 0, 1)
        
        # Create binary progression status for classification
        df['Fast_Progression'] = (df['Progression_Rate'] > 0.6).astype(int)
        
        # Create survival months (secondary target for regression)
        base_survival = np.random.exponential(36, n_samples)  # Base survival ~3 years
        survival_modifier = 1 - df['Progression_Rate'] * 0.7  # Faster progression = shorter survival
        df['Survival_Months'] = np.clip(base_survival * survival_modifier, 3, 120)  # 3 months to 10 years
        
        # Add patient IDs
        df['Subject_ID'] = [f'ALS_{i:04d}' for i in range(n_samples)]
        
        return df

    def preprocess_data(self, target_column: str = None, task_type: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the ALS dataset for machine learning
        
        Args:
            target_column: Name of target column (auto-detects if None)
            task_type: 'regression', 'classification', or 'auto' to detect
            
        Returns:
            Tuple of (X_processed, y_processed)
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data().")
        
        logger.info("Starting data preprocessing...")
        
        # Auto-detect target column and task type if not specified
        if target_column is None:
            if task_type == 'regression' or task_type == 'auto':
                regression_targets = ['Progression_Rate', 'ALSFRS_R_Total', 'FVC_percent', 'Survival_Months']
                for col in regression_targets:
                    if col in self.data.columns:
                        target_column = col
                        self.task_type = 'regression'
                        break
            
            if target_column is None:  # Still none, try classification
                classification_targets = ['Fast_Progression', 'status', 'class', 'target', 'label']
                for col in classification_targets:
                    if col in self.data.columns:
                        target_column = col
                        self.task_type = 'classification'
                        break
            
            if target_column is None:
                # Use last numeric column as default
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                target_column = numeric_cols[-1]
                self.task_type = 'regression'
                logger.warning(f"Could not auto-detect target column. Using: {target_column} for regression")
        else:
            # Determine task type from target column
            if task_type == 'auto':
                if target_column in ['Fast_Progression', 'status', 'class', 'label']:
                    self.task_type = 'classification'
                elif len(self.data[target_column].unique()) <= 10:
                    self.task_type = 'classification'
                else:
                    self.task_type = 'regression'
            else:
                self.task_type = task_type
        
        logger.info(f"Using target column: {target_column} for {self.task_type}")
        
        # Remove non-feature columns
        feature_df = self.data.copy()
        
        non_feature_cols = ['Subject_ID', 'subject_id', 'patient_id', 'id', 'name']
        for col in non_feature_cols:
            if col in feature_df.columns:
                feature_df = feature_df.drop(columns=[col])
                logger.info(f"Removed non-feature column: {col}")
        
        # Separate features and target
        X = feature_df.drop(columns=[target_column])
        y = feature_df[target_column]
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Handle target encoding
        if self.task_type == 'classification':
            if y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                y_processed = self.label_encoder.fit_transform(y)
                logger.info(f"Encoded target labels: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
            else:
                y_processed = y.values
                self.label_encoder = None
        else:  # regression
            y_processed = y.values.astype(float)
            self.label_encoder = None
        
        # Identify categorical and numerical columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        logger.info(f"Found {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
        
        # Create preprocessing pipeline
        preprocessors = []
        
        if numerical_features:
            numerical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            preprocessors.append(('num', numerical_transformer, numerical_features))
        
        if categorical_features:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            preprocessors.append(('cat', categorical_transformer, categorical_features))
        
        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=preprocessors,
            remainder='drop'
        )
        
        # Fit and transform the data
        X_processed = self.preprocessor.fit_transform(X)
        
        # Store processed data
        self.X_processed = X_processed
        self.y_processed = y_processed
        
        logger.info(f"Preprocessing completed. Shape: {X_processed.shape}")
        
        if self.task_type == 'classification':
            logger.info(f"Target distribution: {np.bincount(y_processed.astype(int))}")
        else:
            logger.info(f"Target statistics: mean={y_processed.mean():.3f}, std={y_processed.std():.3f}")
        
        return X_processed, y_processed

    def train_regression_models(self, n_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Train regression models for ALS progression prediction
        
        Args:
            n_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary with model results
        """
        if self.X_processed is None or self.y_processed is None:
            raise ValueError("Data must be preprocessed first. Call preprocess_data().")
        
        if self.task_type != 'regression':
            logger.warning("Current task type is not regression. Skipping regression models.")
            return {}
        
        logger.info("Training regression models...")
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf')
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(random_state=42, eval_metric='rmse')
        
        # Scoring metrics for regression
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        results = {}
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                cv_results = cross_validate(model, self.X_processed, self.y_processed, 
                                          cv=cv, scoring=scoring, return_train_score=False)
                
                results[name] = {
                    'mse_mean': -cv_results['test_neg_mean_squared_error'].mean(),
                    'mse_std': cv_results['test_neg_mean_squared_error'].std(),
                    'mae_mean': -cv_results['test_neg_mean_absolute_error'].mean(),
                    'mae_std': cv_results['test_neg_mean_absolute_error'].std(),
                    'r2_mean': cv_results['test_r2'].mean(),
                    'r2_std': cv_results['test_r2'].std(),
                    'rmse_mean': np.sqrt(-cv_results['test_neg_mean_squared_error'].mean())
                }
                
                # Train final model on full dataset
                model.fit(self.X_processed, self.y_processed)
                
                # Save model
                model_path = self.output_dir / "models" / f"regression_{name.replace(' ', '_').lower()}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
                logger.info(f"{name} - RMSE: {results[name]['rmse_mean']:.3f}, R²: {results[name]['r2_mean']:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                
        self.regression_results = results
        return results

    def train_classification_models(self, n_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Train classification models for ALS progression prediction
        
        Args:
            n_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary with model results
        """
        if self.X_processed is None or self.y_processed is None:
            raise ValueError("Data must be preprocessed first. Call preprocess_data().")
        
        if self.task_type != 'classification':
            logger.warning("Current task type is not classification. Skipping classification models.")
            return {}
        
        logger.info("Training classification models...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        # Scoring metrics
        scoring = ['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc']
        
        results = {}
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                cv_results = cross_validate(model, self.X_processed, self.y_processed, 
                                          cv=cv, scoring=scoring, return_train_score=False)
                
                results[name] = {
                    'accuracy_mean': cv_results['test_accuracy'].mean(),
                    'accuracy_std': cv_results['test_accuracy'].std(),
                    'balanced_accuracy_mean': cv_results['test_balanced_accuracy'].mean(),
                    'balanced_accuracy_std': cv_results['test_balanced_accuracy'].std(),
                    'f1_macro_mean': cv_results['test_f1_macro'].mean(),
                    'f1_macro_std': cv_results['test_f1_macro'].std(),
                    'roc_auc_mean': cv_results['test_roc_auc'].mean(),
                    'roc_auc_std': cv_results['test_roc_auc'].std()
                }
                
                # Train final model on full dataset
                model.fit(self.X_processed, self.y_processed)
                
                # Save model
                model_path = self.output_dir / "models" / f"classification_{name.replace(' ', '_').lower()}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
                logger.info(f"{name} - Accuracy: {results[name]['accuracy_mean']:.3f} (±{results[name]['accuracy_std']:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                
        self.classification_results = results
        return results

    def train_neural_network(self, epochs: int = 100, batch_size: int = 32) -> Dict[str, float]:
        """
        Train neural network for ALS prediction
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training results
        """
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping neural network training")
            return {}
        
        if self.X_processed is None or self.y_processed is None:
            raise ValueError("Data must be preprocessed first. Call preprocess_data().")
        
        logger.info(f"Training neural network for {self.task_type}...")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(self.X_processed)
        
        if self.task_type == 'classification':
            y_tensor = torch.LongTensor(self.y_processed.astype(int))
        else:  # regression
            y_tensor = torch.FloatTensor(self.y_processed.astype(float))
        
        # Split data
        if self.task_type == 'classification':
            X_train, X_val, y_train, y_val = train_test_split(
                X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_tensor, y_tensor, test_size=0.2, random_state=42
            )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_size = X_tensor.shape[1]
        
        if self.task_type == 'classification':
            num_classes = len(np.unique(self.y_processed))
            model = ALSMLPClassifier(input_size, num_classes)
            criterion = nn.CrossEntropyLoss()
        else:
            model = ALSMLPRegressor(input_size)
            criterion = nn.MSELoss()
        
        # Optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_metrics = []
        best_val_metric = float('inf') if self.task_type == 'regression' else 0.0
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                if self.task_type == 'regression':
                    outputs = outputs.squeeze()
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                if self.task_type == 'classification':
                    val_correct = 0
                    val_total = 0
                    
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                    
                    val_metric = val_correct / val_total  # accuracy
                    val_metrics.append(val_metric)
                    scheduler.step(-val_metric)  # For ReduceLROnPlateau
                    
                    # Early stopping (maximize accuracy)
                    if val_metric > best_val_metric:
                        best_val_metric = val_metric
                        patience_counter = 0
                        torch.save(model.state_dict(), self.output_dir / "models" / "neural_network_best.pth")
                    else:
                        patience_counter += 1
                        
                else:  # regression
                    val_predictions = []
                    val_targets = []
                    
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        val_predictions.extend(outputs.cpu().numpy())
                        val_targets.extend(batch_y.cpu().numpy())
                    
                    val_metric = np.sqrt(mean_squared_error(val_targets, val_predictions))  # RMSE
                    val_metrics.append(val_metric)
                    scheduler.step(val_metric)  # For ReduceLROnPlateau
                    
                    # Early stopping (minimize RMSE)
                    if val_metric < best_val_metric:
                        best_val_metric = val_metric
                        patience_counter = 0
                        torch.save(model.state_dict(), self.output_dir / "models" / "neural_network_best.pth")
                    else:
                        patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 20 == 0:
                metric_name = 'Accuracy' if self.task_type == 'classification' else 'RMSE'
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Val {metric_name}: {val_metric:.4f}")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            if self.task_type == 'classification':
                val_outputs = model(X_val)
                _, val_pred = torch.max(val_outputs, 1)
                
                val_accuracy = (val_pred == y_val).float().mean().item()
                val_pred_np = val_pred.numpy()
                y_val_np = y_val.numpy()
                val_f1 = f1_score(y_val_np, val_pred_np, average='macro')
                
                results = {
                    'best_validation_accuracy': best_val_metric,
                    'final_validation_accuracy': val_accuracy,
                    'validation_f1_macro': val_f1,
                    'epochs_trained': len(train_losses)
                }
                
                logger.info(f"Neural Network - Best Val Acc: {best_val_metric:.3f}, Final F1: {val_f1:.3f}")
                
            else:  # regression
                val_outputs = model(X_val).squeeze()
                val_predictions = val_outputs.numpy()
                y_val_np = y_val.numpy()
                
                final_rmse = np.sqrt(mean_squared_error(y_val_np, val_predictions))
                final_mae = mean_absolute_error(y_val_np, val_predictions)
                final_r2 = r2_score(y_val_np, val_predictions)
                
                results = {
                    'best_validation_rmse': best_val_metric,
                    'final_validation_rmse': final_rmse,
                    'validation_mae': final_mae,
                    'validation_r2': final_r2,
                    'epochs_trained': len(train_losses)
                }
                
                logger.info(f"Neural Network - Best Val RMSE: {best_val_metric:.3f}, Final R²: {final_r2:.3f}")
        
        self.neural_results = results
        return results

    def save_preprocessor(self):
        """Save the data preprocessor and label encoder"""
        if self.preprocessor is not None:
            preprocessor_path = self.output_dir / "preprocessors" / "preprocessor.pkl"
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
        
        if self.label_encoder is not None:
            encoder_path = self.output_dir / "preprocessors" / "label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive training report
        
        Returns:
            Dictionary containing all results and metadata
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'task_type': self.task_type,
            'dataset_info': {
                'shape': self.data.shape if self.data is not None else None,
                'features': len(self.feature_names) if self.feature_names else None,
                'feature_names': self.feature_names
            },
            'regression_results': self.regression_results,
            'classification_results': self.classification_results,
            'neural_results': self.neural_results,
            'output_directory': str(self.output_dir)
        }
        
        # Save report
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create summary
        self._create_summary_report(report)
        
        return report

    def _create_summary_report(self, report: Dict[str, Any]):
        """Create human-readable summary report"""
        summary_path = self.output_dir / "summary_report.txt"
        
        with open(summary_path, 'w') as f:
            f.write("ALS PROGRESSION PREDICTION TRAINING REPORT\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Generated: {report['timestamp']}\n")
            f.write(f"Task Type: {report['task_type'].title()}\n\n")
            
            if report['dataset_info']['shape']:
                f.write(f"Dataset Information:\n")
                f.write(f"- Shape: {report['dataset_info']['shape']}\n")
                f.write(f"- Features: {report['dataset_info']['features']}\n\n")
            
            if report['regression_results']:
                f.write("Regression Model Results:\n")
                f.write("-" * 26 + "\n")
                for model, metrics in report['regression_results'].items():
                    f.write(f"{model}:\n")
                    f.write(f"  RMSE: {metrics['rmse_mean']:.3f}\n")
                    f.write(f"  MAE: {metrics['mae_mean']:.3f} (±{metrics['mae_std']:.3f})\n")
                    f.write(f"  R²: {metrics['r2_mean']:.3f} (±{metrics['r2_std']:.3f})\n\n")
            
            if report['classification_results']:
                f.write("Classification Model Results:\n")
                f.write("-" * 30 + "\n")
                for model, metrics in report['classification_results'].items():
                    f.write(f"{model}:\n")
                    f.write(f"  Accuracy: {metrics['accuracy_mean']:.3f} (±{metrics['accuracy_std']:.3f})\n")
                    f.write(f"  Balanced Accuracy: {metrics['balanced_accuracy_mean']:.3f} (±{metrics['balanced_accuracy_std']:.3f})\n")
                    f.write(f"  F1 Score: {metrics['f1_macro_mean']:.3f} (±{metrics['f1_macro_std']:.3f})\n")
                    f.write(f"  ROC AUC: {metrics['roc_auc_mean']:.3f} (±{metrics['roc_auc_std']:.3f})\n\n")
            
            if report['neural_results']:
                f.write("Neural Network Results:\n")
                f.write("-" * 22 + "\n")
                if report['task_type'] == 'classification':
                    f.write(f"Best Validation Accuracy: {report['neural_results']['best_validation_accuracy']:.3f}\n")
                    f.write(f"Final Validation F1: {report['neural_results']['validation_f1_macro']:.3f}\n")
                else:
                    f.write(f"Best Validation RMSE: {report['neural_results']['best_validation_rmse']:.3f}\n")
                    f.write(f"Final Validation R²: {report['neural_results']['validation_r2']:.3f}\n")
                f.write(f"Epochs Trained: {report['neural_results']['epochs_trained']}\n\n")
            
            f.write(f"Models and outputs saved in: {report['output_directory']}\n")

    def run_full_pipeline(self, data_path: str = None, epochs: int = 100, 
                         n_folds: int = 5, dataset_choice: str = "als-progression", 
                         task_type: str = 'auto', target_column: str = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            data_path: Path to dataset CSV (optional)
            epochs: Number of epochs for neural network
            n_folds: Number of folds for cross-validation
            dataset_choice: Which Kaggle dataset to use
            task_type: 'regression', 'classification', or 'auto'
            target_column: Name of target column (auto-detects if None)
            
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting ALS Progression Prediction Training Pipeline...")
        
        try:
            # Load data
            self.load_data(data_path, dataset_choice)
            
            # Preprocess data
            self.preprocess_data(target_column, task_type)
            
            # Train models based on task type
            regression_results = {}
            classification_results = {}
            
            if self.task_type == 'regression':
                regression_results = self.train_regression_models(n_folds)
            elif self.task_type == 'classification':
                classification_results = self.train_classification_models(n_folds)
            
            # Train neural network
            neural_results = self.train_neural_network(epochs)
            
            # Save preprocessors
            self.save_preprocessor()
            
            # Generate report
            report = self.generate_report()
            
            logger.info("ALS training pipeline completed successfully!")
            logger.info(f"Results saved to: {self.output_dir}")
            
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """
    Main entry point for the training script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="ALS Progression Prediction Training Pipeline")
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to dataset CSV file (optional, will download from Kaggle if not provided)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs for neural network training')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--dataset-choice', type=str, default='als-progression',
                       choices=['als-progression', 'bram-als'],
                       help='Which Kaggle dataset to use')
    parser.add_argument('--task-type', type=str, default='auto',
                       choices=['regression', 'classification', 'auto'],
                       help='Type of ML task to perform')
    parser.add_argument('--target-column', type=str, default=None,
                       help='Name of target column (auto-detects if not provided)')
    parser.add_argument('--output-dir', type=str, default='als_outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    pipeline = ALSTrainingPipeline(output_dir=args.output_dir)
    
    try:
        results = pipeline.run_full_pipeline(
            data_path=args.data_path,
            epochs=args.epochs,
            n_folds=args.folds,
            dataset_choice=args.dataset_choice,
            task_type=args.task_type,
            target_column=args.target_column
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Task Type: {results.get('task_type', 'Unknown')}")
        print(f"Results saved to: {args.output_dir}")
        
        # Print quick summary
        if results.get('regression_results'):
            print(f"\nBest Regression Model Performance:")
            best_model = min(results['regression_results'].items(), 
                           key=lambda x: x[1]['rmse_mean'])
            print(f"  {best_model[0]}: {best_model[1]['rmse_mean']:.3f} RMSE, {best_model[1]['r2_mean']:.3f} R²")
        
        if results.get('classification_results'):
            print(f"\nBest Classification Model Performance:")
            best_model = max(results['classification_results'].items(), 
                           key=lambda x: x[1]['accuracy_mean'])
            print(f"  {best_model[0]}: {best_model[1]['accuracy_mean']:.3f} accuracy")
        
        if results.get('neural_results'):
            if results.get('task_type') == 'classification':
                print(f"Neural Network: {results['neural_results']['best_validation_accuracy']:.3f} best validation accuracy")
            else:
                print(f"Neural Network: {results['neural_results']['best_validation_rmse']:.3f} best validation RMSE")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
