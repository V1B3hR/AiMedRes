#!/usr/bin/env python3
"""
Parkinson's Disease Classification Training Pipeline

This script implements a comprehensive machine learning pipeline for Parkinson's disease classification
using the Kaggle datasets:
- https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set
- https://www.kaggle.com/datasets/anonymous6623/uci-parkinsons-datasets

Features:
- Downloads datasets using kagglehub
- Comprehensive data preprocessing  
- Training of classical models with 5-fold cross-validation
- Tabular neural network training (MLP) 
- Detailed metrics reporting
- Model and preprocessing pipeline persistence
"""

import os
import sys
import logging
import warnings
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report
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
    # Create dummy objects when PyTorch is not available
    class nn:
        class Module:
            def __init__(self):
                pass
        class Sequential:
            def __init__(self, *args):
                pass
        class Linear:
            def __init__(self, *args, **kwargs):
                pass
        class BatchNorm1d:
            def __init__(self, *args, **kwargs):
                pass
        class ReLU:
            def __init__(self, *args, **kwargs):
                pass
        class Dropout:
            def __init__(self, *args, **kwargs):
                pass

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
        logging.FileHandler('parkinsons_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ParkinsonsMLPClassifier(nn.Module):
    """
    Multi-layer perceptron optimized for Parkinson's disease classification
    """
    
    def __init__(self, input_size: int, num_classes: int = 2):
        super(ParkinsonsMLPClassifier, self).__init__()
        
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second hidden layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Third hidden layer
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class ParkinsonsTrainingPipeline:
    """
    Complete training pipeline for Parkinson's disease classification
    """
    
    def __init__(self, output_dir: str = "parkinsons_outputs"):
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
        
        # Results storage
        self.classical_results = {}
        self.neural_results = {}
        
        logger.info(f"Initialized Parkinson's Training Pipeline. Outputs will be saved to {self.output_dir}")

    def load_data(self, data_path: str = None, dataset_choice: str = "vikasukani") -> pd.DataFrame:
        """
        Load Parkinson's disease dataset
        
        Args:
            data_path: Path to local dataset CSV (optional)
            dataset_choice: Which Kaggle dataset to use ("vikasukani" or "uci-parkinsons")
        
        Returns:
            Loaded DataFrame
        """
        if data_path and os.path.exists(data_path):
            logger.info(f"Loading data from local path: {data_path}")
            self.data = pd.read_csv(data_path)
        elif KAGGLEHUB_AVAILABLE:
            logger.info(f"Downloading Parkinson's dataset from Kaggle ({dataset_choice})...")
            try:
                if dataset_choice == "vikasukani":
                    # Primary Parkinson's dataset
                    dataset_path = kagglehub.dataset_download("vikasukani/parkinsons-disease-data-set")
                    # Look for both CSV and data files
                    data_files = list(Path(dataset_path).glob("*.csv")) + list(Path(dataset_path).glob("*.data"))
                    if data_files:
                        self.data = pd.read_csv(data_files[0])
                        logger.info(f"Loaded dataset from: {data_files[0]}")
                    else:
                        raise FileNotFoundError("No CSV or data files found in downloaded dataset")
                        
                elif dataset_choice == "uci-parkinsons":
                    # Alternative UCI Parkinson's dataset
                    dataset_path = kagglehub.dataset_download("anonymous6623/uci-parkinsons-datasets")
                    data_files = list(Path(dataset_path).glob("*.csv")) + list(Path(dataset_path).glob("*.data"))
                    if data_files:
                        self.data = pd.read_csv(data_files[0])
                        logger.info(f"Loaded dataset from: {data_files[0]}")
                    else:
                        raise FileNotFoundError("No CSV or data files found in downloaded dataset")
                        
                elif dataset_choice == "leilahasan":
                    # Dataset mentioned in problem statement
                    dataset_path = kagglehub.dataset_download("leilahasan/parkinson-dataset")
                    data_files = list(Path(dataset_path).glob("*.csv")) + list(Path(dataset_path).glob("*.data"))
                    if data_files:
                        self.data = pd.read_csv(data_files[0])
                        logger.info(f"Loaded dataset from: {data_files[0]}")
                    else:
                        raise FileNotFoundError("No CSV or data files found in downloaded dataset")
                        
            except Exception as e:
                logger.warning(f"Failed to download from Kaggle: {e}")
                logger.info("Creating sample Parkinson's dataset for demonstration")
                self.data = self._create_sample_parkinsons_data()
        else:
            logger.warning("Kagglehub not available, creating sample dataset")
            self.data = self._create_sample_parkinsons_data()
        
        logger.info(f"Dataset loaded successfully. Shape: {self.data.shape}")
        logger.info(f"Columns: {list(self.data.columns)}")
        
        return self.data

    def _create_sample_parkinsons_data(self) -> pd.DataFrame:
        """Create sample Parkinson's dataset for demonstration"""
        np.random.seed(42)
        
        n_samples = 500
        
        # Typical Parkinson's features based on voice measurements
        data = {
            'MDVP:Fo(Hz)': np.random.normal(154, 30, n_samples),  # Fundamental frequency
            'MDVP:Fhi(Hz)': np.random.normal(200, 40, n_samples),  # Maximum vocal fundamental frequency
            'MDVP:Flo(Hz)': np.random.normal(120, 25, n_samples),  # Minimum vocal fundamental frequency
            'MDVP:Jitter(%)': np.random.exponential(0.01, n_samples),  # Frequency variation
            'MDVP:Jitter(Abs)': np.random.exponential(0.0001, n_samples),
            'MDVP:RAP': np.random.exponential(0.005, n_samples),
            'MDVP:PPQ': np.random.exponential(0.005, n_samples),
            'Jitter:DDP': np.random.exponential(0.015, n_samples),
            'MDVP:Shimmer': np.random.exponential(0.03, n_samples),  # Amplitude variation
            'MDVP:Shimmer(dB)': np.random.exponential(0.3, n_samples),
            'Shimmer:APQ3': np.random.exponential(0.02, n_samples),
            'Shimmer:APQ5': np.random.exponential(0.025, n_samples),
            'MDVP:APQ': np.random.exponential(0.03, n_samples),
            'Shimmer:DDA': np.random.exponential(0.06, n_samples),
            'NHR': np.random.exponential(0.03, n_samples),  # Noise-to-harmonic ratio
            'HNR': np.random.normal(20, 5, n_samples),  # Harmonic-to-noise ratio
            'RPDE': np.random.uniform(0.3, 0.7, n_samples),  # Recurrence period density entropy
            'DFA': np.random.uniform(0.5, 0.8, n_samples),  # Detrended fluctuation analysis
            'spread1': np.random.normal(-5, 2, n_samples),  # Fundamental frequency variation
            'spread2': np.random.normal(0.2, 0.1, n_samples),
            'D2': np.random.uniform(1.5, 3.5, n_samples),  # Correlation dimension
            'PPE': np.random.uniform(0.1, 0.4, n_samples)  # Pitch period entropy
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable (status: 1 = Parkinson's, 0 = Healthy)
        # Higher jitter, shimmer, and NHR typically indicate Parkinson's
        parkinson_prob = (
            (df['MDVP:Jitter(%)'] > 0.006) * 0.3 +
            (df['MDVP:Shimmer'] > 0.03) * 0.3 +
            (df['NHR'] > 0.025) * 0.3 +
            np.random.random(n_samples) * 0.1
        )
        df['status'] = (parkinson_prob > 0.5).astype(int)
        
        # Add name column (typically present in Parkinson's datasets)
        df['name'] = [f'patient_{i}' for i in range(n_samples)]
        
        return df

    def preprocess_data(self, target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the Parkinson's dataset for machine learning
        
        Args:
            target_column: Name of target column (auto-detects if None)
            
        Returns:
            Tuple of (X_processed, y_processed)
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data().")
        
        logger.info("Starting data preprocessing...")
        
        # Auto-detect target column if not specified
        if target_column is None:
            possible_targets = ['status', 'class', 'target', 'label', 'diagnosis']
            
            for col in possible_targets:
                if col in self.data.columns:
                    target_column = col
                    break
            
            if target_column is None:
                # Use last column as default
                target_column = self.data.columns[-1]
                logger.warning(f"Could not auto-detect target column. Using: {target_column}")
        
        logger.info(f"Using target column: {target_column}")
        
        # Remove non-feature columns (like name, id)
        feature_df = self.data.copy()
        
        # Common non-feature columns to remove
        non_feature_cols = ['name', 'id', 'patient_id', 'subject_id']
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
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y_processed = self.label_encoder.fit_transform(y)
            logger.info(f"Encoded target labels: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        else:
            y_processed = y.values
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
        logger.info(f"Target distribution: {np.bincount(y_processed.astype(int))}")
        
        return X_processed, y_processed

    def train_classical_models(self, n_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Train classical machine learning models with cross-validation
        
        Args:
            n_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary with model results
        """
        if self.X_processed is None or self.y_processed is None:
            raise ValueError("Data must be preprocessed first. Call preprocess_data().")
        
        logger.info("Training classical models...")
        
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
                model_path = self.output_dir / "models" / f"{name.replace(' ', '_').lower()}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
                logger.info(f"{name} - Accuracy: {results[name]['accuracy_mean']:.3f} (±{results[name]['accuracy_std']:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                
        self.classical_results = results
        return results

    def train_neural_network(self, epochs: int = 100, batch_size: int = 32) -> Dict[str, float]:
        """
        Train neural network classifier
        
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
        
        logger.info("Training neural network...")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(self.X_processed)
        y_tensor = torch.LongTensor(self.y_processed.astype(int))
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_size = X_tensor.shape[1]
        num_classes = len(np.unique(self.y_processed))
        model = ParkinsonsMLPClassifier(input_size, num_classes)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / val_total
            val_accuracies.append(val_acc)
            scheduler.step(-val_acc)  # For ReduceLROnPlateau
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.output_dir / "models" / "neural_network_best.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Evaluate on full validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, val_pred = torch.max(val_outputs, 1)
            
            # Calculate metrics
            val_accuracy = (val_pred == y_val).float().mean().item()
            
            # Convert to numpy for sklearn metrics
            val_pred_np = val_pred.numpy()
            y_val_np = y_val.numpy()
            
            val_f1 = f1_score(y_val_np, val_pred_np, average='macro')
        
        results = {
            'best_validation_accuracy': best_val_acc,
            'final_validation_accuracy': val_accuracy,
            'validation_f1_macro': val_f1,
            'epochs_trained': len(train_losses)
        }
        
        self.neural_results = results
        logger.info(f"Neural Network - Best Val Acc: {best_val_acc:.3f}, Final F1: {val_f1:.3f}")
        
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
            'dataset_info': {
                'shape': self.data.shape if self.data is not None else None,
                'features': len(self.feature_names) if self.feature_names else None,
                'feature_names': self.feature_names
            },
            'classical_results': self.classical_results,
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
            f.write("PARKINSON'S DISEASE CLASSIFICATION TRAINING REPORT\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            if report['dataset_info']['shape']:
                f.write(f"Dataset Information:\n")
                f.write(f"- Shape: {report['dataset_info']['shape']}\n")
                f.write(f"- Features: {report['dataset_info']['features']}\n\n")
            
            if report['classical_results']:
                f.write("Classical Model Results:\n")
                f.write("-" * 25 + "\n")
                for model, metrics in report['classical_results'].items():
                    f.write(f"{model}:\n")
                    f.write(f"  Accuracy: {metrics['accuracy_mean']:.3f} (±{metrics['accuracy_std']:.3f})\n")
                    f.write(f"  Balanced Accuracy: {metrics['balanced_accuracy_mean']:.3f} (±{metrics['balanced_accuracy_std']:.3f})\n")
                    f.write(f"  F1 Score: {metrics['f1_macro_mean']:.3f} (±{metrics['f1_macro_std']:.3f})\n")
                    f.write(f"  ROC AUC: {metrics['roc_auc_mean']:.3f} (±{metrics['roc_auc_std']:.3f})\n\n")
            
            if report['neural_results']:
                f.write("Neural Network Results:\n")
                f.write("-" * 22 + "\n")
                f.write(f"Best Validation Accuracy: {report['neural_results']['best_validation_accuracy']:.3f}\n")
                f.write(f"Final Validation F1: {report['neural_results']['validation_f1_macro']:.3f}\n")
                f.write(f"Epochs Trained: {report['neural_results']['epochs_trained']}\n\n")
            
            f.write(f"Models and outputs saved in: {report['output_directory']}\n")

    def run_full_pipeline(self, data_path: str = None, epochs: int = 100, 
                         n_folds: int = 5, dataset_choice: str = "vikasukani") -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            data_path: Path to dataset CSV (optional)
            epochs: Number of epochs for neural network
            n_folds: Number of folds for cross-validation
            dataset_choice: Which Kaggle dataset to use
            
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting Parkinson's Disease Classification Training Pipeline...")
        
        try:
            # Load data
            self.load_data(data_path, dataset_choice)
            
            # Preprocess data  
            self.preprocess_data()
            
            # Train classical models
            classical_results = self.train_classical_models(n_folds)
            
            # Train neural network
            neural_results = self.train_neural_network(epochs)
            
            # Save preprocessors
            self.save_preprocessor()
            
            # Generate report
            report = self.generate_report()
            
            logger.info("Parkinson's training pipeline completed successfully!")
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
    
    parser = argparse.ArgumentParser(description="Parkinson's Disease Classification Training Pipeline")
    parser.add_argument('--data-path', type=str, default=None, 
                       help='Path to dataset CSV file (optional, will download from Kaggle if not provided)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of epochs for neural network training')
    parser.add_argument('--folds', type=int, default=5, 
                       help='Number of folds for cross-validation')
    parser.add_argument('--dataset-choice', type=str, default='vikasukani',
                       choices=['vikasukani', 'uci-parkinsons', 'leilahasan'],
                       help='Which Kaggle dataset to use')
    parser.add_argument('--output-dir', type=str, default='parkinsons_outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    pipeline = ParkinsonsTrainingPipeline(output_dir=args.output_dir)
    
    try:
        results = pipeline.run_full_pipeline(
            data_path=args.data_path,
            epochs=args.epochs,
            n_folds=args.folds,
            dataset_choice=args.dataset_choice
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        # Print quick summary
        if results.get('classical_results'):
            print(f"\nBest Classical Model Performance:")
            best_model = max(results['classical_results'].items(), 
                           key=lambda x: x[1]['accuracy_mean'])
            print(f"  {best_model[0]}: {best_model[1]['accuracy_mean']:.3f} accuracy")
        
        if results.get('neural_results'):
            print(f"Neural Network: {results['neural_results']['best_validation_accuracy']:.3f} best validation accuracy")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
