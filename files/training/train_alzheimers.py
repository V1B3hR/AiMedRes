#!/usr/bin/env python3
"""
Alzheimer's Disease Classification Training Pipeline

This script implements a comprehensive machine learning pipeline for Alzheimer's disease classification
using the Kaggle dataset (rabieelkharoua/alzheimers-disease-dataset).

Features:
- Downloads dataset using kagglehub
- Comprehensive data preprocessing
- Training of 4 classical models with 5-fold cross-validation
- Tabular neural network training (MLP)
- Detailed metrics reporting
- Model and preprocessing pipeline persistence
"""

# Essential imports only - heavy libraries imported on demand
import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def _lazy_import_ml():
    """Lazy import of ML libraries to improve startup time"""
    global np, pd, pickle, json
    global train_test_split, cross_val_score, StratifiedKFold
    global accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    global classification_report, confusion_matrix
    global LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, SVC
    global LabelEncoder, StandardScaler, SimpleImputer
    global MLPClassifier
    global kagglehub, requests
    
    import numpy as np
    import pandas as pd
    import pickle
    import json
    
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier
    
    try:
        import kagglehub
    except ImportError:
        kagglehub = None
        
    try:
        import requests
    except ImportError:
        requests = None

# ML Libraries
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Advanced ML Libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# PyTorch for neural networks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Kaggle dataset loading
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AlzheimerMLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron for Alzheimer's disease classification
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
        layers = []
        current_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class AlzheimerTrainingPipeline:
    """
    Complete training pipeline for Alzheimer's disease classification
    """
    
    def __init__(self, output_dir: str = "outputs"):
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
        
        # Initialize components
        self.data = None
        self.preprocessor = None
        self.label_encoder = None
        self.X_processed = None
        self.y_encoded = None
        self.feature_names = None
        
        # Results storage
        self.classical_results = {}
        self.neural_results = {}
        
    def download_dataset(self) -> str:
        """
        Download the Alzheimer's dataset from Kaggle using kagglehub
        
        Returns:
            Path to the downloaded dataset
        """
        if not KAGGLEHUB_AVAILABLE:
            raise ImportError("kagglehub is required but not installed. Run: pip install kagglehub")
        
        logger.info("Downloading Alzheimer's dataset from Kaggle...")
        
        try:
            # Download the dataset
            path = kagglehub.dataset_download("rabieelkharoua/alzheimers-disease-dataset")
            logger.info(f"Dataset downloaded to: {path}")
            
            # Find the CSV file in the downloaded directory
            dataset_path = Path(path)
            csv_files = list(dataset_path.glob("*.csv"))
            
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the downloaded dataset")
            
            # Use the first CSV file (typically alzheimer_dataset.csv)
            dataset_file = csv_files[0]
            logger.info(f"Using dataset file: {dataset_file}")
            
            return str(dataset_file)
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load the Alzheimer's dataset
        
        Args:
            data_path: Optional path to dataset. If None, downloads from Kaggle
            
        Returns:
            Loaded dataset as pandas DataFrame
        """
        if data_path is None:
            data_path = self.download_dataset()
        
        logger.info(f"Loading dataset from: {data_path}")
        
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess_data(self, target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data: drop ID columns, encode categorical, impute missing, standardize
        
        Args:
            target_column: Name of the target column. If None, auto-detects from common names.
            
        Returns:
            Tuple of (X_processed, y_encoded)
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data().")
        
        logger.info("Starting data preprocessing...")
        
        # Make a copy to avoid modifying original data
        df = self.data.copy()
        
        # Auto-detect target column if not provided
        if target_column is None:
            possible_targets = ['Class', 'Diagnosis', 'Target', 'Label', 'class', 'diagnosis', 'target', 'label']
            for col in possible_targets:
                if col in df.columns:
                    target_column = col
                    logger.info(f"Auto-detected target column: '{target_column}'")
                    break
            
            if target_column is None:
                available_cols = list(df.columns)
                logger.error(f"Could not auto-detect target column. Available columns: {available_cols}")
                logger.error("Please specify the target column explicitly using the target_column parameter.")
                raise ValueError("Target column not found. Common target column names not detected in dataset")
        
        # Check if target column exists
        if target_column not in df.columns:
            available_cols = list(df.columns)
            logger.error(f"Target column '{target_column}' not found. Available columns: {available_cols}")
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Drop ID columns (typically columns with 'id' or 'ID' in the name)
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        if id_columns:
            logger.info(f"Dropping ID columns: {id_columns}")
            df = df.drop(columns=id_columns)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Target classes: {y.unique()}")
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.y_encoded = y_encoded
        
        logger.info(f"Target encoded classes: {np.unique(y_encoded)}")
        logger.info(f"Class mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        # Identify numeric and categorical columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(include=[object, 'category']).columns.tolist()
        
        logger.info(f"Numeric columns ({len(numeric_columns)}): {numeric_columns}")
        logger.info(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")
        
        # Create preprocessing pipeline
        preprocessors = []
        
        # Numeric preprocessing: impute then standardize
        if numeric_columns:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            preprocessors.append(('numeric', numeric_pipeline, numeric_columns))
        
        # Categorical preprocessing: impute then one-hot encode
        if categorical_columns:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            preprocessors.append(('categorical', categorical_pipeline, categorical_columns))
        
        # Create the column transformer
        self.preprocessor = ColumnTransformer(
            transformers=preprocessors,
            remainder='drop'
        )
        
        # Fit and transform the features
        logger.info("Fitting preprocessing pipeline...")
        X_processed = self.preprocessor.fit_transform(X)
        self.X_processed = X_processed
        
        # Get feature names after preprocessing
        feature_names = []
        
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'numeric':
                feature_names.extend(columns)
            elif name == 'categorical':
                # Get feature names from OneHotEncoder
                encoder = transformer.named_steps['encoder']
                encoded_features = encoder.get_feature_names_out(columns)
                feature_names.extend(encoded_features)
        
        self.feature_names = feature_names
        
        logger.info(f"Preprocessed data shape: {X_processed.shape}")
        logger.info(f"Number of features after preprocessing: {len(feature_names)}")
        
        # Check for any remaining missing values
        if np.isnan(X_processed).any():
            logger.warning("Warning: Missing values detected after preprocessing!")
        
        return X_processed, y_encoded
    
    def train_classical_models(self, n_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Train classical ML models with cross-validation
        
        Args:
            n_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary containing results for each model
        """
        if self.X_processed is None or self.y_encoded is None:
            raise ValueError("Data must be preprocessed first. Call preprocess_data().")
        
        logger.info(f"Training classical models with {n_folds}-fold cross-validation...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            logger.warning("XGBoost not available, skipping...")
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
        else:
            logger.warning("LightGBM not available, skipping...")
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'balanced_accuracy': 'balanced_accuracy',
            'macro_f1': 'f1_macro',
            'roc_auc': 'roc_auc_ovr' if len(np.unique(self.y_encoded)) > 2 else 'roc_auc'
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Perform cross-validation
                cv_results = cross_validate(
                    model, self.X_processed, self.y_encoded,
                    cv=cv, scoring=scoring, return_train_score=False
                )
                
                # Calculate mean and std for each metric
                model_results = {}
                for metric_name, metric_key in scoring.items():
                    scores = cv_results[f'test_{metric_name}']
                    model_results[metric_name] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'scores': scores.tolist()
                    }
                
                results[model_name] = model_results
                
                # Log results
                logger.info(f"{model_name} results:")
                for metric_name, metric_results in model_results.items():
                    mean_score = metric_results['mean']
                    std_score = metric_results['std']
                    logger.info(f"  {metric_name}: {mean_score:.4f} ± {std_score:.4f}")
                
                # Train final model on full dataset and save
                logger.info(f"Training final {model_name} model on full dataset...")
                model.fit(self.X_processed, self.y_encoded)
                
                # Save model
                model_path = self.output_dir / "models" / f"{model_name.lower().replace(' ', '_')}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {model_name} model to {model_path}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.classical_results = results
        return results
    
    def train_neural_network(self, epochs: int = 30, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train a tabular neural network (MLP)
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training results
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available, skipping neural network training")
            return {'error': 'PyTorch not available'}
        
        if self.X_processed is None or self.y_encoded is None:
            raise ValueError("Data must be preprocessed first. Call preprocess_data().")
        
        logger.info(f"Training neural network for {epochs} epochs...")
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(self.X_processed)
        y_tensor = torch.LongTensor(self.y_encoded)
        
        # Get data dimensions
        input_dim = X_tensor.shape[1]
        num_classes = len(np.unique(self.y_encoded))
        
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Number of classes: {num_classes}")
        
        # Create model
        model = AlzheimerMLPClassifier(input_dim, num_classes)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        model.train()
        training_losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Calculate epoch loss
            epoch_loss = np.mean(epoch_losses)
            training_losses.append(epoch_loss)
            
            # Update learning rate
            scheduler.step(epoch_loss)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        # Evaluate on full dataset
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_encoded, predictions)
        balanced_acc = balanced_accuracy_score(self.y_encoded, predictions)
        macro_f1 = f1_score(self.y_encoded, predictions, average='macro')
        
        # ROC AUC (handle multiclass)
        try:
            if num_classes == 2:
                proba = torch.softmax(outputs, dim=1)[:, 1].numpy()
                roc_auc = roc_auc_score(self.y_encoded, proba)
            else:
                proba = torch.softmax(outputs, dim=1).numpy()
                roc_auc = roc_auc_score(self.y_encoded, proba, multi_class='ovr')
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
            roc_auc = 0.0
        
        results = {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'macro_f1': float(macro_f1),
            'roc_auc': float(roc_auc),
            'final_loss': float(training_losses[-1]),
            'training_losses': training_losses,
            'epochs': epochs
        }
        
        logger.info("Neural network training completed:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
        logger.info(f"  Macro F1: {macro_f1:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        # Save model
        model_path = self.output_dir / "models" / "neural_network.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved neural network model to {model_path}")
        
        # Also save the model architecture info
        model_info = {
            'input_dim': input_dim,
            'num_classes': num_classes,
            'hidden_dims': [256, 128, 64, 32]
        }
        
        info_path = self.output_dir / "models" / "neural_network_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        self.neural_results = results
        return results
    
    def save_preprocessing_pipeline(self):
        """
        Save the preprocessing pipeline and label encoder for inference
        """
        if self.preprocessor is None:
            logger.warning("No preprocessor to save")
            return
        
        # Save preprocessor
        preprocessor_path = self.output_dir / "preprocessors" / "preprocessor.pkl"
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        logger.info(f"Saved preprocessor to {preprocessor_path}")
        
        # Save label encoder
        if self.label_encoder is not None:
            encoder_path = self.output_dir / "preprocessors" / "label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            logger.info(f"Saved label encoder to {encoder_path}")
        
        # Save feature names
        if self.feature_names is not None:
            features_path = self.output_dir / "preprocessors" / "feature_names.json"
            with open(features_path, 'w') as f:
                json.dump(self.feature_names, f, indent=2)
            logger.info(f"Saved feature names to {features_path}")
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive training report
        
        Returns:
            Dictionary containing all results and metadata
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'shape': self.data.shape if self.data is not None else None,
                'columns': list(self.data.columns) if self.data is not None else None,
                'target_classes': self.label_encoder.classes_.tolist() if self.label_encoder else None
            },
            'preprocessing_info': {
                'processed_shape': self.X_processed.shape if self.X_processed is not None else None,
                'num_features': len(self.feature_names) if self.feature_names else None
            },
            'classical_models': self.classical_results,
            'neural_network': self.neural_results,
            'dependencies': {
                'xgboost_available': XGBOOST_AVAILABLE,
                'lightgbm_available': LIGHTGBM_AVAILABLE,
                'torch_available': TORCH_AVAILABLE,
                'kagglehub_available': KAGGLEHUB_AVAILABLE
            }
        }
        
        # Save report
        report_path = self.output_dir / "metrics" / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved training report to {report_path}")
        
        # Generate summary text report
        self._generate_summary_report(report)
        
        return report
    
    def _generate_summary_report(self, report: Dict[str, Any]):
        """
        Generate a human-readable summary report
        
        Args:
            report: Full training report dictionary
        """
        summary_path = self.output_dir / "metrics" / "training_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Alzheimer's Disease Classification Training Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Training Date: {report['timestamp']}\n\n")
            
            # Dataset info
            if report['dataset_info']['shape']:
                f.write("Dataset Information:\n")
                f.write(f"  Shape: {report['dataset_info']['shape']}\n")
                f.write(f"  Target classes: {report['dataset_info']['target_classes']}\n")
                f.write(f"  Features after preprocessing: {report['preprocessing_info']['num_features']}\n\n")
            
            # Classical models results
            f.write("Classical Models Performance (5-fold CV):\n")
            f.write("-" * 40 + "\n")
            
            for model_name, results in report['classical_models'].items():
                if 'error' in results:
                    f.write(f"{model_name}: ERROR - {results['error']}\n")
                else:
                    f.write(f"{model_name}:\n")
                    for metric, values in results.items():
                        mean_val = values['mean']
                        std_val = values['std']
                        f.write(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}\n")
                    f.write("\n")
            
            # Neural network results
            if report['neural_network'] and 'error' not in report['neural_network']:
                f.write("Neural Network Performance:\n")
                f.write("-" * 30 + "\n")
                nn_results = report['neural_network']
                f.write(f"Accuracy: {nn_results['accuracy']:.4f}\n")
                f.write(f"Balanced Accuracy: {nn_results['balanced_accuracy']:.4f}\n")
                f.write(f"Macro F1: {nn_results['macro_f1']:.4f}\n")
                f.write(f"ROC AUC: {nn_results['roc_auc']:.4f}\n")
                f.write(f"Training Epochs: {nn_results['epochs']}\n")
                f.write(f"Final Loss: {nn_results['final_loss']:.4f}\n\n")
            
            # Output files
            f.write("Output Files:\n")
            f.write("-" * 15 + "\n")
            f.write("Models: outputs/models/\n")
            f.write("Preprocessors: outputs/preprocessors/\n")
            f.write("Metrics: outputs/metrics/\n")
        
        logger.info(f"Saved summary report to {summary_path}")
    
    def run_full_pipeline(self, data_path: str = None, target_column: str = None, epochs: int = 100, n_folds: int = 5) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            data_path: Optional path to dataset
            target_column: Name of target column (auto-detects if None)
            epochs: Number of epochs for neural network
            n_folds: Number of folds for cross-validation
            
        Returns:
            Complete training report
        """
        logger.info("Starting Alzheimer's Disease Classification Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Load data
            self.load_data(data_path)
            
            # Preprocess data
            self.preprocess_data(target_column)
            
            # Train classical models
            self.train_classical_models(n_folds)
            
            # Train neural network
            self.train_neural_network(epochs)
            
            # Save preprocessing pipeline
            self.save_preprocessing_pipeline()
            
            # Generate report
            report = self.generate_report()
            
            logger.info("=" * 60)
            logger.info("Training pipeline completed successfully!")
            logger.info(f"All outputs saved to: {self.output_dir}")
            
            return report
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """
    Main entry point for the training script
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Alzheimer's Disease Classification Training Pipeline"
    )
    parser.add_argument(
        '--data-path', 
        type=str, 
        default=None,
        help='Path to dataset CSV file (if None, downloads from Kaggle)'
    )
    parser.add_argument(
        '--target-column', 
        type=str, 
        default=None,
        help='Name of target column (auto-detects if None)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=30,
        help='Number of epochs for neural network training (default: 30 for small datasets)'
    )
    parser.add_argument(
        '--folds', 
        type=int, 
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = AlzheimerTrainingPipeline(output_dir=args.output_dir)
    
    try:
        report = pipeline.run_full_pipeline(
            data_path=args.data_path,
            target_column=args.target_column,
            epochs=args.epochs,
            n_folds=args.folds
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {args.output_dir}")
        print("\nKey Metrics Summary:")
        
        # Print classical models summary
        if pipeline.classical_results:
            print("\nClassical Models (Cross-Validation):")
            for model_name, results in pipeline.classical_results.items():
                if 'error' not in results and 'accuracy' in results:
                    acc = results['accuracy']['mean']
                    f1 = results['macro_f1']['mean']
                    print(f"  {model_name}: Accuracy={acc:.4f}, F1={f1:.4f}")
        
        # Print neural network summary
        if pipeline.neural_results and 'error' not in pipeline.neural_results:
            nn = pipeline.neural_results
            print(f"\nNeural Network: Accuracy={nn['accuracy']:.4f}, F1={nn['macro_f1']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)