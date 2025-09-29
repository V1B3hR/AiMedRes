#!/usr/bin/env python3
"""
Cardiovascular Disease Risk Classification Training Pipeline

This script implements a comprehensive machine learning pipeline for cardiovascular disease risk classification
using the specified Kaggle datasets:
- https://www.kaggle.com/datasets/colewelkins/cardiovascular-disease
- https://www.kaggle.com/datasets/thedevastator/exploring-risk-factors-for-cardiovascular-diseas
- https://www.kaggle.com/datasets/jocelyndumlao/cardiovascular-disease-dataset

Features:
- Downloads datasets using kagglehub
- Comprehensive data preprocessing  
- Training of classical models with 5-fold cross-validation
- Tabular neural network training (MLP) with 50 epochs
- Early stopping for neural network training
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cardiovascular_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CardiovascularMLPClassifier(nn.Module):
    """
    Multi-layer perceptron optimized for cardiovascular disease classification
    """
    
    def __init__(self, input_size: int, num_classes: int = 2):
        super(CardiovascularMLPClassifier, self).__init__()
        
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
            nn.Dropout(0.3),
            
            # Fourth hidden layer
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class CardiovascularTrainingPipeline:
    """
    Complete training pipeline for cardiovascular disease risk classification
    """
    
    def __init__(self, output_dir: str = "cardiovascular_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "preprocessors").mkdir(exist_ok=True)
        
        self.data = None
        self.target = None
        self.feature_names = None
        self.target_names = None
        self.preprocessor = None
        self.X_processed = None
        self.y_processed = None
        
        logger.info(f"Cardiovascular training pipeline initialized. Output directory: {self.output_dir}")
    
    def load_data(self, data_path: str = None, dataset_choice: str = "colewelkins") -> pd.DataFrame:
        """
        Load cardiovascular disease dataset from Kaggle or local file
        
        Args:
            data_path: Path to local CSV file (optional)
            dataset_choice: Which Kaggle dataset to use
                - "colewelkins": colewelkins/cardiovascular-disease
                - "thedevastator": thedevastator/exploring-risk-factors-for-cardiovascular-diseas
                - "jocelyndumlao": jocelyndumlao/cardiovascular-disease-dataset
        
        Returns:
            pandas DataFrame with loaded data
        """
        
        if data_path and os.path.exists(data_path):
            logger.info(f"Loading data from local file: {data_path}")
            try:
                # Try different separators as cardiovascular datasets may use semicolons
                try:
                    df = pd.read_csv(data_path, sep=',')
                    if len(df.columns) == 1:
                        df = pd.read_csv(data_path, sep=';')
                except:
                    df = pd.read_csv(data_path, sep=';')
                
                logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns from local file")
                return df
                
            except Exception as e:
                logger.error(f"Error loading local file: {e}")
                logger.info("Falling back to Kaggle dataset download...")
        
        if not KAGGLEHUB_AVAILABLE:
            logger.warning("kagglehub not available. Creating sample data for demonstration.")
            return self._create_sample_data()
        
        try:
            logger.info("Downloading cardiovascular disease dataset from Kaggle...")
            
            if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
                logger.warning("Kaggle credentials not found. Creating sample data for demonstration.")
                return self._create_sample_data()
            
            if dataset_choice == "colewelkins":
                logger.info("Downloading Cole Welkins cardiovascular disease dataset...")
                path = kagglehub.dataset_download("colewelkins/cardiovascular-disease")
            elif dataset_choice == "thedevastator":
                logger.info("Downloading TheDevastator cardiovascular risk factors dataset...")
                path = kagglehub.dataset_download("thedevastator/exploring-risk-factors-for-cardiovascular-diseas")
            elif dataset_choice == "jocelyndumlao":
                logger.info("Downloading Jocelyn Dumlao cardiovascular disease dataset...")
                path = kagglehub.dataset_download("jocelyndumlao/cardiovascular-disease-dataset")
            else:
                logger.warning(f"Unknown dataset choice: {dataset_choice}. Using colewelkins dataset.")
                path = kagglehub.dataset_download("colewelkins/cardiovascular-disease")
            
            # Find CSV file in downloaded path
            csv_files = list(Path(path).glob("*.csv"))
            if csv_files:
                csv_file = csv_files[0]
                logger.info(f"Loading data from: {csv_file}")
                # Try different separators as cardiovascular datasets may use semicolons
                try:
                    df = pd.read_csv(csv_file, sep=',')
                    if len(df.columns) == 1:
                        df = pd.read_csv(csv_file, sep=';')
                except:
                    df = pd.read_csv(csv_file, sep=';')
                
                logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
                return df
            else:
                logger.error("No CSV files found in downloaded dataset")
                return self._create_sample_data()
                
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            logger.info("Creating sample data for demonstration")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample cardiovascular dataset for demonstration"""
        logger.info("Creating sample cardiovascular disease dataset...")
        
        n_samples = 5000
        np.random.seed(42)
        
        # Cardiovascular-specific features
        data = {
            'age': np.random.randint(18, 90, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'height': np.random.normal(170, 15, n_samples),
            'weight': np.random.normal(70, 15, n_samples),
            'ap_hi': np.random.normal(120, 20, n_samples),  # systolic blood pressure
            'ap_lo': np.random.normal(80, 15, n_samples),   # diastolic blood pressure
            'cholesterol': np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.3, 0.2]),  # 1-normal, 2-above normal, 3-well above normal
            'gluc': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.25, 0.15]),  # glucose levels
            'smoke': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'alco': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'active': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),  # physical activity
            'heart_rate': np.random.normal(72, 12, n_samples),
            'chest_pain': np.random.choice(['No', 'Yes'], n_samples, p=[0.6, 0.4]),
            'shortness_of_breath': np.random.choice(['No', 'Yes'], n_samples, p=[0.7, 0.3]),
            'fatigue': np.random.choice(['No', 'Yes'], n_samples, p=[0.6, 0.4]),
        }
        
        # Create target variable with some correlation to features
        cardio_prob = np.zeros(n_samples)
        for i in range(n_samples):
            prob = 0.1  # base probability
            if data['age'][i] > 50: prob += 0.3
            if data['gender'][i] == 'Male': prob += 0.1
            if data['ap_hi'][i] > 140: prob += 0.4  # high systolic BP
            if data['ap_lo'][i] > 90: prob += 0.3   # high diastolic BP
            if data['cholesterol'][i] == 3: prob += 0.3  # high cholesterol
            if data['smoke'][i] == 1: prob += 0.2
            if data['alco'][i] == 1: prob += 0.1
            if data['active'][i] == 0: prob += 0.15  # sedentary lifestyle
            if data['chest_pain'][i] == 'Yes': prob += 0.3
            if data['shortness_of_breath'][i] == 'Yes': prob += 0.2
            if data['fatigue'][i] == 'Yes': prob += 0.1
            
            # BMI calculation and risk
            bmi = data['weight'][i] / ((data['height'][i]/100) ** 2)
            if bmi > 30: prob += 0.2  # obesity
            elif bmi > 25: prob += 0.1  # overweight
            
            cardio_prob[i] = min(prob, 0.95)  # cap at 95%
        
        data['cardio'] = np.random.binomial(1, cardio_prob, n_samples)
        
        df = pd.DataFrame(data)
        logger.info(f"Created sample dataset with {len(df)} rows, cardiovascular disease prevalence: {df['cardio'].mean():.2%}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess cardiovascular disease data
        """
        logger.info("Starting data preprocessing...")
        
        # Auto-detect target column if not specified
        if target_column is None:
            potential_targets = ['cardio', 'target', 'disease', 'outcome', 'result', 'diagnosis']
            for col in potential_targets:
                if col in df.columns:
                    target_column = col
                    break
            
            if target_column is None:
                # Use last column as target
                target_column = df.columns[-1]
                logger.info(f"No explicit target column found. Using last column: {target_column}")
        
        logger.info(f"Using target column: {target_column}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        self.feature_names = list(X.columns)
        
        # Handle target variable
        if y.dtype == 'object' or len(np.unique(y)) > 2:
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.target_names = list(le.classes_)
        else:
            self.target_names = ['No Disease', 'Disease']
        
        logger.info(f"Target distribution: {np.bincount(y)}")
        
        # Identify column types
        numeric_features = []
        categorical_features = []
        
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (like cholesterol levels 1,2,3)
                unique_vals = X[col].nunique()
                if unique_vals <= 10 and X[col].dtype == 'int64':
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        logger.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
        logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        
        # Combine preprocessors
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Fit and transform the data
        X_processed = self.preprocessor.fit_transform(X)
        
        # Store processed data
        self.X_processed = X_processed
        self.y_processed = y
        
        logger.info(f"Preprocessing completed. Shape: {X_processed.shape}")
        
        return X_processed, y
    
    def train_classical_models(self, n_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Train classical machine learning models with cross-validation
        """
        logger.info("Training classical machine learning models...")
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(random_state=42)
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            cv_results = cross_validate(
                model, self.X_processed, self.y_processed,
                cv=cv,
                scoring=['accuracy', 'balanced_accuracy', 'f1', 'roc_auc'],
                return_train_score=False
            )
            
            results[name] = {
                'accuracy': cv_results['test_accuracy'].mean(),
                'accuracy_std': cv_results['test_accuracy'].std(),
                'balanced_accuracy': cv_results['test_balanced_accuracy'].mean(),
                'f1': cv_results['test_f1'].mean(),
                'roc_auc': cv_results['test_roc_auc'].mean()
            }
            
            logger.info(f"{name} - Accuracy: {results[name]['accuracy']:.4f} (+/- {results[name]['accuracy_std']*2:.4f})")
            
            # Save the model
            model.fit(self.X_processed, self.y_processed)
            model_path = self.output_dir / "models" / f"{name.lower().replace(' ', '_')}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        return results
    
    def train_neural_network(self, epochs: int = 50, early_stopping_patience: int = 7) -> Dict[str, float]:
        """
        Train neural network classifier with early stopping
        """
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping neural network training.")
            return {}
        
        logger.info(f"Training neural network for up to {epochs} epochs (early stopping enabled)...")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(self.X_processed.toarray() if hasattr(self.X_processed, 'toarray') else self.X_processed)
        y_tensor = torch.LongTensor(self.y_processed)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=self.y_processed
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Initialize model
        input_size = X_tensor.shape[1]
        num_classes = len(np.unique(self.y_processed))
        model = CardiovascularMLPClassifier(input_size, num_classes)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # Training loop with early stopping
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                # Save best model
                torch.save(model.state_dict(), self.output_dir / "models" / "best_cardiovascular_nn.pth")
            else:
                epochs_no_improve += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            
            if epochs_no_improve >= early_stopping_patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs (best epoch: {best_epoch+1}, best val loss: {best_val_loss:.4f})')
                break
        
        # Final evaluation on validation set
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        # Save final model
        torch.save(model.state_dict(), self.output_dir / "models" / "final_cardiovascular_nn.pth")
        
        results = {
            'accuracy': accuracy,
            'f1': f1,
            'balanced_accuracy': balanced_acc,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch+1
        }
        
        logger.info(f"Neural Network - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Best Epoch: {best_epoch+1}")
        
        return results
    
    def save_preprocessor(self):
        """Save the data preprocessor"""
        preprocessor_path = self.output_dir / "preprocessors" / "cardiovascular_preprocessor.pkl"
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    def save_training_report(self, classical_results: Dict, nn_results: Dict):
        """Save comprehensive training report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'n_samples': len(self.y_processed),
                'n_features': self.X_processed.shape[1],
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'class_distribution': {
                    str(k): int(v) for k, v in zip(*np.unique(self.y_processed, return_counts=True))
                }
            },
            'classical_models': classical_results,
            'neural_network': nn_results
        }
        
        # Save JSON report
        json_path = self.output_dir / "metrics" / "cardiovascular_training_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save text summary
        text_path = self.output_dir / "metrics" / "cardiovascular_training_summary.txt"
        with open(text_path, 'w') as f:
            f.write("CARDIOVASCULAR DISEASE CLASSIFICATION TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training completed: {report['timestamp']}\n")
            f.write(f"Dataset: {report['dataset_info']['n_samples']} samples, {report['dataset_info']['n_features']} features\n\n")
            
            f.write("CLASSICAL MODELS PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for model_name, metrics in classical_results.items():
                f.write(f"{model_name}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"  ROC AUC: {metrics['roc_auc']:.4f}\n\n")
            
            if nn_results:
                f.write("NEURAL NETWORK PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {nn_results['accuracy']:.4f}\n")
                f.write(f"F1 Score: {nn_results['f1']:.4f}\n")
                f.write(f"Balanced Accuracy: {nn_results['balanced_accuracy']:.4f}\n")
                f.write(f"Best Epoch: {nn_results.get('best_epoch', 'N/A')}\n")
        
        logger.info(f"Training report saved to {json_path} and {text_path}")
    
    def run_full_pipeline(self, data_path: str = None, target_column: str = None, 
                         epochs: int = 50, n_folds: int = 5, dataset_choice: str = "colewelkins") -> Dict[str, Any]:
        """
        Run the complete cardiovascular disease classification pipeline
        """
        logger.info("Starting cardiovascular disease classification pipeline...")
        
        # Load and preprocess data
        df = self.load_data(data_path, dataset_choice)
        X, y = self.preprocess_data(df, target_column)
        
        # Train classical models
        classical_results = self.train_classical_models(n_folds)
        
        # Train neural network
        nn_results = self.train_neural_network(epochs)
        
        # Save preprocessor
        self.save_preprocessor()
        
        # Save comprehensive report
        self.save_training_report(classical_results, nn_results)
        
        logger.info("Full pipeline completed successfully!")
        
        return {
            'classical_models': classical_results,
            'neural_network': nn_results,
            'output_directory': str(self.output_dir)
        }

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cardiovascular Disease Risk Classification Training Pipeline")
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
        '--dataset-choice',
        type=str,
        default='colewelkins',
        choices=['colewelkins', 'thedevastator', 'jocelyndumlao'],
        help='Which dataset to use: colewelkins, thedevastator, or jocelyndumlao (default: colewelkins)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='cardiovascular_outputs',
        help='Output directory for results (default: cardiovascular_outputs)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='Number of epochs for neural network training (default: 50)'
    )
    parser.add_argument(
        '--folds', 
        type=int, 
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = CardiovascularTrainingPipeline(output_dir=args.output_dir)
    
    try:
        report = pipeline.run_full_pipeline(
            data_path=args.data_path,
            target_column=args.target_column,
            epochs=args.epochs,
            n_folds=args.folds,
            dataset_choice=args.dataset_choice
        )
        
        print("\n" + "=" * 60)
        print("CARDIOVASCULAR TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {args.output_dir}")
        print("\nKey Metrics Summary:")
        
        # Display classical model results
        if 'classical_models' in report:
            print("\nClassical Models:")
            for model_name, metrics in report['classical_models'].items():
                print(f"  {model_name}: Accuracy={metrics['accuracy']:.2%}, F1={metrics['f1']:.2%}")
        
        # Display neural network results
        if 'neural_network' in report and report['neural_network']:
            nn_metrics = report['neural_network']
            print(f"\nNeural Network: Accuracy={nn_metrics['accuracy']:.2%}, F1={nn_metrics['f1']:.2%}, Best Epoch={nn_metrics.get('best_epoch', 'N/A')}")
        
        print("\n✅ Training pipeline completed successfully!")
        print("📊 Models and metrics saved to output directory")
        print("🔬 Ready for cardiovascular disease risk prediction!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
