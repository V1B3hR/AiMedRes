#!/usr/bin/env python3
"""
ALS (Amyotrophic Lateral Sclerosis) Classification Training Pipeline

This script implements a comprehensive machine learning pipeline for ALS classification.
This is a placeholder implementation that follows the same patterns as the Alzheimer's pipeline.

Features:
- Comprehensive data preprocessing for ALS-specific features
- Training of classical models with cross-validation
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
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ALSTrainingPipeline:
    """
    Complete training pipeline for ALS (Amyotrophic Lateral Sclerosis) classification
    """
    
    def __init__(self, output_dir: str = "outputs/als"):
        """
        Initialize the training pipeline
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
    def create_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create synthetic ALS dataset for demonstration
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Synthetic dataset DataFrame
        """
        logger.info(f"Creating synthetic ALS dataset with {n_samples} samples...")
        
        np.random.seed(42)
        
        # Generate synthetic features based on common ALS indicators
        data = {}
        
        # Demographics
        data['Age'] = np.random.normal(62, 10, n_samples).clip(30, 85)
        data['Gender'] = np.random.choice(['M', 'F'], n_samples, p=[0.65, 0.35])  # Male predominance
        
        # ALSFRS-R (ALS Functional Rating Scale-Revised) - 48 is normal
        data['ALSFRS_R_Total'] = np.random.normal(35, 8, n_samples).clip(0, 48)
        data['ALSFRS_R_Bulbar'] = np.random.normal(10, 2, n_samples).clip(0, 12)
        data['ALSFRS_R_Fine_Motor'] = np.random.normal(9, 3, n_samples).clip(0, 12)
        data['ALSFRS_R_Gross_Motor'] = np.random.normal(8, 3, n_samples).clip(0, 12)
        data['ALSFRS_R_Respiratory'] = np.random.normal(10, 3, n_samples).clip(0, 12)
        
        # Disease progression rate
        data['Disease_Duration_Months'] = np.random.exponential(12, n_samples).clip(1, 60)
        data['Progression_Rate'] = (48 - data['ALSFRS_R_Total']) / data['Disease_Duration_Months']
        
        # Clinical features
        data['Onset_Type'] = np.random.choice(['Limb', 'Bulbar', 'Respiratory'], n_samples, p=[0.7, 0.25, 0.05])
        data['Site_of_Onset'] = np.random.choice(['Upper_Limb', 'Lower_Limb', 'Bulbar', 'Axial'], n_samples, p=[0.4, 0.3, 0.25, 0.05])
        
        # Muscle strength (MRC scale 0-5)
        data['MRC_Upper_Right'] = np.random.uniform(2, 5, n_samples)
        data['MRC_Upper_Left'] = np.random.uniform(2, 5, n_samples)
        data['MRC_Lower_Right'] = np.random.uniform(2, 5, n_samples)
        data['MRC_Lower_Left'] = np.random.uniform(2, 5, n_samples)
        
        # Pulmonary function
        data['FVC_Percent'] = np.random.normal(75, 20, n_samples).clip(20, 120)
        data['FEV1_Percent'] = np.random.normal(70, 20, n_samples).clip(20, 120)
        
        # Neurophysiology
        data['EMG_Abnormal_Regions'] = np.random.poisson(2.5, n_samples).clip(0, 4)
        data['Fasciculation_Score'] = np.random.uniform(0, 4, n_samples)
        
        # Blood biomarkers (simulated)
        data['Creatinine'] = np.random.normal(1.1, 0.3, n_samples).clip(0.5, 3.0)
        data['CK_Level'] = np.random.lognormal(5, 0.5, n_samples).clip(50, 2000)  # Creatine kinase
        data['Neurofilament_Light'] = np.random.lognormal(3, 0.8, n_samples).clip(10, 500)
        
        # Cognitive assessment
        data['Edinburgh_Cognitive_Screen'] = np.random.normal(120, 15, n_samples).clip(60, 130)
        
        # Genetic factors
        data['Family_History'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 10% familial
        data['Gene_Mutation'] = np.random.choice(['None', 'SOD1', 'C9ORF72', 'FUS', 'TARDBP', 'Other'], 
                                                n_samples, p=[0.85, 0.04, 0.06, 0.02, 0.01, 0.02])
        
        # Lifestyle factors
        data['Smoking_History'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        data['Physical_Activity_Level'] = np.random.uniform(1, 5, n_samples)  # 1=sedentary, 5=very active
        data['Military_Service'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        
        # Survival and disease staging
        data['King_Stage'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.3, 0.25, 0.15])
        
        # Create target variable based on features (simplified logic)
        # Lower ALSFRS-R, faster progression, more EMG abnormalities -> ALS
        risk_score = (
            ((48 - data['ALSFRS_R_Total']) / 48) * 0.3 +
            (data['Progression_Rate'] / 2) * 0.25 +
            (data['EMG_Abnormal_Regions'] / 4) * 0.2 +
            ((120 - data['FVC_Percent']) / 100) * 0.15 +
            (data['Fasciculation_Score'] / 4) * 0.1
        )
        
        # Add some noise and create binary target
        risk_score += np.random.normal(0, 0.1, n_samples)
        data['Diagnosis'] = (risk_score > 0.4).astype(int)  # Lower threshold due to rare disease
        
        df = pd.DataFrame(data)
        
        # Adjust class balance to be more realistic (approximately 85% controls, 15% ALS)
        positive_indices = np.where(df['Diagnosis'] == 1)[0]
        negative_indices = np.where(df['Diagnosis'] == 0)[0]
        
        target_positive = int(n_samples * 0.15)
        if len(positive_indices) > target_positive:
            # Randomly select subset of positive cases
            selected_positive = np.random.choice(positive_indices, target_positive, replace=False)
            keep_indices = np.concatenate([negative_indices, selected_positive])
            df = df.iloc[keep_indices].reset_index(drop=True)
        
        logger.info(f"Created dataset with {len(df)} samples")
        logger.info(f"Class distribution: {df['Diagnosis'].value_counts().to_dict()}")
        
        return df
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load ALS dataset
        
        Args:
            data_path: Path to dataset file
            
        Returns:
            Loaded DataFrame
        """
        if data_path and os.path.exists(data_path):
            logger.info(f"Loading data from {data_path}")
            self.data = pd.read_csv(data_path)
        else:
            logger.info("No dataset provided, creating synthetic data...")
            self.data = self.create_synthetic_data()
            
            # Save synthetic data for reproducibility
            synthetic_path = self.output_dir / "synthetic_als_data.csv"
            self.data.to_csv(synthetic_path, index=False)
            logger.info(f"Synthetic data saved to {synthetic_path}")
        
        logger.info(f"Dataset shape: {self.data.shape}")
        logger.info(f"Columns: {list(self.data.columns)}")
        
        return self.data
    
    def preprocess_data(self, target_column: str = 'Diagnosis') -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the dataset
        
        Args:
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_processed, y_encoded)
        """
        logger.info("Starting data preprocessing...")
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Handle categorical variables
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()
        
        logger.info(f"Categorical features: {categorical_features}")
        logger.info(f"Numerical features: {numerical_features}")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ]
        )
        
        # Fit and transform the data
        self.X_processed = preprocessor.fit_transform(X)
        
        # Get feature names
        if categorical_features:
            # Get categorical feature names from the encoder
            cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
            self.feature_names = numerical_features + cat_feature_names
        else:
            self.feature_names = numerical_features
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(y)
        
        self.preprocessor = preprocessor
        
        logger.info(f"Processed data shape: {self.X_processed.shape}")
        logger.info(f"Target distribution: {np.bincount(self.y_encoded)}")
        
        return self.X_processed, self.y_encoded
    
    def train_classical_models(self, n_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Train classical machine learning models
        
        Args:
            n_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary containing results for each model
        """
        logger.info(f"Training classical models with {n_folds}-fold cross-validation...")
        
        if self.X_processed is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        # Define models (with class_weight='balanced' due to imbalanced nature of ALS)
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        }
        
        # Add optional models if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scoring = ['accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Perform cross-validation
                cv_results = cross_validate(
                    model, self.X_processed, self.y_encoded,
                    cv=cv, scoring=scoring, return_train_score=True
                )
                
                # Calculate mean and std for each metric
                results[name] = {}
                for metric in scoring:
                    test_scores = cv_results[f'test_{metric}']
                    results[name][f'{metric}_mean'] = float(np.mean(test_scores))
                    results[name][f'{metric}_std'] = float(np.std(test_scores))
                
                # Fit model on full dataset and save
                model.fit(self.X_processed, self.y_encoded)
                model_path = self.output_dir / "models" / f"{name.lower()}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} model to {model_path}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.classical_results = results
        
        # Save results
        results_path = self.output_dir / "metrics" / "classical_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Classical model training completed")
        return results
    
    def train_neural_network(self, epochs: int = 30, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train a neural network for ALS classification
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training results dictionary
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping neural network training")
            return {'error': 'TensorFlow not available'}
        
        logger.info(f"Training neural network for {epochs} epochs...")
        
        if self.X_processed is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        try:
            # Calculate class weights for imbalanced data
            from sklearn.utils.class_weight import compute_class_weight
            
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.y_encoded),
                y=self.y_encoded
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            
            # Create model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(self.X_processed.shape[1],)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                self.X_processed, self.y_encoded,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                class_weight=class_weight_dict,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Save model
            model_path = self.output_dir / "models" / "neural_network_model.keras"
            model.save(model_path)
            logger.info(f"Saved neural network model to {model_path}")
            
            # Prepare results
            final_metrics = {
                'final_loss': float(history.history['loss'][-1]),
                'final_accuracy': float(history.history['accuracy'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'epochs_trained': len(history.history['loss'])
            }
            
            # Save training history
            history_path = self.output_dir / "metrics" / "neural_network_history.json"
            history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
            
            self.neural_results = final_metrics
            return final_metrics
            
        except Exception as e:
            logger.error(f"Neural network training failed: {e}")
            return {'error': str(e)}
    
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
        if self.feature_names:
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
            'condition': 'als',
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
                'tensorflow_available': TENSORFLOW_AVAILABLE
            }
        }
        
        # Save report
        report_path = self.output_dir / "als_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {report_path}")
        
        self._generate_summary_report(report)
        
        return report
    
    def _generate_summary_report(self, report: Dict[str, Any]):
        """Generate a human-readable summary of results"""
        
        print("\n" + "="*60)
        print("ALS TRAINING PIPELINE SUMMARY")
        print("="*60)
        
        # Dataset info
        if report['dataset_info']['shape']:
            print(f"Dataset: {report['dataset_info']['shape']} samples")
            print(f"Features: {report['preprocessing_info']['num_features']}")
        
        # Classical models results
        print("\nCLASSICAL MODELS PERFORMANCE:")
        print("-" * 50)
        if self.classical_results:
            for model_name, metrics in self.classical_results.items():
                if 'error' not in metrics:
                    accuracy = metrics.get('accuracy_mean', 0)
                    f1 = metrics.get('f1_mean', 0)
                    auc = metrics.get('roc_auc_mean', 0)
                    precision = metrics.get('precision_mean', 0)
                    recall = metrics.get('recall_mean', 0)
                    print(f"{model_name:15} | Acc: {accuracy:.3f} | F1: {f1:.3f} | AUC: {auc:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f}")
                else:
                    print(f"{model_name:15} | ERROR: {metrics['error']}")
        
        # Neural network results
        print("\nNEURAL NETWORK PERFORMANCE:")
        print("-" * 40)
        if self.neural_results:
            if 'error' not in self.neural_results:
                val_acc = self.neural_results.get('final_val_accuracy', 0)
                val_loss = self.neural_results.get('final_val_loss', 0)
                epochs = self.neural_results.get('epochs_trained', 0)
                print(f"Validation Accuracy: {val_acc:.3f}")
                print(f"Validation Loss: {val_loss:.3f}")
                print(f"Epochs Trained: {epochs}")
            else:
                print(f"ERROR: {self.neural_results['error']}")
        
        print("\n" + "="*60)
    
    def run_full_pipeline(self, data_path: str = None, target_column: str = 'Diagnosis', 
                         epochs: int = 30, n_folds: int = 5) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            data_path: Path to dataset
            target_column: Target column name
            epochs: Number of epochs for neural network
            n_folds: Number of folds for cross-validation
            
        Returns:
            Complete training report
        """
        try:
            logger.info("Starting ALS training pipeline...")
            
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
            
            logger.info("==" * 30)
            logger.info("Training pipeline completed successfully!")
            logger.info(f"All outputs saved to: {self.output_dir}")
            
            return report
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    pipeline = ALSTrainingPipeline()
    
    try:
        report = pipeline.run_full_pipeline(epochs=20, n_folds=3)
        print(f"\nTraining completed! Report saved to: {pipeline.output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)