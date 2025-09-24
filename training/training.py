#!/usr/bin/env python3
"""
Training module for duetmind_adaptive
Integrates machine learning training with the existing neural network framework
"""

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, List, Tuple, Optional
import pickle
import os
from pathlib import Path
import random

# Import existing components
from neuralnet import UnifiedAdaptiveAgent, AliveLoopNode, ResourceRoom, NetworkMetrics, MazeMaster
# from files.dataset.create_test_data import create_test_alzheimer_data  # Removed redundant files
from scripts.data_loaders import DataLoader, CSVDataLoader, MockDataLoader, create_data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DuetMindTraining")


def create_test_alzheimer_data(n_samples: int = 100) -> pd.DataFrame:
    """Create synthetic test data for Alzheimer's classification"""
    np.random.seed(42)
    
    # Create synthetic features similar to real Alzheimer's data
    age = np.random.normal(75, 10, n_samples)
    cognitive_score = np.random.normal(25, 5, n_samples)  # MMSE-like score
    brain_volume = np.random.normal(1200, 150, n_samples)
    glucose_metabolism = np.random.normal(100, 20, n_samples)
    
    # Create correlation between features and label
    # Lower cognitive scores and brain volumes increase risk
    risk_score = (30 - cognitive_score) / 10 + (1300 - brain_volume) / 500 + age / 100
    labels = (risk_score + np.random.normal(0, 0.5, n_samples)) > 1.5
    
    df = pd.DataFrame({
        'age': age,
        'cognitive_score': cognitive_score, 
        'brain_volume': brain_volume,
        'glucose_metabolism': glucose_metabolism,
        'diagnosis': labels.astype(int)  # 0 = Normal, 1 = Alzheimer's
    })
    
    return df


class TrainingConfig:
    """Configuration class for deterministic training"""
    
    def __init__(self, 
                 random_seed: int = 42,
                 test_size: float = 0.3,
                 n_estimators: int = 100,
                 max_depth: int = 5,
                 min_samples_split: int = 2):
        self.random_seed = random_seed
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def set_random_seeds(self):
        """Set all random seeds for deterministic behavior"""
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)


class AlzheimerTrainer:
    """Training system for Alzheimer disease prediction integrated with duetmind_adaptive agents"""
    
    def __init__(self, 
                 data_loader: Optional[DataLoader] = None,
                 config: Optional[TrainingConfig] = None):
        self.data_loader = data_loader
        self.config = config or TrainingConfig()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'diagnosis'
        
        # Set random seeds for deterministic behavior
        self.config.set_random_seeds()
        
    def load_data(self) -> pd.DataFrame:
        """Load Alzheimer dataset using data loader or create test data"""
        if self.data_loader:
            logger.info("Loading data using provided data loader")
            df = self.data_loader.load_data()
        else:
            logger.info("No data loader provided, creating test data for training")
            df = create_test_alzheimer_data()
            
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data for training"""
        # Encode categorical variables
        df_processed = df.copy()
        
        # Handle gender encoding
        if 'gender' in df_processed.columns:
            df_processed['gender'] = df_processed['gender'].map({'M': 1, 'F': 0})
        
        # Handle APOE genotype encoding
        if 'apoe_genotype' in df_processed.columns:
            apoe_mapping = {
                'E2/E2': 0, 'E2/E3': 1, 'E2/E4': 2,
                'E3/E3': 3, 'E3/E4': 4, 'E4/E4': 5
            }
            df_processed['apoe_genotype'] = df_processed['apoe_genotype'].map(apoe_mapping)
        
        # Separate features and target
        X = df_processed.drop(columns=[self.target_column])
        y = df_processed[self.target_column]
        
        # Store feature column names
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"Preprocessed {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
        logger.info(f"Target classes: {self.label_encoder.classes_}")
        
        return X_scaled, y_encoded
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the Alzheimer prediction model"""
        # Ensure deterministic behavior
        self.config.set_random_seeds()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_seed, 
            stratify=y
        )
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_seed,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split
        )
        
        logger.info("Training Random Forest classifier...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        logger.info(f"Training accuracy: {train_accuracy:.3f}")
        logger.info(f"Test accuracy: {test_accuracy:.3f}")
        
        # Get feature importance
        feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
        
        metrics = {
            'accuracy': test_accuracy,  # Use test accuracy as primary metric
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': feature_importance,
            'classification_report': classification_report(
                y_test, test_predictions, 
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
        }
        
        return metrics
    
    def save_model(self, model_path: str = "alzheimer_model.pkl"):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_scaler': self.feature_scaler,
            'feature_columns': self.feature_columns
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = "alzheimer_model.pkl"):
        """Load a trained model and preprocessors"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_scaler = model_data['feature_scaler']
        self.feature_columns = model_data['feature_columns']
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, features: Dict[str, Any]) -> str:
        """Make prediction for a single case"""
        if self.model is None:
            raise ValueError("No trained model available. Please train or load a model first.")
        
        # Check for required features
        missing_features = set(self.feature_columns) - set(features.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Select only the features used during training
        feature_df = feature_df[self.feature_columns]
        
        # Apply same preprocessing
        if 'gender' in feature_df.columns:
            feature_df['gender'] = feature_df['gender'].map({'M': 1, 'F': 0})
        
        if 'apoe_genotype' in feature_df.columns:
            apoe_mapping = {
                'E2/E2': 0, 'E2/E3': 1, 'E2/E4': 2,
                'E3/E3': 3, 'E3/E4': 4, 'E4/E4': 5
            }
            feature_df['apoe_genotype'] = feature_df['apoe_genotype'].map(apoe_mapping)
        
        # Scale features
        X_scaled = self.feature_scaler.transform(feature_df)
        
        # Make prediction
        prediction_encoded = self.model.predict(X_scaled)[0]
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        return prediction
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters for secure transmission"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return {
            'model_state': {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'min_samples_split': self.model.min_samples_split,
                'random_state': self.model.random_state
            },
            'feature_columns': self.feature_columns,
            'model_type': 'RandomForestClassifier',
            'preprocessing': {
                'scaler_params': {
                    'mean_': self.feature_scaler.mean_.tolist() if hasattr(self.feature_scaler, 'mean_') else None,
                    'scale_': self.feature_scaler.scale_.tolist() if hasattr(self.feature_scaler, 'scale_') else None
                },
                'label_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None
            }
        }


class TrainingIntegratedAgent(UnifiedAdaptiveAgent):
    """Enhanced agent that can use trained models for reasoning"""
    
    def __init__(self, name: str, style: Dict[str, float], alive_node: AliveLoopNode, 
                 resource_room: ResourceRoom, trainer: AlzheimerTrainer):
        super().__init__(name, style, alive_node, resource_room)
        self.trainer = trainer
        
    def enhanced_reason_with_ml(self, task: str, patient_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced reasoning that incorporates ML predictions"""
        # Get base reasoning from parent
        base_result = self.reason(task)
        
        # If we have patient features and a trained model, add ML prediction
        if patient_features and self.trainer.model is not None:
            try:
                ml_prediction = self.trainer.predict(patient_features)
                base_result['ml_prediction'] = ml_prediction
                base_result['confidence'] = min(1.0, base_result.get('confidence', 0.5) + 0.2)
                self.log_event(f"ML prediction: {ml_prediction}")
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
                base_result['ml_prediction'] = "Unknown"
        
        return base_result


def run_training_simulation():
    """Run a complete training simulation with agents"""
    logger.info("=== DuetMind Adaptive Training Simulation ===")
    
    # Initialize trainer
    trainer = AlzheimerTrainer()
    
    # Load and train model
    df = trainer.load_data()
    X, y = trainer.preprocess_data(df)
    results = trainer.train_model(X, y)
    
    # Save model
    trainer.save_model()
    
    # Print training results
    logger.info("\n=== Training Results ===")
    logger.info(f"Training Accuracy: {results['train_accuracy']:.3f}")
    logger.info(f"Test Accuracy: {results['test_accuracy']:.3f}")
    logger.info("\nFeature Importance:")
    for feature, importance in sorted(results['feature_importance'].items(), 
                                    key=lambda x: x[1], reverse=True):
        logger.info(f"  {feature}: {importance:.3f}")
    
    # Create agents with ML capabilities
    resource_room = ResourceRoom()
    maze_master = MazeMaster()
    metrics = NetworkMetrics()
    
    agents = [
        TrainingIntegratedAgent(
            "MLAgentA", 
            {"logic": 0.8, "analytical": 0.9}, 
            AliveLoopNode((0,0), (0.5,0), 15.0, node_id=1), 
            resource_room, 
            trainer
        ),
        TrainingIntegratedAgent(
            "MLAgentB", 
            {"creativity": 0.7, "logic": 0.6}, 
            AliveLoopNode((2,0), (0,0.5), 12.0, node_id=2), 
            resource_room, 
            trainer
        )
    ]
    
    # Simulate reasoning with ML predictions
    logger.info("\n=== Agent Reasoning with ML ===")
    test_patient = {
        'age': 72,
        'gender': 'F',
        'education_level': 12,
        'mmse_score': 24,
        'cdr_score': 0.5,
        'apoe_genotype': 'E3/E4'
    }
    
    for agent in agents:
        result = agent.enhanced_reason_with_ml(
            "Assess patient risk for cognitive decline", 
            test_patient
        )
        logger.info(f"{agent.name} reasoning result: {result}")
    
    logger.info("\n=== Training Simulation Complete ===")
    return results, agents


class ExtendedTrainingRunner:
    """Extended training with enhanced ML capabilities including cross-validation and hyperparameter tuning"""
    
    def __init__(self, trainer: AlzheimerTrainer):
        self.trainer = trainer
        
    def run_extended_training(self) -> Dict[str, Any]:
        """Run extended training with cross-validation and hyperparameter optimization"""
        logger.info("🔬 Starting Extended Training...")
        
        # Load and preprocess data
        df = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define hyperparameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Perform grid search with cross-validation
        logger.info("🔍 Performing hyperparameter optimization...")
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Perform cross-validation on best model
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Final evaluation
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred)
        
        # Save enhanced model
        self.trainer.model = best_model
        self.trainer.save_model("extended_alzheimer_model.pkl")
        
        results = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'final_accuracy': final_accuracy,
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=self.trainer.label_encoder.classes_,
                                                         output_dict=True)
        }
        
        logger.info(f"✅ Extended training completed with accuracy: {final_accuracy:.3f}")
        return results


class AdvancedTrainingRunner:
    """Advanced training with multiple models and ensemble methods"""
    
    def __init__(self, trainer: AlzheimerTrainer):
        self.trainer = trainer
        
    def run_advanced_training(self) -> Dict[str, Any]:
        """Run advanced training with multiple models and ensemble methods"""
        logger.info("🚀 Starting Advanced Training...")
        
        # Load and preprocess data
        df = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # Train and evaluate each model
        model_results = {}
        trained_models = {}
        
        for name, model in models.items():
            logger.info(f"📊 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            model_results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, 
                                                             target_names=self.trainer.label_encoder.classes_,
                                                             output_dict=True)
            }
            trained_models[name] = model
            
            logger.info(f"   {name}: Accuracy = {accuracy:.3f}, CV = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Create ensemble model
        logger.info("🤝 Creating ensemble model...")
        ensemble_models = [
            ('rf', trained_models['Random Forest']),
            ('gb', trained_models['Gradient Boosting']),
            ('lr', trained_models['Logistic Regression'])
        ]
        
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        ensemble_cv = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
        
        # Find best individual model
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
        best_accuracy = model_results[best_model_name]['accuracy']
        
        # Save best model and ensemble
        self.trainer.model = trained_models[best_model_name]
        self.trainer.save_model("advanced_best_model.pkl")
        
        with open("advanced_ensemble_model.pkl", 'wb') as f:
            pickle.dump(ensemble, f)
        
        results = {
            'model_results': model_results,
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_cv_mean': ensemble_cv.mean(),
            'ensemble_cv_std': ensemble_cv.std(),
            'trained_models': trained_models,
            'ensemble_model': ensemble
        }
        
        logger.info(f"✅ Advanced training completed. Best: {best_model_name} ({best_accuracy:.3f}), Ensemble: {ensemble_accuracy:.3f}")
        return results


if __name__ == "__main__":
    run_training_simulation()