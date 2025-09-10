#!/usr/bin/env python3
"""
Training module for duetmind_adaptive
Integrates machine learning training with the existing neural network framework
"""

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any, List, Tuple, Optional
import pickle
import os
from pathlib import Path

# Import existing components
from neuralnet import UnifiedAdaptiveAgent, AliveLoopNode, ResourceRoom, NetworkMetrics, MazeMaster
from files.dataset.create_test_data import create_test_alzheimer_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DuetMindTraining")

class AlzheimerTrainer:
    """Training system for Alzheimer disease prediction integrated with duetmind_adaptive agents"""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'diagnosis'
        
    def load_data(self) -> pd.DataFrame:
        """Load Alzheimer dataset from file or create test data"""
        if self.data_path and os.path.exists(self.data_path):
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
        else:
            logger.info("Creating test data for training")
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
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """Train the Alzheimer prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            min_samples_split=2
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
        
        return self.model, metrics
    
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
            raise ValueError("Model not trained or loaded")
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        
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
    model, results = trainer.train_model(X, y)
    
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


if __name__ == "__main__":
    run_training_simulation()