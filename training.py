#!/usr/bin/env python3
"""
Machine Learning Training System for duetmind_adaptive framework
Enables AI agents to learn from and make predictions on medical datasets,
specifically focusing on Alzheimer's disease assessment.
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

# Import existing framework components
from labyrinth_adaptive import UnifiedAdaptiveAgent, AliveLoopNode, ResourceRoom

logger = logging.getLogger(__name__)


class AlzheimerTrainer:
    """
    Complete machine learning trainer for Alzheimer's disease assessment.
    Provides data loading, preprocessing, training, and model persistence.
    """
    
    def __init__(self, data_path: str = "alzheimer_features_test.csv"):
        """Initialize the trainer with data path."""
        self.data_path = data_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'diagnosis'
        
    def load_data(self) -> pd.DataFrame:
        """
        Load Alzheimer dataset from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            # Try loading from current directory first
            if os.path.exists(self.data_path):
                df = pd.read_csv(self.data_path)
            else:
                # Try loading from files/dataset directory
                alt_path = os.path.join('files', 'dataset', self.data_path)
                if os.path.exists(alt_path):
                    df = pd.read_csv(alt_path)
                else:
                    # Create test data if file doesn't exist
                    logger.warning(f"Data file {self.data_path} not found. Creating test data.")
                    df = self._create_sample_data()
            
            logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Fallback to creating sample data
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample Alzheimer dataset for demonstration."""
        np.random.seed(42)  # For reproducible results
        
        n_samples = 100
        data = {
            'age': np.random.normal(70, 10, n_samples).astype(int).clip(50, 90),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'education_level': np.random.normal(14, 4, n_samples).astype(int).clip(8, 22),
            'mmse_score': np.random.normal(24, 5, n_samples).clip(0, 30),
            'cdr_score': np.random.choice([0.0, 0.5, 1.0, 2.0], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'apoe_genotype': np.random.choice(['E2/E2', 'E2/E3', 'E3/E3', 'E3/E4', 'E4/E4'], 
                                           n_samples, p=[0.05, 0.15, 0.5, 0.25, 0.05])
        }
        
        # Create realistic diagnosis based on features
        diagnoses = []
        for i in range(n_samples):
            if data['mmse_score'][i] >= 27 and data['cdr_score'][i] == 0.0:
                diagnoses.append('Normal')
            elif data['mmse_score'][i] >= 21 and data['cdr_score'][i] <= 0.5:
                diagnoses.append('MCI')
            else:
                diagnoses.append('Dementia')
        
        data['diagnosis'] = diagnoses
        
        df = pd.DataFrame(data)
        logger.info(f"Created sample dataset with {len(df)} rows")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the dataset for machine learning.
        
        Args:
            df: Raw dataset
            
        Returns:
            Tuple of (X, y) where X is features and y is target labels
        """
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle categorical variables
        if 'gender' in df_processed.columns:
            df_processed['gender_encoded'] = (df_processed['gender'] == 'M').astype(int)
        
        if 'apoe_genotype' in df_processed.columns:
            # Create APOE risk encoding (E4 alleles increase risk)
            apoe_risk = df_processed['apoe_genotype'].apply(self._encode_apoe_risk)
            df_processed['apoe_risk'] = apoe_risk
        
        # Define feature columns (exclude target)
        self.feature_columns = [col for col in df_processed.columns 
                               if col not in [self.target_column, 'gender', 'apoe_genotype']]
        
        # Extract features and target
        X = df_processed[self.feature_columns].values
        y = df_processed[self.target_column].values
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Preprocessed data: X shape {X_scaled.shape}, y shape {y_encoded.shape}")
        logger.info(f"Feature columns: {self.feature_columns}")
        logger.info(f"Target classes: {self.label_encoder.classes_}")
        
        return X_scaled, y_encoded
    
    def _encode_apoe_risk(self, genotype: str) -> int:
        """Encode APOE genotype as risk score."""
        if 'E4/E4' in genotype:
            return 3  # Highest risk
        elif 'E4' in genotype:
            return 2  # Moderate risk
        elif 'E2' in genotype:
            return 0  # Protective
        else:
            return 1  # Neutral (E3/E3)
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train machine learning model on the preprocessed data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Check if dataset is large enough for train/test split
        if len(X) < 20:
            logger.warning("Small dataset detected. Using entire dataset for training and testing.")
            X_train, X_test = X, X
            y_train, y_test = y, y
        else:
            # Split data into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        # Initialize and train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Get feature importance
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': feature_importance,
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'classes': self.label_encoder.classes_.tolist(),
            'classification_report': classification_report(y_test, y_pred_test, 
                                                         target_names=self.label_encoder.classes_,
                                                         output_dict=True)
        }
        
        logger.info(f"Training completed. Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
        logger.info(f"Top features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        return results
    
    def save_model(self, filename: str) -> bool:
        """
        Save the trained model and preprocessing components.
        
        Args:
            filename: Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No trained model to save. Train a model first.")
            return False
        
        try:
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved successfully to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filename: str) -> bool:
        """
        Load a previously trained model.
        
        Args:
            filename: Path to the saved model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            
            logger.info(f"Model loaded successfully from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions on new data.
        
        Args:
            features: Feature matrix for prediction
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        if self.model is None:
            raise ValueError("No trained model available. Train or load a model first.")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Convert to readable labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return {
            'predictions': predicted_labels.tolist(),
            'probabilities': probabilities.tolist(),
            'classes': self.label_encoder.classes_.tolist()
        }


class TrainingIntegratedAgent(UnifiedAdaptiveAgent):
    """
    Enhanced agent that integrates machine learning training capabilities
    with the existing duetmind_adaptive framework.
    """
    
    def __init__(self, name: str, style: Dict[str, float], alive_node: AliveLoopNode, 
                 resource_room: ResourceRoom, trainer: AlzheimerTrainer):
        """
        Initialize the training-integrated agent.
        
        Args:
            name: Agent name
            style: Agent style parameters
            alive_node: Neural network node
            resource_room: Shared resource room
            trainer: Trained AlzheimerTrainer instance
        """
        super().__init__(name, style, alive_node, resource_room)
        self.trainer = trainer
        self.ml_predictions_cache = {}
        
    def enhanced_reason_with_ml(self, task: str, patient_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced reasoning that combines traditional agent reasoning with ML predictions.
        
        Args:
            task: Reasoning task
            patient_features: Optional patient features for ML prediction
            
        Returns:
            Dictionary containing both traditional reasoning and ML prediction results
        """
        # Perform traditional reasoning
        traditional_result = self.reason(task)
        
        # Enhanced result structure
        enhanced_result = {
            'task': task,
            'traditional_reasoning': traditional_result,
            'ml_prediction': None,
            'confidence_combined': traditional_result.get('confidence', 0.5),
            'agent': self.name,
            'enhancement_type': 'ml_integrated'
        }
        
        # Add ML prediction if patient features provided
        if patient_features and self.trainer.model is not None:
            try:
                ml_result = self._make_ml_prediction(patient_features)
                enhanced_result['ml_prediction'] = ml_result
                
                # Combine confidences (weighted average)
                ml_confidence = ml_result.get('max_probability', 0.3)
                traditional_confidence = traditional_result.get('confidence', 0.5)
                combined_confidence = 0.7 * ml_confidence + 0.3 * traditional_confidence
                enhanced_result['confidence_combined'] = combined_confidence
                
                # Log the ML enhancement
                self.log_event(f"ML prediction integrated: {ml_result['prediction']} (conf: {ml_confidence:.3f})")
                
            except Exception as e:
                self.log_event(f"ML prediction failed: {e}")
                enhanced_result['ml_prediction'] = {'error': str(e)}
        
        # Store in knowledge graph
        key = f"{self.name}_enhanced_reason_{len(self.knowledge_graph)}"
        self.knowledge_graph[key] = enhanced_result
        
        return enhanced_result
    
    def _make_ml_prediction(self, patient_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make ML prediction from patient features.
        
        Args:
            patient_features: Dictionary of patient features
            
        Returns:
            Dictionary containing prediction results
        """
        # Convert patient features to the format expected by the model
        feature_vector = self._extract_feature_vector(patient_features)
        
        # Make prediction
        prediction_result = self.trainer.predict(feature_vector.reshape(1, -1))
        
        return {
            'prediction': prediction_result['predictions'][0],
            'probabilities': prediction_result['probabilities'][0],
            'classes': prediction_result['classes'],
            'max_probability': max(prediction_result['probabilities'][0]),
            'features_used': self.trainer.feature_columns
        }
    
    def _extract_feature_vector(self, patient_features: Dict[str, Any]) -> np.ndarray:
        """
        Extract feature vector from patient features dictionary.
        
        Args:
            patient_features: Dictionary of patient features
            
        Returns:
            numpy array of features in the correct order
        """
        # Initialize feature vector
        features = []
        
        # Map patient features to model features
        feature_mapping = {
            'age': patient_features.get('age', 70),
            'education_level': patient_features.get('education_level', 14),
            'mmse_score': patient_features.get('mmse_score', 24),
            'cdr_score': patient_features.get('cdr_score', 0.5),
            'gender_encoded': 1 if patient_features.get('gender', 'M') == 'M' else 0,
            'apoe_risk': self._encode_apoe_risk(patient_features.get('apoe_genotype', 'E3/E3'))
        }
        
        # Extract features in the order expected by the model
        for feature_name in self.trainer.feature_columns:
            features.append(feature_mapping.get(feature_name, 0))
        
        return np.array(features)
    
    def _encode_apoe_risk(self, genotype: str) -> int:
        """Encode APOE genotype as risk score (same as trainer)."""
        if 'E4/E4' in genotype:
            return 3
        elif 'E4' in genotype:
            return 2
        elif 'E2' in genotype:
            return 0
        else:
            return 1
    
    def get_ml_insights(self) -> Dict[str, Any]:
        """
        Get insights about the integrated ML model.
        
        Returns:
            Dictionary containing model insights
        """
        if self.trainer.model is None:
            return {'error': 'No trained model available'}
        
        return {
            'model_type': type(self.trainer.model).__name__,
            'feature_columns': self.trainer.feature_columns,
            'target_classes': self.trainer.label_encoder.classes_.tolist(),
            'feature_importance': dict(zip(self.trainer.feature_columns, 
                                         self.trainer.model.feature_importances_)) if hasattr(self.trainer.model, 'feature_importances_') else None,
            'training_integrated': True
        }


# Demo function to show the complete system in action
def demo_training_system():
    """Demonstrate the complete machine learning training system."""
    print("=== DuetMind Adaptive ML Training System Demo ===\n")
    
    # Step 1: Basic Training
    print("1. Basic Training Example:")
    trainer = AlzheimerTrainer()
    
    # Use larger sample dataset for better training
    print("   Creating larger sample dataset for demonstration...")
    df = trainer._create_sample_data()  # This creates 100 samples
    X, y = trainer.preprocess_data(df)
    results = trainer.train_model(X, y)
    trainer.save_model("my_model.pkl")
    
    print(f"   Training completed with {results['test_accuracy']:.3f} test accuracy")
    print(f"   Top features: {sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Step 2: Enhanced Agent
    print("\n2. Enhanced Agent Example:")
    
    # Create framework components
    alive_node = AliveLoopNode((0, 0), (0.1, 0.1), initial_energy=15.0)
    resource_room = ResourceRoom()
    
    # Create enhanced agent
    agent = TrainingIntegratedAgent("MedicalAI", {"logic": 0.9}, alive_node, resource_room, trainer)
    
    # Example patient features
    patient_features = {
        'age': 75,
        'gender': 'F',
        'education_level': 12,
        'mmse_score': 22,
        'cdr_score': 1.0,
        'apoe_genotype': 'E3/E4'
    }
    
    # Enhanced reasoning with ML
    result = agent.enhanced_reason_with_ml("Assess patient", patient_features)
    
    print(f"   Task: {result['task']}")
    print(f"   Traditional reasoning: {result['traditional_reasoning']['insight']}")
    
    if result['ml_prediction'] and 'error' not in result['ml_prediction']:
        print(f"   ML Prediction: {result['ml_prediction']['prediction']}")
        print(f"   ML Confidence: {result['ml_prediction']['max_probability']:.3f}")
        print(f"   Combined Confidence: {result['confidence_combined']:.3f}")
    else:
        print(f"   ML Prediction: Failed ({result['ml_prediction'].get('error', 'Unknown error') if result['ml_prediction'] else 'No prediction made'})")
        print(f"   Traditional Confidence: {result['traditional_reasoning'].get('confidence', 0.5):.3f}")
    
    # Show ML insights
    print("\n3. ML Model Insights:")
    insights = agent.get_ml_insights()
    print(f"   Model: {insights['model_type']}")
    print(f"   Features: {insights['feature_columns']}")
    print(f"   Classes: {insights['target_classes']}")
    
    print("\n=== Training System Demo Complete ===")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demo
    demo_training_system()